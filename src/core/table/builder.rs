// Copyright (c) 2026 Richard Albright. All rights reserved.

use crate::core::catalog::Catalog;
use crate::core::index::memory::InMemoryVectorIndex;
use crate::core::manifest::ManifestManager;
use crate::core::query::QueryConfig;
use crate::core::storage::create_object_store;
use crate::core::wal::WriteAheadLog;
use anyhow::Result;
use arrow::array::Array;
use arrow::datatypes::{Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use object_store::ObjectStore;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;
use tracing;

use super::Table;

/// Remove internal WAL tracking metadata from schema
fn clean_wal_metadata(schema: &Schema) -> Schema {
    let mut meta = schema.metadata().clone();
    meta.remove("benostream:tx_id");
    meta.remove("benostream:seq");
    schema.clone().with_metadata(meta)
}

/// Robust schema merge that handles column additions, nullability relaxation,
/// and metadata reconciliation without failing on transaction tags.
fn merge_arrow_schemas(base: &Schema, incoming: &Schema) -> Schema {
    let base_clean = clean_wal_metadata(base);
    let incoming_clean = clean_wal_metadata(incoming);

    if let Ok(merged) = Schema::try_merge(vec![base_clean.clone(), incoming_clean.clone()]) {
        return merged;
    }

    let mut fields: Vec<arrow::datatypes::Field> =
        base_clean.fields().iter().map(|f| (**f).clone()).collect();
    for field in incoming_clean.fields() {
        if let Some(idx) = fields.iter().position(|f| f.name() == field.name()) {
            let existing = &fields[idx];
            let is_nullable = existing.is_nullable() || field.is_nullable();
            let mut updated = (**field).clone();
            updated.set_nullable(is_nullable);
            fields[idx] = updated;
        } else {
            fields.push((**field).clone());
        }
    }

    let mut merged_meta = base_clean.metadata().clone();
    for (k, v) in incoming_clean.metadata() {
        merged_meta.entry(k.clone()).or_insert_with(|| v.clone());
    }

    Schema::new_with_metadata(fields, merged_meta)
}

/// Shared WAL recovery logic used by both sync and async Table constructors.
/// Promotes schema to the widest version, aligns all recovered batches, and
/// rebuilds the in-memory vector index from recovered data.
/// Returns (aligned_buffer, optional_memory_index, promoted_schema).
pub(crate) fn recover_wal_state(
    recovered_stream: Box<dyn Iterator<Item = Result<RecordBatch>>>,
    schema_val: SchemaRef,
) -> (Vec<RecordBatch>, Option<InMemoryVectorIndex>, SchemaRef) {
    let mut aligned_buffer = Vec::new();
    let mut total_rows = 0;

    // 1. Collect batches from stream
    let mut batches = Vec::new();
    for batch_res in recovered_stream {
        match batch_res {
            Ok(batch) => batches.push(batch),
            Err(e) => tracing::error!("WAL Replay Error: {}", e),
        }
    }

    if batches.is_empty() {
        return (Vec::new(), None, schema_val);
    }

    tracing::info!("Recovering {} batches from WAL...", batches.len());

    // 2. Compute widest merged schema across all recovered batches
    let mut merged = clean_wal_metadata(&schema_val);
    if merged.fields().is_empty() {
        if let Some(first) = batches.first() {
            merged = clean_wal_metadata(first.schema().as_ref());
        }
    }

    for batch in &batches {
        merged = merge_arrow_schemas(&merged, batch.schema().as_ref());
    }
    let schema_val = std::sync::Arc::new(merged);

    // 3. Align all recovered batches to the widest schema
    for b in batches {
        let aligned = if b.schema().fields() != schema_val.fields() {
            let mut cols = Vec::with_capacity(schema_val.fields().len());
            for field in schema_val.fields() {
                let col = if let Some(c) = b.column_by_name(field.name()) {
                    c.clone()
                } else {
                    arrow::array::new_null_array(field.data_type(), b.num_rows())
                };
                cols.push(col);
            }
            RecordBatch::try_new(schema_val.clone(), cols).unwrap_or(b)
        } else if b.schema().as_ref() != schema_val.as_ref() {
            RecordBatch::try_new(schema_val.clone(), b.columns().to_vec()).unwrap_or(b)
        } else {
            b
        };
        aligned_buffer.push(aligned);
    }

    // Rebuild in-memory vector index from recovered data.
    // Look for an "embedding" column (the most common convention), supporting
    // both FixedSizeList and variable-length List arrays.
    let col_name = aligned_buffer.first().and_then(|b| {
        b.schema()
            .fields()
            .iter()
            .find(|f| f.name() == "embedding")
            .map(|f| f.name().clone())
    });

    let mut mem_index = None;
    if let Some(ref col_name) = col_name {
        if let Some(first) = aligned_buffer.first() {
            if let Some(col) = first.column_by_name(col_name) {
                let dim = if let Some(fsl) = col
                    .as_any()
                    .downcast_ref::<arrow::array::FixedSizeListArray>()
                {
                    Some(fsl.value_length() as usize)
                } else if let Some(list) = col.as_any().downcast_ref::<arrow::array::ListArray>() {
                    (0..list.len()).find_map(|i| {
                        if list.is_null(i) {
                            None
                        } else {
                            list.value(i)
                                .as_any()
                                .downcast_ref::<arrow::array::Float32Array>()
                                .map(|v| v.len())
                        }
                    })
                } else {
                    None
                };

                if let Some(d) = dim {
                    let mut idx = InMemoryVectorIndex::new(d);
                    for batch in &aligned_buffer {
                        let _ = idx.insert_batch(batch, col_name, total_rows);
                        total_rows += batch.num_rows();
                    }
                    mem_index = Some(idx);
                }
            }
        }
    }

    (aligned_buffer, mem_index, schema_val)
}

// ============================================================================
// Table Builder
// ============================================================================

pub struct TableBuilder {
    uri: String,
    catalog: Option<Arc<dyn Catalog>>,
    catalog_namespace: Option<String>,
    catalog_table_name: Option<String>,
    runtime: Option<Arc<Runtime>>,
    index_all: bool,
    default_device: Option<String>,
    query_config: QueryConfig,
    /// Override for the manifest/metadata object store. When `None`, the store
    /// is derived from the URI. Set via [`TableBuilder::with_store`] to share a
    /// store across tables (e.g. an in-memory or fault-injecting store) — the
    /// WS3 concurrency harness relies on this.
    store: Option<Arc<dyn ObjectStore>>,
    data_store: Option<Arc<dyn ObjectStore>>,
    label_pattern: crate::core::table::LabelPattern,
    wal_dir: Option<std::path::PathBuf>,
    durability: crate::core::table::WalDurability,
    streaming_flush_interval: Option<std::time::Duration>,
    max_ingest_ram_gb: Option<f64>,
}

impl TableBuilder {
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            catalog: None,
            catalog_namespace: None,
            catalog_table_name: None,
            runtime: None,
            index_all: false,
            default_device: None,
            query_config: QueryConfig::default(),
            store: None,
            data_store: None,
            label_pattern: crate::core::table::LabelPattern::default(),
            wal_dir: None,
            durability: std::env::var("BENOSTREAM_WAL_DURABILITY")
                .or_else(|_| std::env::var("BENOSEARCH_WAL_DURABILITY"))
                .ok()
                .as_deref()
                .map(|v| match v.to_ascii_lowercase().as_str() {
                    "async" => crate::core::table::WalDurability::Async,
                    _ => crate::core::table::WalDurability::Sync,
                })
                .unwrap_or_default(),
            streaming_flush_interval: std::env::var("BENOSTREAM_STREAMING_FLUSH_INTERVAL_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .map(std::time::Duration::from_secs),
            // Always active: an explicit `BSDB_MAX_INGEST_RAM_GB` wins, otherwise
            // the high-water mark is 80% of the memory actually available to the
            // process (the container's cgroup limit when set, else host RAM).
            max_ingest_ram_gb: Some(
                std::env::var("BSDB_MAX_INGEST_RAM_GB")
                    .ok()
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|gb| *gb > 0.0)
                    .unwrap_or_else(crate::core::resources::default_max_ingest_ram_gb),
            ),
        }
    }

    pub fn with_wal_dir<P: Into<std::path::PathBuf>>(mut self, path: P) -> Self {
        self.wal_dir = Some(path.into());
        self
    }

    pub fn with_durability(mut self, durability: crate::core::table::WalDurability) -> Self {
        self.durability = durability;
        self
    }

    pub fn with_catalog(
        mut self,
        catalog: Arc<dyn Catalog>,
        namespace: &str,
        table_name: &str,
    ) -> Self {
        self.catalog = Some(catalog);
        self.catalog_namespace = Some(namespace.to_string());
        self.catalog_table_name = Some(table_name.to_string());
        self
    }

    pub fn with_runtime(mut self, rt: Arc<Runtime>) -> Self {
        self.runtime = Some(rt);
        self
    }

    pub fn with_index_all(mut self, index_all: bool) -> Self {
        self.index_all = index_all;
        self
    }

    pub fn with_default_device(mut self, device: &str) -> Self {
        self.default_device = Some(device.to_string());
        self
    }

    pub fn with_max_ingest_ram_gb(mut self, gb: f64) -> Self {
        self.max_ingest_ram_gb = Some(gb);
        self
    }

    pub fn with_query_config(mut self, config: QueryConfig) -> Self {
        self.query_config = config;
        self
    }

    pub fn with_data_store(mut self, store: Arc<dyn ObjectStore>) -> Self {
        self.data_store = Some(store);
        self
    }

    /// Override the manifest/metadata object store instead of deriving it from
    /// the URI. Enables sharing one store across multiple `Table` handles (the
    /// multi-writer concurrency harness) and injecting a fault-injecting store.
    pub fn with_store(mut self, store: Arc<dyn ObjectStore>) -> Self {
        self.store = Some(store);
        self
    }

    pub fn with_auto_label_columns(mut self, pattern: crate::core::table::LabelPattern) -> Self {
        self.label_pattern = pattern;
        self
    }

    pub fn with_streaming_flush_interval(mut self, interval: std::time::Duration) -> Self {
        self.streaming_flush_interval = Some(interval);
        self
    }

    pub async fn build_async(self) -> Result<Table> {
        // Normalize URI to absolute path if it is local
        let uri = if !self.uri.contains("://") || self.uri.starts_with("file://") {
            let path = self.uri.strip_prefix("file://").unwrap_or(&self.uri);
            let abs_path = std::fs::canonicalize(path).unwrap_or_else(|_| {
                if let Ok(current) = std::env::current_dir() {
                    current.join(path)
                } else {
                    std::path::PathBuf::from(path)
                }
            });
            format!("file://{}", abs_path.display())
        } else {
            self.uri.clone()
        };

        if let Some((base, prefix, ns, table)) = Table::detect_iceberg_rest(&uri) {
            return Box::pin(Table::new_from_rest(base, prefix, ns, table, &uri)).await;
        }

        let store = match self.store {
            Some(s) => s,
            None => create_object_store(&uri)?,
        };

        let manifest_manager = ManifestManager::new(store.clone(), "", &uri);
        let (manifest, version) = manifest_manager.load_latest().await.unwrap_or_default();
        let schema_val = if version > 0 {
            Table::load_initial_schema(store.clone(), &uri).await
        } else {
            Arc::new(Schema::new(Vec::<arrow::datatypes::Field>::new()))
        };
        let partition_spec = Arc::new(manifest.partition_spec.clone());

        // Initialize WAL
        let wal_dir = if let Some(dir) = self.wal_dir {
            dir
        } else if let Ok(env_dir) = std::env::var("BENOSTREAM_WAL_DIR") {
            std::path::PathBuf::from(env_dir)
        } else if uri.starts_with("file://") {
            let path = uri.strip_prefix("file://").unwrap_or(&uri);
            std::path::PathBuf::from(path).join("_wal")
        } else {
            let safe_uri = uri.replace("://", "_").replace("/", "_");
            let dir = std::env::temp_dir().join("benostream_wal").join(safe_uri);
            // `warn!`, not `info!`: falling back to a temp-dir WAL means the
            // writes are not durable across a machine loss, which operators
            // must see in production logs.
            tracing::warn!(
                "Table initialized with remote URI '{}' using default WAL directory '{}'. \
                For persistent machine-loss durability, configure a persistent WAL path using with_wal_dir() or BENOSTREAM_WAL_DIR.",
                uri,
                dir.display()
            );
            dir
        };

        if !wal_dir.exists() {
            std::fs::create_dir_all(&wal_dir).unwrap_or_default();
        }

        let mut wal = WriteAheadLog::new(wal_dir);
        let _ = wal.spawn_worker();

        // Replay WAL (Recovery) - single pass to avoid double reads
        let (recovered_batches, recovered_paths) = wal.replay().unwrap_or_else(|e| {
            tracing::warn!("WAL Recovery Warning: {}", e);
            (vec![], vec![])
        });

        // Idempotent recovery: skip WAL records whose transaction was already
        // committed to the manifest. This is the "manifest-before-WAL-truncation"
        // crash case (review case E): the commit succeeded but the process died
        // before the WAL was truncated, so replaying the record would duplicate
        // the committed rows. The committed tx ids are recorded in the manifest
        // property `benostream.committed_wal_tx` at commit time.
        let committed_tx: std::collections::HashSet<uuid::Uuid> = manifest
            .properties
            .get("benostream.committed_wal_tx")
            .map(|s| {
                s.split(',')
                    .filter_map(|t| uuid::Uuid::parse_str(t.trim()).ok())
                    .collect()
            })
            .unwrap_or_default();
        let recovered_batches: Vec<RecordBatch> = if committed_tx.is_empty() {
            recovered_batches
        } else {
            let before = recovered_batches.len();
            let filtered: Vec<RecordBatch> = recovered_batches
                .into_iter()
                .filter(|b| match crate::core::wal::extract_wal_tx(b) {
                    Some(h) => !committed_tx.contains(&h.tx_id),
                    None => true,
                })
                .collect();
            if filtered.len() != before {
                tracing::info!(
                    "WAL recovery: skipped {} already-committed record(s) (idempotent replay)",
                    before - filtered.len()
                );
            }
            filtered
        };

        // Track the tx ids of the recovered (uncommitted) WAL records. Their rows
        // are now in the write buffer, so the *next* commit must record them as
        // committed — otherwise a later crash would re-replay them and duplicate
        // the rows (the recovered batch is committed as part of the buffer, but
        // its tx id would not be in `benostream.committed_wal_tx`).
        let recovered_tx_ids: Vec<uuid::Uuid> = recovered_batches
            .iter()
            .filter_map(|b| crate::core::wal::extract_wal_tx(b).map(|h| h.tx_id))
            .collect();

        let recovered_stream = Box::new(recovered_batches.into_iter().map(Ok));

        let (initial_buffer, initial_mem_index, schema_val) =
            recover_wal_state(recovered_stream, schema_val);

        // Restore the persisted index configuration from the manifest schema so
        // an opened table inherits the indexes its segments were built with
        // (previously the config was in-memory only and lost on reopen).
        let (restored_index_columns, restored_index_configs) = {
            let mut cols: Vec<String> = Vec::new();
            let mut cfgs: HashMap<String, crate::core::table::state::ColumnIndexConfig> =
                HashMap::new();
            if let Some(schema) = manifest.schemas.last() {
                for f in &schema.fields {
                    if !f.indexes.is_empty() {
                        cols.push(f.name.clone());
                        cfgs.insert(
                            f.name.clone(),
                            crate::core::table::state::ColumnIndexConfig {
                                device: None,
                                tokenizer: None,
                                enabled: true,
                                algorithms: f.indexes.clone(),
                            },
                        );
                    }
                }
            }
            if !cols.is_empty() {
                tracing::info!(
                    "Restored index configuration for {} column(s) from manifest: {:?}",
                    cols.len(),
                    cols
                );
            }
            (cols, cfgs)
        };

        let table = Table {
            uri: uri.clone(),
            store,
            data_store: self.data_store,
            rt: self.runtime,
            query_config: self.query_config,

            indexing: crate::core::table::TableIndexState {
                index_all: self.index_all,
                index_columns: Arc::new(parking_lot::RwLock::new(restored_index_columns)),
                index_configs: Arc::new(parking_lot::RwLock::new(restored_index_configs)),
                default_device: Arc::new(parking_lot::RwLock::new(self.default_device)),
                memory_index: Arc::new(parking_lot::RwLock::new(initial_mem_index)),
            },

            catalog_state: crate::core::table::TableCatalogState {
                catalog: self.catalog,
                namespace: self.catalog_namespace,
                table_name: self.catalog_table_name,
            },

            schema: Arc::new(parking_lot::RwLock::new(schema_val)),
            write_buffer: Arc::new(parking_lot::RwLock::new(initial_buffer)),
            wal: Arc::new(Mutex::new(wal)),
            background_tasks: Arc::new(Mutex::new(Vec::new())),
            index_build_gate: super::new_index_build_gate(),
            sort_order: Arc::new(parking_lot::RwLock::new(None)),
            sort_order_columns: Arc::new(parking_lot::RwLock::new(None)),
            #[cfg(feature = "enterprise")]
            enterprise_license: None,
            primary_key: Arc::new(parking_lot::RwLock::new(Vec::new())),
            autocommit: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            recovered_wal_paths: Arc::new(parking_lot::Mutex::new(recovered_paths)),
            partition_spec,
            label_pattern: self.label_pattern,
            durability: self.durability,
            max_ingest_ram_gb: self.max_ingest_ram_gb,
            memory_reclaimed: Arc::new(tokio::sync::Notify::new()),
            format_version: Arc::new(std::sync::atomic::AtomicI32::new(manifest.format_version)),
            pending_wal_tx_ids: Arc::new(parking_lot::Mutex::new(recovered_tx_ids)),
        };

        table.sync_primary_key_from_schema_async().await.ok();
        let _ = table.infer_index_metadata_from_physical_async().await;

        // One-time migration of legacy v1 graph indexes (see
        // `migrate_legacy_graph_indexes_async`). Run in the background so
        // opening a large table is not blocked; queries during the rebuild use
        // the SQL BFS fallback, which is correct. Set
        // BENOSTREAM_DISABLE_GRAPH_MIGRATION=1 to opt out.
        if std::env::var("BENOSTREAM_DISABLE_GRAPH_MIGRATION").as_deref() != Ok("1") {
            let migration_table = table.clone();
            tokio::spawn(async move {
                if let Err(e) = migration_table.migrate_legacy_graph_indexes_async().await {
                    tracing::warn!("legacy graph index migration failed: {e}");
                }
            });
        }

        if let Some(interval) = self.streaming_flush_interval {
            table.start_streaming_flush_task(interval);
        }

        Ok(table)
    }

    pub fn build(mut self) -> Result<Table> {
        let rt = match self.runtime {
            Some(ref r) => r.clone(),
            None => {
                let r = Arc::new(Runtime::new()?);
                self.runtime = Some(r.clone());
                r
            }
        };
        rt.block_on(self.build_async())
    }
}
