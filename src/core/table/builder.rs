// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use std::sync::Arc;
use object_store::ObjectStore;
use tokio::runtime::Runtime;
use std::collections::HashMap;
use tokio::sync::Mutex;
use tracing;
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{Schema, SchemaRef};
use arrow::array::Array;
use crate::core::catalog::Catalog;
use crate::core::storage::create_object_store;
use crate::core::manifest::ManifestManager;
use crate::core::query::QueryConfig;
use crate::core::wal::WriteAheadLog;
use crate::core::index::memory::InMemoryVectorIndex;

use super::Table;

/// Shared WAL recovery logic used by both sync and async Table constructors.
/// Promotes schema to the widest version, aligns all recovered batches, and
/// rebuilds the in-memory vector index from recovered data.
/// Returns (aligned_buffer, optional_memory_index, promoted_schema).
pub(crate) fn recover_wal_state(
    recovered_stream: Box<dyn Iterator<Item = Result<RecordBatch>>>,
    mut schema_val: SchemaRef,
) -> (Vec<RecordBatch>, Option<InMemoryVectorIndex>, SchemaRef) {
    let mut aligned_buffer = Vec::new();
    let mut total_rows = 0;
    
    // 1. First pass: Collect batches and merge schema
    let mut batches = Vec::new();
    for batch_res in recovered_stream {
        match batch_res {
            Ok(batch) => {
                // Safely merge schema
                match arrow::datatypes::Schema::try_merge(vec![schema_val.as_ref().clone(), batch.schema().as_ref().clone()]) {
                    Ok(s) => schema_val = Arc::new(s),
                    Err(e) => tracing::warn!("Failed to merge WAL batch schema: {}", e),
                }
                batches.push(batch);
            }
            Err(e) => tracing::error!("WAL Replay Error: {}", e),
        }
    }

    if batches.is_empty() {
        return (Vec::new(), None, schema_val);
    }

    tracing::info!("Recovering {} batches from WAL...", batches.len());

    // Use first batch schema if current schema is empty
    if schema_val.fields().is_empty() {
        if let Some(first) = batches.first() {
            schema_val = first.schema();
        }
    }

    // Safely attempt to merge all WAL schemas to capture any column additions
    // or type evolutions instead of fragile field count comparisons.
    let mut merged_schema = schema_val.as_ref().clone();
    for batch in &batches {
        match arrow::datatypes::Schema::try_merge(vec![merged_schema.clone(), batch.schema().as_ref().clone()]) {
            Ok(s) => merged_schema = s,
            Err(e) => tracing::warn!("Failed to merge WAL batch schema: {}", e),
        }
    }
    let schema_val = std::sync::Arc::new(merged_schema);

    // Align all recovered batches to the widest schema
    for b in batches {
        let aligned = if b.schema() != schema_val {
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
        } else {
            b
        };
        aligned_buffer.push(aligned);
    }

    // Rebuild in-memory vector index from recovered data.
    // Look for an "embedding" column (the most common convention), supporting
    // both FixedSizeList and variable-length List arrays.
    let col_name = aligned_buffer.first().and_then(|b| {
        b.schema().fields().iter()
            .find(|f| f.name() == "embedding")
            .map(|f| f.name().clone())
    });

    let mut mem_index = None;
    if let Some(ref col_name) = col_name {
        if let Some(first) = aligned_buffer.first() {
            if let Some(col) = first.column_by_name(col_name) {
                let dim = if let Some(fsl) = col.as_any().downcast_ref::<arrow::array::FixedSizeListArray>() {
                    Some(fsl.value_length() as usize)
                } else if let Some(list) = col.as_any().downcast_ref::<arrow::array::ListArray>() {
                    (0..list.len()).find_map(|i| {
                        if list.is_null(i) { None } else {
                            list.value(i).as_any().downcast_ref::<arrow::array::Float32Array>().map(|v| v.len())
                        }
                    })
                } else { None };

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
    data_store: Option<Arc<dyn ObjectStore>>,
    label_pattern: crate::core::table::LabelPattern,
}

impl TableBuilder {
    pub fn new(uri: impl Into<String>) -> Self {
        Self {
            uri: uri.into(),
            catalog: None,
            catalog_namespace: None,
            catalog_table_name: None,
            runtime: None,
            index_all: true,
            default_device: None,
            query_config: QueryConfig::default(),
            data_store: None,
            label_pattern: crate::core::table::LabelPattern::default(),
        }
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

    pub fn with_query_config(mut self, config: QueryConfig) -> Self {
        self.query_config = config;
        self
    }

    pub fn with_data_store(mut self, store: Arc<dyn ObjectStore>) -> Self {
        self.data_store = Some(store);
        self
    }

    pub fn with_auto_label_columns(mut self, pattern: crate::core::table::LabelPattern) -> Self {
        self.label_pattern = pattern;
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

        let store = create_object_store(&uri)?;
        
        let manifest_manager = ManifestManager::new(store.clone(), "", &uri);
        let (manifest, version) = manifest_manager.load_latest().await.unwrap_or_default();
        let schema_val = if version > 0 {
             Table::load_initial_schema(store.clone(), &uri).await
        } else {
             Arc::new(Schema::new(Vec::<arrow::datatypes::Field>::new()))
        };
        let partition_spec = Arc::new(manifest.partition_spec.clone());

        // Initialize WAL
        let wal_dir = if uri.starts_with("file://") {
            let path = uri.strip_prefix("file://").unwrap();
            std::path::PathBuf::from(path).join("_wal")
        } else {
             let safe_uri = uri.replace("://", "_").replace("/", "_");
             std::env::temp_dir().join("hyperstream_wal").join(safe_uri)
        };

        if !wal_dir.exists() {
            std::fs::create_dir_all(&wal_dir).unwrap_or_default();
        }

        let mut wal = WriteAheadLog::new(wal_dir);
        let _ = wal.spawn_worker();
        
        // Replay WAL (Recovery)
        let recovered_stream = wal.replay_stream().unwrap_or_else(|e| {
            tracing::warn!("WAL Recovery Warning: {}" , e);
            Box::new(std::iter::empty())
        });
        
        let (_, recovered_paths) = wal.replay().unwrap_or_else(|_| (vec![], vec![])); // For paths cleanup only

        let (initial_buffer, initial_mem_index, schema_val) = recover_wal_state(
            recovered_stream, schema_val,
        );

        let table = Table { 
            uri: uri.clone(), 
            store, 
            data_store: self.data_store,
            rt: self.runtime,
            query_config: self.query_config,
            
            indexing: crate::core::table::TableIndexState {
                index_all: self.index_all,
                index_columns: Arc::new(std::sync::RwLock::new(Vec::new())),
                index_configs: Arc::new(std::sync::RwLock::new(HashMap::new())),
                default_device: Arc::new(std::sync::RwLock::new(self.default_device)),
                memory_index: Arc::new(std::sync::RwLock::new(initial_mem_index)),
            },

            catalog_state: crate::core::table::TableCatalogState {
                catalog: self.catalog,
                namespace: self.catalog_namespace,
                table_name: self.catalog_table_name,
            },

            schema: Arc::new(std::sync::RwLock::new(schema_val)),
            write_buffer: Arc::new(std::sync::RwLock::new(initial_buffer)),
            wal: Arc::new(Mutex::new(wal)),
            background_tasks: Arc::new(Mutex::new(Vec::new())),
            sort_order: Arc::new(std::sync::RwLock::new(None)),
            sort_order_columns: Arc::new(std::sync::RwLock::new(None)),
            #[cfg(feature = "enterprise")]
            enterprise_license: None,
            primary_key: Arc::new(std::sync::RwLock::new(Vec::new())),
            autocommit: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            recovered_wal_paths: Arc::new(std::sync::Mutex::new(recovered_paths)),
            partition_spec,
            label_pattern: self.label_pattern,
        };
        
        table.sync_primary_key_from_schema_async().await.ok();
        let _ = table.infer_index_metadata_from_physical_async().await;
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
