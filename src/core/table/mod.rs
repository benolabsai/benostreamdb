// Copyright (c) 2026 Richard Albright. All rights reserved.

/// Core Table API - High-level interface for HyperStreamDB tables
///
/// This module provides the main Table abstraction that encapsulates:
/// - Query execution (with filters, vector search)
/// - Write operations
/// - Delete operations
/// - Merge-on-read & Merge-on-write
/// - Compaction & maintenance
/// - External catalog integration
///
/// Language bindings (Python, Java, etc.) are thin wrappers around this core API.
use anyhow::Result;
use arrow::record_batch::RecordBatch;
use object_store::ObjectStore;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use tracing;

pub mod builder;
pub use builder::TableBuilder;
pub mod catalog;
pub mod fluent;
pub use fluent::TableQuery;
pub mod index_config;
pub mod maintenance;
pub mod merge;
pub use merge::MergeMode;
pub mod primary_key;
pub mod read;
pub mod schema;
pub mod state;
pub use state::{ColumnIndexConfig, LabelPattern};
pub(crate) use state::{TableCatalogState, TableIndexState};
pub mod stats;
pub use stats::{DataFileInfo, IndexCoverage, Split, TableStatistics};
pub mod write;

#[cfg(test)]
mod tests;

pub use crate::core::planner::VectorSearchParams;

use crate::core::manifest::{
    ManifestManager, NullOrder, PartitionSpec, SortDirection, SortField, SortOrder,
};
use crate::core::metadata::TableMetadata;
use crate::core::query::QueryConfig;
use crate::core::reader::HybridReader;
use crate::core::storage::create_object_store;
use crate::core::wal::WriteAheadLog;
use crate::SegmentConfig;
use arrow::datatypes::{Schema, SchemaRef};

/// Main Table struct - represents a HyperStreamDB table
pub struct Table {
    pub uri: String,
    pub store: Arc<dyn ObjectStore>,
    /// Optional separate store for external data (Layered Indexing)
    pub data_store: Option<Arc<dyn ObjectStore>>,
    pub rt: Option<Arc<Runtime>>,
    pub(crate) query_config: QueryConfig,

    // Sub-modules (Internal State Groups)
    pub(crate) indexing: TableIndexState,
    pub(crate) catalog_state: TableCatalogState,

    pub(crate) schema: Arc<parking_lot::RwLock<SchemaRef>>,
    pub(crate) write_buffer: Arc<parking_lot::RwLock<Vec<RecordBatch>>>,
    pub(crate) wal: Arc<Mutex<WriteAheadLog>>,
    pub(crate) background_tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,

    /// Sort order to apply when writing data (Iceberg V2 spec compliance)
    pub(crate) sort_order: Arc<parking_lot::RwLock<Option<SortOrder>>>,
    /// Column names for sort order (needed for column lookup)
    pub(crate) sort_order_columns: Arc<parking_lot::RwLock<Option<Vec<String>>>>,
    #[cfg(feature = "enterprise")]
    pub(crate) enterprise_license: Option<String>,
    pub(crate) primary_key: Arc<parking_lot::RwLock<Vec<String>>>,
    pub(crate) autocommit: Arc<std::sync::atomic::AtomicBool>,
    pub(crate) recovered_wal_paths: Arc<parking_lot::Mutex<Vec<String>>>,
    pub(crate) partition_spec: Arc<PartitionSpec>,
    /// Naming pattern to use for unnamed columns
    pub(crate) label_pattern: LabelPattern,
}

/// Generates an Excel-style column label (A, B, C... AA, AB...) for a given index.
pub fn excel_column_label(mut index: usize) -> String {
    let mut label = String::new();
    loop {
        let remainder = index % 26;
        label.push((b'A' + remainder as u8) as char);
        if index < 26 {
            break;
        }
        index = (index / 26) - 1;
    }
    label.chars().rev().collect()
}

impl Clone for Table {
    fn clone(&self) -> Self {
        Self {
            uri: self.uri.clone(),
            store: self.store.clone(),
            data_store: self.data_store.clone(),
            rt: self.rt.clone(),
            query_config: self.query_config.clone(),
            indexing: self.indexing.clone(),
            catalog_state: self.catalog_state.clone(),
            schema: self.schema.clone(),
            write_buffer: self.write_buffer.clone(),
            wal: self.wal.clone(),
            background_tasks: self.background_tasks.clone(),
            sort_order: self.sort_order.clone(),
            sort_order_columns: self.sort_order_columns.clone(),
            #[cfg(feature = "enterprise")]
            enterprise_license: self.enterprise_license.clone(),
            primary_key: self.primary_key.clone(),
            autocommit: self.autocommit.clone(),
            recovered_wal_paths: self.recovered_wal_paths.clone(),
            partition_spec: self.partition_spec.clone(),
            label_pattern: self.label_pattern,
        }
    }
}

impl std::fmt::Debug for Table {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Table")
            .field("uri", &self.uri)
            .field("index_all", &self.indexing.index_all)
            .finish()
    }
}

impl Drop for Table {
    fn drop(&mut self) {
        if let Ok(tasks) = self.background_tasks.try_lock() {
            let pending = tasks.iter().filter(|t| !t.is_finished()).count();
            if pending > 0 {
                tracing::warn!(
                    "Table instance for '{}' dropped with {} pending background tasks. These tasks are now detached.",
                    self.uri,
                    pending
                );
            }
        }
        crate::telemetry::tracing::flush_telemetry();
    }
}

impl Table {
    pub fn builder(uri: impl Into<String>) -> builder::TableBuilder {
        builder::TableBuilder::new(uri)
    }

    pub fn new(uri: String) -> Result<Self> {
        TableBuilder::new(uri).with_index_all(true).build()
    }

    pub async fn new_async(uri: String) -> Result<Self> {
        if let Some((base_url, prefix, namespace, table_name)) = Self::detect_iceberg_rest(&uri) {
            tracing::info!(
                "Detected Iceberg REST URI. Automatically configuring catalog: {}{}",
                base_url,
                prefix
                    .as_ref()
                    .map(|p| format!("/{}", p))
                    .unwrap_or_default()
            );
            return Self::new_from_rest(base_url, prefix, namespace, table_name, &uri).await;
        }

        TableBuilder::new(uri)
            .with_index_all(false)
            .build_async()
            .await
    }

    pub fn create(uri: String, schema: SchemaRef) -> Result<Self> {
        let rt = Arc::new(Runtime::new()?);
        let mut table = rt.clone().block_on(Self::create_async(uri, schema))?;
        table.rt = Some(rt);
        Ok(table)
    }

    pub async fn create_async(uri: String, schema: SchemaRef) -> Result<Self> {
        let store = create_object_store(&uri)?;
        let manifest_manager = ManifestManager::new(store.clone(), "", &uri);

        let (_, version) = manifest_manager.load_latest().await?;
        if version > 0 {
            return Err(anyhow::anyhow!("Table already exists at {}", uri));
        }

        let manifest_schema = crate::core::manifest::Schema::from_arrow(&schema, 1);
        let max_id = manifest_schema
            .fields
            .iter()
            .map(|f| f.id)
            .max()
            .unwrap_or(0);

        manifest_manager
            .update_schema(vec![manifest_schema.clone()], 1, Some(max_id))
            .await?;

        let mut metadata = TableMetadata::new(
            2,
            uuid::Uuid::new_v4().to_string(),
            uri.clone(),
            manifest_schema,
            PartitionSpec::default(),
            SortOrder::default(),
        );
        metadata.save_to_store(store.as_ref(), 1).await?;

        TableBuilder::new(uri)
            .with_index_all(false)
            .build_async()
            .await
    }

    pub async fn create_partitioned_async(
        uri: String,
        schema: SchemaRef,
        spec: crate::core::manifest::PartitionSpec,
    ) -> Result<Self> {
        let store = create_object_store(&uri)?;
        let manifest_manager = ManifestManager::new(store.clone(), "", &uri);

        let (_, version) = manifest_manager.load_latest().await?;
        if version > 0 {
            return Err(anyhow::anyhow!("Table already exists at {}", uri));
        }

        let manifest_schema = crate::core::manifest::Schema::from_arrow(&schema, 1);

        let genesis_manifest = crate::core::manifest::Manifest::new_with_spec(
            1,
            vec![],
            None,
            vec![manifest_schema.clone()],
            1,
            spec.clone(),
        );

        manifest_manager.commit_manifest(genesis_manifest).await?;

        let mut metadata = TableMetadata::new(
            2,
            uuid::Uuid::new_v4().to_string(),
            uri.clone(),
            manifest_schema,
            spec,
            SortOrder::default(),
        );
        metadata.save_to_store(store.as_ref(), 1).await?;

        TableBuilder::new(uri)
            .with_index_all(false)
            .build_async()
            .await
    }

    pub async fn wait_for_background_tasks_async(&self) -> Result<()> {
        let mut tasks = {
            let mut t = self.background_tasks.lock().await;
            std::mem::take(&mut *t)
        };

        for task in tasks.drain(..) {
            task.await
                .map_err(|e| anyhow::anyhow!("Background task failed: {}", e))?;
        }
        Ok(())
    }

    pub fn wait_for_background_tasks(&self) -> Result<()> {
        if let Some(ref rt) = self.rt {
            rt.block_on(self.wait_for_background_tasks_async())
        } else {
            anyhow::bail!(
                "No runtime configured for Table to wait for background tasks synchronously"
            )
        }
    }

    #[cfg(feature = "enterprise")]
    pub fn enable_enterprise(&mut self, license_key: String) -> Result<()> {
        crate::core::license::verify_license(&license_key)?;
        self.enterprise_license = Some(license_key);
        Ok(())
    }

    pub fn is_enterprise_enabled(&self) -> bool {
        #[cfg(feature = "enterprise")]
        {
            self.enterprise_license.is_some()
        }
        #[cfg(not(feature = "enterprise"))]
        {
            false
        }
    }

    pub fn replace_sort_order(&self, columns: &[&str], ascending: &[bool]) -> Result<()> {
        if columns.len() != ascending.len() {
            anyhow::bail!("columns and ascending arrays must have same length");
        }

        let fields: Vec<SortField> = columns
            .iter()
            .zip(ascending.iter())
            .enumerate()
            .map(|(i, (_col, asc))| SortField {
                source_id: i as i32 + 1,
                transform: "identity".to_string(),
                direction: if *asc {
                    SortDirection::Asc
                } else {
                    SortDirection::Desc
                },
                null_order: if *asc {
                    NullOrder::NullsFirst
                } else {
                    NullOrder::NullsLast
                },
            })
            .collect();

        let order = SortOrder {
            order_id: 1,
            fields,
        };

        let mut guard = self.sort_order.write();
        *guard = Some(order);

        self.sort_order_columns
            .write()
            .replace(columns.iter().map(|s| s.to_string()).collect());

        Ok(())
    }

    pub fn get_sort_order(&self) -> Option<SortOrder> {
        self.sort_order.read().clone()
    }

    pub(crate) fn apply_sort_order(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let guard = self.sort_order.read();
        let order = match guard.as_ref() {
            Some(o) if !o.fields.is_empty() => o,
            _ => return Ok(batch.clone()),
        };

        let columns_guard = self.sort_order_columns.read();
        let column_names = match columns_guard.as_ref() {
            Some(names) => names,
            None => return Ok(batch.clone()),
        };

        let mut sort_columns = Vec::new();
        for (field, col_name) in order.fields.iter().zip(column_names.iter()) {
            if let Some((idx, _)) = batch.schema().column_with_name(col_name) {
                let column = batch.column(idx).clone();
                let options = arrow::compute::SortOptions {
                    descending: matches!(field.direction, SortDirection::Desc),
                    nulls_first: matches!(field.null_order, NullOrder::NullsFirst),
                };
                sort_columns.push(arrow::compute::SortColumn {
                    values: column,
                    options: Some(options),
                });
            }
        }

        if sort_columns.is_empty() {
            return Ok(batch.clone());
        }

        let indices = arrow::compute::lexsort_to_indices(&sort_columns, None)?;

        let sorted_columns: Vec<Arc<dyn arrow::array::Array>> = batch
            .columns()
            .iter()
            .map(|col| arrow::compute::take(col.as_ref(), &indices, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(RecordBatch::try_new(batch.schema(), sorted_columns)?)
    }

    #[allow(dead_code)]
    pub(crate) fn has_v3_metadata_columns(schema: &arrow::datatypes::SchemaRef) -> bool {
        schema.column_with_name("_row_id").is_some()
            && schema
                .column_with_name("_last_updated_sequence_number")
                .is_some()
    }

    #[allow(dead_code)]
    pub(crate) fn add_v3_metadata_columns(
        &self,
        batch: &RecordBatch,
        sequence_number: i64,
    ) -> Result<RecordBatch> {
        use arrow::array::{Int64Array, StringArray};
        use arrow::datatypes::{DataType, Field};

        let num_rows = batch.num_rows();

        let row_ids: Vec<String> = (0..num_rows)
            .map(|_| uuid::Uuid::new_v4().to_string())
            .collect();
        let row_id_array = Arc::new(StringArray::from(row_ids));

        let seq_numbers = vec![sequence_number; num_rows];
        let seq_array = Arc::new(Int64Array::from(seq_numbers));

        let mut new_fields: Vec<Arc<Field>> = batch.schema().fields().iter().cloned().collect();
        new_fields.push(Arc::new(Field::new("_row_id", DataType::Utf8, false)));
        new_fields.push(Arc::new(Field::new(
            "_last_updated_sequence_number",
            DataType::Int64,
            false,
        )));
        let new_schema = Arc::new(arrow::datatypes::Schema::new(new_fields));

        let mut new_columns: Vec<Arc<dyn arrow::array::Array>> = batch.columns().to_vec();
        new_columns.push(row_id_array);
        new_columns.push(seq_array);

        Ok(RecordBatch::try_new(new_schema, new_columns)?)
    }

    pub(crate) async fn load_initial_schema(store: Arc<dyn ObjectStore>, _uri: &str) -> SchemaRef {
        let manifest_manager = ManifestManager::new(store.clone(), "", _uri);
        if let Ok((manifest, version)) = manifest_manager.load_latest().await {
            if version > 0 && !manifest.schemas.is_empty() {
                if let Some(latest) = manifest
                    .schemas
                    .iter()
                    .find(|s| s.schema_id == manifest.current_schema_id)
                {
                    return Arc::new(latest.to_arrow());
                } else if let Some(latest) = manifest.schemas.last() {
                    return Arc::new(latest.to_arrow());
                }
            }

            if !manifest.entries.is_empty() {
                if let Some(entry) = manifest.entries.first() {
                    let file_path = &entry.file_path;
                    let parts: Vec<&str> = file_path.split('/').collect();
                    let filename = parts.last().unwrap_or(&"wrapper");
                    let segment_id = filename.replace(".parquet", "");
                    let config = SegmentConfig::new("", &segment_id);
                    let reader = HybridReader::new(config, store.clone(), _uri);
                    if let Ok(mut s) = reader
                        .stream_all(None as Option<arrow::datatypes::SchemaRef>)
                        .await
                    {
                        use futures::StreamExt;
                        if let Some(Ok(batch)) = s.next().await {
                            return batch.schema();
                        }
                    }
                }
            }
        }

        use futures::StreamExt;
        let mut stream = store.list(None);
        let mut first_file = None;
        while let Some(res) = stream.next().await {
            if let Ok(meta) = res {
                let p = meta.location.to_string();
                if p.ends_with(".parquet") && !p.contains(".inv.parquet") && !p.contains(".hnsw.") {
                    first_file = Some(p);
                    break;
                }
            }
        }

        if let Some(path) = first_file {
            let filename = path.split('/').next_back().unwrap_or(&path);
            let segment_id = filename.replace(".parquet", "");
            let config = SegmentConfig::new("", &segment_id);
            let reader = HybridReader::new(config, store.clone(), _uri);
            if let Ok(mut s) = reader
                .stream_all(None as Option<arrow::datatypes::SchemaRef>)
                .await
            {
                if let Some(Ok(batch)) = s.next().await {
                    return batch.schema();
                }
            }
        }
        Arc::new(Schema::empty())
    }

    pub fn set_max_parallel_readers(&mut self, max: usize) {
        self.query_config = self.query_config.clone().with_max_parallel_readers(max);
    }

    pub fn auto_detect_parallel_readers(&mut self) {
        self.query_config.max_parallel_readers = None;
    }

    pub fn get_max_parallel_readers(&self) -> Option<usize> {
        self.query_config.max_parallel_readers
    }
}
