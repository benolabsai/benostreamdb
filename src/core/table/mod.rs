// Copyright (c) 2026 Richard Albright. All rights reserved.

/// Core Table API - High-level interface for HyperStreamDB tables
/// 
/// This module provides the main Table abstraction that encapsulates:
/// - Query execution (with filters, vector search)
/// - Write operations
/// - Delete operations  
/// - Merge-on-read
/// - Compaction
/// - Maintenance
///
/// Language bindings (Python, Java, etc.) should be thin wrappers around this core API.
use anyhow::{Result, Context};
use std::collections::HashMap;
use arrow::record_batch::RecordBatch;
use arrow::array::Array;
use object_store::ObjectStore;
use std::sync::Arc;
use tokio::runtime::Runtime;
use tracing;


pub mod builder;
pub use builder::TableBuilder;
pub mod fluent;
pub use fluent::TableQuery;
pub mod schema;
pub mod read;
pub mod write;

use crate::core::storage::create_object_store;
use crate::core::manifest::{Manifest, ManifestEntry, ManifestManager, PartitionSpec, SortOrder, SortField, SortDirection, NullOrder};
use crate::core::metadata::TableMetadata;
use crate::core::planner::{QueryPlanner, QueryFilter, FilterExpr};
use crate::core::reader::HybridReader;
use crate::core::segment::HybridSegmentWriter;
use crate::core::compaction::{Compactor, CompactionOptions};
use crate::core::maintenance::Maintenance;
use crate::SegmentConfig;
use crate::core::query::QueryConfig;
use serde::{Serialize, Deserialize};
use serde_json::Value; // Keep Value
use arrow::datatypes::{Schema, SchemaRef};
use crate::core::index::memory::InMemoryVectorIndex;
use crate::core::wal::WriteAheadLog;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use rayon::prelude::*;
use crate::core::index::gpu::{get_global_gpu_context};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct ColumnIndexConfig {
    pub device: Option<String>,
    pub tokenizer: Option<String>,
    pub enabled: bool,
}


// TableBuilder logic is now in builder.rs

/// Main Table struct - represents a HyperStreamDB table
pub struct Table {
    pub uri: String,
    pub store: Arc<dyn ObjectStore>,
    /// Optional separate store for external data (Layered Indexing)
    pub data_store: Option<Arc<dyn ObjectStore>>,
    pub rt: Option<Arc<Runtime>>,
    query_config: QueryConfig,
    // Indexing Configuration
    index_all: bool,
    index_columns: Arc<std::sync::RwLock<Vec<String>>>,
    index_configs: Arc<std::sync::RwLock<HashMap<String, ColumnIndexConfig>>>,
    default_device: Arc<std::sync::RwLock<Option<String>>>,
    schema: Arc<std::sync::RwLock<SchemaRef>>,
    write_buffer: Arc<std::sync::RwLock<Vec<RecordBatch>>>,
    memory_index: Arc<std::sync::RwLock<Option<InMemoryVectorIndex>>>,
    wal: Arc<Mutex<WriteAheadLog>>,
    background_tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,
    pub catalog: Option<Arc<dyn crate::core::catalog::Catalog>>,
    pub catalog_namespace: Option<String>,
    pub catalog_table_name: Option<String>,
    /// Sort order to apply when writing data (Iceberg V2 spec compliance)
    sort_order: Arc<std::sync::RwLock<Option<SortOrder>>>,
    /// Column names for sort order (needed for column lookup)
    sort_order_columns: Arc<std::sync::RwLock<Option<Vec<String>>>>,
    #[cfg(feature = "enterprise")]
    enterprise_license: Option<String>,
    primary_key: Arc<std::sync::RwLock<Vec<String>>>,
    autocommit: Arc<std::sync::atomic::AtomicBool>,
    recovered_wal_paths: Arc<std::sync::Mutex<Vec<String>>>,
    pub(crate) partition_spec: Arc<PartitionSpec>,
}

impl Clone for Table {
    fn clone(&self) -> Self {
        Self {
            uri: self.uri.clone(),
            store: self.store.clone(),
            data_store: self.data_store.clone(),
            rt: self.rt.clone(),
            query_config: self.query_config.clone(),
            index_all: self.index_all,
            index_columns: self.index_columns.clone(),
            index_configs: self.index_configs.clone(),
            default_device: self.default_device.clone(),
            schema: self.schema.clone(),
            write_buffer: self.write_buffer.clone(),
            memory_index: self.memory_index.clone(),
            wal: self.wal.clone(),
            background_tasks: self.background_tasks.clone(),
            catalog: self.catalog.clone(),
            catalog_namespace: self.catalog_namespace.clone(),
            catalog_table_name: self.catalog_table_name.clone(),
            sort_order: self.sort_order.clone(),
            sort_order_columns: self.sort_order_columns.clone(),
            #[cfg(feature = "enterprise")]
            enterprise_license: self.enterprise_license.clone(),
            primary_key: self.primary_key.clone(),
            autocommit: self.autocommit.clone(),
            recovered_wal_paths: self.recovered_wal_paths.clone(),
            partition_spec: self.partition_spec.clone(),
        }
    }
}

impl std::fmt::Debug for Table {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Table")
            .field("uri", &self.uri)
            .field("index_all", &self.index_all)
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
    }
}


impl Table {
    pub fn set_index_all(&mut self, enabled: bool) {
        self.index_all = enabled;
    }

    pub fn get_index_all(&self) -> bool {
        self.index_all
    }
    
    pub fn set_autocommit(&self, enabled: bool) {
        self.autocommit.store(enabled, std::sync::atomic::Ordering::Relaxed);
    }
    
    pub fn get_autocommit(&self) -> bool {
        self.autocommit.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn set_primary_key(&self, columns: Vec<String>) {
        let mut pk = self.primary_key.write().unwrap();
        *pk = columns;
    }

    pub fn get_primary_key(&self) -> Vec<String> {
        self.primary_key.read().unwrap().clone()
    }

    pub fn set_index_config(&self, column: String, enabled: bool, tokenizer: Option<String>, device: Option<String>) {
        let mut configs = self.index_configs.write().unwrap();
        configs.insert(column.clone(), ColumnIndexConfig {
            device,
            tokenizer,
            enabled,
        });
        
        let mut cols = self.index_columns.write().unwrap();
        if enabled {
            if !cols.contains(&column) {
                cols.push(column);
            }
        } else {
            cols.retain(|c| c != &column);
        }
    }

    /// Convenience wrapper for set_index_config (enabled=true)
    pub fn add_index(&self, column: String, tokenizer: Option<String>, device: Option<String>) {
        self.set_index_config(column, true, tokenizer, device);
    }

    /// Convenience wrapper for set_index_config (enabled=false)
    pub fn drop_index(&self, column: String) {
        self.set_index_config(column, false, None, None);
    }

    /// Add a column to the primary key. 
    /// This is an atomic operation that commits a new manifest version.
    /// Validation: Ensures no duplicate keys exist for the new definition.
    pub async fn add_primary_key(&self, column: String) -> Result<()> {
        let manifest = self.manifest().await?;
        let latest_schema = manifest.schemas.last().ok_or_else(|| anyhow::anyhow!("No schema found"))?;
        
        // Find field ID for column
        let field_id = latest_schema.fields.iter()
            .find(|f| f.name == column)
            .map(|f| f.id)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' not found in schema", column))?;

        let mut next_ids = latest_schema.identifier_field_ids.clone();
        if next_ids.contains(&field_id) {
            return Ok(()); // Already in PK
        }
        next_ids.push(field_id);
        
        // Validate uniqueness before committing
        self._validate_pk_uniqueness(&next_ids, &latest_schema).await?;
        
        // Atomic commit to manifest
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager.update_identifier_fields(next_ids).await?;
        
        // Update in-memory state
        let mut pk = self.primary_key.write().unwrap();
        if !pk.contains(&column) {
            pk.push(column);
        }
        Ok(())
    }

    /// Remove a column from the primary key.
    /// This is an atomic operation that commits a new manifest version.
    pub async fn drop_primary_key(&self, column: String) -> Result<()> {
        let manifest = self.manifest().await?;
        let latest_schema = manifest.schemas.last().ok_or_else(|| anyhow::anyhow!("No schema found"))?;
        
        let field_id = latest_schema.fields.iter()
            .find(|f| f.name == column)
            .map(|f| f.id)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' not found in schema", column))?;

        let mut next_ids = latest_schema.identifier_field_ids.clone();
        next_ids.retain(|id| id != &field_id);

        // Atomic commit to manifest
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager.update_identifier_fields(next_ids).await?;

        // Update in-memory state
        let mut pk = self.primary_key.write().unwrap();
        pk.retain(|c| c != &column);
        Ok(())
    }

    /// Internal helper to validate that a set of field IDs form a unique key across existing data.
    async fn _validate_pk_uniqueness(&self, field_ids: &[i32], schema: &crate::core::manifest::Schema) -> Result<()> {
        let col_names: Vec<String> = field_ids.iter()
            .map(|id| schema.fields.iter().find(|f| f.id == *id).map(|f| f.name.clone()).unwrap())
            .collect();
            
        // For now, we perform a scan to check for duplicates. 
        // In a production environment, this could be optimized using indexes.
        let batches = self.read_with_columns(None, None, col_names.clone()).map_err(|e| anyhow::anyhow!("Validation read failed: {}", e))?;
        
        let mut seen = std::collections::HashSet::new();
        
        for batch in batches {
            let sort_fields = batch.schema().fields().iter()
                .map(|f| arrow::row::SortField::new(f.data_type().clone()))
                .collect::<Vec<_>>();
            let converter = arrow::row::RowConverter::new(sort_fields)
                .map_err(|e| anyhow::anyhow!("RowConverter error: {}", e))?;
            
            let rows = converter.convert_columns(batch.columns())
                .map_err(|e| anyhow::anyhow!("Row conversion error: {}", e))?;
                
            for row in rows.iter() {
                if !seen.insert(row.as_ref().to_vec()) {
                    return Err(anyhow::anyhow!("Primary key violation detected for columns {:?}: Duplicate row values found.", col_names));
                }
            }
        }
        
        Ok(())
    }
}

// TableQuery logic is now in query.rs

impl Table {
    pub fn object_store(&self) -> Arc<dyn ObjectStore> {
        self.store.clone()
    }

    pub fn table_uri(&self) -> String {
        self.uri.clone()
    }

    pub fn query_config(&self) -> &crate::core::query::QueryConfig {
        &self.query_config
    }

    /// Get the number of rows currently in the write buffer (not yet flushed)
    pub fn write_buffer_row_count(&self) -> usize {
        self.write_buffer.read().unwrap().iter().map(|b| b.num_rows()).sum()
    }

    /// Check if the table currently has an active in-memory vector index
    pub fn has_memory_index(&self) -> bool {
        self.memory_index.read().unwrap().is_some()
    }

    pub fn new(uri: String) -> Result<Self> {
        TableBuilder::new(uri).with_index_all(true).build()
    }

    /// Load a table from Nessie Catalog at specific branch/tag/hash
    pub async fn from_nessie(
        nessie_config: crate::core::nessie::NessieConfig,
        namespace: &str,
        table: &str,
        _ref_hash: Option<String>
    ) -> Result<Self> {
        use crate::core::catalog::Catalog;
        let client = crate::core::catalog::nessie::NessieClient::new(nessie_config.uri.clone());
        let metadata = client.load_table(namespace, table).await?;
        
        // We use a local path for the HyperStreamDB artifacts for this external table
        // e.g. /local/path/{namespace}_{table}
        let local_uri = format!("file:///tmp/hyperstream_nessie_{}_{}", namespace, table);
        
        tracing::info!("Resolved Nessie table {} to metadata: {}", table, metadata.location);
        
        let mut table_obj = Self::register_external(local_uri, &metadata.location).await?;
        table_obj.catalog = Some(Arc::new(client));
        table_obj.catalog_namespace = Some(namespace.to_string());
        table_obj.catalog_table_name = Some(table.to_string());
        Ok(table_obj)
    }

    /// Load a table from AWS Glue Catalog
    pub async fn from_glue(
        catalog_id: Option<String>,
        namespace: &str,
        table: &str
    ) -> Result<Self> {
        use crate::core::catalog::Catalog;
        let client = crate::core::catalog::glue::GlueCatalogClient::new(catalog_id).await?;
        let metadata = client.load_table(namespace, table).await?;
        
        // Use local path for artifacts
        let local_uri = format!("file:///tmp/hyperstream_glue_{}_{}", namespace, table);
        tracing::info!("Resolved Glue table {}.{} to metadata: {}", namespace, table, metadata.location);
        
        let mut table_obj = Self::register_external(local_uri, &metadata.location).await?;
        table_obj.catalog = Some(Arc::new(client));
        table_obj.catalog_namespace = Some(namespace.to_string());
        table_obj.catalog_table_name = Some(table.to_string());
        Ok(table_obj)
    }

    /// Load a table from Hive Metastore
    /// address: "host:port" or "thrift://host:port"
    pub async fn from_hive(
        address: &str,
        namespace: &str,
        table: &str
    ) -> Result<Self> {
        use crate::core::catalog::Catalog;
        let client = crate::core::catalog::hive::HiveMetastoreClient::new(address.to_string())?;
        let metadata = client.load_table(namespace, table).await?;
        
        let local_uri = format!("file:///tmp/hyperstream_hive_{}_{}", namespace, table);
        tracing::info!("Resolved Hive table {}.{} to metadata: {}", namespace, table, metadata.location);
        
        let mut table_obj = Self::register_external(local_uri, &metadata.location).await?;
        table_obj.catalog = Some(Arc::new(client));
        table_obj.catalog_namespace = Some(namespace.to_string());
        table_obj.catalog_table_name = Some(table.to_string());
        Ok(table_obj)
    }

    /// Register an existing Iceberg table for Layered Indexing
    pub async fn register_external(uri: String, iceberg_metadata_uri: &str) -> Result<Self> {
        create_object_store(&uri)?;
        
        // Determine if iceberg_metadata_uri is a directory (table location) or a file (metadata file)
        // If it's a directory, we need to find the metadata file inside it
        let (meta_store_uri, filename) = {
            let meta_url = url::Url::parse(iceberg_metadata_uri).context("Invalid iceberg_metadata_uri")?;
            let meta_path_str = meta_url.path();
            let meta_path = std::path::Path::new(meta_path_str);
            
            // Check if the path looks like a metadata file (ends with .json and contains "metadata")
            let path_str = meta_path_str.to_string();
            if path_str.ends_with(".metadata.json") || path_str.ends_with(".json") && path_str.contains("metadata") {
                // It's a metadata file path
                let parent_dir = meta_path.parent().context("No parent directory for metadata file")?;
                let fname = meta_path.file_name().context("No filename for metadata file")?.to_str().unwrap().to_string();
                
                let store_uri = if iceberg_metadata_uri.starts_with("file://") {
                    format!("file://{}", parent_dir.display())
                } else {
                    let mut base_url = meta_url.clone();
                    base_url.set_path(parent_dir.to_str().unwrap());
                    base_url.to_string()
                };
                (store_uri, fname)
            } else {
                // It's a table directory, look for metadata file inside metadata/ subdirectory
                // Try to find the latest metadata file
                let table_dir = meta_path_str;
                let metadata_dir = if iceberg_metadata_uri.starts_with("file://") {
                    format!("file://{}/metadata", table_dir)
                } else {
                    format!("{}/metadata", iceberg_metadata_uri.trim_end_matches('/'))
                };
                
                tracing::debug!("Table location appears to be a directory. Looking for metadata in: {}", metadata_dir);
                
                // Use v1.metadata.json as default
                (metadata_dir, "v1.metadata.json".to_string())
            }
        };

        let iceberg_meta_store = create_object_store(&meta_store_uri)?;
        
        tracing::info!("Linking external Iceberg table: {}", iceberg_metadata_uri);
        
        // 1. Load Iceberg Metadata
        let path = object_store::path::Path::from(filename.as_str());
        let ret = iceberg_meta_store.get(&path).await?;
        let bytes = ret.bytes().await?;
        let iceberg_meta: crate::core::iceberg::IcebergTableMetadata = serde_json::from_slice(&bytes)?;
        
        // 2. Map Iceberg Schema to HyperStreamDB Schema
        // For simplicity, we use the current schema
        let current_schema_json = iceberg_meta.schemas.iter()
            .find(|s| s.get("schema-id").and_then(|v| v.as_i64()) == Some(iceberg_meta.current_schema_id as i64))
            .unwrap_or_else(|| {
                tracing::warn!("Could not find schema with ID {}, falling back to first schema", iceberg_meta.current_schema_id);
                &iceberg_meta.schemas[0]
            });
        let hdb_schema: crate::core::manifest::Schema = serde_json::from_value(current_schema_json.clone())?;
        let schema_ref: SchemaRef = Arc::new(hdb_schema.to_arrow());

        // 3. Initialize HyperStream Table
        // This creates the directory and the initial manifest v0
        let mut table = Self::create_async(uri.clone(), schema_ref.clone()).await?;
        
        // Root data_store carefully: if file, root at / to support absolute paths in manifests.
        // If s3, root at bucket level.
        let data_store_uri = if iceberg_meta.location.starts_with("file://") {
            "file:///".to_string()
        } else if iceberg_meta.location.starts_with("s3://") {
             let url = url::Url::parse(&iceberg_meta.location)?;
             format!("s3://{}/", url.host_str().unwrap_or(""))
        } else {
             iceberg_meta.location.clone()
        };
        table.data_store = Some(create_object_store(&data_store_uri)?);
        
        // 4. Import Snapshot (Incremental Indexing Trigger)
        if let Some(snapshot_id) = iceberg_meta.current_snapshot_id {
            table.import_iceberg_snapshot(snapshot_id, &iceberg_meta, iceberg_meta_store).await?;
        }

        Ok(table)
    }

    /// Detect if a URI is an Iceberg REST Catalog endpoint
    fn detect_iceberg_rest(uri: &str) -> Option<(String, Option<String>, String, String)> {
        if !uri.starts_with("http") { return None; }
        
        let ns_marker = "/namespaces/";
        let t_marker = "/tables/";
        
        let ns_idx = uri.find(ns_marker)?;
        let t_idx = uri.find(t_marker)?;
        
        if ns_idx >= t_idx { return None; }
        
        // Find /v1/ which precedes namespaces
        let v1_idx = uri.find("/v1")?;
        if v1_idx >= ns_idx { return None; }
        
        let base_url = uri[..v1_idx].to_string();
        
        // Extract optional prefix (e.g., /v1/warehouse/namespaces/...)
        let prefix_start = v1_idx + 3;
        let prefix_end = ns_idx;
        let prefix = if prefix_end > prefix_start {
            let p = &uri[prefix_start..prefix_end];
            let trimmed = p.trim_matches('/');
            if trimmed.is_empty() { None } else { Some(trimmed.to_string()) }
        } else {
            None
        };
        
        let namespace = uri[ns_idx + ns_marker.len() .. t_idx].to_string();
        let table_name = uri[t_idx + t_marker.len()..].to_string();
        
        Some((base_url, prefix, namespace, table_name))
    }

    async fn new_from_rest(base_url: String, prefix: Option<String>, namespace: String, table_name: String, rest_uri: &str) -> Result<Self> {
        use crate::core::catalog::rest::RestCatalogClient;
        use crate::core::catalog::Catalog;
        
        let client = RestCatalogClient::new(base_url, prefix);
        let metadata = client.load_table(&namespace, &table_name).await?;
        
        // Derive local native URI (Cache location for layered index)
        let cache_dir = std::env::var("HYPERSTREAM_CACHE_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("hyperstream_cache"));
        
        // Ensure directory exists
        if !cache_dir.exists() {
             let _ = std::fs::create_dir_all(&cache_dir);
        }
        
        let safe_name = rest_uri.replace("://", "_").replace("/", "_").replace(":", "_");
        let native_uri_path = cache_dir.join(safe_name);
        let native_uri = format!("file://{}", native_uri_path.display());
        
        let store = create_object_store(&native_uri)?;
        
        // Check if already registered
        let manager = ManifestManager::new(store.clone(), "", &native_uri);
        let (_, version) = manager.load_latest().await.unwrap_or((Manifest::default(), 0));
        
        if version > 0 {
            let mut table = TableBuilder::new(native_uri).with_index_all(false).build_async().await?;
            table.catalog = Some(Arc::new(client));
            table.catalog_namespace = Some(namespace);
            table.catalog_table_name = Some(table_name);
            Ok(table)
        } else {
            // Try to open the warehouse location as a HyperStreamDB native table first
            // (REST-created tables are stored as HyperStreamDB tables in the warehouse)
            tracing::debug!("Checking if warehouse location is a HyperStreamDB table: {}", metadata.location);
            
            match create_object_store(&metadata.location) {
                Ok(warehouse_store) => {
                    let warehouse_manager = ManifestManager::new(
                        warehouse_store,
                        "",
                        &metadata.location
                    );
                    
                    if let Ok((_, warehouse_version)) = warehouse_manager.load_latest().await {
                        if warehouse_version > 0 {
                            // It's a HyperStreamDB table in the warehouse, open it directly
                            tracing::info!("✅ Opening existing HyperStreamDB table from REST catalog: {}", metadata.location);
                            let mut table = TableBuilder::new(metadata.location).with_index_all(false).build_async().await?;
                            table.catalog = Some(Arc::new(client));
                            table.catalog_namespace = Some(namespace);
                            table.catalog_table_name = Some(table_name);
                            return Ok(table);
                        } else {
                            // Version=0, but this could be an empty newly-created table
                            // Create a new native table at the warehouse location using the metadata schema
                            tracing::info!("✅ Creating new HyperStreamDB table at warehouse location: {}", metadata.location);
                            let current_schema = metadata.schemas.iter()
                                .find(|s| s.schema_id == metadata.current_schema_id)
                                .or_else(|| metadata.schemas.last())
                                .ok_or_else(|| anyhow::anyhow!("No schema found in table metadata"))?;
                            let schema_ref = Arc::new(current_schema.to_arrow());
                            let table = Self::create_async(metadata.location.clone(), schema_ref).await?;
                            // Don't set catalog fields - this is a newly-created empty table and committing back
                            // to the catalog would fail. On next load, we'll detect the existing manifest and open normally.
                            return Ok(table);
                        }
                    } else {
                        tracing::debug!("ℹ️  Warehouse location has no manifest, trying external Iceberg import or creating new table");
                    }
                },
                Err(e) => {
                    tracing::debug!("ℹ️  Could not access warehouse location: {}. Trying external Iceberg import.", e);
                }
            }
            
            // If not a HyperStreamDB table, try to import as external Iceberg table
            tracing::info!("Auto-registering Iceberg table from REST catalog: {}", rest_uri);
            let mut table = Self::register_external(native_uri, &metadata.location).await?;

            table.catalog = Some(Arc::new(client));
            table.catalog_namespace = Some(namespace);
            table.catalog_table_name = Some(table_name);
            Ok(table)
        }
    }

    /// Start a background observer to watch an external Iceberg table for changes
    pub async fn spawn_iceberg_observer(&self, iceberg_metadata_uri: String, interval: std::time::Duration) -> Result<()> {
        let table = self.clone();
        
        let handle = tokio::spawn(async move {
            let mut last_processed_snapshot = None;
            
            // Try to set initial state from current manifest
            if let Ok((_, _)) = table.get_snapshot_segments_with_version().await {
                 // For now, we don't store the external snapshot id in our manifest explicitly,
                 // but we could. For simplicity, we just start fresh or process what's new.
            }

            loop {
                match table.check_and_import_new_snapshot(&iceberg_metadata_uri, &mut last_processed_snapshot).await {
                    Ok(true) => tracing::debug!("Snapshot Observer: New snapshot processed."),
                    Ok(false) => {}, // No new snapshot
                    Err(e) => tracing::error!("Snapshot Observer Error: {}", e),
                }
                tokio::time::sleep(interval).await;
            }
        });
        
        self.background_tasks.lock().await.push(handle);
        Ok(())
    }

    async fn check_and_import_new_snapshot(
        &self, 
        iceberg_metadata_uri: &str,
        last_snapshot_id: &mut Option<i64>
    ) -> Result<bool> {
        let meta_url = url::Url::parse(iceberg_metadata_uri).context("Invalid iceberg_metadata_uri")?;
        let meta_path_str = meta_url.path();
        let meta_path_obj = std::path::Path::new(meta_path_str);
        let parent_dir = meta_path_obj.parent().context("No parent directory")?;
        let filename = meta_path_obj.file_name().context("No filename")?.to_str().unwrap();
        
        let meta_store_uri = if iceberg_metadata_uri.starts_with("file://") {
            format!("file://{}", parent_dir.display())
        } else {
            let mut base_url = meta_url.clone();
            base_url.set_path(parent_dir.to_str().unwrap());
            base_url.to_string()
        };

        let iceberg_meta_store = create_object_store(&meta_store_uri)?;
        let path = object_store::path::Path::from(filename);
        let ret = iceberg_meta_store.get(&path).await?;
        let bytes = ret.bytes().await?;
        let iceberg_meta: crate::core::iceberg::IcebergTableMetadata = serde_json::from_slice(&bytes)?;
        
        if let Some(current_id) = iceberg_meta.current_snapshot_id {
            if Some(current_id) != *last_snapshot_id {
                self.import_iceberg_snapshot(current_id, &iceberg_meta, iceberg_meta_store).await?;
                *last_snapshot_id = Some(current_id);
                return Ok(true);
            }
        }
        
        Ok(false)
    }

    async fn import_iceberg_snapshot(
        &self, 
        snapshot_id: i64, 
        meta: &crate::core::iceberg::IcebergTableMetadata,
        iceberg_store: Arc<dyn ObjectStore>
    ) -> Result<()> {
        let snapshot = meta.snapshots.iter().find(|s| s.snapshot_id == snapshot_id)
            .context("Snapshot not found")?;
            
        tracing::info!("Importing Iceberg Snapshot {}...", snapshot_id);
        
        // Load Manifest List
        let ml_path_str = snapshot.manifest_list.trim_start_matches("file://");
        let ml_path = object_store::path::Path::from(ml_path_str);
        // If ml_path_str is absolute and store is relative, we might need more effort.
        // But for local files, object_store handles absolute paths if rooted correctly.
        // However, a better way is to make it relative to the store.
        
        let ret = iceberg_store.get(&ml_path).await?;
        let bytes = ret.bytes().await?;
        let manifest_list = crate::core::iceberg::read_manifest_list(&bytes[..])?;
        
        let mut all_entries = Vec::new();
        
        for ml_entry in manifest_list {
            // Load Manifest
            let m_path_str = ml_entry.manifest_path.trim_start_matches("file://");
            let m_path = object_store::path::Path::from(m_path_str);
            let ret = iceberg_store.get(&m_path).await?;
            let bytes = ret.bytes().await?;
            let manifest = crate::core::iceberg::read_manifest(&bytes[..])?;
            
            let iceberg_schema = crate::core::manifest::Schema {
                schema_id: meta.current_schema_id,
                fields: meta.schemas.iter()
                    .find(|s| s["schema-id"].as_i64().map(|id| id as i32) == Some(meta.current_schema_id))
                    .map(|s| {
                        s["fields"].as_array().unwrap_or(&Vec::new()).iter().map(|f| {
                            crate::core::manifest::SchemaField {
                                id: f["id"].as_i64().unwrap_or(0) as i32,
                                name: f["name"].as_str().unwrap_or("").to_string(),
                                type_str: f["type"].clone().to_string().replace("\"", ""),
                                 required: f["required"].as_bool().unwrap_or(false),
                                 fields: Vec::new(),
                                 initial_default: None,
                                 write_default: None,
                             }
                        }).collect()
                    }).unwrap_or_else(Vec::new),
                identifier_field_ids: Vec::new(),
            };

            let iceberg_spec = meta.partition_specs.iter()
                .find(|s| s["spec-id"].as_i64().map(|id| id as i32) == Some(meta.default_spec_id))
                .and_then(|s| crate::core::iceberg::iceberg_partition_spec_to_hyperstream(s).ok())
                .unwrap_or_default();

            let mut data_entries = Vec::new();
            let mut delete_files = Vec::new();

            for entry in manifest {
                 if entry.status != 2 { // Not deleted
                     match crate::core::iceberg::convert_iceberg_to_object(&entry, &iceberg_schema, &iceberg_spec) {
                         Ok(crate::core::iceberg::IcebergManifestObject::Data(me)) => {
                             data_entries.push(me);
                         }
                         Ok(crate::core::iceberg::IcebergManifestObject::Delete(df)) => {
                             delete_files.push(df);
                         }
                         Err(e) => tracing::warn!("Error converting Iceberg entry: {}", e),
                     }
                 }
            }

            // Associate deletes with data entries (simplification: same partition)
            for df in delete_files {
                for data in &mut data_entries {
                    if data.partition_values == df.partition_values {
                        data.delete_files.push(df.clone());
                    }
                }
            }
            all_entries.extend(data_entries);
        }
        
        // Commit as new manifest version in HyperStream
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager.commit_imported_entries(all_entries).await?;
        
        Ok(())
    }

    /// Create a new Table instance (Asynchronous)
    /// This is safe to call from within an async runtime.
    pub async fn new_async(uri: String) -> Result<Self> {
        TableBuilder::new(uri).with_index_all(false).build_async().await
    }
    
    /// Create a new table with an explicit schema (Asynchronous)
    /// This is safe to call from within an async runtime.
    pub async fn create_async(uri: String, schema: SchemaRef) -> Result<Self> {
        let store = create_object_store(&uri)?;
        let manifest_manager = ManifestManager::new(store.clone(), "", &uri);

        // Check if table already exists (has a manifest)
        let (_, version) = manifest_manager.load_latest().await?;
        if version > 0 {
            return Err(anyhow::anyhow!("Table already exists at {}", uri));
        }

        // Convert Arrow Schema to Manifest Schema
        let manifest_schema = crate::core::manifest::Schema::from_arrow(&schema, 1);
        let max_id = manifest_schema.fields.iter().map(|f| f.id).max().unwrap_or(0);

        // Initialize Manifest with Schema
        // This will create v1.json
        manifest_manager.update_schema(vec![manifest_schema.clone()], 1, Some(max_id)).await?;

        // Initialize Table Metadata (Iceberg v2 Spec)
        let mut metadata = TableMetadata::new(
            2, 
            uuid::Uuid::new_v4().to_string(), 
            uri.clone(), 
            manifest_schema, 
            PartitionSpec::default(), 
            SortOrder::default()
        );
        metadata.save_to_store(store.as_ref(), 1).await?;

        // Return a table instance pointing to the now-initialized location
        TableBuilder::new(uri).with_index_all(false).build_async().await
    }

    /// Create a new table with an explicit schema and partition specification (Asynchronous)
    pub async fn create_partitioned_async(
        uri: String, 
        schema: SchemaRef, 
        spec: crate::core::manifest::PartitionSpec
    ) -> Result<Self> {
        let store = create_object_store(&uri)?;
        let manifest_manager = ManifestManager::new(store.clone(), "", &uri);

        // Check if table already exists (has a manifest)
        let (_, version) = manifest_manager.load_latest().await?;
        if version > 0 {
            return Err(anyhow::anyhow!("Table already exists at {}", uri));
        }

        // Convert Arrow Schema to Manifest Schema
        let manifest_schema = crate::core::manifest::Schema::from_arrow(&schema, 1);

        // Create Genesis Manifest with Spec
        let genesis_manifest = crate::core::manifest::Manifest::new_with_spec(
            1, 
            vec![], 
            None, 
            vec![manifest_schema.clone()], 
            1, 
            spec.clone()
        );

        manifest_manager.commit_manifest(genesis_manifest).await?;

        // Initialize Table Metadata (Iceberg v2 Spec)
        let mut metadata = TableMetadata::new(
            2, 
            uuid::Uuid::new_v4().to_string(), 
            uri.clone(), 
            manifest_schema, 
            spec, 
            SortOrder::default()
        );
        metadata.save_to_store(store.as_ref(), 1).await?;

        // Return a table instance pointing to the now-initialized location
        TableBuilder::new(uri).with_index_all(false).build_async().await
    }

    pub fn create(uri: String, schema: SchemaRef) -> Result<Self> {
        let rt = Arc::new(Runtime::new()?);
        let mut table = rt.clone().block_on(Self::create_async(uri, schema))?;
        table.rt = Some(rt);
        Ok(table)
    }


    /// Wait for all background tasks to complete
    pub async fn wait_for_background_tasks_async(&self) -> Result<()> {
        let mut tasks = {
            let mut t = self.background_tasks.lock().await;
            std::mem::take(&mut *t)
        };
        
        for task in tasks.drain(..) {
            task.await.map_err(|e| anyhow::anyhow!("Background task failed: {}", e))?;
        }
        Ok(())
    }

    pub fn wait_for_background_tasks(&self) -> Result<()> {
        if let Some(ref rt) = self.rt {
            rt.block_on(self.wait_for_background_tasks_async())
        } else {
            // Fallback for async-only instances: they should use wait_for_background_tasks_async directly
            anyhow::bail!("No runtime configured for Table to wait for background tasks synchronously")
        }
    }

    #[cfg(feature = "enterprise")]
    pub fn enable_enterprise(&mut self, license_key: String) -> Result<()> {
        crate::enterprise::license::validate_license(&license_key)?;
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

    /// Replace the table's sort order (Iceberg V2/V3 evolution)
    /// 
    /// # Arguments
    /// * `columns` - Column names to sort by
    /// * `ascending` - Whether each column should be sorted in ascending order
    /// 
    /// # Example
    /// ```no_run
    /// # use hyperstreamdb::core::table::Table;
    /// # fn example(table: &Table) -> anyhow::Result<()> {
    /// table.replace_sort_order(&["timestamp", "user_id"], &[false, true])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn replace_sort_order(&self, columns: &[&str], ascending: &[bool]) -> Result<()> {
        if columns.len() != ascending.len() {
            anyhow::bail!("columns and ascending arrays must have same length");
        }
        
        let fields: Vec<SortField> = columns.iter().zip(ascending.iter())
            .enumerate()
            .map(|(i, (_col, asc))| SortField {
                source_id: i as i32 + 1, // Field IDs typically start at 1
                transform: "identity".to_string(),
                direction: if *asc { SortDirection::Asc } else { SortDirection::Desc },
                null_order: if *asc { NullOrder::NullsFirst } else { NullOrder::NullsLast },
            })
            .collect();
        
        // Store column names for lookup during apply
        let order = SortOrder {
            order_id: 1,
            fields,
        };
        
        let mut guard = self.sort_order.write().unwrap();
        *guard = Some(order);
        
        // Also store column names for apply_sort_order
        self.sort_order_columns.write().unwrap().replace(columns.iter().map(|s| s.to_string()).collect());
        
        Ok(())
    }
    
    /// Get the current sort order
    pub fn get_sort_order(&self) -> Option<SortOrder> {
        self.sort_order.read().unwrap().clone()
    }

    /// Apply sort order to a RecordBatch before writing
    fn apply_sort_order(&self, batch: &RecordBatch) -> Result<RecordBatch> {
        let guard = self.sort_order.read().unwrap();
        let order = match guard.as_ref() {
            Some(o) if !o.fields.is_empty() => o,
            _ => return Ok(batch.clone()), // No sort order configured
        };
        
        let columns_guard = self.sort_order_columns.read().unwrap();
        let column_names = match columns_guard.as_ref() {
            Some(names) => names,
            None => return Ok(batch.clone()),
        };
        
        // Build sort columns
        let mut sort_columns = Vec::new();
        for (field, col_name) in order.fields.iter().zip(column_names.iter()) {
            if let Some((idx, _)) = batch.schema().column_with_name(col_name) {
                let column = batch.column(idx).clone();
                let options = arrow::compute::SortOptions {
                    descending: matches!(field.direction, SortDirection::Desc),
                    nulls_first: matches!(field.null_order, NullOrder::NullsFirst),
                };
                sort_columns.push(arrow::compute::SortColumn { values: column, options: Some(options) });
            }
        }
        
        if sort_columns.is_empty() {
            return Ok(batch.clone());
        }
        
        // Compute sorted indices
        let indices = arrow::compute::lexsort_to_indices(&sort_columns, None)?;
        
        // Apply sort to all columns
        let sorted_columns: Vec<Arc<dyn arrow::array::Array>> = batch.columns()
            .iter()
            .map(|col| arrow::compute::take(col.as_ref(), &indices, None))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        
        Ok(RecordBatch::try_new(batch.schema(), sorted_columns)?)
    }

    /// Check if schema has V3 metadata columns (_row_id, _last_updated_sequence_number)
    #[allow(dead_code)]
    fn has_v3_metadata_columns(schema: &arrow::datatypes::SchemaRef) -> bool {
        schema.column_with_name("_row_id").is_some() 
            && schema.column_with_name("_last_updated_sequence_number").is_some()
    }

    /// Add V3 metadata columns to a RecordBatch
    /// Generates UUID v7 for _row_id and uses current sequence number
    fn add_v3_metadata_columns(&self, batch: &RecordBatch, sequence_number: i64) -> Result<RecordBatch> {
        use arrow::array::{StringArray, Int64Array};
        use arrow::datatypes::{Field, DataType};
        
        let num_rows = batch.num_rows();
        
        // Generate UUID v4 for each row (random UUIDs)
        let row_ids: Vec<String> = (0..num_rows)
            .map(|_| uuid::Uuid::new_v4().to_string())
            .collect();
        let row_id_array = Arc::new(StringArray::from(row_ids));
        
        // Set sequence number for all rows
        let seq_numbers = vec![sequence_number; num_rows];
        let seq_array = Arc::new(Int64Array::from(seq_numbers));
        
        // Build new schema with metadata columns
        let mut new_fields: Vec<Arc<Field>> = batch.schema().fields().iter().cloned().collect();
        new_fields.push(Arc::new(Field::new("_row_id", DataType::Utf8, false)));
        new_fields.push(Arc::new(Field::new("_last_updated_sequence_number", DataType::Int64, false)));
        let new_schema = Arc::new(arrow::datatypes::Schema::new(new_fields));
        
        // Build new columns
        let mut new_columns: Vec<Arc<dyn arrow::array::Array>> = batch.columns().to_vec();
        new_columns.push(row_id_array);
        new_columns.push(seq_array);
        
        Ok(RecordBatch::try_new(new_schema, new_columns)?)
    }

    async fn load_initial_schema(store: Arc<dyn ObjectStore>, _uri: &str) -> SchemaRef {
        let manifest_manager = ManifestManager::new(store.clone(), "", _uri);
        if let Ok((manifest, version)) = manifest_manager.load_latest().await {
            if version > 0 && !manifest.schemas.is_empty() {
                // Use the latest schema from manifest
                if let Some(latest) = manifest.schemas.iter().find(|s| s.schema_id == manifest.current_schema_id) {
                    return Arc::new(latest.to_arrow());
                } else if let Some(latest) = manifest.schemas.last() {
                    return Arc::new(latest.to_arrow());
                }
            }
            
            // If manifest exists but entries exist without schemas (Legacy)
            if !manifest.entries.is_empty() {
                // Try reading from the first entry's file
                if let Some(entry) = manifest.entries.first() {
                    let file_path = &entry.file_path;
                    let parts: Vec<&str> = file_path.split('/').collect();
                    let filename = parts.last().unwrap_or(&"wrapper");
                    let segment_id = filename.replace(".parquet", "");
                    let config = SegmentConfig::new("", &segment_id);
                    let reader = HybridReader::new(config, store.clone(), _uri);
                    if let Ok(mut s) = reader.stream_all(None as Option<arrow::datatypes::SchemaRef>).await {
                        use futures::StreamExt;
                        if let Some(Ok(batch)) = s.next().await {
                            return batch.schema();
                        }
                    }
                }
            }
        }

        // Deep fallback: Manual scan (original logic)
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
             let filename = path.split('/').next_back().unwrap();
             let segment_id = filename.replace(".parquet", "");
             let config = SegmentConfig::new("", &segment_id); 
             let reader = HybridReader::new(config, store.clone(), _uri);
             if let Ok(mut s) = reader.stream_all(None as Option<arrow::datatypes::SchemaRef>).await {
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

    pub fn add_index_columns(&mut self, columns: Vec<String>, device: Option<String>) -> Result<()> {
        {
            let mut index_cols = self.index_columns.write().unwrap();
            let mut index_configs = self.index_configs.write().unwrap();
            
            for col in &columns {
                if !index_cols.contains(col) {
                    index_cols.push(col.clone());
                }
                index_configs.insert(col.clone(), ColumnIndexConfig { device: device.clone(), enabled: true, tokenizer: None });
            }
            index_cols.sort();
            index_cols.dedup();
        }
        self.backfill_indexes(columns)
    }

    pub async fn add_index_columns_async(&mut self, columns: Vec<String>, device: Option<String>) -> Result<()> {
        {
            let mut index_cols = self.index_columns.write().unwrap();
            let mut index_configs = self.index_configs.write().unwrap();
            
            for col in &columns {
                if !index_cols.contains(col) {
                    index_cols.push(col.clone());
                }
                index_configs.insert(col.clone(), ColumnIndexConfig { device: device.clone(), enabled: true, tokenizer: None });
            }
            index_cols.sort();
            index_cols.dedup();
        }
        self.backfill_indexes_async(columns).await
    }

    pub fn set_default_device(&mut self, device: Option<String>) {
        let mut d = self.default_device.write().unwrap();
        *d = device;
    }

    pub fn get_default_device(&self) -> Option<String> {
        self.default_device.read().unwrap().clone()
    }

    pub fn remove_index_columns(&mut self, columns: Vec<String>) {
        let mut index_cols = self.index_columns.write().unwrap();
        index_cols.retain(|c| !columns.contains(c));
    }

    pub fn remove_all_index_columns(&mut self) {
        let mut index_cols = self.index_columns.write().unwrap();
        index_cols.clear();
        self.index_all = false;
    }

    pub fn index_all_columns(&mut self) -> Result<()> {
        self.index_all = true;
        self.backfill_indexes(Vec::new()) 
    }

    pub async fn index_all_columns_async(&mut self) -> Result<()> {
        self.index_all = true;
        self.backfill_indexes_async(Vec::new()).await
    }

    pub fn get_index_columns(&self) -> Vec<String> {
        self.index_columns.read().unwrap().clone()
    }

    pub fn runtime(&self) -> Arc<Runtime> {
        self.rt.as_ref().expect("Runtime not available on async Table").clone()
    }

    pub fn arrow_schema(&self) -> SchemaRef {
        self.schema.read().unwrap().clone()
    }

    pub async fn manifest(&self) -> Result<crate::core::manifest::Manifest> {
        let (manifest, _) = crate::core::manifest::ManifestManager::new(
            self.store.clone(),
            "",
            &self.uri,
        ).load_latest().await?;
        Ok(manifest)
    }

    fn backfill_indexes(&self, target_columns: Vec<String>) -> Result<()> {
        self.runtime().block_on(self.backfill_indexes_async(target_columns))
    }

    async fn backfill_indexes_async(&self, target_columns: Vec<String>) -> Result<()> {
        use futures::StreamExt;
        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, all_entries, _) = manager.load_latest_full().await?;
        
        if all_entries.is_empty() {
            return Ok(());
        }

        let entries_results: Vec<Result<ManifestEntry>> = futures::future::join_all(all_entries.iter().map(|entry| {
            let entry = entry.clone();
            let table_uri = self.uri.clone();
            let store = self.store.clone();
            let data_store = self.data_store.clone().unwrap_or(self.store.clone());
            let target_cols = target_columns.clone();
            
            async move {
                let mut current_entry = entry.clone();
                let file_path_str = current_entry.file_path.clone();
                let segment_id = file_path_str.split('/').next_back().unwrap_or(&file_path_str)
                    .strip_suffix(".parquet").unwrap_or(&file_path_str);

                let mut cols_to_index = self.index_columns.read().unwrap().clone();
                for col in target_cols {
                    if !cols_to_index.contains(&col) {
                        cols_to_index.push(col);
                    }
                }

                let config = SegmentConfig::new(&table_uri, segment_id)
                    .with_parquet_path(current_entry.file_path.clone())
                    .with_data_store(data_store)
                    .with_index_all(self.index_all)
                    .with_columns_to_index(cols_to_index);
                
                let reader = HybridReader::new(config.clone(), store.clone(), &table_uri);
                let mut writer = HybridSegmentWriter::new(config);
                writer.primary_key = self.primary_key.read().unwrap().clone();
                writer.set_store(store.clone());
                
                let mut stream = reader.stream_all(None as Option<arrow::datatypes::SchemaRef>).await?;
                while let Some(batch) = stream.next().await {
                    let batch = batch?;
                    writer.write_batch(&batch)?;
                    writer.build_indexes(&batch)?;
                }
                
                writer.upload_to_store().await?;
                
                // Invalidate the cache for this segment
                let cache_key = format!("{}/{}", table_uri, current_entry.file_path);
                crate::core::cache::PARQUET_META_CACHE.invalidate(&cache_key).await;
                
                let updated_entry = writer.to_manifest_entry();
                current_entry.index_files = updated_entry.index_files;
                
                Ok(current_entry)
            }
        })).await;

        let mut updated_entries = Vec::new();
        for res in entries_results {
            updated_entries.push(res?);
        }

        if !updated_entries.is_empty() {
            manager.commit_imported_entries(updated_entries).await?;
        }

        Ok(())
    }

    // Read operations moved to read.rs

    // Write operations moved to write.rs

    /// Rewrite data files to optimize snapshots (Compaction)
    pub fn rewrite_data_files(&self, options: Option<CompactionOptions>) -> Result<()> {
        self.runtime().block_on(self.rewrite_data_files_async(options))
    }

    /// Legacy alias for rewrite_data_files
    pub fn compact(&self, options: Option<CompactionOptions>) -> Result<()> {
        self.rewrite_data_files(options)
    }

    /// Rewrite data files (Asynchronous)
    pub async fn rewrite_data_files_async(&self, options: Option<CompactionOptions>) -> Result<()> {
        // Flush before compaction to include recent writes
        self.flush_async().await?;
        
        let opts = options.unwrap_or_default();
        let compactor = Compactor::new(&self.uri, opts)?;
        compactor.rewrite_data_files().await
    }

    /// Update the table schema (Evolution)
    pub async fn update_schema(&self, new_schema: crate::core::manifest::Schema) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, _, _) = manifest_manager.load_latest_full().await?;
        
        let new_schema_id = manifest.schemas.iter()
            .map(|s| s.schema_id)
            .max()
            .unwrap_or(0) + 1;
            
        let mut new_schemas = manifest.schemas.clone();
        let mut schema_to_add = new_schema.clone();
        schema_to_add.schema_id = new_schema_id;
        new_schemas.push(schema_to_add);

        let max_id = new_schema.fields.iter().map(|f| f.id).max().unwrap_or(0);
        
        manifest_manager.update_schema(new_schemas, new_schema_id, Some(max_id)).await?;
        
        // Reload local schema
        let mut local_schema = self.schema.write().unwrap();
        *local_schema = Arc::new(new_schema.to_arrow());
        
        Ok(())
    }

    /// Rollback table to a specific snapshot ID
    pub async fn rollback_to_snapshot(&self, snapshot_id: i64) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager.rollback_to_snapshot(snapshot_id as u64).await?;
        Ok(())
    }

    /// Physically delete unreferenced data and manifest files
    pub fn vacuum(&self, retention_versions: usize) -> Result<usize> {
        self.runtime().block_on(self.vacuum_async(retention_versions))
    }

    /// Async implementation of vacuum
    pub async fn vacuum_async(&self, retention_versions: usize) -> Result<usize> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager.vacuum(retention_versions).await
    }

    /// Async implementation of delete
    pub async fn delete_async(&self, filter: &str) -> Result<()> {
        use futures::StreamExt;
        
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, all_entries, _) = manifest_manager.load_latest_full().await?;
        
        if all_entries.is_empty() {
            return Ok(());
        }

        let planner = QueryPlanner::new();
        let arrow_schema = self.arrow_schema();
        let expr = FilterExpr::parse_sql(filter, arrow_schema).await.context("Failed to parse delete filter")?;
        
        let candidates = planner.prune_entries(&all_entries, Some(&expr), None);
        let candidate_paths: std::collections::HashSet<String> = candidates.iter().map(|(e, _)| e.file_path.clone()).collect();
        tracing::debug!("delete_async: filter='{}', potential candidates: {}", filter, candidates.len());
        
        let mut all_updated_entries = Vec::new();

        for entry in all_entries {
            if !candidate_paths.contains(&entry.file_path) {
                // Preserve non-candidate segments as-is
                all_updated_entries.push(entry);
                continue;
            }

            tracing::debug!("delete_async: processing segment {}", entry.file_path);
            let file_path_str = entry.file_path.clone();
            
            // Fix path resolution: find correct physical subdirectory
            let path = std::path::Path::new(&file_path_str);
            let rel_parent = path.parent().and_then(|p| p.to_str()).unwrap_or("");
            let full_base_path = if rel_parent.is_empty() {
                self.uri.clone()
            } else {
                format!("{}/{}", self.uri, rel_parent)
            };

            let segment_id = file_path_str.split('/').next_back().unwrap_or(&file_path_str)
                .strip_suffix(".parquet").unwrap_or(&file_path_str);
            
            let config = SegmentConfig::new(&full_base_path, segment_id)
                .with_index_files(entry.index_files.clone())
                .with_record_count(entry.record_count as u64);
            let reader = HybridReader::new(config, self.store.clone(), &self.uri);
            
            let mut new_deletes = Vec::new();
            
            // OPTIMIZATION: Check for Sidecar Scalar Index (Inverted Index)
            let and_filters = expr.extract_and_conditions();
            let mut bitmap_opt: Option<roaring::RoaringBitmap> = None;

            if !and_filters.is_empty() {
                for filter in and_filters {
                    if let Ok(Some(bm)) = reader.get_scalar_filter_bitmap(&filter).await {
                        if let Some(current) = bitmap_opt {
                            bitmap_opt = Some(current & bm);
                        } else {
                            bitmap_opt = Some(bm);
                        }
                    } else {
                        // If any part of the AND is MISSING an index, we might need a scan
                        // or we just skip this optimization.
                        bitmap_opt = None;
                        break;
                    }
                }
            }

            if let Some(bitmap) = bitmap_opt {
                // Index HIT! We found the deleted rows instantly.
                for row_id in bitmap.iter() {
                    new_deletes.push(row_id as i64);
                }
            } else {
                // Index MISS: Full file scan (fallback)
                let mut stream = reader.stream_all(None).await?;
                let mut current_row_offset = 0;
                while let Some(batch_res) = stream.next().await {
                    let batch = batch_res?;
                    let num_rows = batch.num_rows();
                    let mask = planner.evaluate_expr(&batch, &expr)?;
                    for i in 0..num_rows {
                        if mask.value(i) {
                            new_deletes.push((current_row_offset + i) as i64);
                        }
                    }
                    current_row_offset += num_rows;
                }
            }
            
            if !new_deletes.is_empty() {
                // Generate NEW Position Delete File
                let mut file_paths = arrow::array::StringBuilder::new();
                let mut positions = arrow::array::Int64Builder::new();
                
                for &pos in &new_deletes {
                    file_paths.append_value(&entry.file_path);
                    positions.append_value(pos);
                }
                
                let file_path_array = file_paths.finish();
                let pos_array = positions.finish();
                
                let delete_writer = crate::core::iceberg::iceberg_delete::IcebergDeleteWriter::new(
                    self.uri.clone(),
                    2 // Format V2
                );

                let partition_data = if !entry.partition_values.is_empty() {
                    let path = std::path::Path::new(&entry.file_path);
                    let rel_path = path.parent()
                        .and_then(|p| p.to_str())
                        .unwrap_or("")
                        .trim_start_matches('/')
                        .to_string();
                        
                    Some((rel_path, entry.partition_values.clone()))
                } else {
                    None
                };

                let delete_file = delete_writer.write_position_delete(
                    partition_data, 
                    &file_path_array,
                    &pos_array
                ).await?;
                
                let mut new_entry = entry.clone();
                new_entry.delete_files.push(delete_file);
                all_updated_entries.push(new_entry);
            } else {
                all_updated_entries.push(entry.clone());
            }
        }

        if !all_updated_entries.is_empty() {
            // Commit the entire updated state. 
            // In HyperStreamDB manifest management, this acts as an atomic swap of the segment list.
            manifest_manager.commit(&all_updated_entries, &[], crate::core::manifest::CommitMetadata::default()).await?;
        }

        Ok(())
    }

    /// Check if any keys in the batch already exist in the table (Primary Key Enforcement)
    async fn check_primary_key_uniqueness_async(&self, batch: &RecordBatch, columns: &[String]) -> Result<()> {
        
        if batch.num_rows() == 0 { return Ok(()); }
        
        let schema = batch.schema();
        let col_indices: Vec<usize> = columns.iter()
            .map(|c| schema.index_of(c))
            .collect::<Result<Vec<usize>, _>>()?;

        // OPTIMIZATION: Use IN clause for batches (efficient via Inverted Index)
        // For now, we take the first row as a sample check to avoid huge expression generation 
        // until we have a proper Row-Value In-List implementation.
        for i in 0..batch.num_rows().min(100) { // Limit samples for performance in MVP
            let mut filters_str_vec = Vec::new();
            for (col_name, col_idx) in columns.iter().zip(col_indices.iter()) {
                let col = batch.column(*col_idx);
                let val = if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int32Array>() {
                    format!("{}", arr.value(i))
                } else if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int64Array>() {
                    format!("{}", arr.value(i))
                } else if let Some(arr) = col.as_any().downcast_ref::<arrow::array::StringArray>() {
                    format!("'{}'", arr.value(i).replace("'", "''"))
                } else {
                    continue;
                };
                filters_str_vec.push(format!("{} = {}", col_name, val));
            }
            
            if !filters_str_vec.is_empty() {
                let filter_str = filters_str_vec.join(" AND ");
                let expr = FilterExpr::parse_sql(&filter_str, self.arrow_schema()).await?;
                
                // Check manifests
                let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
                let (_, all_entries, _) = manifest_manager.load_latest_full().await?;
                let planner = QueryPlanner::new();
                let candidates = planner.prune_entries(&all_entries, Some(&expr), None);
                
                if !candidates.is_empty() {
                    // Refine search within candidates (Index lookup)
                    for (entry, _) in candidates {
                        // Resolve partition-aware path for PK lookup
                        let path = std::path::Path::new(&entry.file_path);
                        let rel_parent = path.parent().and_then(|p| p.to_str()).unwrap_or("");
                        let full_base_path = if rel_parent.is_empty() {
                             self.uri.clone()
                        } else {
                             format!("{}/{}", self.uri, rel_parent)
                        };
                        
                        let seg_id = entry.file_path.split('/').next_back().unwrap_or(&entry.file_path)
                            .replace(".parquet", "");

                        let config = SegmentConfig::new(&full_base_path, &seg_id)
                            .with_index_files(entry.index_files.clone())
                            .with_delete_files(entry.delete_files.clone());
                        let reader = HybridReader::new(config, self.store.clone(), &self.uri);
                        
                        let filters = expr.extract_and_conditions();
                        let mut bitmap_opt: Option<roaring::RoaringBitmap> = None;
                        
                        for f in filters {
                            if let Ok(Some(bm)) = reader.get_scalar_filter_bitmap(&f).await {
                                // Subtract logically deleted rows!
                                let deleted = reader.load_merged_deletes().await?;
                                let alive_bm = bm.clone() - deleted.clone();
                                
                                tracing::debug!("PK Check for {}: Index bits: {}, Deleted bits: {}, Alive bits: {}", 
                                    f.column, bm.len(), deleted.len(), alive_bm.len());
                                
                                if let Some(current) = bitmap_opt {
                                    bitmap_opt = Some(current & alive_bm);
                                } else {
                                    bitmap_opt = Some(alive_bm);
                                }
                            } else {
                                bitmap_opt = None;
                                break;
                            }
                        }

                        if let Some(bm) = bitmap_opt {
                            if !bm.is_empty() {
                                let pk_val = columns.iter().zip(filters_str_vec.iter())
                                    .map(|(c, f)| format!("{}={}", c, f))
                                    .collect::<Vec<_>>().join(", ");
                                return Err(anyhow::anyhow!("Duplicate primary key error: {} already exists", pk_val));
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    /// Synchronize PK columns from Iceberg schema identifier-field-ids (Internal Async)
    async fn sync_primary_key_from_schema_async(&self) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let latest_manifest = manifest_manager.load_latest_full().await?.0;
        
        if let Some(schema) = latest_manifest.schemas.last() {
            let pk_cols: Vec<String> = schema.identifier_field_ids.iter().map(|id| {
                schema.fields.iter().find(|f| f.id == *id).map(|f| f.name.clone()).unwrap_or_default()
            }).filter(|n| !n.is_empty()).collect();
            
            if !pk_cols.is_empty() {
                self.set_primary_key(pk_cols);
            }
        }
        Ok(())
    }

    /// Synchronize PK columns (Public Sync)
    pub fn sync_primary_key_from_schema(&self) -> Result<()> {
        self.runtime().block_on(self.sync_primary_key_from_schema_async())
    }

    /// Merge (Upsert) batches into the table
    pub fn merge(&self, batches: Vec<RecordBatch>, key_column: &str, mode: MergeMode) -> Result<()> {
        match mode {
            MergeMode::MergeOnRead => self.merge_on_read(batches, key_column),
            MergeMode::MergeOnWrite => self.merge_on_write(batches, key_column),
        }
    }

    fn merge_on_read(&self, batches: Vec<RecordBatch>, key_column: &str) -> Result<()> {
        let key_cols: Vec<&str> = key_column.split(',').collect();
        self.runtime().block_on(async {
            for batch in &batches {
                let schema = batch.schema();
                let col_indices: Vec<usize> = key_cols.iter()
                    .map(|&c| schema.index_of(c))
                    .collect::<Result<Vec<usize>, _>>()?;
                
                for i in 0..batch.num_rows() {
                    let mut filters = Vec::new();
                    for (&col_name, &col_idx) in key_cols.iter().zip(col_indices.iter()) {
                        let col = batch.column(col_idx);
                        let val = if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int32Array>() {
                            format!("{}", arr.value(i))
                        } else if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int64Array>() {
                            format!("{}", arr.value(i))
                        } else if let Some(arr) = col.as_any().downcast_ref::<arrow::array::StringArray>() {
                            format!("'{}'", arr.value(i))
                        } else {
                            // Fallback for other types
                            "".to_string()
                        };
                        
                        if !val.is_empty() {
                            filters.push(format!("{} = {}", col_name, val));
                        }
                    }
                    
                    if !filters.is_empty() {
                        let filter_expr = filters.join(" AND ");
                        self.delete_async(&filter_expr).await?;
                    }
                }
            }
            
            // Step 2: Write the new data (Append)
            self.write_async(batches).await?;
            Ok(())
        })
    }

    fn merge_on_write(&self, batches: Vec<RecordBatch>, key_column: &str) -> Result<()> {
        // MoW uses MergePlanner to rewrite segments
        use crate::core::merge::MergePlanner;
        
        if batches.is_empty() { return Ok(()); }
        
        let schema = batches[0].schema();
        let source_batch = arrow::compute::concat_batches(&schema, &batches)?;
        
        // Extract keys as JSON values for MergePlanner
        let mut source_keys = Vec::new();
        let col_idx = schema.index_of(key_column)?;
        let col = source_batch.column(col_idx);
        
        if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int32Array>() {
            for i in 0..arr.len() { source_keys.push(Value::Number(arr.value(i).into())); }
        } else if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int64Array>() {
            for i in 0..arr.len() { source_keys.push(Value::Number(arr.value(i).into())); }
        } else {
            return Err(anyhow::anyhow!("MoW currently only supports integer keys"));
        }

        self.runtime().block_on(async {
            
            let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
            let (_manifest, all_entries, _) = manifest_manager.load_latest_full().await?;
            
            let segment_ids: Vec<String> = all_entries.iter().map(|e| {
                e.file_path.split('/').next_back().unwrap().replace(".parquet", "")
            }).collect();
            
            let planner = MergePlanner::new();
            let commit_actions = planner.execute_merge(
                &self.uri,
                key_column,
                &source_keys,
                &source_batch,
                &segment_ids,
                |_, _| Ok(None) // index_provider
            )?;
            
            let mut new_entries = Vec::new();
            let mut removed_paths = Vec::new();
            
            // Note: MergePlanner returns Segment IDs. We need ManifestEntries.
            // This part is a bit tricky because MergePlanner is legacy.
            // For now, let's assume we can reconstruct basic entries.
            // A better way would be to have MergePlanner work with ManifestEntries.
            
            for (old_seg, new_seg) in commit_actions {
                if let Some(old) = old_seg {
                    removed_paths.push(format!("{}.parquet", old));
                }
                
                // Create a dummy entry for the new segment (will be updated by compactor later)
                // In a perfect system, writer.to_manifest_entry() should be used.
                let entry = ManifestEntry {
                    file_path: format!("{}.parquet", new_seg),
                    ..Default::default()
                };
                new_entries.push(entry);
            }
            
            manifest_manager.commit(&new_entries, &removed_paths, crate::core::manifest::CommitMetadata::default()).await?;
            Ok(())
        })
    }

    /// Remove orphan files
    pub fn remove_orphan_files(&self, older_than_days: u64) -> Result<()> {
        self.runtime().block_on(async {
            let maintenance = Maintenance::new(&self.uri)?;
            let older_than_ms = older_than_days * 24 * 60 * 60 * 1000;
            maintenance.remove_orphan_files(older_than_ms as i64).await
        })
    }
}

/// Merge strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeMode {
    MergeOnRead,
    MergeOnWrite,
}

pub use crate::core::planner::VectorSearchParams;

// ============================================================================
// Connector APIs for Spark/Trino Integration
// ============================================================================

/// Information about a data file (for Spark/Trino file-level parallelism)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataFileInfo {
    pub file_path: String,
    pub row_count: u64,
    pub file_size_bytes: u64,
    pub min_values: std::collections::HashMap<String, String>,
    pub max_values: std::collections::HashMap<String, String>,
    
    // Index metadata
    pub has_scalar_indexes: bool,
    pub has_vector_indexes: bool,
    pub indexed_columns: Vec<String>,
}

/// Split information (for Trino split-level parallelism)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Split {
    pub file_path: String,
    pub start_offset: u64,
    pub length: u64,
    pub row_group_ids: Vec<usize>,
    
    // Index metadata
    pub index_file_path: Option<String>,
    pub can_use_indexes: bool,
}

/// Table-level statistics (for query planning)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableStatistics {
    pub row_count: u64,
    pub file_count: usize,
    pub total_size_bytes: u64,
    
    // Index coverage
    pub index_coverage: IndexCoverage,
}

/// Index coverage information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexCoverage {
    pub scalar_indexed_columns: Vec<String>,
    pub vector_indexed_columns: Vec<String>,
    pub inverted_indexed_columns: Vec<String>,
    pub total_index_size_bytes: u64,
}

impl Table {
    /// List all data files in the table (with index metadata)
    pub fn list_data_files(&self) -> Result<Vec<DataFileInfo>> {
        self.runtime().block_on(async {
            let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
            let (_manifest, all_entries, _) = manifest_manager.load_latest_full().await?;
            
            let mut results = Vec::new();

            for entry in all_entries {
                let mut min_values = std::collections::HashMap::new();
                let mut max_values = std::collections::HashMap::new();
                
                for (col, stats) in &entry.column_stats {
                    if let Some(min) = &stats.min {
                        min_values.insert(col.clone(), min.to_string());
                    }
                    if let Some(max) = &stats.max {
                        max_values.insert(col.clone(), max.to_string());
                    }
                }

                let has_scalar_indexes = entry.index_files.iter().any(|f| f.index_type == "scalar" || f.index_type == "inverted");
                let has_vector_indexes = entry.index_files.iter().any(|f| f.index_type == "vector" || f.index_type == "hnsw");
                
                let indexed_columns = entry.index_files.iter()
                    .filter_map(|f| f.column_name.clone())
                    .collect();

                // Ensure absolute path if possible. 
                // Using entry.file_path directly as it's usually relative to table root
                // or whatever was written. 
                // Connectors might need full path. 
                // For now, construct full path if it doesn't look like one.
                let file_path = if entry.file_path.contains("://") {
                    entry.file_path.clone()
                } else {
                    // Try to join with uri
                    // simple heuristic
                    let base = self.uri.trim_end_matches('/');
                    let relative = entry.file_path.trim_start_matches('/');
                    format!("{}/{}", base, relative)
                };

                results.push(DataFileInfo {
                    file_path,
                    row_count: entry.record_count as u64,
                    file_size_bytes: entry.file_size_bytes as u64,
                    min_values,
                    max_values,
                    has_scalar_indexes,
                    has_vector_indexes,
                    indexed_columns,
                });
            }
            Ok(results)
        })
    }

    /// Get splits for parallel reading (index-aware)
    pub fn get_splits(&self, max_split_size: usize) -> Result<Vec<Split>> {
        let files = self.list_data_files()?;
        let mut splits = Vec::new();

        for file in files {
             // For PoC, find first index file if any
             // Ideally we pass specific index file for the column needed.
             // But Split struct has generic index_file_path.
             // We'll leave it None or find via Manifest inspection if we had it here.
             // But we lost ManifestEntry context. 
             // Ideally list_data_files should return more info, OR we do it here.
             // For now, simpler approach:
            
             if file.file_size_bytes > max_split_size as u64 {
                let num_splits = (file.file_size_bytes / max_split_size as u64) + 1;
                for i in 0..num_splits {
                    splits.push(Split {
                        file_path: file.file_path.clone(),
                        start_offset: i * max_split_size as u64,
                        length: max_split_size as u64,
                        row_group_ids: vec![i as usize],
                        index_file_path: file.indexed_columns.first().map(|_| {
                            file.file_path.replace(".parquet", "")
                        }),
                        can_use_indexes: file.has_scalar_indexes || file.has_vector_indexes,
                    });
                }
            } else {
                splits.push(Split {
                    file_path: file.file_path.clone(),
                    start_offset: 0,
                    length: file.file_size_bytes,
                    row_group_ids: vec![0],
                    index_file_path: file.indexed_columns.first().map(|_| {
                        file.file_path.replace(".parquet", "")
                    }),
                    can_use_indexes: file.has_scalar_indexes || file.has_vector_indexes,
                });
            }
        }
        Ok(splits)
    }
    /// List all data files in the table (Async)
    pub async fn list_data_files_async(&self) -> Result<Vec<DataFileInfo>> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, all_entries, _) = manifest_manager.load_latest_full().await?;

        let mut results = Vec::new();
        for entry in all_entries {
            let mut min_values = std::collections::HashMap::new();
            let mut max_values = std::collections::HashMap::new();
            
            for (col, stats) in &entry.column_stats {
                if let Some(min) = &stats.min {
                    min_values.insert(col.clone(), min.to_string());
                }
                if let Some(max) = &stats.max {
                    max_values.insert(col.clone(), max.to_string());
                }
            }

            let has_scalar_indexes = entry.index_files.iter().any(|f| f.index_type == "scalar" || f.index_type == "inverted");
            let has_vector_indexes = entry.index_files.iter().any(|f| f.index_type == "vector" || f.index_type == "hnsw");
            
            let indexed_columns = entry.index_files.iter()
                .filter_map(|f| f.column_name.clone())
                .collect();

            let file_path = if entry.file_path.contains("://") {
                entry.file_path.clone()
            } else {
                let base = self.uri.trim_end_matches('/');
                let relative = entry.file_path.trim_start_matches('/');
                format!("{}/{}", base, relative)
            };

            results.push(DataFileInfo {
                file_path,
                row_count: entry.record_count as u64,
                file_size_bytes: entry.file_size_bytes as u64,
                min_values,
                max_values,
                has_scalar_indexes,
                has_vector_indexes,
                indexed_columns,
            });
        }
        Ok(results)
    }

    /// Get splits for parallel reading (Index-aware, Async)
    pub async fn get_splits_async(&self, max_split_size: usize) -> Result<Vec<Split>> {
        let files = self.list_data_files_async().await?;
        let mut splits = Vec::new();
        for file in files {
             if file.file_size_bytes > max_split_size as u64 {
                let num_splits = (file.file_size_bytes / max_split_size as u64) + 1;
                for i in 0..num_splits {
                    splits.push(Split {
                        file_path: file.file_path.clone(),
                        start_offset: i * max_split_size as u64,
                        length: max_split_size as u64,
                        row_group_ids: vec![i as usize],
                        index_file_path: file.indexed_columns.first().map(|_| {
                            file.file_path.replace(".parquet", "")
                        }),
                        can_use_indexes: file.has_scalar_indexes || file.has_vector_indexes,
                    });
                }
            } else {
                splits.push(Split {
                    file_path: file.file_path.clone(),
                    start_offset: 0,
                    length: file.file_size_bytes,
                    row_group_ids: vec![0],
                    index_file_path: file.indexed_columns.first().map(|_| {
                        file.file_path.replace(".parquet", "")
                    }),
                    can_use_indexes: file.has_scalar_indexes || file.has_vector_indexes,
                });
            }
        }
        Ok(splits)
    }

    /// Read a specific data file (with index acceleration)
    pub fn read_file(&self, file_path: &str, columns: Option<Vec<String>>, filter: Option<&str>) -> Result<Vec<RecordBatch>> {
        self.runtime().block_on(async {
             // Use HybridReader
             // We need to construct SegmentConfig.
             // Assuming file_path ends with segment_id.parquet
             let parts: Vec<&str> = file_path.split('/').collect();
             let filename = parts.last().unwrap_or(&"wrapper");
             let segment_id = filename.replace(".parquet", "");
             
             // Base path: assume table logic. 
             // Actually HybridReader takes config and store.
             // Store handles URI.
             // If we reuse self.store, it points to table root.
             // If file_path is absolute, we might need new store or relative path.
             
             // For PoC: assume file_path is compatible with Table's store (relative or same bucket)
             // or creates new store if needed.
             // Let's use `create_object_store` if it looks like full URI.
             let (store, mut config) = if file_path.contains("://") {
                 let parse_res = url::Url::parse(file_path);
                 match parse_res {
                     Ok(url) => {
                         let scheme = url.scheme();
                         let store = if scheme == "file" {
                             let path_str = url.path();
                             let path = std::path::Path::new(path_str);
                             let parent = path.parent().unwrap_or(std::path::Path::new("/"));
                             let parent_uri = format!("file://{}", parent.to_string_lossy());
                             create_object_store(&parent_uri)?
                         } else {
                             create_object_store(file_path)?
                         };

                         let relative_path = if scheme == "file" {
                             let path = std::path::Path::new(url.path());
                             path.file_name().and_then(|s| s.to_str()).unwrap_or("wrapper").to_string()
                         } else {
                             let p = url.path();
                             p.trim_start_matches('/').to_string()
                         };
                         
                         let segment_id_full = relative_path;
                         let segment_id = segment_id_full.strip_suffix(".parquet").unwrap_or(&segment_id_full).to_string();
                         
                         let config = SegmentConfig::new("", &segment_id);
                         (store, config)
                     },
                     Err(_) => {
                         let s = create_object_store(file_path)?;
                         let config = SegmentConfig::new("", &segment_id);
                         (s, config)
                     }
                 }
             } else {
                 let config = SegmentConfig::new("", &segment_id);
                 (self.store.clone(), config)
             };

             // Try to enrich config from manifest
             let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
             let (_manifest, all_entries, _) = manager.load_latest_full().await.unwrap_or_default();
             if let Some(entry) = all_entries.iter().find(|e| e.file_path == file_path || e.file_path.ends_with(file_path)) {
                 config = config.with_parquet_path(entry.file_path.clone())
                     .with_delete_files(entry.delete_files.clone())
                     .with_index_files(entry.index_files.clone())
                     .with_file_size(entry.file_size_bytes as u64);
             }

              let reader = HybridReader::new(config, store, &self.uri);
             // stream_all calls index read if filter provided
             // Just read all columns for now, ignoring filter in reader (apply later)
             use futures::StreamExt;
             
             // Resolve Target Schema (Projection)
             let target_schema = if let Some(cols) = columns {
                  let current_schema = self.arrow_schema();
                  let fields: Vec<arrow::datatypes::Field> = cols.iter()
                      .filter_map(|name| current_schema.field_with_name(name).ok().cloned())
                      .collect();
                   // If requested columns not found in schema, we might return empty or error?
                   // For now, if non-empty projection requested but no matching fields, explicit empty schema (returns 0 cols).
                   if fields.is_empty() { 
                       // Check if columns were requested but not found.
                       // Maybe fallback to full if we can't resolve? No, explicit projection.
                       Some(Arc::new(Schema::new(Vec::<arrow::datatypes::Field>::new())))  
                   } else { 
                       Some(Arc::new(Schema::new(fields))) 
                   }
             } else {
                  None
             };

             let mut batches = Vec::new();

             // 1. Try Index Read if filter is present
             let mut index_used = false;
             if let Some(filter_str) = filter {
                 if let Some(qf) = crate::core::planner::QueryFilter::parse(filter_str) {
                     if let Ok(indexed_batches) = reader.query_index_first(&qf, target_schema.clone()).await {
                         batches = indexed_batches;
                         index_used = true;
                     }
                 }
             }

             // 2. Fallback to Full Scan
             if !index_used {
                 let mut stream = reader.stream_all(target_schema).await?;
                 while let Some(batch) = stream.next().await {
                     batches.push(batch?);
                 }
             }

             // 3. Apply post-filtering if filter is present
             if let Some(filter_str) = filter {
                 let planner = crate::core::planner::QueryPlanner::new();
                 let filter_expr = crate::core::planner::FilterExpr::parse_sql(filter_str, self.arrow_schema()).await?;
                 
                 let mut filtered_batches = Vec::new();
                 for batch in batches {
                     let filtered = planner.filter_expr(&batch, &filter_expr)?;
                     if filtered.num_rows() > 0 {
                         filtered_batches.push(filtered);
                     }
                 }
                 Ok(filtered_batches)
             } else {
                 Ok(batches)
             }

        })
    }

    /// Read a specific split (with index acceleration)
    pub fn read_split(&self, split: &Split, columns: Vec<String>, _filter: Option<&str>) -> Result<Vec<RecordBatch>> {
        self.runtime().block_on(async {
            // New Implementation: Use stream_row_groups with column pushdown
            
            // 1. Setup Reader (Duplicated logic from read_file - should refactor helper)
            let file_path = &split.file_path;
            
             let (store, config) = if file_path.contains("://") {
                 match url::Url::parse(file_path) {
                     Ok(url) => {
                         let scheme = url.scheme();
                         let store = if scheme == "file" {
                             let path_str = url.path();
                             let path = std::path::Path::new(path_str);
                             let parent = path.parent().unwrap_or(std::path::Path::new("/"));
                             let parent_uri = format!("file://{}", parent.to_string_lossy());
                             create_object_store(&parent_uri)?
                         } else {
                             create_object_store(file_path)?
                         };
                         
                         let relative_path = if scheme == "file" {
                             let path = std::path::Path::new(url.path());
                             path.file_name().and_then(|s| s.to_str()).unwrap_or("wrapper").to_string()
                         } else {
                             let p = url.path();
                             p.trim_start_matches('/').to_string()
                         };
                         let segment_id_full = relative_path;
                         let segment_id = segment_id_full.strip_suffix(".parquet").unwrap_or(&segment_id_full).to_string();
                         let config = SegmentConfig::new("", &segment_id);
                         (store, config)
                     },
                     Err(_) => {
                         let s = create_object_store(file_path)?;
                         let parts: Vec<&str> = file_path.split('/').collect();
                         let filename = parts.last().unwrap_or(&"wrapper");
                         let segment_id = filename.replace(".parquet", "");
                         let config = SegmentConfig::new("", &segment_id);
                         (s, config)
                     }
                 }
             } else {
                 let parts: Vec<&str> = file_path.split('/').collect();
                 let filename = parts.last().unwrap_or(&"wrapper");
                 let segment_id = filename.replace(".parquet", "");
                 let config = SegmentConfig::new("", &segment_id);
                 (self.store.clone(), config)
             };

              let reader = HybridReader::new(config, store, &self.uri);
             use futures::StreamExt;
             
             // Resolve Target Schema (Projection)
             let target_schema = if columns.is_empty() {
                 None
             } else {
                  let current_schema = self.arrow_schema();
                  let fields: Vec<arrow::datatypes::Field> = columns.iter()
                      .filter_map(|name| current_schema.field_with_name(name).ok().cloned())
                      .collect();
                   if fields.is_empty() { 
                       // Explicit empty schema
                       Some(Arc::new(Schema::new(Vec::<arrow::datatypes::Field>::new()))) 
                   } else { 
                       Some(Arc::new(Schema::new(fields))) 
                   }
             };

             // If filter is simple/indexed, we MIGHT try index read. 
             // BUT: index read (query_index_first) aims for whole file or needs row mapping.
             // If we are reading a split (subset), index reading whole file then filtering might be wrong?
             // Actually index returns row selection.
             // If we intersect index selection with row group selection it works.
             // But existing query_index_first() effectively does global selection.
             // For now, let's stick to stream_row_groups which applies deletes.
             // TODO: Index filtering on Split level needs combining RowGroups + Index Bitmap.
             
             let mut stream = reader.stream_row_groups(Some(&split.row_group_ids), target_schema).await?;
             
             let mut batches = Vec::new();
             while let Some(batch) = stream.next().await {
                 batches.push(batch?);
             }
             Ok(batches)
        })
    }

    /// Get table-level statistics (with index info)
    pub fn get_table_statistics(&self) -> Result<TableStatistics> {
        self.runtime().block_on(async {
            let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
            let (_manifest, all_entries, _) = manifest_manager.load_latest_full().await?;
            
            let row_count = all_entries.iter().map(|e| e.record_count).sum::<i64>() as u64;
            let total_size = all_entries.iter().map(|e| e.file_size_bytes).sum::<i64>() as u64;
            
            // Calculate basic index coverage
            // Assuming simplified logic: union of indexed columns across all files
            let mut scalar_idx = std::collections::HashSet::new();
            let mut vector_idx = std::collections::HashSet::new();
            
            let total_index_size = 0;

            for entry in &all_entries {
                for idx in &entry.index_files {
                    if let Some(col) = &idx.column_name {
                        if idx.index_type == "scalar" || idx.index_type == "inverted" {
                            scalar_idx.insert(col.clone());
                        } else if idx.index_type == "vector" || idx.index_type == "hnsw" {
                            vector_idx.insert(col.clone());
                        }
                    }
                    // Entry doesn't store index file size in IndexFile struct?
                    // It only stores file_path.
                    // Accessing object metadata for every index file is expensive here.
                    // Assuming 0 or approximated from somewhere else.
                }
            }

            let index_coverage = IndexCoverage {
                scalar_indexed_columns: scalar_idx.clone().into_iter().collect(),
                vector_indexed_columns: vector_idx.into_iter().collect(),
                inverted_indexed_columns: scalar_idx.into_iter().collect(),
                total_index_size_bytes: total_index_size,
            };
            
            Ok(TableStatistics {
                row_count,
                file_count: all_entries.len(),
                total_size_bytes: total_size,
                index_coverage,
            })
        })
    }

    /// Get table-level statistics (Asynchronous)
    pub async fn get_table_statistics_async(&self) -> Result<TableStatistics> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, all_entries, _) = manifest_manager.load_latest_full().await?;
        
        let row_count = all_entries.iter().map(|e| e.record_count).sum::<i64>() as u64;
        let total_size = all_entries.iter().map(|e| e.file_size_bytes).sum::<i64>() as u64;
        
        let mut scalar_idx = std::collections::HashSet::new();
        let mut vector_idx = std::collections::HashSet::new();
        
        let total_index_size = 0;

        for entry in &all_entries {
            for idx in &entry.index_files {
                if let Some(col) = &idx.column_name {
                    if idx.index_type == "scalar" || idx.index_type == "inverted" {
                        scalar_idx.insert(col.clone());
                    } else if idx.index_type == "vector" || idx.index_type == "hnsw" {
                        vector_idx.insert(col.clone());
                    }
                }
            }
        }

        let index_coverage = IndexCoverage {
            scalar_indexed_columns: scalar_idx.clone().into_iter().collect(),
            vector_indexed_columns: vector_idx.into_iter().collect(),
            inverted_indexed_columns: scalar_idx.into_iter().collect(), // For now, inverted are considered scalar too
            total_index_size_bytes: total_index_size,
        };
        
        Ok(TableStatistics {
            row_count,
            file_count: all_entries.len(),
            total_size_bytes: total_size,
            index_coverage,
        })
    }

    pub async fn get_snapshot_segments(&self) -> Result<Vec<ManifestEntry>> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_, all_entries, _) = manifest_manager.load_latest_full().await?;
        Ok(all_entries)
    }

    pub async fn get_snapshot_segments_with_version(&self) -> Result<(Manifest, u64)> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager.load_latest().await
    }

    /// Read from the in-memory write buffer with optional filter and projection
    /// Used by HyperStreamExec to include uncommitted data in SQL queries
    pub fn read_write_buffer(
        &self,
        filter: Option<&QueryFilter>,
        columns: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>> {
        let mut result = Vec::new();
        let buffer = self.write_buffer.read().unwrap();
        
        if buffer.is_empty() {
            return Ok(result);
        }

        let planner = QueryPlanner::new();
        
        for batch in buffer.iter() {
            // Apply projection first
            let batch_to_filter = if let Some(cols) = columns {
                let indices: Vec<usize> = cols.iter()
                    .filter_map(|name| batch.schema().index_of(name).ok())
                    .collect();
                batch.project(&indices).unwrap_or(batch.clone())
            } else {
                batch.clone()
            };

            // Apply filter if present
            if let Some(f) = filter {
                if let Ok(filtered) = planner.filter_batch(&batch_to_filter, f) {
                    if filtered.num_rows() > 0 {
                        result.push(filtered);
                    }
                }
            } else {
                result.push(batch_to_filter);
            }
        }
        Ok(result)
    }

    fn get_vector_column_for_shuffling(&self, batch: &RecordBatch) -> Option<String> {
        // Use first index column if it's a vector
        let index_cols = self.index_columns.read().unwrap();
        for col_name in index_cols.iter() {
            if let Ok(idx) = batch.schema().index_of(col_name) {
                let col = batch.column(idx);
                if matches!(col.data_type(), arrow::datatypes::DataType::FixedSizeList(inner, _) 
                  if *inner.data_type() == arrow::datatypes::DataType::Float32) {
                    return Some(col_name.clone());
                }
            }
        }

        // Fallback: Pick first vector column (FixedSizeList<Float32>)
        batch.schema().fields().iter()
            .find(|f| matches!(f.data_type(), arrow::datatypes::DataType::FixedSizeList(inner, _) 
                  if *inner.data_type() == arrow::datatypes::DataType::Float32))
            .map(|f| f.name().clone())
    }

    async fn shuffle_batch_by_centroids(&self, batch: &RecordBatch, col_name: &str) -> Result<RecordBatch> {
        use crate::core::index::ivf::simple_kmeans;
        use arrow::array::{Array, Int32Array};
        
        let col_idx = batch.schema().index_of(col_name)?;
        let list_array = batch.column(col_idx).as_any().downcast_ref::<arrow::array::FixedSizeListArray>().unwrap();
        
        let n = list_array.len();
        if n < 1024 { return Ok(batch.clone()); } // Too small to benefit from shuffling
        
        // 1. Convert vectors to Vec<Vec<f32>> for K-Means (Training step)
        // Optimization: Port this to handle FixedSizeList directly to avoid copies
        let vectors: Vec<Vec<f32>> = (0..n).into_par_iter().step_by(n / 1000 + 1).map(|i| {
            list_array.value(i).as_any().downcast_ref::<arrow::array::Float32Array>().unwrap().values().to_vec()
        }).collect();

        // 2. Train centroids (Sampled)
        let k = (n as f64).sqrt() as usize;
        let k = k.clamp(16, 1024);
        let (centroids, _) = simple_kmeans(&vectors, k, 3)?; // Fast 3-iter training

        // 3. Assign all vectors (GPU Accelerated!)
        let _ = get_global_gpu_context().unwrap_or_else(crate::core::index::gpu::ComputeContext::auto_detect);


        let dim = list_array.value_length() as usize;
        let flat_vectors: Vec<f32> = (0..n).into_par_iter().flat_map(|i| {
            list_array.value(i).as_any().downcast_ref::<arrow::array::Float32Array>().unwrap().values().to_vec()
        }).collect();
        
        let flat_centroids: Vec<f32> = centroids.iter().flatten().copied().collect();
        
        let assignments = crate::core::index::gpu::compute_kmeans_assignment(&flat_vectors, &flat_centroids, dim)?;
        
        // 4. Sort batch by assignments
        let assignment_array = Int32Array::from(assignments.into_iter().map(|a| a as i32).collect::<Vec<i32>>());
        let sort_indices = arrow::compute::sort_to_indices(&assignment_array, None, None)?;
        
        let mut columns = Vec::new();
        for i in 0..batch.num_columns() {
            columns.push(arrow::compute::take(batch.column(i), &sort_indices, None)?);
        }
        
        RecordBatch::try_new(batch.schema(), columns).context("Failed to reconstruct shuffled batch")
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use std::sync::Arc;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_table_lifecycle() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().to_str().unwrap().to_string();
        // Use local file system uri
        let uri = format!("file://{}", path);

        // 1. Create Table (async)
        let table = Table::new_async(uri.clone()).await?;
        
        // 2. Write Data
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]);
        
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["a", "b", "c"])),
            ]
        )?;
        
        // Use write_async since table was created with new_async
        table.write_async(vec![batch.clone()]).await?;
        table.commit_async().await?;

        // 3. Read Data
        let batches = table.read_async(None, None, None).await?;
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 3);


        Ok(())
    }

    #[tokio::test]
    async fn test_multi_column_bucketing() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().to_str().unwrap().to_string();
        let uri = format!("file://{}", path);
        let _table = Table::new_async(uri.clone()).await?;

        // 1. Setup schema with metadata IDs
        let mut fields = Vec::new();
        let mut id_meta = std::collections::HashMap::new();
        id_meta.insert("iceberg.id".to_string(), "1".to_string());
        fields.push(Field::new("col1", DataType::Int32, false).with_metadata(id_meta));
        
        let mut type_meta = std::collections::HashMap::new();
        type_meta.insert("iceberg.id".to_string(), "2".to_string());
        fields.push(Field::new("col2", DataType::Utf8, false).with_metadata(type_meta));
        
        let schema = Arc::new(Schema::new(fields));

        // 2. Define multi-column bucket partition spec
        let spec = crate::core::manifest::PartitionSpec {
            spec_id: 0,
            fields: vec![
                crate::core::manifest::PartitionField::new_multi(
                    vec![1, 2],
                    Some(1000),
                    "combined_bucket".to_string(),
                    "bucket[10]".to_string(),
                )
            ],
        };

        // 3. Create batch
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 2])),
                Arc::new(StringArray::from(vec!["a", "b", "a"])),
            ]
        )?;

        // 4. Split by partition
        let results = spec.partition_batch(&batch)?;
        
        // Each uniquely combined (col1, col2) should have a stable hash
        // (1, "a"), (1, "b"), (2, "a") are all different, so they should return 3 partitions
        // unless there's a hash collision (unlikely with only 10 buckets and these values)
        assert!(results.len() >= 2); 
        
        for (key, sub_batch) in results {
            assert!(key.contains_key("combined_bucket"));
            assert!(sub_batch.num_rows() >= 1);
        }

        Ok(())
    }

    #[tokio::test]
    async fn test_admin_ops() -> Result<()> {
        let dir = tempdir()?;
        let path = dir.path().to_str().unwrap().to_string();
        let uri = format!("file://{}", path);
        let table = Table::new_async(uri.clone()).await?;

        // 1. Initial State: Autocommit is True
        assert!(table.get_autocommit());

        // 2. Write Data (Autocommit)
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]
        )?;
        
        table.write_async(vec![batch.clone()]).await?;
        
        // Should be committed automatically
        let batches = table.read_async(None, None, None).await?;
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].num_rows(), 3);

        // 3. Truncate
        table.truncate_async().await?;
        let batches = table.read_async(None, None, None).await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        tracing::info!("After truncate, read {} records in {} batches", total_rows, batches.len());
        assert!(total_rows == 0, "Table should be empty after truncate, but found {} rows!", total_rows);

        // 4. Autocommit
        table.set_autocommit(false);
        table.write_async(vec![batch.clone()]).await?;
        
        // Visible in read (from buffer)
        let batches = table.read_async(None, None, None).await?;
        assert!(!batches.is_empty(), "Should see data in buffer");
        
        let manifest_manager = ManifestManager::new(table.store.clone(), "", &table.uri);
        let (_, _, ver_pre) = manifest_manager.load_latest_full().await.unwrap_or_default();
        assert_eq!(ver_pre, 2, "Should still be v2 before manual commit");
        
        table.commit_async().await?;
        let (_, _, ver_post) = manifest_manager.load_latest_full().await.unwrap_or_default();
        assert_eq!(ver_post, 3, "Should be v3 after manual commit");

        // 5. Vacuum
        // Note: vacuum_async might not delete anything if within retention, but let's test it works
        table.vacuum_async(1).await?; 
        
        Ok(())
    }
}
