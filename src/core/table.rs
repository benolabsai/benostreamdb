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
use roaring::RoaringBitmap;
use std::collections::HashMap;
use arrow::record_batch::RecordBatch;
use arrow::array::Array;
use object_store::ObjectStore;
use std::sync::Arc;
use tokio::runtime::Runtime;

use crate::core::storage::create_object_store;
use crate::core::manifest::{Manifest, ManifestEntry, ManifestManager, PartitionSpec, SortOrder, SortField, SortDirection, NullOrder};
use crate::core::metadata::TableMetadata;
use crate::core::planner::{QueryPlanner, QueryFilter, FilterExpr};
use crate::core::reader::HybridReader;
use crate::core::segment::HybridSegmentWriter;
use crate::core::compaction::{Compactor, CompactionOptions};
use crate::core::maintenance::Maintenance;
use crate::SegmentConfig;
use crate::core::query::{self, QueryConfig};
use serde::{Serialize, Deserialize};
use serde_json::Value; // Keep Value
use arrow::datatypes::{Schema, SchemaRef};
use crate::core::index::memory::InMemoryVectorIndex;
use crate::core::index::VectorMetric;
use crate::core::wal::WriteAheadLog;
use tokio::sync::Mutex;
use tokio::task::JoinHandle;
use crate::telemetry::metrics::INGEST_ROWS_TOTAL;

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

    /// Create a new Table instance (Synchronous)
    /// This blocks the current thread to load the schema.
    /// CAUTION: Do not call this from within an async runtime. Use `new_async` instead.
    pub fn new(uri: String) -> Result<Self> {
        // 1. Check if it's an Iceberg REST URI
        if let Some((base, prefix, ns, table)) = Self::detect_iceberg_rest(&uri) {
            let rt = Arc::new(Runtime::new()?);
            let rest_uri = uri.clone();
            return rt.block_on(async {
                Self::new_from_rest(base, prefix, ns, table, &rest_uri).await
            });
        }

        let store = create_object_store(&uri)?;
        let rt = Arc::new(Runtime::new()?);
        
        // Load schema eagerly (blocking)
        let schema_val = rt.block_on(Self::load_initial_schema(store.clone(), &uri));

        // Initialize WAL (Sync wrapper)
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
        let _ = rt.block_on(async {
            wal.spawn_worker().unwrap_or_default();
        });
        
        // Replay WAL (Recovery)
        let recovered_batches = wal.replay().unwrap_or_else(|e| {
            println!("WAL Recovery Warning: {}", e);
            Vec::new()
        });
        
        let mut initial_buffer = Vec::new();
        let mut initial_mem_index = None;

        if !recovered_batches.is_empty() {
            println!("recovering {} batches from WAL...", recovered_batches.len());
            initial_buffer = recovered_batches;
             
             // Simple recovery of memory index (brute force)
             if let Some(first) = initial_buffer.first() {
                 if let Some(col) = first.column_by_name("embedding") {
                     if let Some(fsl) = col.as_any().downcast_ref::<arrow::array::FixedSizeListArray>() {
                         let dim = fsl.value_length() as usize;
                         let mut idx = InMemoryVectorIndex::new(dim);
                         let mut offset = 0;
                         for batch in &initial_buffer {
                             let _ = idx.insert_batch(batch, "embedding", offset);
                             offset += batch.num_rows();
                         }
                         initial_mem_index = Some(idx);
                     }
                 }
             }
        }

        Ok(Table { 
            uri, 
            store, 
            data_store: None,
            rt: Some(rt),
            query_config: QueryConfig::default(),
            index_all: false, // Default: Index Nothing
            index_columns: Arc::new(std::sync::RwLock::new(Vec::new())),
            schema: Arc::new(std::sync::RwLock::new(schema_val)),
            write_buffer: Arc::new(std::sync::RwLock::new(initial_buffer)),
            memory_index: Arc::new(std::sync::RwLock::new(initial_mem_index)),
            wal: Arc::new(Mutex::new(wal)),
            background_tasks: Arc::new(Mutex::new(Vec::new())),
            catalog: None,
            catalog_namespace: None,
            catalog_table_name: None,
            sort_order: Arc::new(std::sync::RwLock::new(None)),
            sort_order_columns: Arc::new(std::sync::RwLock::new(None)),
            #[cfg(feature = "enterprise")]
            enterprise_license: None,
        })
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
        
        println!("Resolved Nessie table {} to metadata: {}", table, metadata.location);
        
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
        println!("Resolved Glue table {}.{} to metadata: {}", namespace, table, metadata.location);
        
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
        println!("Resolved Hive table {}.{} to metadata: {}", namespace, table, metadata.location);
        
        let mut table_obj = Self::register_external(local_uri, &metadata.location).await?;
        table_obj.catalog = Some(Arc::new(client));
        table_obj.catalog_namespace = Some(namespace.to_string());
        table_obj.catalog_table_name = Some(table.to_string());
        Ok(table_obj)
    }

    /// Register an existing Iceberg table for Layered Indexing
    pub async fn register_external(uri: String, iceberg_metadata_uri: &str) -> Result<Self> {
        create_object_store(&uri)?;
        
        // Derive iceberg_meta_store from the parent directory of iceberg_metadata_uri
        let meta_url = url::Url::parse(iceberg_metadata_uri).context("Invalid iceberg_metadata_uri")?;
        let meta_path_str = meta_url.path();
        let meta_path = std::path::Path::new(meta_path_str);
        let parent_dir = meta_path.parent().context("No parent directory for metadata file")?;
        let filename = meta_path.file_name().context("No filename for metadata file")?.to_str().unwrap();
        
        let meta_store_uri = if iceberg_metadata_uri.starts_with("file://") {
            format!("file://{}", parent_dir.display())
        } else {
            // Reconstruct URI without the filename
            let mut base_url = meta_url.clone();
            base_url.set_path(parent_dir.to_str().unwrap());
            base_url.to_string()
        };

        let iceberg_meta_store = create_object_store(&meta_store_uri)?;
        
        println!("Linking external Iceberg table: {}", iceberg_metadata_uri);
        
        // 1. Load Iceberg Metadata
        let path = object_store::path::Path::from(filename);
        let ret = iceberg_meta_store.get(&path).await?;
        let bytes = ret.bytes().await?;
        let iceberg_meta: crate::core::iceberg::IcebergTableMetadata = serde_json::from_slice(&bytes)?;
        
        // 2. Map Iceberg Schema to HyperStreamDB Schema
        // For simplicity, we use the current schema
        let current_schema_json = iceberg_meta.schemas.iter()
            .find(|s| s.get("schema-id").and_then(|v| v.as_i64()) == Some(iceberg_meta.current_schema_id as i64))
            .unwrap_or_else(|| {
                eprintln!("WARNING: Could not find schema with ID {}, falling back to first schema", iceberg_meta.current_schema_id);
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
            let mut table = Self::new_native_async(native_uri).await?;
            table.catalog = Some(Arc::new(client));
            table.catalog_namespace = Some(namespace);
            table.catalog_table_name = Some(table_name);
            Ok(table)
        } else {
            println!("Auto-registering Iceberg table from REST catalog: {}", rest_uri);
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
                    Ok(true) => println!("Snapshot Observer: New snapshot processed."),
                    Ok(false) => {}, // No new snapshot
                    Err(e) => eprintln!("Snapshot Observer Error: {}", e),
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
            
        println!("Importing Iceberg Snapshot {}...", snapshot_id);
        
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
                    }).unwrap_or_default()
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
                         Err(e) => println!("Error converting Iceberg entry: {}", e),
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
        // 1. Check if it's an Iceberg REST URI
        if let Some((base, prefix, ns, table)) = Self::detect_iceberg_rest(&uri) {
            return Self::new_from_rest(base, prefix, ns, table, &uri).await;
        }

        Self::new_native_async(uri).await
    }

    /// Internal core logic for opening a native HyperStreamDB table asynchronously
    async fn new_native_async(uri: String) -> Result<Self> {
        let store = create_object_store(&uri)?;
        // For async usage, we do NOT create a dedicated runtime.
        // This avoids the panic on Drop when running inside an async context.
        // NOTE: Sync methods (read, write) will panic if called on this instance.
        
        let mut schema_val = Self::load_initial_schema(store.clone(), &uri).await;
        let index_cols = Arc::new(std::sync::RwLock::new(Vec::<String>::new()));

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
        let recovered_batches = wal.replay().unwrap_or_else(|e| {
            println!("WAL Recovery Warning: {}", e);
            Vec::new()
        });
        
        let mut initial_buffer = Vec::new();
        let mut initial_mem_index = None;

        if !recovered_batches.is_empty() {
            println!("recovering {} batches from WAL...", recovered_batches.len());
            // Restore Schema from first batch if unknown
            if schema_val.fields().is_empty() {
                 if let Some(first) = recovered_batches.first() {
                     schema_val = first.schema();
                 }
            }

            // Restore Index
            // (Copy-paste logic from write_async backfill, simplified)
             let target_col = index_cols.read().unwrap().first().cloned()
                 .or_else(|| {
                     recovered_batches.first().and_then(|b| {
                         b.schema().fields().iter()
                             .find(|f| f.name() == "embedding")
                             .map(|f| f.name().clone())
                     })
                 });
            
            if let Some(col_name) = target_col {
                let mut offset = 0;
                // Init index
                if let Some(first) = recovered_batches.first() {
                     if let Some(col) = first.column_by_name(&col_name) {
                         // Support FixedSizeList, List, LargeList
                         let dim = if let Some(fsl) = col.as_any().downcast_ref::<arrow::array::FixedSizeListArray>() {
                              Some(fsl.value_length() as usize)
                         } else if let Some(list) = col.as_any().downcast_ref::<arrow::array::ListArray>() {
                              // Probe first non-null
                              (0..list.len()).find_map(|i| {
                                  if list.is_null(i) { None } else {
                                      list.value(i).as_any().downcast_ref::<arrow::array::Float32Array>().map(|v| v.len())
                                  }
                              })
                         } 
                         // Time32 range - Temporarily disabled due to type ambiguity
                         /*
                         else if let Some(time32) = col.as_any().downcast_ref::<arrow::array::Time32Array>() {
                             // This branch is for indexing, not vector search, so dim is not applicable here.
                             // This block seems misplaced if it's trying to determine vector dimension.
                             // It should be handled by the `insert_batch` logic if Time32/Time64 indexing is supported.
                             None
                         }
                         // Time64 range
                         else if let Some(time64) = col.as_any().downcast_ref::<arrow::array::Time64Array>() {
                             None
                         }
                         */
                         else { None }; // TODO: LargeList support here too if needed generically

                         if let Some(d) = dim {
                             let mut idx = InMemoryVectorIndex::new(d);
                             for batch in &recovered_batches {
                                 let _ = idx.insert_batch(batch, &col_name, offset);
                                 offset += batch.num_rows();
                             }
                             initial_mem_index = Some(idx);
                         }
                     }
                }
            }
            
            initial_buffer = recovered_batches;
            
            // Note: We DO NOT truncate WAL here. 
            // We kept the data in memory (dirty). If we crash now, we need WAL again.
            // We only truncate when we Flush to Parquet.
            // However, since we re-read them, the `wal` writer needs to know to Append?
            // `wal.append` opens in Append mode, so it's fine.
            // BUT: `wal.truncate` happens on flush.
            // If we have previous WAL data, and we Append new data, the file grows.
            // Correct.
        }

        Ok(Table { 
            uri: uri.to_string(), 
            store, 
            data_store: None,
            rt: None,
            query_config: QueryConfig::default(),
            index_all: false, 
            index_columns: index_cols,
            schema: Arc::new(std::sync::RwLock::new(schema_val)),
            write_buffer: Arc::new(std::sync::RwLock::new(initial_buffer)),
            memory_index: Arc::new(std::sync::RwLock::new(initial_mem_index)),
            wal: Arc::new(Mutex::new(wal)),
            background_tasks: Arc::new(Mutex::new(Vec::new())),
            catalog: None,
            catalog_namespace: None,
            catalog_table_name: None,
            sort_order: Arc::new(std::sync::RwLock::new(None)),
            sort_order_columns: Arc::new(std::sync::RwLock::new(None)),
            #[cfg(feature = "enterprise")]
            enterprise_license: None,
        })
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
        manifest_manager.update_schema(vec![manifest_schema.clone()], 1, Some(max_id as i32)).await?;

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
        Self::new_native_async(uri).await
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
        Self::new_native_async(uri).await
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
    /// ```
    /// table.replace_sort_order(&["timestamp", "user_id"], &[false, true])?;
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
             let filename = path.split('/').last().unwrap();
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

    pub fn add_index_columns(&mut self, columns: Vec<String>) -> Result<()> {
        let mut index_cols = self.index_columns.write().unwrap();
        index_cols.extend(columns.clone());
        index_cols.sort();
        index_cols.dedup();
        drop(index_cols);
        self.backfill_indexes(columns)
    }

    pub async fn add_index_columns_async(&mut self, columns: Vec<String>) -> Result<()> {
        let mut index_cols = self.index_columns.write().unwrap();
        index_cols.extend(columns.clone());
        index_cols.sort();
        index_cols.dedup();
        drop(index_cols);
        self.backfill_indexes_async(columns).await
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
                let segment_id = file_path_str.split('/').last().unwrap_or(&file_path_str)
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
                writer.set_store(store.clone());
                
                let mut stream = reader.stream_all(None as Option<arrow::datatypes::SchemaRef>).await?;
                while let Some(batch) = stream.next().await {
                    let batch = batch?;
                    writer.write_batch(&batch)?;
                    writer.build_indexes(&batch)?;
                }
                
                writer.upload_to_store().await?;
                
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

    pub fn read(&self, filter: Option<&str>, vector_filter: Option<VectorSearchParams>) -> Result<Vec<RecordBatch>> {
        self.runtime().block_on(self.read_async(filter, vector_filter, None))
    }

    pub fn read_with_columns(&self, filter: Option<&str>, vector_filter: Option<VectorSearchParams>, columns: Vec<String>) -> Result<Vec<RecordBatch>> {
        let columns_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
        self.runtime().block_on(self.read_async(filter, vector_filter, Some(&columns_refs)))
    }

    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        use datafusion::prelude::SessionContext;
        use crate::core::sql::HyperStreamTableProvider;

        let ctx = SessionContext::new();
        let provider = Arc::new(HyperStreamTableProvider::new(Arc::new(self.clone())));
        ctx.register_table("t", provider)?;
        let df = ctx.sql(query).await?;
        Ok(df.collect().await?)
    }

    pub async fn read_async(&self, filter_str: Option<&str>, vector_filter: Option<VectorSearchParams>, columns: Option<&[&str]>) -> Result<Vec<RecordBatch>> {
        self.read_with_config_async(filter_str, vector_filter, columns, self.query_config.clone()).await
    }

    pub async fn read_with_config_async(
        &self, 
        filter_str: Option<&str>, 
        vector_filter: Option<VectorSearchParams>, 
        columns: Option<&[&str]>,
        config: QueryConfig
    ) -> Result<Vec<RecordBatch>> {
        let expr = match filter_str {
            Some(f) => {
                let schema = self.arrow_schema();
                Some(FilterExpr::parse_sql(f, schema).await?)
            }
            _ => None,
        };
        self.read_expr_with_config_async(expr, vector_filter, columns, config).await
    }

    pub async fn read_expr_with_config_async(
        &self,
        expr: Option<FilterExpr>,
        vector_filter: Option<VectorSearchParams>,
        columns: Option<&[&str]>,
        config: QueryConfig,
    ) -> Result<Vec<RecordBatch>> {
        use futures::StreamExt;

        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, all_entries, version) = manifest_manager.load_latest_full().await.unwrap_or_default();

        let entries_to_read = if manifest.version > 0 {
            if expr.is_some() {
                let planner = QueryPlanner::new();
                planner.prune_entries(&all_entries, expr.as_ref()).into_iter().map(|(e, _)| e).collect()
            } else {
                all_entries.clone()
            }
        } else {
            self.list_segments_from_store().await?
        };
        // Handle vector search
        if let Some(ref vs_params) = vector_filter {
             // 1. Search Disk
             let mut results = query::execute_vector_search_with_config(
                entries_to_read,
                self.store.clone(),
                self.data_store.clone(),
                &self.uri,
                &vs_params.column,
                &vs_params.query,
                vs_params.k,
                expr.clone(),
                vs_params.metric,
                config.clone(),
                vs_params.ef_search,
            ).await?;

             // 2. Search Memory
             let memory_hits = {
                 let idx = self.memory_index.read().unwrap();
                 if let Some(mem_idx) = idx.as_ref() {
                     let filter_bitmap = if let Some(ref e) = expr {
                         let buffer = self.write_buffer.read().unwrap();
                         let mut bitmap = RoaringBitmap::new();
                          let mut offset = 0;
                          let planner = QueryPlanner::new();
                          for batch in buffer.iter() {
                               if let Ok(mask) = planner.evaluate_expr(batch, e) {
                                   for i in 0..batch.num_rows() {
                                      if mask.value(i) {
                                          bitmap.insert((offset + i) as u32);
                                      }
                                  }
                              }
                              offset += batch.num_rows();
                          }
                          Some(bitmap)
                      } else {
                          None
                      };
                      mem_idx.search(&vs_params.query, vs_params.k, filter_bitmap.as_ref())
                  } else {
                      vec![]
                  }
              };

              if !memory_hits.is_empty() {
                  let buffer = self.write_buffer.read().unwrap();
                  if let Some(first) = buffer.first() {
                      let schema = first.schema();
                      let batch_offsets: Vec<usize> = buffer.iter().scan(0, |state, b| {
                          let start = *state;
                          *state += b.num_rows();
                          Some(start)
                      }).collect();
                      
                      let mut result_rows = Vec::new();
                      for (id, _dist) in memory_hits {
                          for (i, offset) in batch_offsets.iter().enumerate().rev() {
                              if id >= *offset {
                                  let row_idx = id - offset;
                                  if i < buffer.len() && row_idx < buffer[i].num_rows() {
                                      result_rows.push(buffer[i].slice(row_idx, 1));
                                  }
                                  break;
                              }
                          }
                      }
                      
                      if !result_rows.is_empty() {
                          let mem_batch = arrow::compute::concat_batches(&schema, result_rows.iter().collect::<Vec<&RecordBatch>>())?;
                          results.push(mem_batch);
                      }
                  }
              }

              return Ok(results);
        }

        let expr_arc = expr.map(Arc::new);
        let stream = futures::stream::iter(entries_to_read)
            .map(|entry| {
                let expr_clone = expr_arc.clone();
                async move {
                    self.read_segment_expr(&entry, expr_clone.as_deref(), version, columns).await
                }
            })
            .buffer_unordered(16); 

        let results: Vec<Result<Vec<RecordBatch>>> = stream.collect().await;
        let mut all_batches = Vec::new();
        for (i, res) in results.into_iter().enumerate() {
            match res {
                Ok(b_vec) => {
                     all_batches.extend(b_vec);
                },
                Err(e) => eprintln!("Error reading batch {}: {}", i, e),
            }
        }

        // --- Read from In-Memory Write Buffer ---
        {
            let buffer = self.write_buffer.read().unwrap();
            if !buffer.is_empty() {
                if let Some(ref e) = expr_arc {
                    let planner = QueryPlanner::new();
                    for batch in buffer.iter() {
                         let batch_to_scan = if let Some(cols) = columns {
                             let indices: Vec<usize> = cols.iter()
                                .filter_map(|name| batch.schema().index_of(name).ok())
                                .collect();
                             batch.project(&indices).unwrap_or(batch.clone())
                         } else {
                             batch.clone()
                         };

                         if let Ok(filtered) = planner.filter_expr(&batch_to_scan, e) {
                             if filtered.num_rows() > 0 {
                                 all_batches.push(filtered);
                             }
                         }
                    }
                } else {
                     for batch in buffer.iter() {
                         if let Some(cols) = columns {
                             let indices: Vec<usize> = cols.iter()
                                .filter_map(|name| batch.schema().index_of(name).ok())
                                .collect();
                             if let Ok(projected) = batch.project(&indices) {
                                 all_batches.push(projected);
                             }
                         } else {
                             all_batches.push(batch.clone());
                         }
                     }
                }
            }
        }
        

        Ok(all_batches)
    }

    pub async fn read_filter_async(
        &self,
        filters: Vec<QueryFilter>,
        vector_filter: Option<VectorSearchParams>,
        columns: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>> {
        self.read_filter_with_config_async(filters, vector_filter, columns, self.query_config.clone()).await
    }

    pub async fn read_filter_with_config_async(
        &self,
        filters: Vec<QueryFilter>,
        vector_filter: Option<VectorSearchParams>,
        columns: Option<&[&str]>,
        config: QueryConfig,
    ) -> Result<Vec<RecordBatch>> {
        let expr = FilterExpr::from_filters(filters);
        self.read_expr_with_config_async(expr, vector_filter, columns, config).await
    }

    pub async fn read_segment_expr(
        &self,
        entry: &ManifestEntry,
        expr: Option<&FilterExpr>,
        manifest_version: u64,
        columns: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>> {
        use futures::StreamExt;
        
        let file_path_str = entry.file_path.clone();
        let segment_id = file_path_str.split('/').last().unwrap_or(&file_path_str)
            .strip_suffix(".parquet").unwrap_or(&file_path_str);

        // Load Manifest to get Schema
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, _, _) = manifest_manager.load_latest_full().await.unwrap_or_default();
        
        let iceberg_schema = manifest.schemas.iter().find(|s| s.schema_id == manifest.current_schema_id).cloned();

        let config = SegmentConfig::new(&self.uri, segment_id)
            .with_parquet_path(entry.file_path.clone())
            .with_data_store(self.data_store.clone().unwrap_or(self.store.clone()))
            .with_delete_files(entry.delete_files.clone())
            .with_index_files(entry.index_files.clone())
            .with_file_size(entry.file_size_bytes as u64)
            .with_index_all(self.index_all)
            .with_columns_to_index(self.index_columns.read().unwrap().clone());

        let mut reader = HybridReader::new(config, self.store.clone(), &self.uri);
        if let Some(s) = &iceberg_schema {
            reader = reader.with_iceberg_schema(s.clone());
        }

        let full_schema = if let Some(schema) = &iceberg_schema {
             Arc::new(schema.to_arrow())
        } else {
             self.arrow_schema()
        };

        let target_schema = if let Some(cols) = columns {
             let fields: Vec<arrow::datatypes::Field> = cols.iter()
                 .filter_map(|name| full_schema.field_with_name(name).ok().cloned())
                 .collect();
             Some(Arc::new(Schema::new(fields)))
        } else {
             Some(full_schema)
        };

        if manifest_version == 0 || expr.is_none() {
            let mut stream = reader.stream_all(target_schema).await?;
            let mut batches = Vec::new();
            while let Some(batch_result) = stream.next().await {
                batches.push(batch_result?);
            }
            return Ok(batches);
        }

        let expr = expr.unwrap();
        let and_filters = expr.extract_and_conditions();

        // Try to use index for the FIRST filter that has one
        let mut batches = Vec::new();
        let mut index_used = false;

        for filter in and_filters {
             if let Ok(indexed_batches) = reader.query_index_first(&filter, target_schema.clone()).await {
                 batches = indexed_batches;
                 index_used = true;
                 break;
             } else {

             }
        }

        if !index_used {
            let mut stream = reader.stream_all(target_schema).await?;
            while let Some(batch_result) = stream.next().await {
                batches.push(batch_result?);
            }
        }

        let planner = QueryPlanner::new();
        let mut filtered_batches = Vec::new();
        for batch in batches {
            if let Ok(filtered) = planner.filter_expr(&batch, expr) {
                if filtered.num_rows() > 0 {
                     filtered_batches.push(filtered);
                }
            } else if let Err(_e) = planner.filter_expr(&batch, expr) {

            }
        }
        Ok(filtered_batches)
    }

    pub async fn read_segment_multi(
        &self,
        entry: &ManifestEntry,
        filters: &[QueryFilter],
        manifest_version: u64,
        columns: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>> {
        let expr = FilterExpr::from_filters(filters.to_vec());
        self.read_segment_expr(entry, expr.as_ref(), manifest_version, columns).await
    }

    pub async fn read_segment(
        &self,
        entry: &ManifestEntry,
        query_filter_opt: Option<&QueryFilter>,
        manifest_version: u64,
        columns: Option<&[&str]>,
    ) -> Result<Vec<RecordBatch>> {
        let filters = match query_filter_opt {
            Some(f) => vec![f.clone()],
            None => vec![],
        };
        self.read_segment_multi(entry, &filters, manifest_version, columns).await
    }

    pub async fn stream_all(&self, columns: Option<&[&str]>) -> Result<futures::stream::BoxStream<'static, Result<RecordBatch>>> {
        use futures::StreamExt;
        let batches = self.read_async(None, None, columns).await?;
        Ok(futures::stream::iter(batches.into_iter().map(Ok)).boxed())
    }

    async fn list_segments_from_store(&self) -> Result<Vec<ManifestEntry>> {
        use futures::StreamExt;
        let mut entries = Vec::new();
        let mut stream = self.store.list(None);
        while let Some(res) = stream.next().await {
            if let Ok(meta) = res {
                let path = meta.location.to_string();
                if (!path.contains("/") || path.contains("data/")) && path.ends_with(".parquet") 
                   && !path.contains(".inv.parquet") && !path.contains(".hnsw.") {
                       entries.push(ManifestEntry {
                           file_path: path,
                           file_size_bytes: meta.size as i64,
                           record_count: 0,
                           index_files: vec![],
                           delete_files: vec![],
                           column_stats: std::collections::HashMap::new(),
                           partition_values: std::collections::HashMap::new(),
                           clustering_strategy: None,
                           clustering_columns: None,
                           min_clustering_score: None,
                           max_clustering_score: None,
                           normalization_mins: None,
                           normalization_maxs: None,
                       });
                }
            }
        }
        Ok(entries)
    }

    /// Write Arrow RecordBatches to the table (Buffered)
    /// 
    /// Data is written to an in-memory buffer. It is NOT persisted to disk until:
    /// 1. The buffer exceeds `HYPERSTREAM_CACHE_GB`
    /// 2. `commit()` is called explicitly
    pub fn write(&self, batches: Vec<RecordBatch>) -> Result<()> {
        self.runtime().block_on(self.write_async(batches))
    }

    /// Commit buffered writes to disk
    pub fn commit(&self) -> Result<()> {
        self.runtime().block_on(self.flush_async())
    }
    
    /// Async commit
    pub async fn commit_async(&self) -> Result<()> {
        self.flush_async().await
    }

    /// Compact the WAL (consolidate log entries)
    pub fn checkpoint(&self) -> Result<()> {
        let mut wal = self.wal.blocking_lock();
        wal.compact()
    }

    /// Explicit Schema Evolution: Add a new column
    pub async fn add_column(&self, name: &str, data_type: arrow::datatypes::DataType) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (mut manifest, _, _) = manifest_manager.load_latest_full().await?;
        
        let mut current_schema = if let Some(schema) = manifest.schemas.iter().find(|s| s.schema_id == manifest.current_schema_id) {
             schema.clone()
        } else {
             // Bootstrap schema from Arrow schema if not tracking yet
             let arrow_schema = self.arrow_schema();
             crate::core::manifest::Schema::new(0, arrow_schema.fields().iter().enumerate().map(|(i, f)| {
                 crate::core::manifest::SchemaField {
                     id: i as i32 + 1,
                     name: f.name().clone(),
                     type_str: f.data_type().to_string(),
                      required: !f.is_nullable(),
                      fields: Vec::new(),
                      initial_default: None,
                      write_default: None,
                  }
             }).collect())
        };

        // Check if exists
        if current_schema.fields.iter().any(|f| f.name == name) {
            return Err(anyhow::anyhow!("Column '{}' already exists", name));
        }

        // New Field ID
        let new_id = manifest.last_column_id + 1;
        let new_field = crate::core::manifest::SchemaField::from_arrow_field(
            &arrow::datatypes::Field::new(name, data_type.clone(), true),
            new_id
        );
        current_schema.fields.push(new_field);

        // Update Manifest
        let new_schema_id = manifest.current_schema_id + 1;
        current_schema.schema_id = new_schema_id;
        
        manifest.schemas.push(current_schema);
        manifest.current_schema_id = new_schema_id;
        manifest.last_column_id = new_id;
        
        // Commit Metadata Only Change
        manifest_manager.update_schema(manifest.schemas, manifest.current_schema_id, Some(new_id)).await?;
        println!("Schema Evolution: Added column '{}' (Schema ID: {})", name, new_schema_id);
        
        Ok(())
    }

    /// Partition Spec Evolution: Set a new partition spec (Iceberg V2 spec compliance)
    /// 
    /// The previous spec is retained in history for reading old data files.
    /// 
    /// # Arguments
    /// * `fields` - Partition field definitions (source_id, name, transform)
    /// 
    /// # Example
    /// ```
    /// table.set_partition_spec(&[
    ///     PartitionField { source_id: 1, field_id: 1000, name: "month".into(), transform: "month".into() },
    /// ]).await?;
    /// ```
    pub async fn update_spec(&self, fields: &[crate::core::manifest::PartitionField]) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, _, _) = manifest_manager.load_latest_full().await?;
        
        // Determine next spec_id
        let new_spec_id = manifest.partition_specs.iter()
            .map(|s| s.spec_id)
            .max()
            .unwrap_or(-1) + 1;
        
        // Create new spec
        let new_spec = PartitionSpec {
            spec_id: new_spec_id,
            fields: fields.to_vec(),
        };
        
        // Build updated specs list (append new spec to history)
        let mut updated_specs = manifest.partition_specs.clone();
        updated_specs.push(new_spec);
        
        // Commit the partition spec evolution
        let metadata = crate::core::manifest::CommitMetadata {
            updated_partition_specs: Some(updated_specs),
            updated_default_spec_id: Some(new_spec_id),
            ..Default::default()
        };
        
        manifest_manager.commit(&[], &[], metadata).await?;
        println!("Partition Evolution: New spec ID {} with {} fields", new_spec_id, fields.len());
        
        Ok(())
    }

    /// Explicit Schema Evolution: Drop a column
    pub async fn drop_column(&self, name: &str) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (mut manifest, _, _) = manifest_manager.load_latest_full().await?;
        
        let mut current_schema = manifest.schemas.iter().find(|s| s.schema_id == manifest.current_schema_id)
            .ok_or_else(|| anyhow::anyhow!("No active schema found for evolution"))?.clone();
            
        if !current_schema.fields.iter().any(|f| f.name == name) {
            return Err(anyhow::anyhow!("Column '{}' does not exist", name));
        }
        
        // Remove field
        current_schema.fields.retain(|f| f.name != name);
        
        // Update Manifest
        let new_schema_id = manifest.current_schema_id + 1;
        current_schema.schema_id = new_schema_id;
        manifest.schemas.push(current_schema);
        manifest.current_schema_id = new_schema_id;
        manifest_manager.update_schema(manifest.schemas, manifest.current_schema_id, Some(manifest.last_column_id)).await?;
        println!("Schema Evolution: Dropped column '{}' (Schema ID: {})", name, new_schema_id);
        Ok(())
    }
    
    /// Explicit Schema Evolution: Rename a column
    pub async fn rename_column(&self, old_name: &str, new_name: &str) -> Result<()> {
         let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (mut manifest, _, _) = manifest_manager.load_latest_full().await?;

        let mut current_schema = manifest.schemas.iter().find(|s| s.schema_id == manifest.current_schema_id)
            .ok_or_else(|| anyhow::anyhow!("No active schema found for evolution"))?.clone();
            
        if let Some(field) = current_schema.fields.iter_mut().find(|f| f.name == old_name) {
             field.name = new_name.to_string();
        } else {
             return Err(anyhow::anyhow!("Column '{}' does not exist", old_name));
        }

        let new_schema_id = manifest.current_schema_id + 1;
        current_schema.schema_id = new_schema_id;
        manifest.schemas.push(current_schema);
        manifest.current_schema_id = new_schema_id;
        manifest_manager.update_schema(manifest.schemas, manifest.current_schema_id, Some(manifest.last_column_id)).await?;
        println!("Schema Evolution: Renamed '{}' -> '{}' (Schema ID: {})", old_name, new_name, new_schema_id);
        Ok(())
    }

    /// Explicit Schema Evolution: Update column type (Type Promotion)
    /// Widens the type of an existing column (e.g., int -> long, float -> double)
    pub async fn update_column_type(&self, name: &str, new_type: &str) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (mut manifest, _, _) = manifest_manager.load_latest_full().await?;
        
        let mut current_schema = manifest.schemas.iter().find(|s| s.schema_id == manifest.current_schema_id)
            .ok_or_else(|| anyhow::anyhow!("No active schema found for evolution"))?.clone();
            
        if let Some(field) = current_schema.fields.iter_mut().find(|f| f.name == name) {
             if !Self::can_promote(&field.type_str, new_type) {
                 return Err(anyhow::anyhow!("Invalid type promotion: {} -> {}", field.type_str, new_type));
             }
             field.type_str = new_type.to_string();
        } else {
             return Err(anyhow::anyhow!("Column '{}' does not exist", name));
        }

        let new_schema_id = manifest.current_schema_id + 1;
        current_schema.schema_id = new_schema_id;
        manifest.schemas.push(current_schema);
        manifest.current_schema_id = new_schema_id;
        
        manifest_manager.update_schema(manifest.schemas, manifest.current_schema_id, Some(manifest.last_column_id)).await?;
        println!("Schema Evolution: Updated column type '{}' to '{}' (Schema ID: {})", name, new_type, new_schema_id);
        Ok(())
    }

    fn can_promote(old_type: &str, new_type: &str) -> bool {
        match (old_type.to_lowercase().as_str(), new_type.to_lowercase().as_str()) {
            ("int" | "int32", "long" | "int64") => true,
            ("float" | "float32", "double" | "float64") => true,
            (o, n) if (o.contains("decimal") || o.contains("decimal")) && (n.contains("decimal") || n.contains("decimal")) => {
                // Simplified: allow any decimal to any decimal for now (usually widening precision)
                true
            },
            _ => false
        }
    }

    /// Explicit Schema Evolution: Move a column to a new position (0-based index)
    pub async fn move_column(&self, name: &str, new_index: usize) -> Result<()> {
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (mut manifest, _, _) = manifest_manager.load_latest_full().await?;
        
        let mut current_schema = manifest.schemas.iter().find(|s| s.schema_id == manifest.current_schema_id)
            .ok_or_else(|| anyhow::anyhow!("No active schema found for evolution"))?.clone();
            
        let old_index = current_schema.fields.iter().position(|f| f.name == name)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' does not exist", name))?;
            
        if new_index >= current_schema.fields.len() {
             return Err(anyhow::anyhow!("Invalid new index {}", new_index));
        }
        
        if old_index == new_index {
            return Ok(()); // No-op
        }

        let field = current_schema.fields.remove(old_index);
        current_schema.fields.insert(new_index, field);

        let new_schema_id = manifest.current_schema_id + 1;
        current_schema.schema_id = new_schema_id;
        manifest.schemas.push(current_schema);
        manifest.current_schema_id = new_schema_id;
        
        manifest_manager.update_schema(manifest.schemas, manifest.current_schema_id, Some(manifest.last_column_id)).await?;
        println!("Schema Evolution: Moved column '{}' to index {} (Schema ID: {})", name, new_index, new_schema_id);
        Ok(())
    }

    /// Async implementation of write (Buffered) with Schema Validation
    pub async fn write_async(&self, batches: Vec<RecordBatch>) -> Result<()> {
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        INGEST_ROWS_TOTAL.inc_by(total_rows as u64);
        
        if batches.is_empty() {
             return Ok(());
        }

        let mut is_empty_schema = false;
        if let Some(first_batch) = batches.first() {
            let mut lock = self.schema.write().unwrap();
            is_empty_schema = lock.fields().is_empty();
            if is_empty_schema {
                *lock = first_batch.schema();
            }
            drop(lock);
        }

        let target_schema = self.arrow_schema();
        let batches: Vec<RecordBatch> = batches.into_iter().map(|b| {
            if b.schema() != target_schema {
                // Remap batch to target schema (adds iceberg.id metadata)
                RecordBatch::try_new(target_schema.clone(), b.columns().to_vec()).unwrap_or(b)
            } else {
                b
            }
        }).collect();

        if is_empty_schema {
            // Schema was empty, now we have detected column types from the data
            // This enables schema-on-write behavior
        }

        // 1. Write-Ahead Log (Durability) & 2. Indexing (In-Memory) in PARALLEL
        let wal = self.wal.clone();
        let memory_index = self.memory_index.clone();
        let target_col = self.index_columns.read().unwrap().first().cloned()
             .or_else(|| {
                 batches.first().and_then(|b| {
                     b.schema().fields().iter()
                         .find(|f| f.name() == "embedding")
                         .map(|f| f.name().clone())
                 })
             });

        let buffer_len_before = { 
            let buffer = self.write_buffer.read().unwrap();
            buffer.iter().map(|b| b.num_rows()).sum()
        };

        let batches_for_wal = batches.clone();
        let batches_for_idx = batches.clone();
        
        let (wal_res, idx_res) = tokio::join!(
            // WAL Task
            async move {
                let wal_lock = wal.lock().await;
                for batch in batches_for_wal {
                    wal_lock.append_async(batch).await?;
                }
                
                // Check if WAL needs compaction (sync check for now)
                if wal_lock.should_compact()? {
                }
                Ok::<(), anyhow::Error>(())
            },
            // Indexing Task
            async move {
                if let Some(col_name) = target_col {
                    let mut idx_lock = memory_index.write().unwrap();
                    
                    if idx_lock.is_none() {
                        if let Some(first) = batches_for_idx.first() {
                            if let Some(col) = first.column_by_name(&col_name) {
                                if let Some(fsl) = col.as_any().downcast_ref::<arrow::array::FixedSizeListArray>() {
                                     let dim = fsl.value_length() as usize;
                                     *idx_lock = Some(InMemoryVectorIndex::new(dim));
                                }
                                // Time32 Not Supported
                                /*
                                else if let Some(time32) = col.as_any().downcast_ref::<arrow::array::Time32Array>() {
                                    // This block is for indexing, not vector search, so dim is not applicable here.
                                    // It should be handled by the `insert_batch` logic if Time32/Time64 indexing is supported.
                                }
                                */
                                // Time64 Not Supported
                                /*
                                else if let Some(time64) = col.as_any().downcast_ref::<arrow::array::Time64Array>() {
                                }
                                */
                            }
                        }
                    }

                    if let Some(idx) = idx_lock.as_mut() {
                        let mut current_offset = buffer_len_before;
                        for batch in &batches_for_idx {
                            let _ = idx.insert_batch(batch, &col_name, current_offset);
                            current_offset += batch.num_rows();
                        }
                    }
                }
                Ok::<(), anyhow::Error>(())
            }
        );

        wal_res?;
        idx_res?;
        // -----------------------------

        // Buffer the batches
        {
            let mut buffer = self.write_buffer.write().unwrap();
            buffer.extend(batches);
        }

        // Check if we should flush (spillover)
        let should_flush = {
            let buffer = self.write_buffer.read().unwrap();
            
            // Calculate size in bytes (approximate)
            let total_bytes: usize = buffer.iter()
                .map(|b| b.get_array_memory_size())
                .sum();
            
            let cache_gb: usize = std::env::var("HYPERSTREAM_CACHE_GB")
                .unwrap_or_else(|_| "2".to_string())
                .parse()
                .unwrap_or(2);
            let limit_bytes = cache_gb * 1024 * 1024 * 1024;
            
            total_bytes > limit_bytes
        };

        if should_flush {
            println!("Write buffer exceeded limit. Flushing to disk (Spillover)...");
            self.flush_async().await?;
        }

        Ok(())
    }

    /// Flush buffer to disk
    /// Flush buffer to disk
    pub async fn flush_async(&self) -> Result<()> {
        // Extract batches from buffer
        let batches_to_write: Vec<RecordBatch> = {
            let mut buffer = self.write_buffer.write().unwrap();
            if buffer.is_empty() {
                return Ok(());
            }
            std::mem::take(&mut *buffer)
        };

        // Reset memory index
        {
            let mut idx = self.memory_index.write().unwrap();
            *idx = None;
        }

        if batches_to_write.is_empty() {
            return Ok(());
        }

        // --- NEW: Coalesce Batches for larger segments ---
        let schema = batches_to_write[0].schema();
        let coalesced_batch = arrow::compute::concat_batches(&schema, &batches_to_write)?;
        
        // Apply sort order if configured (Iceberg V2 spec compliance)
        let sorted_batch = self.apply_sort_order(&coalesced_batch)?;
        
        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, _, _) = manifest_manager.load_latest_full().await.unwrap_or_default();
        let spec = &manifest.partition_spec;
        
        // Add V3 metadata columns if format_version >= 3 (Iceberg V3 Row Lineage)
        let batch_with_metadata = if manifest.format_version >= 3 {
            let sequence_number = manifest.version as i64;
            self.add_v3_metadata_columns(&sorted_batch, sequence_number)?
        } else {
            sorted_batch.clone()
        };

        // Split batches by partition
        let partitioned_batches = self.split_batch_by_partition(&batch_with_metadata, spec)?;
        
        // Extract local path from URI for writer
        let base_path = self.uri.strip_prefix("file://").unwrap_or(&self.uri);
        std::fs::create_dir_all(base_path)?;
        
        let mut all_new_entries = Vec::new();
        let mut all_generated_files = Vec::new();
        let index_cols = self.index_columns.read().unwrap().clone();
        let index_all_flag = self.index_all;

        for (partition_values, batch) in partitioned_batches {
            let segment_id = format!("seg_{}", uuid::Uuid::new_v4());

            // 1. Create writer for data write (no index building yet)
            let config_write = SegmentConfig::new(base_path, &segment_id)
                .with_index_all(false)
                .with_columns_to_index(Vec::new())
                .with_partition_values(partition_values.clone());
            let writer_write = HybridSegmentWriter::new(config_write);

            // Copy needed data before spawn_blocking
            let batch_for_write = batch.clone();
            
            let (entry, generated_files) = tokio::task::spawn_blocking(move || {
                writer_write.write_batch(&batch_for_write)?;
                let entry = writer_write.to_manifest_entry();
                let files = writer_write.get_generated_files(); 
                Ok::<(crate::core::manifest::ManifestEntry, Vec<String>), anyhow::Error>((entry, files))
            }).await.context("Flush task panicked")??;
            
            all_generated_files.extend(generated_files);
            all_new_entries.push(entry.clone());

            // 2. Queue index building asynchronously (if needed)
            if index_all_flag || !index_cols.is_empty() {
                let index_cols_clone = index_cols.clone();
                let base_path_clone = base_path.to_string();
                let segment_id_clone = segment_id.clone();
                let batch_for_indexing = batch.clone();
                let partition_values_clone = partition_values.clone();
                
                let entry_clone = entry.clone();
                let manifest_manager_clone = manifest_manager.clone();
                let index_all_flag = index_all_flag;

                let handle = tokio::spawn(async move {
                    let _start = std::time::Instant::now();
                    
                    let config_index = SegmentConfig::new(&base_path_clone, &segment_id_clone)
                        .with_index_all(index_all_flag)
                        .with_columns_to_index(index_cols_clone)
                        .with_partition_values(partition_values_clone);
                    
                    let index_res = tokio::task::spawn_blocking(move || {
                        let index_writer = HybridSegmentWriter::new(config_index);
                        index_writer.build_indexes(&batch_for_indexing)?;
                        let files = index_writer.get_generated_files();
                        let updated_entry_info = index_writer.to_manifest_entry();
                        Ok::<(crate::core::manifest::ManifestEntry, Vec<String>), anyhow::Error>((updated_entry_info, files))
                    }).await;

                    match index_res {
                        Ok(Ok((updated_entry, _files))) => {
                            // Atomic metadata update: Add index files to the existing entry
                            // (Using commit logic that handles merges)
                            let mut merged_entry = entry_clone;
                            merged_entry.index_files = updated_entry.index_files;
                            merged_entry.column_stats = updated_entry.column_stats; 
                            
                            match manifest_manager_clone.commit(&[merged_entry], &[], crate::core::manifest::CommitMetadata::default()).await {
                                Ok(_) => println!("Successfully attached indexes to manifest for segment {}", 
                                                 segment_id_clone),
                                Err(e) => eprintln!("Failed to attach indexes for segment {}: {}", segment_id_clone, e),
                            }
                        }
                        _ => {
                            eprintln!("Index building failed for segment {}", segment_id_clone);
                        }
                    }
                });
                self.background_tasks.lock().await.push(handle);
            }
        }
        
        // Detect if schema has evolved since last manifest load
        let (manifest, _, _) = manifest_manager.load_latest_full().await.unwrap_or_default();
        let current_schema = self.arrow_schema();
        
        let should_update_schema = if manifest.schemas.is_empty() {
            true
        } else {
            let latest_schema = manifest.schemas.last().unwrap().to_arrow();
            // Compare schemas, but ignore metadata if necessary. 
            // Simple != check works for basic evolution.
            latest_schema != *current_schema
        };

        let (final_schemas, final_schema_id) = if should_update_schema {
            let mut new_schemas = manifest.schemas.clone();
            let new_id = if manifest.schemas.is_empty() { 0 } else { manifest.current_schema_id + 1 };
            new_schemas.push(crate::core::manifest::Schema::from_arrow(&current_schema, new_id));
            (Some(new_schemas), Some(new_id))
        } else {
            (None, None)
        };

        // Final commit for all data segments (with possible schema update)
        let commit_metadata = crate::core::manifest::CommitMetadata {
            updated_schemas: final_schemas,
            updated_schema_id: final_schema_id,
            ..Default::default()
        };
        let new_manifest = manifest_manager.commit(&all_new_entries, &[], commit_metadata).await?;

        // 4. Update Table Metadata (Iceberg v2 Spec)
        // Determine the root for metadata. If we have a catalog, use its reported location.
        let meta_location = if let Some(catalog) = &self.catalog {
            if let (Some(ns), Some(t)) = (&self.catalog_namespace, &self.catalog_table_name) {
                catalog.load_table(ns, t).await.map(|m| m.location).unwrap_or_else(|_| self.uri.clone())
            } else {
                self.uri.clone()
            }
        } else {
            self.uri.clone()
        };

        let meta_store_arc = create_object_store(&meta_location)?;
        let meta_store = meta_store_arc.as_ref();
        
        let mut table_meta = match TableMetadata::load_latest(meta_store).await {
            Ok(meta) => meta,
            Err(_) => {
                // Initialize skeleton if not found
                TableMetadata::new(
                    2, 
                    uuid::Uuid::new_v4().to_string(), 
                    self.uri.clone(), 
                    new_manifest.schemas.last().cloned().unwrap_or_else(|| {
                        crate::core::manifest::Schema::from_arrow(&current_schema, 0)
                    }), 
                    new_manifest.partition_spec.clone(),
                    new_manifest.sort_orders.first().cloned().unwrap_or_default()
                )
            }
        };

        // Add a new snapshot pointing to the latest manifest
        let snapshot = crate::core::metadata::Snapshot {
            snapshot_id: new_manifest.version as i64,
            parent_snapshot_id: table_meta.current_snapshot_id,
            timestamp_ms: new_manifest.timestamp_ms,
            sequence_number: Some(new_manifest.version as i64),
            summary: HashMap::from([("operation".to_string(), "append".to_string())]),
            manifest_list: new_manifest.manifest_list_path.clone().unwrap_or_default(),
            schema_id: Some(new_manifest.current_schema_id),
            first_row_id: table_meta.next_row_id,
            added_rows: None, // TODO: Calculate from manifest
        };
        table_meta.add_snapshot(snapshot);
        
        // Save metadata file (vX.metadata.json)
        let new_meta_version = (new_manifest.version) as i32; // Sync with manifest version for simplicity
        table_meta.save_to_store(meta_store, new_meta_version).await?;
        
        // 5. Commit to Catalog if configured (Iceberg Atomic Swap)
        if let Some(catalog) = &self.catalog {
            if let (Some(ns), Some(table)) = (&self.catalog_namespace, &self.catalog_table_name) {
                let updates = vec![
                    serde_json::json!({
                        "action": "add-snapshot",
                        "snapshot": table_meta.snapshots.last().unwrap()
                    }),
                    serde_json::json!({
                        "action": "set-current-snapshot",
                        "snapshot-id": table_meta.current_snapshot_id
                    })
                ];
                catalog.commit_table(ns, table, updates).await?;
                println!("Committed snapshot {} to catalog {}.{}", new_manifest.version, ns, table);
            }
        }

        // 3. Upload data files asynchronously if remote
        if self.uri.contains("://") && !self.uri.starts_with("file://") {
             let store_clone = self.store.clone();
             let files_to_upload = all_generated_files;
             let handle = tokio::spawn(async move {
                 for file_path in files_to_upload {
                      let local_path = std::path::Path::new(file_path.as_str());
                      if let Ok(data) = tokio::fs::read(&local_path).await {
                           let remote_path = object_store::path::Path::from(file_path.as_str());
                           let _ = store_clone.put(&remote_path, data.into()).await;
                      }
                 }
             });
             self.background_tasks.lock().await.push(handle);
        }
        
        // 5. Truncate WAL (Durability Checkpoint)
        {
             let mut wal = self.wal.lock().await;
             wal.truncate().context("Failed to truncate WAL")?;
        }

        Ok(())
    }

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

    /// Delete rows matching the filter (Merge-on-Read)
    pub fn delete(&self, filter: &str) -> Result<()> {
        self.runtime().block_on(self.delete_async(filter))
    }

    /// Async implementation of delete
/// Async implementation of delete
pub async fn delete_async(&self, filter: &str) -> Result<()> {
    use futures::StreamExt;
    // use roaring::RoaringBitmap; // No longer needed for persistence, only temp
    
    let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
    let (_manifest, all_entries, _) = manifest_manager.load_latest_full().await?;
    
    if all_entries.is_empty() {
        return Ok(());
    }

    let planner = QueryPlanner::new();
    let arrow_schema = self.arrow_schema();
    let expr = FilterExpr::parse_sql(filter, arrow_schema).await.context("Failed to parse delete filter")?;
    
    let candidates = planner.prune_entries(&all_entries, Some(&expr));
    // println!("Delete Plan: Scanning {} candidate segments for deletions.", candidates.len());

    let mut all_new_entries = Vec::new();

    for (entry, _index_file) in candidates {
        let file_path_str = entry.file_path.clone();
        let segment_id = file_path_str.split('/').last().unwrap_or(&file_path_str)
            .strip_suffix(".parquet").unwrap_or(&file_path_str);
        
        // Critical: We MUST scan WITHOUT applying existing deletes to get correct absolute Row IDs.
        // We create a reader that only sees the base data file.
        let config = SegmentConfig::new(&self.uri, segment_id); 
        // Note: .with_delete_files() is NOT called, so existing deletes are ignored during scan.

        let reader = HybridReader::new(config, self.store.clone(), &self.uri);
        
        let mut new_deletes = Vec::new();
        let mut current_row_offset = 0;
        
            // Scan full file to find absolute positions matching the filter
        let mut stream = reader.stream_all(None).await?;
        
        while let Some(batch_res) = stream.next().await {
            let batch = batch_res?;
            let num_rows = batch.num_rows();
            
            // Evaluate filter on this batch
            let mask = planner.evaluate_expr(&batch, &expr)?;
            
            for i in 0..num_rows {
                if mask.value(i) {
                    new_deletes.push((current_row_offset + i) as i64);
                }
            }
            
            current_row_offset += num_rows;
        }
        
        if !new_deletes.is_empty() {
            println!("Segment {}: Found {} NEW rows to delete.", segment_id, new_deletes.len());
            
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
                // Determine relative partition path from data file path
                // Heuristic: remove filename, treat remainder relative to table uri as partition path
                let path = std::path::Path::new(&entry.file_path);
                let parent = path.parent().unwrap(); 
                let parent_str = parent.to_str().unwrap();
                
                // Attempt to strip base uri
                // Handle "file://" prefix difference
                let base_clean = self.uri.replace("file://", "");
                let parent_clean = parent_str.replace("file://", "");
                
                let rel_path = if parent_clean.starts_with(&base_clean) {
                    parent_clean.strip_prefix(&base_clean).unwrap_or("").trim_start_matches('/').to_string()
                } else {
                    // Fallback: just use directory name (works for single level)
                    // Or if different store, we might be in trouble, but let's assume valid structure.
                    parent.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string()
                };

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
            all_new_entries.push(new_entry);
        } else {
            all_new_entries.push(entry.clone());
        }
    }

    if !all_new_entries.is_empty() {
        // Commit Update
        manifest_manager.commit(&all_new_entries, &[], crate::core::manifest::CommitMetadata::default()).await?;
    }

    Ok(())
}

    /// Merge (Upsert) batches into the table
    pub fn merge(&self, batches: Vec<RecordBatch>, key_column: &str, mode: MergeMode) -> Result<()> {
        match mode {
            MergeMode::MergeOnRead => self.merge_on_read(batches, key_column),
            MergeMode::MergeOnWrite => self.merge_on_write(batches, key_column),
        }
    }

    fn merge_on_read(&self, batches: Vec<RecordBatch>, key_column: &str) -> Result<()> {
        self.runtime().block_on(async {
            // For MoR:
            // 1. Write the new batches as new segments (Append)
            // 2. Identify the keys being updated and delete them from old segments
            
            // Step 1: Delete existing keys
            // This is complex because we need to delete specific keys, not a range.
            // For now, we'll do it naively: for each key in the batch, call delete("column = key")
            // Optimization: Batch the deletes using an IN clause (if supported) or manual logic.
            
            for batch in &batches {
                let schema = batch.schema();
                let col_idx = schema.index_of(key_column)?;
                let col = batch.column(col_idx);
                
                // For simplicity, handle Int32/Int64 keys for integration tests
                if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int32Array>() {
                    for i in 0..arr.len() {
                        let filter = format!("{} = {}", key_column, arr.value(i));
                        self.delete_async(&filter).await?;
                    }
                } else if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int64Array>() {
                    for i in 0..arr.len() {
                        let filter = format!("{} = {}", key_column, arr.value(i));
                        self.delete_async(&filter).await?;
                    }
                }
            }

            // Step 2: Write new batches
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
                e.file_path.split('/').last().unwrap().replace(".parquet", "")
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
                    file_size_bytes: 0, // Unknown
                    record_count: 0,
                    index_files: vec![],
                    delete_files: vec![],
                    column_stats: std::collections::HashMap::new(),
                    partition_values: std::collections::HashMap::new(),
                    clustering_strategy: None,
                    clustering_columns: None,
                    min_clustering_score: None,
                    max_clustering_score: None,
                    normalization_mins: None,
                    normalization_maxs: None,
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

/// Vector search parameters
#[derive(Debug, Clone)]
pub struct VectorSearchParams {
    pub column: String,
    pub query: crate::core::index::VectorValue,
    pub k: usize,
    pub metric: VectorMetric,
    pub ef_search: Option<usize>,
    pub probes: Option<usize>,
}

impl VectorSearchParams {
    pub fn new(column: &str, query: crate::core::index::VectorValue, k: usize) -> Self {
        Self {
            column: column.to_string(),
            query,
            k,
            metric: VectorMetric::L2,
            ef_search: None,
            probes: None,
        }
    }
    
    pub fn with_metric(mut self, metric: VectorMetric) -> Self {
        self.metric = metric;
        self
    }

    pub fn with_ef_search(mut self, ef_search: usize) -> Self {
        self.ef_search = Some(ef_search);
        self
    }

    pub fn with_probes(mut self, probes: usize) -> Self {
        self.probes = Some(probes);
        self
    }
}

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
                     match reader.query_index_first(&qf, target_schema.clone()).await {
                         Ok(indexed_batches) => {
                             batches = indexed_batches;
                             index_used = true;
                         },
                         Err(_) => {}
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

    /// Helper to shard a RecordBatch into multiple batches based on PartitionSpec
    fn split_batch_by_partition(
        &self, 
        batch: &RecordBatch, 
        spec: &crate::core::manifest::PartitionSpec
    ) -> Result<Vec<(HashMap<String, serde_json::Value>, RecordBatch)>> {
        use arrow::compute::take;
        use arrow::array::UInt32Array;
        use std::collections::BTreeMap;
        
        if spec.fields.is_empty() {
             return Ok(vec![(HashMap::new(), batch.clone())]);
        }

        let mut row_groups: HashMap<String, Vec<u32>> = HashMap::new();
        let mut key_cache: HashMap<String, HashMap<String, serde_json::Value>> = HashMap::new();
        
        for row_i in 0..batch.num_rows() {
            let mut key_map = BTreeMap::new();
            for field in &spec.fields {
                // Find columns by source_ids (Iceberg Field IDs)
                let source_ids = field.get_source_ids();
                let mut col_indices = Vec::new();
                
                for id in &source_ids {
                    let idx = batch.schema().fields().iter().position(|f| {
                        f.metadata().get("iceberg.id")
                            .and_then(|id_str| id_str.parse::<i32>().ok())
                            .map(|found_id| found_id == *id)
                            .unwrap_or(false)
                    });
                    if let Some(i) = idx {
                        col_indices.push(i);
                    }
                }

                // Fallback to name-based if no source_ids matched and field has a name
                if col_indices.is_empty() {
                    if let Ok(idx) = batch.schema().index_of(&field.name) {
                        col_indices.push(idx);
                    }
                }

                let val = if !col_indices.is_empty() {
                    let arrays: Vec<&dyn arrow::array::Array> = col_indices.iter()
                        .map(|idx| batch.column(*idx).as_ref())
                        .collect();
                    let transform = crate::core::iceberg::IcebergTransform::parse(&field.transform);
                    transform.apply_multi(&arrays, row_i)
                } else {
                    serde_json::Value::Null
                };
                
                key_map.insert(field.name.clone(), val);
            }
            let key_str = serde_json::to_string(&key_map)?;
            row_groups.entry(key_str.clone()).or_default().push(row_i as u32);
            if !key_cache.contains_key(&key_str) {
                let mut final_key = HashMap::new();
                for (k, v) in key_map {
                    final_key.insert(k, v);
                }
                key_cache.insert(key_str, final_key);
            }
        }
        
        let mut result = Vec::new();
        for (key_str, row_indices) in row_groups {
            let key = key_cache.remove(&key_str).unwrap();
            let indices = UInt32Array::from(row_indices);
            let sub_batch = RecordBatch::try_new(
                batch.schema(),
                batch.columns().iter()
                    .map(|c| take(c.as_ref(), &indices, None))
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| anyhow::anyhow!("Arrow take failed: {}", e))?
            )?;
            result.push((key, sub_batch));
        }
        
        Ok(result)
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
        let table = Table::new_async(uri.clone()).await?;

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
        let results = table.split_batch_by_partition(&batch, &spec)?;
        
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
}
