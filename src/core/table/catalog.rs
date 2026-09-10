// Copyright (c) 2026 Richard Albright. All rights reserved.

/// External catalog integrations and Iceberg table registration.
///
/// Contains methods on `Table` for:
/// - `from_nessie`, `from_glue`, `from_hive`
/// - `register_external`, `detect_iceberg_rest`, `new_from_rest`
/// - `spawn_iceberg_observer`, `check_and_import_new_snapshot`, `import_iceberg_snapshot`
use anyhow::{Context, Result};
use arrow::datatypes::SchemaRef;
use object_store::ObjectStore;
use std::sync::Arc;

use super::Table;
use crate::core::manifest::{Manifest, ManifestManager};
use crate::core::storage::create_object_store;
use crate::core::table::builder::TableBuilder;

impl Table {
    /// Load a table from Nessie Catalog at specific branch/tag/hash
    pub async fn from_nessie(
        nessie_config: crate::core::nessie::NessieConfig,
        namespace: &str,
        table: &str,
        _ref_hash: Option<String>,
    ) -> Result<Self> {
        use crate::core::catalog::Catalog;
        let client = crate::core::catalog::nessie::NessieClient::new(nessie_config.uri.clone());
        let metadata = client.load_table(namespace, table).await?;

        // We use a local path for the HyperStreamDB artifacts for this external table
        let local_uri = format!("file:///tmp/hyperstream_nessie_{}_{}", namespace, table);

        tracing::info!(
            "Resolved Nessie table {} to metadata: {}",
            table,
            metadata.location
        );

        let mut table_obj = Self::register_external(local_uri, &metadata.location).await?;
        table_obj.catalog_state.catalog = Some(Arc::new(client));
        table_obj.catalog_state.namespace = Some(namespace.to_string());
        table_obj.catalog_state.table_name = Some(table.to_string());
        Ok(table_obj)
    }

    /// Load a table from AWS Glue Catalog
    pub async fn from_glue(
        catalog_id: Option<String>,
        namespace: &str,
        table: &str,
    ) -> Result<Self> {
        use crate::core::catalog::Catalog;
        let client = crate::core::catalog::glue::GlueCatalogClient::new(catalog_id).await?;
        let metadata = client.load_table(namespace, table).await?;

        // Use local path for artifacts
        let local_uri = format!("file:///tmp/hyperstream_glue_{}_{}", namespace, table);
        tracing::info!(
            "Resolved Glue table {}.{} to metadata: {}",
            namespace,
            table,
            metadata.location
        );

        let mut table_obj = Self::register_external(local_uri, &metadata.location).await?;
        table_obj.catalog_state.catalog = Some(Arc::new(client));
        table_obj.catalog_state.namespace = Some(namespace.to_string());
        table_obj.catalog_state.table_name = Some(table.to_string());
        Ok(table_obj)
    }

    /// Load a table from Hive Metastore
    /// address: "host:port" or "thrift://host:port"
    pub async fn from_hive(address: &str, namespace: &str, table: &str) -> Result<Self> {
        use crate::core::catalog::Catalog;
        let client = crate::core::catalog::hive::HiveMetastoreClient::new(address.to_string())?;
        let metadata = client.load_table(namespace, table).await?;

        let local_uri = format!("file:///tmp/hyperstream_hive_{}_{}", namespace, table);
        tracing::info!(
            "Resolved Hive table {}.{} to metadata: {}",
            namespace,
            table,
            metadata.location
        );

        let mut table_obj = Self::register_external(local_uri, &metadata.location).await?;
        table_obj.catalog_state.catalog = Some(Arc::new(client));
        table_obj.catalog_state.namespace = Some(namespace.to_string());
        table_obj.catalog_state.table_name = Some(table.to_string());
        Ok(table_obj)
    }

    /// Register an existing Iceberg table for Layered Indexing
    pub async fn register_external(uri: String, iceberg_metadata_uri: &str) -> Result<Self> {
        create_object_store(&uri)?;

        // Determine if iceberg_metadata_uri is a directory (table location) or a file (metadata file)
        let (meta_store_uri, filename) = {
            let meta_url =
                url::Url::parse(iceberg_metadata_uri).context("Invalid iceberg_metadata_uri")?;
            let meta_path_str = meta_url.path();
            let meta_path = std::path::Path::new(meta_path_str);

            let path_str = meta_path_str.to_string();
            if path_str.ends_with(".metadata.json")
                || path_str.ends_with(".json") && path_str.contains("metadata")
            {
                // It's a metadata file path
                let parent_dir = meta_path
                    .parent()
                    .context("No parent directory for metadata file")?;
                let fname = meta_path
                    .file_name()
                    .context("No filename for metadata file")?
                    .to_str()
                    .ok_or_else(|| anyhow::anyhow!("Non-UTF8 metadata filename"))?
                    .to_string();

                let store_uri = if iceberg_metadata_uri.starts_with("file://") {
                    format!("file://{}", parent_dir.display())
                } else {
                    let mut base_url = meta_url.clone();
                    if let Some(parent_str) = parent_dir.to_str() {
                        base_url.set_path(parent_str);
                    }
                    base_url.to_string()
                };
                (store_uri, fname)
            } else {
                // It's a table directory, look for metadata file inside metadata/ subdirectory
                let metadata_dir = if iceberg_metadata_uri.starts_with("file://") {
                    format!("file://{}/metadata", meta_path_str)
                } else {
                    format!("{}/metadata", iceberg_metadata_uri.trim_end_matches('/'))
                };

                tracing::debug!(
                    "Table location appears to be a directory. Looking for metadata in: {}",
                    metadata_dir
                );

                (metadata_dir, "v1.metadata.json".to_string())
            }
        };

        let iceberg_meta_store = create_object_store(&meta_store_uri)?;

        tracing::info!("Linking external Iceberg table: {}", iceberg_metadata_uri);

        // 1. Load Iceberg Metadata
        let path = object_store::path::Path::from(filename.as_str());
        let ret = iceberg_meta_store.get(&path).await?;
        let bytes = ret.bytes().await?;
        let iceberg_meta: crate::core::iceberg::IcebergTableMetadata =
            serde_json::from_slice(&bytes)?;

        // 2. Map Iceberg Schema to HyperStreamDB Schema
        let current_schema_json = iceberg_meta
            .schemas
            .iter()
            .find(|s| {
                s.get("schema-id").and_then(|v| v.as_i64())
                    == Some(iceberg_meta.current_schema_id as i64)
            })
            .unwrap_or_else(|| {
                tracing::warn!(
                    "Could not find schema with ID {}, falling back to first schema",
                    iceberg_meta.current_schema_id
                );
                &iceberg_meta.schemas[0]
            });
        let hdb_schema: crate::core::manifest::Schema =
            serde_json::from_value(current_schema_json.clone())?;
        let schema_ref: SchemaRef = Arc::new(hdb_schema.to_arrow());

        // 3. Initialize HyperStream Table
        let mut table = Self::create_async(uri.clone(), schema_ref.clone()).await?;

        // Root data_store carefully: if file, root at / to support absolute paths in manifests.
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
            table
                .import_iceberg_snapshot(snapshot_id, &iceberg_meta, iceberg_meta_store)
                .await?;
        }

        Ok(table)
    }

    /// Detect if a URI is an Iceberg REST Catalog endpoint
    pub(crate) fn detect_iceberg_rest(
        uri: &str,
    ) -> Option<(String, Option<String>, String, String)> {
        if !uri.starts_with("http") {
            return None;
        }

        let ns_marker = "/namespaces/";
        let t_marker = "/tables/";

        let ns_idx = uri.find(ns_marker)?;
        let t_idx = uri.find(t_marker)?;

        if ns_idx >= t_idx {
            return None;
        }

        // Find /v1/ which precedes namespaces
        let v1_idx = uri.find("/v1")?;
        if v1_idx >= ns_idx {
            return None;
        }

        let base_url = uri[..v1_idx].to_string();

        let prefix_start = v1_idx + 3;
        let prefix_end = ns_idx;
        let prefix = if prefix_end > prefix_start {
            let p = &uri[prefix_start..prefix_end];
            let trimmed = p.trim_matches('/');
            if trimmed.is_empty() {
                None
            } else {
                Some(trimmed.to_string())
            }
        } else {
            None
        };

        let namespace = uri[ns_idx + ns_marker.len()..t_idx].to_string();
        let table_name = uri[t_idx + t_marker.len()..].to_string();

        Some((base_url, prefix, namespace, table_name))
    }

    pub(crate) async fn new_from_rest(
        base_url: String,
        prefix: Option<String>,
        namespace: String,
        table_name: String,
        rest_uri: &str,
    ) -> Result<Self> {
        use crate::core::catalog::rest::RestCatalogClient;
        use crate::core::catalog::Catalog;

        let client = RestCatalogClient::new(base_url, prefix);
        let metadata = client.load_table(&namespace, &table_name).await?;

        // Derive local native URI (Cache location for layered index)
        let cache_dir = std::env::var("HYPERSTREAM_CACHE_DIR")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| std::env::temp_dir().join("hyperstream_cache"));

        if !cache_dir.exists() {
            let _ = std::fs::create_dir_all(&cache_dir);
        }

        let safe_name = rest_uri
            .replace("://", "_")
            .replace("/", "_")
            .replace(":", "_");
        let native_uri_path = cache_dir.join(safe_name);
        let native_uri = format!("file://{}", native_uri_path.display());

        let store = create_object_store(&native_uri)?;

        // Check if already registered
        let manager = ManifestManager::new(store.clone(), "", &native_uri);
        let (_, version) = manager
            .load_latest()
            .await
            .unwrap_or((Manifest::default(), 0));

        if version > 0 {
            let mut table = TableBuilder::new(native_uri)
                .with_index_all(false)
                .build_async()
                .await?;
            table.catalog_state.catalog = Some(Arc::new(client));
            table.catalog_state.namespace = Some(namespace);
            table.catalog_state.table_name = Some(table_name);
            Ok(table)
        } else {
            tracing::debug!(
                "Checking if warehouse location is a HyperStreamDB table: {}",
                metadata.location
            );

            match create_object_store(&metadata.location) {
                Ok(warehouse_store) => {
                    let warehouse_manager =
                        ManifestManager::new(warehouse_store, "", &metadata.location);

                    if let Ok((_, warehouse_version)) = warehouse_manager.load_latest().await {
                        if warehouse_version > 0 {
                            tracing::info!(
                                "✅ Opening existing HyperStreamDB table from REST catalog: {}",
                                metadata.location
                            );
                            let mut table = TableBuilder::new(metadata.location)
                                .with_index_all(false)
                                .build_async()
                                .await?;
                            table.catalog_state.catalog = Some(Arc::new(client));
                            table.catalog_state.namespace = Some(namespace);
                            table.catalog_state.table_name = Some(table_name);
                            return Ok(table);
                        } else {
                            tracing::info!(
                                "✅ Creating new HyperStreamDB table at warehouse location: {}",
                                metadata.location
                            );
                            let current_schema = metadata
                                .schemas
                                .iter()
                                .find(|s| s.schema_id == metadata.current_schema_id)
                                .or_else(|| metadata.schemas.last())
                                .ok_or_else(|| {
                                    anyhow::anyhow!("No schema found in table metadata")
                                })?;
                            let schema_ref = Arc::new(current_schema.to_arrow());
                            let table =
                                Self::create_async(metadata.location.clone(), schema_ref).await?;
                            return Ok(table);
                        }
                    } else {
                        tracing::debug!("ℹ️  Warehouse location has no manifest, trying external Iceberg import or creating new table");
                    }
                }
                Err(e) => {
                    tracing::debug!("ℹ️  Could not access warehouse location: {}. Trying external Iceberg import.", e);
                }
            }

            tracing::info!(
                "Auto-registering Iceberg table from REST catalog: {}",
                rest_uri
            );
            let mut table = Self::register_external(native_uri, &metadata.location).await?;

            table.catalog_state.catalog = Some(Arc::new(client));
            table.catalog_state.namespace = Some(namespace);
            table.catalog_state.table_name = Some(table_name);
            Ok(table)
        }
    }

    /// Start a background observer to watch an external Iceberg table for changes
    pub async fn spawn_iceberg_observer(
        &self,
        iceberg_metadata_uri: String,
        interval: std::time::Duration,
    ) -> Result<()> {
        let table = self.clone();

        let handle = tokio::spawn(async move {
            let mut last_processed_snapshot = None;

            loop {
                match table
                    .check_and_import_new_snapshot(
                        &iceberg_metadata_uri,
                        &mut last_processed_snapshot,
                    )
                    .await
                {
                    Ok(true) => tracing::debug!("Snapshot Observer: New snapshot processed."),
                    Ok(false) => {}
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
        last_snapshot_id: &mut Option<i64>,
    ) -> Result<bool> {
        let meta_url =
            url::Url::parse(iceberg_metadata_uri).context("Invalid iceberg_metadata_uri")?;
        let meta_path_str = meta_url.path();
        let meta_path_obj = std::path::Path::new(meta_path_str);
        let parent_dir = meta_path_obj.parent().context("No parent directory")?;
        let filename = meta_path_obj
            .file_name()
            .context("No filename")?
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Non-UTF8 metadata filename"))?;

        let meta_store_uri = if iceberg_metadata_uri.starts_with("file://") {
            format!("file://{}", parent_dir.display())
        } else {
            let mut base_url = meta_url.clone();
            if let Some(parent_str) = parent_dir.to_str() {
                base_url.set_path(parent_str);
            }
            base_url.to_string()
        };

        let iceberg_meta_store = create_object_store(&meta_store_uri)?;
        let path = object_store::path::Path::from(filename);
        let ret = iceberg_meta_store.get(&path).await?;
        let bytes = ret.bytes().await?;
        let iceberg_meta: crate::core::iceberg::IcebergTableMetadata =
            serde_json::from_slice(&bytes)?;

        if let Some(current_id) = iceberg_meta.current_snapshot_id {
            if Some(current_id) != *last_snapshot_id {
                self.import_iceberg_snapshot(current_id, &iceberg_meta, iceberg_meta_store)
                    .await?;
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
        iceberg_store: Arc<dyn ObjectStore>,
    ) -> Result<()> {
        let snapshot = meta
            .snapshots
            .iter()
            .find(|s| s.snapshot_id == snapshot_id)
            .context("Snapshot not found")?;

        tracing::info!("Importing Iceberg Snapshot {}...", snapshot_id);

        // Load Manifest List
        let ml_path_str_clean = if let Ok(url) = url::Url::parse(&snapshot.manifest_list) {
            let path = url.path();
            if let Ok(loc_url) = url::Url::parse(&meta.location) {
                let loc_path = format!("{}/metadata/", loc_url.path().trim_end_matches('/'));
                if path.starts_with(&loc_path) {
                    path.strip_prefix(&loc_path).unwrap_or(path).to_string()
                } else {
                    path.trim_start_matches('/').to_string()
                }
            } else {
                path.trim_start_matches('/').to_string()
            }
        } else {
            snapshot
                .manifest_list
                .trim_start_matches("file://")
                .to_string()
        };
        let ml_path = object_store::path::Path::from(ml_path_str_clean.as_str());

        let ret = iceberg_store.get(&ml_path).await?;
        let bytes = ret.bytes().await?;
        let manifest_list = crate::core::iceberg::read_manifest_list(&bytes[..])?;

        let mut all_entries = Vec::new();

        for ml_entry in manifest_list {
            let m_path_str_clean = if let Ok(url) = url::Url::parse(&ml_entry.manifest_path) {
                let path = url.path();
                if let Ok(loc_url) = url::Url::parse(&meta.location) {
                    let loc_path = format!("{}/metadata/", loc_url.path().trim_end_matches('/'));
                    if path.starts_with(&loc_path) {
                        path.strip_prefix(&loc_path).unwrap_or(path).to_string()
                    } else {
                        path.trim_start_matches('/').to_string()
                    }
                } else {
                    path.trim_start_matches('/').to_string()
                }
            } else {
                ml_entry
                    .manifest_path
                    .trim_start_matches("file://")
                    .to_string()
            };
            let m_path = object_store::path::Path::from(m_path_str_clean.as_str());
            let ret = iceberg_store.get(&m_path).await?;
            let bytes = ret.bytes().await?;
            let manifest = crate::core::iceberg::read_manifest(&bytes[..])?;

            let iceberg_schema = crate::core::manifest::Schema {
                schema_id: meta.current_schema_id,
                fields: meta
                    .schemas
                    .iter()
                    .find(|s| {
                        s["schema-id"].as_i64().map(|id| id as i32) == Some(meta.current_schema_id)
                    })
                    .map(|s| {
                        s["fields"]
                            .as_array()
                            .unwrap_or(&Vec::new())
                            .iter()
                            .map(|f| crate::core::manifest::SchemaField {
                                id: f["id"].as_i64().unwrap_or(0) as i32,
                                name: f["name"].as_str().unwrap_or("").to_string(),
                                type_str: f["type"].clone().to_string().replace('\"', ""),
                                required: f["required"].as_bool().unwrap_or(false),
                                fields: Vec::new(),
                                initial_default: None,
                                write_default: None,
                                indexes: Vec::new(),
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
                identifier_field_ids: Vec::new(),
            };

            let iceberg_spec = meta
                .partition_specs
                .iter()
                .find(|s| s["spec-id"].as_i64().map(|id| id as i32) == Some(meta.default_spec_id))
                .and_then(|s| crate::core::iceberg::iceberg_partition_spec_to_hyperstream(s).ok())
                .unwrap_or_default();

            let mut data_entries = Vec::new();
            let mut delete_files = Vec::new();

            for entry in manifest {
                if entry.status != 2 {
                    match crate::core::iceberg::convert_iceberg_to_object(
                        &entry,
                        &iceberg_schema,
                        &iceberg_spec,
                    ) {
                        Ok(crate::core::iceberg::IcebergManifestObject::Data(me)) => {
                            data_entries.push(*me);
                        }
                        Ok(crate::core::iceberg::IcebergManifestObject::Delete(df)) => {
                            delete_files.push(df);
                        }
                        Err(e) => tracing::warn!("Error converting Iceberg entry: {}", e),
                    }
                }
            }

            for df in delete_files {
                for data in &mut data_entries {
                    if data.partition_values == df.partition_values {
                        data.delete_files.push(df.clone());
                    }
                }
            }
            all_entries.extend(data_entries);
        }

        let manifest_manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        manifest_manager
            .commit_imported_entries(all_entries)
            .await?;

        Ok(())
    }
}
