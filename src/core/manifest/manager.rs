// Copyright (c) 2026 Richard Albright. All rights reserved.

use crate::core::cache::CacheExt;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use object_store::{path::Path, ObjectStore};
use anyhow::Result;
use futures::StreamExt;
use chrono::Utc;
use tracing;
use arrow::record_batch::RecordBatch;
use arrow::array::Array;

use super::types::*;
#[derive(Clone)]
pub struct ManifestManager {
    store: Arc<dyn ObjectStore>,
    manifest_dir: Path,
    root_uri: String,
}

#[derive(Debug, Default, Clone)]
pub struct CommitMetadata {
    pub updated_schemas: Option<Vec<Schema>>,
    pub updated_schema_id: Option<i32>,
    pub updated_partition_specs: Option<Vec<PartitionSpec>>,
    pub updated_default_spec_id: Option<i32>,
    pub updated_properties: Option<HashMap<String, String>>,
    pub removed_properties: Option<Vec<String>>,
    pub updated_sort_orders: Option<Vec<SortOrder>>,
    pub updated_default_sort_order_id: Option<i32>,
    pub updated_last_column_id: Option<i32>,
    pub is_fast_append: bool,
}

impl ManifestManager {
    pub fn new(store: Arc<dyn ObjectStore>, base_path: &str, root_uri: &str) -> Self {
        // Manifest directory is typically `_manifest/` under the table root
        let manifest_dir = if base_path.is_empty() {
             Path::from("_manifest/")
        } else {
             Path::from(format!("{}/_manifest/", base_path))
        };
        
        Self {
            store,
            manifest_dir,
            root_uri: root_uri.trim_end_matches('/').to_string(),
        }
    }

    fn get_cache_key(&self, path: &Path) -> String {
        format!("{}/{}", self.root_uri, path)
    }

    fn get_dir_cache_key(&self) -> String {
        let mut key = format!("{}/{}", self.root_uri, self.manifest_dir);
        while key.ends_with('/') {
            key.pop();
        }
        key
    }

    /// Check if any manifests exist in the directory
    pub async fn exists(&self) -> Result<bool> {
        let mut stream = self.store.list(Some(&self.manifest_dir));
        if let Some(meta) = stream.next().await {
            let _n = meta?.location.as_ref().len();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Load the latest manifest bypassing LATEST_VERSION_CACHE.
    pub async fn load_latest_direct(&self) -> Result<(Manifest, u64)> {
        let mut max_ver = 0;
        let mut latest_loc = None;
        
        let mut stream = self.store.list(Some(&self.manifest_dir));
        while let Some(meta) = stream.next().await {
            let meta = meta?;
            let path = meta.location.as_ref();
            if path.ends_with(".json") {
                if let Some(filename) = path.split('/').last() {
                    if filename.starts_with('v') && filename.ends_with(".json") {
                        if let Ok(ver) = filename[1..filename.len()-5].parse::<u64>() {
                            if ver >= max_ver {
                                max_ver = ver;
                                latest_loc = Some(meta.location);
                            }
                        }
                    }
                }
            }
        }

        if let Some(loc) = latest_loc {
             let manifest_bytes = self.store.get(&loc).await?.bytes().await?;
             let manifest: Manifest = serde_json::from_slice(&manifest_bytes)?;
             Ok((manifest, max_ver))
        } else {
             Ok((Manifest::default(), 0))
        }
    }

    /// Load the latest manifest. Returns (Manifest, version_number).
    /// If no manifest exists, returns an empty Manifest with version 0.
    pub async fn load_latest(&self) -> Result<(Manifest, u64)> {
        let cache_key = self.get_dir_cache_key();

        // 1. Check Version Cache (Fast Path)
        if let Some(ver) = crate::core::cache::LATEST_VERSION_CACHE.get_with_metrics(&cache_key, "latest_version").await {
            tracing::debug!("ManifestManager::load_latest: Found version {} in LATEST_VERSION_CACHE", ver);
            if let Ok(manifest) = self.load_version(ver).await {
                tracing::debug!("ManifestManager::load_latest: Cache hit v{} (entries={})", ver, manifest.entries.len());
                return Ok((manifest, ver));
            }
        }

        // 2. Slow Path: List files in _manifest/
        let mut stream = self.store.list(Some(&self.manifest_dir));
        let mut max_ver = 0;
        let mut latest_path = None;

        while let Some(meta) = stream.next().await {
            let meta = meta?;
            let path_str = meta.location.to_string();
            // Expected format: v{N}.json
            if let Some(filename) = path_str.split('/').next_back() {
                if filename.starts_with('v') && filename.ends_with(".json") {
                    let ver_str = &filename[1..filename.len()-5]; // strip 'v' and '.json'
                    if let Ok(ver) = ver_str.parse::<u64>() {
                        if ver > max_ver {
                            max_ver = ver;
                            latest_path = Some(meta.location);
                        }
                    }
                }
            }
        }

        // Update Version Cache
        if max_ver > 0 {
            crate::core::cache::LATEST_VERSION_CACHE.insert(cache_key, max_ver).await;
        }

        if let Some(path) = latest_path { 
             tracing::debug!("ManifestManager::load_latest: Found version {} on disk at {:?}", max_ver, path);
              return match self.load_version(max_ver).await {
                 Ok(m) => {
                     tracing::debug!("ManifestManager::load_latest: Successfully loaded v{} (entries={})", max_ver, m.entries.len());
                     Ok((m, max_ver))
                 },
                 Err(e) => {
                     tracing::error!("ManifestManager::load_latest: Failed to load v{} via load_version: {}", max_ver, e);
                     // Fallback if somehow listing said it exists but we can't read it
                     let bytes = self.store.get(&path).await?.bytes().await?;
                     let manifest: Manifest = serde_json::from_slice(&bytes)?;
                     Ok((manifest, max_ver))
                 }
             }
        } else {
            // No manifest found, return empty genesis
            Ok((Manifest::new(0, Vec::new(), None), 0))
        }
    }

    /// Load the latest manifest and ALL its entries (including sharded ones)
    pub async fn load_latest_full(&self) -> Result<(Manifest, Vec<ManifestEntry>, u64)> {
        let (manifest, ver) = self.load_latest().await?;
        let entries = self.load_all_entries(&manifest).await?;
        Ok((manifest, entries, ver))
    }

    /// Load a specific version of the manifest
    pub async fn load_version(&self, version: u64) -> Result<Manifest> {
        let filename = format!("v{}.json", version);
        let path = self.manifest_dir.child(filename);
        let cache_key = self.get_cache_key(&path);

        // 1. Check Data Cache
        if let Some(manifest) = crate::core::cache::MANIFEST_CACHE.get_with_metrics(&cache_key, "manifest").await {
            return Ok(manifest.as_ref().clone());
        }

        // 2. Fetch from S3
        let bytes = self.store.get(&path).await?.bytes().await?;
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        
        // 3. Populate Cache
        crate::core::cache::MANIFEST_CACHE.insert(cache_key, Arc::new(manifest.clone())).await;

        Ok(manifest)
    }

    /// Load a manifest list from a specific path
    pub async fn load_manifest_list(&self, path_str: &str) -> Result<ManifestList> {
        let path = Path::from(path_str);
        let cache_key = format!("{}/{}", self.root_uri, path);

        if let Some(list) = crate::core::cache::MANIFEST_LIST_CACHE.get_with_metrics(&cache_key, "manifest_list").await {
            return Ok(list.as_ref().clone());
        }

        let bytes = self.store.get(&path).await?.bytes().await?;
        
        if path_str.ends_with(".avro") {
            let iceberg_list = crate::core::iceberg::read_manifest_list(&bytes[..])?;
            let manifest_files = iceberg_list.into_iter().map(|e| {
                ManifestListEntry {
                    manifest_path: e.manifest_path,
                    manifest_length: e.manifest_length,
                    partition_spec_id: e.partition_spec_id,
                    content: e.content,
                    sequence_number: e.sequence_number,
                    min_sequence_number: e.min_sequence_number,
                    added_snapshot_id: e.added_snapshot_id,
                    added_files_count: e.added_files_count,
                    existing_files_count: e.existing_files_count,
                    deleted_files_count: e.deleted_files_count,
                    added_rows_count: e.added_rows_count,
                    existing_rows_count: e.existing_rows_count,
                    deleted_rows_count: e.deleted_rows_count,
                    partition_stats: HashMap::new(), // Stats not parsed yet
                }
            }).collect();
            let list = ManifestList {
                manifest_files,
            };
            crate::core::cache::MANIFEST_LIST_CACHE.insert(cache_key, Arc::new(list.clone())).await;
            return Ok(list);
        }

        let list: ManifestList = serde_json::from_slice(&bytes)?;
        
        crate::core::cache::MANIFEST_LIST_CACHE.insert(cache_key, Arc::new(list.clone())).await;
        Ok(list)
    }

    /// Save a manifest list to storage
    pub async fn save_manifest_list(&self, list: &ManifestList, version: u64) -> Result<String> {
        let filename = format!("list-v{}.json", version);
        let path = self.manifest_dir.child(filename);
        let bytes = serde_json::to_vec_pretty(list)?;
        
        self.store.put(&path, bytes.into()).await?;
        Ok(path.to_string())
    }

    pub async fn load_all_entries(&self, manifest: &Manifest) -> Result<Vec<ManifestEntry>> {
        // Use a HashMap to deduplicate segments by file_path.
        // Entries directly in the Manifest (staged/small tables) should override 
        // entries found in Manifest Lists (sharded segments).
        let mut entry_map: HashMap<String, ManifestEntry> = HashMap::new();
        
        // 1. Process manifest lists first (lower priority)
        if let Some(list_path) = &manifest.manifest_list_path {
            let list = self.load_manifest_list(list_path).await?;
            
            // Parallelize manifest loading using FuturesUnordered
            let mut futures = futures::stream::FuturesUnordered::new();
            
            // Resolve schema for stats decoding
            let schema = manifest.schemas.iter()
                .find(|s| s.schema_id == manifest.current_schema_id)
                .or(manifest.schemas.last())
                .cloned();

            for entry in list.manifest_files {
                let entry_path = entry.manifest_path.clone();
                let table_spec = manifest.partition_spec.clone();
                let table_schema = schema.clone();
                let store = self.store.clone();
                let root_uri = self.root_uri.clone();

                futures.push(async move {
                    if entry_path.ends_with(".avro") {
                        if let Some(s) = table_schema {
                             Self::load_avro_manifest_static(store, entry_path, s, table_spec, root_uri).await
                        } else {
                             Self::load_avro_manifest_static(store, entry_path, Schema::default(), table_spec, root_uri).await
                        }
                    } else {
                        Self::load_manifest_static(store, entry_path, root_uri).await
                    }
                });
            }

            while let Some(res) = futures.next().await {
                let sub_manifest = res?;
                for e in sub_manifest.entries {
                    entry_map.insert(e.file_path.clone(), e);
                }
            }
        }

        // 2. Process manifest.entries last (higher priority - overrides manifest lists)
        for e in &manifest.entries {
            entry_map.insert(e.file_path.clone(), e.clone());
        }
        
        Ok(entry_map.into_values().collect())
    }

    /// Helper to load a manifest from an arbitrary path
    pub async fn load_manifest_from_path(&self, path_str: &str) -> Result<Manifest> {
        Self::load_manifest_static(self.store.clone(), path_str.to_string(), self.root_uri.clone()).await
    }

    pub async fn load_avro_manifest(&self, path_str: &str, schema: &Schema, spec: &PartitionSpec) -> Result<Manifest> {
          Self::load_avro_manifest_static(self.store.clone(), path_str.to_string(), schema.clone(), spec.clone(), self.root_uri.clone()).await
    }

    async fn load_avro_manifest_static(store: Arc<dyn ObjectStore>, path_str: String, schema: Schema, spec: PartitionSpec, root_uri: String) -> Result<Manifest> {
         let path = Path::from(path_str);
         let cache_key = format!("{}/{}", root_uri, path);
         
         if let Some(manifest) = crate::core::cache::MANIFEST_CACHE.get_with_metrics(&cache_key, "manifest").await {
             return Ok(manifest.as_ref().clone());
         }

         let bytes = store.get(&path).await?.bytes().await?;
         let iceberg_entries = crate::core::iceberg::read_manifest(&bytes[..])?;
         
         let mut data_entries = Vec::new();
         let mut delete_files = Vec::new();
         
         for ie in iceberg_entries {
             if ie.status == 0 || ie.status == 1 { // EXISTING or ADDED
                 match crate::core::iceberg::convert_iceberg_to_object(&ie, &schema, &spec)? {
                     crate::core::iceberg::IcebergManifestObject::Data(me) => data_entries.push(me),
                     crate::core::iceberg::IcebergManifestObject::Delete(df) => delete_files.push(df),
                 }
             }
         }
         
         // Simple linking of equality deletes to data files in same partition
         for data in &mut data_entries {
             for delete in &delete_files {
                 if data.partition_values == delete.partition_values {
                     data.delete_files.push(delete.clone());
                 }
             }
         }
         
         let manifest = Manifest::new(0, data_entries, None);
         crate::core::cache::MANIFEST_CACHE.insert(cache_key, Arc::new(manifest.clone())).await;
         Ok(manifest)
    }

    async fn load_manifest_static(store: Arc<dyn ObjectStore>, path_str: String, root_uri: String) -> Result<Manifest> {
        let path = Path::from(path_str);
        let cache_key = format!("{}/{}", root_uri, path);

        if let Some(manifest) = crate::core::cache::MANIFEST_CACHE.get_with_metrics(&cache_key, "manifest").await {
            return Ok(manifest.as_ref().clone());
        }

        let bytes = store.get(&path).await?.bytes().await?;
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        
        crate::core::cache::MANIFEST_CACHE.insert(cache_key, Arc::new(manifest.clone())).await;
        Ok(manifest)
    }

    /// Walk back history starting from the latest version.
    /// Returns a list of Manifests [Latest, Latest-1, ... Genesis]
    pub async fn walk_history(&self) -> Result<Vec<Manifest>> {
        let (latest, _) = self.load_latest().await?;
        if latest.version == 0 {
            return Ok(vec![]);
        }

        let mut history = Vec::new();
        history.push(latest.clone());

        let mut current = latest;
        while let Some(prev) = current.prev_version {
            // Safety break for now, though u64 prevents inf loops
            if prev == 0 { break; } 
            
            match self.load_version(prev).await {
                Ok(m) => {
                    history.push(m.clone());
                    current = m;
                },
                Err(e) => {
                    tracing::warn!("Broken manifest chain at v{}: {}", prev, e);
                    break;
                }
            }
        }
        
        Ok(history)
    }

    /// Rollback the table state to a previous manifest version
    pub async fn rollback_to_snapshot(&self, version: u64) -> Result<Manifest> {
        let max_retries = 10;
        let mut attempt = 0;
        
        loop {
            let (_current_manifest, current_ver) = self.load_latest().await?;
            let new_ver = current_ver + 1;
            
            // Load the target version we want to rollback to
            let target_manifest = self.load_version(version).await?;
            
            // Create a new manifest that is a copy of the target, 
            // but with a new version number and pointing to the current version as previous.
            let mut new_manifest = target_manifest.clone();
            new_manifest.version = new_ver;
            new_manifest.prev_version = Some(current_ver);
            
            let filename = format!("v{}.json", new_ver);
            let path = self.manifest_dir.child(filename);
            let bytes = serde_json::to_vec_pretty(&new_manifest)?;
            
            use object_store::{PutMode, PutOptions};
            let opts = PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            };
            
            match self.store.put_opts(&path, bytes.into(), opts).await {
                Ok(_) => {
                    tracing::info!("Rolled back Manifest to v{} (from snapshot {})", new_ver, version);
                    let dir_key = format!("{}/{}", self.root_uri, self.manifest_dir);
                    crate::core::cache::LATEST_VERSION_CACHE.invalidate(&dir_key).await;
                    let file_key = format!("{}/{}", self.root_uri, path);
                    crate::core::cache::MANIFEST_CACHE.insert(file_key, Arc::new(new_manifest.clone())).await;
                    return Ok(new_manifest);
                }
                Err(e) if is_already_exists(&e) => {
                     attempt += 1;
                     if attempt >= max_retries { break; }
                     tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                     continue;
                }
                Err(e) => return Err(e.into())
            }
        }
        Err(anyhow::anyhow!("Failed to rollback after {} attempts", max_retries))
    }



    /// Commit a change to the timeline.
    /// Uses optimistic concurrency control with retries and PutMode::Create
    /// to ensure atomicity under high concurrency.
    #[tracing::instrument(skip(self, add_entries, remove_paths, metadata))]
    pub async fn commit(
        &self, 
        add_entries: &[ManifestEntry], 
        remove_paths: &[String],
        metadata: CommitMetadata
    ) -> Result<Manifest> {
        let cache_key = self.get_dir_cache_key();
        let lock = COMMIT_LOCKS.entry(cache_key)
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .value().clone();
            
        let _guard = lock.lock().await;
        
        let dist_lock_path = Path::from(format!("{}/commit.lock", self.manifest_dir));
        let dist_lock = crate::core::lock::FileBasedLock::new(self.store.clone(), dist_lock_path, 30);
        dist_lock.acquire().await?;

        let max_retries = 100;
        for attempt in 0..max_retries {
            let (current_manifest, current_ver) = self.load_latest_direct().await?;
            
            // 1. Calculate new state
            let all_entries = self.load_all_entries(&current_manifest).await?;
            let mut active_map: HashMap<String, ManifestEntry> = all_entries.into_iter().map(|e| (e.file_path.clone(), e)).collect();
            let new_ver = current_ver + 1;
            
            // Hardened De-duplication
            for path in remove_paths {
                active_map.remove(path);
            }
            
            // Add new entries to state (overwrites if path exists, but preserves indexes if we are adding unindexed version)
            for entry in add_entries {
                // HyperStream Optimization: If the existing entry already has indexes, 
                // and the new one doesn't (or has fewer), PRESERVE the indexes!
                // This prevents 'flush_async' main thread from overwriting background indexing results.
                if let Some(existing) = active_map.get(&entry.file_path) {
                    if existing.index_files.len() > entry.index_files.len() {
                        let mut merged = entry.clone();
                        merged.index_files = existing.index_files.clone();
                        active_map.insert(entry.file_path.clone(), merged);
                        continue;
                    }
                }
                active_map.insert(entry.file_path.clone(), entry.clone());
            }
            let new_entries: Vec<ManifestEntry> = active_map.into_values().collect();
            
            // 2. Decide if we need a ManifestList (Scalability)
            let (final_entries, manifest_list_path) = if new_entries.len() > MAX_ENTRIES_PER_MANIFEST {
                // Split entries into multiple manifests
                let mut manifest_files = Vec::new();
                let mut futures = futures::stream::FuturesUnordered::new();
                let chunks = new_entries.chunks(MAX_ENTRIES_PER_MANIFEST);
                
                for (chunk_idx, chunk) in chunks.enumerate() {
                    let uuid = uuid::Uuid::new_v4();
                    let filename = format!("{}-m{}.avro", uuid, chunk_idx);
                    let path = self.manifest_dir.child(filename);
                    
                    let writer = crate::core::iceberg::IcebergWriter::new();
                    let default_schema = crate::core::manifest::Schema::default();
                    let table_schema = current_manifest.schemas.last().unwrap_or(&default_schema).clone();
                    let table_spec = current_manifest.partition_spec.clone();
                    let new_ver_i64 = new_ver as i64;
                    let store = self.store.clone();
                    let chunk_owned: Vec<ManifestEntry> = chunk.to_vec();

                    futures.push(async move {
                        let bytes = writer.write_manifest_file(&chunk_owned, &table_spec, &table_schema, new_ver_i64, new_ver_i64)?;
                        let manifest_length = bytes.len() as i64;
                        let rows_count: i64 = chunk_owned.iter().map(|e| e.record_count).sum();
                        
                        store.put(&path, bytes.into()).await?;
                        
                        Result::<ManifestListEntry>::Ok(ManifestListEntry {
                            manifest_path: path.to_string(),
                            manifest_length,
                            partition_spec_id: table_spec.spec_id,
                            content: 0, // Data
                            sequence_number: new_ver_i64,
                            min_sequence_number: new_ver_i64,
                            added_snapshot_id: new_ver_i64,
                            added_files_count: chunk_owned.len() as i32,
                            existing_files_count: 0,
                            deleted_files_count: 0,
                            added_rows_count: rows_count,
                            existing_rows_count: 0,
                            deleted_rows_count: 0,
                            partition_stats: HashMap::new(), 
                        })
                    });
                }

                while let Some(res) = futures.next().await {
                    manifest_files.push(res?);
                }
                
                let list_uuid = uuid::Uuid::new_v4();
                let list_filename = format!("snap-{}-{}.avro", new_ver, list_uuid);
                let list_path_loc = self.manifest_dir.child(list_filename);
                
                let writer = crate::core::iceberg::IcebergWriter::new();
                let list_bytes = writer.write_manifest_list(&manifest_files)?;
                self.store.put(&list_path_loc, list_bytes.into()).await?;
                
                (Vec::new(), Some(list_path_loc.to_string()))
            } else {
                (new_entries, None)
            };
            
            // 3. Create new Manifest
            let final_schemas = metadata.updated_schemas.as_ref().cloned().unwrap_or_else(|| current_manifest.schemas.clone());
            let final_schema_id = metadata.updated_schema_id.unwrap_or(current_manifest.current_schema_id);

            let final_partition_spec = if let Some(specs) = &metadata.updated_partition_specs {
                 specs.last().cloned().unwrap_or(current_manifest.partition_spec.clone()) 
            } else {
                 current_manifest.partition_spec.clone()
            };
            
            let final_sort_orders = metadata.updated_sort_orders.as_ref().cloned().unwrap_or_else(|| current_manifest.sort_orders.clone());
            let final_default_sort_order_id = metadata.updated_default_sort_order_id.unwrap_or(current_manifest.default_sort_order_id);
            
            let mut new_manifest = Manifest::new_with_spec(
                new_ver, 
                final_entries, 
                Some(current_ver),
                final_schemas,
                final_schema_id,
                final_partition_spec,
            );
            
            new_manifest.sort_orders = final_sort_orders;
            new_manifest.default_sort_order_id = final_default_sort_order_id;
            
            new_manifest.properties = current_manifest.properties.clone();
            if let Some(props) = &metadata.updated_properties {
                tracing::debug!("Applying property updates: {:?}", props);
                new_manifest.properties.extend(props.clone().into_iter());
            }
            if let Some(removals) = &metadata.removed_properties {
                 for key in removals {
                     new_manifest.properties.remove(key);
                 }
            }
            
            new_manifest.partition_specs = if let Some(specs) = &metadata.updated_partition_specs {
                specs.clone()
            } else {
                current_manifest.partition_specs.clone()
            };
            new_manifest.default_spec_id = metadata.updated_default_spec_id.unwrap_or(current_manifest.default_spec_id);
            new_manifest.last_column_id = metadata.updated_last_column_id.unwrap_or(current_manifest.last_column_id);

            new_manifest.manifest_list_path = manifest_list_path;
            
            // 4. Write v{N+1}.json with PutMode::Create (Atomic)
            let filename = format!("v{}.json", new_ver);
            let path = self.manifest_dir.child(filename);
            
            let bytes = serde_json::to_vec_pretty(&new_manifest)?;

            use object_store::{PutMode, PutOptions};
            let opts = PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            };

            match self.store.put_opts(&path, bytes.into(), opts).await {
                Ok(_) => {
                    tracing::info!("Committed Manifest: {}", path);
                    // 5. Update Caches
                    let dir_key = format!("{}/{}", self.root_uri, self.manifest_dir);
                    crate::core::cache::LATEST_VERSION_CACHE.invalidate(&dir_key).await;
                    crate::core::cache::LATEST_VERSION_CACHE.insert(dir_key, new_ver).await;
                    
                    // Cache the new manifest file eagerly
                    let file_key = format!("{}/{}", self.root_uri, path);
                    crate::core::cache::MANIFEST_CACHE.insert(file_key, Arc::new(new_manifest.clone())).await;

                    let _ = dist_lock.release().await;
                    return Ok(new_manifest);
                }
                Err(e) if is_already_exists(&e) => {
                    if attempt % 10 == 0 || attempt > 90 {
                        tracing::debug!("Conflict committing Manifest v{} (attempt {}), retrying...", new_ver, attempt + 1);
                    }
                    let base_delay = 10 * (2u64.pow(attempt.min(5) as u32));
                    let jitter = rand::random::<u64>() % base_delay;
                    tokio::time::sleep(std::time::Duration::from_millis(base_delay + jitter)).await;
                    continue;
                }
                Err(e) => {
                    let _ = dist_lock.release().await;
                    return Err(e.into());
                }
            }
        }
        
        let _ = dist_lock.release().await;
        Err(anyhow::anyhow!("Failed to commit manifest after {} attempts due to concurrent updates", max_retries))
    }

    /// Commit a set of imported entries (merges with current state)
    pub async fn commit_imported_entries(&self, entries: Vec<ManifestEntry>) -> Result<Manifest> {
        let dist_lock_path = Path::from(format!("{}/commit.lock", self.manifest_dir));
        let dist_lock = crate::core::lock::FileBasedLock::new(self.store.clone(), dist_lock_path, 30);
        dist_lock.acquire().await?;

        let max_retries = 10;
        let mut attempt = 0;
        loop {
            let (current_manifest, current_ver) = self.load_latest().await?;
            let all_existing = self.load_all_entries(&current_manifest).await?;
            
            // Merge entries, avoid duplicates, favor NEW entries for the SAME file_path
            let mut entry_map: HashMap<String, ManifestEntry> = all_existing.into_iter()
                .map(|e| (e.file_path.clone(), e))
                .collect();
                
            for entry in &entries {
                entry_map.insert(entry.file_path.clone(), entry.clone());
            }
            
            let merged_entries: Vec<ManifestEntry> = entry_map.into_values().collect();
            let new_ver = current_ver + 1;
            
            let mut new_manifest = Manifest::new_with_spec(
                new_ver,
                merged_entries,
                Some(current_ver),
                current_manifest.schemas.clone(),
                current_manifest.current_schema_id,
                current_manifest.partition_spec.clone(),
            );

            new_manifest.partition_specs = current_manifest.partition_specs.clone();
            new_manifest.default_spec_id = current_manifest.default_spec_id;
            new_manifest.properties = current_manifest.properties.clone();
            new_manifest.sort_orders = current_manifest.sort_orders.clone();
            new_manifest.default_sort_order_id = current_manifest.default_sort_order_id;
            new_manifest.manifest_list_path = current_manifest.manifest_list_path.clone();
            
            // Write to storage with conflict detection
            let filename = format!("v{}.json", new_ver);
            let path = self.manifest_dir.child(filename);
            let bytes = serde_json::to_vec_pretty(&new_manifest)?;
            
            use object_store::{PutMode, PutOptions};
            let opts = PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            };
            
            match self.store.put_opts(&path, bytes.into(), opts).await {
                Ok(_) => {
                    tracing::info!("Imported {} external entries into Manifest v{}", entries.len(), new_ver);
                    let dir_key = self.get_dir_cache_key();
                    crate::core::cache::LATEST_VERSION_CACHE.invalidate(&dir_key).await;
                    let file_key = self.get_cache_key(&path);
                    crate::core::cache::MANIFEST_CACHE.insert(file_key, Arc::new(new_manifest.clone())).await;
                    let _ = dist_lock.release().await;
                    return Ok(new_manifest);
                }
                Err(e) if is_already_exists(&e) => {
                    attempt += 1;
                    if attempt >= max_retries { break; }
                    tracing::warn!("Manifest conflict during entry import. Retrying attempt {}/{}", attempt, max_retries);
                    tokio::time::sleep(std::time::Duration::from_millis(20 * attempt)).await;
                    continue;
                }
                Err(e) => {
                    let _ = dist_lock.release().await;
                    return Err(e.into());
                }
            }
        }
        let _ = dist_lock.release().await;
        Err(anyhow::anyhow!("Failed to commit imported entries after {} attempts", max_retries))
    }

    pub async fn update_schema(&self, new_schemas: Vec<Schema>, new_schema_id: i32, last_column_id: Option<i32>) -> Result<Manifest> {
        let dist_lock_path = Path::from(format!("{}/commit.lock", self.manifest_dir));
        let dist_lock = crate::core::lock::FileBasedLock::new(self.store.clone(), dist_lock_path, 30);
        dist_lock.acquire().await?;

        let max_retries = 10;
        let mut attempt = 0;
        
        loop {
            // Optimistic Concurrency Control Loop
            let (current_manifest, current_ver) = self.load_latest().await?;
            let new_ver = current_ver + 1;
            
            // Re-use entries from latest
            let entries = current_manifest.entries.clone();
            
            let mut new_manifest = Manifest::new_with_spec(
                new_ver, 
                entries, 
                Some(current_ver),
                new_schemas.clone(),
                new_schema_id,
                current_manifest.partition_spec.clone(),
            );
            new_manifest.manifest_list_path = current_manifest.manifest_list_path.clone();
            new_manifest.last_column_id = last_column_id.unwrap_or(current_manifest.last_column_id);
            
            let filename = format!("v{}.json", new_ver);
            let path = self.manifest_dir.child(filename);
            let bytes = serde_json::to_vec_pretty(&new_manifest)?;
            
            use object_store::{PutMode, PutOptions};
            let opts = PutOptions {
                mode: PutMode::Create,
                ..Default::default()
              };
            
            match self.store.put_opts(&path, bytes.into(), opts).await {
                Ok(_) => {
                    tracing::info!("Committed Manifest v{} (Schema Update)", new_ver);
                    let dir_key = format!("{}/{}", self.root_uri, self.manifest_dir);
                    crate::core::cache::LATEST_VERSION_CACHE.invalidate(&dir_key).await;
                    let file_key = format!("{}/{}", self.root_uri, path);
                    crate::core::cache::MANIFEST_CACHE.insert(file_key, Arc::new(new_manifest.clone())).await;
                    let _ = dist_lock.release().await;
                    return Ok(new_manifest);
                }
                Err(e) if is_already_exists(&e) => {
                     // Conflict, retry
                     if attempt >= max_retries {
                         break;
                     }
                     attempt += 1;
                     let base_delay = 10 * (2u64.pow(attempt.min(5) as u32));
                     let jitter = rand::random::<u64>() % base_delay;
                     tokio::time::sleep(std::time::Duration::from_millis(base_delay + jitter)).await;
                     continue;
                }
                Err(e) => {
                    let _ = dist_lock.release().await;
                    return Err(e.into());
                }
            }
        }
        let _ = dist_lock.release().await;
        Err(anyhow::anyhow!("Failed to commit schema update after {} attempts", max_retries))
    }

    /// Update the primary key (identifier fields) for the table.
    /// This creates a new schema version with the updated field IDs.
    #[tracing::instrument(skip(self, new_ids))]
    pub async fn update_identifier_fields(&self, new_ids: Vec<i32>) -> Result<Manifest> {
        let (current_manifest, _) = self.load_latest().await?;
        let mut schemas = current_manifest.schemas.clone();
        
        // Get latest schema and update its identifier fields
        if let Some(latest_schema) = schemas.last_mut() {
            // Check if it's already exactly the same to avoid redundant commits
            if latest_schema.identifier_field_ids == new_ids {
                return Ok(current_manifest);
            }
            
            // Create a new schema version
            let mut new_schema = latest_schema.clone();
            new_schema.schema_id += 1;
            new_schema.identifier_field_ids = new_ids.clone();
            
            // AUTOMATIC EVOLUTION: Set identifier fields to required: true (NOT NULL)
            for field in &mut new_schema.fields {
                if new_ids.contains(&field.id) {
                    field.required = true;
                }
            }
            
            let new_schema_id = new_schema.schema_id;
            schemas.push(new_schema);
            
            self.update_schema(schemas, new_schema_id, None).await
        } else {
            Err(anyhow::anyhow!("No existing schema found to update identifier fields"))
        }
    }

    /// Atomically commit a full manifest (optimistic concurrency)
    pub async fn commit_manifest(&self, manifest: Manifest) -> Result<()> {
        let max_retries = 10;
        let mut attempt = 0;
        loop {
            let (_, current_ver) = self.load_latest().await?;
            if manifest.version != current_ver + 1 {
                return Err(anyhow::anyhow!("Manifest version mismatch: expected {}, got {}", current_ver + 1, manifest.version));
            }
            
            let filename = format!("v{}.json", manifest.version);
            let path = self.manifest_dir.child(filename);
            let bytes = serde_json::to_vec_pretty(&manifest)?;
            
            use object_store::{PutMode, PutOptions};
            let opts = PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            };
            
            match self.store.put_opts(&path, bytes.into(), opts).await {
                Ok(_) => {
                    tracing::info!("Committed Manifest v{}", manifest.version);
                    let dir_key = self.get_dir_cache_key();
                    crate::core::cache::LATEST_VERSION_CACHE.invalidate(&dir_key).await;
                    let file_key = self.get_cache_key(&path);
                    crate::core::cache::MANIFEST_CACHE.insert(file_key, Arc::new(manifest.clone())).await;
                    return Ok(());
                }
                Err(e) if is_already_exists(&e) => {
                     attempt += 1;
                     if attempt >= max_retries { break; }
                     tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                     continue;
                }
                Err(e) => return Err(e.into())
            }
        }
        Err(anyhow::anyhow!("Failed to commit manifest after {} attempts", max_retries))
    }

    /// Update indexing specifications for columns
    pub async fn update_index_specs(&self, column_indexes: HashMap<String, Vec<IndexAlgorithm>>) -> Result<Manifest> {
        let max_retries = 10;
        let mut attempt = 0;
        loop {
            let (current_manifest, current_ver) = self.load_latest().await?;
            let current_schema = current_manifest.schemas.last()
                .ok_or_else(|| anyhow::anyhow!("No schema found in manifest"))?;
            
            let mut new_fields = current_schema.fields.clone();
            let mut changed = false;
            
            for field in &mut new_fields {
                if let Some(new_indexes) = column_indexes.get(&field.name) {
                    field.indexes = new_indexes.clone();
                    changed = true;
                }
            }
            
            if !changed {
                return Ok(current_manifest);
            }
            
            let new_schema = Schema {
                schema_id: current_schema.schema_id + 1,
                fields: new_fields,
                identifier_field_ids: current_schema.identifier_field_ids.clone(),
            };
            
            let mut new_schemas = current_manifest.schemas.clone();
            new_schemas.push(new_schema.clone());
            
            let new_ver = current_ver + 1;
            let mut new_manifest = Manifest::new_with_spec(
                new_ver, 
                current_manifest.entries.clone(), 
                Some(current_ver),
                new_schemas,
                new_schema.schema_id,
                current_manifest.partition_spec.clone(),
            );
            new_manifest.manifest_list_path = current_manifest.manifest_list_path.clone();
            new_manifest.properties = current_manifest.properties.clone();
            
            let filename = format!("v{}.json", new_ver);
            let path = self.manifest_dir.child(filename);
            let bytes = serde_json::to_vec_pretty(&new_manifest)?;
            
            use object_store::{PutMode, PutOptions};
            let opts = PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            };
            
            match self.store.put_opts(&path, bytes.into(), opts).await {
                Ok(_) => {
                    tracing::info!("Committed Manifest v{} (Index Spec Update)", new_ver);
                    let dir_key = self.get_dir_cache_key();
                    crate::core::cache::LATEST_VERSION_CACHE.invalidate(&dir_key).await;
                    let file_key = self.get_cache_key(&path);
                    crate::core::cache::MANIFEST_CACHE.insert(file_key, Arc::new(new_manifest.clone())).await;
                    return Ok(new_manifest);
                }
                Err(e) if is_already_exists(&e) => {
                     attempt += 1;
                     if attempt >= max_retries { break; }
                     tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                     continue;
                }
                Err(e) => return Err(e.into())
            }
        }
        Err(anyhow::anyhow!("Failed to commit index spec update"))
    }

    /// Update partition specification
    pub async fn update_partition_spec(&self, new_spec: PartitionSpec) -> Result<Manifest> {
        let max_retries = 10;
        let mut attempt = 0;
        loop {
            let (current_manifest, current_ver) = self.load_latest().await?;
            let new_ver = current_ver + 1;
            
            let new_manifest = Manifest::new_with_spec(
                new_ver, 
                current_manifest.entries.clone(), 
                Some(current_ver),
                current_manifest.schemas.clone(),
                current_manifest.current_schema_id,
                new_spec.clone(),
            );
            
            let filename = format!("v{}.json", new_ver);
            let path = self.manifest_dir.child(filename);
            let bytes = serde_json::to_vec_pretty(&new_manifest)?;
            
            use object_store::{PutMode, PutOptions};
            let opts = PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            };
            
            match self.store.put_opts(&path, bytes.into(), opts).await {
                Ok(_) => {
                    tracing::info!("Committed Manifest v{} (Partition Spec Update)", new_ver);
                    let dir_key = self.get_dir_cache_key();
                    crate::core::cache::LATEST_VERSION_CACHE.invalidate(&dir_key).await;
                    let file_key = self.get_cache_key(&path);
                    crate::core::cache::MANIFEST_CACHE.insert(file_key, Arc::new(new_manifest.clone())).await;
                    return Ok(new_manifest);
                }
                Err(e) if is_already_exists(&e) => {
                     attempt += 1;
                     if attempt >= max_retries { break; }
                     tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                     continue;
                }
                Err(e) => return Err(e.into())
            }
        }
        Err(anyhow::anyhow!("Failed to commit partition spec update"))
    }

    /// Delete old manifest files and any data/index files NOT referenced by the latest N versions.
    /// Returns the number of files deleted.
    pub async fn vacuum(&self, retention_versions: usize) -> Result<usize> {
        if retention_versions == 0 {
            return Err(anyhow::anyhow!("Retention versions must be at least 1"));
        }

        let (_latest_m, latest_ver) = self.load_latest().await?;
        if latest_ver == 0 {
            return Ok(0);
        }

        // 1. Identify active files in the retention window
        let mut active_files = HashSet::new();
        let mut manifest_files_to_keep = HashSet::new();

        let start_ver = latest_ver.saturating_sub(retention_versions as u64 - 1).max(1);
        
        for v in start_ver..=latest_ver {
            let m = match self.load_version(v).await {
                Ok(m) => m,
                Err(_) => continue, // Skip missing versions in history gaps
            };
            
            // Collect all data and index files
            for entry in m.entries {
                active_files.insert(entry.file_path.clone());
                for index in entry.index_files {
                    active_files.insert(index.file_path.clone());
                }
                for del in entry.delete_files {
                    active_files.insert(del.file_path.clone());
                }
            }
            
            // Keep the manifest file itself
            let m_name = format!("v{}.json", v);
            let m_path = self.manifest_dir.child(m_name);
            manifest_files_to_keep.insert(m_path.to_string());
        }

        // 2. Discover all files in the storage
        // We list the root but skip the _manifest/ dir contents (except handled separately)
        let mut deleted_count = 0;
        let mut stream = self.store.list(None);
        
        while let Some(meta) = stream.next().await {
            let meta = meta?;
            let path_str = meta.location.to_string();
            
            // Skip the current manifest directory itself but check files inside
            if path_str.contains("_manifest/v") {
                // If it's a manifest file, check if we keep it
                if !manifest_files_to_keep.contains(&path_str) {
                    tracing::info!("Vacuum: Deleting old manifest {}", path_str);
                    self.store.delete(&meta.location).await?;
                    deleted_count += 1;
                }
                continue;
            }

            // Skip other files in _manifest/ (e.g. checkpoints if added later)
            if path_str.contains("_manifest/") {
                continue;
            }

            // check if it's a data file we should care about
            // segment files (seg_...), compacted files (compacted_...), or .tmp files
            let is_data_file = path_str.ends_with(".parquet") || 
                              path_str.ends_with(".hnsw") || 
                              path_str.ends_with(".idx") ||
                              path_str.ends_with(".tmp");

            if is_data_file {
                // If it's not in the active set, delete it
                if !active_files.contains(&path_str) {
                    // Small safety: don't delete very young .tmp files (leeway for active writers)
                    // If it's a .tmp file and less than 1 hour old, skip.
                    if path_str.ends_with(".tmp") {
                        let age = Utc::now() - chrono::DateTime::from_timestamp(meta.last_modified.timestamp(), 0).unwrap_or(Utc::now());
                        if age.num_minutes() < 60 {
                            continue;
                        }
                    }

                    tracing::info!("Vacuum: Deleting unreferenced file {}", path_str);
                    self.store.delete(&meta.location).await?;
                    deleted_count += 1;
                }
            }
        }

        Ok(deleted_count)
    }
}

impl PartitionSpec {
    pub fn partition_batch(&self, batch: &RecordBatch) -> Result<Vec<(HashMap<String, Value>, RecordBatch)>> {
        if self.fields.is_empty() {
             return Ok(vec![(HashMap::new(), batch.clone())]);
        }

        // 1. Group row indices by partition key
        let mut row_groups: HashMap<Vec<Value>, Vec<u32>> = HashMap::new();
        
        for i in 0..batch.num_rows() {
            let mut key = Vec::with_capacity(self.fields.len());
            for field in &self.fields {
                                // Determine source columns
                let source_ids = field.get_source_ids();
                let mut cols = Vec::new();
                
                // Prioritize finding by Name (most intuitive for users)
                let mut found = false;
                if let Ok(idx) = batch.schema().index_of(&field.name) {
                    cols.push(batch.column(idx));
                    found = true;
                }

                // Fallback to Iceberg IDs only if name lookup failed
                if !found {
                    for id in &source_ids {
                        let idx = batch.schema().fields().iter().position(|f| {
                            f.metadata().get("iceberg.id")
                                .and_then(|id_str| id_str.parse::<i32>().ok())
                                .map(|found_id| found_id == *id)
                                .unwrap_or(false)
                        });
                        if let Some(i) = idx {
                            cols.push(batch.column(i));
                        }
                    }
                }

                if cols.is_empty() {
                    anyhow::bail!("Partition column {} (IDs {:?}) missing from batch", field.name, source_ids);
                }

                let val = match field.transform.as_str() {
                    "identity" if cols.len() == 1 => {
                        let col = cols[0];
                        if col.is_null(i) {
                            serde_json::Value::Null
                        } else {
                             if let Some(s) = col.as_any().downcast_ref::<arrow::array::StringArray>() {
                                 serde_json::Value::String(s.value(i).to_string())
                             } else if let Some(s) = col.as_any().downcast_ref::<arrow::array::LargeStringArray>() {
                                 serde_json::Value::String(s.value(i).to_string())
                             } else if let Some(dict) = col.as_any().downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::Int32Type>>() {
                                 let values = dict.values().as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
                                 if dict.is_null(i) {
                                     serde_json::Value::Null
                                 } else {
                                     let key = dict.key(i).unwrap();
                                     serde_json::Value::String(values.value(key as usize).to_string())
                                 }
                             } else if let Some(dict) = col.as_any().downcast_ref::<arrow::array::DictionaryArray<arrow::datatypes::Int64Type>>() {
                                 let values = dict.values().as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
                                 if dict.is_null(i) {
                                     serde_json::Value::Null
                                 } else {
                                     let key = dict.key(i).unwrap() as usize;
                                     serde_json::Value::String(values.value(key).to_string())
                                 }
                             } else if let Some(n) = col.as_any().downcast_ref::<arrow::array::Int64Array>() {
                                 serde_json::Value::Number(serde_json::Number::from(n.value(i)))
                             } else if let Some(n) = col.as_any().downcast_ref::<arrow::array::Int32Array>() {
                                 serde_json::Value::Number(serde_json::Number::from(n.value(i)))
                             } else if let Some(n) = col.as_any().downcast_ref::<arrow::array::Float64Array>() {
                                 serde_json::Number::from_f64(n.value(i)).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null)
                             } else if let Some(n) = col.as_any().downcast_ref::<arrow::array::Float32Array>() {
                                 serde_json::Number::from_f64(n.value(i) as f64).map(serde_json::Value::Number).unwrap_or(serde_json::Value::Null)
                             } else {
                                 serde_json::to_value(format!("{:?}", col.slice(i, 1))).unwrap_or(serde_json::Value::Null)
                             }
                        }
                    },
                    _ => {
                        // Handle bucket[N] or bucket(N)
                        let mut num_buckets = None;
                        if field.transform.starts_with("bucket") {
                            let parts: Vec<&str> = field.transform.split(|c| c == '[' || c == ']' || c == '(' || c == ')').collect();
                            for part in parts {
                                if let Ok(n) = part.trim().parse::<u64>() {
                                    num_buckets = Some(n);
                                    break;
                                }
                            }
                        }

                        // Fallback: Stable hash of all source values for unknown/multi-column transforms
                        let mut hash_input = String::new();
                        for col in cols {
                             hash_input.push_str(&format!("{:?}", col.slice(i, 1)));
                        }
                        use std::collections::hash_map::DefaultHasher;
                        use std::hash::{Hash, Hasher};
                        let mut hasher = DefaultHasher::new();
                        hash_input.hash(&mut hasher);
                        let full_hash = hasher.finish();
                        
                        if let Some(n) = num_buckets {
                            serde_json::Value::Number(serde_json::Number::from((full_hash % n) as i32))
                        } else {
                            serde_json::Value::Number(serde_json::Number::from(full_hash))
                        }
                    }
                };
                key.push(val);
            }
            row_groups.entry(key).or_insert_with(Vec::new).push(i as u32);
        }

    // 2. Create sharded RecordBatches
    let mut result = Vec::with_capacity(row_groups.len());
    for (key_vec, rows) in row_groups {
        let indices = arrow::array::UInt32Array::from(rows);
        let sharded_batch = arrow::compute::take_record_batch(batch, &indices)?;

        let mut key_map = HashMap::with_capacity(self.fields.len());
        for (f, v) in self.fields.iter().zip(key_vec) {
            key_map.insert(f.name.clone(), v);
        }
        result.push((key_map, sharded_batch));
    }

    Ok(result)
}

/// Generate a Hive-style partition path string (e.g., "year=2024/month=04") from partition values.
pub fn partition_to_path(&self, values: &std::collections::HashMap<String, serde_json::Value>) -> String {
    let mut path_parts = Vec::new();
    for field in &self.fields {
        if let Some(val) = values.get(&field.name) {
            let val_str = match val {
                serde_json::Value::String(s) => s.clone(),
                serde_json::Value::Number(n) => n.to_string(),
                serde_json::Value::Bool(b) => b.to_string(),
                serde_json::Value::Null => "null".to_string(),
                _ => val.to_string().replace("\"", ""), // Remove quotes for cleaner directory names
            };
            path_parts.push(format!("{}={}", field.name, val_str));
        }
    }
    path_parts.join("/")
}
}

fn is_already_exists(e: &object_store::Error) -> bool {
    match e {
        object_store::Error::AlreadyExists { .. } => true,
        _ => e.to_string().contains("already exists") // Fallback for some store implementations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    fn create_entry(id: &str) -> ManifestEntry {
         ManifestEntry {
            file_path: format!("{}.parquet", id),
            file_size_bytes: 100,
            record_count: 10,
            index_files: vec![],
            delete_files: vec![],
            column_stats: HashMap::new(),
            partition_values: HashMap::new(),
            ..Default::default()
        }
    }

    #[tokio::test]
    async fn test_manifest_flow() -> Result<()> {
        // Use unique path to avoid cache conflicts
        let test_id = uuid::Uuid::new_v4();
        let test_uri = format!("memory://test_{}", test_id);
        let store = Arc::new(InMemory::new());
        let manager = ManifestManager::new(store, "", &test_uri);

        // 1. Initial State (Empty)
        let (m0, v0) = manager.load_latest().await?;
        assert_eq!(v0, 0);
        assert!(m0.entries.is_empty());

        // 2. Commit Add
        let entry_a = create_entry("seg_a");
        let m1 = manager.commit(std::slice::from_ref(&entry_a), &[], CommitMetadata::default()).await?;
        assert_eq!(m1.version, 1);
        
        // Load all entries (including those in manifest lists)
        let all_entries_1 = manager.load_all_entries(&m1).await?;
        assert_eq!(all_entries_1.len(), 1);
        assert_eq!(all_entries_1[0].file_path, "seg_a.parquet");

        // 3. Commit Add + Remove
        let entry_b = create_entry("seg_b");
        // Remove seg_a by path
        let m2 = manager.commit(std::slice::from_ref(&entry_b), &["seg_a.parquet".to_string()], CommitMetadata::default()).await?;
        assert_eq!(m2.version, 2);
        
        // Load all entries (including those in manifest lists)
        let all_entries_2 = manager.load_all_entries(&m2).await?;
        assert_eq!(all_entries_2.len(), 1);
        assert_eq!(all_entries_2[0].file_path, "seg_b.parquet");

        // 4. Reload
        let (latest, ver) = manager.load_latest().await?;
        assert_eq!(ver, 2);
        
        // Load all entries and compare
        let latest_entries = manager.load_all_entries(&latest).await?;
        let m2_entries = manager.load_all_entries(&m2).await?;
        assert_eq!(latest_entries.len(), m2_entries.len());
        assert_eq!(latest_entries[0].file_path, m2_entries[0].file_path);

        // Cleanup cache
        let cache_key = format!("{}/{}", test_uri, "");
        crate::core::cache::LATEST_VERSION_CACHE.invalidate(&cache_key).await;

        Ok(())
    }

    #[tokio::test]
    async fn test_verify_manifest_history() -> Result<()> {
        let store = Arc::new(InMemory::new());
        let root_uri = "memory://test";
        let manager = ManifestManager::new(store.clone(), "test_table", root_uri);

        // Commit 1
        let entry1 = create_entry("seg1");
        manager.commit(&[entry1], &[], CommitMetadata::default()).await?;

        // Commit 2
        let entry2 = create_entry("seg2");
        manager.commit(&[entry2], &[], CommitMetadata::default()).await?;

        // Load specific version (v2)
        // Load specific version (v2)
        let (_manifest, entries, version) = manager.load_latest_full().await?;
        assert_eq!(version, 2);
        assert_eq!(entries.len(), 2);

        // Walk history
        let history = manager.walk_history().await?;
        // walk_history returns [v2, v1] - it stops before v0 (genesis) since prev_version of v1 would be Some(0)
        // and the loop breaks when prev == 0
        assert_eq!(history.len(), 2); // v2, v1
        assert_eq!(history[0].version, 2);
        assert_eq!(history[1].version, 1);
        
        Ok(())
    }
}
