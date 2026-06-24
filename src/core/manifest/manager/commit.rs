// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Manifest commit operations: optimistic concurrency, rollback, vacuum, and import.

use anyhow::Result;
use chrono::Utc;
use futures::StreamExt;
use object_store::{path::Path, ObjectStore};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::super::types::*;
use super::{CommitMetadata, ManifestManager};

fn is_already_exists(err: &object_store::Error) -> bool {
    err.to_string().contains("already exists")
}

impl ManifestManager {
    /// Commit a change to the timeline.
    /// Uses optimistic concurrency control with retries and PutMode::Create
    /// to ensure atomicity under high concurrency.
    #[tracing::instrument(skip(self, add_entries, remove_paths, metadata))]
    pub async fn commit(
        &self,
        add_entries: &[ManifestEntry],
        remove_paths: &[String],
        metadata: CommitMetadata,
    ) -> Result<Manifest> {
        let cache_key = self.get_dir_cache_key();
        let lock = COMMIT_LOCKS
            .entry(cache_key)
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
            .value()
            .clone();

        let _guard = lock.lock().await;

        let dist_lock_path = Path::from(format!("{}/commit.lock", self.manifest_dir));
        let dist_lock =
            crate::core::lock::FileBasedLock::new(self.store.clone(), dist_lock_path, 30);
        let _dist_guard = dist_lock.acquire().await?;

        let max_retries = 100;
        for attempt in 0..max_retries {
            let (current_manifest, current_ver) = self.load_latest_direct().await?;

            // 1. Calculate new state
            let all_entries = self.load_all_entries(&current_manifest).await?;
            let mut active_map: HashMap<String, ManifestEntry> = all_entries
                .into_iter()
                .map(|e| (e.file_path.clone(), e))
                .collect();
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
            // HyperStreamDB v0.4: Always use Tiered Manifests (ManifestList -> ManifestFile)
            // chunked by 8MB to ensure 100% Iceberg Spec compatibility.
            let (final_entries, manifest_list_path) = if !new_entries.is_empty() {
                let mut manifest_files = Vec::new();
                let mut futures = futures::stream::FuturesUnordered::new();

                let writer = crate::core::iceberg::IcebergWriter::new();
                let default_schema = crate::core::manifest::Schema::default();
                let table_schema = current_manifest
                    .schemas
                    .last()
                    .unwrap_or(&default_schema)
                    .clone();
                let table_spec = current_manifest.partition_spec.clone();
                let new_ver_i64 = new_ver as i64;
                let store = self.store.clone();

                let chunks = writer.write_manifest_chunks(
                    &new_entries,
                    &table_spec,
                    &table_schema,
                    new_ver_i64,
                    new_ver_i64,
                    crate::core::manifest::types::MANIFEST_TARGET_SIZE_BYTES,
                )?;

                for (chunk_idx, (bytes, file_count, row_count)) in chunks.into_iter().enumerate() {
                    let uuid = uuid::Uuid::new_v4();
                    let filename = format!("{}-m{}.avro", uuid, chunk_idx);
                    let path = self.manifest_dir.child(filename);
                    let store = store.clone();
                    let partition_spec_id = table_spec.spec_id;

                    futures.push(async move {
                        let manifest_length = bytes.len() as i64;
                        store.put(&path, bytes.into()).await?;

                        Result::<ManifestListEntry>::Ok(ManifestListEntry {
                            manifest_path: path.to_string(),
                            manifest_length,
                            partition_spec_id,
                            content: 0, // Data
                            sequence_number: new_ver_i64,
                            min_sequence_number: new_ver_i64,
                            added_snapshot_id: new_ver_i64,
                            added_files_count: file_count as i32,
                            existing_files_count: 0,
                            deleted_files_count: 0,
                            added_rows_count: row_count,
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
                (Vec::new(), None)
            };

            // 3. Create new Manifest
            let final_schemas = metadata
                .updated_schemas
                .as_ref()
                .cloned()
                .unwrap_or_else(|| current_manifest.schemas.clone());
            let final_schema_id = metadata
                .updated_schema_id
                .unwrap_or(current_manifest.current_schema_id);

            let final_partition_spec = if let Some(specs) = &metadata.updated_partition_specs {
                specs
                    .last()
                    .cloned()
                    .unwrap_or(current_manifest.partition_spec.clone())
            } else {
                current_manifest.partition_spec.clone()
            };

            let final_sort_orders = metadata
                .updated_sort_orders
                .as_ref()
                .cloned()
                .unwrap_or_else(|| current_manifest.sort_orders.clone());
            let final_default_sort_order_id = metadata
                .updated_default_sort_order_id
                .unwrap_or(current_manifest.default_sort_order_id);

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
            new_manifest.default_spec_id = metadata
                .updated_default_spec_id
                .unwrap_or(current_manifest.default_spec_id);
            new_manifest.last_column_id = metadata
                .updated_last_column_id
                .unwrap_or(current_manifest.last_column_id);

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
                    crate::core::cache::LATEST_VERSION_CACHE
                        .invalidate(&dir_key)
                        .await;
                    crate::core::cache::LATEST_VERSION_CACHE
                        .insert(dir_key, new_ver)
                        .await;

                    let file_key = format!("{}/{}", self.root_uri, path);
                    crate::core::cache::MANIFEST_CACHE
                        .insert(file_key, Arc::new(new_manifest.clone()))
                        .await;

                    return Ok(new_manifest);
                }
                Err(e) if is_already_exists(&e) => {
                    metrics::counter!("hyperstreamdb_manifest_commit_retries_total").increment(1);
                    if attempt % 10 == 0 || attempt > 90 {
                        tracing::debug!(
                            "Conflict committing Manifest v{} (attempt {}), retrying...",
                            new_ver,
                            attempt + 1
                        );
                    }
                    let base_delay = 10 * (2u64.pow(attempt.min(5) as u32));
                    let jitter = rand::random::<u64>() % base_delay;
                    tokio::time::sleep(std::time::Duration::from_millis(base_delay + jitter)).await;
                    continue;
                }
                Err(e) => {
                    return Err(e.into());
                }
            }
        }

        Err(anyhow::anyhow!(
            "Failed to commit manifest after {} attempts due to concurrent updates",
            max_retries
        ))
    }

    /// Commit a set of imported entries (merges with current state)
    pub async fn commit_imported_entries(&self, entries: Vec<ManifestEntry>) -> Result<Manifest> {
        let dist_lock_path = Path::from(format!("{}/commit.lock", self.manifest_dir));
        let dist_lock =
            crate::core::lock::FileBasedLock::new(self.store.clone(), dist_lock_path, 30);
        let _dist_guard = dist_lock.acquire().await?;

        let max_retries = 10;
        let mut attempt = 0;
        loop {
            let (current_manifest, current_ver) = self.load_latest().await?;
            let all_existing = self.load_all_entries(&current_manifest).await?;

            // Merge entries, avoid duplicates, favor NEW entries for the SAME file_path
            let mut entry_map: HashMap<String, ManifestEntry> = all_existing
                .into_iter()
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
                    tracing::info!(
                        "Imported {} external entries into Manifest v{}",
                        entries.len(),
                        new_ver
                    );
                    let dir_key = self.get_dir_cache_key();
                    crate::core::cache::LATEST_VERSION_CACHE
                        .invalidate(&dir_key)
                        .await;
                    let file_key = self.get_cache_key(&path);
                    crate::core::cache::MANIFEST_CACHE
                        .insert(file_key, Arc::new(new_manifest.clone()))
                        .await;
                    return Ok(new_manifest);
                }
                Err(e) if is_already_exists(&e) => {
                    attempt += 1;
                    if attempt >= max_retries {
                        break;
                    }
                    tracing::warn!(
                        "Manifest conflict during entry import. Retrying attempt {}/{}",
                        attempt,
                        max_retries
                    );
                    tokio::time::sleep(std::time::Duration::from_millis(20 * attempt)).await;
                    continue;
                }
                Err(e) => {
                    return Err(e.into());
                }
            }
        }
        Err(anyhow::anyhow!(
            "Failed to commit imported entries after {} attempts",
            max_retries
        ))
    }

    /// Atomically commit a full manifest (optimistic concurrency)
    pub async fn commit_manifest(&self, manifest: Manifest) -> Result<()> {
        let max_retries = 10;
        let mut attempt = 0;
        loop {
            let (_, current_ver) = self.load_latest().await?;
            if manifest.version != current_ver + 1 {
                return Err(anyhow::anyhow!(
                    "Manifest version mismatch: expected {}, got {}",
                    current_ver + 1,
                    manifest.version
                ));
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
                    crate::core::cache::LATEST_VERSION_CACHE
                        .invalidate(&dir_key)
                        .await;
                    let file_key = self.get_cache_key(&path);
                    crate::core::cache::MANIFEST_CACHE
                        .insert(file_key, Arc::new(manifest.clone()))
                        .await;
                    return Ok(());
                }
                Err(e) if is_already_exists(&e) => {
                    attempt += 1;
                    if attempt >= max_retries {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    continue;
                }
                Err(e) => return Err(e.into()),
            }
        }
        Err(anyhow::anyhow!(
            "Failed to commit manifest after {} attempts",
            max_retries
        ))
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
                    tracing::info!(
                        "Rolled back Manifest to v{} (from snapshot {})",
                        new_ver,
                        version
                    );
                    let dir_key = format!("{}/{}", self.root_uri, self.manifest_dir);
                    crate::core::cache::LATEST_VERSION_CACHE
                        .invalidate(&dir_key)
                        .await;
                    let file_key = format!("{}/{}", self.root_uri, path);
                    crate::core::cache::MANIFEST_CACHE
                        .insert(file_key, Arc::new(new_manifest.clone()))
                        .await;
                    return Ok(new_manifest);
                }
                Err(e) if is_already_exists(&e) => {
                    attempt += 1;
                    if attempt >= max_retries {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                    continue;
                }
                Err(e) => return Err(e.into()),
            }
        }
        Err(anyhow::anyhow!(
            "Failed to rollback after {} attempts",
            max_retries
        ))
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

        let start_ver = latest_ver
            .saturating_sub(retention_versions as u64 - 1)
            .max(1);

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
            let is_data_file = path_str.ends_with(".parquet")
                || path_str.ends_with(".hnsw")
                || path_str.ends_with(".idx")
                || path_str.ends_with(".tmp");

            if is_data_file {
                // If it's not in the active set, delete it
                if !active_files.contains(&path_str) {
                    // Small safety: don't delete very young .tmp files (leeway for active writers)
                    if path_str.ends_with(".tmp") {
                        let age = Utc::now()
                            - chrono::DateTime::from_timestamp(meta.last_modified.timestamp(), 0)
                                .unwrap_or(Utc::now());
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
