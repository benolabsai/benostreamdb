// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Manifest loading operations: fetching, caching, and walking manifest history.

use crate::core::cache::CacheExt;
use anyhow::Result;
use futures::StreamExt;
use object_store::path::Path;
use std::collections::HashMap;

use super::super::types::*;
use super::ManifestManager;

impl ManifestManager {
    /// Load the latest manifest bypassing LATEST_VERSION_CACHE.
    pub async fn load_latest_direct(&self) -> Result<(Manifest, u64)> {
        let mut max_ver = 0;
        let mut latest_loc = None;

        let mut stream = self.store.list(Some(&self.manifest_dir));
        while let Some(meta) = stream.next().await {
            let meta = meta?;
            let path = meta.location.as_ref();
            if path.ends_with(".json") {
                if let Some(filename) = path.split('/').next_back() {
                    if filename.starts_with('v') && filename.ends_with(".json") {
                        if let Ok(ver) = filename[1..filename.len() - 5].parse::<u64>() {
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
        if let Some(ver) = crate::core::cache::LATEST_VERSION_CACHE
            .get_with_metrics(&cache_key, "latest_version")
            .await
        {
            tracing::debug!(
                "ManifestManager::load_latest: Found version {} in LATEST_VERSION_CACHE",
                ver
            );
            if let Ok(manifest) = self.load_version(ver).await {
                tracing::debug!(
                    "ManifestManager::load_latest: Cache hit v{} (entries={})",
                    ver,
                    manifest.entries.len()
                );
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
                    let ver_str = &filename[1..filename.len() - 5]; // strip 'v' and '.json'
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
            crate::core::cache::LATEST_VERSION_CACHE
                .insert(cache_key, max_ver)
                .await;
        }

        if let Some(path) = latest_path {
            tracing::debug!(
                "ManifestManager::load_latest: Found version {} on disk at {:?}",
                max_ver,
                path
            );
            return match self.load_version(max_ver).await {
                Ok(m) => {
                    tracing::debug!(
                        "ManifestManager::load_latest: Successfully loaded v{} (entries={})",
                        max_ver,
                        m.entries.len()
                    );
                    Ok((m, max_ver))
                }
                Err(e) => {
                    tracing::error!(
                        "ManifestManager::load_latest: Failed to load v{} via load_version: {}",
                        max_ver,
                        e
                    );
                    // Fallback if somehow listing said it exists but we can't read it
                    let bytes = self.store.get(&path).await?.bytes().await?;
                    let manifest: Manifest = serde_json::from_slice(&bytes)?;
                    Ok((manifest, max_ver))
                }
            };
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
        if let Some(manifest) = crate::core::cache::MANIFEST_CACHE
            .get_with_metrics(&cache_key, "manifest")
            .await
        {
            return Ok(manifest.as_ref().clone());
        }

        // 2. Fetch from S3
        let bytes = self.store.get(&path).await?.bytes().await?;
        let manifest: Manifest = serde_json::from_slice(&bytes)?;

        // 3. Populate Cache
        crate::core::cache::MANIFEST_CACHE
            .insert(cache_key, std::sync::Arc::new(manifest.clone()))
            .await;

        Ok(manifest)
    }

    /// Load a manifest list from a specific path
    pub async fn load_manifest_list(&self, path_str: &str) -> Result<ManifestList> {
        let path = Path::from(path_str);
        let cache_key = format!("{}/{}", self.root_uri, path);

        if let Some(list) = crate::core::cache::MANIFEST_LIST_CACHE
            .get_with_metrics(&cache_key, "manifest_list")
            .await
        {
            return Ok(list.as_ref().clone());
        }

        let bytes = self.store.get(&path).await?.bytes().await?;

        if path_str.ends_with(".avro") {
            let iceberg_list = crate::core::iceberg::read_manifest_list(&bytes[..])?;
            let manifest_files = iceberg_list
                .into_iter()
                .map(|e| {
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
                })
                .collect();
            let list = ManifestList { manifest_files };
            crate::core::cache::MANIFEST_LIST_CACHE
                .insert(cache_key, std::sync::Arc::new(list.clone()))
                .await;
            return Ok(list);
        }

        let list: ManifestList = serde_json::from_slice(&bytes)?;

        crate::core::cache::MANIFEST_LIST_CACHE
            .insert(cache_key, std::sync::Arc::new(list.clone()))
            .await;
        Ok(list)
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
            let schema = manifest
                .schemas
                .iter()
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
                            Self::load_avro_manifest_static(
                                store, entry_path, s, table_spec, root_uri,
                            )
                            .await
                        } else {
                            Self::load_avro_manifest_static(
                                store,
                                entry_path,
                                Schema::default(),
                                table_spec,
                                root_uri,
                            )
                            .await
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
        Self::load_manifest_static(
            self.store.clone(),
            path_str.to_string(),
            self.root_uri.clone(),
        )
        .await
    }

    pub async fn load_avro_manifest(
        &self,
        path_str: &str,
        schema: &Schema,
        spec: &PartitionSpec,
    ) -> Result<Manifest> {
        Self::load_avro_manifest_static(
            self.store.clone(),
            path_str.to_string(),
            schema.clone(),
            spec.clone(),
            self.root_uri.clone(),
        )
        .await
    }

    async fn load_avro_manifest_static(
        store: std::sync::Arc<dyn object_store::ObjectStore>,
        path_str: String,
        schema: Schema,
        spec: PartitionSpec,
        root_uri: String,
    ) -> Result<Manifest> {
        let path = Path::from(path_str);
        let cache_key = format!("{}/{}", root_uri, path);

        if let Some(manifest) = crate::core::cache::MANIFEST_CACHE
            .get_with_metrics(&cache_key, "manifest")
            .await
        {
            return Ok(manifest.as_ref().clone());
        }

        let bytes = store.get(&path).await?.bytes().await?;
        let iceberg_entries = crate::core::iceberg::read_manifest(&bytes[..])?;

        let mut data_entries = Vec::new();
        let mut delete_files = Vec::new();

        for ie in iceberg_entries {
            if ie.status == 0 || ie.status == 1 {
                // EXISTING or ADDED
                match crate::core::iceberg::convert_iceberg_to_object(&ie, &schema, &spec)? {
                    crate::core::iceberg::IcebergManifestObject::Data(me) => data_entries.push(me),
                    crate::core::iceberg::IcebergManifestObject::Delete(df) => {
                        delete_files.push(df)
                    }
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
        crate::core::cache::MANIFEST_CACHE
            .insert(cache_key, std::sync::Arc::new(manifest.clone()))
            .await;
        Ok(manifest)
    }

    async fn load_manifest_static(
        store: std::sync::Arc<dyn object_store::ObjectStore>,
        path_str: String,
        root_uri: String,
    ) -> Result<Manifest> {
        let path = Path::from(path_str);
        let cache_key = format!("{}/{}", root_uri, path);

        if let Some(manifest) = crate::core::cache::MANIFEST_CACHE
            .get_with_metrics(&cache_key, "manifest")
            .await
        {
            return Ok(manifest.as_ref().clone());
        }

        let bytes = store.get(&path).await?.bytes().await?;
        let manifest: Manifest = serde_json::from_slice(&bytes)?;

        crate::core::cache::MANIFEST_CACHE
            .insert(cache_key, std::sync::Arc::new(manifest.clone()))
            .await;
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
            if prev == 0 {
                break;
            }

            match self.load_version(prev).await {
                Ok(m) => {
                    history.push(m.clone());
                    current = m;
                }
                Err(e) => {
                    tracing::warn!("Broken manifest chain at v{}: {}", prev, e);
                    break;
                }
            }
        }

        Ok(history)
    }
}
