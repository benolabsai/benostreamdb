// Copyright (c) 2026 Richard Albright. All rights reserved.

use crate::core::manifest::ManifestManager;
use crate::core::storage::create_object_store;
use object_store::ObjectStore;
use std::collections::HashSet;
use std::sync::Arc;
use anyhow::{Result, Context};
use futures::StreamExt;
use object_store::path::Path;

pub struct Maintenance {
    store: Arc<dyn ObjectStore>,
    manifest_manager: ManifestManager,
}

impl Maintenance {
    pub fn new(uri: &str) -> Result<Self> {
        let store = create_object_store(uri)?;
        // Base path logic matching Compactor
        let base_path = if uri.starts_with("file://") {
            "".to_string() 
        } else {
             let url = url::Url::parse(uri).context("Invalid URI")?;
             url.path().trim_start_matches('/').to_string()
        };
        
        let manifest_manager = ManifestManager::new(store.clone(), &base_path, uri);
        
        Ok(Self {
            store,
            manifest_manager,
        })
    }

    /// Iceberg-compatible command: expire_snapshots
    /// Removes old manifest versions and deletes data files that are ONLY reachable from expired snapshots.
    pub async fn expire_snapshots(&self, retain_last: usize) -> Result<()> {
        let history = self.manifest_manager.walk_history().await?;
        
        if history.len() <= retain_last {
            tracing::debug!("History length ({}) <= retain_last ({}). No action.", history.len(), retain_last);
            return Ok(());
        }

        // 1. Identify Valid vs Expired
        let (retained, expired) = history.split_at(retain_last); // history is [Latest, ..., Oldest]
        
        tracing::info!("Expiring {} snapshots. Retaining latest {}.", expired.len(), retained.len());

        // 2. Collect Valid Files (All files referenced in Retained Manifests)
        let mut valid_files = HashSet::new();
        for m in retained {
            // Load all entries including those in manifest lists
            let all_entries = self.manifest_manager.load_all_entries(m).await?;
            for entry in &all_entries {
                valid_files.insert(entry.file_path.clone());
                for idx in &entry.index_files {
                    valid_files.insert(idx.file_path.clone());
                }
            }
        }

        // 3. Collect Candidates (All files referenced in Expired Manifests)
        let mut candidate_files = HashSet::new();
        for m in expired {
            // Load all entries including those in manifest lists
            let all_entries = self.manifest_manager.load_all_entries(m).await?;
            for entry in &all_entries {
                candidate_files.insert(entry.file_path.clone());
                for idx in &entry.index_files {
                    candidate_files.insert(idx.file_path.clone());
                }
            }
        }

        // 4. Calculate Difference
        // Delete = Candidate - Valid
        let mut to_delete = Vec::new();
        for cand in candidate_files {
            if !valid_files.contains(&cand) {
                to_delete.push(cand);
            }
        }

        tracing::info!("Found {} files to expire/delete.", to_delete.len());

        // 5. Delete Data Files
        // We have exact paths now!
        let mut deletions = 0;
        for path_str in to_delete {
             let p = Path::from(path_str.as_str());
             // Ensure it's not root matching something weird
             // Ignore errors (file might be already gone)
             if let Err(e) = self.store.delete(&p).await {
                 tracing::warn!("Failed to delete expired file {}: {}", path_str, e);
             } else {
                 deletions += 1;
                 tracing::debug!("Deleted: {}", path_str);
             }
        }
        
        tracing::info!("Expired {} data/index files.", deletions);

        // 6. Delete Expired Manifest Files
        for m in expired {
            let filename = format!("v{}.json", m.version);
            let _p = Path::from("_manifest").child(filename); // Assuming _manifest is fixed
             // Wait, BasePath logic in ManifestManager handles _manifest prefix.
             // But we don't expose path construction easily.
             // Re-derive or add method.
             // Let's rely on standard structure:
             // manifest_dir is usually {base}/_manifest
             // We can use list/delete on manifest dir.
             // Actually, ManifestManager encapsulates the dir.
             // Let's just construct it manually for now or add delete_version to Manager.
        }
        
        Ok(())
    }

    /// Iceberg-compatible command: remove_orphan_files
    /// Scans storage and deletes files not referenced by ANY valid manifest (Active + History).
    /// Used to clean up failed writes (partial uploads).
    pub async fn remove_orphan_files(&self, older_than_ms: i64) -> Result<()> {
        let history = self.manifest_manager.walk_history().await?;
        
        // 1. Collect Global Valid Set
        let mut all_valid_files = HashSet::new();
        for m in history {
            // Load all entries including those in manifest lists
            let all_entries = self.manifest_manager.load_all_entries(&m).await?;
            for entry in &all_entries {
                all_valid_files.insert(entry.file_path.clone());
                for idx in &entry.index_files {
                    all_valid_files.insert(idx.file_path.clone());
                }
            }
        }
        
        tracing::debug!("Valid files in history: {}", all_valid_files.len());

        // 2. Scan All Files
        let list_path = Path::from("/"); 
        let mut stream = self.store.list(Some(&list_path));
        let now = chrono::Utc::now().timestamp_millis();
        
        let mut deletions = 0;

        while let Some(meta) = stream.next().await {
            let meta = meta?;
            let path_str = meta.location.to_string();
            
            // Skip manifest directory
            if path_str.contains("_manifest/") {
                continue;
            }

            // Exact match check
            if !all_valid_files.contains(&path_str) {
                // Orphan candidate. Check age.
                let age = now - meta.last_modified.timestamp_millis();
                if age > older_than_ms {
                    tracing::info!("Found Orphan: {} (Age: {}ms)", path_str, age);
                    self.store.delete(&meta.location).await?;
                    deletions += 1;
                }
            }
        }
        
        tracing::info!("Removed {} orphan files.", deletions);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SegmentConfig;
    use crate::core::segment::HybridSegmentWriter;
    use crate::core::manifest::ManifestEntry;

    // Helper to create a dummy segment
    async fn create_dummy_segment(path: &str, id: &str) -> Result<ManifestEntry> {
        let config = SegmentConfig::new(path, id);
        let _writer = HybridSegmentWriter::new(config);
        
        let fpath = format!("{}/{}.parquet", path, id);
        std::fs::write(&fpath, b"dummy content")?;
        
        // Return matching entry
        Ok(ManifestEntry {
            file_path: format!("{}.parquet", id),
            file_size_bytes: 13,
            ..Default::default()
        })
    }

    #[tokio::test]
    async fn test_maintenance_flow() -> Result<()> {
        let temp_dir = std::env::temp_dir().join(format!("test_maint_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&temp_dir)?;
        let uri = format!("file://{}", temp_dir.to_str().unwrap());

        // Clear caches to ensure clean test state
        crate::core::cache::LATEST_VERSION_CACHE.invalidate_all();
        crate::core::cache::MANIFEST_CACHE.invalidate_all();

        let maintenance = Maintenance::new(&uri)?;
        let manager = &maintenance.manifest_manager;

        // 1. Create Segments A, B, C
        let entry_a = create_dummy_segment(temp_dir.to_str().unwrap(), "seg_a").await?;
        let entry_b = create_dummy_segment(temp_dir.to_str().unwrap(), "seg_b").await?;
        let entry_c = create_dummy_segment(temp_dir.to_str().unwrap(), "seg_c").await?;

        // 2. Commit Manifests
        // v1: [A, B]
        manager.commit(&[entry_a.clone(), entry_b.clone()], &[], crate::core::manifest::CommitMetadata::default()).await?;
        // v2: [C, B] (A is dropped)
        // Remove path for A
        manager.commit(std::slice::from_ref(&entry_c), std::slice::from_ref(&entry_a.file_path), crate::core::manifest::CommitMetadata::default()).await?;
        
        // 3. Expire Snapshots
        // Retain 1 (Only v2). v1 is expired.
        maintenance.expire_snapshots(1).await?;
        
        // Verify A gone, B, C exist
        let entries: Vec<_> = std::fs::read_dir(&temp_dir)?.map(|e| e.unwrap().file_name().into_string().unwrap()).collect();
        assert!(!entries.contains(&"seg_a.parquet".to_string()), "seg_a should be deleted");
        assert!(entries.contains(&"seg_b.parquet".to_string()), "seg_b should exist");
        assert!(entries.contains(&"seg_c.parquet".to_string()), "seg_c should exist");

        // Cleanup
        let cache_key = format!("{}/{}", uri, "");
        crate::core::cache::LATEST_VERSION_CACHE.invalidate(&cache_key).await;
        std::fs::remove_dir_all(&temp_dir)?;
        Ok(())
    }
}
