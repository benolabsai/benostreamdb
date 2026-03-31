// Copyright (c) 2026 Richard Albright. All rights reserved.

use moka::future::Cache;
use once_cell::sync::Lazy;
use std::sync::Arc;
use crate::core::manifest::{Manifest, ManifestList};
use crate::core::index::hnsw_ivf::HnswIvfIndex;
use roaring::RoaringBitmap;
use std::time::Duration;
use hnsw_rs::prelude::Hnsw;
use hnsw_rs::dist::DistL2;
use arrow::record_batch::RecordBatch;
use parquet::file::metadata::ParquetMetaData;
use std::path::PathBuf;
use object_store::ObjectStore;
use anyhow::Result;

pub static DISK_CACHE_DIR: Lazy<Option<PathBuf>> = Lazy::new(|| {
    std::env::var("HYPERSTREAM_DISK_CACHE_DIR")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            let path = std::path::Path::new("/tmp/hdb_cache");
            if std::fs::create_dir_all(path).is_ok() {
                Some(path.to_path_buf())
            } else {
                None
            }
        })
});

#[derive(Clone)]
pub struct DiskCache {
    store: Arc<dyn ObjectStore>,
    cache_dir: Option<PathBuf>,
}

impl DiskCache {
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        Self {
            store,
            cache_dir: DISK_CACHE_DIR.clone(),
        }
    }

    pub async fn get_bytes(&self, path: &str) -> Result<bytes::Bytes> {
        use sha2::{Digest, Sha256};
        
        if let Some(cache_dir) = &self.cache_dir {
            let mut hasher = Sha256::new();
            hasher.update(path);
            let hash = format!("{:x}", hasher.finalize());
            let cache_path = cache_dir.join(&hash);
            
            if cache_path.exists() {
                if let Ok(b) = std::fs::read(&cache_path) {
                    return Ok(bytes::Bytes::from(b));
                }
            }

            let b = self.store.get(&object_store::path::Path::from(path)).await?.bytes().await?;
            let _ = std::fs::write(&cache_path, &b);
            Ok(b)
        } else {
            let res = self.store.get(&object_store::path::Path::from(path)).await
                .map_err(|e| anyhow::anyhow!(e))?;
            res.bytes().await.map_err(|e| anyhow::anyhow!(e))
        }
    }
}

// Cache Keys
// Manifest: "s3://bucket/path/_manifest" -> Manifest
// Version: "s3://bucket/path/_manifest" -> u64 (Latest Version)
// Index: "s3://bucket/path/segment_id/column" -> RoaringBitmap
//
// NOTE: We intentionally do NOT cache HNSW indexes or Parquet data.
// At petabyte scale, caching would exhaust memory. The design relies on:
// - Streaming only needed data via object store range requests
// - Efficient indexes (roaring bitmaps, HNSW graphs) that are small
// - Client-side filtering to minimize data transfer

pub static MANIFEST_CACHE: Lazy<Cache<String, Arc<Manifest>>> = Lazy::new(|| {
    Cache::builder()
        .max_capacity(1000)
        .time_to_live(Duration::from_secs(60 * 60)) // 1 hour for immutable vN.json
        .build()
});

pub static MANIFEST_LIST_CACHE: Lazy<Cache<String, Arc<ManifestList>>> = Lazy::new(|| {
    Cache::builder()
        .max_capacity(1000)
        .time_to_live(Duration::from_secs(60 * 60))
        .build()
});

pub static LATEST_VERSION_CACHE: Lazy<Cache<String, u64>> = Lazy::new(|| {
    Cache::builder()
        .max_capacity(500)
        .time_to_live(Duration::from_secs(2)) // 2 seconds TTL for consistency
        .build()
});

pub static INDEX_CACHE: Lazy<Cache<String, Arc<RoaringBitmap>>> = Lazy::new(|| {
    Cache::builder()
        .max_capacity(10_000) // Objects - bitmaps are tiny (~KB each)
        .time_to_idle(Duration::from_secs(60 * 5)) // 5 mins idle
        .build()
});

pub static BYTE_CACHE: Lazy<Cache<String, Arc<Vec<u8>>>> = Lazy::new(|| {
    Cache::builder()
        .max_capacity(1000) // 1000 small files (manifests, etc)
        .time_to_idle(Duration::from_secs(60 * 30)) // 30 mins
        .build()
});

pub static HNSW_CACHE: Lazy<Cache<String, Arc<Hnsw<f32, DistL2>>>> = Lazy::new(|| {
    Cache::builder()
        .max_capacity(500) // Cache up to 500 HNSW graphs
        .time_to_idle(Duration::from_secs(60 * 10)) // 10 mins idle
        .build()
});

/// Cache for HNSW-IVF hybrid indexes
/// These are more memory-efficient than plain HNSW since they only load needed clusters
pub static HNSW_IVF_CACHE: Lazy<Cache<String, Arc<HnswIvfIndex>>> = Lazy::new(|| {
    // Default to 2GB cache if not set
    let cache_gb: u64 = std::env::var("HYPERSTREAM_CACHE_GB")
        .unwrap_or_else(|_| "2".to_string())
        .parse()
        .unwrap_or(2);
    
    // Convert to KB to avoid u32 overflow in weigher (moka requirement)
    // u32::MAX KB = 4TB, which is plenty for a single item.
    let max_kb = cache_gb * 1024 * 1024; 

    tracing::info!("Initializing HNSW-IVF Cache with {} GB limit", cache_gb);

    Cache::builder()
        .weigher(|_key, value: &Arc<HnswIvfIndex>| -> u32 {
            (value.size_in_bytes() / 1024) as u32
        })
        .max_capacity(max_kb) 
        .time_to_idle(Duration::from_secs(60 * 15)) // 15 mins idle
        .build()
});

pub static INVERTED_INDEX_CACHE: Lazy<Cache<String, Arc<Vec<RecordBatch>>>> = Lazy::new(|| {
    Cache::builder()
        .max_capacity(1000) // Cache 1000 decoded inverted index files
        .time_to_idle(Duration::from_secs(60 * 5)) 
        .build()
});

pub static PARQUET_META_CACHE: Lazy<Cache<String, (Arc<ParquetMetaData>, usize)>> = Lazy::new(|| {
    Cache::builder()
        .max_capacity(1000) // 1000 file footers (schema, row groups)
        .time_to_idle(Duration::from_secs(60 * 30)) 
        .build()
});

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Schema, Field, DataType};
    use arrow::array::Int32Array;
    use std::time::Duration as StdDuration;
    use tokio::time::sleep;

    #[tokio::test]
    async fn test_byte_cache_hit_miss() {
        let key = "test_byte_cache_key_1".to_string();
        let data = Arc::new(vec![1u8, 2, 3, 4, 5]);
        
        // Miss
        assert!(BYTE_CACHE.get(&key).await.is_none());
        
        // Insert
        BYTE_CACHE.insert(key.clone(), data.clone()).await;
        
        // Hit
        let cached = BYTE_CACHE.get(&key).await;
        assert!(cached.is_some());
        assert_eq!(*cached.unwrap(), *data);
        
        // Cleanup
        BYTE_CACHE.invalidate(&key).await;
    }

    #[tokio::test]
    async fn test_index_cache_hit_miss() {
        let key = "test_index_cache_key_1".to_string();
        let mut bitmap = RoaringBitmap::new();
        bitmap.insert(1);
        bitmap.insert(5);
        bitmap.insert(100);
        let bitmap = Arc::new(bitmap);
        
        // Miss
        assert!(INDEX_CACHE.get(&key).await.is_none());
        
        // Insert
        INDEX_CACHE.insert(key.clone(), bitmap.clone()).await;
        
        // Hit
        let cached = INDEX_CACHE.get(&key).await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().len(), 3);
        
        // Cleanup
        INDEX_CACHE.invalidate(&key).await;
    }

    #[tokio::test]
    async fn test_inverted_index_cache_hit_miss() {
        let key = "test_inverted_index_key_1".to_string();
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
        ]));
        
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        ).unwrap();
        
        let batches = Arc::new(vec![batch]);
        
        // Miss
        assert!(INVERTED_INDEX_CACHE.get(&key).await.is_none());
        
        // Insert
        INVERTED_INDEX_CACHE.insert(key.clone(), batches.clone()).await;
        
        // Hit
        let cached = INVERTED_INDEX_CACHE.get(&key).await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().len(), 1);
        
        // Cleanup
        INVERTED_INDEX_CACHE.invalidate(&key).await;
    }

    #[tokio::test]
    async fn test_latest_version_cache_hit_miss() {
        let key = "test_version_cache_key_1".to_string();
        let version = 42u64;
        
        // Miss
        assert!(LATEST_VERSION_CACHE.get(&key).await.is_none());
        
        // Insert
        LATEST_VERSION_CACHE.insert(key.clone(), version).await;
        
        // Hit
        let cached = LATEST_VERSION_CACHE.get(&key).await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), version);
        
        // Cleanup
        LATEST_VERSION_CACHE.invalidate(&key).await;
    }

    #[tokio::test]
    async fn test_cache_eviction_lru() {
        // Create a small cache for testing eviction
        let test_cache: Cache<String, Arc<Vec<u8>>> = Cache::builder()
            .max_capacity(3)
            .build();
        
        // Insert 3 items (at capacity)
        test_cache.insert("key1".to_string(), Arc::new(vec![1])).await;
        test_cache.insert("key2".to_string(), Arc::new(vec![2])).await;
        test_cache.insert("key3".to_string(), Arc::new(vec![3])).await;
        
        // Give cache time to process
        sleep(StdDuration::from_millis(200)).await;
        test_cache.run_pending_tasks().await;
        
        // All 3 should be present
        assert!(test_cache.get("key1").await.is_some());
        assert!(test_cache.get("key2").await.is_some());
        assert!(test_cache.get("key3").await.is_some());
        
        // Insert 4th and 5th items, should trigger evictions
        test_cache.insert("key4".to_string(), Arc::new(vec![4])).await;
        test_cache.insert("key5".to_string(), Arc::new(vec![5])).await;
        
        // Give significant time for eviction to process
        sleep(StdDuration::from_millis(500)).await;
        test_cache.run_pending_tasks().await;
        sleep(StdDuration::from_millis(200)).await;
        
        // Total entries should not exceed capacity
        let entry_count = test_cache.entry_count();
        assert!(entry_count <= 3, "Cache exceeded capacity: {} > 3", entry_count);
    }

    #[tokio::test]
    async fn test_cache_concurrent_access() {
        let key = "test_concurrent_key".to_string();
        let data = Arc::new(vec![1u8, 2, 3]);
        
        // Insert initial data
        BYTE_CACHE.insert(key.clone(), data.clone()).await;
        
        // Spawn multiple concurrent readers
        let mut handles = vec![];
        for i in 0..10 {
            let k = key.clone();
            let handle = tokio::spawn(async move {
                for _ in 0..100 {
                    let result = BYTE_CACHE.get(&k).await;
                    assert!(result.is_some(), "Reader {} failed to get cached value", i);
                }
            });
            handles.push(handle);
        }
        
        // Wait for all readers
        for handle in handles {
            handle.await.unwrap();
        }
        
        // Verify data is still correct
        let final_data = BYTE_CACHE.get(&key).await;
        assert!(final_data.is_some());
        assert_eq!(*final_data.unwrap(), *data);
        
        // Cleanup
        BYTE_CACHE.invalidate(&key).await;
    }

    #[tokio::test]
    async fn test_cache_invalidation() {
        let key = "test_invalidation_key".to_string();
        let data = Arc::new(vec![1u8, 2, 3]);
        
        // Insert
        BYTE_CACHE.insert(key.clone(), data.clone()).await;
        assert!(BYTE_CACHE.get(&key).await.is_some());
        
        // Invalidate
        BYTE_CACHE.invalidate(&key).await;
        
        // Should be gone
        assert!(BYTE_CACHE.get(&key).await.is_none());
    }

    #[tokio::test]
    async fn test_cache_key_uniqueness() {
        // Test that different tables/segments have unique cache keys
        let table1_key = "s3://bucket/table1/segment1.parquet".to_string();
        let table2_key = "s3://bucket/table2/segment1.parquet".to_string();
        
        let data1 = Arc::new(vec![1u8]);
        let data2 = Arc::new(vec![2u8]);
        
        BYTE_CACHE.insert(table1_key.clone(), data1.clone()).await;
        BYTE_CACHE.insert(table2_key.clone(), data2.clone()).await;
        
        // Both should be cached independently
        let cached1 = BYTE_CACHE.get(&table1_key).await.unwrap();
        let cached2 = BYTE_CACHE.get(&table2_key).await.unwrap();
        
        assert_eq!(*cached1, vec![1u8]);
        assert_eq!(*cached2, vec![2u8]);
        
        // Cleanup
        BYTE_CACHE.invalidate(&table1_key).await;
        BYTE_CACHE.invalidate(&table2_key).await;
    }
}
