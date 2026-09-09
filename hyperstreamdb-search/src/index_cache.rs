// Copyright (c) 2026 Richard Albright. All rights reserved.

//! On-demand index-file cache for the search API.
//!
//! Index files (inverted `*.inv.parquet`, BM25 stats `*.inv.meta.json`, and
//! HNSW `*.hnsw*`) are fetched from the object store on demand and cached
//! locally so repeated searches do not re-fetch them. The cache is:
//!
//! - **LRU** — least-recently-used entries are evicted first.
//! - **Size-capped** — bounded by `HYPERSEARCH_INDEX_CACHE_GB` (default 2 GB).
//! - **Versioned** — keyed by `(index, segment_id, column, file, manifest_version)`
//!   so a new segment / refresh (which bumps the manifest version) invalidates
//!   stale entries.
//!
//! This is a self-contained, in-process cache (the spec permits adding the cache
//! layer in the search crate). It is unit-tested for eviction and invalidation.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Instant;

/// Default cache cap in GiB when `HYPERSEARCH_INDEX_CACHE_GB` is unset.
const DEFAULT_CACHE_GB: u64 = 2;

/// Identifies a single cached index file.
///
/// Two entries are distinct unless all five components match; a change in
/// `manifest_version` (a new segment or refresh) yields a new key, which is how
/// staleness is avoided without explicit invalidation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct IndexFileKey {
    pub index: String,
    pub segment_id: String,
    pub column: String,
    /// File discriminator, e.g. `"inv.parquet"`, `"inv.meta.json"`, `"hnsw"`.
    pub file: String,
    pub manifest_version: u64,
}

impl IndexFileKey {
    pub fn new(
        index: impl Into<String>,
        segment_id: impl Into<String>,
        column: impl Into<String>,
        file: impl Into<String>,
        manifest_version: u64,
    ) -> Self {
        Self {
            index: index.into(),
            segment_id: segment_id.into(),
            column: column.into(),
            file: file.into(),
            manifest_version,
        }
    }
}

struct Entry {
    bytes: Vec<u8>,
    size: u64,
    #[allow(dead_code)]
    last_access: Instant,
}

struct Inner {
    entries: HashMap<IndexFileKey, Entry>,
    /// LRU order: index 0 is the least-recently-used, last is most-recent.
    order: Vec<IndexFileKey>,
    current_size: u64,
    max_size: u64,
}

/// An LRU, size-capped cache of index-file bytes.
#[derive(Clone)]
pub struct IndexFileCache {
    inner: std::sync::Arc<Mutex<Inner>>,
}

impl IndexFileCache {
    /// Create a cache with an explicit byte cap.
    pub fn with_capacity(max_bytes: u64) -> Self {
        Self {
            inner: std::sync::Arc::new(Mutex::new(Inner {
                entries: HashMap::new(),
                order: Vec::new(),
                current_size: 0,
                max_size: max_bytes,
            })),
        }
    }

    /// Create a cache sized from `HYPERSEARCH_INDEX_CACHE_GB` (default 2 GiB).
    pub fn from_env() -> Self {
        let gb = std::env::var("HYPERSEARCH_INDEX_CACHE_GB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(DEFAULT_CACHE_GB);
        Self::with_capacity(gb.saturating_mul(1024 * 1024 * 1024))
    }

    /// The configured byte cap.
    pub fn max_size(&self) -> u64 {
        self.inner.lock().unwrap().max_size
    }

    /// Total bytes currently cached.
    pub fn current_size(&self) -> u64 {
        self.inner.lock().unwrap().current_size
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().entries.len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Look up a cached file, marking it most-recently-used. Returns a clone of
    /// the bytes on a hit, or `None` on a miss.
    pub fn get(&self, key: &IndexFileKey) -> Option<Vec<u8>> {
        let mut inner = self.inner.lock().unwrap();
        // Move to the back (most-recent) of the LRU order.
        if let Some(pos) = inner.order.iter().position(|k| k == key) {
            inner.order.swap_remove(pos);
            inner.order.push(key.clone());
        }
        inner.entries.get(key).map(|e| e.bytes.clone())
    }

    /// Insert (or replace) a file, then evict least-recently-used entries until
    /// the cache is within its byte cap.
    pub fn put(&self, key: IndexFileKey, bytes: Vec<u8>) {
        let size = bytes.len() as u64;
        let mut inner = self.inner.lock().unwrap();

        // Replace an existing entry: account for the size delta first.
        if let Some(old) = inner.entries.remove(&key) {
            inner.current_size = inner.current_size.saturating_sub(old.size);
            if let Some(pos) = inner.order.iter().position(|k| k == &key) {
                inner.order.swap_remove(pos);
            }
        }

        inner.entries.insert(
            key.clone(),
            Entry {
                bytes,
                size,
                last_access: Instant::now(),
            },
        );
        inner.order.push(key.clone());
        inner.current_size += size;

        // Evict LRU entries while over the cap. A single entry larger than the
        // cap is kept (it was just requested) but will be evicted on the next
        // insert; this avoids an infinite loop.
        while inner.current_size > inner.max_size && inner.order.len() > 1 {
            if let Some(pos) = inner.order.iter().position(|k| k == &inner.order[0]) {
                let lru = inner.order.swap_remove(0);
                if let Some(evicted) = inner.entries.remove(&lru) {
                    inner.current_size = inner.current_size.saturating_sub(evicted.size);
                }
                let _ = pos;
            } else {
                break;
            }
        }
    }

    /// Remove all entries for an index (e.g. after `DELETE /{index}` or a
    /// refresh that rewrites its segments). Returns the number removed.
    pub fn invalidate_index(&self, index: &str) -> usize {
        let mut inner = self.inner.lock().unwrap();
        let to_remove: Vec<IndexFileKey> = inner
            .entries
            .keys()
            .filter(|k| k.index == index)
            .cloned()
            .collect();
        let mut removed = 0;
        for key in to_remove {
            if let Some(evicted) = inner.entries.remove(&key) {
                inner.current_size = inner.current_size.saturating_sub(evicted.size);
                removed += 1;
            }
            if let Some(pos) = inner.order.iter().position(|k| k == &key) {
                inner.order.swap_remove(pos);
            }
        }
        removed
    }

    /// Remove entries for an index whose `manifest_version` is strictly older
    /// than `version` (a refresh bumped the version, so older segments' cached
    /// files are stale). Returns the number removed.
    pub fn invalidate_older_versions(&self, index: &str, version: u64) -> usize {
        let mut inner = self.inner.lock().unwrap();
        let to_remove: Vec<IndexFileKey> = inner
            .entries
            .keys()
            .filter(|k| k.index == index && k.manifest_version < version)
            .cloned()
            .collect();
        let mut removed = 0;
        for key in to_remove {
            if let Some(evicted) = inner.entries.remove(&key) {
                inner.current_size = inner.current_size.saturating_sub(evicted.size);
                removed += 1;
            }
            if let Some(pos) = inner.order.iter().position(|k| k == &key) {
                inner.order.swap_remove(pos);
            }
        }
        removed
    }
}

impl Default for IndexFileCache {
    fn default() -> Self {
        Self::from_env()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(index: &str, seg: &str, col: &str, file: &str, ver: u64) -> IndexFileKey {
        IndexFileKey::new(index, seg, col, file, ver)
    }

    #[test]
    fn get_miss_returns_none() {
        let cache = IndexFileCache::with_capacity(1024);
        assert!(cache.get(&key("i", "s", "c", "f", 1)).is_none());
    }

    #[test]
    fn put_then_get_roundtrips() {
        let cache = IndexFileCache::with_capacity(1024);
        cache.put(key("i", "s", "c", "f", 1), vec![1, 2, 3]);
        assert_eq!(cache.get(&key("i", "s", "c", "f", 1)), Some(vec![1, 2, 3]));
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.current_size(), 3);
    }

    #[test]
    fn distinct_keys_are_distinct_entries() {
        let cache = IndexFileCache::with_capacity(4096);
        cache.put(key("i", "s1", "c", "f", 1), vec![1]);
        cache.put(key("i", "s2", "c", "f", 1), vec![2]);
        cache.put(key("i", "s1", "c", "f", 2), vec![3]); // newer manifest version
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.get(&key("i", "s1", "c", "f", 1)), Some(vec![1]));
        assert_eq!(cache.get(&key("i", "s1", "c", "f", 2)), Some(vec![3]));
    }

    #[test]
    fn evicts_lru_when_over_cap() {
        // Cap of 10 bytes; each entry is 4 bytes.
        let cache = IndexFileCache::with_capacity(10);
        cache.put(key("i", "a", "c", "f", 1), vec![0; 4]);
        cache.put(key("i", "b", "c", "f", 1), vec![0; 4]);
        // Third insert pushes to 12 > 10, evicting the LRU ("a").
        cache.put(key("i", "c", "c", "f", 1), vec![0; 4]);
        assert!(cache.get(&key("i", "a", "c", "f", 1)).is_none());
        assert!(cache.get(&key("i", "b", "c", "f", 1)).is_some());
        assert!(cache.get(&key("i", "c", "c", "f", 1)).is_some());
        assert!(cache.current_size() <= 10);
    }

    #[test]
    fn access_refreshes_lru_order() {
        let cache = IndexFileCache::with_capacity(10);
        cache.put(key("i", "a", "c", "f", 1), vec![0; 4]);
        cache.put(key("i", "b", "c", "f", 1), vec![0; 4]);
        // Touch "a" so "b" becomes the LRU.
        let _ = cache.get(&key("i", "a", "c", "f", 1));
        cache.put(key("i", "c", "c", "f", 1), vec![0; 4]);
        // "b" (now LRU) is evicted, "a" survives.
        assert!(cache.get(&key("i", "b", "c", "f", 1)).is_none());
        assert!(cache.get(&key("i", "a", "c", "f", 1)).is_some());
    }

    #[test]
    fn replace_same_key_does_not_double_count_size() {
        let cache = IndexFileCache::with_capacity(1024);
        cache.put(key("i", "s", "c", "f", 1), vec![0; 8]);
        assert_eq!(cache.current_size(), 8);
        cache.put(key("i", "s", "c", "f", 1), vec![0; 4]);
        assert_eq!(cache.current_size(), 4);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn invalidate_index_removes_only_that_index() {
        let cache = IndexFileCache::with_capacity(4096);
        cache.put(key("i1", "s", "c", "f", 1), vec![0; 4]);
        cache.put(key("i1", "s2", "c", "f", 1), vec![0; 4]);
        cache.put(key("i2", "s", "c", "f", 1), vec![0; 4]);
        let removed = cache.invalidate_index("i1");
        assert_eq!(removed, 2);
        assert!(cache.get(&key("i1", "s", "c", "f", 1)).is_none());
        assert!(cache.get(&key("i2", "s", "c", "f", 1)).is_some());
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn invalidate_older_versions_keeps_current() {
        let cache = IndexFileCache::with_capacity(4096);
        cache.put(key("i", "s", "c", "f", 1), vec![0; 4]);
        cache.put(key("i", "s", "c", "f", 2), vec![0; 4]);
        cache.put(key("i", "s", "c", "f", 3), vec![0; 4]);
        // A refresh bumped the version to 3; versions < 3 are stale.
        let removed = cache.invalidate_older_versions("i", 3);
        assert_eq!(removed, 2);
        assert!(cache.get(&key("i", "s", "c", "f", 1)).is_none());
        assert!(cache.get(&key("i", "s", "c", "f", 2)).is_none());
        assert!(cache.get(&key("i", "s", "c", "f", 3)).is_some());
    }

    #[test]
    fn from_env_uses_default_when_unset() {
        // Remove the var so the default (2 GiB) applies.
        std::env::remove_var("HYPERSEARCH_INDEX_CACHE_GB");
        let cache = IndexFileCache::from_env();
        assert_eq!(cache.max_size(), 2 * 1024 * 1024 * 1024);
    }
}
