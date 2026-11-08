// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::index::csr_graph::MmapCsrGraph;
use crate::core::index::hnsw_ivf::HnswIvfIndex;
use crate::core::index::hnsw_rs::dist::DistL2;
use crate::core::index::hnsw_rs::hnsw::Hnsw;
use crate::core::manifest::{Manifest, ManifestList};
use anyhow::Result;
use arrow::record_batch::RecordBatch;
use dashmap::DashMap;
use moka::future::Cache;
use object_store::ObjectStore;
use once_cell::sync::Lazy;
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::bloom_filter::Sbbf;
use roaring::RoaringBitmap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[async_trait::async_trait]
pub trait CacheExt<K, V> {
    async fn get_with_metrics(&self, key: &K, cache_name: &str) -> Option<V>;
}

#[async_trait::async_trait]
impl<K, V> CacheExt<K, V> for moka::future::Cache<K, V>
where
    K: std::hash::Hash + Eq + Send + Sync + 'static,
    V: Clone + Send + Sync + 'static,
{
    async fn get_with_metrics(&self, key: &K, cache_name: &str) -> Option<V> {
        if let Some(val) = self.get(key).await {
            crate::telemetry::metrics::CACHE_HITS_TOTAL
                .with_label_values(&[cache_name])
                .inc();
            Some(val)
        } else {
            crate::telemetry::metrics::CACHE_MISSES_TOTAL
                .with_label_values(&[cache_name])
                .inc();
            None
        }
    }
}

const GIB: u64 = 1024 * 1024 * 1024;

/// Default on-disk budget for [`DiskCache`] when `BSDB_DISK_CACHE_GB` is unset.
///
/// The disk cache exists to keep *remote* objects (S3/Azure/GCS) off the
/// network. It is not scratch space: with no budget it grows without bound, and
/// a single 10M-vector benchmark run wrote 409 GB of copies of indexes that were
/// already sitting on a local disk.
pub const DEFAULT_DISK_CACHE_BUDGET_GB: u64 = 16;

/// Whether the secondary [`DiskCache`] is enabled at all.
///
/// `BSDB_DISK_CACHE=off|0|false|no` and `BSDB_DISK_CACHE_GB=0` both disable it.
/// Local (`file://`) stores are always served in place regardless — see
/// [`DiskCache::new`].
fn disk_cache_enabled() -> bool {
    if let Ok(v) = std::env::var("BSDB_DISK_CACHE") {
        if matches!(
            v.trim().to_ascii_lowercase().as_str(),
            "0" | "off" | "false" | "no"
        ) {
            return false;
        }
    }
    disk_cache_budget_bytes() > 0
}

/// On-disk budget for [`DiskCache`], from `BSDB_DISK_CACHE_GB` (default 16 GiB).
pub fn disk_cache_budget_bytes() -> u64 {
    match std::env::var("BSDB_DISK_CACHE_GB") {
        Ok(v) => match v.trim().parse::<u64>() {
            Ok(gb) => gb.saturating_mul(GIB),
            Err(_) => {
                tracing::warn!(
                    value = %v,
                    "BSDB_DISK_CACHE_GB is not a number; using the default budget"
                );
                DEFAULT_DISK_CACHE_BUDGET_GB * GIB
            }
        },
        Err(_) => DEFAULT_DISK_CACHE_BUDGET_GB * GIB,
    }
}

/// Bytes currently resident in [`DISK_CACHE_DIR`], maintained on write.
static DISK_CACHE_BYTES: AtomicU64 = AtomicU64::new(0);

/// One gate per cache key so concurrent readers of the same object fetch and
/// write it once instead of once per thread.
static DISK_CACHE_INFLIGHT: Lazy<DashMap<String, Arc<tokio::sync::Semaphore>>> =
    Lazy::new(DashMap::new);

/// Subtract `budget` from `dir` by deleting its oldest entries.
///
/// `remove_temp` also drops `*.tmp` files left behind by an interrupted writer.
/// It must stay `false` during a live sweep: another task's in-progress temp
/// file is not garbage.
fn sweep_disk_cache(dir: &Path, budget: u64, remove_temp: bool) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };

    let mut files: Vec<(PathBuf, u64, Option<std::time::SystemTime>)> = Vec::new();
    let mut total = 0u64;

    for entry in entries.flatten() {
        let path = entry.path();
        let Ok(meta) = entry.metadata() else { continue };
        if !meta.is_file() {
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("tmp") {
            if remove_temp {
                let _ = std::fs::remove_file(&path);
            }
            continue;
        }
        total = total.saturating_add(meta.len());
        files.push((path, meta.len(), meta.modified().ok()));
    }

    if total <= budget {
        return total;
    }

    files.sort_by_key(|(_, _, mtime)| *mtime);
    for (path, len, _) in files {
        if total <= budget {
            break;
        }
        if std::fs::remove_file(&path).is_ok() {
            total = total.saturating_sub(len);
        }
    }
    total
}

/// Account for `bytes` just written into `dir` and evict if over budget.
fn charge_disk_cache(dir: &Path, bytes: u64) {
    static EVICTING: AtomicBool = AtomicBool::new(false);

    let budget = disk_cache_budget_bytes();
    if budget == 0 {
        return;
    }
    let total = DISK_CACHE_BYTES.fetch_add(bytes, Ordering::Relaxed) + bytes;
    if total <= budget {
        return;
    }
    if EVICTING.swap(true, Ordering::AcqRel) {
        return; // another task is already trimming
    }

    let dir = dir.to_path_buf();
    // Directory scans are blocking I/O; keep them off the async worker threads.
    tokio::task::spawn_blocking(move || {
        let remaining = sweep_disk_cache(&dir, budget, false);
        DISK_CACHE_BYTES.store(remaining, Ordering::Relaxed);
        EVICTING.store(false, Ordering::Release);
    });
}

fn inflight_gate(hash: &str) -> Arc<tokio::sync::Semaphore> {
    DISK_CACHE_INFLIGHT
        .entry(hash.to_string())
        .or_insert_with(|| Arc::new(tokio::sync::Semaphore::new(1)))
        .clone()
}

/// Drop a single-flight gate, forgetting it once no task is waiting on it.
fn release_gate(hash: &str, gate: Arc<tokio::sync::Semaphore>) {
    drop(gate);
    // The map holds one reference; anything above that means another task is
    // still waiting on the same gate.
    DISK_CACHE_INFLIGHT.remove_if(hash, |_, v| Arc::strong_count(v) == 1);
}

/// True when `cache_path` exists and matches the object's current length.
///
/// `expected_size == None` means the HEAD failed (the store does not support it
/// or the object is gone), in which case the entry is trusted as-is.
async fn cached_entry_is_valid(cache_path: &Path, expected_size: Option<u64>) -> bool {
    match tokio::fs::metadata(cache_path).await {
        Ok(meta) => expected_size.is_none() || expected_size == Some(meta.len()),
        Err(_) => false,
    }
}

/// Read a cached entry after validating it against the object's current length.
async fn read_cached_entry(cache_path: &Path, expected_size: Option<u64>) -> Option<bytes::Bytes> {
    if !cached_entry_is_valid(cache_path, expected_size).await {
        return None;
    }
    tokio::fs::read(cache_path)
        .await
        .ok()
        .map(bytes::Bytes::from)
}

/// Write `data` into the cache atomically (unique temp file, then rename).
///
/// The temp name must be unique per write attempt. It used to be keyed by
/// thread id, which is not unique under a work-stealing runtime: two tasks on
/// the same worker truncated each other's temp file.
async fn write_cache_entry(hash: &str, cache_dir: &Path, cache_path: &Path, data: &[u8]) {
    static WRITE_NONCE: AtomicU64 = AtomicU64::new(0);

    let nonce = WRITE_NONCE.fetch_add(1, Ordering::Relaxed);
    let temp_path = cache_dir.join(format!("{}.{}.{}.tmp", hash, std::process::id(), nonce));

    let written = {
        use tokio::io::AsyncWriteExt;
        match tokio::fs::File::create(&temp_path).await {
            Ok(mut f) => f.write_all(data).await.is_ok() && f.flush().await.is_ok(),
            Err(_) => false,
        }
    };

    if written {
        // rename is atomic on POSIX, so readers never observe a partial file.
        if tokio::fs::rename(&temp_path, cache_path).await.is_err() {
            let _ = tokio::fs::remove_file(&temp_path).await;
        }
    } else {
        let _ = tokio::fs::remove_file(&temp_path).await;
    }
}

/// Root directory of `store` when it is backed by the local filesystem.
///
/// `ObjectStore` is `Display` — the only store-type introspection the stable
/// trait exposes — and `LocalFileSystem` renders itself as
/// `LocalFileSystem(<root>)`.
fn local_store_root(store: &dyn ObjectStore) -> Option<PathBuf> {
    let display = store.to_string();
    let root = display
        .strip_prefix("LocalFileSystem(")
        .and_then(|rest| rest.strip_suffix(')'))?;
    // The rendered root is a URL (`file:///path/to/table`).
    Some(PathBuf::from(root.strip_prefix("file://").unwrap_or(root)))
}

pub static DISK_CACHE_DIR: Lazy<Option<PathBuf>> = Lazy::new(|| {
    if !disk_cache_enabled() {
        tracing::info!("secondary disk cache disabled by configuration");
        return None;
    }

    let dir = std::env::var("BSDB_DISK_CACHE_DIR")
        .ok()
        .filter(|p| !p.trim().is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("hdb_cache"));

    if let Err(e) = std::fs::create_dir_all(&dir) {
        tracing::warn!(
            dir = %dir.display(),
            error = %e,
            "cannot create disk cache directory; disabling disk cache"
        );
        return None;
    }

    // Bring a pre-existing directory back inside the budget and drop the temp
    // files of interrupted runs, otherwise the cache grows forever across runs.
    let budget = disk_cache_budget_bytes();
    let used = sweep_disk_cache(&dir, budget, true);
    DISK_CACHE_BYTES.store(used, Ordering::Relaxed);

    tracing::info!(
        dir = %dir.display(),
        used_mb = used / (1024 * 1024),
        budget_mb = budget / (1024 * 1024),
        "secondary disk cache ready"
    );
    Some(dir)
});

#[derive(Clone)]
pub struct DiskCache {
    store: Arc<dyn ObjectStore>,
    cache_dir: Option<PathBuf>,
    /// Root of the backing store when it is a local directory. Such objects are
    /// read/mmap'd in place instead of being copied into `cache_dir`.
    local_root: Option<PathBuf>,
}

impl DiskCache {
    /// Build a cache for `store`.
    ///
    /// A local (`file://`, `/path`, bare-directory) store is **never** cached:
    /// the object already lives on disk, so copying it into `$TMPDIR/hdb_cache`
    /// turns one read into one read plus a full write of the same bytes. That
    /// duplication is what saturated a spinning disk with 300 MB/s of writes
    /// while the source data sat on an SSD next to it.
    pub fn new(store: Arc<dyn ObjectStore>) -> Self {
        let local_root = local_store_root(store.as_ref());
        let cache_dir = if local_root.is_some() {
            None
        } else {
            DISK_CACHE_DIR.clone()
        };
        Self {
            store,
            cache_dir,
            local_root,
        }
    }

    /// Bypass the disk cache explicitly (objects are still served from `store`,
    /// and mmap'd in place when the store is local).
    pub fn disabled(store: Arc<dyn ObjectStore>) -> Self {
        let local_root = local_store_root(store.as_ref());
        Self {
            store,
            cache_dir: None,
            local_root,
        }
    }

    /// Whether objects are being copied into the on-disk cache.
    pub fn is_enabled(&self) -> bool {
        self.cache_dir.is_some()
    }

    pub async fn get_bytes(&self, path: &str) -> Result<bytes::Bytes> {
        self.get_bytes_with_key(path, path).await
    }

    /// Fetch `path` from the store, hashing the on-disk entry by `cache_key`.
    ///
    /// [`DISK_CACHE_DIR`] is shared by every store on the host, so hashing the
    /// bare object path lets unrelated stores collide whenever they happen to
    /// use the same relative layout — two temp directories, two buckets, or two
    /// CI steps reusing `/tmp`. Callers that know the store-qualified URI must
    /// pass it here so a stale graph can never be paired with a fresh row-id
    /// mapping for the same relative path.
    pub async fn get_bytes_with_key(&self, path: &str, cache_key: &str) -> Result<bytes::Bytes> {
        use sha2::{Digest, Sha256};

        // No cache dir means "read through": either the store is local (read the
        // file where it lives) or the cache was disabled on purpose.
        let Some(cache_dir) = self.cache_dir.clone() else {
            let res = self
                .store
                .get(&object_store::path::Path::from(path))
                .await
                .map_err(|e| anyhow::anyhow!(e))?;
            return res.bytes().await.map_err(|e| anyhow::anyhow!(e));
        };

        let mut hasher = Sha256::new();
        hasher.update(cache_key);
        let hash = format!("{:x}", hasher.finalize());
        let cache_path = cache_dir.join(&hash);

        let expected_size: Option<u64> = self
            .store
            .head(&object_store::path::Path::from(path))
            .await
            .map(|m| m.size)
            .ok();

        if let Some(hit) = read_cached_entry(&cache_path, expected_size).await {
            return Ok(hit);
        }

        // Every search thread asks for the same shard at the same instant.
        // Without a single-flight gate each one downloads and writes its own
        // full copy, so one 2 GB object becomes 32 concurrent 2 GB writes.
        let gate = inflight_gate(&hash);
        let permit = gate.acquire().await.ok();

        // The task in front of us may have completed the write while we waited.
        if let Some(hit) = read_cached_entry(&cache_path, expected_size).await {
            return Ok(hit);
        }

        let b = self
            .store
            .get(&object_store::path::Path::from(path))
            .await?
            .bytes()
            .await?;

        write_cache_entry(&hash, &cache_dir, &cache_path, &b).await;
        let len = b.len() as u64;
        drop(permit);
        release_gate(&hash, gate);
        charge_disk_cache(&cache_dir, len);

        Ok(b)
    }

    pub async fn get_mmap(&self, path: &str) -> Result<Arc<memmap2::Mmap>> {
        self.get_mmap_with_key(path, path).await
    }

    /// Memory maps an object straight out of a local store directory.
    ///
    /// No copy is made: the map points at the store's own file, so serving is
    /// out-of-core without ever doubling the bytes on disk.
    fn mmap_local_object(&self, path: &str) -> Result<Arc<memmap2::Mmap>> {
        let root = self.local_root.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Disk cache disabled and the store is not a local filesystem; \
                 cannot memory map {}",
                path
            )
        })?;

        let src = root.join(path.trim_start_matches('/'));
        let file = std::fs::File::open(&src)
            .map_err(|e| anyhow::anyhow!("failed to open local object {}: {e}", src.display()))?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        // Same reasoning as the cache-dir path below: consumers are random
        // access, so speculative readahead only pollutes the page cache.
        #[cfg(unix)]
        let _ = mmap.advise(memmap2::Advice::Random);
        Ok(Arc::new(mmap))
    }

    /// Mmap-backed counterpart to [`DiskCache::get_bytes_with_key`].
    pub async fn get_mmap_with_key(
        &self,
        path: &str,
        cache_key: &str,
    ) -> Result<Arc<memmap2::Mmap>> {
        use sha2::{Digest, Sha256};

        let Some(cache_dir) = self.cache_dir.clone() else {
            return self.mmap_local_object(path);
        };

        let mut hasher = Sha256::new();
        hasher.update(cache_key);
        let hash = format!("{:x}", hasher.finalize());
        let cache_path = cache_dir.join(&hash);

        let expected_size: Option<u64> = self
            .store
            .head(&object_store::path::Path::from(path))
            .await
            .map(|m| m.size)
            .ok();

        // Metadata only: this must never read a multi-GB entry just to check it.
        let needs_fetch = !cached_entry_is_valid(&cache_path, expected_size).await;

        if needs_fetch {
            let gate = inflight_gate(&hash);
            let permit = gate.acquire().await.ok();

            // A leader may have filled the entry while we waited for the gate.
            if !cached_entry_is_valid(&cache_path, expected_size).await {
                let b = self
                    .store
                    .get(&object_store::path::Path::from(path))
                    .await?
                    .bytes()
                    .await?;
                write_cache_entry(&hash, &cache_dir, &cache_path, &b).await;
                let len = b.len() as u64;
                drop(permit);
                release_gate(&hash, gate);
                charge_disk_cache(&cache_dir, len);
            } else {
                drop(permit);
                release_gate(&hash, gate);
            }
        }

        let file = std::fs::File::open(&cache_path)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        // Every consumer of get_mmap is a random-access serving structure
        // (HNSW graph links/vectors, CSR offsets/edges/dict). Default kernel
        // readahead speculatively pulls ~128 KB windows around each page
        // fault that are evicted before reuse during pointer-chasing,
        // polluting the page cache and thrashing out-of-core serving.
        // MADV_RANDOM disables readahead so only the pages actually touched
        // are faulted in — this is what makes mmap-backed (out-of-core)
        // vector/graph search cache-efficient on memory-constrained hosts.
        #[cfg(unix)]
        let _ = mmap.advise(memmap2::Advice::Random);
        Ok(Arc::new(mmap))
    }
}

// Cache Keys
// Manifest: "s3://bucket/path/_manifest" -> Manifest
// Version: "s3://bucket/path/_manifest" -> u64 (Latest Version)
// Index: "s3://bucket/path/segment_id/column" -> RoaringBitmap
//
// NOTE: We intentionally do NOT cache HNSW indexes or Parquet data.
// At large scale, caching would exhaust memory. The design relies on:
// - Streaming only needed data via object store range requests
// - Efficient indexes (roaring bitmaps, HNSW graphs) that are small
// - Client-side filtering to minimize data transfer

pub static MANIFEST_CACHE: Lazy<Cache<String, Arc<Manifest>>> = Lazy::new(|| {
    Cache::builder()
        .max_capacity(1000)
        .time_to_live(Duration::from_secs(5)) // Short TTL; commit() explicitly invalidates
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
        .time_to_live(Duration::from_secs(2)) // Short TTL; commit() explicitly invalidates
        .build()
});

pub static INDEX_CACHE: Lazy<Cache<String, Arc<RoaringBitmap>>> = Lazy::new(|| {
    Cache::builder()
        .max_capacity(10_000) // Objects - bitmaps are tiny (~KB each)
        .time_to_idle(Duration::from_secs(60 * 5)) // 5 mins idle
        .build()
});

pub static BYTE_CACHE: Lazy<Cache<String, Arc<Vec<u8>>>> = Lazy::new(|| {
    let cache_gb: u64 = std::env::var("BSDB_CACHE_GB")
        .unwrap_or_else(|_| "1".to_string())
        .parse()
        .unwrap_or(1);

    // Allocate 10% of global cache to Byte Cache (max 512MB)
    let limit_bytes = (cache_gb * 1024 * 1024 * 1024 / 10).min(512 * 1024 * 1024);
    let max_kb = limit_bytes / 1024;

    tracing::info!(
        "Initializing Byte Cache with {} MB limit",
        limit_bytes / (1024 * 1024)
    );

    Cache::builder()
        // moka's weigher is u32 and the value is in KB, so clamp to avoid a
        // wrap for a value above 4 TiB (u32::MAX KB).
        .weigher(|_key, value: &Arc<Vec<u8>>| -> u32 {
            (value.len() / 1024).min(u32::MAX as usize) as u32
        })
        .max_capacity(max_kb)
        .time_to_idle(Duration::from_secs(60 * 30)) // 30 mins
        .build()
});

/// Cache of whole small parquet files, keyed by object path.
///
/// A scan touches every segment, and a workload issues several scans per step,
/// so small segments are read repeatedly. Serving their byte ranges from memory
/// avoids the per-range object-store overhead (a `spawn_blocking` + open per
/// range). Only files at or below `PARQUET_BYTES_CACHE_MAX_FILE` are cached;
/// larger files are read through `ParquetObjectReader`, which fetches only the
/// needed column chunks.
pub static PARQUET_BYTES_CACHE: Lazy<Cache<String, bytes::Bytes>> = Lazy::new(|| {
    let cache_gb: u64 = std::env::var("BSDB_CACHE_GB")
        .unwrap_or_else(|_| "1".to_string())
        .parse()
        .unwrap_or(1);

    // Allocate 10% of the global cache to whole-file parquet bytes (max 512MB).
    let limit_bytes = (cache_gb * 1024 * 1024 * 1024 / 10).min(512 * 1024 * 1024);
    let max_kb = limit_bytes / 1024;

    Cache::builder()
        // moka's weigher is u32 and the value is in KB, so clamp to avoid a
        // wrap for a value above 4 TiB (u32::MAX KB).
        .weigher(|_key, value: &bytes::Bytes| -> u32 {
            (value.len() / 1024).min(u32::MAX as usize) as u32
        })
        .max_capacity(max_kb)
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
    // Default to 1GB cache if not set
    let cache_gb: u64 = std::env::var("BSDB_CACHE_GB")
        .unwrap_or_else(|_| "1".to_string())
        .parse()
        .unwrap_or(1);

    // Convert to KB to avoid u32 overflow in weigher (moka requirement)
    // u32::MAX KB = 4TB, which is plenty for a single item.
    let max_kb = cache_gb * 1024 * 1024;

    tracing::info!("Initializing HNSW-IVF Cache with {} GB limit", cache_gb);

    Cache::builder()
        .weigher(|_key, value: &Arc<HnswIvfIndex>| -> u32 {
            (value.size_in_bytes() / 1024).min(u32::MAX as usize) as u32
        })
        .max_capacity(max_kb)
        .time_to_idle(Duration::from_secs(60 * 15)) // 15 mins idle
        .build()
});

/// Cache for CSR graph indexes loaded from Puffin compound bundles.
///
/// The loose-sidecar path mmap'd the three CSR files directly; the Puffin path
/// reads the blobs into owned bytes, so caching the decoded graph avoids
/// re-reading and re-copying the bundle on every `load_graph_index` call.
/// Budgeted from `BSDB_CACHE_GB` (25% of the global cache, like the inverted
/// index cache).
pub static CSR_GRAPH_CACHE: Lazy<Cache<String, Arc<MmapCsrGraph>>> = Lazy::new(|| {
    let cache_gb: u64 = std::env::var("BSDB_CACHE_GB")
        .unwrap_or_else(|_| "1".to_string())
        .parse()
        .unwrap_or(1);

    // Allocate 25% of the global cache to CSR graphs.
    let limit_bytes = cache_gb * 1024 * 1024 * 1024 / 4;
    let max_kb = limit_bytes / 1024;

    tracing::info!(
        "Initializing CSR Graph Cache with {} MB limit",
        limit_bytes / (1024 * 1024)
    );

    Cache::builder()
        .weigher(|_key, value: &Arc<MmapCsrGraph>| -> u32 {
            (value.size_in_bytes() / 1024).min(u32::MAX as usize) as u32
        })
        .max_capacity(max_kb)
        .time_to_idle(Duration::from_secs(60 * 15))
        .build()
});

pub static INVERTED_INDEX_CACHE: Lazy<Cache<String, Arc<Vec<RecordBatch>>>> = Lazy::new(|| {
    let cache_gb: u64 = std::env::var("BSDB_CACHE_GB")
        .unwrap_or_else(|_| "1".to_string())
        .parse()
        .unwrap_or(1);

    // Allocate 25% of global cache to Inverted Index Cache
    let limit_bytes = cache_gb * 1024 * 1024 * 1024 / 4;
    let max_kb = limit_bytes / 1024;

    tracing::info!(
        "Initializing Inverted Index Cache with {} MB limit",
        limit_bytes / (1024 * 1024)
    );

    Cache::builder()
        .weigher(|_key, value: &Arc<Vec<RecordBatch>>| -> u32 {
            let bytes: usize = value.iter().map(|b| b.get_array_memory_size()).sum();
            (bytes / 1024).min(u32::MAX as usize) as u32
        })
        .max_capacity(max_kb)
        .time_to_idle(Duration::from_secs(60 * 15))
        .build()
});

/// Analyzer-name cache keyed by inverted-index file path. Readers pull the
/// analyzer name from the inv file's key/value metadata once and cache it so
/// repeated queries re-tokenize identically without re-reading the footer.
pub static ANALYZER_META_CACHE: Lazy<Cache<String, String>> = Lazy::new(|| {
    Cache::builder()
        .max_capacity(10_000)
        .time_to_idle(Duration::from_secs(60 * 15))
        .build()
});

/// Parsed parquet metadata, keyed by object path, paired with the file size.
///
/// Stores the `ArrowReaderMetadata` (not the raw `ParquetMetaData`) so a read
/// can skip re-deriving the Arrow schema from the parquet schema on every
/// access — the read path's `meta` phase.
pub static PARQUET_META_CACHE: Lazy<Cache<String, (Arc<ArrowReaderMetadata>, usize)>> =
    Lazy::new(|| {
        Cache::builder()
            .max_capacity(1000) // 1000 file footers (schema, row groups)
            .time_to_idle(Duration::from_secs(60 * 30))
            .build()
    });

pub static BLOOM_FILTER_CACHE: Lazy<Cache<String, Arc<Sbbf>>> = Lazy::new(|| {
    Cache::builder()
        .max_capacity(2048) // Roughly 250MB if each is 128KB
        .time_to_idle(Duration::from_secs(60 * 30))
        .build()
});

/// Global cache for parsed Iceberg position-delete files.
///
/// Delete files are immutable once written, so their content can be cached
/// indefinitely (within TTI bounds). The key is `"{delete_file_path}::{target_data_file_path}"`
/// because the same physical delete file may contain positions for multiple data
/// files and we filter to the target during parsing.
pub static POSITION_DELETE_CACHE: Lazy<Cache<String, Arc<std::collections::HashSet<i64>>>> =
    Lazy::new(|| {
        Cache::builder()
            .max_capacity(50_000)
            .time_to_idle(Duration::from_secs(60 * 10)) // 10 mins idle
            .build()
    });

pub static FULL_DELETE_FILE_CACHE: Lazy<
    Cache<String, Arc<std::collections::HashMap<String, std::collections::HashSet<i64>>>>,
> = Lazy::new(|| {
    Cache::builder()
        .max_capacity(50_000)
        .time_to_idle(Duration::from_secs(60 * 10)) // 10 mins idle
        .build()
});

/// Global cache for parsed equality-delete values, keyed by the segment's
/// delete-file set (same key as the merged position-delete cache).
///
/// `load_equality_deletes` is called on every segment read; without this cache
/// it re-reads and re-parses every equality delete file each time.
pub static EQUALITY_DELETE_CACHE: Lazy<
    Cache<String, Arc<Vec<crate::core::reader::EqualityDelete>>>,
> = Lazy::new(|| {
    Cache::builder()
        .max_capacity(50_000)
        .time_to_idle(Duration::from_secs(60 * 10)) // 10 mins idle
        .build()
});

/// Doris-inspired Block Cache for decoded RecordBatches.
/// Bypasses Parquet decoding/decompression for frequently accessed blocks.
pub static BLOCK_CACHE: Lazy<Cache<String, Arc<RecordBatch>>> = Lazy::new(|| {
    // Default to 1GB cache if not set
    let cache_gb: u64 = std::env::var("BSDB_BLOCK_CACHE_GB")
        .unwrap_or_else(|_| "1".to_string())
        .parse()
        .unwrap_or(1);

    let max_kb = cache_gb * 1024 * 1024;

    tracing::info!("Initializing Block Cache with {} GB limit", cache_gb);

    Cache::builder()
        // moka's weigher is u32 and the value is in KB; clamp so a batch larger
        // than 4 TiB cannot wrap the weight to a tiny value.
        .weigher(|_key, value: &Arc<RecordBatch>| -> u32 {
            (value.get_array_memory_size() / 1024).min(u32::MAX as usize) as u32
        })
        .max_capacity(max_kb)
        .time_to_idle(Duration::from_secs(60 * 15))
        .build()
});

/// Invalidate all memory caches keyed by an object/file path or (root_uri + path).
pub async fn invalidate_file_caches(root_uri: &str, path: &str) {
    let key1 = path.to_string();
    let key2 = format!("{}/{}", root_uri, path);
    let key3 = if let Some(stripped) = path.strip_prefix('/') {
        format!("{}/{}", root_uri, stripped)
    } else {
        path.to_string()
    };
    let clean_path = path.trim_start_matches('/');
    let key4 = format!("{}/{}", root_uri.trim_end_matches('/'), clean_path);

    for k in &[&key1, &key2, &key3, &key4] {
        BLOCK_CACHE.invalidate(*k).await;
        PARQUET_BYTES_CACHE.invalidate(*k).await;
        PARQUET_META_CACHE.invalidate(*k).await;
        BYTE_CACHE.invalidate(*k).await;
        INDEX_CACHE.invalidate(*k).await;
        BLOOM_FILTER_CACHE.invalidate(*k).await;
        HNSW_CACHE.invalidate(*k).await;
        HNSW_IVF_CACHE.invalidate(*k).await;
        INVERTED_INDEX_CACHE.invalidate(*k).await;
        ANALYZER_META_CACHE.invalidate(*k).await;
        POSITION_DELETE_CACHE.invalidate(*k).await;
        FULL_DELETE_FILE_CACHE.invalidate(*k).await;
        EQUALITY_DELETE_CACHE.invalidate(*k).await;
    }
}

/// Invalidate the index and metadata caches for a single table, leaving other
/// tables' entries resident.
///
/// `rebuild_index` used to call `invalidate_all()` on every cache, which evicted
/// every other table's indexes in the same process. That is both a performance
/// cliff (every concurrent table reloads from storage) and a correctness hazard
/// (a reader can observe a cache miss mid-search and fall back to a stale
/// artifact). Cache keys are prefixed with the table URI, so scope the
/// invalidation to that prefix.
pub async fn invalidate_table_caches(uri: &str) {
    let uri = uri.to_string();
    // A table's own manager uses the normalized `file://` URI, but callers that
    // construct a `ManifestManager` from the raw path (tests, CLI helpers) key
    // the same caches without the scheme. Match both so neither view goes stale.
    let bare = uri.strip_prefix("file://").unwrap_or(&uri).to_string();
    macro_rules! invalidate_scoped {
        ($cache:expr) => {{
            let u = uri.clone();
            let b = bare.clone();
            let _ =
                $cache.invalidate_entries_if(move |k: &String, _| k.contains(&u) || k.contains(&b));
            // `invalidate_entries_if` is applied lazily; flush it so a caller
            // that reads the index immediately after a rebuild sees the new
            // bytes rather than the cached ones.
            $cache.run_pending_tasks().await;
        }};
    }
    // The expensive artifacts (graphs, vectors, inverted lists) are scoped to
    // this table so concurrent tables keep their resident indexes.
    invalidate_scoped!(HNSW_IVF_CACHE);
    invalidate_scoped!(HNSW_CACHE);
    invalidate_scoped!(INDEX_CACHE);
    invalidate_scoped!(CSR_GRAPH_CACHE);
    invalidate_scoped!(INVERTED_INDEX_CACHE);
    // The manifest caches are tiny (bounded entry counts, short TTLs) and are
    // keyed by both the normalized and raw table path depending on the caller.
    // Clearing them globally is cheap and avoids a stale-version read.
    MANIFEST_CACHE.invalidate_all();
    MANIFEST_LIST_CACHE.invalidate_all();
    LATEST_VERSION_CACHE.invalidate_all();
}

/// Invalidate all index and metadata caches across all tables without a rebuild.
pub fn invalidate_index_caches() {
    HNSW_IVF_CACHE.invalidate_all();
    HNSW_CACHE.invalidate_all();
    INDEX_CACHE.invalidate_all();
    CSR_GRAPH_CACHE.invalidate_all();
    INVERTED_INDEX_CACHE.invalidate_all();
    MANIFEST_CACHE.invalidate_all();
    MANIFEST_LIST_CACHE.invalidate_all();
    LATEST_VERSION_CACHE.invalidate_all();
}

/// Invalidate all in-memory caches including block, parquet, delete and metadata caches.
pub fn invalidate_all_caches() {
    invalidate_index_caches();
    BYTE_CACHE.invalidate_all();
    PARQUET_BYTES_CACHE.invalidate_all();
    PARQUET_META_CACHE.invalidate_all();
    BLOCK_CACHE.invalidate_all();
    BLOOM_FILTER_CACHE.invalidate_all();
    POSITION_DELETE_CACHE.invalidate_all();
    FULL_DELETE_FILE_CACHE.invalidate_all();
    EQUALITY_DELETE_CACHE.invalidate_all();
    ANALYZER_META_CACHE.invalidate_all();
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
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

        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();

        let batches = Arc::new(vec![batch]);

        // Miss
        assert!(INVERTED_INDEX_CACHE.get(&key).await.is_none());

        // Insert
        INVERTED_INDEX_CACHE
            .insert(key.clone(), batches.clone())
            .await;

        // Hit
        let cached = INVERTED_INDEX_CACHE.get(&key).await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().len(), 1);

        // Cleanup
        INVERTED_INDEX_CACHE.invalidate(&key).await;
    }

    #[tokio::test]
    async fn test_latest_version_cache_hit_miss() {
        // Use a local cache instance for this test because the global LATEST_VERSION_CACHE
        // has a TTL of 0 which breaks hit/miss assertions.
        let test_cache: Cache<String, u64> = Cache::builder()
            .max_capacity(10)
            .time_to_live(StdDuration::from_secs(60))
            .build();

        let key = "test_version_cache_key_1".to_string();
        let version = 42u64;

        // Miss
        assert!(test_cache.get(&key).await.is_none());

        // Insert
        test_cache.insert(key.clone(), version).await;

        // Hit
        let cached = test_cache.get(&key).await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap(), version);

        // Cleanup
        test_cache.invalidate(&key).await;
    }

    #[tokio::test]
    async fn test_cache_eviction_lru() {
        // Create a small cache for testing eviction
        let test_cache: Cache<String, Arc<Vec<u8>>> = Cache::builder().max_capacity(3).build();

        // Insert 3 items (at capacity)
        test_cache
            .insert("key1".to_string(), Arc::new(vec![1]))
            .await;
        test_cache
            .insert("key2".to_string(), Arc::new(vec![2]))
            .await;
        test_cache
            .insert("key3".to_string(), Arc::new(vec![3]))
            .await;

        // Give cache time to process
        sleep(StdDuration::from_millis(200)).await;
        test_cache.run_pending_tasks().await;

        // All 3 should be present
        assert!(test_cache.get("key1").await.is_some());
        assert!(test_cache.get("key2").await.is_some());
        assert!(test_cache.get("key3").await.is_some());

        // Insert 4th and 5th items, should trigger evictions
        test_cache
            .insert("key4".to_string(), Arc::new(vec![4]))
            .await;
        test_cache
            .insert("key5".to_string(), Arc::new(vec![5]))
            .await;

        // Give significant time for eviction to process
        sleep(StdDuration::from_millis(500)).await;
        test_cache.run_pending_tasks().await;
        sleep(StdDuration::from_millis(200)).await;

        // Total entries should not exceed capacity
        let entry_count = test_cache.entry_count();
        assert!(
            entry_count <= 3,
            "Cache exceeded capacity: {} > 3",
            entry_count
        );
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

    /// A local store must be served in place. Copying a local file into the
    /// cache is pure duplication: the bytes already are on disk, so the copy
    /// doubles both the write volume and the space used by the index.
    #[tokio::test]
    async fn test_local_store_bypasses_disk_cache() {
        use object_store::local::LocalFileSystem;
        use object_store::path::Path;
        use object_store::ObjectStore;

        let dir = tempfile::tempdir().unwrap();
        let rel = "seg.embedding.cluster_0.hnsw.graph";
        std::fs::write(dir.path().join(rel), b"graph-bytes").unwrap();

        let store: Arc<dyn ObjectStore> =
            Arc::new(LocalFileSystem::new_with_prefix(dir.path()).unwrap());
        let cache = DiskCache::new(store);

        assert!(
            !cache.is_enabled(),
            "a file:// store is already on disk; it must not be copied into the cache"
        );

        let object = Path::from(rel).to_string();
        assert_eq!(&cache.get_bytes(&object).await.unwrap()[..], b"graph-bytes");
        assert_eq!(&cache.get_mmap(&object).await.unwrap()[..], b"graph-bytes");
    }

    /// The disk cache must stay inside its budget and drop the temp files of
    /// interrupted writers. Unbounded growth is what produced 409 GB of copies.
    #[test]
    fn test_disk_cache_sweep_enforces_budget() {
        let dir = tempfile::tempdir().unwrap();

        for (name, age_secs) in [("old", 300u64), ("mid", 200), ("new", 100)] {
            let path = dir.path().join(name);
            std::fs::write(&path, vec![0u8; 4096]).unwrap();
            let mtime = std::time::SystemTime::now()
                .checked_sub(std::time::Duration::from_secs(age_secs))
                .unwrap();
            let f = std::fs::File::options().write(true).open(&path).unwrap();
            f.set_modified(mtime).unwrap();
        }
        std::fs::write(dir.path().join("deadbeef.1.2.tmp"), vec![0u8; 4096]).unwrap();

        let used = sweep_disk_cache(dir.path(), 8192, true);

        assert_eq!(
            used, 8192,
            "sweep must trim the directory down to the budget"
        );
        assert!(
            !dir.path().join("old").exists(),
            "the oldest entry is evicted first"
        );
        assert!(
            !dir.path().join("deadbeef.1.2.tmp").exists(),
            "stale temp files from interrupted writers are removed"
        );
        assert!(dir.path().join("new").exists());
    }

    /// The host-global disk cache must not serve bytes from another store just
    /// because both use the same relative object path and byte length.
    #[tokio::test]
    async fn test_disk_cache_namespaces_by_qualified_key() {
        use object_store::memory::InMemory;
        use object_store::path::Path;
        use object_store::ObjectStore;

        let store_a = Arc::new(InMemory::new());
        let store_b = Arc::new(InMemory::new());
        let object = Path::from("seg.embedding.cluster_0.hnsw.graph");

        store_a
            .put(&object, bytes::Bytes::from_static(b"graph-a").into())
            .await
            .unwrap();
        store_b
            .put(&object, bytes::Bytes::from_static(b"graph-b").into())
            .await
            .unwrap();

        // Same relative path and same length on purpose.
        assert_eq!(b"graph-a".len(), b"graph-b".len());

        let nonce = std::process::id();
        let a = DiskCache::new(store_a)
            .get_bytes_with_key(
                "seg.embedding.cluster_0.hnsw.graph",
                &format!("store-a/{nonce}/seg.embedding.cluster_0.hnsw.graph"),
            )
            .await
            .unwrap();
        let b = DiskCache::new(store_b)
            .get_bytes_with_key(
                "seg.embedding.cluster_0.hnsw.graph",
                &format!("store-b/{nonce}/seg.embedding.cluster_0.hnsw.graph"),
            )
            .await
            .unwrap();

        assert_eq!(&a[..], b"graph-a");
        assert_eq!(&b[..], b"graph-b");
    }
}
