// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Index preloading — warm the read-path caches at startup.
//!
//! Dgraph keeps its posting-list indexes resident in RAM; OpenSearch/Elasticsearch
//! keep their Lucene segments warm. BenoStreamDB's indexes are Iceberg data files
//! served through bounded moka caches, so the first query after a cold start pays
//! the object-store fetch + decode cost. `preload_indexes_async` pays that cost
//! up front, at table-open time, so the first user query is instant.
//!
//! ## Memory budget and disk spillover
//!
//! The in-memory caches are bounded (`BENOSTREAM_CACHE_GB`). Preload respects an
//! explicit `max_memory_bytes` budget: indexes are warmed into RAM until the
//! budget is exhausted, then the remainder is warmed into the **disk cache**
//! (`BENOSTREAM_DISK_CACHE_DIR`, mmap-backed with `MADV_RANDOM`) so it is served
//! out-of-core instead of re-fetched from the object store. This is the same
//! two-tier design the read path already uses for HNSW/CSR serving.
//!
//! ## LRU
//!
//! The moka caches evict by capacity (TinyLFU/LRU) and time-to-idle, so warming
//! more than fits simply evicts the least-recently-used entries — the most
//! recently warmed indexes stay resident. Preload warms in manifest order, so
//! the last-warmed (most recently committed) segments win the budget.

use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;

use super::Table;
use crate::core::cache::{DiskCache, INVERTED_INDEX_CACHE};
use crate::core::manifest::{IndexFile, ManifestManager};

/// Options controlling an index preload pass.
#[derive(Debug, Clone)]
pub struct PreloadOptions {
    /// Maximum bytes to warm into the in-memory caches. Indexes beyond this
    /// budget are warmed into the disk (mmap) cache instead when
    /// `spill_to_disk` is set.
    pub max_memory_bytes: u64,
    /// Warm vector (HNSW / HNSW-IVF / TQ) indexes.
    pub include_vector: bool,
    /// Warm inverted (BM25 / exact) indexes.
    pub include_inverted: bool,
    /// Warm CSR graph indexes (offsets/edges/dict).
    pub include_graph: bool,
    /// Populate the disk (mmap) cache for indexes that exceed the memory budget.
    pub spill_to_disk: bool,
}

impl Default for PreloadOptions {
    fn default() -> Self {
        Self {
            // 4 GiB default in-memory budget; the rest spills to disk.
            max_memory_bytes: 4 * 1024 * 1024 * 1024,
            include_vector: true,
            include_inverted: true,
            include_graph: true,
            spill_to_disk: true,
        }
    }
}

impl PreloadOptions {
    /// Budget derived from `BENOSTREAM_CACHE_GB` (default 1 GiB), leaving the
    /// disk cache to absorb the overflow.
    pub fn from_env() -> Self {
        let cache_gb: u64 = std::env::var("BENOSTREAM_CACHE_GB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        Self {
            max_memory_bytes: cache_gb * 1024 * 1024 * 1024,
            ..Self::default()
        }
    }
}

/// Result of a preload pass.
#[derive(Debug, Clone, Default)]
pub struct PreloadStats {
    /// Index files discovered in the manifest.
    pub indexes_seen: usize,
    /// Index files successfully warmed (memory or disk).
    pub indexes_warmed: usize,
    /// Index files skipped (type filtered out, or load error).
    pub indexes_skipped: usize,
    /// Bytes warmed into the in-memory caches.
    pub bytes_in_memory: u64,
    /// Bytes warmed into the disk (mmap) cache.
    pub bytes_on_disk: u64,
    /// Wall-clock time for the pass.
    pub elapsed_ms: u128,
}

impl Table {
    /// Warm the read-path index caches for this table.
    ///
    /// Enumerates every index file in the current manifest and loads it through
    /// the same cache path a query would use, so the first query is served from
    /// memory (or the mmap disk cache) instead of the object store.
    ///
    /// Safe to call repeatedly; already-cached indexes are cache hits and cost
    /// only a lookup. Intended to be called once at server startup by the
    /// `benostream-search` gateway.
    pub async fn preload_indexes_async(&self, opts: PreloadOptions) -> Result<PreloadStats> {
        let t0 = Instant::now();
        let mut stats = PreloadStats::default();

        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, entries, _version) = manager.load_latest_full().await?;

        // Collect (segment_dir, index_file) pairs. The index file path may be
        // relative to the segment's data-file directory or already absolute
        // (relative to the table root); resolve both the same way the reader does.
        let mut work: Vec<(String, IndexFile)> = Vec::new();
        for entry in &entries {
            let seg_dir = entry
                .file_path
                .rsplit_once('/')
                .map(|(dir, _)| dir.to_string())
                .unwrap_or_default();
            for idx in &entry.index_files {
                work.push((seg_dir.clone(), idx.clone()));
            }
        }
        stats.indexes_seen = work.len();

        let disk_cache = DiskCache::new(self.store.clone());

        for (seg_dir, idx) in work {
            let kind = classify(&idx);
            let wanted = match kind {
                IndexKind::Vector => opts.include_vector,
                IndexKind::Inverted => opts.include_inverted,
                IndexKind::Graph => opts.include_graph,
                IndexKind::Other => false,
            };
            if !wanted {
                stats.indexes_skipped += 1;
                continue;
            }

            // Resolve the full path (relative to the table root).
            let full_path = if idx.file_path.contains('/') || seg_dir.is_empty() {
                idx.file_path.clone()
            } else {
                format!("{}/{}", seg_dir, idx.file_path)
            };

            // Size the index so we can decide memory vs disk.
            let size = self.index_size_bytes(&full_path, &idx).await;

            let fits_memory = stats.bytes_in_memory.saturating_add(size) <= opts.max_memory_bytes;

            let warmed = if fits_memory {
                match self.warm_in_memory(&full_path, &idx, kind).await {
                    Ok(()) => {
                        stats.bytes_in_memory = stats.bytes_in_memory.saturating_add(size);
                        true
                    }
                    Err(e) => {
                        tracing::debug!(
                            path = %full_path,
                            error = %e,
                            "preload: in-memory warm failed"
                        );
                        false
                    }
                }
            } else if opts.spill_to_disk {
                match self.warm_on_disk(&disk_cache, &full_path, kind).await {
                    Ok(()) => {
                        stats.bytes_on_disk = stats.bytes_on_disk.saturating_add(size);
                        true
                    }
                    Err(e) => {
                        tracing::debug!(
                            path = %full_path,
                            error = %e,
                            "preload: disk warm failed"
                        );
                        false
                    }
                }
            } else {
                false
            };

            if warmed {
                stats.indexes_warmed += 1;
            } else {
                stats.indexes_skipped += 1;
            }
        }

        stats.elapsed_ms = t0.elapsed().as_millis();
        tracing::info!(
            seen = stats.indexes_seen,
            warmed = stats.indexes_warmed,
            skipped = stats.indexes_skipped,
            mem_mb = stats.bytes_in_memory / (1024 * 1024),
            disk_mb = stats.bytes_on_disk / (1024 * 1024),
            elapsed_ms = stats.elapsed_ms,
            "preload: index caches warmed"
        );
        Ok(stats)
    }

    /// Best-effort size of an index file (or its multi-file family).
    async fn index_size_bytes(&self, full_path: &str, idx: &IndexFile) -> u64 {
        // Puffin blob: the manifest records the exact length.
        if let Some(len) = idx.length {
            if len > 0 {
                return len as u64;
            }
        }
        // Single-file index: HEAD the object.
        if let Ok(meta) = self
            .store
            .head(&object_store::path::Path::from(full_path))
            .await
        {
            return meta.size;
        }
        // Multi-file HNSW family: sum the known siblings.
        let mut total = 0u64;
        for suffix in [".centroids.parquet", ".hnsw.graph", ".hnsw.data"] {
            let p = format!("{}{}", full_path, suffix);
            if let Ok(meta) = self
                .store
                .head(&object_store::path::Path::from(p.as_str()))
                .await
            {
                total += meta.size;
            }
        }
        total
    }

    /// Warm an index into the in-memory caches (the query fast path).
    async fn warm_in_memory(
        &self,
        full_path: &str,
        idx: &IndexFile,
        kind: IndexKind,
    ) -> Result<()> {
        let cache_key = cache_key_for(&self.uri, full_path, idx);
        match kind {
            IndexKind::Vector => {
                // Populates HNSW_IVF_CACHE (the reader's vector fast path).
                let _ = crate::core::index::hnsw_ivf::HnswIvfIndex::load_async_with_cache_key(
                    self.store.clone(),
                    full_path,
                    &cache_key,
                    false,
                )
                .await?;
                Ok(())
            }
            IndexKind::Inverted => {
                if INVERTED_INDEX_CACHE.get(&cache_key).await.is_some() {
                    return Ok(());
                }
                let batches = self.read_inverted_batches(full_path, idx).await?;
                INVERTED_INDEX_CACHE
                    .insert(cache_key, Arc::new(batches))
                    .await;
                Ok(())
            }
            IndexKind::Graph => {
                // CSR families are mmap-served; warming the disk cache is the
                // in-memory-equivalent (page cache) warm for them.
                let disk = DiskCache::new(self.store.clone());
                self.warm_on_disk(&disk, full_path, kind).await
            }
            IndexKind::Other => Ok(()),
        }
    }

    /// Warm an index into the disk (mmap) cache for out-of-core serving.
    async fn warm_on_disk(&self, disk: &DiskCache, full_path: &str, kind: IndexKind) -> Result<()> {
        match kind {
            IndexKind::Vector => {
                // Load with mmap so the HNSW graph is served from the disk cache.
                let cache_key = format!("{}/{}", self.uri, full_path);
                let _ = crate::core::index::hnsw_ivf::HnswIvfIndex::load_async_with_cache_key(
                    self.store.clone(),
                    full_path,
                    &cache_key,
                    true,
                )
                .await?;
                Ok(())
            }
            IndexKind::Inverted => {
                // Pull the file into the disk cache (mmap-backed).
                let _ = disk.get_mmap(full_path).await?;
                Ok(())
            }
            IndexKind::Graph => {
                // CSR is a 3-file family; warm each sibling.
                for suffix in [
                    ".graph_v2.csr.offsets",
                    ".graph_v2.csr.edges",
                    ".graph_v2.csr.dict",
                ] {
                    let p = format!("{}{}", full_path, suffix);
                    let _ = disk.get_mmap(&p).await;
                }
                Ok(())
            }
            IndexKind::Other => Ok(()),
        }
    }

    /// Read and decode an inverted-index parquet file into RecordBatches.
    async fn read_inverted_batches(
        &self,
        full_path: &str,
        idx: &IndexFile,
    ) -> Result<Vec<arrow::record_batch::RecordBatch>> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let bytes = if let (Some(offset), Some(length)) = (idx.offset, idx.length) {
            self.store
                .get_range(
                    &object_store::path::Path::from(full_path),
                    (offset as u64)..(offset as u64 + length as u64),
                )
                .await?
        } else {
            self.store
                .get(&object_store::path::Path::from(full_path))
                .await?
                .bytes()
                .await?
        };

        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
        let reader = builder.build()?;
        let mut batches = Vec::new();
        for b in reader {
            batches.push(b?);
        }
        Ok(batches)
    }
}

/// The kind of index, derived from the manifest's `index_type` / `blob_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexKind {
    Vector,
    Inverted,
    Graph,
    Other,
}

fn classify(idx: &IndexFile) -> IndexKind {
    let blob = idx.blob_type.as_deref().unwrap_or("");
    let itype = idx.index_type.as_str();

    if blob.starts_with("hnsw") || blob.starts_with("tq") || itype == "vector" {
        return IndexKind::Vector;
    }
    if blob == "csr_graph" || itype == "graph" {
        return IndexKind::Graph;
    }
    if itype == "inverted" || itype == "bm25" {
        return IndexKind::Inverted;
    }
    IndexKind::Other
}

fn cache_key_for(root_uri: &str, full_path: &str, idx: &IndexFile) -> String {
    if let Some(offset) = idx.offset {
        format!("{}/{}:{}", root_uri, full_path, offset)
    } else {
        format!("{}/{}", root_uri, full_path)
    }
}
