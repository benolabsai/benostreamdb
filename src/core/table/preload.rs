// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

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
//! The in-memory caches are bounded (`BSDB_CACHE_GB`). Preload respects an
//! explicit `max_memory_bytes` budget: indexes are warmed into RAM until the
//! budget is exhausted, then the remainder is warmed into the **disk cache**
//! (`BSDB_DISK_CACHE_DIR`, mmap-backed with `MADV_RANDOM`) so it is served
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
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
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
    /// Budget derived from `BSDB_CACHE_GB` (default 1 GiB), leaving the
    /// disk cache to absorb the overflow.
    pub fn from_env() -> Self {
        let cache_gb: u64 = std::env::var("BSDB_CACHE_GB")
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
    /// First warm failure, so a caller can tell an empty index from a broken one
    /// without a tracing subscriber attached (the Python module installs none).
    pub last_error: Option<String>,
}

/// Result of a table warmup pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WarmupStats {
    /// True if the table has vector or graph indices.
    pub has_vector_indices: bool,
    /// Number of synthetic probes executed.
    pub probes_run: usize,
    /// Number of unique candidate clusters warmed.
    pub clusters_warmed: usize,
    /// Wall-clock time in milliseconds.
    pub duration_ms: u128,
}

impl Table {
    fn cache_root(&self) -> &str {
        self.uri.trim_end_matches('/')
    }

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

        // Collect (rel_entry_path, index_file) pairs. The index file path may be
        // relative to the segment's data-file directory or already relative
        // to the table root; resolve both the same way the reader does.
        let mut work: Vec<(String, IndexFile)> = Vec::new();
        for entry in &entries {
            let rel_entry = crate::core::segment::relativize_against(&self.uri, &entry.file_path);
            for idx in &entry.index_files {
                work.push((rel_entry.clone(), idx.clone()));
            }
        }
        stats.indexes_seen = work.len();

        let disk_cache = DiskCache::new(self.store.clone());
        let cache_root = self.cache_root();

        // A Puffin container carries one vector index split across many blobs
        // (centroids + a graph and a mapping per IVF cluster), and loading any
        // one of them loads the whole index. Warming each blob therefore warmed
        // the same index hundreds of times — materializing the entire bundle on
        // each pass. Warm each (container, column, algorithm) exactly once.
        let mut warmed_vector_indexes: HashSet<(String, String, String)> = HashSet::new();

        let mut router_records: std::collections::HashMap<
            String,
            (
                usize,
                crate::core::index::VectorMetric,
                Vec<crate::core::index::vector_router::CentroidRecord>,
            ),
        > = std::collections::HashMap::new();

        for (rel_entry, idx) in work {
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
            let full_path =
                crate::core::segment::resolve_entry_index_path(&rel_entry, &idx.file_path);

            if kind == IndexKind::Vector && idx.blob_offset.is_some() {
                let family = (
                    full_path.clone(),
                    idx.column_name.clone(),
                    idx.algorithm.clone(),
                );
                if !warmed_vector_indexes.insert(family) {
                    stats.indexes_skipped += 1;
                    continue;
                }
            }

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
                        tracing::warn!(
                            path = %full_path,
                            error = %e,
                            "preload: in-memory warm failed"
                        );
                        if stats.last_error.is_none() {
                            stats.last_error = Some(format!("{full_path}: {e}"));
                        }
                        false
                    }
                }
            } else if opts.spill_to_disk {
                match self.warm_on_disk(&disk_cache, &full_path, &idx, kind).await {
                    Ok(()) => {
                        stats.bytes_on_disk = stats.bytes_on_disk.saturating_add(size);
                        true
                    }
                    Err(e) => {
                        tracing::warn!(
                            path = %full_path,
                            error = %e,
                            "preload: disk warm failed"
                        );
                        if stats.last_error.is_none() {
                            stats.last_error = Some(format!("{full_path} (disk): {e}"));
                        }
                        false
                    }
                }
            } else {
                false
            };

            if warmed {
                stats.indexes_warmed += 1;
                if kind == IndexKind::Vector {
                    let cache_key = if idx.blob_offset.is_some() {
                        crate::core::index::hnsw_ivf::HnswIvfIndex::puffin_vector_cache_key(
                            cache_root,
                            &full_path,
                            &idx.column_name,
                            &idx.algorithm,
                        )
                    } else {
                        cache_key_for(cache_root, &full_path, &idx)
                    };
                    if let Some(cached) = crate::core::cache::HNSW_IVF_CACHE.get(&cache_key).await {
                        let seg_id =
                            crate::core::segment::segment_id_from_path(&rel_entry).to_string();

                        let entry = router_records
                            .entry(idx.column_name.clone())
                            .or_insert_with(|| (cached.dim(), cached.metric(), Vec::new()));

                        for (c_id, centroid) in cached.centroids().iter().enumerate() {
                            entry
                                .2
                                .push(crate::core::index::vector_router::CentroidRecord {
                                    segment_id: seg_id.clone(),
                                    cluster_id: c_id,
                                    centroid: centroid.clone(),
                                });
                        }
                    }
                }
            } else {
                stats.indexes_skipped += 1;
            }
        }

        // Populate table-level vector routers from loaded segment centroids
        for (col, (dim, metric, records)) in router_records {
            if !records.is_empty() {
                let router = Arc::new(crate::core::index::TableVectorRouter::new(
                    dim, metric, records,
                ));
                tracing::info!(
                    column = %col,
                    centroids = router.len(),
                    "preload: initialized table-level vector router"
                );
                self.indexing.vector_routers.write().insert(col, router);
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

    /// Check whether the current table contains any vector or graph index files.
    /// If false, the table is a generic tabular/relational Iceberg table and all vector
    /// warming/allocation is bypassed.
    pub async fn has_vector_or_graph_indices(&self) -> Result<bool> {
        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, entries, _version) = manager.load_latest_full().await?;
        Ok(entries.iter().any(|e| {
            e.index_files
                .iter()
                .any(|idx| idx.index_category == "vector" || idx.index_category == "graph")
        }))
    }

    /// 3-Tier Warmup:
    /// Check if the table has vector/graph indices. If none exist (generic Iceberg table),
    /// this is a zero-cost no-op returning immediately.
    /// If vector indices exist, uses the index's own centroids to execute untimed synthetic
    /// warmup probes, pre-faulting the most critical Voronoi clusters and entrypoints into RAM.
    pub async fn warmup_from_centroids_async(
        &self,
        n_probes: usize,
        _max_memory_bytes: Option<u64>,
    ) -> Result<WarmupStats> {
        let t0 = Instant::now();
        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, entries, _version) = match manager.load_latest_full().await {
            Ok(v) => v,
            Err(_) => {
                return Ok(WarmupStats {
                    has_vector_indices: false,
                    probes_run: 0,
                    clusters_warmed: 0,
                    duration_ms: t0.elapsed().as_millis(),
                })
            }
        };

        // 1. Conditionality Check: Generic Iceberg table gatekeeper
        let mut vector_cols_set = std::collections::HashSet::new();
        for e in &entries {
            for idx in &e.index_files {
                if idx.index_category == "vector" || idx.index_category == "graph" {
                    vector_cols_set.insert(idx.column_name.clone());
                }
            }
        }

        if vector_cols_set.is_empty() {
            return Ok(WarmupStats {
                has_vector_indices: false,
                probes_run: 0,
                clusters_warmed: 0,
                duration_ms: t0.elapsed().as_millis(),
            });
        }

        let mut total_probes = 0;
        let mut total_clusters = 0;

        for col in vector_cols_set {
            // Tier 1: Ensure router is built and resident in memory (< 1 ms)
            if let Ok(Some(router)) = self.ensure_vector_router(&col).await {
                let n_centroids = router.len();
                if n_centroids == 0 {
                    continue;
                }

                // Sample up to n_probes centroids evenly across the centroid matrix
                let probes_to_run = n_probes.min(n_centroids);
                let stride = if n_centroids > probes_to_run {
                    n_centroids / probes_to_run
                } else {
                    1
                };

                for step in 0..probes_to_run {
                    let idx = step * stride;
                    if let Some(centroid) = router.get_centroid(idx) {
                        let mut params = crate::core::planner::VectorSearchParams::new(
                            &col,
                            crate::core::index::VectorValue::Float32(centroid.to_vec()),
                            10,
                        );
                        params.ef_search = Some(32);
                        params.use_mmap = true;
                        if let Ok(res) = self.execute_vector_search_as_scored(params).await {
                            total_clusters += res.len();
                        }
                        total_probes += 1;
                    }
                }
            }
        }

        Ok(WarmupStats {
            has_vector_indices: true,
            probes_run: total_probes,
            clusters_warmed: total_clusters,
            duration_ms: t0.elapsed().as_millis(),
        })
    }

    /// Explicit Tier 3 Warmup API:
    /// Pre-warms the index caches using synthetic centroid probes.
    pub async fn warmup(
        &self,
        n_probes: Option<usize>,
        max_memory_gb: Option<f64>,
    ) -> Result<WarmupStats> {
        let probes = n_probes.unwrap_or(100);
        let max_bytes = max_memory_gb.map(|gb| (gb * 1024.0 * 1024.0 * 1024.0) as u64);
        self.warmup_from_centroids_async(probes, max_bytes).await
    }

    /// Ensure a table-level vector router exists for the given column, constructing
    /// it from segment index centroids if not already resident.
    pub async fn ensure_vector_router(
        &self,
        column: &str,
    ) -> Result<Option<Arc<crate::core::index::TableVectorRouter>>> {
        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (_manifest, entries, _version) = manager.load_latest_full().await?;

        // The router is only valid for the exact set of segments that carry a
        // vector index for `column` in the current manifest. A commit that adds
        // or removes a segment changes that set, and a stale router silently
        // prunes the new segments out of every query: `might_match_vector` drops
        // any segment absent from `candidate_segments`, so a router built before
        // the segment existed can never route to it. The in-process cache is
        // cleared on this process's own commits, but `preload_indexes_async` can
        // also populate it from a partially-warmed cache, so validate against the
        // manifest before serving it.
        let indexed_segments: std::collections::HashSet<String> = entries
            .iter()
            .filter_map(|entry| {
                let has_vec = entry
                    .index_files
                    .iter()
                    .any(|idx| idx.column_name == column && idx.index_category == "vector");
                if has_vec {
                    let rel_entry =
                        crate::core::segment::relativize_against(&self.uri, &entry.file_path);
                    Some(crate::core::segment::segment_id_from_path(&rel_entry).to_string())
                } else {
                    None
                }
            })
            .collect();

        // Bind the clone before the `if let`: a temporary read guard in the
        // scrutinee would live for the whole block and deadlock the `write()`
        // below.
        let cached_router = self.indexing.vector_routers.read().get(column).cloned();
        if let Some(router) = cached_router {
            let covers = {
                let router_segments: std::collections::HashSet<&str> = router
                    .segment_indices
                    .iter()
                    .map(|(seg, _)| seg.as_str())
                    .collect();
                router_segments.len() == indexed_segments.len()
                    && indexed_segments
                        .iter()
                        .all(|s| router_segments.contains(s.as_str()))
            };
            if covers {
                return Ok(Some(router));
            }
            tracing::warn!(
                column = %column,
                router_segments = router.segment_indices.len(),
                indexed_segments = indexed_segments.len(),
                "ensure_vector_router: cached router does not cover the current indexed segments; dropping"
            );
            self.indexing.vector_routers.write().remove(column);
        }

        let mut records = Vec::new();
        let mut dim = 0;
        let mut metric = crate::core::index::VectorMetric::L2;
        let mut loaded_segments = std::collections::HashSet::new();
        let cache_root = self.cache_root();

        for entry in &entries {
            let rel_entry = crate::core::segment::relativize_against(&self.uri, &entry.file_path);
            let seg_id = crate::core::segment::segment_id_from_path(&rel_entry).to_string();

            for idx in &entry.index_files {
                if idx.column_name == column && idx.index_category == "vector" {
                    let full_path =
                        crate::core::segment::resolve_entry_index_path(&rel_entry, &idx.file_path);

                    if !loaded_segments.insert((seg_id.clone(), idx.column_name.clone())) {
                        continue;
                    }

                    if idx.blob_offset.is_some() {
                        // Read only the centroids blob. Loading the full index
                        // here deserializes the graph, vectors and quantizer to
                        // answer a question that needs none of them — 42 GB of
                        // index at 100M, paid again after every rebuild.
                        match crate::core::index::hnsw_ivf::HnswIvfIndex::load_centroids_from_puffin(
                            self.store.clone(),
                            &full_path,
                            true,
                        )
                        .await
                        {
                            Ok((centroids, loaded_metric)) => {
                                if let Some(first) = centroids.first() {
                                    dim = first.len();
                                }
                                metric = loaded_metric;
                                for (c_id, centroid) in centroids.into_iter().enumerate() {
                                    records.push(
                                        crate::core::index::vector_router::CentroidRecord {
                                            segment_id: seg_id.clone(),
                                            cluster_id: c_id,
                                            centroid,
                                        },
                                    );
                                }
                            }
                            Err(e) => {
                                tracing::warn!(
                                    puffin = %full_path,
                                    error = %e,
                                    "ensure_vector_router: could not read the centroids blob"
                                );
                            }
                        }
                    } else {
                        // Legacy non-Puffin index: fall back to the full loader.
                        let cache_key = format!("{}/{}", cache_root, full_path);
                        if let Ok(index) =
                            crate::core::index::hnsw_ivf::HnswIvfIndex::load_async_with_cache_key(
                                self.store.clone(),
                                &full_path,
                                &cache_key,
                                true,
                            )
                            .await
                        {
                            dim = index.dim();
                            metric = index.metric();
                            for (c_id, centroid) in index.centroids().iter().enumerate() {
                                records.push(crate::core::index::vector_router::CentroidRecord {
                                    segment_id: seg_id.clone(),
                                    cluster_id: c_id,
                                    centroid: centroid.clone(),
                                });
                            }
                        }
                    }
                    break;
                }
            }
        }

        if !records.is_empty() {
            let router = Arc::new(crate::core::index::TableVectorRouter::new(
                dim, metric, records,
            ));
            self.indexing
                .vector_routers
                .write()
                .insert(column.to_string(), router.clone());
            Ok(Some(router))
        } else {
            Ok(None)
        }
    }

    /// Ensure the table-level, cluster-keyed index for `column` is resident.
    ///
    /// Mirrors [`Self::ensure_vector_router`]: returns the cached artifact when
    /// present, otherwise loads it from the manifest's `table_indexes` list.
    ///
    /// The artifact is rejected when the manifest's current data files are not
    /// exactly the set it was built from. That is the table-level lineage
    /// contract: a data commit drops `table_indexes`, but a reader holding a
    /// stale in-process cache must also refuse to serve it.
    pub async fn ensure_cluster_index(
        &self,
        column: &str,
    ) -> Result<Option<Arc<crate::core::index::ClusterIndex>>> {
        let manager = ManifestManager::new(self.store.clone(), "", &self.uri);
        let (manifest, entries, version) = manager.load_latest_full().await?;

        // The set of live data files, normalised to bare segment ids. Both the
        // cached-artifact check and the manifest-artifact check compare against
        // this same set.
        let current_segments: Vec<String> = entries
            .iter()
            .map(|e| {
                let name = e.file_path.rsplit('/').next().unwrap_or(&e.file_path);
                let name = name.strip_suffix(".puffin").unwrap_or(name);
                name.strip_suffix(".parquet").unwrap_or(name).to_string()
            })
            .collect();

        // Validate a cached artifact against the *current* segment set before
        // serving it. The in-process cache is only invalidated by this process's
        // own commits; another process may have compacted since. Loading the
        // manifest is cheap (it is already cached) — only the artifact bytes are
        // expensive, and those stay cached when the lineage still matches.
        if let Some(index) = self.indexing.cluster_indexes.read().get(column).cloned() {
            if index.segment_table.len() == current_segments.len()
                && index
                    .segment_table
                    .iter()
                    .all(|s| current_segments.iter().any(|c| c == s))
            {
                return Ok(Some(index));
            }
            tracing::warn!(
                column = %column,
                artifact_segments = index.segment_table.len(),
                current_segments = current_segments.len(),
                "ensure_cluster_index: cached artifact does not cover the current data files; dropping"
            );
            self.indexing.cluster_indexes.write().remove(column);
        }

        let Some(entry) = manifest
            .table_indexes
            .iter()
            .find(|idx| idx.column_name == column && idx.index_category == "vector")
        else {
            return Ok(None);
        };

        // Reject an artifact that does not cover exactly the current data files.
        if !entry.covers_segments(&current_segments) {
            tracing::warn!(
                column = %column,
                artifact_segments = entry.source_segments.len(),
                current_segments = current_segments.len(),
                "ensure_cluster_index: artifact does not cover the current data files; ignoring"
            );
            return Ok(None);
        }

        let index = if !entry.shard_files.is_empty() {
            crate::core::index::ClusterIndex::load_sharded(
                self.store.clone(),
                &entry.shard_files,
                column,
                version,
            )
            .await?
        } else {
            crate::core::index::ClusterIndex::load(
                self.store.clone(),
                &entry.file_path,
                column,
                version,
            )
            .await?
        };
        let index = Arc::new(index);
        self.indexing
            .cluster_indexes
            .write()
            .insert(column.to_string(), index.clone());
        Ok(Some(index))
    }

    /// Best-effort size of an index file (or its multi-file family).
    async fn index_size_bytes(&self, full_path: &str, idx: &IndexFile) -> u64 {
        // Puffin blob: the manifest records the exact length.
        if let Some(len) = idx.blob_length {
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
        let cache_root = self.cache_root();
        // Puffin-packed vector index: one cache entry per container index, not
        // per blob (see `puffin_vector_cache_key`).
        let cache_key = if idx.blob_offset.is_some() {
            crate::core::index::hnsw_ivf::HnswIvfIndex::puffin_vector_cache_key(
                cache_root,
                full_path,
                &idx.column_name,
                &idx.algorithm,
            )
        } else {
            cache_key_for(cache_root, full_path, idx)
        };
        match kind {
            IndexKind::Vector => {
                // Populates HNSW_IVF_CACHE (the reader's vector fast path).
                if idx.blob_offset.is_some() {
                    let _ = crate::core::index::hnsw_ivf::HnswIvfIndex::load_from_puffin(
                        self.store.clone(),
                        full_path,
                        &cache_key,
                        true,
                        &idx.column_name,
                    )
                    .await?;
                } else {
                    let _ = crate::core::index::hnsw_ivf::HnswIvfIndex::load_async_with_cache_key(
                        self.store.clone(),
                        full_path,
                        &cache_key,
                        true,
                    )
                    .await?;
                }
                Ok(())
            }
            IndexKind::Inverted => {
                use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

                let source_type_key = format!("{cache_key}:source_type");
                let extract_meta = |metadata: &parquet::file::metadata::ParquetMetaData| {
                    let mut analyzer_val: Option<String> = None;
                    let mut source_type_val: Option<String> = None;
                    if let Some(kv_list) = metadata.file_metadata().key_value_metadata() {
                        for kv in kv_list {
                            if kv.key == "analyzer" {
                                analyzer_val = kv.value.clone();
                            } else if kv.key == "source_type" {
                                source_type_val = kv.value.clone();
                            }
                        }
                    }
                    let analyzer = analyzer_val.unwrap_or_else(|| {
                        if idx.algorithm == "bm25" {
                            "analyzer:english".to_string()
                        } else {
                            "identity".to_string()
                        }
                    });
                    let source_type = source_type_val.unwrap_or_default();
                    (analyzer, source_type)
                };

                if INVERTED_INDEX_CACHE.get(&cache_key).await.is_some() {
                    if crate::core::cache::ANALYZER_META_CACHE
                        .get(&cache_key)
                        .await
                        .is_none()
                        || crate::core::cache::ANALYZER_META_CACHE
                            .get(&source_type_key)
                            .await
                            .is_none()
                    {
                        let bytes = self.read_inverted_bytes_raw(full_path, idx).await?;
                        let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
                        let (analyzer, source_type) = extract_meta(builder.metadata());
                        crate::core::cache::ANALYZER_META_CACHE
                            .insert(cache_key.clone(), analyzer)
                            .await;
                        crate::core::cache::ANALYZER_META_CACHE
                            .insert(source_type_key, source_type)
                            .await;
                    }
                    return Ok(());
                }

                let bytes = self.read_inverted_bytes_raw(full_path, idx).await?;
                let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)?;
                let (analyzer, source_type) = extract_meta(builder.metadata());
                crate::core::cache::ANALYZER_META_CACHE
                    .insert(cache_key.clone(), analyzer)
                    .await;
                crate::core::cache::ANALYZER_META_CACHE
                    .insert(source_type_key, source_type)
                    .await;

                let reader = builder.build()?;
                let mut batches = Vec::new();
                for b in reader {
                    batches.push(b?);
                }
                INVERTED_INDEX_CACHE
                    .insert(cache_key, Arc::new(batches))
                    .await;
                Ok(())
            }
            IndexKind::Graph => {
                // CSR families are mmap-served; warming the disk cache is the
                // in-memory-equivalent (page cache) warm for them.
                let disk = DiskCache::new(self.store.clone());
                self.warm_on_disk(&disk, full_path, idx, kind).await
            }
            IndexKind::Other => Ok(()),
        }
    }

    /// Warm an index into the disk (mmap) cache for out-of-core serving.
    async fn warm_on_disk(
        &self,
        disk: &DiskCache,
        full_path: &str,
        idx: &IndexFile,
        kind: IndexKind,
    ) -> Result<()> {
        let cache_root = self.cache_root();
        match kind {
            IndexKind::Vector => {
                // Load with mmap so the HNSW graph is served from the disk cache.
                if idx.blob_offset.is_some() {
                    let cache_key =
                        crate::core::index::hnsw_ivf::HnswIvfIndex::puffin_vector_cache_key(
                            cache_root,
                            full_path,
                            &idx.column_name,
                            &idx.algorithm,
                        );
                    let _ = crate::core::index::hnsw_ivf::HnswIvfIndex::load_from_puffin(
                        self.store.clone(),
                        full_path,
                        &cache_key,
                        true,
                        &idx.column_name,
                    )
                    .await?;
                } else {
                    let cache_key = format!("{}/{}", cache_root, full_path);
                    let _ = crate::core::index::hnsw_ivf::HnswIvfIndex::load_async_with_cache_key(
                        self.store.clone(),
                        full_path,
                        &cache_key,
                        true,
                    )
                    .await?;
                }
                Ok(())
            }
            IndexKind::Inverted => {
                // Pull the file into the disk cache (mmap-backed).
                let _ = disk.get_mmap(full_path).await?;
                Ok(())
            }
            IndexKind::Graph => {
                if idx.blob_offset.is_some() || full_path.ends_with(".puffin") {
                    let _ = disk.get_mmap(full_path).await?;
                } else {
                    let base = full_path
                        .strip_suffix(".graph_v2.csr.offsets")
                        .or_else(|| full_path.strip_suffix(".graph_v2.csr.edges"))
                        .or_else(|| full_path.strip_suffix(".graph_v2.csr.dict"))
                        .or_else(|| full_path.strip_suffix(".csr.offsets"))
                        .or_else(|| full_path.strip_suffix(".csr.edges"))
                        .or_else(|| full_path.strip_suffix(".csr.dict"))
                        .unwrap_or(full_path);
                    let csr_prefix =
                        if full_path.contains(".csr.") && !full_path.contains(".graph_v2.csr.") {
                            ".csr"
                        } else {
                            ".graph_v2.csr"
                        };
                    for suffix in [".offsets", ".edges", ".dict"] {
                        let p = format!("{}{}{}", base, csr_prefix, suffix);
                        let _ = disk.get_mmap(&p).await;
                    }
                }
                Ok(())
            }
            IndexKind::Other => Ok(()),
        }
    }

    /// Read raw inverted-index Parquet bytes (handling Puffin blob ranges and standalone files).
    async fn read_inverted_bytes_raw(
        &self,
        full_path: &str,
        idx: &IndexFile,
    ) -> Result<bytes::Bytes> {
        if let (Some(offset), Some(length)) = (idx.blob_offset, idx.blob_length) {
            Ok(self
                .store
                .get_range(
                    &object_store::path::Path::from(full_path),
                    (offset as u64)..(offset as u64 + length as u64),
                )
                .await?)
        } else {
            Ok(self
                .store
                .get(&object_store::path::Path::from(full_path))
                .await?
                .bytes()
                .await?)
        }
    }
}

/// The kind of index, derived from the manifest's `index_category` / `blob_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum IndexKind {
    Vector,
    Inverted,
    Graph,
    Other,
}

fn classify(idx: &IndexFile) -> IndexKind {
    let blob = idx.blob_type.as_deref().unwrap_or("");
    let itype = idx.index_category.as_str();

    if blob.starts_with("hnsw") || blob.starts_with("tq") || itype == "vector" {
        return IndexKind::Vector;
    }
    if blob == "csr_graph" || itype == "graph" {
        return IndexKind::Graph;
    }
    if idx.is_lexical() {
        return IndexKind::Inverted;
    }
    IndexKind::Other
}

fn cache_key_for(root_uri: &str, full_path: &str, idx: &IndexFile) -> String {
    if let Some(offset) = idx.blob_offset {
        format!("{}/{}:{}", root_uri, full_path, offset)
    } else {
        format!("{}/{}", root_uri, full_path)
    }
}
