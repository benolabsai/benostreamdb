// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

use anyhow::Result;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use crate::backend::{ComputeContext, GpuBackend};
use crate::kmeans::train_kmeans;
use crate::metric::Metric;

/// Search result item containing the vector ID and distance to query.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: u64,
    pub distance: f32,
}

impl Eq for SearchResult {}

impl PartialOrd for SearchResult {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for SearchResult {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap by distance so smallest distances remain
        self.distance.total_cmp(&other.distance)
    }
}

/// GPU-Accelerated IVF-Flat Index.
///
/// Divides vector space into Voronoi cells via coarse k-means centroids.
/// During query time, coarse centroid search prunes the search space to `n_probe`
/// buckets, which are then scanned in parallel on the GPU with zero graph-building overhead.
pub struct IvfFlatIndex {
    centroids: Vec<f32>,
    bucket_vectors: Vec<Vec<f32>>,
    bucket_ids: Vec<Vec<u64>>,
    dim: usize,
    metric: Metric,
    ctx: ComputeContext,
}

impl IvfFlatIndex {
    /// Build an IVF-Flat index from a contiguous array of vectors (`n_vectors * dim`).
    pub fn build(
        vectors: &[f32],
        ids: Option<&[u64]>,
        dim: usize,
        n_lists: Option<usize>,
        metric: Metric,
        ctx: Option<ComputeContext>,
    ) -> Result<Self> {
        let n_vectors = vectors.len() / dim;
        if n_vectors == 0 {
            anyhow::bail!("Cannot build index from empty vector array");
        }

        let ctx = ctx.unwrap_or_else(ComputeContext::auto_detect);

        // Optimal cluster count heuristic if not specified
        let n_lists = n_lists
            .unwrap_or_else(|| ((n_vectors as f64).sqrt() as usize).clamp(1, 4096))
            .min(n_vectors);

        tracing::info!(
            "Building GPU IVF-Flat index: {} vectors, dim={}, clusters={}, backend={}",
            n_vectors,
            dim,
            n_lists,
            ctx.name()
        );

        // 1. Train cluster centroids on GPU
        let centroids = train_kmeans(vectors, dim, n_lists, 10, &ctx)?;
        let actual_k = centroids.len() / dim;

        // 2. Assign all vectors to nearest centroids in bulk on GPU
        let assignments = ctx
            .compute_kmeans_assignment(vectors, &centroids, dim)
            .unwrap_or_else(|_| {
                crate::backend::cpu::CpuBackend::new()
                    .compute_kmeans_assignment(vectors, &centroids, dim)
                    .unwrap_or_default()
            });

        // 3. Bucket vectors into flat contiguous arrays
        let mut bucket_vectors: Vec<Vec<f32>> = vec![Vec::new(); actual_k];
        let mut bucket_ids: Vec<Vec<u64>> = vec![Vec::new(); actual_k];

        for (i, &cluster_id) in assignments.iter().enumerate() {
            let cid = cluster_id as usize;
            if cid < actual_k {
                let id = ids.map(|s| s[i]).unwrap_or(i as u64);
                let v = &vectors[i * dim..(i + 1) * dim];
                bucket_vectors[cid].extend_from_slice(v);
                bucket_ids[cid].push(id);
            }
        }

        Ok(Self {
            centroids,
            bucket_vectors,
            bucket_ids,
            dim,
            metric,
            ctx,
        })
    }

    /// Query the index for the top-k nearest neighbors.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        n_probe: usize,
        filter: Option<&roaring::RoaringBitmap>,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            anyhow::bail!("Query dimension {} does not match index {}", query.len(), self.dim);
        }

        let n_clusters = self.centroids.len() / self.dim;
        let n_probe = n_probe.max(1).min(n_clusters);

        // Step 1: Coarse Search - rank centroids on GPU
        let centroid_distances = self.ctx.compute_distance(
            query,
            &self.centroids,
            self.dim,
            self.metric,
        )?;

        let mut ranked_centroids: Vec<(usize, f32)> =
            centroid_distances.into_iter().enumerate().collect();
        ranked_centroids.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Step 2: Fine Search - scan vectors inside top n_probe buckets
        let mut heap: BinaryHeap<SearchResult> = BinaryHeap::with_capacity(k);

        for (cluster_id, _) in ranked_centroids.into_iter().take(n_probe) {
            let bucket_vecs = &self.bucket_vectors[cluster_id];
            let bucket_ids = &self.bucket_ids[cluster_id];

            if bucket_vecs.is_empty() {
                continue;
            }

            // GPU distance evaluation for the entire cluster in parallel
            let distances = self.ctx.compute_distance(
                query,
                bucket_vecs,
                self.dim,
                self.metric,
            )?;

            for (local_idx, &dist) in distances.iter().enumerate() {
                let id = bucket_ids[local_idx];

                if let Some(f) = filter {
                    if !f.contains(id as u32) {
                        continue;
                    }
                }

                if heap.len() < k {
                    heap.push(SearchResult { id, distance: dist });
                } else if let Some(top) = heap.peek() {
                    if dist < top.distance {
                        heap.pop();
                        heap.push(SearchResult { id, distance: dist });
                    }
                }
            }
        }

        let mut results = heap.into_sorted_vec();
        // into_sorted_vec returns in ascending order for min-comparison or descending for max
        results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        Ok(results)
    }

    /// Number of vectors indexed.
    pub fn len(&self) -> usize {
        self.bucket_ids.iter().map(|b| b.len()).sum()
    }

    /// Check if index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Vector dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Distance metric.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Active compute accelerator backend name.
    pub fn backend_name(&self) -> &str {
        self.ctx.name()
    }
}
