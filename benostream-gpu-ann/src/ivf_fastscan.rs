// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

use crate::backend::{ComputeContext, GpuBackend};
use crate::fastscan::{FastScanCodebook, FASTSCAN_BLOCK_SIZE};
use crate::ivf_flat::SearchResult;
use crate::kmeans::train_kmeans;
use crate::metric::Metric;
use anyhow::Result;
use std::collections::BinaryHeap;

/// High-Throughput IVF FastScan Index with In-Register SIMD Traversal.
///
/// Combines coarse Voronoi partition pruning (via e-means / k-means) with
/// 4-bit in-register SIMD Block Inverted List scanning. Evaluates 32 vector
/// distances per cycle via AVX2 / ARM NEON byte shuffles directly in registers.
pub struct IvfFastScanIndex {
    centroids: Vec<f32>,
    codebook: FastScanCodebook,
    /// Inverted list codes packed into contiguous 32-vector blocks (ceil(L/32) * M * 16 bytes per cluster).
    list_codes: Vec<Vec<u8>>,
    /// Vector IDs in each cluster.
    list_ids: Vec<Vec<u64>>,
    dim: usize,
    metric: Metric,
    use_residuals: bool,
    ctx: ComputeContext,
}

impl IvfFastScanIndex {
    /// Build an IVF FastScan index.
    pub fn build(
        vectors: &[f32],
        ids: Option<&[u64]>,
        dim: usize,
        n_lists: Option<usize>,
        m: Option<usize>,
        metric: Metric,
        ctx: Option<ComputeContext>,
    ) -> Result<Self> {
        Self::build_with_options(vectors, ids, dim, n_lists, m, metric, true, ctx)
    }

    /// Build an IVF FastScan index with explicit residual encoding options.
    pub fn build_with_options(
        vectors: &[f32],
        ids: Option<&[u64]>,
        dim: usize,
        n_lists: Option<usize>,
        m: Option<usize>,
        metric: Metric,
        use_residuals: bool,
        ctx: Option<ComputeContext>,
    ) -> Result<Self> {
        let n_vectors = vectors.len() / dim;
        if n_vectors == 0 {
            anyhow::bail!("Cannot build index from empty vector array");
        }

        let ctx = ctx.unwrap_or_else(ComputeContext::auto_detect);

        // Optimal cluster count heuristic
        let n_lists = n_lists
            .unwrap_or_else(|| ((n_vectors as f64).sqrt() as usize).clamp(1, 4096))
            .min(n_vectors);

        // Determine subvector count m (must divide dim)
        let m = m.unwrap_or(if dim.is_multiple_of(4) && dim / 4 >= 1 {
            dim / 4
        } else if dim.is_multiple_of(2) {
            dim / 2
        } else {
            1
        });

        tracing::info!(
            "Building IVF-FastScan index: {} vectors, dim={}, clusters={}, m={}, residuals={}, backend={}",
            n_vectors,
            dim,
            n_lists,
            m,
            use_residuals,
            ctx.name()
        );

        // 1. Train coarse centroids (routes to e-means when eligible)
        let centroids = train_kmeans(vectors, dim, n_lists, 10, &ctx)?;
        let actual_k = centroids.len() / dim;

        // 2. Assign all vectors to nearest centroids in bulk
        let assignments = ctx
            .compute_kmeans_assignment(vectors, &centroids, dim)
            .unwrap_or_else(|_| {
                crate::backend::cpu::CpuBackend::new()
                    .compute_kmeans_assignment(vectors, &centroids, dim)
                    .unwrap_or_default()
            });

        // 3. Partition vectors into clusters (optionally computing residuals r = x - c)
        let mut cluster_raw: Vec<Vec<f32>> = vec![Vec::new(); actual_k];
        let mut cluster_ids: Vec<Vec<u64>> = vec![Vec::new(); actual_k];

        for (i, &cluster_id) in assignments.iter().enumerate() {
            let cid = cluster_id as usize;
            if cid < actual_k {
                let id = ids.map(|s| s[i]).unwrap_or(i as u64);
                let v = &vectors[i * dim..(i + 1) * dim];

                if use_residuals && metric == Metric::L2 {
                    let c = &centroids[cid * dim..(cid + 1) * dim];
                    let r: Vec<f32> = v.iter().zip(c.iter()).map(|(&x, &cent)| x - cent).collect();
                    cluster_raw[cid].extend_from_slice(&r);
                } else {
                    cluster_raw[cid].extend_from_slice(v);
                }
                cluster_ids[cid].push(id);
            }
        }

        // 4. Train 4-bit FastScan codebook on all partitioned vectors/residuals
        let mut training_sample = Vec::with_capacity(vectors.len());
        for c_vecs in &cluster_raw {
            training_sample.extend_from_slice(c_vecs);
        }
        if training_sample.is_empty() {
            training_sample.extend_from_slice(vectors);
        }

        let codebook = FastScanCodebook::train(&training_sample, dim, m, 42)?;

        // 5. Pack each cluster into 32-vector interleaved FastScan blocks
        let mut list_codes = Vec::with_capacity(actual_k);
        let mut list_ids = Vec::with_capacity(actual_k);

        for (cid, raw_vecs) in cluster_raw.into_iter().enumerate() {
            if raw_vecs.is_empty() {
                list_codes.push(Vec::new());
                list_ids.push(Vec::new());
            } else {
                let (encoded, _) = codebook.encode_blocks(&raw_vecs);
                list_codes.push(encoded);
                list_ids.push(cluster_ids[cid].clone());
            }
        }

        Ok(Self {
            centroids,
            codebook,
            list_codes,
            list_ids,
            dim,
            metric,
            use_residuals,
            ctx,
        })
    }

    /// Query the index for the top-k nearest neighbors using in-register SIMD FastScan.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        n_probe: usize,
        filter: Option<&roaring::RoaringBitmap>,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            anyhow::bail!(
                "Query dimension {} does not match index {}",
                query.len(),
                self.dim
            );
        }

        let n_clusters = self.centroids.len() / self.dim;
        let n_probe = n_probe.max(1).min(n_clusters);

        // Step 1: Coarse Search - rank centroids on GPU / hardware accelerator
        let centroid_distances =
            self.ctx
                .compute_distance(query, &self.centroids, self.dim, self.metric)?;

        let mut ranked_centroids: Vec<(usize, f32)> =
            centroid_distances.into_iter().enumerate().collect();
        ranked_centroids.sort_by(|a, b| a.1.total_cmp(&b.1));

        // Step 2: Fine Search - scan block inverted lists using in-register byte shuffles
        let mut heap: BinaryHeap<SearchResult> = BinaryHeap::with_capacity(k);
        let block_bytes = self.codebook.m * 16;

        for (cluster_id, _) in ranked_centroids.into_iter().take(n_probe) {
            let encoded_blocks = &self.list_codes[cluster_id];
            let ids = &self.list_ids[cluster_id];

            if encoded_blocks.is_empty() || ids.is_empty() {
                continue;
            }

            // Compute query residual if using residuals
            let query_buf: Vec<f32>;
            let q_to_use: &[f32] = if self.use_residuals && self.metric == Metric::L2 {
                let c = &self.centroids[cluster_id * self.dim..(cluster_id + 1) * self.dim];
                query_buf = query
                    .iter()
                    .zip(c.iter())
                    .map(|(&q, &cent)| q - cent)
                    .collect();
                &query_buf
            } else {
                query
            };

            // Compute LUT once per probed cluster
            let (lut_u8, min_dist_sum, scale) = self.codebook.compute_lut(q_to_use);

            let n_blocks = encoded_blocks.len() / block_bytes;
            let total_cluster_vectors = ids.len();

            for b in 0..n_blocks {
                let block_slice = &encoded_blocks[b * block_bytes..(b + 1) * block_bytes];
                let block_dists = self.codebook.scan_block(block_slice, &lut_u8);

                let base_idx = b * FASTSCAN_BLOCK_SIZE;
                for (i, &dist_int) in block_dists.iter().enumerate().take(FASTSCAN_BLOCK_SIZE) {
                    let local_idx = base_idx + i;
                    if local_idx >= total_cluster_vectors {
                        break;
                    }

                    let id = ids[local_idx];
                    if let Some(f) = filter {
                        if !f.contains(id as u32) {
                            continue;
                        }
                    }

                    let dist = min_dist_sum + (dist_int as f32) * scale;

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
        }

        let mut results = heap.into_sorted_vec();
        results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        Ok(results)
    }

    /// Number of vectors indexed.
    pub fn len(&self) -> usize {
        self.list_ids.iter().map(|b| b.len()).sum()
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
