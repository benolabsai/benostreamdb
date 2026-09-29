// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

use crate::backend::ComputeContext;
use crate::ivf_flat::SearchResult;
use crate::metric::Metric;
use anyhow::Result;
use rand::{thread_rng, Rng};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

/// Single node in the HNSW multilayer graph.
#[derive(Debug, Clone)]
pub struct HnswNode {
    /// Maximum layer this node exists on (0 <= layer <= max_layer)
    pub layer: usize,
    /// Per-layer neighbor indices: `neighbors[l]` is the list of node IDs at layer `l`
    pub neighbors: Vec<Vec<usize>>,
}

/// Candidate point during graph search or insertion.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    node_idx: usize,
    distance: f32,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// Min-heap ordering (smallest distance first)
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.distance.total_cmp(&self.distance)
    }
}

/// Max-heap candidate for keeping the closest ef points (largest distance evicted first).
#[derive(Debug, Clone, Copy, PartialEq)]
struct FurthestCandidate {
    node_idx: usize,
    distance: f32,
}

impl Eq for FurthestCandidate {}

impl PartialOrd for FurthestCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FurthestCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.total_cmp(&other.distance)
    }
}

/// GPU-Accelerated Hierarchical Navigable Small World (HNSW) Index.
///
/// Accelerates candidate frontier distance computations during index construction
/// and search using the active hardware accelerator (CUDA, Apple Metal, WGPU, or CPU SIMD).
pub struct HnswIndex {
    nodes: Vec<HnswNode>,
    vectors: Vec<f32>,
    ids: Vec<u64>,
    entry_point: Option<usize>,
    max_layer: usize,
    m: usize,
    m0: usize,
    ef_construction: usize,
    ml: f64,
    dim: usize,
    metric: Metric,
    ctx: ComputeContext,
}

impl HnswIndex {
    /// Create a new, empty HNSW index.
    pub fn new(
        dim: usize,
        metric: Metric,
        m: usize,
        ef_construction: usize,
        ctx: Option<ComputeContext>,
    ) -> Self {
        let m = m.max(4);
        let m0 = m * 2;
        let ml = 1.0 / (m as f64).ln();
        let ctx = ctx.unwrap_or_else(ComputeContext::auto_detect);

        Self {
            nodes: Vec::new(),
            vectors: Vec::new(),
            ids: Vec::new(),
            entry_point: None,
            max_layer: 0,
            m,
            m0,
            ef_construction: ef_construction.max(m),
            ml,
            dim,
            metric,
            ctx,
        }
    }

    /// Sample a random maximum layer for a new node using exponential decay: -ln(unif) * mL
    fn sample_layer(&self) -> usize {
        let mut rng = thread_rng();
        let unif: f64 = rng.gen_range(1e-7..1.0);
        ((-unif.ln()) * self.ml).floor() as usize
    }

    /// Evaluate distance between a query vector and a candidate node on CPU or GPU.
    #[inline]
    fn dist(&self, q: &[f32], node_idx: usize) -> f32 {
        let v = &self.vectors[node_idx * self.dim..(node_idx + 1) * self.dim];
        match self.metric {
            Metric::L2 => {
                let sum: f32 = q
                    .iter()
                    .zip(v.iter())
                    .map(|(a, b)| {
                        let d = a - b;
                        d * d
                    })
                    .sum();
                sum.sqrt()
            }
            Metric::Cosine => {
                let mut dot = 0.0f32;
                let mut nq = 0.0f32;
                let mut nv = 0.0f32;
                for (&a, &b) in q.iter().zip(v.iter()) {
                    dot += a * b;
                    nq += a * a;
                    nv += b * b;
                }
                if nq == 0.0 || nv == 0.0 {
                    1.0
                } else {
                    1.0 - (dot / (nq.sqrt() * nv.sqrt()))
                }
            }
            Metric::InnerProduct => -q.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<f32>(),
            Metric::L1 => q.iter().zip(v.iter()).map(|(a, b)| (a - b).abs()).sum(),
            Metric::Hamming => {
                let mut dist = 0.0f32;
                for (&a, &b) in q.iter().zip(v.iter()) {
                    if a != b {
                        dist += 1.0;
                    }
                }
                dist
            }
            Metric::Jaccard => {
                let mut inter = 0.0f32;
                let mut union_count = 0.0f32;
                for (&a, &b) in q.iter().zip(v.iter()) {
                    if a > 0.0 || b > 0.0 {
                        if a == b && a > 0.0 {
                            inter += 1.0;
                        }
                        union_count += 1.0;
                    }
                }
                if union_count == 0.0 {
                    0.0
                } else {
                    1.0 - (inter / union_count)
                }
            }
        }
    }

    /// Batched distance evaluation against a slice of candidate node indices.
    /// Dispatches to the GPU accelerator when candidate count exceeds the GPU threshold.
    fn dist_batch(&self, q: &[f32], candidate_nodes: &[usize]) -> Vec<f32> {
        const GPU_DISPATCH_THRESHOLD: usize = 32;

        if candidate_nodes.len() >= GPU_DISPATCH_THRESHOLD {
            // Gather candidate vectors into a flat contiguous buffer
            let mut flat_buf = Vec::with_capacity(candidate_nodes.len() * self.dim);
            for &c_idx in candidate_nodes {
                flat_buf.extend_from_slice(&self.vectors[c_idx * self.dim..(c_idx + 1) * self.dim]);
            }

            if let Ok(gpu_dists) = self
                .ctx
                .compute_distance(q, &flat_buf, self.dim, self.metric)
            {
                return gpu_dists;
            }
        }

        // Fallback to inline calculation
        candidate_nodes
            .iter()
            .map(|&idx| self.dist(q, idx))
            .collect()
    }

    /// Search layer `lc` for the `ef` closest neighbors to `q` starting from `entry_points`.
    fn search_layer(
        &self,
        q: &[f32],
        entry_points: &[usize],
        ef: usize,
        lc: usize,
    ) -> Vec<Candidate> {
        let mut visited: HashSet<usize> = HashSet::with_capacity(ef * 2);
        let mut candidates: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut w: BinaryHeap<FurthestCandidate> = BinaryHeap::with_capacity(ef);

        for &ep in entry_points {
            let d = self.dist(q, ep);
            visited.insert(ep);
            candidates.push(Candidate {
                node_idx: ep,
                distance: d,
            });
            w.push(FurthestCandidate {
                node_idx: ep,
                distance: d,
            });
        }

        while let Some(c) = candidates.pop() {
            if let Some(furthest) = w.peek() {
                if c.distance > furthest.distance {
                    break;
                }
            }

            let neighbors = &self.nodes[c.node_idx].neighbors[lc];
            let unvisited: Vec<usize> = neighbors
                .iter()
                .copied()
                .filter(|&n| visited.insert(n))
                .collect();

            if unvisited.is_empty() {
                continue;
            }

            // Batched GPU distance evaluation
            let dists = self.dist_batch(q, &unvisited);

            for (&n_idx, &dist) in unvisited.iter().zip(dists.iter()) {
                let furthest_dist = w.peek().map(|f| f.distance).unwrap_or(f32::MAX);
                if dist < furthest_dist || w.len() < ef {
                    candidates.push(Candidate {
                        node_idx: n_idx,
                        distance: dist,
                    });
                    w.push(FurthestCandidate {
                        node_idx: n_idx,
                        distance: dist,
                    });
                    if w.len() > ef {
                        w.pop();
                    }
                }
            }
        }

        w.into_iter()
            .map(|f| Candidate {
                node_idx: f.node_idx,
                distance: f.distance,
            })
            .collect()
    }

    /// Select M closest neighbors from candidate list.
    fn select_neighbors(&self, candidates: &mut [Candidate], m: usize) -> Vec<usize> {
        candidates.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        candidates.iter().take(m).map(|c| c.node_idx).collect()
    }

    /// Insert a single vector into the HNSW graph.
    pub fn insert(&mut self, vector: &[f32], id: u64) -> Result<usize> {
        if vector.len() != self.dim {
            anyhow::bail!(
                "Vector dimension {} does not match index {}",
                vector.len(),
                self.dim
            );
        }

        let new_idx = self.nodes.len();
        let target_layer = self.sample_layer();

        let mut node = HnswNode {
            layer: target_layer,
            neighbors: vec![Vec::new(); target_layer + 1],
        };

        self.vectors.extend_from_slice(vector);
        self.ids.push(id);

        if let Some(curr_entry) = self.entry_point {
            let mut ep = curr_entry;
            let mut ep_dist = self.dist(vector, ep);
            let mut curr_layer = self.max_layer;

            // 1. Greedily descend from top layer down to target_layer + 1
            while curr_layer > target_layer {
                let mut changed = true;
                while changed {
                    changed = false;
                    let neighbors = &self.nodes[ep].neighbors[curr_layer];
                    let dists = self.dist_batch(vector, neighbors);
                    for (&n_idx, &d) in neighbors.iter().zip(dists.iter()) {
                        if d < ep_dist {
                            ep_dist = d;
                            ep = n_idx;
                            changed = true;
                        }
                    }
                }
                if curr_layer == 0 {
                    break;
                }
                curr_layer -= 1;
            }

            // 2. Insert into layers from min(target_layer, max_layer) down to 0
            let mut eps = vec![ep];
            for l in (0..=target_layer.min(self.max_layer)).rev() {
                let mut candidates = self.search_layer(vector, &eps, self.ef_construction, l);
                let max_m = if l == 0 { self.m0 } else { self.m };
                let selected = self.select_neighbors(&mut candidates, max_m);

                // Form bidirectional connections
                node.neighbors[l] = selected.clone();

                for &neighbor_idx in &selected {
                    self.nodes[neighbor_idx].neighbors[l].push(new_idx);

                    // Shrink neighbor connections if exceeding max_m
                    if self.nodes[neighbor_idx].neighbors[l].len() > max_m {
                        let n_vec = self.vectors
                            [neighbor_idx * self.dim..(neighbor_idx + 1) * self.dim]
                            .to_vec();
                        let nbrs = self.nodes[neighbor_idx].neighbors[l].clone();
                        let mut n_cands: Vec<Candidate> = nbrs
                            .iter()
                            .map(|&nbr| Candidate {
                                node_idx: nbr,
                                distance: self.dist(&n_vec, nbr),
                            })
                            .collect();
                        let pruned = self.select_neighbors(&mut n_cands, max_m);
                        self.nodes[neighbor_idx].neighbors[l] = pruned;
                    }
                }

                eps = selected;
            }

            self.nodes.push(node);

            // Update entry point if new node has higher layer
            if target_layer > self.max_layer {
                self.max_layer = target_layer;
                self.entry_point = Some(new_idx);
            }
        } else {
            // First node in the graph
            self.nodes.push(node);
            self.entry_point = Some(new_idx);
            self.max_layer = target_layer;
        }

        Ok(new_idx)
    }

    /// Build HNSW graph from a flat slice of vectors.
    pub fn build(
        vectors: &[f32],
        ids: Option<&[u64]>,
        dim: usize,
        metric: Metric,
        m: usize,
        ef_construction: usize,
        ctx: Option<ComputeContext>,
    ) -> Result<Self> {
        let n = vectors.len() / dim;
        if n == 0 {
            anyhow::bail!("Cannot build HNSW index from empty vector array");
        }

        let mut index = Self::new(dim, metric, m, ef_construction, ctx);
        tracing::info!(
            "Building GPU-accelerated HNSW index: {} vectors, dim={}, M={}, efConstruction={}, backend={}",
            n,
            dim,
            m,
            ef_construction,
            index.ctx.name()
        );

        for i in 0..n {
            let v = &vectors[i * dim..(i + 1) * dim];
            let id = ids.map(|s| s[i]).unwrap_or(i as u64);
            index.insert(v, id)?;
        }

        Ok(index)
    }

    /// Search HNSW graph for top-k nearest neighbors.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        filter: Option<&roaring::RoaringBitmap>,
    ) -> Result<Vec<SearchResult>> {
        if query.len() != self.dim {
            anyhow::bail!(
                "Query dimension {} does not match index {}",
                query.len(),
                self.dim
            );
        }

        let Some(mut ep) = self.entry_point else {
            return Ok(Vec::new());
        };

        let mut ep_dist = self.dist(query, ep);
        let ef = ef_search.max(k);

        // 1. Greedily descend from max_layer down to layer 1
        for l in (1..=self.max_layer).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let neighbors = &self.nodes[ep].neighbors[l];
                let dists = self.dist_batch(query, neighbors);
                for (&n_idx, &d) in neighbors.iter().zip(dists.iter()) {
                    if d < ep_dist {
                        ep_dist = d;
                        ep = n_idx;
                        changed = true;
                    }
                }
            }
        }

        // 2. Search bottom layer 0 with beam search size ef
        let candidates = self.search_layer(query, &[ep], ef, 0);

        // 3. Filter and sort top-k
        let mut results: Vec<SearchResult> = candidates
            .into_iter()
            .filter_map(|c| {
                let id = self.ids[c.node_idx];
                if let Some(f) = filter {
                    if !f.contains(id as u32) {
                        return None;
                    }
                }
                Some(SearchResult {
                    id,
                    distance: c.distance,
                })
            })
            .collect();

        if let Some(f) = filter {
            if results.len() < k && (f.len() as usize) <= 1024 {
                let existing: HashSet<u64> = results.iter().map(|r| r.id).collect();
                for id in f.iter() {
                    let id64 = id as u64;
                    if !existing.contains(&id64) {
                        if let Some(node_idx) = self.ids.iter().position(|&x| x == id64) {
                            let d = self.dist(query, node_idx);
                            results.push(SearchResult {
                                id: id64,
                                distance: d,
                            });
                        }
                    }
                }
            }
        }

        results.sort_by(|a, b| a.distance.total_cmp(&b.distance));
        results.truncate(k);
        Ok(results)
    }

    /// Number of nodes in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if graph is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Dimensionality of vectors.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Distance metric.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// Backend accelerator name.
    pub fn backend_name(&self) -> &str {
        self.ctx.name()
    }
}
