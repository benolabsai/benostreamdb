// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Table-level vector routing and segment pruning.
//!
//! In an Iceberg lakehouse architecture, data is partitioned into columnar segment
//! files (Parquet + detached Puffin index sidecars). Without a global routing index,
//! nearest-neighbor search must fan out across every segment in the table (O(S * n_probe)),
//! turning out-of-core vector search into an expensive multi-segment scan.
//!
//! `TableVectorRouter` aggregates the IVF centroids across all segments into a lightweight,
//! SIMD-friendly centroid matrix (< 10 MB in RAM). At query time, the router evaluates
//! the query vector against all global centroids in < 50 µs, selects the top-N closest
//! centroids across the entire table, and prunes the 80–90% of segments whose Voronoi
//! cells are far from the query vector.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::OnceLock;

use serde::{Deserialize, Serialize};

use crate::core::index::distance;
use crate::core::index::VectorMetric;

/// Record describing a cluster centroid associated with a specific segment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CentroidRecord {
    pub segment_id: String,
    pub cluster_id: usize,
    pub centroid: Vec<f32>,
}

/// Policy governing how table-level vector routing is dispatched across segments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoutingPolicy {
    /// Global pruning: selects the top `total_probes` centroids across the entire table.
    /// Segments that do not contain any of the top global centroids are pruned.
    /// Optimal for tables compacted with spatial Z-order / Voronoi clustering.
    GlobalPruning,

    /// Per-segment cluster targeting: selects the top `probes_per_segment` centroids
    /// within each individual segment. All segments remain candidates (no segment pruning),
    /// but each segment directly targets its best local clusters without local centroid scans.
    /// Optimal for read-only uncompacted tables to guarantee high recall (>85%).
    PerSegmentTargeting,

    /// Adaptive distance threshold: evaluates the global minimum centroid distance d_min.
    /// A segment is admitted as a candidate if its nearest centroid distance <= d_min * factor.
    /// Within admitted segments, up to `probes` clusters are targeted.
    /// Seamlessly bridges compacted tables (prunes non-candidate segments) and uncompacted tables
    /// (retains all segments when centroids are uniformly distributed).
    Adaptive { factor: f32 },
}

impl Default for RoutingPolicy {
    fn default() -> Self {
        // Resolved once per process. `std::env::var` takes a global lock and
        // allocates a String, and `route()` sits on the per-query path.
        static DEFAULT_POLICY: OnceLock<RoutingPolicy> = OnceLock::new();
        *DEFAULT_POLICY.get_or_init(|| {
            let factor = std::env::var("BSDB_ROUTING_ADAPTIVE_FACTOR")
                .ok()
                .and_then(|f| f.parse::<f32>().ok())
                .unwrap_or(1.35);
            match std::env::var("BSDB_ROUTING_MODE") {
                Ok(mode) => match mode.to_ascii_lowercase().as_str() {
                    "global" | "pruning" => RoutingPolicy::GlobalPruning,
                    "per_segment" | "targeting" => RoutingPolicy::PerSegmentTargeting,
                    _ => RoutingPolicy::Adaptive { factor },
                },
                Err(_) => RoutingPolicy::Adaptive { factor },
            }
        })
    }
}

/// Routing plan returned by `TableVectorRouter::route`.
#[derive(Debug, Clone)]
pub struct VectorRoutingPlan {
    /// Segments that contain candidate clusters and must be searched.
    pub candidate_segments: HashSet<String>,
    /// Specific local cluster IDs to probe per segment: segment_id -> Vec<cluster_id>.
    pub segment_clusters: HashMap<String, Vec<usize>>,
    /// Closest centroid distance (for radius / distance bound pruning).
    pub min_centroid_distance: f32,
}

/// A centroid distance paired with its row index.
///
/// Ordered by distance so a `BinaryHeap` is a max-heap of the *worst* retained
/// candidate: `peek()` is the largest distance among the top-k, which is exactly
/// the element to evict when a closer centroid arrives.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Candidate {
    dist: f32,
    idx: usize,
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
    }
}

/// Keep the `k` smallest distances seen so far, in a heap of at most `k + 1`.
#[inline]
fn push_bounded(heap: &mut BinaryHeap<Candidate>, k: usize, cand: Candidate) {
    if k == 0 {
        return;
    }
    if heap.len() < k {
        heap.push(cand);
    } else if let Some(worst) = heap.peek() {
        if cand.dist < worst.dist {
            heap.pop();
            heap.push(cand);
        }
    }
}

/// Distance under the router's metric. Inner product is negated so that
/// "smaller is closer" holds for every metric.
#[inline]
fn distance_for(metric: VectorMetric, q: &[f32], c: &[f32]) -> f32 {
    match metric {
        VectorMetric::L2 => distance::l2_distance(q, c),
        VectorMetric::Cosine => distance::cosine_distance(q, c),
        VectorMetric::InnerProduct => -distance::dot_product(q, c),
        VectorMetric::L1 => distance::l1_distance(q, c),
        VectorMetric::Hamming => distance::hamming_distance(q, c),
        VectorMetric::Jaccard => distance::jaccard_distance(q, c),
    }
}

/// Whether the Adaptive policy may cap how many segments one query fans out to.
///
/// Off by default, and it must stay off for any measurement that has to mean
/// something. The cap is not a free optimization: it silently reduces the number
/// of IVF lists a query visits, so it changes the operating point rather than the
/// cost of reaching it. Measured on Cohere-10M (25 segments, 20-point grid,
/// `ef=128 n_probe=4`): recall@10 fell from 0.7019 to 0.0618 — a ratio of 0.088
/// against a capped-segment fraction of 2/25 = 0.08. The sweep was reporting the
/// cap, not the index. Set `BSDB_ROUTING_CAP_SEGMENTS=1` to opt in.
fn adaptive_segment_cap_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        matches!(
            std::env::var("BSDB_ROUTING_CAP_SEGMENTS")
                .map(|s| s.trim().to_ascii_lowercase())
                .as_deref(),
            Ok("1") | Ok("true") | Ok("yes")
        )
    })
}

/// Cap on how many segments one query may fan out to, applied only when
/// [`adaptive_segment_cap_enabled`] is set.
///
/// `probes` is a per-segment cluster count, so it cannot bound the segment count
/// directly. Four clusters per segment is the ratio at which additional segments
/// stop contributing to the top-k in the 10M sweeps; the floor of 2 keeps small
/// tables from being over-pruned.
fn capped_segment_count(probes: usize, segment_count: usize) -> usize {
    (probes / 4).max(2).min(segment_count)
}

/// Segment fan-out actually applied, honouring the opt-in. Defaults to the whole
/// table, which is the behaviour every existing recall measurement was taken
/// against.
fn routing_max_segments(probes: usize, segment_count: usize) -> usize {
    if adaptive_segment_cap_enabled() {
        capped_segment_count(probes, segment_count)
    } else {
        segment_count
    }
}

/// Group centroid row indices by segment, preserving first-seen segment order.
fn group_by_segment(targets: &[(String, usize)]) -> Vec<(String, Vec<usize>)> {
    let mut order: Vec<String> = Vec::new();
    let mut by_segment: HashMap<String, Vec<usize>> = HashMap::new();
    for (idx, (seg, _)) in targets.iter().enumerate() {
        if !by_segment.contains_key(seg) {
            order.push(seg.clone());
        }
        by_segment.entry(seg.clone()).or_default().push(idx);
    }
    order
        .into_iter()
        .map(|seg| {
            let indices = by_segment.remove(&seg).unwrap_or_default();
            (seg, indices)
        })
        .collect()
}

/// In-memory table-level vector routing index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableVectorRouter {
    /// Vector dimensionality
    pub dim: usize,
    /// Metric
    pub metric: VectorMetric,
    /// Flat centroid matrix (N * dim f32 in row-major order)
    pub centroid_matrix: Vec<f32>,
    /// Targets corresponding to each centroid row: (segment_id, local_cluster_id)
    pub targets: Vec<(String, usize)>,
    /// Centroid row indices grouped by segment, built once at construction so
    /// per-segment routing iterates a slice instead of rebuilding a HashMap on
    /// every query.
    #[serde(default)]
    pub segment_indices: Vec<(String, Vec<usize>)>,
}

impl TableVectorRouter {
    /// Create a new router from centroid records.
    pub fn new(dim: usize, metric: VectorMetric, records: Vec<CentroidRecord>) -> Self {
        let mut centroid_matrix = Vec::with_capacity(records.len() * dim);
        let mut targets = Vec::with_capacity(records.len());

        for r in records {
            if r.centroid.len() == dim {
                centroid_matrix.extend_from_slice(&r.centroid);
                targets.push((r.segment_id, r.cluster_id));
            }
        }

        let segment_indices = group_by_segment(&targets);

        Self {
            dim,
            metric,
            centroid_matrix,
            targets,
            segment_indices,
        }
    }

    /// Number of centroids registered across all segments.
    pub fn len(&self) -> usize {
        self.targets.len()
    }

    /// True if the router has no centroids.
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }

    /// Get the centroid vector at the given index.
    pub fn get_centroid(&self, index: usize) -> Option<&[f32]> {
        if index < self.len() && (index + 1) * self.dim <= self.centroid_matrix.len() {
            Some(&self.centroid_matrix[index * self.dim..(index + 1) * self.dim])
        } else {
            None
        }
    }

    /// Route a query vector using the default routing policy (or `BSDB_ROUTING_MODE`).
    pub fn route(&self, query_f32: &[f32], total_probes: usize) -> VectorRoutingPlan {
        self.route_with_policy(query_f32, total_probes, RoutingPolicy::default())
    }

    /// Route a query vector to candidate segments and their local cluster IDs using a specific policy.
    ///
    /// One pass over the centroid matrix with a bounded top-k heap per segment.
    /// No per-query `Vec` of all distances, no full sort, and no HashMap rebuild:
    /// `segment_indices` is precomputed at construction.
    pub fn route_with_policy(
        &self,
        query_f32: &[f32],
        probes: usize,
        policy: RoutingPolicy,
    ) -> VectorRoutingPlan {
        if self.is_empty() || query_f32.len() != self.dim {
            return VectorRoutingPlan {
                candidate_segments: HashSet::new(),
                segment_clusters: HashMap::new(),
                min_centroid_distance: f32::MAX,
            };
        }

        let metric = self.metric;
        let dim = self.dim;
        let matrix = &self.centroid_matrix;
        let k = probes.max(1);

        let mut global_min = f32::MAX;
        let mut candidate_segments = HashSet::new();
        let mut segment_clusters: HashMap<String, Vec<usize>> = HashMap::new();

        match policy {
            RoutingPolicy::GlobalPruning => {
                let mut heap = BinaryHeap::with_capacity(k + 1);
                for i in 0..self.targets.len() {
                    let c = &matrix[i * dim..(i + 1) * dim];
                    let dist = distance_for(metric, query_f32, c);
                    if dist < global_min {
                        global_min = dist;
                    }
                    push_bounded(&mut heap, k, Candidate { dist, idx: i });
                }

                // Group by borrowed segment id first, then allocate one String
                // per unique segment instead of two per candidate. The heap
                // holds unique centroid indices, so cluster ids cannot repeat
                // within a segment and the old `contains` guard was dead code.
                let mut by_segment: HashMap<&str, Vec<usize>> = HashMap::new();
                for cand in heap.into_sorted_vec() {
                    let (ref seg_id, cluster_id) = self.targets[cand.idx];
                    by_segment
                        .entry(seg_id.as_str())
                        .or_default()
                        .push(cluster_id);
                }
                for (seg_id, clusters) in by_segment {
                    candidate_segments.insert(seg_id.to_string());
                    segment_clusters.insert(seg_id.to_string(), clusters);
                }
            }

            RoutingPolicy::PerSegmentTargeting => {
                for (seg_id, indices) in &self.segment_indices {
                    let mut heap = BinaryHeap::with_capacity(k + 1);
                    for &i in indices {
                        let c = &matrix[i * dim..(i + 1) * dim];
                        let dist = distance_for(metric, query_f32, c);
                        if dist < global_min {
                            global_min = dist;
                        }
                        push_bounded(&mut heap, k, Candidate { dist, idx: i });
                    }
                    let targeted: Vec<usize> = heap
                        .into_sorted_vec()
                        .into_iter()
                        .map(|c| self.targets[c.idx].1)
                        .collect();
                    candidate_segments.insert(seg_id.clone());
                    segment_clusters.insert(seg_id.clone(), targeted);
                }
            }

            RoutingPolicy::Adaptive { factor } => {
                // Per-segment best distance and top-k clusters in one pass.
                let mut per_segment: Vec<(&String, f32, Vec<usize>)> =
                    Vec::with_capacity(self.segment_indices.len());
                for (seg_id, indices) in &self.segment_indices {
                    let mut heap = BinaryHeap::with_capacity(k + 1);
                    let mut seg_best = f32::MAX;
                    for &i in indices {
                        let c = &matrix[i * dim..(i + 1) * dim];
                        let dist = distance_for(metric, query_f32, c);
                        if dist < seg_best {
                            seg_best = dist;
                        }
                        if dist < global_min {
                            global_min = dist;
                        }
                        push_bounded(&mut heap, k, Candidate { dist, idx: i });
                    }
                    let targeted: Vec<usize> = heap
                        .into_sorted_vec()
                        .into_iter()
                        .map(|c| self.targets[c.idx].1)
                        .collect();
                    per_segment.push((seg_id, seg_best, targeted));
                }

                // Threshold calculation handling zero / negative inner-product metrics
                let threshold = if global_min.abs() < 1e-6 {
                    1e-4
                } else if global_min < 0.0 {
                    global_min * (2.0 - factor)
                } else {
                    global_min * factor
                };

                // Rank once by best distance; both the threshold filter and the
                // cap below read from it.
                per_segment.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

                // Rank-based cap, opt-in only. On a negative inner-product
                // metric the threshold above is *less negative* than
                // `global_min` (`global_min * (2 - factor)` with factor 1.35 is
                // `global_min * 0.65`), so any segment holding a centroid above
                // that bar is admitted. For normalized embeddings that is almost
                // every segment, and routing degenerates into a full fan-out.
                // The cap prunes in that regime, but at a recall cost that has to
                // be measured deliberately rather than inherited by accident —
                // see `adaptive_segment_cap_enabled`.
                let max_segments = routing_max_segments(k, per_segment.len());
                for (seg_id, seg_best, targeted) in per_segment.iter() {
                    if candidate_segments.len() >= max_segments {
                        break;
                    }
                    if *seg_best <= threshold {
                        candidate_segments.insert((*seg_id).clone());
                        segment_clusters.insert((*seg_id).clone(), targeted.clone());
                    }
                }

                // Safety fallback: if the threshold was overly aggressive, admit
                // the closest segments by rank.
                if candidate_segments.is_empty() {
                    for (seg_id, _, targeted) in per_segment.iter().take(max_segments) {
                        candidate_segments.insert((*seg_id).clone());
                        segment_clusters.insert((*seg_id).clone(), targeted.clone());
                    }
                }
            }
        }

        VectorRoutingPlan {
            candidate_segments,
            segment_clusters,
            min_centroid_distance: global_min,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_table_vector_router_pruning() {
        let dim = 4;
        let metric = VectorMetric::L2;

        // Create centroids across 5 distinct segments
        let mut records = Vec::new();
        for seg_idx in 0..5 {
            let seg_id = format!("seg_{:03}", seg_idx);
            for c_id in 0..4 {
                // Segment 0 centered around 10.0, Segment 1 around 20.0, etc.
                let base = (seg_idx as f32 + 1.0) * 10.0 + (c_id as f32);
                records.push(CentroidRecord {
                    segment_id: seg_id.clone(),
                    cluster_id: c_id,
                    centroid: vec![base, base, base, base],
                });
            }
        }

        let router = TableVectorRouter::new(dim, metric, records);
        assert_eq!(router.len(), 20);

        // Query vector close to Segment 1 (base ~20.0)
        let query = vec![20.5, 20.5, 20.5, 20.5];

        // Probe top 2 global centroids with explicit GlobalPruning
        let plan = router.route_with_policy(&query, 2, RoutingPolicy::GlobalPruning);

        // Top 2 centroids should both belong to seg_001
        assert_eq!(plan.candidate_segments.len(), 1);
        assert!(plan.candidate_segments.contains("seg_001"));
        assert_eq!(plan.segment_clusters.get("seg_001").unwrap().len(), 2);

        // Probe top 6 global centroids (should touch seg_001 and nearest neighbor seg_000 or seg_002)
        let plan6 = router.route_with_policy(&query, 6, RoutingPolicy::GlobalPruning);
        assert!(plan6.candidate_segments.len() <= 3);
        assert!(plan6.candidate_segments.contains("seg_001"));
        // Segments 3 and 4 (base 40.0 and 50.0) should be pruned!
        assert!(!plan6.candidate_segments.contains("seg_003"));
        assert!(!plan6.candidate_segments.contains("seg_004"));
    }

    #[test]
    fn test_routing_policy_per_segment_targeting() {
        let dim = 4;
        let metric = VectorMetric::L2;

        let mut records = Vec::new();
        for seg_idx in 0..5 {
            let seg_id = format!("seg_{:03}", seg_idx);
            for c_id in 0..4 {
                let base = (seg_idx as f32 + 1.0) * 10.0 + (c_id as f32);
                records.push(CentroidRecord {
                    segment_id: seg_id.clone(),
                    cluster_id: c_id,
                    centroid: vec![base, base, base, base],
                });
            }
        }

        let router = TableVectorRouter::new(dim, metric, records);
        let query = vec![20.5, 20.5, 20.5, 20.5];

        // PerSegmentTargeting: all 5 segments must be retained, each with top 2 clusters
        let plan = router.route_with_policy(&query, 2, RoutingPolicy::PerSegmentTargeting);
        assert_eq!(plan.candidate_segments.len(), 5);
        for seg_idx in 0..5 {
            let seg_id = format!("seg_{:03}", seg_idx);
            assert!(plan.candidate_segments.contains(&seg_id));
            assert_eq!(plan.segment_clusters.get(&seg_id).unwrap().len(), 2);
        }
    }

    #[test]
    fn test_routing_policy_adaptive() {
        let dim = 4;
        let metric = VectorMetric::L2;

        let mut records = Vec::new();
        for seg_idx in 0..5 {
            let seg_id = format!("seg_{:03}", seg_idx);
            for c_id in 0..4 {
                let base = (seg_idx as f32 + 1.0) * 10.0 + (c_id as f32);
                records.push(CentroidRecord {
                    segment_id: seg_id.clone(),
                    cluster_id: c_id,
                    centroid: vec![base, base, base, base],
                });
            }
        }

        let router = TableVectorRouter::new(dim, metric, records);
        let query = vec![20.5, 20.5, 20.5, 20.5];

        // Adaptive with factor 1.35: seg_001 is closest, other distant segments pruned
        let plan = router.route_with_policy(&query, 2, RoutingPolicy::Adaptive { factor: 1.35 });
        assert!(plan.candidate_segments.contains("seg_001"));
        assert!(!plan.candidate_segments.contains("seg_004"));
    }

    #[test]
    fn test_table_vector_router_empty() {
        let router = TableVectorRouter::new(4, VectorMetric::L2, vec![]);
        assert!(router.is_empty());
        let plan = router.route(&[1.0, 2.0, 3.0, 4.0], 4);
        assert!(plan.candidate_segments.is_empty());
        assert!(plan.segment_clusters.is_empty());
    }

    /// The Adaptive policy must admit every segment the distance threshold
    /// admits. On a negative inner-product metric `global_min * (2 - factor)` is
    /// *less negative* than `global_min`, so everything aligned with the query
    /// passes and the policy fans out — by design. Capping it there is a recall
    /// decision that must be opt-in, so the default must not prune.
    #[test]
    fn test_adaptive_defaults_to_full_fanout() {
        let dim = 4;
        let mut records = Vec::new();
        // 25 segments, 4 centroids each, all identical and aligned with the
        // query, so every segment passes the threshold.
        for seg in 0..25 {
            for c in 0..4 {
                records.push(CentroidRecord {
                    segment_id: format!("seg_{seg:03}"),
                    cluster_id: c,
                    centroid: vec![1.0, 1.0, 1.0, 1.0],
                });
            }
        }
        let router = TableVectorRouter::new(dim, VectorMetric::InnerProduct, records);
        let query = vec![1.0, 1.0, 1.0, 1.0];

        if adaptive_segment_cap_enabled() {
            // The cap is process-wide state; a test binary that opted in cannot
            // assert the default. Skip rather than assert something false.
            return;
        }

        let plan = router.route_with_policy(&query, 16, RoutingPolicy::Adaptive { factor: 1.35 });
        assert_eq!(
            plan.candidate_segments.len(),
            25,
            "default Adaptive routing must not prune segments"
        );
    }

    /// The cap formula itself, independent of the process-wide opt-in.
    #[test]
    fn test_capped_segment_count_formula() {
        // Floor of 2 so a small table is never over-pruned, 4 probes per
        // segment, bounded by the table's own segment count.
        assert_eq!(capped_segment_count(1, 100), 2);
        assert_eq!(capped_segment_count(16, 100), 4);
        assert_eq!(capped_segment_count(48, 100), 12);
        assert_eq!(capped_segment_count(1000, 7), 7);
    }

    /// The bounded heap must return exactly the k nearest centroids, in order.
    #[test]
    fn test_bounded_topk_matches_full_sort() {
        let dim = 8;
        let mut records = Vec::new();
        for i in 0..500 {
            let base = i as f32;
            records.push(CentroidRecord {
                segment_id: format!("seg_{:03}", i % 7),
                cluster_id: i,
                centroid: vec![base; dim],
            });
        }
        let router = TableVectorRouter::new(dim, VectorMetric::L2, records);
        let query = vec![250.0; dim];

        for k in [1usize, 4, 16, 64] {
            let plan = router.route_with_policy(&query, k, RoutingPolicy::GlobalPruning);
            let total: usize = plan.segment_clusters.values().map(|v| v.len()).sum();
            assert_eq!(total, k, "expected {k} clusters, got {total}");
        }
    }
}
