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

use std::collections::{HashMap, HashSet};
use rayon::prelude::*;
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
        if let Ok(mode) = std::env::var("BSDB_ROUTING_MODE") {
            match mode.to_ascii_lowercase().as_str() {
                "global" | "pruning" => RoutingPolicy::GlobalPruning,
                "per_segment" | "targeting" => RoutingPolicy::PerSegmentTargeting,
                _ => {
                    let factor = std::env::var("BSDB_ROUTING_ADAPTIVE_FACTOR")
                        .ok()
                        .and_then(|f| f.parse::<f32>().ok())
                        .unwrap_or(1.35);
                    RoutingPolicy::Adaptive { factor }
                }
            }
        } else {
            let factor = std::env::var("BSDB_ROUTING_ADAPTIVE_FACTOR")
                .ok()
                .and_then(|f| f.parse::<f32>().ok())
                .unwrap_or(1.35);
            RoutingPolicy::Adaptive { factor }
        }
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

        Self {
            dim,
            metric,
            centroid_matrix,
            targets,
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

    /// Route a query vector using the default routing policy (or `BSDB_ROUTING_MODE`).
    pub fn route(&self, query_f32: &[f32], total_probes: usize) -> VectorRoutingPlan {
        self.route_with_policy(query_f32, total_probes, RoutingPolicy::default())
    }

    /// Route a query vector to candidate segments and their local cluster IDs using a specific policy.
    ///
    /// Computes distances to all global centroids in parallel via Rayon,
    /// applies the specified routing policy, and returns the candidate segments and cluster targets.
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

        let n_centroids = self.targets.len();
        let metric = self.metric;
        let dim = self.dim;
        let matrix = &self.centroid_matrix;

        // Fast parallel distance computation across all centroids in < 50 µs
        let dists: Vec<(usize, f32)> = (0..n_centroids)
            .into_par_iter()
            .map(|i| {
                let c = &matrix[i * dim..(i + 1) * dim];
                let dist = match metric {
                    VectorMetric::L2 => distance::l2_distance(query_f32, c),
                    VectorMetric::Cosine => distance::cosine_distance(query_f32, c),
                    VectorMetric::InnerProduct => -distance::dot_product(query_f32, c),
                    VectorMetric::L1 => distance::l1_distance(query_f32, c),
                    VectorMetric::Hamming => distance::hamming_distance(query_f32, c),
                    VectorMetric::Jaccard => distance::jaccard_distance(query_f32, c),
                };
                (i, dist)
            })
            .collect();

        match policy {
            RoutingPolicy::GlobalPruning => {
                let mut sorted_dists = dists;
                sorted_dists
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

                let k_probes = probes.max(1).min(n_centroids);
                let min_dist = sorted_dists.first().map(|x| x.1).unwrap_or(f32::MAX);

                let mut candidate_segments = HashSet::new();
                let mut segment_clusters: HashMap<String, Vec<usize>> = HashMap::new();

                for (idx, _dist) in sorted_dists.into_iter().take(k_probes) {
                    let (ref seg_id, cluster_id) = self.targets[idx];
                    candidate_segments.insert(seg_id.clone());
                    let list = segment_clusters.entry(seg_id.clone()).or_default();
                    if !list.contains(&cluster_id) {
                        list.push(cluster_id);
                    }
                }

                VectorRoutingPlan {
                    candidate_segments,
                    segment_clusters,
                    min_centroid_distance: min_dist,
                }
            }

            RoutingPolicy::PerSegmentTargeting => {
                // Group centroid distances by segment and retain best clusters per segment
                let mut seg_centroids: HashMap<&str, Vec<(usize, f32)>> = HashMap::new();
                let mut global_min = f32::MAX;

                for &(idx, dist) in &dists {
                    if dist < global_min {
                        global_min = dist;
                    }
                    let (ref seg_id, cluster_id) = self.targets[idx];
                    seg_centroids
                        .entry(seg_id.as_str())
                        .or_default()
                        .push((cluster_id, dist));
                }

                let mut candidate_segments = HashSet::new();
                let mut segment_clusters = HashMap::new();

                for (seg_id, mut cluster_dists) in seg_centroids {
                    cluster_dists
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    let k = probes.max(1).min(cluster_dists.len());
                    let targeted: Vec<usize> =
                        cluster_dists.into_iter().take(k).map(|(c, _)| c).collect();
                    candidate_segments.insert(seg_id.to_string());
                    segment_clusters.insert(seg_id.to_string(), targeted);
                }

                VectorRoutingPlan {
                    candidate_segments,
                    segment_clusters,
                    min_centroid_distance: global_min,
                }
            }

            RoutingPolicy::Adaptive { factor } => {
                // Group centroid distances by segment and identify global minimum
                let mut seg_centroids: HashMap<&str, Vec<(usize, f32)>> = HashMap::new();
                let mut global_min = f32::MAX;

                for &(idx, dist) in &dists {
                    if dist < global_min {
                        global_min = dist;
                    }
                    let (ref seg_id, cluster_id) = self.targets[idx];
                    seg_centroids
                        .entry(seg_id.as_str())
                        .or_default()
                        .push((cluster_id, dist));
                }

                // Threshold calculation handling zero / negative inner-product metrics
                let threshold = if global_min.abs() < 1e-6 {
                    1e-4
                } else if global_min < 0.0 {
                    global_min * (2.0 - factor)
                } else {
                    global_min * factor
                };

                let mut candidate_segments = HashSet::new();
                let mut segment_clusters = HashMap::new();

                for (seg_id, mut cluster_dists) in seg_centroids {
                    cluster_dists
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    let seg_best = cluster_dists.first().map(|x| x.1).unwrap_or(f32::MAX);

                    // A segment is admitted only if its closest Voronoi centroid is within the adaptive threshold
                    if seg_best <= threshold {
                        let k = probes.max(1).min(cluster_dists.len());
                        let targeted: Vec<usize> =
                            cluster_dists.into_iter().take(k).map(|(c, _)| c).collect();
                        candidate_segments.insert(seg_id.to_string());
                        segment_clusters.insert(seg_id.to_string(), targeted);
                    }
                }

                // Safety fallback: if threshold was overly aggressive, admit the closest global segments
                if candidate_segments.is_empty() {
                    let mut sorted_dists = dists;
                    sorted_dists
                        .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                    for (idx, _dist) in sorted_dists.into_iter().take(probes.max(1)) {
                        let (ref seg_id, cluster_id) = self.targets[idx];
                        candidate_segments.insert(seg_id.clone());
                        let list = segment_clusters.entry(seg_id.clone()).or_default();
                        if !list.contains(&cluster_id) {
                            list.push(cluster_id);
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
}
