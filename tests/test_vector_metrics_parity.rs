// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use hyperstreamdb::core::index::hnsw_ivf::HnswIvfIndex;
use hyperstreamdb::core::index::{VectorMetric, VectorValue};
use hyperstreamdb::core::manifest::IndexAlgorithm;
use std::collections::HashSet;

fn normalize(vec: &mut [f32]) {
    let norm = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

fn compute_brute_force(
    query: &[f32],
    vectors: &[Vec<f32>],
    metric: VectorMetric,
    k: usize,
) -> Vec<(usize, f32)> {
    let mut dists: Vec<(usize, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d = match metric {
                VectorMetric::L2 => hyperstreamdb::core::index::distance::l2_distance(query, v),
                VectorMetric::Cosine => {
                    hyperstreamdb::core::index::distance::cosine_distance(query, v)
                }
                VectorMetric::InnerProduct => {
                    (1.0 - hyperstreamdb::core::index::distance::dot_product(query, v)).max(0.0)
                }
                VectorMetric::L1 => hyperstreamdb::core::index::distance::l1_distance(query, v),
                VectorMetric::Hamming => {
                    hyperstreamdb::core::index::distance::hamming_distance(query, v)
                }
                VectorMetric::Jaccard => {
                    hyperstreamdb::core::index::distance::jaccard_distance(query, v)
                }
            };
            (i, d)
        })
        .collect();

    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.truncate(k);
    dists
}

#[test]
fn test_all_vector_metrics_against_brute_force() -> Result<()> {
    let dim = 16;
    let n_vectors = 200;
    let k = 10;
    let n_lists = 4;

    // Generate reproducible synthetic dataset
    let mut raw_vectors = Vec::with_capacity(n_vectors);
    for i in 0..n_vectors {
        let mut vec = Vec::with_capacity(dim);
        for d in 0..dim {
            let val = (((i * 31 + d * 17 + 7) % 1000) as f32) / 1000.0;
            vec.push(val);
        }
        raw_vectors.push(vec);
    }

    let raw_query: Vec<f32> = (0..dim)
        .map(|d| (((d * 23 + 13) % 1000) as f32) / 1000.0)
        .collect();

    let metrics = [
        VectorMetric::L2,
        VectorMetric::Cosine,
        VectorMetric::InnerProduct,
        VectorMetric::L1,
        VectorMetric::Hamming,
        VectorMetric::Jaccard,
    ];

    for metric in metrics {
        let mut vectors = raw_vectors.clone();
        let mut query = raw_query.clone();

        if matches!(metric, VectorMetric::Cosine | VectorMetric::InnerProduct) {
            // Cosine and InnerProduct assume unit-normalized vectors
            for v in &mut vectors {
                normalize(v);
            }
            normalize(&mut query);
        } else if matches!(metric, VectorMetric::Hamming | VectorMetric::Jaccard) {
            // Hamming and Jaccard operate on binary indicator values (0.0 or 1.0)
            for v in &mut vectors {
                for val in v.iter_mut() {
                    *val = if *val > 0.5 { 1.0 } else { 0.0 };
                }
            }
            for val in query.iter_mut() {
                *val = if *val > 0.5 { 1.0 } else { 0.0 };
            }
        }

        let algo = IndexAlgorithm::Hnsw {
            metric: metric.to_string(),
            complexity: 16,
            quality: 64,
            build_device: None,
            search_device: None,
        };

        let index = HnswIvfIndex::build(vectors.clone(), metric, Some(n_lists), Some(16), &algo)?;

        // Search with n_probe = n_lists to check fine HNSW search accuracy against exact ground truth
        let results = index.search(&VectorValue::Float32(query.clone()), k, n_lists, None)?;

        let brute_force = compute_brute_force(&query, &vectors, metric, k);
        assert!(
            !brute_force.is_empty(),
            "Brute force returned empty results"
        );
        let max_brute_dist = brute_force.last().unwrap().1;

        // Check distance optimality: all returned points should be as good as the k-th nearest neighbor
        let mut valid_neighbors = 0;
        for (_, dist) in &results {
            if *dist <= max_brute_dist + 1e-5 {
                valid_neighbors += 1;
            }
        }
        let accuracy = valid_neighbors as f32 / k as f32;

        let brute_set: HashSet<usize> = brute_force.iter().map(|(id, _)| *id).collect();
        let indexed_set: HashSet<usize> = results.iter().map(|(id, _)| *id).collect();
        let overlap = brute_set.intersection(&indexed_set).count();

        println!(
            "Metric: {:?} -> Accuracy@{}: {:.2}% (overlap: {}/{}, max_dist: {:.4})",
            metric,
            k,
            accuracy * 100.0,
            overlap,
            k,
            max_brute_dist
        );

        // High accuracy expected (>= 80% nearest neighbors within top-k distance bound)
        assert!(
            accuracy >= 0.8,
            "Metric {:?} achieved accuracy {:.2}%, expected >= 80%",
            metric,
            accuracy * 100.0
        );

        // Ensure returned results are sorted in ascending order of distance
        for i in 1..results.len() {
            assert!(
                results[i - 1].1 <= results[i].1 + 1e-6,
                "Results for {:?} not sorted by distance: {} > {}",
                metric,
                results[i - 1].1,
                results[i].1
            );
        }
    }

    Ok(())
}
