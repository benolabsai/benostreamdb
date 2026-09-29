// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

use benostream_gpu_ann::{Algorithm, ComputeContext, IndexBuilder, Metric, VectorIndex};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn create_test_vectors(n: usize, dim: usize, metric: Metric, seed: u64) -> Vec<f32> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let binary = matches!(metric, Metric::Hamming | Metric::Jaccard);
    let mut v = Vec::with_capacity(n * dim);
    for _ in 0..n {
        let mut row: Vec<f32> = (0..dim)
            .map(|_| {
                if binary {
                    if rng.gen_bool(0.5) {
                        1.0
                    } else {
                        0.0
                    }
                } else {
                    rng.gen_range(-1.0..1.0)
                }
            })
            .collect();

        // Inner product embeddings are unit-normalized in practice
        if metric == Metric::InnerProduct {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for x in row.iter_mut() {
                    *x /= norm;
                }
            }
        }
        v.extend(row);
    }
    v
}

#[test]
fn test_all_algorithms_across_all_metrics_cpu_and_gpu() {
    let dim = 32;
    let n_vectors = 300;

    let algorithms = [
        ("IVF-Flat", Algorithm::IvfFlat { n_lists: Some(16) }),
        (
            "HNSW",
            Algorithm::Hnsw {
                m: 12,
                ef_construction: 50,
            },
        ),
        (
            "CAGRA",
            Algorithm::Cagra {
                graph_degree: 16,
                intermediate_degree: 32,
            },
        ),
    ];

    let metrics = [
        Metric::L2,
        Metric::Cosine,
        Metric::InnerProduct,
        Metric::L1,
        Metric::Hamming,
        Metric::Jaccard,
    ];

    let gpu_ctx = ComputeContext::auto_detect();
    let cpu_ctx = ComputeContext::cpu();

    for (algo_name, algo) in &algorithms {
        for metric in metrics {
            let vectors = create_test_vectors(n_vectors, dim, metric, 777);

            // 1. Build CPU index
            let cpu_index: VectorIndex = IndexBuilder::new(dim, metric)
                .algorithm(algo.clone())
                .context(cpu_ctx.clone())
                .build(&vectors, None)
                .unwrap_or_else(|e| {
                    panic!("CPU build failed for {} {:?}: {:?}", algo_name, metric, e)
                });

            // 2. Build GPU index
            let gpu_index: VectorIndex = IndexBuilder::new(dim, metric)
                .algorithm(algo.clone())
                .context(gpu_ctx.clone())
                .build(&vectors, None)
                .unwrap_or_else(|e| {
                    panic!("GPU build failed for {} {:?}: {:?}", algo_name, metric, e)
                });

            assert_eq!(cpu_index.len(), n_vectors);
            assert_eq!(gpu_index.len(), n_vectors);

            // 3. Query vector at index 5
            let query = &vectors[5 * dim..6 * dim];
            let k = 5;
            let beam = 30;

            let cpu_res = cpu_index.search(query, k, beam, None).expect("CPU search");
            let gpu_res = gpu_index.search(query, k, beam, None).expect("GPU search");

            assert!(
                !cpu_res.is_empty(),
                "CPU search returned empty for {} {:?}",
                algo_name,
                metric
            );
            assert!(
                !gpu_res.is_empty(),
                "GPU search returned empty for {} {:?}",
                algo_name,
                metric
            );

            // In all algorithms, the nearest neighbor to a point already in the index should be itself
            assert_eq!(
                cpu_res[0].id, 5,
                "CPU {} {:?} top hit should be query id 5, got {}",
                algo_name, metric, cpu_res[0].id
            );
            assert_eq!(
                gpu_res[0].id, 5,
                "GPU {} {:?} top hit should be query id 5, got {}",
                algo_name, metric, gpu_res[0].id
            );

            // Check distance parity for the self-match
            let tol = 1e-4;
            assert!(
                (cpu_res[0].distance - gpu_res[0].distance).abs() <= tol,
                "Distance mismatch for {} {:?}: cpu={} gpu={}",
                algo_name,
                metric,
                cpu_res[0].distance,
                gpu_res[0].distance
            );

            // Verify monotonic ascending distance order
            for i in 1..cpu_res.len() {
                assert!(
                    cpu_res[i].distance >= cpu_res[i - 1].distance - 1e-5,
                    "CPU distances not sorted for {} {:?}",
                    algo_name,
                    metric
                );
            }
            for i in 1..gpu_res.len() {
                assert!(
                    gpu_res[i].distance >= gpu_res[i - 1].distance - 1e-5,
                    "GPU distances not sorted for {} {:?}",
                    algo_name,
                    metric
                );
            }
        }
    }
}
