// Copyright (c) 2026 BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

//! Fuzz & property-style test to guarantee no panics across untrusted / adversarial inputs.

use benostream_gpu_ann::{Algorithm, ComputeContext, IndexBuilder, Metric};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::panic::catch_unwind;

#[test]
fn test_fuzz_random_adversarial_inputs_no_panic() {
    let mut rng = ChaCha8Rng::seed_from_u64(0xDEADBEEF);
    let cpu_ctx = ComputeContext::cpu();

    let metrics = [
        Metric::L2,
        Metric::Cosine,
        Metric::InnerProduct,
        Metric::L1,
        Metric::Hamming,
        Metric::Jaccard,
    ];

    for iteration in 0..300 {
        let dim = rng.gen_range(1..=64);
        let n = rng.gen_range(0..=40);
        let metric = metrics[rng.gen_range(0..metrics.len())];

        let mut vectors: Vec<f32> = Vec::with_capacity(n * dim);
        for _ in 0..(n * dim) {
            let choice = rng.gen_range(0..10);
            let val = match choice {
                0 => f32::NAN,
                1 => f32::INFINITY,
                2 => f32::NEG_INFINITY,
                3 => 0.0,
                4 => -0.0,
                5 => 1e-15,
                6 => 1e15,
                _ => rng.gen_range(-10.0..10.0),
            };
            vectors.push(val);
        }

        let algo_choice = rng.gen_range(0..3);
        let algo = match algo_choice {
            0 => Algorithm::IvfFlat {
                n_lists: if rng.gen_bool(0.5) { Some(rng.gen_range(1..=8)) } else { None },
            },
            1 => Algorithm::Hnsw {
                m: rng.gen_range(2..=16),
                ef_construction: rng.gen_range(4..=32),
            },
            _ => Algorithm::Cagra {
                graph_degree: rng.gen_range(2..=16),
                intermediate_degree: rng.gen_range(4..=32),
            },
        };

        // Ensure build never panics under adversarial floats or zero lengths
        let build_result = catch_unwind(std::panic::AssertUnwindSafe(|| {
            IndexBuilder::new(dim, metric)
                .algorithm(algo)
                .context(cpu_ctx.clone())
                .build(&vectors, None)
        }));

        assert!(
            build_result.is_ok(),
            "Panic caught during index build at iteration {} with n={}, dim={}, metric={:?}",
            iteration, n, dim, metric
        );

        if let Ok(Ok(index)) = build_result {
            // Generate adversarial query
            let query_dim = if rng.gen_bool(0.1) {
                // Dimension mismatch test
                rng.gen_range(1..=dim + 10)
            } else {
                dim
            };

            let query: Vec<f32> = (0..query_dim).map(|_| rng.gen_range(-2.0..2.0)).collect();
            let k = rng.gen_range(0..=n + 10);
            let search_width = rng.gen_range(0..=100);

            // Generate randomized filter
            let filter = if rng.gen_bool(0.5) {
                let mut bm = roaring::RoaringBitmap::new();
                for _ in 0..rng.gen_range(0..=n + 5) {
                    bm.insert(rng.gen_range(0..=n as u32 + 10));
                }
                Some(bm)
            } else {
                None
            };

            // Ensure search never panics
            let search_result = catch_unwind(std::panic::AssertUnwindSafe(|| {
                index.search(&query, k, search_width, filter.as_ref())
            }));

            assert!(
                search_result.is_ok(),
                "Panic caught during search at iteration {} with k={}, search_width={}",
                iteration, k, search_width
            );
        }
    }
}
