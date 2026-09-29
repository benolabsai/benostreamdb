// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

//! Long-running soak test for benostream-gpu-ann.
//!
//! Ignored by default so `cargo test` stays fast. Run explicitly with:
//!
//! ```bash
//! cargo test -p benostream-gpu-ann --features all --test test_soak -- --ignored --nocapture
//! ```

use benostream_gpu_ann::{Algorithm, ComputeContext, IndexBuilder, Metric};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::time::{Duration, Instant};

#[test]
#[ignore]
fn test_long_running_ann_soak() {
    let soak_seconds = std::env::var("ANN_SOAK_SECONDS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(30);

    println!("Starting benostream-gpu-ann soak test for {} seconds...", soak_seconds);

    let start = Instant::now();
    let duration = Duration::from_secs(soak_seconds);
    let mut rng = ChaCha8Rng::seed_from_u64(0xCAFE);

    let dim = 64;
    let n_vectors = 300;
    let ctx = ComputeContext::auto_detect();
    println!("Running soak test against accelerator: {}", ctx.name());

    let mut iterations = 0;
    let mut total_searches = 0;

    while start.elapsed() < duration {
        let metric = match iterations % 3 {
            0 => Metric::L2,
            1 => Metric::Cosine,
            _ => Metric::InnerProduct,
        };

        let mut vectors = vec![0.0f32; n_vectors * dim];
        for v in vectors.iter_mut() {
            *v = rng.gen_range(-1.0..1.0);
        }

        if metric == Metric::InnerProduct {
            for i in 0..n_vectors {
                let slice = &mut vectors[i * dim..(i + 1) * dim];
                let norm: f32 = slice.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-6 {
                    for x in slice.iter_mut() {
                        *x /= norm;
                    }
                }
            }
        }

        let algo = match iterations % 3 {
            0 => Algorithm::IvfFlat { n_lists: Some(8) },
            1 => Algorithm::Hnsw { m: 8, ef_construction: 24 },
            _ => Algorithm::Cagra { graph_degree: 8, intermediate_degree: 16 },
        };

        // Build index under load
        let index = IndexBuilder::new(dim, metric)
            .algorithm(algo)
            .context(ctx.clone())
            .build(&vectors, None)
            .expect("Soak build must succeed without leak or panic");

        assert_eq!(index.len(), n_vectors);

        // Run batch queries
        for q_idx in 0..10 {
            let query = &vectors[q_idx * dim..(q_idx + 1) * dim];
            let results = index.search(query, 5, 20, None)
                .expect("Soak query must succeed");
            assert_eq!(results.len(), 5);
            assert!(results[0].distance.is_finite());
            assert!(results[0].distance <= results[1].distance + 1e-4);
            total_searches += 1;
        }

        iterations += 1;
        if iterations % 20 == 0 {
            println!(
                "Soak progress: {} iterations, {} searches completed ({:.1}s elapsed)",
                iterations,
                total_searches,
                start.elapsed().as_secs_f64()
            );
        }
    }

    println!(
        "Soak test completed successfully: {} iterations, {} total searches in {:.2}s",
        iterations,
        total_searches,
        start.elapsed().as_secs_f64()
    );
}
