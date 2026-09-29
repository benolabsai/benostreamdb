// Copyright (c) 2026 BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

use benostream_gpu_ann::backend::cpu::CpuBackend;
use benostream_gpu_ann::backend::GpuBackend;
use benostream_gpu_ann::{Algorithm, ComputeContext, IndexBuilder, Metric, VectorIndex};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[test]
fn test_cagra_basic_cpu() {
    let dim = 32;
    let n_vectors = 200;
    let mut rng = ChaCha8Rng::seed_from_u64(404);

    let mut vectors = vec![0.0f32; n_vectors * dim];
    for v in vectors.iter_mut() {
        *v = rng.gen_range(-1.0..1.0);
    }

    let cagra = IndexBuilder::new(dim, Metric::L2)
        .algorithm(Algorithm::Cagra {
            graph_degree: 16,
            intermediate_degree: 32,
        })
        .context(ComputeContext::cpu())
        .build_cagra(&vectors, None)
        .expect("Build CAGRA on CPU");

    assert_eq!(cagra.len(), n_vectors);
    assert_eq!(cagra.dim(), dim);
    assert_eq!(cagra.graph_degree(), 16);

    // Query 15th vector: exact match should have distance ~ 0.0 and id == 15
    let query = &vectors[15 * dim..16 * dim];
    let results = cagra.search(query, 5, 24, None).expect("CAGRA search");

    assert!(!results.is_empty());
    assert_eq!(results[0].id, 15);
    assert!(results[0].distance.abs() < 1e-4);
}

#[test]
fn test_cagra_filter() {
    let dim = 16;
    let n_vectors = 150;
    let mut rng = ChaCha8Rng::seed_from_u64(505);

    let mut vectors = vec![0.0f32; n_vectors * dim];
    for v in vectors.iter_mut() {
        *v = rng.gen();
    }

    let cagra = IndexBuilder::new(dim, Metric::L2)
        .algorithm(Algorithm::Cagra {
            graph_degree: 12,
            intermediate_degree: 24,
        })
        .context(ComputeContext::cpu())
        .build_cagra(&vectors, None)
        .expect("Build CAGRA");

    let mut filter = roaring::RoaringBitmap::new();
    filter.insert(5);
    filter.insert(15);
    filter.insert(25);

    let query = &vectors[0..dim];
    let results = cagra
        .search(query, 5, 32, Some(&filter))
        .expect("Filtered CAGRA search");

    for r in &results {
        assert!(filter.contains(r.id as u32));
    }
}

#[test]
fn test_cagra_gpu_accelerated_recall() {
    let dim = 32;
    let n_vectors = 800;
    let mut rng = ChaCha8Rng::seed_from_u64(606);

    let mut vectors = vec![0.0f32; n_vectors * dim];
    for v in vectors.iter_mut() {
        *v = rng.gen_range(-1.0..1.0);
    }

    // Auto-detect fastest GPU accelerator (CUDA / Metal / WGPU)
    let ctx = ComputeContext::auto_detect();
    println!("Building Stage 3 CAGRA on active hardware: {}", ctx.name());

    let cagra = IndexBuilder::new(dim, Metric::L2)
        .algorithm(Algorithm::Cagra {
            graph_degree: 32,
            intermediate_degree: 64,
        })
        .context(ctx)
        .build_cagra(&vectors, None)
        .expect("Build GPU CAGRA");

    assert_eq!(cagra.graph_degree(), 32);

    let cpu = CpuBackend::new();
    let k = 10;
    let search_width = 48;
    let mut total_hits = 0;
    let n_queries = 40;

    for q_idx in 0..n_queries {
        let query = &vectors[q_idx * dim..(q_idx + 1) * dim];

        // 1. Ground truth exact scan
        let all_dists = cpu.compute_distance(query, &vectors, dim, Metric::L2).expect("CPU scan");
        let mut exact: Vec<(usize, f32)> = all_dists.into_iter().enumerate().collect();
        exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let exact_ids: std::collections::HashSet<u64> =
            exact.iter().take(k).map(|(id, _)| *id as u64).collect();

        // 2. GPU CAGRA search
        let results = cagra.search(query, k, search_width, None).expect("CAGRA search");
        let found_ids: std::collections::HashSet<u64> = results.iter().map(|r| r.id).collect();

        let hits = exact_ids.intersection(&found_ids).count();
        total_hits += hits;
    }

    let recall = total_hits as f64 / (n_queries * k) as f64;
    println!(
        "GPU CAGRA Recall@{}: {:.2}% on {}",
        k,
        recall * 100.0,
        cagra.backend_name()
    );
    assert!(
        recall >= 0.90,
        "GPU CAGRA Recall@{} was {:.2}%, expected >= 90%",
        k,
        recall * 100.0
    );
}

#[test]
fn test_cagra_unified_vector_index() {
    let dim = 16;
    let n_vectors = 100;
    let vectors = vec![0.3f32; n_vectors * dim];

    let index: VectorIndex = IndexBuilder::new(dim, Metric::Cosine)
        .algorithm(Algorithm::Cagra {
            graph_degree: 8,
            intermediate_degree: 16,
        })
        .build(&vectors, None)
        .expect("Build CAGRA through VectorIndex");

    assert_eq!(index.len(), n_vectors);
    assert_eq!(index.dim(), dim);

    let query = vec![0.3f32; dim];
    let results = index.search(&query, 5, 16, None).expect("Search on VectorIndex CAGRA");
    assert!(!results.is_empty());
}
