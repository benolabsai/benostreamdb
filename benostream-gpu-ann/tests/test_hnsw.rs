// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

use benostream_gpu_ann::backend::cpu::CpuBackend;
use benostream_gpu_ann::backend::GpuBackend;
use benostream_gpu_ann::{Algorithm, ComputeContext, IndexBuilder, Metric, VectorIndex};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[test]
fn test_hnsw_basic_cpu() {
    let dim = 32;
    let n_vectors = 300;
    let mut rng = ChaCha8Rng::seed_from_u64(101);

    let mut vectors = vec![0.0f32; n_vectors * dim];
    for v in vectors.iter_mut() {
        *v = rng.gen_range(-1.0..1.0);
    }

    let hnsw = IndexBuilder::new(dim, Metric::L2)
        .algorithm(Algorithm::Hnsw {
            m: 16,
            ef_construction: 64,
        })
        .context(ComputeContext::cpu())
        .build_hnsw(&vectors, None)
        .expect("Build HNSW on CPU");

    assert_eq!(hnsw.len(), n_vectors);
    assert_eq!(hnsw.dim(), dim);

    // Query 7th vector: exact match should have distance ~ 0.0 and id == 7
    let query = &vectors[7 * dim..8 * dim];
    let results = hnsw.search(query, 5, 30, None).expect("Search on HNSW");

    assert!(!results.is_empty());
    assert_eq!(results[0].id, 7);
    assert!(results[0].distance.abs() < 1e-5);
}

#[test]
fn test_hnsw_filter_cpu() {
    let dim = 16;
    let n_vectors = 200;
    let mut rng = ChaCha8Rng::seed_from_u64(202);

    let mut vectors = vec![0.0f32; n_vectors * dim];
    for v in vectors.iter_mut() {
        *v = rng.gen();
    }

    let hnsw = IndexBuilder::new(dim, Metric::L2)
        .algorithm(Algorithm::Hnsw {
            m: 12,
            ef_construction: 50,
        })
        .context(ComputeContext::cpu())
        .build_hnsw(&vectors, None)
        .expect("Build HNSW");

    let mut filter = roaring::RoaringBitmap::new();
    filter.insert(10);
    filter.insert(20);
    filter.insert(30);

    let query = &vectors[0..dim];
    let results = hnsw
        .search(query, 5, 40, Some(&filter))
        .expect("Filtered HNSW search");

    for r in &results {
        assert!(filter.contains(r.id as u32));
    }
}

#[test]
fn test_hnsw_gpu_accelerated_recall() {
    let dim = 32;
    let n_vectors = 1_000;
    let mut rng = ChaCha8Rng::seed_from_u64(303);

    let mut vectors = vec![0.0f32; n_vectors * dim];
    for v in vectors.iter_mut() {
        *v = rng.gen_range(-1.0..1.0);
    }

    // Auto-detect fastest GPU accelerator (CUDA / Metal / WGPU)
    let ctx = ComputeContext::auto_detect();
    println!("Building GPU HNSW on active hardware: {}", ctx.name());

    let hnsw = IndexBuilder::new(dim, Metric::L2)
        .algorithm(Algorithm::Hnsw {
            m: 16,
            ef_construction: 100,
        })
        .context(ctx)
        .build_hnsw(&vectors, None)
        .expect("Build GPU-accelerated HNSW");

    let cpu = CpuBackend::new();
    let k = 10;
    let ef_search = 50;
    let mut total_hits = 0;
    let n_queries = 50;

    for q_idx in 0..n_queries {
        let query = &vectors[q_idx * dim..(q_idx + 1) * dim];

        // 1. Exact ground truth linear scan
        let all_dists = cpu
            .compute_distance(query, &vectors, dim, Metric::L2)
            .expect("CPU scan");
        let mut exact: Vec<(usize, f32)> = all_dists.into_iter().enumerate().collect();
        exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let exact_ids: std::collections::HashSet<u64> =
            exact.iter().take(k).map(|(id, _)| *id as u64).collect();

        // 2. GPU HNSW search
        let results = hnsw.search(query, k, ef_search, None).expect("HNSW search");
        let found_ids: std::collections::HashSet<u64> = results.iter().map(|r| r.id).collect();

        let hits = exact_ids.intersection(&found_ids).count();
        total_hits += hits;
    }

    let recall = total_hits as f64 / (n_queries * k) as f64;
    println!(
        "GPU HNSW Recall@{}: {:.2}% on {}",
        k,
        recall * 100.0,
        hnsw.backend_name()
    );
    assert!(
        recall >= 0.95,
        "GPU HNSW Recall@{} was {:.2}%, expected >= 95%",
        k,
        recall * 100.0
    );
}

#[test]
fn test_unified_vector_index_enum() {
    let dim = 16;
    let n_vectors = 100;
    let vectors = vec![0.5f32; n_vectors * dim];

    // Build unified VectorIndex
    let index: VectorIndex = IndexBuilder::new(dim, Metric::Cosine)
        .algorithm(Algorithm::Hnsw {
            m: 8,
            ef_construction: 32,
        })
        .build(&vectors, None)
        .expect("Build through VectorIndex");

    assert_eq!(index.len(), n_vectors);
    assert_eq!(index.dim(), dim);

    let query = vec![0.5f32; dim];
    let results = index.search(&query, 5, 20, None).expect("Unified search");
    assert!(!results.is_empty());
}
