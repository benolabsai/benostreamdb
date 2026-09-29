// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

use benostream_gpu_ann::{Algorithm, ComputeContext, IndexBuilder, Metric};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Arc;

#[test]
fn test_empty_vectors_error_handling() {
    let empty: Vec<f32> = Vec::new();
    let res = IndexBuilder::new(16, Metric::L2).build(&empty, None);
    assert!(res.is_err(), "Building from empty vectors must error gracefully");
}

#[test]
fn test_dimension_mismatch_error_handling() {
    let dim = 16;
    let vectors = vec![0.1f32; 10 * dim];
    let index = IndexBuilder::new(dim, Metric::L2)
        .build(&vectors, None)
        .expect("Build should succeed");

    let bad_query = vec![0.1f32; dim + 4]; // Wrong dimension
    let search_res = index.search(&bad_query, 5, 10, None);
    assert!(search_res.is_err(), "Search with mismatched dimension must error");
}

#[test]
fn test_single_vector_index() {
    let dim = 32;
    let single_vector = vec![0.42f32; dim];

    for algo in [
        Algorithm::IvfFlat { n_lists: None },
        Algorithm::Hnsw { m: 4, ef_construction: 16 },
        Algorithm::Cagra { graph_degree: 4, intermediate_degree: 8 },
    ] {
        let index = IndexBuilder::new(dim, Metric::L2)
            .algorithm(algo)
            .build(&single_vector, None)
            .expect("Building index with 1 vector should succeed");

        assert_eq!(index.len(), 1);
        let results = index.search(&single_vector, 5, 10, None).expect("Search single vector");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 0);
        assert!(results[0].distance.abs() < 1e-5);
    }
}

#[test]
fn test_duplicate_identical_vectors() {
    let dim = 16;
    let n = 50;
    // All 50 vectors are identical
    let vectors = vec![1.23f32; n * dim];

    for algo in [
        Algorithm::IvfFlat { n_lists: Some(4) },
        Algorithm::Hnsw { m: 8, ef_construction: 24 },
        Algorithm::Cagra { graph_degree: 8, intermediate_degree: 16 },
    ] {
        let index = IndexBuilder::new(dim, Metric::L2)
            .algorithm(algo)
            .build(&vectors, None)
            .expect("Building with duplicate vectors should not divide by zero");

        let query = vec![1.23f32; dim];
        let results = index.search(&query, 10, 20, None).expect("Search duplicates");
        assert!(!results.is_empty());
        for r in &results {
            assert!(r.distance.abs() < 1e-4);
        }
    }
}

#[test]
fn test_high_dimension_transformer_embeddings() {
    for &dim in &[384, 768] {
        let n_vectors = 150;
        let mut rng = ChaCha8Rng::seed_from_u64(12345);
        let mut vectors = vec![0.0f32; n_vectors * dim];
        for v in vectors.iter_mut() {
            *v = rng.gen_range(-1.0..1.0);
        }

        // Build on active GPU hardware
        let ctx = ComputeContext::auto_detect();
        println!("Testing dim={} embeddings on {}", dim, ctx.name());

        for algo in [
            Algorithm::IvfFlat { n_lists: Some(8) },
            Algorithm::Hnsw { m: 16, ef_construction: 64 },
            Algorithm::Cagra { graph_degree: 16, intermediate_degree: 32 },
        ] {
            let index = IndexBuilder::new(dim, Metric::Cosine)
                .algorithm(algo)
                .context(ctx.clone())
                .build(&vectors, None)
                .unwrap_or_else(|e| panic!("Build failed for dim={}: {:?}", dim, e));

            let query = &vectors[12 * dim..13 * dim];
            let results = index.search(query, 5, 20, None).expect("Search high-dim");
            assert!(!results.is_empty());
            assert_eq!(results[0].id, 12);
            assert!(results[0].distance.abs() < 1e-4);
        }
    }
}

#[test]
fn test_k_greater_than_n() {
    let dim = 16;
    let n_vectors = 15;
    let vectors = vec![0.5f32; n_vectors * dim];

    let index = IndexBuilder::new(dim, Metric::L2)
        .algorithm(Algorithm::Hnsw { m: 8, ef_construction: 20 })
        .build(&vectors, None)
        .expect("Build index");

    // Request 50 neighbors when index only has 15
    let query = vec![0.5f32; dim];
    let results = index.search(&query, 50, 60, None).expect("Search with k > n");
    assert_eq!(results.len(), 15, "Must return at most n vectors without panicking");
}

#[test]
fn test_bitset_filter_boundaries() {
    let dim = 16;
    let n_vectors = 100;
    let vectors = vec![0.1f32; n_vectors * dim];

    let index = IndexBuilder::new(dim, Metric::L2)
        .algorithm(Algorithm::Hnsw { m: 8, ef_construction: 32 })
        .build(&vectors, None)
        .expect("Build index");

    let query = vec![0.1f32; dim];

    // Case 1: Empty filter (0 matches)
    let empty_filter = roaring::RoaringBitmap::new();
    let res = index.search(&query, 5, 20, Some(&empty_filter)).expect("Search empty filter");
    assert!(res.is_empty(), "Must return 0 results when filter has no matches");

    // Case 2: Exact 1 match
    let mut single_filter = roaring::RoaringBitmap::new();
    single_filter.insert(42);
    let res = index.search(&query, 5, 20, Some(&single_filter)).expect("Search single filter");
    assert_eq!(res.len(), 1);
    assert_eq!(res[0].id, 42);

    // Case 3: All matches
    let mut all_filter = roaring::RoaringBitmap::new();
    for i in 0..100 {
        all_filter.insert(i);
    }
    let res = index.search(&query, 5, 20, Some(&all_filter)).expect("Search all filter");
    assert_eq!(res.len(), 5);
}

#[test]
fn test_multithreaded_concurrent_search() {
    use std::thread;

    let dim = 32;
    let n_vectors = 500;
    let mut rng = ChaCha8Rng::seed_from_u64(999);
    let mut vectors = vec![0.0f32; n_vectors * dim];
    for v in vectors.iter_mut() {
        *v = rng.gen_range(-1.0..1.0);
    }

    let index = Arc::new(
        IndexBuilder::new(dim, Metric::L2)
            .algorithm(Algorithm::Cagra { graph_degree: 16, intermediate_degree: 32 })
            .build(&vectors, None)
            .expect("Build index")
    );

    let mut handles = Vec::new();

    // Spawn 8 threads querying concurrently
    for thread_id in 0..8 {
        let index_clone = Arc::clone(&index);
        let q_vec = vectors[thread_id * dim..(thread_id + 1) * dim].to_vec();

        let handle = thread::spawn(move || {
            for _ in 0..10 {
                let results = index_clone.search(&q_vec, 5, 20, None).expect("Concurrent search");
                assert!(!results.is_empty());
                assert_eq!(results[0].id, thread_id as u64);
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().expect("Thread should not panic");
    }
}
