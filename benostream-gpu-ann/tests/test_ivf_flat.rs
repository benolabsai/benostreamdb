// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

use benostream_gpu_ann::{ComputeContext, IndexBuilder, Metric};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[test]
fn test_ivf_flat_basic_cpu() {
    let dim = 32;
    let n_vectors = 500;
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut vectors = vec![0.0f32; n_vectors * dim];
    for v in vectors.iter_mut() {
        *v = rng.gen_range(-1.0..1.0);
    }

    let index = IndexBuilder::new(dim, Metric::L2)
        .n_lists(16)
        .context(ComputeContext::cpu())
        .build(&vectors, None)
        .expect("Build should succeed");

    assert_eq!(index.len(), n_vectors);
    assert_eq!(index.dim(), dim);

    // Query with the 10th vector: exact match should have distance ~ 0.0 and id == 10
    let query = &vectors[10 * dim..11 * dim];
    let results = index
        .search(query, 5, 16, None)
        .expect("Search should succeed");

    assert!(!results.is_empty());
    assert_eq!(results[0].id, 10);
    assert!(results[0].distance.abs() < 1e-5);
}

#[test]
fn test_ivf_flat_filter_cpu() {
    let dim = 16;
    let n_vectors = 200;
    let mut rng = ChaCha8Rng::seed_from_u64(123);

    let mut vectors = vec![0.0f32; n_vectors * dim];
    for v in vectors.iter_mut() {
        *v = rng.gen();
    }

    let index = IndexBuilder::new(dim, Metric::L2)
        .n_lists(8)
        .context(ComputeContext::cpu())
        .build(&vectors, None)
        .expect("Build should succeed");

    // Create a filter allowing only IDs in {5, 6, 7}
    let mut filter = roaring::RoaringBitmap::new();
    filter.insert(5);
    filter.insert(6);
    filter.insert(7);

    let query = &vectors[0..dim];
    let results = index
        .search(query, 5, 8, Some(&filter))
        .expect("Filtered search should succeed");

    for r in &results {
        assert!(filter.contains(r.id as u32));
    }
}

#[test]
fn test_ivf_flat_auto_detect_hardware() {
    let dim = 64;
    let n_vectors = 1000;
    let mut rng = ChaCha8Rng::seed_from_u64(999);

    let mut vectors = vec![0.0f32; n_vectors * dim];
    for v in vectors.iter_mut() {
        *v = rng.gen_range(-1.0..1.0);
    }

    // Auto-detect fastest accelerator (CUDA / Metal / WGPU / CPU)
    let ctx = ComputeContext::auto_detect();
    println!("Testing IVF-Flat on active backend: {}", ctx.name());

    let index = IndexBuilder::new(dim, Metric::Cosine)
        .n_lists(32)
        .context(ctx)
        .build(&vectors, None)
        .expect("Build on hardware should succeed");

    let query = &vectors[42 * dim..43 * dim];
    let results = index
        .search(query, 5, 32, None)
        .expect("Search should succeed");

    assert!(!results.is_empty());
    assert_eq!(results[0].id, 42);
    assert!(results[0].distance.abs() < 1e-4);
}
