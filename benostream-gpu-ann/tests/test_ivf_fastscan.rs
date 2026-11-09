// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

use benostream_gpu_ann::{Algorithm, ComputeContext, IndexBuilder, Metric};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[test]
fn test_ivf_fastscan_basic_cpu() {
    let dim = 32;
    let n_vectors = 600;
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut vectors = vec![0.0f32; n_vectors * dim];
    for v in vectors.iter_mut() {
        *v = rng.gen_range(-1.0..1.0);
    }

    let index = IndexBuilder::new(dim, Metric::L2)
        .algorithm(Algorithm::IvfFastScan {
            n_lists: Some(16),
            m: Some(8),
        })
        .context(ComputeContext::cpu())
        .build(&vectors, None)
        .expect("Build should succeed");

    assert_eq!(index.len(), n_vectors);
    assert_eq!(index.dim(), dim);

    // Query with the 10th vector: 4-bit FastScan should retrieve vector 10 in top-5
    let query = &vectors[10 * dim..11 * dim];
    let results = index
        .search(query, 5, 16, None)
        .expect("Search should succeed");

    assert!(!results.is_empty());
    let ids: Vec<u64> = results.iter().map(|r| r.id).collect();
    assert!(
        ids.contains(&10),
        "Top-5 results must contain the query vector itself, got {:?}",
        ids
    );
}

#[test]
fn test_ivf_fastscan_filter_cpu() {
    let dim = 16;
    let n_vectors = 300;
    let mut rng = ChaCha8Rng::seed_from_u64(123);

    let mut vectors = vec![0.0f32; n_vectors * dim];
    for v in vectors.iter_mut() {
        *v = rng.gen();
    }

    let index = IndexBuilder::new(dim, Metric::L2)
        .algorithm(Algorithm::IvfFastScan {
            n_lists: Some(8),
            m: Some(4),
        })
        .context(ComputeContext::cpu())
        .build(&vectors, None)
        .expect("Build should succeed");

    // Filter allowing only IDs in {15, 16, 17}
    let mut filter = roaring::RoaringBitmap::new();
    filter.insert(15);
    filter.insert(16);
    filter.insert(17);

    let query = &vectors[15 * dim..16 * dim];
    let results = index
        .search(query, 5, 8, Some(&filter))
        .expect("Filtered search should succeed");

    assert!(!results.is_empty());
    for r in &results {
        assert!(filter.contains(r.id as u32));
    }
}
