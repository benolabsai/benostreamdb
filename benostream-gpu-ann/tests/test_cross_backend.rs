// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

use benostream_gpu_ann::backend::cpu::CpuBackend;
use benostream_gpu_ann::backend::GpuBackend;
use benostream_gpu_ann::metric::Metric;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

fn generate_random_vectors(n: usize, dim: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let vectors: Vec<f32> = (0..n * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    (query, vectors)
}

fn get_active_backends() -> Vec<Box<dyn GpuBackend>> {
    let mut backends: Vec<Box<dyn GpuBackend>> = vec![Box::new(CpuBackend::new())];

    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    {
        if let Ok(b) = benostream_gpu_ann::backend::cuda::CudaBackend::new(0) {
            backends.push(Box::new(b));
        }
    }

    #[cfg(target_os = "macos")]
    {
        if let Ok(b) = benostream_gpu_ann::backend::metal::MetalBackend::new() {
            backends.push(Box::new(b));
        }
    }

    #[cfg(feature = "wgpu")]
    {
        if let Ok(b) = benostream_gpu_ann::backend::wgpu::WgpuBackend::new("WGPU_Test", None) {
            backends.push(Box::new(b));
        }
    }

    backends
}

#[test]
fn test_gpu_matches_cpu_all_metrics() {
    let backends = get_active_backends();
    println!("Active backends for differential testing: {:?}", backends.iter().map(|b| b.name()).collect::<Vec<_>>());

    let cpu = CpuBackend::new();
    let dim = 64;
    let n_vectors = 1_000;
    let (query, vectors) = generate_random_vectors(n_vectors, dim, 42);

    let metrics = [
        Metric::L2,
        Metric::Cosine,
        Metric::InnerProduct,
        Metric::L1,
        Metric::Hamming,
        Metric::Jaccard,
    ];

    for metric in metrics {
        let gold = cpu.compute_distance(&query, &vectors, dim, metric).expect("CPU gold standard");

        for b in &backends {
            if b.name() == cpu.name() {
                continue;
            }

            println!("Testing metric {:?} on backend {}", metric, b.name());
            let got = b.compute_distance(&query, &vectors, dim, metric)
                .unwrap_or_else(|e| panic!("Failed on {}: {:?}", b.name(), e));

            assert_eq!(got.len(), gold.len(), "{}: length mismatch", b.name());

            for (i, (g, c)) in gold.iter().zip(got.iter()).enumerate() {
                let tol = 1e-3 * g.abs().max(1.0);
                assert!(
                    (g - c).abs() <= tol,
                    "Backend {} metric {:?} idx={}: cpu={} gpu={} (diff={})",
                    b.name(),
                    metric,
                    i,
                    g,
                    c,
                    (g - c).abs()
                );
            }
        }
    }
}

#[test]
fn test_gpu_matches_cpu_odd_dimensions() {
    // Tests dimensions that are not powers-of-two or multiples of 16 to verify thread-tail bounds guards
    let backends = get_active_backends();
    let cpu = CpuBackend::new();

    for odd_dim in [13, 37, 71, 127] {
        let n_vectors = 250;
        let (query, vectors) = generate_random_vectors(n_vectors, odd_dim, 100 + odd_dim as u64);
        let gold = cpu.compute_distance(&query, &vectors, odd_dim, Metric::L2).expect("CPU gold");

        for b in &backends {
            if b.name() == cpu.name() {
                continue;
            }
            let got = b.compute_distance(&query, &vectors, odd_dim, Metric::L2).expect("GPU distance");
            assert_eq!(got.len(), gold.len());
            for (i, (g, c)) in gold.iter().zip(got.iter()).enumerate() {
                let tol = 1e-3 * g.abs().max(1.0);
                assert!(
                    (g - c).abs() <= tol,
                    "Backend {} odd_dim={} idx={}: cpu={} gpu={}",
                    b.name(),
                    odd_dim,
                    i,
                    g,
                    c
                );
            }
        }
    }
}

#[test]
fn test_kmeans_assignment_parity() {
    let backends = get_active_backends();
    let cpu = CpuBackend::new();
    let dim = 32;
    let n_vectors = 500;
    let n_clusters = 16;

    let mut rng = ChaCha8Rng::seed_from_u64(888);
    let vectors: Vec<f32> = (0..n_vectors * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let centroids: Vec<f32> = (0..n_clusters * dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

    let gold = cpu.compute_kmeans_assignment(&vectors, &centroids, dim).expect("CPU assignment");

    for b in &backends {
        if b.name() == cpu.name() {
            continue;
        }
        let got = b.compute_kmeans_assignment(&vectors, &centroids, dim).expect("GPU assignment");
        assert_eq!(got.len(), gold.len(), "{}: count mismatch", b.name());
        assert_eq!(got, gold, "{}: assignments did not match CPU reference", b.name());
    }
}

#[test]
fn test_ivf_flat_gpu_vs_cpu_recall() {
    use benostream_gpu_ann::{ComputeContext, IndexBuilder};

    let dim = 32;
    let n_vectors = 2_000;
    let mut rng = ChaCha8Rng::seed_from_u64(777);

    let mut vectors = vec![0.0f32; n_vectors * dim];
    for v in vectors.iter_mut() {
        *v = rng.gen_range(-1.0..1.0);
    }

    // Build GPU IVF-Flat index
    let ctx = ComputeContext::auto_detect();
    println!("Testing Recall@10 on {}", ctx.name());

    let index = IndexBuilder::new(dim, Metric::L2)
        .n_lists(32)
        .context(ctx)
        .build(&vectors, None)
        .expect("Build index on GPU");

    let cpu = CpuBackend::new();
    let k = 10;
    let n_probe = 16;
    let mut total_hits = 0;
    let n_queries = 50;

    for q_idx in 0..n_queries {
        let query = &vectors[q_idx * dim..(q_idx + 1) * dim];

        // 1. Ground truth exact scan on CPU
        let all_dists = cpu.compute_distance(query, &vectors, dim, Metric::L2).expect("CPU scan");
        let mut exact: Vec<(usize, f32)> = all_dists.into_iter().enumerate().collect();
        exact.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let exact_ids: std::collections::HashSet<u64> =
            exact.iter().take(k).map(|(id, _)| *id as u64).collect();

        // 2. GPU IVF-Flat search
        let results = index.search(query, k, n_probe, None).expect("GPU search");
        let found_ids: std::collections::HashSet<u64> = results.iter().map(|r| r.id).collect();

        let hits = exact_ids.intersection(&found_ids).count();
        total_hits += hits;
    }

    let recall = total_hits as f64 / (n_queries * k) as f64;
    println!("Measured Recall@{}: {:.2}% on {}", k, recall * 100.0, index.backend_name());
    assert!(recall >= 0.90, "Recall@{} was {:.2}%, expected >= 90%", k, recall * 100.0);
}
