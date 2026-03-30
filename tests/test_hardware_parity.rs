// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use hyperstreamdb::core::index::gpu::{compute_distance, ComputeBackend, ComputeContext};
use rand::Rng;

fn generate_random_vectors(n: usize, dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    (0..n * dim).map(|_| rng.gen::<f32>()).collect()
}

fn assert_parity(
    label: &str,
    query: &[f32],
    vectors: &[f32],
    dim: usize,
    backend_a: ComputeBackend,
    backend_b: ComputeBackend,
) -> Result<()> {
    let ctx_a = ComputeContext { backend: backend_a, device_id: 0 };
    let ctx_b = ComputeContext { backend: backend_b, device_id: 0 };

    use hyperstreamdb::core::index::VectorMetric;
    let dist_a = compute_distance(query, vectors, dim, VectorMetric::L2, &ctx_a)?;
    let dist_b = compute_distance(query, vectors, dim, VectorMetric::L2, &ctx_b)?;

    assert_eq!(
        dist_a.len(),
        dist_b.len(),
        "{} - Backend results length mismatch: {:?} vs {:?}",
        label, backend_a, backend_b
    );

    for (i, (&a, &b)) in dist_a.iter().zip(dist_b.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-5,
            "{} - Numerical divergence at index {} (backend {:?} vs {:?}): {} != {} (diff: {})",
            label, i, backend_a, backend_b, a, b, diff
        );
    }
    
    println!("{} - PASSED (backend {:?} vs {:?})", label, backend_a, backend_b);
    Ok(())
}

#[test]
fn test_l2_parity_cpu_vs_other_backends() -> Result<()> {
    let dim = 128;
    let n_vectors = 100;
    #[allow(unused_variables)]
    let query = generate_random_vectors(1, dim);
    #[allow(unused_variables)]
    let vectors = generate_random_vectors(n_vectors, dim);

    // Always test CPU (reference)
    let _ref_backend = ComputeBackend::Cpu;
 
    // 1. Test CUDA if enabled
    #[cfg(feature = "cuda")]
    {
        assert_parity("CUDA Parity", &query, &vectors, dim, _ref_backend, ComputeBackend::Cuda)?;
    }
 
    // 2. Test MPS if enabled
    #[cfg(feature = "mps")]
    {
        assert_parity("MPS Parity", &query, &vectors, dim, _ref_backend, ComputeBackend::Mps)?;
    }
     
    // 3. Test ROCm if enabled (OpenCL based)
    #[cfg(feature = "rocm")]
    {
        assert_parity("ROCm Parity", &query, &vectors, dim, _ref_backend, ComputeBackend::Rocm)?;
    }

    Ok(())
}

#[test]
fn test_l2_parity_different_dimensions() -> Result<()> {
    // Test that the kernels handle non-power-of-two dimensions correctly
    let dims = vec![3, 64, 127, 1536];
    let n_vectors = 10;
    
    for dim in dims {
        let query = generate_random_vectors(1, dim);
        let vectors = generate_random_vectors(n_vectors, dim);
        
        // CPU vs CPU (sanity check)
        assert_parity(
            &format!("CPU Sanity (dim={})", dim),
            &query, &vectors, dim, 
            ComputeBackend::Cpu, ComputeBackend::Cpu
        )?;

        // If specific backends are enabled, parity with CPU
        #[cfg(feature = "cuda")]
        assert_parity(&format!("CUDA dim={}", dim), &query, &vectors, dim, ComputeBackend::Cpu, ComputeBackend::Cuda)?;
        
        #[cfg(feature = "mps")]
        assert_parity(&format!("MPS dim={}", dim), &query, &vectors, dim, ComputeBackend::Cpu, ComputeBackend::Mps)?;
    }

    Ok(())
}
