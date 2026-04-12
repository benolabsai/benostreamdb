// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use hyperstreamdb::core::index::gpu::{compute_distance, ComputeBackend, ComputeContext, set_global_gpu_context};
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
    use hyperstreamdb::core::index::VectorMetric;

    // Compute with backend A
    let ctx_a = ComputeContext::from_backend(backend_a)?;
    set_global_gpu_context(Some(ctx_a));
    let dist_a = compute_distance(query, vectors, dim, VectorMetric::L2)?;

    // Compute with backend B
    let ctx_b = ComputeContext::from_backend(backend_b)?;
    set_global_gpu_context(Some(ctx_b));
    let dist_b = compute_distance(query, vectors, dim, VectorMetric::L2)?;

    // Clean up
    set_global_gpu_context(None);

    assert_eq!(
        dist_a.len(),
        dist_b.len(),
        "{} - Backend results length mismatch: {:?} vs {:?}",
        label, backend_a, backend_b
    );

    for (i, (&a, &b)) in dist_a.iter().zip(dist_b.iter()).enumerate() {
        let diff = (a - b).abs();
        assert!(
            diff < 1e-4, // Slightly looser tolerance for different backends
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
    let _query = generate_random_vectors(1, dim);
    let _vectors = generate_random_vectors(n_vectors, dim);

    // Always test CPU (reference)
    let _ref_backend = ComputeBackend::Cpu;
 
    // 1. Test CUDA if enabled
    #[cfg(feature = "cuda")]
    {
        assert_parity("CUDA Parity", &_query, &_vectors, dim, _ref_backend, ComputeBackend::Cuda)?;
    }
 
    // 2. Test MPS if enabled
    #[cfg(feature = "mps")]
    {
        assert_parity("MPS Parity", &_query, &_vectors, dim, _ref_backend, ComputeBackend::Mps)?;
    }
     
    // 3. Test ROCm and Intel WGPU interfaces on Linux
    #[cfg(target_os = "linux")]
    {
        if let Ok(_) = ComputeContext::from_backend(ComputeBackend::Rocm) {
            assert_parity("ROCm Parity", &_query, &_vectors, dim, _ref_backend, ComputeBackend::Rocm)?;
        }
        if let Ok(_) = ComputeContext::from_backend(ComputeBackend::Intel) {
            assert_parity("Intel Parity", &_query, &_vectors, dim, _ref_backend, ComputeBackend::Intel)?;
        }
    }

    Ok(())
}

#[test]
fn test_l2_parity_different_dimensions() -> Result<()> {
    // Test that the kernels handle non-power-of-two dimensions correctly
    let dims = vec![3, 64, 127, 1024]; // Reduced 1536 to 1024 for speed in parity tests
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
