// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use hyperstreamdb::core::index::gpu::{compute_distance, ComputeContext, set_global_gpu_context};
use hyperstreamdb::core::index::VectorMetric;

#[test]
fn test_inner_product_cpu() -> Result<()> {
    // Test Inner Product computation on CPU
    let query = vec![1.0, 2.0, 3.0];
    let vectors = vec![
        1.0, 2.0, 3.0,  // IP: 1*1 + 2*2 + 3*3 = 1+4+9 = 14
        1.0, 0.0, 0.0,  // IP: 1*1 + 2*0 + 3*0 = 1
        0.0, 1.0, 0.0,  // IP: 1*0 + 2*1 + 3*0 = 2
        0.0, 0.0, 1.0,  // IP: 1*0 + 2*0 + 3*1 = 3
    ];
    let dim = 3;
    let context = ComputeContext::default();
    set_global_gpu_context(Some(context));
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::InnerProduct)?;
    
    assert_eq!(distances.len(), 4);
    assert!((distances[0] - 14.0).abs() < 1e-5, "Expected 14.0, got {}", distances[0]);
    assert!((distances[1] - 1.0).abs() < 1e-5, "Expected 1.0, got {}", distances[1]);
    assert!((distances[2] - 2.0).abs() < 1e-5, "Expected 2.0, got {}", distances[2]);
    assert!((distances[3] - 3.0).abs() < 1e-5, "Expected 3.0, got {}", distances[3]);
    
    println!("Inner product CPU test PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_inner_product_cuda() -> Result<()> {
    // Test Inner Product computation on CUDA
    let query = vec![1.0, 2.0, 3.0];
    let vectors = vec![
        1.0, 2.0, 3.0,  // IP: 14
        1.0, 0.0, 0.0,  // IP: 1
        0.0, 1.0, 0.0,  // IP: 2
        0.0, 0.0, 1.0,  // IP: 3
    ];
    let dim = 3;
    let context = ComputeContext::from_device_str("cuda:0")?;
    set_global_gpu_context(Some(context));
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::InnerProduct)?;
    
    assert_eq!(distances.len(), 4);
    assert!((distances[0] - 14.0).abs() < 1e-5, "Expected 14.0, got {}", distances[0]);
    assert!((distances[1] - 1.0).abs() < 1e-5, "Expected 1.0, got {}", distances[1]);
    assert!((distances[2] - 2.0).abs() < 1e-5, "Expected 2.0, got {}", distances[2]);
    assert!((distances[3] - 3.0).abs() < 1e-5, "Expected 3.0, got {}", distances[3]);
    
    println!("Inner product CUDA test PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_inner_product_cuda_vs_cpu_parity() -> Result<()> {
    // Test that CUDA and CPU produce the same results
    use rand::Rng;
    
    let dim = 128;
    let n_vectors = 100;
    
    // Generate random vectors
    let mut rng = rand::thread_rng();
    let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let vectors: Vec<f32> = (0..n_vectors * dim).map(|_| rng.gen::<f32>()).collect();
    
    // Compute on CPU
    let cpu_context = ComputeContext::default();
    set_global_gpu_context(Some(cpu_context));
    let cpu_distances = compute_distance(&query, &vectors, dim, VectorMetric::InnerProduct)?;
    
    // Compute on CUDA
    let cuda_context = ComputeContext::from_device_str("cuda:0")?;
    set_global_gpu_context(Some(cuda_context));
    let cuda_distances = compute_distance(&query, &vectors, dim, VectorMetric::InnerProduct)?;
    
    // Compare results
    assert_eq!(cpu_distances.len(), cuda_distances.len());
    for i in 0..n_vectors {
        let diff = (cpu_distances[i] - cuda_distances[i]).abs();
        assert!(
            diff < 1e-4,
            "Mismatch at index {}: CPU={}, CUDA={}, diff={}",
            i, cpu_distances[i], cuda_distances[i], diff
        );
    }
    
    println!("Inner product CUDA vs CPU parity test PASSED");
    Ok(())
}
