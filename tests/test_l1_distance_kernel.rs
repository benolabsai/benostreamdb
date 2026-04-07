// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use hyperstreamdb::core::index::gpu::{compute_distance, ComputeContext, set_global_gpu_context};
use hyperstreamdb::core::index::VectorMetric;

#[test]
fn test_l1_distance_cpu() -> Result<()> {
    // Test L1 (Manhattan) distance computation on CPU
    let query = vec![1.0, 2.0, 3.0];
    let vectors = vec![
        1.0, 2.0, 3.0,  // L1 distance: |1-1| + |2-2| + |3-3| = 0
        4.0, 5.0, 6.0,  // L1 distance: |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9
        0.0, 0.0, 0.0,  // L1 distance: |1-0| + |2-0| + |3-0| = 1 + 2 + 3 = 6
        -1.0, -2.0, -3.0, // L1 distance: |1-(-1)| + |2-(-2)| + |3-(-3)| = 2 + 4 + 6 = 12
    ];
    let dim = 3;
    let context = ComputeContext::default();
    set_global_gpu_context(Some(context));
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::L1)?;
    
    assert_eq!(distances.len(), 4);
    assert!((distances[0] - 0.0).abs() < 1e-5, "Expected 0.0, got {}", distances[0]);
    assert!((distances[1] - 9.0).abs() < 1e-5, "Expected 9.0, got {}", distances[1]);
    assert!((distances[2] - 6.0).abs() < 1e-5, "Expected 6.0, got {}", distances[2]);
    assert!((distances[3] - 12.0).abs() < 1e-5, "Expected 12.0, got {}", distances[3]);
    
    println!("L1 distance CPU test PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_l1_distance_cuda() -> Result<()> {
    // Test L1 (Manhattan) distance computation on CUDA
    let query = vec![1.0, 2.0, 3.0];
    let vectors = vec![
        1.0, 2.0, 3.0,  // L1 distance: 0
        4.0, 5.0, 6.0,  // L1 distance: 9
        0.0, 0.0, 0.0,  // L1 distance: 6
        -1.0, -2.0, -3.0, // L1 distance: 12
    ];
    let dim = 3;
    let context = ComputeContext::from_device_str("cuda:0")?;
    set_global_gpu_context(Some(context));
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::L1)?;
    
    assert_eq!(distances.len(), 4);
    assert!((distances[0] - 0.0).abs() < 1e-5, "Expected 0.0, got {}", distances[0]);
    assert!((distances[1] - 9.0).abs() < 1e-5, "Expected 9.0, got {}", distances[1]);
    assert!((distances[2] - 6.0).abs() < 1e-5, "Expected 6.0, got {}", distances[2]);
    assert!((distances[3] - 12.0).abs() < 1e-5, "Expected 12.0, got {}", distances[3]);
    
    println!("L1 distance CUDA test PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_l1_distance_cuda_vs_cpu_parity() -> Result<()> {
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
    let cpu_distances = compute_distance(&query, &vectors, dim, VectorMetric::L1)?;
    
    // Compute on CUDA
    let cuda_context = ComputeContext::from_device_str("cuda:0")?;
    set_global_gpu_context(Some(cuda_context));
    let cuda_distances = compute_distance(&query, &vectors, dim, VectorMetric::L1)?;
    
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
    
    println!("L1 distance CUDA vs CPU parity test PASSED");
    Ok(())
}

#[test]
fn test_l1_distance_various_dimensions() -> Result<()> {
    // Test L1 distance with various dimensions
    let dims = vec![1, 3, 16, 64, 128, 256, 512, 1024];
    
    for dim in dims {
        let query = vec![1.0; dim];
        let vectors = vec![2.0; dim * 2]; // Two vectors, each with value 2.0
        
        let context = ComputeContext::default();
        set_global_gpu_context(Some(context));
        let distances = compute_distance(&query, &vectors, dim, VectorMetric::L1)?;
        
        assert_eq!(distances.len(), 2);
        // L1 distance of [1,1,...,1] with [2,2,...,2] = |1-2| * dim = dim
        let expected = dim as f32;
        assert!((distances[0] - expected).abs() < 1e-4, 
            "Dim {}: Expected {}, got {}", dim, expected, distances[0]);
        assert!((distances[1] - expected).abs() < 1e-4,
            "Dim {}: Expected {}, got {}", dim, expected, distances[1]);
    }
    
    println!("L1 distance various dimensions test PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_l1_distance_cuda_large_batch() -> Result<()> {
    // Test L1 distance with a large batch to verify chunking works
    use rand::Rng;
    
    let dim = 128;
    let n_vectors = 5000; // Large enough to test batch processing
    
    let mut rng = rand::thread_rng();
    let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let vectors: Vec<f32> = (0..n_vectors * dim).map(|_| rng.gen::<f32>()).collect();
    
    // Compute on CPU (reference)
    let cpu_context = ComputeContext::default();
    set_global_gpu_context(Some(cpu_context));
    let cpu_distances = compute_distance(&query, &vectors, dim, VectorMetric::L1)?;
    
    // Compute on CUDA
    let cuda_context = ComputeContext::from_device_str("cuda:0")?;
    set_global_gpu_context(Some(cuda_context));
    let cuda_distances = compute_distance(&query, &vectors, dim, VectorMetric::L1)?;
    
    // Compare results
    assert_eq!(cpu_distances.len(), cuda_distances.len());
    for i in 0..n_vectors {
        let diff = (cpu_distances[i] - cuda_distances[i]).abs();
        assert!(
            diff < 1e-3,
            "Mismatch at index {}: CPU={}, CUDA={}, diff={}",
            i, cpu_distances[i], cuda_distances[i], diff
        );
    }
    
    println!("L1 distance CUDA large batch test PASSED");
    Ok(())
}
