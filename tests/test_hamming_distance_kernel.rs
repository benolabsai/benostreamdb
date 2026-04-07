// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use hyperstreamdb::core::index::gpu::{compute_distance, ComputeContext, set_global_gpu_context};
use hyperstreamdb::core::index::VectorMetric;

#[test]
fn test_hamming_distance_cpu() -> Result<()> {
    // Test Hamming distance computation on CPU
    let query = vec![1.0, 2.0, 3.0];
    let vectors = vec![
        1.0, 2.0, 3.0,  // Hamming distance: 0 (all elements match)
        1.0, 2.0, 4.0,  // Hamming distance: 1 (one element differs)
        1.0, 5.0, 6.0,  // Hamming distance: 2 (two elements differ)
        4.0, 5.0, 6.0,  // Hamming distance: 3 (all elements differ)
    ];
    let dim = 3;
    let context = ComputeContext::default();
    set_global_gpu_context(Some(context));
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::Hamming)?;
    
    assert_eq!(distances.len(), 4);
    assert!((distances[0] - 0.0).abs() < 1e-5, "Expected 0.0, got {}", distances[0]);
    assert!((distances[1] - 1.0).abs() < 1e-5, "Expected 1.0, got {}", distances[1]);
    assert!((distances[2] - 2.0).abs() < 1e-5, "Expected 2.0, got {}", distances[2]);
    assert!((distances[3] - 3.0).abs() < 1e-5, "Expected 3.0, got {}", distances[3]);
    
    println!("Hamming distance CPU test PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_hamming_distance_cuda() -> Result<()> {
    // Test Hamming distance computation on CUDA
    let query = vec![1.0, 2.0, 3.0];
    let vectors = vec![
        1.0, 2.0, 3.0,  // Hamming distance: 0
        1.0, 2.0, 4.0,  // Hamming distance: 1
        1.0, 5.0, 6.0,  // Hamming distance: 2
        4.0, 5.0, 6.0,  // Hamming distance: 3
    ];
    let dim = 3;
    let context = ComputeContext::from_device_str("cuda:0")?;
    set_global_gpu_context(Some(context));
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::Hamming)?;
    
    assert_eq!(distances.len(), 4);
    assert!((distances[0] - 0.0).abs() < 1e-5, "Expected 0.0, got {}", distances[0]);
    assert!((distances[1] - 1.0).abs() < 1e-5, "Expected 1.0, got {}", distances[1]);
    assert!((distances[2] - 2.0).abs() < 1e-5, "Expected 2.0, got {}", distances[2]);
    assert!((distances[3] - 3.0).abs() < 1e-5, "Expected 3.0, got {}", distances[3]);
    
    println!("Hamming distance CUDA test PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_hamming_distance_cuda_vs_cpu_parity() -> Result<()> {
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
    let cpu_distances = compute_distance(&query, &vectors, dim, VectorMetric::Hamming)?;
    
    // Compute on CUDA
    let cuda_context = ComputeContext::from_device_str("cuda:0")?;
    set_global_gpu_context(Some(cuda_context));
    let cuda_distances = compute_distance(&query, &vectors, dim, VectorMetric::Hamming)?;
    
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
    
    println!("Hamming distance CUDA vs CPU parity test PASSED");
    Ok(())
}

#[test]
fn test_hamming_distance_various_dimensions() -> Result<()> {
    // Test Hamming distance with various dimensions
    let dims = vec![1, 3, 16, 64, 128, 256, 512, 1024];
    
    for dim in dims {
        let query = vec![1.0; dim];
        let vectors = vec![2.0; dim * 2]; // Two vectors, each with value 2.0
        
        let context = ComputeContext::default();
        set_global_gpu_context(Some(context));
        let distances = compute_distance(&query, &vectors, dim, VectorMetric::Hamming)?;
        
        assert_eq!(distances.len(), 2);
        // Hamming distance: all elements differ, so distance = dim
        let expected = dim as f32;
        assert!((distances[0] - expected).abs() < 1e-4, 
            "Dim {}: Expected {}, got {}", dim, expected, distances[0]);
        assert!((distances[1] - expected).abs() < 1e-4,
            "Dim {}: Expected {}, got {}", dim, expected, distances[1]);
    }
    
    println!("Hamming distance various dimensions test PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_hamming_distance_cuda_large_batch() -> Result<()> {
    // Test Hamming distance with a large batch to verify chunking works
    use rand::Rng;
    
    let dim = 128;
    let n_vectors = 5000; // Large enough to test batch processing
    
    let mut rng = rand::thread_rng();
    let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let vectors: Vec<f32> = (0..n_vectors * dim).map(|_| rng.gen::<f32>()).collect();
    
    // Compute on CPU (reference)
    let cpu_context = ComputeContext::default();
    set_global_gpu_context(Some(cpu_context));
    let cpu_distances = compute_distance(&query, &vectors, dim, VectorMetric::Hamming)?;
    
    // Compute on CUDA
    let cuda_context = ComputeContext::from_device_str("cuda:0")?;
    set_global_gpu_context(Some(cuda_context));
    let cuda_distances = compute_distance(&query, &vectors, dim, VectorMetric::Hamming)?;
    
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
    
    println!("Hamming distance CUDA large batch test PASSED");
    Ok(())
}

#[test]
fn test_hamming_distance_binary_vectors() -> Result<()> {
    // Test Hamming distance with binary vectors (0s and 1s)
    let query = vec![1.0, 0.0, 1.0, 1.0, 0.0];
    let vectors = vec![
        1.0, 0.0, 1.0, 1.0, 0.0,  // Hamming distance: 0 (identical)
        0.0, 1.0, 0.0, 0.0, 1.0,  // Hamming distance: 5 (all bits flipped)
        1.0, 0.0, 1.0, 0.0, 0.0,  // Hamming distance: 1 (one bit differs)
        1.0, 1.0, 1.0, 1.0, 1.0,  // Hamming distance: 2 (two bits differ)
    ];
    let dim = 5;
    let context = ComputeContext::default();
    set_global_gpu_context(Some(context));
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::Hamming)?;
    
    assert_eq!(distances.len(), 4);
    assert!((distances[0] - 0.0).abs() < 1e-5, "Expected 0.0, got {}", distances[0]);
    assert!((distances[1] - 5.0).abs() < 1e-5, "Expected 5.0, got {}", distances[1]);
    assert!((distances[2] - 1.0).abs() < 1e-5, "Expected 1.0, got {}", distances[2]);
    assert!((distances[3] - 2.0).abs() < 1e-5, "Expected 2.0, got {}", distances[3]);
    
    println!("Hamming distance binary vectors test PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_hamming_distance_cuda_binary_vectors() -> Result<()> {
    // Test Hamming distance with binary vectors on CUDA
    let query = vec![1.0, 0.0, 1.0, 1.0, 0.0];
    let vectors = vec![
        1.0, 0.0, 1.0, 1.0, 0.0,  // Hamming distance: 0
        0.0, 1.0, 0.0, 0.0, 1.0,  // Hamming distance: 5
        1.0, 0.0, 1.0, 0.0, 0.0,  // Hamming distance: 1
        1.0, 1.0, 1.0, 1.0, 1.0,  // Hamming distance: 2
    ];
    let dim = 5;
    let context = ComputeContext::from_device_str("cuda:0")?;
    set_global_gpu_context(Some(context));
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::Hamming)?;
    
    assert_eq!(distances.len(), 4);
    assert!((distances[0] - 0.0).abs() < 1e-5, "Expected 0.0, got {}", distances[0]);
    assert!((distances[1] - 5.0).abs() < 1e-5, "Expected 5.0, got {}", distances[1]);
    assert!((distances[2] - 1.0).abs() < 1e-5, "Expected 1.0, got {}", distances[2]);
    assert!((distances[3] - 2.0).abs() < 1e-5, "Expected 2.0, got {}", distances[3]);
    
    println!("Hamming distance CUDA binary vectors test PASSED");
    Ok(())
}
