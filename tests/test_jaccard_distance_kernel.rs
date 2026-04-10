// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use hyperstreamdb::core::index::gpu::{compute_distance, ComputeContext, set_global_gpu_context};
use hyperstreamdb::core::index::VectorMetric;

#[test]
fn test_jaccard_distance_cpu() -> Result<()> {
    // Test Jaccard distance computation on CPU
    let query = vec![1.0, 2.0, 0.0, 3.0];
    let vectors = vec![
        1.0, 2.0, 0.0, 3.0,  // Identical: intersection=3, union=3, distance=0.0
        1.0, 2.0, 0.0, 0.0,  // Partial overlap: intersection=2, union=3, distance=1/3
        0.0, 0.0, 4.0, 5.0,  // No overlap: intersection=0, union=5, distance=1.0
        0.0, 0.0, 0.0, 0.0,  // All zeros in vector: intersection=0, union=3, distance=1.0
    ];
    let dim = 4;
    let context = ComputeContext::default();
    set_global_gpu_context(Some(context));
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard)?;
    
    assert_eq!(distances.len(), 4);
    assert!((distances[0] - 0.0).abs() < 1e-5, "Expected 0.0, got {}", distances[0]);
    assert!((distances[1] - (1.0/3.0)).abs() < 1e-5, "Expected 0.333, got {}", distances[1]);
    assert!((distances[2] - 1.0).abs() < 1e-5, "Expected 1.0, got {}", distances[2]);
    assert!((distances[3] - 1.0).abs() < 1e-5, "Expected 1.0 (no overlap), got {}", distances[3]);
    
    println!("Jaccard distance CPU test PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_jaccard_distance_cuda() -> Result<()> {
    // Test Jaccard distance computation on CUDA
    let query = vec![1.0, 2.0, 0.0, 3.0];
    let vectors = vec![
        1.0, 2.0, 0.0, 3.0,  // Identical
        1.0, 2.0, 0.0, 0.0,  // Partial overlap
        0.0, 0.0, 4.0, 5.0,  // No overlap
        0.0, 0.0, 0.0, 0.0,  // All zeros in vector
    ];
    let dim = 4;
    let context = ComputeContext::from_device_str("cuda:0")?;
    set_global_gpu_context(Some(context));
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard)?;
    
    assert_eq!(distances.len(), 4);
    assert!((distances[0] - 0.0).abs() < 1e-5, "Expected 0.0, got {}", distances[0]);
    assert!((distances[1] - (1.0/3.0)).abs() < 1e-5, "Expected 0.333, got {}", distances[1]);
    assert!((distances[2] - 1.0).abs() < 1e-5, "Expected 1.0, got {}", distances[2]);
    assert!((distances[3] - 1.0).abs() < 1e-5, "Expected 1.0 (no overlap), got {}", distances[3]);
    
    println!("Jaccard distance CUDA test PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_jaccard_distance_cuda_vs_cpu_parity() -> Result<()> {
    // Test that CUDA and CPU produce the same results
    use rand::Rng;
    
    let dim = 128;
    let n_vectors = 100;
    
    // Generate random vectors with some zeros
    let mut rng = rand::thread_rng();
    let query: Vec<f32> = (0..dim).map(|_| {
        if rng.gen::<f32>() > 0.3 { rng.gen::<f32>() } else { 0.0 }
    }).collect();
    let vectors: Vec<f32> = (0..n_vectors * dim).map(|_| {
        if rng.gen::<f32>() > 0.3 { rng.gen::<f32>() } else { 0.0 }
    }).collect();
    
    // Compute on CPU
    let cpu_context = ComputeContext::default();
    set_global_gpu_context(Some(cpu_context));
    let cpu_distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard)?;
    
    // Compute on CUDA
    let cuda_context = ComputeContext::from_device_str("cuda:0")?;
    set_global_gpu_context(Some(cuda_context));
    let cuda_distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard)?;
    
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
    
    println!("Jaccard distance CUDA vs CPU parity test PASSED");
    Ok(())
}

#[test]
fn test_jaccard_distance_various_dimensions() -> Result<()> {
    // Test Jaccard distance with various dimensions
    let dims = vec![1, 3, 16, 64, 128, 256, 512, 1024];
    
    for dim in dims {
        let query = vec![1.0; dim];
        let mut vectors = vec![1.0; dim];
        vectors.extend(vec![2.0; dim]);
        
        let context = ComputeContext::default();
        set_global_gpu_context(Some(context));
        let distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard)?;
        
        assert_eq!(distances.len(), 2);
        assert!((distances[0] - 0.0).abs() < 1e-4);
        assert!((distances[1] - 1.0).abs() < 1e-4);
    }
    
    println!("Jaccard distance various dimensions test PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_jaccard_distance_cuda_large_batch() -> Result<()> {
    // Test Jaccard distance with a large batch to verify chunking works
    use rand::Rng;
    
    let dim = 128;
    let n_vectors = 5000;
    
    let mut rng = rand::thread_rng();
    let query: Vec<f32> = (0..dim).map(|_| {
        if rng.gen::<f32>() > 0.3 { rng.gen::<f32>() } else { 0.0 }
    }).collect();
    let vectors: Vec<f32> = (0..n_vectors * dim).map(|_| {
        if rng.gen::<f32>() > 0.3 { rng.gen::<f32>() } else { 0.0 }
    }).collect();
    
    // Compute on CPU (reference)
    let cpu_context = ComputeContext::default();
    set_global_gpu_context(Some(cpu_context));
    let cpu_distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard)?;
    
    // Compute on CUDA
    let cuda_context = ComputeContext::from_device_str("cuda:0")?;
    set_global_gpu_context(Some(cuda_context));
    let cuda_distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard)?;
    
    // Compare results
    assert_eq!(cpu_distances.len(), cuda_distances.len());
    for i in 0..n_vectors {
        let diff = (cpu_distances[i] - cuda_distances[i]).abs();
        assert!(diff < 1e-3);
    }
    
    println!("Jaccard distance CUDA large batch test PASSED");
    Ok(())
}

#[test]
fn test_jaccard_distance_binary_sets() -> Result<()> {
    // Test Jaccard distance with binary vectors (0s and 1s) representing sets
    let query = vec![1.0, 0.0, 1.0, 1.0, 0.0];
    let vectors = vec![
        1.0, 0.0, 1.0, 1.0, 0.0,  // Identical: intersection=3, union=3, distance=0.0
        1.0, 0.0, 1.0, 0.0, 0.0,  // Subset: intersection=2, union=3, distance=1/3
        0.0, 1.0, 0.0, 0.0, 1.0,  // Disjoint: intersection=0, union=5, distance=1.0
        1.0, 1.0, 1.0, 1.0, 1.0,  // Superset: intersection=3, union=5, distance=2/5
    ];
    let dim = 5;
    let context = ComputeContext::default();
    set_global_gpu_context(Some(context));
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard)?;
    
    assert_eq!(distances.len(), 4);
    assert!((distances[0] - 0.0).abs() < 1e-5, "Expected 0.0, got {}", distances[0]);
    assert!((distances[1] - (1.0/3.0)).abs() < 1e-5, "Expected 0.333, got {}", distances[1]);
    assert!((distances[2] - 1.0).abs() < 1e-5, "Expected 1.0, got {}", distances[2]);
    assert!((distances[3] - (2.0/5.0)).abs() < 1e-5, "Expected 0.4, got {}", distances[3]);
    
    println!("Jaccard distance binary sets test PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_jaccard_distance_cuda_binary_sets() -> Result<()> {
    // Test Jaccard distance with binary vectors on CUDA
    let query = vec![1.0, 0.0, 1.0, 1.0, 0.0];
    let vectors = vec![
        1.0, 0.0, 1.0, 1.0, 0.0,  // Identical
        1.0, 0.0, 1.0, 0.0, 0.0,  // Subset
        0.0, 1.0, 0.0, 0.0, 1.0,  // Disjoint
        1.0, 1.0, 1.0, 1.0, 1.0,  // Superset
    ];
    let dim = 5;
    let context = ComputeContext::from_device_str("cuda:0")?;
    set_global_gpu_context(Some(context));
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard)?;
    
    assert_eq!(distances.len(), 4);
    assert!((distances[0] - 0.0).abs() < 1e-5, "Expected 0.0, got {}", distances[0]);
    assert!((distances[1] - (1.0/3.0)).abs() < 1e-5, "Expected 0.333, got {}", distances[1]);
    assert!((distances[2] - 1.0).abs() < 1e-5, "Expected 1.0, got {}", distances[2]);
    assert!((distances[3] - (2.0/5.0)).abs() < 1e-5, "Expected 0.4, got {}", distances[3]);
    
    println!("Jaccard distance CUDA binary sets test PASSED");
    Ok(())
}

#[test]
fn test_jaccard_distance_edge_cases() -> Result<()> {
    // Test edge cases for Jaccard distance
    let query = vec![0.0, 0.0, 0.0];
    let vectors = vec![0.0, 0.0, 0.0];
    let context = ComputeContext::default();
    set_global_gpu_context(Some(context));
    let distances = compute_distance(&query, &vectors, 3, VectorMetric::Jaccard)?;
    assert!((distances[0] - 0.0).abs() < 1e-5, "All zeros should give distance 0.0");
    
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_jaccard_distance_cuda_edge_cases() -> Result<()> {
    // Test edge cases for Jaccard distance on CUDA
    let query = vec![0.0, 0.0, 0.0];
    let vectors = vec![0.0, 0.0, 0.0];
    let context = ComputeContext::from_device_str("cuda:0")?;
    set_global_gpu_context(Some(context));
    let distances = compute_distance(&query, &vectors, 3, VectorMetric::Jaccard)?;
    assert!((distances[0] - 0.0).abs() < 1e-5, "All zeros should give distance 0.0");
    
    Ok(())
}
