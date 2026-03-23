use anyhow::Result;
use hyperstreamdb::core::index::gpu::{compute_distance, ComputeBackend, ComputeContext};
use hyperstreamdb::core::index::VectorMetric;

#[test]
fn test_jaccard_distance_cpu() -> Result<()> {
    // Test Jaccard distance computation on CPU
    // Jaccard distance = 1 - (intersection / union)
    // For vectors with non-zero elements treated as sets
    
    let query = vec![1.0, 2.0, 0.0, 3.0];
    let vectors = vec![
        1.0, 2.0, 0.0, 3.0,  // Identical: intersection=3, union=3, distance=0.0
        1.0, 2.0, 0.0, 0.0,  // Partial overlap: intersection=2, union=3, distance=1/3
        0.0, 0.0, 4.0, 5.0,  // No overlap: intersection=0, union=5, distance=1.0
        0.0, 0.0, 0.0, 0.0,  // All zeros in vector: intersection=0, union=3, distance=1.0
    ];
    let dim = 4;
    let context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard, &context)?;
    
    println!("Distances: {:?}", distances);
    
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
    let context = ComputeContext { backend: ComputeBackend::Cuda, device_id: 0 };
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard, &context)?;
    
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
    let cpu_context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
    let cpu_distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard, &cpu_context)?;
    
    // Compute on CUDA
    let cuda_context = ComputeContext { backend: ComputeBackend::Cuda, device_id: 0 };
    let cuda_distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard, &cuda_context)?;
    
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
        // Create query with all 1s
        let query = vec![1.0; dim];
        // Create two vectors: one identical, one with all 2s (different values but same positions)
        let mut vectors = vec![1.0; dim]; // First vector identical
        vectors.extend(vec![2.0; dim]); // Second vector different values
        
        let context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
        let distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard, &context)?;
        
        assert_eq!(distances.len(), 2);
        // First vector: identical, so intersection=dim, union=dim, distance=0
        assert!((distances[0] - 0.0).abs() < 1e-4, 
            "Dim {}: Expected 0.0, got {}", dim, distances[0]);
        // Second vector: different values, so intersection=0, union=dim, distance=1.0
        assert!((distances[1] - 1.0).abs() < 1e-4,
            "Dim {}: Expected 1.0, got {}", dim, distances[1]);
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
    let n_vectors = 5000; // Large enough to test batch processing
    
    let mut rng = rand::thread_rng();
    let query: Vec<f32> = (0..dim).map(|_| {
        if rng.gen::<f32>() > 0.3 { rng.gen::<f32>() } else { 0.0 }
    }).collect();
    let vectors: Vec<f32> = (0..n_vectors * dim).map(|_| {
        if rng.gen::<f32>() > 0.3 { rng.gen::<f32>() } else { 0.0 }
    }).collect();
    
    // Compute on CPU (reference)
    let cpu_context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
    let cpu_distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard, &cpu_context)?;
    
    // Compute on CUDA
    let cuda_context = ComputeContext { backend: ComputeBackend::Cuda, device_id: 0 };
    let cuda_distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard, &cuda_context)?;
    
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
    let context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard, &context)?;
    
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
    let context = ComputeContext { backend: ComputeBackend::Cuda, device_id: 0 };
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard, &context)?;
    
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
    
    // Case 1: Both vectors are all zeros
    let query = vec![0.0, 0.0, 0.0];
    let vectors = vec![0.0, 0.0, 0.0];
    let context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
    let distances = compute_distance(&query, &vectors, 3, VectorMetric::Jaccard, &context)?;
    assert!((distances[0] - 0.0).abs() < 1e-5, "All zeros should give distance 0.0");
    
    // Case 2: Query has elements, vector is all zeros
    let query = vec![1.0, 2.0, 3.0];
    let vectors = vec![0.0, 0.0, 0.0];
    let distances = compute_distance(&query, &vectors, 3, VectorMetric::Jaccard, &context)?;
    assert!((distances[0] - 1.0).abs() < 1e-5, "No overlap should give distance 1.0");
    
    // Case 3: Query is all zeros, vector has elements
    let query = vec![0.0, 0.0, 0.0];
    let vectors = vec![1.0, 2.0, 3.0];
    let distances = compute_distance(&query, &vectors, 3, VectorMetric::Jaccard, &context)?;
    assert!((distances[0] - 1.0).abs() < 1e-5, "No overlap should give distance 1.0");
    
    println!("Jaccard distance edge cases test PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_jaccard_distance_cuda_edge_cases() -> Result<()> {
    // Test edge cases for Jaccard distance on CUDA
    
    // Case 1: Both vectors are all zeros
    let query = vec![0.0, 0.0, 0.0];
    let vectors = vec![0.0, 0.0, 0.0];
    let context = ComputeContext { backend: ComputeBackend::Cuda, device_id: 0 };
    let distances = compute_distance(&query, &vectors, 3, VectorMetric::Jaccard, &context)?;
    assert!((distances[0] - 0.0).abs() < 1e-5, "All zeros should give distance 0.0");
    
    // Case 2: Query has elements, vector is all zeros
    let query = vec![1.0, 2.0, 3.0];
    let vectors = vec![0.0, 0.0, 0.0];
    let distances = compute_distance(&query, &vectors, 3, VectorMetric::Jaccard, &context)?;
    assert!((distances[0] - 1.0).abs() < 1e-5, "No overlap should give distance 1.0");
    
    // Case 3: Query is all zeros, vector has elements
    let query = vec![0.0, 0.0, 0.0];
    let vectors = vec![1.0, 2.0, 3.0];
    let distances = compute_distance(&query, &vectors, 3, VectorMetric::Jaccard, &context)?;
    assert!((distances[0] - 1.0).abs() < 1e-5, "No overlap should give distance 1.0");
    
    println!("Jaccard distance CUDA edge cases test PASSED");
    Ok(())
}
