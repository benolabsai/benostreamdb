use anyhow::Result;
use hyperstreamdb::core::index::gpu::{compute_distance, ComputeBackend, ComputeContext};
use hyperstreamdb::core::index::VectorMetric;

#[test]
fn test_inner_product_cpu() -> Result<()> {
    // Test inner product computation on CPU
    let query = vec![1.0, 2.0, 3.0];
    let vectors = vec![
        1.0, 2.0, 3.0,  // Inner product: 1*1 + 2*2 + 3*3 = 1 + 4 + 9 = 14
        4.0, 5.0, 6.0,  // Inner product: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        0.0, 0.0, 0.0,  // Inner product: 0
        -1.0, -2.0, -3.0, // Inner product: -1 -4 -9 = -14
    ];
    let dim = 3;
    let context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::InnerProduct, &context)?;
    
    assert_eq!(distances.len(), 4);
    assert!((distances[0] - 14.0).abs() < 1e-5, "Expected 14.0, got {}", distances[0]);
    assert!((distances[1] - 32.0).abs() < 1e-5, "Expected 32.0, got {}", distances[1]);
    assert!((distances[2] - 0.0).abs() < 1e-5, "Expected 0.0, got {}", distances[2]);
    assert!((distances[3] - (-14.0)).abs() < 1e-5, "Expected -14.0, got {}", distances[3]);
    
    println!("Inner product CPU test PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_inner_product_cuda() -> Result<()> {
    // Test inner product computation on CUDA
    let query = vec![1.0, 2.0, 3.0];
    let vectors = vec![
        1.0, 2.0, 3.0,  // Inner product: 14
        4.0, 5.0, 6.0,  // Inner product: 32
        0.0, 0.0, 0.0,  // Inner product: 0
        -1.0, -2.0, -3.0, // Inner product: -14
    ];
    let dim = 3;
    let context = ComputeContext { backend: ComputeBackend::Cuda, device_id: 0 };
    
    let distances = compute_distance(&query, &vectors, dim, VectorMetric::InnerProduct, &context)?;
    
    assert_eq!(distances.len(), 4);
    assert!((distances[0] - 14.0).abs() < 1e-5, "Expected 14.0, got {}", distances[0]);
    assert!((distances[1] - 32.0).abs() < 1e-5, "Expected 32.0, got {}", distances[1]);
    assert!((distances[2] - 0.0).abs() < 1e-5, "Expected 0.0, got {}", distances[2]);
    assert!((distances[3] - (-14.0)).abs() < 1e-5, "Expected -14.0, got {}", distances[3]);
    
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
    let cpu_context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
    let cpu_distances = compute_distance(&query, &vectors, dim, VectorMetric::InnerProduct, &cpu_context)?;
    
    // Compute on CUDA
    let cuda_context = ComputeContext { backend: ComputeBackend::Cuda, device_id: 0 };
    let cuda_distances = compute_distance(&query, &vectors, dim, VectorMetric::InnerProduct, &cuda_context)?;
    
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

#[test]
fn test_inner_product_various_dimensions() -> Result<()> {
    // Test inner product with various dimensions
    let dims = vec![1, 3, 16, 64, 128, 256, 512, 1024];
    
    for dim in dims {
        let query = vec![1.0; dim];
        let vectors = vec![2.0; dim * 2]; // Two vectors, each with value 2.0
        
        let context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
        let distances = compute_distance(&query, &vectors, dim, VectorMetric::InnerProduct, &context)?;
        
        assert_eq!(distances.len(), 2);
        // Inner product of [1,1,...,1] with [2,2,...,2] = 2*dim
        let expected = 2.0 * dim as f32;
        assert!((distances[0] - expected).abs() < 1e-4, 
            "Dim {}: Expected {}, got {}", dim, expected, distances[0]);
        assert!((distances[1] - expected).abs() < 1e-4,
            "Dim {}: Expected {}, got {}", dim, expected, distances[1]);
    }
    
    println!("Inner product various dimensions test PASSED");
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_inner_product_cuda_large_batch() -> Result<()> {
    // Test inner product with a large batch to verify chunking works
    use rand::Rng;
    
    let dim = 128;
    let n_vectors = 5000; // Large enough to test batch processing
    
    let mut rng = rand::thread_rng();
    let query: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>()).collect();
    let vectors: Vec<f32> = (0..n_vectors * dim).map(|_| rng.gen::<f32>()).collect();
    
    // Compute on CPU (reference)
    let cpu_context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
    let cpu_distances = compute_distance(&query, &vectors, dim, VectorMetric::InnerProduct, &cpu_context)?;
    
    // Compute on CUDA
    let cuda_context = ComputeContext { backend: ComputeBackend::Cuda, device_id: 0 };
    let cuda_distances = compute_distance(&query, &vectors, dim, VectorMetric::InnerProduct, &cuda_context)?;
    
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
    
    println!("Inner product CUDA large batch test PASSED");
    Ok(())
}
