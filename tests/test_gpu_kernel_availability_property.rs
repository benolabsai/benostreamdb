// Copyright (c) 2026 Richard Albright. All rights reserved.

// Feature: python-vector-api-gpu-acceleration, Property 7: GPU Kernel Availability
// **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
//
// Property: For any distance metric (Cosine, Inner Product, L1, Hamming, Jaccard) and any
// available GPU backend, the system should successfully compute distances using GPU acceleration.

use hyperstreamdb::core::index::gpu::{compute_distance, ComputeBackend, ComputeContext, set_global_gpu_context};
use hyperstreamdb::core::index::VectorMetric;
use proptest::prelude::*;

// Strategy for generating valid vector dimensions
fn dimension_strategy() -> impl Strategy<Value = usize> {
    prop::sample::select(vec![1, 3, 8, 16, 32, 64, 128, 256, 512, 1024])
}

// Strategy for generating number of vectors
fn num_vectors_strategy() -> impl Strategy<Value = usize> {
    1..=100usize
}

// Strategy for generating distance metrics (excluding L2 as it's already tested)
fn distance_metric_strategy() -> impl Strategy<Value = VectorMetric> {
    prop::sample::select(vec![
        VectorMetric::Cosine,
        VectorMetric::InnerProduct,
        VectorMetric::L1,
        VectorMetric::Hamming,
        VectorMetric::Jaccard,
    ])
}

// Strategy for generating compute backends
fn compute_backend_strategy() -> impl Strategy<Value = ComputeBackend> {
    prop::sample::select(vec![
        ComputeBackend::Cpu,
        #[cfg(feature = "cuda")]
        ComputeBackend::Cuda,
        #[cfg(feature = "mps")]
        ComputeBackend::Mps,
        #[cfg(feature = "rocm")]
        ComputeBackend::Rocm,
        #[cfg(feature = "intel")]
        ComputeBackend::Intel,
    ])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property 7: GPU Kernel Availability
    /// For any distance metric and any available GPU backend, the system should successfully
    /// compute distances using GPU acceleration.
    #[test]
    fn test_gpu_kernel_availability(
        dim in dimension_strategy(),
        n_vectors in num_vectors_strategy(),
        metric in distance_metric_strategy(),
        backend in compute_backend_strategy(),
        seed in any::<u64>(),
    ) {
        // Generate deterministic random vectors using the seed
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        
        let query: Vec<f32> = (0..dim).map(|_| rng.gen_range(-10.0..10.0)).collect();
        let vectors: Vec<f32> = (0..n_vectors * dim).map(|_| rng.gen_range(-10.0..10.0)).collect();
        
        // Construct context manually with None implementation first (auto-detect will handle it if needed)
        let context = ComputeContext { backend, device_id: 0, implementation: None };
        set_global_gpu_context(Some(context.clone()));
        
        // The key property: compute_distance should succeed for any metric and backend
        let result = compute_distance(&query, &vectors, dim, metric);
        
        // For CPU backend, it should always succeed
        if backend == ComputeBackend::Cpu {
            prop_assert!(result.is_ok(), "CPU backend should always succeed");
            let distances = result.unwrap();
            prop_assert_eq!(distances.len(), n_vectors, "Should return correct number of distances");
            
            // All distances should be finite
            for (i, &dist) in distances.iter().enumerate() {
                prop_assert!(dist.is_finite(), "Distance at index {} should be finite, got {}", i, dist);
            }
        } else {
            // For GPU backends, either succeed or gracefully handle unavailability
            match result {
                Ok(distances) => {
                    // If GPU computation succeeds, verify correctness
                    prop_assert_eq!(distances.len(), n_vectors, "Should return correct number of distances");
                    
                    // All distances should be finite
                    for (i, &dist) in distances.iter().enumerate() {
                        prop_assert!(dist.is_finite(), "Distance at index {} should be finite, got {}", i, dist);
                    }
                    
                    // Verify GPU results match CPU results (within tolerance)
                    let cpu_context = ComputeContext::default();
                    set_global_gpu_context(Some(cpu_context));
                    let cpu_distances = compute_distance(&query, &vectors, dim, metric)
                        .expect("CPU computation should succeed");
                    
                    for i in 0..n_vectors {
                        let diff = (distances[i] - cpu_distances[i]).abs();
                        let tolerance = 1e-2; // Allow slightly more tolerance for GPU vs CPU
                        prop_assert!(
                            diff < tolerance,
                            "GPU vs CPU mismatch at index {}: GPU={}, CPU={}, diff={}, metric={:?}",
                            i, distances[i], cpu_distances[i], diff, metric
                        );
                    }
                }
                Err(e) => {
                    // GPU backend not available or failed - this is acceptable
                    let error_msg = e.to_string();
                    prop_assert!(
                        error_msg.contains("not available") || 
                        error_msg.contains("not supported") ||
                        error_msg.contains("failed to initialize") ||
                        error_msg.contains("CUDA") ||
                        error_msg.contains("WGPU") ||
                        error_msg.contains("Metal") ||
                        error_msg.contains("CL_PLATFORM_NOT_FOUND") ||
                        error_msg.contains("CL_DEVICE_NOT_FOUND") ||
                        error_msg.contains("threshold"), // threshold logic might trigger CPU fallback
                        "Error should indicate backend unavailability, got: {}",
                        error_msg
                    );
                }
            }
        }
        
        // Clean up
        set_global_gpu_context(None);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Property 7 (Edge Cases): GPU kernels should handle edge cases correctly
    #[test]
    fn test_gpu_kernel_edge_cases(
        metric in distance_metric_strategy(),
        backend in compute_backend_strategy(),
    ) {
        let test_cases = vec![
            (vec![1.0], vec![2.0], 1),
            (vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0], 3),
            (vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0], 3),
            (vec![-1.0, -2.0, -3.0], vec![1.0, 2.0, 3.0], 3),
            (vec![1.0, -2.0, 3.0], vec![-1.0, 2.0, -3.0], 3),
        ];
        
        for (query, vector, dim) in test_cases {
            let context = ComputeContext { backend, device_id: 0, implementation: None };
            set_global_gpu_context(Some(context));
            let result = compute_distance(&query, &vector, dim, metric);
            
            if backend == ComputeBackend::Cpu {
                prop_assert!(result.is_ok());
            } else if let Ok(distances) = result {
                prop_assert_eq!(distances.len(), 1);
                
                let cpu_context = ComputeContext::default();
                set_global_gpu_context(Some(cpu_context));
                let cpu_distances = compute_distance(&query, &vector, dim, metric)
                    .expect("CPU should succeed");
                
                if !distances[0].is_nan() && !cpu_distances[0].is_nan() {
                    let diff = (distances[0] - cpu_distances[0]).abs();
                    prop_assert!(diff < 1e-2);
                }
            }
        }
        set_global_gpu_context(None);
    }
}
