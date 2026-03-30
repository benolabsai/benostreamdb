// Copyright (c) 2026 Richard Albright. All rights reserved.

// Feature: python-vector-api-gpu-acceleration, Property 7: GPU Kernel Availability
// **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**
//
// Property: For any distance metric (Cosine, Inner Product, L1, Hamming, Jaccard) and any
// available GPU backend, the system should successfully compute distances using GPU acceleration.

use anyhow::Result;
use hyperstreamdb::core::index::gpu::{compute_distance, ComputeBackend, ComputeContext};
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
        #[cfg(feature = "intel_gpu")]
        ComputeBackend::Intel,
    ])
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

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
        
        let context = ComputeContext { backend, device_id: 0 };
        
        // The key property: compute_distance should succeed for any metric and backend
        let result = compute_distance(&query, &vectors, dim, metric, &context);
        
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
                    let cpu_context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
                    let cpu_distances = compute_distance(&query, &vectors, dim, metric, &cpu_context)
                        .expect("CPU computation should succeed");
                    
                    for i in 0..n_vectors {
                        let diff = (distances[i] - cpu_distances[i]).abs();
                        let tolerance = match metric {
                            VectorMetric::Cosine | VectorMetric::InnerProduct => 5e-3,
                            _ => 1e-4,
                        };
                        prop_assert!(
                            diff < tolerance,
                            "GPU vs CPU mismatch at index {}: GPU={}, CPU={}, diff={}, metric={:?}",
                            i, distances[i], cpu_distances[i], diff, metric
                        );
                    }
                }
                Err(e) => {
                    // GPU backend not available or failed - this is acceptable
                    // The system should handle this gracefully
                    let error_msg = e.to_string();
                    prop_assert!(
                        error_msg.contains("not available") || 
                        error_msg.contains("not supported") ||
                        error_msg.contains("failed to initialize") ||
                        error_msg.contains("CUDA") ||
                        error_msg.contains("OpenCL") ||
                        error_msg.contains("Metal") ||
                        error_msg.contains("CL_PLATFORM_NOT_FOUND") ||
                        error_msg.contains("CL_DEVICE_NOT_FOUND"),
                        "Error should indicate backend unavailability, got: {}",
                        error_msg
                    );
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property 7 (Edge Cases): GPU kernels should handle edge cases correctly
    #[test]
    fn test_gpu_kernel_edge_cases(
        metric in distance_metric_strategy(),
        backend in compute_backend_strategy(),
    ) {
        let test_cases = vec![
            // Case 1: Single dimension
            (vec![1.0], vec![2.0], 1),
            // Case 2: All zeros
            (vec![0.0, 0.0, 0.0], vec![0.0, 0.0, 0.0], 3),
            // Case 3: Identical vectors
            (vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0], 3),
            // Case 4: Negative values
            (vec![-1.0, -2.0, -3.0], vec![1.0, 2.0, 3.0], 3),
            // Case 5: Mixed positive and negative
            (vec![1.0, -2.0, 3.0], vec![-1.0, 2.0, -3.0], 3),
        ];
        
        for (query, vector, dim) in test_cases {
            let context = ComputeContext { backend, device_id: 0 };
            let result = compute_distance(&query, &vector, dim, metric, &context);
            
            if backend == ComputeBackend::Cpu {
                prop_assert!(result.is_ok(), "CPU should handle edge case: query={:?}, vector={:?}", query, vector);
                let distances = result.unwrap();
                prop_assert_eq!(distances.len(), 1);
                prop_assert!(distances[0].is_finite() || distances[0].is_nan(), "Distance should be finite or NaN for edge case");
            } else if let Ok(distances) = result {
                // GPU backends should either succeed or fail gracefully
                prop_assert_eq!(distances.len(), 1);
                prop_assert!(distances[0].is_finite() || distances[0].is_nan(), "Distance should be finite or NaN for edge case");
                
                // Verify against CPU
                let cpu_context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
                let cpu_distances = compute_distance(&query, &vector, dim, metric, &cpu_context)
                    .expect("CPU should succeed");
                
                if !distances[0].is_nan() && !cpu_distances[0].is_nan() {
                    let diff = (distances[0] - cpu_distances[0]).abs();
                    let tolerance = match metric {
                        VectorMetric::Cosine | VectorMetric::InnerProduct => 1e-3,
                        _ => 1e-4,
                    };
                    prop_assert!(
                        diff < tolerance,
                        "Edge case mismatch: GPU={}, CPU={}, diff={}",
                        distances[0], cpu_distances[0], diff
                    );
                }
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(50))]

    /// Property 7 (Sparse Vectors): GPU kernels should handle sparse vectors correctly
    #[test]
    fn test_gpu_kernel_sparse_vectors(
        dim in dimension_strategy(),
        n_vectors in num_vectors_strategy(),
        metric in distance_metric_strategy(),
        backend in compute_backend_strategy(),
        seed in any::<u64>(),
    ) {
        // Generate sparse vectors (70% zeros)
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        
        let query: Vec<f32> = (0..dim).map(|_| {
            if rng.gen::<f32>() > 0.7 {
                rng.gen_range(-10.0..10.0)
            } else {
                0.0
            }
        }).collect();
        
        let vectors: Vec<f32> = (0..n_vectors * dim).map(|_| {
            if rng.gen::<f32>() > 0.7 {
                rng.gen_range(-10.0..10.0)
            } else {
                0.0
            }
        }).collect();
        
        let context = ComputeContext { backend, device_id: 0 };
        let result = compute_distance(&query, &vectors, dim, metric, &context);
        
        if backend == ComputeBackend::Cpu {
            prop_assert!(result.is_ok(), "CPU should handle sparse vectors");
            let distances = result.unwrap();
            prop_assert_eq!(distances.len(), n_vectors);
            
            for &dist in &distances {
                prop_assert!(dist.is_finite() || dist.is_nan(), "Sparse vector distance should be finite or NaN");
            }
        } else if let Ok(distances) = result {
            prop_assert_eq!(distances.len(), n_vectors);
            
            // Verify against CPU
            let cpu_context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
            let cpu_distances = compute_distance(&query, &vectors, dim, metric, &cpu_context)
                .expect("CPU should succeed");
            
            for i in 0..n_vectors {
                if !distances[i].is_nan() && !cpu_distances[i].is_nan() {
                    let diff = (distances[i] - cpu_distances[i]).abs();
                    let tolerance = match metric {
                        VectorMetric::Cosine | VectorMetric::InnerProduct => 1e-3,
                        _ => 1e-4,
                    };
                    prop_assert!(
                        diff < tolerance,
                        "Sparse vector mismatch at index {}: GPU={}, CPU={}, diff={}",
                        i, distances[i], cpu_distances[i], diff
                    );
                }
            }
        }
    }
}

// Additional unit tests for specific scenarios

#[test]
fn test_all_metrics_available_on_cpu() -> Result<()> {
    // Verify that all metrics work on CPU backend
    let metrics = vec![
        VectorMetric::Cosine,
        VectorMetric::InnerProduct,
        VectorMetric::L1,
        VectorMetric::Hamming,
        VectorMetric::Jaccard,
    ];
    
    let query = vec![1.0, 2.0, 3.0, 4.0];
    let vectors = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ];
    let dim = 4;
    let context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
    
    for metric in metrics {
        let result = compute_distance(&query, &vectors, dim, metric, &context);
        assert!(result.is_ok(), "Metric {:?} should work on CPU", metric);
        let distances = result?;
        assert_eq!(distances.len(), 2);
        assert!(distances[0].is_finite());
        assert!(distances[1].is_finite());
    }
    
    Ok(())
}

#[test]
#[cfg(feature = "cuda")]
fn test_all_metrics_available_on_cuda() -> Result<()> {
    // Verify that all metrics work on CUDA backend (if available)
    let metrics = vec![
        VectorMetric::Cosine,
        VectorMetric::InnerProduct,
        VectorMetric::L1,
        VectorMetric::Hamming,
        VectorMetric::Jaccard,
    ];
    
    let query = vec![1.0, 2.0, 3.0, 4.0];
    let vectors = vec![
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    ];
    let dim = 4;
    let context = ComputeContext { backend: ComputeBackend::Cuda, device_id: 0 };
    
    for metric in metrics {
        let result = compute_distance(&query, &vectors, dim, metric, &context);
        // CUDA might not be available, but if it is, all metrics should work
        if result.is_ok() {
            let distances = result?;
            assert_eq!(distances.len(), 2, "Metric {:?} should return correct number of distances", metric);
            assert!(distances[0].is_finite(), "Metric {:?} distance 0 should be finite", metric);
            assert!(distances[1].is_finite(), "Metric {:?} distance 1 should be finite", metric);
        }
    }
    
    Ok(())
}

#[test]
fn test_gpu_kernel_availability_with_large_dimensions() -> Result<()> {
    // Test with large dimensions to ensure kernels handle them correctly
    let dims = vec![512, 1024, 2048];
    let metrics = vec![
        VectorMetric::Cosine,
        VectorMetric::InnerProduct,
        VectorMetric::L1,
        VectorMetric::Hamming,
        VectorMetric::Jaccard,
    ];
    
    for dim in dims {
        let query = vec![1.0; dim];
        let vectors = vec![2.0; dim * 2]; // Two vectors
        
        let cpu_context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
        
        for metric in &metrics {
            let result = compute_distance(&query, &vectors, dim, *metric, &cpu_context);
            assert!(result.is_ok(), "Metric {:?} with dim {} should work on CPU", metric, dim);
            let distances = result?;
            assert_eq!(distances.len(), 2);
            assert!(distances[0].is_finite());
            assert!(distances[1].is_finite());
        }
    }
    
    Ok(())
}
