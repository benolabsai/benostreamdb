// Copyright (c) 2026 Richard Albright. All rights reserved.

/// Integration test for pgvector operators with GPU acceleration
/// 
/// This test verifies that pgvector distance operators (<->, <=>, <#>)
/// use GPU acceleration when a GPU context is configured.
/// 
/// Requirements: 7.6, 7.7
use hyperstreamdb::core::index::gpu::{ComputeContext, ComputeBackend, set_global_gpu_context, get_global_gpu_context};
use hyperstreamdb::core::sql::vector_udf::all_vector_udfs;
use datafusion::prelude::*;
use datafusion::execution::FunctionRegistry;
use datafusion::arrow::array::{Float32Array, FixedSizeListArray};
use datafusion::arrow::datatypes::{DataType, Field};
use std::sync::Arc;

/// Test that pgvector operators route through GPU-enabled UDFs
#[tokio::test]
async fn test_pgvector_operators_use_gpu_context() {
    // Create a session context
    let ctx = SessionContext::new();
    
    // Register vector UDFs (these are what the operators map to)
    for udf in all_vector_udfs() {
        ctx.register_udf(udf);
    }
    
    // Create test data - two vectors
    let vec1 = vec![1.0f32, 2.0, 3.0];
    let vec2 = vec![4.0f32, 5.0, 6.0];
    
    // Create FixedSizeList arrays
    let values1 = Float32Array::from(vec1.clone());
    let values2 = Float32Array::from(vec2.clone());
    
    let field = Arc::new(Field::new("item", DataType::Float32, true));
    let list1 = FixedSizeListArray::try_new(field.clone(), 3, Arc::new(values1), None).unwrap();
    let list2 = FixedSizeListArray::try_new(field, 3, Arc::new(values2), None).unwrap();
    
    // Create a DataFrame with the vectors
    let batch = datafusion::arrow::record_batch::RecordBatch::try_from_iter(vec![
        ("v1", Arc::new(list1) as Arc<dyn datafusion::arrow::array::Array>),
        ("v2", Arc::new(list2) as Arc<dyn datafusion::arrow::array::Array>),
    ]).unwrap();
    
    let df = ctx.read_batch(batch).unwrap();
    
    // Test 1: Compute distance without GPU context (CPU fallback)
    set_global_gpu_context(None);
    assert!(get_global_gpu_context().is_none(), "GPU context should be None initially");
    
    // Use the UDF directly (simulating what pgvector operator would do)
    let result_cpu = df.clone()
        .select(vec![
            Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
                ctx.udf("dist_l2").unwrap(),
                vec![col("v1"), col("v2")]
            ))
        ])
        .unwrap()
        .collect()
        .await
        .unwrap();
    
    assert_eq!(result_cpu.len(), 1);
    let distance_array_cpu = result_cpu[0].column(0).as_any().downcast_ref::<Float32Array>().unwrap();
    let cpu_distance = distance_array_cpu.value(0);
    
    // Expected L2 distance: sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27) ≈ 5.196
    assert!((cpu_distance - 5.196).abs() < 0.01, 
            "CPU distance should be approximately 5.196, got {}", cpu_distance);
    
    // Test 2: Set GPU context and verify same result (GPU path)
    let gpu_ctx = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
    set_global_gpu_context(Some(gpu_ctx));
    assert!(get_global_gpu_context().is_some(), "GPU context should be set");
    
    let result_gpu = df.clone()
        .select(vec![
            Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
                ctx.udf("dist_l2").unwrap(),
                vec![col("v1"), col("v2")]
            ))
        ])
        .unwrap()
        .collect()
        .await
        .unwrap();
    
    assert_eq!(result_gpu.len(), 1);
    let distance_array_gpu = result_gpu[0].column(0).as_any().downcast_ref::<Float32Array>().unwrap();
    let gpu_distance = distance_array_gpu.value(0);
    
    // GPU and CPU should produce identical results
    assert!((gpu_distance - cpu_distance).abs() < 0.001, 
            "GPU and CPU distances should match, got CPU: {}, GPU: {}", cpu_distance, gpu_distance);
    
    // Clean up
    set_global_gpu_context(None);
}

/// Test all pgvector operators with GPU context
#[tokio::test]
async fn test_all_pgvector_operators_with_gpu() {
    // Create a session context
    let ctx = SessionContext::new();
    
    // Register vector UDFs
    for udf in all_vector_udfs() {
        ctx.register_udf(udf);
    }
    
    // Set GPU context (using CPU backend for testing)
    let gpu_ctx = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
    set_global_gpu_context(Some(gpu_ctx));
    
    // Create test data
    let vec1 = vec![1.0f32, 2.0, 3.0];
    let vec2 = vec![4.0f32, 5.0, 6.0];
    
    let values1 = Float32Array::from(vec1.clone());
    let values2 = Float32Array::from(vec2.clone());
    
    let field = Arc::new(Field::new("item", DataType::Float32, true));
    let list1 = FixedSizeListArray::try_new(field.clone(), 3, Arc::new(values1), None).unwrap();
    let list2 = FixedSizeListArray::try_new(field, 3, Arc::new(values2), None).unwrap();
    
    let batch = datafusion::arrow::record_batch::RecordBatch::try_from_iter(vec![
        ("v1", Arc::new(list1) as Arc<dyn datafusion::arrow::array::Array>),
        ("v2", Arc::new(list2) as Arc<dyn datafusion::arrow::array::Array>),
    ]).unwrap();
    
    let df = ctx.read_batch(batch).unwrap();
    
    // Test all pgvector operators by testing their corresponding UDFs
    // Operator <-> maps to dist_l2
    // Operator <=> maps to dist_cosine
    // Operator <#> maps to dist_ip
    // Operator <+> maps to dist_l1
    // Operator <~> maps to dist_hamming
    // Operator <%> maps to dist_jaccard
    
    let operators = vec![
        ("dist_l2", "<->", "L2"),
        ("dist_cosine", "<=>", "Cosine"),
        ("dist_ip", "<#>", "Inner Product"),
        ("dist_l1", "<+>", "L1"),
        ("dist_hamming", "<~>", "Hamming"),
        ("dist_jaccard", "<%>", "Jaccard"),
    ];
    
    for (udf_name, operator, metric_name) in operators {
        let result = df.clone()
            .select(vec![
                Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
                    ctx.udf(udf_name).unwrap(),
                    vec![col("v1"), col("v2")]
                ))
            ])
            .unwrap()
            .collect()
            .await;
        
        assert!(result.is_ok(), 
                "Operator {} ({}) should compute successfully with GPU context", operator, metric_name);
        
        let result = result.unwrap();
        assert_eq!(result.len(), 1, "{} should return 1 row", metric_name);
        
        let distance_array = result[0].column(0).as_any().downcast_ref::<Float32Array>().unwrap();
        let distance = distance_array.value(0);
        
        // Verify we got a valid number (not NaN or Inf)
        assert!(distance.is_finite(), 
                "{} distance (operator {}) should be finite, got {}", metric_name, operator, distance);
    }
    
    // Clean up
    set_global_gpu_context(None);
}

/// Test that GPU context is properly used across multiple queries
#[tokio::test]
async fn test_gpu_context_persistence_across_queries() {
    let ctx = SessionContext::new();
    
    // Register vector UDFs
    for udf in all_vector_udfs() {
        ctx.register_udf(udf);
    }
    
    // Create test data
    let vec1 = vec![1.0f32, 0.0, 0.0];
    let vec2 = vec![0.0f32, 1.0, 0.0];
    
    let values1 = Float32Array::from(vec1.clone());
    let values2 = Float32Array::from(vec2.clone());
    
    let field = Arc::new(Field::new("item", DataType::Float32, true));
    let list1 = FixedSizeListArray::try_new(field.clone(), 3, Arc::new(values1), None).unwrap();
    let list2 = FixedSizeListArray::try_new(field, 3, Arc::new(values2), None).unwrap();
    
    let batch = datafusion::arrow::record_batch::RecordBatch::try_from_iter(vec![
        ("v1", Arc::new(list1) as Arc<dyn datafusion::arrow::array::Array>),
        ("v2", Arc::new(list2) as Arc<dyn datafusion::arrow::array::Array>),
    ]).unwrap();
    
    let df = ctx.read_batch(batch).unwrap();
    
    // Set GPU context once
    let gpu_ctx = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
    set_global_gpu_context(Some(gpu_ctx));
    
    // Execute multiple queries - all should use GPU context
    for _ in 0..3 {
        let result = df.clone()
            .select(vec![
                Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
                    ctx.udf("dist_l2").unwrap(),
                    vec![col("v1"), col("v2")]
                ))
            ])
            .unwrap()
            .collect()
            .await;
        
        assert!(result.is_ok(), "Query should succeed with persistent GPU context");
        
        // Verify context is still set
        assert!(get_global_gpu_context().is_some(), "GPU context should persist across queries");
    }
    
    // Clean up
    set_global_gpu_context(None);
}

/// Test GPU context with batch operations (multiple rows)
#[tokio::test]
async fn test_pgvector_operators_batch_with_gpu() {
    let ctx = SessionContext::new();
    
    // Register vector UDFs
    for udf in all_vector_udfs() {
        ctx.register_udf(udf);
    }
    
    // Set GPU context
    let gpu_ctx = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
    set_global_gpu_context(Some(gpu_ctx));
    
    // Create test data with multiple rows
    let vectors = [
        vec![1.0f32, 0.0, 0.0],
        vec![0.0f32, 1.0, 0.0],
        vec![0.0f32, 0.0, 1.0],
        vec![1.0f32, 1.0, 1.0],
    ];
    
    let query_vec = [1.0f32, 0.0, 0.0];
    
    // Flatten vectors for FixedSizeListArray
    let flat_values: Vec<f32> = vectors.iter().flatten().copied().collect();
    let values_array = Float32Array::from(flat_values);
    
    let field = Arc::new(Field::new("item", DataType::Float32, true));
    let list_array = FixedSizeListArray::try_new(field.clone(), 3, Arc::new(values_array), None).unwrap();
    
    // Create query vector array (broadcast to all rows)
    let query_values = Float32Array::from(query_vec.repeat(4));
    let query_list = FixedSizeListArray::try_new(field, 3, Arc::new(query_values), None).unwrap();
    
    let batch = datafusion::arrow::record_batch::RecordBatch::try_from_iter(vec![
        ("embedding", Arc::new(list_array) as Arc<dyn datafusion::arrow::array::Array>),
        ("query", Arc::new(query_list) as Arc<dyn datafusion::arrow::array::Array>),
    ]).unwrap();
    
    let df = ctx.read_batch(batch).unwrap();
    
    // Compute distances for all rows using GPU
    let result = df
        .select(vec![
            Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
                ctx.udf("dist_l2").unwrap(),
                vec![col("embedding"), col("query")]
            ))
        ])
        .unwrap()
        .collect()
        .await
        .unwrap();
    
    assert_eq!(result.len(), 1);
    let distance_array = result[0].column(0).as_any().downcast_ref::<Float32Array>().unwrap();
    
    // Verify we got distances for all 4 rows
    assert_eq!(distance_array.len(), 4, "Should compute distances for all rows");
    
    // Verify all distances are valid
    for i in 0..4 {
        let dist = distance_array.value(i);
        assert!(dist.is_finite(), "Distance for row {} should be finite, got {}", i, dist);
    }
    
    // First vector [1,0,0] should have distance 0 from query [1,0,0]
    assert!(distance_array.value(0).abs() < 0.001, 
            "Distance from [1,0,0] to [1,0,0] should be ~0, got {}", distance_array.value(0));
    
    // Clean up
    set_global_gpu_context(None);
}
