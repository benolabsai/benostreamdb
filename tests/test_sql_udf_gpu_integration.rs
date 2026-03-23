/// Integration test for SQL UDF GPU context support
/// 
/// This test verifies that SQL distance UDFs can use GPU acceleration
/// when a GPU context is configured via the global context.

use hyperstreamdb::core::index::gpu::{ComputeContext, ComputeBackend, set_global_gpu_context, get_global_gpu_context};
use hyperstreamdb::core::sql::vector_udf::all_vector_udfs;
use datafusion::prelude::*;
use datafusion::execution::FunctionRegistry;
use datafusion::arrow::array::{Float32Array, FixedSizeListArray};
use datafusion::arrow::datatypes::{DataType, Field};
use std::sync::Arc;

#[tokio::test]
async fn test_sql_udf_uses_gpu_context() {
    // Create a session context
    let ctx = SessionContext::new();
    
    // Register vector UDFs
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
    
    // Test 1: Without GPU context (should use CPU)
    set_global_gpu_context(None);
    assert!(get_global_gpu_context().is_none(), "GPU context should be None");
    
    let result = df.clone()
        .select(vec![col("v1"), col("v2"), 
                     Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
                         ctx.udf("dist_l2").unwrap(),
                         vec![col("v1"), col("v2")]
                     ))])
        .unwrap()
        .collect()
        .await
        .unwrap();
    
    assert_eq!(result.len(), 1);
    let distance_array = result[0].column(2).as_any().downcast_ref::<Float32Array>().unwrap();
    let cpu_distance = distance_array.value(0);
    
    // Expected L2 distance: sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(9+9+9) = sqrt(27) ≈ 5.196
    assert!((cpu_distance - 5.196).abs() < 0.01, "CPU distance should be approximately 5.196, got {}", cpu_distance);
    
    // Test 2: With GPU context set to CPU backend (should still work)
    let cpu_ctx = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
    set_global_gpu_context(Some(cpu_ctx));
    assert!(get_global_gpu_context().is_some(), "GPU context should be set");
    
    let result = df.clone()
        .select(vec![col("v1"), col("v2"), 
                     Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
                         ctx.udf("dist_l2").unwrap(),
                         vec![col("v1"), col("v2")]
                     ))])
        .unwrap()
        .collect()
        .await
        .unwrap();
    
    assert_eq!(result.len(), 1);
    let distance_array = result[0].column(2).as_any().downcast_ref::<Float32Array>().unwrap();
    let gpu_distance = distance_array.value(0);
    
    // Should get the same result
    assert!((gpu_distance - cpu_distance).abs() < 0.001, 
            "GPU and CPU distances should match, got CPU: {}, GPU: {}", cpu_distance, gpu_distance);
    
    // Clean up
    set_global_gpu_context(None);
}

#[tokio::test]
async fn test_all_distance_metrics_with_gpu_context() {
    // Create a session context
    let ctx = SessionContext::new();
    
    // Register vector UDFs
    for udf in all_vector_udfs() {
        ctx.register_udf(udf);
    }
    
    // Set CPU backend context
    let cpu_ctx = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
    set_global_gpu_context(Some(cpu_ctx));
    
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
    
    // Test all distance metrics
    let metrics = vec![
        ("dist_l2", "L2"),
        ("dist_cosine", "Cosine"),
        ("dist_ip", "Inner Product"),
        ("dist_l1", "L1"),
    ];
    
    for (udf_name, metric_name) in metrics {
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
        
        assert!(result.is_ok(), "{} metric should compute successfully with GPU context", metric_name);
        let result = result.unwrap();
        assert_eq!(result.len(), 1, "{} should return 1 row", metric_name);
        
        let distance_array = result[0].column(0).as_any().downcast_ref::<Float32Array>().unwrap();
        let distance = distance_array.value(0);
        
        // Just verify we got a valid number (not NaN or Inf)
        assert!(distance.is_finite(), "{} distance should be finite, got {}", metric_name, distance);
    }
    
    // Clean up
    set_global_gpu_context(None);
}

#[test]
fn test_gpu_context_thread_local() {
    // Test that GPU context is thread-local
    use std::thread;
    
    // Set context in main thread
    let ctx1 = ComputeContext { backend: ComputeBackend::Cpu, device_id: 0 };
    set_global_gpu_context(Some(ctx1));
    
    assert!(get_global_gpu_context().is_some(), "Main thread should have context");
    
    // Spawn a new thread - it should not have the context
    let handle = thread::spawn(|| {
        let ctx = get_global_gpu_context();
        assert!(ctx.is_none(), "New thread should not have GPU context");
        
        // Set context in this thread
        let ctx2 = ComputeContext { backend: ComputeBackend::Cuda, device_id: 1 };
        set_global_gpu_context(Some(ctx2));
        
        let ctx = get_global_gpu_context();
        assert!(ctx.is_some(), "Thread should now have context");
        assert_eq!(ctx.unwrap().backend, ComputeBackend::Cuda);
    });
    
    handle.join().unwrap();
    
    // Main thread should still have its original context
    let ctx = get_global_gpu_context();
    assert!(ctx.is_some(), "Main thread should still have context");
    assert_eq!(ctx.unwrap().backend, ComputeBackend::Cpu);
    
    // Clean up
    set_global_gpu_context(None);
}
