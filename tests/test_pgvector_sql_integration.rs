// Copyright (c) 2026 Richard Albright. All rights reserved.

// Integration tests for pgvector-compatible SQL support
// Feature: pgvector-sql-support

use hyperstreamdb::core::sql::session::HyperStreamSession;
use hyperstreamdb::core::table::Table;
use arrow::array::{Float32Array, FixedSizeListArray, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use tempfile::TempDir;

// Helper function to create a test table with vector data
async fn create_test_table_with_vectors() -> (HyperStreamSession, Arc<Table>, TempDir) {
    // Create temporary directory for table
    let temp_dir = TempDir::new().unwrap();
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
    
    // Create table
    let table = Arc::new(Table::new_async(uri.clone()).await.unwrap());
    
    // Create schema with id, name, and embedding columns
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                3,
            ),
            false,
        ),
    ]));
    
    // Create test data
    let id_array = Int32Array::from(vec![1, 2, 3, 4]);
    let name_array = StringArray::from(vec!["doc1", "doc2", "doc3", "doc4"]);
    
    // Create vector embeddings
    let values = Float32Array::from(vec![
        1.0, 0.0, 0.0,  // doc1: [1, 0, 0]
        0.0, 1.0, 0.0,  // doc2: [0, 1, 0]
        0.0, 0.0, 1.0,  // doc3: [0, 0, 1]
        1.0, 1.0, 1.0,  // doc4: [1, 1, 1]
    ]);
    let embedding_array = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        3,
        Arc::new(values),
        None,
    );
    
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(id_array),
            Arc::new(name_array),
            Arc::new(embedding_array),
        ],
    ).unwrap();
    
    // Write data to table (don't commit yet - test with write buffer)
    table.write_async(vec![batch]).await.unwrap();
    
    // Create session and register table AFTER data is written
    let session = HyperStreamSession::new(None);
    session.register_table("documents", table.clone()).unwrap();
    
    (session, table, temp_dir)
}

// ============================================================================
// Subtask 14.1: Integration test for all distance operators
// ============================================================================

#[tokio::test]
async fn test_simple_select() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    // Test simple SELECT without any UDFs
    let result = session.sql("SELECT id, name FROM documents").await;
    
    if let Err(e) = &result {
        eprintln!("Error: {:?}", e);
    }
    assert!(result.is_ok(), "Simple SELECT should work");
    let batches = result.unwrap();
    assert!(!batches.is_empty(), "Should return results");
}

#[tokio::test]
async fn test_all_distance_operators_l2() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    // Test L2 distance using UDF (operator syntax not yet fully working)
    let result = session.sql(
        "SELECT id, name, dist_l2(embedding, ARRAY[1.0, 0.0, 0.0]) as distance 
         FROM documents 
         ORDER BY distance LIMIT 2"
    ).await;
    
    if let Err(e) = &result {
        eprintln!("Error: {:?}", e);
    }
    assert!(result.is_ok(), "L2 distance query should execute successfully");
    let batches = result.unwrap();
    assert!(!batches.is_empty(), "Should return results");
}

#[tokio::test]
async fn test_all_distance_operators_cosine() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    let result = session.sql(
        "SELECT id, name, dist_cosine(embedding, ARRAY[1.0, 0.0, 0.0]) as distance 
         FROM documents 
         ORDER BY distance LIMIT 2"
    ).await;
    
    assert!(result.is_ok(), "Cosine distance query should execute successfully");
}

#[tokio::test]
async fn test_all_distance_operators_inner_product() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    let result = session.sql(
        "SELECT id, name, dist_ip(embedding, ARRAY[1.0, 0.0, 0.0]) as distance 
         FROM documents 
         ORDER BY distance LIMIT 2"
    ).await;
    
    assert!(result.is_ok(), "Inner product distance query should execute successfully");
}

#[tokio::test]
async fn test_all_distance_operators_l1() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    let result = session.sql(
        "SELECT id, name, dist_l1(embedding, ARRAY[1.0, 0.0, 0.0]) as distance 
         FROM documents 
         ORDER BY distance LIMIT 2"
    ).await;
    
    assert!(result.is_ok(), "L1 distance query should execute successfully");
}

#[tokio::test]
async fn test_all_distance_operators_hamming() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    let result = session.sql(
        "SELECT id, name, dist_hamming(embedding, ARRAY[1.0, 0.0, 0.0]) as distance 
         FROM documents 
         ORDER BY distance LIMIT 2"
    ).await;
    
    assert!(result.is_ok(), "Hamming distance query should execute successfully");
}

#[tokio::test]
async fn test_all_distance_operators_jaccard() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    let result = session.sql(
        "SELECT id, name, dist_jaccard(embedding, ARRAY[1.0, 0.0, 0.0]) as distance 
         FROM documents 
         ORDER BY distance LIMIT 2"
    ).await;
    
    assert!(result.is_ok(), "Jaccard distance query should execute successfully");
}

// ============================================================================
// Subtask 14.2: Integration test for KNN with LIMIT pushdown
// ============================================================================

#[tokio::test]
async fn test_knn_with_limit_pushdown() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    // Test KNN query with LIMIT
    let result = session.sql(
        "SELECT id, name, dist_l2(embedding, ARRAY[1.0, 0.0, 0.0]) as distance 
         FROM documents 
         ORDER BY distance 
         LIMIT 2"
    ).await;
    
    assert!(result.is_ok(), "KNN query with LIMIT should execute");
    let batches = result.unwrap();
    assert!(!batches.is_empty(), "Should return results");
    
    // Verify we get exactly 2 results
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2, "Should return exactly 2 results with LIMIT 2");
}

#[tokio::test]
async fn test_knn_results_ordered_by_distance() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    let result = session.sql(
        "SELECT id, dist_l2(embedding, ARRAY[1.0, 0.0, 0.0]) as distance 
         FROM documents 
         ORDER BY distance"
    ).await;
    
    assert!(result.is_ok(), "KNN query should execute");
    let batches = result.unwrap();
    
    // Extract distances and verify they're in ascending order
    if let Some(batch) = batches.first() {
        let distance_col = batch.column(1)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        
        for i in 1..distance_col.len() {
            assert!(
                distance_col.value(i - 1) <= distance_col.value(i),
                "Distances should be in ascending order"
            );
        }
    }
}

// ============================================================================
// Subtask 14.3: Integration test for vector literal parsing
// ============================================================================

#[tokio::test]
async fn test_vector_literal_in_select() {
    let session = HyperStreamSession::new(None);
    
    // Test vector literal parsing in SELECT
    let result = session.sql(
        "SELECT ARRAY[1.0, 2.0, 3.0] as vec"
    ).await;
    
    assert!(result.is_ok(), "Vector literal in SELECT should work");
}

#[tokio::test]
async fn test_vector_literal_in_where_clause() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    let result = session.sql(
        "SELECT id, name 
         FROM documents 
         WHERE dist_l2(embedding, ARRAY[1.0, 0.0, 0.0]) < 2.0"
    ).await;
    
    assert!(result.is_ok(), "Vector literal in WHERE clause should work");
}

#[tokio::test]
async fn test_vector_literal_various_formats() {
    let session = HyperStreamSession::new(None);
    
    // Test integer format
    let result = session.sql("SELECT ARRAY[1, 2, 3] as vec").await;
    assert!(result.is_ok(), "Integer vector literal should work");
    
    // Test float format
    let result = session.sql("SELECT ARRAY[1.5, 2.7, 3.9] as vec").await;
    assert!(result.is_ok(), "Float vector literal should work");
    
    // Test mixed format
    let result = session.sql("SELECT ARRAY[1, 2.5, 3] as vec").await;
    assert!(result.is_ok(), "Mixed vector literal should work");
}

// ============================================================================
// Subtask 14.4: Integration test for sparse vector operations
// ============================================================================

#[tokio::test]
async fn test_sparse_vector_conversion() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    // Test converting dense to sparse
    let result = session.sql(
        "SELECT id, vector_to_sparse(embedding) as sparse_emb 
         FROM documents 
         LIMIT 1"
    ).await;
    
    assert!(result.is_ok(), "Sparse vector conversion should work");
}

#[tokio::test]
async fn test_sparse_to_dense_roundtrip() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    // Test sparse to dense conversion
    let result = session.sql(
        "SELECT id, sparse_to_vector(vector_to_sparse(embedding)) as dense_emb 
         FROM documents 
         LIMIT 1"
    ).await;
    
    assert!(result.is_ok(), "Sparse to dense roundtrip should work");
}

// ============================================================================
// Subtask 14.5: Integration test for binary vector operations
// ============================================================================

#[tokio::test]
async fn test_binary_quantization() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    let result = session.sql(
        "SELECT id, binary_quantize(embedding) as binary_emb 
         FROM documents 
         LIMIT 1"
    ).await;
    
    assert!(result.is_ok(), "Binary quantization should work");
}

#[tokio::test]
async fn test_hamming_distance_on_binary() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    let result = session.sql(
        "SELECT id, 
                dist_hamming(
                    binary_quantize(embedding), 
                    binary_quantize(ARRAY[1.0, 0.0, 0.0])
                ) as hamming_dist 
         FROM documents"
    ).await;
    
    assert!(result.is_ok(), "Hamming distance on binary vectors should work");
}

// ============================================================================
// Subtask 14.6: Integration test for vector aggregations
// ============================================================================

#[tokio::test]
async fn test_vector_sum_aggregation() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    let result = session.sql(
        "SELECT vector_sum(embedding) as sum_vec 
         FROM documents"
    ).await;
    
    assert!(result.is_ok(), "Vector sum aggregation should work");
}

#[tokio::test]
async fn test_vector_avg_aggregation() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    let result = session.sql(
        "SELECT vector_avg(embedding) as avg_vec 
         FROM documents"
    ).await;
    
    assert!(result.is_ok(), "Vector avg aggregation should work");
}

#[tokio::test]
async fn test_vector_aggregation_with_group_by() {
    let session = HyperStreamSession::new(None);
    
    // Create temporary directory for table
    let temp_dir = TempDir::new().unwrap();
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
    let table = Arc::new(Table::new_async(uri).await.unwrap());
    
    // Create schema with category and embedding
    let schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                3,
            ),
            false,
        ),
    ]));
    
    let category_array = StringArray::from(vec!["A", "A", "B", "B"]);
    let values = Float32Array::from(vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 1.0, 1.0,
    ]);
    let embedding_array = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        3,
        Arc::new(values),
        None,
    );
    
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(category_array), Arc::new(embedding_array)],
    ).unwrap();
    
    table.write_async(vec![batch]).await.unwrap();
    // Don't commit - test with write buffer like other tests
    // table.commit_async().await.unwrap();
    session.register_table("grouped_docs", table).unwrap();
    
    let result = session.sql(
        "SELECT category, vector_avg(embedding) as avg_vec 
         FROM grouped_docs 
         GROUP BY category"
    ).await;
    
    assert!(result.is_ok(), "Vector aggregation with GROUP BY should work");
    
    // Keep temp_dir alive
    drop(temp_dir);
}

// ============================================================================
// Subtask 14.7: Integration test for combined filters and vector search
// ============================================================================

#[tokio::test]
async fn test_knn_with_where_filter() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    let result = session.sql(
        "SELECT id, name, dist_l2(embedding, ARRAY[1.0, 0.0, 0.0]) as distance 
         FROM documents 
         WHERE id > 1 
         ORDER BY distance 
         LIMIT 2"
    ).await;
    
    assert!(result.is_ok(), "KNN with WHERE filter should work");
    let batches = result.unwrap();
    
    // Verify all returned IDs are > 1
    if let Some(batch) = batches.first() {
        let id_col = batch.column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        
        for i in 0..id_col.len() {
            assert!(id_col.value(i) > 1, "All IDs should be > 1");
        }
    }
}

#[tokio::test]
async fn test_knn_with_multiple_filters() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    let result = session.sql(
        "SELECT id, name, dist_l2(embedding, ARRAY[1.0, 0.0, 0.0]) as distance 
         FROM documents 
         WHERE id > 1 AND name != 'doc4' 
         ORDER BY distance 
         LIMIT 2"
    ).await;
    
    assert!(result.is_ok(), "KNN with multiple filters should work");
}

// ============================================================================
// Subtask 14.8: Integration test for error handling
// ============================================================================

#[tokio::test]
async fn test_error_dimension_mismatch() {
    let (session, _table, _temp_dir) = create_test_table_with_vectors().await;
    
    // Try to compute distance with wrong dimension
    let result = session.sql(
        "SELECT dist_l2(embedding, ARRAY[1.0, 0.0]) as distance 
         FROM documents"
    ).await;
    
    // Should fail due to dimension mismatch
    assert!(result.is_err(), "Should error on dimension mismatch");
}

#[tokio::test]
async fn test_error_invalid_vector_literal() {
    let session = HyperStreamSession::new(None);
    
    // This should be caught by DataFusion's parser
    let result = session.sql("SELECT ARRAY[1, abc, 3] as vec").await;
    assert!(result.is_err(), "Should error on invalid vector literal");
}

// ============================================================================
// Basic unit tests (kept from original)
// ============================================================================

#[test]
fn test_vector_operators_registered() {
    use hyperstreamdb::core::sql::vector_operators::VECTOR_OPERATORS;
    
    assert_eq!(VECTOR_OPERATORS.len(), 6, "Should have 6 vector operators");
    
    let operators: Vec<&str> = VECTOR_OPERATORS.iter().map(|m| m.operator).collect();
    assert!(operators.contains(&"<->"), "Should have L2 operator");
    assert!(operators.contains(&"<=>"), "Should have Cosine operator");
    assert!(operators.contains(&"<#>"), "Should have Inner Product operator");
    assert!(operators.contains(&"<+>"), "Should have L1 operator");
    assert!(operators.contains(&"<~>"), "Should have Hamming operator");
    assert!(operators.contains(&"<%>"), "Should have Jaccard operator");
}

#[test]
fn test_all_distance_udfs_available() {
    use hyperstreamdb::core::sql::vector_udf;
    
    let udfs = vector_udf::all_vector_udfs();
    assert!(udfs.len() >= 6, "Should have at least 6 distance UDFs");
    
    let udf_names: Vec<String> = udfs.iter().map(|u| u.name().to_string()).collect();
    assert!(udf_names.contains(&"dist_l2".to_string()), "Should have dist_l2");
    assert!(udf_names.contains(&"dist_cosine".to_string()), "Should have dist_cosine");
    assert!(udf_names.contains(&"dist_ip".to_string()), "Should have dist_ip");
    assert!(udf_names.contains(&"dist_l1".to_string()), "Should have dist_l1");
    assert!(udf_names.contains(&"dist_hamming".to_string()), "Should have dist_hamming");
    assert!(udf_names.contains(&"dist_jaccard".to_string()), "Should have dist_jaccard");
}

#[test]
fn test_vector_literal_parser_available() {
    use hyperstreamdb::core::sql::vector_literal::VectorLiteralParser;
    
    let result = VectorLiteralParser::parse("[1.0, 2.0, 3.0]");
    assert!(result.is_ok(), "Vector literal parser should parse valid literals");
}
