// Copyright (c) 2026 Richard Albright. All rights reserved.

use hyperstreamdb::Table;
use std::sync::Arc;

#[tokio::test]
async fn test_datafusion_integration() -> Result<(), Box<dyn std::error::Error>> {
    let path = "/tmp/hs_df_test_rust_v2";
    let _ = std::fs::remove_dir_all(path);
    // Use async constructor to avoid blocking thread
    let table = Table::new_async(format!("file://{}", path)).await?;

    // Create some data using Arrow/Pandas (simulated)
    use datafusion::arrow::array::{Int32Array, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::arrow::record_batch::RecordBatch;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(datafusion::arrow::array::Int64Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec!["A", "B", "C", "D", "E"])),
        ],
    )?;

    // Write async (updates schema cache)
    table.write_async(vec![batch]).await?;

    // High-level SQL API
    // "t" is the default table name registered inside sql()
    let batches = table.sql("SELECT * FROM t WHERE id > 1").await?;
    assert!(!batches.is_empty());
    
    // Test Limit/Offset
    let limited = table.sql("SELECT * FROM t ORDER BY id LIMIT 2 OFFSET 1").await?;
    // Data: 1, 2, 3, 4, 5. Filter > 1 -> 2,3,4,5. 
    // Wait, previous test was just "SELECT *". 
    // Testing specific Limit:
    
    let batch = &limited[0]; // Assuming single batch for small data
    assert_eq!(batch.num_rows(), 2);
    // Offset 1 means skip "A" (id=1)? No, Order By ID.
    // 1, 2, 3, 4, 5.
    // Offset 1 skips 1. Starts at 2.
    // Limit 2 takes 2, 3.
    let ids = batch.column(0).as_any().downcast_ref::<datafusion::arrow::array::Int64Array>().unwrap();
    assert_eq!(ids.value(0), 2);
    assert_eq!(ids.value(1), 3);

    // Test Joins via Session
    use hyperstreamdb::core::sql::session::HyperStreamSession;
    let session = HyperStreamSession::new(None);
    session.register_table("t1", Arc::new(table.clone()))?;
    
    // Create second table
    let path2 = "/tmp/hs_df_test_rust_v2_orders";
    let _ = std::fs::remove_dir_all(path2);
    let table2 = Table::new_async(format!("file://{}", path2)).await?;
    
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("user_id", DataType::Int64, false),
        Field::new("amount", DataType::Float64, false),
    ]));
    
    let batch2 = RecordBatch::try_new(
        schema2.clone(),
        vec![
            Arc::new(datafusion::arrow::array::Int64Array::from(vec![2, 4])),
            Arc::new(datafusion::arrow::array::Float64Array::from(vec![20.5, 40.0])),
        ],
    )?;
    table2.write_async(vec![batch2]).await?;
    
    session.register_table("t2", Arc::new(table2))?;
    
    let joined = session.sql("SELECT t1.id, t1.name, t2.amount FROM t1 JOIN t2 ON t1.id = t2.user_id ORDER BY t1.id").await?;
    // t1: 1, 2, 3, 4, 5
    // t2: 2, 4
    // Matches: 2, 4
    assert!(!joined.0.is_empty());
    let j_batch = &joined.0[0];
    assert_eq!(j_batch.num_rows(), 2);
    
    println!("Rust SQL Test & Join Passed!");

    Ok(())
}
