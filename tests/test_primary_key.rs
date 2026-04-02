// Copyright (c) 2026 Richard Albright. All rights reserved.

use hyperstreamdb::core::table::Table;
use arrow::record_batch::RecordBatch;
use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use tempfile::tempdir;

#[tokio::test]
async fn test_primary_key_uniqueness_enforcement() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    // 1. Setup Table with Primary Key
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));
    
    let table = Table::create_async(uri.clone(), schema.clone()).await?;
    
    // Set the PK explicitly for this test
    table.set_primary_key(vec!["id".to_string()]);
    
    // 2. Write first batch (valid)
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))]
    )?;
    table.write_async(vec![batch1]).await?;
    
    // IMPORTANT: Commit and wait for background indexing to finish
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;
    
    // 3. Write second batch with DUPLICATE (id=3 already exists on disk)
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![3, 4, 5]))]
    )?;
    
    let res = table.write_async(vec![batch2]).await;
    
    assert!(res.is_err(), "Should have failed due to duplicate PK '3'");
    let err_msg = res.unwrap_err().to_string();
    println!("Caught expected error: {}", err_msg);
    assert!(err_msg.contains("Duplicate primary key error"));
    assert!(err_msg.contains("id = 3"));
    
    Ok(())
}

#[tokio::test]
async fn test_compound_primary_key_uniqueness() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let schema = Arc::new(Schema::new(vec![
        Field::new("pk1", DataType::Int32, false),
        Field::new("pk2", DataType::Int32, false),
        Field::new("val", DataType::Int32, true),
    ]));
    
    let table = Table::create_async(uri.clone(), schema.clone()).await?;
    table.set_primary_key(vec!["pk1".to_string(), "pk2".to_string()]);
    
    // Write (1, 100)
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![100])),
            Arc::new(Int32Array::from(vec![10])),
        ]
    )?;
    table.write_async(vec![batch1]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;
    
    // Write (1, 200) - Should SUCCEED (partial overlap)
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![200])),
            Arc::new(Int32Array::from(vec![20])),
        ]
    )?;
    table.write_async(vec![batch2]).await?;
    
    // Write (1, 100) - Should FAIL (exact overlap)
    let batch3 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1])),
            Arc::new(Int32Array::from(vec![100])),
            Arc::new(Int32Array::from(vec![30])),
        ]
    )?;
    
    let res = table.write_async(vec![batch3]).await;
    assert!(res.is_err(), "Should have failed due to duplicate compound PK (1, 100)");
    assert!(res.unwrap_err().to_string().contains("Duplicate primary key error"));
    
    Ok(())
}
