// Copyright (c) 2026 Richard Albright. All rights reserved.
#![cfg(feature = "java")]

use hyperstreamdb::Table;
use hyperstreamdb::core::ffi::HyperStreamSession;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::array::{Int32Array, StringArray};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use std::path::Path;

async fn create_test_table(uri: &str, _row_count: usize) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Create Schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    // 2. Clear old data
    if uri.starts_with("file://") {
        let path = uri.strip_prefix("file://").unwrap();
        if Path::new(path).exists() {
            std::fs::remove_dir_all(path)?;
        }
    }

    // 3. Create Table
    let table = Table::create_async(uri.to_string(), schema.clone()).await?;

    // 4. Write Data (2 commits)
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["A", "B"])),
        ],
    )?;
    table.write_async(vec![batch1]).await?;
    table.commit_async().await?;

    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![3, 4, 5])),
            Arc::new(StringArray::from(vec!["C", "D", "E"])),
        ],
    )?;
    table.write_async(vec![batch2]).await?;
    table.commit_async().await?;

    Ok(())
}

#[tokio::test]
async fn test_connector_simulation() -> Result<(), Box<dyn std::error::Error>> {
    let table_uri = "file:///tmp/test_connector_ffi";
    create_test_table(table_uri, 5).await?;

    // Coordinator
    let table = Table::new_async(table_uri.to_string()).await?;
    let splits = table.get_splits_async(1024).await?; 
    
    println!("Got {} splits", splits.len());
    // Expect at least 2 splits (one per file/commit)
    assert!(splits.len() >= 2, "Expected at least 2 splits for 2 segments");

    // Worker Simulation
    let (total_rows, seen_ids): (usize, std::collections::HashSet<i32>) = tokio::task::spawn_blocking(move || {
        let mut total_rows = 0;
        let mut seen_ids = std::collections::HashSet::new();

        for split in splits {
            println!("Processing Split: {:?}", split);
            let path = &split.file_path;
            
            // Note: In real Spark/Trino, this happens in a thread that is NOT a Tokio async runtime thread.
            // Hence why HyperStreamSession uses its own RUNTIME.block_on internally.
            // spawn_blocking moves us to a thread where blocking is allowed.
            let mut session = HyperStreamSession::new(path).expect("Failed to create session");
            
            while let Some(batch) = session.next_batch() {
                 println!("Read batch with {} rows", batch.num_rows());
                 total_rows += batch.num_rows();
                 
                 let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                 for i in 0..ids.len() {
                     seen_ids.insert(ids.value(i));
                 }
            }
        }
        (total_rows, seen_ids)
    }).await?;

    // Verification
    assert_eq!(total_rows, 5, "Total rows read should be 5");
    assert!(seen_ids.contains(&1));
    assert!(seen_ids.contains(&5));
    
    println!("Connector Simulation Passed!");
    Ok(())
}
