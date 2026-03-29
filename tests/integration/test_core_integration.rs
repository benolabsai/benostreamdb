// Copyright (c) 2026 Richard Albright. All rights reserved.

use hyperstream::{Table, RecordBatch};
use hyperstream::manifest::ManifestEntry;
use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use tempfile::tempdir;
use anyhow::Result;

#[tokio::test]
async fn test_full_table_lifecycle_e2e() -> Result<()> {
    // 1. Setup Table on Local FS
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);
    // Add wait duration for FS consistency if needed, but usually fine on local
    
    let table = Table::new_async(uri.clone()).await?;

    // 2. Define Schema & Data
    let schema = Arc::new(Schema::new(vec![
        Field::new("user_id", DataType::Int32, false),
        Field::new("event", DataType::Utf8, false),
    ]));

    // Batch 1: users 1-10
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from((1..=10).collect::<Vec<i32>>())),
            Arc::new(StringArray::from(vec!["login"; 10])),
        ]
    )?;

    // Batch 2: users 11-20
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from((11..=20).collect::<Vec<i32>>())),
            Arc::new(StringArray::from(vec!["logout"; 10])),
        ]
    )?;

    // 3. Write Data
    table.write(vec![batch1])?;
    table.commit_async().await?;

    table.write(vec![batch2])?;
    table.commit_async().await?;

    // 4. Read Verification (Scan All)
    let all_batches = table.read_async(None, None, None).await?;
    let total_rows: usize = all_batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 20);

    // 5. Read with Filter (Simulated Pushdown)
    // Assuming the table supports simple filtering or the mocks do.
    // Ideally, we'd test the `table.read_with_filter` if exposed perfectly.
    // For now, we trust `read_async` optional filter argument if implemented.
    
    // Test Index Coverage Check
    let stats = table.get_table_statistics().await?;
    println!("Table Stats: {:?}", stats);
    assert_eq!(stats.file_count, 2);
    assert_eq!(stats.row_count, 20);

    Ok(())
}
