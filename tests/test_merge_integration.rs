// Copyright (c) 2026 Richard Albright. All rights reserved.

use hyperstreamdb::Table;
use arrow::array::{Int32Array, StringArray, Float64Array};
use arrow::datatypes::{Schema, Field, DataType};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use tempfile::tempdir;

#[tokio::test]
async fn test_merge_with_disk_segments() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap();
    let uri = format!("file://{}", path);

    let table = Table::new_async(uri.clone()).await?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("value", DataType::Float64, false),
    ]));

    // Write initial segment to disk
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec!["A", "B", "C", "D", "E"])),
            Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0, 5.0])),
        ],
    )?;

    table.write_async(vec![batch1]).await?;
    table.commit_async().await?;

    // Write second segment with overlapping IDs (merge scenario)
    let batch2 = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![3, 4, 5, 6, 7])), // IDs 3,4,5 overlap
            Arc::new(StringArray::from(vec!["C2", "D2", "E2", "F", "G"])),
            Arc::new(Float64Array::from(vec![30.0, 40.0, 50.0, 60.0, 70.0])),
        ],
    )?;

    table.write_async(vec![batch2]).await?;
    table.commit_async().await?;

    // Read all data
    let result = table.read_async(None, None, None).await?;
    let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();

    // Should have 10 rows (5 from each segment)
    // In a real merge/upsert scenario, duplicates would be handled
    assert_eq!(total_rows, 10);

    Ok(())
}

#[tokio::test]
async fn test_merge_large_dataset_disk_spill() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap();
    let uri = format!("file://{}", path);

    let table = Table::new_async(uri.clone()).await?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("data", DataType::Utf8, false),
    ]));

    // Write large dataset (100K rows) that will spill to disk
    let batch_size = 10_000;
    let num_batches = 10;

    for batch_idx in 0..num_batches {
        let start_id = (batch_idx * batch_size) as i32;
        let end_id = start_id + batch_size as i32;
        let ids: Vec<i32> = (start_id..end_id).collect();
        let data: Vec<String> = ids.iter().map(|i| format!("data_{}", i)).collect();
        let data_refs: Vec<&str> = data.iter().map(|s| s.as_str()).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(data_refs)),
            ],
        )?;

        table.write_async(vec![batch]).await?;
    }

    table.commit_async().await?;

    // Verify all data was written
    let result = table.read_async(None, None, None).await?;
    let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();

    assert_eq!(total_rows, (batch_size * num_batches) as usize);

    Ok(())
}

#[tokio::test]
async fn test_merge_memory_pressure() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap();
    let uri = format!("file://{}", path);

    let table = Table::new_async(uri.clone()).await?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]));

    // Create batches with large string payloads to simulate memory pressure
    let num_rows = 1_000;
    let large_string = "x".repeat(1000); // 1KB per row = 1MB total per batch

    let ids: Vec<i32> = (0..num_rows).collect();
    let payloads: Vec<&str> = vec![large_string.as_str(); num_rows as usize];

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(payloads)),
        ],
    )?;

    // Write multiple batches to create memory pressure
    for _ in 0..5 {
        table.write_async(vec![batch.clone()]).await?;
        table.commit_async().await?;
    }

    // Verify data integrity under memory pressure
    let result = table.read_async(None, None, None).await?;
    let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();

    assert_eq!(total_rows, (num_rows * 5) as usize);

    Ok(())
}

#[tokio::test]
async fn test_merge_with_compaction() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap();
    let uri = format!("file://{}", path);

    let table = Table::new_async(uri.clone()).await?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("value", DataType::Int32, false),
    ]));

    // Write multiple small segments
    for i in 0..10 {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![i * 100, i * 100 + 1, i * 100 + 2])),
                Arc::new(Int32Array::from(vec![i, i + 1, i + 2])),
            ],
        )?;

        table.write_async(vec![batch]).await?;
        table.commit_async().await?;
    }

    // Compact to merge segments (pass None for default options)
    table.rewrite_data_files_async(None).await?;

    // Verify data after compaction
    let result = table.read_async(None, None, None).await?;
    let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();

    assert_eq!(total_rows, 30); // 10 segments * 3 rows each

    Ok(())
}

#[tokio::test]
async fn test_merge_preserves_data_integrity() -> Result<(), Box<dyn std::error::Error>> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap();
    let uri = format!("file://{}", path);

    let table = Table::new_async(uri.clone()).await?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec!["A", "B", "A", "C", "B"])),
        ],
    )?;

    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    // Query with filter
    // Cast id to bigint to match literal '2' (Int64)
    let filtered = table.read_async(Some("cast(id as bigint) > 2"), None, None).await?;
    let total_rows: usize = filtered.iter().map(|b| b.num_rows()).sum();

    assert_eq!(total_rows, 3); // IDs 3, 4, 5

    Ok(())
}
