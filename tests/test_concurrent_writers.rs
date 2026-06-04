// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use arrow::array::{FixedSizeListArray, Float32Array, Int32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use hyperstreamdb::Table;
use std::sync::Arc;

async fn create_test_batch(start_id: i32, num_rows: usize) -> RecordBatch {
    let dim = 4;
    let id_array = Int32Array::from_iter_values(start_id..start_id + num_rows as i32);

    let mut values = Vec::with_capacity(num_rows * dim);
    for i in 0..num_rows {
        for j in 0..dim {
            values.push((i + j) as f32);
        }
    }
    let values_array = Float32Array::from(values);
    let vectors_array = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
        Arc::new(values_array),
        None,
    )
    .unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]));

    RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(vectors_array)]).unwrap()
}

#[tokio::test]
async fn test_massive_concurrent_writers() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

    // 1. Initialize Table
    let table = Arc::new(Table::new_async(uri.clone()).await?);

    // Establishing schema first!
    let initial_batch = create_test_batch(0, 1).await;
    table.write_async(vec![initial_batch]).await?;
    table.commit_async().await?;

    let num_writers = 50; // High concurrency
    let rows_per_batch = 10;

    // 2. Spawn Massive Writers
    let mut writer_handles = Vec::new();
    for w in 0..num_writers {
        let t = table.clone();
        let handle = tokio::spawn(async move {
            let start_id = (w as i32 * 1000) + 1;
            let batch = create_test_batch(start_id, rows_per_batch).await;
            t.write_async(vec![batch]).await.unwrap();
            t.commit_async().await.unwrap();
        });
        writer_handles.push(handle);
    }

    // Wait for all to finish
    for h in writer_handles {
        h.await?;
    }

    // 3. Final validation
    let batches = table.sql("SELECT count(*) FROM t").await?;
    let count = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap()
        .value(0);

    let expected_count = (num_writers * rows_per_batch + 1) as i64;
    println!("Final count: {}, Expected: {}", count, expected_count);
    assert_eq!(count, expected_count);

    Ok(())
}
