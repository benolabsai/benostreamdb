use anyhow::Result;
use arrow::array::{Int32Array, Float32Array, FixedSizeListArray};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{Schema, DataType, Field};
use hyperstreamdb::Table;
use hyperstreamdb::core::table::VectorSearchParams;
use std::sync::Arc;
use tokio::time::{sleep, Duration};

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
        None
    ).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("embedding", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32
        ), false),
    ]));

    RecordBatch::try_new(schema, vec![
        Arc::new(id_array),
        Arc::new(vectors_array),
    ]).unwrap()
}

#[tokio::test]
async fn test_high_concurrency_readers_writers() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
    
    // 1. Initialize Table
    let table = Arc::new(Table::new_async(uri.clone()).await?);
    
    // Establishing schema first!
    let initial_batch = create_test_batch(0, 1).await;
    table.write_async(vec![initial_batch]).await?;
    table.commit_async().await?;
    
    let num_writers = 5;
    let batches_per_writer = 10;
    let rows_per_batch = 100;
    
    // 2. Spawn Writers
    let mut writer_handles = Vec::new();
    for w in 0..num_writers {
        let t = table.clone();
        let handle = tokio::spawn(async move {
            for b in 0..batches_per_writer {
                let start_id = (w as i32 * 1000) + (b as i32 * rows_per_batch as i32) + 1; // +1 to avoid overlap with initial
                let batch = create_test_batch(start_id, rows_per_batch).await;
                t.write_async(vec![batch]).await.unwrap();
                t.commit_async().await.unwrap();
                sleep(Duration::from_millis(10)).await;
            }
        });
        writer_handles.push(handle);
    }
    
    // 3. Spawn Readers
    let num_readers = 10;
    let mut reader_handles = Vec::new();
    for _ in 0..num_readers {
        let t = table.clone();
        let handle = tokio::spawn(async move {
            for _ in 0..20 {
                // Mix of SQL and Vector search
                let _ = t.sql("SELECT * FROM t WHERE id > 0").await.unwrap();
                
                let query_vec = vec![0.5; 4];
                let vs_params = VectorSearchParams::new("embedding", query_vec, 5);
                let _ = t.read_async(None, Some(vs_params), None).await.unwrap();
                
                sleep(Duration::from_millis(15)).await;
            }
        });
        reader_handles.push(handle);
    }
    
    // 4. Spawn a Compactor
    let t_compactor = table.clone();
    let compactor_handle = tokio::spawn(async move {
        for _ in 0..3 {
            sleep(Duration::from_secs(1)).await;
            println!("Triggering background compaction...");
            let _ = t_compactor.compact(None);
        }
    });
    
    // Wait for all to finish
    for h in writer_handles { h.await?; }
    for h in reader_handles { h.await?; }
    compactor_handle.await?;
    
    // 5. Final validation
    let batches = table.sql("SELECT count(*) FROM t").await?;
    let count = batches[0].column(0).as_any().downcast_ref::<arrow::array::Int64Array>().unwrap().value(0);
    
    let expected_count = (num_writers * batches_per_writer * rows_per_batch + 1) as i64;
    println!("Final count: {}, Expected: {}", count, expected_count);
    assert_eq!(count, expected_count);
    
    Ok(())
}
