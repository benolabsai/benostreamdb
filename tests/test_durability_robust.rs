// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use arrow::array::Int32Array;
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{Schema, DataType, Field};
use hyperstreamdb::Table;
use std::sync::Arc;

async fn create_simple_batch(start_id: i32, num_rows: usize) -> RecordBatch {
    let id_array = Int32Array::from_iter_values(start_id..start_id + num_rows as i32);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));

    RecordBatch::try_new(schema, vec![
        Arc::new(id_array),
    ]).unwrap()
}

#[tokio::test]
async fn test_wal_crash_recovery_uncommitted() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
    
    // 1. Create table and write data, but DO NOT COMMIT
    {
        let table = Table::new_async(uri.clone()).await?;
        let batch = create_simple_batch(1, 10).await;
        
        // write_async appends to WAL and memory buffer
        table.write_async(vec![batch]).await?;
        
        // We drop the table here. Memory buffer is lost, but WAL should persist.
        // In a real crash, the process would terminate.
    }

    // 2. Re-open the table. It should replay the WAL.
    {
        let table = Table::new_async(uri.clone()).await?;
        
        // Verify that the data is available in the write buffer (recovered from WAL)
        let results = table.read_async(None, None, None).await?;
        let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
        
        assert_eq!(total_rows, 10, "WAL replay should have recovered 10 rows");
        
        // Verify contents
        let mut ids = Vec::new();
        for b in results {
            let id_col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..b.num_rows() {
                ids.push(id_col.value(i));
            }
        }
        ids.sort();
        assert_eq!(ids, (1..11).collect::<Vec<i32>>());
    }

    Ok(())
}

#[tokio::test]
async fn test_wal_truncate_after_commit() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
    
    // 1. Write and COMMIT
    {
        let table = Table::new_async(uri.clone()).await?;
        let batch = create_simple_batch(1, 10).await;
        table.write_async(vec![batch]).await?;
        table.commit_async().await?;
        
        // WAL should be truncated/cleaned up after successful commit
    }

    // 2. Re-open and verify data is still there (from main storage, not WAL)
    {
        let table = Table::new_async(uri.clone()).await?;
        let results = table.read_async(None, None, None).await?;
        let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
        
        assert_eq!(total_rows, 10);
    }
    
    // 3. Write more, DON'T commit, then check WAL recovery again
    {
         let table = Table::new_async(uri.clone()).await?;
         let batch = create_simple_batch(11, 5).await;
         table.write_async(vec![batch]).await?;
    }
    
    {
        let table = Table::new_async(uri.clone()).await?;
        let results = table.read_async(None, None, None).await?;
        let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
        
        // 10 from Parquet + 5 from WAL
        assert_eq!(total_rows, 15, "WAL replay should stack on top of persisted data");
    }

    Ok(())
}
