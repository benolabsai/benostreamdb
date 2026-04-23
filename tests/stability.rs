use hyperstreamdb::Table;
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use tempfile::tempdir;
use futures::StreamExt;

#[tokio::test]
async fn test_streaming_read_stability() {
    let dir = tempdir().unwrap();
    let table_path = dir.path().to_str().unwrap().to_string();
    
    // 1. Create Table
    let table = Table::new_async(table_path).await.unwrap();
    
    // 2. Write some data
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, false),
    ]));
    
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(arrow::array::Int32Array::from(vec![1, 2, 3])),
            Arc::new(arrow::array::StringArray::from(vec!["a", "b", "c"])),
        ],
    ).unwrap();
    
    table.write_async(vec![batch.clone()]).await.unwrap();
    table.commit_async().await.unwrap();
    
    // 3. Test Streaming Read
    let mut stream = table.read_stream_async(None, None, None).await.unwrap();
    let mut count = 0;
    while let Some(batch_res) = stream.next().await {
        let b = batch_res.unwrap();
        count += b.num_rows();
    }
    
    assert_eq!(count, 3);
}

#[tokio::test]
async fn test_wal_incremental_recovery() {
    let dir = tempdir().unwrap();
    let table_path = dir.path().to_str().unwrap().to_string();
    
    {
        // 1. Create Table and write to WAL without committing
        let table = Table::new_async(table_path.clone()).await.unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(arrow::array::Int32Array::from(vec![1, 2, 3]))],
        ).unwrap();
        
        table.write_async(vec![batch]).await.unwrap();
        // Drop table without commit - data remains in WAL
    }
    
    // 2. Re-open table - should recover from WAL
    let table = Table::new_async(table_path).await.unwrap();
    let results = table.read_async(None, None, None).await.unwrap();
    
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 3);
}
