// Copyright (c) 2026 Richard Albright. All rights reserved.

use hyperstreamdb::core::table::Table;
use arrow::record_batch::RecordBatch;
use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use tempfile::tempdir;


#[tokio::test]
async fn test_bloom_filter_caching() -> anyhow::Result<()> {
    // 1. Initialize Tracing to see the Cache Miss/Hit logs
    let _ = tracing_subscriber::fmt()
        .with_env_filter("hyperstreamdb=debug")
        .with_test_writer()
        .try_init();

    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));
    
    let table = Table::create_async(uri.clone(), schema.clone()).await?;
    table.set_primary_key(vec!["id".to_string()]);

    // 2. Write initial record (id=10)
    let batch1 = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![10]))]
    )?;
    table.write_async(vec![batch1]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    // 3. Perform a check (via write_async with duplicate)
    // This will trigger a Bloom Cache MISS on the first call
    let duplicate_batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![10]))]
    )?;
    
    println!("--- FIRST CHECK (MISS EXPECTED) ---");
    let res1 = table.write_async(vec![duplicate_batch.clone()]).await;
    match &res1 {
        Err(e) => println!("Error 1: {}", e),
        Ok(_) => panic!("Should catch duplicate id=10"),
    }
    assert!(res1.is_err());

    // 4. Perform same check again
    // This should trigger a Bloom Cache HIT
    println!("--- SECOND CHECK (HIT EXPECTED) ---");
    let res2 = table.write_async(vec![duplicate_batch]).await;
    assert!(res2.is_err(), "Should catch duplicate id=10 again");

    // 5. Verify different value (id=20)
    // This should be a MISS (different Bloom block maybe, but definitely different check logic)
    // Actually Bloom check is per-Segment. If it's the same segment, it's the same Bloom filter.
    // So hitting a different value in the SAME SEGMENT should be a HIT for the Bloom Filter object itself!
    // Since the Sbbf object is cached for the segment+offset.
    
    let next_batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![20]))]
    )?;
    println!("--- THIRD CHECK (DIFFERENT VALUE, SAME SEGMENT -> HIT EXPECTED) ---");
    let _ = table.write_async(vec![next_batch]).await;

    Ok(())
}
