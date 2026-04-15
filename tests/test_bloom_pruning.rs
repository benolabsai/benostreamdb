use hyperstreamdb::core::table::Table;
use arrow::array::{Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use tempfile::tempdir;

#[tokio::test]
async fn test_bloom_filter_general_query_pruning() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();
    let tmp = tempdir()?;
    let uri = format!("file://{}", tmp.path().to_str().unwrap());
    
    // 1. Schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));
    
    let table = Table::create_async(uri.clone(), schema.clone()).await?;
    table.set_indexed_columns(vec!["id".to_string()]);

    // 2. Write 3 separate segments
    // Segment 1: ids 1-10
    let b1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))])?;
    table.write_async(vec![b1]).await?;
    table.commit_async().await?;
    
    // Segment 2: ids 11-20
    let b2 = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![11, 12, 13, 14, 15, 16, 17, 18, 19, 20]))])?;
    table.write_async(vec![b2]).await?;
    table.commit_async().await?;

    // Segment 3: ids 21-30
    let b3 = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![21, 22, 23, 24, 25, 26, 27, 28, 29, 30]))])?;
    table.write_async(vec![b3]).await?;
    table.commit_async().await?;

    table.wait_for_background_tasks_async().await?;

    println!("--- Table state: 3 segments committed ---");
    // We'll use tracing or just check the number of rows/batches
    let results = table.read_async(Some("id = 5"), None, None).await?;
    
    assert_eq!(results.len(), 1, "Should find 1 batch");
    assert_eq!(results[0].num_rows(), 1, "Should find 1 row");
    
    // Verify using "explain" (which we just enabled Bloom stats for)
    let explanation = table.explain(Some("id = 5"), None).await;
    println!("Explanation:\n{}", explanation);
    
    // Note: HybridReader logs "Bloom Filter Pruned segment" to debug!
    // If it's pruned, it won't even try to read Parquet.
    
    println!("--- Executing Query for id=99 (Should skip ALL segments) ---");
    let results_none = table.read_async(Some("id = 99"), None, None).await?;
    assert!(results_none.is_empty(), "Should return no results");
    
    Ok(())
}
