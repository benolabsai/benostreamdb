use hyperstreamdb::core::table::Table;
use hyperstreamdb::core::manifest::{PartitionSpec, PartitionField};
use arrow::record_batch::RecordBatch;
use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use std::sync::Arc;
use futures::StreamExt;

#[tokio::test]
async fn test_partitioned_write_and_delete() -> anyhow::Result<()> {
    // 1. Setup Table with Partitioning
    let table_name = "test_partitioned_writes";
    let uri = format!("file:///tmp/{}", table_name);
    let _ = std::fs::remove_dir_all(format!("/tmp/{}", table_name));
    
    // Schema: id (int64), data (string)
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("data", DataType::Utf8, false),
    ]));

    // Partition Spec: Bucket[16] on id
    // Source ID for "id" is 1 (first field)
    let spec = PartitionSpec {
        spec_id: 1,
        fields: vec![
            PartitionField {
                source_id: 1,
                field_id: Some(1000),
                name: "id_bucket".to_string(),
                transform: "bucket[16]".to_string(),
            }
        ]
    };

    // 2. Create Table
    let table = Table::create_partitioned_async(uri.clone(), arrow_schema.clone(), spec).await?;

    // 3. Write Data (100 rows)
    // IDs 0..100
    let ids: Vec<i64> = (0..100).collect();
    let data: Vec<String> = ids.iter().map(|i| format!("val_{}", i)).collect();
    
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            Arc::new(arrow::array::Int64Array::from(ids.clone())),
            Arc::new(StringArray::from(data)),
        ],
    )?;
    
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    // 4. Verify Physical Layout (Partitioned Directories)
    // Should see "id_bucket=N/" folders in data directory?
    // HyperStreamDB currently writes files into `data/` but encodes partition in path?
    // Or uses Hive-style partitioning structure?
    // Checking implementation... Reader uses `partition_values` from manifest.
    // Writer... let's check if it respects partition paths.
    // Ideally it should. If not, verification is harder (must read manifest).
    // But for verification, we just check row counts after delete.

    // 5. Delete specific row (ID=10)
    // The planner should identify which partition ID=10 belongs to, and only scan/rewrite that partition's files?
    // Or write a Position Delete file linked to that partition.
    println!("Deleting id=10...");
    table.delete_async("id = 10").await?;

    // 6. Verify Deletion
    let mut stream = table.stream_all(None).await?;
    let mut count = 0;
    while let Some(batch) = stream.next().await {
        let b = batch?;
        let id_col = b.column(0).as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
        for i in 0..b.num_rows() {
            if id_col.value(i) == 10 {
                panic!("Row with id=10 should have been deleted!");
            }
            count += 1;
        }
    }
    
    assert_eq!(count, 99, "Row count should be 99");
    
    // 7. Test Range Delete crossing partitions (if buckets allow)
    // Deleting 20..30.
    println!("Deleting 20..30");
    table.delete_async("id >= 20 AND id < 30").await?;
    
    let mut count2 = 0;
    let mut stream2 = table.stream_all(None).await?;
    while let Some(batch) = stream2.next().await {
        let b = batch?;
        count2 += b.num_rows();
    }
    assert_eq!(count2, 89, "Row count should be 89");

    Ok(())
}
