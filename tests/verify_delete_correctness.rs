use hyperstreamdb::core::table::Table;
// use hyperstreamdb::core::manifest::Schema; // Unused
use arrow::record_batch::RecordBatch;
use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use std::sync::Arc;
use tokio::runtime::Runtime;
use futures::StreamExt;

#[tokio::test]
async fn test_delete_correctness_standard_compliance() -> anyhow::Result<()> {
    // 1. Setup Table
    let table_name = "test_delete_correctness";
    let uri = format!("file:///tmp/{}", table_name);
    let _ = std::fs::remove_dir_all(format!("/tmp/{}", table_name));
    
    // We used to define manifest schema here, but create_async takes Arrow schema
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("data", DataType::Utf8, false),
    ]));

    let table = Table::create_async(uri.clone(), arrow_schema.clone()).await?;

    // 2. Insert Data (100 rows)
    let ids: Vec<i64> = (0..100).collect();
    let data: Vec<String> = ids.iter().map(|i| format!("val_{}", i)).collect();
    
    
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            Arc::new(arrow::array::Int64Array::from(ids)),
            Arc::new(StringArray::from(data)),
        ],
    )?;
    
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    // Verify Initial Count
    let count = scan_count(&table).await?;
    assert_eq!(count, 100, "Initial count should be 100");

    // 3. Perform Delete 1: Delete ID = 10
    println!("Deleting ID = 10...");
    table.delete_async("id = 10").await?;

    // Verify Count
    let count = scan_count(&table).await?;
    assert_eq!(count, 99, "Count should be 99 after deleting 1 row");
    
    // Verify ID 10 is gone
    let results = scan_all(&table).await?;
    let ids = results.column(0).as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
    for i in 0..ids.len() {
        if ids.value(i) == 10 {
            panic!("ID 10 should have been deleted!");
        }
    }

    // 4. Verify Physical File Format
    // We expect an Avro file in the directory
    verify_delete_files_format(&uri, 1).await?;

    // 5. Perform Delete 2: Delete IDs 20-29 (Range)
    println!("Deleting IDs 20-29...");
    table.delete_async("id >= 20 AND id < 30").await?;

    // Verify Count
    let count = scan_count(&table).await?;
    assert_eq!(count, 89, "Count should be 89 after deleting 10 more rows");

    // 6. Verify Duplicate Delete (Idempotency)
    // Delete ID 10 again (already deleted)
    // The current implementation scans RAW file, so it WILL find row 10 (at offset 10)
    // and write another delete file for it. This is wasteful but compliant.
    println!("Deleting ID = 10 AGAIN...");
    table.delete_async("id = 10").await?;
    
    let count = scan_count(&table).await?;
    assert_eq!(count, 89, "Count should still be 89");

    Ok(())
}

async fn scan_count(table: &Table) -> anyhow::Result<usize> {
    let mut stream = table.stream_all(None).await?;
    let mut count = 0;
    while let Some(batch) = stream.next().await {
        let b = batch?;
        count += b.num_rows();
    }
    Ok(count)
}

async fn scan_all(table: &Table) -> anyhow::Result<RecordBatch> {
    let mut stream = table.stream_all(None).await?;
    let mut batches: Vec<RecordBatch> = Vec::new();
    while let Some(batch) = stream.next().await {
        batches.push(batch?);
    }
    if batches.is_empty() {
        // Return empty batch with correct schema if possible, or error?
        // For test, just return empty with hardcoded schema or panic if unexpected empty
        // But deletes might empty the table.
        // We can't easily reconstruction schema from empty vec unless we pass it.
        // Let's assume at least one batch if not empty, or handle gracefully.
        // Actually arrow concat requires at least one batch or schema. 
        // We can just return empty Result if expected.
        return Ok(RecordBatch::new_empty(Arc::new(ArrowSchema::empty()))); // Incomplete schema but safe for test check?
    }
    use arrow::compute::concat_batches;
    let schema = batches[0].schema();
    Ok(concat_batches(&schema, &batches)?)
}

async fn verify_delete_files_format(uri: &str, expected_min_count: usize) -> anyhow::Result<()> {
    let path = uri.strip_prefix("file://").unwrap();
    let mut count = 0;
    
    use std::fs;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with("del-pos-") && name.ends_with(".avro") {
                count += 1;
                println!("Found valid position delete file: {}", name);
                
                // Optional: Read it with avro-rs to verify schema
                let file = std::fs::File::open(&path)?;
                let reader = apache_avro::Reader::new(file)?;
                let schema = reader.writer_schema();
                // Check if schema has "file_path" and "pos"
                // Simplified check
                // println!("Schema: {:?}", schema);
            } else if name.ends_with(".roaring") {
                return Err(anyhow::anyhow!("Found Legacy RoaringBitmap delete file! Violation: {}", name));
            }
        }
    }
    
    if count < expected_min_count {
        return Err(anyhow::anyhow!("Found {} delete files, expected at least {}", count, expected_min_count));
    }
    Ok(())
}
