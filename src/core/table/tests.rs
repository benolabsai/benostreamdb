// Copyright (c) 2026 Richard Albright. All rights reserved.

use super::*;
use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use tempfile::tempdir;

#[tokio::test]
async fn test_table_lifecycle() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    // Use local file system uri
    let uri = format!("file://{}", path);

    // 1. Create Table (async)
    let table = Table::new_async(uri.clone()).await?;

    // 2. Write Data
    let schema = Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]);

    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["a", "b", "c"])),
        ],
    )?;

    // Use write_async since table was created with new_async
    table.write_async(vec![batch.clone()]).await?;
    table.commit_async().await?;

    // 3. Read Data
    let batches = table.read_async(None, None, None).await?;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);

    Ok(())
}

#[tokio::test]
async fn test_multi_column_bucketing() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);
    let _table = Table::new_async(uri.clone()).await?;

    // 1. Setup schema with metadata IDs
    let mut fields = Vec::new();
    let mut id_meta = std::collections::HashMap::new();
    id_meta.insert("iceberg.id".to_string(), "1".to_string());
    fields.push(Field::new("col1", DataType::Int32, false).with_metadata(id_meta));

    let mut type_meta = std::collections::HashMap::new();
    type_meta.insert("iceberg.id".to_string(), "2".to_string());
    fields.push(Field::new("col2", DataType::Utf8, false).with_metadata(type_meta));

    let schema = Arc::new(Schema::new(fields));

    // 2. Define multi-column bucket partition spec
    let spec = crate::core::manifest::PartitionSpec {
        spec_id: 0,
        fields: vec![crate::core::manifest::PartitionField::new_multi(
            vec![1, 2],
            Some(1000),
            "combined_bucket".to_string(),
            "bucket[10]".to_string(),
        )],
    };

    // 3. Create batch
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 1, 2])),
            Arc::new(StringArray::from(vec!["a", "b", "a"])),
        ],
    )?;

    // 4. Split by partition
    let results = spec.partition_batch(&batch)?;

    // Each uniquely combined (col1, col2) should have a stable hash
    // (1, "a"), (1, "b"), (2, "a") are all different, so they should return 3 partitions
    // unless there's a hash collision (unlikely with only 10 buckets and these values)
    assert!(results.len() >= 2);

    for (key, sub_batch) in results {
        assert!(key.contains_key("combined_bucket"));
        assert!(sub_batch.num_rows() >= 1);
    }

    Ok(())
}

#[tokio::test]
async fn test_admin_ops() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);
    let table = Table::new_async(uri.clone()).await?;

    // 1. Initial State: Autocommit is now false by default (opt-in)
    assert!(!table.get_autocommit());

    // 2. Write Data with explicit autocommit enabled
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )?;

    table.set_autocommit(true);
    table.write_async(vec![batch.clone()]).await?;

    // Should be committed automatically
    let batches = table.read_async(None, None, None).await?;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 3);

    // 3. Truncate
    table.truncate_async().await?;
    let batches = table.read_async(None, None, None).await?;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    tracing::info!(
        "After truncate, read {} records in {} batches",
        total_rows,
        batches.len()
    );
    assert!(
        total_rows == 0,
        "Table should be empty after truncate, but found {} rows!",
        total_rows
    );

    // 4. Manual commit (autocommit=false)
    table.set_autocommit(false);
    table.write_async(vec![batch.clone()]).await?;

    // Visible in read (from buffer)
    let batches = table.read_async(None, None, None).await?;
    assert!(!batches.is_empty(), "Should see data in buffer");

    let manifest_manager = ManifestManager::new(table.store.clone(), "", &table.uri);
    let (_, _, ver_pre) = manifest_manager
        .load_latest_full()
        .await
        .unwrap_or_default();
    assert_eq!(ver_pre, 2, "Should still be v2 before manual commit");

    table.commit_async().await?;
    let (_, _, ver_post) = manifest_manager
        .load_latest_full()
        .await
        .unwrap_or_default();
    assert_eq!(ver_post, 3, "Should be v3 after manual commit");

    // 5. Vacuum
    // Note: vacuum_async might not delete anything if within retention, but let's test it works
    table.vacuum_async(1).await?;

    Ok(())
}
