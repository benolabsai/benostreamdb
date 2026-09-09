// Copyright (c) 2026 Richard Albright. All rights reserved.

use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use hyperstreamdb::core::table::Table;
use std::sync::Arc;

#[tokio::test]
async fn test_recover_indexes_empty_table() -> anyhow::Result<()> {
    let tmp = tempfile::tempdir()?;
    let uri = format!("file://{}", tmp.path().display());
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

    let table = Table::create_async(uri, schema).await?;
    let recovered = table.recover_indexes_async().await?;
    assert_eq!(
        recovered, 0,
        "Empty table should have 0 segments to recover"
    );

    Ok(())
}

#[tokio::test]
async fn test_recover_indexes_after_unindexed_ingest() -> anyhow::Result<()> {
    let tmp = tempfile::tempdir()?;
    let uri = format!("file://{}", tmp.path().display());
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("text", DataType::Utf8, false),
    ]));

    let id_arr = arrow::array::Int32Array::from(vec![1, 2, 3]);
    let text_arr =
        arrow::array::StringArray::from(vec!["vector search", "hybrid index", "lakehouse engine"]);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(id_arr), Arc::new(text_arr)])?;

    let mut table = Table::create_async(uri.clone(), schema.clone()).await?;

    // Configure indexing on the table
    table
        .add_index_columns_async(vec!["text".to_string()], None)
        .await?;

    // Ingest initial batch
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    // Verify index files exist initially
    let manager =
        hyperstreamdb::core::manifest::ManifestManager::new(table.store.clone(), "", &table.uri);
    let (_manifest, mut all_entries, _) = manager.load_latest_full().await?;
    assert!(!all_entries.is_empty());
    assert!(
        !all_entries[0].index_files.is_empty(),
        "Initial write should have built index files"
    );

    // Simulate an external Iceberg compaction engine (Spark/Trino) rewriting data files without sidecars
    for entry in &mut all_entries {
        entry.index_files = Vec::new();
    }
    manager.commit_imported_entries(all_entries).await?;

    // Verify query continues to return correct rows (brute force Parquet fallback)
    let rows_before = table.read_async(Some("id > 1"), None, None).await?;
    assert_eq!(rows_before.iter().map(|b| b.num_rows()).sum::<usize>(), 2);

    // Recover indexes: should detect the compacted segment lacking sidecars and rebuild it
    let recovered = table.recover_indexes_async().await?;
    assert_eq!(
        recovered, 1,
        "Should have recovered 1 compacted segment lacking index sidecars"
    );

    // Calling again should report 0 needed
    let recovered_again = table.recover_indexes_async().await?;
    assert_eq!(
        recovered_again, 0,
        "All segments should now have valid index sidecars"
    );

    // Verify query correctness after index recovery
    let rows_after = table.read_async(Some("id = 3"), None, None).await?;
    assert_eq!(rows_after.iter().map(|b| b.num_rows()).sum::<usize>(), 1);

    Ok(())
}

#[test]
fn test_recover_indexes_sync_api() -> anyhow::Result<()> {
    let tmp = tempfile::tempdir()?;
    let uri = format!("file://{}", tmp.path().display());
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

    let table = Table::create(uri, schema)?;
    let recovered = table.recover_indexes()?;
    assert_eq!(recovered, 0);

    Ok(())
}
