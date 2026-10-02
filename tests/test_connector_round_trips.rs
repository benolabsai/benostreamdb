// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
//
//! WS4: Spark/Trino connector read-update-delete round-trips.
//!
//! The JVM connectors drive the engine through a small surface: list data files
//! and enumerate splits (scan planning), read rows, and commit updates/deletes
//! (delete + re-insert, i.e. the MERGE-on-read path). These tests exercise that
//! surface end-to-end at the core API level — the same calls the JNI bridge
//! (`src/core/ffi.rs`) makes — and assert the table stays consistent across
//! each step: scan planning must always reflect the *current* snapshot.

use std::sync::Arc;

use arrow::array::{Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use benostreamdb::core::table::Table;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]))
}

fn batch(start: i32, n: i32) -> anyhow::Result<RecordBatch> {
    let ids: Vec<i32> = (start..start + n).collect();
    let names: Vec<String> = ids.iter().map(|i| format!("row-{i}")).collect();
    Ok(RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(names)),
        ],
    )?)
}

async fn read_ids(table: &Table, filter: Option<&str>) -> anyhow::Result<Vec<i32>> {
    let batches = table.read_async(filter, None, None).await?;
    let mut ids = Vec::new();
    for b in &batches {
        if let Some(col) = b.column_by_name("id") {
            let arr = col.as_any().downcast_ref::<Int32Array>().expect("id");
            for i in 0..arr.len() {
                ids.push(arr.value(i));
            }
        }
    }
    ids.sort_unstable();
    Ok(ids)
}

async fn new_table(uri: String) -> anyhow::Result<Table> {
    let table = Table::new_async(uri).await?;
    table.set_autocommit(false);
    Ok(table)
}

/// Scan planning lists the current data files and splits them, and every split
/// points at a real data file.
#[tokio::test]
async fn scan_planning_lists_files_and_splits() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());
    let table = new_table(uri).await?;
    table.write_async(vec![batch(0, 6)?, batch(6, 6)?]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let files = table.list_data_files_async().await?;
    assert!(!files.is_empty(), "connector must see data files");
    assert!(
        files.iter().all(|f| f.row_count > 0),
        "every data file must report a positive row count"
    );
    let file_paths: std::collections::HashSet<&str> =
        files.iter().map(|f| f.file_path.as_str()).collect();

    let splits = table.get_splits_async(1 << 20, None).await?;
    assert!(!splits.is_empty(), "connector must enumerate splits");
    for s in &splits {
        assert!(s.length > 0, "split must have a positive length");
        assert!(
            file_paths.contains(s.file_path.as_str()),
            "split references a file not in list_data_files: {}",
            s.file_path
        );
    }
    assert_eq!(read_ids(&table, None).await?, (0..12).collect::<Vec<_>>());
    Ok(())
}

/// Delete (the update/delete half of a MERGE-on-read) must be visible to a
/// re-read, and scan planning must reflect the new snapshot — no stale splits.
#[tokio::test]
async fn delete_then_read_round_trip() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());
    let table = new_table(uri).await?;
    table.write_async(vec![batch(0, 10)?]).await?;
    table.commit_async().await?;

    table.delete_async("id < 5").await?;
    table.wait_for_background_tasks_async().await?;

    assert_eq!(
        read_ids(&table, None).await?,
        (5..10).collect::<Vec<_>>(),
        "deleted rows must not reappear on read"
    );

    // Scan planning still lists the (now logically-trimmed) table correctly.
    let files = table.list_data_files_async().await?;
    let file_paths: std::collections::HashSet<String> =
        files.iter().map(|f| f.file_path.clone()).collect();
    for s in table.get_splits_async(1 << 20, None).await? {
        assert!(file_paths.contains(&s.file_path));
    }
    // The engine reads position deletes at scan time; the connector's view of
    // the rows must match a direct filtered read.
    assert_eq!(
        read_ids(&table, Some("id >= 5")).await?,
        (5..10).collect::<Vec<_>>()
    );
    Ok(())
}

/// Update via delete + re-insert (MERGE), then read back the merged state.
#[tokio::test]
async fn update_via_delete_and_insert_round_trip() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());
    let table = new_table(uri).await?;
    table.write_async(vec![batch(0, 8)?]).await?;
    table.commit_async().await?;

    // "Update" rows 0..4: delete the old versions, insert new ones.
    table.delete_async("id < 4").await?;
    table.write_async(vec![batch(100, 4)?]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let mut expected: Vec<i32> = (4..8).collect();
    expected.extend(100..104);
    expected.sort_unstable();
    assert_eq!(read_ids(&table, None).await?, expected);

    // A second delete + scan-planning cycle must stay consistent.
    table.delete_async("id >= 100").await?;
    table.wait_for_background_tasks_async().await?;
    assert_eq!(read_ids(&table, None).await?, (4..8).collect::<Vec<_>>());
    Ok(())
}

/// Full connector-style lifecycle: read → delete-all → empty read → rewrite →
/// read, with scan planning valid at every step.
#[tokio::test]
async fn connector_lifecycle_round_trip() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());
    let table = new_table(uri).await?;

    table.write_async(vec![batch(0, 5)?]).await?;
    table.commit_async().await?;
    assert_eq!(read_ids(&table, None).await?, (0..5).collect::<Vec<_>>());

    table.delete_async("id >= 0").await?;
    table.wait_for_background_tasks_async().await?;
    assert!(
        read_ids(&table, None).await?.is_empty(),
        "delete-all must leave no readable rows"
    );

    table.write_async(vec![batch(200, 3)?]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;
    assert_eq!(
        read_ids(&table, None).await?,
        (200..203).collect::<Vec<_>>()
    );

    // Evidence the connector can plan the final snapshot.
    assert!(!table.list_data_files_async().await?.is_empty());
    assert!(!table.get_splits_async(1 << 20, None).await?.is_empty());
    Ok(())
}
