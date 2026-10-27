// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Covers `Table::list_scan_splits`, the primitive behind the Spark and Trino
//! multi-partition reads.
//!
//! The contract the connectors rely on:
//!
//! * one split per data file, carrying the absolute path and its size;
//! * `row_group_ids` empty means "all row groups" (file-granular split);
//! * `has_deletes` is set for a file that has merge-on-read delete files, so a
//!   connector never reads it raw (that would resurrect deleted rows).

use anyhow::Result;
use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::Table;
use std::sync::Arc;

fn batch(ids: Vec<i64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(ids))]).expect("sample batch")
}

async fn table_with_two_files() -> Result<(tempfile::TempDir, Table)> {
    let dir = tempfile::tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());

    let table = Table::new_async(uri).await?;
    table.write_async(vec![batch(vec![1, 2])]).await?;
    table.commit_async().await?;
    // A second commit writes a second data file, which is what makes the scan
    // splittable at all.
    table.write_async(vec![batch(vec![3, 4])]).await?;
    table.commit_async().await?;

    Ok((dir, table))
}

#[tokio::test]
async fn scan_splits_are_one_per_data_file() -> Result<()> {
    let (_dir, table) = table_with_two_files().await?;

    // The manifest itself must not list a file twice: if it does, every reader
    // (SQL included) duplicates that file's rows.
    let files = table.list_data_files_async().await?;
    let mut file_paths: Vec<&str> = files.iter().map(|f| f.file_path.as_str()).collect();
    file_paths.sort_unstable();
    file_paths.dedup();
    assert_eq!(
        files.len(),
        file_paths.len(),
        "manifest must not repeat a data file"
    );

    let splits = table.list_scan_splits().await?;

    assert_eq!(splits.len(), 2, "one split per committed data file");
    for split in &splits {
        assert!(
            split.file_path.ends_with(".parquet"),
            "split should point at a data file, got {}",
            split.file_path
        );
        assert!(split.length > 0, "split should carry the file size");
        assert!(
            split.row_group_ids.is_empty(),
            "file-granular splits mean 'all row groups'"
        );
        assert!(!split.has_deletes, "no deletes were written yet");
    }

    // Distinct files, otherwise the fan-out would double-count rows.
    let mut paths: Vec<&str> = splits.iter().map(|s| s.file_path.as_str()).collect();
    paths.sort_unstable();
    paths.dedup();
    assert_eq!(paths.len(), 2, "splits must reference distinct files");

    Ok(())
}

#[tokio::test]
async fn scan_splits_flag_files_with_deletes() -> Result<()> {
    let (_dir, table) = table_with_two_files().await?;

    table.delete_async("id = 1").await?;
    table.commit_async().await?;

    let splits = table.list_scan_splits().await?;

    assert!(
        splits.iter().any(|split| split.has_deletes),
        "a file with a delete file must be flagged so connectors do not fan out"
    );

    Ok(())
}
