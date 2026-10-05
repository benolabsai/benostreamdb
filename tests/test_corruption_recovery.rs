// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::Table;
use std::fs;
use std::io::Write;
use std::sync::Arc;
use tempfile::tempdir;

fn batch(start: i32, n: i32) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let ids = Int32Array::from_iter_values(start..start + n);
    RecordBatch::try_new(schema, vec![Arc::new(ids)]).unwrap()
}

#[tokio::test]
async fn test_wal_corruption_aborts_startup() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    // 1. Initialize table and write some data
    {
        // By default, new_async uses WalDurability::Sync, so write_async writes synchronously to _wal
        let table = Table::new_async(uri.clone()).await?;
        table.write_async(vec![batch(0, 10)]).await?;
        // DO NOT call flush_async(), otherwise it moves the WAL into a parquet file
    }

    // 2. Corrupt the WAL
    let mut corrupted_a_file = false;
    let mut dirs = vec![dir.path().to_path_buf()];
    while let Some(d) = dirs.pop() {
        if let Ok(entries) = fs::read_dir(&d) {
            for e in entries.flatten() {
                let path = e.path();
                if path.is_file() && path.extension().unwrap_or_default() == "arrow" {
                    let mut file = fs::OpenOptions::new().write(true).open(&path)?;
                    file.write_all(b"GARBAGE_DATA_CORRUPTION_TEST")?;
                    corrupted_a_file = true;
                } else if path.is_dir() {
                    dirs.push(path);
                }
            }
        }
    }
    assert!(corrupted_a_file, "Could not find WAL file to corrupt");

    // 3. Attempt to reopen - should fail due to corruption
    let open_res = Table::new_async(uri.clone()).await;
    assert!(
        open_res.is_err(),
        "Table should refuse to start with a corrupted WAL file to prevent data loss"
    );

    // 4. Manual intervention: remove the corrupted WAL file
    let mut dirs2 = vec![dir.path().to_path_buf()];
    while let Some(d) = dirs2.pop() {
        if let Ok(entries) = fs::read_dir(&d) {
            for e in entries.flatten() {
                let path = e.path();
                if path.is_file() && path.extension().unwrap_or_default() == "arrow" {
                    fs::remove_file(&path)?;
                } else if path.is_dir() {
                    dirs2.push(path);
                }
            }
        }
    }

    // 5. Reopen should now succeed
    let table = Table::new_async(uri).await?;
    let batches = table.read_async(None, None, None).await?;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 0,
        "Table should be empty since un-manifested writes were lost along with the WAL file"
    );

    Ok(())
}

#[tokio::test]
async fn test_index_corruption_fallback() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    // 1. Initialize table, write data, add index
    {
        let table = Table::new_async(uri.clone()).await?;
        table.write_async(vec![batch(0, 10)]).await?;
        table.commit_async().await?;
        table
            .add_index("id".to_string(), IndexAlgorithm::Bitmap)
            .await?;
        table.wait_for_background_tasks_async().await?;
    }

    // 2. Corrupt the index bundle (all secondary indexes live in `.puffin`).
    let mut corrupted = false;
    let mut dirs_to_visit = vec![dir.path().to_path_buf()];

    while let Some(current_dir) = dirs_to_visit.pop() {
        if let Ok(entries) = fs::read_dir(&current_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    dirs_to_visit.push(path);
                } else if path.is_file() {
                    let file_name = path.file_name().unwrap_or_default().to_string_lossy();
                    if file_name.ends_with(".puffin") {
                        let mut file = fs::OpenOptions::new().write(true).open(&path)?;
                        file.write_all(b"CORRUPTED_INDEX_GARBAGE_DATA")?;
                        corrupted = true;
                    }
                }
            }
        }
    }
    assert!(corrupted, "Could not find index bundle to corrupt");

    // 3. Reopen table and verify fallback behavior
    let table = Table::new_async(uri.clone()).await?;

    // We should be able to read data despite the corrupt index (graceful fallback)
    let batches = table.read_async(None, None, None).await?;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 10,
        "Data must be readable via full scan despite corrupted index"
    );

    // 4. Drop the corrupted index
    table.drop_index("id".to_string()).await?;

    // 5. Rebuild the index
    table
        .add_index("id".to_string(), IndexAlgorithm::Bitmap)
        .await?;
    table.wait_for_background_tasks_async().await?;

    Ok(())
}
