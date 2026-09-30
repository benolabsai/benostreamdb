// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Concurrency and Crash-Resilience Regression Tests (H1, H2, H3).
//!
//! Specifically validates:
//! 1. H2: `truncate_async()` concurrent write barrier under active multithreaded ingestion.
//! 2. H1: `PendingWrite` (batch + WAL transaction ID) atomicity under high-throughput parallel writers with flushes.
//! 3. H3: Safe error bubbling during destructive operations without corrupting manifest states.

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::Table;

fn create_batch(start_id: i32, num_rows: usize) -> RecordBatch {
    let id_array = Int32Array::from_iter_values(start_id..start_id + num_rows as i32);
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(id_array)]).unwrap()
}

/// H2 Regression Test: Concurrent writers racing with `truncate_async()`.
///
/// Verifies that with the `maintenance_lock` write barrier, `truncate_async` does not
/// race with concurrent `write_async` calls to produce orphaned WAL entries, torn manifest
/// snapshots, or corrupted read states.
#[tokio::test]
async fn test_truncate_vs_write_race_h2() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

    let table = Arc::new(Table::new_async(uri.clone()).await?);

    // Initial write to establish schema and base snapshot
    let base_batch = create_batch(1, 20);
    table.write_async(vec![base_batch]).await?;
    table.commit_async().await?;

    let stop_signal = Arc::new(AtomicBool::new(false));
    let num_writers = 4;
    let mut writer_handles = Vec::new();

    for w in 0..num_writers {
        let t = table.clone();
        let stop = stop_signal.clone();
        let handle = tokio::spawn(async move {
            let mut iter = 0;
            while !stop.load(Ordering::Relaxed) && iter < 50 {
                let start_id = (w + 1) * 100_000 + iter * 10;
                let batch = create_batch(start_id, 10);
                let _ = t.write_async(vec![batch]).await;
                if iter % 5 == 0 {
                    let _ = t.commit_async().await;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
                iter += 1;
            }
        });
        writer_handles.push(handle);
    }

    // Allow writers to start producing data
    tokio::time::sleep(Duration::from_millis(25)).await;

    // Trigger truncate while writers are actively appending
    table.truncate_async().await?;

    // Signal writers to stop and await them
    stop_signal.store(true, Ordering::Relaxed);
    for h in writer_handles {
        let _ = h.await;
    }

    // Flush any in-flight pending writes that completed after truncate
    table.flush_async().await?;

    // Now query the table: the table must be in a consistent state
    let batches = table.read_async(None, None, None).await?;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();

    // Verify snapshot and manifest health
    let version = table.snapshot_version().await?;
    assert!(version >= 1, "Snapshot version should be valid");

    // Close and re-open table to ensure WAL and manifest load cleanly without errors
    let table_reopened = Table::new_async(uri.clone()).await?;
    let reopened_batches = table_reopened.read_async(None, None, None).await?;
    let reopened_rows: usize = reopened_batches.iter().map(|b| b.num_rows()).sum();

    let reopened_version = table_reopened.snapshot_version().await?;
    assert_eq!(version, reopened_version);
    assert_eq!(total_rows, reopened_rows);

    Ok(())
}

/// H1 Regression Test: `PendingWrite` atomicity under high-throughput concurrent producers.
///
/// Verifies that batch data and its corresponding WAL transaction IDs never diverge, get dropped,
/// or corrupt each other when multiple async writers push data concurrently with periodic flushes.
#[tokio::test]
async fn test_pending_write_atomicity_high_throughput_h1() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

    let table = Arc::new(Table::new_async(uri.clone()).await?);
    // Disable autocommit to exercise the pending write buffer and WAL staging
    table.set_autocommit(false);

    let num_producers = 6;
    let batches_per_producer = 15;
    let rows_per_batch = 20;
    let total_expected_rows = num_producers * batches_per_producer * rows_per_batch;

    let mut producer_handles = Vec::new();
    for p in 0..num_producers {
        let t = table.clone();
        let handle = tokio::spawn(async move {
            for b in 0..batches_per_producer {
                let start_id = (p as i32 * 10_000) + (b as i32 * rows_per_batch as i32);
                let batch = create_batch(start_id, rows_per_batch);
                t.write_async(vec![batch]).await.unwrap();
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
        });
        producer_handles.push(handle);
    }

    // Spawn a background flusher
    let flusher_table = table.clone();
    let flusher_stop = Arc::new(AtomicBool::new(false));
    let stop_for_flusher = flusher_stop.clone();
    let flusher_handle = tokio::spawn(async move {
        while !stop_for_flusher.load(Ordering::Relaxed) {
            tokio::time::sleep(Duration::from_millis(15)).await;
            let _ = flusher_table.flush_async().await;
        }
    });

    for h in producer_handles {
        h.await.unwrap();
    }

    flusher_stop.store(true, Ordering::Relaxed);
    let _ = flusher_handle.await;

    // Final flush to ensure all remaining pending writes are persisted
    table.flush_async().await?;
    table.commit_async().await?;

    // Read back all rows and assert 100% data integrity
    let results = table.read_async(None, None, None).await?;
    let mut collected_ids = HashSet::new();

    for b in results {
        let id_col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..b.num_rows() {
            let id = id_col.value(i);
            assert!(
                collected_ids.insert(id),
                "Duplicate row ID detected: {}",
                id
            );
        }
    }

    assert_eq!(
        collected_ids.len(),
        total_expected_rows,
        "All produced rows must be accounted for without loss"
    );

    Ok(())
}

/// H3 Regression Test: Truncate error propagation.
///
/// Ensures truncate fails cleanly and returns an Err when target storage is unreadable
/// instead of silently continuing with an unwrap_or_default() empty manifest.
#[tokio::test]
async fn test_truncate_error_propagation_h3() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

    let table = Table::new_async(uri.clone()).await?;
    let batch = create_batch(1, 10);
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    // Delete the manifest directory directly behind the table's back to simulate store failure/corruption
    let manifest_dir = temp_dir.path().join("_manifest");
    if manifest_dir.exists() {
        std::fs::remove_dir_all(&manifest_dir)?;
        // Create an unreadable file with the same name to cause an I/O error
        std::fs::write(&manifest_dir, b"corrupted non-directory")?;
    }

    // truncate_async should bubble up the I/O error rather than silently succeeding
    let result = table.truncate_async().await;
    assert!(
        result.is_err(),
        "truncate_async must propagate storage errors"
    );

    Ok(())
}

/// H2 Barrier Test: Concurrent `flush_async` / `commit_async` racing with `truncate_async()`.
///
/// Verifies that `flush_async` and `commit_async` respect the `maintenance_lock` barrier,
/// preventing resurrected data or deadlocks.
#[tokio::test]
async fn test_truncate_vs_flush_commit_race() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
    let table = Arc::new(Table::new_async(uri.clone()).await?);

    // Initial write
    table.write_async(vec![create_batch(1, 50)]).await?;
    table.commit_async().await?;

    let stop = Arc::new(AtomicBool::new(false));
    let mut tasks = Vec::new();

    // 2 background flushers/committers
    for _ in 0..2 {
        let t = table.clone();
        let s = stop.clone();
        tasks.push(tokio::spawn(async move {
            let mut i = 0;
            while !s.load(Ordering::Relaxed) && i < 30 {
                let _ = t.write_async(vec![create_batch(1000 + i * 5, 5)]).await;
                let _ = t.flush_async().await;
                let _ = t.commit_async().await;
                tokio::time::sleep(Duration::from_millis(2)).await;
                i += 1;
            }
        }));
    }

    // Trigger truncate in the middle
    tokio::time::sleep(Duration::from_millis(15)).await;
    table.truncate_async().await?;

    stop.store(true, Ordering::Relaxed);
    for task in tasks {
        let _ = task.await;
    }

    // Final flush and commit
    table.flush_async().await?;
    let version = table.snapshot_version().await?;
    assert!(version >= 1);

    Ok(())
}

/// Deadlock Regression Test: Verify `rewrite_data_files_async` does not self-deadlock
/// when flushing before compaction.
#[tokio::test]
async fn test_compaction_rewrite_data_files_no_deadlock() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
    let table = Table::new_async(uri.clone()).await?;

    // Write several batches without explicit flush
    for i in 0..5 {
        table.write_async(vec![create_batch(i * 10, 10)]).await?;
    }

    // rewrite_data_files_async acquires maintenance_lock.write() and calls flush_unlocked_async()
    // It should complete promptly without deadlocking.
    tokio::time::timeout(
        Duration::from_secs(10),
        table.rewrite_data_files_async(None),
    )
    .await
    .expect("compaction timed out - deadlock detected")?;

    let rows = table.read_async(None, None, None).await?;
    let total: usize = rows.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 50);

    Ok(())
}
