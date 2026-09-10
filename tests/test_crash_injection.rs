// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Crash injection and durability integration tests.
//! Validates the Four Core Invariants under failure scenarios:
//! 1. The Overlay Invariant
//! 2. Publication Invariant
//! 3. Durability Invariant
//! 4. Maintenance Invariant

use anyhow::Result;
use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use hyperstreamdb::core::manifest::{CommitMetadata, ManifestEntry, ManifestManager};
use hyperstreamdb::core::storage::create_object_store;
use hyperstreamdb::core::table::builder::TableBuilder;
use hyperstreamdb::core::table::WalDurability;
use hyperstreamdb::core::wal::{extract_wal_tx, tag_batch_with_wal_tx, WriteAheadLog};
use hyperstreamdb::Table;
use std::sync::Arc;

async fn create_test_batch(start_id: i32, num_rows: usize) -> RecordBatch {
    let id_array = Int32Array::from_iter_values(start_id..start_id + num_rows as i32);
    let name_array = StringArray::from_iter_values(
        (start_id..start_id + num_rows as i32).map(|i| format!("item_{}", i)),
    );
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));

    RecordBatch::try_new(schema, vec![Arc::new(id_array), Arc::new(name_array)]).unwrap()
}

/// Test 1: WAL Transaction Identity & Sequence Tagging
/// Verifies that WAL records can be stamped with transaction IDs and sequence numbers,
/// and retrieved upon replay.
#[tokio::test]
async fn test_wal_transaction_identity_and_sequence() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let wal_path = temp_dir.path().to_str().unwrap().to_string();

    let mut wal = WriteAheadLog::new(wal_path);
    wal.spawn_worker()?;

    let tx_id_1 = uuid::Uuid::new_v4();
    let batch_1 = create_test_batch(1, 10).await;
    let tagged_1 = tag_batch_with_wal_tx(&batch_1, tx_id_1, 1)?;

    let tx_id_2 = uuid::Uuid::new_v4();
    let batch_2 = create_test_batch(11, 10).await;
    let tagged_2 = tag_batch_with_wal_tx(&batch_2, tx_id_2, 2)?;

    wal.append_sync(tagged_1).await?;
    wal.append_sync(tagged_2).await?;
    wal.flush_async().await?;

    // Replay and verify headers
    let (recovered_batches, _) = wal.replay()?;
    assert_eq!(recovered_batches.len(), 2);

    let header_1 = extract_wal_tx(&recovered_batches[0]).expect("header 1 must exist");
    assert_eq!(header_1.tx_id, tx_id_1);
    assert_eq!(header_1.sequence_number, 1);

    let header_2 = extract_wal_tx(&recovered_batches[1]).expect("header 2 must exist");
    assert_eq!(header_2.tx_id, tx_id_2);
    assert_eq!(header_2.sequence_number, 2);

    Ok(())
}

/// Test 2: Custom Persistent WAL Directory via TableBuilder
/// Verifies that remote or local tables can use an explicit persistent WAL directory,
/// and that re-opening the table recovers uncommitted writes directly from it.
#[tokio::test]
async fn test_persistent_wal_directory_configuration() -> Result<()> {
    let temp_table_dir = tempfile::tempdir()?;
    let temp_wal_dir = tempfile::tempdir()?;

    let table_uri = format!("file://{}", temp_table_dir.path().to_str().unwrap());
    let wal_dir = temp_wal_dir.path().to_str().unwrap().to_string();

    // 1. Create table with custom WAL dir, write without committing, then drop
    {
        let table = TableBuilder::new(table_uri.clone())
            .with_wal_dir(wal_dir.clone())
            .with_durability(WalDurability::Sync)
            .build_async()
            .await?;

        let batch = create_test_batch(100, 25).await;
        table.write_async(vec![batch]).await?;
        // Dropping table simulates crash before commit
    }

    // 2. Reopen table with the same custom WAL dir
    {
        let table = TableBuilder::new(table_uri.clone())
            .with_wal_dir(wal_dir)
            .build_async()
            .await?;

        let batches = table.read_async(None, None, None).await?;
        let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(
            total_rows, 25,
            "Reopened table must recover uncommitted WAL rows from custom WAL dir"
        );
    }

    Ok(())
}

/// Test 3: Compaction Precondition Validation
/// Verifies that compaction fails cleanly if any candidate file being replaced
/// has been concurrently removed or replaced.
#[tokio::test]
async fn test_compaction_precondition_aborts_on_missing_candidate() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

    let store = create_object_store(&uri)?;
    let manifest = ManifestManager::new(store, "", &uri);

    // Commit snapshot v1 with file A
    let entry_a = ManifestEntry {
        file_path: "data/file_a.parquet".to_string(),
        file_size_bytes: 1024,
        record_count: 100,
        ..Default::default()
    };
    manifest
        .commit(&[entry_a.clone()], &[], CommitMetadata::default())
        .await?;

    // Verify snapshot v1 contains file A
    let (_, entries_v1, _) = manifest.load_latest_full().await?;
    assert!(entries_v1.iter().any(|e| e.file_path == "data/file_a.parquet"));

    // Attempt compaction replacing file_a AND a non-existent file_b with require_remove_paths_exist = true
    let entry_c = ManifestEntry {
        file_path: "data/file_c.parquet".to_string(),
        file_size_bytes: 2048,
        record_count: 200,
        ..Default::default()
    };
    let mut commit_meta = CommitMetadata::default();
    commit_meta.require_remove_paths_exist = true;

    let stale_remove_paths = vec![
        "data/file_a.parquet".to_string(),
        "data/non_existent_file_b.parquet".to_string(),
    ];

    let result = manifest
        .commit(&[entry_c], &stale_remove_paths, commit_meta)
        .await;

    assert!(
        result.is_err(),
        "Compaction commit must fail when candidate file does not exist in current snapshot"
    );
    let err_str = result.err().unwrap().to_string();
    assert!(
        err_str.contains("Compaction precondition failed"),
        "Error message should clearly state compaction precondition failure: {}",
        err_str
    );

    // Verify file A is STILL intact in active files (no partial removal)
    let (_, entries_after, _) = manifest.load_latest_full().await?;
    assert!(
        entries_after.iter().any(|e| e.file_path == "data/file_a.parquet"),
        "Active files must remain unchanged after aborted compaction commit"
    );

    Ok(())
}

/// Test 4: Maintenance Coordination and Staging Protection
/// Verifies that `remove_orphan_files` ignores `_staging/`, `_wal/`, and `commit.lock`.
#[tokio::test]
async fn test_maintenance_staging_and_wal_exclusion() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

    let table = Table::new_async(uri.clone()).await?;

    // Create active committed data
    let batch = create_test_batch(1, 50).await;
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    // Manually create staging and WAL dummy files
    let staging_path = temp_dir.path().join("_staging").join("indexes").join("idx_1.bin");
    std::fs::create_dir_all(staging_path.parent().unwrap())?;
    std::fs::write(&staging_path, b"in-flight index data")?;

    let wal_file = temp_dir.path().join("_wal").join("wal_001.arrow");
    std::fs::create_dir_all(wal_file.parent().unwrap())?;
    std::fs::write(&wal_file, b"wal data")?;

    // Run orphan cleanup with older_than_ms = -1 (clean everything unreferenced immediately)
    table.remove_orphan_files_async(-1).await?;

    // Verify staging file and wal file still exist!
    assert!(
        staging_path.exists(),
        "Files in _staging/ must NOT be deleted by remove_orphan_files"
    );
    assert!(
        wal_file.exists(),
        "Files in _wal/ must NOT be deleted by remove_orphan_files"
    );

    Ok(())
}

/// Test 5: Durability Modes - Sync vs Async
/// Verifies that `write_async` with `WalDurability::Sync` guarantees synchronous disk write,
/// and `write_buffered_async` allows async buffering.
#[tokio::test]
async fn test_wal_durability_modes() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

    // 1. Sync Table
    {
        let table = TableBuilder::new(uri.clone())
            .with_durability(WalDurability::Sync)
            .build_async()
            .await?;

        let batch = create_test_batch(1, 20).await;
        table.write_async(vec![batch]).await?;
        // No explicit flush or commit needed for sync WAL durability
    }

    // Recover immediately without graceful shutdown
    {
        let table = Table::new_async(uri.clone()).await?;
        let rows = table.read_async(None, None, None).await?;
        let total: usize = rows.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 20, "Sync durability must recover all 20 rows");
    }

    // 2. Async Table using write_buffered_async
    let temp_dir2 = tempfile::tempdir()?;
    let uri2 = format!("file://{}", temp_dir2.path().to_str().unwrap());
    {
        let table = TableBuilder::new(uri2.clone())
            .with_durability(WalDurability::Async)
            .build_async()
            .await?;

        let batch = create_test_batch(100, 30).await;
        table.write_buffered_async(vec![batch]).await?;
        table.flush_wal_async().await?; // Explicit sync point
    }

    // Reopen and check
    {
        let table = Table::new_async(uri2).await?;
        let rows = table.read_async(None, None, None).await?;
        let total: usize = rows.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 30, "Buffered async with flush_wal_async must recover 30 rows");
    }

    Ok(())
}

/// Test 6: Atomic Write Buffer Preservation on Flush Error
/// Verifies that if flush fails during upload/commit, the write buffer is
/// not lost: in-memory readers still see the data, and retry succeeds.
#[tokio::test]
async fn test_write_buffer_preservation_on_flush_error() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

    let table = Table::new_async(uri.clone()).await?;

    let batch = create_test_batch(1, 15).await;
    table.write_async(vec![batch]).await?;

    // In-memory buffer contains the 15 rows before flush
    let initial_read = table.read_async(None, None, None).await?;
    let initial_rows: usize = initial_read.iter().map(|b| b.num_rows()).sum();
    assert_eq!(initial_rows, 15);

    // Make table root directory read-only to inject a failure on manifest commit or parquet write
    let table_path = temp_dir.path();
    let original_perms = std::fs::metadata(table_path)?.permissions();
    let mut read_only_perms = original_perms.clone();
    read_only_perms.set_readonly(true);
    std::fs::set_permissions(table_path, read_only_perms)?;

    // Attempt flush — this must fail due to permission error
    let flush_res = table.flush_async().await;
    assert!(flush_res.is_err(), "Flush must fail when storage is read-only");

    // Restore permissions so cleanup and retry can proceed
    std::fs::set_permissions(table_path, original_perms)?;

    // CRITICAL: The write buffer MUST still contain the 15 rows despite the failed flush!
    let read_after_fail = table.read_async(None, None, None).await?;
    let rows_after_fail: usize = read_after_fail.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        rows_after_fail, 15,
        "Write buffer must retain in-memory rows when flush fails"
    );

    // Now retry flush with write permissions restored — must succeed!
    table.flush_async().await?;

    // Verify all rows remain accessible after successful retry
    let read_final = table.read_async(None, None, None).await?;
    let rows_final: usize = read_final.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows_final, 15);

    Ok(())
}
