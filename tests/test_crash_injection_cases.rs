// Copyright (c) 2026 Richard Albright. All rights reserved.

//! WS2: explicit crash-injection tests for the production-readiness review's
//! cases A–F. Each test kills the write at one named boundary and asserts the
//! recovery invariant for that specific failure mode.
//!
//! | Case | Boundary | Invariant |
//! |------|----------|-----------|
//! | A | data written, index not | data is readable (full-scan fallback) |
//! | B | index built, manifest not | no dangling reference; reads correct |
//! | C | manifest committed, caller not acked | data visible after reopen |
//! | D | WAL durable, manifest not | data recovered from WAL |
//! | E | manifest committed, WAL not truncated | no duplicate rows (idempotent) |
//! | F | two concurrent writers | no lost updates |
//!
//! See `plans/production_readiness_plan.md` §WS2.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::fault_injection::{arm, disarm, CrashPoint};
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::Table;
use tempfile::tempdir;

fn batch(start: i32, n: i32) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let ids = Int32Array::from_iter_values(start..start + n);
    RecordBatch::try_new(schema, vec![Arc::new(ids)]).unwrap()
}

async fn read_ids(table: &Table) -> anyhow::Result<Vec<i32>> {
    let batches = table.read_async(None, None, None).await?;
    let mut ids = Vec::new();
    for b in &batches {
        if let Some(col) = b.column_by_name("id") {
            if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
                ids.extend(arr.iter().flatten());
            }
        }
    }
    Ok(ids)
}

/// Case A — data written, index build aborted.
///
/// The data commit must still succeed and the rows must be readable via the
/// full-scan fallback even though the index was never uploaded.
#[tokio::test]
async fn case_a_data_without_index_is_readable() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    let table = Table::new_async(uri.clone()).await?;
    table
        .add_index("id".to_string(), IndexAlgorithm::Bitmap)
        .await?;

    arm(CrashPoint::IndexUpload);
    let res = async {
        table.write_async(vec![batch(0, 20)]).await?;
        table.commit_async().await
    }
    .await;
    disarm();
    let _ = table.wait_for_background_tasks_async().await;
    drop(table);

    // The data commit is independent of the index build, so it must have
    // succeeded; the rows must be readable regardless of the missing index.
    let table = Table::new_async(uri).await?;
    let ids = read_ids(&table).await?;
    assert_eq!(
        ids.len(),
        20,
        "data must be readable even when the index build was aborted (case A); op={res:?}"
    );
    Ok(())
}

/// Case C — manifest committed, caller not acknowledged (delayed visibility).
///
/// A crash after the manifest commit but before the caller sees success must
/// still leave the rows durable and visible on reopen.
#[tokio::test]
async fn case_c_delayed_visibility_is_durable() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    {
        let table = Table::new_async(uri.clone()).await?;
        arm(CrashPoint::ManifestVisible);
        let _ = async {
            table.write_async(vec![batch(0, 12)]).await?;
            table.commit_async().await
        }
        .await;
        disarm();
    }

    let table = Table::new_async(uri).await?;
    let ids = read_ids(&table).await?;
    assert_eq!(
        ids.len(),
        12,
        "a committed-but-unacknowledged write must be durable (case C)"
    );
    Ok(())
}

/// Case D — WAL durable, manifest not committed.
///
/// The rows must be recovered from the WAL on reopen (durability), and the
/// table must be at the post-write state.
#[tokio::test]
async fn case_d_wal_before_manifest_recovers() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    {
        let table = Table::new_async(uri.clone()).await?;
        arm(CrashPoint::WalFlush);
        let _ = async {
            table.write_async(vec![batch(0, 9)]).await?;
            table.commit_async().await
        }
        .await;
        disarm();
    }

    let table = Table::new_async(uri).await?;
    let ids = read_ids(&table).await?;
    assert_eq!(
        ids.len(),
        9,
        "rows durable in the WAL must be recovered on reopen (case D)"
    );
    Ok(())
}

/// Case E — manifest committed, WAL not truncated.
///
/// This is the idempotency case: replaying the WAL must NOT re-apply the
/// already-committed batch. Before the fix this produced duplicate rows.
#[tokio::test]
async fn case_e_manifest_before_wal_truncation_is_idempotent() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    // Seed 10 committed rows.
    {
        let table = Table::new_async(uri.clone()).await?;
        table.write_async(vec![batch(0, 10)]).await?;
        table.commit_async().await?;
    }

    // Second write: commit succeeds, crash before the WAL is truncated.
    {
        let table = Table::new_async(uri.clone()).await?;
        arm(CrashPoint::WalTruncate);
        let _ = async {
            table.write_async(vec![batch(100, 5)]).await?;
            table.commit_async().await
        }
        .await;
        disarm();
    }

    let table = Table::new_async(uri).await?;
    let ids = read_ids(&table).await?;
    let unique: HashSet<i32> = ids.iter().copied().collect();
    assert_eq!(
        ids.len(),
        15,
        "expected exactly 15 rows after idempotent recovery (case E)"
    );
    assert_eq!(
        unique.len(),
        ids.len(),
        "WAL replay must not duplicate already-committed rows (case E)"
    );
    Ok(())
}

/// Case F — two concurrent writers must not lose updates.
#[tokio::test]
async fn case_f_two_writers_no_lost_updates() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    // Seed.
    {
        let table = Table::new_async(uri.clone()).await?;
        table.write_async(vec![batch(0, 1)]).await?;
        table.commit_async().await?;
    }

    eprintln!("[case-f] seed committed");

    use benostreamdb::core::table::builder::TableBuilder;
    let wal1 = tempdir()?;
    let wal2 = tempdir()?;
    let t1 = TableBuilder::new(&uri)
        .with_wal_dir(wal1.path())
        .build_async()
        .await?;
    let t2 = TableBuilder::new(&uri)
        .with_wal_dir(wal2.path())
        .build_async()
        .await?;

    let w1 = tokio::spawn(async move {
        t1.write_async(vec![batch(1000, 3)]).await.map_err(|e| {
            eprintln!("[case-f] writer 1 write error: {e}");
            e
        })?;
        t1.commit_async().await.map_err(|e| {
            eprintln!("[case-f] writer 1 commit error: {e}");
            e
        })
    });
    let w2 = tokio::spawn(async move {
        t2.write_async(vec![batch(2000, 4)]).await.map_err(|e| {
            eprintln!("[case-f] writer 2 write error: {e}");
            e
        })?;
        t2.commit_async().await.map_err(|e| {
            eprintln!("[case-f] writer 2 commit error: {e}");
            e
        })
    });

    eprintln!("[case-f] waiting for writers");
    w1.await??;
    eprintln!("[case-f] writer 1 done");
    w2.await??;
    eprintln!("[case-f] writer 2 done");

    let table = Table::new_async(uri).await?;
    let ids = read_ids(&table).await?;
    let unique: HashSet<i32> = ids.iter().copied().collect();
    eprintln!("[case-f] read {} ids: {ids:?}", ids.len());
    assert_eq!(
        ids.len(),
        8,
        "both concurrent writers' rows must survive (case F): {ids:?}"
    );
    assert_eq!(unique.len(), ids.len(), "no duplicate rows (case F)");
    Ok(())
}
