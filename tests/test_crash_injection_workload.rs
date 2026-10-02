// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! WS2: crash injection across the full maintenance lifecycle.
//!
//! [`test_crash_injection_sweep`](test_crash_injection_sweep.rs) covers a plain
//! write+commit. This extends the sweep to the operations that mutate the
//! manifest in more complex ways — **delete, compaction, vacuum, and index
//! build** — and asserts the full invariant set after recovery:
//!
//! * **Atomicity** — the table is at the pre-op or post-op state, never torn.
//! * **No duplication** — WAL replay is idempotent.
//! * **Delete semantics** — a deleted row never resurfaces.
//! * **Index correctness** — a vector search still returns live rows.
//! * **Manifest consistency** — the manifest loads and its entries resolve.
//!
//! See `plans/production_readiness_plan.md` §WS2.

// The `SERIAL` guard below is deliberately held across `.await` points: the
// fault injector is process-global, so cases must run one-at-a-time. A
// std `MutexGuard` is intentional here (there is no async lock to hold and no
// re-entrancy), so silence the lint rather than restructuring the guard.
#![allow(clippy::await_holding_lock)]

use std::collections::HashSet;
use std::sync::{Arc, Mutex, MutexGuard};

use arrow::array::{FixedSizeListArray, Int32Array};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::fault_injection::{arm, disarm, CrashPoint};
use benostreamdb::core::index::VectorValue;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::core::table::VectorSearchParams;
use benostreamdb::Table;
use tempfile::tempdir;

const DIM: usize = 8;
const SEGMENTS: i32 = 3;
const ROWS_PER_SEGMENT: i32 = 20;
const SEED_ROWS: i32 = SEGMENTS * ROWS_PER_SEGMENT; // 60

/// The fault injector uses a single process-global armed point, so tests that
/// arm it must not run concurrently. Serialize them.
static SERIAL: Mutex<()> = Mutex::new(());

fn serial_guard() -> MutexGuard<'static, ()> {
    SERIAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// A batch of `n` rows starting at `start`, each with a deterministic embedding.
fn vec_batch(start: i32, n: i32) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ),
    ]));
    let ids = Int32Array::from_iter_values(start..start + n);
    let embedding = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        (start..start + n).map(|i| {
            Some(
                (0..DIM)
                    .map(|d| Some(((i as usize * DIM + d) % 31) as f32 / 31.0))
                    .collect::<Vec<_>>(),
            )
        }),
        DIM as i32,
    );
    RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(embedding) as _]).unwrap()
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

/// Seed a table with a vector index and `SEGMENTS` committed segments.
async fn seed(uri: &str) -> anyhow::Result<()> {
    let table = Table::new_async(uri.to_string()).await?;
    table
        .add_index("embedding".to_string(), IndexAlgorithm::hnsw_tq8())
        .await?;
    for r in 0..SEGMENTS {
        table
            .write_async(vec![vec_batch(r * 100, ROWS_PER_SEGMENT)])
            .await?;
        table.commit_async().await?;
    }
    table.wait_for_background_tasks_async().await?;
    Ok(())
}

/// Run `op` with a crash injected at `point`, then reopen the table.
///
/// Returns `(op_result, reopened_table)`.
async fn run_with_crash<F, Fut>(
    uri: &str,
    point: CrashPoint,
    op: F,
) -> anyhow::Result<(anyhow::Result<()>, Table)>
where
    F: FnOnce(Table) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<()>>,
{
    let op_result = {
        let table = Table::new_async(uri.to_string()).await?;
        arm(point);
        let r = op(table.clone()).await;
        disarm();
        let _ = table.wait_for_background_tasks_async().await;
        r
    };
    let reopened = Table::new_async(uri.to_string()).await?;
    Ok((op_result, reopened))
}

/// Crash during a delete+commit: the table must be at 60 (pre) or 59 (post)
/// rows, never torn, never duplicated, and a successful delete must stick.
#[tokio::test]
async fn crash_during_delete_preserves_semantics() -> anyhow::Result<()> {
    let _guard = serial_guard();
    let mut failures: Vec<String> = Vec::new();

    for &point in CrashPoint::ALL {
        let dir = tempdir()?;
        let uri = format!("file://{}", dir.path().display());
        seed(&uri).await?;

        let (op, table) = run_with_crash(&uri, point, |t| async move {
            t.delete_async("id = 100").await?;
            t.commit_async().await
        })
        .await?;

        let ids = read_ids(&table).await?;
        let unique: HashSet<i32> = ids.iter().copied().collect();

        if ids.len() != SEED_ROWS as usize && ids.len() != (SEED_ROWS - 1) as usize {
            failures.push(format!(
                "[{point}] delete atomicity: {} rows (expected {SEED_ROWS} or {})",
                ids.len(),
                SEED_ROWS - 1
            ));
        }
        if unique.len() != ids.len() {
            failures.push(format!("[{point}] delete duplicated rows"));
        }
        if op.is_ok() && ids.contains(&100) {
            failures.push(format!("[{point}] deleted row 100 resurfaced"));
        }
    }

    assert!(
        failures.is_empty(),
        "delete crash sweep found {} violation(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

/// Crash during compaction: committed rows must be preserved (no loss, no
/// duplication, no resurrection of deleted rows).
#[tokio::test]
async fn crash_during_compaction_preserves_rows() -> anyhow::Result<()> {
    let _guard = serial_guard();
    let mut failures: Vec<String> = Vec::new();

    for &point in CrashPoint::ALL {
        let dir = tempdir()?;
        let uri = format!("file://{}", dir.path().display());
        seed(&uri).await?;

        // Delete one row first so compaction has a delete to materialize.
        {
            let t = Table::new_async(uri.clone()).await?;
            t.delete_async("id = 100").await?;
            t.commit_async().await?;
        }

        let (op, table) = run_with_crash(&uri, point, |t| async move {
            t.rewrite_data_files_async(None).await
        })
        .await?;

        let ids = read_ids(&table).await?;
        let unique: HashSet<i32> = ids.iter().copied().collect();

        if ids.len() != (SEED_ROWS - 1) as usize {
            failures.push(format!(
                "[{point}] compaction changed row count: {} (expected {})",
                ids.len(),
                SEED_ROWS - 1
            ));
        }
        if unique.len() != ids.len() {
            failures.push(format!("[{point}] compaction duplicated rows"));
        }
        if ids.contains(&100) {
            failures.push(format!("[{point}] compaction resurrected deleted row 100"));
        }
        let _ = op;
    }

    assert!(
        failures.is_empty(),
        "compaction crash sweep found {} violation(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

/// Crash during vacuum: live rows must survive; only unreferenced artifacts may
/// be deleted.
#[tokio::test]
async fn crash_during_vacuum_preserves_live_rows() -> anyhow::Result<()> {
    let _guard = serial_guard();
    let mut failures: Vec<String> = Vec::new();

    for &point in CrashPoint::ALL {
        let dir = tempdir()?;
        let uri = format!("file://{}", dir.path().display());
        seed(&uri).await?;

        let (op, table) = run_with_crash(&uri, point, |t| async move {
            t.vacuum_async(1).await.map(|_| ())
        })
        .await?;

        let ids = read_ids(&table).await?;
        let unique: HashSet<i32> = ids.iter().copied().collect();

        if ids.len() != SEED_ROWS as usize {
            failures.push(format!(
                "[{point}] vacuum lost live rows: {} (expected {SEED_ROWS})",
                ids.len()
            ));
        }
        if unique.len() != ids.len() {
            failures.push(format!("[{point}] vacuum duplicated rows"));
        }
        let _ = op;
    }

    assert!(
        failures.is_empty(),
        "vacuum crash sweep found {} violation(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

/// Crash during an index build: the data commit must still succeed and a vector
/// search must return live rows (full-scan fallback when the index is missing).
#[tokio::test]
async fn crash_during_index_build_preserves_recall() -> anyhow::Result<()> {
    let _guard = serial_guard();
    let mut failures: Vec<String> = Vec::new();

    for &point in CrashPoint::ALL {
        let dir = tempdir()?;
        let uri = format!("file://{}", dir.path().display());
        seed(&uri).await?;

        // A new segment with an index build that may be aborted.
        let (op, table) = run_with_crash(&uri, point, |t| async move {
            t.write_async(vec![vec_batch(1000, ROWS_PER_SEGMENT)])
                .await?;
            t.commit_async().await
        })
        .await?;

        let ids = read_ids(&table).await?;
        let unique: HashSet<i32> = ids.iter().copied().collect();

        // Atomicity: 60 (pre) or 80 (post).
        if ids.len() != SEED_ROWS as usize && ids.len() != (SEED_ROWS + ROWS_PER_SEGMENT) as usize {
            failures.push(format!(
                "[{point}] index-build atomicity: {} rows (expected {SEED_ROWS} or {})",
                ids.len(),
                SEED_ROWS + ROWS_PER_SEGMENT
            ));
        }
        if unique.len() != ids.len() {
            failures.push(format!("[{point}] index-build duplicated rows"));
        }

        // A vector search must return live rows regardless of index state.
        let params = VectorSearchParams::new("embedding", VectorValue::Float32(vec![0.1; DIM]), 5);
        match table.read_async(None, Some(vec![params]), None).await {
            Ok(batches) => {
                let hits: usize = batches.iter().map(|b| b.num_rows()).sum();
                if hits == 0 {
                    failures.push(format!("[{point}] vector search returned no rows"));
                }
            }
            Err(e) => failures.push(format!("[{point}] vector search failed: {e}")),
        }
        let _ = op;
    }

    assert!(
        failures.is_empty(),
        "index-build crash sweep found {} violation(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}
