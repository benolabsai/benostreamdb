// Copyright (c) 2026 Richard Albright. All rights reserved.

//! WS2: swept crash-injection across every write/commit/maintenance boundary.
//!
//! For each named [`CrashPoint`], we run a write+commit that is aborted at that
//! boundary (modelling a `SIGKILL`), then re-open the table and assert the
//! **atomicity** and **durability** invariants:
//!
//! * Atomicity — the reopened table is either at the pre-write state (`N` rows)
//!   or the post-write state (`N + M` rows), never a torn in-between.
//! * No duplication — WAL replay must be idempotent: a batch that is already
//!   committed to the manifest must not be replayed a second time.
//! * Durability — if the operation returned `Ok`, the rows must be present.
//! * Recoverability — the table must re-open and its manifest must load.
//!
//! See `plans/production_readiness_plan.md` §WS2.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::fault_injection::{arm, disarm, CrashPoint};
use benostreamdb::Table;
use tempfile::tempdir;

const SEED_ROWS: i32 = 10;
const NEW_ROWS: i32 = 5;

fn batch(start: i32, n: i32) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let ids = Int32Array::from_iter_values(start..start + n);
    RecordBatch::try_new(schema, vec![Arc::new(ids)]).unwrap()
}

/// Read every `id` currently visible (committed segments + recovered WAL buffer).
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

/// Seed a table with `SEED_ROWS` committed rows and return its URI.
async fn seed_table(uri: &str) -> anyhow::Result<()> {
    let table = Table::new_async(uri.to_string()).await?;
    table.write_async(vec![batch(0, SEED_ROWS)]).await?;
    table.commit_async().await?;
    Ok(())
}

/// The core sweep: one injection point at a time.
#[tokio::test]
async fn crash_injection_sweep_preserves_atomicity_and_durability() -> anyhow::Result<()> {
    let mut failures: Vec<String> = Vec::new();

    for &point in CrashPoint::ALL {
        let dir = tempdir()?;
        let uri = format!("file://{}", dir.path().display());

        // 1. Baseline: SEED_ROWS committed.
        seed_table(&uri).await?;

        // 2. Inject a crash at `point` during a second write+commit.
        let op_result = {
            let table = Table::new_async(uri.clone()).await?;
            arm(point);
            let r = async {
                table.write_async(vec![batch(1000, NEW_ROWS)]).await?;
                table.commit_async().await
            }
            .await;
            disarm();
            // Let any background index task finish (or fail) before we reopen.
            let _ = table.wait_for_background_tasks_async().await;
            r
        };

        // 3. Re-open and verify invariants.
        let table = match Table::new_async(uri.clone()).await {
            Ok(t) => t,
            Err(e) => {
                failures.push(format!("[{point}] table failed to re-open: {e}"));
                continue;
            }
        };

        let ids = match read_ids(&table).await {
            Ok(v) => v,
            Err(e) => {
                failures.push(format!("[{point}] read failed after recovery: {e}"));
                continue;
            }
        };

        let total = ids.len() as i32;
        let unique: HashSet<i32> = ids.iter().copied().collect();

        // Atomicity: never a torn state.
        if total != SEED_ROWS && total != SEED_ROWS + NEW_ROWS {
            failures.push(format!(
                "[{point}] atomicity violated: {total} rows (expected {SEED_ROWS} or {})",
                SEED_ROWS + NEW_ROWS
            ));
        }

        // Idempotency: no duplicated rows from WAL replay.
        if unique.len() != ids.len() {
            failures.push(format!(
                "[{point}] WAL replay duplicated rows: {} rows, {} unique",
                ids.len(),
                unique.len()
            ));
        }

        // Durability: a successful operation must be fully visible.
        if op_result.is_ok() && total != SEED_ROWS + NEW_ROWS {
            failures.push(format!(
                "[{point}] operation returned Ok but only {total} rows are visible"
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "crash-injection sweep found {} violation(s):\n{}",
        failures.len(),
        failures.join("\n")
    );

    Ok(())
}
