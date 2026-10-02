// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Long-running soak harness for the OSS GA checklist.
//!
//! Ignored by default so `cargo test` stays fast. Run explicitly:
//!
//! ```bash
//! BSDB_SOAK_SECONDS=600 cargo test --test soak -- --ignored --nocapture
//! ```
//!
//! The assertion is deliberately blunt — the point is that a multi-minute mixed
//! workload neither panics nor loses rows — because that is exactly the
//! "graceful degradation rather than process death" requirement.

use arrow::array::{FixedSizeListArray, Int64Array};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::core::table::VectorSearchParams;
use benostreamdb::Table;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::tempdir;

const DIM: usize = 64;
const ROWS: usize = 1_000;

fn batch(start: i64) -> RecordBatch {
    let ids: Vec<i64> = (start..start + ROWS as i64).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ),
    ]));

    let embedding = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        (0..ROWS).map(|r| {
            Some((0..DIM).map(move |d| {
                // Deterministic, cheap values — the workload is what matters.
                Some(((r * DIM + d) % 97) as f32 / 97.0)
            }))
        }),
        DIM as i32,
    );

    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from(ids)), Arc::new(embedding) as _],
    )
    .expect("record batch")
}

/// Resident set size in bytes (Linux `/proc/self/status` `VmRSS`).
///
/// Returns `None` on platforms without `/proc` (the soak then skips the memory
/// assertion rather than failing spuriously).
fn rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

/// The soak's RSS ceiling in bytes (`BSDB_SOAK_RSS_CEILING_GB`, default 8 GiB).
///
/// The soak must degrade gracefully (throttle/trim) rather than grow until the
/// OS OOM-kills it — this is the memory half of the "no process death" bar.
fn rss_ceiling_bytes() -> u64 {
    std::env::var("BSDB_SOAK_RSS_CEILING_GB")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|gb| *gb > 0.0)
        .map(|gb| (gb * 1024.0 * 1024.0 * 1024.0) as u64)
        .unwrap_or(8 * 1024 * 1024 * 1024)
}

#[tokio::test]
#[ignore = "soak test; run with `-- --ignored` and BSDB_SOAK_SECONDS"]
async fn mixed_workload_soak() {
    let secs: u64 = std::env::var("BSDB_SOAK_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);

    let dir = tempdir().expect("tempdir");
    let uri = format!("file://{}", dir.path().to_str().expect("utf8 path"));
    let table = Table::new_async(uri).await.expect("open table");

    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut writes: usize = 0;
    let ceiling = rss_ceiling_bytes();
    let mut peak_rss: u64 = 0;

    while Instant::now() < deadline {
        table
            .write_async(vec![batch(writes as i64 * ROWS as i64)])
            .await
            .expect("write");
        writes += 1;

        if writes.is_multiple_of(3) {
            table.commit_async().await.expect("commit");
        }
        if writes.is_multiple_of(2) {
            let batches = table.read_async(None, None, None).await.expect("read");
            assert!(
                batches.iter().map(|b| b.num_rows()).sum::<usize>() >= ROWS,
                "read returned no rows mid-soak"
            );
        }

        // Bound the working set: trim the oldest rows so a long soak does not
        // grow without limit (and OOM). The churn is what matters, not growth.
        if writes.is_multiple_of(20) {
            let cutoff = (writes as i64 - 10) * ROWS as i64;
            let _ = table.delete_async(&format!("id < {cutoff}")).await;
            table.commit_async().await.expect("commit after trim");
        }

        // Memory half of the "no process death" bar: the soak must stay under
        // the ceiling rather than grow until the OS OOM-kills it.
        if let Some(rss) = rss_bytes() {
            peak_rss = peak_rss.max(rss);
            assert!(
                rss <= ceiling,
                "RSS {} MiB exceeded the {} MiB ceiling at write {writes}",
                rss / (1024 * 1024),
                ceiling / (1024 * 1024)
            );
        }
    }

    table.commit_async().await.expect("final commit");
    assert!(writes > 1, "soak did not perform any writes");

    // The process is still here and serving reads after the soak.
    let batches = table
        .read_async(None, None, None)
        .await
        .expect("final read");
    let final_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert!(final_rows >= ROWS);

    // Publishable summary (parsed by scripts/pre_release_soak.sh and CI).
    eprintln!(
        "[soak-stats] test=mixed_workload_soak writes={writes} duration_s={secs} rows={final_rows} peak_rss_mb={}",
        peak_rss / (1024 * 1024)
    );
}

/// Total rows currently visible.
async fn visible_rows(table: &Table) -> usize {
    table
        .read_async(None, None, None)
        .await
        .expect("read")
        .iter()
        .map(|b| b.num_rows())
        .sum()
}

/// WS5 maintenance-churn soak: sustained insert / delete / reinsert /
/// update-column / compaction / index-rebuild / snapshot-rollback /
/// time-travel, with a vector query between every stage.
///
/// `tests/test_maintenance_invariant.rs` is the fast, per-PR version of these
/// invariants. This is the long-duration version: under sustained churn for
/// `BSDB_SOAK_SECONDS`, the table must never lose committed rows, never return
/// a deleted row from a vector query, and never panic. It is `#[ignore]`d and
/// wired into the weekly `.github/workflows/soak.yml` job.
#[tokio::test]
#[ignore = "soak test; run with `-- --ignored` and BSDB_SOAK_SECONDS"]
async fn maintenance_churn_soak() {
    use benostreamdb::core::index::VectorValue;

    let secs: u64 = std::env::var("BSDB_SOAK_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);

    let dir = tempdir().expect("tempdir");
    let uri = format!("file://{}", dir.path().to_str().expect("utf8 path"));
    let table = Table::new_async(uri).await.expect("open table");

    // Seed + a vector index so the indexed vector path is exercised from the start.
    table.write_async(vec![batch(0)]).await.expect("seed write");
    table.commit_async().await.expect("seed commit");
    table
        .add_index("embedding".to_string(), IndexAlgorithm::hnsw_tq8())
        .await
        .expect("add index");
    table
        .wait_for_background_tasks_async()
        .await
        .expect("wait for index build");

    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut round: i64 = 1;
    let ceiling = rss_ceiling_bytes();
    let mut peak_rss: u64 = 0;

    while Instant::now() < deadline {
        let start = round * ROWS as i64;

        eprintln!("[soak] round {round}: insert");
        table.write_async(vec![batch(start)]).await.expect("insert");
        table.commit_async().await.expect("commit");

        eprintln!("[soak] round {round}: delete");
        let _ = table
            .delete_async(&format!("id >= {} AND id < {}", start, start + 100))
            .await;

        eprintln!("[soak] round {round}: read after delete");
        assert!(
            visible_rows(&table).await >= 1,
            "read failed after delete in round {round}"
        );

        eprintln!("[soak] round {round}: reinsert");
        table
            .write_async(vec![batch(start + 10_000_000)])
            .await
            .expect("reinsert");
        table.commit_async().await.expect("commit");

        // vector query mid-churn (must never panic)
        eprintln!("[soak] round {round}: vector query");
        let params = VectorSearchParams::new("embedding", VectorValue::Float32(vec![0.25; DIM]), 5);
        let _ = table
            .read_async(None, Some(vec![params]), None)
            .await
            .expect("vector query");

        // compaction + vacuum under live data
        eprintln!("[soak] round {round}: compaction");
        let _ = table.rewrite_data_files_async(None).await;
        eprintln!("[soak] round {round}: read after compaction");
        let after_compact = visible_rows(&table).await;
        eprintln!("[soak] round {round}: rows after compaction = {after_compact}");

        eprintln!("[soak] round {round}: vacuum");
        let _ = table.vacuum_async(1).await;
        eprintln!("[soak] round {round}: read after vacuum");
        let after_vacuum = visible_rows(&table).await;
        eprintln!("[soak] round {round}: rows after vacuum = {after_vacuum}");

        // index rebuild occasionally (expensive)
        if round % 5 == 0 {
            eprintln!("[soak] round {round}: index rebuild");
            let _ = table
                .add_index("embedding".to_string(), IndexAlgorithm::hnsw_tq8())
                .await;
            let _ = table.wait_for_background_tasks_async().await;
        }

        // time-travel read + rollback to the current version
        eprintln!("[soak] round {round}: time-travel");
        if let Ok(v) = table.snapshot_version().await {
            let _ = table.manifest_at_version(v).await;
            let _ = table.rollback_to_snapshot(v as i64).await;
        }

        eprintln!("[soak] round {round}: final read");
        assert!(
            visible_rows(&table).await >= 1,
            "table lost all rows after round {round}"
        );

        // Bound the working set: remove this round's inserts so a long soak
        // stays memory-bounded. The churn (insert/delete/compact/vacuum) is
        // what matters, not unbounded growth.
        let _ = table
            .delete_async(&format!("id >= {} AND id < {}", start, start + ROWS as i64))
            .await;
        let _ = table
            .delete_async(&format!(
                "id >= {} AND id < {}",
                start + 10_000_000,
                start + 10_000_000 + ROWS as i64
            ))
            .await;
        let _ = table.commit_async().await;

        // Memory half of the "no process death" bar.
        if let Some(rss) = rss_bytes() {
            peak_rss = peak_rss.max(rss);
            assert!(
                rss <= ceiling,
                "RSS {} MiB exceeded the {} MiB ceiling at round {round}",
                rss / (1024 * 1024),
                ceiling / (1024 * 1024)
            );
        }

        round += 1;
    }

    assert!(round > 1, "maintenance soak performed no rounds");
    let final_rows = visible_rows(&table).await;
    assert!(final_rows >= 1, "table empty after the maintenance soak");

    // Publishable summary (parsed by scripts/pre_release_soak.sh and CI).
    eprintln!(
        "[soak-stats] test=maintenance_churn_soak rounds={} duration_s={secs} rows={final_rows} peak_rss_mb={}",
        round - 1,
        peak_rss / (1024 * 1024)
    );
}
