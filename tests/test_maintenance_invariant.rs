// Copyright (c) 2026 Richard Albright. All rights reserved.

//! WS5: the Maintenance Invariant under sustained churn.
//!
//! Covers the three deliverables of WS5 in
//! `plans/production_readiness_plan.md`:
//!
//! 1. A bounded soak: insert → delete → reinsert → compaction → vacuum →
//!    vector query, asserting the table never loses committed rows.
//! 2. The **GC-vs-reader race**: a version is pinned, maintenance runs, and the
//!    pinned version's artifacts must survive. The mechanism is **version-based
//!    retention** (the vacuum window `[latest-retention+1, latest]` plus a
//!    re-validation against the latest manifest) — *not* explicit reader
//!    leases, so a reader must pin a version inside the retention window. The
//!    companion test documents that bound.
//! 3. Delete semantics crossed with the vector (HNSW) path: a deleted row must
//!    never surface as a search candidate.

use std::sync::Arc;

use arrow::array::{FixedSizeListArray, Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::index::VectorValue;
use benostreamdb::core::manifest::{IndexAlgorithm, ManifestManager};
use benostreamdb::core::storage::create_object_store;
use benostreamdb::core::table::VectorSearchParams;
use benostreamdb::Table;
use object_store::path::Path as ObjPath;
use object_store::ObjectStore;
use tempfile::tempdir;

const DIM: usize = 8;

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

async fn count(table: &Table) -> anyhow::Result<i64> {
    let batches = table.sql("SELECT count(*) FROM t").await?;
    Ok(batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0))
}

async fn table_version(table: &Table) -> anyhow::Result<u64> {
    table.snapshot_version().await
}

/// Data + delete file paths referenced by a specific manifest version.
async fn data_files_at_version(table: &Table, version: u64) -> anyhow::Result<Vec<String>> {
    let manifest = table.manifest_at_version(version).await?;
    let manager = ManifestManager::new(table.store.clone(), "", &table.uri);
    let entries = manager.load_all_entries(&manifest).await?;
    let mut paths: Vec<String> = entries.iter().map(|e| e.file_path.clone()).collect();
    for e in &entries {
        for f in &e.delete_files {
            paths.push(f.file_path.clone());
        }
    }
    Ok(paths)
}

/// A full-column read after a *range* delete + compaction must still succeed
/// (regression for a non-nullable column gaining nulls through the read path).
#[tokio::test]
async fn compaction_after_range_delete_keeps_columns_readable() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());
    let table = Table::new_async(uri).await?;

    table
        .add_index("embedding".to_string(), IndexAlgorithm::hnsw_tq8())
        .await?;

    table.write_async(vec![vec_batch(0, 20)]).await?;
    table.commit_async().await?;
    table.write_async(vec![vec_batch(1000, 20)]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    table.delete_async("id >= 1000 AND id < 1005").await?;
    table.rewrite_data_files_async(None).await?;

    // Full-column read (materializes `id` and `embedding`).
    let batches = table.read_async(None, None, None).await?;
    let total: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 35, "range delete + compaction must leave 35 rows");

    Ok(())
}

/// Regression for the WS5 finding (plan §2.6): compaction must physically
/// remove deleted rows, not resurrect them by re-reading the raw data files.
#[tokio::test]
async fn compaction_does_not_resurrect_deleted_rows() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());
    let table = Table::new_async(uri).await?;

    for r in 0..3 {
        table.write_async(vec![vec_batch(r * 100, 20)]).await?;
        table.commit_async().await?;
    }
    assert_eq!(count(&table).await?, 60);

    // Delete one row in the middle segment (id=100).
    table.delete_async("id = 100").await?;
    assert_eq!(count(&table).await?, 59);

    // Compact: the delete must be materialized, not lost.
    table.rewrite_data_files_async(None).await?;
    assert_eq!(
        count(&table).await?,
        59,
        "compaction must not resurrect a deleted row"
    );

    Ok(())
}

/// 1. Bounded soak: churn must never lose committed rows.
#[tokio::test]
async fn soak_churn_never_loses_committed_rows() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());
    let table = Table::new_async(uri).await?;

    // Round 1: 3 segments, each 20 rows.
    for r in 0..3 {
        table.write_async(vec![vec_batch(r * 100, 20)]).await?;
        table.commit_async().await?;
    }
    assert_eq!(count(&table).await?, 60);

    // Delete one row; reinsert it as a new segment.
    table.delete_async("id = 100").await?;
    assert_eq!(count(&table).await?, 59);

    table.write_async(vec![vec_batch(500, 20)]).await?;
    table.commit_async().await?;
    assert_eq!(count(&table).await?, 79);

    // Maintenance under live reads.
    table.rewrite_data_files_async(None).await?;
    assert_eq!(count(&table).await?, 79, "compaction must preserve rows");
    table.vacuum_async(1).await?;
    assert_eq!(count(&table).await?, 79, "vacuum must preserve live rows");

    // Vector query still works after churn + maintenance.
    let params = VectorSearchParams::new("embedding", VectorValue::Float32(vec![0.1; DIM]), 5);
    let rows: usize = table
        .read_async(None, Some(vec![params]), None)
        .await?
        .iter()
        .map(|b| b.num_rows())
        .sum();
    assert!(rows > 0, "vector query must return rows after churn");

    Ok(())
}

/// 2a. GC-vs-reader race: a version pinned *inside* the retention window must
/// survive vacuum — its manifest and its data files must not be reclaimed.
#[tokio::test]
async fn pinned_version_within_retention_survives_vacuum() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());
    let store: Arc<dyn ObjectStore> = create_object_store(&uri)?;
    let table = Table::new_async(uri).await?;

    for r in 0..3 {
        table.write_async(vec![vec_batch(r * 100, 10)]).await?;
        table.commit_async().await?;
    }
    let pinned = table_version(&table).await?;
    let pinned_files = data_files_at_version(&table, pinned).await?;
    assert!(!pinned_files.is_empty());

    // Advance to v5 so the window [latest-2, latest] still covers the pinned v3.
    for r in 3..5 {
        table.write_async(vec![vec_batch(r * 100, 10)]).await?;
        table.commit_async().await?;
    }
    let latest = table_version(&table).await?;
    assert!(latest > pinned);

    // Retention 3 keeps [latest-2, latest], which includes the pinned version.
    table.vacuum_async(3).await?;

    for f in &pinned_files {
        assert!(
            store.head(&ObjPath::from(f.as_str())).await.is_ok(),
            "vacuum reclaimed a file referenced by the pinned version {pinned}: {f}"
        );
    }
    // The pinned manifest itself must still be readable.
    table.manifest_at_version(pinned).await?;

    Ok(())
}

/// 2b. Documents the retention *mechanism*: vacuum reclaims manifest versions
/// outside the retention window. A long-lived reader must pin a version inside
/// the window (there are no reader leases); this asserts the bound explicitly.
#[tokio::test]
async fn vacuum_reclaims_manifests_outside_retention_window() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());
    let store: Arc<dyn ObjectStore> = create_object_store(&uri)?;
    let table = Table::new_async(uri).await?;

    for r in 0..5 {
        table.write_async(vec![vec_batch(r * 100, 10)]).await?;
        table.commit_async().await?;
    }
    let latest = table_version(&table).await?;

    // Retention 2 keeps only the two newest manifest versions.
    table.vacuum_async(2).await?;

    let manifest_exists = |v: u64| {
        let store = store.clone();
        async move {
            let p = ObjPath::from(format!("_manifest/v{v}.json"));
            store.head(&p).await.is_ok()
        }
    };

    assert!(
        manifest_exists(latest).await && manifest_exists(latest - 1).await,
        "manifests inside the retention window must be kept"
    );
    assert!(
        !manifest_exists(latest - 4).await,
        "manifests outside the retention window must be reclaimed"
    );

    Ok(())
}

/// 3. Delete semantics crossed with the vector path: a deleted row must never be
/// returned as a vector-search candidate.
#[tokio::test]
async fn deleted_row_is_not_a_vector_search_candidate() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());
    let table = Table::new_async(uri.clone()).await?;

    // A vector index so the search uses the index path, not a flat scan.
    table
        .add_index("embedding".to_string(), IndexAlgorithm::hnsw_tq8())
        .await?;

    table.write_async(vec![vec_batch(0, 40)]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    // Delete one row, then query with its exact embedding.
    table.delete_async("id = 7").await?;

    let target = (0..DIM)
        .map(|d| ((7usize * DIM + d) % 31) as f32 / 31.0)
        .collect::<Vec<_>>();
    let params = VectorSearchParams::new("embedding", VectorValue::Float32(target), 10);
    let out = table.read_async(None, Some(vec![params]), None).await?;

    let mut ids = Vec::new();
    for b in &out {
        if let Some(col) = b.column_by_name("id") {
            if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
                ids.extend(arr.iter().flatten());
            }
        }
    }
    assert!(
        !ids.contains(&7),
        "deleted row id=7 must not be a vector-search candidate: {ids:?}"
    );
    // Other rows must still be returned.
    assert!(!ids.is_empty(), "search must still return live rows");

    Ok(())
}

/// A write after a rollback must produce a consistent, readable table.
#[tokio::test]
async fn write_after_rollback_is_consistent() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());
    let table = Table::new_async(uri).await?;

    table.write_async(vec![vec_batch(0, 10)]).await?;
    table.commit_async().await?;
    let v1 = table_version(&table).await?;

    table.write_async(vec![vec_batch(100, 10)]).await?;
    table.commit_async().await?;

    table.rollback_to_snapshot(v1 as i64).await?;
    assert_eq!(count(&table).await?, 10);

    // Write again after the rollback, then read.
    table.write_async(vec![vec_batch(200, 10)]).await?;
    table.commit_async().await?;
    let c = count(&table).await?;
    assert_eq!(c, 20, "write after rollback must be readable and additive");

    Ok(())
}

/// Snapshot rollback restores the row count of an earlier version.
#[tokio::test]
async fn rollback_restores_earlier_snapshot() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());
    let table = Table::new_async(uri).await?;

    table.write_async(vec![vec_batch(0, 10)]).await?;
    table.commit_async().await?;
    let v1 = table_version(&table).await?;
    assert_eq!(count(&table).await?, 10);

    for r in 1..3 {
        table.write_async(vec![vec_batch(r * 100, 10)]).await?;
        table.commit_async().await?;
    }
    assert_eq!(count(&table).await?, 30);

    table.rollback_to_snapshot(v1 as i64).await?;
    assert_eq!(
        count(&table).await?,
        10,
        "rollback must restore the snapshot's row count"
    );

    Ok(())
}
