// Copyright (c) 2026 Richard Albright. All rights reserved.

//! WS1 regression: index-management code paths must resolve the **tiered**
//! manifest list, not the (always-empty) inline `Manifest.entries`.
//!
//! BenoStreamDB commits through tiered manifests (`ManifestList` → `*.avro`),
//! so `Manifest.entries` is empty by design. Any code that iterates it directly
//! silently sees zero segments. Two such paths were fixed:
//!
//! 1. `drop_index` — collected index file paths from `manifest.entries`, so it
//!    never found any and left every index file orphaned on disk.
//! 2. `infer_index_metadata_from_physical_async` — the "recover indexes from
//!    physical files" path was a no-op for tiered tables.
//!
//! See `plans/production_readiness_plan.md` §2.2.

use std::sync::Arc;

use arrow::array::{Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::manifest::{IndexAlgorithm, ManifestManager};
use benostreamdb::core::storage::create_object_store;
use benostreamdb::core::table::Table;
use tempfile::tempdir;

fn id_batch(start: i32, n: i32) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let ids = Int32Array::from_iter_values(start..start + n);
    RecordBatch::try_new(schema, vec![Arc::new(ids)]).unwrap()
}

fn edge_batch() -> anyhow::Result<RecordBatch> {
    let schema = Arc::new(Schema::new(vec![
        Field::new("source", DataType::Int64, false),
        Field::new("target", DataType::Int64, false),
    ]));
    // 0 -> 1, 2 ; 1 -> 2 ; 2 -> 0, 1, 3 ; 3 -> 0
    let src: Vec<i64> = vec![0, 0, 1, 2, 2, 2, 3];
    let dst: Vec<i64> = vec![1, 2, 2, 0, 1, 3, 0];
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(src)),
            Arc::new(Int64Array::from(dst)),
        ],
    )?)
}

/// `drop_index` must delete the physical index files even though the segment
/// entry lives in the tiered manifest list (not inline in `Manifest.entries`).
#[tokio::test]
async fn drop_index_removes_files_for_tiered_manifest() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let table = Table::new_async(uri.clone()).await?;
    table.set_autocommit(false);
    table
        .add_index(
            "source".to_string(),
            IndexAlgorithm::CsrGraph {
                src_column: "source".to_string(),
                dst_column: "target".to_string(),
            },
        )
        .await?;

    table.write_async(vec![edge_batch()?]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // The CSR artifacts must exist on disk before we drop the index.
    let csr_suffixes = [
        ".graph_v2.csr.offsets",
        ".graph_v2.csr.edges",
        ".graph_v2.csr.dict",
    ];
    let find_csr = |suffix: &str| -> Option<std::path::PathBuf> {
        std::fs::read_dir(&path)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .find(|p| {
                p.file_name()
                    .map(|n| n.to_string_lossy().ends_with(suffix))
                    .unwrap_or(false)
            })
    };
    for suffix in csr_suffixes {
        assert!(
            find_csr(suffix).is_some(),
            "expected a `{suffix}` file on disk before drop_index"
        );
    }

    // Drop the index. With the tiered-manifest bug this was a no-op: the files
    // stayed on disk forever.
    table.drop_index("source".to_string()).await?;

    for suffix in csr_suffixes {
        assert!(
            find_csr(suffix).is_none(),
            "drop_index left `{suffix}` orphaned on disk — it did not resolve the \
             tiered manifest list"
        );
    }

    Ok(())
}

/// The append-only fast path references the previous manifest files instead of
/// re-encoding every entry (see `plans/production_readiness_plan.md` §8.7).
/// The accessor base (`load_all_entries`) — which every index-management path
/// builds on — must walk the *whole* manifest list, not just the newest
/// manifest file.
///
/// Note: an index-attach commit carries `remove_paths` (it replaces the
/// unindexed entry with the indexed one), so it always takes the full-rewrite
/// path and consolidates the list. A multi-file list therefore only arises from
/// pure data appends, which is what this test exercises.
#[tokio::test]
async fn load_all_entries_resolves_append_only_manifest_list() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    let table = Table::new_async(uri.clone()).await?;

    // Two pure appends -> the manifest list references the previous manifest
    // file plus a new one.
    table.write_async(vec![id_batch(0, 10)]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;
    table.write_async(vec![id_batch(10, 10)]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let store = create_object_store(&uri)?;
    let manager = ManifestManager::new(store, "", &uri);
    let (manifest, _) = manager.load_latest().await?;
    let list = manager
        .load_manifest_list(manifest.manifest_list_path.as_ref().expect("list path"))
        .await?;
    assert!(
        list.manifest_files.len() >= 2,
        "expected an append-only manifest list with >= 2 files, got {}",
        list.manifest_files.len()
    );

    // Every segment, from every referenced manifest file, must be visible.
    let entries = manager.load_all_entries(&manifest).await?;
    assert_eq!(entries.len(), 2, "expected two segments");
    let rows: i64 = entries.iter().map(|e| e.record_count).sum();
    assert_eq!(
        rows, 20,
        "all segments across the manifest list must be visible"
    );

    Ok(())
}

/// The read-path byte cache keys on the parquet path, so it relies on segment
/// files being immutable: a path must never be reused for different content.
/// Assert that a later commit neither drops nor reuses an existing segment path,
/// and that repeated reads return identical data.
#[tokio::test]
async fn segment_paths_are_immutable_across_commits() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    let table = Table::new_async(uri.clone()).await?;
    table.write_async(vec![id_batch(0, 10)]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let store = create_object_store(&uri)?;
    let manager = ManifestManager::new(store, "", &uri);
    let (m1, _) = manager.load_latest().await?;
    let paths1: Vec<String> = manager
        .load_all_entries(&m1)
        .await?
        .iter()
        .map(|e| e.file_path.clone())
        .collect();
    assert!(!paths1.is_empty(), "first commit produced no segment");

    table.write_async(vec![id_batch(10, 10)]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let (m2, _) = manager.load_latest().await?;
    let paths2: Vec<String> = manager
        .load_all_entries(&m2)
        .await?
        .iter()
        .map(|e| e.file_path.clone())
        .collect();

    // The first commit's segment path must still be present and unchanged.
    for p in &paths1 {
        assert!(
            paths2.contains(p),
            "segment path {p} disappeared after a later commit"
        );
    }
    // The new commit must add a fresh path, never reuse an existing one.
    let new_paths: Vec<&String> = paths2.iter().filter(|p| !paths1.contains(p)).collect();
    assert!(
        !new_paths.is_empty(),
        "the second commit added no new segment path"
    );

    // Repeated reads must be identical (the byte cache must not serve stale
    // bytes for a reused path).
    let a = table.sql("SELECT id FROM t").await?;
    let b = table.sql("SELECT id FROM t").await?;
    let rows_a: usize = a.iter().map(|batch| batch.num_rows()).sum();
    let rows_b: usize = b.iter().map(|batch| batch.num_rows()).sum();
    assert_eq!(rows_a, 20, "expected 20 rows after two commits");
    assert_eq!(rows_a, rows_b, "repeated reads diverged");

    Ok(())
}
