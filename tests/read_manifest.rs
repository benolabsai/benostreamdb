// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Manifest read-path integration test.
//!
//! Originally a one-off debug binary that dumped a hardcoded pytest temp
//! directory. Converted into a self-contained test: it builds a table, commits
//! data plus an index, then reads the manifest back through the same accessor
//! the readers use (`ManifestManager::load_all_entries`) and asserts the data
//! entry and its index file are registered.
//!
//! This exercises the tiered-manifest read path (entries live in the manifest
//! list, not inline in `Manifest.entries`) — see
//! `plans/production_readiness_plan.md` §2.2.

use std::sync::Arc;

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::manifest::{IndexAlgorithm, ManifestManager};
use benostreamdb::core::storage::create_object_store;
use benostreamdb::Table;
use tempfile::tempdir;

fn batch(start: i32, n: i32) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let ids = Int32Array::from_iter_values(start..start + n);
    RecordBatch::try_new(schema, vec![Arc::new(ids)]).unwrap()
}

#[tokio::test]
async fn manifest_round_trip_registers_data_and_index() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    // Build a table with a bitmap index on `id`, then commit data.
    {
        let table = Table::new_async(uri.clone()).await?;
        table
            .add_index("id".to_string(), IndexAlgorithm::Bitmap)
            .await?;
        table.write_async(vec![batch(0, 25)]).await?;
        table.commit_async().await?;
        table.wait_for_background_tasks_async().await?;
    }

    // Read the manifest back through the reader's accessor.
    let store = create_object_store(&uri)?;
    let manager = ManifestManager::new(store, "", &uri);
    let (manifest, _) = manager.load_latest().await?;
    let entries = manager.load_all_entries(&manifest).await?;

    assert!(
        !entries.is_empty(),
        "the committed segment must be visible via load_all_entries"
    );

    let total_rows: i64 = entries.iter().map(|e| e.record_count).sum();
    assert_eq!(
        total_rows, 25,
        "manifest must report the committed row count"
    );

    // The data file must be registered and exist on disk.
    for entry in &entries {
        let full = format!("{}/{}", dir.path().display(), entry.file_path);
        assert!(
            std::path::Path::new(&full).exists(),
            "manifest references a missing data file: {}",
            entry.file_path
        );
    }

    // An index on the `id` column must be registered in the manifest.
    let all_index_files: Vec<String> = entries
        .iter()
        .flat_map(|e| e.index_files.iter())
        .map(|f| format!("{}:{}:{:?}", f.index_type, f.file_path, f.column_name))
        .collect();
    let has_id_index = entries
        .iter()
        .flat_map(|e| e.index_files.iter())
        .any(|f| f.column_name.as_deref() == Some("id"));
    assert!(
        has_id_index,
        "an index on `id` must be registered in the manifest; got {all_index_files:?}"
    );

    Ok(())
}

/// The append-only fast path must (a) reference the previous manifest files
/// instead of re-encoding every entry, and (b) consolidate back to a single
/// manifest file once a commit carries removes (compaction).
#[tokio::test]
async fn append_only_commit_grows_then_consolidates() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    let table = Table::new_async(uri.clone()).await?;

    // First commit: one manifest file.
    table.write_async(vec![batch(0, 10)]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let store = create_object_store(&uri)?;
    let manager = ManifestManager::new(store, "", &uri);

    let (m1, _) = manager.load_latest().await?;
    let list1 = manager
        .load_manifest_list(m1.manifest_list_path.as_ref().expect("list path"))
        .await?;
    assert_eq!(
        list1.manifest_files.len(),
        1,
        "first commit should produce exactly one manifest file"
    );

    // Second commit: a pure append. The fast path must reference the previous
    // manifest file and add exactly one new one (no full re-encode).
    table.write_async(vec![batch(10, 10)]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let (m2, _) = manager.load_latest().await?;
    let list2 = manager
        .load_manifest_list(m2.manifest_list_path.as_ref().expect("list path"))
        .await?;
    assert_eq!(
        list2.manifest_files.len(),
        2,
        "append-only commit should reference the previous manifest file plus one new one"
    );

    // Correctness: both segments must still be visible.
    let entries2 = manager.load_all_entries(&m2).await?;
    let rows2: i64 = entries2.iter().map(|e| e.record_count).sum();
    assert_eq!(rows2, 20, "append-only commit must preserve all rows");

    // Compaction carries removes, so it must fall back to the full rewrite and
    // consolidate the manifest list back to a single file.
    table.rewrite_data_files_async(None).await?;
    table.wait_for_background_tasks_async().await?;

    let (m3, _) = manager.load_latest().await?;
    let list3 = manager
        .load_manifest_list(m3.manifest_list_path.as_ref().expect("list path"))
        .await?;
    assert_eq!(
        list3.manifest_files.len(),
        1,
        "a commit with removes must consolidate the manifest list"
    );

    let entries3 = manager.load_all_entries(&m3).await?;
    let rows3: i64 = entries3.iter().map(|e| e.record_count).sum();
    assert_eq!(rows3, 20, "compaction must preserve all rows");

    Ok(())
}
