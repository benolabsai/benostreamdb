// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Index-lifecycle invariants: secondary indexes must survive **compaction** and
//! **snapshot rollback**, must be **removed by `DROP TABLE`**, and must not be
//! left **orphaned** by `vacuum`.
//!
//! These lock in the guarantees the overlay architecture depends on:
//!   * compaction rebuilds the merged segment's indexes (never silently drops
//!     them, which would collapse every query to a full scan);
//!   * a rollback restores the target snapshot's data *and* its indexes (the
//!     index is bound to the data file by checksum, so it stays valid);
//!   * `DROP TABLE` deletes the whole table prefix, indexes included;
//!   * `vacuum` deletes index files no longer referenced by a retained version.

use std::sync::Arc;

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::core::sql::session::BenoStreamSession;
use benostreamdb::core::table::Table;
use tempfile::tempdir;

fn edge_batch(offset: i64) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("source", DataType::Int64, false),
        Field::new("target", DataType::Int64, false),
    ]));
    let src: Vec<i64> = vec![0, 0, 1, 2, 2, 2, 3]
        .into_iter()
        .map(|v| v + offset)
        .collect();
    let dst: Vec<i64> = vec![1, 2, 2, 0, 1, 3, 0]
        .into_iter()
        .map(|v| v + offset)
        .collect();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(src)),
            Arc::new(Int64Array::from(dst)),
        ],
    )
    .unwrap()
}

fn csr_index() -> IndexAlgorithm {
    IndexAlgorithm::CsrGraph {
        src_column: "source".to_string(),
        dst_column: "target".to_string(),
    }
}

fn count_files_with_suffix(dir: &str, suffix: &str) -> usize {
    std::fs::read_dir(dir)
        .map(|rd| {
            rd.flatten()
                .filter(|e| e.file_name().to_string_lossy().ends_with(suffix))
                .count()
        })
        .unwrap_or(0)
}

/// Compaction must rebuild the merged segment's indexes, not drop them.
#[tokio::test]
async fn compaction_preserves_indexes() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let table = Table::new_async(uri.clone()).await?;
    table.set_autocommit(false);
    table.add_index("source".to_string(), csr_index()).await?;

    // Two separate segments so compaction has something to merge.
    table.write_async(vec![edge_batch(0)]).await?;
    table.commit_async().await?;
    table.write_async(vec![edge_batch(100)]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let before = table.get_snapshot_segments().await?;
    assert!(
        before.iter().any(|e| !e.index_files.is_empty()),
        "expected at least one segment with index files before compaction"
    );

    table.rewrite_data_files_async(None).await?;
    table.wait_for_background_tasks_async().await?;

    let after = table.get_snapshot_segments().await?;
    assert_eq!(after.len(), 1, "compaction should merge into one segment");
    assert!(
        !after[0].index_files.is_empty(),
        "compaction dropped the merged segment's indexes — every query would \
         fall back to a full scan"
    );
    // The rebuilt index files must actually exist on disk.
    for idx in &after[0].index_files {
        let name = idx.file_path.rsplit('/').next().unwrap_or(&idx.file_path);
        assert!(
            std::path::Path::new(&path).join(name).exists()
                || std::path::Path::new(&idx.file_path).exists(),
            "compaction referenced index file {} that does not exist",
            idx.file_path
        );
    }
    Ok(())
}

/// A rollback restores the target snapshot's data *and* its indexes.
#[tokio::test]
async fn rollback_preserves_indexes() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let table = Table::new_async(uri.clone()).await?;
    table.set_autocommit(false);
    table.add_index("source".to_string(), csr_index()).await?;

    table.write_async(vec![edge_batch(0)]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;
    let (_, v_a) = table.get_snapshot_segments_with_version().await?;

    table.write_async(vec![edge_batch(100)]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    // Roll back to the first snapshot.
    table.rollback_to_snapshot(v_a as i64).await?;
    table.wait_for_background_tasks_async().await?;

    let entries = table.get_snapshot_segments().await?;
    assert!(
        entries.iter().any(|e| !e.index_files.is_empty()),
        "rollback lost the target snapshot's indexes"
    );
    // The restored index files must still be present on disk.
    let puffins = count_files_with_suffix(&path, ".puffin");
    assert!(
        puffins > 0,
        "rollback left no index bundle on disk for the restored snapshot"
    );
    Ok(())
}

/// `DROP TABLE` deletes the whole table prefix, indexes included.
#[tokio::test]
async fn drop_table_removes_index_files() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let wh = dir.path().to_str().unwrap().to_string();
    let table_dir = format!("{}/default/t", wh);

    let mut session = BenoStreamSession::new(None);
    session.set_warehouse(Some(wh.clone()));

    let table = Table::new_async(format!("file://{}", table_dir)).await?;
    table.set_autocommit(false);
    table.add_index("source".to_string(), csr_index()).await?;
    table.write_async(vec![edge_batch(0)]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    assert!(
        count_files_with_suffix(&table_dir, ".puffin") > 0,
        "expected an index bundle before DROP TABLE"
    );

    session.register_table("t", Arc::new(table))?;
    session.sql("DROP TABLE t").await?;

    assert!(
        !std::path::Path::new(&table_dir).exists()
            || count_files_with_suffix(&table_dir, ".puffin") == 0,
        "DROP TABLE left index files behind under {}",
        table_dir
    );
    Ok(())
}

/// `vacuum` deletes index files no longer referenced by a retained version.
#[tokio::test]
async fn vacuum_removes_orphaned_index_files() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let table = Table::new_async(uri.clone()).await?;
    table.set_autocommit(false);
    table.add_index("source".to_string(), csr_index()).await?;

    // v1: one segment + its index.
    table.write_async(vec![edge_batch(0)]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    // v2: compact, which writes a new segment + index and retires v1's.
    table.rewrite_data_files_async(None).await?;
    table.wait_for_background_tasks_async().await?;

    // Keep only the latest version; v1's index files become unreferenced.
    let deleted = table.vacuum_async(1).await?;
    assert!(deleted > 0, "vacuum should have deleted retired files");

    // The live segment's index must survive.
    let entries = table.get_snapshot_segments().await?;
    assert!(
        entries.iter().any(|e| !e.index_files.is_empty()),
        "vacuum deleted the live segment's indexes"
    );
    Ok(())
}
