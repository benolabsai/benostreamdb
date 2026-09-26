// Copyright (c) 2026 Richard Albright. All rights reserved.

//! WS1: the graph CSR index must be registered in the manifest under the **v2**
//! on-disk format, and the files the reader looks for must actually exist.
//!
//! This is the path-convention check for the graph index — the bug class that
//! produced the v1→v2 migration issue: the writer and the reader must agree on
//! the artifact naming, or the reader silently falls back to the SQL BFS path.
//! See `plans/production_readiness_plan.md` §3 (WS1).

use std::sync::Arc;

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::core::table::Table;
use tempfile::tempdir;

/// WS1 path-conformance: after `add_index` + `write_async` + `commit_async`, the
/// CSR files must be registered in the manifest under the **v2** on-disk format
/// and the files the reader looks for must actually exist.
///
/// NOTE on the accessor: BenoStreamDB commits through *tiered manifests*
/// (`ManifestList` → `*.avro` manifest files), so `Manifest.entries` is empty by
/// design — the real entries live in the manifest list. The reader
/// (`load_multi_csr`, `get_scalar_filter_bitmap`, `explain`) resolves them via
/// `ManifestManager::load_all_entries(&manifest)`. This test therefore uses the
/// same accessor the reader uses; asserting on `manifest.entries` directly would
/// always see an empty list and produce a false negative.
#[tokio::test]
async fn graph_csr_index_is_registered_as_v2_with_files_on_disk() -> anyhow::Result<()> {
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

    let schema = Arc::new(Schema::new(vec![
        Field::new("source", DataType::Int64, false),
        Field::new("target", DataType::Int64, false),
    ]));
    // 0 -> 1, 2 ; 1 -> 2 ; 2 -> 0, 1, 3 ; 3 -> 0
    let src: Vec<i64> = vec![0, 0, 1, 2, 2, 2, 3];
    let dst: Vec<i64> = vec![1, 2, 2, 0, 1, 3, 0];
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(src)),
            Arc::new(Int64Array::from(dst)),
        ],
    )?;
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    // The index build runs in a background task that commits a follow-up
    // manifest version; give it a moment to land before re-opening.
    tokio::time::sleep(std::time::Duration::from_secs(2)).await;

    // Re-open the table: index registration is deferred to open-time physical
    // inference (`infer_index_metadata_from_physical_async`), not commit time.
    drop(table);
    let table = Table::new_async(uri).await?;

    let manifest = table.manifest().await?;
    eprintln!("manifest version: {}", manifest.version);

    // Resolve the tiered manifest list exactly as the reader does.
    let manager =
        benostreamdb::core::manifest::ManifestManager::new(table.store.clone(), "", &table.uri);
    let entries = manager.load_all_entries(&manifest).await?;

    let all_index_files: Vec<_> = entries
        .iter()
        .flat_map(|e| e.index_files.iter())
        .map(|f| format!("{}:{}", f.index_type, f.file_path))
        .collect();
    eprintln!("resolved index_files: {all_index_files:?}");
    let dir_files: Vec<_> = std::fs::read_dir(&path)?
        .flatten()
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    eprintln!("dir files: {dir_files:?}");

    let graph_entries: Vec<_> = entries
        .iter()
        .flat_map(|e| e.index_files.iter())
        .filter(|f| f.index_type == "graph_v2")
        .collect();

    assert!(
        !graph_entries.is_empty(),
        "no `graph_v2` index registered in the manifest — the reader's \
         `load_multi_csr` only trusts v2, so the CSR fast path would be silently \
         unavailable"
    );

    for f in &graph_entries {
        for suffix in [
            ".graph_v2.csr.offsets",
            ".graph_v2.csr.edges",
            ".graph_v2.csr.dict",
        ] {
            let rel = format!("{}{}", f.file_path, suffix);
            let full = if rel.starts_with('/') {
                rel.clone()
            } else {
                format!("{}/{}", path, rel)
            };
            assert!(
                std::path::Path::new(&full).exists(),
                "reader expects `{full}` but it does not exist — writer/reader \
                 path divergence for the graph CSR"
            );
        }
    }

    Ok(())
}
