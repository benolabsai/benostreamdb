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

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::core::table::Table;
use tempfile::tempdir;

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
        std::fs::read_dir(&path).ok()?.flatten().map(|e| e.path()).find(|p| {
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
