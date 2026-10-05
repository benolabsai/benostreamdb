// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Regression test for multi-column Puffin index selection.
//!
//! A segment's Puffin bundle holds one index per column (e.g. a forward
//! `source` graph and a reverse `target` graph). The readers must select the
//! blobs belonging to the requested column; picking the last blob of each type
//! loaded the wrong adjacency, and because the blob order is not stable the
//! result was intermittent.

use std::sync::Arc;

use arrow::array::UInt64Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::core::table::Table;
use tempfile::tempdir;

/// Build a directed chain 0 -> 1 -> 2 -> ... -> n with BOTH a forward
/// (`source`) and a reverse (`target`) CSR graph index in the same segment.
async fn build_two_graph_table(uri: &str, n: u64) -> anyhow::Result<Table> {
    let table = Table::new_async(uri.to_string()).await?;
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
    table
        .add_index(
            "target".to_string(),
            IndexAlgorithm::CsrGraph {
                src_column: "target".to_string(),
                dst_column: "source".to_string(),
            },
        )
        .await?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("source", DataType::UInt64, false),
        Field::new("target", DataType::UInt64, false),
    ]));

    let mut src: Vec<u64> = Vec::new();
    let mut dst: Vec<u64> = Vec::new();
    for i in 0..n {
        src.push(i);
        dst.push(i + 1);
    }
    // Fan-out from 0 so the edge count is large enough to span partitions.
    for i in 2..n {
        src.push(0);
        dst.push(i);
    }

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(UInt64Array::from(src)),
            Arc::new(UInt64Array::from(dst)),
        ],
    )?;
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;
    drop(table);

    Table::new_async(uri.to_string()).await
}

/// `Table::shortest_path` must always resolve the forward `source` graph, even
/// when a reverse `target` graph shares the same Puffin bundle.
#[tokio::test]
async fn shortest_path_selects_the_requested_graph_column() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let table = build_two_graph_table(&uri, 25000).await?;

    // 0 -> 3 is a direct edge (the fan-out), and the reverse graph has no
    // out-edges from node 0, so loading it would return an empty path.
    for run in 0..20 {
        let path = table.shortest_path(0, 3, true, None).await?;
        assert_eq!(
            path,
            vec![0, 3],
            "shortest_path returned the wrong path on run {run}"
        );
    }

    Ok(())
}
