// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Covers `Table::sql_multi`, the primitive behind multi-table table functions
//! such as `graph_rag_search`.
//!
//! The interesting case is a statement that references two registered tables by
//! different aliases at once: the document table as a normal relation and the
//! edge table through the graph table function. Single-table `sql()` cannot
//! express this because it registers only `t`.

use anyhow::Result;
use arrow::array::{Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::Table;
use std::sync::Arc;

fn docs_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("title", DataType::Utf8, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["graph rag", "vector search", "graph database"])),
        ],
    )
    .expect("docs batch")
}

fn edges_batch() -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("source", DataType::Int64, false),
        Field::new("target", DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(Int64Array::from(vec![2, 3])),
        ],
    )
    .expect("edges batch")
}

#[tokio::test]
async fn sql_multi_joins_document_and_edge_tables() -> Result<()> {
    let root = tempfile::tempdir()?;
    let docs_uri = format!("file://{}/docs", root.path().display());
    let edges_uri = format!("file://{}/edges", root.path().display());

    let docs = Table::new_async(docs_uri.clone()).await?;
    docs.write_async(vec![docs_batch()]).await?;
    docs.commit_async().await?;

    let edges = Table::new_async(edges_uri.clone()).await?;
    edges.write_async(vec![edges_batch()]).await?;
    edges.commit_async().await?;

    let anchor = Table::new_async(docs_uri.clone()).await?;
    let batches = anchor
        .sql_multi(
            vec![("t".to_string(), docs_uri), ("e".to_string(), edges_uri)],
            "SELECT t.id FROM t \
             JOIN graph_neighbors('e', '1', 1, 'auto') g ON t.id = g.node",
        )
        .await?;

    let rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    assert!(
        rows >= 1,
        "expected at least one document reachable from seed 1, got {rows}"
    );

    Ok(())
}

#[tokio::test]
async fn sql_multi_rejects_an_empty_table_set() -> Result<()> {
    let root = tempfile::tempdir()?;
    let uri = format!("file://{}/docs", root.path().display());
    let table = Table::new_async(uri).await?;

    let error = table
        .sql_multi(vec![], "SELECT 1")
        .await
        .expect_err("an empty alias set must be rejected");
    assert!(
        error.to_string().contains("at least one table"),
        "unexpected error: {error}"
    );

    Ok(())
}
