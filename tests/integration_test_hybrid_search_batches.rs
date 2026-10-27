// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Covers `Table::hybrid_search_batches`, the entry point behind the Trino
//! `system.hybrid_search` table function and its JNI shim.
//!
//! The Java-side tests prove the SQL surface; this proves the engine contract:
//! vector and BM25 results are fused with RRF and the hydrated rows carry an
//! `rrf_score` column. Keeping it here means a regression in the fusion or the
//! column rename fails without needing a JVM.

use anyhow::Result;
use arrow::array::{Float32Builder, Int64Array, ListBuilder, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::Table;
use std::sync::Arc;

fn sample_batch() -> RecordBatch {
    let ids = Int64Array::from(vec![1, 2, 3]);
    let titles = StringArray::from(vec!["graph rag", "vector search", "graph database"]);

    let mut embedding = ListBuilder::new(Float32Builder::new());
    for vector in [[1.0f32, 0.0f32], [0.0, 1.0], [0.9, 0.1]] {
        embedding.values().append_value(vector[0]);
        embedding.values().append_value(vector[1]);
        embedding.append(true);
    }

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("title", DataType::Utf8, false),
        Field::new(
            "embedding",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            false,
        ),
    ]));

    RecordBatch::try_new(
        schema,
        vec![Arc::new(ids), Arc::new(titles), Arc::new(embedding.finish())],
    )
    .expect("sample batch")
}

#[tokio::test]
async fn hybrid_search_batches_fuses_vector_and_keyword() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());

    let table = Table::new_async(uri).await?;
    table.write_async(vec![sample_batch()]).await?;
    table.commit_async().await?;

    table
        .add_index("embedding".to_string(), IndexAlgorithm::hnsw())
        .await?;
    table
        .add_index(
            "title".to_string(),
            IndexAlgorithm::Bm25 {
                k1: 1.2,
                b: 0.75,
                tokenizer: "standard".to_string(),
            },
        )
        .await?;
    // Index building is a separate phase; without this the search below can run
    // against an unbuilt index. rebuild_index takes a column (the JNI
    // `build_index('all')` form maps to recovery, not to a column named "all").
    table.rebuild_index("embedding".to_string()).await?;
    table.rebuild_index("title".to_string()).await?;

    let batches = table
        .hybrid_search_batches("title", "graph", "embedding", vec![1.0, 0.0], 2, 60.0)
        .await?;

    let rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
    assert!(rows >= 1, "expected at least one fused row, got {rows}");
    assert!(
        rows <= 2,
        "k=2 must cap the fused result set, got {rows} rows"
    );

    // The connector renames the engine's `distance` column to `rrf_score`; the
    // engine method must already expose the fused score under that name.
    let schema = batches[0].schema();
    let score_index = schema
        .index_of("rrf_score")
        .expect("rrf_score column should be present");
    let scores = batches[0].column(score_index);
    assert!(scores.len() >= 1, "rrf_score should have a value per row");

    Ok(())
}

#[tokio::test]
async fn hybrid_search_with_zero_k_returns_nothing() -> Result<()> {
    let dir = tempfile::tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());

    let table = Table::new_async(uri).await?;
    table.write_async(vec![sample_batch()]).await?;
    table.commit_async().await?;

    let batches = table
        .hybrid_search_batches("title", "graph", "embedding", vec![1.0, 0.0], 0, 60.0)
        .await?;
    assert!(batches.is_empty(), "k=0 must short-circuit to no batches");

    Ok(())
}
