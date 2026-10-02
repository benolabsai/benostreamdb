// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
//
//! Vector-search latency smoke test.
//!
//! Builds a small vector table with an HNSW/TQ8 sidecar index and asserts that
//! `execute_vector_search_as_scored` returns a bounded, non-empty result set.

use std::sync::Arc;

use arrow::array::{FixedSizeListArray, Int32Array};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;

use benostreamdb::core::index::VectorValue;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::core::table::Table;
use benostreamdb::VectorSearchParams;

fn vector_schema(dim: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]))
}

fn vector_batch(n: i32, dim: usize) -> anyhow::Result<RecordBatch> {
    let ids: Vec<i32> = (0..n).collect();
    let vectors: Vec<Option<Vec<Option<f32>>>> =
        (0..n).map(|i| Some(vec![Some(i as f32); dim])).collect();
    Ok(RecordBatch::try_new(
        vector_schema(dim),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vectors, dim as i32),
            ),
        ],
    )?)
}

fn hnsw_tq8() -> IndexAlgorithm {
    IndexAlgorithm::HnswTq8 {
        metric: "l2".to_string(),
        complexity: 16,
        quality: 200,
    }
}

#[tokio::test]
async fn vector_search_returns_bounded_results() -> anyhow::Result<()> {
    let dim = 8usize;
    let dir = tempfile::tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());
    let table = Table::new_async(uri).await?;

    table.add_index("embedding".to_string(), hnsw_tq8()).await?;

    table.set_autocommit(false);
    table.write_async(vec![vector_batch(64, dim)?]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let params = VectorSearchParams::new("embedding", VectorValue::Float32(vec![0.5f32; dim]), 10);
    let res = table.execute_vector_search_as_scored(params).await?;

    assert!(res.len() <= 10, "k=10 must bound the result set");
    assert!(!res.is_empty(), "expected at least one scored result");
    Ok(())
}
