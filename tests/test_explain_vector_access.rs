// Copyright (c) 2026 Richard Albright. All rights reserved.

//! WS1 regression: `Table::explain()` must report the vector access path from
//! the manifest's `index_files` — the same source of truth the search path uses
//! (`reader/scan.rs::vector_search`) — not from a filesystem glob.
//!
//! The old implementation globbed `{seg}.{col}.cluster_*.hnsw.graph`, which
//! misses the quantization infix (`.tq8.` / `.tq4.` / `.pq.`) and therefore
//! always reported "Brute Force Scan (No Index)" for a quantized index. See
//! `plans/production_readiness_plan.md` §2.

use std::sync::Arc;

use arrow::array::{FixedSizeListArray, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::index::VectorValue;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::core::table::{Table, VectorSearchParams};
use tempfile::tempdir;

fn embedding_schema(dim: usize) -> Arc<Schema> {
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

fn embedding_batch(schema: Arc<Schema>, n: i32, dim: usize) -> anyhow::Result<RecordBatch> {
    let ids: Vec<i32> = (0..n).collect();
    let vectors: Vec<Option<Vec<Option<f32>>>> = (0..n)
        .map(|i| {
            let v = (i as f32) * 0.1;
            Some(vec![Some(v); dim])
        })
        .collect();
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vectors, dim as i32),
            ),
        ],
    )?)
}

/// Build a table with `algo` on `embedding`, then assert `explain()` reports
/// `expected` as the access path and does not misreport a full scan.
async fn assert_access_path(algo: IndexAlgorithm, expected: &str) -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());

    let table = Table::new_async(uri).await?;
    table.set_autocommit(false);
    table.add_index("embedding".to_string(), algo).await?;

    let dim = 4usize;
    let schema = embedding_schema(dim);
    table
        .write_async(vec![embedding_batch(schema, 32, dim)?])
        .await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let plan = table
        .explain(
            None,
            Some(vec![VectorSearchParams::new(
                "embedding",
                VectorValue::Float32(vec![0.5; dim]),
                5,
            )]),
        )
        .await;

    assert!(
        plan.contains(expected),
        "explain() should report `{expected}`, got:\n{plan}"
    );
    assert!(
        !plan.contains("Brute Force Scan (No Index)"),
        "explain() must not misreport a full scan when an index exists, got:\n{plan}"
    );

    Ok(())
}

#[tokio::test]
async fn explain_reports_tq8_index_access_path() -> anyhow::Result<()> {
    assert_access_path(
        IndexAlgorithm::HnswTq8 {
            metric: "l2".to_string(),
            complexity: 16,
            quality: 8,
        },
        "HNSW-TQ8 Cluster Index",
    )
    .await
}

#[tokio::test]
async fn explain_reports_tq4_index_access_path() -> anyhow::Result<()> {
    assert_access_path(
        IndexAlgorithm::HnswTq4 {
            metric: "l2".to_string(),
            complexity: 16,
            quality: 8,
        },
        "HNSW-TQ4 Cluster Index",
    )
    .await
}

#[tokio::test]
async fn explain_reports_pq_index_access_path() -> anyhow::Result<()> {
    assert_access_path(
        IndexAlgorithm::HnswPq {
            metric: "l2".to_string(),
            complexity: 16,
            quality: 8,
            // dim (4) must be divisible by `compression`; 4 / 2 = 2 subspaces.
            compression: 2,
        },
        "HNSW-PQ Cluster Index",
    )
    .await
}

#[tokio::test]
async fn explain_reports_ivf_index_access_path() -> anyhow::Result<()> {
    assert_access_path(
        IndexAlgorithm::Hnsw {
            metric: "l2".to_string(),
            complexity: 16,
            quality: 200,
            build_device: None,
            search_device: None,
        },
        "HNSW-IVF Cluster Index",
    )
    .await
}

/// A table with an inverted/BM25 index must report an index access path, not a
/// full scan. This is the scalar counterpart of the vector regression: the old
/// `explain()` globbed `{seg}.{col}.inv.parquet` / `{seg}.{col}.idx` on disk
/// instead of reading the manifest.
#[tokio::test]
async fn explain_reports_inverted_index_access_path() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());

    let table = Table::new_async(uri).await?;
    table.set_autocommit(false);
    table
        .add_index(
            "category".to_string(),
            IndexAlgorithm::Bm25 {
                k1: 1.2,
                b: 0.75,
                tokenizer: "default".to_string(),
            },
        )
        .await?;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    let ids: Vec<i32> = (0..32).collect();
    let cats: Vec<String> = (0..32).map(|i| format!("cat_{}", i % 4)).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(cats)),
        ],
    )?;
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let plan = table.explain(Some("category = 'cat_1'"), None).await;

    assert!(
        plan.contains("Inverted Index") || plan.contains("BM25"),
        "explain() should report an inverted/BM25 access path, got:\n{plan}"
    );
    assert!(
        !plan.contains("access: Full Scan"),
        "explain() must not report a full scan when an inverted index exists, got:\n{plan}"
    );

    Ok(())
}

/// A table with no vector index must still report a brute-force scan.
#[tokio::test]
async fn explain_reports_brute_force_without_index() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());

    let table = Table::new_async(uri).await?;
    table.set_autocommit(false);

    let dim = 4usize;
    let schema = embedding_schema(dim);
    table
        .write_async(vec![embedding_batch(schema, 8, dim)?])
        .await?;
    table.commit_async().await?;

    let plan = table
        .explain(
            None,
            Some(vec![VectorSearchParams::new(
                "embedding",
                VectorValue::Float32(vec![0.5; dim]),
                5,
            )]),
        )
        .await;

    assert!(
        plan.contains("Brute Force Scan (No Index)"),
        "explain() should report a brute-force scan when no vector index exists, got:\n{plan}"
    );

    Ok(())
}

/// The performance half of the §2 regression: an indexed vector query must be
/// materially faster than the full scan. Builds two tables over identical data,
/// one with an HNSW-TQ8 index and one without, and compares the search latency.
///
/// This is a benchmark-style assertion; it uses a large enough row count that
/// the index's sub-linear search dominates its fixed overhead.
#[tokio::test]
async fn indexed_vector_query_is_faster_than_full_scan() -> anyhow::Result<()> {
    let dim = 32usize;
    let n = 5000i32;
    let batch = embedding_batch(embedding_schema(dim), n, dim)?;

    let dir_a = tempdir()?;
    let table_a = Table::new_async(format!("file://{}", dir_a.path().display())).await?;
    table_a
        .add_index("embedding".to_string(), IndexAlgorithm::hnsw_tq8())
        .await?;
    table_a.write_async(vec![batch.clone()]).await?;
    table_a.commit_async().await?;
    table_a.wait_for_background_tasks_async().await?;

    let dir_b = tempdir()?;
    let table_b = Table::new_async(format!("file://{}", dir_b.path().display())).await?;
    table_b.write_async(vec![batch]).await?;
    table_b.commit_async().await?;
    table_b.wait_for_background_tasks_async().await?;

    let q = vec![0.5f32; dim];
    let params = || {
        vec![VectorSearchParams::new(
            "embedding",
            VectorValue::Float32(q.clone()),
            5,
        )]
    };

    // Warm up (index load, caches) so the comparison is steady-state.
    let _ = table_a.read_async(None, Some(params()), None).await?;
    let _ = table_b.read_async(None, Some(params()), None).await?;

    const M: usize = 20;
    let t = std::time::Instant::now();
    for _ in 0..M {
        let _ = table_a.read_async(None, Some(params()), None).await?;
    }
    let indexed = t.elapsed();

    let t = std::time::Instant::now();
    for _ in 0..M {
        let _ = table_b.read_async(None, Some(params()), None).await?;
    }
    let scan = t.elapsed();

    assert!(
        indexed < scan,
        "indexed vector query ({indexed:?}) should be faster than the full scan ({scan:?})"
    );

    Ok(())
}
