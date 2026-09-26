// Copyright (c) 2026 Richard Albright. All rights reserved.

//! WS1 differential oracle: **the index must never change the answer.**
//!
//! The review's highest-value testing idea: because the overlay index is
//! advisory, the full scan is a correctness oracle. For each index type we run
//! the same query against two tables with identical data — one indexed, one
//! not — and compare.
//!
//! * **Exact indexes** (scalar bitmap, inverted/BM25): the indexed result must
//!   be *identical* to the full-scan result.
//! * **ANN indexes** (HNSW / IVF / TurboQuant / PQ): the result is approximate
//!   by design, so we assert *recall* — the exact nearest neighbour must appear
//!   in the indexed top-k.
//!
//! See `plans/production_readiness_plan.md` §3 (WS1).

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{FixedSizeListArray, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::index::VectorValue;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::core::table::Table;
use tempfile::tempdir;

// ── Scalar / inverted ───────────────────────────────────────────────────────

fn scalar_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]))
}

fn scalar_batch(n: i32) -> anyhow::Result<RecordBatch> {
    let ids: Vec<i32> = (0..n).collect();
    let cats: Vec<String> = (0..n).map(|i| format!("cat_{}", i % 4)).collect();
    Ok(RecordBatch::try_new(
        scalar_schema(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(cats)),
        ],
    )?)
}

async fn build_scalar_table(uri: String, algo: Option<IndexAlgorithm>) -> anyhow::Result<Table> {
    let table = Table::new_async(uri).await?;
    table.set_autocommit(false);
    if let Some(a) = algo {
        table.add_index("category".to_string(), a).await?;
    }
    table.write_async(vec![scalar_batch(64)?]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;
    Ok(table)
}

async fn assert_scalar_matches_full_scan(algo: Option<IndexAlgorithm>, label: &str) -> anyhow::Result<()> {
    let d1 = tempdir()?;
    let d2 = tempdir()?;
    let plain =
        build_scalar_table(format!("file://{}", d1.path().to_str().unwrap()), None).await?;
    let indexed =
        build_scalar_table(format!("file://{}", d2.path().to_str().unwrap()), algo).await?;

    for filter in ["category = 'cat_1'", "category = 'cat_3'", "id >= 30"] {
        let a = ids_of(&plain.filter(filter).to_batches().await?);
        let b = ids_of(&indexed.filter(filter).to_batches().await?);
        assert_eq!(
            a, b,
            "{label}: indexed result diverged from full scan for `{filter}`: \
             scan={a:?} indexed={b:?}"
        );
        assert!(!a.is_empty(), "{label}: filter `{filter}` returned no rows");
    }
    Ok(())
}

fn ids_of(batches: &[RecordBatch]) -> HashSet<i32> {
    let mut out = HashSet::new();
    for b in batches {
        if let Some(col) = b.column_by_name("id") {
            if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
                for i in 0..arr.len() {
                    out.insert(arr.value(i));
                }
            }
        }
    }
    out
}

/// A BM25 (tokenized) index must not change the answer for an exact filter.
#[tokio::test]
async fn differential_bm25_index_matches_full_scan() -> anyhow::Result<()> {
    assert_scalar_matches_full_scan(
        Some(IndexAlgorithm::Bm25 {
            k1: 1.2,
            b: 0.75,
            tokenizer: "default".to_string(),
        }),
        "bm25",
    )
    .await
}

/// A scalar bitmap index must equal the full-scan result exactly.
#[tokio::test]
async fn differential_bitmap_index_matches_full_scan() -> anyhow::Result<()> {
    assert_scalar_matches_full_scan(Some(IndexAlgorithm::Bitmap), "bitmap").await
}

// ── Vector (ANN) ────────────────────────────────────────────────────────────

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
    // Distinct vectors: vector i is `[i, i, ...]`, so the nearest neighbour of
    // `[q, q, ...]` is unambiguously id = round(q).
    let vectors: Vec<Option<Vec<Option<f32>>>> = (0..n)
        .map(|i| Some(vec![Some(i as f32); dim]))
        .collect();
    Ok(RecordBatch::try_new(
        vector_schema(dim),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                vectors,
                dim as i32,
            )),
        ],
    )?)
}

async fn build_vector_table(
    uri: String,
    algo: Option<IndexAlgorithm>,
    n: i32,
    dim: usize,
) -> anyhow::Result<Table> {
    let table = Table::new_async(uri).await?;
    table.set_autocommit(false);
    if let Some(a) = algo {
        table.add_index("embedding".to_string(), a).await?;
    }
    table.write_async(vec![vector_batch(n, dim)?]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;
    Ok(table)
}

fn top_ids(batches: &[RecordBatch], k: usize) -> Vec<i32> {
    let mut out = Vec::new();
    for b in batches {
        if let Some(col) = b.column_by_name("id") {
            if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
                for i in 0..arr.len() {
                    if out.len() < k {
                        out.push(arr.value(i));
                    }
                }
            }
        }
    }
    out
}

/// The exact nearest neighbour (from the full scan) must appear in the indexed
/// top-k. This is the ANN form of "the index must not change the answer".
async fn assert_ann_recall(algo: IndexAlgorithm, label: &str) -> anyhow::Result<()> {
    let dim = 8usize;
    let n = 64i32;
    let k = 5usize;
    let query = vec![10.0f32; dim]; // exact nearest neighbour is id = 10

    let d1 = tempdir()?;
    let d2 = tempdir()?;
    let plain =
        build_vector_table(format!("file://{}", d1.path().to_str().unwrap()), None, n, dim).await?;
    let indexed = build_vector_table(
        format!("file://{}", d2.path().to_str().unwrap()),
        Some(algo),
        n,
        dim,
    )
    .await?;

    let exact = top_ids(
        &plain
            .query()
            .vector_search("embedding", VectorValue::Float32(query.clone()), k)
            .to_batches()
            .await?,
        k,
    );
    let approx = top_ids(
        &indexed
            .query()
            .vector_search("embedding", VectorValue::Float32(query), k)
            .to_batches()
            .await?,
        k,
    );

    assert!(!exact.is_empty(), "{label}: full scan returned no rows");
    assert!(
        approx.contains(&exact[0]),
        "{label}: indexed top-{k} {approx:?} missed the exact nearest neighbour {} (full scan {exact:?})",
        exact[0]
    );

    Ok(())
}

#[tokio::test]
async fn differential_ann_tq8_recall() -> anyhow::Result<()> {
    assert_ann_recall(
        IndexAlgorithm::HnswTq8 {
            metric: "l2".to_string(),
            complexity: 16,
            quality: 8,
        },
        "tq8",
    )
    .await
}

#[tokio::test]
async fn differential_ann_tq4_recall() -> anyhow::Result<()> {
    assert_ann_recall(
        IndexAlgorithm::HnswTq4 {
            metric: "l2".to_string(),
            complexity: 16,
            quality: 8,
        },
        "tq4",
    )
    .await
}

#[tokio::test]
async fn differential_ann_pq_recall() -> anyhow::Result<()> {
    assert_ann_recall(
        IndexAlgorithm::HnswPq {
            metric: "l2".to_string(),
            complexity: 16,
            quality: 8,
            compression: 2,
        },
        "pq",
    )
    .await
}

#[tokio::test]
async fn differential_ann_ivf_recall() -> anyhow::Result<()> {
    assert_ann_recall(
        IndexAlgorithm::Hnsw {
            metric: "l2".to_string(),
            complexity: 16,
            quality: 200,
            build_device: None,
            search_device: None,
        },
        "ivf",
    )
    .await
}
