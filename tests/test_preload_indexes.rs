// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Tests for `Table::preload_indexes_async` — the Dgraph-style cache warm.
//!
//! Covers:
//! 1. A vector index is discovered and warmed within budget.
//! 2. A tight memory budget forces indexes onto the disk (mmap) tier.
//! 3. A preloaded index still answers a query correctly (no corruption).
//! 4. Warm calls are idempotent (second pass is cheap cache hits).

use std::sync::Arc;

use arrow::array::{FixedSizeListArray, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::index::VectorValue;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::core::table::{PreloadOptions, Table, VectorSearchParams};
use tempfile::tempdir;

fn schema(dim: usize) -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("title", DataType::Utf8, false),
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

fn batch(schema: Arc<Schema>, n: i32, dim: usize) -> anyhow::Result<RecordBatch> {
    let ids: Vec<i32> = (0..n).collect();
    let titles: Vec<String> = (0..n).map(|i| format!("doc-{i}")).collect();
    let vectors: Vec<Option<Vec<Option<f32>>>> = (0..n)
        .map(|i| Some(vec![Some((i as f32) * 0.01); dim]))
        .collect();
    Ok(RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(titles)),
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vectors, dim as i32),
            ),
        ],
    )?)
}

/// Build a table with a vector index + BM25 index over 64 rows.
async fn build_indexed_table() -> anyhow::Result<(tempfile::TempDir, Table)> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());
    let dim = 8usize;

    let mut table = Table::new_async(uri).await?;
    table.set_autocommit(false);
    table
        .add_index(
            "embedding".to_string(),
            IndexAlgorithm::HnswTq8 {
                metric: "l2".to_string(),
                complexity: 16,
                quality: 8,
            },
        )
        .await?;
    table
        .add_index_columns_async(vec!["title".to_string()], None)
        .await?;

    table
        .write_async(vec![batch(schema(dim), 64, dim)?])
        .await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;
    Ok((dir, table))
}

#[tokio::test]
async fn preload_warms_vector_index_within_budget() -> anyhow::Result<()> {
    let (_dir, table) = build_indexed_table().await?;

    let stats = table
        .preload_indexes_async(PreloadOptions {
            // Generous budget: everything should land in memory.
            max_memory_bytes: 1 << 30,
            ..PreloadOptions::default()
        })
        .await?;

    assert!(
        stats.indexes_seen > 0,
        "preload should discover index files, saw {}",
        stats.indexes_seen
    );
    assert!(
        stats.indexes_warmed > 0,
        "preload should warm at least one index, warmed {}",
        stats.indexes_warmed
    );
    assert!(
        stats.bytes_in_memory > 0,
        "a generous budget should warm indexes in memory"
    );
    Ok(())
}

#[tokio::test]
async fn preload_spills_to_disk_when_budget_is_zero() -> anyhow::Result<()> {
    let (_dir, table) = build_indexed_table().await?;

    let stats = table
        .preload_indexes_async(PreloadOptions {
            // Zero in-memory budget: everything must go to the disk tier.
            max_memory_bytes: 0,
            spill_to_disk: true,
            ..PreloadOptions::default()
        })
        .await?;

    assert!(
        stats.indexes_seen > 0,
        "preload should discover index files even with a zero budget"
    );
    assert_eq!(
        stats.bytes_in_memory, 0,
        "a zero budget must not warm anything into memory"
    );
    assert!(
        stats.bytes_on_disk > 0 || stats.indexes_warmed > 0,
        "with spill_to_disk, indexes should be warmed onto the disk tier \
         (on_disk={}, warmed={})",
        stats.bytes_on_disk,
        stats.indexes_warmed
    );
    Ok(())
}

#[tokio::test]
async fn preload_preserves_query_correctness() -> anyhow::Result<()> {
    let (_dir, table) = build_indexed_table().await?;

    let _ = table
        .preload_indexes_async(PreloadOptions::default())
        .await?;

    // `explain()` must still report the index access path (not a brute-force
    // scan) after preload — i.e. preload did not perturb index metadata.
    let plan = table
        .explain(
            None,
            Some(vec![VectorSearchParams::new(
                "embedding",
                VectorValue::Float32(vec![0.0; 8]),
                5,
            )]),
        )
        .await;
    assert!(
        !plan.contains("Brute Force Scan (No Index)"),
        "after preload, explain() must still find the index, got:\n{plan}"
    );

    // And a full read must still work.
    let batches = table.read_async(None, None, None).await?;
    let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(rows, 64, "all rows should be readable after preload");
    Ok(())
}

#[tokio::test]
async fn preload_is_idempotent() -> anyhow::Result<()> {
    let (_dir, table) = build_indexed_table().await?;

    let first = table
        .preload_indexes_async(PreloadOptions::default())
        .await?;
    let second = table
        .preload_indexes_async(PreloadOptions::default())
        .await?;

    // Both passes must see the same indexes; the second is served from
    // cache and must not error.
    assert_eq!(
        first.indexes_seen, second.indexes_seen,
        "both preload passes should enumerate the same index set"
    );
    assert!(
        second.indexes_warmed <= first.indexes_seen,
        "second pass must not warm more than were seen"
    );
    Ok(())
}

#[tokio::test]
async fn preload_respects_type_filters() -> anyhow::Result<()> {
    let (_dir, table) = build_indexed_table().await?;

    // Exclude everything: nothing should be warmed.
    let stats = table
        .preload_indexes_async(PreloadOptions {
            include_vector: false,
            include_inverted: false,
            include_graph: false,
            ..PreloadOptions::default()
        })
        .await?;

    assert!(
        stats.indexes_seen > 0,
        "index files should still be discovered"
    );
    assert_eq!(
        stats.indexes_warmed, 0,
        "with all types filtered out, nothing should be warmed"
    );
    Ok(())
}
