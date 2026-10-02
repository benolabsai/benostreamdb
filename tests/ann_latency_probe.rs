// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
//
//! Diagnostic probe for the ANN-Benchmarks per-query latency. Times each stage
//! of the read path independently so the fixed overhead is separated from the
//! actual search/I-O cost. Not a correctness test; `#[ignore]`d by default.

use std::sync::Arc;
use std::time::Instant;

use arrow::array::{FixedSizeListArray, Int32Array};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;

use benostreamdb::core::index::{VectorMetric, VectorValue};
use benostreamdb::core::manifest::{IndexAlgorithm, ManifestManager};
use benostreamdb::core::planner::VectorSearchParams;
use benostreamdb::core::table::Table;

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
    let vectors: Vec<Option<Vec<Option<f32>>>> = (0..n)
        .map(|i| Some(vec![Some((i as f32) / n as f32); dim]))
        .collect();
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

fn params(query: &VectorValue, k: usize) -> VectorSearchParams {
    VectorSearchParams {
        column: "embedding".to_string(),
        query: query.clone(),
        k,
        metric: VectorMetric::L2,
        ef_search: Some(200),
        probes: None,
        stats_only: false,
        use_mmap: true,
        radius: None,
    }
}

#[tokio::test]
#[ignore = "diagnostic probe; run explicitly with --ignored --nocapture"]
async fn ann_latency_probe() -> anyhow::Result<()> {
    let dim = 128usize;
    let n = 20_000i32;
    let k = 10usize;

    let dir = tempfile::tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());
    let table = Table::new_async(uri.clone()).await?;
    table.set_autocommit(false);
    table
        .add_index(
            "embedding".to_string(),
            IndexAlgorithm::HnswTq8 {
                metric: "l2".to_string(),
                complexity: 16,
                quality: 200,
            },
        )
        .await?;
    table.write_async(vec![vector_batch(n, dim)?]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let query = VectorValue::Float32(vec![0.5f32; dim]);

    // Warm all caches once.
    let _ = table.read_async(None, None, None).await?;

    for (label, iters) in [
        ("manifest_full", 6usize),
        ("scan_all", 6),
        ("vector_read", 6),
        ("pure_index", 6),
    ] {
        for i in 0..iters {
            let t = Instant::now();
            match label {
                "manifest_full" => {
                    let mm = ManifestManager::new(table.store.clone(), "", &table.uri);
                    let (m, e, _v) = mm.load_latest_full().await?;
                    println!(
                        "{label} #{i}: {:?} (entries={}, inline={})",
                        t.elapsed(),
                        e.len(),
                        m.entries.len()
                    );
                    continue;
                }
                "scan_all" => {
                    let _ = table.read_async(None, None, None).await?;
                }
                "vector_read" => {
                    let _ = table
                        .read_async(None, Some(vec![params(&query, k)]), None)
                        .await?;
                }
                _ => {
                    let _ = table
                        .execute_vector_search_as_scored(params(&query, k))
                        .await?;
                }
            }
            println!("{label} #{i}: {:?}", t.elapsed());
        }
    }
    Ok(())
}
