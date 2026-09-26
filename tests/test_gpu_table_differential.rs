// Copyright (c) 2026 Richard Albright. All rights reserved.

//! GPU table-level differential: a vector search executed with a GPU compute
//! context must return the same top-k as the CPU path.
//!
//! The existing GPU tests ([`test_hardware_parity.rs`], `test_*_distance_kernel.rs`)
//! cover the distance *kernels* in isolation. This test covers the end-to-end
//! table read path (index load → cluster search → distance → top-k) with a GPU
//! context, which is where a context-propagation bug would show up.
//!
//! Skipped unless a working CUDA device is present (see `gpu_test_helpers`).

mod gpu_test_helpers;

use std::sync::Arc;

use arrow::array::{FixedSizeListArray, Int32Array};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::index::gpu::{set_thread_gpu_context, ComputeBackend, ComputeContext};
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
    // Well-separated vectors so the top-k is unambiguous.
    let vectors: Vec<Option<Vec<Option<f32>>>> = (0..n)
        .map(|i| Some(vec![Some((i as f32) * 10.0); dim]))
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

async fn vector_topk(table: &Table, q: Vec<f32>, k: usize) -> anyhow::Result<Vec<i32>> {
    let params = VectorSearchParams::new("embedding", VectorValue::Float32(q), k);
    let out = table.read_async(None, Some(vec![params]), None).await?;
    let mut ids = Vec::new();
    for b in &out {
        if let Some(col) = b.column_by_name("id") {
            if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
                ids.extend(arr.iter().flatten());
            }
        }
    }
    Ok(ids)
}

#[tokio::test]
async fn gpu_vector_search_matches_cpu() -> anyhow::Result<()> {
    if gpu_test_helpers::should_skip_gpu_tests() {
        return Ok(());
    }

    let dim = 16usize;
    let n = 200i32;
    let batch = embedding_batch(embedding_schema(dim), n, dim)?;

    // Build the index on the CPU so the comparison isolates the search path.
    set_thread_gpu_context(Some(ComputeContext::from_backend(ComputeBackend::Cpu)?));
    let dir = tempdir()?;
    let table = Table::new_async(format!("file://{}", dir.path().display())).await?;
    table
        .add_index("embedding".to_string(), IndexAlgorithm::hnsw_tq8())
        .await?;
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let cuda = ComputeContext::from_backend(ComputeBackend::Cuda)?;
    let cpu = ComputeContext::from_backend(ComputeBackend::Cpu)?;

    for i in 0..5 {
        let q = vec![(i as f32) * 10.0; dim];

        set_thread_gpu_context(Some(cpu.clone()));
        let cpu_ids = vector_topk(&table, q.clone(), 5).await?;

        set_thread_gpu_context(Some(cuda.clone()));
        let gpu_ids = vector_topk(&table, q, 5).await?;

        assert_eq!(
            cpu_ids, gpu_ids,
            "GPU and CPU vector search top-k diverged for query {i}"
        );
    }

    set_thread_gpu_context(None);
    Ok(())
}
