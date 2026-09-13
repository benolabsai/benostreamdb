// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Verify that multi-chunk HNSW search returns correct results when the
//! vector index is split into multiple chunks (chunked build for large
//! datasets). Uses a small `HYPERSTREAM_HNSW_CHUNK_SIZE` so a modest number
//! of vectors produces multiple chunks.

use std::sync::Arc;

use arrow::array::{FixedSizeListArray, Int32Array};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use hyperstreamdb::core::index::VectorValue;
use hyperstreamdb::core::table::Table;
use tempfile::tempdir;

/// Serialize tests that mutate the global HYPERSTREAM_HNSW_CHUNK_SIZE env var.
static CHUNK_SIZE_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

#[tokio::test]
async fn test_multi_chunk_hnsw_search() -> anyhow::Result<()> {
    let _guard = CHUNK_SIZE_LOCK.lock().await;

    // Use a tiny chunk size so 25 vectors -> 3 chunks (10, 10, 5).
    std::env::set_var("HYPERSTREAM_HNSW_CHUNK_SIZE", "10");

    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let table = Table::new_async(uri.clone()).await?;
    table.set_autocommit(false);
    table
        .add_index(
            "embedding".to_string(),
            hyperstreamdb::core::manifest::IndexAlgorithm::Hnsw {
                metric: "l2".to_string(),
                complexity: 16,
                quality: 200,
                build_device: None,
                search_device: None,
            },
        )
        .await?;

    let dim = 4usize;
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]));

    // 25 vectors. Vector at id=24 (last, in chunk 3) is [9.0, 9.0, 9.0, 9.0].
    // The query [9.0, 9.0, 9.0, 9.0] should match id=24 (in the LAST chunk).
    let n = 25i32;
    let ids: Vec<i32> = (0..n).collect();
    let vectors: Vec<Option<Vec<Option<f32>>>> = (0..n)
        .map(|i| {
            let v = (i as f32) * 0.1;
            Some(vec![Some(v), Some(v), Some(v), Some(v)])
        })
        .collect();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(vectors, dim as i32),
            ),
        ],
    )?;

    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    // Clear caches so the search loads the (chunked) index from disk.
    hyperstreamdb::core::cache::HNSW_IVF_CACHE.invalidate_all();

    // Query closest to the LAST vector (id=24, in chunk 3).
    let hits = table
        .query()
        .vector_search(
            "embedding",
            VectorValue::Float32(vec![9.0, 9.0, 9.0, 9.0]),
            1,
        )
        .to_batches()
        .await?;

    assert!(!hits.is_empty(), "expected at least one hit");
    let id_col = hits[0].column_by_name("id").expect("id column present");
    let id_arr = id_col
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("id is Int32");
    let top_id = id_arr.value(0);

    // The nearest neighbor should be id=24 (the last vector, in the last chunk).
    // Before the multi-chunk fix, only chunk 1 was searched, so this would
    // return id=9 (the closest in chunk 1) instead of id=24.
    assert_eq!(
        top_id, 24,
        "multi-chunk search should find the global nearest neighbor (id=24 in the last chunk), got id={top_id}"
    );

    // Clean up env var.
    std::env::remove_var("HYPERSTREAM_HNSW_CHUNK_SIZE");

    Ok(())
}
