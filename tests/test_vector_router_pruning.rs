// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Integration tests for TableVectorRouter segment pruning and targeted cluster search.

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

fn cluster_batch(
    schema: Arc<Schema>,
    start_id: i32,
    count: i32,
    center_val: f32,
    dim: usize,
) -> anyhow::Result<RecordBatch> {
    let ids: Vec<i32> = (start_id..start_id + count).collect();
    let titles: Vec<String> = (start_id..start_id + count)
        .map(|i| format!("doc-{i}"))
        .collect();
    let vectors: Vec<Option<Vec<Option<f32>>>> = (0..count)
        .map(|i| Some(vec![Some(center_val + (i as f32) * 0.001); dim]))
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

#[tokio::test]
async fn test_multi_segment_vector_router_pruning() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());
    let dim = 8usize;

    let table = Table::new_async(uri).await?;
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

    let s = schema(dim);

    // Segment 0: IDs 0..50 centered around 10.0
    table
        .write_async(vec![cluster_batch(s.clone(), 0, 50, 10.0, dim)?])
        .await?;
    table.commit_async().await?;

    // Segment 1: IDs 50..100 centered around 50.0
    table
        .write_async(vec![cluster_batch(s.clone(), 50, 50, 50.0, dim)?])
        .await?;
    table.commit_async().await?;

    // Segment 2: IDs 100..150 centered around 100.0
    table
        .write_async(vec![cluster_batch(s.clone(), 100, 50, 100.0, dim)?])
        .await?;
    table.commit_async().await?;

    table.wait_for_background_tasks_async().await?;

    // Preload indexes and initialize table-level vector router
    let preload_stats = table
        .preload_indexes_async(PreloadOptions {
            max_memory_bytes: 100 * 1024 * 1024,
            spill_to_disk: true,
            ..Default::default()
        })
        .await?;

    assert!(preload_stats.indexes_warmed >= 3, "All 3 segment vector indexes should be warmed");

    // Verify router exists and has registered all segment centroids
    let router_opt = table.ensure_vector_router("embedding").await?;
    assert!(router_opt.is_some(), "Vector router must be initialized");
    let router = router_opt.unwrap();
    assert!(router.len() >= 3, "Router must contain centroids from all segments");

    // Query 1: Vector near Segment 0 center (10.0)
    let q1 = vec![10.01; dim];
    let vp1 = VectorSearchParams::new("embedding", VectorValue::Float32(q1), 5)
        .with_metric(benostreamdb::core::index::VectorMetric::L2)
        .with_probes(2);

    let plan1 = router.route(&vec![10.01; dim], 2);
    eprintln!("Plan1 candidate segments: {:?}", plan1.candidate_segments);
    eprintln!("Plan1 targeted clusters: {:?}", plan1.segment_clusters);

    let manifest = table.manifest().await?;
    for e in &manifest.entries {
        eprintln!("ManifestEntry file_path: {}", e.file_path);
        for idx in &e.index_files {
            eprintln!("  idx file_path: {}, blob_type: {:?}", idx.file_path, idx.blob_type);
        }
    }

    let results1 = table
        .read_expr_with_config_async(
            None,
            Some(vec![vp1]),
            Some(&["id"]),
            table.query_config().clone(),
            None,
        )
        .await?;

    assert!(!results1.is_empty());
    let ids1: Vec<i32> = results1
        .iter()
        .flat_map(|b| {
            b.column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();

    assert!(ids1.iter().all(|&id| id < 50), "All top-k hits for query 1 must come from Segment 0 (id < 50), got: {:?}", ids1);

    // Query 2: Vector near Segment 2 center (100.0)
    let q2 = vec![100.01; dim];
    let vp2 = VectorSearchParams::new("embedding", VectorValue::Float32(q2), 5)
        .with_metric(benostreamdb::core::index::VectorMetric::L2)
        .with_probes(2);

    let results2 = table
        .read_expr_with_config_async(
            None,
            Some(vec![vp2]),
            Some(&["id"]),
            table.query_config().clone(),
            None,
        )
        .await?;

    assert!(!results2.is_empty());
    let ids2: Vec<i32> = results2
        .iter()
        .flat_map(|b| {
            b.column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();

    assert!(ids2.iter().all(|&id| id >= 100), "All top-k hits for query 2 must come from Segment 2 (id >= 100), got: {:?}", ids2);

    Ok(())
}
