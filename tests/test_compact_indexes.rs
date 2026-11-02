// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Integration tests for Table::compact_indexes manual index compaction in Community Edition.
//!
//! Verifies:
//! 1. Multi-segment index sidecars are consolidated into Voronoi Puffin shards.
//! 2. Underlying Parquet data files remain 100% untouched and immutable.
//! 3. Vector queries continue to execute with correct results after compaction.
//! 4. Compaction statistics adhere to target shard size constraints.

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
async fn test_manual_index_compaction_sidecars_only() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());
    let dim = 8usize;

    let table = Table::new_async(uri.clone()).await?;
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

    // Write 3 distinct segments
    table
        .write_async(vec![cluster_batch(s.clone(), 0, 50, 10.0, dim)?])
        .await?;
    table.commit_async().await?;

    table
        .write_async(vec![cluster_batch(s.clone(), 50, 50, 50.0, dim)?])
        .await?;
    table.commit_async().await?;

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

    // Record initial Parquet data file paths and sizes
    let mut data_files_before = table.list_data_files_async().await?;
    data_files_before.sort_by(|a, b| a.file_path.cmp(&b.file_path));
    assert_eq!(data_files_before.len(), 3, "Expected 3 initial segments");

    // Execute Manual Index Compaction (advisory Puffin sidecars only)
    let stats = table
        .compact_indexes_async("embedding", Some(50_000_000))
        .await?;

    assert_eq!(stats.column, "embedding");
    assert_eq!(stats.input_segments, 3, "Should have processed 3 input segments");
    assert!(stats.output_shards >= 1, "Should have generated consolidated shards");
    assert_eq!(stats.total_records, 150, "Should reflect all 150 records");
    assert_eq!(stats.target_shard_bytes, 50_000_000);

    // Verify invariant: Underlying Parquet data files MUST be 100% untouched
    let mut data_files_after = table.list_data_files_async().await?;
    data_files_after.sort_by(|a, b| a.file_path.cmp(&b.file_path));
    assert_eq!(
        data_files_before.len(),
        data_files_after.len(),
        "Number of Parquet data files must remain identical"
    );
    for (before, after) in data_files_before.iter().zip(data_files_after.iter()) {
        assert_eq!(
            before.file_path, after.file_path,
            "Parquet data file path must not change"
        );
        assert_eq!(
            before.file_size_bytes, after.file_size_bytes,
            "Parquet data file size must remain identical"
        );
    }

    // Verify vector search continues to operate correctly after compaction
    let q = vec![10.01; dim];
    let vp = VectorSearchParams::new("embedding", VectorValue::Float32(q), 5)
        .with_metric(benostreamdb::core::index::VectorMetric::L2)
        .with_probes(2);

    let results = table
        .read_expr_with_config_async(
            None,
            Some(vec![vp]),
            Some(&["id"]),
            table.query_config().clone(),
            None,
        )
        .await?;

    assert!(!results.is_empty(), "Query must return results");
    let ids: Vec<i32> = results
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

    assert!(
        ids.iter().all(|&id| id < 50),
        "Nearest neighbors for cluster 10.0 must be id < 50, got {:?}",
        ids
    );

    Ok(())
}

#[test]
fn test_sync_compact_indexes_api() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());
    let dim = 8usize;

    let table = Table::new(uri)?;
    table.set_autocommit(false);
    table.runtime().block_on(table.add_index(
        "embedding".to_string(),
        IndexAlgorithm::HnswTq8 {
            metric: "l2".to_string(),
            complexity: 16,
            quality: 8,
        },
    ))?;

    let s = schema(dim);
    table.write(vec![cluster_batch(s.clone(), 0, 50, 10.0, dim)?])?;
    table.commit()?;

    table.write(vec![cluster_batch(s.clone(), 50, 50, 50.0, dim)?])?;
    table.commit()?;
    table.wait_for_background_tasks()?;

    let sync_stats = table.compact_indexes("embedding", None)?;
    assert_eq!(sync_stats.column, "embedding");
    assert_eq!(sync_stats.input_segments, 2);
    assert_eq!(sync_stats.total_records, 100);
    assert_eq!(sync_stats.target_shard_bytes, 536_870_912); // 512 MiB Iceberg standard

    Ok(())
}

