// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Integration tests for Table::compact_indexes manual index compaction in Community Edition.
//!
//! Verifies:
//! 1. Multi-segment index sidecars are consolidated into Voronoi Puffin shards.
//! 2. Underlying Parquet data files remain 100% untouched and immutable.
//! 3. Vector queries continue to execute with correct results after compaction.
//! 4. Compaction statistics adhere to target shard size constraints.

use std::collections::HashSet;
use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{FixedSizeListArray, Int32Array, StringArray, UInt32Array};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::index::VectorValue;
use benostreamdb::core::manifest::{IndexAlgorithm, ManifestManager};
use benostreamdb::core::puffin::{
    PuffinReader, PUFFIN_BLOB_VECTOR_CLUSTER_NODE_META, PUFFIN_BLOB_VECTOR_HNSW_GRAPH,
};
use benostreamdb::core::storage::create_object_store;
use benostreamdb::core::table::{PreloadOptions, Table, VectorSearchParams};
use bytes::Bytes;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
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

fn mixed_cluster_batch(
    schema: Arc<Schema>,
    start_id: i32,
    count_per_cluster: i32,
    centers: &[f32],
    segment_idx: usize,
    dim: usize,
) -> anyhow::Result<RecordBatch> {
    let mut ids = Vec::new();
    let mut titles = Vec::new();
    let mut vectors = Vec::new();

    let mut current_id = start_id;
    for &center in centers {
        for i in 0..count_per_cluster {
            ids.push(current_id);
            titles.push(format!("doc-{current_id}"));
            let offset_val = ((i as usize) * 3 + segment_idx) as f32 * 0.005;
            vectors.push(Some(vec![Some(center + offset_val); dim]));
            current_id += 1;
        }
    }

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

    assert!(
        preload_stats.indexes_warmed >= 3,
        "All 3 segment vector indexes should be warmed"
    );

    // Record initial Parquet data file paths and sizes
    let mut data_files_before = table.list_data_files_async().await?;
    data_files_before.sort_by(|a, b| a.file_path.cmp(&b.file_path));
    assert_eq!(data_files_before.len(), 3, "Expected 3 initial segments");

    // Execute Manual Index Compaction (advisory Puffin sidecars only)
    let stats = table
        .compact_indexes_async("embedding", Some(50_000_000))
        .await?;

    assert_eq!(stats.column, "embedding");
    assert_eq!(
        stats.input_segments, 3,
        "Should have processed 3 input segments"
    );
    assert!(
        stats.output_shards >= 1,
        "Should have generated consolidated shards"
    );
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

#[tokio::test]
async fn test_contiguous_clusters_in_index_after_compaction() -> anyhow::Result<()> {
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

    // Ingest 3 distinct segments. In each segment, write rows belonging to two well-separated
    // global cluster centers (10.0 and 100.0).
    // Segment 0: 25 rows near 10.0 (IDs 0..24), 25 rows near 100.0 (IDs 25..49)
    table
        .write_async(vec![mixed_cluster_batch(
            s.clone(),
            0,
            25,
            &[10.0, 100.0],
            0,
            dim,
        )?])
        .await?;
    table.commit_async().await?;

    // Segment 1: 25 rows near 10.0 (IDs 50..74), 25 rows near 100.0 (IDs 75..99)
    table
        .write_async(vec![mixed_cluster_batch(
            s.clone(),
            50,
            25,
            &[10.0, 100.0],
            1,
            dim,
        )?])
        .await?;
    table.commit_async().await?;

    // Segment 2: 25 rows near 10.0 (IDs 100..124), 25 rows near 100.0 (IDs 125..149)
    table
        .write_async(vec![mixed_cluster_batch(
            s.clone(),
            100,
            25,
            &[10.0, 100.0],
            2,
            dim,
        )?])
        .await?;
    table.commit_async().await?;

    table.wait_for_background_tasks_async().await?;

    // 1. Verify initial layout: Data rows are physically split across 3 separate Parquet files
    let mut data_files_before = table.list_data_files_async().await?;
    data_files_before.sort_by(|a, b| a.file_path.cmp(&b.file_path));
    assert_eq!(
        data_files_before.len(),
        3,
        "Expected 3 independent data files"
    );

    // 2. Execute table-level index compaction for k=2 global clusters
    table
        .rebuild_cluster_index("embedding".to_string(), 2, 16, 100)
        .await?;

    // 3. Invariant: Parquet data files MUST remain 100% untouched and byte-identical
    let mut data_files_after = table.list_data_files_async().await?;
    data_files_after.sort_by(|a, b| a.file_path.cmp(&b.file_path));
    assert_eq!(
        data_files_before.len(),
        data_files_after.len(),
        "Parquet data files count must remain identical"
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

    // 4. Verify manifest registration of the table-level cluster index
    let store = create_object_store(&uri)?;
    let manager = ManifestManager::new(store.clone(), "", &uri);
    let (manifest, _entries, _version) = manager.load_latest_full().await?;
    assert_eq!(
        manifest.table_indexes.len(),
        1,
        "Manifest must register exactly 1 table-level cluster index"
    );
    let table_idx = &manifest.table_indexes[0];
    assert_eq!(table_idx.column_name, "embedding");
    assert_eq!(table_idx.cluster_count, 2);
    assert_eq!(table_idx.source_segments.len(), 3);
    assert_eq!(table_idx.source_record_count, 150);

    // 5. Invariant: SHOW CONTIGUOUS CLUSTERS ARE IN THE INDEX ARTIFACT
    // Read the Puffin container directly from disk
    let artifact_path = dir.path().join(&table_idx.file_path);
    assert!(
        artifact_path.exists(),
        "Cluster index artifact file must exist on disk"
    );
    let file_bytes = std::fs::read(&artifact_path)?;
    let puffin_reader = PuffinReader::new(Cursor::new(&file_bytes))?;
    let blobs = &puffin_reader.footer().blobs;

    // Collect cluster graph and node metadata blobs
    let mut cluster_graphs: std::collections::HashMap<usize, (u64, u64)> =
        std::collections::HashMap::new();
    let mut cluster_metas: std::collections::HashMap<usize, (u64, u64)> =
        std::collections::HashMap::new();

    for blob in blobs {
        if let Some(cid_str) = blob.properties.get("cluster_id") {
            let cid: usize = cid_str.parse()?;
            if blob.r#type == PUFFIN_BLOB_VECTOR_CLUSTER_NODE_META {
                cluster_metas.insert(cid, (blob.offset as u64, blob.length as u64));
            } else if blob.r#type == PUFFIN_BLOB_VECTOR_HNSW_GRAPH {
                cluster_graphs.insert(cid, (blob.offset as u64, blob.length as u64));
            }
        }
    }

    assert_eq!(
        cluster_graphs.len(),
        2,
        "Expected 2 contiguous cluster graph blobs"
    );
    assert_eq!(
        cluster_metas.len(),
        2,
        "Expected 2 contiguous cluster node_meta blobs"
    );

    // Inspect each cluster's contiguous layout and member provenance
    for cid in 0..2 {
        let (g_off, g_len) = cluster_graphs[&cid];
        let (m_off, m_len) = cluster_metas[&cid];

        // A. Assert each blob is a single contiguous, non-empty byte slice within the Puffin container
        assert!(g_len > 0, "Cluster {cid} graph blob must be non-empty");
        assert!(m_len > 0, "Cluster {cid} node_meta blob must be non-empty");
        assert!(g_off + g_len <= file_bytes.len() as u64);
        assert!(m_off + m_len <= file_bytes.len() as u64);

        // Disjoint contiguous slices
        assert!(
            g_off + g_len <= m_off || m_off + m_len <= g_off,
            "Cluster {cid} blobs must occupy distinct contiguous byte intervals"
        );

        // B. Assert node_meta unifies rows across ALL THREE distinct segments into this single contiguous cluster partition
        let meta_slice =
            Bytes::copy_from_slice(&file_bytes[m_off as usize..(m_off + m_len) as usize]);
        let meta_reader = ParquetRecordBatchReaderBuilder::try_new(meta_slice)?.build()?;
        let mut segments_in_cluster = HashSet::new();
        let mut total_cluster_members = 0usize;

        for batch in meta_reader {
            let batch = batch?;
            let seg_col = batch
                .column(0)
                .as_any()
                .downcast_ref::<UInt32Array>()
                .expect("segment_idx column");
            for i in 0..batch.num_rows() {
                segments_in_cluster.insert(seg_col.value(i));
                total_cluster_members += 1;
            }
        }

        assert_eq!(
            total_cluster_members, 75,
            "Cluster {cid} must group all 75 members contiguously in the index"
        );
        assert_eq!(
            segments_in_cluster.len(),
            3,
            "Cluster {cid} must contain members originating from all 3 separate Parquet data files (got {segments_in_cluster:?})"
        );
        assert!(segments_in_cluster.contains(&0));
        assert!(segments_in_cluster.contains(&1));
        assert!(segments_in_cluster.contains(&2));
    }

    // 6. Verify End-to-End Vector Query Resolution via Global Cluster Index
    // Querying near cluster 10.0 should retrieve nearest neighbors spanning multiple segments
    let q = vec![10.15; dim];
    let vp = VectorSearchParams::new("embedding", VectorValue::Float32(q), 15)
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

    // Verify all returned IDs belong to cluster 10.0: IDs in [0..25), [50..75), or [100..125)
    for id in &ids {
        let is_cluster_10 =
            (*id >= 0 && *id < 25) || (*id >= 50 && *id < 75) || (*id >= 100 && *id < 125);
        assert!(
            is_cluster_10,
            "Returned ID {id} should belong to cluster center 10.0"
        );
    }

    // Verify that the query hydrated results across multiple distinct segments
    let seg0_hits = ids.iter().filter(|&&id| (0..25).contains(&id)).count();
    let seg1_hits = ids.iter().filter(|&&id| (50..75).contains(&id)).count();
    let seg2_hits = ids.iter().filter(|&&id| (100..125).contains(&id)).count();
    let segments_represented =
        (seg0_hits > 0) as u32 + (seg1_hits > 0) as u32 + (seg2_hits > 0) as u32;

    assert!(
        segments_represented >= 2,
        "Query must resolve hits across multiple data files without file rewrite: seg0={seg0_hits}, seg1={seg1_hits}, seg2={seg2_hits}"
    );

    Ok(())
}

#[tokio::test]
async fn test_sequential_sharded_cluster_index_traversal() -> anyhow::Result<()> {
    // Verifies:
    // 1. Sharded table-level index creation: partitions clusters across sequential Puffin shards
    //    whenever target_shard_bytes is reached.
    // 2. Sequential cluster order: each shard covers a contiguous, monotonic cluster range [c_start..=c_end].
    // 3. Shard traversal: vector query engine traverses across multiple shards on demand,
    //    resolving correct hits across segments with zero Parquet file rewrites.
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    let dim = 8usize;
    let s = schema(dim);
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

    // Ingest 3 distinct segments with 4 distinct cluster centers: [10.0, 30.0, 50.0, 70.0]
    let centers = vec![10.0, 30.0, 50.0, 70.0];
    for seg_idx in 0..3 {
        let batch = mixed_cluster_batch(
            s.clone(),
            (seg_idx * 100) as i32,
            15, // 15 per cluster * 4 = 60 rows per segment -> 180 rows total
            &centers,
            seg_idx,
            dim,
        )?;
        table.write_async(vec![batch]).await?;
        table.commit_async().await?;
    }

    // Force rolling sequential shards with a 5 KB target shard threshold
    table
        .rebuild_cluster_index_with_target("embedding".to_string(), 4, 16, 100, Some(5000))
        .await?;

    // 1. Verify manifest registers the multi-shard table index
    let store = create_object_store(&uri)?;
    let manager = ManifestManager::new(store.clone(), "", &uri);
    let (manifest, _entries, _version) = manager.load_latest_full().await?;
    assert_eq!(manifest.table_indexes.len(), 1);
    let table_idx = &manifest.table_indexes[0];
    assert_eq!(table_idx.column_name, "embedding");
    assert_eq!(table_idx.cluster_count, 4);

    // Verify multiple sequential shards were generated
    assert!(
        table_idx.shard_files.len() >= 2,
        "Target shard threshold must partition index into >= 2 sequential shards (got {})",
        table_idx.shard_files.len()
    );

    // 2. Inspect each shard on disk to verify sequential, monotonic cluster ranges
    let mut last_max_cluster = None;
    for (shard_idx, shard_rel_path) in table_idx.shard_files.iter().enumerate() {
        let shard_abs_path = dir.path().join(shard_rel_path);
        assert!(
            shard_abs_path.exists(),
            "Shard file {shard_rel_path} must exist on disk"
        );

        let shard_bytes = std::fs::read(&shard_abs_path)?;
        let reader = PuffinReader::new(Cursor::new(&shard_bytes))?;

        let mut shard_clusters = Vec::new();
        for blob in &reader.footer().blobs {
            if let Some(cid_str) = blob.properties.get("cluster_id") {
                if blob.r#type == PUFFIN_BLOB_VECTOR_HNSW_GRAPH {
                    shard_clusters.push(cid_str.parse::<usize>()?);
                }
            }
        }
        shard_clusters.sort_unstable();
        shard_clusters.dedup();

        assert!(
            !shard_clusters.is_empty(),
            "Shard {shard_idx} must contain at least one cluster"
        );

        let min_c = *shard_clusters.first().unwrap();
        let max_c = *shard_clusters.last().unwrap();

        // Sequential invariant: this shard's minimum cluster must be strictly greater
        // than the previous shard's maximum cluster
        if let Some(prev_max) = last_max_cluster {
            assert!(
                min_c > prev_max,
                "Shard {shard_idx} min cluster {min_c} must be strictly greater than previous shard max cluster {prev_max}"
            );
        }
        last_max_cluster = Some(max_c);
    }

    // 3. Verify End-to-End Search Traversal across Sequential Shards
    // Query 1: probe cluster 10.0 (in early shard)
    let q1 = vec![10.15; dim];
    let vp1 = VectorSearchParams::new("embedding", VectorValue::Float32(q1), 10)
        .with_metric(benostreamdb::core::index::VectorMetric::L2)
        .with_probes(2);

    let res1 = table
        .read_expr_with_config_async(
            None,
            Some(vec![vp1]),
            Some(&["id"]),
            table.query_config().clone(),
            None,
        )
        .await?;
    assert!(!res1.is_empty(), "Query 1 must return hits");
    let ids1: Vec<i32> = res1
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
    for id in &ids1 {
        let in_c10 =
            (*id >= 0 && *id < 15) || (*id >= 100 && *id < 115) || (*id >= 200 && *id < 215);
        assert!(in_c10, "Hit {id} must belong to cluster 10.0");
    }

    // Query 2: probe cluster 70.0 (in later shard)
    let q2 = vec![70.15; dim];
    let vp2 = VectorSearchParams::new("embedding", VectorValue::Float32(q2), 10)
        .with_metric(benostreamdb::core::index::VectorMetric::L2)
        .with_probes(2);

    let res2 = table
        .read_expr_with_config_async(
            None,
            Some(vec![vp2]),
            Some(&["id"]),
            table.query_config().clone(),
            None,
        )
        .await?;
    assert!(!res2.is_empty(), "Query 2 must return hits");
    let ids2: Vec<i32> = res2
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
    for id in &ids2 {
        let in_c70 =
            (*id >= 45 && *id < 60) || (*id >= 145 && *id < 160) || (*id >= 245 && *id < 260);
        assert!(in_c70, "Hit {id} must belong to cluster 70.0");
    }

    Ok(())
}

#[tokio::test]
async fn test_sql_execute_cluster_index_and_warmup() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = dir.path().to_str().unwrap().to_string();
    let dim = 8;
    let s = schema(dim);

    let table = Table::create_async(uri.clone(), s.clone()).await?;

    let b1 = cluster_batch(s.clone(), 0, 50, 1.0, dim)?;
    let b2 = cluster_batch(s.clone(), 50, 50, 50.0, dim)?;
    table.write_async(vec![b1]).await?;
    table.commit_async().await?;
    table.write_async(vec![b2]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let session = benostreamdb::core::sql::session::BenoStreamSession::new(None);
    session.register_table("events", Arc::new(table.clone()))?;

    // 1. Run rebuild_cluster_index via standard SQL DDL
    let sql_rebuild = "ALTER TABLE events EXECUTE rebuild_cluster_index(column => 'embedding', k => 2, m => 16, ef_c => 100, target_shard_mb => 512)";
    let (res, _) = session.sql(sql_rebuild).await?;
    assert!(res.is_empty() || res[0].num_rows() == 0);

    // Verify manifest has table-level cluster index
    let store = create_object_store(&uri)?;
    let manager = ManifestManager::new(store.clone(), "", &uri);
    let (manifest, _, _) = manager.load_latest_full().await?;
    assert_eq!(manifest.table_indexes.len(), 1);
    assert_eq!(manifest.table_indexes[0].column_name, "embedding");
    assert_eq!(manifest.table_indexes[0].cluster_count, 2);

    // 2. Run warmup via standard SQL DDL
    let sql_warmup = "ALTER TABLE events EXECUTE warmup(n_probes => 2, max_memory_gb => 1.0)";
    let (res_warm, _) = session.sql(sql_warmup).await?;
    assert!(res_warm.is_empty() || res_warm[0].num_rows() == 0);

    Ok(())
}
