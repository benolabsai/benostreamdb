// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! End-to-end verification matrix for Hive-partitioned indexing, query pushdown,
//! ANN/KNN routing, maintenance (backfill/reindex/compaction/vacuum/merge), and
//! non-local URI prefix handling (Step 1D).

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use arrow::array::{Array, Date32Array, FixedSizeListArray, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::compaction::Compactor;
use benostreamdb::core::index::VectorValue;
use benostreamdb::core::manifest::{
    IndexAlgorithm, Manifest, ManifestEntry, ManifestManager, PartitionField, PartitionSpec,
};
use benostreamdb::core::planner::FilterExpr;
use benostreamdb::core::puffin::{
    PuffinReader, PUFFIN_BLOB_GRAPH_CSR_DICT, PUFFIN_BLOB_GRAPH_CSR_EDGES,
    PUFFIN_BLOB_GRAPH_CSR_OFFSETS, PUFFIN_BLOB_HNSW_IVF_CENTROIDS, PUFFIN_BLOB_JSON_PATH,
    PUFFIN_BLOB_LEXICAL_BM25, PUFFIN_BLOB_LEXICAL_DOCLEN, PUFFIN_BLOB_VECTOR_HNSW_GRAPH,
};
use benostreamdb::core::search::KeywordSearchParams;
use benostreamdb::core::sql::session::BenoStreamSession;
use benostreamdb::core::storage::create_object_store;
use benostreamdb::core::table::{MergeMode, PreloadOptions, Table, VectorSearchParams};
use tempfile::tempdir;
use tokio::sync::Mutex;

/// Serialize cache/env-sensitive tests within this test binary.
static SERIAL: Mutex<()> = Mutex::const_new(());

fn days_since_epoch(year: i32, month: u32, day: u32) -> i32 {
    let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1).expect("epoch");
    let d = chrono::NaiveDate::from_ymd_opt(year, month, day).expect("valid date");
    d.signed_duration_since(epoch).num_days() as i32
}

fn fsl_array(vecs: Vec<Vec<f32>>, dim: usize) -> FixedSizeListArray {
    let data: Vec<Option<Vec<Option<f32>>>> = vecs
        .into_iter()
        .map(|v| Some(v.into_iter().map(Some).collect()))
        .collect();
    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(data, dim as i32)
}

fn walk_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(root) else {
        return out;
    };
    for entry in rd.flatten() {
        let p = entry.path();
        if p.is_dir() {
            out.extend(walk_files(&p));
        } else {
            out.push(p);
        }
    }
    out
}

fn ids_from_batches(batches: &[RecordBatch]) -> HashSet<i32> {
    let mut out = HashSet::new();
    for b in batches {
        if let Some(col) = b.column_by_name("id") {
            if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        out.insert(arr.value(i));
                    }
                }
            } else if let Some(arr) = col.as_any().downcast_ref::<Int64Array>() {
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        out.insert(arr.value(i) as i32);
                    }
                }
            }
        }
    }
    out
}

fn ordered_ids_from_batches(batches: &[RecordBatch]) -> Vec<i32> {
    let mut out = Vec::new();
    for b in batches {
        if let Some(col) = b.column_by_name("id") {
            if let Some(arr) = col.as_any().downcast_ref::<Int32Array>() {
                for i in 0..arr.len() {
                    if !arr.is_null(i) {
                        out.push(arr.value(i));
                    }
                }
            }
        }
    }
    out
}

async fn load_entries(table: &Table) -> Result<(Manifest, Vec<ManifestEntry>)> {
    let manager = ManifestManager::new(table.store.clone(), "", &table.uri);
    let (manifest, entries, _) = manager.load_latest_full().await?;
    Ok((manifest, entries))
}

#[tokio::test]
async fn partitioned_hive_layout_and_all_indexes() -> Result<()> {
    let _guard = SERIAL.lock().await;
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());
    let dim = 4usize;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("event_date", DataType::Date32, false),
        Field::new("body", DataType::Utf8, false),
        Field::new("meta", DataType::Utf8, false),
        Field::new("src", DataType::Int64, false),
        Field::new("dst", DataType::Int64, false),
        Field::new(
            "emb_ivf",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
        Field::new(
            "emb_tq8",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]));

    let spec = PartitionSpec {
        spec_id: 1,
        fields: vec![
            PartitionField::new_single(
                2,
                Some(1000),
                "category".to_string(),
                "identity".to_string(),
            ),
            PartitionField::new_single(
                3,
                Some(1001),
                "event_date".to_string(),
                "identity".to_string(),
            ),
        ],
    };

    let table = Table::create_partitioned_async(uri.clone(), schema.clone(), spec).await?;
    table.set_autocommit(false);

    table
        .add_index("id".to_string(), IndexAlgorithm::Bitmap)
        .await?;
    table
        .add_index("category".to_string(), IndexAlgorithm::Bitmap)
        .await?;
    table
        .add_index("event_date".to_string(), IndexAlgorithm::Bitmap)
        .await?;
    table
        .add_index(
            "body".to_string(),
            IndexAlgorithm::Bm25 {
                k1: 1.5,
                b: 0.75,
                tokenizer: "default".to_string(),
            },
        )
        .await?;
    table
        .add_index(
            "meta".to_string(),
            IndexAlgorithm::JsonPath {
                paths: vec!["$.kind".to_string()],
            },
        )
        .await?;
    table
        .add_index(
            "src".to_string(),
            IndexAlgorithm::CsrGraph {
                src_column: "src".to_string(),
                dst_column: "dst".to_string(),
            },
        )
        .await?;
    table
        .add_index("emb_ivf".to_string(), IndexAlgorithm::hnsw())
        .await?;
    table
        .add_index("emb_tq8".to_string(), IndexAlgorithm::hnsw_tq8())
        .await?;

    let d_a = days_since_epoch(2025, 12, 1);
    let d_b = days_since_epoch(2026, 1, 1);

    let make_batch =
        |start: i32, count: i32, cat: &str, date_days: i32, base_vec: f32| -> Result<RecordBatch> {
            let ids: Vec<i32> = (start..start + count).collect();
            let cats: Vec<&str> = vec![cat; count as usize];
            let dates: Vec<i32> = vec![date_days; count as usize];
            let bodies: Vec<String> = ids
                .iter()
                .map(|i| format!("partition {cat} telemetry signal doc{i}"))
                .collect();
            let metas: Vec<String> = ids
                .iter()
                .map(|i| {
                    let kind = if i % 2 == 0 { "alert" } else { "metric" };
                    format!(r#"{{"kind":"{kind}","seq":{i}}}"#)
                })
                .collect();
            let srcs: Vec<i64> = ids.iter().map(|&i| i as i64).collect();
            let dsts: Vec<i64> = ids.iter().map(|&i| (i + 1) as i64).collect();
            let vecs: Vec<Vec<f32>> = (0..count)
                .map(|i| vec![base_vec + (i as f32) * 0.01, 0.1, 0.2, 0.3])
                .collect();

            Ok(RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int32Array::from(ids)),
                    Arc::new(StringArray::from(cats)),
                    Arc::new(Date32Array::from(dates)),
                    Arc::new(StringArray::from(bodies)),
                    Arc::new(StringArray::from(metas)),
                    Arc::new(Int64Array::from(srcs)),
                    Arc::new(Int64Array::from(dsts)),
                    Arc::new(fsl_array(vecs.clone(), dim)),
                    Arc::new(fsl_array(vecs, dim)),
                ],
            )?)
        };

    table
        .write_async(vec![
            make_batch(0, 12, "A", d_a, 0.0)?,
            make_batch(100, 12, "B", d_b, 10.0)?,
        ])
        .await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    // Reopen table cleanly to ensure metadata persists across reopen.
    drop(table);
    let table = Table::new_async(uri.clone()).await?;
    let (_manifest, entries) = load_entries(&table).await?;
    assert_eq!(
        entries.len(),
        2,
        "expected 2 partitioned segments (A and B)"
    );

    let prefix_a = format!("category=A/event_date={d_a}/");
    let prefix_b = format!("category=B/event_date={d_b}/");

    for entry in &entries {
        assert!(
            entry.file_path.starts_with(&prefix_a) || entry.file_path.starts_with(&prefix_b),
            "segment file_path `{}` must live inside Hive partition directory",
            entry.file_path
        );
        let pq_full = dir.path().join(&entry.file_path);
        assert!(pq_full.exists(), "missing Parquet segment at {pq_full:?}");

        let expected_prefix = if entry.file_path.starts_with(&prefix_a) {
            &prefix_a
        } else {
            &prefix_b
        };

        let mut categories: HashSet<String> = HashSet::new();
        let mut puffin_paths: HashSet<String> = HashSet::new();
        for idx in &entry.index_files {
            assert!(
                idx.file_path.starts_with(expected_prefix),
                "sidecar `{}` must share Hive prefix `{expected_prefix}`",
                idx.file_path
            );
            assert!(
                idx.file_path.ends_with(".puffin"),
                "sidecar `{}` must use Puffin bundle",
                idx.file_path
            );
            categories.insert(idx.index_category.clone());
            puffin_paths.insert(idx.file_path.clone());
        }

        for req in ["lexical", "json_path", "graph_v2", "vector"] {
            assert!(
                categories.contains(req),
                "segment `{}` missing index category `{req}`; found {categories:?}",
                entry.file_path
            );
        }

        for col in ["id", "category", "event_date"] {
            assert!(
                entry
                    .index_files
                    .iter()
                    .any(|f| f.column_name == col && f.is_lexical() && f.algorithm == "inverted"),
                "segment `{}` missing scalar inverted index for `{col}`; index_files: {:?}",
                entry.file_path,
                entry.index_files
            );
        }
        assert!(
            entry
                .index_files
                .iter()
                .any(|f| f.column_name == "body" && f.is_lexical() && f.algorithm == "inverted"),
            "segment `{}` missing BM25 inverted index for `body`",
            entry.file_path
        );
        assert!(
            entry
                .index_files
                .iter()
                .any(|f| f.column_name == "body" && f.is_lexical() && f.algorithm == "doclen"),
            "segment `{}` missing BM25 doclen blob for `body`",
            entry.file_path
        );
        assert!(
            entry
                .index_files
                .iter()
                .any(|f| f.column_name == "meta" && f.index_category == "json_path"),
            "segment `{}` missing json_path index for `meta`",
            entry.file_path
        );
        assert!(
            entry
                .index_files
                .iter()
                .any(|f| f.column_name == "src" && f.index_category == "graph_v2"),
            "segment `{}` missing graph_v2 index for `src`",
            entry.file_path
        );
        assert!(
            entry.index_files.iter().any(|f| f.column_name == "emb_ivf"
                && f.index_category == "vector"
                && f.algorithm == "hnsw"),
            "segment `{}` missing vector hnsw index for `emb_ivf`",
            entry.file_path
        );
        assert!(
            entry.index_files.iter().any(|f| f.column_name == "emb_tq8"
                && f.index_category == "vector"
                && f.algorithm == "tq8"),
            "segment `{}` missing vector tq8 index for `emb_tq8`",
            entry.file_path
        );

        assert_eq!(
            puffin_paths.len(),
            1,
            "expected 1 Puffin sidecar per segment"
        );
        let puffin_full = dir
            .path()
            .join(puffin_paths.iter().next().expect("one puffin"));
        assert!(
            puffin_full.exists(),
            "missing Puffin file at {puffin_full:?}"
        );
        let bytes = std::fs::read(&puffin_full)?;
        let pr = PuffinReader::new(std::io::Cursor::new(bytes))?;
        let blob_types: HashSet<&str> = pr
            .footer()
            .blobs
            .iter()
            .map(|b| b.r#type.as_str())
            .collect();
        for expected_blob in [
            PUFFIN_BLOB_LEXICAL_BM25,
            PUFFIN_BLOB_LEXICAL_DOCLEN,
            PUFFIN_BLOB_JSON_PATH,
            PUFFIN_BLOB_GRAPH_CSR_OFFSETS,
            PUFFIN_BLOB_GRAPH_CSR_EDGES,
            PUFFIN_BLOB_GRAPH_CSR_DICT,
            PUFFIN_BLOB_VECTOR_HNSW_GRAPH,
            PUFFIN_BLOB_HNSW_IVF_CENTROIDS,
        ] {
            assert!(
                blob_types.contains(expected_blob),
                "Puffin sidecar {:?} missing expected blob `{expected_blob}`; has {blob_types:?}",
                puffin_full
            );
        }
        let identity_blobs = pr
            .footer()
            .blobs
            .iter()
            .filter(|b| {
                b.r#type == PUFFIN_BLOB_LEXICAL_BM25
                    && b.properties.get("analyzer").map(String::as_str) == Some("identity")
            })
            .count();
        assert!(
            identity_blobs >= 3,
            "expected >= 3 identity scalar posting blobs (id, category, event_date), got {identity_blobs}"
        );
    }

    // Exercise functional queries against the Hive-partitioned indexes.
    let alert_ids = ids_from_batches(
        &table
            .filter(r#"category = 'B' AND json_extract_path_text(meta, 'kind') = 'alert'"#)
            .to_batches()
            .await?,
    );
    assert_eq!(alert_ids, HashSet::from([100, 102, 104, 106, 108, 110]));

    let kw_hits = table
        .execute_keyword_search_as_scored(KeywordSearchParams::new("body", "doc105"))
        .await?;
    assert_eq!(kw_hits.len(), 1, "BM25 keyword hit on partitioned segment");
    assert_eq!(
        kw_hits[0].row_id, 5,
        "expected row_id 5 (id=105) in partition B segment"
    );
    assert!(kw_hits[0].score > 0.0);

    let id_filtered = ids_from_batches(&table.filter("id = 105").to_batches().await?);
    assert_eq!(id_filtered, HashSet::from([105]));

    let date_filtered = ids_from_batches(
        &table
            .filter("event_date >= DATE '2026-01-01'")
            .to_batches()
            .await?,
    );
    assert_eq!(date_filtered, (100..112).collect::<HashSet<i32>>());

    Ok(())
}

#[tokio::test]
async fn partitioned_backfill_reindex_and_preload_preserve_hive_layout() -> Result<()> {
    let _guard = SERIAL.lock().await;
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());
    let dim = 4usize;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("body", DataType::Utf8, false),
        Field::new(
            "emb",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]));

    let spec = PartitionSpec {
        spec_id: 1,
        fields: vec![PartitionField::new_single(
            2,
            Some(1000),
            "category".to_string(),
            "identity".to_string(),
        )],
    };

    let table = Table::create_partitioned_async(uri.clone(), schema.clone(), spec).await?;
    table.set_autocommit(false);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 10, 11, 12, 13])),
            Arc::new(StringArray::from(vec![
                "A", "A", "A", "A", "B", "B", "B", "B",
            ])),
            Arc::new(StringArray::from(vec![
                "alpha signal one",
                "alpha signal two",
                "alpha signal three",
                "alpha signal four",
                "beta signal one",
                "beta signal two",
                "beta signal three",
                "beta signal four",
            ])),
            Arc::new(fsl_array(
                vec![
                    vec![1.0, 0.0, 0.0, 0.0],
                    vec![0.9, 0.1, 0.0, 0.0],
                    vec![0.8, 0.2, 0.0, 0.0],
                    vec![0.7, 0.3, 0.0, 0.0],
                    vec![0.0, 0.0, 1.0, 0.0],
                    vec![0.0, 0.0, 0.9, 0.1],
                    vec![0.0, 0.0, 0.8, 0.2],
                    vec![0.0, 0.0, 0.7, 0.3],
                ],
                dim,
            )),
        ],
    )?;

    // Write BEFORE configuring indexes so add_index triggers partitioned backfill.
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    table
        .add_index(
            "body".to_string(),
            IndexAlgorithm::Bm25 {
                k1: 1.5,
                b: 0.75,
                tokenizer: "default".to_string(),
            },
        )
        .await?;
    table.wait_for_background_tasks_async().await?;

    table
        .add_index("emb".to_string(), IndexAlgorithm::hnsw_tq8())
        .await?;
    table.wait_for_background_tasks_async().await?;

    let (_, entries_after_backfill) = load_entries(&table).await?;
    assert_eq!(entries_after_backfill.len(), 2);
    for entry in &entries_after_backfill {
        let dir_prefix = entry
            .file_path
            .rsplit_once('/')
            .map(|(p, _)| p)
            .expect("hive directory prefix");
        assert!(dir_prefix == "category=A" || dir_prefix == "category=B");
        assert!(entry
            .index_files
            .iter()
            .any(|f| f.is_lexical() && f.file_path.starts_with(dir_prefix)));
        assert!(entry
            .index_files
            .iter()
            .any(|f| f.index_category == "vector" && f.file_path.starts_with(dir_prefix)));
    }

    // Reindex only the inverted column on the partitioned table; vector blobs & Hive paths must remain intact.
    let rebuilt = table.reindex_inverted_column("body").await?;
    assert_eq!(
        rebuilt, 2,
        "must rebuild inverted index for both partitions"
    );

    let (_, entries_after_reindex) = load_entries(&table).await?;
    for entry in &entries_after_reindex {
        let dir_prefix = entry
            .file_path
            .rsplit_once('/')
            .map(|(p, _)| p)
            .expect("hive directory prefix");
        assert!(
            entry
                .index_files
                .iter()
                .any(|f| f.is_lexical() && f.file_path.starts_with(dir_prefix)),
            "lexical index missing or misplaced after reindex: {:?}",
            entry.index_files
        );
        assert!(
            entry
                .index_files
                .iter()
                .any(|f| f.index_category == "vector" && f.file_path.starts_with(dir_prefix)),
            "vector index dropped during reindex_inverted_column: {:?}",
            entry.index_files
        );
    }

    // No orphan `.puffin` should exist at the flat table root.
    let root_puffins: Vec<PathBuf> = std::fs::read_dir(dir.path())?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.ends_with(".puffin"))
        })
        .collect();
    assert!(
        root_puffins.is_empty(),
        "unexpected root puffin files outside Hive dirs: {root_puffins:?}"
    );

    // Preload and ensure_vector_router must resolve nested Puffin paths and warm caches cleanly.
    let stats = table
        .preload_indexes_async(PreloadOptions {
            include_vector: true,
            include_inverted: true,
            include_graph: false,
            max_memory_bytes: 128 * 1024 * 1024,
            spill_to_disk: true,
        })
        .await?;
    assert!(
        stats.indexes_warmed >= 4,
        "expected both lexical and vector indexes across 2 partitions to warm, got {stats:?}"
    );
    assert!(
        stats.last_error.is_none(),
        "preload reported unexpected error on Hive paths: {:?}",
        stats.last_error
    );

    let router = table
        .ensure_vector_router("emb")
        .await?
        .expect("vector router must be built");
    assert!(
        router.len() >= 2,
        "router should index both partition segments"
    );

    Ok(())
}

#[tokio::test]
async fn partitioned_knn_routing_prunes_decoy_and_sql_projection_safety() -> Result<()> {
    let _guard = SERIAL.lock().await;
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());
    let dim = 4usize;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("event_date", DataType::Date32, false),
        Field::new(
            "emb",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]));

    let spec = PartitionSpec {
        spec_id: 1,
        fields: vec![
            PartitionField::new_single(
                2,
                Some(1000),
                "category".to_string(),
                "identity".to_string(),
            ),
            PartitionField::new_single(
                3,
                Some(1001),
                "event_date".to_string(),
                "identity".to_string(),
            ),
        ],
    };

    let table = Table::create_partitioned_async(uri.clone(), schema.clone(), spec).await?;
    table.set_autocommit(false);
    table
        .add_index("category".to_string(), IndexAlgorithm::Bitmap)
        .await?;
    table
        .add_index("event_date".to_string(), IndexAlgorithm::Bitmap)
        .await?;
    table
        .add_index(
            "emb".to_string(),
            IndexAlgorithm::HnswTq8 {
                metric: "cosine".to_string(),
                complexity: 16,
                quality: 32,
            },
        )
        .await?;

    let d_old = days_since_epoch(2025, 12, 1);
    let d_new = days_since_epoch(2026, 1, 15);

    // Partition A (decoy): vectors directly aligned with query [1, 0, 0, 0] (cosine dist ~ 0.0)
    let decoy_ids: Vec<i32> = (100..112).collect();
    let decoy_vecs: Vec<Vec<f32>> = (0..12)
        .map(|i| vec![1.0, (i as f32) * 0.001, 0.0, 0.0])
        .collect();
    let batch_a = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(decoy_ids)),
            Arc::new(StringArray::from(vec!["A"; 12])),
            Arc::new(Date32Array::from(vec![d_old; 12])),
            Arc::new(fsl_array(decoy_vecs, dim)),
        ],
    )?;

    // Partition B (target): 2026-01-15 has nearest match [0.95, 0.31, 0, 0] (id 200..202) and orthogonal noise
    let mut b_ids: Vec<i32> = Vec::new();
    let mut b_vecs: Vec<Vec<f32>> = Vec::new();
    for i in 0..12 {
        b_ids.push(200 + i);
        if i < 4 {
            b_vecs.push(vec![
                0.95 - (i as f32) * 0.02,
                0.31 + (i as f32) * 0.02,
                0.0,
                0.0,
            ]);
        } else {
            b_vecs.push(vec![0.0, 0.0, 0.8 + (i as f32) * 0.01, 0.2]);
        }
    }
    let batch_b = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(b_ids)),
            Arc::new(StringArray::from(vec!["B"; 12])),
            Arc::new(Date32Array::from(vec![d_new; 12])),
            Arc::new(fsl_array(b_vecs, dim)),
        ],
    )?;

    // Commit in separate commits so router records separate segment centroids cleanly.
    table.write_async(vec![batch_a]).await?;
    table.commit_async().await?;
    table.write_async(vec![batch_b]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let (_, entries) = load_entries(&table).await?;
    let mut b_seg_ids = HashSet::new();
    let mut a_seg_ids = HashSet::new();
    for e in &entries {
        let fname = e.file_path.rsplit('/').next().unwrap_or(&e.file_path);
        let seg_id = fname.strip_suffix(".parquet").unwrap_or(fname).to_string();
        if e.partition_values.get("category").and_then(|v| v.as_str()) == Some("B") {
            b_seg_ids.insert(seg_id);
        } else {
            a_seg_ids.insert(seg_id);
        }
    }
    assert!(!a_seg_ids.is_empty() && !b_seg_ids.is_empty());

    let query = vec![1.0f32, 0.0, 0.0, 0.0];
    let router = table
        .ensure_vector_router("emb")
        .await?
        .expect("router must exist");

    // 1. Verify route_allowed excludes decoy segment A even at probes=1.
    let plan = router.route_allowed(&query, 1, Some(&b_seg_ids));
    assert!(
        !plan.candidate_segments.is_empty(),
        "allowed partition B must retain candidates"
    );
    assert!(
        plan.candidate_segments.is_subset(&b_seg_ids),
        "route_allowed must exclude pruned decoy partition A: {:?}",
        plan.candidate_segments
    );

    // 2. Verify SQL pushdown with partition filter + SELECT projecting ONLY `id` (disjoint from filter & vector col).
    let table = Arc::new(table);
    let session = BenoStreamSession::new(None);
    session.register_table("t", table.clone())?;

    let (batches, out_schema) = session
        .sql(
            "SELECT id FROM t \
             WHERE category = 'B' AND event_date >= DATE '2026-01-01' \
             ORDER BY emb <=> [1.0, 0.0, 0.0, 0.0] \
             LIMIT 3",
        )
        .await?;

    assert_eq!(out_schema.fields().len(), 1);
    assert_eq!(out_schema.field(0).name(), "id");

    let returned_ids = ordered_ids_from_batches(&batches);
    assert_eq!(returned_ids.len(), 3, "LIMIT 3 must return 3 rows");
    for id in &returned_ids {
        assert!(
            (200..212).contains(id),
            "SQL KNN leaked row id {id} from decoy partition A"
        );
    }
    assert!(
        returned_ids.contains(&200),
        "exact closest neighbor inside surviving partition B (id 200) must be recalled; got {returned_ids:?}"
    );

    // 3. Verify explain() reports the quantized index path and pruning summary cleanly.
    let plan_txt = table
        .explain(
            Some("category = 'B'"),
            Some(vec![VectorSearchParams::new(
                "emb",
                VectorValue::Float32(query),
                3,
            )]),
        )
        .await;
    assert!(
        plan_txt.contains("HNSW-TQ8 Cluster Index"),
        "explain() must report quantized TQ8 path, got:\n{plan_txt}"
    );
    assert!(
        !plan_txt.contains("Brute Force Scan (No Index)"),
        "explain() must not fall back to Brute Force Scan on indexed partition, got:\n{plan_txt}"
    );

    Ok(())
}

#[tokio::test]
async fn partitioned_cluster_index_deletes_and_hybrid_search() -> Result<()> {
    let _guard = SERIAL.lock().await;
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());
    let dim = 4usize;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("body", DataType::Utf8, false),
        Field::new(
            "emb",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32,
            ),
            false,
        ),
    ]));

    let spec = PartitionSpec {
        spec_id: 1,
        fields: vec![PartitionField::new_single(
            2,
            Some(1000),
            "category".to_string(),
            "identity".to_string(),
        )],
    };

    let table = Table::create_partitioned_async(uri, schema.clone(), spec).await?;
    table.set_autocommit(false);
    table
        .add_index("id".to_string(), IndexAlgorithm::Bitmap)
        .await?;
    table
        .add_index("category".to_string(), IndexAlgorithm::Bitmap)
        .await?;
    table
        .add_index(
            "body".to_string(),
            IndexAlgorithm::Bm25 {
                k1: 1.2,
                b: 0.75,
                tokenizer: "default".to_string(),
            },
        )
        .await?;
    table
        .add_index("emb".to_string(), IndexAlgorithm::hnsw_tq8())
        .await?;

    let mut ids = Vec::new();
    let mut cats = Vec::new();
    let mut bodies = Vec::new();
    let mut vecs = Vec::new();

    for i in 1..=8 {
        ids.push(i);
        cats.push("A");
        bodies.push(format!("alpha storage engine doc {i}"));
        vecs.push(vec![0.0, 1.0, (i as f32) * 0.05, 0.0]);
    }
    for i in 101..=108 {
        ids.push(i);
        cats.push("B");
        let text = match i {
            101 => "quantum ledger routing exact deleted winner",
            102 => "quantum ledger routing runner up surviving",
            _ => "beta storage partition surviving note",
        };
        bodies.push(text.to_string());
        // Make 101 closest to [1,0,0,0], and 102 second closest.
        let vx = match i {
            101 => 1.0,
            102 => 0.9,
            _ => 0.1 + (i - 101) as f32 * 0.05,
        };
        let vy = match i {
            101 => 0.0,
            102 => 0.2,
            _ => 0.8,
        };
        vecs.push(vec![vx, vy, 0.0, 0.0]);
    }

    let body_refs: Vec<&str> = bodies.iter().map(String::as_str).collect();
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(cats)),
            Arc::new(StringArray::from(body_refs)),
            Arc::new(fsl_array(vecs, dim)),
        ],
    )?;

    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    // Build table-level cluster index over both Hive partitions.
    table
        .rebuild_cluster_index("emb".to_string(), 2, 16, 32)
        .await?;

    let q = vec![1.0f32, 0.0, 0.0, 0.0];
    let pre_delete = table
        .query()
        .vector_search("emb", VectorValue::Float32(q.clone()), 2)
        .to_batches()
        .await?;
    let pre_ids = ids_from_batches(&pre_delete);
    assert!(pre_ids.contains(&101), "pre-delete search should hit 101");

    // Delete exact nearest neighbor id = 101.
    table.delete_async("id = 101").await?;

    // Unfiltered vector search must step aside from stale ClusterIndex when has_deletes == true.
    let post_vec_batches = table
        .query()
        .vector_search("emb", VectorValue::Float32(q.clone()), 3)
        .to_batches()
        .await?;
    let post_vec_ids = ids_from_batches(&post_vec_batches);
    assert!(
        !post_vec_ids.contains(&101),
        "deleted id=101 resurrected from ClusterIndex fast path: {post_vec_ids:?}"
    );
    assert!(
        post_vec_ids.contains(&102),
        "surviving neighbor id=102 expected in post-delete top-3: {post_vec_ids:?}"
    );

    // Filtered keyword & phrase searches must skip deleted row 101 and prune partition A.
    let filter_b = FilterExpr::parse_sql("category = 'B'", table.arrow_schema()).await?;

    let kw_scored = table
        .execute_keyword_search_as_scored_filtered(
            KeywordSearchParams::new("body", "quantum"),
            Some(filter_b.clone()),
        )
        .await?;
    let kw_batches = table.fetch_results_by_id(kw_scored, Some(&["id"])).await?;
    let kw_ids = ids_from_batches(&kw_batches);
    assert_eq!(kw_ids, HashSet::from([102]));

    let phrase_scored = table
        .execute_phrase_search_as_scored_filtered(
            "body",
            "quantum ledger routing",
            0,
            Some("default"),
            Some(filter_b),
        )
        .await?;
    let phrase_batches = table
        .fetch_results_by_id(phrase_scored, Some(&["id"]))
        .await?;
    let phrase_ids = ids_from_batches(&phrase_batches);
    assert_eq!(phrase_ids, HashSet::from([102]));

    // Hybrid RRF filtered by partition must honor both delete files and partition filter.
    let hybrid_batches = table
        .hybrid_search_batches_filtered(
            "body",
            "quantum ledger",
            "emb",
            q,
            5,
            60.0,
            Some("category = 'B'"),
        )
        .await?;
    let hybrid_ids = ids_from_batches(&hybrid_batches);
    assert!(
        !hybrid_ids.contains(&101),
        "deleted id=101 must never appear in hybrid search: {hybrid_ids:?}"
    );
    assert!(
        hybrid_ids.contains(&102),
        "surviving match id=102 missing in hybrid search: {hybrid_ids:?}"
    );

    Ok(())
}

#[tokio::test]
async fn partitioned_write_buffer_and_split_reader_edge_cases() -> Result<()> {
    let _guard = SERIAL.lock().await;
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("note", DataType::Utf8, false),
    ]));

    let spec = PartitionSpec {
        spec_id: 1,
        fields: vec![PartitionField::new_single(
            2,
            Some(1000),
            "category".to_string(),
            "identity".to_string(),
        )],
    };

    let table = Arc::new(Table::create_partitioned_async(uri, schema.clone(), spec).await?);
    table.set_autocommit(false);
    table
        .add_index("id".to_string(), IndexAlgorithm::Bitmap)
        .await?;
    table
        .add_index("category".to_string(), IndexAlgorithm::Bitmap)
        .await?;
    // Deliberately leave `note` UNINDEXED to test row-group fallback filtering.

    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4])),
            Arc::new(StringArray::from(vec!["A", "A", "B", "B"])),
            Arc::new(StringArray::from(vec![
                "plain",
                "target_note",
                "other",
                "target_note",
            ])),
        ],
    )?;

    // Part A: Query uncommitted write buffer with SELECT disjoint from WHERE predicate.
    table.write_async(vec![batch]).await?;

    let expr_a = FilterExpr::parse_sql("category = 'A'", table.arrow_schema()).await?;
    let wb_batches = table.read_write_buffer_expr(Some(&expr_a), Some(&["id"]))?;
    assert_eq!(ids_from_batches(&wb_batches), HashSet::from([1, 2]));
    for b in &wb_batches {
        assert_eq!(b.schema().fields().len(), 1);
        assert_eq!(b.schema().field(0).name(), "id");
    }

    let session = BenoStreamSession::new(None);
    session.register_table("t", table.clone())?;
    let (sql_batches, sql_schema) = session.sql("SELECT id FROM t WHERE category = 'A'").await?;
    assert_eq!(sql_schema.fields().len(), 1);
    assert_eq!(sql_schema.field(0).name(), "id");
    assert_eq!(ids_from_batches(&sql_batches), HashSet::from([1, 2]));

    // Part B: Commit & verify read_split_async behavior on empty-index hits and unindexed fallback filters.
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let splits = table
        .get_splits_async(64 * 1024 * 1024, Some("category = 'A'"))
        .await?;
    assert_eq!(
        splits.len(),
        1,
        "partition filter must prune to only split A"
    );
    let split_a = &splits[0];
    assert!(
        split_a.file_path.contains("category=A/"),
        "expected split inside category=A/, got {}",
        split_a.file_path
    );

    use futures::StreamExt;
    // Empty-index hit on indexed `id`: MUST NOT fall through to unfiltered `stream_row_groups`.
    let empty_hit_batches: Vec<RecordBatch> = table
        .read_split_async(split_a, vec!["id".to_string()], Some("id = 999999"))
        .await?
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(
        ids_from_batches(&empty_hit_batches),
        HashSet::new(),
        "empty-index hit must return 0 rows instead of falling back to unfiltered scan"
    );

    // Filter on UNINDEXED `note` with projection disjoint (`SELECT id`): fallback must filter accurately.
    let unindexed_batches: Vec<RecordBatch> = table
        .read_split_async(
            split_a,
            vec!["id".to_string()],
            Some("note = 'target_note'"),
        )
        .await?
        .collect::<Vec<_>>()
        .await
        .into_iter()
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(
        ids_from_batches(&unindexed_batches),
        HashSet::from([2]),
        "unindexed column filter must post-filter fallback scan before projecting"
    );
    for b in &unindexed_batches {
        assert_eq!(b.schema().fields().len(), 1);
        assert_eq!(b.schema().field(0).name(), "id");
    }

    Ok(())
}

#[tokio::test]
async fn partitioned_merge_compaction_vacuum_schema_evolution_and_memory_uri() -> Result<()> {
    let _guard = SERIAL.lock().await;

    // ── Subtest A: Partitioned Merge (CoW & MoR) + Schema Evolution + Rebuilt Indexes ──
    {
        let dir = tempdir()?;
        let uri = format!("file://{}", dir.path().display());
        let dim = 4usize;

        let base_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("category", DataType::Utf8, false),
            Field::new("val", DataType::Int32, false),
            Field::new(
                "emb",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
        ]));

        let spec = PartitionSpec {
            spec_id: 1,
            fields: vec![PartitionField::new_single(
                2,
                Some(1000),
                "category".to_string(),
                "identity".to_string(),
            )],
        };

        let table = Table::create_partitioned_async(uri.clone(), base_schema.clone(), spec).await?;
        table.set_autocommit(false);
        table
            .add_index("id".to_string(), IndexAlgorithm::Bitmap)
            .await?;
        table
            .add_index("category".to_string(), IndexAlgorithm::Bitmap)
            .await?;
        table
            .add_index("emb".to_string(), IndexAlgorithm::hnsw_tq8())
            .await?;

        let init_batch = RecordBatch::try_new(
            base_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 6, 7, 8])),
                Arc::new(StringArray::from(vec!["A", "A", "A", "B", "B", "B"])),
                Arc::new(Int32Array::from(vec![10, 20, 30, 60, 70, 80])),
                Arc::new(fsl_array(vec![vec![1.0, 0.0, 0.0, 0.0]; 6], dim)),
            ],
        )?;
        table.write_async(vec![init_batch]).await?;
        table.commit_async().await?;
        table.wait_for_background_tasks_async().await?;

        // Evolve table schema BEFORE merge.
        table.add_column("extra", DataType::Utf8).await?;

        let evolved_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("category", DataType::Utf8, false),
            Field::new("val", DataType::Int32, false),
            Field::new(
                "emb",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    dim as i32,
                ),
                false,
            ),
            Field::new("extra", DataType::Utf8, true),
        ]));

        // Merge (CopyOnWrite): update id=2 (A), id=7 (B), insert id=20 (new partition C).
        let cow_batch = RecordBatch::try_new(
            evolved_schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![2, 7, 20])),
                Arc::new(StringArray::from(vec!["A", "B", "C"])),
                Arc::new(Int32Array::from(vec![200, 700, 2000])),
                Arc::new(fsl_array(vec![vec![0.0, 1.0, 0.0, 0.0]; 3], dim)),
                Arc::new(StringArray::from(vec![
                    Some("cow_a"),
                    Some("cow_b"),
                    Some("cow_c"),
                ])),
            ],
        )?;
        let table_cow = table.clone();
        tokio::task::spawn_blocking(move || {
            table_cow.merge(vec![cow_batch], "id", MergeMode::CopyOnWrite)
        })
        .await??;

        let (_, cow_entries) = load_entries(&table).await?;
        assert!(
            cow_entries
                .iter()
                .any(|e| e.file_path.starts_with("category=C/seg_")),
            "CoW unmatched insert for C must land in category=C/, entries: {:?}",
            cow_entries.iter().map(|e| &e.file_path).collect::<Vec<_>>()
        );
        for entry in &cow_entries {
            let prefix = entry
                .file_path
                .rsplit_once('/')
                .map(|(p, _)| p)
                .unwrap_or("");
            assert!(
                !prefix.is_empty(),
                "CoW entry `{}` must preserve Hive partition subdir",
                entry.file_path
            );
            assert!(
                entry
                    .index_files
                    .iter()
                    .any(|i| i.column_name == "id" && i.file_path.starts_with(prefix)),
                "CoW segment `{}` must carry rebuilt scalar index inside `{prefix}/`",
                entry.file_path
            );
            assert!(
                entry
                    .index_files
                    .iter()
                    .any(|i| i.column_name == "emb" && i.file_path.starts_with(prefix)),
                "CoW segment `{}` must carry rebuilt vector index inside `{prefix}/`",
                entry.file_path
            );
        }

        // Merge (MergeOnRead): update id=3 (A), insert id=30 (new partition D).
        let mor_batch = RecordBatch::try_new(
            evolved_schema,
            vec![
                Arc::new(Int32Array::from(vec![3, 30])),
                Arc::new(StringArray::from(vec!["A", "D"])),
                Arc::new(Int32Array::from(vec![300, 3000])),
                Arc::new(fsl_array(vec![vec![0.0, 0.0, 1.0, 0.0]; 2], dim)),
                Arc::new(StringArray::from(vec![Some("mor_a"), Some("mor_d")])),
            ],
        )?;
        let table_mor = table.clone();
        tokio::task::spawn_blocking(move || {
            table_mor.merge(vec![mor_batch], "id", MergeMode::MergeOnRead)
        })
        .await??;

        let table_arc = Arc::new(table);
        let session = BenoStreamSession::new(None);
        session.register_table("t", table_arc.clone())?;
        let (rows, _) = session
            .sql("SELECT id, category, val, extra FROM t ORDER BY id")
            .await?;

        let mut got: HashMap<i32, (String, i32, Option<String>)> = HashMap::new();
        for b in &rows {
            let ids = b
                .column_by_name("id")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let cats = b
                .column_by_name("category")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            let vals = b
                .column_by_name("val")
                .unwrap()
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let extras = b
                .column_by_name("extra")
                .unwrap()
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            for i in 0..b.num_rows() {
                got.insert(
                    ids.value(i),
                    (
                        cats.value(i).to_string(),
                        vals.value(i),
                        if extras.is_null(i) {
                            None
                        } else {
                            Some(extras.value(i).to_string())
                        },
                    ),
                );
            }
        }
        assert_eq!(
            got.len(),
            8,
            "expected 6 original updated/kept + 2 inserts (20, 30)"
        );
        assert_eq!(
            got.get(&2),
            Some(&("A".to_string(), 200, Some("cow_a".to_string())))
        );
        assert_eq!(
            got.get(&3),
            Some(&("A".to_string(), 300, Some("mor_a".to_string())))
        );
        assert_eq!(
            got.get(&7),
            Some(&("B".to_string(), 700, Some("cow_b".to_string())))
        );
        assert_eq!(
            got.get(&20),
            Some(&("C".to_string(), 2000, Some("cow_c".to_string())))
        );
        assert_eq!(
            got.get(&30),
            Some(&("D".to_string(), 3000, Some("mor_d".to_string())))
        );
        assert_eq!(got.get(&1), Some(&("A".to_string(), 10, None)));
    }

    // ── Subtest B: Partitioned Compaction + Vacuum Preserving Live Hive Indexes ──
    {
        let prev_vacuum_age = std::env::var("BSDB_VACUUM_MIN_FILE_AGE_SECS").ok();
        std::env::set_var("BSDB_VACUUM_MIN_FILE_AGE_SECS", "0");

        let res: Result<()> = async {
            let dir = tempdir()?;
            let uri = format!("file://{}", dir.path().display());

            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("category", DataType::Utf8, false),
            ]));
            let spec = PartitionSpec {
                spec_id: 1,
                fields: vec![PartitionField::new_single(
                    2,
                    Some(1000),
                    "category".to_string(),
                    "identity".to_string(),
                )],
            };
            let table = Table::create_partitioned_async(uri.clone(), schema.clone(), spec).await?;
            table.set_autocommit(false);
            table
                .add_index("id".to_string(), IndexAlgorithm::Bitmap)
                .await?;

            for step in 0..3 {
                let id_a = step * 10 + 1;
                let id_b = step * 10 + 2;
                let b = RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from(vec![id_a, id_b])),
                        Arc::new(StringArray::from(vec!["A", "B"])),
                    ],
                )?;
                table.write_async(vec![b]).await?;
                table.commit_async().await?;
            }
            table.wait_for_background_tasks_async().await?;

            let (_, pre_entries) = load_entries(&table).await?;
            assert_eq!(
                pre_entries.len(),
                6,
                "3 writes across 2 partitions = 6 segments"
            );
            let pre_paths: HashSet<String> =
                pre_entries.iter().map(|e| e.file_path.clone()).collect();

            table.rewrite_data_files_async(None).await?;
            let (_, post_entries) = load_entries(&table).await?;
            assert_eq!(
                post_entries.len(),
                2,
                "compaction should coalesce into 1 segment per partition"
            );
            for e in &post_entries {
                assert!(
                    e.file_path.starts_with("category=A/")
                        || e.file_path.starts_with("category=B/"),
                    "compacted entry must keep Hive path: {}",
                    e.file_path
                );
                assert!(
                    e.index_files.iter().any(|i| i.column_name == "id"),
                    "compacted entry `{}` must preserve index",
                    e.file_path
                );
            }

            let reaped = table.vacuum_async(1).await?;
            assert!(
                reaped > 0,
                "vacuum should reap pre-compaction segments/manifests"
            );

            for old_path in &pre_paths {
                assert!(
                    !dir.path().join(old_path).exists(),
                    "obsolete compacted segment `{old_path}` should be reaped by vacuum"
                );
            }
            for live in &post_entries {
                assert!(
                    dir.path().join(&live.file_path).exists(),
                    "live compacted segment `{}` missing after vacuum",
                    live.file_path
                );
                for idx in &live.index_files {
                    assert!(
                        dir.path().join(&idx.file_path).exists(),
                        "live index `{}` missing after vacuum",
                        idx.file_path
                    );
                }
            }

            let ids = ids_from_batches(&table.filter("category = 'B'").to_batches().await?);
            assert_eq!(ids, HashSet::from([2, 12, 22]));
            Ok(())
        }
        .await;

        match prev_vacuum_age {
            Some(v) => std::env::set_var("BSDB_VACUUM_MIN_FILE_AGE_SECS", v),
            None => std::env::remove_var("BSDB_VACUUM_MIN_FILE_AGE_SECS"),
        }
        res?;
    }

    // ── Subtest C: load_initial_schema Fallback on Hive Directory ──
    {
        let dir = tempdir()?;
        let uri = format!("file://{}", dir.path().display());

        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("category", DataType::Utf8, false),
            Field::new("score", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringArray::from(vec!["A", "A"])),
                Arc::new(Int64Array::from(vec![42, 99])),
            ],
        )?;

        let hive_dir = dir.path().join("category=A");
        std::fs::create_dir_all(&hive_dir)?;
        let pq_path = hive_dir.join("seg_fallback.parquet");
        let file = std::fs::File::create(&pq_path)?;
        let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema.clone(), None)?;
        writer.write(&batch)?;
        writer.close()?;

        // Commit a schema-less v1 manifest so TableBuilder::build_async falls back to load_initial_schema.
        let store = create_object_store(&uri)?;
        let manager = ManifestManager::new(store, "", &uri);
        manager
            .commit_manifest(Manifest::new(1, vec![], None))
            .await?;

        let reopened = Table::new_async(uri).await?;
        let recovered = reopened.arrow_schema();
        let field_names: Vec<&str> = recovered
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert_eq!(
            field_names,
            vec!["id", "category", "score"],
            "load_initial_schema must discover Parquet files inside Hive subdirectories"
        );
    }

    // ── Subtest D: memory:// URI Partitioned Write + Compactor Does Not Create CWD `memory:` ──
    {
        let tmp_cwd = tempdir()?;
        let orig_cwd = std::env::current_dir()?;
        std::env::set_current_dir(tmp_cwd.path())?;

        let mem_res: Result<()> = async {
            let unique_name = format!("part_idx_{}", uuid::Uuid::new_v4().simple());
            let uri = format!("memory://{unique_name}");

            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int32, false),
                Field::new("category", DataType::Utf8, false),
            ]));
            let spec = PartitionSpec {
                spec_id: 1,
                fields: vec![PartitionField::new_single(
                    2,
                    Some(1000),
                    "category".to_string(),
                    "identity".to_string(),
                )],
            };
            let table = Table::create_partitioned_async(uri.clone(), schema.clone(), spec).await?;
            table.set_autocommit(false);
            table.add_index("id".to_string(), IndexAlgorithm::Bitmap).await?;

            for round in 0..2 {
                let b = RecordBatch::try_new(
                    schema.clone(),
                    vec![
                        Arc::new(Int32Array::from(vec![round * 10 + 1, round * 10 + 2])),
                        Arc::new(StringArray::from(vec!["A", "B"])),
                    ],
                )?;
                table.write_async(vec![b]).await?;
                table.commit_async().await?;
            }
            table.wait_for_background_tasks_async().await?;

            let compactor = Compactor::new(&uri, Default::default())?;
            let compacted = compactor.rewrite_data_files().await?;
            assert!(
                compacted.is_some(),
                "memory:// Compactor should compact small partitioned segments without double-prefixing"
            );

            assert!(
                !tmp_cwd.path().join("memory:").exists(),
                "flush_internal_async created bogus CWD directory `{}`",
                tmp_cwd.path().join("memory:").display()
            );
            let files_in_cwd = walk_files(tmp_cwd.path());
            assert!(
                files_in_cwd.is_empty(),
                "memory:// operations must not pollute CWD: {files_in_cwd:?}"
            );
            Ok(())
        }
        .await;

        std::env::set_current_dir(&orig_cwd)?;
        mem_res?;
    }

    Ok(())
}
