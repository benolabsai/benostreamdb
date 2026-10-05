// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Differential oracle for the JSON-path overlay index: **the index must never
//! change the answer.**
//!
//! The same `json_*` query runs against an indexed and an unindexed table with
//! identical data; the results must be identical. Because the overlay is
//! advisory and the filter is re-applied above the scan, the full scan is a
//! correctness oracle for the index.
//!
//! Also asserts the index is actually registered in the manifest under the
//! `json_path` category, so the planner path (not just the fallback scan) is
//! exercised.

use std::collections::HashSet;
use std::sync::Arc;

use arrow::array::{Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::manifest::{IndexAlgorithm, ManifestManager};
use benostreamdb::core::storage::create_object_store;
use benostreamdb::core::table::Table;
use tempfile::tempdir;

fn schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("payload", DataType::Utf8, false),
    ]))
}

fn batch() -> anyhow::Result<RecordBatch> {
    let ids: Vec<i32> = (0..12).collect();
    let payloads: Vec<String> = vec![
        r#"{"level":"error","user":{"id":1},"tags":["a","b"]}"#.to_string(),
        r#"{"level":"info","user":{"id":2},"tags":["b"]}"#.to_string(),
        r#"{"level":"error","user":{"id":3},"tags":["c"]}"#.to_string(),
        r#"{"level":"warn","user":{"id":4},"tags":["a"]}"#.to_string(),
        r#"{"level":"error","user":{"id":5},"tags":["b","c"]}"#.to_string(),
        r#"{"level":"info","user":{"id":6},"tags":[]}"#.to_string(),
        r#"{"level":"error","user":{"id":7},"tags":["a"]}"#.to_string(),
        r#"{"level":"debug","user":{"id":8},"tags":["b"]}"#.to_string(),
        r#"{"level":"error","user":{"id":9},"tags":["c"]}"#.to_string(),
        r#"{"level":"info","user":{"id":10},"tags":["a","b"]}"#.to_string(),
        r#"{"level":"error","user":{"id":11},"tags":["b"]}"#.to_string(),
        r#"{"level":"warn","user":{"id":12},"tags":["c"]}"#.to_string(),
    ];
    Ok(RecordBatch::try_new(
        schema(),
        vec![
            Arc::new(Int32Array::from(ids)),
            Arc::new(StringArray::from(payloads)),
        ],
    )?)
}

async fn build(uri: String, indexed: bool) -> anyhow::Result<Table> {
    let table = Table::new_async(uri).await?;
    table.set_autocommit(false);
    if indexed {
        table
            .add_index(
                "payload".to_string(),
                IndexAlgorithm::JsonPath {
                    paths: vec![
                        "$.level".to_string(),
                        "$.user.id".to_string(),
                        "$.tags".to_string(),
                    ],
                },
            )
            .await?;
    }
    table.write_async(vec![batch()?]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;
    Ok(table)
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

/// The `json_path` overlay must be registered in the manifest so the reader can
/// find it (a loose file with no `IndexFile` entry would be invisible).
#[tokio::test]
async fn json_path_index_is_registered_in_manifest() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());
    {
        let table = build(uri.clone(), true).await?;
        drop(table);
    }

    let store = create_object_store(&uri)?;
    let manager = ManifestManager::new(store, "", &uri);
    let (manifest, _) = manager.load_latest().await?;
    let entries = manager.load_all_entries(&manifest).await?;

    let has_json_path = entries.iter().any(|e| {
        e.index_files
            .iter()
            .any(|f| f.index_category == "json_path" && f.column_name == "payload")
    });
    assert!(
        has_json_path,
        "the json_path index must be registered in the manifest"
    );
    Ok(())
}

/// Every `json_*` predicate must return the same rows with and without the
/// index.
#[tokio::test]
async fn json_path_index_matches_full_scan() -> anyhow::Result<()> {
    let d1 = tempdir()?;
    let d2 = tempdir()?;
    let plain = build(format!("file://{}", d1.path().to_str().unwrap()), false).await?;
    let indexed = build(format!("file://{}", d2.path().to_str().unwrap()), true).await?;

    let filters = [
        r#"json_contains(payload, '{"level":"error"}')"#,
        r#"json_exists(payload, 'level')"#,
        r#"json_path_exists(payload, '$.user.id')"#,
        r#"json_extract_path_text(payload, 'level') = 'error'"#,
        r#"json_contains(payload, '{"tags":"b"}')"#,
    ];

    for filter in filters {
        let a = ids_of(&plain.filter(filter).to_batches().await?);
        let b = ids_of(&indexed.filter(filter).to_batches().await?);
        assert_eq!(
            a, b,
            "json_path index diverged from full scan for `{filter}`: scan={a:?} indexed={b:?}"
        );
        assert!(!a.is_empty(), "filter `{filter}` returned no rows");
    }
    Ok(())
}

/// A query on a path the index does not cover must fall back to a full scan and
/// still return the correct rows (an empty index bitmap would be a subset).
#[tokio::test]
async fn unindexed_path_falls_back_to_scan() -> anyhow::Result<()> {
    let d1 = tempdir()?;
    let d2 = tempdir()?;
    let plain = build(format!("file://{}", d1.path().to_str().unwrap()), false).await?;
    let indexed = build(format!("file://{}", d2.path().to_str().unwrap()), true).await?;

    // `$.user.id` is indexed, but `$.missing` is not.
    let filter = r#"json_path_exists(payload, '$.missing')"#;
    let a = ids_of(&plain.filter(filter).to_batches().await?);
    let b = ids_of(&indexed.filter(filter).to_batches().await?);
    assert_eq!(a, b, "unindexed path must fall back to a full scan");
    assert!(a.is_empty(), "no row has `$.missing`");
    Ok(())
}
