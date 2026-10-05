// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
//
//! Indexing defaults and the `index_all` opt-in.
//!
//! Two properties must hold:
//!
//! 1. **Indexing is opt-in.** Constructing a table must NOT default to building
//!    indexes for every column — that caused large HNSW builds on every write.
//!    (`Table::new` used to default `index_all = true` while `new_async` did
//!    not; they now agree on `false`.)
//! 2. **`index_all` still works.** If a caller explicitly opts in — via
//!    `with_index_all(true)` or `index_all_columns()` — every eligible column is
//!    indexed even without `add_index`.

use std::sync::Arc;

use arrow::array::{FixedSizeListArray, Int32Array};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;

use benostreamdb::core::manifest::IndexAlgorithm;
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
    let vectors: Vec<Option<Vec<Option<f32>>>> =
        (0..n).map(|i| Some(vec![Some(i as f32); dim])).collect();
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

fn hnsw_tq8() -> IndexAlgorithm {
    IndexAlgorithm::HnswTq8 {
        metric: "l2".to_string(),
        complexity: 16,
        quality: 200,
    }
}

/// Count vector index artifacts. Indexes are packed into Puffin compound
/// bundles, so a `.puffin` file containing a vector blob counts as one.
fn count_vector_index_files(dir: &std::path::Path) -> usize {
    fn walk(p: &std::path::Path, n: &mut usize) {
        let Ok(rd) = std::fs::read_dir(p) else { return };
        for e in rd.flatten() {
            let path = e.path();
            if path.is_dir() {
                walk(&path, n);
            } else if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                if name.ends_with(".puffin") {
                    if let Ok(bytes) = std::fs::read(&path) {
                        if let Ok(reader) = benostreamdb::core::puffin::PuffinReader::new(
                            std::io::Cursor::new(bytes),
                        ) {
                            if reader
                                .footer()
                                .blobs
                                .iter()
                                .any(|b| b.r#type.starts_with("org.apache.iceberg.vector"))
                            {
                                *n += 1;
                            }
                        }
                    }
                } else if name.contains(".tq8.")
                    || name.contains(".hnsw.")
                    || name.contains(".centroids.")
                {
                    *n += 1;
                }
            }
        }
    }
    let mut n = 0;
    walk(dir, &mut n);
    n
}

async fn write_vector_rows(table: &Table, d: &std::path::Path) -> anyhow::Result<()> {
    table.set_autocommit(false);
    table.write_async(vec![vector_batch(64, 8)?]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;
    let _ = d;
    Ok(())
}

/// The default must be opt-out. The sync constructor drives its own runtime, so
/// it is exercised from a plain (non-async) test to avoid a nested `block_on`.
#[test]
fn sync_new_does_not_default_to_index_all() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());
    let sync = Table::new(uri)?;
    assert!(
        !sync.get_index_all(),
        "Table::new must not default to index_all"
    );
    Ok(())
}

#[tokio::test]
async fn new_async_does_not_default_to_index_all() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());
    let async_t = Table::new_async(uri).await?;
    assert!(
        !async_t.get_index_all(),
        "Table::new_async must not default to index_all"
    );
    Ok(())
}

/// `add_index` on an otherwise-default table builds the vector index.
#[tokio::test]
async fn add_index_builds_vector_index() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());
    let table = Table::new_async(uri).await?;
    assert!(!table.get_index_all());

    table.set_autocommit(false);
    table.add_index("embedding".to_string(), hnsw_tq8()).await?;
    table.write_async(vec![vector_batch(64, 8)?]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    assert!(
        count_vector_index_files(dir.path()) > 0,
        "add_index must produce vector index files"
    );
    Ok(())
}

/// Explicitly opting in with `with_index_all(true)` indexes the vector column
/// even though `add_index` was never called.
#[tokio::test]
async fn index_all_builds_vector_index_without_add_index() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());
    let table = Table::builder(uri)
        .with_index_all(true)
        .build_async()
        .await?;
    assert!(table.get_index_all(), "explicit opt-in must be honoured");

    write_vector_rows(&table, dir.path()).await?;

    assert!(
        count_vector_index_files(dir.path()) > 0,
        "index_all=true must build the vector index without add_index"
    );
    Ok(())
}

/// Default (no `add_index`, no `index_all`) writes rows but builds no index.
#[tokio::test]
async fn default_write_builds_no_vector_index() -> anyhow::Result<()> {
    let dir = tempfile::tempdir()?;
    let uri = format!("file://{}", dir.path().to_str().unwrap());
    let table = Table::new_async(uri).await?;
    write_vector_rows(&table, dir.path()).await?;

    assert_eq!(
        count_vector_index_files(dir.path()),
        0,
        "a default table must not build indexes without an explicit request"
    );
    Ok(())
}
