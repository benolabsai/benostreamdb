// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! `Table::reindex_inverted_column` upgrades a legacy 2-column (position-less)
//! inverted index to the 3-column position-aware format *in place*, without
//! rebuilding the segment's other indexes.

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::core::search::KeywordSearchParams;
use benostreamdb::Table;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::tempdir;

/// Locate the `{segment}.{col}.inv.parquet` sidecar in a table directory.
fn find_inv_file(dir: &Path, col: &str) -> PathBuf {
    let suffix = format!(".{}.inv.parquet", col);
    std::fs::read_dir(dir)
        .expect("read table dir")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.ends_with(&suffix))
                .unwrap_or(false)
        })
        .expect("inverted index sidecar")
}

/// Column names of a parquet file, in order.
fn inv_columns(path: &Path) -> Vec<String> {
    let f = std::fs::File::open(path).expect("open inv file");
    let builder =
        parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(f).expect("reader");
    builder
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

/// Rewrite a 3-column inverted index as the legacy 2-column layout, preserving
/// the `analyzer` footer metadata (as real legacy files do).
fn downgrade_to_two_columns(path: &Path) {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::arrow::ArrowWriter;

    let f = std::fs::File::open(path).expect("open inv file");
    let builder = ParquetRecordBatchReaderBuilder::try_new(f).expect("reader");
    let schema = builder.schema().clone();
    let analyzer_kv: Option<Vec<parquet::file::metadata::KeyValue>> = builder
        .metadata()
        .file_metadata()
        .key_value_metadata()
        .map(|kvs| {
            kvs.iter()
                .filter(|kv| kv.key == "analyzer")
                .cloned()
                .collect()
        });
    let batches: Vec<RecordBatch> = builder
        .build()
        .expect("build reader")
        .map(|b| b.expect("batch"))
        .collect();

    let two_col_schema = Arc::new(Schema::new(vec![
        schema.field(0).clone(),
        schema.field(1).clone(),
    ]));
    let out = std::fs::File::create(path).expect("create inv file");
    let props = parquet::file::properties::WriterProperties::builder()
        .set_key_value_metadata(analyzer_kv)
        .build();
    let mut writer =
        ArrowWriter::try_new(out, two_col_schema.clone(), Some(props)).expect("writer");
    for b in batches {
        let trimmed = RecordBatch::try_new(
            two_col_schema.clone(),
            vec![b.column(0).clone(), b.column(1).clone()],
        )
        .expect("trim batch");
        writer.write(&trimmed).expect("write batch");
    }
    writer.close().expect("close writer");
}

#[tokio::test]
async fn reindex_upgrades_legacy_two_column_index() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    let schema = Arc::new(Schema::new(vec![Field::new(
        "content",
        DataType::Utf8,
        false,
    )]));
    let table = Arc::new(Table::create_async(uri.clone(), schema.clone()).await?);

    table
        .add_index(
            "content".to_string(),
            IndexAlgorithm::Bm25 {
                tokenizer: "default".to_string(),
                k1: 1.2,
                b: 0.75,
            },
        )
        .await?;

    let docs = vec![
        "the quick brown fox jumps over the lazy dog",
        "quick brown dogs are faster than brown foxes",
        "the lazy fox was neither quick nor brown",
        "an unrelated document about databases and streaming data",
    ];
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(StringArray::from(docs))])?;
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    let inv_path = find_inv_file(dir.path(), "content");
    assert_eq!(
        inv_columns(&inv_path),
        vec!["key", "row_ids", "positions"],
        "freshly built index must be position-aware"
    );

    // Simulate a pre-positions index.
    downgrade_to_two_columns(&inv_path);
    assert_eq!(inv_columns(&inv_path), vec!["key", "row_ids"]);

    // Reindex: restores the 3-column layout in place.
    let rebuilt = table.reindex_inverted_column("content").await?;
    assert_eq!(rebuilt, 1, "one segment should be rebuilt");
    assert_eq!(
        inv_columns(&inv_path),
        vec!["key", "row_ids", "positions"],
        "reindex must restore the position-aware layout"
    );

    // Search still returns the same rows.
    let results = table
        .execute_keyword_search_as_scored(KeywordSearchParams {
            column: "content".to_string(),
            query: "quick brown".to_string(),
            ..Default::default()
        })
        .await?;
    assert!(
        results.len() >= 3,
        "expected >=3 matches for 'quick brown', got {}",
        results.len()
    );

    Ok(())
}
