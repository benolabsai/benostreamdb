// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! `Table::reindex_inverted_column` upgrades a legacy 2-column (position-less)
//! inverted index to the 3-column position-aware format *in place*, without
//! rebuilding the segment's other indexes.
//!
//! Indexes are stored inside Puffin compound bundles, so the test manipulates
//! the inverted blob within the `.puffin` container.

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::core::puffin::{PuffinReader, PuffinWriter, PUFFIN_BLOB_LEXICAL_BM25};
use benostreamdb::core::search::KeywordSearchParams;
use benostreamdb::Table;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tempfile::tempdir;

/// Locate the `{segment}.puffin` compound bundle in a table directory.
fn find_puffin_file(dir: &Path) -> PathBuf {
    std::fs::read_dir(dir)
        .expect("read table dir")
        .filter_map(Result::ok)
        .map(|e| e.path())
        .find(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.ends_with(".puffin"))
                .unwrap_or(false)
        })
        .expect("puffin index bundle")
}

/// Read a blob of `blob_type` out of a Puffin file.
fn read_puffin_blob(path: &Path, blob_type: &str) -> Vec<u8> {
    let bytes = std::fs::read(path).expect("read puffin");
    let mut reader = PuffinReader::new(Cursor::new(bytes)).expect("puffin reader");
    let idx = reader
        .footer()
        .blobs
        .iter()
        .position(|b| b.r#type == blob_type)
        .expect("blob present");
    reader.read_blob(idx).expect("read blob")
}

/// Rewrite a single blob of a Puffin file, preserving all other blobs and their
/// metadata (including the `analyzer` property).
fn rewrite_puffin_blob(path: &Path, blob_type: &str, new_data: Vec<u8>) {
    let bytes = std::fs::read(path).expect("read puffin");
    let mut reader = PuffinReader::new(Cursor::new(bytes)).expect("puffin reader");
    let blobs = reader.footer().blobs.clone();

    let mut writer = PuffinWriter::new(Cursor::new(Vec::new())).expect("puffin writer");
    for (i, blob) in blobs.iter().enumerate() {
        let data = if blob.r#type == blob_type {
            new_data.clone()
        } else {
            reader.read_blob(i).expect("read blob")
        };
        writer
            .add_blob(
                blob.r#type.clone(),
                blob.fields.clone(),
                blob.snapshot_id,
                blob.sequence_number,
                &data,
                blob.properties.clone(),
            )
            .expect("add blob");
    }
    let cursor = writer.finish().expect("finish puffin");
    std::fs::write(path, cursor.into_inner()).expect("write puffin");
}

/// Column names of a Parquet byte buffer, in order.
fn inv_columns_from_bytes(bytes: &[u8]) -> Vec<String> {
    let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
        bytes::Bytes::copy_from_slice(bytes),
    )
    .expect("reader");
    builder
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect()
}

/// Rewrite a 3-column inverted index as the legacy 2-column layout, preserving
/// the `analyzer` footer metadata (as real legacy files do).
fn downgrade_to_two_columns(bytes: &[u8]) -> Vec<u8> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::arrow::ArrowWriter;

    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::copy_from_slice(bytes))
        .expect("reader");
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
        .expect("build")
        .map(|r| r.expect("batch"))
        .collect();

    let two_col_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Utf8, false),
        Field::new(
            "row_ids",
            DataType::List(Arc::new(Field::new("item", DataType::UInt32, true))),
            false,
        ),
    ]));

    let mut out = Vec::new();
    let props = parquet::file::properties::WriterProperties::builder()
        .set_key_value_metadata(analyzer_kv)
        .build();
    let mut writer =
        ArrowWriter::try_new(&mut out, two_col_schema.clone(), Some(props)).expect("writer");
    for b in batches {
        let trimmed = RecordBatch::try_new(
            two_col_schema.clone(),
            vec![b.column(0).clone(), b.column(1).clone()],
        )
        .expect("trim batch");
        writer.write(&trimmed).expect("write batch");
    }
    writer.close().expect("close writer");
    out
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

    let puffin_path = find_puffin_file(dir.path());
    let inv_bytes = read_puffin_blob(&puffin_path, PUFFIN_BLOB_LEXICAL_BM25);
    assert_eq!(
        inv_columns_from_bytes(&inv_bytes),
        vec!["key", "row_ids", "positions"],
        "freshly built index must be position-aware"
    );

    // Simulate a pre-positions index by rewriting the blob in the bundle.
    let downgraded = downgrade_to_two_columns(&inv_bytes);
    rewrite_puffin_blob(&puffin_path, PUFFIN_BLOB_LEXICAL_BM25, downgraded);
    assert_eq!(
        inv_columns_from_bytes(&read_puffin_blob(&puffin_path, PUFFIN_BLOB_LEXICAL_BM25)),
        vec!["key", "row_ids"]
    );

    // Reindex: restores the 3-column layout in place.
    let rebuilt = table.reindex_inverted_column("content").await?;
    assert_eq!(rebuilt, 1, "one segment should be rebuilt");
    assert_eq!(
        inv_columns_from_bytes(&read_puffin_blob(&puffin_path, PUFFIN_BLOB_LEXICAL_BM25)),
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
