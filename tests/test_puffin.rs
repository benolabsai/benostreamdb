// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use benostreamdb::core::puffin::{PuffinReader, PuffinWriter};
use std::collections::HashMap;
use std::io::Cursor;

#[test]
fn test_puffin_write_read() -> Result<(), Box<dyn std::error::Error>> {
    let mut buffer = Cursor::new(Vec::new());

    let mut writer = PuffinWriter::new(&mut buffer).unwrap();
    writer.add_blob(
        "test-blob".to_string(),
        vec![1],
        100,
        1,
        b"Hello Puffin!",
        HashMap::new(),
    )?;

    writer.add_blob(
        "another-blob".to_string(),
        vec![2, 3],
        100,
        1,
        b"Binary Data",
        HashMap::new(),
    )?;

    writer.finish().unwrap();

    let bytes = buffer.into_inner();
    let mut reader = PuffinReader::new(Cursor::new(bytes)).unwrap();

    assert_eq!(reader.footer().blobs.len(), 2);

    let data1 = reader.read_blob(0).unwrap();
    assert_eq!(data1, b"Hello Puffin!");
    assert_eq!(reader.footer().blobs[0].r#type, "test-blob");
    assert_eq!(reader.footer().blobs[0].fields, vec![1]);

    let data2 = reader.read_blob(1).unwrap();
    assert_eq!(data2, b"Binary Data");
    assert_eq!(reader.footer().blobs[1].r#type, "another-blob");

    Ok(())
}

#[test]
fn test_puffin_compound_index_writer_reader() -> Result<(), Box<dyn std::error::Error>> {
    use benostreamdb::core::puffin::{
        PuffinIndexReader, PuffinIndexWriter, PUFFIN_BLOB_INDEX_ROARING, PUFFIN_BLOB_LEXICAL_BM25,
        PUFFIN_BLOB_VECTOR_HNSW_TQ8,
    };
    use sha2::{Digest, Sha256};

    let container_file = "seg_test_123.puffin".to_string();
    let snapshot_id = 42i64;
    let data_checksum =
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855".to_string();
    let record_count = 100_000i64;

    let mut index_writer = PuffinIndexWriter::new_in_memory(
        container_file.clone(),
        snapshot_id,
        data_checksum.clone(),
        record_count,
    )?;

    // 1. Pack Vector Index Blob
    let vector_payload = b"mock-hnsw-tq8-binary-graph-payload-1234567890";
    index_writer.add_index_blob(
        "vector",
        "hnsw_tq8",
        "embedding",
        PUFFIN_BLOB_VECTOR_HNSW_TQ8,
        vector_payload,
        vec![1],
        HashMap::new(),
    )?;

    // 2. Pack BM25 Lexical Index Blob
    let bm25_payload = b"mock-inverted-postings-parquet-bytes";
    index_writer.add_index_blob(
        "lexical",
        "bm25",
        "title",
        PUFFIN_BLOB_LEXICAL_BM25,
        bm25_payload,
        vec![2],
        HashMap::new(),
    )?;

    // 3. Pack Roaring Bitmap Scalar Index Blob
    let bitmap_payload = b"mock-roaring-bitmap-payload";
    index_writer.add_index_blob(
        "scalar",
        "bitmap",
        "status",
        PUFFIN_BLOB_INDEX_ROARING,
        bitmap_payload,
        vec![3],
        HashMap::new(),
    )?;

    // Finish writing and inspect IndexFile entries
    let (puffin_bytes, index_entries) = index_writer.finish_to_bytes()?;

    assert_eq!(index_entries.len(), 3);

    // Verify lineage & range metadata
    for (i, entry) in index_entries.iter().enumerate() {
        assert_eq!(entry.format_version, 2);
        assert_eq!(entry.file_path, container_file);
        assert_eq!(entry.source_snapshot_id, snapshot_id);
        assert_eq!(entry.source_data_checksum, data_checksum);
        assert_eq!(entry.source_record_count, record_count);
        assert!(entry.blob_offset.is_some());
        assert!(entry.blob_length.is_some());
        assert!(entry.index_checksum.is_some());

        // Byte-range extraction from Puffin container matches payload
        let offset = entry.blob_offset.unwrap() as usize;
        let length = entry.blob_length.unwrap() as usize;
        let slice = &puffin_bytes[offset..offset + length];

        let expected_payload: &[u8] = match i {
            0 => vector_payload,
            1 => bm25_payload,
            2 => bitmap_payload,
            _ => unreachable!(),
        };

        assert_eq!(slice, expected_payload);

        // Verify SHA-256 matches index_checksum
        let mut hasher = Sha256::new();
        hasher.update(expected_payload);
        let expected_hash = format!("{:x}", hasher.finalize());
        assert_eq!(entry.index_checksum.as_ref().unwrap(), &expected_hash);
    }

    // Verify Reader
    let mut reader = PuffinIndexReader::new(Cursor::new(puffin_bytes))?;
    assert_eq!(reader.blobs().len(), 3);

    let read_vector = reader
        .read_blob_by_type(PUFFIN_BLOB_VECTOR_HNSW_TQ8)?
        .unwrap();
    assert_eq!(read_vector, vector_payload);

    let read_bm25 = reader.read_blob_by_type(PUFFIN_BLOB_LEXICAL_BM25)?.unwrap();
    assert_eq!(read_bm25, bm25_payload);

    let read_bitmap = reader
        .read_blob_by_type(PUFFIN_BLOB_INDEX_ROARING)?
        .unwrap();
    assert_eq!(read_bitmap, bitmap_payload);

    Ok(())
}

#[tokio::test]
async fn test_puffin_hybrid_segment_writer_and_reader() -> Result<(), Box<dyn std::error::Error>> {
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use benostreamdb::core::reader::HybridReader;
    use benostreamdb::core::segment::HybridSegmentWriter;
    use benostreamdb::SegmentConfig;
    use std::sync::Arc;

    let tmp_dir = tempfile::tempdir()?;
    let tmp_dir_path = tmp_dir.path().to_str().unwrap().to_string();
    let segment_id = "puffin_seg_001";

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("title", DataType::Utf8, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec![
                "apache iceberg lakehouse",
                "puffin compound secondary index",
                "benostreamdb high performance",
                "vector hnsw and bm25 search",
                "open open lakehouse standard",
            ])),
        ],
    )?;

    let config = SegmentConfig::new(&tmp_dir_path, segment_id).with_index_all(true);

    let mut index_configs = std::collections::HashMap::new();
    index_configs.insert(
        "title".to_string(),
        benostreamdb::core::table::ColumnIndexConfig {
            enabled: true,
            algorithms: vec![benostreamdb::core::manifest::IndexAlgorithm::Bm25 {
                k1: 1.2,
                b: 0.75,
                tokenizer: "analyzer:english".to_string(),
            }],
            tokenizer: Some("analyzer:english".to_string()),
            device: None,
        },
    );

    let store: Arc<dyn object_store::ObjectStore> =
        Arc::new(object_store::local::LocalFileSystem::new());
    let writer = HybridSegmentWriter::new(config.clone())
        .with_store(store.clone())
        .with_index_configs(index_configs);

    writer.write_batch(&batch)?;
    writer.build_indexes(&batch, 0)?;
    writer.finish_indexing().await?;

    let base = format!("{}/{}", tmp_dir_path, segment_id);

    // 1. Data file should exist
    assert!(std::path::Path::new(&format!("{}.parquet", base)).exists());

    // 2. Puffin bundle container should exist
    let puffin_path = format!("{}.puffin", base);
    assert!(
        std::path::Path::new(&puffin_path).exists(),
        "Expected Puffin bundle container at {}",
        puffin_path
    );

    // 3. Loose inverted index should NOT exist (bundled into puffin)
    let loose_inv_path = format!("{}.title.inv.parquet", base);
    assert!(
        !std::path::Path::new(&loose_inv_path).exists(),
        "Loose inverted index should not exist when puffin bundling is enabled"
    );

    // 4. Inspect Manifest Entry
    let manifest_entry = writer.to_manifest_entry();
    assert!(!manifest_entry.index_files.is_empty());

    let puffin_entry = manifest_entry
        .index_files
        .iter()
        .find(|f| f.column_name == "title" && f.algorithm == "inverted")
        .expect("title inverted index should be recorded in manifest");

    assert_eq!(puffin_entry.format_version, 2);
    assert!(puffin_entry
        .file_path
        .ends_with(&format!("{}.puffin", segment_id)));
    assert!(puffin_entry.blob_offset.is_some());
    assert!(puffin_entry.blob_length.is_some());
    assert!(puffin_entry.index_checksum.is_some());

    // 5. Query via HybridReader with byte-range read from Puffin container
    let reader_config = SegmentConfig::new(&tmp_dir_path, segment_id)
        .with_parquet_path(format!("{}/{}.parquet", tmp_dir_path, segment_id))
        .with_index_files(manifest_entry.index_files.clone())
        .with_record_count(manifest_entry.record_count as u64);

    let reader = HybridReader::new(reader_config, store, &tmp_dir_path);
    let params = benostreamdb::core::index::bm25::Bm25Params::default();
    let results = reader
        .keyword_search_index("title", "lakehouse", 10, &params, None, None)
        .await?;

    assert!(!results.is_empty(), "Expected keyword search results");
    // Row 0 ("apache iceberg lakehouse") and Row 4 ("open open lakehouse standard") contain "lakehouse"
    let matched_row_ids: Vec<usize> = results.iter().map(|(r, _)| *r).collect();
    assert!(
        matched_row_ids.contains(&0) || matched_row_ids.contains(&4),
        "Expected row 0 or 4 to match 'lakehouse', got {:?}",
        matched_row_ids
    );

    Ok(())
}
