// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::core::reader::HybridReader;
use benostreamdb::core::search::KeywordSearchParams;
use benostreamdb::SegmentConfig;
use benostreamdb::Table;
use std::sync::Arc;
use tempfile::tempdir;

#[tokio::test]
async fn test_position_aware_inverted_and_phrase_search() -> anyhow::Result<()> {
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

    // 1. Verify standard keyword BM25 search
    let results = table
        .execute_keyword_search_as_scored(KeywordSearchParams {
            column: "content".to_string(),
            query: "quick brown".to_string(),
            ..Default::default()
        })
        .await?;

    assert!(!results.is_empty(), "Should find matches for 'quick brown'");
    // doc 0, 1, 2 should all have matches
    assert!(results.len() >= 3);

    // 2. Verify phrase search directly via reader
    let manifest = table.manifest().await?;
    let manifest_manager =
        benostreamdb::core::manifest::ManifestManager::new(table.store.clone(), "", &table.uri);
    let all_entries = manifest_manager.load_all_entries(&manifest).await?;
    assert_eq!(all_entries.len(), 1);

    let entry = &all_entries[0];
    let file_path_str = entry.file_path.clone();
    let segment_id = file_path_str
        .split('/')
        .next_back()
        .unwrap_or(&file_path_str)
        .strip_suffix(".parquet")
        .unwrap_or(&file_path_str);

    let config = SegmentConfig::new(&table.uri, segment_id)
        .with_parquet_path(entry.file_path.clone())
        .with_index_files(entry.index_files.clone())
        .with_record_count(entry.record_count as u64);

    let reader = HybridReader::new(config, table.store.clone(), &table.uri);

    // Exact phrase "quick brown fox":
    // Doc 0: "the (0) quick (1) brown (2) fox (3) ..." -> matches at pos 1, 2, 3!
    // Doc 1: "quick (0) brown (1) dogs ... brown (5) foxes (6)" -> no exact "quick brown fox"
    // Doc 2: "the lazy fox was neither quick nor brown" -> no "quick brown fox"
    let phrase_matches = reader
        .phrase_search_index("content", "quick brown fox", 10, 0, None)
        .await?;

    assert_eq!(
        phrase_matches.len(),
        1,
        "Only doc 0 has exact phrase 'quick brown fox'"
    );
    assert_eq!(phrase_matches[0].0, 0, "Doc 0 must match");

    // Phrase with slop=2: "quick fox"
    // Doc 0: quick (1), brown (2), fox (3) -> distance 2 -> matches with slop >= 1
    let slop_matches = reader
        .phrase_search_index("content", "quick fox", 10, 1, None)
        .await?;
    assert!(
        slop_matches.iter().any(|(rid, _)| *rid == 0),
        "Doc 0 should match 'quick fox' with slop=1"
    );

    Ok(())
}
