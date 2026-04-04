// Copyright (c) 2026 Richard Albright. All rights reserved.

use hyperstreamdb::core::table::Table;
use hyperstreamdb::core::manifest::{PartitionSpec, PartitionField};
use arrow::record_batch::RecordBatch;
use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use tempfile::tempdir;
use anyhow::Result;

#[tokio::test]
async fn test_multi_column_null_partitioning() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    // 1. Define schema
    let schema = Arc::new(Schema::new(vec![
        Field::new("year", DataType::Int64, true),
        Field::new("month", DataType::Int64, true),
        Field::new("category", DataType::Utf8, true),
        Field::new("value", DataType::Int64, false),
    ]));

    // 2. Define Partition Spec: (year, category)
    let spec = PartitionSpec {
        spec_id: 1,
        fields: vec![
            PartitionField { source_ids: vec![1], source_id: Some(1), field_id: None, name: "year".to_string(), transform: "identity".to_string() },
            PartitionField { source_ids: vec![3], source_id: Some(3), field_id: None, name: "category".to_string(), transform: "identity".to_string() },
        ],
    };

    // 3. Create Partitioned Table
    let table = Table::create_partitioned_async(uri.clone(), schema.clone(), spec).await?;

    // 4. Ingest Data with mixing values (including NULLs)
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(arrow::array::Int64Array::from(vec![Some(2022), Some(2022), Some(2023), None])),
            Arc::new(arrow::array::Int64Array::from(vec![1, 2, 1, 1])),
            Arc::new(StringArray::from(vec![Some("A"), Some("B"), Some("A"), Some("C")])),
            Arc::new(arrow::array::Int64Array::from(vec![10, 20, 30, 40])),
        ]
    )?;

    table.write_async(vec![batch]).await?;
    table.flush_async().await?;
    table.commit_async().await?;

    // 5. Verify physical sharding in manifest
    let entries = table.get_snapshot_segments().await?;
    // We expect 4 segments (2022/A, 2022/B, 2023/A, NULL/C)
    // Note: If they happen to fall in same batch, they are sharded in flush_async.
    assert_eq!(entries.len(), 4, "Should have 4 partitioned segments");

    // 6. Test Pruning: Query by year=2022
    let filter_year = hyperstreamdb::core::planner::QueryFilter {
        column: "year".to_string(),
        min: Some(serde_json::json!(2022)),
        min_inclusive: true,
        max: Some(serde_json::json!(2022)),
        max_inclusive: true,
        values: None,
        negated: false,
    };
    
    let results_year = table.read_filter_async(vec![filter_year], None, None).await?;
    let total_rows_year: usize = results_year.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows_year, 2, "year=2022 should return 2 rows");

    // 7. Test Pruning: Query by category=A
    let filter_cat = hyperstreamdb::core::planner::QueryFilter {
        column: "category".to_string(),
        min: Some(serde_json::json!("A")),
        min_inclusive: true,
        max: Some(serde_json::json!("A")),
        max_inclusive: true,
        values: None,
        negated: false,
    };
    let results_cat = table.read_filter_async(vec![filter_cat], None, None).await?;
    let total_rows_cat: usize = results_cat.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows_cat, 2, "category=A should return 2 rows (one in 2022, one in 2023)");

    // 8. Test NULL Pruning (Implicitly)
    let results_all = table.read_filter_async(vec![], None, None).await?;
    let total_rows_all: usize = results_all.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows_all, 4, "Should read all 4 rows including NULL partition");

    Ok(())
}

#[tokio::test]
async fn test_compaction_preserves_partitions() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let schema = Arc::new(Schema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("value", DataType::Int64, false),
    ]));

    let spec = PartitionSpec {
        spec_id: 1,
        fields: vec![
            PartitionField { source_ids: vec![1], source_id: Some(1), field_id: None, name: "category".to_string(), transform: "identity".to_string() },
        ],
    };

    let table = Table::create_partitioned_async(uri.clone(), schema.clone(), spec).await?;

    // Write 2 segments for the same partition 'A'
    for _ in 0..2 {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["A"])),
                Arc::new(arrow::array::Int64Array::from(vec![1])),
            ]
        )?;
        table.write_async(vec![batch]).await?;
        table.flush_async().await?;
        table.commit_async().await?;
    }

    // Verify we have 2 segments
    let entries_pre = table.get_snapshot_segments().await?;
    assert_eq!(entries_pre.len(), 2);

    // Trigger Compaction
    table.rewrite_data_files_async(None).await?;

    // Verify we have 1 segment now, and it STILL HAS partition_values = {category: A}
    let entries_post = table.get_snapshot_segments().await?;
    assert_eq!(entries_post.len(), 1, "Compaction should merge segments");
    
    let entry = &entries_post[0];
    assert!(entry.partition_values.contains_key("category"), "Compacted entry must have partition key");
    assert_eq!(entry.partition_values.get("category").unwrap(), &serde_json::json!("A"));

    Ok(())
}
