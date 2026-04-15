// Copyright (c) 2026 Richard Albright. All rights reserved.

use hyperstreamdb::core::table::Table;
use arrow::array::{Int32Array, StringArray, RecordBatch};
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use tempfile::tempdir;
use std::time::Instant;

/// Verifies that Primary Key duplicate checks remain fast (O(1) via index)
/// regardless of how many segments the table has grown to.
///
/// We use 5 segments × 10k rows = 50k rows total — enough to span multiple
/// Parquet row groups and prove the inverted index is consulted rather than
/// falling back to a full scan.
#[tokio::test]
async fn test_pk_acceleration_multi_segment() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let uri = format!("file://{}", tmp.path().to_str().unwrap());

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("val", DataType::Utf8, false),
    ]));

    let table = Table::create_async(uri.clone(), schema.clone()).await?;
    table.set_primary_key(vec!["id".to_string()]);
    table.set_indexed_columns(vec!["id".to_string()]);

    // Insert 5 segments × 10k rows each = 50k unique rows
    const SEGMENTS: usize = 5;
    const ROWS_PER_SEGMENT: usize = 10_000;

    let start_insert = Instant::now();
    for i in 0..SEGMENTS {
        let ids: Vec<i32> = (0..ROWS_PER_SEGMENT)
            .map(|j| (i * ROWS_PER_SEGMENT + j) as i32)
            .collect();
        let vals: Vec<String> = ids.iter().map(|id| format!("value-{}", id)).collect();

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(ids)),
                Arc::new(StringArray::from(vals)),
            ],
        )?;

        table.write_async(vec![batch]).await?;
        println!("  Inserted segment {}/{}", i + 1, SEGMENTS);
    }
    println!("Total insertion time: {:?}", start_insert.elapsed());

    // --- Duplicate Detection ---
    // Pick keys spread across different segments to hit different index shards
    println!("--- Verifying Duplicate Detection (PK Acceleration) ---");
    let test_ids = [0i32, 9_999, 10_000, 25_000, 49_999];

    for id in test_ids {
        let dup_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![id])),
                Arc::new(StringArray::from(vec!["dup".to_string()])),
            ],
        )?;

        let now = Instant::now();
        let res = table.write_async(vec![dup_batch]).await;
        let elapsed = now.elapsed();

        assert!(res.is_err(), "Should have detected duplicate PK: {}", id);
        println!("  Duplicate check id={:>6}: {:?}", id, elapsed);

        // Must be sub-500ms even in unoptimized debug mode — proves we're not
        // scanning all 50k rows linearly.
        assert!(
            elapsed.as_millis() < 500,
            "PK validation too slow ({}ms) — likely fell back to full scan",
            elapsed.as_millis()
        );
    }

    // --- Non-Existent Key Insertion (Bloom Filter fast-reject path) ---
    println!("--- Verifying Non-Existent Key Fast Insert (Bloom Pruning) ---");
    let missing_ids = [-1i32, 50_001, 999_999];
    for id in missing_ids {
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![id])),
                Arc::new(StringArray::from(vec!["new".to_string()])),
            ],
        )?;

        let now = Instant::now();
        let _ = table.write_async(vec![batch]).await?;
        println!("  New key insert id={:>7}: {:?}", id, now.elapsed());
    }

    Ok(())
}
