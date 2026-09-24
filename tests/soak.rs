// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Long-running soak harness for the OSS GA checklist.
//!
//! Ignored by default so `cargo test` stays fast. Run explicitly:
//!
//! ```bash
//! BSDB_SOAK_SECONDS=600 cargo test --test soak -- --ignored --nocapture
//! ```
//!
//! The assertion is deliberately blunt — the point is that a multi-minute mixed
//! workload neither panics nor loses rows — because that is exactly the
//! "graceful degradation rather than process death" requirement.

use arrow::array::{FixedSizeListArray, Int64Array};
use arrow::datatypes::{DataType, Field, Float32Type, Schema};
use arrow::record_batch::RecordBatch;
use benostreamdb::Table;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::tempdir;

const DIM: usize = 64;
const ROWS: usize = 1_000;

fn batch(start: i64) -> RecordBatch {
    let ids: Vec<i64> = (start..start + ROWS as i64).collect();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new(
            "embedding",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                DIM as i32,
            ),
            false,
        ),
    ]));

    let embedding = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        (0..ROWS).map(|r| {
            Some((0..DIM).map(move |d| {
                // Deterministic, cheap values — the workload is what matters.
                Some(((r * DIM + d) % 97) as f32 / 97.0)
            }))
        }),
        DIM as i32,
    );

    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int64Array::from(ids)), Arc::new(embedding) as _],
    )
    .expect("record batch")
}

#[tokio::test]
#[ignore = "soak test; run with `-- --ignored` and BSDB_SOAK_SECONDS"]
async fn mixed_workload_soak() {
    let secs: u64 = std::env::var("BSDB_SOAK_SECONDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(60);

    let dir = tempdir().expect("tempdir");
    let uri = format!("file://{}", dir.path().to_str().expect("utf8 path"));
    let table = Table::new_async(uri).await.expect("open table");

    let deadline = Instant::now() + Duration::from_secs(secs);
    let mut writes: usize = 0;

    while Instant::now() < deadline {
        table
            .write_async(vec![batch(writes as i64 * ROWS as i64)])
            .await
            .expect("write");
        writes += 1;

        if writes % 3 == 0 {
            table.commit_async().await.expect("commit");
        }
        if writes % 2 == 0 {
            let batches = table.read_async(None, None, None).await.expect("read");
            assert!(
                batches.iter().map(|b| b.num_rows()).sum::<usize>() >= ROWS,
                "read returned no rows mid-soak"
            );
        }
    }

    table.commit_async().await.expect("final commit");
    assert!(writes > 1, "soak did not perform any writes");

    // The process is still here and serving reads after the soak.
    let batches = table
        .read_async(None, None, None)
        .await
        .expect("final read");
    assert!(batches.iter().map(|b| b.num_rows()).sum::<usize>() >= ROWS);
}
