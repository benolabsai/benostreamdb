use anyhow::Result;
use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use hyperstreamdb::core::wal::{WalConfig, WriteAheadLog};
use std::sync::Arc;
use tempfile::tempdir;

fn create_test_batch(start: i32, count: i32) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));
    let ids = Int32Array::from((start..start + count).collect::<Vec<i32>>());
    RecordBatch::try_new(schema, vec![Arc::new(ids)]).unwrap()
}

#[tokio::test]
async fn test_wal_compaction_deletes_stale_files() -> Result<()> {
    let temp_dir = tempdir()?;
    let wal_dir = temp_dir.path().join("wal_stale_test");
    std::fs::create_dir_all(&wal_dir)?;

    // 1. Create multiple WAL files by creating multiple instances
    // Instance 1
    {
        let mut wal1 = WriteAheadLog::new(&wal_dir);
        wal1.append(&create_test_batch(0, 10))?;
        // Dropping wal1 finishes the stream
    }

    // Instance 2
    {
        let mut wal2 = WriteAheadLog::new(&wal_dir);
        wal2.append(&create_test_batch(10, 10))?;
    }

    // Check that we have 2 .arrow files
    let arrow_files: Vec<_> = std::fs::read_dir(&wal_dir)?
        .map(|res| res.unwrap().path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("arrow"))
        .collect();
    assert_eq!(
        arrow_files.len(),
        2,
        "Should have 2 log files before compaction"
    );

    // 2. Perform compaction using Instance 3
    let wal3 = WriteAheadLog::new(&wal_dir);
    // Force compaction by setting a very low threshold
    let config = WalConfig {
        compact_threshold_mb: 0,
        ..Default::default()
    };
    let mut wal3 = wal3.with_config(config);

    assert!(wal3.should_compact()?);
    wal3.compact()?;

    // 3. Verify that only ONE .arrow file remains
    let arrow_files_after: Vec<_> = std::fs::read_dir(&wal_dir)?
        .map(|res| res.unwrap().path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("arrow"))
        .collect();

    assert_eq!(
        arrow_files_after.len(),
        1,
        "Should have exactly 1 log file after compaction"
    );

    // 4. Verify data integrity (total 20 rows)
    let (batches, _) = wal3.replay()?;
    let total_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 20,
        "Should have all data from both original files"
    );

    Ok(())
}
