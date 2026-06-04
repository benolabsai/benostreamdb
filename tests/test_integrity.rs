use anyhow::Result;
use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use tempfile::tempdir;

#[tokio::test]
async fn test_integrity_validation() -> Result<()> {
    let temp_dir = tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

    let table = hyperstreamdb::Table::new_async(uri.clone()).await?;

    // 1. Write some data
    let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
    )?;

    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    // 2. Verify integrity (should succeed)
    let initial_verify = table.verify_integrity_async().await;
    assert!(
        initial_verify.is_ok(),
        "Data integrity should be valid after normal write, but got error: {:?}",
        initial_verify.err().unwrap()
    );

    // 3. Tamper with the data file
    let segments = table.get_snapshot_segments().await?;
    assert_eq!(segments.len(), 1);

    let mut full_path = std::path::PathBuf::from(uri.strip_prefix("file://").unwrap());
    full_path.push(
        segments[0]
            .file_path
            .strip_prefix("file://")
            .unwrap_or(&segments[0].file_path),
    );

    let mut file = std::fs::OpenOptions::new().write(true).open(&full_path)?;
    use std::io::Write;
    file.write_all(b"corrupted_data_padding")?;
    file.sync_all()?;

    // 4. Verify integrity (should fail)
    let result = table.verify_integrity_async().await;
    assert!(
        result.is_err(),
        "Data integrity should fail after tampering"
    );

    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("Data integrity validation failed"),
        "Unexpected error message: {}",
        err_msg
    );

    Ok(())
}
