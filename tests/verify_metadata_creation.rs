use hyperstreamdb::core::table::Table;
use arrow::datatypes::{Field, DataType, Schema};
use std::sync::Arc;
use object_store::path::Path;

#[tokio::test]
async fn test_table_metadata_creation() -> anyhow::Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
    
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    
    // Create table
    let table = Table::create_async(uri.clone(), schema).await?;
    
    // 1. Verify legacy manifest v1.json exists (in base dir or metadata/?)
    // Actually our ManifestManager uses "manifest.json" or "vX.json"
    // Let's check TableMetadata v1.metadata.json
    let metadata_path = Path::from("metadata/v1.metadata.json");
    let res = table.store.get(&metadata_path).await?;
    let bytes = res.bytes().await?;
    
    let metadata: serde_json::Value = serde_json::from_slice(&bytes)?;
    assert_eq!(metadata["format-version"], 2);
    assert_eq!(metadata["location"], uri);
    
    // 2. Verify version-hint.text
    let hint_path = Path::from("metadata/version-hint.text");
    let res = table.store.get(&hint_path).await?;
    let bytes = res.bytes().await?;
    let version_str = String::from_utf8(bytes.to_vec())?;
    assert_eq!(version_str.trim(), "1");
    
    println!("Successfully verified metadata creation at {}", uri);
    Ok(())
}
