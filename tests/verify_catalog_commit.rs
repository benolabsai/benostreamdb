use hyperstreamdb::core::catalog::Catalog;
use hyperstreamdb::core::catalog::rest::RestCatalogClient;
use hyperstreamdb::Table;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use tokio::time::{sleep, Duration};
use tempfile::tempdir;
use std::process::Command;

#[tokio::test]
async fn test_rest_catalog_commit_flow() -> anyhow::Result<()> {
    let tmp = tempdir()?;
    let warehouse_path = tmp.path().join("warehouse");
    std::fs::create_dir_all(&warehouse_path)?;
    let warehouse_uri = format!("file://{}", warehouse_path.display());
    
    // 1. Start REST server in background (using pre-built binary)
    let port = 8183;
    let exe_path = std::env::current_dir()?.join("target/debug/iceberg_rest");
    if !exe_path.exists() {
        return Err(anyhow::anyhow!("iceberg_rest binary not found at {:?}. Please run 'cargo build --bin iceberg_rest' first.", exe_path));
    }

    let mut server_child = Command::new(exe_path)
        .env("HYPERSTREAM_STORAGE_URI", &warehouse_uri)
        .env("PORT", port.to_string())
        .spawn()?;
    
    // Wait for server to start
    let mut success = false;
    for _ in 0..20 {
        if std::net::TcpStream::connect(format!("localhost:{}", port)).is_ok() {
            success = true;
            break;
        }
        sleep(Duration::from_secs(1)).await;
    }
    assert!(success, "Server failed to start on port {}", port);
    
    let base_url = format!("http://localhost:{}", port);
    let client = RestCatalogClient::new(base_url.clone(), Some("hdb".to_string()));
    
    let namespace = "test_ns";
    let table_name = "test_table";
    
    // 2. Create Table via Catalog
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    
    let table_location = format!("{}/{}/{}", warehouse_uri, namespace, table_name);
    client.create_table(namespace, table_name, schema.clone(), Some(&table_location)).await?;
    
    // 3. Load Table via Catalog
    let rest_uri = format!("{}/v1/hdb/namespaces/{}/tables/{}", base_url, namespace, table_name);
    let table = Table::new_async(rest_uri).await?;
    
    // 4. Write Data and Commit
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(arrow::array::Int32Array::from(vec![1, 2, 3])),
            Arc::new(arrow::array::StringArray::from(vec!["A", "B", "C"])),
        ],
    )?;
    
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    
    // 5. Verify Metadata Persistence
    // Check if v1.metadata.json exists
    let meta_v1 = warehouse_path.join(namespace).join(table_name).join("metadata/v1.metadata.json");
    assert!(meta_v1.exists(), "v1.metadata.json should exist");
    
    // 6. Verify Catalog reflects the update
    let updated_metadata = client.load_table(namespace, table_name).await?;
    assert!(updated_metadata.current_snapshot_id.is_some(), "Snapshot should be registered in catalog");
    assert!(updated_metadata.current_snapshot_id.unwrap() > 0, "Snapshot ID should be positive");
    
    // Cleanup
    let _ = server_child.kill();
    
    Ok(())
}
