// Example: Using catalogs in native Rust
// This demonstrates that all catalog implementations work natively in Rust

use hyperstreamdb::{Catalog, CatalogType, create_catalog};
use std::collections::HashMap;
use arrow::datatypes::{Schema, Field, DataType};
use std::sync::Arc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Example 1: Nessie Catalog
    let mut nessie_config = HashMap::new();
    nessie_config.insert("url".to_string(), "http://localhost:19120".to_string());
    
    let nessie = create_catalog(CatalogType::Nessie, nessie_config)?;
    
    // Create table
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
    ]));
    nessie.create_table("main", "my_table", schema.clone(), Some("s3://bucket/table")).await?;
    
    // Load table
    let metadata = nessie.load_table("main", "my_table").await?;
    println!("Table location: {}", metadata.location);
    
    // Example 2: REST Catalog
    let mut rest_config = HashMap::new();
    rest_config.insert("url".to_string(), "http://localhost:8181".to_string());
    rest_config.insert("prefix".to_string(), "warehouse".to_string());
    
    let rest = create_catalog(CatalogType::Rest, rest_config)?;
    
    // Same API!
    rest.create_table("db", "table", schema.clone(), Some("s3://bucket/table")).await?;
    let metadata = rest.load_table("db", "table").await?;
    println!("Table location: {}", metadata.location);
    
    // Example 3: Direct usage (without factory)
    // Note: RestCatalogClient is likely in core::catalog::rest
    use hyperstreamdb::core::catalog::rest::RestCatalogClient;
    
    let rest_client = RestCatalogClient::new(
        "http://localhost:8181".to_string(),
        Some("warehouse".to_string())
    );
    
    rest_client.create_table("db", "table2", schema, Some("s3://bucket/table2")).await?;
    
    Ok(())
}
