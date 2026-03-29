// Copyright (c) 2026 Richard Albright. All rights reserved.

use hyperstreamdb::Table;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::array::{Int32Array, StringArray};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let uri = std::env::var("HYPERSTREAM_STORAGE_URI").unwrap_or_else(|_| "file:///tmp/test_table".to_string());
    
    // Clean up previous run
    let _ = std::fs::remove_dir_all("/tmp/test_table");
    
    println!("Creating table at {}", uri);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    
    let table = Table::create_async(uri.to_string(), schema.clone()).await?;
    
    println!("Inserting data...");
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])),
            Arc::new(StringArray::from(vec!["Alice", "Bob", "Charlie"])),
        ],
    )?;
    
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    
    println!("Data seeded.");
    Ok(())
}
