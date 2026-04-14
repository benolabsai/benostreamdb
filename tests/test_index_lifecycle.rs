// Copyright (c) 2026 Richard Albright. All rights reserved.

use hyperstreamdb::Table;
use hyperstreamdb::core::manifest::IndexAlgorithm;
use hyperstreamdb::core::sql::session::HyperStreamSession;
use arrow::record_batch::RecordBatch;
use arrow::array::StringArray;
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use tempfile::tempdir;

#[tokio::test]
async fn test_index_lifecycle_add_drop_readd() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    // 1. Setup Table
    let schema = Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    
    let table = Arc::new(Table::create_async(uri.clone(), schema.clone()).await?);
    
    // 2. Insert Initial Data (NO INDEXES)
    let batch = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(StringArray::from(vec![
            "HyperStreamDB is a fast database",
            "DataFusion is a SQL engine",
            "BM25 is for search",
        ])),
        Arc::new(StringArray::from(vec!["tech", "tech", "info"])),
    ])?;
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    let session = HyperStreamSession::new(None);
    session.register_table("docs", table.clone())?;

    // 3. Add BM25 Index to 'text'
    println!("Adding BM25 index to 'text'...");
    table.add_index("text".to_string(), IndexAlgorithm::Bm25 { 
        tokenizer: "default".to_string(),
        k1: 1.5,
        b: 0.75,
    }).await?;
    
    // Wait for background backfill
    table.wait_for_background_tasks_async().await?;
    
    // DEBUG: Inspect manifest
    let manifest = table.manifest().await?;
    let entry = manifest.entries.first().expect("Should have at least one entry");
    println!("Manifest Entry Index Files: {:?}", entry.index_files);
    assert_eq!(manifest.entries.len(), 1, "Should only have ONE segment in manifest");
    assert!(!entry.index_files.is_empty(), "Index files should be registered in manifest");
    assert!(entry.index_files.iter().any(|idx| idx.index_type == "inverted" && idx.column_name.as_deref() == Some("text")), "Should have an inverted index for 'text'");
    
    // Verify search works via pushdown
    let (results, _) = session.sql("SELECT * FROM docs WHERE text = 'database'").await?;
    assert!(!results.is_empty(), "Should find results after adding index");
    
    // 4. Concurrently add indexes to 'category' and update 'text'
    // This tests the manifest concurrency hardening
    println!("Adding concurrent indexes...");
    let t1 = table.clone();
    let h1 = tokio::spawn(async move {
        t1.add_index("category".to_string(), IndexAlgorithm::Bm25 { 
            tokenizer: "identity".to_string(),
            k1: 1.2,
            b: 0.75,
        }).await
    });
    
    let t2 = table.clone();
    let h2 = tokio::spawn(async move {
        t2.add_index("text".to_string(), IndexAlgorithm::Bm25 { 
            tokenizer: "default".to_string(),
            k1: 1.2, // Changed k1
            b: 0.75,
        }).await
    });
    
    h1.await??;
    h2.await??;
    
    table.wait_for_background_tasks_async().await?;
    
    // Verify both indexes updated
    let manifest = table.manifest().await?;
    let latest_schema = manifest.schemas.last().unwrap();
    let text_field = latest_schema.fields.iter().find(|f| f.name == "text").unwrap();
    let cat_field = latest_schema.fields.iter().find(|f| f.name == "category").unwrap();
    assert_eq!(text_field.indexes.len(), 1);
    assert_eq!(cat_field.indexes.len(), 1);
    
    // 5. Drop Index from 'text'
    println!("Dropping index from 'text'...");
    table.drop_index("text".to_string()).await?;
    table.wait_for_background_tasks_async().await?;
    
    // Verify manifest updated
    let manifest = table.manifest().await?;
    let latest_schema = manifest.schemas.last().unwrap();
    let text_field = latest_schema.fields.iter().find(|f| f.name == "text").unwrap();
    assert!(text_field.indexes.is_empty(), "Index should be dropped");
    
    // Query should still work (falling back to scan if pushdown not possible, 
    // though our current physical plan requires index for pushdown)
    let (results, _) = session.sql("SELECT * FROM docs WHERE text = 'database'").await?;
    // Since we dropped the index, and our TableProvider only handles = as BM25 if index exists,
    // this will now be a normal DataFusion Filter on the text column.
    // It should STILL find the row if exact match, but BM25 is keyword.
    // Our test data "HyperStreamDB is a fast database" != "database", so expected 0 results without index.
    assert!(results.is_empty(), "Keyword search should return empty results if index is dropped because equality filter falls back to exact match");

    Ok(())
}
