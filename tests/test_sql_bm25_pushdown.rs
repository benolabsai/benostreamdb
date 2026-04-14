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
async fn test_sql_bm25_pushdown() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    // 1. Setup Table with BM25 Index
    let schema = Arc::new(Schema::new(vec![
        Field::new("text", DataType::Utf8, false),
    ]));
    
    let table = Arc::new(Table::create_async(uri.clone(), schema.clone()).await?);
    
    // Add BM25 Index
    table.add_index("text".to_string(), IndexAlgorithm::Bm25 { 
        tokenizer: "default".to_string(),
        k1: 1.5,
        b: 0.75,
    }).await?;
    
    // 2. Insert Data
    let batch = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(StringArray::from(vec![
            "HyperStreamDB is a blazing fast vector database",
            "DataFusion provides the SQL engine",
            "BM25 is used for keyword search",
        ])),
    ])?;
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    table.wait_for_background_tasks_async().await?;

    // 3. Execute SQL Query with Equality Filter
    // This should trigger BM25 pushdown because 'text' has a BM25 index
    let session = HyperStreamSession::new(None);
    session.register_table("documents", table.clone())?;
    
    let (results, _) = session.sql("SELECT * FROM documents WHERE text = 'database'").await?;
    
    assert!(!results.is_empty(), "Should have found results via BM25 pushdown");
    let text_col = results[0].column(0).as_any().downcast_ref::<StringArray>().unwrap();
    assert!(text_col.value(0).contains("database"), "Result should contain 'database'");

    Ok(())
}
