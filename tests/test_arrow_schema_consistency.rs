// Copyright (c) 2026 Richard Albright. All rights reserved.

use hyperstreamdb::core::table::Table;
use hyperstreamdb::core::index::VectorValue;
use arrow::record_batch::RecordBatch;
use arrow::array::{Int32Array, FixedSizeListArray};
use arrow::datatypes::{DataType, Field, Schema, Float32Type};
use std::sync::Arc;
use tempfile::tempdir;

#[tokio::test]
async fn test_vector_search_schema_consistency() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    // Schema with Vector
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("embedding", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            3
        ), false),
    ]));

    // 1. Setup Table with explicit schema
    let table = Table::create_async(uri.clone(), schema.clone()).await?;
    
    // 2. Write Data
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int32Array::from(vec![1, 2])),
            Arc::new(FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                vec![
                    Some(vec![Some(0.1), Some(0.2), Some(0.3)]),
                    Some(vec![Some(0.4), Some(0.5), Some(0.6)]),
                ],
                3
            )),
        ]
    )?;
    
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    
    // 3. Test Normal Search Result Schema
    {
        let query_vec = vec![0.1, 0.2, 0.3];
        let hits = table.query()
            .vector_search("embedding", VectorValue::Float32(query_vec), 10)
            .to_batches().await?;
        
        assert!(!hits.is_empty(), "Should have results");
        let result_schema = hits[0].schema();
        
        // Assert schema contains distance
        assert!(result_schema.field_with_name("distance").is_ok(), "Schema should contain 'distance' column");
        assert_eq!(result_schema.field_with_name("distance").unwrap().data_type(), &DataType::Float32);
        
        // Check contents
        assert!(result_schema.field_with_name("id").is_ok());
        assert!(result_schema.field_with_name("embedding").is_ok());
    }

    // 4. Test EMPTY Search Result Schema (The Regression Case)
    {
        let query_vec = vec![0.9, 0.9, 0.9];
        let hits = table.query()
            .filter("id > 100") // No rows match
            .vector_search("embedding", VectorValue::Float32(query_vec), 10)
            .to_batches().await?;
        
        assert!(hits.is_empty(), "Should have no results");
    }
    
    // 5. Test Projection with Vector Search
    {
        let query_vec = vec![0.1, 0.2, 0.3];
        let hits = table.query()
            .select(vec!["id".to_string()]) // Project only ID
            .vector_search("embedding", VectorValue::Float32(query_vec), 1)
            .to_batches().await?;
        
        assert!(!hits.is_empty());
        let result_schema = hits[0].schema();
        
        assert!(result_schema.field_with_name("id").is_ok());
        assert!(result_schema.field_with_name("distance").is_ok(), "Distance should be present even if not explicitly selected");
        assert!(result_schema.field_with_name("embedding").is_err(), "Embedding should NOT be present if not selected");
    }

    Ok(())
}
