// Copyright (c) 2026 Richard Albright. All rights reserved.

use hyperstreamdb::core::table::Table;
use arrow::record_batch::RecordBatch;
use arrow::array::{Int32Array, StringArray, FixedSizeListArray};
use arrow::datatypes::{DataType, Field, Schema, Float32Type};
use std::sync::Arc;
use tempfile::tempdir;

#[tokio::test]
async fn test_core_ingestion_buffered_and_wal() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    // 1. Setup Table
    let table = Table::new_async(uri.clone()).await?;
    table.set_autocommit(false);
    
    // Schema with Vector for Indexing
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("embedding", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            3
        ), false),
    ]));
    
    // 2. Write Data (100 rows)
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
    
    table.write_async(vec![batch.clone()]).await?;
    
    // Verify it's in buffer
    assert_eq!(table.write_buffer_row_count(), 2);
    
    // Verify vector search works (searches memory)
    {
        let hits = table.query()
            .vector_search("embedding", hyperstreamdb::core::index::VectorValue::Float32(vec![0.1, 0.2, 0.3]), 1)
            .to_batches().await?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].num_rows(), 1);
    }

    // 3. Simulate Crash/Reopen (WAL Recovery)
    // Drop the table and reopen
    drop(table);
    
    let table2 = Table::new_async(uri.clone()).await?;
    
    // Check if data was recovered from WAL
    assert_eq!(table2.write_buffer_row_count(), 2, "Should have recovered 2 rows from WAL");
    
    // Verify memory index was also re-built during recovery
    assert!(table2.has_memory_index(), "Memory index should be re-built from WAL");
    
    {
        let hits = table2.query()
            .vector_search("embedding", hyperstreamdb::core::index::VectorValue::Float32(vec![0.4, 0.5, 0.6]), 1)
            .to_batches().await?;
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].num_rows(), 1);
    }

    // 4. Commit (Flush to Disk)
    table2.commit_async().await?;
    
    // Buffer should be empty now
    assert_eq!(table2.write_buffer_row_count(), 0);
    
    // Data should be in manifest (sharded or not)
    let manifest = table2.manifest().await?;
    println!("Manifest version: {}, sharded: {}", manifest.version, manifest.manifest_list_path.is_some());
    
    // Read from disk to verify
    let results = table2.read_async(None, None, None).await?;
    let count: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(count, 2, "Should have 2 rows on disk after commit");

    Ok(())
}

#[tokio::test]
async fn test_core_ingestion_schema_evolution() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    let table = Table::new_async(uri.clone()).await?;
    
    // Write first batch with schema1
    let schema1 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
    ]));
    let batch1 = RecordBatch::try_new(
        schema1.clone(),
        vec![Arc::new(Int32Array::from(vec![1, 2]))]
    )?;
    table.write_async(vec![batch1]).await?;
    
    // Write second batch with evolve schema (added column)
    let schema2 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, true),
    ]));
    let batch2 = RecordBatch::try_new(
        schema2.clone(),
        vec![
            Arc::new(Int32Array::from(vec![3, 4])),
            Arc::new(StringArray::from(vec![Some("Alice"), Some("Bob")])),
        ]
    )?;
    table.write_async(vec![batch2]).await?;
    
    // Verify schema evolved
    let current_schema = table.arrow_schema();
    assert_eq!(current_schema.fields().len(), 2);
    assert!(current_schema.field_with_name("name").is_ok());
    
    table.commit_async().await?;
    
    // Read all data
    let results = table.read_async(None, None, None).await?;
    let _total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    
    Ok(())
}

#[tokio::test]
async fn test_recovered_evolved_wal() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let path = dir.path().to_str().unwrap().to_string();
    let uri = format!("file://{}", path);

    // Initial Write (Schema A: id)
    {
        let table = Table::new_async(uri.clone()).await?;
        table.set_autocommit(false);
        let schema1 = Arc::new(Schema::new(vec![Field::new("id", arrow::datatypes::DataType::Int32, false)]));
        let batch1 = RecordBatch::try_new(schema1.clone(), vec![Arc::new(Int32Array::from(vec![1, 2]))])?;
        table.write_async(vec![batch1]).await?;
        assert_eq!(table.write_buffer_row_count(), 2);
    }

    // Evolved Write (Schema B: id, name)
    {
        let table = Table::new_async(uri.clone()).await?;
        table.set_autocommit(false);
        assert_eq!(table.write_buffer_row_count(), 2);
        
        let schema2 = Arc::new(Schema::new(vec![
            Field::new("id", arrow::datatypes::DataType::Int32, false),
            Field::new("name", arrow::datatypes::DataType::Utf8, true),
        ]));
        let batch2 = RecordBatch::try_new(
            schema2.clone(), 
            vec![
                Arc::new(Int32Array::from(vec![3, 4])),
                Arc::new(StringArray::from(vec![Some("Alice"), Some("Bob")])),
            ]
        )?;
        
        table.write_async(vec![batch2]).await?;
        assert_eq!(table.write_buffer_row_count(), 4);
        assert_eq!(table.arrow_schema().fields().len(), 2);
    }

    // Reopen and Recover everything
    {
        let table = Table::new_async(uri.clone()).await?;
        assert_eq!(table.write_buffer_row_count(), 4);
        assert_eq!(table.arrow_schema().fields().len(), 2);
        
        let batches = table.read_async(None, None, None).await?;
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 4);
        
        let name_col_exists = batches.iter().any(|b| b.schema().field_with_name("name").is_ok());
        assert!(name_col_exists);
    }

    // 4. Test "Narrower" batch correctly promotes to widest schema
    {
        let table = Table::new_async(uri.clone()).await?;
        table.set_autocommit(false);
        assert_eq!(table.arrow_schema().fields().len(), 2);
        
        let schema_thin = Arc::new(Schema::new(vec![Field::new("id", arrow::datatypes::DataType::Int32, false)]));
        let batch_thin = RecordBatch::try_new(schema_thin.clone(), vec![Arc::new(Int32Array::from(vec![5, 6]))])?;
        
        // This should NOT shrink the table, but backfill the thin batch to 2 columns
        table.write_async(vec![batch_thin]).await?;
        assert_eq!(table.arrow_schema().fields().len(), 2);
        assert_eq!(table.write_buffer_row_count(), 6);
        
        let batches = table.read_async(None, None, None).await?;
        let total: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total, 6);
        
        // Verify last 2 rows have null names
        let total_name_nulls: usize = batches.iter().map(|b| {
            if let Some(col) = b.column_by_name("name") {
                col.null_count()
            } else {
                0
            }
        }).sum();
        // First 2 rows (recovered) had [1,2] -> name: null, [3,4] -> name: Some, [5,6] -> name: null
        // Total nulls should be 2 (from first batch) + 2 (from thin batch) = 4
        assert_eq!(total_name_nulls, 4);
    }

    Ok(())
}
