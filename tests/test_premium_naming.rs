// Copyright (c) 2026 Richard Albright. All rights reserved.

use hyperstreamdb::Table;
use arrow::record_batch::RecordBatch;
use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use std::sync::Arc;
use tempfile::tempdir;

#[tokio::test]
async fn test_excel_naming_logic() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    // 1. Create Data with numeric/unnamed columns (0, 1)
    let schema = Arc::new(Schema::new(vec![
        Field::new("0", DataType::Int32, false),
        Field::new("1", DataType::Int32, false),
    ]));
    
    let batch = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(Int32Array::from(vec![10, 20, 30])),
        Arc::new(Int32Array::from(vec![100, 200, 300])),
    ])?;

    // 2. Open Table (Empty)
    let table = Table::new_async(uri.clone()).await?;
    
    // 3. Write data - should trigger Excel auto-naming (Default)
    table.write_async(vec![batch]).await?;
    
    // 4. Verify Schema has names "A" and "B" instead of "0" and "1"
    let updated_schema = table.arrow_schema();
    println!("Schema fields: {:?}", updated_schema.fields().iter().map(|f| f.name()).collect::<Vec<_>>());
    
    // Use find to check names to be robust against metadata columns
    assert!(updated_schema.fields().iter().any(|f| f.name() == "A"), "Field 'A' not found");
    assert!(updated_schema.fields().iter().any(|f| f.name() == "B"), "Field 'B' not found");
    
    Ok(())
}

#[tokio::test]
async fn test_polars_naming_logic() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());

    // 1. Create Data
    let schema = Arc::new(Schema::new(vec![
        Field::new("0", DataType::Int32, false),
    ]));
    let batch = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(Int32Array::from(vec![1])),
    ])?;

    // 2. Open Table with Polars Naming Pattern
    let table = Table::builder(uri.clone())
        .with_auto_label_columns(hyperstreamdb::core::table::LabelPattern::Polars)
        .build_async()
        .await?;
    
    // 3. Write data
    table.write_async(vec![batch]).await?;
    
    // 4. Verify name is "column_1"
    let updated_schema = table.arrow_schema();
    assert!(updated_schema.fields().iter().any(|f| f.name() == "column_1"), "Field 'column_1' not found");
    
    Ok(())
}
