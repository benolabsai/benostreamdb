use anyhow::Result;
use arrow::array::{Int32Array, BooleanArray, StringArray};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{Schema, DataType, Field};
use hyperstreamdb::Table;
use std::sync::Arc;

#[tokio::test]
async fn test_boolean_sql_filtering() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
    
    // 1. Initialize Table
    let table = Table::new_async(uri.clone()).await?;
    
    // 2. Create Data with Boolean column
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("is_active", DataType::Boolean, false),
        Field::new("category", DataType::Utf8, false),
    ]));
    
    let id_array = Int32Array::from(vec![1, 2, 3, 4, 5]);
    let active_array = BooleanArray::from(vec![true, false, true, false, true]);
    let category_array = StringArray::from(vec!["A", "B", "A", "B", "A"]);
    
    let batch = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(id_array),
        Arc::new(active_array),
        Arc::new(category_array),
    ])?;
    
    // 3. Write and Commit
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    
    // 4. Test filtering for true values
    let batches_true = table.sql("SELECT * FROM t WHERE is_active = true").await?;
    let total_true: usize = batches_true.iter().map(|b| b.num_rows()).sum();
    
    println!("True results: {} rows", total_true);
    assert_eq!(total_true, 3);
    
    // 5. Test filtering for false values
    let batches_false = table.sql("SELECT * FROM t WHERE is_active = false").await?;
    let total_false: usize = batches_false.iter().map(|b| b.num_rows()).sum();
    
    println!("False results: {} rows", total_false);
    assert_eq!(total_false, 2);
    
    // 6. Test filtering with other columns
    let batches_cat = table.sql("SELECT * FROM t WHERE category = 'A'").await?;
    let total_cat: usize = batches_cat.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_cat, 3);

    Ok(())
}
