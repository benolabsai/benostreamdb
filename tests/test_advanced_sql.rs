use anyhow::Result;
use arrow::array::{Int32Array, StringArray, Float32Array, FixedSizeListArray};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{Schema, DataType, Field, SchemaRef};
use hyperstreamdb::Table;
use hyperstreamdb::core::table::VectorSearchParams;
use hyperstreamdb::core::index::VectorMetric;
use hyperstreamdb::core::manifest::{PartitionSpec, PartitionField};
use std::sync::Arc;

async fn clear_caches() {
    hyperstreamdb::core::cache::MANIFEST_CACHE.invalidate_all();
    hyperstreamdb::core::cache::LATEST_VERSION_CACHE.invalidate_all();
}

async fn get_complex_schema(dim: usize) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("year", DataType::Int32, false),
        Field::new("embedding", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32
        ), false),
    ]))
}

async fn create_complex_batch(start_id: i32, num_rows: usize, category: &str, year: i32, dim: usize) -> RecordBatch {
    let id_array = Int32Array::from_iter_values(start_id..start_id + num_rows as i32);
    let category_array = StringArray::from(vec![category; num_rows]);
    let year_array = Int32Array::from(vec![year; num_rows]);
    
    let mut values = Vec::with_capacity(num_rows * dim);
    for i in 0..num_rows {
        for j in 0..dim {
            values.push((i + j) as f32 * 0.1);
        }
    }
    let values_array = Float32Array::from(values);
    let vectors_array = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
        Arc::new(values_array),
        None
    ).unwrap();

    let schema = get_complex_schema(dim).await;

    RecordBatch::try_new(schema, vec![
        Arc::new(id_array),
        Arc::new(category_array),
        Arc::new(year_array),
        Arc::new(vectors_array),
    ]).unwrap()
}

#[tokio::test]
async fn test_hybrid_sql_vector_search() -> Result<()> {
    clear_caches().await;
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
    
    let dim = 4;
    let schema = get_complex_schema(dim).await;
    
    // 1. Create partitioned table
    let spec = PartitionSpec {
        spec_id: 0,
        fields: vec![
            PartitionField { source_id: 3, field_id: None, name: "year".to_string(), transform: "identity".to_string() },
            PartitionField { source_id: 2, field_id: None, name: "category".to_string(), transform: "identity".to_string() },
        ]
    };
    let mut table = Table::create_partitioned_async(uri.clone(), schema, spec).await?;
    table.index_all_columns_async().await?;

    // 2. Write data across 3 partitions
    table.write_async(vec![create_complex_batch(1, 10, "A", 2022, dim).await]).await?;
    table.write_async(vec![create_complex_batch(11, 10, "B", 2022, dim).await]).await?;
    table.write_async(vec![create_complex_batch(21, 10, "A", 2023, dim).await]).await?;
    table.commit_async().await?;

    // 3. Complex Query: year=2022 AND category='A' AND id > 5 + Vector Search
    let query_vec = vec![0.1; dim];
    let vs_params = VectorSearchParams::new("embedding", query_vec, 5);
    
    let filter = "year = 2022 AND category = 'A' AND id > 5";
    let results = table.read_async(Some(filter), Some(vs_params), None).await?;
    
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    // ids 1..10 in 2022/A. id > 5 => ids 6, 7, 8, 9, 10 (count 5).
    // Vector search k=5, so we should get all 5.
    assert_eq!(total_rows, 5);

    Ok(())
}

#[tokio::test]
async fn test_schema_evolution_addition() -> Result<()> {
    clear_caches().await;
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
    
    let table = Table::new_async(uri.clone()).await?;
    let dim = 4;
    
    // 1. Initial schema write
    let batch1 = create_complex_batch(1, 10, "A", 2022, dim).await;
    table.write_async(vec![batch1]).await?;
    table.commit_async().await?;

    // 2. Write with NEW column (schema evolution)
    let schema_v2 = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
        Field::new("year", DataType::Int32, false),
        Field::new("new_col", DataType::Utf8, true), // Added column
        Field::new("embedding", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32
        ), false),
    ]));

    let batch2 = RecordBatch::try_new(schema_v2, vec![
        Arc::new(Int32Array::from(vec![11, 12])),
        Arc::new(StringArray::from(vec!["A", "A"])),
        Arc::new(Int32Array::from(vec![2022, 2022])),
        Arc::new(StringArray::from(vec!["val1", "val2"])),
        Arc::new(FixedSizeListArray::try_new(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32,
            Arc::new(Float32Array::from(vec![0.1; 2 * dim])),
            None
        ).unwrap()),
    ])?;

    table.write_async(vec![batch2]).await?;
    table.commit_async().await?;

    // 3. Read back - old rows should have NULL for new_col
    let results = table.read_async(None, None, None).await?;
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 12);
    
    // Verify all columns are present in the final reloaded schema
    clear_caches().await;
    let table_reloaded = Table::new_async(uri.clone()).await?;
    let final_schema: SchemaRef = table_reloaded.arrow_schema();
    assert!(final_schema.fields().iter().any(|f| f.name() == "new_col"));

    Ok(())
}
#[tokio::test]
async fn test_cosine_similarity_search() -> Result<()> {
    clear_caches().await;
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
    
    let dim = 4;
    let schema = get_complex_schema(dim).await;
    let mut table = Table::new_async(uri.clone()).await?;
    table.index_all_columns_async().await?;

    // Create vectors with specific directions
    // Vec1: [1, 0, 0, 0]
    // Vec2: [0, 1, 0, 0]
    // Query: [1, 0.1, 0, 0] (should be closer to Vec1 in Cosine)
    
    let _row1 = vec![1.0, 0.0, 0.0, 0.0];
    let _row2 = vec![0.0, 1.0, 0.0, 0.0];
    
    let id_array = Int32Array::from(vec![1, 2]);
    let category_array = StringArray::from(vec!["A", "B"]);
    let year_array = Int32Array::from(vec![2022, 2022]);
    let values_array = Float32Array::from(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
    let vectors_array = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
        Arc::new(values_array),
        None
    ).unwrap();

    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(id_array),
        Arc::new(category_array),
        Arc::new(year_array),
        Arc::new(vectors_array),
    ])?;

    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    // Wait for background indexing task to finish
    table.wait_for_background_tasks_async().await?;

    let query_vec = vec![1.0, 0.1, 0.0, 0.0];
    let vs_params = VectorSearchParams::new("embedding", query_vec, 1)
        .with_metric(VectorMetric::Cosine);
    
    let results = table.read_async(None, Some(vs_params), None).await?;
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 1);
    
    // id should be 1
    let id = results[0].column(0).as_any().downcast_ref::<Int32Array>().unwrap().value(0);
    assert_eq!(id, 1);

    Ok(())
}
