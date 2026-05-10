// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use arrow::array::{Int32Array, Float32Array, FixedSizeListArray};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{Schema, DataType, Field};
use hyperstreamdb::Table;
use hyperstreamdb::core::table::VectorSearchParams;
use std::sync::Arc;
use std::fs;

async fn create_test_batch(start_id: i32, num_rows: usize) -> RecordBatch {
    let dim = 4;
    let id_array = Int32Array::from_iter_values(start_id..start_id + num_rows as i32);
    
    let mut values = Vec::with_capacity(num_rows * dim);
    for i in 0..num_rows {
        for j in 0..dim {
            values.push((i + j) as f32);
        }
    }
    let values_array = Float32Array::from(values);
    let vectors_array = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        dim as i32,
        Arc::new(values_array),
        None
    ).unwrap();

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("embedding", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            dim as i32
        ), false),
    ]));

    RecordBatch::try_new(schema, vec![
        Arc::new(id_array),
        Arc::new(vectors_array),
    ]).unwrap()
}

#[tokio::test]
async fn test_chaos_missing_index_files() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri_path = temp_dir.path().to_str().unwrap();
    let uri = format!("file://{}", uri_path);
    
    // 1. Initialize Table
    let table = Arc::new(Table::new_async(uri.clone()).await?);
    
    table.set_indexed_columns(vec!["embedding".to_string()]);
    
    // Establish schema and index
    let batch = create_test_batch(0, 2000).await;
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    
    // Force write to parquet
    let _ = table.rewrite_data_files_async(None).await;
    
    // 2. Locate Data files and delete them (Chaos!)
    let mut deleted_something = false;
    let entries = table.get_snapshot_segments().await?;
    for entry in entries {
        let p = entry.file_path;
        let local_path = p.replace("file://", "").replace("file:", "");
        let full_path = format!("{}/{}", uri_path, local_path);
        println!("Chaos: Checking file {}", full_path);
        if fs::metadata(&full_path).is_ok() {
            let _ = fs::remove_file(&full_path);
            println!("Chaos: Deleted parquet data file {}", full_path);
            deleted_something = true;
        }
    }
    
    assert!(deleted_something, "Expected to find and delete data files for chaos testing");
    
    // 3. Attempt search (should fail cleanly, not panic)
    let query_vec = vec![0.5; 4];
    let vs_params = VectorSearchParams::new("embedding", hyperstreamdb::core::index::VectorValue::Float32(query_vec), 5);
    
    let search_res = table.read_async(None, Some(vs_params), None).await;
    
    // Should result in explicit Error or graceful fallback
    match search_res {
        Ok(res) => println!("Chaos survived: gracefully fell back. Rows: {}", res.len()),
        Err(e) => println!("Chaos survived: error returned cleanly. Error: {}", e),
    }
    
    Ok(())
}
