// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use arrow::array::{Float32Array, FixedSizeListArray, Int32Array};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{Schema, Field, DataType};
use hyperstreamdb::Table;
use hyperstreamdb::core::table::VectorSearchParams;
use hyperstreamdb::core::query::QueryConfig;
use std::sync::Arc;

async fn create_vector_batch(start_id: i32, num_rows: usize, dim: usize, val_offset: f32) -> RecordBatch {
    let id_array = Int32Array::from_iter_values(start_id..start_id + num_rows as i32);
    
    let mut values = Vec::with_capacity(num_rows * dim);
    for i in 0..num_rows {
        for j in 0..dim {
            values.push((i + j) as f32 + val_offset);
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
async fn test_parallel_vs_sequential_consistency() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
    
    let mut table = Table::new_async(uri.clone()).await?;
    let dim = 8;
    table.index_all_columns_async().await?;

    // 1. Write multiple segments to ensure parallel execution is possible
    for i in 0..5 {
        let batch = create_vector_batch(i * 100, 100, dim, i as f32 * 0.1).await;
        table.write_async(vec![batch]).await?;
        table.commit_async().await?;
    }
    
    // Wait for all background indexing tasks to complete
    table.wait_for_background_tasks_async().await?;

    let query_vec = vec![0.1; dim];
    let vs_params = VectorSearchParams::new(
        "embedding",
        hyperstreamdb::core::index::VectorValue::Float32(query_vec),
        10,
    );

    // 2. Search with parallelism = 1 (Sequential)
    let config_seq = QueryConfig::new().with_max_parallel_readers(1);
    let results_seq = table.read_with_config_async(None, Some(vs_params.clone()), None, config_seq).await?;

    // 3. Search with parallelism = 4 (Parallel)
    let config_par = QueryConfig::new().with_max_parallel_readers(4);
    let results_par = table.read_with_config_async(None, Some(vs_params), None, config_par).await?;

    // 4. Verify results are identical (after sorting by ID to compare batches)
    let collect_ids = |batches: Vec<RecordBatch>| {
        let mut ids = Vec::new();
        for b in batches {
            let id_col = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            for i in 0..b.num_rows() {
                ids.push(id_col.value(i));
            }
        }
        ids.sort();
        ids
    };

    let ids_seq = collect_ids(results_seq);
    let ids_par = collect_ids(results_par);

    assert_eq!(ids_seq.len(), 10);
    assert_eq!(ids_seq, ids_par, "Parallel and sequential search results must be identical");

    Ok(())
}
