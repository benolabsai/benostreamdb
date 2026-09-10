// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use arrow::array::{Float32Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use hyperstreamdb::Table;
use std::sync::Arc;

fn create_composite_test_batch(
    start_id: i32,
    num_rows: usize,
    tenant: &str,
    status: &str,
) -> RecordBatch {
    let id_array = Int32Array::from_iter_values(start_id..start_id + num_rows as i32);
    let tenant_array = StringArray::from(vec![tenant; num_rows]);
    let status_array = StringArray::from(vec![status; num_rows]);
    let score_array = Float32Array::from(
        (0..num_rows)
            .map(|i| (start_id + i as i32) as f32 * 1.5)
            .collect::<Vec<f32>>(),
    );

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("tenant_id", DataType::Utf8, false),
        Field::new("status", DataType::Utf8, false),
        Field::new("score", DataType::Float32, false),
    ]));

    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(id_array),
            Arc::new(tenant_array),
            Arc::new(status_array),
            Arc::new(score_array),
        ],
    )
    .unwrap()
}

#[tokio::test]
async fn test_composite_scalar_roaring_bitmap_index() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

    let mut table = Table::new_async(uri.clone()).await?;

    // 1. Ingest initial batches with different tenants and statuses
    let batch1 = create_composite_test_batch(0, 50, "tenant_alpha", "active");
    let batch2 = create_composite_test_batch(50, 50, "tenant_alpha", "pending");
    let batch3 = create_composite_test_batch(100, 50, "tenant_beta", "active");
    let batch4 = create_composite_test_batch(150, 50, "tenant_beta", "archived");

    table
        .write_async(vec![batch1, batch2, batch3, batch4])
        .await?;
    table.commit_async().await?;

    // 2. Register composite index on (tenant_id, status)
    table
        .add_composite_index_async(vec!["tenant_id".to_string(), "status".to_string()])
        .await?;

    // 3. Index columns and wait for background index builds
    table.index_all_columns_async().await?;
    table.wait_for_background_tasks_async().await?;

    // 4. Query matching both columns in the composite index
    let filter = "tenant_id = 'tenant_alpha' AND status = 'active'";
    let results = table.read_async(Some(filter), None, None).await?;

    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(
        total_rows, 50,
        "Expected exactly 50 rows matching alpha active"
    );

    for batch in &results {
        let tenant_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let status_col = batch
            .column(2)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        for i in 0..batch.num_rows() {
            assert_eq!(tenant_col.value(i), "tenant_alpha");
            assert_eq!(status_col.value(i), "active");
        }
    }

    // 5. Query for a combination that does not exist
    let filter_empty = "tenant_id = 'tenant_alpha' AND status = 'archived'";
    let results_empty = table.read_async(Some(filter_empty), None, None).await?;
    let empty_rows: usize = results_empty.iter().map(|b| b.num_rows()).sum();
    assert_eq!(empty_rows, 0, "Expected 0 rows for alpha archived");

    // 6. Query matching beta active
    let filter_beta = "tenant_id = 'tenant_beta' AND status = 'active'";
    let results_beta = table.read_async(Some(filter_beta), None, None).await?;
    let beta_rows: usize = results_beta.iter().map(|b| b.num_rows()).sum();
    assert_eq!(beta_rows, 50, "Expected 50 rows for beta active");

    // 7. Verify explain output reflects query
    let explain_output = table.explain(Some(filter), None).await;
    assert!(!explain_output.is_empty());

    Ok(())
}
