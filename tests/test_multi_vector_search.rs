// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use arrow::array::{FixedSizeListArray, Float32Array, Int32Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use hyperstreamdb::core::index::VectorValue;
use hyperstreamdb::core::table::VectorSearchParams;
use hyperstreamdb::Table;
use std::sync::Arc;

#[tokio::test]
async fn test_multi_vector_search_rrf_scoring() -> Result<()> {
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

    let table = Table::new_async(uri.clone()).await?;

    let item_field = Arc::new(Field::new("item", DataType::Float32, true));
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
        Field::new(
            "title_vec",
            DataType::FixedSizeList(item_field.clone(), 3),
            false,
        ),
        Field::new(
            "body_vec",
            DataType::FixedSizeList(item_field.clone(), 3),
            false,
        ),
    ]));

    let id_array = Int32Array::from(vec![1, 2, 3, 4]);
    let name_array = StringArray::from(vec![
        "match_both",
        "match_title_only",
        "match_body_only",
        "match_neither",
    ]);

    // doc1: title_vec=[1,0,0], body_vec=[0,1,0] -> exact match on BOTH
    // doc2: title_vec=[1,0,0], body_vec=[0,0,1] -> exact match on title only
    // doc3: title_vec=[0,0,1], body_vec=[0,1,0] -> exact match on body only
    // doc4: title_vec=[0,0,1], body_vec=[0,0,1] -> far from both
    let title_vals = Float32Array::from(vec![
        1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0,
    ]);
    let body_vals = Float32Array::from(vec![
        0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
    ]);

    let title_array = FixedSizeListArray::new(item_field.clone(), 3, Arc::new(title_vals), None);
    let body_array = FixedSizeListArray::new(item_field, 3, Arc::new(body_vals), None);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(id_array),
            Arc::new(name_array),
            Arc::new(title_array),
            Arc::new(body_array),
        ],
    )?;

    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    // Multi-vector query searching for title near [1, 0, 0] and body near [0, 1, 0]
    let query_title = vec![1.0, 0.0, 0.0];
    let query_body = vec![0.0, 1.0, 0.0];

    let vector_filters = vec![
        VectorSearchParams::new("title_vec", VectorValue::Float32(query_title), 4),
        VectorSearchParams::new("body_vec", VectorValue::Float32(query_body), 4),
    ];

    let results = table.read_async(None, Some(vector_filters), None).await?;
    assert!(
        !results.is_empty(),
        "Expected results from multi-vector search"
    );

    let first_batch = &results[0];
    let id_col = first_batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let name_col = first_batch
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();

    // The top result MUST be doc 1 ("match_both") because it scored high in both vector ranks
    assert_eq!(id_col.value(0), 1);
    assert_eq!(name_col.value(0), "match_both");

    // The last result should be doc 4 ("match_neither")
    let last_idx = first_batch.num_rows() - 1;
    assert_eq!(id_col.value(last_idx), 4);
    assert_eq!(name_col.value(last_idx), "match_neither");

    // Also test via HyperStreamSession SQL execution
    let session = hyperstreamdb::core::sql::session::HyperStreamSession::new(None);
    session.register_table("documents", Arc::new(table.clone()))?;

    let sql_results = session
        .sql(
            "SELECT id, name, dist_l2(title_vec, ARRAY[1.0, 0.0, 0.0]) as dist1, dist_l2(body_vec, ARRAY[0.0, 1.0, 0.0]) as dist2
             FROM documents
             ORDER BY dist1, dist2 LIMIT 4",
        )
        .await?;

    let sql_batch = &sql_results.0[0];
    let sql_id_col = sql_batch
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(sql_id_col.value(0), 1, "Top match in SQL should be doc 1");

    Ok(())
}
