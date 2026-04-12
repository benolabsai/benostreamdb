// Copyright (c) 2026 Richard Albright. All rights reserved.

use hyperstreamdb::core::table::Table;
use hyperstreamdb::core::manifest::{PartitionSpec, PartitionField};
use arrow::record_batch::RecordBatch;
use arrow::array::{StringArray, Int64Array, DictionaryArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema, Int32Type};
use std::sync::Arc;

#[tokio::test]
async fn test_string_identity_partitioning() -> anyhow::Result<()> {
    // 1. Setup Table with String Identity Partitioning
    let table_name = "test_string_partition";
    let uri = format!("file:///tmp/{}", table_name);
    let _ = std::fs::remove_dir_all(format!("/tmp/{}", table_name));
    
    // Schema: category (string), value (int64)
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("category", DataType::Utf8, false),
        Field::new("value", DataType::Int64, false),
    ]));

    // Partition Spec: Identity on category (source_id: 1)
    let spec = PartitionSpec {
        spec_id: 1,
        fields: vec![
            PartitionField {
                source_ids: vec![1],
                source_id: Some(1),
                field_id: Some(1000),
                name: "category".to_string(),
                transform: "identity".to_string(),
            }
        ]
    };

    // 2. Create Table
    let table = Table::create_partitioned_async(uri.clone(), arrow_schema.clone(), spec.clone()).await?;

    // 3. Write Data with normal strings
    let categories = vec!["A".to_string(), "B".to_string(), "A".to_string()];
    let values = vec![1i64, 2, 3];
    
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            Arc::new(StringArray::from(categories)),
            Arc::new(Int64Array::from(values)),
        ],
    )?;
    
    // This used to fail with RuntimeError: Unsupported value-schema combination!
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    println!("Successfully wrote with normal strings");

    // 4. Test with Dictionary strings (Categorical)
    let dict_table_name = "test_dict_partition";
    let dict_uri = format!("file:///tmp/{}", dict_table_name);
    let _ = std::fs::remove_dir_all(format!("/tmp/{}", dict_table_name));

    let dict_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("category", DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)), false),
        Field::new("value", DataType::Int64, false),
    ]));

    let dict_table = Table::create_partitioned_async(dict_uri.clone(), dict_schema.clone(), spec).await?;

    let dict_values = vec![10i64, 20, 30];
    let dict_batch = RecordBatch::try_new(
        dict_schema.clone(),
        vec![
            Arc::new(DictionaryArray::<Int32Type>::new(
                arrow::array::Int32Array::from(vec![0, 1, 0]),
                Arc::new(StringArray::from(vec!["C", "D"]))
            )),
            Arc::new(Int64Array::from(dict_values)),
        ],
    )?;

    dict_table.write_async(vec![dict_batch]).await?;
    dict_table.commit_async().await?;
    
    println!("Successfully wrote with dictionary-encoded strings");

    Ok(())
}
