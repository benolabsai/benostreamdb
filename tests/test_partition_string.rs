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

#[tokio::test]
async fn test_mismatched_id_name_partitioning() -> anyhow::Result<()> {
    // This test verifies the hardening fix for when source_id points to an int column
    // but the partition field's name points to a string column (as seen in the guide error).
    
    let table_name = "test_mismatch_partition";
    let uri = format!("file:///tmp/{}", table_name);
    let _ = std::fs::remove_dir_all(format!("/tmp/{}", table_name));
    
    // Schema: id (int32), category (string)
    // id -> ID 1
    // category -> ID 2
    let arrow_schema = Arc::new(ArrowSchema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("category", DataType::Utf8, false),
    ]));

    // INCORRECT Partition Spec (from the guide): Identity on category BUT points to source_id 1 (id column)
    let spec = PartitionSpec {
        spec_id: 1,
        fields: vec![
            PartitionField {
                source_ids: vec![1],
                source_id: Some(1),
                field_id: Some(1000),
                name: "category".to_string(), // Mismatched! Name matches category (String), ID matches id (Int)
                transform: "identity".to_string(),
            }
        ]
    };

    // Create Table
    let table = Table::create_partitioned_async(uri.clone(), arrow_schema.clone(), spec).await?;

    // Write Data
    let ids = vec![1i32, 2, 3];
    let categories = vec!["A".to_string(), "B".to_string(), "A".to_string()];
    
    let batch = RecordBatch::try_new(
        arrow_schema.clone(),
        vec![
            Arc::new(arrow::array::Int32Array::from(ids)),
            Arc::new(StringArray::from(categories)),
        ],
    )?;
    
    // This used to fail with RuntimeError: Value String("A"), schema: Int
    // because iceberg.rs only looked at source_id (1 -> Int).
    // Now it should fallback to name "category" -> String.
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    println!("Successfully handled mismatched ID/Name partitioning!");

    Ok(())
}
