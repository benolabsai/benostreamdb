// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use arrow::array::{
    Array, BooleanArray, Date32Array, Float32Array, Float64Array, 
    Int32Array, Int64Array, LargeBinaryArray, ListArray, StringArray, 
    StructArray, Time64MicrosecondArray, TimestampMicrosecondArray, TimestampNanosecondArray
};
use arrow::record_batch::RecordBatch;
use arrow::datatypes::{Schema, DataType, Field, TimeUnit};
use hyperstreamdb::Table;
use hyperstreamdb::core::manifest::{PartitionSpec, PartitionField};
use std::sync::Arc;

async fn clear_caches() {
    hyperstreamdb::core::cache::MANIFEST_CACHE.invalidate_all();
    hyperstreamdb::core::cache::LATEST_VERSION_CACHE.invalidate_all();
}

#[tokio::test]
async fn test_primitive_types() -> Result<()> {
    clear_caches().await;
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

    let schema = Arc::new(Schema::new(vec![
        Field::new("f_bool", DataType::Boolean, false),
        Field::new("f_int64", DataType::Int64, false),
        Field::new("f_float32", DataType::Float32, false),
        Field::new("f_float64", DataType::Float64, false),
        Field::new("f_binary", DataType::LargeBinary, false),
    ]));

    let table = Table::create_async(uri.clone(), schema.clone()).await?;

    let batch = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(BooleanArray::from(vec![true, false])),
        Arc::new(Int64Array::from(vec![100, 200])),
        Arc::new(Float32Array::from(vec![1.1, 2.2])),
        Arc::new(Float64Array::from(vec![10.1, 20.2])),
        Arc::new(LargeBinaryArray::from(vec![b"hello".as_ref(), b"world".as_ref()])),
    ])?;

    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    // Reload and verify
    let table_reloaded = Table::new_async(uri.clone()).await?;
    let batches: Vec<RecordBatch> = table_reloaded.read_filter_async(vec![], None, None).await?;
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);

    let f_bool = batches[0].column(0).as_any().downcast_ref::<BooleanArray>().unwrap();
    assert!(f_bool.value(0));
    
    Ok(())
}

#[tokio::test]
async fn test_temporal_types() -> Result<()> {
    clear_caches().await;
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
    
    // Iceberg supports: Date, Time(micro), Timestamp(micro/nano, with/without TZ)
    let schema = Arc::new(Schema::new(vec![
        Field::new("f_date", DataType::Date32, false),
        Field::new("f_time", DataType::Time64(TimeUnit::Microsecond), false),
        Field::new("f_ts_micro", DataType::Timestamp(TimeUnit::Microsecond, None), false),
        Field::new("f_ts_nano", DataType::Timestamp(TimeUnit::Nanosecond, None), false),
        Field::new("f_ts_tz", DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into())), false),
    ]));

    let table = Table::create_async(uri.clone(), schema.clone()).await?;

    let batch = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(Date32Array::from(vec![18_628])), // 2021-01-01
        Arc::new(Time64MicrosecondArray::from(vec![3_600_000_000])), // 01:00:00
        Arc::new(TimestampMicrosecondArray::from(vec![1_609_459_200_000_000])), // 2021-01-01 00:00:00
        Arc::new(TimestampNanosecondArray::from(vec![1_609_459_200_000_000_000])), 
        Arc::new(TimestampMicrosecondArray::from(vec![1_609_459_200_000_000]).with_timezone("UTC")),
    ])?;

    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    let table_reloaded = Table::new_async(uri.clone()).await?;
    let manifest = table_reloaded.manifest().await?;
    
    // Check Iceberg Type Mapping in Manifest
    let schema_msg = &manifest.schemas[0];
    assert!(schema_msg.fields.iter().any(|f| f.name == "f_date"));
    assert!(schema_msg.fields.iter().any(|f| f.name == "f_ts_tz" && f.type_str.contains("timestamp")));

    let batches: Vec<RecordBatch> = table_reloaded.read_filter_async(vec![], None, None).await?;
    assert_eq!(batches[0].num_rows(), 1);
    
    Ok(())
}

#[tokio::test]
async fn test_uuid_and_fixed() -> Result<()> {
    clear_caches().await;
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
    
    // UUID is often mapped to Fixed(16) or String in Arrow. 
    // HyperStream currently treats it as String/Utf8 for compatibility.
    let schema = Arc::new(Schema::new(vec![
        Field::new("f_uuid", DataType::Utf8, false),
        Field::new("f_fixed", DataType::FixedSizeBinary(3), false),
    ]));

    let table = Table::create_async(uri.clone(), schema.clone()).await?;

    let uuid_val = uuid::Uuid::new_v4().to_string();
    
    let mut fixed_builder = arrow::array::builder::FixedSizeBinaryBuilder::new(3);
    fixed_builder.append_value([1u8, 2, 3])?;
    let fixed_array = fixed_builder.finish();

    let batch = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(StringArray::from(vec![uuid_val.clone()])),
        Arc::new(fixed_array),
    ])?;

    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    let table_reloaded = Table::new_async(uri.clone()).await?;
    let batches: Vec<RecordBatch> = table_reloaded.read_filter_async(vec![], None, None).await?;
    
    let f_uuid = batches[0].column(0).as_any().downcast_ref::<StringArray>().unwrap();
    assert_eq!(f_uuid.value(0), uuid_val);

    Ok(())
}

#[tokio::test]
async fn test_complex_types() -> Result<()> {
    clear_caches().await;
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

    // Map<String, Int>
    let key_field = Field::new("keys", DataType::Utf8, false);
    let value_field = Field::new("values", DataType::Int32, true);
    let entry_struct = DataType::Struct(vec![key_field, value_field].into());
    let map_type = DataType::Map(Arc::new(Field::new("entries", entry_struct, false)), false);

    let schema = Arc::new(Schema::new(vec![
        Field::new("f_list", DataType::List(Arc::new(Field::new("item", DataType::Int32, true))), false),
        Field::new("f_map", map_type.clone(), false),
    ]));

    let table = Table::create_async(uri.clone(), schema.clone()).await?;

    // Helper for offsets
    use arrow::buffer::OffsetBuffer;

    // Construct List Array
    let list_values = Int32Array::from(vec![Some(1), Some(2), Some(3)]);
    let list_offsets = OffsetBuffer::new(vec![0, 3].into());
    let list_array = ListArray::new(
        Arc::new(Field::new("item", DataType::Int32, true)),
        list_offsets,
        Arc::new(list_values),
        None
    );

    // Construct Map Array using Builder
    use arrow::array::builder::{MapBuilder, StringBuilder, Int32Builder};

    let mut map_builder = MapBuilder::new(
        Some(arrow::array::builder::MapFieldNames {
            entry: "entries".to_string(),
            key: "keys".to_string(),
            value: "values".to_string(),
        }), 
        StringBuilder::new(), 
        Int32Builder::new()
    );

    // Row 1: {"a": 1}
    map_builder.keys().append_value("a");
    map_builder.values().append_value(1);
    map_builder.append(true)?; 
    
    let map_array = map_builder.finish();

    let batch = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(list_array),
        Arc::new(map_array),
    ])?;

    table.write_async(vec![batch]).await?;
    table.commit_async().await?;

    let table_reloaded = Table::new_async(uri.clone()).await?;
    let manifest = table_reloaded.manifest().await?;
    
    // Check conversions in manifest
    let schema_def = &manifest.schemas[0];
    assert!(schema_def.fields.iter().any(|f| f.type_str == "list"));
    assert!(schema_def.fields.iter().any(|f| f.type_str == "map"));

    Ok(())
}

#[tokio::test]
async fn test_schema_evolution_full_lifecycle() -> Result<()> {
    clear_caches().await;
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());

    // 1. Create Initial Table
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("data", DataType::Utf8, false),
    ]));
    let table = Table::create_async(uri.clone(), schema).await?;

    // 2. Add Column
    table.add_column("age", DataType::Int32).await?;
    let table = Table::new_async(uri.clone()).await?; // Reload to refresh schema
    assert!(table.arrow_schema().field_with_name("age").is_ok());

    // 3. Rename Column
    table.rename_column("data", "info").await?;
    let table = Table::new_async(uri.clone()).await?;
    assert!(table.arrow_schema().field_with_name("info").is_ok());
    assert!(table.arrow_schema().field_with_name("data").is_err());

    // 4. Update Type (Promote Int32 -> Int64)
    table.update_column_type("age", "long").await?;
    let table = Table::new_async(uri.clone()).await?;
    assert_eq!(table.arrow_schema().field_with_name("age")?.data_type(), &DataType::Int64);

    // 5. Move Column (Reorder)
    // Current Order: id, info, age
    // Move 'age' to index 0. New Order: age, id, info
    table.move_column("age", 0).await?;
    let table = Table::new_async(uri.clone()).await?;
    assert_eq!(table.arrow_schema().fields()[0].name(), "age");
    assert_eq!(table.arrow_schema().fields()[1].name(), "id");
    assert_eq!(table.arrow_schema().fields()[2].name(), "info");

    // 6. Drop Column
    table.drop_column("info").await?;
    let table = Table::new_async(uri.clone()).await?;
    assert!(table.arrow_schema().field_with_name("info").is_err());
    assert_eq!(table.arrow_schema().fields().len(), 2); // age, id

    Ok(())
}

#[tokio::test]
async fn test_nested_schema_and_field_id_preservation() -> Result<()> {
    // Keeping original test for regression safety
    clear_caches().await;
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
    
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("info", DataType::Struct(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int32, true),
        ].into()), true),
    ]));
    
    let table = Table::create_async(uri.clone(), schema).await?;
    
    let arrow_schema = table.arrow_schema();
    let info_field = arrow_schema.field_with_name("info").unwrap();
    let struct_fields = match info_field.data_type() {
        DataType::Struct(fields) => fields,
        _ => panic!("Expected Struct"),
    };

    let batch = RecordBatch::try_new(arrow_schema.clone(), vec![
        Arc::new(Int32Array::from(vec![1])),
        Arc::new(StructArray::try_new(
            struct_fields.clone(),
            vec![
                Arc::new(StringArray::from(vec!["Alice"])) as Arc<dyn Array>,
                Arc::new(Int32Array::from(vec![30])) as Arc<dyn Array>,
            ],
            None
        )?),
    ])?;
    
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    
    clear_caches().await;
    let table_reloaded = Table::new_async(uri.clone()).await?;
    let manifest = table_reloaded.manifest().await?;
    
    let schema_field = manifest.schemas[0].fields.iter().find(|f| f.name == "info").unwrap();
    assert_eq!(schema_field.type_str, "struct");
    
    // Field IDs should be stable check
    assert_eq!(manifest.schemas[0].fields[0].id, 1); // id
    assert_eq!(manifest.schemas[0].fields[1].id, 2); // info
    
    Ok(())
}

#[tokio::test]
async fn test_partition_transforms() -> Result<()> {
    // Keeping original test
    clear_caches().await;
    let temp_dir = tempfile::tempdir()?;
    let uri = format!("file://{}", temp_dir.path().to_str().unwrap());
    
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    
    let spec = PartitionSpec {
        spec_id: 0,
        fields: vec![
            PartitionField { source_ids: vec![1], source_id: Some(1), field_id: None, name: "id_bucket".to_string(), transform: "bucket[4]".to_string() },
        ]
    };
    
    let table = Table::create_partitioned_async(uri.clone(), schema, spec).await?;
    
    let batch = RecordBatch::try_new(table.arrow_schema(), vec![
        Arc::new(Int32Array::from(vec![1, 2, 100, 200])),
        Arc::new(StringArray::from(vec!["A", "B", "C", "D"])),
    ])?;
    
    table.write_async(vec![batch]).await?;
    table.commit_async().await?;
    
    let stats = table.get_table_statistics_async().await?;
    // With bucket[4] and 4 diverse integer values, we expect multiple partition files
    assert!(stats.file_count >= 2, "Expected at least 2 partition files, got {}", stats.file_count);
    
    Ok(())
}

