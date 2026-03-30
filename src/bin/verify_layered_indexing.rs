// Copyright (c) 2026 Richard Albright. All rights reserved.

use hyperstreamdb::Table;
use anyhow::Result;
use apache_avro::{Writer, Schema as AvroSchema, types::Value as AvroValue};
use std::fs::File;
use arrow::record_batch::RecordBatch;
use arrow::array::{Int32Array, FixedSizeListArray, Float32Array};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<()> {
    let base_dir = "/tmp/iceberg_test";
    let _ = std::fs::remove_dir_all(base_dir);
    std::fs::create_dir_all(format!("{}/metadata", base_dir))?;
    std::fs::create_dir_all(format!("{}/data", base_dir))?;

    println!("1. Creating dummy data file f1 (Parquet)...");
    let data_path1 = format!("{}/data/f1.parquet", base_dir);
    create_dummy_parquet(&data_path1)?;

    println!("2. Creating Iceberg Metadata v1...");
    create_iceberg_snapshot(base_dir, 1, &data_path1, "m1.avro", "ml1.avro")?;
    
    let metadata_path = format!("{}/metadata/v1.metadata.json", base_dir);
    let metadata_json = iceberg_metadata_json(base_dir, 1, "ml1.avro");
    std::fs::write(&metadata_path, serde_json::to_string_pretty(&metadata_json)?)?;

    println!("3. Registering external table in HyperStreamDB...");
    let hdb_uri = "file:///tmp/hdb_shadow";
    let _ = std::fs::remove_dir_all("/tmp/hdb_shadow");
    
    let mut table = Table::register_external(
        hdb_uri.to_string(), 
        &format!("file://{}", metadata_path)
    ).await?;

    println!("4. Verifying initial shadowed entries...");
    let segments = table.get_snapshot_segments().await?;
    assert_eq!(segments.len(), 1, "Should have 1 segment initially");
    println!("Found 1 segment as expected.");

    println!("5. Building Scalar Index for 'id' column...");
    // This should build a sidecar index in /tmp/hdb_shadow
    table.add_index_columns_async(vec!["id".to_string()], None).await?;
    
    println!("6. Checking for local sidecar index file...");
    let mut found_idx = false;
    for entry in std::fs::read_dir("/tmp/hdb_shadow")? {
        let entry = entry?;
        let name = entry.file_name().into_string().unwrap();
        if name.contains(".id.idx") || name.contains(".id.inv.parquet") {
            println!("Found sidecar index: {}", name);
            found_idx = true;
        }
    }
    assert!(found_idx, "Sidecar index file was not created in local HDB directory");

    println!("7. Verifying Query using sidecar index...");
    // The dummy data has id=1..100. Let's filter id=10.
    // The scalar index (in its current mock-like state) might just filter > 0,
    // but the point is to see if it executes.
    let results = table.read_async(Some("id = 10"), None, None).await?;
    println!("Query results: {} batches", results.len());
    let mut total_rows = 0;
    for batch in results {
        total_rows += batch.num_rows();
        let id_col = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..batch.num_rows() {
             assert_eq!(id_col.value(i), 10, "Query returned wrong ID");
        }
    }
    assert!(total_rows > 0, "Query should have returned at least one row");
    println!("Universal Indexing Verified Successfully!");

    Ok(())
}

fn create_dummy_parquet(path: &str) -> Result<()> {
    use parquet::arrow::ArrowWriter;
    use arrow::datatypes::{DataType, Field, Schema};
    
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("vector", DataType::FixedSizeList(
            Arc::new(Field::new("item", DataType::Float32, true)),
            4
        ), false),
    ]));

    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;

    let ids = Int32Array::from((1..=100).collect::<Vec<i32>>());
    let mut vector_values = Vec::new();
    for i in 1..=100 {
        for _ in 0..4 {
            vector_values.push(i as f32);
        }
    }
    let flattened_vectors = Float32Array::from(vector_values);
    let vectors = FixedSizeListArray::try_new(
        Arc::new(Field::new("item", DataType::Float32, true)),
        4,
        Arc::new(flattened_vectors),
        None
    )?;

    let batch = RecordBatch::try_new(schema, vec![Arc::new(ids), Arc::new(vectors)])?;
    writer.write(&batch)?;
    writer.close()?;
    Ok(())
}

fn create_iceberg_snapshot(base_dir: &str, snapshot_id: i64, data_path: &str, m_name: &str, ml_name: &str) -> Result<()> {
    let manifest_schema_str = r#"{
        "type": "record",
        "name": "manifest_entry",
        "fields": [
            {"name": "status", "type": "int"},
            {"name": "snapshot_id", "type": ["null", "long"]},
            {
                "name": "data_file",
                "type": {
                    "type": "record",
                    "name": "data_file",
                    "fields": [
                        {"name": "file_path", "type": "string"},
                        {"name": "file_format", "type": "string"},
                        {"name": "partition", "type": {"type": "record", "name": "partition", "fields": []}},
                        {"name": "record_count", "type": "long"},
                        {"name": "file_size_in_bytes", "type": "long"}
                    ]
                }
            }
        ]
    }"#;
    let manifest_schema = AvroSchema::parse_str(manifest_schema_str)?;
    let manifest_path = format!("{}/metadata/{}", base_dir, m_name);
    let mut writer = Writer::new(&manifest_schema, File::create(&manifest_path)?);

    let data_file = vec![
        ("file_path".to_string(), AvroValue::String(data_path.to_string())),
        ("file_format".to_string(), AvroValue::String("PARQUET".to_string())),
        ("partition".to_string(), AvroValue::Record(Vec::new())),
        ("record_count".to_string(), AvroValue::Long(100)),
        ("file_size_in_bytes".to_string(), AvroValue::Long(1024)),
    ];

    let entry = vec![
        ("status".to_string(), AvroValue::Int(1)), // Added
        ("snapshot_id".to_string(), AvroValue::Union(1, Box::new(AvroValue::Long(snapshot_id)))),
        ("data_file".to_string(), AvroValue::Record(data_file)),
    ];

    writer.append(AvroValue::Record(entry))?;
    writer.flush()?;

    let ml_schema_str = r#"{
        "type": "record",
        "name": "manifest_list_entry",
        "fields": [
            {"name": "manifest_path", "type": "string"},
            {"name": "manifest_length", "type": "long"},
            {"name": "partition_spec_id", "type": "int"},
            {"name": "added_snapshot_id", "type": "long"}
        ]
    }"#;
    let ml_schema = AvroSchema::parse_str(ml_schema_str)?;
    let ml_path = format!("{}/metadata/{}", base_dir, ml_name);
    let mut ml_writer = Writer::new(&ml_schema, File::create(&ml_path)?);

    let ml_entry = vec![
        ("manifest_path".to_string(), AvroValue::String(m_name.to_string())),
        ("manifest_length".to_string(), AvroValue::Long(512)),
        ("partition_spec_id".to_string(), AvroValue::Int(0)),
        ("added_snapshot_id".to_string(), AvroValue::Long(snapshot_id)),
    ];

    ml_writer.append(AvroValue::Record(ml_entry))?;
    ml_writer.flush()?;
    Ok(())
}

fn iceberg_metadata_json(base_dir: &str, snapshot_id: i64, ml_name: &str) -> serde_json::Value {
    serde_json::json!({
        "format-version": 2,
        "table-uuid": "test-uuid",
        "location": format!("file://{}", base_dir),
        "last-sequence-number": snapshot_id,
        "last-updated-ms": 123456789,
        "current-snapshot-id": snapshot_id,
        "current-schema-id": 0,
        "schemas": [
            {
                "schema-id": 0, 
                "fields": [
                    {"id": 1, "name": "id", "required": true, "type": "int"},
                    {"id": 2, "name": "vector", "required": true, "type": "fixed_list<float, 4>"}
                ]
            }
        ],
        "snapshots": [
            {
                "snapshot-id": snapshot_id,
                "timestamp-ms": 123456789,
                "manifest-list": ml_name,
                "summary": {"operation": "append"}
            }
        ],
        "partition-specs": [{"spec-id": 0, "fields": []}],
        "default-spec-id": 0,
        "last-partition-id": 1000,
        "properties": {}
    })
}
