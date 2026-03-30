// Copyright (c) 2026 Richard Albright. All rights reserved.

use apache_avro::{Schema, Writer, types::{Record, Value}, Codec};
use std::fs::File;
use std::env;
use std::sync::Arc;
use arrow::array::{StringArray, Int64Array};
use arrow::datatypes::{Schema as ArrowSchema, Field as ArrowField, DataType};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::file::properties::WriterProperties;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 {
        eprintln!("Usage: {} <output_dir> <base_uri>", args[0]);
        std::process::exit(1);
    }
    let output_dir = &args[1];
    let base_uri = &args[2];
    std::fs::create_dir_all(output_dir)?;

    // --- 0. Generate Parquet Data File ---
    let data_file_name = "data1.parquet";
    let data_file_path = format!("{}/{}", output_dir, data_file_name);
    // Construct the path that will be stored in the manifest and thus in the delete file.
    // If base_uri is "file://generated_manifests" (relative), then the path is "generated_manifests/data1.parquet".
    // We strip "file://" prefix logic in generator for the content of delete file?
    // Let's assume input base_uri is proper.
    
    // Check if base_uri starts with file://, strip it for internal logic if needed
    let clean_base = base_uri.strip_prefix("file://").unwrap_or(base_uri);
    
    // The path stored in manifest and delete file
    let _data_rel_path = format!("{}/{}", clean_base, data_file_name); 

    // Schema: category (string)
    let schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("category", DataType::Utf8, true),
    ]));
    
    let mut categories = Vec::new();
    for i in 0..100 {
        categories.push(format!("row_{}", i));
    }
    let category_array = StringArray::from(categories);
    let batch = RecordBatch::try_new(schema.clone(), vec![Arc::new(category_array)])?;
    
    let file = File::create(&data_file_path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
    writer.write(&batch)?;
    writer.close()?;
    let data_file_len = std::fs::metadata(&data_file_path)?.len();
    
    // --- 0b. Generate Parquet Position Delete File ---
    let delete_file_name = "delete1.parquet";
    let delete_file_path = format!("{}/{}", output_dir, delete_file_name);
    
    // Schema: file_path (string), pos (int64)
    let del_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("file_path", DataType::Utf8, false),
        ArrowField::new("pos", DataType::Int64, false),
    ]));
    
    // Delete row 0 and row 5
    // The file_path must match EXACTLY what is in the manifest entry.
    // In manifest we put `base_uri/data1.parquet` (full uri or relative).
    // Let's put the full `base_uri/data1.parquet` string as passed in args.
    
    let target_path_in_manifest = format!("{}/{}", base_uri, data_file_name);
    
    let file_paths = StringArray::from(vec![target_path_in_manifest.clone(), target_path_in_manifest.clone()]);
    let positions = Int64Array::from(vec![0, 5]);
    
    let del_batch = RecordBatch::try_new(del_schema.clone(), vec![Arc::new(file_paths), Arc::new(positions)])?;
    
    let file = File::create(&delete_file_path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, del_schema, Some(props))?;
    writer.write(&del_batch)?;
    writer.close()?;
    let delete_file_len = std::fs::metadata(&delete_file_path)?.len();

    // --- 0c. Generate Parquet Equality Delete File ---
    let eq_delete_file_name = "delete_eq.parquet";
    let eq_delete_file_path = format!("{}/{}", output_dir, eq_delete_file_name);
    
    // Schema: category (string). Match name with data file.
    let eq_del_schema = Arc::new(ArrowSchema::new(vec![
        ArrowField::new("category", DataType::Utf8, false),
    ]));
    
    // Delete row_10 using equality
    let categories = StringArray::from(vec!["row_10"]);
    let eq_batch = RecordBatch::try_new(eq_del_schema.clone(), vec![Arc::new(categories)])?;
    
    let file = File::create(&eq_delete_file_path)?;
    let props = WriterProperties::builder().build();
    let mut writer = ArrowWriter::try_new(file, eq_del_schema, Some(props))?;
    writer.write(&eq_batch)?;
    writer.close()?;
    let eq_delete_file_len = std::fs::metadata(&eq_delete_file_path)?.len();

    // --- 1. Define Manifest Schemas ---
    let manifest_schema_str = r#"
    {
        "type": "record",
        "name": "manifest_entry",
        "fields": [
            {"name": "status", "type": "int"},
            {"name": "snapshot_id", "type": ["null", "long"], "default": null},
            {"name": "data_file", "type": {
                "type": "record",
                "name": "data_file",
                "fields": [
                    {"name": "content", "type": "int"},
                    {"name": "file_path", "type": "string"},
                    {"name": "file_format", "type": "string"},
                    {"name": "partition", "type": {
                        "type": "record",
                        "name": "r102",
                        "fields": [
                            {"name": "category", "type": ["null", "string"], "default": null}
                        ]
                    }},
                    {"name": "record_count", "type": "long"},
                    {"name": "file_size_in_bytes", "type": "long"},
                    {"name": "equality_ids", "type": ["null", {"type": "array", "items": "int"}], "default": null}
                ]
            }}
        ]
    }"#;
    let manifest_schema = Schema::parse_str(manifest_schema_str).unwrap();

    let manifest_list_schema_str = r#"
    {
        "type": "record",
        "name": "manifest_file",
        "fields": [
            {"name": "manifest_path", "type": "string"},
            {"name": "manifest_length", "type": "long"},
            {"name": "partition_spec_id", "type": "int"},
            {"name": "content", "type": "int"},
            {"name": "added_snapshot_id", "type": "long"},
            {"name": "added_data_files_count", "type": "int", "default": 0},
            {"name": "existing_data_files_count", "type": "int", "default": 0},
            {"name": "deleted_data_files_count", "type": "int", "default": 0},
            {"name": "added_rows_count", "type": "long", "default": 0},
            {"name": "existing_rows_count", "type": "long", "default": 0},
            {"name": "deleted_rows_count", "type": "long", "default": 0}
        ]
    }"#;
    let manifest_list_schema = Schema::parse_str(manifest_list_schema_str).unwrap();

    // --- 2. Write Data Manifest (content=0) ---
    let data_manifest_path = format!("{}/manifest-data.avro", output_dir);
    let file = File::create(&data_manifest_path)?;
    let mut writer = Writer::with_codec(&manifest_schema, file, Codec::Null);

    let mut record = Record::new(&manifest_schema).unwrap();
    record.put("status", 1); // ADDED
    record.put("snapshot_id", Value::Union(1, Box::new(Value::Long(1))));
    
    let partition_val = Value::Record(vec![
        ("category".to_string(), Value::Union(1, Box::new(Value::String("electronics".to_string()))))
    ]);

    let data_file_val = Value::Record(vec![
        ("content".to_string(), Value::Int(0)), // DATA
        ("file_path".to_string(), Value::String(target_path_in_manifest.clone())),
        ("file_format".to_string(), Value::String("PARQUET".to_string())),
        ("partition".to_string(), partition_val.clone()),
        ("record_count".to_string(), Value::Long(100)),
        ("file_size_in_bytes".to_string(), Value::Long(data_file_len as i64)),
        ("equality_ids".to_string(), Value::Union(0, Box::new(Value::Null))),
    ]);
    
    record.put("data_file", data_file_val);
    writer.append(record)?;
    writer.flush()?;
    let data_manifest_len = std::fs::metadata(&data_manifest_path)?.len();

    // --- 3. Write Delete Manifest (content=1, Position Delete) ---
    // Note: Previous version wrote content=2 (Equality). Switching to 1 (Position)
    let delete_manifest_path = format!("{}/manifest-delete.avro", output_dir);
    let file = File::create(&delete_manifest_path)?;
    let mut writer = Writer::with_codec(&manifest_schema, file, Codec::Null);

    let mut record = Record::new(&manifest_schema).unwrap();
    record.put("status", 1); // ADDED
    record.put("snapshot_id", Value::Union(1, Box::new(Value::Long(1))));
    
    let delete_file_val = Value::Record(vec![
        ("content".to_string(), Value::Int(1)), // POSITION DELETE
        ("file_path".to_string(), Value::String(format!("{}/{}", base_uri, delete_file_name))),
        ("file_format".to_string(), Value::String("PARQUET".to_string())),
        ("partition".to_string(), partition_val.clone()), // SAME PARTITION
        ("record_count".to_string(), Value::Long(2)),
        ("file_size_in_bytes".to_string(), Value::Long(delete_file_len as i64)),
        ("equality_ids".to_string(), Value::Union(0, Box::new(Value::Null))),
    ]);
    
    record.put("data_file", delete_file_val);
    writer.append(record)?;

    // Entry 2: Equality Delete
    let eq_delete_file_val = Value::Record(vec![
        ("content".to_string(), Value::Int(2)), // EQUALITY DELETE
        ("file_path".to_string(), Value::String(format!("{}/{}", base_uri, eq_delete_file_name))),
        ("file_format".to_string(), Value::String("PARQUET".to_string())),
        ("partition".to_string(), partition_val.clone()), // SAME PARTITION
        ("record_count".to_string(), Value::Long(1)),
        ("file_size_in_bytes".to_string(), Value::Long(eq_delete_file_len as i64)),
        ("equality_ids".to_string(), Value::Union(1, Box::new(Value::Array(vec![Value::Int(1)])))), // ID 1
    ]);
    
    let mut record = Record::new(&manifest_schema).unwrap();
    record.put("status", 1);
    record.put("snapshot_id", Value::Union(1, Box::new(Value::Long(1))));
    record.put("data_file", eq_delete_file_val);
    writer.append(record)?;

    writer.flush()?;
    let delete_manifest_len = std::fs::metadata(&delete_manifest_path)?.len();

    // --- 4. Write Manifest List ---
    let list_path = format!("{}/snap-1.avro", output_dir);
    let file = File::create(&list_path)?;
    let mut writer = Writer::with_codec(&manifest_list_schema, file, Codec::Null);

    // Entry 1: Data Manifest
    let mut record = Record::new(&manifest_list_schema).unwrap();
    record.put("manifest_path", Value::String(format!("{}/manifest-data.avro", base_uri))); 
    record.put("manifest_length", Value::Long(data_manifest_len as i64));
    record.put("partition_spec_id", 0);
    record.put("content", 0); // DATA
    record.put("added_snapshot_id", 1i64);
    record.put("added_data_files_count", 0i32); // skipped exact counts
    record.put("existing_data_files_count", 0i32);
    record.put("deleted_data_files_count", 0i32);
    record.put("added_rows_count", 0i64);
    record.put("existing_rows_count", 0i64);
    record.put("deleted_rows_count", 0i64);
    writer.append(record)?;

    // Entry 2: Delete Manifest
    let mut record = Record::new(&manifest_list_schema).unwrap();
    record.put("manifest_path", Value::String(format!("{}/manifest-delete.avro", base_uri)));
    record.put("manifest_length", Value::Long(delete_manifest_len as i64));
    record.put("partition_spec_id", 0);
    record.put("content", 1); // DELETES
    record.put("added_snapshot_id", 1i64);
    record.put("added_data_files_count", 0i32);
    record.put("existing_data_files_count", 0i32);
    record.put("deleted_data_files_count", 0i32);
    record.put("added_rows_count", 0i64);
    record.put("existing_rows_count", 0i64);
    record.put("deleted_rows_count", 0i64);
    writer.append(record)?;

    writer.flush()?;

    println!("Generated manifests and parquet files in {}", output_dir);
    Ok(())
}
