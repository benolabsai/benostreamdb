// Copyright (c) 2026 Richard Albright. All rights reserved.

use hyperstreamdb::core::table::Table;
use hyperstreamdb::core::manifest::ManifestManager;
use hyperstreamdb::core::iceberg::iceberg_delete::IcebergDeleteWriter;
use arrow::array::{Int32Array, StringArray};
use arrow::record_batch::RecordBatch;
use std::sync::Arc;
use tempfile::tempdir;

#[tokio::test]
async fn test_mor_mixed_deletes_avro() -> anyhow::Result<()> {
    let dir = tempdir()?;
    let uri = format!("file://{}", dir.path().display());
    
    // 1. Create Table and Insert Data
    let schema = Arc::new(arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int32, false),
        arrow::datatypes::Field::new("name", arrow::datatypes::DataType::Utf8, false),
    ]));
    
    let table = Table::create_async(uri.clone(), schema.clone()).await?;
    
    let batch = RecordBatch::try_new(schema.clone(), vec![
        Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
        Arc::new(StringArray::from(vec!["A", "B", "C", "D", "E"])),
    ])?;
    
    table.write_async(vec![batch.clone()]).await?;
    table.flush_async().await?;
    
    // 2. Perform Position Delete (Delete row 0: id=1)
    table.delete_async("id = 1").await?;
    
    // 3. Perform Equality Delete (Delete id=3)
    let manifest_manager = ManifestManager::new(table.store.clone(), "", &uri);
    let (_manifest, all_entries, _) = manifest_manager.load_latest_full().await?;
    let entry = all_entries.first().expect("Should have one data file");
    
    let delete_writer = IcebergDeleteWriter::new(
        uri.clone(),
        2,
    );
    
    let eq_batch = RecordBatch::try_new(
        Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("id", arrow::datatypes::DataType::Int32, false),
        ])),
        vec![Arc::new(Int32Array::from(vec![3]))]
    )?;
    
    let table_schema_manifest = hyperstreamdb::core::manifest::Schema {
        schema_id: 0,
        fields: vec![
            hyperstreamdb::core::manifest::SchemaField { id: 1, name: "id".to_string(), type_str: "int".to_string(), required: true, fields: vec![], initial_default: None, write_default: None },
            hyperstreamdb::core::manifest::SchemaField { id: 2, name: "name".to_string(), type_str: "string".to_string(), required: true, fields: vec![], initial_default: None, write_default: None },
        ],
        identifier_field_ids: vec![],
    };

    let eq_delete_file = delete_writer.write_equality_delete(
        None,
        &eq_batch,
        &[1], // field id 1 is "id"
        &table_schema_manifest,
    ).await?;
    
    let mut updated_entry = entry.clone();
    updated_entry.delete_files.push(eq_delete_file);
    
    manifest_manager.commit(&[updated_entry], std::slice::from_ref(&entry.file_path), hyperstreamdb::core::manifest::CommitMetadata::default()).await?;
    
    // 4. Verify Reads
    let results = table.read_filter_async(vec![], None, None).await?;
    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    
    println!("Final Rows: {}", total_rows);
    for batch in &results {
        let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..batch.num_rows() {
            println!("Row: id={}", ids.value(i));
        }
    }
    
    assert_eq!(total_rows, 3, "Should have 3 rows remaining");
    
    // Ensure ids 1 and 3 are missing
    for batch in results {
        let ids = batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        for i in 0..batch.num_rows() {
            let id = ids.value(i);
            assert!(id != 1, "ID 1 should be deleted (Position)");
            assert!(id != 3, "ID 3 should be deleted (Equality)");
        }
    }
    
    Ok(())
}
