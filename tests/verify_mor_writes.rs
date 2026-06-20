// Copyright (c) 2026 Richard Albright. All rights reserved.

#[cfg(test)]
mod tests {
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use hyperstreamdb::Table;
    use std::fs;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_mor_position_deletes() -> anyhow::Result<()> {
        let test_dir = "/tmp/hyperstream_mor_test";
        let _ = fs::remove_dir_all(test_dir);
        fs::create_dir_all(test_dir)?;

        let uri = format!("file://{}", test_dir);

        // 1. Create Table
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("data", DataType::Utf8, false),
        ]));

        let table = Table::create_async(uri.clone(), schema.clone()).await?;

        // 2. Insert Data
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
                Arc::new(StringArray::from(vec!["A", "B", "C", "D", "E"])),
            ],
        )?;
        table.write_async(vec![batch]).await?;
        table.commit_async().await?;

        // 3. Delete Rows (ID = 2 and ID = 4)
        // This should generate a Position Delete File
        table.delete_async("id = 2").await?;
        table.delete_async("id = 4").await?;

        // 4. Verify Delete Files exist
        let paths = fs::read_dir(test_dir)?;
        let mut found_del_file = false;

        for path in paths {
            let p = path?.path();
            let name = p.file_name().unwrap().to_str().unwrap();
            // Look for del-pos-*.avro
            if name.starts_with("del-pos-") && name.ends_with(".avro") {
                found_del_file = true;
                println!("Found Delete File: {}", name);
            }
        }

        assert!(found_del_file, "Position Delete File should be created");

        // 5. Verify Manifest contains Delete Entry
        let manifest_manager = hyperstreamdb::core::manifest::ManifestManager::new(
            hyperstreamdb::core::storage::create_object_store(&uri)?,
            "",
            &uri,
        );
        let (_, entries, _) = manifest_manager.load_latest_full().await?;

        let mut delete_entries_count = 0;
        for entry in &entries {
            delete_entries_count += entry.delete_files.len();
        }

        assert!(
            delete_entries_count >= 1,
            "Manifest should reference Delete Files"
        );

        // Verify Manifest Avro file content (optional, deeper check)
        // We trust load_latest_full if it parsed it back?
        // Actually load_latest_full reads the JSON/Avro manifest.
        // If it successfully loaded 'delete_files', it means we wrote them correctly to Avro
        // AND read them back correctly.

        Ok(())
    }
}
