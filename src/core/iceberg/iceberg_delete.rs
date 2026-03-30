// Copyright (c) 2026 Richard Albright. All rights reserved.


use anyhow::Result;
use arrow::array::Array;
use apache_avro::types::Record;

/// Writer for Iceberg Delete Files (Position and Equality Deletes)
pub struct IcebergDeleteWriter {
    base_path: String,
    format_version: i32,
}

impl IcebergDeleteWriter {
    pub fn new(
        base_path: String,
        format_version: i32,
    ) -> Self {
        Self {
            base_path: base_path.replace("file://", ""),
            format_version,
        }
    }

    /// Write a Position Delete File
    /// 
    /// Schema:
    /// - file_path: string (path of the data file where row is deleted)
    /// - pos: long (ordinal position of the deleted row)
    /// - row: optional (the deleted row itself, strictly optional)
    pub async fn write_position_delete(
        &self,
        partition_data: Option<(String, std::collections::HashMap<String, serde_json::Value>)>, 
        file_path_column: &arrow::array::StringArray,
        pos_column: &arrow::array::Int64Array,
    ) -> Result<crate::core::manifest::DeleteFile> {
        if file_path_column.len() != pos_column.len() {
             return Err(anyhow::anyhow!("Mismatch in file_path and pos column lengths"));
        }
        
        let file_name = format!("del-pos-{}-{}.avro", uuid::Uuid::new_v4(), self.format_version);
        let full_path = if let Some((ref part_path, _)) = partition_data {
             format!("{}/{}/{}", self.base_path, part_path, file_name)
        } else {
             format!("{}/{}", self.base_path, file_name)
        };
        
        // Construct Avro Schema for Position Deletes
        let schema_json = r#"
        {
            "type": "record",
            "name": "position_delete",
            "fields": [
                {"name": "file_path", "type": "string"},
                {"name": "pos", "type": "long"}
            ]
        }
        "#;
        let schema = apache_avro::Schema::parse_str(schema_json)?;
        
        // Write file
        let file = std::fs::File::create(&full_path)?;
        let mut writer = apache_avro::Writer::new(&schema, file);
        
        for i in 0..file_path_column.len() {
             let mut record = Record::new(&schema).unwrap();
             record.put("file_path", apache_avro::types::Value::String(file_path_column.value(i).to_string()));
             record.put("pos", apache_avro::types::Value::Long(pos_column.value(i)));
             writer.append(record)?;
        }
        
        let len = writer.flush()?;
        
        // Create DeleteFile metadata
        // Note: For Metadata, we prefer the full URI
        let mut metadata_path = full_path.clone();
        if !metadata_path.starts_with("file://") && !metadata_path.starts_with("s3://") {
             metadata_path = format!("file://{}", full_path);
        }

        let partition_values = partition_data.map(|(_, map)| map).unwrap_or_default();

        Ok(crate::core::manifest::DeleteFile {
            file_path: metadata_path,
            content: crate::core::manifest::DeleteContent::Position,
            file_size_bytes: len as i64,
            record_count: file_path_column.len() as i64,
            partition_values, 
        })
    }

    /// Write an Equality Delete File
    pub async fn write_equality_delete(
        &self,
        partition_value: Option<&str>,
        batch: &arrow::record_batch::RecordBatch,
        equality_ids: &[i32],
        table_schema: &crate::core::manifest::Schema,
    ) -> Result<crate::core::manifest::DeleteFile> {
        let file_name = format!("del-eq-{}-{}.avro", uuid::Uuid::new_v4(), self.format_version);
        let full_path = if let Some(part) = partition_value {
             format!("{}/{}/{}", self.base_path, part, file_name)
        } else {
             format!("{}/{}", self.base_path, file_name)
        };

        // 1. Construct Avro Schema based on equality IDs
        let mut fields_json = Vec::new();
        let mut column_names = Vec::new();

        for &id in equality_ids {
            let field = table_schema.fields.iter().find(|f| f.id == id)
                .ok_or_else(|| anyhow::anyhow!("Field ID {} not found in schema", id))?;
            
            column_names.push(field.name.clone());

            let avro_type = match field.type_str.as_str() {
                "Int32" | "int" => "int",
                "Int64" | "long" => "long",
                "Float32" | "float" => "float",
                "Float64" | "double" => "double",
                "Utf8" | "string" => "string",
                "Boolean" | "bool" => "boolean",
                _ => "string", // Fallback
            };

            fields_json.push(format!(r#"{{"name": "{}", "type": "{}"}}"#, field.name, avro_type));
        }

        let schema_json = format!(r#"
        {{
            "type": "record",
            "name": "equality_delete",
            "fields": [{}]
        }}
        "#, fields_json.join(","));

        let schema = apache_avro::Schema::parse_str(&schema_json)?;
        
        // 2. Write file
        let file = std::fs::File::create(&full_path)?;
        let mut writer = apache_avro::Writer::new(&schema, file);

        for i in 0..batch.num_rows() {
            let mut record = Record::new(&schema).unwrap();
            for &id in equality_ids.iter() {
                let field = &table_schema.fields.iter().find(|f| f.id == id).unwrap();
                let col = batch.column_by_name(&field.name)
                    .ok_or_else(|| anyhow::anyhow!("Column {} not found in batch", field.name))?;
                
                let value = self.arrow_to_avro_value(col, i)?;
                record.put(&field.name, value);
            }
            writer.append(record)?;
        }

        let len = writer.flush()?;

        let mut metadata_path = full_path.clone();
        if !metadata_path.starts_with("file://") && !metadata_path.starts_with("s3://") {
             metadata_path = format!("file://{}", full_path);
        }

        Ok(crate::core::manifest::DeleteFile {
            file_path: metadata_path,
            content: crate::core::manifest::DeleteContent::Equality { equality_ids: equality_ids.to_vec() },
            file_size_bytes: len as i64,
            record_count: batch.num_rows() as i64,
            partition_values: std::collections::HashMap::new(),
        })
    }

    fn arrow_to_avro_value(&self, array: &arrow::array::ArrayRef, i: usize) -> Result<apache_avro::types::Value> {
        use arrow::array::*;
        use apache_avro::types::Value;

        if array.is_null(i) {
            return Ok(Value::Null);
        }

        let val = match array.data_type() {
            arrow::datatypes::DataType::Int32 => {
                let a = array.as_any().downcast_ref::<Int32Array>().unwrap();
                Value::Int(a.value(i))
            },
            arrow::datatypes::DataType::Int64 => {
                let a = array.as_any().downcast_ref::<Int64Array>().unwrap();
                Value::Long(a.value(i))
            },
            arrow::datatypes::DataType::Float32 => {
                let a = array.as_any().downcast_ref::<Float32Array>().unwrap();
                Value::Float(a.value(i))
            },
            arrow::datatypes::DataType::Float64 => {
                let a = array.as_any().downcast_ref::<Float64Array>().unwrap();
                Value::Double(a.value(i))
            },
            arrow::datatypes::DataType::Utf8 => {
                let a = array.as_any().downcast_ref::<StringArray>().unwrap();
                Value::String(a.value(i).to_string())
            },
            arrow::datatypes::DataType::Boolean => {
                let a = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                Value::Boolean(a.value(i))
            },
            _ => Value::String(format!("{:?}", array)), // Fallback
        };
        Ok(val)
    }
}
