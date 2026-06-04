// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::{Context, Result};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use std::sync::Arc;

/// Convert Iceberg JSON schema to Arrow Schema
pub fn iceberg_json_to_arrow_schema(
    schema_json: &serde_json::Value,
) -> Result<arrow::datatypes::SchemaRef> {
    let fields_json = schema_json
        .get("fields")
        .and_then(|f| f.as_array())
        .ok_or_else(|| anyhow::anyhow!("Invalid Iceberg schema: missing fields"))?;

    let mut fields = Vec::new();

    for field in fields_json {
        let name = field
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("unknown");
        let type_json = field
            .get("type")
            .ok_or_else(|| anyhow::anyhow!("Field missing type"))?;
        let required = field
            .get("required")
            .and_then(|r| r.as_bool())
            .unwrap_or(false);
        let id = field.get("id").and_then(|i| i.as_i64()).unwrap_or(0);

        let dt = convert_iceberg_type_to_arrow(type_json)?;
        let mut arrow_field = Field::new(name, dt, !required);

        // Store Iceberg ID in metadata
        if id > 0 {
            arrow_field.set_metadata(std::collections::HashMap::from([(
                "iceberg.id".to_string(),
                id.to_string(),
            )]));
        }

        fields.push(arrow_field);
    }

    Ok(Arc::new(Schema::new(fields)))
}

fn convert_iceberg_type_to_arrow(type_json: &serde_json::Value) -> Result<DataType> {
    if let Some(type_str) = type_json.as_str() {
        match type_str {
            "boolean" => Ok(DataType::Boolean),
            "int" => Ok(DataType::Int32),
            "long" => Ok(DataType::Int64),
            "float" => Ok(DataType::Float32),
            "double" => Ok(DataType::Float64),
            "string" => Ok(DataType::Utf8),
            "binary" | "fixed" => Ok(DataType::Binary), // Simplified
            "date" => Ok(DataType::Date32),
            "timestamp" => Ok(DataType::Timestamp(TimeUnit::Microsecond, None)),
            "timestamptz" => Ok(DataType::Timestamp(
                TimeUnit::Microsecond,
                Some("UTC".into()),
            )),
            "uuid" => Ok(DataType::Utf8), // Arrow doesn't have native UUID
            _ => {
                // Check for complex types like list, map, struct which might be encoded as strings in some contexts?
                // Usually they are objects.
                Err(anyhow::anyhow!("Unsupported primitive type: {}", type_str))
            }
        }
    } else if let Some(obj) = type_json.as_object() {
        // Complex types: list, map, struct
        let type_name = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match type_name {
            "struct" => {
                let fields = obj
                    .get("fields")
                    .and_then(|f| f.as_array())
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                let mut arrow_fields = Vec::new();
                for f in fields {
                    let name = f.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
                    let field_type = f.get("type").context("Missing type")?;
                    let required = f.get("required").and_then(|r| r.as_bool()).unwrap_or(false);
                    arrow_fields.push(arrow::datatypes::Field::new(
                        name,
                        convert_iceberg_type_to_arrow(field_type)?,
                        !required,
                    ));
                }
                Ok(DataType::Struct(arrow_fields.into()))
            }
            "list" => {
                let element_type = obj
                    .get("element")
                    .or_else(|| obj.get("element-type"))
                    .context("Missing element")?; // 'element' or 'element-type'
                let required = obj
                    .get("element-required")
                    .and_then(|r| r.as_bool())
                    .unwrap_or(true); // Default true?
                let dt = convert_iceberg_type_to_arrow(element_type)?;
                Ok(DataType::List(std::sync::Arc::new(
                    arrow::datatypes::Field::new("item", dt, !required),
                )))
            }
            "decimal" => {
                let precision = obj.get("precision").and_then(|p| p.as_u64()).unwrap_or(38) as u8;
                let scale = obj.get("scale").and_then(|s| s.as_u64()).unwrap_or(10) as i8;
                Ok(DataType::Decimal128(precision, scale))
            }
            "map" => {
                let key_type = obj
                    .get("key")
                    .or_else(|| obj.get("key-type"))
                    .context("Missing key-type")?;
                let value_type = obj
                    .get("value")
                    .or_else(|| obj.get("value-type"))
                    .context("Missing value-type")?;
                let value_required = obj
                    .get("value-required")
                    .and_then(|r| r.as_bool())
                    .unwrap_or(true);

                let kt = convert_iceberg_type_to_arrow(key_type)?;
                let vt = convert_iceberg_type_to_arrow(value_type)?;

                Ok(DataType::Map(
                    std::sync::Arc::new(Field::new(
                        "entries",
                        DataType::Struct(
                            vec![
                                Field::new("key", kt, false), // Keys are usually non-nullable in Iceberg
                                Field::new("value", vt, !value_required),
                            ]
                            .into(),
                        ),
                        false,
                    )),
                    false, // Not sorted by default
                ))
            }
            _ => Err(anyhow::anyhow!("Unsupported complex type: {:?}", type_json)),
        }
    } else {
        Err(anyhow::anyhow!("Invalid types definition: {:?}", type_json))
    }
}

/// Convert Iceberg partition spec to HyperStream PartitionSpec
pub fn iceberg_partition_spec_to_hyperstream(
    spec_json: &serde_json::Value,
) -> Result<crate::core::manifest::PartitionSpec> {
    use crate::core::manifest::{PartitionField, PartitionSpec};

    let spec_id = spec_json
        .get("spec-id")
        .and_then(|id| id.as_i64())
        .unwrap_or(0) as i32;
    let fields_json = spec_json
        .get("fields")
        .and_then(|f| f.as_array())
        .ok_or_else(|| anyhow::anyhow!("Invalid Iceberg partition spec: missing fields"))?;

    let mut fields = Vec::new();
    for field in fields_json {
        let name = field
            .get("name")
            .and_then(|n| n.as_str())
            .unwrap_or("unknown")
            .to_string();
        let source_id = field
            .get("source-id")
            .and_then(|id| id.as_i64())
            .unwrap_or(0) as i32;
        let transform = field
            .get("transform")
            .and_then(|t| t.as_str())
            .unwrap_or("identity")
            .to_string();
        let field_id = field
            .get("field-id")
            .and_then(|id| id.as_i64())
            .map(|id| id as i32);

        fields.push(PartitionField::new_single(
            source_id, field_id, name, transform,
        ));
    }

    Ok(PartitionSpec { spec_id, fields })
}
