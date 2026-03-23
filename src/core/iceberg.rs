use anyhow::Result;
use apache_avro::{Reader, types::Value as AvroValue};
use base64::Engine;
use std::io::Read;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
pub mod iceberg_delete;
use arrow::record_batch::RecordBatch;
use arrow::datatypes::Field;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct IcebergTableMetadata {
    pub format_version: i32,
    pub table_uuid: String,
    pub location: String,
    pub last_sequence_number: i64,
    pub last_updated_ms: i64,
    pub current_snapshot_id: Option<i64>,
    pub snapshots: Vec<IcebergSnapshot>,
    pub schemas: Vec<serde_json::Value>,
    pub current_schema_id: i32,
    pub partition_specs: Vec<serde_json::Value>,
    pub default_spec_id: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub struct IcebergSnapshot {
    pub snapshot_id: i64,
    pub timestamp_ms: i64,
    pub manifest_list: String,
    pub summary: std::collections::HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct IcebergManifestListEntry {
    pub manifest_path: String,
    pub manifest_length: i64,
    pub partition_spec_id: i32,
    pub added_snapshot_id: i64,
    pub content: i32,
    pub sequence_number: i64,
    pub min_sequence_number: i64,
    pub added_files_count: i32,
    pub existing_files_count: i32,
    pub deleted_files_count: i32,
    pub added_rows_count: i64,
    pub existing_rows_count: i64,
    pub deleted_rows_count: i64,
}

#[derive(Debug, Clone)]
pub struct IcebergManifestEntry {
    pub status: i32,
    pub snapshot_id: Option<i64>,
    pub data_file: IcebergDataFile,
}

#[derive(Debug, Clone)]
pub struct IcebergDataFile {
    pub content: i32,
    pub file_path: String,
    pub file_format: String,
    pub partition: Vec<serde_json::Value>,
    pub record_count: i64,
    pub file_size_in_bytes: i64,
    // Column stats (Field ID -> Value)
    pub column_sizes: Option<std::collections::HashMap<i32, i64>>,
    pub value_counts: Option<std::collections::HashMap<i32, i64>>,
    pub null_value_counts: Option<std::collections::HashMap<i32, i64>>,
    pub nan_value_counts: Option<std::collections::HashMap<i32, i64>>,
    pub lower_bounds: Option<std::collections::HashMap<i32, Vec<u8>>>,
    pub upper_bounds: Option<std::collections::HashMap<i32, Vec<u8>>>,
    pub equality_ids: Option<Vec<i32>>,
    // V3 Deletion Vector fields
    pub referenced_data_file: Option<String>,
    pub content_offset: Option<i64>,
    pub content_size_in_bytes: Option<i64>,
}

pub fn read_manifest_list<R: Read>(reader: R) -> Result<Vec<IcebergManifestListEntry>> {
    let avro_reader = Reader::new(reader)?;
    let mut entries = Vec::new();

    for record in avro_reader {
        let value = record?;
        if let AvroValue::Record(fields) = value {
            let mut manifest_path = String::new();
            let mut manifest_length = 0;
            let mut partition_spec_id = 0;
            let mut added_snapshot_id = 0;
            let mut content = 0;
            let mut sequence_number = 0;
            let mut min_sequence_number = 0;
            let mut added_files_count = 0;
            let mut existing_files_count = 0;
            let mut deleted_files_count = 0;
            let mut added_rows_count = 0;
            let mut existing_rows_count = 0;
            let mut deleted_rows_count = 0;

            for (name, val) in fields {
                match name.as_str() {
                    "manifest_path" => if let AvroValue::String(s) = val { manifest_path = s; },
                    "manifest_length" => if let AvroValue::Long(s) = val { manifest_length = s; },
                    "partition_spec_id" => if let AvroValue::Int(s) = val { partition_spec_id = s; },
                    "added_snapshot_id" => if let AvroValue::Long(s) = val { added_snapshot_id = s; },
                    "content" => if let AvroValue::Int(s) = val { content = s; },
                    "sequence_number" => if let AvroValue::Long(s) = val { sequence_number = s; },
                    "min_sequence_number" => if let AvroValue::Long(s) = val { min_sequence_number = s; },
                    "added_data_files_count" => if let AvroValue::Int(s) = val { added_files_count = s; },
                    "existing_data_files_count" => if let AvroValue::Int(s) = val { existing_files_count = s; },
                    "deleted_data_files_count" => if let AvroValue::Int(s) = val { deleted_files_count = s; },
                    "added_rows_count" => if let AvroValue::Long(s) = val { added_rows_count = s; },
                    "existing_rows_count" => if let AvroValue::Long(s) = val { existing_rows_count = s; },
                    "deleted_rows_count" => if let AvroValue::Long(s) = val { deleted_rows_count = s; },
                    _ => {}
                }
            }

            entries.push(IcebergManifestListEntry {
                manifest_path,
                manifest_length,
                partition_spec_id,
                added_snapshot_id,
                content,
                sequence_number,
                min_sequence_number,
                added_files_count,
                existing_files_count,
                deleted_files_count,
                added_rows_count,
                existing_rows_count,
                deleted_rows_count,
            });
        }
    }
    Ok(entries)
}

pub fn read_manifest<R: Read>(reader: R) -> Result<Vec<IcebergManifestEntry>> {
    let avro_reader = Reader::new(reader)?;
    let mut entries = Vec::new();

    for record in avro_reader {
        let value = record?;
        if let AvroValue::Record(fields) = value {
            let mut status = 0;
            let mut snapshot_id = None;
            let mut data_file = None;

            for (name, val) in fields {
                match name.as_str() {
                    "status" => if let AvroValue::Int(s) = val { status = s; },
                    "snapshot_id" => if let AvroValue::Long(s) = val { snapshot_id = Some(s); },
                    "data_file" => {
                        if let AvroValue::Record(df_fields) = val {
                            data_file = Some(parse_data_file(df_fields)?);
                        }
                    },
                    _ => {}
                }
            }

            if let Some(df) = data_file {
                entries.push(IcebergManifestEntry {
                    status,
                    snapshot_id,
                    data_file: df,
                });
            }
        }
    }
    Ok(entries)
}
fn parse_data_file(fields: Vec<(String, AvroValue)>) -> Result<IcebergDataFile> {
    let mut content = 0;
    let mut file_path = String::new();
    let mut file_format = String::new();
    let mut partition = Vec::new();
    let mut record_count = 0;
    let mut file_size_in_bytes = 0;
    
    let mut column_sizes = None;
    let mut value_counts = None;
    let mut null_value_counts = None;
    let mut nan_value_counts = None;
    let mut lower_bounds = None;
    let mut upper_bounds = None;
    let mut equality_ids = None;
    
    // V3 Deletion Vector fields
    let mut referenced_data_file = None;
    let mut content_offset = None;
    let mut content_size_in_bytes = None;

    for (name, val) in fields {
        match name.as_str() {
            "content" => if let AvroValue::Int(c) = val { content = c; },
            "file_path" => if let AvroValue::String(s) = val { file_path = s; },
            "file_format" => if let AvroValue::String(s) = val { file_format = s; },
            "partition" => if let AvroValue::Record(p_fields) = val {
                partition = p_fields.into_iter().map(|(_, v)| avro_to_json(v)).collect();
            },
            "record_count" => if let AvroValue::Long(c) = val { record_count = c; },
            "file_size_in_bytes" => if let AvroValue::Long(s) = val { file_size_in_bytes = s; },
            "column_sizes" => column_sizes = parse_map_int_long(val),
            "value_counts" => value_counts = parse_map_int_long(val),
            "null_value_counts" => null_value_counts = parse_map_int_long(val),
            "nan_value_counts" => nan_value_counts = parse_map_int_long(val),
            "lower_bounds" => lower_bounds = parse_map_int_bytes(val),
            "upper_bounds" => upper_bounds = parse_map_int_bytes(val),
            "equality_ids" => {
                let inner = if let AvroValue::Union(_, b) = val { *b } else { val };
                if let AvroValue::Array(items) = inner {
                    let mut ids = Vec::new();
                    for item in items {
                        if let AvroValue::Int(id) = item {
                            ids.push(id);
                        }
                    }
                    equality_ids = Some(ids);
                }
            },
            // V3 Deletion Vector fields
            "referenced_data_file" => {
                let inner = if let AvroValue::Union(_, b) = val { *b } else { val };
                if let AvroValue::String(s) = inner {
                    referenced_data_file = Some(s);
                }
            },
            "content_offset" => {
                let inner = if let AvroValue::Union(_, b) = val { *b } else { val };
                if let AvroValue::Long(o) = inner {
                    content_offset = Some(o);
                }
            },
            "content_size_in_bytes" => {
                let inner = if let AvroValue::Union(_, b) = val { *b } else { val };
                if let AvroValue::Long(s) = inner {
                    content_size_in_bytes = Some(s);
                }
            },
            _ => {}
        }
    }

    Ok(IcebergDataFile {
        content,
        file_path,
        file_format,
        partition,
        record_count,
        file_size_in_bytes,
        column_sizes,
        value_counts,
        null_value_counts,
        nan_value_counts,
        lower_bounds,
        upper_bounds,
        equality_ids,
        referenced_data_file,
        content_offset,
        content_size_in_bytes,
    })
}

fn parse_map_int_long(val: AvroValue) -> Option<std::collections::HashMap<i32, i64>> {
    if let AvroValue::Array(items) = val {
        let mut map = std::collections::HashMap::new();
        for item in items {
            if let AvroValue::Record(fields) = item {
                let mut key = 0;
                let mut value = 0;
                for (k, v) in fields {
                    match k.as_str() {
                        "key" => if let AvroValue::Int(i) = v { key = i; },
                        "value" => if let AvroValue::Long(l) = v { value = l; },
                        _ => {}
                    }
                }
                map.insert(key, value);
            }
        }
        return Some(map);
    }
    None
}

fn parse_map_int_bytes(val: AvroValue) -> Option<std::collections::HashMap<i32, Vec<u8>>> {
    if let AvroValue::Array(items) = val {
        let mut map = std::collections::HashMap::new();
        for item in items {
            if let AvroValue::Record(fields) = item {
                let mut key = 0;
                let mut value = Vec::new();
                for (k, v) in fields {
                    match k.as_str() {
                        "key" => if let AvroValue::Int(i) = v { key = i; },
                        "value" => if let AvroValue::Bytes(b) = v { value = b; },
                        _ => {}
                    }
                }
                map.insert(key, value);
            }
        }
        return Some(map);
    }
    None
}

pub fn avro_to_json(val: AvroValue) -> serde_json::Value {
    match val {
        AvroValue::Null => serde_json::Value::Null,
        AvroValue::Boolean(b) => serde_json::json!(b),
        AvroValue::Int(i) => serde_json::json!(i),
        AvroValue::Long(l) => serde_json::json!(l),
        AvroValue::Float(f) => serde_json::json!(f),
        AvroValue::Double(d) => serde_json::json!(d),
        AvroValue::String(s) => serde_json::json!(s),
        AvroValue::Bytes(b) => serde_json::json!(b),
        AvroValue::Union(_, b) => avro_to_json(*b),
        _ => serde_json::Value::Null, 
    }
}

pub fn decode_iceberg_value(type_json: &serde_json::Value, bytes: &[u8]) -> crate::core::manifest::ManifestValue {
    use crate::core::manifest::ManifestValue;
    
    let type_str = if let Some(s) = type_json.as_str() {
        s
    } else if let Some(obj) = type_json.as_object() {
        obj.get("type").and_then(|t| t.as_str()).unwrap_or("unknown")
    } else {
        "unknown"
    };

    match type_str {
        "boolean" => {
            if !bytes.is_empty() {
                ManifestValue::Boolean(bytes[0] != 0)
            } else {
                ManifestValue::Null
            }
        }
        "int" | "date" => {
            if bytes.len() >= 4 {
                ManifestValue::Int32(i32::from_le_bytes(bytes[0..4].try_into().unwrap()))
            } else {
                ManifestValue::Null
            }
        }
        "long" | "timestamp" | "timestamptz" => {
            if bytes.len() >= 8 {
                ManifestValue::Int64(i64::from_le_bytes(bytes[0..8].try_into().unwrap()))
            } else {
                ManifestValue::Null
            }
        }
        "float" => {
            if bytes.len() >= 4 {
                ManifestValue::Float32(f32::from_le_bytes(bytes[0..4].try_into().unwrap()))
            } else {
                ManifestValue::Null
            }
        }
        "double" => {
            if bytes.len() >= 8 {
                ManifestValue::Float64(f64::from_le_bytes(bytes[0..8].try_into().unwrap()))
            } else {
                ManifestValue::Null
            }
        }
        "string" | "uuid" => {
            if let Ok(s) = std::str::from_utf8(bytes) {
                ManifestValue::String(s.to_string())
            } else {
                ManifestValue::String(base64::engine::general_purpose::STANDARD.encode(bytes))
            }
        }
        "binary" | "fixed" => {
            ManifestValue::String(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
        _ => {
            // Best effort fallback
            parse_avro_value_bytes(bytes)
        }
    }
}

pub fn parse_avro_value_bytes(bytes: &[u8]) -> crate::core::manifest::ManifestValue {
    // Iceberg stores min/max bounds as the serialized binary of the type.
    // Without strict type info passed down, we might struggle.
    // For now, let's attempt to guess or store as stringified bytes?
    // Actually, `ManifestValue` is a wrapper around serde_json::Value.
    // If it's a string, it's just bytes. If it's an int, it's little endian bytes (usually).
    // Let's implement a best-effort simple one or just wrap bytes.
    // TODO: Pass type context for better parsing.
    
    if bytes.len() == 4 {
        // Could be int or float
        let val = i32::from_le_bytes(bytes.try_into().unwrap());
        crate::core::manifest::ManifestValue::Int64(val as i64)
    } else if bytes.len() == 8 {
        // Could be long or double
        let val = i64::from_le_bytes(bytes.try_into().unwrap());
        crate::core::manifest::ManifestValue::Int64(val)
    } else {
        // String or Binary
        if let Ok(s) = std::str::from_utf8(bytes) {
            crate::core::manifest::ManifestValue::String(s.to_string())
        } else {
             // Fallback
             crate::core::manifest::ManifestValue::String(base64::engine::general_purpose::STANDARD.encode(bytes))
        }
    }
}

/// Convert Iceberg JSON schema to Arrow Schema
pub fn iceberg_json_to_arrow_schema(schema_json: &serde_json::Value) -> Result<arrow::datatypes::SchemaRef> {
    use arrow::datatypes::{Schema, Field};
    use std::sync::Arc;

    let fields_json = schema_json.get("fields")
        .and_then(|f| f.as_array())
        .ok_or_else(|| anyhow::anyhow!("Invalid Iceberg schema: missing fields"))?;

    let mut fields = Vec::new();

    for field in fields_json {
        let name = field.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
        let type_json = field.get("type").ok_or_else(|| anyhow::anyhow!("Field missing type"))?;
        let required = field.get("required").and_then(|r| r.as_bool()).unwrap_or(false);
        let id = field.get("id").and_then(|i| i.as_i64()).unwrap_or(0);

        let dt = convert_iceberg_type_to_arrow(type_json)?;
        let mut arrow_field = Field::new(name, dt, !required);
        
        // Store Iceberg ID in metadata
        if id > 0 {
            arrow_field.set_metadata(std::collections::HashMap::from([
                ("iceberg.id".to_string(), id.to_string())
            ]));
        }
        
        fields.push(arrow_field);
    }

    Ok(Arc::new(Schema::new(fields)))
}

fn convert_iceberg_type_to_arrow(type_json: &serde_json::Value) -> Result<arrow::datatypes::DataType> {
    use arrow::datatypes::{DataType, TimeUnit};

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
            "timestamptz" => Ok(DataType::Timestamp(TimeUnit::Microsecond, Some("UTC".into()))),
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
                 let fields = obj.get("fields")
                    .and_then(|f| f.as_array())
                    .map(|v| v.as_slice())
                    .unwrap_or(&[]);
                 let mut arrow_fields = Vec::new();
                 for f in fields {
                     let name = f.get("name").and_then(|n| n.as_str()).unwrap_or("unknown");
                     let field_type = f.get("type").unwrap();
                     let required = f.get("required").and_then(|r| r.as_bool()).unwrap_or(false);
                     arrow_fields.push(arrow::datatypes::Field::new(name, convert_iceberg_type_to_arrow(field_type)?, !required));
                 }
                 Ok(DataType::Struct(arrow_fields.into()))
            },
            "list" => {
                let element_type = obj.get("element").unwrap_or(obj.get("element-type").unwrap()); // 'element' or 'element-type'
                let required = obj.get("element-required").and_then(|r| r.as_bool()).unwrap_or(true); // Default true?
                let dt = convert_iceberg_type_to_arrow(element_type)?;
                Ok(DataType::List(std::sync::Arc::new(arrow::datatypes::Field::new("item", dt, !required))))
            },
            "decimal" => {
                let precision = obj.get("precision").and_then(|p| p.as_u64()).unwrap_or(38) as u8;
                let scale = obj.get("scale").and_then(|s| s.as_u64()).unwrap_or(10) as i8;
                Ok(DataType::Decimal128(precision, scale))
            },
            "map" => {
                let key_type = obj.get("key").unwrap_or(obj.get("key-type").unwrap());
                let value_type = obj.get("value").unwrap_or(obj.get("value-type").unwrap());
                let value_required = obj.get("value-required").and_then(|r| r.as_bool()).unwrap_or(true);
                
                let kt = convert_iceberg_type_to_arrow(key_type)?;
                let vt = convert_iceberg_type_to_arrow(value_type)?;
                
                Ok(DataType::Map(
                    std::sync::Arc::new(Field::new("entries", DataType::Struct(vec![
                        Field::new("key", kt, false), // Keys are usually non-nullable in Iceberg
                        Field::new("value", vt, !value_required),
                    ].into()), false)),
                    false // Not sorted by default
                ))
            },
            _ => Err(anyhow::anyhow!("Unsupported complex type: {:?}", type_json))
        }
    } else {
        Err(anyhow::anyhow!("Invalid types definition: {:?}", type_json))
    }
}

pub fn iceberg_partition_spec_to_hyperstream(
    spec_json: &serde_json::Value,
) -> Result<crate::core::manifest::PartitionSpec> {
    use crate::core::manifest::{PartitionSpec, PartitionField};
    
    let spec_id = spec_json.get("spec-id").and_then(|id| id.as_i64()).unwrap_or(0) as i32;
    let fields_json = spec_json.get("fields")
        .and_then(|f| f.as_array())
        .ok_or_else(|| anyhow::anyhow!("Invalid Iceberg partition spec: missing fields"))?;

    let mut fields = Vec::new();
    for field in fields_json {
        let name = field.get("name").and_then(|n| n.as_str()).unwrap_or("unknown").to_string();
        let source_id = field.get("source-id").and_then(|id| id.as_i64()).unwrap_or(0) as i32;
        let transform = field.get("transform").and_then(|t| t.as_str()).unwrap_or("identity").to_string();
        let field_id = field.get("field-id").and_then(|id| id.as_i64()).map(|id| id as i32);
        
        
        fields.push(PartitionField::new_single(
            source_id,
            field_id,
            name,
            transform,
        ));
    }

    Ok(PartitionSpec {
        spec_id,
        fields,
    })
}

pub enum IcebergManifestObject {
    Data(crate::core::manifest::ManifestEntry),
    Delete(crate::core::manifest::DeleteFile),
}

/// Convert Iceberg Manifest Entry to HyperStream Manifest Entry or Delete File
pub fn convert_iceberg_to_object(
    iceberg_entry: &IcebergManifestEntry,
    schema: &crate::core::manifest::Schema,
    partition_spec: &crate::core::manifest::PartitionSpec,
) -> Result<IcebergManifestObject> {
    use crate::core::manifest::{ManifestEntry, DeleteFile, DeleteContent, ColumnStats};
    use std::collections::HashMap;

    let df = &iceberg_entry.data_file;
    
    // Process partition values
    let mut partition_values = HashMap::new();
    if !df.partition.is_empty() && partition_spec.fields.len() == df.partition.len() {
        for (i, p_field) in partition_spec.fields.iter().enumerate() {
            partition_values.insert(p_field.name.clone(), df.partition[i].clone());
        }
    }

    if df.content == 0 {
        // Data File
        let mut column_stats = HashMap::new();
        let field_map: HashMap<i32, &crate::core::manifest::SchemaField> = schema.fields.iter()
            .map(|f| (f.id, f))
            .collect();

        if let (Some(lowers), Some(uppers), Some(nulls)) = (&df.lower_bounds, &df.upper_bounds, &df.null_value_counts) {
            for (id, lower_bytes) in lowers {
                if let Some(field) = field_map.get(id) {
                    let min_val = decode_iceberg_value(&serde_json::json!(field.type_str), lower_bytes); 
                    let max_val = if let Some(upper_bytes) = uppers.get(id) {
                        Some(decode_iceberg_value(&serde_json::json!(field.type_str), upper_bytes))
                    } else {
                        None
                    };
                    let null_count = *nulls.get(id).unwrap_or(&0);
                    column_stats.insert(field.name.clone(), ColumnStats {
                        min: Some(min_val),
                        max: max_val,
                        null_count,
                        distinct_count: None,
                    });
                }
            }
        }

        Ok(IcebergManifestObject::Data(ManifestEntry {
            file_path: df.file_path.clone(),
            file_size_bytes: df.file_size_in_bytes,
            record_count: df.record_count,
            index_files: Vec::new(),
            delete_files: Vec::new(),
            column_stats,
            partition_values,
            clustering_strategy: None,
            clustering_columns: None,
            min_clustering_score: None,
            max_clustering_score: None,
            normalization_mins: None,
            normalization_maxs: None,
        }))
    } else {
        // Delete File
        let content = if df.content == 1 {
            DeleteContent::Position
        } else if df.content == 2 {
            DeleteContent::Equality {
                equality_ids: df.equality_ids.clone().unwrap_or_default()
            }
        } else if df.content == 3 {
            // V3 Deletion Vector (content=3)
            // Check if we have the required DV fields
            if let (Some(ref_file), Some(offset), Some(size)) = (
                df.referenced_data_file.clone(),
                df.content_offset,
                df.content_size_in_bytes
            ) {
                DeleteContent::DeletionVector {
                    puffin_file_path: df.file_path.clone(),
                    content_offset: offset,
                    content_size_in_bytes: size,
                }
            } else {
                // Fallback to position delete if DV fields are missing
                DeleteContent::Position
            }
        } else {
            // Unknown content type, default to position
            DeleteContent::Position
        };

        Ok(IcebergManifestObject::Delete(DeleteFile {
            file_path: df.file_path.clone(),
            content,
            file_size_bytes: df.file_size_in_bytes,
            record_count: df.record_count,
            partition_values,
        }))
    }
}

pub struct PositionDeleteReader {
    store: Arc<dyn object_store::ObjectStore>,
}

impl PositionDeleteReader {
    pub fn new(store: Arc<dyn object_store::ObjectStore>) -> Self {
        Self { store }
    }

    pub async fn read_deletes(&self, path: &str, target_data_file_path: &str) -> Result<HashSet<i64>> {
        let is_avro = path.ends_with(".avro");
        let path_obj = object_store::path::Path::from(path);
        let res = self.store.get(&path_obj).await?;
        let bytes = res.bytes().await?;
        
        if is_avro {
            self.read_deletes_avro(bytes, target_data_file_path)
        } else {
            self.read_deletes_parquet(bytes, target_data_file_path)
        }
    }

    fn read_deletes_avro(&self, bytes: bytes::Bytes, target_data_file_path: &str) -> Result<HashSet<i64>> {
        let reader = apache_avro::Reader::new(&bytes[..])?;
        let mut deleted_positions = HashSet::new();

        for record in reader {
            let value = record?;
            if let apache_avro::types::Value::Record(fields) = value {
                let mut file_path = None;
                let mut pos = None;

                for (name, val) in fields {
                    match name.as_str() {
                        "file_path" => if let apache_avro::types::Value::String(s) = val { file_path = Some(s); },
                        "pos" => if let apache_avro::types::Value::Long(p) = val { pos = Some(p); },
                        _ => {}
                    }
                }

                if let (Some(fp), Some(p)) = (file_path, pos) {
                    if fp == target_data_file_path {
                        deleted_positions.insert(p);
                    }
                }
            }
        }
        Ok(deleted_positions)
    }

    fn read_deletes_parquet(&self, bytes: bytes::Bytes, target_data_file_path: &str) -> Result<HashSet<i64>> {
        use arrow::array::{StringArray, Int64Array};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        let cursor = bytes;
        let builder = ParquetRecordBatchReaderBuilder::try_new(cursor)?;
        let mut reader = builder.build()?;
        
        let mut deleted_positions = HashSet::new();
        
        while let Some(batch_res) = reader.next() {
            let batch = batch_res?;
            
            if let (Ok(file_path_col), Ok(pos_col)) = (batch.column_by_name("file_path").ok_or(()), batch.column_by_name("pos").ok_or(())) {
                let file_paths = file_path_col.as_any().downcast_ref::<StringArray>()
                    .ok_or_else(|| anyhow::anyhow!("file_path column is not string"))?;
                let positions = pos_col.as_any().downcast_ref::<Int64Array>()
                    .ok_or_else(|| anyhow::anyhow!("pos column is not int64"))?;
                
                for i in 0..batch.num_rows() {
                    if file_paths.value(i) == target_data_file_path {
                        deleted_positions.insert(positions.value(i));
                    }
                }
            }
        }
        Ok(deleted_positions)
    }
}

pub struct EqualityDeleteReader {
    store: Arc<dyn object_store::ObjectStore>,
}

impl EqualityDeleteReader {
    pub fn new(store: Arc<dyn object_store::ObjectStore>) -> Self {
        Self { store }
    }


    /// Read equality delete records from a Parquet file.
    /// Returns batches containing ONLY the columns specified by `equality_ids`.
    pub async fn read_equality_deletes(
        &self, 
        path: &str, 
        equality_ids: &[i32], 
        schema: &crate::core::manifest::Schema
    ) -> Result<Vec<RecordBatch>> {
        let is_avro = path.ends_with(".avro");
        let path_obj = object_store::path::Path::from(path);
        let res = self.store.get(&path_obj).await?;
        let bytes = res.bytes().await?;
        
        if is_avro {
            self.read_equality_deletes_avro(bytes, equality_ids, schema)
        } else {
            self.read_equality_deletes_parquet(bytes, equality_ids, schema)
        }
    }

    fn read_equality_deletes_avro(
        &self,
        bytes: bytes::Bytes,
        equality_ids: &[i32],
        schema: &crate::core::manifest::Schema
    ) -> Result<Vec<RecordBatch>> {
        use std::collections::HashMap;
        use arrow::array::*;
        use arrow::datatypes::{Field, Schema as ArrowSchema};
        use std::sync::Arc;

        let reader = apache_avro::Reader::new(&bytes[..])?;
        
        let mut column_names = Vec::new();
        let field_map: HashMap<i32, &crate::core::manifest::SchemaField> = schema.fields.iter()
            .map(|f| (f.id, f))
            .collect();

        for &id in equality_ids {
            if let Some(field) = field_map.get(&id) {
                column_names.push(field.name.clone());
            } else {
                return Err(anyhow::anyhow!("Equality Delete ID {} not found in schema", id));
            }
        }

        // Build columns
        let mut columns_data: Vec<Vec<apache_avro::types::Value>> = vec![Vec::new(); equality_ids.len()];
        let mut row_count = 0;

        for record in reader {
            let value = record?;
            if let apache_avro::types::Value::Record(fields) = value {
                let field_map_record: HashMap<String, apache_avro::types::Value> = fields.into_iter().collect();
                
                for (idx, name) in column_names.iter().enumerate() {
                    if let Some(val) = field_map_record.get(name) {
                        columns_data[idx].push(val.clone());
                    } else {
                         columns_data[idx].push(apache_avro::types::Value::Null);
                    }
                }
                row_count += 1;
            }
        }

        if row_count == 0 {
            return Ok(vec![]);
        }

        // Convert to Arrow RecordBatch
        let mut arrow_columns = Vec::new();
        let mut arrow_fields = Vec::new();

        for (idx, name) in column_names.iter().enumerate() {
            let field = field_map.get(&equality_ids[idx]).unwrap();
            let arrow_field = Field::new(name, self.map_type_to_arrow(&field.type_str), true);
            arrow_fields.push(arrow_field);
            
            let array = self.avro_to_arrow_array(&columns_data[idx], &field.type_str)?;
            arrow_columns.push(array);
        }

        let arrow_schema = Arc::new(ArrowSchema::new(arrow_fields));
        let batch = RecordBatch::try_new(arrow_schema, arrow_columns)?;
        Ok(vec![batch])
    }

    fn map_type_to_arrow(&self, type_str: &str) -> arrow::datatypes::DataType {
        match type_str {
            "Int32" | "int" => arrow::datatypes::DataType::Int32,
            "Int64" | "long" => arrow::datatypes::DataType::Int64,
            "Float32" | "float" => arrow::datatypes::DataType::Float32,
            "Float64" | "double" => arrow::datatypes::DataType::Float64,
            "Utf8" | "string" => arrow::datatypes::DataType::Utf8,
            "Boolean" | "bool" => arrow::datatypes::DataType::Boolean,
            _ => arrow::datatypes::DataType::Utf8,
        }
    }

    fn avro_to_arrow_array(&self, values: &[apache_avro::types::Value], type_str: &str) -> Result<arrow::array::ArrayRef> {
        use arrow::array::*;
        use apache_avro::types::Value;

        match type_str {
            "Int32" | "int" => {
                let mut builder = Int32Builder::new();
                for v in values {
                    match v {
                        Value::Int(i) => builder.append_value(*i),
                        _ => builder.append_null(),
                    }
                }
                Ok(Arc::new(builder.finish()))
            },
            "Int64" | "long" => {
                let mut builder = Int64Builder::new();
                for v in values {
                    match v {
                        Value::Long(i) => builder.append_value(*i),
                        _ => builder.append_null(),
                    }
                }
                Ok(Arc::new(builder.finish()))
            },
            "Float32" | "float" => {
                let mut builder = Float32Builder::new();
                for v in values {
                    match v {
                        Value::Float(i) => builder.append_value(*i),
                        _ => builder.append_null(),
                    }
                }
                Ok(Arc::new(builder.finish()))
            },
            "Float64" | "double" => {
                let mut builder = Float64Builder::new();
                for v in values {
                    match v {
                        Value::Double(i) => builder.append_value(*i),
                        _ => builder.append_null(),
                    }
                }
                Ok(Arc::new(builder.finish()))
            },
            "Utf8" | "string" => {
                let mut builder = StringBuilder::new();
                for v in values {
                    match v {
                        Value::String(s) => builder.append_value(s),
                        _ => builder.append_null(),
                    }
                }
                Ok(Arc::new(builder.finish()))
            },
            "Boolean" | "bool" => {
                let mut builder = BooleanBuilder::new();
                for v in values {
                    match v {
                        Value::Boolean(b) => builder.append_value(*b),
                        _ => builder.append_null(),
                    }
                }
                Ok(Arc::new(builder.finish()))
            },
            _ => {
                let mut builder = StringBuilder::new();
                for v in values {
                    builder.append_value(format!("{:?}", v));
                }
                Ok(Arc::new(builder.finish()))
            }
        }
    }

    fn read_equality_deletes_parquet(
        &self,
        bytes: bytes::Bytes,
        equality_ids: &[i32],
        schema: &crate::core::manifest::Schema
    ) -> Result<Vec<RecordBatch>> {
        use std::collections::HashMap;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use bytes::Bytes;

        let cursor = Bytes::from(bytes);

        // 1. Resolve column names from IDs
        let mut column_names = Vec::new();
        let field_map: HashMap<i32, &crate::core::manifest::SchemaField> = schema.fields.iter()
            .map(|f| (f.id, f))
            .collect();

        for &id in equality_ids {
            if let Some(field) = field_map.get(&id) {
                column_names.push(field.name.clone());
            } else {
                // If ID not found in schema, we can't read it. Ideally shouldn't happen for valid manifests.
                return Err(anyhow::anyhow!("Equality Delete ID {} not found in schema", id));
            }
        }

        // 2. Open Parquet Reader
        let builder = ParquetRecordBatchReaderBuilder::try_new(cursor)?;
        let arrow_schema = builder.schema();
        
        // 3. Find projection mask indices
        let mut indices = Vec::new();
        for col_name in &column_names {
            if let Ok(idx) = arrow_schema.index_of(col_name) {
                indices.push(idx);
            } else {
                // If column missing in delete file, it implies no deletes for that ID? 
                // Or maybe partial schema match. For now, strict requirement.
                return Err(anyhow::anyhow!("Column {} not found in equality delete file", col_name));
            }
        }
        
        // 4. Project and Read
        let projection = parquet::arrow::ProjectionMask::roots(builder.parquet_schema(), indices);
        let reader = builder.with_projection(projection).build()?;

        let mut batches = Vec::new();
        for batch_res in reader {
            batches.push(batch_res?);
        }

        Ok(batches)
    }
}

/// GPU Accelerated Puffin Index Writer for Iceberg
pub struct GpuPuffinWriter {
    // Orchestrates GPU-based index builds (HNSW, Bloom)
}

impl GpuPuffinWriter {
    pub fn new() -> Self {
        Self {}
    }

    pub async fn build_index(&self, _column: &str) -> Result<String> {
        // GPU build...
        Ok("sidecar_path".to_string())
    }
}

pub const MANIFEST_LIST_SCHEMA_V2: &str = r#"
{
    "type": "record",
    "name": "manifest_list",
    "fields": [
        {"name": "manifest_path", "type": "string"},
        {"name": "manifest_length", "type": "long"},
        {"name": "partition_spec_id", "type": "int"},
        {"name": "content", "type": "int", "doc": "0=data, 1=deletes"},
        {"name": "sequence_number", "type": "long", "default": 0},
        {"name": "min_sequence_number", "type": "long", "default": 0},
        {"name": "added_snapshot_id", "type": "long"},
        {"name": "added_data_files_count", "type": "int"},
        {"name": "existing_data_files_count", "type": "int"},
        {"name": "deleted_data_files_count", "type": "int"},
        {"name": "added_rows_count", "type": "long"},
        {"name": "existing_rows_count", "type": "long"},
        {"name": "deleted_rows_count", "type": "long"},
        {"name": "partitions", "type": ["null", {
            "type": "array",
            "items": {
                "type": "record",
                "name": "field_summary",
                "fields": [
                    {"name": "contains_null", "type": "boolean"},
                    {"name": "contains_nan", "type": ["null", "boolean"]},
                    {"name": "lower_bound", "type": ["null", "bytes"]},
                    {"name": "upper_bound", "type": ["null", "bytes"]}
                ]
            }
        }]}
    ]
}
"#;

pub struct IcebergWriter;

impl IcebergWriter {
    pub fn new() -> Self {
        Self {}
    }

    /// Write a Manifest List (snap-*.avro)
    pub fn write_manifest_list(&self, entries: &[crate::core::manifest::ManifestListEntry]) -> Result<Vec<u8>> {
        let schema = apache_avro::Schema::parse_str(MANIFEST_LIST_SCHEMA_V2)?;
        let mut writer = apache_avro::Writer::new(&schema, Vec::new());
        
        for entry in entries {
            let mut record = apache_avro::types::Record::new(&schema).unwrap();
            record.put("manifest_path", entry.manifest_path.clone());
            record.put("manifest_length", entry.manifest_length);
            record.put("partition_spec_id", entry.partition_spec_id);
            record.put("content", entry.content); 
            record.put("sequence_number", entry.sequence_number);
            record.put("min_sequence_number", entry.min_sequence_number); 
            record.put("added_snapshot_id", entry.added_snapshot_id);
            record.put("added_data_files_count", entry.added_files_count);
            record.put("existing_data_files_count", entry.existing_files_count);
            record.put("deleted_data_files_count", entry.deleted_files_count);
            record.put("added_rows_count", entry.added_rows_count); 
            record.put("existing_rows_count", entry.existing_rows_count);
            record.put("deleted_rows_count", entry.deleted_rows_count);
            record.put("partitions", apache_avro::types::Value::Null); // TODO: Summaries

            writer.append(record)?;
        }
        
        Ok(writer.into_inner()?)
    }

    /// Write a Manifest File (*.avro)
    pub fn write_manifest_file(
        &self, 
        entries: &[crate::core::manifest::ManifestEntry], 
        partition_spec: &crate::core::manifest::PartitionSpec,
        snapshot_id: i64,
        seq_num: i64
    ) -> Result<Vec<u8>> {
        // 1. Generate Schema based on Partition Spec
        let schema_json = self.generate_manifest_schema(partition_spec);
        let schema = apache_avro::Schema::parse_str(&schema_json)?;
        let mut writer = apache_avro::Writer::new(&schema, Vec::new());

        for entry in entries {
            let mut record = apache_avro::types::Record::new(&schema).unwrap();
            record.put("status", apache_avro::types::Value::Int(1)); // 1=ADDED
            record.put("snapshot_id", apache_avro::types::Value::Union(1, Box::new(apache_avro::types::Value::Long(snapshot_id))));
            record.put("sequence_number", apache_avro::types::Value::Union(1, Box::new(apache_avro::types::Value::Long(seq_num))));
            record.put("file_sequence_number", apache_avro::types::Value::Union(1, Box::new(apache_avro::types::Value::Long(seq_num))));
            
            let data_file_schema = match &schema {
                apache_avro::Schema::Record(r) => &r.fields.iter().find(|f| f.name == "data_file").unwrap().schema,
                _ => unreachable!(),
            };
            let mut data_file = apache_avro::types::Record::new(data_file_schema).unwrap();

            data_file.put("content", apache_avro::types::Value::Int(0)); // 0=Data
            data_file.put("file_path", apache_avro::types::Value::String(entry.file_path.clone()));
            data_file.put("file_format", apache_avro::types::Value::String("PARQUET".to_string()));
            data_file.put("record_count", apache_avro::types::Value::Long(entry.record_count));
            data_file.put("file_size_in_bytes", apache_avro::types::Value::Long(entry.file_size_bytes));
            
            // Statistics (Unions)
            data_file.put("column_sizes", apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)));
            data_file.put("value_counts", apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)));
            data_file.put("null_value_counts", apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)));
            data_file.put("nan_value_counts", apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)));
            data_file.put("lower_bounds", apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)));
            data_file.put("upper_bounds", apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)));
            
            // Partition Data
            // We need to fetch the inner record schema for 'partition' field inside data_file
            // This is getting complex to reflect via API, sticking to JSON value construction?
            // Avro-rs Record::put takes Into<Value>. 
            // Better to construct Value::Record manually matching the schema structure.
            
            let mut partition_record_values = Vec::new();
            for field in &partition_spec.fields {
                let val = entry.partition_values.get(&field.name).unwrap_or(&serde_json::Value::Null);
                let avro_val = json_to_avro_value(val);
                let union_val = match avro_val {
                    apache_avro::types::Value::Null => apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                    _ => apache_avro::types::Value::Union(1, Box::new(avro_val)),
                };
                partition_record_values.push((field.name.clone(), union_val));
            }
            data_file.put("partition", apache_avro::types::Value::Record(partition_record_values));
            data_file.put("equality_ids", apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)));

            record.put("data_file", data_file);
            writer.append(record)?;

            // Write associated Delete Files
            for del_file in &entry.delete_files {
                 let mut record = apache_avro::types::Record::new(&schema).unwrap();
                 record.put("status", apache_avro::types::Value::Int(1)); // 1=ADDED
                 record.put("snapshot_id", apache_avro::types::Value::Union(1, Box::new(apache_avro::types::Value::Long(snapshot_id))));
                 record.put("sequence_number", apache_avro::types::Value::Union(1, Box::new(apache_avro::types::Value::Long(seq_num))));
                 record.put("file_sequence_number", apache_avro::types::Value::Union(1, Box::new(apache_avro::types::Value::Long(seq_num))));
                  
                 let data_file_schema = match &schema {
                     apache_avro::Schema::Record(r) => &r.fields.iter().find(|f| f.name == "data_file").unwrap().schema,
                     _ => unreachable!(),
                 };
                 let mut data_file = apache_avro::types::Record::new(data_file_schema).unwrap();

                 let content_id = match del_file.content {
                     crate::core::manifest::DeleteContent::Position => 1,
                     crate::core::manifest::DeleteContent::Equality { .. } => 2,
                     crate::core::manifest::DeleteContent::DeletionVector { .. } => 3,
                 };

                 data_file.put("content", apache_avro::types::Value::Int(content_id));
                 data_file.put("file_path", apache_avro::types::Value::String(del_file.file_path.clone()));
                 data_file.put("file_format", apache_avro::types::Value::String("AVRO".to_string()));
                  data_file.put("record_count", apache_avro::types::Value::Long(del_file.record_count));
                  data_file.put("file_size_in_bytes", apache_avro::types::Value::Long(del_file.file_size_bytes));
                  
                  // Statistics (Unions)
                  data_file.put("column_sizes", apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)));
                  data_file.put("value_counts", apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)));
                  data_file.put("null_value_counts", apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)));
                  data_file.put("nan_value_counts", apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)));
                  data_file.put("lower_bounds", apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)));
                  data_file.put("upper_bounds", apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)));

                 // Use Delete File's partition values (or inherit from parent if empty?)
                 // DeleteFile has partition_values.
                 let mut partition_record_values = Vec::new();
                 for field in &partition_spec.fields {
                     let val = del_file.partition_values.get(&field.name).unwrap_or(&serde_json::Value::Null);
                     let avro_val = json_to_avro_value(val);
                     let union_val = match avro_val {
                         apache_avro::types::Value::Null => apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)),
                         _ => apache_avro::types::Value::Union(1, Box::new(avro_val)),
                     };
                     partition_record_values.push((field.name.clone(), union_val));
                 }
                 data_file.put("partition", apache_avro::types::Value::Record(partition_record_values));

                  // Equality IDs
                  if let crate::core::manifest::DeleteContent::Equality { equality_ids } = &del_file.content {
                      let avro_ids: Vec<apache_avro::types::Value> = equality_ids.iter().map(|&i| apache_avro::types::Value::Int(i)).collect();
                      data_file.put("equality_ids", apache_avro::types::Value::Union(1, Box::new(apache_avro::types::Value::Array(avro_ids))));
                  } else {
                      data_file.put("equality_ids", apache_avro::types::Value::Union(0, Box::new(apache_avro::types::Value::Null)));
                  }

                  record.put("data_file", data_file);
                  writer.append(record)?;
            }
        }

        Ok(writer.into_inner()?)
    }

    fn generate_manifest_schema(&self, spec: &crate::core::manifest::PartitionSpec) -> String {
        let mut partition_fields = Vec::new();
        for field in &spec.fields {
             // For now assume all partitions are int, string or long. 
             // Need full type mapping from source schema implicitly?
             // Or generic "string" / "int" based on transform.
             let type_str = match field.transform.as_str() {
                 "year" | "month" | "day" => r#"["null", "int"]"#,
                 s if s.starts_with("bucket[") => r#"["null", "int"]"#,
                 s if s.starts_with("truncate[") => r#"["null", "string"]"#,
                 _ => r#"["null", "string"]"#,
             };
             partition_fields.push(format!(r#"{{"name": "{}", "type": {}, "default": null}}"#, field.name, type_str));
        }
        let partition_fields_json = partition_fields.join(",");

        format!(r#"
{{
    "type": "record",
    "name": "manifest",
    "fields": [
        {{"name": "status", "type": "int", "doc": "0=EXISTING, 1=ADDED, 2=DELETED"}},
        {{"name": "snapshot_id", "type": ["null", "long"]}},
        {{"name": "sequence_number", "type": ["null", "long"]}},
        {{"name": "file_sequence_number", "type": ["null", "long"]}},
        {{"name": "data_file", "type": {{
            "type": "record",
            "name": "r2",
            "fields": [
                {{"name": "content", "type": "int", "doc": "0=DATA, 1=POSITION DELETES, 2=EQUALITY DELETES"}},
                {{"name": "file_path", "type": "string"}},
                {{"name": "file_format", "type": "string"}},
                {{"name": "partition", "type": {{
                    "type": "record",
                    "name": "r102",
                    "fields": [{}]
                }}}},

                {{"name": "record_count", "type": "long"}},
                {{"name": "file_size_in_bytes", "type": "long"}},
                {{"name": "column_sizes", "type": ["null", {{"type": "array", "items": {{"type": "record", "name": "k1", "fields": [{{"name":"key", "type":"int"}}, {{"name":"value", "type":"long"}}]}}}}], "default": null}},
                {{"name": "value_counts", "type": ["null", {{"type": "array", "items": {{"type": "record", "name": "k2", "fields": [{{"name":"key", "type":"int"}}, {{"name":"value", "type":"long"}}]}}}}], "default": null}},
                {{"name": "null_value_counts", "type": ["null", {{"type": "array", "items": {{"type": "record", "name": "k3", "fields": [{{"name":"key", "type":"int"}}, {{"name":"value", "type":"long"}}]}}}}], "default": null}},
                {{"name": "nan_value_counts", "type": ["null", {{"type": "array", "items": {{"type": "record", "name": "k4", "fields": [{{"name":"key", "type":"int"}}, {{"name":"value", "type":"long"}}]}}}}], "default": null}},
                {{"name": "lower_bounds", "type": ["null", {{"type": "array", "items": {{"type": "record", "name": "k5", "fields": [{{"name":"key", "type":"int"}}, {{"name":"value", "type":"bytes"}}]}}}}], "default": null}},
                {{"name": "upper_bounds", "type": ["null", {{"type": "array", "items": {{"type": "record", "name": "k6", "fields": [{{"name":"key", "type":"int"}}, {{"name":"value", "type":"bytes"}}]}}}}], "default": null}},
                {{"name": "equality_ids", "type": ["null", {{"type": "array", "items": "int"}}], "default": null}}
            ]
        }}
    }}]
}}
"#, partition_fields_json)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum IcebergTransform {
    Identity,
    Bucket(u32),
    Truncate(u32),
    Year,
    Month,
    Day,
    Hour,
    Void,
}

impl IcebergTransform {
    pub fn parse(s: &str) -> Self {
        match s {
            "identity" => Self::Identity,
            "year" => Self::Year,
            "month" => Self::Month,
            "day" => Self::Day,
            "hour" => Self::Hour,
            "void" => Self::Void,
            _ if s.starts_with("bucket[") => {
                 let n = s.trim_start_matches("bucket[").trim_end_matches(']').parse().unwrap_or(0);
                 Self::Bucket(n)
            },
            _ if s.starts_with("truncate[") => {
                 let n = s.trim_start_matches("truncate[").trim_end_matches(']').parse().unwrap_or(0);
                 Self::Truncate(n)
            },
            _ => Self::Void
        }
    }


    pub fn apply(&self, array: &dyn arrow::array::Array, row_i: usize) -> serde_json::Value {
        self.apply_multi(&[array], row_i)
    }

    pub fn apply_multi(&self, arrays: &[&dyn arrow::array::Array], row_i: usize) -> serde_json::Value {
        if arrays.is_empty() { return serde_json::Value::Null; }
        if arrays.iter().any(|a| a.is_null(row_i)) { return serde_json::Value::Null; }
        
        use chrono::Datelike;

        match self {
            Self::Identity => {
                let array = arrays[0];
                match array.data_type() {
                    arrow::datatypes::DataType::Int32 => {
                        let a = array.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
                        serde_json::json!(a.value(row_i))
                    },
                    arrow::datatypes::DataType::Int64 => {
                        let a = array.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
                        serde_json::json!(a.value(row_i))
                    },
                    arrow::datatypes::DataType::Utf8 => {
                        let a = array.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
                        serde_json::json!(a.value(row_i))
                    },
                    arrow::datatypes::DataType::Date32 => {
                        let a = array.as_any().downcast_ref::<arrow::array::Date32Array>().unwrap();
                        serde_json::json!(a.value(row_i))
                    },
                    arrow::datatypes::DataType::Timestamp(_, _) => {
                         let a = array.as_any().downcast_ref::<arrow::array::TimestampMicrosecondArray>().unwrap();
                         serde_json::json!(a.value(row_i))
                    },
                    _ => serde_json::Value::Null
                }
            },
            Self::Bucket(n) => {
                let mut hash_val: u32 = 0;
                for array in arrays {
                    let field_hash = match array.data_type() {
                        arrow::datatypes::DataType::Int32 => {
                            let a = array.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
                            let val = a.value(row_i);
                            let bytes = (val as i64).to_le_bytes();
                            murmur3_32_x86(&bytes, hash_val)
                        },
                        arrow::datatypes::DataType::Int64 => {
                            let a = array.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
                            let val = a.value(row_i);
                            let bytes = val.to_le_bytes();
                            murmur3_32_x86(&bytes, hash_val)
                        },
                        arrow::datatypes::DataType::Utf8 => {
                            let a = array.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
                            let s = a.value(row_i);
                            murmur3_32_x86(s.as_bytes(), hash_val)
                        },
                        arrow::datatypes::DataType::Date32 => {
                            let a = array.as_any().downcast_ref::<arrow::array::Date32Array>().unwrap();
                            let val = a.value(row_i);
                            let bytes = (val as i64).to_le_bytes();
                            murmur3_32_x86(&bytes, hash_val)
                        },
                        _ => 0,
                    };
                    hash_val = field_hash;
                }
                // Iceberg Bucketing: (hash & Integer.MAX_VALUE) % N
                serde_json::json!((hash_val & 0x7FFFFFFF) % *n)
            },
            Self::Truncate(w) => {
                let array = arrays[0];
                match array.data_type() {
                    arrow::datatypes::DataType::Utf8 => {
                        let a = array.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
                        let s = a.value(row_i);
                        let limit = (*w as usize).min(s.len());
                        serde_json::json!(&s[..limit])
                    },
                    arrow::datatypes::DataType::Int32 => {
                        let a = array.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
                        let v = a.value(row_i);
                        serde_json::json!(v - (v % (*w as i32)))
                    },
                    arrow::datatypes::DataType::Int64 => {
                        let a = array.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
                        let v = a.value(row_i);
                        serde_json::json!(v - (v % (*w as i64)))
                    },
                    _ => serde_json::Value::Null,
                }
            },
            Self::Year => {
                let array = arrays[0];
                // Years from 1970
                if let arrow::datatypes::DataType::Date32 = array.data_type() {
                    let a = array.as_any().downcast_ref::<arrow::array::Date32Array>().unwrap();
                    let days = a.value(row_i);
                    // 1970-01-01 is epoch.
                    let opt_date = chrono::NaiveDate::from_num_days_from_ce_opt(days + 719163);
                    if let Some(d) = opt_date {
                        serde_json::json!(d.year() - 1970)
                    } else { serde_json::Value::Null }
                } else if let arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, _) = array.data_type() {
                     let a = array.as_any().downcast_ref::<arrow::array::TimestampMicrosecondArray>().unwrap();
                     let micros = a.value(row_i);
                     let seconds = micros / 1_000_000;
                     let opt_dt = chrono::DateTime::from_timestamp(seconds, 0);
                     if let Some(dt) = opt_dt {
                         serde_json::json!(dt.year() - 1970)
                     } else { serde_json::Value::Null }
                } else { serde_json::Value::Null }
            },
            Self::Month => {
                let array = arrays[0];
                // Months from 1970-01-01
                if let arrow::datatypes::DataType::Date32 = array.data_type() {
                    let a = array.as_any().downcast_ref::<arrow::array::Date32Array>().unwrap();
                    let days = a.value(row_i);
                    let opt_date = chrono::NaiveDate::from_num_days_from_ce_opt(days + 719163);
                    if let Some(d) = opt_date {
                        serde_json::json!((d.year() - 1970) * 12 + (d.month() as i32) - 1)
                    } else { serde_json::Value::Null }
                } else if let arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, _) = array.data_type() {
                     let a = array.as_any().downcast_ref::<arrow::array::TimestampMicrosecondArray>().unwrap();
                     let micros = a.value(row_i);
                     let seconds = micros / 1_000_000;
                     let opt_dt = chrono::DateTime::from_timestamp(seconds, 0);
                     if let Some(dt) = opt_dt {
                         serde_json::json!((dt.year() - 1970) * 12 + (dt.month() as i32) - 1)
                     } else { serde_json::Value::Null }
                } else { serde_json::Value::Null }
            },
            Self::Day => {
                let array = arrays[0];
                // Days from 1970-01-01
                if let arrow::datatypes::DataType::Date32 = array.data_type() {
                    let a = array.as_any().downcast_ref::<arrow::array::Date32Array>().unwrap();
                     serde_json::json!(a.value(row_i))
                } else if let arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, _) = array.data_type() {
                     let a = array.as_any().downcast_ref::<arrow::array::TimestampMicrosecondArray>().unwrap();
                     let micros = a.value(row_i);
                     // Spec: input timestamp (micros) -> days from epoch
                     serde_json::json!(micros / (1_000_000 * 60 * 60 * 24))
                } else { serde_json::Value::Null }
            },
            Self::Hour => {
                let array = arrays[0];
                // Hours from 1970-01-01 00:00:00
                if let arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, _) = array.data_type() {
                     let a = array.as_any().downcast_ref::<arrow::array::TimestampMicrosecondArray>().unwrap();
                     let micros = a.value(row_i);
                     serde_json::json!(micros / (1_000_000 * 60 * 60))
                } else { serde_json::Value::Null }
            },
            Self::Void => serde_json::Value::Null,
        }
    }
}

/// Wrapper around murmur3 crate to ensure x86 32-bit implementation
/// Aligned with official `iceberg-rust` implementation.
pub fn murmur3_32_x86(data: &[u8], seed: u32) -> u32 {
    murmur3::murmur3_32(&mut std::io::Cursor::new(data), seed).unwrap()
}

pub fn json_to_avro_value(v: &serde_json::Value) -> apache_avro::types::Value {
    match v {
        serde_json::Value::Null => apache_avro::types::Value::Null,
        serde_json::Value::Bool(b) => apache_avro::types::Value::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                if i >= (i32::MIN as i64) && i <= (i32::MAX as i64) {
                    apache_avro::types::Value::Int(i as i32)
                } else {
                    apache_avro::types::Value::Long(i)
                }
            } else if let Some(f) = n.as_f64() {
                apache_avro::types::Value::Double(f)
            } else {
                apache_avro::types::Value::Null
            }
        },
        serde_json::Value::String(s) => apache_avro::types::Value::String(s.clone()),
        _ => apache_avro::types::Value::Null
    }
}

