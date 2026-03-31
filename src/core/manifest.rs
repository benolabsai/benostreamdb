// Copyright (c) 2026 Richard Albright. All rights reserved.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use object_store::{path::Path, ObjectStore};
use anyhow::Result;
use futures::StreamExt;
use chrono::Utc;

pub type SegmentId = String;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct IndexFile {
    pub file_path: String,
    pub index_type: String, // e.g. "scalar", "vector", "bloom"
    pub column_name: Option<String>,
    /// HyperStream Extension: Puffin blob details if this is a Puffin file
    #[serde(default)]
    pub blob_type: Option<String>,
    #[serde(default)]
    pub offset: Option<i64>,
    #[serde(default)]
    pub length: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum DeleteContent {
    Position,
    Equality { equality_ids: Vec<i32> },
    /// V3 Deletion Vector: Reference to a Puffin file containing a deletion vector
    #[serde(rename = "deletion-vector")]
    DeletionVector {
        /// Path to the Puffin file containing the deletion vector
        puffin_file_path: String,
        /// Offset within the Puffin file where the deletion vector blob starts
        content_offset: i64,
        /// Size of the deletion vector blob in bytes
        content_size_in_bytes: i64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DeleteFile {
    pub file_path: String,
    pub content: DeleteContent,
    pub file_size_bytes: i64,
    pub record_count: i64,
    #[serde(default)]
    pub partition_values: HashMap<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct PartitionField {
    /// Source column ID(s). For single-column transforms, this is a single-element vec.
    /// For multi-column transforms like bucket(N, col1, col2), this contains multiple IDs.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub source_ids: Vec<i32>,
    
    /// Legacy single source_id for backward compatibility
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source_id: Option<i32>,
    
    pub field_id: Option<i32>,
    pub name: String,
    pub transform: String, // e.g. "identity", "year", "month", "day", "bucket(256)"
}

impl PartitionField {
    /// Create a single-column partition field
    pub fn new_single(source_id: i32, field_id: Option<i32>, name: String, transform: String) -> Self {
        Self {
            source_ids: vec![source_id],
            source_id: Some(source_id),
            field_id,
            name,
            transform,
        }
    }
    
    /// Create a multi-column partition field (e.g., composite bucketing)
    pub fn new_multi(source_ids: Vec<i32>, field_id: Option<i32>, name: String, transform: String) -> Self {
        Self {
            source_ids: source_ids.clone(),
            source_id: source_ids.first().copied(),
            field_id,
            name,
            transform,
        }
    }
    
    /// Get source IDs, handling both old and new format
    pub fn get_source_ids(&self) -> Vec<i32> {
        if !self.source_ids.is_empty() {
            self.source_ids.clone()
        } else if let Some(id) = self.source_id {
            vec![id]
        } else {
            vec![]
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct PartitionSpec {
    pub spec_id: i32,
    pub fields: Vec<PartitionField>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SortDirection {
    Asc,
    Desc,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum NullOrder {
    NullsFirst,
    NullsLast,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "kebab-case")]
pub struct SortField {
    pub source_id: i32,
    pub transform: String, // e.g. "identity"
    pub direction: SortDirection,
    pub null_order: NullOrder,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub struct SortOrder {
    pub order_id: i32,
    pub fields: Vec<SortField>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManifestEntry {
    pub file_path: String,
    pub file_size_bytes: i64,
    pub record_count: i64,
    /// HyperStream Extension: Sidecar Index Files
    pub index_files: Vec<IndexFile>,
    /// HyperStream Extension: Merge-on-Read Delete Files (Iceberg v2 compliant)
    #[serde(default)]
    pub delete_files: Vec<DeleteFile>,
    /// Column Statistics for Pruning (Min/Max/Nulls)
    #[serde(default)]
    pub column_stats: HashMap<String, ColumnStats>,
    /// Partition values for this file
    #[serde(default)]
    pub partition_values: HashMap<String, Value>,
    /// HyperStream Extension: Clustering metadata for advanced pruning
    #[serde(default)]
    pub clustering_strategy: Option<String>,
    #[serde(default)]
    pub clustering_columns: Option<Vec<String>>,
    #[serde(default)]
    pub min_clustering_score: Option<u64>,
    #[serde(default)]
    pub max_clustering_score: Option<u64>,
    #[serde(default)]
    pub normalization_mins: Option<Vec<Value>>,
    #[serde(default)]
    pub normalization_maxs: Option<Vec<Value>>,
}

impl From<&ManifestValue> for Value {
    fn from(val: &ManifestValue) -> Self {
        match val {
            ManifestValue::String(s) => Value::String(s.clone()),
            ManifestValue::Int32(i) => Value::Number((*i).into()),
            ManifestValue::Int64(i) => Value::Number((*i).into()),
            ManifestValue::Float32(f) => serde_json::Number::from_f64(*f as f64).map(Value::Number).unwrap_or(Value::Null),
            ManifestValue::Float64(f) => serde_json::Number::from_f64(*f).map(Value::Number).unwrap_or(Value::Null),
            ManifestValue::Boolean(b) => Value::Bool(*b),
            ManifestValue::Null => Value::Null,
        }
    }
}

impl std::fmt::Display for ManifestValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ManifestValue::String(s) => write!(f, "{}", s),
            ManifestValue::Int32(i) => write!(f, "{}", i),
            ManifestValue::Int64(i) => write!(f, "{}", i),
            ManifestValue::Float32(v) => write!(f, "{}", v),
            ManifestValue::Float64(v) => write!(f, "{}", v),
            ManifestValue::Boolean(b) => write!(f, "{}", b),
            ManifestValue::Null => write!(f, "null"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(untagged)]
pub enum ManifestValue {
    String(String),
    Int32(i32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    Boolean(bool),
    Null,
}

impl From<Value> for ManifestValue {
    fn from(val: Value) -> Self {
        match val {
            Value::String(s) => ManifestValue::String(s),
            Value::Number(n) => {
                if n.is_i64() {
                    ManifestValue::Int64(n.as_i64().unwrap())
                } else if n.is_f64() {
                     ManifestValue::Float64(n.as_f64().unwrap())
                } else {
                     // Fallback, treated as f64 or 0 if nan/inf (json doesn't have nan)
                     ManifestValue::Float64(n.as_f64().unwrap_or(0.0))
                }
            },
            Value::Bool(b) => ManifestValue::Boolean(b),
            Value::Null => ManifestValue::Null,
            _ => ManifestValue::String(val.to_string()), // Fallback for arrays/objects
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ColumnStats {
    pub min: Option<ManifestValue>,
    pub max: Option<ManifestValue>,
    pub null_count: i64,
    /// Number of distinct values (NDV) - Iceberg V2 spec field
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distinct_count: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManifestListEntry {
    pub manifest_path: String,
    pub manifest_length: i64,
    pub partition_spec_id: i32,
    /// 0=Data, 1=Deletes
    pub content: i32,
    pub sequence_number: i64,
    pub min_sequence_number: i64,
    pub added_snapshot_id: i64,
    pub added_files_count: i32,
    pub existing_files_count: i32,
    pub deleted_files_count: i32,
    pub added_rows_count: i64,
    pub existing_rows_count: i64,
    pub deleted_rows_count: i64,
    pub partition_stats: HashMap<String, ColumnStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ManifestList {
    pub manifest_files: Vec<ManifestListEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SchemaField {
    pub id: i32,
    pub name: String,
    #[serde(rename = "type")]
    pub type_str: String, // "int", "string", "struct", "list", "map"
    pub required: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fields: Vec<SchemaField>, // For nested types (struct fields, list element, map key/value)
    /// Iceberg V3: Default value for rows written before this column was added
    #[serde(skip_serializing_if = "Option::is_none")]
    pub initial_default: Option<Value>,
    /// Iceberg V3: Default value for new rows when this column is null
    #[serde(skip_serializing_if = "Option::is_none")]
    pub write_default: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Schema {
    #[serde(alias = "schema-id")]
    pub schema_id: i32,
    pub fields: Vec<SchemaField>,
    #[serde(rename = "identifier-field-ids", default, skip_serializing_if = "Vec::is_empty")]
    pub identifier_field_ids: Vec<i32>,
}

impl Schema {
    pub fn new(id: i32, fields: Vec<SchemaField>, identifier_field_ids: Vec<i32>) -> Self {
        Self { schema_id: id, fields, identifier_field_ids }
    }

    pub fn to_arrow(&self) -> arrow::datatypes::Schema {
        let fields: Vec<arrow::datatypes::Field> = self.fields.iter().map(|f| f.to_arrow()).collect();
        arrow::datatypes::Schema::new(fields)
    }

    pub fn from_arrow(schema: &arrow::datatypes::Schema, id: i32) -> Self {
        let fields = schema.fields().iter().enumerate().map(|(i, f)| {
            SchemaField::from_arrow_field(f, i as i32 + 1)
        }).collect();
        Self { schema_id: id, fields, identifier_field_ids: Vec::new() }
    }
}

impl SchemaField {
    pub fn from_arrow_field(f: &arrow::datatypes::Field, id: i32) -> Self {
        use arrow::datatypes::DataType;
        
        let mut nested_fields = Vec::new();
        let type_str = match f.data_type() {
            DataType::Struct(fields) => {
                nested_fields = fields.iter().enumerate().map(|(i, sf)| {
                    SchemaField::from_arrow_field(sf, id * 100 + i as i32 + 1) // Simple nested ID logic
                }).collect();
                "struct".to_string()
            },
            DataType::List(field) => {
                nested_fields.push(SchemaField::from_arrow_field(field, id * 100 + 1));
                "list".to_string()
            },
            DataType::Map(field, _) => {
                // Arrow Map has a Struct field "entries" with "key" and "value"
                if let DataType::Struct(fields) = field.data_type() {
                    nested_fields = fields.iter().enumerate().map(|(i, sf)| {
                        SchemaField::from_arrow_field(sf, id * 100 + i as i32 + 1)
                    }).collect();
                }
                "map".to_string()
            },
            DataType::FixedSizeBinary(len) => format!("fixed[{}]", len),
            DataType::Int32 => "int".to_string(),
            DataType::Int64 => "long".to_string(),
            DataType::Float32 => "float".to_string(),
            DataType::Float64 => "double".to_string(),
            DataType::Utf8 => "string".to_string(),
            DataType::LargeUtf8 => "largeutf8".to_string(),
            DataType::Binary => "binary".to_string(),
            DataType::LargeBinary => "largebinary".to_string(),
            DataType::Boolean => "boolean".to_string(),
            DataType::Date32 => "date".to_string(),
            DataType::Date64 => "date64".to_string(),
            DataType::Timestamp(unit, tz) => {
                let unit_str = match unit {
                    arrow::datatypes::TimeUnit::Second => "second",
                    arrow::datatypes::TimeUnit::Millisecond => "millisecond",
                    arrow::datatypes::TimeUnit::Microsecond => "microsecond",
                    arrow::datatypes::TimeUnit::Nanosecond => "nanosecond",
                };
                if let Some(tz_val) = tz {
                    format!("timestamp({}, {})", unit_str, tz_val.to_lowercase())
                } else {
                    format!("timestamp({}, none)", unit_str)
                }
            },
            DataType::Time32(unit) => {
                let unit_str = match unit {
                    arrow::datatypes::TimeUnit::Second => "second",
                    arrow::datatypes::TimeUnit::Millisecond => "millisecond",
                    _ => "millisecond",
                };
                format!("time32({})", unit_str)
            },
            DataType::Time64(unit) => {
                let unit_str = match unit {
                    arrow::datatypes::TimeUnit::Microsecond => "microsecond",
                    arrow::datatypes::TimeUnit::Nanosecond => "nanosecond",
                    _ => "microsecond",
                };
                format!("time64({})", unit_str)
            },
            dt => dt.to_string().to_lowercase(),
        };

        let field_id = f.metadata().get("iceberg.id")
            .and_then(|id_str| id_str.parse::<i32>().ok())
            .unwrap_or(id);

        SchemaField {
            id: field_id,
            name: f.name().clone(),
            type_str,
            required: !f.is_nullable(),
            fields: nested_fields,
            initial_default: None,
            write_default: None,
        }
    }

    pub fn to_arrow(&self) -> arrow::datatypes::Field {
         let dt = match self.type_str.to_lowercase().as_str() {
             "int32" | "int" => arrow::datatypes::DataType::Int32,
             "int64" | "long" => arrow::datatypes::DataType::Int64,
             "utf8" | "string" => arrow::datatypes::DataType::Utf8,
             "float32" | "float" => arrow::datatypes::DataType::Float32,
             "float64" | "double" => arrow::datatypes::DataType::Float64,
             "boolean" | "bool" => arrow::datatypes::DataType::Boolean,
             "timestamp(microsecond, none)" => arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None),
             "timestamp(nanosecond, none)" => arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None),
             // Handle UTC timezone specifically if requested
             "timestamp(microsecond, utc)" => arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("UTC".into())),
             "date" | "date32" => arrow::datatypes::DataType::Date32,
             "date64" => arrow::datatypes::DataType::Date64,
             "binary" => arrow::datatypes::DataType::Binary,
             "largebinary" => arrow::datatypes::DataType::LargeBinary,
             "largeutf8" => arrow::datatypes::DataType::LargeUtf8,
             // Handle all timestamp variants: "timestamp", "timestamptz", "timestamp(unit, tz)"
             s if s == "timestamp" || s == "timestamptz" || s.starts_with("timestamp(") => {
                 if s == "timestamptz" {
                     arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, Some("UTC".into()))
                 } else if s == "timestamp" {
                     arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None)
                 } else {
                     // Parse "timestamp(unit, tz_or_none)"
                     let inner = s.trim_start_matches("timestamp(").trim_end_matches(')');
                     let parts: Vec<&str> = inner.splitn(2, ',').map(|p| p.trim()).collect();
                     let unit = match parts.first().map(|s| *s) {
                         Some("second") => arrow::datatypes::TimeUnit::Second,
                         Some("millisecond") => arrow::datatypes::TimeUnit::Millisecond,
                         Some("nanosecond") => arrow::datatypes::TimeUnit::Nanosecond,
                         _ => arrow::datatypes::TimeUnit::Microsecond,
                     };
                     let tz = parts.get(1).and_then(|t| {
                         if *t == "none" { None } else { Some(t.to_string().into()) }
                     });
                     arrow::datatypes::DataType::Timestamp(unit, tz)
                 }
             },
             s if s.contains("time32") => {
                 if s.contains("millisecond") {
                     arrow::datatypes::DataType::Time32(arrow::datatypes::TimeUnit::Millisecond)
                 } else {
                     arrow::datatypes::DataType::Time32(arrow::datatypes::TimeUnit::Second)
                 }
             },
             s if s.contains("time64") => {
                 if s.contains("nanosecond") {
                     arrow::datatypes::DataType::Time64(arrow::datatypes::TimeUnit::Nanosecond)
                 } else {
                     arrow::datatypes::DataType::Time64(arrow::datatypes::TimeUnit::Microsecond)
                 }
             },
             s if s.contains("fixedsizelist") || s.contains("fixed_list") => {
                 let dim = s.split(|c: char| !c.is_numeric())
                    .filter_map(|p| p.parse::<i32>().ok())
                    .next()
                    .unwrap_or(0);
                 arrow::datatypes::DataType::FixedSizeList(
                     Arc::new(arrow::datatypes::Field::new("item", arrow::datatypes::DataType::Float32, true)),
                     dim
                 )
             },
             s if s.starts_with("fixed[") => {
                 let len = s.trim_start_matches("fixed[").trim_end_matches(']')
                    .parse::<i32>().unwrap_or(0);
                 arrow::datatypes::DataType::FixedSizeBinary(len)
             },
             "struct" => {
                 let arrow_fields = self.fields.iter().map(|f| f.to_arrow()).collect();
                 arrow::datatypes::DataType::Struct(arrow_fields)
             },
             "list" => {
                 let item_field = self.fields.first().map(|f| f.to_arrow())
                    .unwrap_or(arrow::datatypes::Field::new("item", arrow::datatypes::DataType::Utf8, true));
                 arrow::datatypes::DataType::List(Arc::new(item_field))
             },
             "map" => {
                 let key_field = self.fields.first().map(|f| f.to_arrow())
                    .unwrap_or(arrow::datatypes::Field::new("key", arrow::datatypes::DataType::Utf8, false));
                 let value_field = self.fields.get(1).map(|f| f.to_arrow())
                    .unwrap_or(arrow::datatypes::Field::new("value", arrow::datatypes::DataType::Utf8, true));
                 
                 arrow::datatypes::DataType::Map(
                     Arc::new(arrow::datatypes::Field::new("entries", arrow::datatypes::DataType::Struct(vec![
                         key_field, value_field
                     ].into()), false)),
                     false
                 )
             },
             s if s == "decimal" || s.starts_with("decimal(") || s.starts_with("decimal128(") => {
                 let parts: Vec<&str> = if s.starts_with("decimal(") {
                     s.trim_start_matches("decimal(").trim_end_matches(')')
                 } else {
                     s.trim_start_matches("decimal128(").trim_end_matches(')')
                 }
                     .split(',')
                     .map(|p| p.trim())
                     .collect();
                 let precision = parts.first().and_then(|p| p.parse::<u8>().ok()).unwrap_or(38);
                 let scale = parts.get(1).and_then(|p| p.parse::<i8>().ok()).unwrap_or(10);
                 arrow::datatypes::DataType::Decimal128(precision, scale)
             },
             _ => arrow::datatypes::DataType::Utf8
         };
         let mut f = arrow::datatypes::Field::new(&self.name, dt, !self.required);
         f.set_metadata(std::collections::HashMap::from([("iceberg.id".to_string(), self.id.to_string())]));
         f
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Manifest {
    pub version: u64,
    /// Iceberg table format version (1, 2, or 3)
    #[serde(default = "default_format_version")]
    pub format_version: i32,
    pub timestamp_ms: i64,
    /// Path to the ManifestList file (Iceberg-style scalability)
    #[serde(default)]
    pub manifest_list_path: Option<String>,
    /// List of active entries (Directly in manifest for small tables, otherwise in ManifestList)
    pub entries: Vec<ManifestEntry>, 
    /// Pointer to previous version (for history/rollback)
    pub prev_version: Option<u64>,
    /// Explicit Schema Tracking (Iceberg-style)
    #[serde(default)]
    pub schemas: Vec<Schema>,
    #[serde(default)]
    pub current_schema_id: i32,
    /// Partition Specification
    #[serde(default)]
    pub partition_spec: PartitionSpec,
    #[serde(default)]
    pub partition_specs: Vec<PartitionSpec>,
    #[serde(default)]
    pub default_spec_id: i32,
    /// Sort Orders (Iceberg spec)
    #[serde(default)]
    pub sort_orders: Vec<SortOrder>,
    #[serde(default)]
    pub default_sort_order_id: i32,
    #[serde(default)]
    pub properties: HashMap<String, String>,
    #[serde(default)]
    pub last_column_id: i32,
}

fn default_format_version() -> i32 {
    2 // Default to V2 for backward compatibility
}

impl Manifest {
    pub fn new(version: u64, entries: Vec<ManifestEntry>, prev_version: Option<u64>) -> Self {
        Self {
            version,
            format_version: 2,
            timestamp_ms: Utc::now().timestamp_millis(),
            manifest_list_path: None,
            entries,
            prev_version,
            schemas: Vec::new(),
            current_schema_id: 0,
            partition_spec: PartitionSpec::default(),
            partition_specs: Vec::new(),
            default_spec_id: 0,
            sort_orders: Vec::new(),
            default_sort_order_id: 0,
            properties: HashMap::new(),
            last_column_id: 0,
        }
    }
    
    pub fn new_with_schema(version: u64, entries: Vec<ManifestEntry>, prev_version: Option<u64>, schemas: Vec<Schema>, current_schema_id: i32) -> Self {
        let last_id = schemas.iter().flat_map(|s| s.fields.iter().map(|f| f.id)).max().unwrap_or(0);
        Self {
            version,
            format_version: 2,
            timestamp_ms: Utc::now().timestamp_millis(),
            manifest_list_path: None,
            entries,
            prev_version,
            schemas,
            current_schema_id,
            partition_spec: PartitionSpec::default(),
            partition_specs: Vec::new(),
            default_spec_id: 0,
            sort_orders: Vec::new(),
            default_sort_order_id: 0,
            properties: HashMap::new(),
            last_column_id: last_id,
        }
    }

    pub fn new_with_spec(
        version: u64, 
        entries: Vec<ManifestEntry>, 
        prev_version: Option<u64>, 
        schema_list: Vec<Schema>, 
        current_schema_id: i32,
        partition_spec: PartitionSpec
    ) -> Self {
        let last_id = schema_list.iter().flat_map(|s| s.fields.iter().map(|f| f.id)).max().unwrap_or(0);
        let spec_id = partition_spec.spec_id;
        Self {
            version,
            format_version: 2,
            timestamp_ms: Utc::now().timestamp_millis(),
            manifest_list_path: None,
            entries,
            prev_version,
            schemas: schema_list,
            current_schema_id,
            partition_spec: partition_spec.clone(),
            partition_specs: vec![partition_spec], // Track spec history
            default_spec_id: spec_id,
            sort_orders: Vec::new(),
            default_sort_order_id: 0,
            properties: HashMap::new(),
            last_column_id: last_id,
        }
    }
}

const MAX_ENTRIES_PER_MANIFEST: usize = 10000;

#[derive(Clone)]
pub struct ManifestManager {
    store: Arc<dyn ObjectStore>,
    manifest_dir: Path,
    root_uri: String,
}

#[derive(Debug, Default, Clone)]
pub struct CommitMetadata {
    pub updated_schemas: Option<Vec<Schema>>,
    pub updated_schema_id: Option<i32>,
    pub updated_partition_specs: Option<Vec<PartitionSpec>>,
    pub updated_default_spec_id: Option<i32>,
    pub updated_properties: Option<HashMap<String, String>>,
    pub removed_properties: Option<Vec<String>>,
    pub updated_sort_orders: Option<Vec<SortOrder>>,
    pub updated_default_sort_order_id: Option<i32>,
    pub updated_last_column_id: Option<i32>,
}

impl ManifestManager {
    pub fn new(store: Arc<dyn ObjectStore>, base_path: &str, root_uri: &str) -> Self {
        // Manifest directory is typically `_manifest/` under the table root
        let manifest_dir = if base_path.is_empty() {
             Path::from("_manifest/")
        } else {
             Path::from(format!("{}/_manifest/", base_path))
        };
        
        Self {
            store,
            manifest_dir,
            root_uri: root_uri.trim_end_matches('/').to_string(),
        }
    }

    fn get_cache_key(&self, path: &Path) -> String {
        format!("{}/{}", self.root_uri, path)
    }

    fn get_dir_cache_key(&self) -> String {
        format!("{}/{}", self.root_uri, self.manifest_dir)
    }

    /// Check if any manifests exist in the directory
    pub async fn exists(&self) -> Result<bool> {
        let mut stream = self.store.list(Some(&self.manifest_dir));
        if let Some(meta) = stream.next().await {
            let _ = meta?;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Load the latest manifest. Returns (Manifest, version_number).
    /// If no manifest exists, returns an empty Manifest with version 0.
    pub async fn load_latest(&self) -> Result<(Manifest, u64)> {
        let cache_key = self.get_dir_cache_key();

        // 1. Check Version Cache (Fast Path)
        if let Some(ver) = crate::core::cache::LATEST_VERSION_CACHE.get(&cache_key).await {
            // We have a "hint" of what the latest version is. 
            // We can try to load that directly.
            // Note: In strict consistency systems, we might want to re-check List occasionally?
            // But for now, we trust the cache for its TTL duration (2s).
            if let Ok(manifest) = self.load_version(ver).await {
                return Ok((manifest, ver));
            }
        }

        // 2. Slow Path: List files in _manifest/
        let mut stream = self.store.list(Some(&self.manifest_dir));
        let mut max_ver = 0;
        let mut latest_path = None;

        while let Some(meta) = stream.next().await {
            let meta = meta?;
            let path_str = meta.location.to_string();
            // Expected format: v{N}.json
            if let Some(filename) = path_str.split('/').next_back() {
                if filename.starts_with('v') && filename.ends_with(".json") {
                    let ver_str = &filename[1..filename.len()-5]; // strip 'v' and '.json'
                    if let Ok(ver) = ver_str.parse::<u64>() {
                        if ver > max_ver {
                            max_ver = ver;
                            latest_path = Some(meta.location);
                        }
                    }
                }
            }
        }

        // Update Version Cache
        if max_ver > 0 {
            crate::core::cache::LATEST_VERSION_CACHE.insert(cache_key, max_ver).await;
        }

        if let Some(path) = latest_path { // Only used if not using cache for retrieving body?
            // Actually, we can just use load_version(max_ver) which handles caching of body
             return match self.load_version(max_ver).await {
                 Ok(m) => Ok((m, max_ver)),
                 Err(_) => {
                     // Fallback if somehow listing said it exists but we can't read it
                     let bytes = self.store.get(&path).await?.bytes().await?;
                     let manifest: Manifest = serde_json::from_slice(&bytes)?;
                     Ok((manifest, max_ver))
                 }
             }
        } else {
            // No manifest found, return empty genesis
            Ok((Manifest::new(0, Vec::new(), None), 0))
        }
    }

    /// Load the latest manifest and ALL its entries (including sharded ones)
    pub async fn load_latest_full(&self) -> Result<(Manifest, Vec<ManifestEntry>, u64)> {
        let (manifest, ver) = self.load_latest().await?;
        let entries = self.load_all_entries(&manifest).await?;
        Ok((manifest, entries, ver))
    }

    /// Load a specific version of the manifest
    pub async fn load_version(&self, version: u64) -> Result<Manifest> {
        let filename = format!("v{}.json", version);
        let path = self.manifest_dir.child(filename);
        let cache_key = self.get_cache_key(&path);

        // 1. Check Data Cache
        if let Some(manifest) = crate::core::cache::MANIFEST_CACHE.get(&cache_key).await {
            return Ok(manifest.as_ref().clone());
        }

        // 2. Fetch from S3
        let bytes = self.store.get(&path).await?.bytes().await?;
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        
        // 3. Populate Cache
        crate::core::cache::MANIFEST_CACHE.insert(cache_key, Arc::new(manifest.clone())).await;

        Ok(manifest)
    }

    /// Load a manifest list from a specific path
    pub async fn load_manifest_list(&self, path_str: &str) -> Result<ManifestList> {
        let path = Path::from(path_str);
        let cache_key = format!("{}/{}", self.root_uri, path);

        if let Some(list) = crate::core::cache::MANIFEST_LIST_CACHE.get(&cache_key).await {
            return Ok(list.as_ref().clone());
        }

        let bytes = self.store.get(&path).await?.bytes().await?;
        
        if path_str.ends_with(".avro") {
            let iceberg_list = crate::core::iceberg::read_manifest_list(&bytes[..])?;
            let manifest_files = iceberg_list.into_iter().map(|e| {
                ManifestListEntry {
                    manifest_path: e.manifest_path,
                    manifest_length: e.manifest_length,
                    partition_spec_id: e.partition_spec_id,
                    content: e.content,
                    sequence_number: e.sequence_number,
                    min_sequence_number: e.min_sequence_number,
                    added_snapshot_id: e.added_snapshot_id,
                    added_files_count: e.added_files_count,
                    existing_files_count: e.existing_files_count,
                    deleted_files_count: e.deleted_files_count,
                    added_rows_count: e.added_rows_count,
                    existing_rows_count: e.existing_rows_count,
                    deleted_rows_count: e.deleted_rows_count,
                    partition_stats: HashMap::new(), // Stats not parsed yet
                }
            }).collect();
            let list = ManifestList {
                manifest_files,
            };
            crate::core::cache::MANIFEST_LIST_CACHE.insert(cache_key, Arc::new(list.clone())).await;
            return Ok(list);
        }

        let list: ManifestList = serde_json::from_slice(&bytes)?;
        
        crate::core::cache::MANIFEST_LIST_CACHE.insert(cache_key, Arc::new(list.clone())).await;
        Ok(list)
    }

    /// Save a manifest list to storage
    pub async fn save_manifest_list(&self, list: &ManifestList, version: u64) -> Result<String> {
        let filename = format!("list-v{}.json", version);
        let path = self.manifest_dir.child(filename);
        let bytes = serde_json::to_vec_pretty(list)?;
        
        self.store.put(&path, bytes.into()).await?;
        Ok(path.to_string())
    }

    /// Recursively load all entries from a manifest (including those in manifest_list_path)
    pub async fn load_all_entries(&self, manifest: &Manifest) -> Result<Vec<ManifestEntry>> {
        let mut all_entries = manifest.entries.clone();
        
        if let Some(list_path) = &manifest.manifest_list_path {
            let list = self.load_manifest_list(list_path).await?;
            
            // Resolve schema for stats decoding
            let schema = manifest.schemas.iter()
                .find(|s| s.schema_id == manifest.current_schema_id)
                .or(manifest.schemas.last()); // Fallback

            for entry in list.manifest_files {
                if entry.manifest_path.ends_with(".avro") {
                    if let Some(s) = schema {
                        let sub_manifest = self.load_avro_manifest(&entry.manifest_path, s, &manifest.partition_spec).await?;
                        all_entries.extend(sub_manifest.entries);
                    } else {
                        // Log warning or skip stats?
                         tracing::warn!("No schema available to decode Avro manifest");
                          let sub_manifest = self.load_avro_manifest(&entry.manifest_path, &Schema { schema_id: 0, fields: vec![], identifier_field_ids: vec![] }, &manifest.partition_spec).await?;
                         all_entries.extend(sub_manifest.entries);
                    }
                } else {
                    let sub_manifest = self.load_manifest_from_path(&entry.manifest_path).await?;
                    all_entries.extend(sub_manifest.entries);
                }
            }
        }
        
        Ok(all_entries)
    }

    /// Helper to load a manifest from an arbitrary path
    async fn load_manifest_from_path(&self, path_str: &str) -> Result<Manifest> {
        let path = Path::from(path_str);
        let cache_key = format!("{}/{}", self.root_uri, path);

        if let Some(manifest) = crate::core::cache::MANIFEST_CACHE.get(&cache_key).await {
            return Ok(manifest.as_ref().clone());
        }

        let bytes = self.store.get(&path).await?.bytes().await?;
        let manifest: Manifest = serde_json::from_slice(&bytes)?;
        
        crate::core::cache::MANIFEST_CACHE.insert(cache_key, Arc::new(manifest.clone())).await;
        Ok(manifest)
    }

    async fn load_avro_manifest(&self, path_str: &str, schema: &Schema, spec: &PartitionSpec) -> Result<Manifest> {
         let path = Path::from(path_str);
         let cache_key = format!("{}/{}", self.root_uri, path);
         
         if let Some(manifest) = crate::core::cache::MANIFEST_CACHE.get(&cache_key).await {
             return Ok(manifest.as_ref().clone());
         }

         let bytes = self.store.get(&path).await?.bytes().await?;
         let iceberg_entries = crate::core::iceberg::read_manifest(&bytes[..])?;
         
         let mut data_entries = Vec::new();
         let mut delete_files = Vec::new();
         
         for ie in iceberg_entries {
             if ie.status == 0 || ie.status == 1 { // EXISTING or ADDED
                 match crate::core::iceberg::convert_iceberg_to_object(&ie, schema, spec)? {
                     crate::core::iceberg::IcebergManifestObject::Data(me) => data_entries.push(me),
                     crate::core::iceberg::IcebergManifestObject::Delete(df) => delete_files.push(df),
                 }
             }
         }
         
         // Simple linking of equality deletes to data files in same partition
         // Matches logic in iceberg_rest.rs
         for data in &mut data_entries {
             for delete in &delete_files {
                 if data.partition_values == delete.partition_values {
                     data.delete_files.push(delete.clone());
                 }
             }
         }
         
         let manifest = Manifest::new(0, data_entries, None);
         crate::core::cache::MANIFEST_CACHE.insert(cache_key, Arc::new(manifest.clone())).await;
         Ok(manifest)
    }

    /// Walk back history starting from the latest version.
    /// Returns a list of Manifests [Latest, Latest-1, ... Genesis]
    pub async fn walk_history(&self) -> Result<Vec<Manifest>> {
        let (latest, _) = self.load_latest().await?;
        if latest.version == 0 {
            return Ok(vec![]);
        }

        let mut history = Vec::new();
        history.push(latest.clone());

        let mut current = latest;
        while let Some(prev) = current.prev_version {
            // Safety break for now, though u64 prevents inf loops
            if prev == 0 { break; } 
            
            match self.load_version(prev).await {
                Ok(m) => {
                    history.push(m.clone());
                    current = m;
                },
                Err(e) => {
                    tracing::warn!("Broken manifest chain at v{}: {}", prev, e);
                    break;
                }
            }
        }
        
        Ok(history)
    }

    /// Rollback the table state to a previous manifest version
    pub async fn rollback_to_snapshot(&self, version: u64) -> Result<Manifest> {
        let max_retries = 10;
        let mut attempt = 0;
        
        loop {
            let (_current_manifest, current_ver) = self.load_latest().await?;
            let new_ver = current_ver + 1;
            
            // Load the target version we want to rollback to
            let target_manifest = self.load_version(version).await?;
            
            // Create a new manifest that is a copy of the target, 
            // but with a new version number and pointing to the current version as previous.
            let mut new_manifest = target_manifest.clone();
            new_manifest.version = new_ver;
            new_manifest.prev_version = Some(current_ver);
            
            let filename = format!("v{}.json", new_ver);
            let path = self.manifest_dir.child(filename);
            let bytes = serde_json::to_vec_pretty(&new_manifest)?;
            
            use object_store::{PutMode, PutOptions};
            let opts = PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            };
            
            match self.store.put_opts(&path, bytes.into(), opts).await {
                Ok(_) => {
                    tracing::info!("Rolled back Manifest to v{} (from snapshot {})", new_ver, version);
                    let dir_key = format!("{}/{}", self.root_uri, self.manifest_dir);
                    crate::core::cache::LATEST_VERSION_CACHE.invalidate(&dir_key).await;
                    let file_key = format!("{}/{}", self.root_uri, path);
                    crate::core::cache::MANIFEST_CACHE.insert(file_key, Arc::new(new_manifest.clone())).await;
                    return Ok(new_manifest);
                }
                Err(e) if is_already_exists(&e) => {
                     attempt += 1;
                     if attempt >= max_retries { break; }
                     tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                     continue;
                }
                Err(e) => return Err(e.into())
            }
        }
        Err(anyhow::anyhow!("Failed to rollback after {} attempts", max_retries))
    }



    /// Commit a change to the timeline.
    /// Uses optimistic concurrency control with retries and PutMode::Create
    /// to ensure atomicity under high concurrency.
    pub async fn commit(
        &self, 
        add_entries: &[ManifestEntry], 
        remove_paths: &[String],
        metadata: CommitMetadata
    ) -> Result<Manifest> {
        let max_retries = 100;
        
        for attempt in 0..max_retries {
            let (current_manifest, current_ver) = self.load_latest().await?;
            
            // 1. Calculate new state
            // Load ALL entries including those in manifest lists
            let mut all_entries = self.load_all_entries(&current_manifest).await?;
            
            // --- BOOTSTRAP: If this is the first commit (v1), discover existing files ---
            if current_ver == 0 && all_entries.is_empty() {
                // If it's a genesis manifest, check if there's any existing data on disk
                // This bridges the discovery flow with the versioned flow.
                let mut stream = self.store.list(None);
                while let Some(meta) = stream.next().await {
                    let meta = meta?;
                    let path_str = meta.location.to_string();
                    if path_str.ends_with(".parquet") && !path_str.contains("_wal/") {
                         // Create a basic entry for the discovered file
                         all_entries.push(ManifestEntry {
                             file_path: path_str,
                             record_count: 0, 
                             file_size_bytes: meta.size as i64,
                             column_stats: HashMap::new(),
                             index_files: Vec::new(),
                             partition_values: HashMap::new(),
                             clustering_strategy: None,
                             clustering_columns: None,
                             min_clustering_score: None,
                             max_clustering_score: None,
                             normalization_mins: None,
                             normalization_maxs: None,
                             delete_files: Vec::new(),
                         });
                    }
                }
            }
            
            // Map for easy removal
            let mut active_map: HashMap<String, ManifestEntry> = all_entries.into_iter()
                .map(|e| (e.file_path.clone(), e))
                .collect();
            
            let mut some_paths_missing = false;
            for path in remove_paths {
                if active_map.remove(path).is_none() {
                    some_paths_missing = true;
                }
            }
            
            let new_ver = current_ver + 1;
            
            // If this is a compaction/replacement (has add_entries and remove_paths)
            // and some paths we intended to remove are ALREADY gone, it means
            // a concurrent compaction or delete happened. We must ABORT to avoid duplication.
            if !add_entries.is_empty() && !remove_paths.is_empty() && some_paths_missing {
                tracing::debug!("Aborting commit v{} attempt {}: some remove_paths were already gone", new_ver, attempt);
                return Ok(current_manifest); 
            }

            for entry in add_entries {
                active_map.insert(entry.file_path.clone(), entry.clone());
            }
            
            let new_entries: Vec<ManifestEntry> = active_map.into_values().collect();
            
            // 2. Decide if we need a ManifestList (Scalability)
            let (final_entries, manifest_list_path) = if new_entries.len() > MAX_ENTRIES_PER_MANIFEST {
                // Split entries into multiple manifests
                let mut manifest_files = Vec::new();
                let chunks = new_entries.chunks(MAX_ENTRIES_PER_MANIFEST);

                for chunk in chunks {
                    let uuid = uuid::Uuid::new_v4();
                    let filename = format!("{}-m0.avro", uuid);
                    let path = self.manifest_dir.child(filename);
                    
                    let writer = crate::core::iceberg::IcebergWriter::new();
                    let default_schema = crate::core::manifest::Schema::default();
                    let table_schema = current_manifest.schemas.last().unwrap_or(&default_schema);
                    let bytes = writer.write_manifest_file(chunk, &current_manifest.partition_spec, table_schema, new_ver as i64, new_ver as i64)?;
                    let manifest_length = bytes.len() as i64;
                    let rows_count: i64 = chunk.iter().map(|e| e.record_count).sum();
                    
                    self.store.put(&path, bytes.into()).await?;
                    
                    manifest_files.push(ManifestListEntry {
                        manifest_path: path.to_string(),
                        manifest_length,
                        partition_spec_id: current_manifest.partition_spec.spec_id,
                        content: 0, // Data
                        sequence_number: new_ver as i64,
                        min_sequence_number: new_ver as i64,
                        added_snapshot_id: new_ver as i64,
                        added_files_count: chunk.len() as i32,
                        existing_files_count: 0,
                        deleted_files_count: 0,
                        added_rows_count: rows_count,
                        existing_rows_count: 0,
                        deleted_rows_count: 0,
                        partition_stats: HashMap::new(), 
                    });
                }
                
                let list_uuid = uuid::Uuid::new_v4();
                let list_filename = format!("snap-{}-{}.avro", new_ver, list_uuid);
                let list_path_loc = self.manifest_dir.child(list_filename);
                
                let writer = crate::core::iceberg::IcebergWriter::new();
                let list_bytes = writer.write_manifest_list(&manifest_files)?;
                self.store.put(&list_path_loc, list_bytes.into()).await?;
                
                (Vec::new(), Some(list_path_loc.to_string()))
            } else {
                // Determine if we should force Avro for single chunk too?
                // Yes, for consistency.
                let uuid = uuid::Uuid::new_v4();
                let filename = format!("{}-m0.avro", uuid);
                let path = self.manifest_dir.child(filename);
                
                let writer = crate::core::iceberg::IcebergWriter::new();
                let default_schema = crate::core::manifest::Schema::default();
                let table_schema = current_manifest.schemas.last().unwrap_or(&default_schema);
                let bytes = writer.write_manifest_file(&new_entries, &current_manifest.partition_spec, table_schema, new_ver as i64, new_ver as i64)?;
                let manifest_length = bytes.len() as i64;
                let rows_count: i64 = new_entries.iter().map(|e| e.record_count).sum();
                
                self.store.put(&path, bytes.into()).await?;
                
                let manifest_files = vec![ManifestListEntry {
                    manifest_path: path.to_string(),
                    manifest_length,
                    partition_spec_id: current_manifest.partition_spec.spec_id,
                    content: 0,
                    sequence_number: new_ver as i64,
                    min_sequence_number: new_ver as i64,
                    added_snapshot_id: new_ver as i64,
                    added_files_count: new_entries.len() as i32,
                    existing_files_count: 0,
                    deleted_files_count: 0,
                    added_rows_count: rows_count,
                    existing_rows_count: 0,
                    deleted_rows_count: 0,
                    partition_stats: HashMap::new(),
                }];
                
                let list_uuid = uuid::Uuid::new_v4();
                let list_filename = format!("snap-{}-{}.avro", new_ver, list_uuid);
                let list_path_loc = self.manifest_dir.child(list_filename);
                
                let list_bytes = writer.write_manifest_list(&manifest_files)?;
                self.store.put(&list_path_loc, list_bytes.into()).await?;
                
                (Vec::new(), Some(list_path_loc.to_string()))
            };
            
            // 3. Create new Manifest
            let final_schemas = metadata.updated_schemas.as_ref().cloned().unwrap_or_else(|| current_manifest.schemas.clone());
            let final_schema_id = metadata.updated_schema_id.unwrap_or(current_manifest.current_schema_id);

            // Apply Partition Spec Updates

            // But Manifest struct currently only has `partition_spec`. 
            // Iceberg Manifest File implies spec is per manifest, but Table Metadata has list.
            // Here we are creating Table Metadata essentially? No, this is Manifest (vN.json is Table Metadata in our hybrid model).
            // Yes, vN.json IS Table Metadata. So we should update fields like partition_specs, properties etc.
            
            // Wait, Manifest struct has:
            // pub partition_spec: PartitionSpec,
            // pub sort_orders: Vec<SortOrder>,
            
            // It seems we need to update Manifest struct to hold list of specs if we want to be fully compliant, 
            // or we just switch the current `partition_spec`.
            // The `Manifest` struct in `manifest.rs` (vN.json) roughly maps to Iceberg Table Metadata.
            
            let final_partition_spec = if let Some(specs) = &metadata.updated_partition_specs {
                 // If specs provided, pick the one matching default_spec_id or just use the last one?
                 // Usually we add a spec and set it as default.
                 // For now, let's assume if specs are updated, we use the last one as current.
                 specs.last().cloned().unwrap_or(current_manifest.partition_spec.clone()) 
            } else {
                 current_manifest.partition_spec.clone()
            };
            
            let final_sort_orders = metadata.updated_sort_orders.as_ref().cloned().unwrap_or_else(|| current_manifest.sort_orders.clone());
            let final_default_sort_order_id = metadata.updated_default_sort_order_id.unwrap_or(current_manifest.default_sort_order_id);
            
            // Properties
            // Not in Manifest struct? Checking definition...
            // Manifest struct line 297 doesn't show `properties`.
            // I need to add `properties` to Manifest struct if I want to support them!
            // Line 128 shows `pub properties: HashMap<String, String>` in IcebergTableMetadata, but Manifest is our internal representation.
            // Let's assume for now I will add it to Manifest struct in a separate edit if missing.
            // Looking at `view_file` output from earlier (step 335), Manifest struct ends at line 320.
            // It DOES NOT have properties.
            
            let mut new_manifest = Manifest::new_with_spec(
                new_ver, 
                final_entries, 
                Some(current_ver),
                final_schemas,
                final_schema_id,
                final_partition_spec,
            );
            
            new_manifest.sort_orders = final_sort_orders;
            new_manifest.default_sort_order_id = final_default_sort_order_id;
            
            // Apply final properties
            new_manifest.properties = current_manifest.properties.clone();
            if let Some(props) = &metadata.updated_properties {
                tracing::debug!("Applying property updates: {:?}", props);
                new_manifest.properties.extend(props.clone().into_iter());
            }
            if let Some(removals) = &metadata.removed_properties {
                 for key in removals {
                     new_manifest.properties.remove(key);
                 }
            }
            
            new_manifest.partition_specs = if let Some(specs) = &metadata.updated_partition_specs {
                specs.clone()
            } else {
                current_manifest.partition_specs.clone()
            };
            new_manifest.default_spec_id = metadata.updated_default_spec_id.unwrap_or(current_manifest.default_spec_id);
            new_manifest.last_column_id = metadata.updated_last_column_id.unwrap_or(current_manifest.last_column_id);

            // Manifest Metadata Updates logic (partial)
            new_manifest.manifest_list_path = manifest_list_path;
            
            // 4. Write v{N+1}.json with PutMode::Create (Atomic)
            let filename = format!("v{}.json", new_ver);
            let path = self.manifest_dir.child(filename);
            
            let bytes = serde_json::to_vec_pretty(&new_manifest)?;

            use object_store::{PutMode, PutOptions};
            let opts = PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            };

            match self.store.put_opts(&path, bytes.into(), opts).await {
                Ok(_) => {
                    println!("Committed Manifest: {}", path);
                    // 5. Update Caches
                    let dir_key = format!("{}/{}", self.root_uri, self.manifest_dir);
                    crate::core::cache::LATEST_VERSION_CACHE.invalidate(&dir_key).await;
                    crate::core::cache::LATEST_VERSION_CACHE.insert(dir_key, new_ver).await;
                    
                    // Cache the new manifest file eagerly
                    let file_key = format!("{}/{}", self.root_uri, path);
                    crate::core::cache::MANIFEST_CACHE.insert(file_key, Arc::new(new_manifest.clone())).await;

                    return Ok(new_manifest);
                }
                Err(e) if is_already_exists(&e) => {
                    // Conflict logic...
                    if attempt % 10 == 0 || attempt > 90 {
                        tracing::debug!("Conflict committing Manifest v{} (attempt {}), retrying...", new_ver, attempt + 1);
                    }
                    let base_delay = 10 * (2u64.pow(attempt.min(5) as u32));
                    let jitter = rand::random::<u64>() % base_delay;
                    tokio::time::sleep(std::time::Duration::from_millis(base_delay + jitter)).await;
                    continue;
                }
                Err(e) => return Err(e.into()),
            }
        }
        
        Err(anyhow::anyhow!("Failed to commit manifest after {} attempts due to concurrent updates", max_retries))
    }

    /// Commit a set of imported entries (merges with current state)
    pub async fn commit_imported_entries(&self, entries: Vec<ManifestEntry>) -> Result<Manifest> {
        let (current_manifest, current_ver) = self.load_latest().await?;
        let all_existing = self.load_all_entries(&current_manifest).await?;
        
        // Merge entries, avoid duplicates
        let mut entry_map: HashMap<String, ManifestEntry> = all_existing.into_iter()
            .map(|e| (e.file_path.clone(), e))
            .collect();
            
        for entry in entries {
            entry_map.insert(entry.file_path.clone(), entry);
        }
        
        let merged_entries: Vec<ManifestEntry> = entry_map.into_values().collect();
        let new_ver = current_ver + 1;
        
        let mut new_manifest = Manifest::new_with_spec(
            new_ver,
            merged_entries,
            Some(current_ver),
            current_manifest.schemas.clone(),
            current_manifest.current_schema_id,
            current_manifest.partition_spec.clone(),
        );

        new_manifest.partition_specs = current_manifest.partition_specs.clone();
        new_manifest.default_spec_id = current_manifest.default_spec_id;
        new_manifest.properties = current_manifest.properties.clone();
        new_manifest.sort_orders = current_manifest.sort_orders.clone();
        new_manifest.default_sort_order_id = current_manifest.default_sort_order_id;
        
        // Write to storage
        let filename = format!("v{}.json", new_ver);
        let path = self.manifest_dir.child(filename);
        let bytes = serde_json::to_vec_pretty(&new_manifest)?;
        
        use object_store::{PutMode, PutOptions};
        let opts = PutOptions {
            mode: PutMode::Create,
            ..Default::default()
        };
        
        self.store.put_opts(&path, bytes.into(), opts).await?;
        tracing::info!("Imported {} external entries into Manifest v{}", new_manifest.entries.len(), new_ver);
        
        // Update Caches
        let dir_key = format!("{}/{}", self.root_uri, self.manifest_dir);
        crate::core::cache::LATEST_VERSION_CACHE.invalidate(&dir_key).await;
        
        // Cache the new manifest file eagerly
        let file_key = format!("{}/{}", self.root_uri, path);
        crate::core::cache::MANIFEST_CACHE.insert(file_key, Arc::new(new_manifest.clone())).await;
        
        Ok(new_manifest)
    }

    pub async fn update_schema(&self, new_schemas: Vec<Schema>, new_schema_id: i32, last_column_id: Option<i32>) -> Result<Manifest> {
        let max_retries = 10;
        let mut attempt = 0;
        
        loop {
            // Optimistic Concurrency Control Loop
            let (current_manifest, current_ver) = self.load_latest().await?;
            let new_ver = current_ver + 1;
            
            // Re-use entries from latest
            let entries = current_manifest.entries.clone();
            
            let mut new_manifest = Manifest::new_with_spec(
                new_ver, 
                entries, 
                Some(current_ver),
                new_schemas.clone(),
                new_schema_id,
                current_manifest.partition_spec.clone(),
            );
            new_manifest.last_column_id = last_column_id.unwrap_or(current_manifest.last_column_id);
            
            let filename = format!("v{}.json", new_ver);
            let path = self.manifest_dir.child(filename);
            let bytes = serde_json::to_vec_pretty(&new_manifest)?;
            
            use object_store::{PutMode, PutOptions};
            let opts = PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            };
            
            match self.store.put_opts(&path, bytes.into(), opts).await {
                Ok(_) => {
                    println!("Committed Manifest v{} (Schema Update)", new_ver);
                    let dir_key = format!("{}/{}", self.root_uri, self.manifest_dir);
                    crate::core::cache::LATEST_VERSION_CACHE.invalidate(&dir_key).await;
                    let file_key = format!("{}/{}", self.root_uri, path);
                    crate::core::cache::MANIFEST_CACHE.insert(file_key, Arc::new(new_manifest.clone())).await;
                    return Ok(new_manifest);
                }
                Err(e) if is_already_exists(&e) => {
                     // Conflict, retry
                     if attempt >= max_retries {
                         break;
                     }
                     attempt += 1;
                     let base_delay = 10 * (2u64.pow(attempt.min(5) as u32));
                     let jitter = rand::random::<u64>() % base_delay;
                     tokio::time::sleep(std::time::Duration::from_millis(base_delay + jitter)).await;
                     continue;
                }
                Err(e) => return Err(e.into())
            }
        }
        Err(anyhow::anyhow!("Failed to commit schema update after {} attempts", max_retries))
    }

    /// Atomically commit a full manifest (optimistic concurrency)
    pub async fn commit_manifest(&self, manifest: Manifest) -> Result<()> {
        let max_retries = 10;
        let mut attempt = 0;
        loop {
            let (_, current_ver) = self.load_latest().await?;
            if manifest.version != current_ver + 1 {
                return Err(anyhow::anyhow!("Manifest version mismatch: expected {}, got {}", current_ver + 1, manifest.version));
            }
            
            let filename = format!("v{}.json", manifest.version);
            let path = self.manifest_dir.child(filename);
            let bytes = serde_json::to_vec_pretty(&manifest)?;
            
            use object_store::{PutMode, PutOptions};
            let opts = PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            };
            
            match self.store.put_opts(&path, bytes.into(), opts).await {
                Ok(_) => {
                    tracing::info!("Committed Manifest v{}", manifest.version);
                    let dir_key = self.get_dir_cache_key();
                    crate::core::cache::LATEST_VERSION_CACHE.invalidate(&dir_key).await;
                    let file_key = self.get_cache_key(&path);
                    crate::core::cache::MANIFEST_CACHE.insert(file_key, Arc::new(manifest.clone())).await;
                    return Ok(());
                }
                Err(e) if is_already_exists(&e) => {
                     attempt += 1;
                     if attempt >= max_retries { break; }
                     tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                     continue;
                }
                Err(e) => return Err(e.into())
            }
        }
        Err(anyhow::anyhow!("Failed to commit manifest after {} attempts", max_retries))
    }

    /// Update partition specification
    pub async fn update_partition_spec(&self, new_spec: PartitionSpec) -> Result<Manifest> {
        let max_retries = 10;
        let mut attempt = 0;
        loop {
            let (current_manifest, current_ver) = self.load_latest().await?;
            let new_ver = current_ver + 1;
            
            let new_manifest = Manifest::new_with_spec(
                new_ver, 
                current_manifest.entries.clone(), 
                Some(current_ver),
                current_manifest.schemas.clone(),
                current_manifest.current_schema_id,
                new_spec.clone(),
            );
            
            let filename = format!("v{}.json", new_ver);
            let path = self.manifest_dir.child(filename);
            let bytes = serde_json::to_vec_pretty(&new_manifest)?;
            
            use object_store::{PutMode, PutOptions};
            let opts = PutOptions {
                mode: PutMode::Create,
                ..Default::default()
            };
            
            match self.store.put_opts(&path, bytes.into(), opts).await {
                Ok(_) => {
                    tracing::info!("Committed Manifest v{} (Partition Spec Update)", new_ver);
                    let dir_key = format!("{}/{}", self.root_uri, self.manifest_dir);
                    crate::core::cache::LATEST_VERSION_CACHE.invalidate(&dir_key).await;
                    let file_key = format!("{}/{}", self.root_uri, path);
                    crate::core::cache::MANIFEST_CACHE.insert(file_key, Arc::new(new_manifest.clone())).await;
                    return Ok(new_manifest);
                }
                Err(e) if is_already_exists(&e) => {
                     attempt += 1;
                     if attempt >= max_retries { break; }
                     tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                     continue;
                }
                Err(e) => return Err(e.into())
            }
        }
        Err(anyhow::anyhow!("Failed to commit partition spec update"))
    }

    /// Delete old manifest files and any data/index files NOT referenced by the latest N versions.
    /// Returns the number of files deleted.
    pub async fn vacuum(&self, retention_versions: usize) -> Result<usize> {
        if retention_versions == 0 {
            return Err(anyhow::anyhow!("Retention versions must be at least 1"));
        }

        let (_latest_m, latest_ver) = self.load_latest().await?;
        if latest_ver == 0 {
            return Ok(0);
        }

        // 1. Identify active files in the retention window
        let mut active_files = HashSet::new();
        let mut manifest_files_to_keep = HashSet::new();

        let start_ver = latest_ver.saturating_sub(retention_versions as u64 - 1).max(1);
        
        for v in start_ver..=latest_ver {
            let m = match self.load_version(v).await {
                Ok(m) => m,
                Err(_) => continue, // Skip missing versions in history gaps
            };
            
            // Collect all data and index files
            for entry in m.entries {
                active_files.insert(entry.file_path.clone());
                for index in entry.index_files {
                    active_files.insert(index.file_path.clone());
                }
                for del in entry.delete_files {
                    active_files.insert(del.file_path.clone());
                }
            }
            
            // Keep the manifest file itself
            let m_name = format!("v{}.json", v);
            let m_path = self.manifest_dir.child(m_name);
            manifest_files_to_keep.insert(m_path.to_string());
        }

        // 2. Discover all files in the storage
        // We list the root but skip the _manifest/ dir contents (except handled separately)
        let mut deleted_count = 0;
        let mut stream = self.store.list(None);
        
        while let Some(meta) = stream.next().await {
            let meta = meta?;
            let path_str = meta.location.to_string();
            
            // Skip the current manifest directory itself but check files inside
            if path_str.contains("_manifest/v") {
                // If it's a manifest file, check if we keep it
                if !manifest_files_to_keep.contains(&path_str) {
                    tracing::info!("Vacuum: Deleting old manifest {}", path_str);
                    self.store.delete(&meta.location).await?;
                    deleted_count += 1;
                }
                continue;
            }

            // Skip other files in _manifest/ (e.g. checkpoints if added later)
            if path_str.contains("_manifest/") {
                continue;
            }

            // check if it's a data file we should care about
            // segment files (seg_...), compacted files (compacted_...), or .tmp files
            let is_data_file = path_str.ends_with(".parquet") || 
                              path_str.ends_with(".hnsw") || 
                              path_str.ends_with(".idx") ||
                              path_str.ends_with(".tmp");

            if is_data_file {
                // If it's not in the active set, delete it
                if !active_files.contains(&path_str) {
                    // Small safety: don't delete very young .tmp files (leeway for active writers)
                    // If it's a .tmp file and less than 1 hour old, skip.
                    if path_str.ends_with(".tmp") {
                        let age = Utc::now() - chrono::DateTime::from_timestamp(meta.last_modified.timestamp(), 0).unwrap_or(Utc::now());
                        if age.num_minutes() < 60 {
                            continue;
                        }
                    }

                    tracing::info!("Vacuum: Deleting unreferenced file {}", path_str);
                    self.store.delete(&meta.location).await?;
                    deleted_count += 1;
                }
            }
        }

        Ok(deleted_count)
    }
}

fn is_already_exists(e: &object_store::Error) -> bool {
    match e {
        object_store::Error::AlreadyExists { .. } => true,
        _ => e.to_string().contains("already exists") // Fallback for some store implementations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;

    fn create_entry(id: &str) -> ManifestEntry {
        ManifestEntry {
            file_path: format!("{}.parquet", id),
            file_size_bytes: 100,
            record_count: 10,
            index_files: vec![],
            delete_files: vec![],
            column_stats: HashMap::new(),
            partition_values: HashMap::new(),
            clustering_strategy: None,
            clustering_columns: None,
            min_clustering_score: None,
            max_clustering_score: None,
            normalization_mins: None,
            normalization_maxs: None,
        }
    }

    #[tokio::test]
    async fn test_manifest_flow() -> Result<()> {
        // Use unique path to avoid cache conflicts
        let test_id = uuid::Uuid::new_v4();
        let test_uri = format!("memory://test_{}", test_id);
        let store = Arc::new(InMemory::new());
        let manager = ManifestManager::new(store, "", &test_uri);

        // 1. Initial State (Empty)
        let (m0, v0) = manager.load_latest().await?;
        assert_eq!(v0, 0);
        assert!(m0.entries.is_empty());

        // 2. Commit Add
        let entry_a = create_entry("seg_a");
        let m1 = manager.commit(std::slice::from_ref(&entry_a), &[], CommitMetadata::default()).await?;
        assert_eq!(m1.version, 1);
        
        // Load all entries (including those in manifest lists)
        let all_entries_1 = manager.load_all_entries(&m1).await?;
        assert_eq!(all_entries_1.len(), 1);
        assert_eq!(all_entries_1[0].file_path, "seg_a.parquet");

        // 3. Commit Add + Remove
        let entry_b = create_entry("seg_b");
        // Remove seg_a by path
        let m2 = manager.commit(std::slice::from_ref(&entry_b), &["seg_a.parquet".to_string()], CommitMetadata::default()).await?;
        assert_eq!(m2.version, 2);
        
        // Load all entries (including those in manifest lists)
        let all_entries_2 = manager.load_all_entries(&m2).await?;
        assert_eq!(all_entries_2.len(), 1);
        assert_eq!(all_entries_2[0].file_path, "seg_b.parquet");

        // 4. Reload
        let (latest, ver) = manager.load_latest().await?;
        assert_eq!(ver, 2);
        
        // Load all entries and compare
        let latest_entries = manager.load_all_entries(&latest).await?;
        let m2_entries = manager.load_all_entries(&m2).await?;
        assert_eq!(latest_entries.len(), m2_entries.len());
        assert_eq!(latest_entries[0].file_path, m2_entries[0].file_path);

        // Cleanup cache
        let cache_key = format!("{}/{}", test_uri, "");
        crate::core::cache::LATEST_VERSION_CACHE.invalidate(&cache_key).await;

        Ok(())
    }

    #[tokio::test]
    async fn test_verify_manifest_history() -> Result<()> {
        let store = Arc::new(InMemory::new());
        let root_uri = "memory://test";
        let manager = ManifestManager::new(store.clone(), "test_table", root_uri);

        // Commit 1
        let entry1 = create_entry("seg1");
        manager.commit(&[entry1], &[], CommitMetadata::default()).await?;

        // Commit 2
        let entry2 = create_entry("seg2");
        manager.commit(&[entry2], &[], CommitMetadata::default()).await?;

        // Load specific version (v2)
        // Load specific version (v2)
        let (_manifest, entries, version) = manager.load_latest_full().await?;
        assert_eq!(version, 2);
        assert_eq!(entries.len(), 2);

        // Walk history
        let history = manager.walk_history().await?;
        // walk_history returns [v2, v1] - it stops before v0 (genesis) since prev_version of v1 would be Some(0)
        // and the loop breaks when prev == 0
        assert_eq!(history.len(), 2); // v2, v1
        assert_eq!(history[0].version, 2);
        assert_eq!(history[1].version, 1);
        
        Ok(())
    }
}
