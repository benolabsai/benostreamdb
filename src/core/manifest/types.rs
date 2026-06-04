#![allow(unused)]
// Copyright (c) 2026 Richard Albright. All rights reserved.

use crate::core::cache::CacheExt;
use anyhow::Result;
use arrow::array::Array;
use arrow::record_batch::RecordBatch;
use chrono::Utc;
use futures::StreamExt;
use object_store::{path::Path, ObjectStore};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tracing;

pub type SegmentId = String;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
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
    Equality {
        equality_ids: Vec<i32>,
    },
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
    pub fn new_single(
        source_id: i32,
        field_id: Option<i32>,
        name: String,
        transform: String,
    ) -> Self {
        Self {
            source_ids: vec![source_id],
            source_id: Some(source_id),
            field_id,
            name,
            transform,
        }
    }

    /// Create a multi-column partition field (e.g., composite bucketing)
    pub fn new_multi(
        source_ids: Vec<i32>,
        field_id: Option<i32>,
        name: String,
        transform: String,
    ) -> Self {
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ManifestEntry {
    pub file_path: String,
    pub file_size_bytes: i64,
    pub record_count: i64,
    /// HyperStream Extension: File checksum (e.g. SHA256) for data integrity verification
    #[serde(default)]
    pub file_checksum: Option<String>,
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
            ManifestValue::Float32(f) => serde_json::Number::from_f64(*f as f64)
                .map(Value::Number)
                .unwrap_or(Value::Null),
            ManifestValue::Float64(f) => serde_json::Number::from_f64(*f)
                .map(Value::Number)
                .unwrap_or(Value::Null),
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

impl ManifestValue {
    /// Extract a ManifestValue from an Arrow array at a specific index
    pub fn from_array(array: &std::sync::Arc<dyn arrow::array::Array>, i: usize) -> Self {
        if array.is_null(i) {
            return ManifestValue::Null;
        }

        use arrow::datatypes::DataType;
        match array.data_type() {
            DataType::Utf8 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow::array::StringArray>()
                    .unwrap();
                ManifestValue::String(arr.value(i).to_string())
            }
            DataType::Int32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int32Array>()
                    .unwrap();
                ManifestValue::Int32(arr.value(i))
            }
            DataType::Int64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow::array::Int64Array>()
                    .unwrap();
                ManifestValue::Int64(arr.value(i))
            }
            DataType::Float32 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow::array::Float32Array>()
                    .unwrap();
                ManifestValue::Float32(arr.value(i))
            }
            DataType::Float64 => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .unwrap();
                ManifestValue::Float64(arr.value(i))
            }
            DataType::Boolean => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow::array::BooleanArray>()
                    .unwrap();
                ManifestValue::Boolean(arr.value(i))
            }
            _ => ManifestValue::Null, // Unsupported complex types for equality/stats
        }
    }
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
            }
            Value::Bool(b) => ManifestValue::Boolean(b),
            Value::Null => ManifestValue::Null,
            _ => ManifestValue::String(val.to_string()), // Fallback for arrays/objects
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ColumnStats {
    pub min: Option<ManifestValue>,
    pub max: Option<ManifestValue>,
    pub null_count: i64,
    /// Number of distinct values (NDV) - Iceberg V2 spec field
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distinct_count: Option<i64>,
    /// HyperStream Extension: Vector-specific statistics
    #[serde(default)]
    pub vector_stats: Option<VectorStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct VectorStats {
    pub min_norm: f32,
    pub max_norm: f32,
    pub mean_norm: f32,
    /// HyperStream Extension: Per-dimension ranges for advanced pruning
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dim_min: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dim_max: Option<Vec<f32>>,
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
#[serde(rename_all = "snake_case", tag = "type")]
pub enum IndexAlgorithm {
    Hnsw {
        #[serde(default = "default_metric")]
        metric: String,
        #[serde(default = "default_complexity", alias = "m")]
        complexity: usize,
        #[serde(default = "default_quality", alias = "ef_construction")]
        quality: usize,
        #[serde(default)]
        build_device: Option<String>,
        #[serde(default)]
        search_device: Option<String>,
    },
    HnswPq {
        #[serde(default = "default_metric")]
        metric: String,
        #[serde(default = "default_complexity", alias = "m")]
        complexity: usize,
        #[serde(default = "default_quality", alias = "ef_construction")]
        quality: usize,
        #[serde(default = "default_compression", alias = "subspaces")]
        compression: usize,
    },
    HnswTq4 {
        #[serde(default = "default_metric")]
        metric: String,
        #[serde(default = "default_complexity", alias = "m")]
        complexity: usize,
        #[serde(default = "default_quality", alias = "ef_construction")]
        quality: usize,
    },
    HnswTq8 {
        #[serde(default = "default_metric")]
        metric: String,
        #[serde(default = "default_complexity", alias = "m")]
        complexity: usize,
        #[serde(default = "default_quality", alias = "ef_construction")]
        quality: usize,
    },
    Bm25 {
        #[serde(default)]
        k1: f32,
        #[serde(default)]
        b: f32,
        #[serde(default)]
        tokenizer: String,
    },
    Bloom {
        #[serde(default)]
        fpr: f32,
    },
    Bitmap,
}

impl std::fmt::Display for IndexAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexAlgorithm::Hnsw { .. } => write!(f, "hnsw"),
            IndexAlgorithm::HnswPq { .. } => write!(f, "hnsw_pq"),
            IndexAlgorithm::HnswTq4 { .. } => write!(f, "hnsw_tq4"),
            IndexAlgorithm::HnswTq8 { .. } => write!(f, "hnsw_tq8"),
            IndexAlgorithm::Bm25 { .. } => write!(f, "bm25"),
            IndexAlgorithm::Bloom { .. } => write!(f, "bloom"),
            IndexAlgorithm::Bitmap => write!(f, "bitmap"),
        }
    }
}

impl Default for IndexAlgorithm {
    fn default() -> Self {
        Self::hnsw_tq8()
    }
}

fn default_metric() -> String {
    "l2".to_string()
}
fn default_complexity() -> usize {
    16
}
fn default_quality() -> usize {
    200
}
fn default_compression() -> usize {
    8
}

impl IndexAlgorithm {
    /// Create a standard HNSW index configuration.
    pub fn hnsw() -> Self {
        Self::Hnsw {
            metric: default_metric(),
            complexity: default_complexity(),
            quality: default_quality(),
            build_device: None,
            search_device: None,
        }
    }

    /// Create an 8-bit TurboQuant optimized index (4x compression).
    pub fn hnsw_tq8() -> Self {
        Self::HnswTq8 {
            metric: default_metric(),
            complexity: default_complexity(),
            quality: default_quality(),
        }
    }

    /// Create a 4-bit TurboQuant optimized index (8x compression).
    pub fn hnsw_tq4() -> Self {
        Self::HnswTq4 {
            metric: default_metric(),
            complexity: default_complexity(),
            quality: default_quality(),
        }
    }

    /// Create a Product Quantization index with a specific compression ratio.
    pub fn hnsw_pq() -> Self {
        Self::HnswPq {
            metric: default_metric(),
            complexity: default_complexity(),
            quality: default_quality(),
            compression: default_compression(),
        }
    }

    pub fn with_metric(mut self, metric: impl Into<String>) -> Self {
        let m = metric.into();
        match &mut self {
            Self::Hnsw { metric, .. }
            | Self::HnswPq { metric, .. }
            | Self::HnswTq4 { metric, .. }
            | Self::HnswTq8 { metric, .. } => *metric = m,
            _ => {}
        }
        self
    }

    pub fn with_complexity(mut self, complexity: usize) -> Self {
        match &mut self {
            Self::Hnsw { complexity: c, .. }
            | Self::HnswPq { complexity: c, .. }
            | Self::HnswTq4 { complexity: c, .. }
            | Self::HnswTq8 { complexity: c, .. } => *c = complexity,
            _ => {}
        }
        self
    }

    pub fn with_quality(mut self, quality: usize) -> Self {
        match &mut self {
            Self::Hnsw { quality: q, .. }
            | Self::HnswPq { quality: q, .. }
            | Self::HnswTq4 { quality: q, .. }
            | Self::HnswTq8 { quality: q, .. } => *q = quality,
            _ => {}
        }
        self
    }

    pub fn with_compression(mut self, compression: usize) -> Self {
        if let Self::HnswPq { compression: c, .. } = &mut self {
            *c = compression;
        }
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
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
    /// HyperStream Extension: Multiple indexing algorithms for this column
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub indexes: Vec<IndexAlgorithm>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct Schema {
    #[serde(alias = "schema-id")]
    pub schema_id: i32,
    pub fields: Vec<SchemaField>,
    #[serde(
        rename = "identifier-field-ids",
        default,
        skip_serializing_if = "Vec::is_empty"
    )]
    pub identifier_field_ids: Vec<i32>,
}

impl Schema {
    pub fn new(id: i32, fields: Vec<SchemaField>, identifier_field_ids: Vec<i32>) -> Self {
        Self {
            schema_id: id,
            fields,
            identifier_field_ids,
        }
    }

    pub fn to_arrow(&self) -> arrow::datatypes::Schema {
        let fields: Vec<arrow::datatypes::Field> =
            self.fields.iter().map(|f| f.to_arrow()).collect();
        arrow::datatypes::Schema::new(fields)
    }

    pub fn from_arrow(schema: &arrow::datatypes::Schema, id: i32) -> Self {
        let fields = schema
            .fields()
            .iter()
            .enumerate()
            .map(|(i, f)| SchemaField::from_arrow_field(f, i as i32 + 1))
            .collect();
        Self {
            schema_id: id,
            fields,
            identifier_field_ids: Vec::new(),
        }
    }
}

impl SchemaField {
    pub fn from_arrow_field(f: &arrow::datatypes::Field, id: i32) -> Self {
        use arrow::datatypes::DataType;

        let mut nested_fields = Vec::new();
        let type_str = match f.data_type() {
            DataType::Struct(fields) => {
                nested_fields = fields
                    .iter()
                    .enumerate()
                    .map(|(i, sf)| {
                        SchemaField::from_arrow_field(sf, id * 100 + i as i32 + 1)
                        // Simple nested ID logic
                    })
                    .collect();
                "struct".to_string()
            }
            DataType::List(field) => {
                nested_fields.push(SchemaField::from_arrow_field(field, id * 100 + 1));
                "list".to_string()
            }
            DataType::Map(field, _) => {
                // Arrow Map has a Struct field "entries" with "key" and "value"
                if let DataType::Struct(fields) = field.data_type() {
                    nested_fields = fields
                        .iter()
                        .enumerate()
                        .map(|(i, sf)| SchemaField::from_arrow_field(sf, id * 100 + i as i32 + 1))
                        .collect();
                }
                "map".to_string()
            }
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
            }
            DataType::Time32(unit) => {
                let unit_str = match unit {
                    arrow::datatypes::TimeUnit::Second => "second",
                    arrow::datatypes::TimeUnit::Millisecond => "millisecond",
                    _ => "millisecond",
                };
                format!("time32({})", unit_str)
            }
            DataType::Time64(unit) => {
                let unit_str = match unit {
                    arrow::datatypes::TimeUnit::Microsecond => "microsecond",
                    arrow::datatypes::TimeUnit::Nanosecond => "nanosecond",
                    _ => "microsecond",
                };
                format!("time64({})", unit_str)
            }
            dt => dt.to_string().to_lowercase(),
        };

        let field_id = f
            .metadata()
            .get("iceberg.id")
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
            indexes: vec![],
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
            "timestamp(microsecond, none)" => {
                arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, None)
            }
            "timestamp(nanosecond, none)" => {
                arrow::datatypes::DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None)
            }
            // Handle UTC timezone specifically if requested
            "timestamp(microsecond, utc)" => arrow::datatypes::DataType::Timestamp(
                arrow::datatypes::TimeUnit::Microsecond,
                Some("UTC".into()),
            ),
            "date" | "date32" => arrow::datatypes::DataType::Date32,
            "date64" => arrow::datatypes::DataType::Date64,
            "binary" => arrow::datatypes::DataType::Binary,
            "largebinary" => arrow::datatypes::DataType::LargeBinary,
            "largeutf8" => arrow::datatypes::DataType::LargeUtf8,
            // Handle all timestamp variants: "timestamp", "timestamptz", "timestamp(unit, tz)"
            s if s == "timestamp" || s == "timestamptz" || s.starts_with("timestamp(") => {
                if s == "timestamptz" {
                    arrow::datatypes::DataType::Timestamp(
                        arrow::datatypes::TimeUnit::Microsecond,
                        Some("UTC".into()),
                    )
                } else if s == "timestamp" {
                    arrow::datatypes::DataType::Timestamp(
                        arrow::datatypes::TimeUnit::Microsecond,
                        None,
                    )
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
                        if *t == "none" {
                            None
                        } else {
                            Some(t.to_string().into())
                        }
                    });
                    arrow::datatypes::DataType::Timestamp(unit, tz)
                }
            }
            s if s.contains("time32") => {
                if s.contains("millisecond") {
                    arrow::datatypes::DataType::Time32(arrow::datatypes::TimeUnit::Millisecond)
                } else {
                    arrow::datatypes::DataType::Time32(arrow::datatypes::TimeUnit::Second)
                }
            }
            s if s.contains("time64") => {
                if s.contains("nanosecond") {
                    arrow::datatypes::DataType::Time64(arrow::datatypes::TimeUnit::Nanosecond)
                } else {
                    arrow::datatypes::DataType::Time64(arrow::datatypes::TimeUnit::Microsecond)
                }
            }
            s if s.contains("fixedsizelist") || s.contains("fixed_list") => {
                let dim = s
                    .split(|c: char| !c.is_numeric())
                    .filter_map(|p| p.parse::<i32>().ok())
                    .next()
                    .unwrap_or(0);
                arrow::datatypes::DataType::FixedSizeList(
                    Arc::new(arrow::datatypes::Field::new(
                        "item",
                        arrow::datatypes::DataType::Float32,
                        true,
                    )),
                    dim,
                )
            }
            s if s.starts_with("fixed[") => {
                let len = s
                    .trim_start_matches("fixed[")
                    .trim_end_matches(']')
                    .parse::<i32>()
                    .unwrap_or(0);
                arrow::datatypes::DataType::FixedSizeBinary(len)
            }
            "struct" => {
                let arrow_fields = self.fields.iter().map(|f| f.to_arrow()).collect();
                arrow::datatypes::DataType::Struct(arrow_fields)
            }
            "list" => {
                let item_field = self.fields.first().map(|f| f.to_arrow()).unwrap_or(
                    arrow::datatypes::Field::new("item", arrow::datatypes::DataType::Utf8, true),
                );
                arrow::datatypes::DataType::List(Arc::new(item_field))
            }
            "map" => {
                let key_field = self.fields.first().map(|f| f.to_arrow()).unwrap_or(
                    arrow::datatypes::Field::new("key", arrow::datatypes::DataType::Utf8, false),
                );
                let value_field = self.fields.get(1).map(|f| f.to_arrow()).unwrap_or(
                    arrow::datatypes::Field::new("value", arrow::datatypes::DataType::Utf8, true),
                );

                arrow::datatypes::DataType::Map(
                    Arc::new(arrow::datatypes::Field::new(
                        "entries",
                        arrow::datatypes::DataType::Struct(vec![key_field, value_field].into()),
                        false,
                    )),
                    false,
                )
            }
            s if s == "decimal" || s.starts_with("decimal(") || s.starts_with("decimal128(") => {
                let parts: Vec<&str> = if s.starts_with("decimal(") {
                    s.trim_start_matches("decimal(").trim_end_matches(')')
                } else {
                    s.trim_start_matches("decimal128(").trim_end_matches(')')
                }
                .split(',')
                .map(|p| p.trim())
                .collect();
                let precision = parts
                    .first()
                    .and_then(|p| p.parse::<u8>().ok())
                    .unwrap_or(38);
                let scale = parts
                    .get(1)
                    .and_then(|p| p.parse::<i8>().ok())
                    .unwrap_or(10);
                arrow::datatypes::DataType::Decimal128(precision, scale)
            }
            _ => arrow::datatypes::DataType::Utf8,
        };
        let mut f = arrow::datatypes::Field::new(&self.name, dt, !self.required);
        f.set_metadata(std::collections::HashMap::from([(
            "iceberg.id".to_string(),
            self.id.to_string(),
        )]));
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

    pub fn new_with_schema(
        version: u64,
        entries: Vec<ManifestEntry>,
        prev_version: Option<u64>,
        schemas: Vec<Schema>,
        current_schema_id: i32,
    ) -> Self {
        let last_id = schemas
            .iter()
            .flat_map(|s| s.fields.iter().map(|f| f.id))
            .max()
            .unwrap_or(0);
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
        partition_spec: PartitionSpec,
    ) -> Self {
        let last_id = schema_list
            .iter()
            .flat_map(|s| s.fields.iter().map(|f| f.id))
            .max()
            .unwrap_or(0);
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
// Iceberg Target Manifest Size = 8MB
pub(crate) const MANIFEST_TARGET_SIZE_BYTES: usize = 8 * 1024 * 1024;

lazy_static::lazy_static! {
    /// Global registry of commit locks to serialize manifest updates per directory.
    pub(crate) static ref COMMIT_LOCKS: dashmap::DashMap<String, Arc<tokio::sync::Mutex<()>>> = dashmap::DashMap::new();
}

impl PartitionSpec {
    /// Convert partition values to a Hive-style path string
    pub fn partition_to_path(&self, values: &HashMap<String, serde_json::Value>) -> String {
        let mut parts = Vec::new();
        for field in &self.fields {
            if let Some(val) = values.get(&field.name) {
                let val_str = match val {
                    serde_json::Value::String(s) => s.clone(),
                    _ => val.to_string(),
                };

                // Percent-encode special characters to be safe for filenames and match object_store Paths
                let mut encoded = String::new();
                for b in val_str.bytes() {
                    match b {
                        b'a'..=b'z'
                        | b'A'..=b'Z'
                        | b'0'..=b'9'
                        | b'-'
                        | b'_'
                        | b'.'
                        | b'!'
                        | b'~'
                        | b'*'
                        | b'\''
                        | b'('
                        | b')' => {
                            encoded.push(b as char);
                        }
                        _ => {
                            encoded.push_str(&format!("%{:02X}", b));
                        }
                    }
                }
                parts.push(format!("{}={}", field.name, encoded));
            }
        }
        parts.join("/")
    }
}

impl ManifestValue {
    /// Convert to a serde_json::Value
    pub fn to_json_value(&self) -> serde_json::Value {
        match self {
            ManifestValue::String(s) => serde_json::json!(s),
            ManifestValue::Int32(i) => serde_json::json!(i),
            ManifestValue::Int64(i) => serde_json::json!(i),
            ManifestValue::Float32(f) => serde_json::json!(f),
            ManifestValue::Float64(f) => serde_json::json!(f),
            ManifestValue::Boolean(b) => serde_json::json!(b),
            ManifestValue::Null => serde_json::Value::Null,
        }
    }
}
