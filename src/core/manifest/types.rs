// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use arrow::array::Array;
use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

pub type SegmentId = String;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct IndexFile {
    /// Format version of the serialized index structure (e.g., 1 for legacy, 2 for tightened/Puffin)
    #[serde(default = "default_index_format_version")]
    pub format_version: u32,

    /// Unique identifier for this index instance
    pub index_id: String,

    /// High-level index category: "vector", "lexical", "scalar", "composite", "bloom"
    pub index_category: String,

    /// Concrete algorithm implementation: "hnsw_turboquant8", "hnsw_f32", "bm25_v1", "roaring_bitmap_v1"
    pub algorithm: String,

    /// Target column name within the Parquet schema
    pub column_name: String,

    // -----------------------------------------------------------------------
    // Strict Lineage Bindings (Data Integrity Guards)
    // -----------------------------------------------------------------------
    /// Exact Iceberg Snapshot ID active when this index was constructed
    pub source_snapshot_id: i64,

    /// Cryptographic SHA-256 hash of the target Parquet data file
    pub source_data_checksum: String,

    /// Exact record count of the target Parquet data file
    /// Enforces a strict 1-to-1 bijection between index point IDs and Parquet row offsets
    pub source_record_count: i64,

    /// Cryptographic SHA-256 hash of the index artifact itself (detects storage corruption / bitrot)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index_checksum: Option<String>,

    // -----------------------------------------------------------------------
    // Physical Storage Location & Puffin Bundle Offsets
    // -----------------------------------------------------------------------
    /// Physical URI or relative storage path of the index artifact (or Puffin bundle)
    pub file_path: String,

    /// Puffin blob type if packed into an Iceberg Puffin container
    /// e.g., "org.apache.iceberg.vector.hnsw-tq8", "org.apache.iceberg.lexical.bm25"
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob_type: Option<String>,

    /// Byte offset within the container file where the index payload begins
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob_offset: Option<i64>,

    /// Length in bytes of the index payload
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blob_length: Option<i64>,

    /// Compression algorithm applied to the index blob ("zstd", "lz4", "none")
    #[serde(default = "default_index_compression")]
    pub compression: String,
}

fn default_index_format_version() -> u32 {
    2
}
fn default_index_compression() -> String {
    "zstd".to_string()
}

impl IndexFile {
    /// Whether this index belongs to the lexical (full-text) family.
    ///
    /// The canonical category is `"lexical"`. Legacy manifests written before
    /// the rename used `"inverted"`, and some code
    /// paths historically used `"bm25"`. All three denote the same family
    /// (inverted postings / BM25) and must be treated equivalently so that
    /// legacy tables keep working and new tables are recognized by every
    /// consumer.
    pub fn is_lexical(&self) -> bool {
        matches!(
            self.index_category.as_str(),
            "lexical" | "inverted" | "bm25"
        )
    }
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
    /// BenoStream Extension: File checksum (e.g. SHA256) for data integrity verification
    #[serde(default)]
    pub file_checksum: Option<String>,
    /// BenoStream Extension: Sidecar Index Files
    pub index_files: Vec<IndexFile>,
    /// BenoStream Extension: Merge-on-Read Delete Files (Iceberg v2 compliant)
    #[serde(default)]
    pub delete_files: Vec<DeleteFile>,
    /// Column Statistics for Pruning (Min/Max/Nulls)
    #[serde(default)]
    pub column_stats: HashMap<String, ColumnStats>,
    /// Partition values for this file
    #[serde(default)]
    pub partition_values: HashMap<String, Value>,
    /// BenoStream Extension: Clustering metadata for advanced pruning
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
    /// Iceberg V3 Row Lineage: the first `_row_id` assigned to the first row in
    /// this data file. A row's `_row_id` is `first_row_id + row_position`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_row_id: Option<i64>,
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
            // Arrow guarantees the array variant matches `data_type()`, so the
            // downcasts below cannot fail in practice; `map(..).unwrap_or(Null)`
            // keeps this decode path total (a violated invariant degrades to
            // Null instead of panicking a commit/read).
            DataType::Utf8 => array
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .map(|arr| ManifestValue::String(arr.value(i).to_string()))
                .unwrap_or(ManifestValue::Null),
            // `large_string` is the default for PyArrow string columns; without
            // this arm partition values silently became Null.
            DataType::LargeUtf8 => array
                .as_any()
                .downcast_ref::<arrow::array::LargeStringArray>()
                .map(|arr| ManifestValue::String(arr.value(i).to_string()))
                .unwrap_or(ManifestValue::Null),
            DataType::Utf8View => array
                .as_any()
                .downcast_ref::<arrow::array::StringViewArray>()
                .map(|arr| ManifestValue::String(arr.value(i).to_string()))
                .unwrap_or(ManifestValue::Null),
            // Date/time types feed the year/month/day/hour transforms.
            DataType::Date32 => array
                .as_any()
                .downcast_ref::<arrow::array::Date32Array>()
                .map(|arr| ManifestValue::Int32(arr.value(i)))
                .unwrap_or(ManifestValue::Null),
            DataType::Date64 => array
                .as_any()
                .downcast_ref::<arrow::array::Date64Array>()
                .map(|arr| ManifestValue::Int64(arr.value(i)))
                .unwrap_or(ManifestValue::Null),
            DataType::Timestamp(_, _) => {
                let arr = array
                    .as_any()
                    .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
                    .map(|a| a.value(i))
                    .or_else(|| {
                        array
                            .as_any()
                            .downcast_ref::<arrow::array::TimestampMillisecondArray>()
                            .map(|a| a.value(i) * 1_000)
                    })
                    .or_else(|| {
                        array
                            .as_any()
                            .downcast_ref::<arrow::array::TimestampSecondArray>()
                            .map(|a| a.value(i) * 1_000_000)
                    })
                    .or_else(|| {
                        array
                            .as_any()
                            .downcast_ref::<arrow::array::TimestampNanosecondArray>()
                            .map(|a| a.value(i) / 1_000)
                    });
                match arr {
                    Some(v) => ManifestValue::Int64(v),
                    None => ManifestValue::Null,
                }
            }
            DataType::Int32 => array
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .map(|arr| ManifestValue::Int32(arr.value(i)))
                .unwrap_or(ManifestValue::Null),
            DataType::Int64 => array
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .map(|arr| ManifestValue::Int64(arr.value(i)))
                .unwrap_or(ManifestValue::Null),
            DataType::Float32 => array
                .as_any()
                .downcast_ref::<arrow::array::Float32Array>()
                .map(|arr| ManifestValue::Float32(arr.value(i)))
                .unwrap_or(ManifestValue::Null),
            DataType::Float64 => array
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
                .map(|arr| ManifestValue::Float64(arr.value(i)))
                .unwrap_or(ManifestValue::Null),
            DataType::Boolean => array
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
                .map(|arr| ManifestValue::Boolean(arr.value(i)))
                .unwrap_or(ManifestValue::Null),
            _ => ManifestValue::Null, // Unsupported complex types for equality/stats
        }
    }
}

impl From<Value> for ManifestValue {
    fn from(val: Value) -> Self {
        match val {
            Value::String(s) => ManifestValue::String(s),
            Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    ManifestValue::Int64(i)
                } else if let Some(f) = n.as_f64() {
                    ManifestValue::Float64(f)
                } else {
                    // Fallback, treated as f64 or 0 if nan/inf (json doesn't have nan)
                    ManifestValue::Float64(0.0)
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
    /// BenoStream Extension: Vector-specific statistics
    #[serde(default)]
    pub vector_stats: Option<VectorStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct VectorStats {
    pub min_norm: f32,
    pub max_norm: f32,
    pub mean_norm: f32,
    /// BenoStream Extension: Per-dimension ranges for advanced pruning
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
    CompositeBitmap {
        columns: Vec<String>,
    },
    CsrGraph {
        /// Source column (e.g. "src")
        #[serde(default = "default_src_column")]
        src_column: String,
        /// Destination column (e.g. "dst")
        #[serde(default = "default_dst_column")]
        dst_column: String,
    },
    /// A rebuildable inverted index over selected JSON paths in a `Utf8`
    /// column. Each configured path (e.g. `$.user.id`) is extracted per row and
    /// indexed as `(path, value) -> row_ids`, so equality/containment filters
    /// on those paths avoid a full scan.
    JsonPath {
        /// JSON paths to index, e.g. `["$.user.id", "$.level"]`.
        #[serde(default)]
        paths: Vec<String>,
    },
}

impl IndexAlgorithm {
    /// The canonical name of this algorithm (matches `Display`).
    pub fn name(&self) -> &'static str {
        match self {
            IndexAlgorithm::Hnsw { .. } => "hnsw",
            IndexAlgorithm::HnswPq { .. } => "hnsw_pq",
            IndexAlgorithm::HnswTq4 { .. } => "hnsw_tq4",
            IndexAlgorithm::HnswTq8 { .. } => "hnsw_tq8",
            IndexAlgorithm::Bm25 { .. } => "bm25",
            IndexAlgorithm::Bloom { .. } => "bloom",
            IndexAlgorithm::Bitmap => "bitmap",
            IndexAlgorithm::CompositeBitmap { .. } => "composite_bitmap",
            IndexAlgorithm::CsrGraph { .. } => "csr_graph",
            IndexAlgorithm::JsonPath { .. } => "json_path",
        }
    }

    /// Parse a user-supplied index name or physical category into an
    /// [`IndexAlgorithm`].
    ///
    /// Accepts both the canonical names from [`Self::all_names`] and the
    /// physical categories the manifest uses (`"vector"`, `"lexical"`,
    /// `"scalar"`, `"bloom"`, `"graph_v2"`), so the SQL/connector surfaces can
    /// pass whichever spelling their users know. Returns `None` for anything
    /// unrecognized, letting each caller fall back to its own default (the
    /// connectors use [`Self::default`] = `hnsw`, full precision).
    pub fn from_name(name: &str) -> Option<IndexAlgorithm> {
        match name.trim().to_ascii_lowercase().as_str() {
            "hnsw" | "hnsw_f32" => Some(IndexAlgorithm::hnsw()),
            // "vector" is the physical category for the default vector index,
            // which is now full-precision HNSW.
            "vector" => Some(IndexAlgorithm::hnsw()),
            "hnsw_pq" | "pq" => Some(IndexAlgorithm::hnsw_pq()),
            "hnsw_tq4" | "tq4" => Some(IndexAlgorithm::hnsw_tq4()),
            "hnsw_tq8" | "tq8" => Some(IndexAlgorithm::hnsw_tq8()),
            "bm25" | "lexical" | "inverted" => Some(IndexAlgorithm::Bm25 {
                k1: 1.2,
                b: 0.75,
                tokenizer: "default".to_string(),
            }),
            "bloom" => Some(IndexAlgorithm::Bloom { fpr: 0.05 }),
            "bitmap" | "scalar" => Some(IndexAlgorithm::Bitmap),
            "composite_bitmap" | "composite" => Some(IndexAlgorithm::CompositeBitmap {
                columns: Vec::new(),
            }),
            "csr_graph" | "graph" | "graph_v2" => Some(IndexAlgorithm::CsrGraph {
                src_column: "source".to_string(),
                dst_column: "target".to_string(),
            }),
            "json_path" => Some(IndexAlgorithm::JsonPath { paths: Vec::new() }),
            _ => None,
        }
    }

    /// Every index algorithm name the engine understands. Single source of
    /// truth for the connector surface-parity test — adding a variant here
    /// without exposing it in the Python/dbt/connector surfaces fails the test.
    pub fn all_names() -> Vec<String> {
        vec![
            "hnsw",
            "hnsw_pq",
            "hnsw_tq4",
            "hnsw_tq8",
            "bm25",
            "bloom",
            "bitmap",
            "composite_bitmap",
            "csr_graph",
            "json_path",
        ]
        .into_iter()
        .map(String::from)
        .collect()
    }
}

impl std::fmt::Display for IndexAlgorithm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl Default for IndexAlgorithm {
    /// Full-precision HNSW is the default. Quantized variants (TurboQuant
    /// `hnsw_tq4`/`hnsw_tq8`, `hnsw_pq`) are opt-in: they trade recall for a
    /// smaller index and cheaper distance computation. Holding multiple
    /// precisions on one column and selecting one at query time is on the
    /// roadmap.
    fn default() -> Self {
        Self::hnsw()
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
fn default_src_column() -> String {
    "src".to_string()
}
fn default_dst_column() -> String {
    "dst".to_string()
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

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SchemaField {
    pub id: i32,
    pub name: String,
    /// Canonical BenoStreamDB/Arrow spelling of the column type (`int`, `long`,
    /// `double`, `list`, `map`, `struct`, ...). Serialized to Iceberg's type
    /// grammar by [`schema_field_to_json`].
    pub type_str: String,
    pub required: bool,
    /// For nested types: struct members, the list element, or the map
    /// key/value pair, in that order.
    pub fields: Vec<SchemaField>,
    /// Iceberg V3: Default value for rows written before this column was added
    pub initial_default: Option<Value>,
    /// Iceberg V3: Default value for new rows when this column is null
    pub write_default: Option<Value>,
    /// BenoStream Extension: Multiple indexing algorithms for this column
    pub indexes: Vec<IndexAlgorithm>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct Schema {
    pub schema_id: i32,
    pub fields: Vec<SchemaField>,
    pub identifier_field_ids: Vec<i32>,
}

/// Map BenoStreamDB/Arrow type spellings onto Iceberg's primitive type grammar.
///
/// Iceberg's `SchemaParser` only accepts its own names (`int`, `long`, `double`,
/// `timestamp`, `timestamptz`, `fixed[N]`, ...). Anything else is rejected, so
/// Arrow-specific spellings have to be normalised at the metadata boundary.
fn iceberg_primitive_name(raw: &str) -> String {
    let s = raw.trim().to_ascii_lowercase();
    match s.as_str() {
        "int8" | "int16" | "int32" | "int" | "uint8" | "uint16" => "int".to_string(),
        "int64" | "long" | "uint32" | "uint64" => "long".to_string(),
        "float32" | "float" => "float".to_string(),
        "float64" | "double" => "double".to_string(),
        "bool" | "boolean" => "boolean".to_string(),
        "utf8" | "string" | "largeutf8" => "string".to_string(),
        "binary" | "largebinary" => "binary".to_string(),
        "date" | "date32" | "date64" => "date".to_string(),
        "timestamptz" => "timestamptz".to_string(),
        "timestamp" => "timestamp".to_string(),
        "time" | "time32" | "time64" => "time".to_string(),
        "null" => "unknown".to_string(),
        other => {
            if let Some(inner) = other
                .strip_prefix("timestamp(")
                .and_then(|v| v.strip_suffix(')'))
            {
                let mut parts = inner.splitn(2, ',').map(str::trim);
                let unit = parts.next().unwrap_or("microsecond");
                let tz = parts.next().unwrap_or("none");
                let has_tz = !(tz.is_empty() || tz == "none");
                return match (unit, has_tz) {
                    ("nanosecond", true) => "timestamptz_ns".to_string(),
                    ("nanosecond", false) => "timestamp_ns".to_string(),
                    (_, true) => "timestamptz".to_string(),
                    (_, false) => "timestamp".to_string(),
                };
            }
            if other.starts_with("time32(") || other.starts_with("time64(") {
                return "time".to_string();
            }
            other.to_string()
        }
    }
}

/// Serialize one schema field using Iceberg's nested-type encoding.
fn schema_field_to_json(field: &SchemaField) -> Result<Value, String> {
    let mut obj = serde_json::Map::new();
    obj.insert("id".to_string(), Value::from(field.id));
    obj.insert("name".to_string(), Value::from(field.name.clone()));
    obj.insert("required".to_string(), Value::from(field.required));
    obj.insert("type".to_string(), schema_field_type_to_json(field)?);
    if let Some(value) = &field.initial_default {
        obj.insert("initial-default".to_string(), value.clone());
    }
    if let Some(value) = &field.write_default {
        obj.insert("write-default".to_string(), value.clone());
    }
    if !field.indexes.is_empty() {
        let indexes = serde_json::to_value(&field.indexes).map_err(|e| e.to_string())?;
        obj.insert("indexes".to_string(), indexes);
    }
    Ok(Value::Object(obj))
}

fn schema_field_type_to_json(field: &SchemaField) -> Result<Value, String> {
    match field.type_str.trim().to_ascii_lowercase().as_str() {
        "struct" => {
            let fields = field
                .fields
                .iter()
                .map(schema_field_to_json)
                .collect::<Result<Vec<_>, _>>()?;
            let mut obj = serde_json::Map::new();
            obj.insert("type".to_string(), Value::from("struct"));
            obj.insert("fields".to_string(), Value::Array(fields));
            Ok(Value::Object(obj))
        }
        "list" => {
            let element = field.fields.first();
            let mut obj = serde_json::Map::new();
            obj.insert("type".to_string(), Value::from("list"));
            obj.insert(
                "element-id".to_string(),
                Value::from(element.map(|e| e.id).unwrap_or(field.id + 1)),
            );
            obj.insert(
                "element".to_string(),
                match element {
                    Some(e) => schema_field_type_to_json(e)?,
                    None => Value::from("string"),
                },
            );
            obj.insert(
                "element-required".to_string(),
                Value::from(element.map(|e| e.required).unwrap_or(false)),
            );
            Ok(Value::Object(obj))
        }
        "map" => {
            let key = field.fields.first();
            let value = field.fields.get(1);
            let mut obj = serde_json::Map::new();
            obj.insert("type".to_string(), Value::from("map"));
            obj.insert(
                "key-id".to_string(),
                Value::from(key.map(|e| e.id).unwrap_or(field.id + 1)),
            );
            obj.insert(
                "key".to_string(),
                match key {
                    Some(e) => schema_field_type_to_json(e)?,
                    None => Value::from("string"),
                },
            );
            obj.insert(
                "key-required".to_string(),
                Value::from(key.map(|e| e.required).unwrap_or(false)),
            );
            obj.insert(
                "value-id".to_string(),
                Value::from(value.map(|e| e.id).unwrap_or(field.id + 2)),
            );
            obj.insert(
                "value".to_string(),
                match value {
                    Some(e) => schema_field_type_to_json(e)?,
                    None => Value::from("string"),
                },
            );
            obj.insert(
                "value-required".to_string(),
                Value::from(value.map(|e| e.required).unwrap_or(false)),
            );
            Ok(Value::Object(obj))
        }
        other => Ok(Value::from(iceberg_primitive_name(other))),
    }
}

/// Build a nested field from an Iceberg `element`/`key`/`value` node.
fn nested_schema_field(
    id: i32,
    name: &str,
    type_node: Option<&Value>,
    required: bool,
) -> Result<SchemaField, String> {
    match type_node {
        Some(Value::String(primitive)) => Ok(SchemaField {
            id,
            name: name.to_string(),
            type_str: iceberg_primitive_name(primitive),
            required,
            fields: Vec::new(),
            initial_default: None,
            write_default: None,
            indexes: Vec::new(),
        }),
        Some(Value::Object(nested)) => {
            let mut obj = nested.clone();
            obj.insert("id".to_string(), Value::from(id));
            obj.insert("name".to_string(), Value::from(name));
            obj.insert("required".to_string(), Value::from(required));
            schema_field_from_json(&Value::Object(obj))
        }
        _ => Err(format!("nested schema type for `{name}` is not an object")),
    }
}

fn schema_field_from_json(value: &Value) -> Result<SchemaField, String> {
    let obj = value
        .as_object()
        .ok_or_else(|| "schema field must be a JSON object".to_string())?;
    let id = obj
        .get("id")
        .and_then(Value::as_i64)
        .ok_or_else(|| "schema field is missing `id`".to_string())? as i32;
    let name = obj
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| "schema field is missing `name`".to_string())?
        .to_string();
    let required = obj
        .get("required")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let type_node = obj
        .get("type")
        .ok_or_else(|| "schema field is missing `type`".to_string())?;

    let (type_str, fields) = match type_node {
        Value::String(kind) => {
            let kind = kind.to_ascii_lowercase();
            // Legacy BenoStreamDB encoding kept nested members in `fields`
            // alongside a plain `list`/`map`/`struct` type string.
            let nested = if matches!(kind.as_str(), "struct" | "list" | "map") {
                obj.get("fields")
                    .and_then(Value::as_array)
                    .map(|items| {
                        items
                            .iter()
                            .map(schema_field_from_json)
                            .collect::<Result<Vec<_>, _>>()
                    })
                    .transpose()?
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            (kind, nested)
        }
        Value::Object(nested) => {
            let kind = nested
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_ascii_lowercase();
            match kind.as_str() {
                "struct" => {
                    let fields = nested
                        .get("fields")
                        .and_then(Value::as_array)
                        .map(|items| {
                            items
                                .iter()
                                .map(schema_field_from_json)
                                .collect::<Result<Vec<_>, _>>()
                        })
                        .transpose()?
                        .unwrap_or_default();
                    ("struct".to_string(), fields)
                }
                "list" => {
                    let element_id = nested
                        .get("element-id")
                        .and_then(Value::as_i64)
                        .map(|v| v as i32)
                        .unwrap_or_else(|| id + 1);
                    let element_required = nested
                        .get("element-required")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    let element = nested_schema_field(
                        element_id,
                        "element",
                        nested.get("element"),
                        element_required,
                    )?;
                    ("list".to_string(), vec![element])
                }
                "map" => {
                    let key_id = nested
                        .get("key-id")
                        .and_then(Value::as_i64)
                        .map(|v| v as i32)
                        .unwrap_or_else(|| id + 1);
                    let value_id = nested
                        .get("value-id")
                        .and_then(Value::as_i64)
                        .map(|v| v as i32)
                        .unwrap_or_else(|| id + 2);
                    let key_required = nested
                        .get("key-required")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    let value_required = nested
                        .get("value-required")
                        .and_then(Value::as_bool)
                        .unwrap_or(false);
                    let key = nested_schema_field(key_id, "key", nested.get("key"), key_required)?;
                    let value = nested_schema_field(
                        value_id,
                        "value",
                        nested.get("value"),
                        value_required,
                    )?;
                    ("map".to_string(), vec![key, value])
                }
                other => (other.to_string(), Vec::new()),
            }
        }
        _ => return Err("schema field `type` must be a string or an object".to_string()),
    };

    Ok(SchemaField {
        id,
        name,
        type_str,
        required,
        fields,
        initial_default: obj.get("initial-default").cloned(),
        write_default: obj.get("write-default").cloned(),
        indexes: obj
            .get("indexes")
            .cloned()
            .map(serde_json::from_value)
            .transpose()
            .map_err(|e| e.to_string())?
            .unwrap_or_default(),
    })
}

impl Serialize for SchemaField {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let value = schema_field_to_json(self).map_err(serde::ser::Error::custom)?;
        value.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SchemaField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        schema_field_from_json(&value).map_err(serde::de::Error::custom)
    }
}

impl Serialize for Schema {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        // Iceberg requires `type: "struct"` and kebab-case `schema-id`.
        let mut state = serializer.serialize_struct("schema", 4)?;
        state.serialize_field("type", "struct")?;
        state.serialize_field("schema-id", &self.schema_id)?;
        state.serialize_field("fields", &self.fields)?;
        if !self.identifier_field_ids.is_empty() {
            state.serialize_field("identifier-field-ids", &self.identifier_field_ids)?;
        }
        state.end()
    }
}

impl<'de> Deserialize<'de> for Schema {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct SchemaRepr {
            #[serde(rename = "schema-id", alias = "schema_id", default)]
            schema_id: i32,
            #[serde(default)]
            fields: Vec<SchemaField>,
            #[serde(
                rename = "identifier-field-ids",
                alias = "identifier_field_ids",
                default
            )]
            identifier_field_ids: Vec<i32>,
        }

        let repr = SchemaRepr::deserialize(deserializer)?;
        Ok(Schema {
            schema_id: repr.schema_id,
            fields: repr.fields,
            identifier_field_ids: repr.identifier_field_ids,
        })
    }
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
            .or_else(|| f.metadata().get("PARQUET:field_id"))
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
            "uint64" | "ulong" => arrow::datatypes::DataType::UInt64,
            "uint32" | "uint" => arrow::datatypes::DataType::UInt32,
            "uint16" => arrow::datatypes::DataType::UInt16,
            "uint8" => arrow::datatypes::DataType::UInt8,
            "int16" | "short" => arrow::datatypes::DataType::Int16,
            "int8" | "byte" => arrow::datatypes::DataType::Int8,
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
                    let unit = match parts.first().copied() {
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
    /// Authoritative, partition-scoped list of merge-on-read delete files.
    ///
    /// Delete files are stored **once** here (not attached to every data
    /// entry). `ManifestEntry::delete_files` is a read-time *derived view*:
    /// the reader resolves it by matching `partition_values`, and the commit
    /// never persists it. This avoids the read-attach -> commit-persist cycle
    /// that previously made the per-entry list grow without bound.
    #[serde(default)]
    pub delete_files: Vec<DeleteFile>,
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
            delete_files: Vec::new(),
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
            delete_files: Vec::new(),
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
            delete_files: Vec::new(),
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

#[cfg(test)]
mod index_algorithm_name_tests {
    use super::IndexAlgorithm;

    #[test]
    fn from_name_accepts_canonical_names_and_categories() {
        // Canonical names round-trip.
        for name in IndexAlgorithm::all_names() {
            let alg = IndexAlgorithm::from_name(&name)
                .unwrap_or_else(|| panic!("canonical name '{name}' did not parse"));
            assert_eq!(alg.name(), name, "name '{name}' did not round-trip");
        }
        // Physical categories the manifest uses map to the right family.
        assert_eq!(
            IndexAlgorithm::from_name("vector").map(|a| a.name()),
            Some("hnsw")
        );
        assert_eq!(
            IndexAlgorithm::from_name("lexical").map(|a| a.name()),
            Some("bm25")
        );
        assert_eq!(
            IndexAlgorithm::from_name("scalar").map(|a| a.name()),
            Some("bitmap")
        );
        assert_eq!(
            IndexAlgorithm::from_name("graph_v2").map(|a| a.name()),
            Some("csr_graph")
        );
    }

    #[test]
    fn from_name_is_case_insensitive_and_trims() {
        assert_eq!(
            IndexAlgorithm::from_name("  HNSW_PQ ").map(|a| a.name()),
            Some("hnsw_pq")
        );
    }

    #[test]
    fn from_name_returns_none_for_unknown() {
        // Unknown input must be `None` so callers apply their own default
        // rather than silently mis-indexing.
        assert!(IndexAlgorithm::from_name("not_an_index").is_none());
        assert!(IndexAlgorithm::from_name("").is_none());
    }
}
