// Copyright (c) 2026 Richard Albright. All rights reserved.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use regex::Regex;
use once_cell::sync::Lazy;
use crate::core::table::{Table, VectorSearchParams};
use crate::core::compaction::CompactionOptions;
use crate::core::index::VectorMetric;
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use arrow::array::RecordBatchIterator;
use arrow::ffi_stream::{FFI_ArrowArrayStream, ArrowArrayStreamReader};
use pyo3::ffi::Py_uintptr_t;
use crate::core::catalog::{Catalog, nessie::NessieClient, rest::RestCatalogClient, glue::GlueCatalogClient, hive::HiveMetastoreClient, unity::UnityCatalogClient, jdbc::JdbcCatalogClient, CatalogConfig, CatalogType};
use tokio::runtime::Runtime;
use std::sync::Arc;
use std::collections::HashMap;
use futures::StreamExt;

static SQL_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)dist_l2\(([^,]+),\s*\[([^\]]+)\]\)").unwrap());

/// Module-level global Tokio runtime for all Python-bound operations.
/// Sharing a single runtime prevents 'Cannot drop a runtime in a context where blocking is not allowed' panics.
static TOKIO_RUNTIME: Lazy<Arc<Runtime>> = Lazy::new(|| {
    Arc::new(Runtime::new().expect("Failed to create unified Tokio runtime for HyperStreamDB"))
});

fn sanitize_sql(query: &str) -> String {
    SQL_REGEX.replace_all(query, "l2_distance($1, $2)").to_string()
}
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use crate::python_gpu_context::PyDevice;
use crate::core::manifest::IndexAlgorithm;

// Helper function to parse metric string to VectorMetric enum
// Uses native Rust names: L2, Cosine, InnerProduct, L1, Hamming, Jaccard
// Also accepts lowercase aliases for backward compatibility

fn parse_metric(metric_str: &str) -> PyResult<VectorMetric> {
    match metric_str {
        "l2" | "L2" => Ok(VectorMetric::L2),
        "cosine" | "Cosine" => Ok(VectorMetric::Cosine),
        "innerproduct" | "inner_product" | "InnerProduct" => Ok(VectorMetric::InnerProduct),
        "l1" | "L1" => Ok(VectorMetric::L1),
        "hamming" | "Hamming" => Ok(VectorMetric::Hamming),
        "jaccard" | "Jaccard" => Ok(VectorMetric::Jaccard),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!("Invalid metric '{}'", metric_str))),
    }
}

fn parse_index_algorithm(val: Bound<'_, PyAny>) -> PyResult<IndexAlgorithm> {
    if let Ok(s) = val.extract::<String>() {
        match s.to_lowercase().as_str() {
            "hnsw" => Ok(IndexAlgorithm::Hnsw { 
                metric: "l2".to_string(), 
                complexity: 16, 
                quality: 200, 
                build_device: None, 
                search_device: None 
            }),
            "hnsw_pq" | "pq" => Ok(IndexAlgorithm::HnswPq { 
                metric: "l2".to_string(), 
                complexity: 16, 
                quality: 200, 
                compression: 8 
            }),
            "hnsw_tq4" | "tq4" => Ok(IndexAlgorithm::HnswTq4 { 
                metric: "l2".to_string(), 
                complexity: 16, 
                quality: 200 
            }),
            "hnsw_tq8" | "tq8" => Ok(IndexAlgorithm::HnswTq8 { 
                metric: "l2".to_string(), 
                complexity: 16, 
                quality: 200 
            }),
            "bm25" => Ok(IndexAlgorithm::Bm25 {
                k1: 1.2,
                b: 0.75,
                tokenizer: "default".to_string(),
            }),
            "bloom" => Ok(IndexAlgorithm::Bloom { fpr: 0.05 }),
            "bitmap" | "inverted" => Ok(IndexAlgorithm::Bitmap),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!("Unknown index type: {}", s))),
        }
    } else if let Ok(dict) = val.downcast::<PyDict>() {
        let type_str: String = dict.get_item("type")?.ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("Missing 'type' key in index config"))?.extract()?;
        match type_str.to_lowercase().as_str() {
            "hnsw" => {
                let metric = dict.get_item("metric")?.and_then(|v| v.extract::<String>().ok()).unwrap_or_else(|| "l2".to_string());
                let complexity = dict.get_item("complexity")?.and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| dict.get_item("m").ok().flatten().and_then(|v| v.extract::<usize>().ok()))
                    .unwrap_or(16);
                let quality = dict.get_item("quality")?.and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| dict.get_item("ef_construction").ok().flatten().and_then(|v| v.extract::<usize>().ok()))
                    .unwrap_or(200);
                let build_device = dict.get_item("build_device")?.and_then(|v| v.extract::<String>().ok());
                let search_device = dict.get_item("search_device")?.and_then(|v| v.extract::<String>().ok());
                Ok(IndexAlgorithm::Hnsw { metric, complexity, quality, build_device, search_device })
            },
            "hnsw_pq" | "pq" => {
                let metric = dict.get_item("metric")?.and_then(|v| v.extract::<String>().ok()).unwrap_or_else(|| "l2".to_string());
                let compression = dict.get_item("compression")?.and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| dict.get_item("subspaces").ok().flatten().and_then(|v| v.extract::<usize>().ok()))
                    .unwrap_or(8);
                let complexity = dict.get_item("complexity")?.and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| dict.get_item("m").ok().flatten().and_then(|v| v.extract::<usize>().ok()))
                    .unwrap_or(16);
                let quality = dict.get_item("quality")?.and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| dict.get_item("ef_construction").ok().flatten().and_then(|v| v.extract::<usize>().ok()))
                    .unwrap_or(200);
                Ok(IndexAlgorithm::HnswPq { metric, complexity, quality, compression })
            },
            "hnsw_tq4" | "tq4" => {
                let metric = dict.get_item("metric")?.and_then(|v| v.extract::<String>().ok()).unwrap_or_else(|| "l2".to_string());
                let complexity = dict.get_item("complexity")?.and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| dict.get_item("m").ok().flatten().and_then(|v| v.extract::<usize>().ok()))
                    .unwrap_or(16);
                let quality = dict.get_item("quality")?.and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| dict.get_item("ef_construction").ok().flatten().and_then(|v| v.extract::<usize>().ok()))
                    .unwrap_or(200);
                Ok(IndexAlgorithm::HnswTq4 { metric, complexity, quality })
            },
            "hnsw_tq8" | "tq8" => {
                let metric = dict.get_item("metric")?.and_then(|v| v.extract::<String>().ok()).unwrap_or_else(|| "l2".to_string());
                let complexity = dict.get_item("complexity")?.and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| dict.get_item("m").ok().flatten().and_then(|v| v.extract::<usize>().ok()))
                    .unwrap_or(16);
                let quality = dict.get_item("quality")?.and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| dict.get_item("ef_construction").ok().flatten().and_then(|v| v.extract::<usize>().ok()))
                    .unwrap_or(200);
                Ok(IndexAlgorithm::HnswTq8 { metric, complexity, quality })
            },
            "bm25" => {
                let k1 = dict.get_item("k1")?.and_then(|v| v.extract().ok()).unwrap_or(1.2);
                let b = dict.get_item("b")?.and_then(|v| v.extract().ok()).unwrap_or(0.75);
                let tokenizer = dict.get_item("tokenizer")?.and_then(|v| v.extract().ok()).unwrap_or_else(|| "default".to_string());
                Ok(IndexAlgorithm::Bm25 { k1, b, tokenizer })
            },
            "bloom" => {
                let fpr = dict.get_item("fpr")?.and_then(|v| v.extract().ok()).unwrap_or(0.05);
                Ok(IndexAlgorithm::Bloom { fpr })
            },
            "bitmap" | "inverted" => Ok(IndexAlgorithm::Bitmap),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!("Unknown index type: {}", type_str))),
        }
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err("Index algorithm must be a string or a dict"))
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyMergeMode {
    MergeOnRead,
    MergeOnWrite,
}

impl From<PyMergeMode> for crate::core::table::MergeMode {
    fn from(mode: PyMergeMode) -> Self {
        match mode {
            PyMergeMode::MergeOnRead => crate::core::table::MergeMode::MergeOnRead,
            PyMergeMode::MergeOnWrite => crate::core::table::MergeMode::MergeOnWrite,
        }
    }
}

// Connector API Python Bindings

#[pyclass]
pub struct PyDataFileInfo {
    #[pyo3(get)]
    pub file_path: String,
    #[pyo3(get)]
    pub row_count: u64,
    #[pyo3(get)]
    pub file_size_bytes: u64,
    #[pyo3(get)]
    pub min_values: std::collections::HashMap<String, String>,
    #[pyo3(get)]
    pub max_values: std::collections::HashMap<String, String>,
    #[pyo3(get)]
    pub has_scalar_indexes: bool,
    #[pyo3(get)]
    pub has_vector_indexes: bool,
    #[pyo3(get)]
    pub indexed_columns: Vec<String>,
}

#[pyclass]
pub struct PySplit {
    #[pyo3(get)]
    pub file_path: String,
    #[pyo3(get)]
    pub start_offset: u64,
    #[pyo3(get)]
    pub length: u64,
    #[pyo3(get)]
    pub row_group_ids: Vec<usize>,
    #[pyo3(get)]
    pub index_file_path: Option<String>,
    #[pyo3(get)]
    pub can_use_indexes: bool,
}

#[pymethods]
impl PySplit {
    #[new]
    #[pyo3(signature = (file_path, start_offset, length, row_group_ids, index_file_path=None, can_use_indexes=false))]
    fn new(
        file_path: String, 
        start_offset: u64, 
        length: u64, 
        row_group_ids: Vec<usize>, 
        index_file_path: Option<String>, 
        can_use_indexes: bool
    ) -> Self {
        Self {
            file_path,
            start_offset,
            length,
            row_group_ids,
            index_file_path,
            can_use_indexes,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyIndexCoverage {
    #[pyo3(get)]
    pub scalar_indexed_columns: Vec<String>,
    #[pyo3(get)]
    pub vector_indexed_columns: Vec<String>,
    #[pyo3(get)]
    pub inverted_indexed_columns: Vec<String>,
    #[pyo3(get)]
    pub total_index_size_bytes: u64,
}
#[pyclass]
pub struct PyTableStatistics {
    #[pyo3(get)]
    pub row_count: u64,
    #[pyo3(get)]
    pub file_count: usize,
    #[pyo3(get)]
    pub total_size_bytes: u64,
    #[pyo3(get)]
    pub index_coverage: PyIndexCoverage,
}

#[pyclass(name = "DataType")]
#[derive(Clone, Debug)]
pub struct PyDataType {
    pub(crate) dt: DataType,
}

#[pymethods]
impl PyDataType {
    #[staticmethod]
    fn int8() -> Self { Self { dt: DataType::Int8 } }
    #[staticmethod]
    fn int16() -> Self { Self { dt: DataType::Int16 } }
    #[staticmethod]
    fn int32() -> Self { Self { dt: DataType::Int32 } }
    #[staticmethod]
    fn int64() -> Self { Self { dt: DataType::Int64 } }
    #[staticmethod]
    fn uint8() -> Self { Self { dt: DataType::UInt8 } }
    #[staticmethod]
    fn uint16() -> Self { Self { dt: DataType::UInt16 } }
    #[staticmethod]
    fn uint32() -> Self { Self { dt: DataType::UInt32 } }
    #[staticmethod]
    fn uint64() -> Self { Self { dt: DataType::UInt64 } }
    #[staticmethod]
    fn float16() -> Self { Self { dt: DataType::Float16 } }
    #[staticmethod]
    fn float32() -> Self { Self { dt: DataType::Float32 } }
    #[staticmethod]
    fn float64() -> Self { Self { dt: DataType::Float64 } }
    #[staticmethod]
    fn string() -> Self { Self { dt: DataType::Utf8 } }
    #[staticmethod]
    fn binary() -> Self { Self { dt: DataType::Binary } }
    #[staticmethod]
    fn boolean() -> Self { Self { dt: DataType::Boolean } }
    #[staticmethod]
    fn date32() -> Self { Self { dt: DataType::Date32 } }
    #[staticmethod]
    fn date64() -> Self { Self { dt: DataType::Date64 } }
    #[staticmethod]
    fn timestamp_ms() -> Self { Self { dt: DataType::Timestamp(TimeUnit::Millisecond, None) } }
    #[staticmethod]
    fn timestamp_us() -> Self { Self { dt: DataType::Timestamp(TimeUnit::Microsecond, None) } }
    #[staticmethod]
    #[pyo3(signature = (dim, nullable=true))]
    fn vector(dim: usize, nullable: bool) -> Self {
        Self {
            dt: DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, nullable)),
                dim as i32,
            )
        }
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.dt)
    }
}

#[pyclass(name = "Field")]
#[derive(Clone, Debug)]
pub struct PyField {
    pub(crate) inner: Field,
}

#[pymethods]
impl PyField {
    #[new]
    #[pyo3(signature = (name, data_type, nullable=true, metadata=None))]
    fn new(name: String, data_type: PyDataType, nullable: bool, metadata: Option<HashMap<String, String>>) -> Self {
        let mut field = Field::new(name, data_type.dt, nullable);
        if let Some(m) = metadata {
            field = field.with_metadata(m);
        }
        Self { inner: field }
    }

    fn __repr__(&self) -> String {
        format!("Field(name={}, type={:?}, nullable={})", self.inner.name(), self.inner.data_type(), self.inner.is_nullable())
    }
}

#[pyclass(name = "PartitionField")]
#[derive(Clone, Debug)]
pub struct PyPartitionField {
    #[pyo3(get, set)]
    pub source_ids: Vec<i32>,
    #[pyo3(get, set)]
    pub field_id: Option<i32>,
    #[pyo3(get, set)]
    pub name: String,
    #[pyo3(get, set)]
    pub transform: String,
}

#[pymethods]
impl PyPartitionField {
    #[new]
    #[pyo3(signature = (source_ids, name, transform, field_id=None))]
    fn new(source_ids: Vec<i32>, name: String, transform: String, field_id: Option<i32>) -> Self {
        Self { source_ids, field_id, name, transform }
    }
}

#[pyclass(name = "Schema")]
#[derive(Clone, Debug)]
pub struct PySchema {
    pub(crate) inner: arrow::datatypes::SchemaRef,
}

#[pymethods]
impl PySchema {
    #[new]
    #[pyo3(signature = (fields, metadata=None))]
    fn new(fields: Vec<PyField>, metadata: Option<HashMap<String, String>>) -> Self {
        let arrow_fields: Vec<Field> = fields.into_iter().map(|f| f.inner).collect();
        let mut schema = Schema::new(arrow_fields);
        if let Some(m) = metadata {
            schema = schema.with_metadata(m);
        }
        Self {
            inner: Arc::new(schema),
        }
    }

    fn __repr__(&self) -> String {
        format!("Schema(fields={:?})", self.inner.fields())
    }
}


#[pyclass(name = "ManifestEntry")]
#[derive(Clone, Debug)]
pub struct PyManifestEntry {
    #[pyo3(get)]
    pub file_path: String,
    #[pyo3(get)]
    pub file_size_bytes: i64,
    #[pyo3(get)]
    pub record_count: i64,
    #[pyo3(get)]
    pub index_files_count: usize,
    #[pyo3(get)]
    pub delete_files_count: usize,
}

#[pymethods]
impl PyManifestEntry {
    fn __repr__(&self) -> String {
        format!("ManifestEntry(path={}, rows={})", self.file_path, self.record_count)
    }
}

#[pyclass(name = "Manifest")]
#[derive(Clone, Debug)]
pub struct PyManifest {
    #[pyo3(get)]
    pub version: u64,
    #[pyo3(get)]
    pub timestamp_ms: i64,
    #[pyo3(get)]
    pub current_schema_id: i32,
    #[pyo3(get)]
    pub partition_spec_id: i32,
    #[pyo3(get)]
    pub entries: Vec<PyManifestEntry>,
    #[pyo3(get)]
    pub properties: std::collections::HashMap<String, String>,
}

#[pymethods]
impl PyManifest {
    fn __repr__(&self) -> String {
        format!("Manifest(version={}, entries={})", self.version, self.entries.len())
    }
    
    /// Compatibility alias for 'entries'
    #[getter]
    fn files(&self) -> Vec<PyManifestEntry> {
        self.entries.clone()
    }
}


/// High-level Table API - Pandas-compatible interface
/// This is a thin Python wrapper around the core Rust Table struct
#[pyclass(name = "Table")]
pub struct PyTable {
    table: Table,
    device: Option<Py<PyDevice>>,
}

impl PyTable {
    pub fn new_internal(uri: &str, device: Option<Py<PyDevice>>) -> Result<Self, anyhow::Error> {
        let mut table = TOKIO_RUNTIME.block_on(Table::builder(uri.to_string())
            .with_runtime(TOKIO_RUNTIME.clone())
            .build_async())?;
        table.rt = Some(TOKIO_RUNTIME.clone());
        Ok(PyTable { table, device })
    }

    pub fn create_internal(uri: &str, schema: arrow::datatypes::SchemaRef, device: Option<Py<PyDevice>>) -> Result<Self, anyhow::Error> {
        let mut table = TOKIO_RUNTIME.block_on(Table::create_async(uri.to_string(), schema))?;
        table.rt = Some(TOKIO_RUNTIME.clone());
        Ok(PyTable { table, device })
    }
}

#[pymethods]
#[allow(deprecated)]
impl PyTable {
    #[new]
    #[pyo3(signature = (uri, device=None))]
    fn new(uri: &str, device: Option<Py<PyDevice>>) -> PyResult<Self> {
        Self::new_internal(uri, device)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Create a new table with an explicit schema
    #[staticmethod]
    #[pyo3(signature = (uri, schema, device=None))]
    fn create(uri: &str, schema: Bound<'_, PyAny>, device: Option<Py<PyDevice>>) -> PyResult<Self> {
        let rust_schema = extract_schema(schema)?;
        Self::create_internal(uri, rust_schema, device)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    #[staticmethod]
    #[pyo3(signature = (uri, schema, partition_spec, device=None))]
    fn create_partitioned(uri: &str, schema: Bound<'_, PyAny>, partition_spec: Bound<'_, PyAny>, device: Option<Py<PyDevice>>) -> PyResult<Self> {
        let rust_schema = extract_schema(schema)?;
        let rust_spec = extract_partition_spec(partition_spec)?;
        
        let mut table = TOKIO_RUNTIME.block_on(Table::create_partitioned_async(uri.to_string(), rust_schema, rust_spec))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            
        // CRITICAL: Attach the runtime to the table so sync methods don't panic
        table.rt = Some(TOKIO_RUNTIME.clone());
        tracing::debug!("Rust Table created with global runtime");
        
        Ok(PyTable { table, device })
    }

    /// Register an existing Iceberg table
    #[staticmethod]
    #[pyo3(signature = (uri, iceberg_metadata_uri))]
    fn register_external(uri: &str, iceberg_metadata_uri: &str) -> PyResult<Self> {
        let mut table = TOKIO_RUNTIME.block_on(Table::register_external(uri.to_string(), iceberg_metadata_uri))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            
        // CRITICAL: Attach the runtime to the table
        table.rt = Some(TOKIO_RUNTIME.clone());
            
        Ok(PyTable { table, device: None })
    }
    
    /// Override parallel readers for vector search (disables auto-detection)
    /// 
    /// By default, parallelism is AUTO-DETECTED based on:
    /// - Available system memory  
    /// - Segment size (num_vectors × embedding_dim × 4 bytes)
    /// 
    /// Only call this if auto-detection doesn't work for your use case.
    /// 
    /// Example:
    ///     table.set_max_parallel_readers(4)  # Force 4 parallel readers
    fn set_max_parallel_readers(&mut self, max_readers: usize) {
        self.table.set_max_parallel_readers(max_readers);
    }
    
    /// Reset to auto-detect parallel readers based on system memory
    /// 
    /// Example:
    ///     table.auto_detect_parallel_readers()  # Let the system decide
    fn auto_detect_parallel_readers(&mut self) {
        self.table.auto_detect_parallel_readers();
    }

    /// Start a background observer to watch an external Iceberg table for changes
    fn spawn_iceberg_observer(&self, py: Python<'_>, iceberg_metadata_uri: &str, interval_seconds: u64) -> PyResult<()> {
        py.allow_threads(|| {
            TOKIO_RUNTIME.block_on(self.table.spawn_iceberg_observer(
                iceberg_metadata_uri.to_string(),
                std::time::Duration::from_secs(interval_seconds)
            ))
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
    
    /// Get current max parallel readers setting
    /// Returns None if auto-detecting, or the manual override value
    fn get_max_parallel_readers(&self) -> Option<usize> {
        self.table.get_max_parallel_readers()
    }

    fn set_index_all(&mut self, enabled: bool) {
        self.table.set_index_all(enabled);
    }

    fn get_index_all(&self) -> bool {
        self.table.get_index_all()
    }

    fn set_primary_key(&mut self, columns: Vec<String>) {
        self.table.set_primary_key(columns);
    }

    fn get_primary_key(&self) -> Vec<String> {
        self.table.get_primary_key()
    }

    /// Add a column to the primary key.
    fn add_primary_key(&mut self, py: Python<'_>, column: String) -> PyResult<()> {
        py.allow_threads(|| {
            TOKIO_RUNTIME.block_on(self.table.add_primary_key(column))
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Remove a column from the primary key.
    fn drop_primary_key(&mut self, py: Python<'_>, column: String) -> PyResult<()> {
        py.allow_threads(|| {
            TOKIO_RUNTIME.block_on(self.table.drop_primary_key(column))
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Update indexing specifications for multiple columns at once.
    fn set_index_columns(&mut self, py: Python<'_>, config: Bound<'_, PyDict>) -> PyResult<()> {
        let mut rust_config = HashMap::new();
        for (col, val) in config.into_iter() {
            let col_name: String = col.extract()?;
            let mut algs = Vec::new();
            
            if let Ok(list) = val.downcast::<pyo3::types::PyList>() {
                for item in list {
                    algs.push(parse_index_algorithm(item)?);
                }
            } else {
                algs.push(parse_index_algorithm(val)?);
            }
            rust_config.insert(col_name, algs);
        }

        py.allow_threads(|| {
            TOKIO_RUNTIME.block_on(self.table.set_index_columns(rust_config))
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Add an indexing strategy to a column.
    #[pyo3(signature = (column, algorithm = None))]
    fn add_index(&mut self, py: Python<'_>, column: String, algorithm: Option<Bound<'_, PyAny>>) -> PyResult<()> {
        let rust_alg = if let Some(algo_obj) = algorithm {
            parse_index_algorithm(algo_obj)?
        } else {
            // Smart Default based on Column Type
            let schema = self.table.arrow_schema();
            let field = schema.field_with_name(&column)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            
            match field.data_type() {
                arrow::datatypes::DataType::List(_) | arrow::datatypes::DataType::FixedSizeList(_, _) => {
                    IndexAlgorithm::Hnsw {
                        metric: "l2".into(),
                        complexity: 16,
                        quality: 200,
                        build_device: None,
                        search_device: None
                    }
                },
                _ => IndexAlgorithm::Bitmap,
            }
        };

        py.allow_threads(|| {
            TOKIO_RUNTIME.block_on(self.table.add_index(column, rust_alg))
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Explicitly quantize a column using TurboQuant or PQ.
    /// This is an enterprise-grade compression feature.
    #[pyo3(signature = (column, type_ = "TQ8", metric = "l2", complexity = 16, quality = 200))]
    fn quantize(&mut self, py: Python<'_>, column: String, type_: &str, metric: &str, complexity: usize, quality: usize) -> PyResult<()> {
        let algo = match type_.to_lowercase().as_str() {
            "tq8" | "hnsw_tq8" => IndexAlgorithm::HnswTq8 { metric: metric.to_string(), complexity, quality },
            "tq4" | "hnsw_tq4" => IndexAlgorithm::HnswTq4 { metric: metric.to_string(), complexity, quality },
            "pq" | "hnsw_pq" => IndexAlgorithm::HnswPq { metric: metric.to_string(), complexity, quality, compression: 8 },
            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("Unsupported quantization type: {}. Use TQ8, TQ4, or PQ.", type_))),
        };

        py.allow_threads(|| {
            TOKIO_RUNTIME.block_on(self.table.add_index(column, algo))
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Remove all indexing strategies from a column.
    fn drop_index(&mut self, py: Python<'_>, column: String) -> PyResult<()> {
        py.allow_threads(|| {
            TOKIO_RUNTIME.block_on(self.table.drop_index(column))
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Set default device for all future indexes in this table
    #[pyo3(signature = (device=None))]
    fn set_default_device(&mut self, device: Option<String>) {
        self.table.set_default_device(device);
    }

    /// Get current default device
    fn get_default_device(&self) -> Option<String> {
        self.table.get_default_device()
    }

    /// Register a Python-based embedding function into the Rust core.
    /// This allows the Rust core to trigger vectorization even without Python GIL
    /// by re-acquiring it only during the callback.
    fn register_python_embedding(&self, py: Python<'_>, name: String, dim: usize, callback: Py<PyAny>) -> PyResult<()> {
        let callback_clone = callback.clone_ref(py);
        let wrapper = move |texts: Vec<String>| -> anyhow::Result<Vec<Vec<f32>>> {
            Python::with_gil(|py| {
                let args = (texts,);
                let res = callback_clone.call1(py, args)
                    .map_err(|e| anyhow::anyhow!("Python callback error: {}", e))?;
                
                // Convert back to Vec<Vec<f32>>
                let list: Vec<Vec<f32>> = res.extract(py)
                    .map_err(|e| anyhow::anyhow!("Failed to extract embeddings from Python: {}", e))?;
                Ok(list)
            })
        };
        
        crate::core::embeddings::register_embedded_func(
            name.clone(), 
            Arc::new(crate::core::embeddings::PythonCallbackFunction::new(name, dim, wrapper))
        );
        Ok(())
    }

    /// Add columns to indexing configuration
    #[pyo3(signature = (columns, tokenizer=None))]
    fn add_index_columns(&mut self, columns: Vec<String>, tokenizer: Option<String>) -> PyResult<()> {
        self.table.add_index_columns(columns, tokenizer)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Remove columns from indexing configuration
    fn remove_index_columns(&mut self, columns: Vec<String>) {
        self.table.remove_index_columns(columns);
    }

    /// Remove all columns from indexing configuration
    fn remove_all_index_columns(&mut self) {
        self.table.remove_all_index_columns();
    }

    /// Index all columns (triggers backfill)
    fn index_all_columns(&mut self) -> PyResult<()> {
        self.table.index_all_columns()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }



    /// Read table to PyArrow Table with optional filtering and column projection
    /// 
    /// Args:
    ///     filter: Optional SQL-like filter string (e.g., "age > 25 AND city = 'NYC'")
    ///     vector_filter: Optional dict for vector search:
    ///         - column: str (required) - vector column name
    ///         - query: list (required) - query vector
    ///         - k: int (required) - number of results
    ///         - metric: str (optional) - 'l2'|'cosine'|'innerproduct'|'l1'|'hamming'|'jaccard' (default: l2)
    ///         - ef_search: int (optional) - HNSW ef parameter for search quality tuning
    ///         - probes: int (optional) - IVF probes parameter for search speed tuning
    ///     columns: Optional list of column names to read (skips others like embeddings)
    /// 
    /// Returns:
    ///     PyArrow Table (via Arrow C Data Interface)
    /// 
    /// Example:
    ///     # Vector search with cosine metric
    ///     df = table.to_pandas(vector_filter={"column": "embedding", "query": [1.0, 2.0], "k": 3, "metric": "cosine"})

    #[pyo3(signature = (filter=None, vector_filter=None, columns=None, device=None, **kwargs))]
    fn to_pandas(
        &self,
        py: Python<'_>,
        filter: Option<String>,
        vector_filter: Option<Bound<'_, PyDict>>,
        columns: Option<Vec<String>>,
        device: Option<Py<PyDevice>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        // Use streaming API under the hood to stabilize memory usage
        let reader = self.to_arrow_stream(py, filter, vector_filter, columns, device)?;
        
        // Convert to Table first, then to Pandas
        let arrow_table = reader.call_method0(py, "read_all")?;
        
        // Convert to Pandas via PyArrow, passing through kwargs
        arrow_table.call_method(py, "to_pandas", (), kwargs.as_ref())
    }

    /// Read table to PyArrow Table
    #[pyo3(signature = (filter=None, vector_filter=None, columns=None, device=None))]
    fn to_arrow(
        &self,
        py: Python<'_>,
        filter: Option<String>,
        vector_filter: Option<Bound<'_, PyDict>>,
        columns: Option<Vec<String>>,
        device: Option<Py<PyDevice>>,
    ) -> PyResult<Py<PyAny>> {
        // Use streaming API under the hood to stabilize memory usage
        let reader = self.to_arrow_stream(py, filter, vector_filter, columns, device)?;
        reader.call_method0(py, "read_all")
    }

    /// Read table to PyArrow RecordBatchReader (Streaming)
    /// 
    /// This is the recommended way to read large datasets that don't fit in memory.
    /// 
    /// Args:
    ///     filter: Optional SQL-like filter string
    ///     vector_filter: Optional dict for vector search
    ///     columns: List of columns to read
    ///     device: Optional ComputeContext for GPU acceleration
    /// 
    /// Returns:
    ///     pyarrow.RecordBatchReader
    #[pyo3(signature = (filter=None, vector_filter=None, columns=None, device=None))]
    fn to_arrow_stream(
        &self,
        py: Python<'_>,
        filter: Option<String>,
        vector_filter: Option<Bound<'_, PyDict>>,
        columns: Option<Vec<String>>,
        device: Option<Py<PyDevice>>,
    ) -> PyResult<Py<PyAny>> {
        let columns_clone = columns.clone();
        
        let vs_params_combined = if let Some(ref vf) = vector_filter {
            let column: String = vf.get_item("column")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("vector_filter requires 'column' key"))?
                .extract()?;
            let k: usize = vf.get_item("k")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("vector_filter requires 'k' key"))?
                .extract()?;
            let query_obj = vf.get_item("query")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("vector_filter requires 'query' key"))?;
            let query: Vec<f32> = query_obj.extract()?;
            let mut params = VectorSearchParams::new(&column, crate::core::index::VectorValue::Float32(query), k);
            
            if let Ok(Some(metric_obj)) = vf.get_item("metric") {
                if let Ok(metric_str) = metric_obj.extract::<String>() {
                    params = params.with_metric(parse_metric(&metric_str)?);
                }
            }
            
            let rrf_k = if let Ok(Some(rrf_obj)) = vf.get_item("rrf_k") {
                rrf_obj.extract::<usize>().ok()
            } else {
                None
            };
            
            Some((params, rrf_k))
        } else {
            None
        };

        let filter_str = filter.clone();
        let table_schema = self.table.arrow_schema();
        let mut projected_schema = if let Some(cols) = &columns {
            let mut fields = Vec::new();
            for c in cols {
                if let Some((_, field)) = table_schema.column_with_name(c) {
                    fields.push(std::sync::Arc::new(field.clone()));
                }
            }
            std::sync::Arc::new(arrow::datatypes::Schema::new(fields))
        } else {
            table_schema
        };

        if vs_params_combined.is_some() && projected_schema.column_with_name("distance").is_none() {
            let mut fields = projected_schema.fields().to_vec();
            fields.push(std::sync::Arc::new(arrow::datatypes::Field::new("distance", arrow::datatypes::DataType::Float32, true)));
            projected_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(fields));
        }

        let ctx = device.as_ref().or(self.device.as_ref()).map(|c| c.clone_ref(py));
        let rust_context = if let Some(py_ctx) = ctx {
            let ctx_borrow = py_ctx.bind(py).borrow();
            Some(ctx_borrow.context.clone())
        } else {
            None
        };

        let stream_res = py.allow_threads(move || {
            if let Some(c) = rust_context {
                crate::core::index::gpu::set_global_gpu_context(Some(c));
            }
            
            let mut config = self.table.query_config().clone();
            let vs_params = if let Some((p, k)) = vs_params_combined {
                if let Some(val) = k {
                    config = config.with_rrf_k(val as f32);
                }
                Some(p)
            } else {
                None
            };

            TOKIO_RUNTIME.block_on(self.table.read_with_config_stream_async(
                filter_str.as_deref(), 
                vs_params, 
                columns_clone.as_ref().map(|c| c.iter().map(|s| s.as_str()).collect::<Vec<&str>>()).as_deref(),
                config
            ))
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;

        arrow_stream_to_pyarrow(py, stream_res, projected_schema)
    }

    /// Write data to table.
    /// Supports:
    /// - List of PyArrow RecordBatches
    /// - PyArrow Table
    /// - Pandas DataFrame
    #[pyo3(signature = (data, device=None))]
    fn write(&self, py: Python<'_>, data: Bound<'_, PyAny>, device: Option<Py<PyDevice>>) -> PyResult<()> {
        let ctx = device.as_ref().or(self.device.as_ref()).map(|c| c.clone_ref(py));
        let rust_context = if let Some(py_ctx) = ctx {
            let ctx_borrow = py_ctx.bind(py).borrow();
            Some(ctx_borrow.context.clone())
        } else {
            None
        };

        // 1. Check if it's a list (List[RecordBatch])
        if let Ok(list) = data.downcast::<pyo3::types::PyList>() {
             let mut batches = Vec::new();
             for item in list {
                 // Export PyArrow RecordBatch to C Interface
                 let mut array = FFI_ArrowArray::empty();
                 let mut schema = FFI_ArrowSchema::empty();
                 
                 let array_ptr = &mut array as *mut _ as Py_uintptr_t;
                 let schema_ptr = &mut schema as *mut _ as Py_uintptr_t;
                 
                 item.call_method1("_export_to_c", (array_ptr, schema_ptr))?;
                 
                 // Import as Rust RecordBatch
                 // Safety: We just exported it from PyArrow, so it should be valid.
                 let batch = unsafe { import_record_batch_from_c(array, &schema) }
                     .map_err(|e: arrow::error::ArrowError| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
                     
                 batches.push(batch);
             }
             
             // Call core Table API, releasing the GIL
             py.allow_threads(move || {
                if let Some(c) = rust_context {
                    crate::core::index::gpu::set_global_gpu_context(Some(c));
                }
                self.table.write(batches)
             }).map_err(|e: anyhow::Error| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
             
        } else {
             // Fallback to Table/DataFrame handling
             let obj: Py<PyAny> = data.unbind();
             
             let context_clone = device.as_ref().map(|c| c.clone_ref(py));
             // Let's try arrow first as it's lighter
             if self.write_arrow(py, obj.clone_ref(py), context_clone.as_ref().map(|c| c.clone_ref(py))).is_ok() {
                 Ok(())
             } else {
                 // Try pandas
                 self.write_pandas(py, obj, context_clone)
             }
        }
    }

    /// Flush write buffer to disk (triggers vector shuffling and index building)
    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            TOKIO_RUNTIME.block_on(self.table.flush_async())
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    #[getter]
    fn index_columns(&self) -> Vec<String> {
        self.table.get_index_columns()
    }


    /// Read table to Pandas DataFrame (Alias for to_pandas)
    #[pyo3(signature = (filter=None, columns=None, device=None))]
    fn read(
        &self,
        py: Python<'_>,
        filter: Option<String>,
        columns: Option<Vec<String>>,
        device: Option<Py<PyDevice>>,
    ) -> PyResult<Py<PyAny>> {
        self.to_pandas(py, filter, None, columns, device, None)
    }

    /// Vector search on the table
    #[pyo3(signature = (column, query, k=10, filter=None, device=None))]
    fn search(
        &self,
        py: Python<'_>,
        column: String,
        query: Vec<f32>,
        k: usize,
        filter: Option<String>,
        device: Option<Py<PyDevice>>,
    ) -> PyResult<Py<PyAny>> {
        let vf_dict = PyDict::new(py);
        vf_dict.set_item("column", column)?;
        vf_dict.set_item("query", query)?;
        vf_dict.set_item("k", k)?;
        
        self.to_pandas(py, filter, Some(vf_dict), None, device, None)
    }

    /// Parallel vector search - runs multiple queries in parallel in Rust (bypasses Python GIL)
    /// 
    /// This method submits all queries to a dedicated Rust thread pool, allowing
    /// true parallelism that bypasses Python's GIL limitations.
    /// 
    /// Args:
    ///     queries: List of tuples (column, query_vector, k, filter_optional)
    ///              Each tuple is (str, List[float], int, Optional[str])
    /// 
    /// Returns:
    ///     List of PyArrow Tables (one per query)
    /// 
    /// Example:
    ///     queries = [
    ///         ("embedding", [0.1, 0.2, ...], 10, None),
    ///         ("embedding", [0.3, 0.4, ...], 10, "user_id < 100"),
    ///     ]
    ///     results = table.search_parallel(queries)
    ///     # All queries run in parallel in Rust - no GIL blocking!
    #[pyo3(signature = (queries))]
    fn search_parallel(
        &self,
        py: Python<'_>,
        queries: Vec<(String, Vec<f32>, usize, Option<String>)>,
    ) -> PyResult<Py<PyAny>> {
        use futures::future::join_all;
        
        let table = Arc::new(self.table.clone());
        
        // Release GIL and run all queries in parallel in Rust
        let results: Result<Vec<Vec<RecordBatch>>, anyhow::Error> = py.allow_threads(move || {
            TOKIO_RUNTIME.block_on(async {
                // Spawn all queries as concurrent tasks
                let tasks: Vec<_> = queries.into_iter().map(|(column, query, k, filter): (String, Vec<f32>, usize, Option<String>)| {
                    let table_clone = table.clone();
                    tokio::spawn(async move {
                        let vf_params = VectorSearchParams::new(&column, crate::core::index::VectorValue::Float32(query), k);
                        table_clone.read_async(filter.as_deref(), Some(vf_params), None).await
                    })
                }).collect();
                
                // Wait for all queries to complete in parallel
                let join_results = join_all(tasks).await;
                
                // Collect results, handling errors
                let mut all_results = Vec::new();
                for result in join_results {
                    match result {
                        Ok(Ok(batches)) => all_results.push(batches),
                        Ok(Err(e)) => return Err(anyhow::anyhow!("Query failed: {}", e)),
                        Err(e) => return Err(anyhow::anyhow!("Task join failed: {}", e)),
                    }
                }
                Ok(all_results)
            })
        });
        
        // Convert results to PyArrow tables
        match results {
            Ok(batch_vecs) => {
                let schema = self.table.arrow_schema();
                let mut py_tables = Vec::new();
                for batches in batch_vecs {
                    // Convert RecordBatches to Py<PyAny> (PyArrow Table)
                    let py_table = arrow_batches_to_pyarrow(py, batches, schema.clone())?;
                    py_tables.push(py_table);
                }
                // Manually convert to PyList to avoid PyO3 ambiguity
                let list = pyo3::types::PyList::new(py, py_tables)?;
                Ok(list.into())
            }
            Err(e) => {
                let msg = format!("{}", e);
                Err(pyo3::exceptions::PyRuntimeError::new_err(msg))
            },
        }
    }

    /// Wait for all background tasks (like index building) to complete
    fn wait_for_background_tasks(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            TOKIO_RUNTIME.block_on(self.table.wait_for_background_tasks_async())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Write Pandas DataFrame to table
    #[pyo3(signature = (df, device=None))]
    fn write_pandas(&self, py: Python<'_>, df: Py<PyAny>, device: Option<Py<PyDevice>>) -> PyResult<()> {
        // Provide current table schema to PyArrow for correct type inference (especially for vectors)
        let pyarrow = py.import("pyarrow")?;
        let schema = self.table.arrow_schema();
        let table_class = pyarrow.getattr("Table")?;
        let arrow_table = if schema.fields().is_empty() {
             table_class.call_method1("from_pandas", (df.bind(py),))?.unbind()
        } else {
             let py_schema = arrow_schema_to_pyarrow(py, schema)?;
             table_class.call_method1("from_pandas", (df.bind(py), py_schema))?.unbind()
        };
        self.write_arrow(py, arrow_table, device)
    }

    /// Write PyArrow Table to table
    #[pyo3(signature = (table, device=None))]
    fn write_arrow(&self, py: Python<'_>, table: Py<PyAny>, device: Option<Py<PyDevice>>) -> PyResult<()> {
        // Convert PyArrow Table to RecordBatches
        let batches = pyarrow_to_arrow_batches(py, table)?;
        
        let ctx = device.as_ref().or(self.device.as_ref()).map(|c| c.clone_ref(py));
        let rust_context = if let Some(py_ctx) = ctx {
            let ctx_borrow = py_ctx.bind(py).borrow();
            Some(ctx_borrow.context.clone())
        } else {
            None
        };

        // Call core Table API, releasing the GIL
        py.allow_threads(move || {
            if let Some(c) = rust_context {
                crate::core::index::gpu::set_global_gpu_context(Some(c));
            }
            self.table.write(batches)
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Commit buffered writes to disk (automatically flushes first, then finalizes metadata)
    fn commit(&self, py: Python<'_>) -> PyResult<()> {
        // Flush write buffer to disk first (triggers vector shuffling and index building)
        py.allow_threads(|| {
            TOKIO_RUNTIME.block_on(self.table.flush_async())
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        // Finalize metadata
        py.allow_threads(|| {
            self.table.commit()
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    /// Wait for any background indexing or maintenance tasks to complete.
    fn wait_for_indexes(&self, py: Python<'_>) -> PyResult<()> {
        self.wait_for_background_tasks(py)
    }

    /// Async commit (flushes then finalizes metadata asynchronously)
    fn commit_async(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            TOKIO_RUNTIME.block_on(async {
                self.table.flush_async().await?;
                self.table.commit_async().await
            })
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Compact the WAL (consolidate log entries into single batch)
    fn checkpoint(&self, py: Python<'_>) -> PyResult<()> {
        // Release GIL during checkpoint to allow other Python threads to run
        py.allow_threads(|| {
            self.table.checkpoint()
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Remove orphaned files
    fn remove_orphan_files(&self, older_than_days: i64) -> PyResult<()> {
        self.table.remove_orphan_files(older_than_days as u64)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Delete rows matching the filter (Merge-on-Read)
    fn delete(&self, filter: String) -> PyResult<()> {
        let rt = self.table.rt.as_ref().ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("Runtime not initialized"))?;
        rt.block_on(self.table.delete_async(&filter))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Truncate the table (metadata-only operation)
    fn truncate(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            self.table.truncate()
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Physically delete unreferenced data and manifest files
    fn vacuum(&self, py: Python<'_>, retention_versions: usize) -> PyResult<usize> {
        py.allow_threads(|| {
            self.table.vacuum(retention_versions)
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Get autocommit setting
    #[getter]
    fn get_autocommit(&self) -> bool {
        self.table.get_autocommit()
    }

    /// Set autocommit setting
    #[setter]
    fn set_autocommit(&self, enabled: bool) {
        self.table.set_autocommit(enabled)
    }



    /// Merge Pandas DataFrame into the table (Upsert)
    /// 
    /// Args:
    ///     df: Pandas DataFrame
    ///     key_column: Column name to merge on
    ///     mode: Optional PyMergeMode (MergeOnRead or MergeOnWrite)
    ///     device: Optional ComputeContext for GPU acceleration
    #[pyo3(signature = (df, key_column, mode=None, device=None))]
    fn merge_pandas(
        &self,
        py: Python<'_>,
        df: Py<PyAny>,
        key_column: String,
        mode: Option<PyMergeMode>,
        device: Option<Py<PyDevice>>,
    ) -> PyResult<()> {
        // 1. Convert Pandas -> Arrow RecordBatch
        let pyarrow = py.import("pyarrow")?;
        let table_class = pyarrow.getattr("Table")?;
        
        let schema = self.table.arrow_schema();
        let arrow_table = if schema.fields().is_empty() {
             table_class.call_method1("from_pandas", (df,))?.unbind()
        } else {
             let py_schema = arrow_schema_to_pyarrow(py, schema)?;
             table_class.call_method1("from_pandas", (df, py_schema))?.unbind()
        };
        
        let batches = pyarrow_to_arrow_batches(py, arrow_table)?;
        
        let ctx = device.as_ref().or(self.device.as_ref()).map(|c| c.clone_ref(py));
        let rust_context = if let Some(py_ctx) = ctx {
            let ctx_borrow = py_ctx.bind(py).borrow();
            Some(ctx_borrow.context.clone())
        } else {
            None
        };

        // 2. Call core Table API, releasing the GIL
        let merge_mode = mode.unwrap_or(PyMergeMode::MergeOnRead).into();
        py.allow_threads(move || {
            if let Some(c) = rust_context {
                crate::core::index::gpu::set_global_gpu_context(Some(c));
            }
            self.table.merge(batches, &key_column, merge_mode)
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Run Compaction to generate Manifest and optimize files
    #[pyo3(signature = (min_file_size_bytes=None))]
    fn rewrite_data_files(&self, min_file_size_bytes: Option<i64>) -> PyResult<()> {
        let mut options = CompactionOptions::default();
        if let Some(min_size) = min_file_size_bytes {
            options.min_file_size_bytes = min_size;
            options.target_file_size_bytes = min_size * 2;
        }
        
        self.table.rewrite_data_files(Some(options))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Legacy alias for rewrite_data_files
    #[pyo3(signature = (min_file_size_bytes=None))]
    fn compact(&self, min_file_size_bytes: Option<i64>) -> PyResult<()> {
        self.rewrite_data_files(min_file_size_bytes)
    }

    /// Replace the table's sort order
    fn replace_sort_order(&self, columns: Vec<String>, ascending: Vec<bool>) -> PyResult<()> {
        let col_refs: Vec<&str> = columns.iter().map(|s| s.as_str()).collect();
        self.table.replace_sort_order(&col_refs, &ascending)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Update the table's partition specification
    fn update_spec(&self, py: Python<'_>, fields: Vec<PyPartitionField>) -> PyResult<()> {
        let rust_fields: Vec<crate::core::manifest::PartitionField> = fields.into_iter().map(|f| {
            crate::core::manifest::PartitionField::new_multi(
                f.source_ids,
                f.field_id,
                f.name,
                f.transform
            )
        }).collect();

        py.allow_threads(|| {
            TOKIO_RUNTIME.block_on(self.table.update_spec(&rust_fields))
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Update table schema (Evolution)
    fn update_schema(&self, py: Python<'_>, schema: PySchema) -> PyResult<()> {
        let hdb_schema = crate::core::manifest::Schema::from_arrow(&schema.inner, 1);
        py.allow_threads(|| {
            TOKIO_RUNTIME.block_on(self.table.update_schema(hdb_schema))
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Rollback to a specific snapshot
    fn rollback_to_snapshot(&self, py: Python<'_>, snapshot_id: i64) -> PyResult<()> {
        py.allow_threads(|| {
            TOKIO_RUNTIME.block_on(self.table.rollback_to_snapshot(snapshot_id))
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Execute SQL query against the table.
    /// The table is registered as 't'.
    /// 
    /// Example:
    ///     table.sql("SELECT * FROM t WHERE id > 10")
    fn execute_sql(&self, py: Python<'_>, query: String) -> PyResult<Py<PyAny>> {
        let query = sanitize_sql(&query);
        let rt = self.table.runtime();
        
        let batch_result: Result<(Vec<RecordBatch>, arrow::datatypes::SchemaRef), String> = rt.block_on(async {
            use datafusion::prelude::SessionContext;
            let mut ctx = SessionContext::new();
            
            // Register standard functions and aggregates
            datafusion_functions::register_all(&mut ctx).map_err(|e| e.to_string())?;
            datafusion_functions_aggregate::register_all(&mut ctx).map_err(|e| e.to_string())?;
            
            let _ = crate::core::sql::vector_operators::register_vector_operators(&mut ctx);
            
            // Register table as 't' (short alias, safe from keywords)
            let provider = Arc::new(crate::core::sql::HyperStreamTableProvider::new(Arc::new(self.table.clone())));
            ctx.register_table("t", provider).map_err(|e| e.to_string())?;
            
            // Register vector UDFs (dist_l2, dist_cosine, etc.)
            for udf in crate::core::sql::vector_udf::all_vector_udfs() {
                ctx.register_udf(udf);
            }
            
            // Register Vector Aggregate functions (Additive in DF 52)
            for udf in crate::core::sql::vector_udf::all_vector_aggregates() {
                ctx.register_udaf(udf);
            }
            
            // Execute
            let df = ctx.sql(&query).await.map_err(|e| e.to_string())?;
            let schema: arrow::datatypes::SchemaRef = std::sync::Arc::new(df.schema().as_arrow().clone());
            let batches = df.collect().await.map_err(|e| e.to_string())?;
            Ok((batches, schema))
        });
        
        match batch_result {
            Ok((batches, schema)) => arrow_batches_to_pyarrow(py, batches, schema),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    fn manifest(&self, py: Python<'_>) -> PyResult<PyObject> {
        // Load manifest info from table
        let rt = self.table.runtime();
        let manifest_result = rt.block_on(async {
            self.table.get_snapshot_segments_with_version().await
        });
        
        match manifest_result {
            Ok((manifest, _version)) => {
                // Return as a dictionary for subscription support (manifest["schemas"])
                pythonize::pythonize(py, &manifest)
                    .map(|b| b.unbind())
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
            },
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e.to_string())),
        }
    }

    // ============================================================================
    // Connector APIs (Spark/Trino)
    // ============================================================================

    /// List all data files with index metadata (for Spark/Trino)
    fn list_data_files(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let files = self.table.list_data_files()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
            
        let py_files: Vec<PyDataFileInfo> = files.into_iter().map(|f| PyDataFileInfo {
            file_path: f.file_path,
            row_count: f.row_count,
            file_size_bytes: f.file_size_bytes,
            min_values: f.min_values,
            max_values: f.max_values,
            has_scalar_indexes: f.has_scalar_indexes,
            has_vector_indexes: f.has_vector_indexes,
            indexed_columns: f.indexed_columns,
        }).collect();
        
        let list = pyo3::types::PyList::new(py, py_files)?;
        Ok(list.into())
    }
    
    /// Read specific file with optional filter (index-accelerated)
    #[pyo3(signature = (file_path, filter=None, columns=None))]
    fn read_file(&self, py: Python<'_>, file_path: String, filter: Option<String>, columns: Option<Vec<String>>) -> PyResult<Py<PyAny>> {
        let table_schema = self.table.arrow_schema();
        let projected_schema = if let Some(cols) = &columns {
            let mut fields = Vec::new();
            for c in cols {
                if let Some((_, field)) = table_schema.column_with_name(c) {
                    fields.push(std::sync::Arc::new(field.clone()));
                }
            }
            std::sync::Arc::new(arrow::datatypes::Schema::new(fields))
        } else {
            table_schema
        };
        
        let batches = self.table.read_file(&file_path, columns, filter.as_deref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
            
        arrow_batches_to_pyarrow(py, batches, projected_schema)
    }
    
    /// Get splits for parallel reading (Trino)
    #[pyo3(signature = (max_split_size))]
    fn get_splits(&self, py: Python<'_>, max_split_size: usize) -> PyResult<Py<PyAny>> {
        let splits = self.table.get_splits(max_split_size)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
            
        let py_splits: Vec<PySplit> = splits.into_iter().map(|s| PySplit {
            file_path: s.file_path,
            start_offset: s.start_offset,
            length: s.length,
            row_group_ids: s.row_group_ids,
            index_file_path: s.index_file_path,
            can_use_indexes: s.can_use_indexes,
        }).collect();
        
        let list = pyo3::types::PyList::new(py, py_splits)?;
        Ok(list.into())
    }
    
    /// Read specific split with column projection
    fn read_split(&self, py: Python<'_>, split: &PySplit, columns: Vec<String>) -> PyResult<Py<PyAny>> {
        let table_schema = self.table.arrow_schema();
        let projected_schema = {
            let mut fields = Vec::new();
            for c in &columns {
                if let Some((_, field)) = table_schema.column_with_name(c) {
                    fields.push(std::sync::Arc::new(field.clone()));
                }
            }
            std::sync::Arc::new(arrow::datatypes::Schema::new(fields))
        };

        let rust_split = crate::core::table::Split {
            file_path: split.file_path.clone(),
            start_offset: split.start_offset,
            length: split.length,
            row_group_ids: split.row_group_ids.clone(),
            index_file_path: split.index_file_path.clone(),
            can_use_indexes: split.can_use_indexes,
        };
        
        let batches = self.table.read_split(&rust_split, columns, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
            
        arrow_batches_to_pyarrow(py, batches, projected_schema)
    }
    
    /// Get table statistics with index coverage
    fn get_table_statistics(&self) -> PyResult<PyTableStatistics> {
        let stats = self.table.get_table_statistics()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
            
        Ok(PyTableStatistics {
            row_count: stats.row_count,
            file_count: stats.file_count,
            total_size_bytes: stats.total_size_bytes,
            index_coverage: PyIndexCoverage {
                scalar_indexed_columns: stats.index_coverage.scalar_indexed_columns,
                vector_indexed_columns: stats.index_coverage.vector_indexed_columns,
                inverted_indexed_columns: stats.index_coverage.inverted_indexed_columns,
                total_index_size_bytes: stats.index_coverage.total_index_size_bytes,
            },
        })
    }

    fn table_uri(&self) -> String {
        self.table.table_uri()
    }

    /// Explain query plan showing index usage and execution strategy
    /// 
    /// Args:
    ///     filter: Optional SQL-like filter string
    ///     vector_filter: Optional dict for vector search (see to_arrow for parameter details)
    /// 
    /// Returns:
    ///     String explaining the query execution plan with index coverage
    #[pyo3(signature = (filter=None, vector_filter=None))]
    fn explain(&self, filter: Option<String>, vector_filter: Option<Bound<'_, PyDict>>) -> PyResult<String> {
        let vs_params = if let Some(ref vf) = vector_filter {
            let column: String = vf.get_item("column")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("vector_filter requires 'column' key"))?
                .extract()?;
            let k: usize = vf.get_item("k")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("vector_filter requires 'k' key"))?
                .extract()?;
            let query_obj = vf.get_item("query")?
                .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("vector_filter requires 'query' key"))?;
            let query: Vec<f32> = query_obj.extract()?;
            let mut params = VectorSearchParams::new(&column, crate::core::index::VectorValue::Float32(query), k);
            
            // Parse optional metric parameter
            if let Ok(Some(metric_obj)) = vf.get_item("metric") {
                if let Ok(metric_str) = metric_obj.extract::<String>() {
                    params = params.with_metric(parse_metric(&metric_str)?);
                }
            }
            
            // Parse optional ef_search parameter (for HNSW)
            if let Ok(Some(ef_obj)) = vf.get_item("ef_search") {
                if let Ok(ef_search) = ef_obj.extract::<usize>() {
                    params = params.with_ef_search(ef_search);
                }
            }
            
            // Parse optional probes parameter (for IVF)
            if let Ok(Some(probes_obj)) = vf.get_item("probes") {
                if let Ok(probes) = probes_obj.extract::<usize>() {
                    params = params.with_probes(probes);
                }
            }
            
            Some(params)
        } else {
            None
        };

        Ok(TOKIO_RUNTIME.block_on(self.table.explain(filter.as_deref(), vs_params)))
    }
}

/// Python wrapper for Nessie Catalog (Iceberg-compatible)
#[pyclass]
pub struct PyNessieCatalog {
    client: Arc<NessieClient>,
}

#[pymethods]
impl PyNessieCatalog {
    #[new]
    #[pyo3(signature = (url))]
    fn new(url: String) -> PyResult<Self> {
        let client = Arc::new(NessieClient::new(url));
        Ok(PyNessieCatalog { client })
    }

    /// Create a new table
    fn create_table(&self, branch: String, table_name: String, schema: PySchema, location: Option<String>) -> PyResult<()> {
        TOKIO_RUNTIME.block_on(async {
            self.client.create_table(&branch, &table_name, schema.inner, location.as_deref()).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    /// Load a table (returns PyTable)
    fn load_table(&self, branch: String, table_name: String) -> PyResult<PyTable> {
        let metadata = TOKIO_RUNTIME.block_on(async {
            self.client.load_table(&branch, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        // Return a Table instance pointing to the location
        // Note: Table::new expects the root path (uri). 
        // If metadata.location is full path to `metadata.json`, we might need to adjust.
        // Assuming metadata.location is the table root for now, or we need to parse it.
        // Standard Iceberg: metadata_location is path to specific json file.
        // HyperStream Table::new takes a root URI. 
        // We'll pass the location directly.
        PyTable::new_internal(&metadata.location, None).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    fn create_branch(&self, branch_name: String, source_ref: Option<String>) -> PyResult<()> {
        TOKIO_RUNTIME.block_on(async {
            self.client.create_branch(&branch_name, source_ref.as_deref()).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    fn table_exists(&self, branch: String, table_name: String) -> PyResult<bool> {
        TOKIO_RUNTIME.block_on(async {
            self.client.table_exists(&branch, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
}

/// Python wrapper for REST Catalog (Iceberg-compatible)
#[pyclass]
pub struct PyRestCatalog {
    client: Arc<RestCatalogClient>,
}

#[pymethods]
impl PyRestCatalog {
    #[new]
    #[pyo3(signature = (url, prefix=None))]
    fn new(url: String, prefix: Option<String>) -> PyResult<Self> {
        let client = Arc::new(RestCatalogClient::new(url, prefix));
        Ok(PyRestCatalog { client })
    }

    /// Create a new table
    fn create_table(&self, namespace: String, table_name: String, schema: PySchema, location: Option<String>) -> PyResult<()> {
        TOKIO_RUNTIME.block_on(async {
            self.client.create_table(&namespace, &table_name, schema.inner, location.as_deref()).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    /// Load a table (returns PyTable)
    fn load_table(&self, namespace: String, table_name: String) -> PyResult<PyTable> {
        let metadata = TOKIO_RUNTIME.block_on(async {
            self.client.load_table(&namespace, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        // Return a Table instance pointing to the location
        PyTable::new_internal(&metadata.location, None).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    fn table_exists(&self, namespace: String, table_name: String) -> PyResult<bool> {
        TOKIO_RUNTIME.block_on(async {
            self.client.table_exists(&namespace, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
}

/// Python wrapper for AWS Glue Catalog
#[pyclass]
pub struct PyGlueCatalog {
    client: Arc<GlueCatalogClient>,
}

#[pymethods]
impl PyGlueCatalog {
    #[new]
    #[pyo3(signature = (catalog_id=None))]
    fn new(catalog_id: Option<String>) -> PyResult<Self> {
        let client = TOKIO_RUNTIME.block_on(async {
            GlueCatalogClient::new(catalog_id).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        Ok(PyGlueCatalog { 
            client: Arc::new(client),
        })
    }

    /// Create a new table
    fn create_table(&self, database: String, table_name: String, schema: PySchema, location: Option<String>) -> PyResult<()> {
        TOKIO_RUNTIME.block_on(async {
            self.client.create_table(&database, &table_name, schema.inner, location.as_deref()).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    /// Load a table (returns PyTable)
    fn load_table(&self, database: String, table_name: String) -> PyResult<PyTable> {
        let metadata = TOKIO_RUNTIME.block_on(async {
            self.client.load_table(&database, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        // Return a Table instance pointing to the location
        PyTable::new_internal(&metadata.location, None).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    fn table_exists(&self, database: String, table_name: String) -> PyResult<bool> {
        TOKIO_RUNTIME.block_on(async {
            self.client.table_exists(&database, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
}

/// Python wrapper for Hive Metastore Catalog
/// 
/// Note: This is a placeholder implementation. Full Hive Metastore support
/// requires Thrift RPC integration. For production use, consider:
/// - AWS Glue (Hive-compatible)
/// - Iceberg REST Catalog
/// - Nessie Catalog
#[pyclass]
pub struct PyHiveCatalog {
    client: Arc<HiveMetastoreClient>,
}

#[pymethods]
impl PyHiveCatalog {
    #[new]
    #[pyo3(signature = (url))]
    fn new(url: String) -> PyResult<Self> {
        let client = TOKIO_RUNTIME.block_on(async {
            HiveMetastoreClient::new(url)
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        let client = Arc::new(client);
        Ok(PyHiveCatalog { client })
    }

    /// Create a new table
    fn create_table(&self, database: String, table_name: String, schema: PySchema, location: Option<String>) -> PyResult<()> {
        TOKIO_RUNTIME.block_on(async {
            self.client.create_table(&database, &table_name, schema.inner, location.as_deref()).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    /// Load a table (placeholder - returns informative error)
    fn load_table(&self, database: String, table_name: String) -> PyResult<PyTable> {
        let metadata = TOKIO_RUNTIME.block_on(async {
            self.client.load_table(&database, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        PyTable::new_internal(&metadata.location, None).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    fn table_exists(&self, database: String, table_name: String) -> PyResult<bool> {
        TOKIO_RUNTIME.block_on(async {
            self.client.table_exists(&database, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
}

/// Python wrapper for Unity Catalog (Databricks)
#[pyclass]
pub struct PyUnityCatalog {
    client: Arc<UnityCatalogClient>,
}

#[pymethods]
impl PyUnityCatalog {
    #[new]
    #[pyo3(signature = (url, token))]
    fn new(url: String, token: String) -> PyResult<Self> {
        let client = Arc::new(UnityCatalogClient::new(url, token));
        Ok(PyUnityCatalog { client })
    }

    /// Create a new table
    fn create_table(&self, catalog: String, table_name: String, schema: PySchema, location: Option<String>) -> PyResult<()> {
        TOKIO_RUNTIME.block_on(async {
            self.client.create_table(&catalog, &table_name, schema.inner, location.as_deref()).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    /// Load a table (returns PyTable)
    fn load_table(&self, catalog: String, table_name: String) -> PyResult<PyTable> {
        let metadata = TOKIO_RUNTIME.block_on(async {
            self.client.load_table(&catalog, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        PyTable::new_internal(&metadata.location, None).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    fn table_exists(&self, catalog: String, table_name: String) -> PyResult<bool> {
        TOKIO_RUNTIME.block_on(async {
            self.client.table_exists(&catalog, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
}

/// Python wrapper for JDBC Catalog
#[pyclass]
pub struct PyJdbcCatalog {
    client: Arc<JdbcCatalogClient>,
}

#[pymethods]
impl PyJdbcCatalog {
    #[new]
    #[pyo3(signature = (uri, warehouse=None, catalog_name=None))]
    fn new(uri: String, warehouse: Option<String>, catalog_name: Option<String>) -> PyResult<Self> {
        let catalog_name = catalog_name.unwrap_or_else(|| "default".to_string());
        let client = TOKIO_RUNTIME.block_on(async {
            JdbcCatalogClient::new(uri, warehouse, catalog_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        Ok(PyJdbcCatalog { 
            client: Arc::new(client),
        })
    }

    /// Create a new table
    fn create_table(&self, namespace: String, table_name: String, schema: PySchema, location: Option<String>) -> PyResult<()> {
        TOKIO_RUNTIME.block_on(async {
            self.client.create_table(&namespace, &table_name, schema.inner, location.as_deref()).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    /// Load a table (returns PyTable)
    fn load_table(&self, namespace: String, table_name: String) -> PyResult<PyTable> {
        let metadata = TOKIO_RUNTIME.block_on(async {
            self.client.load_table(&namespace, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        PyTable::new_internal(&metadata.location, None).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    fn table_exists(&self, namespace: String, table_name: String) -> PyResult<bool> {
        TOKIO_RUNTIME.block_on(async {
            self.client.table_exists(&namespace, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
}

// Helper functions for Arrow C Data Interface

fn arrow_schema_to_pyarrow(py: Python<'_>, schema: arrow::datatypes::SchemaRef) -> PyResult<Py<PyAny>> {
    let mut ffi_schema = FFI_ArrowSchema::try_from(schema.as_ref())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
    
    let schema_ptr = &mut ffi_schema as *mut _ as Py_uintptr_t;
    let pyarrow = py.import("pyarrow")?;
    let schema_class = pyarrow.getattr("Schema")?;
    let py_schema = schema_class.call_method1("_import_from_c", (schema_ptr,))?.unbind();
    
    Ok(py_schema)
}

fn arrow_batches_to_pyarrow(py: Python<'_>, batches: Vec<RecordBatch>, schema: arrow::datatypes::SchemaRef) -> PyResult<Py<PyAny>> {
    // Use Arrow C Stream Interface for efficient transfer
    let batch_iter = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);
    
    // Export to C Stream
    let stream = FFI_ArrowArrayStream::new(Box::new(batch_iter));
    let stream_ptr = Box::into_raw(Box::new(stream)) as Py_uintptr_t;
    
    // Import in Python via PyArrow
    let pyarrow = py.import("pyarrow")?;
    let reader_class = pyarrow.getattr("RecordBatchReader")?;
    let table = reader_class.call_method1("_import_from_c", (stream_ptr,))?
        .call_method0("read_all")?
        .unbind();
    
    Ok(table)
}

struct StreamRecordBatchReader {
    schema: arrow::datatypes::SchemaRef,
    stream: futures::stream::BoxStream<'static, anyhow::Result<RecordBatch>>,
}

impl RecordBatchReader for StreamRecordBatchReader {
    fn schema(&self) -> arrow::datatypes::SchemaRef {
        self.schema.clone()
    }
}

impl Iterator for StreamRecordBatchReader {
    type Item = Result<RecordBatch, arrow::error::ArrowError>;
    
    fn next(&mut self) -> Option<Self::Item> {
        TOKIO_RUNTIME.block_on(self.stream.next()).map(|res| {
            res.map_err(|e| arrow::error::ArrowError::ExternalError(e.into()))
        })
    }
}

fn arrow_stream_to_pyarrow(py: Python<'_>, stream: futures::stream::BoxStream<'static, anyhow::Result<RecordBatch>>, schema: arrow::datatypes::SchemaRef) -> PyResult<Py<PyAny>> {
    let reader = StreamRecordBatchReader {
        schema: schema.clone(),
        stream,
    };
    
    // Export to C Stream
    let stream = FFI_ArrowArrayStream::new(Box::new(reader));
    let stream_ptr = Box::into_raw(Box::new(stream)) as Py_uintptr_t;
    
    // Import in Python via PyArrow
    let pyarrow = py.import("pyarrow")?;
    let reader_class = pyarrow.getattr("RecordBatchReader")?;
    let reader = reader_class.call_method1("_import_from_c", (stream_ptr,))?.unbind();
    
    Ok(reader)
}

#[pyclass(name = "Session")]
pub struct PySession {
    inner: Arc<crate::core::sql::session::HyperStreamSession>,
}

#[pymethods]
impl PySession {
    #[new]
    #[pyo3(signature = (memory_mb=None))]
    pub fn new(memory_mb: Option<usize>) -> PyResult<Self> {
        let limit_bytes = memory_mb.map(|mb| mb * 1024 * 1024);
        Ok(Self {
            inner: Arc::new(crate::core::sql::session::HyperStreamSession::new(limit_bytes)),
        })
    }

    pub fn register(&self, name: String, table: &PyTable) -> PyResult<()> {
        self.inner.register_table(&name, Arc::new(table.table.clone()))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    pub fn sql(&self, py: Python<'_>, query: String) -> PyResult<Py<PyAny>> {
        let query = sanitize_sql(&query);
        let (batches, schema) = TOKIO_RUNTIME.block_on(self.inner.sql(&query))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        arrow_batches_to_pyarrow(py, batches, schema)
    }
}

#[pyfunction]
#[pyo3(signature = (catalog_type, config))]
pub fn create_catalog(py: Python<'_>, catalog_type: String, config: std::collections::HashMap<String, String>) -> PyResult<Py<PyAny>> {
    match catalog_type.to_lowercase().as_str() {
        "nessie" => {
            let url = config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Nessie catalog"))?;
            let catalog = PyNessieCatalog::new(url)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        "rest" => {
             let url = config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Rest catalog"))?;
             let prefix = config.get("prefix").cloned();
             let catalog = PyRestCatalog::new(url, prefix)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        "glue" => {
             let catalog_id = config.get("catalog_id").cloned();
             let catalog = PyGlueCatalog::new(catalog_id)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        "hive" => {
             let url = config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Hive catalog"))?;
             let catalog = PyHiveCatalog::new(url)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        "unity" => {
            let url = config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Unity catalog"))?;
            let token = config.get("token").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'token' for Unity catalog"))?;
            let catalog = PyUnityCatalog::new(url, token)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        "jdbc" => {
            let uri = config.get("uri").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'uri' for JDBC catalog"))?;
            let warehouse = config.get("warehouse").cloned();
            let catalog_name = config.get("catalog_name").cloned();
            let catalog = PyJdbcCatalog::new(uri, warehouse, catalog_name)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Unknown catalog type: {}", catalog_type))),
    }
}

#[pyfunction]
pub fn create_catalog_from_config(py: Python<'_>, path: String) -> PyResult<Py<PyAny>> {
    let config = CatalogConfig::load_from_file(&path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
    match config.catalog_type {
        CatalogType::Nessie => {
            let url = config.config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Nessie catalog"))?;
            let catalog = PyNessieCatalog::new(url)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Rest => {
             let url = config.config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Rest catalog"))?;
             let prefix = config.config.get("prefix").cloned();
             let catalog = PyRestCatalog::new(url, prefix)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Glue => {
             let catalog_id = config.config.get("catalog_id").cloned();
             let catalog = PyGlueCatalog::new(catalog_id)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Hive => {
             let url = config.config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Hive catalog"))?;
             let catalog = PyHiveCatalog::new(url)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Unity => {
            let url = config.config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Unity catalog"))?;
            let token = config.config.get("token").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'token' for Unity catalog"))?;
            let catalog = PyUnityCatalog::new(url, token)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Jdbc => {
            let uri = config.config.get("uri").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'uri' for JDBC catalog"))?;
            let warehouse = config.config.get("warehouse").cloned();
            let catalog_name = config.config.get("catalog_name").cloned();
            let catalog = PyJdbcCatalog::new(uri, warehouse, catalog_name)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
    }
}

#[pyfunction]
pub fn load_default_catalog(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let config = CatalogConfig::load_default()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
    match config.catalog_type {
        CatalogType::Nessie => {
            let url = config.config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Nessie catalog"))?;
            let catalog = PyNessieCatalog::new(url)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Rest => {
             let url = config.config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Rest catalog"))?;
             let prefix = config.config.get("prefix").cloned();
             let catalog = PyRestCatalog::new(url, prefix)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Glue => {
             let catalog_id = config.config.get("catalog_id").cloned();
             let catalog = PyGlueCatalog::new(catalog_id)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Hive => {
             let url = config.config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Hive catalog"))?;
             let catalog = PyHiveCatalog::new(url)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Unity => {
            let url = config.config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Unity catalog"))?;
            let token = config.config.get("token").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'token' for Unity catalog"))?;
            let catalog = PyUnityCatalog::new(url, token)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Jdbc => {
            let uri = config.config.get("uri").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'uri' for JDBC catalog"))?;
            let warehouse = config.config.get("warehouse").cloned();
            let catalog_name = config.config.get("catalog_name").cloned();
            let catalog = PyJdbcCatalog::new(uri, warehouse, catalog_name)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
    }
}

#[pyfunction]
#[pyo3(signature = (level="INFO"))]
pub fn init_logging(level: &str) -> PyResult<()> {
    crate::telemetry::tracing::update_log_level(level)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    crate::telemetry::tracing::init_tracing("hyperstreamdb")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
}

#[pyfunction]
pub fn open_table(_py: Python<'_>, uri: &str) -> PyResult<PyTable> {
    PyTable::new_internal(uri, None).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
}

// Helper to import RecordBatch from C Interface

// Helper to import RecordBatch from C Interface
unsafe fn import_record_batch_from_c(array: FFI_ArrowArray, schema: &FFI_ArrowSchema) -> Result<RecordBatch, arrow::error::ArrowError> {
    let array_data = arrow::ffi::from_ffi(array, schema)?;
    let struct_array = arrow::array::StructArray::from(array_data);
    Ok(RecordBatch::from(struct_array))
}

fn pyarrow_to_arrow_batches(py: Python<'_>, table: Py<PyAny>) -> PyResult<Vec<RecordBatch>> {
    // Convert PyArrow Table to batches via C Stream Interface
    let _pyarrow = py.import("pyarrow")?;
    
    // Get RecordBatchReader
    let reader = table.call_method0(py, "to_reader")?;
    
    // Create struct to hold the exported stream
    let mut stream = FFI_ArrowArrayStream::empty();
    let stream_ptr = &mut stream as *mut FFI_ArrowArrayStream as Py_uintptr_t;

    // Export to C Stream (pass pointer to python)
    reader.call_method1(py, "_export_to_c", (stream_ptr,))?;
    
    // Import from C Stream
    let stream_reader = ArrowArrayStreamReader::try_new(stream)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
    
    let mut batches = Vec::new();
    for batch_result in stream_reader {
        let batch = batch_result
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        batches.push(batch);
    }
    
    Ok(batches)
}

fn extract_schema(schema_obj: Bound<'_, PyAny>) -> PyResult<arrow::datatypes::SchemaRef> {
    // 1. Try to unwrap if it is a PySchema directly
    if let Ok(py_schema) = schema_obj.extract::<PySchema>() {
         return Ok(py_schema.inner.clone());
    }

    // 2. Try to use Arrow C Data Interface via _export_to_c
    if schema_obj.hasattr("_export_to_c")? {
        let mut ffi_schema = FFI_ArrowSchema::empty();
        let schema_ptr = &mut ffi_schema as *mut FFI_ArrowSchema as Py_uintptr_t;
        schema_obj.call_method1("_export_to_c", (schema_ptr,))?;
        
        let schema = arrow::datatypes::Schema::try_from(&ffi_schema)
            .map_err(|e| pyo3::exceptions::PyTypeError::new_err(format!("Arrow schema extraction failed: {}", e)))?;
        return Ok(Arc::new(schema));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected hyperstreamdb.Schema or pyarrow.Schema object"
    ))
}

fn extract_partition_spec(spec_obj: Bound<'_, PyAny>) -> PyResult<crate::core::manifest::PartitionSpec> {
    let dict = spec_obj.downcast::<pyo3::types::PyDict>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("partition_spec must be a dictionary"))?;

    let fields_obj = dict.get_item("fields")?
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("partition_spec must contain 'fields'"))?;
    let fields_list = fields_obj.downcast::<pyo3::types::PyList>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("'fields' must be a list"))?;

    let mut fields = Vec::new();
    for item in fields_list {
        let f_dict = item.downcast::<pyo3::types::PyDict>()
            .map_err(|_| pyo3::exceptions::PyTypeError::new_err("Each partition field must be a dictionary"))?;

        let name = f_dict.get_item("name")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'name' in partition field"))?
            .extract::<String>()?;
        let transform = f_dict.get_item("transform")?
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Missing 'transform' in partition field"))?
            .extract::<String>()?;
        
        let source_id = f_dict.get_item("source_id")?.and_then(|i| i.extract::<i32>().ok());
        let field_id = f_dict.get_item("field_id")?.and_then(|i| i.extract::<i32>().ok());

        fields.push(crate::core::manifest::PartitionField {
            source_ids: source_id.map(|id| vec![id]).unwrap_or_default(),
            source_id,
            field_id,
            name,
            transform,
        });
    }

    Ok(crate::core::manifest::PartitionSpec {
        spec_id: 0,
        fields,
    })
}
