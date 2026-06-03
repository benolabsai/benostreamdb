// Copyright (c) 2026 Richard Albright. All rights reserved.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use regex::Regex;
use once_cell::sync::Lazy;
use crate::core::index::VectorMetric;
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use arrow::array::RecordBatchIterator;
use arrow::ffi_stream::{FFI_ArrowArrayStream, ArrowArrayStreamReader};
use pyo3::ffi::Py_uintptr_t;
use tokio::runtime::Runtime;
use std::sync::Arc;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use crate::core::manifest::IndexAlgorithm;
use futures::StreamExt;

pub static SQL_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)dist_l2\(([^,]+),\s*\[([^\]]+)\]\)").unwrap());

/// Sanitize SQL query by replacing Python-side helper function `dist_l2` with the
/// native DataFusion UDF `l2_distance`. Additionally, validate the query for
/// common injection patterns and reject any SQL containing `;` (statement
/// termination) or `--` (line comment) that would indicate multi-statement
/// or comment injection.
pub fn sanitize_sql(query: &str) -> PyResult<String> {
    // Reject queries with statement terminators or comment markers
    if query.contains(';') || query.contains("--") {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "SQL query contains disallowed characters (';' or '--'). \
             Use parameterized queries or filter expressions instead."
        ));
    }

    // Reject queries with embedded NULL bytes (common in FFI injection)
    if query.contains('\0') {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "SQL query contains NULL bytes"
        ));
    }

    Ok(SQL_REGEX.replace_all(query, "dist_l2($1, ARRAY[$2])").to_string())
}

/// Module-level global Tokio runtime for all Python-bound operations.
/// Sharing a single runtime prevents 'Cannot drop a runtime in a context where blocking is not allowed' panics.
pub static TOKIO_RUNTIME: Lazy<Arc<Runtime>> = Lazy::new(|| {
    Arc::new(Runtime::new().expect("Failed to create unified Tokio runtime for HyperStreamDB"))
});

/// Helper function to parse metric string to VectorMetric enum
/// Uses native Rust names: L2, Cosine, InnerProduct, L1, Hamming, Jaccard
/// Also accepts lowercase aliases for backward compatibility
pub fn parse_metric(metric_str: &str) -> PyResult<VectorMetric> {
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

pub fn parse_index_algorithm(val: Bound<'_, PyAny>) -> PyResult<IndexAlgorithm> {
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

#[pyfunction]
#[pyo3(signature = (level="INFO"))]
pub fn init_logging(level: &str) -> PyResult<()> {
    crate::telemetry::tracing::update_log_level(level)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let guard = crate::telemetry::tracing::init_tracing("hyperstreamdb")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Box::leak(Box::new(guard));
    Ok(())
}

// ============================================================================
// Arrow C Data Interface helpers
// ============================================================================

pub fn arrow_schema_to_pyarrow(py: Python<'_>, schema: arrow::datatypes::SchemaRef) -> PyResult<Py<PyAny>> {
    let mut ffi_schema = FFI_ArrowSchema::try_from(schema.as_ref())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;

    let schema_ptr = &mut ffi_schema as *mut _ as Py_uintptr_t;
    let pyarrow = py.import("pyarrow")?;
    let schema_class = pyarrow.getattr("Schema")?;
    let py_schema = schema_class.call_method1("_import_from_c", (schema_ptr,))?.unbind();

    Ok(py_schema)
}

pub fn arrow_batches_to_pyarrow(py: Python<'_>, batches: Vec<RecordBatch>, schema: arrow::datatypes::SchemaRef) -> PyResult<Py<PyAny>> {
    // Use Arrow C Stream Interface for efficient transfer
    let actual_schema = if let Some(first) = batches.first() {
        first.schema()
    } else {
        schema
    };
    let batch_iter = RecordBatchIterator::new(batches.into_iter().map(Ok), actual_schema);

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

pub struct StreamRecordBatchReader {
    pub schema: arrow::datatypes::SchemaRef,
    pub stream: futures::stream::BoxStream<'static, anyhow::Result<RecordBatch>>,
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

pub fn arrow_stream_to_pyarrow(py: Python<'_>, stream: futures::stream::BoxStream<'static, anyhow::Result<RecordBatch>>, schema: arrow::datatypes::SchemaRef) -> PyResult<Py<PyAny>> {
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

// Helper to validate that a Python object is a PyArrow RecordBatch before FFI export.
// Returns a clear TypeError if the object does not conform to the expected type.
pub fn validate_record_batch(obj: &Bound<'_, PyAny>) -> PyResult<()> {
    let type_name = obj.get_type().name()
        .map_err(|e| pyo3::exceptions::PyTypeError::new_err(
            format!("Cannot determine type of object passed to table.write(): {}", e)
        ))?;
    // PyArrow RecordBatch reports its type as "RecordBatch" or "pyarrow.lib.RecordBatch"
    let type_name_str = type_name.to_string_lossy();
    if !type_name_str.ends_with("RecordBatch") {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            format!("Expected pyarrow.RecordBatch, got '{}'. \
                     Pass a RecordBatch, a list of RecordBatches, a PyArrow Table, or a Pandas DataFrame.", type_name)
        ));
    }
    Ok(())
}

pub unsafe fn import_record_batch_from_c(array: FFI_ArrowArray, schema: &FFI_ArrowSchema) -> Result<RecordBatch, arrow::error::ArrowError> {
    let array_data = arrow::ffi::from_ffi(array, schema)?;
    let struct_array = arrow::array::StructArray::from(array_data);
    Ok(RecordBatch::from(struct_array))
}

pub fn pyarrow_to_arrow_batches(py: Python<'_>, table: Py<PyAny>) -> PyResult<Vec<RecordBatch>> {
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

pub fn extract_schema(schema_obj: Bound<'_, PyAny>) -> PyResult<arrow::datatypes::SchemaRef> {
    use super::schema::PySchema;

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

pub fn extract_partition_spec(spec_obj: Bound<'_, PyAny>) -> PyResult<crate::core::manifest::PartitionSpec> {
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
