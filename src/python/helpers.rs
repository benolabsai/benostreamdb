// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::index::VectorMetric;
use crate::core::manifest::IndexAlgorithm;
use arrow::array::RecordBatchIterator;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow::record_batch::{RecordBatch, RecordBatchReader};
use futures::StreamExt;
use once_cell::sync::Lazy;
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use regex::Regex;
use std::sync::Arc;
use tokio::runtime::Runtime;

/// Pattern rewriting the Python-side helper `dist_l2(...)` to the native UDF
/// call. A literal pattern cannot fail to compile; `None` exists only so this
/// stays total (the query is returned unchanged) instead of unwrapping.
pub static SQL_REGEX: Lazy<Option<Regex>> =
    Lazy::new(|| Regex::new(r"(?i)dist_l2\(([^,]+),\s*\[([^\]]+)\]\)").ok());

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
             Use parameterized queries or filter expressions instead.",
        ));
    }

    // Reject queries with embedded NULL bytes (common in FFI injection)
    if query.contains('\0') {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "SQL query contains NULL bytes",
        ));
    }

    Ok(match SQL_REGEX.as_ref() {
        Some(re) => re.replace_all(query, "dist_l2($1, ARRAY[$2])").to_string(),
        None => query.to_string(),
    })
}

/// Module-level global Tokio runtime for all Python-bound operations.
/// Sharing a single runtime prevents 'Cannot drop a runtime in a context where blocking is not allowed' panics.
///
/// `Runtime::new()` has no infallible form and fails only if the OS cannot give
/// the process a reactor/worker threads. If that happens the binding layer is
/// unusable, so surfacing it as a clear error at first use is correct — this is
/// a documented residual invariant in NO_PANIC_POLICY.md.
#[allow(clippy::expect_used)]
pub static TOKIO_RUNTIME: Lazy<Arc<Runtime>> = Lazy::new(|| {
    Arc::new(Runtime::new().expect("Failed to create unified Tokio runtime for BenoStreamDB"))
});

/// Load a memory-mapped [`MultiSegmentCsrGraph`] for `graph_column` from the
/// table's graph-index sidecars (`{file}.graph_v2.csr.{offsets,edges,dict}`).
///
/// This is the single source of truth for the CSR fast path, shared by
/// [`crate::python::graph::PyGraphAPI`], `PyTable::subgraph`,
/// `PyTable::graph_neighbors` and `PyTable::drift_search`.
///
/// Returns `Ok(None)` when the table has no graph index on that column, so
/// callers can fall back to the SQL `bfs_visited` path. The CSR is built on the
/// index's *source* column, so `get_neighbors` yields forward (out-)neighbours.
pub(crate) fn load_multi_csr(
    table: &crate::core::table::Table,
    graph_column: &str,
) -> anyhow::Result<Option<crate::core::index::csr_graph::MultiSegmentCsrGraph>> {
    TOKIO_RUNTIME.block_on(table.load_graph_index(graph_column))
}

pub(crate) use crate::core::table::graph::csr_bfs_visited;

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
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Invalid metric '{}'",
            metric_str
        ))),
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
                search_device: None,
            }),
            "hnsw_pq" | "pq" => Ok(IndexAlgorithm::HnswPq {
                metric: "l2".to_string(),
                complexity: 16,
                quality: 200,
                compression: 8,
            }),
            "hnsw_tq4" | "tq4" => Ok(IndexAlgorithm::HnswTq4 {
                metric: "l2".to_string(),
                complexity: 16,
                quality: 200,
            }),
            "hnsw_tq8" | "tq8" => Ok(IndexAlgorithm::HnswTq8 {
                metric: "l2".to_string(),
                complexity: 16,
                quality: 200,
            }),
            "bm25" => Ok(IndexAlgorithm::Bm25 {
                k1: 1.2,
                b: 0.75,
                tokenizer: "default".to_string(),
            }),
            "bloom" => Ok(IndexAlgorithm::Bloom { fpr: 0.05 }),
            "bitmap" | "inverted" => Ok(IndexAlgorithm::Bitmap),
            "csr_graph" | "graph" | "csr" => Ok(IndexAlgorithm::CsrGraph {
                src_column: "src".to_string(),
                dst_column: "dst".to_string(),
            }),
            "composite_bitmap" => Ok(IndexAlgorithm::CompositeBitmap { columns: vec![] }),
            "json_path" => Ok(IndexAlgorithm::JsonPath { paths: vec![] }),
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown index type: {}",
                s
            ))),
        }
    } else if let Ok(dict) = val.downcast::<PyDict>() {
        let type_str: String = dict
            .get_item("type")?
            .ok_or_else(|| {
                pyo3::exceptions::PyKeyError::new_err("Missing 'type' key in index config")
            })?
            .extract()?;
        match type_str.to_lowercase().as_str() {
            "hnsw" => {
                let metric = dict
                    .get_item("metric")?
                    .and_then(|v| v.extract::<String>().ok())
                    .unwrap_or_else(|| "l2".to_string());
                let complexity = dict
                    .get_item("complexity")?
                    .and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| {
                        dict.get_item("m")
                            .ok()
                            .flatten()
                            .and_then(|v| v.extract::<usize>().ok())
                    })
                    .unwrap_or(16);
                let quality = dict
                    .get_item("quality")?
                    .and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| {
                        dict.get_item("ef_construction")
                            .ok()
                            .flatten()
                            .and_then(|v| v.extract::<usize>().ok())
                    })
                    .unwrap_or(200);
                let build_device = dict
                    .get_item("build_device")?
                    .and_then(|v| v.extract::<String>().ok());
                let search_device = dict
                    .get_item("search_device")?
                    .and_then(|v| v.extract::<String>().ok());
                Ok(IndexAlgorithm::Hnsw {
                    metric,
                    complexity,
                    quality,
                    build_device,
                    search_device,
                })
            }
            "hnsw_pq" | "pq" => {
                let metric = dict
                    .get_item("metric")?
                    .and_then(|v| v.extract::<String>().ok())
                    .unwrap_or_else(|| "l2".to_string());
                let compression = dict
                    .get_item("compression")?
                    .and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| {
                        dict.get_item("subspaces")
                            .ok()
                            .flatten()
                            .and_then(|v| v.extract::<usize>().ok())
                    })
                    .unwrap_or(8);
                let complexity = dict
                    .get_item("complexity")?
                    .and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| {
                        dict.get_item("m")
                            .ok()
                            .flatten()
                            .and_then(|v| v.extract::<usize>().ok())
                    })
                    .unwrap_or(16);
                let quality = dict
                    .get_item("quality")?
                    .and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| {
                        dict.get_item("ef_construction")
                            .ok()
                            .flatten()
                            .and_then(|v| v.extract::<usize>().ok())
                    })
                    .unwrap_or(200);
                Ok(IndexAlgorithm::HnswPq {
                    metric,
                    complexity,
                    quality,
                    compression,
                })
            }
            "hnsw_tq4" | "tq4" => {
                let metric = dict
                    .get_item("metric")?
                    .and_then(|v| v.extract::<String>().ok())
                    .unwrap_or_else(|| "l2".to_string());
                let complexity = dict
                    .get_item("complexity")?
                    .and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| {
                        dict.get_item("m")
                            .ok()
                            .flatten()
                            .and_then(|v| v.extract::<usize>().ok())
                    })
                    .unwrap_or(16);
                let quality = dict
                    .get_item("quality")?
                    .and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| {
                        dict.get_item("ef_construction")
                            .ok()
                            .flatten()
                            .and_then(|v| v.extract::<usize>().ok())
                    })
                    .unwrap_or(200);
                Ok(IndexAlgorithm::HnswTq4 {
                    metric,
                    complexity,
                    quality,
                })
            }
            "hnsw_tq8" | "tq8" => {
                let metric = dict
                    .get_item("metric")?
                    .and_then(|v| v.extract::<String>().ok())
                    .unwrap_or_else(|| "l2".to_string());
                let complexity = dict
                    .get_item("complexity")?
                    .and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| {
                        dict.get_item("m")
                            .ok()
                            .flatten()
                            .and_then(|v| v.extract::<usize>().ok())
                    })
                    .unwrap_or(16);
                let quality = dict
                    .get_item("quality")?
                    .and_then(|v| v.extract::<usize>().ok())
                    .or_else(|| {
                        dict.get_item("ef_construction")
                            .ok()
                            .flatten()
                            .and_then(|v| v.extract::<usize>().ok())
                    })
                    .unwrap_or(200);
                Ok(IndexAlgorithm::HnswTq8 {
                    metric,
                    complexity,
                    quality,
                })
            }
            "bm25" => {
                let k1 = dict
                    .get_item("k1")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(1.2);
                let b = dict
                    .get_item("b")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(0.75);
                let tokenizer = dict
                    .get_item("tokenizer")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_else(|| "default".to_string());
                Ok(IndexAlgorithm::Bm25 { k1, b, tokenizer })
            }
            "bloom" => {
                let fpr = dict
                    .get_item("fpr")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or(0.05);
                Ok(IndexAlgorithm::Bloom { fpr })
            }
            "bitmap" | "inverted" => Ok(IndexAlgorithm::Bitmap),
            "csr_graph" | "graph" | "csr" => {
                let src_column = dict
                    .get_item("src_column")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_else(|| "src".to_string());
                let dst_column = dict
                    .get_item("dst_column")?
                    .and_then(|v| v.extract().ok())
                    .unwrap_or_else(|| "dst".to_string());
                Ok(IndexAlgorithm::CsrGraph {
                    src_column,
                    dst_column,
                })
            }
            "composite_bitmap" => {
                let columns = dict
                    .get_item("columns")?
                    .and_then(|v| v.extract::<Vec<String>>().ok())
                    .unwrap_or_default();
                Ok(IndexAlgorithm::CompositeBitmap { columns })
            }
            "json_path" => {
                let paths = dict
                    .get_item("paths")?
                    .and_then(|v| v.extract::<Vec<String>>().ok())
                    .unwrap_or_default();
                Ok(IndexAlgorithm::JsonPath { paths })
            }
            _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unknown index type: {}",
                type_str
            ))),
        }
    } else {
        Err(pyo3::exceptions::PyTypeError::new_err(
            "Index algorithm must be a string or a dict",
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (level="INFO"))]
pub fn init_logging(level: &str) -> PyResult<()> {
    crate::telemetry::tracing::update_log_level(level)
        .map_err(pyo3::exceptions::PyRuntimeError::new_err)?;
    let guard = crate::telemetry::tracing::init_tracing("benostreamdb")
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Box::leak(Box::new(guard));
    Ok(())
}

/// Return the Rust build profile of the loaded extension: `"release"` or
/// `"debug"`.
///
/// `maturin develop` (without `--release`) produces an unoptimised debug build
/// whose graph traversal and vector kernels can be an order of magnitude
/// slower. Benchmarks and Graph RAG pipelines should assert on this before
/// trusting any latency numbers.
#[pyfunction]
pub fn build_profile() -> &'static str {
    if cfg!(debug_assertions) {
        "debug"
    } else {
        "release"
    }
}

/// Return `true` when the loaded extension was compiled without optimisations
/// (i.e. `maturin develop` instead of `maturin develop --release`).
#[pyfunction]
pub fn is_debug_build() -> bool {
    cfg!(debug_assertions)
}

/// Set the process GPU device for subsequent operations.
///
/// Accepts `auto` | `cpu` | `cuda[:N]` | `mps`/`metal` | `intel`/`xpu` |
/// `rocm`/`hip`. This is the same mapping the Spark/Trino JNI bridges and the
/// Flight/MCP servers use, so every surface resolves a device identically.
/// Returns the resolved backend name (e.g. `"cuda"`, `"cpu"`).
#[pyfunction]
pub fn set_gpu_device(device: &str) -> String {
    let ctx = crate::core::index::gpu::context_from_device_str(device);
    let resolved = format!("{:?}", ctx.backend).to_lowercase();
    crate::core::index::gpu::set_thread_gpu_context(Some(ctx));
    resolved
}

/// The currently active GPU backend name (`"cpu"` when none is set).
#[pyfunction]
pub fn gpu_device() -> String {
    crate::core::index::gpu::get_thread_gpu_context()
        .map(|c| format!("{:?}", c.backend).to_lowercase())
        .unwrap_or_else(|| "cpu".to_string())
}

/// Tear down the process cleanly at interpreter exit.
///
/// Registered as a Python `atexit` handler by the package `__init__`. Drains
/// any still-running background tasks first — an index build may be using the
/// GPU, and leaving it running while the tokio runtime is torn down makes a
/// worker thread fault (a flaky segfault on GPU workloads). Then releases the
/// GPU context. Safe to call more than once.
#[pyfunction]
pub fn shutdown_gpu(py: Python<'_>) {
    py.detach(|| {
        TOKIO_RUNTIME.block_on(crate::core::table::drain_background_tasks(
            std::time::Duration::from_secs(5),
        ));
    });
    crate::core::index::gpu::clear_gpu_context();
}

// ============================================================================
// Arrow C Data Interface helpers
// ============================================================================

pub fn arrow_schema_to_pyarrow(
    py: Python<'_>,
    schema: arrow::datatypes::SchemaRef,
) -> PyResult<Py<PyAny>> {
    let mut ffi_schema = FFI_ArrowSchema::try_from(schema.as_ref())
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))?;

    let schema_ptr = &mut ffi_schema as *mut _ as Py_uintptr_t;
    let pyarrow = py.import("pyarrow")?;
    let schema_class = pyarrow.getattr("Schema")?;
    let py_schema = schema_class
        .call_method1("_import_from_c", (schema_ptr,))?
        .unbind();

    Ok(py_schema)
}

pub fn arrow_batches_to_pyarrow(
    py: Python<'_>,
    batches: Vec<RecordBatch>,
    schema: arrow::datatypes::SchemaRef,
) -> PyResult<Py<PyAny>> {
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
    let table = reader_class
        .call_method1("_import_from_c", (stream_ptr,))?
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
        TOKIO_RUNTIME
            .block_on(self.stream.next())
            .map(|res| res.map_err(|e| arrow::error::ArrowError::ExternalError(e.into())))
    }
}

pub fn arrow_stream_to_pyarrow(
    py: Python<'_>,
    stream: futures::stream::BoxStream<'static, anyhow::Result<RecordBatch>>,
    schema: arrow::datatypes::SchemaRef,
) -> PyResult<Py<PyAny>> {
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
    let reader = reader_class
        .call_method1("_import_from_c", (stream_ptr,))?
        .unbind();

    Ok(reader)
}

// Helper to validate that a Python object is a PyArrow RecordBatch before FFI export.
// Returns a clear TypeError if the object does not conform to the expected type.
pub fn validate_record_batch(obj: &Bound<'_, PyAny>) -> PyResult<()> {
    let type_name = obj.get_type().name().map_err(|e| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "Cannot determine type of object passed to table.write(): {}",
            e
        ))
    })?;
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

/// # Safety
/// This function is unsafe because it interprets FFI arrays from C without bounds or type checking guarantees.
pub unsafe fn import_record_batch_from_c(
    array: FFI_ArrowArray,
    schema: &FFI_ArrowSchema,
) -> Result<RecordBatch, arrow::error::ArrowError> {
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
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))?;

    let mut batches = Vec::new();
    for batch_result in stream_reader {
        let batch = batch_result
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))?;
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

        let schema = arrow::datatypes::Schema::try_from(&ffi_schema).map_err(|e| {
            pyo3::exceptions::PyTypeError::new_err(format!("Arrow schema extraction failed: {}", e))
        })?;
        return Ok(Arc::new(schema));
    }

    Err(pyo3::exceptions::PyTypeError::new_err(
        "Expected benostreamdb.Schema or pyarrow.Schema object",
    ))
}

pub fn extract_partition_spec(
    spec_obj: Bound<'_, PyAny>,
) -> PyResult<crate::core::manifest::PartitionSpec> {
    let dict = spec_obj.downcast::<pyo3::types::PyDict>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err("partition_spec must be a dictionary")
    })?;

    let fields_obj = dict.get_item("fields")?.ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("partition_spec must contain 'fields'")
    })?;
    let fields_list = fields_obj
        .downcast::<pyo3::types::PyList>()
        .map_err(|_| pyo3::exceptions::PyTypeError::new_err("'fields' must be a list"))?;

    let mut fields = Vec::new();
    for item in fields_list {
        let f_dict = item.downcast::<pyo3::types::PyDict>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err("Each partition field must be a dictionary")
        })?;

        let name = f_dict
            .get_item("name")?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Missing 'name' in partition field")
            })?
            .extract::<String>()?;
        let transform = f_dict
            .get_item("transform")?
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Missing 'transform' in partition field")
            })?
            .extract::<String>()?;

        let source_id = f_dict
            .get_item("source_id")?
            .and_then(|i| i.extract::<i32>().ok());
        let field_id = f_dict
            .get_item("field_id")?
            .and_then(|i| i.extract::<i32>().ok());

        fields.push(crate::core::manifest::PartitionField {
            source_ids: source_id.map(|id| vec![id]).unwrap_or_default(),
            source_id,
            field_id,
            name,
            transform,
        });
    }

    Ok(crate::core::manifest::PartitionSpec { spec_id: 0, fields })
}

/// Names of every custom function the engine registers with DataFusion
/// (vector scalar UDFs, vector aggregates, JSON UDFs, graph UDAFs).
///
/// This is the core's function surface, consumed by
/// `tests/python/test_function_parity.py` to assert that the Python, dbt,
/// Trino, and Spark surfaces stay in sync as functions are added.
#[pyfunction]
pub fn registered_functions() -> Vec<String> {
    crate::core::sql::udf::registered_function_names()
}

/// Names of every graph traversal **table function** the engine registers
/// (`graph_neighbors`, `graph_shortest_path`, …). The counterpart to
/// [`registered_functions`] for the `FROM graph_*(...)` surface.
#[pyfunction]
pub fn registered_table_functions() -> Vec<String> {
    crate::core::sql::graph_udf::graph_table_function_names()
}

/// Every index algorithm name the engine understands. Source of truth for the
/// connector surface-parity test.
#[pyfunction]
pub fn registered_index_algorithms() -> Vec<String> {
    crate::core::manifest::IndexAlgorithm::all_names()
}

/// The DDL / maintenance statement kinds the engine intercepts. Source of truth
/// for the connector surface-parity test.
#[pyfunction]
pub fn registered_ddl_statements() -> Vec<String> {
    crate::core::sql::catalog_ddl::handled_ddl_statements()
}

/// The `ALTER TABLE ... EXECUTE <action>` procedure names the engine supports.
#[pyfunction]
pub fn registered_table_actions() -> Vec<String> {
    crate::core::sql::catalog_ddl::table_action_names()
}
