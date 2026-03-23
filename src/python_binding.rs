use pyo3::prelude::*;
use pyo3::types::PyDict;
use crate::core::table::{Table, VectorSearchParams};
use crate::core::compaction::CompactionOptions;
use arrow::record_batch::RecordBatch;
use arrow::array::RecordBatchIterator;
use arrow::ffi_stream::{FFI_ArrowArrayStream, ArrowArrayStreamReader};
use pyo3::ffi::Py_uintptr_t;
use crate::core::catalog::{Catalog, nessie::NessieClient, rest::RestCatalogClient, glue::GlueCatalogClient, hive::HiveMetastoreClient, unity::UnityCatalogClient, jdbc::JdbcCatalogClient, CatalogConfig, CatalogType};
use tokio::runtime::Runtime;
use std::sync::Arc;
use arrow::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};

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
    fn vector(dim: usize) -> Self {
        Self {
            dt: DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
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
    #[pyo3(signature = (name, data_type, nullable=true))]
    fn new(name: String, data_type: PyDataType, nullable: bool) -> Self {
        Self {
            inner: Field::new(name, data_type.dt, nullable),
        }
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

impl PySchema {
    pub fn new_internal(fields: Vec<PyField>) -> Self {
        let arrow_fields: Vec<Field> = fields.into_iter().map(|f| f.inner).collect();
        Self {
            inner: Arc::new(Schema::new(arrow_fields)),
        }
    }
}

#[pymethods]
impl PySchema {
    #[new]
    fn new(fields: Vec<PyField>) -> Self {
        Self::new_internal(fields)
    }

    fn __repr__(&self) -> String {
        format!("Schema(fields={:?})", self.inner.fields())
    }
}

/// High-level Table API - Pandas-compatible interface
/// This is a thin Python wrapper around the core Rust Table struct
#[pyclass(name = "Table")]
pub struct PyTable {
    table: Table,
    /// Dedicated runtime for parallel query execution (bypasses Python GIL)
    query_pool: Arc<Runtime>,
}

impl PyTable {
    pub fn new_internal(uri: &str) -> Result<Self, anyhow::Error> {
        let table = Table::new(uri.to_string())?;
        // Create dedicated runtime for parallel query execution
        // This runtime is separate from the table's internal runtime
        // and allows us to run multiple queries in parallel, bypassing Python GIL
        let query_pool = Arc::new(
            Runtime::new()
                .map_err(|e| anyhow::anyhow!("Failed to create query pool runtime: {}", e))?
        );
        Ok(PyTable { table, query_pool })
    }

    pub fn create_internal(uri: &str, schema: arrow::datatypes::SchemaRef) -> Result<Self, anyhow::Error> {
        let table = Table::create(uri.to_string(), schema)?;
        // Create dedicated runtime for parallel query execution
        let query_pool = Arc::new(
            Runtime::new()
                .map_err(|e| anyhow::anyhow!("Failed to create query pool runtime: {}", e))?
        );
        Ok(PyTable { table, query_pool })
    }
}

#[pymethods]
#[allow(deprecated)]
impl PyTable {
    #[new]
    #[pyo3(signature = (uri))]
    fn new(uri: &str) -> PyResult<Self> {
        Self::new_internal(uri)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Create a new table with an explicit schema
    #[staticmethod]
    #[pyo3(signature = (uri, schema))]
    fn create(uri: &str, schema: PySchema) -> PyResult<Self> {
        Self::create_internal(uri, schema.inner)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Register an existing Iceberg table
    #[staticmethod]
    #[pyo3(signature = (uri, iceberg_metadata_uri))]
    fn register_external(uri: &str, iceberg_metadata_uri: &str) -> PyResult<Self> {
        let rt = Runtime::new()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to create runtime: {}", e)))?;
        
        let table = rt.block_on(Table::register_external(uri.to_string(), iceberg_metadata_uri))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            
        let query_pool = Arc::new(rt);
        Ok(PyTable { table, query_pool })
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
            self.query_pool.block_on(self.table.spawn_iceberg_observer(
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

    /// Add columns to be indexed (triggers backfill)
    fn add_index_columns(&mut self, columns: Vec<String>) -> PyResult<()> {
        self.table.add_index_columns(columns)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Register a Python-based embedding function into the Rust core.
    /// This allows the Rust core to trigger vectorization even without Python GIL
    /// by re-acquiring it only during the callback.
    fn register_python_embedding(&self, py: Python<'_>, name: String, dim: usize, callback: PyObject) -> PyResult<()> {
        let callback_clone = callback.clone_ref(py);
        let wrapper = move |texts: Vec<String>| -> Result<Vec<Vec<f32>>> {
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
    ///     vector_filter: Optional dict with {"column": str, "query": list, "k": int}
    ///     columns: Optional list of column names to read (skips others like embeddings)
    /// 
    /// Returns:
    ///     PyArrow Table (via Arrow C Data Interface)
    /// 
    /// Example:
    ///     # Skip reading the 'embedding' column for faster scalar queries
    ///     df = table.to_pandas(filter="category = 'science'", columns=["doc_id", "title", "category"])
    #[pyo3(signature = (filter=None, vector_filter=None, columns=None))]
    fn to_arrow(
        &self,
        py: Python<'_>,
        filter: Option<String>,
        vector_filter: Option<Bound<'_, PyDict>>,
        columns: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        let _pyarrow = py.import("pyarrow")?;
        // Parse vector filter if provided
        let vs_params = if let Some(ref vf) = vector_filter {
            let column: String = vf.get_item("column")?.unwrap().extract()?;
            let k: usize = vf.get_item("k")?.unwrap().extract()?;
            let query_obj = vf.get_item("query")?.unwrap();
            let query: Vec<f32> = query_obj.extract()?;
            Some(VectorSearchParams::new(&column, crate::core::index::VectorValue::Float32(query), k))
        } else {
            None
        };

        // Clone for closure
        let filter_str = filter.clone();
        let vs_params_clone = vs_params.clone();
        let columns_clone = columns.clone();

        // Call core Table API with column projection, releasing the GIL
        let batches = py.allow_threads(move || {
            if let Some(cols) = columns_clone {
                 self.table.read_with_columns(filter_str.as_deref(), vs_params_clone, cols)
            } else {
                 self.table.read(filter_str.as_deref(), vs_params_clone)
            }
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        // Convert to Arrow C Data Interface
        arrow_batches_to_pyarrow(py, batches)
    }

    /// Read table to Pandas DataFrame with optional filtering and column projection
    /// 
    /// This method pushes predicates down to Rust for index acceleration.
    /// Use the `columns` parameter to skip reading large columns like embeddings
    /// when you only need scalar data.
    /// 
    /// Example:
    ///     # Fast scalar query - skip reading 768D embeddings
    ///     df = table.to_pandas(filter="category = 'science'", columns=["doc_id", "title"])
    #[pyo3(signature = (filter=None, vector_filter=None, columns=None))]
    fn to_pandas(
        &self,
        py: Python<'_>,
        filter: Option<String>,
        vector_filter: Option<Bound<'_, PyDict>>,
        columns: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        // Get Arrow table with column projection
        let arrow_table = self.to_arrow(py, filter, vector_filter, columns)?;
        
        // Convert to Pandas via PyArrow
        // Python: arrow_table.to_pandas()
        let _pyarrow = py.import("pyarrow")?;
        arrow_table.call_method0(py, "to_pandas")
    }

    /// Write data to table.
    /// Supports:
    /// - List of PyArrow RecordBatches
    /// - PyArrow Table
    /// - Pandas DataFrame
    #[pyo3(signature = (data))]
    fn write(&self, py: Python<'_>, data: Bound<'_, PyAny>) -> PyResult<()> {
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
             py.allow_threads(|| {
                self.table.write(batches)
             }).map_err(|e: anyhow::Error| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
             
        } else {
             // Fallback to Table/DataFrame handling
             let obj: Py<PyAny> = data.unbind();
             
             // Check if it's a DataFrame (has "values" attr? or check type?)
             // Simple heuristic: try write_arrow, if that fails, maybe pandas?
             // But write_arrow expects object that converts to reader.
             // write_pandas expects object that converts to Table via from_pandas.
             
             // Let's try arrow first as it's lighter
             // Use clone_ref(py) for Py<T>
             if let Ok(_) = self.write_arrow(py, obj.clone_ref(py)) {
                 Ok(())
             } else {
                 // Try pandas
                 self.write_pandas(py, obj)
             }
        }
    }

    #[getter]
    fn index_columns(&self) -> Vec<String> {
        self.table.get_index_columns()
    }

    #[setter]
    fn set_index_columns(&mut self, columns: Vec<String>) -> PyResult<()> {
        self.table.remove_all_index_columns();
        self.table.add_index_columns(columns)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }


    /// Read table to Pandas DataFrame (Alias for to_pandas)
    #[pyo3(signature = (filter=None, columns=None))]
    fn read(
        &self,
        py: Python<'_>,
        filter: Option<String>,
        columns: Option<Vec<String>>,
    ) -> PyResult<Py<PyAny>> {
        self.to_pandas(py, filter, None, columns)
    }

    /// Vector search on the table
    #[pyo3(signature = (column, query, k=10, filter=None))]
    fn search(
        &self,
        py: Python<'_>,
        column: String,
        query: Vec<f32>,
        k: usize,
        filter: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let vf_dict = PyDict::new(py);
        vf_dict.set_item("column", column)?;
        vf_dict.set_item("query", query)?;
        vf_dict.set_item("k", k)?;
        
        self.to_pandas(py, filter, Some(vf_dict), None)
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
        let query_pool = self.query_pool.clone();
        
        // Release GIL and run all queries in parallel in Rust
        let results: Result<Vec<Vec<RecordBatch>>, anyhow::Error> = py.allow_threads(move || {
            query_pool.block_on(async {
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
                let mut py_tables = Vec::new();
                for batches in batch_vecs {
                    // Convert RecordBatches to Py<PyAny> (PyArrow Table)
                    let py_table = arrow_batches_to_pyarrow(py, batches)?;
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
            self.query_pool.block_on(self.table.wait_for_background_tasks_async())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }

    /// Write Pandas DataFrame to table
    fn write_pandas(&self, py: Python<'_>, df: Py<PyAny>) -> PyResult<()> {
        // Convert Pandas -> PyArrow -> Rust Arrow
        let pyarrow = py.import("pyarrow")?;
        let table_class = pyarrow.getattr("Table")?;
        let arrow_table = table_class.call_method1("from_pandas", (df,))?.unbind();
        self.write_arrow(py, arrow_table)
    }

    /// Write PyArrow Table to table
    fn write_arrow(&self, py: Python<'_>, table: Py<PyAny>) -> PyResult<()> {
        // Convert PyArrow Table to RecordBatches
        let batches = pyarrow_to_arrow_batches(py, table)?;
        
        // Call core Table API, releasing the GIL
        py.allow_threads(|| {
            self.table.write(batches)
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Commit buffered writes to disk (flush)
    fn commit(&self, py: Python<'_>) -> PyResult<()> {
        // Release GIL during commit to allow other Python threads to run
        py.allow_threads(|| {
            self.table.commit()
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        // Ensure all background indexing tasks complete before returning to Python
        self.wait_for_background_tasks(py)
    }
    /// Wait for any background indexing or maintenance tasks to complete.
    fn wait_for_indexes(&self, py: Python<'_>) -> PyResult<()> {
        self.wait_for_background_tasks(py)
    }

    /// Async commit
    fn commit_async(&self, py: Python<'_>) -> PyResult<()> {
        py.allow_threads(|| {
            self.query_pool.block_on(self.table.commit_async())
        })
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        self.wait_for_background_tasks(py)
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
        self.table.delete(&filter)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }



    /// Merge Pandas DataFrame into the table (Upsert)
    /// 
    /// Args:
    ///     df: Pandas DataFrame
    ///     key_column: Column name to merge on
    ///     mode: Optional PyMergeMode (MergeOnRead or MergeOnWrite)
    #[pyo3(signature = (df, key_column, mode=None))]
    fn merge_pandas(
        &self,
        py: Python<'_>,
        df: Py<PyAny>,
        key_column: String,
        mode: Option<PyMergeMode>,
    ) -> PyResult<()> {
        // 1. Convert Pandas -> Arrow RecordBatch
        let pyarrow = py.import("pyarrow")?;
        let table_class = pyarrow.getattr("Table")?;
        let arrow_table = table_class.call_method1("from_pandas", (df,))?.unbind();
        let batches = pyarrow_to_arrow_batches(py, arrow_table)?;
        
        // 2. Call core Table API
        let merge_mode = mode.unwrap_or(PyMergeMode::MergeOnRead).into();
        self.table.merge(batches, &key_column, merge_mode)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
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
            self.query_pool.block_on(self.table.update_spec(&rust_fields))
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Update table schema (Evolution)
    fn update_schema(&self, py: Python<'_>, schema: PySchema) -> PyResult<()> {
        let hdb_schema = crate::core::manifest::Schema::from_arrow(&schema.inner, 1);
        py.allow_threads(|| {
            self.query_pool.block_on(self.table.update_schema(hdb_schema))
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Rollback to a specific snapshot
    fn rollback_to_snapshot(&self, py: Python<'_>, snapshot_id: i64) -> PyResult<()> {
        py.allow_threads(|| {
            self.query_pool.block_on(self.table.rollback_to_snapshot(snapshot_id))
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Execute SQL query against the table.
    /// The table is registered as 't'.
    /// 
    /// Example:
    ///     table.sql("SELECT * FROM t WHERE id > 10")
    fn sql(&self, py: Python<'_>, query: String) -> PyResult<Py<PyAny>> {
        let rt = self.table.runtime();
        
        let batch_result: Result<Vec<RecordBatch>, String> = rt.block_on(async {
            use datafusion::prelude::SessionContext;
            let ctx = SessionContext::new();
            
            // Register table as 't' (short alias, safe from keywords)
            let provider = Arc::new(crate::core::sql::HyperStreamTableProvider::new(Arc::new(self.table.clone())));
            ctx.register_table("t", provider).map_err(|e| e.to_string())?;
            
            // Execute
            let df = ctx.sql(&query).await.map_err(|e| e.to_string())?;
            let batches = df.collect().await.map_err(|e| e.to_string())?;
            Ok(batches)
        });
        
        match batch_result {
            Ok(batches) => arrow_batches_to_pyarrow(py, batches),
            Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(e)),
        }
    }

    fn manifest(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Load manifest info from table
        let rt = self.table.runtime();
        let manifest_result: Result<(crate::core::manifest::Manifest, u64), anyhow::Error> = rt.block_on(async {
            self.table.get_snapshot_segments_with_version().await
        });
        
        match manifest_result {
            Ok((manifest, version)) => {
                let dict = PyDict::new(py);
                
                // Basic info
                dict.set_item("version", version)?;
                dict.set_item("timestamp_ms", manifest.timestamp_ms)?;
                dict.set_item("current_schema_id", manifest.current_schema_id)?;
                dict.set_item("partition_spec_id", manifest.partition_spec.spec_id)?;
                dict.set_item("entries_count", manifest.entries.len())?;
                
                // Data files summary
                let total_rows: i64 = manifest.entries.iter().map(|e| e.record_count).sum();
                let total_bytes: i64 = manifest.entries.iter().map(|e| e.file_size_bytes).sum();
                dict.set_item("total_rows", total_rows)?;
                dict.set_item("total_bytes", total_bytes)?;
                
                // List of data files
                let files_list = pyo3::types::PyList::empty(py);
                for entry in &manifest.entries {
                    let file_dict = PyDict::new(py);
                    file_dict.set_item("file_path", &entry.file_path)?;
                    file_dict.set_item("file_size_bytes", entry.file_size_bytes)?;
                    file_dict.set_item("record_count", entry.record_count)?;
                    file_dict.set_item("index_files_count", entry.index_files.len())?;
                    file_dict.set_item("delete_files_count", entry.delete_files.len())?;
                    files_list.append(file_dict)?;
                }
                dict.set_item("files", files_list)?;
                
                // Properties
                let props_dict = PyDict::new(py);
                for (k, v) in &manifest.properties {
                    props_dict.set_item(k, v)?;
                }
                dict.set_item("properties", props_dict)?;
                
                Ok(dict.unbind().into())
            }
            Err(_) => {
                // Empty manifest for new/empty tables
                let dict = PyDict::new(py);
                dict.set_item("version", 0)?;
                dict.set_item("entries_count", 0)?;
                dict.set_item("total_rows", 0)?;
                dict.set_item("total_bytes", 0)?;
                dict.set_item("files", pyo3::types::PyList::empty(py))?;
                Ok(dict.unbind().into())
            }
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
        let batches = self.table.read_file(&file_path, columns, filter.as_deref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
            
        arrow_batches_to_pyarrow(py, batches)
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
            
        arrow_batches_to_pyarrow(py, batches)
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
}

/// Python wrapper for Nessie Catalog (Iceberg-compatible)
#[pyclass]
pub struct PyNessieCatalog {
    client: Arc<NessieClient>,
    rt: Arc<Runtime>,
}

#[pymethods]
impl PyNessieCatalog {
    #[new]
    fn new(url: String) -> PyResult<Self> {
        let client = Arc::new(NessieClient::new(url));
        let rt = Arc::new(Runtime::new().unwrap());
        Ok(PyNessieCatalog { client, rt })
    }

    /// Create a new table
    fn create_table(&self, branch: String, table_name: String, schema: PySchema, location: Option<String>) -> PyResult<()> {
        self.rt.block_on(async {
            self.client.create_table(&branch, &table_name, schema.inner, location.as_deref()).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    /// Load a table (returns PyTable)
    fn load_table(&self, branch: String, table_name: String) -> PyResult<PyTable> {
        let metadata = self.rt.block_on(async {
            self.client.load_table(&branch, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        // Return a Table instance pointing to the location
        // Note: Table::new expects the root path (uri). 
        // If metadata.location is full path to `metadata.json`, we might need to adjust.
        // Assuming metadata.location is the table root for now, or we need to parse it.
        // Standard Iceberg: metadata_location is path to specific json file.
        // HyperStream Table::new takes a root URI. 
        // We'll pass the location directly.
        PyTable::new_internal(&metadata.location).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    fn create_branch(&self, branch_name: String, source_ref: Option<String>) -> PyResult<()> {
        self.rt.block_on(async {
            self.client.create_branch(&branch_name, source_ref.as_deref()).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    fn table_exists(&self, branch: String, table_name: String) -> PyResult<bool> {
        self.rt.block_on(async {
            self.client.table_exists(&branch, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
}

/// Python wrapper for REST Catalog (Iceberg-compatible)
#[pyclass]
pub struct PyRestCatalog {
    client: Arc<RestCatalogClient>,
    rt: Arc<Runtime>,
}

#[pymethods]
impl PyRestCatalog {
    #[new]
    #[pyo3(signature = (url, prefix=None))]
    fn new(url: String, prefix: Option<String>) -> PyResult<Self> {
        let client = Arc::new(RestCatalogClient::new(url, prefix));
        let rt = Arc::new(Runtime::new().unwrap());
        Ok(PyRestCatalog { client, rt })
    }

    /// Create a new table
    fn create_table(&self, namespace: String, table_name: String, schema: PySchema, location: Option<String>) -> PyResult<()> {
        self.rt.block_on(async {
            self.client.create_table(&namespace, &table_name, schema.inner, location.as_deref()).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    /// Load a table (returns PyTable)
    fn load_table(&self, namespace: String, table_name: String) -> PyResult<PyTable> {
        let metadata = self.rt.block_on(async {
            self.client.load_table(&namespace, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        // Return a Table instance pointing to the location
        PyTable::new_internal(&metadata.location).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    fn table_exists(&self, namespace: String, table_name: String) -> PyResult<bool> {
        self.rt.block_on(async {
            self.client.table_exists(&namespace, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
}

/// Python wrapper for AWS Glue Catalog
#[pyclass]
pub struct PyGlueCatalog {
    client: Arc<GlueCatalogClient>,
    rt: Arc<Runtime>,
}

#[pymethods]
impl PyGlueCatalog {
    #[new]
    #[pyo3(signature = (catalog_id=None))]
    fn new(catalog_id: Option<String>) -> PyResult<Self> {
        let rt = Arc::new(Runtime::new().unwrap());
        let client = rt.block_on(async {
            GlueCatalogClient::new(catalog_id).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        Ok(PyGlueCatalog { 
            client: Arc::new(client),
            rt 
        })
    }

    /// Create a new table
    fn create_table(&self, database: String, table_name: String, schema: PySchema, location: Option<String>) -> PyResult<()> {
        self.rt.block_on(async {
            self.client.create_table(&database, &table_name, schema.inner, location.as_deref()).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    /// Load a table (returns PyTable)
    fn load_table(&self, database: String, table_name: String) -> PyResult<PyTable> {
        let metadata = self.rt.block_on(async {
            self.client.load_table(&database, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        // Return a Table instance pointing to the location
        PyTable::new_internal(&metadata.location).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    fn table_exists(&self, database: String, table_name: String) -> PyResult<bool> {
        self.rt.block_on(async {
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
    rt: Arc<Runtime>,
}

#[pymethods]
impl PyHiveCatalog {
    #[new]
    fn new(url: String) -> PyResult<Self> {
        let rt = Arc::new(Runtime::new().unwrap());
        let client = rt.block_on(async {
            HiveMetastoreClient::new(url)
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        let client = Arc::new(client);
        Ok(PyHiveCatalog { client, rt })
    }

    /// Create a new table
    fn create_table(&self, database: String, table_name: String, schema: PySchema, location: Option<String>) -> PyResult<()> {
        self.rt.block_on(async {
            self.client.create_table(&database, &table_name, schema.inner, location.as_deref()).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    /// Load a table (placeholder - returns informative error)
    fn load_table(&self, database: String, table_name: String) -> PyResult<PyTable> {
        let metadata = self.rt.block_on(async {
            self.client.load_table(&database, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        PyTable::new_internal(&metadata.location).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    fn table_exists(&self, database: String, table_name: String) -> PyResult<bool> {
        self.rt.block_on(async {
            self.client.table_exists(&database, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
}

/// Python wrapper for Unity Catalog (Databricks)
#[pyclass]
pub struct PyUnityCatalog {
    client: Arc<UnityCatalogClient>,
    rt: Arc<Runtime>,
}

#[pymethods]
impl PyUnityCatalog {
    #[new]
    fn new(url: String, token: String) -> PyResult<Self> {
        let client = Arc::new(UnityCatalogClient::new(url, token));
        let rt = Arc::new(Runtime::new().unwrap());
        Ok(PyUnityCatalog { client, rt })
    }

    /// Create a new table
    fn create_table(&self, catalog: String, table_name: String, schema: PySchema, location: Option<String>) -> PyResult<()> {
        self.rt.block_on(async {
            self.client.create_table(&catalog, &table_name, schema.inner, location.as_deref()).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    /// Load a table (returns PyTable)
    fn load_table(&self, catalog: String, table_name: String) -> PyResult<PyTable> {
        let metadata = self.rt.block_on(async {
            self.client.load_table(&catalog, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        PyTable::new_internal(&metadata.location).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    fn table_exists(&self, catalog: String, table_name: String) -> PyResult<bool> {
        self.rt.block_on(async {
            self.client.table_exists(&catalog, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
}

/// Python wrapper for JDBC Catalog
#[pyclass]
pub struct PyJdbcCatalog {
    client: Arc<JdbcCatalogClient>,
    rt: Arc<Runtime>,
}

#[pymethods]
impl PyJdbcCatalog {
    #[new]
    #[pyo3(signature = (uri, warehouse=None, catalog_name=None))]
    fn new(uri: String, warehouse: Option<String>, catalog_name: Option<String>) -> PyResult<Self> {
        let rt = Arc::new(Runtime::new().unwrap());
        let catalog_name = catalog_name.unwrap_or_else(|| "default".to_string());
        let client = rt.block_on(async {
            JdbcCatalogClient::new(uri, warehouse, catalog_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        Ok(PyJdbcCatalog { 
            client: Arc::new(client),
            rt 
        })
    }

    /// Create a new table
    fn create_table(&self, namespace: String, table_name: String, schema: PySchema, location: Option<String>) -> PyResult<()> {
        self.rt.block_on(async {
            self.client.create_table(&namespace, &table_name, schema.inner, location.as_deref()).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    /// Load a table (returns PyTable)
    fn load_table(&self, namespace: String, table_name: String) -> PyResult<PyTable> {
        let metadata = self.rt.block_on(async {
            self.client.load_table(&namespace, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        PyTable::new_internal(&metadata.location).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
    
    fn table_exists(&self, namespace: String, table_name: String) -> PyResult<bool> {
        self.rt.block_on(async {
            self.client.table_exists(&namespace, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
}

// Helper functions for Arrow C Data Interface

fn arrow_batches_to_pyarrow(py: Python<'_>, batches: Vec<RecordBatch>) -> PyResult<Py<PyAny>> {
    if batches.is_empty() {
        // Return empty table
        let pyarrow = py.import("pyarrow")?;
        let table_class = pyarrow.getattr("Table")?;
        let empty_list = pyo3::types::PyList::empty(py);
        return Ok(table_class.call_method1("from_pylist", (empty_list,))?.unbind());
    }

    // Use Arrow C Stream Interface for efficient transfer
    let schema = batches[0].schema();
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

#[pyclass(name = "Session")]
pub struct PySession {
    inner: Arc<crate::core::sql::session::HyperStreamSession>,
    rt: Arc<Runtime>,
}

#[pymethods]
impl PySession {
    #[new]
    #[pyo3(signature = (memory_mb=None))]
    pub fn new(memory_mb: Option<usize>) -> PyResult<Self> {
        let limit_bytes = memory_mb.map(|mb| mb * 1024 * 1024);
        Ok(Self {
            inner: Arc::new(crate::core::sql::session::HyperStreamSession::new(limit_bytes)),
            rt: Arc::new(Runtime::new()?),
        })
    }

    pub fn register(&self, name: String, table: &PyTable) -> PyResult<()> {
        self.inner.register_table(&name, Arc::new(table.table.clone()))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    pub fn sql(&self, py: Python<'_>, query: String) -> PyResult<Py<PyAny>> {
        let batches = self.rt.block_on(self.inner.sql(&query))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        arrow_batches_to_pyarrow(py, batches)
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
pub fn open_table(_py: Python<'_>, uri: &str) -> PyResult<PyTable> {
    PyTable::new_internal(uri).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
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
