import re
import os

path = "src/python_binding.rs"
with open(path, "r") as f:
    text = f.read()

# 1. to_arrow
repl1_old = """        let filter_str = filter.clone();
        let vs_params_clone = vs_params.clone();
        let columns_clone = columns.clone();

        // Determine which context to use: per-call context or table-level default"""
repl1_new = """        let filter_str = filter.clone();
        let vs_params_clone = vs_params.clone();
        let columns_clone = columns.clone();

        let table_schema = self.table.arrow_schema();
        let projected_schema = if let Some(cols) = &columns_clone {
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

        // Determine which context to use: per-call context or table-level default"""
text = text.replace(repl1_old, repl1_new)

text = text.replace("arrow_batches_to_pyarrow(py, batches)\n    }\n\n    /// Read table to Pandas DataFrame", "arrow_batches_to_pyarrow(py, batches, projected_schema)\n    }\n\n    /// Read table to Pandas DataFrame")

# 2. search_parallel
repl2_old = """        // Convert results to PyArrow tables
        match results {
            Ok(batch_vecs) => {
                let mut py_tables = Vec::new();
                for batches in batch_vecs {
                    // Convert RecordBatches to Py<PyAny> (PyArrow Table)
                    let py_table = arrow_batches_to_pyarrow(py, batches)?;"""
repl2_new = """        // Convert results to PyArrow tables
        match results {
            Ok(batch_vecs) => {
                let schema = self.table.arrow_schema();
                let mut py_tables = Vec::new();
                for batches in batch_vecs {
                    // Convert RecordBatches to Py<PyAny> (PyArrow Table)
                    let py_table = arrow_batches_to_pyarrow(py, batches, schema.clone())?;"""
text = text.replace(repl2_old, repl2_new)

# 3. PyTable::sql
repl3_old = """    fn sql(&self, py: Python<'_>, query: String) -> PyResult<Py<PyAny>> {
        let rt = self.table.runtime();
        
        let batch_result: Result<Vec<RecordBatch>, String> = rt.block_on(async {
            use datafusion::prelude::SessionContext;
            let ctx = SessionContext::new();
            
            // Register table as 't' (short alias, safe from keywords)
            let provider = std::sync::Arc::new(crate::core::sql::HyperStreamTableProvider::new(std::sync::Arc::new(self.table.clone())));
            ctx.register_table("t", provider).map_err(|e| e.to_string())?;
            
            // Execute
            let df = ctx.sql(&query).await.map_err(|e| e.to_string())?;
            let batches = df.collect().await.map_err(|e| e.to_string())?;
            Ok(batches)
        });
        
        match batch_result {
            Ok(batches) => arrow_batches_to_pyarrow(py, batches),"""
repl3_new = """    fn sql(&self, py: Python<'_>, query: String) -> PyResult<Py<PyAny>> {
        let rt = self.table.runtime();
        
        let batch_result: Result<(Vec<RecordBatch>, arrow::datatypes::SchemaRef), String> = rt.block_on(async {
            use datafusion::prelude::SessionContext;
            let ctx = SessionContext::new();
            
            // Register table as 't' (short alias, safe from keywords)
            let provider = std::sync::Arc::new(crate::core::sql::HyperStreamTableProvider::new(std::sync::Arc::new(self.table.clone())));
            ctx.register_table("t", provider).map_err(|e| e.to_string())?;
            
            // Execute
            let df = ctx.sql(&query).await.map_err(|e| e.to_string())?;
            let schema: arrow::datatypes::SchemaRef = std::sync::Arc::new(df.schema().into());
            let batches = df.collect().await.map_err(|e| e.to_string())?;
            Ok((batches, schema))
        });
        
        match batch_result {
            Ok((batches, schema)) => arrow_batches_to_pyarrow(py, batches, schema),"""
text = text.replace(repl3_old, repl3_new)

# 4. read_file
repl4_old = """    fn read_file(&self, py: Python<'_>, file_path: String, filter: Option<String>, columns: Option<Vec<String>>) -> PyResult<Py<PyAny>> {
        let batches = self.table.read_file(&file_path, columns, filter.as_deref())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
            
        arrow_batches_to_pyarrow(py, batches)
    }"""
repl4_new = """    fn read_file(&self, py: Python<'_>, file_path: String, filter: Option<String>, columns: Option<Vec<String>>) -> PyResult<Py<PyAny>> {
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
    }"""
text = text.replace(repl4_old, repl4_new)

# 5. read_split
repl5_old = """        let py_dict = split.bind(py);
        let file_path: String = py_dict.get_item("file_path")?.unwrap().extract()?;
        let file_size: u64 = py_dict.get_item("file_size_bytes")?.unwrap().extract()?;
        let start_offset: u64 = py_dict.get_item("start_offset")?.unwrap_or_else(|| split.py().None().bind(split.py()).clone()).extract().unwrap_or(0);
        let end_offset: u64 = py_dict.get_item("end_offset")?.unwrap_or_else(|| split.py().None().bind(split.py()).clone()).extract().unwrap_or(file_size);
        
        let rust_split = crate::core::storage::FileSplit {
            file_path,
            start_offset,
            length: end_offset - start_offset,
            file_size_bytes: file_size,
        };
        
        let batches = self.table.read_split(&rust_split, columns, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
            
        arrow_batches_to_pyarrow(py, batches)
    }"""
repl5_new = """        let table_schema = self.table.arrow_schema();
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

        let py_dict = split.bind(py);
        let file_path: String = py_dict.get_item("file_path")?.unwrap().extract()?;
        let file_size: u64 = py_dict.get_item("file_size_bytes")?.unwrap().extract()?;
        let start_offset: u64 = py_dict.get_item("start_offset")?.unwrap_or_else(|| split.py().None().bind(split.py()).clone()).extract().unwrap_or(0);
        let end_offset: u64 = py_dict.get_item("end_offset")?.unwrap_or_else(|| split.py().None().bind(split.py()).clone()).extract().unwrap_or(file_size);
        
        let rust_split = crate::core::storage::FileSplit {
            file_path,
            start_offset,
            length: end_offset - start_offset,
            file_size_bytes: file_size,
        };
        
        let batches = self.table.read_split(&rust_split, columns, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
            
        arrow_batches_to_pyarrow(py, batches, projected_schema)
    }"""
text = text.replace(repl5_old, repl5_new)

# 6. arrow_batches_to_pyarrow
repl6_old = """fn arrow_batches_to_pyarrow(py: Python<'_>, batches: Vec<RecordBatch>) -> PyResult<Py<PyAny>> {
    if batches.is_empty() {
        // Return empty table
        let pyarrow = py.import("pyarrow")?;
        let table_class = pyarrow.getattr("Table")?;
        let empty_list = pyo3::types::PyList::empty(py);
        return Ok(table_class.call_method1("from_pylist", (empty_list,))?.unbind());
    }

    // Use Arrow C Stream Interface for efficient transfer
    let schema = batches[0].schema();
    let batch_iter = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);"""
repl6_new = """fn arrow_batches_to_pyarrow(py: Python<'_>, batches: Vec<RecordBatch>, schema: arrow::datatypes::SchemaRef) -> PyResult<Py<PyAny>> {
    // Use Arrow C Stream Interface for efficient transfer
    let batch_iter = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);"""
text = text.replace(repl6_old, repl6_new)

# 7. PySession::sql
repl7_old = """    pub fn sql(&self, py: Python<'_>, query: String) -> PyResult<Py<PyAny>> {
        let batches = self.rt.block_on(self.inner.sql(&query))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        arrow_batches_to_pyarrow(py, batches)
    }"""
repl7_new = """    pub fn sql(&self, py: Python<'_>, query: String) -> PyResult<Py<PyAny>> {
        let (batches, schema) = self.rt.block_on(self.inner.sql(&query))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;
        
        arrow_batches_to_pyarrow(py, batches, schema)
    }"""
text = text.replace(repl7_old, repl7_new)

with open(path, "w") as f:
    f.write(text)

# Also update src/core/sql/session.rs
session_path = "src/core/sql/session.rs"
with open(session_path, "r") as f:
    session_text = f.read()

sess_repl_old = """    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>> {
        // Parse the SQL query and execute directly
        // Note: pgvector rewriter temporarily disabled due to schema mismatch issues
        let df = self.ctx.sql(query).await?;
        let batches = df.collect().await?;
        Ok(batches)
    }"""
sess_repl_new = """    pub async fn sql(&self, query: &str) -> Result<(Vec<RecordBatch>, arrow::datatypes::SchemaRef)> {
        // Parse the SQL query and execute directly
        // Note: pgvector rewriter temporarily disabled due to schema mismatch issues
        let df = self.ctx.sql(query).await?;
        let schema: arrow::datatypes::SchemaRef = std::sync::Arc::new(df.schema().into());
        let batches = df.collect().await?;
        Ok((batches, schema))
    }"""
session_text = session_text.replace(sess_repl_old, sess_repl_new)

with open(session_path, "w") as f:
    f.write(session_text)

