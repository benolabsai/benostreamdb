// Copyright (c) 2026 Richard Albright. All rights reserved.

use pyo3::prelude::*;
use std::sync::Arc;

use super::helpers::*;
use super::table::PyTable;

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
        let query = sanitize_sql(&query)?;
        let (batches, schema) = TOKIO_RUNTIME.block_on(self.inner.sql(&query))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;

        arrow_batches_to_pyarrow(py, batches, schema)
    }
}
