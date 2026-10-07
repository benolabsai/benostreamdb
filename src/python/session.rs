// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use pyo3::prelude::*;
use std::sync::Arc;

use super::helpers::*;
use super::table::PyTable;

#[pyclass(name = "Session")]
pub struct PySession {
    inner: Arc<crate::core::sql::session::BenoStreamSession>,
}

#[pymethods]
impl PySession {
    #[new]
    #[pyo3(signature = (memory_mb=None, warehouse=None))]
    pub fn new(memory_mb: Option<usize>, warehouse: Option<String>) -> PyResult<Self> {
        let limit_bytes = memory_mb.map(|mb| mb * 1024 * 1024);
        let mut session = crate::core::sql::session::BenoStreamSession::new(limit_bytes);
        // A warehouse base location lets CREATE TABLE / CREATE TABLE AS SELECT
        // derive table URIs (`<warehouse>/<schema>/<table>`), as the Flight
        // server does via BSDB_WAREHOUSE.
        if let Some(w) = warehouse.filter(|w| !w.is_empty()) {
            session.set_warehouse(Some(w));
        }
        Ok(Self {
            inner: Arc::new(session),
        })
    }

    pub fn register(&self, name: String, table: &PyTable) -> PyResult<()> {
        self.inner
            .register_table(&name, Arc::new(table.table.clone()))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    pub fn sql(&self, py: Python<'_>, query: String) -> PyResult<Py<PyAny>> {
        let query = sanitize_sql(&query)?;
        let (batches, schema) = TOKIO_RUNTIME
            .block_on(self.inner.sql(&query))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err((format!("{:#}", e),))
            })?;

        arrow_batches_to_pyarrow(py, batches, schema)
    }
}
