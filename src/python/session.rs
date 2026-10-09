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

    /// List the graph tables (node/edge) registered in the session.
    ///
    /// Returns a list of dicts: `{name, table_type, source_column,
    /// target_column, id_column, label_column}`. This is the discovery primitive
    /// an MCP agent uses to find the graph without being told column names.
    #[pyo3(signature = (schema=None))]
    pub fn list_graph_tables(
        &self,
        py: Python<'_>,
        schema: Option<String>,
    ) -> PyResult<Py<PyAny>> {
        let infos = py
            .detach(|| TOKIO_RUNTIME.block_on(self.inner.list_graph_tables()))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let list = pyo3::types::PyList::empty(py);
        for info in infos {
            if let Some(ref want) = schema {
                // The name is `catalog.schema.table`; filter on the schema part.
                let parts: Vec<&str> = info.name.split('.').collect();
                if parts.len() < 2 || parts[parts.len() - 2] != want {
                    continue;
                }
            }
            let d = pyo3::types::PyDict::new(py);
            d.set_item("name", info.name)?;
            d.set_item("table_type", info.table_type)?;
            d.set_item("source_column", info.source_column)?;
            d.set_item("target_column", info.target_column)?;
            d.set_item("id_column", info.id_column)?;
            d.set_item("label_column", info.label_column)?;
            list.append(d)?;
        }
        Ok(list.into())
    }
}
