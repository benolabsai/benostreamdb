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
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((format!("{:#}", e),)))?;

        arrow_batches_to_pyarrow(py, batches, schema)
    }

    /// List the graph tables (node/edge) registered in the session.
    ///
    /// Returns a list of dicts: `{name, table_type, source_column,
    /// target_column, id_column, label_column}`. This is the discovery primitive
    /// an MCP agent uses to find the graph without being told column names.
    #[pyo3(signature = (schema=None))]
    pub fn list_graph_tables(&self, py: Python<'_>, schema: Option<String>) -> PyResult<Py<PyAny>> {
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

    /// Returns the warehouse catalog policy as a Python dict, or None if no policy exists.
    pub fn catalog_policy(&self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let policy_opt = self
            .inner
            .catalog_policy()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        if let Some(policy) = policy_opt {
            let json_str = serde_json::to_string(&policy)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
            let json_mod = py.import("json")?;
            let dict = json_mod.call_method1("loads", (json_str,))?;
            Ok(Some(dict.into()))
        } else {
            Ok(None)
        }
    }

    /// Set read-only vs read-write access for a specific schema on the warehouse.
    pub fn set_schema_access(
        &self,
        py: Python<'_>,
        schema: String,
        read_only: bool,
    ) -> PyResult<Py<PyAny>> {
        let policy = self
            .inner
            .set_schema_access(&schema, read_only)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let json_str = serde_json::to_string(&policy)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let json_mod = py.import("json")?;
        let dict = json_mod.call_method1("loads", (json_str,))?;
        Ok(dict.into())
    }

    /// Locks down all schemas as read-only except scratchpad.
    pub fn lock_all_read_only(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let wh = self.inner.warehouse().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Warehouse is not set in session")
        })?;
        let policy = crate::core::catalog::lock_catalog_read_only(wh)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;

        let json_str = serde_json::to_string(&policy)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
        let json_mod = py.import("json")?;
        let dict = json_mod.call_method1("loads", (json_str,))?;
        Ok(dict.into())
    }
}
