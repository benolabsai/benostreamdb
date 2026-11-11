// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use pyo3::prelude::*;
use std::sync::Arc;

use super::super::helpers::{extract_schema, TOKIO_RUNTIME};
use super::super::table::PyTable;
use crate::core::catalog::nessie::NessieClient;
use crate::core::catalog::Catalog;

/// Python wrapper for Nessie Catalog (Iceberg-compatible)
#[pyclass]
pub struct PyNessieCatalog {
    client: Arc<NessieClient>,
}

#[pymethods]
impl PyNessieCatalog {
    #[new]
    #[pyo3(signature = (url, token=None))]
    pub(crate) fn new(url: String, token: Option<String>) -> PyResult<Self> {
        let client = Arc::new(NessieClient::with_token(url, token));
        Ok(PyNessieCatalog { client })
    }

    fn create_namespace(&self, branch: String) -> PyResult<()> {
        TOKIO_RUNTIME
            .block_on(async { self.client.create_namespace(&branch).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn namespace_exists(&self, branch: String) -> PyResult<bool> {
        TOKIO_RUNTIME
            .block_on(async { self.client.namespace_exists(&branch).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn drop_namespace(&self, branch: String) -> PyResult<()> {
        TOKIO_RUNTIME
            .block_on(async { self.client.drop_namespace(&branch).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    /// Create a new table
    #[pyo3(signature = (branch, table_name, schema, location=None))]
    fn create_table(
        &self,
        branch: String,
        table_name: String,
        schema: Bound<'_, PyAny>,
        location: Option<String>,
    ) -> PyResult<()> {
        let arrow_schema = extract_schema(schema)?;
        TOKIO_RUNTIME
            .block_on(async {
                self.client
                    .create_table(&branch, &table_name, arrow_schema, location.as_deref())
                    .await
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn drop_table(&self, branch: String, table_name: String) -> PyResult<()> {
        TOKIO_RUNTIME
            .block_on(async { self.client.drop_table(&branch, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn list_tables(&self, branch: String) -> PyResult<Vec<String>> {
        TOKIO_RUNTIME
            .block_on(async { self.client.list_tables(&branch).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    /// Load a table (returns catalog-backed PyTable)
    fn load_table(&self, branch: String, table_name: String) -> PyResult<PyTable> {
        let metadata = TOKIO_RUNTIME
            .block_on(async { self.client.load_table(&branch, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))?;

        PyTable::load_from_catalog(
            &metadata.location,
            self.client.clone(),
            &branch,
            &table_name,
            None,
            None,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    #[pyo3(signature = (branch_name, source_ref=None))]
    fn create_branch(&self, branch_name: String, source_ref: Option<String>) -> PyResult<()> {
        TOKIO_RUNTIME
            .block_on(async {
                self.client
                    .create_branch(&branch_name, source_ref.as_deref())
                    .await
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn table_exists(&self, branch: String, table_name: String) -> PyResult<bool> {
        TOKIO_RUNTIME
            .block_on(async { self.client.table_exists(&branch, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }
}
