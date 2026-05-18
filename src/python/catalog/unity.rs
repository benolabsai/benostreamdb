// Copyright (c) 2026 Richard Albright. All rights reserved.

use pyo3::prelude::*;
use std::sync::Arc;

use super::super::helpers::TOKIO_RUNTIME;
use super::super::table::PyTable;
use super::super::schema::PySchema;
use crate::core::catalog::unity::UnityCatalogClient;
use crate::core::catalog::Catalog;

/// Python wrapper for Unity Catalog (Databricks)
#[pyclass]
pub struct PyUnityCatalog {
    client: Arc<UnityCatalogClient>,
}

#[pymethods]
impl PyUnityCatalog {
    #[new]
    #[pyo3(signature = (url, token))]
    pub(crate) fn new(url: String, token: String) -> PyResult<Self> {
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
