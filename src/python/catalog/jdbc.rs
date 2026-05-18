// Copyright (c) 2026 Richard Albright. All rights reserved.

use pyo3::prelude::*;
use std::sync::Arc;

use super::super::helpers::TOKIO_RUNTIME;
use super::super::table::PyTable;
use super::super::schema::PySchema;
use crate::core::catalog::jdbc::JdbcCatalogClient;
use crate::core::catalog::Catalog;

/// Python wrapper for JDBC Catalog
#[pyclass]
pub struct PyJdbcCatalog {
    client: Arc<JdbcCatalogClient>,
}

#[pymethods]
impl PyJdbcCatalog {
    #[new]
    #[pyo3(signature = (uri, warehouse=None, catalog_name=None))]
    pub(crate) fn new(uri: String, warehouse: Option<String>, catalog_name: Option<String>) -> PyResult<Self> {
        let catalog_name = catalog_name.unwrap_or_else(|| "default".to_string());
        let client = TOKIO_RUNTIME.block_on(async {
            JdbcCatalogClient::new(uri, warehouse, catalog_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;

        Ok(PyJdbcCatalog {
            client: Arc::new(client),
        })
    }

    /// Create a new table
    fn create_table(&self, namespace: String, table_name: String, schema: PySchema, location: Option<String>) -> PyResult<()> {
        TOKIO_RUNTIME.block_on(async {
            self.client.create_table(&namespace, &table_name, schema.inner, location.as_deref()).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    /// Load a table (returns PyTable)
    fn load_table(&self, namespace: String, table_name: String) -> PyResult<PyTable> {
        let metadata = TOKIO_RUNTIME.block_on(async {
            self.client.load_table(&namespace, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;

        PyTable::new_internal(&metadata.location, None).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }

    fn table_exists(&self, namespace: String, table_name: String) -> PyResult<bool> {
        TOKIO_RUNTIME.block_on(async {
            self.client.table_exists(&namespace, &table_name).await
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
    }
}
