// Copyright (c) 2026 Richard Albright. All rights reserved.

use pyo3::prelude::*;
use std::sync::Arc;

use super::super::helpers::TOKIO_RUNTIME;
use super::super::schema::PySchema;
use super::super::table::PyTable;
use crate::core::catalog::rest::RestCatalogClient;
use crate::core::catalog::Catalog;

/// Python wrapper for REST Catalog (Iceberg-compatible)
#[pyclass]
pub struct PyRestCatalog {
    client: Arc<RestCatalogClient>,
}

#[pymethods]
impl PyRestCatalog {
    #[new]
    #[pyo3(signature = (url, prefix=None))]
    pub(crate) fn new(url: String, prefix: Option<String>) -> PyResult<Self> {
        let client = Arc::new(RestCatalogClient::new(url, prefix));
        Ok(PyRestCatalog { client })
    }

    /// Create a new table
    fn create_table(
        &self,
        namespace: String,
        table_name: String,
        schema: PySchema,
        location: Option<String>,
    ) -> PyResult<()> {
        TOKIO_RUNTIME
            .block_on(async {
                self.client
                    .create_table(&namespace, &table_name, schema.inner, location.as_deref())
                    .await
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    /// Load a table (returns PyTable)
    fn load_table(&self, namespace: String, table_name: String) -> PyResult<PyTable> {
        let metadata = TOKIO_RUNTIME
            .block_on(async { self.client.load_table(&namespace, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))?;

        // Return a Table instance pointing to the location
        PyTable::new_internal(&metadata.location, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn table_exists(&self, namespace: String, table_name: String) -> PyResult<bool> {
        TOKIO_RUNTIME
            .block_on(async { self.client.table_exists(&namespace, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }
}
