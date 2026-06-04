// Copyright (c) 2026 Richard Albright. All rights reserved.

use pyo3::prelude::*;
use std::sync::Arc;

use super::super::helpers::TOKIO_RUNTIME;
use super::super::schema::PySchema;
use super::super::table::PyTable;
use crate::core::catalog::glue::GlueCatalogClient;
use crate::core::catalog::Catalog;

/// Python wrapper for AWS Glue Catalog
#[pyclass]
pub struct PyGlueCatalog {
    client: Arc<GlueCatalogClient>,
}

#[pymethods]
impl PyGlueCatalog {
    #[new]
    #[pyo3(signature = (catalog_id=None))]
    pub(crate) fn new(catalog_id: Option<String>) -> PyResult<Self> {
        let client = TOKIO_RUNTIME
            .block_on(async { GlueCatalogClient::new(catalog_id).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))?;

        Ok(PyGlueCatalog {
            client: Arc::new(client),
        })
    }

    /// Create a new table
    fn create_table(
        &self,
        database: String,
        table_name: String,
        schema: PySchema,
        location: Option<String>,
    ) -> PyResult<()> {
        TOKIO_RUNTIME
            .block_on(async {
                self.client
                    .create_table(&database, &table_name, schema.inner, location.as_deref())
                    .await
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    /// Load a table (returns PyTable)
    fn load_table(&self, database: String, table_name: String) -> PyResult<PyTable> {
        let metadata = TOKIO_RUNTIME
            .block_on(async { self.client.load_table(&database, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))?;

        // Return a Table instance pointing to the location
        PyTable::new_internal(&metadata.location, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn table_exists(&self, database: String, table_name: String) -> PyResult<bool> {
        TOKIO_RUNTIME
            .block_on(async { self.client.table_exists(&database, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }
}
