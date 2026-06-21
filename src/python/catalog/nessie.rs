// Copyright (c) 2026 Richard Albright. All rights reserved.

use pyo3::prelude::*;
use std::sync::Arc;

use super::super::helpers::TOKIO_RUNTIME;
use super::super::schema::PySchema;
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
    #[pyo3(signature = (url))]
    pub(crate) fn new(url: String) -> PyResult<Self> {
        let client = Arc::new(NessieClient::new(url));
        Ok(PyNessieCatalog { client })
    }

    /// Create a new table
    fn create_table(
        &self,
        branch: String,
        table_name: String,
        schema: PySchema,
        location: Option<String>,
    ) -> PyResult<()> {
        TOKIO_RUNTIME
            .block_on(async {
                self.client
                    .create_table(&branch, &table_name, schema.inner, location.as_deref())
                    .await
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    /// Load a table (returns PyTable)
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
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

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
