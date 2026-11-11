// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use pyo3::prelude::*;
use std::sync::Arc;

use super::super::helpers::{extract_schema, TOKIO_RUNTIME};
use super::super::table::PyTable;
use crate::core::catalog::hive::HiveMetastoreClient;
use crate::core::catalog::Catalog;

/// Python wrapper for Hive Metastore Catalog
#[pyclass]
pub struct PyHiveCatalog {
    client: Arc<HiveMetastoreClient>,
}

#[pymethods]
impl PyHiveCatalog {
    #[new]
    #[pyo3(signature = (url))]
    pub(crate) fn new(url: String) -> PyResult<Self> {
        let client = TOKIO_RUNTIME
            .block_on(async { HiveMetastoreClient::new(url) })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))?;
        let client = Arc::new(client);
        Ok(PyHiveCatalog { client })
    }

    fn create_namespace(&self, database: String) -> PyResult<()> {
        TOKIO_RUNTIME
            .block_on(async { self.client.create_namespace(&database).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn namespace_exists(&self, database: String) -> PyResult<bool> {
        TOKIO_RUNTIME
            .block_on(async { self.client.namespace_exists(&database).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn drop_namespace(&self, database: String) -> PyResult<()> {
        TOKIO_RUNTIME
            .block_on(async { self.client.drop_namespace(&database).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    /// Create a new table
    #[pyo3(signature = (database, table_name, schema, location=None))]
    fn create_table(
        &self,
        database: String,
        table_name: String,
        schema: Bound<'_, PyAny>,
        location: Option<String>,
    ) -> PyResult<()> {
        let arrow_schema = extract_schema(schema)?;
        TOKIO_RUNTIME
            .block_on(async {
                self.client
                    .create_table(&database, &table_name, arrow_schema, location.as_deref())
                    .await
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn drop_table(&self, database: String, table_name: String) -> PyResult<()> {
        TOKIO_RUNTIME
            .block_on(async { self.client.drop_table(&database, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn list_tables(&self, database: String) -> PyResult<Vec<String>> {
        TOKIO_RUNTIME
            .block_on(async { self.client.list_tables(&database).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    /// Load a table
    fn load_table(&self, database: String, table_name: String) -> PyResult<PyTable> {
        let metadata = TOKIO_RUNTIME
            .block_on(async { self.client.load_table(&database, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))?;

        PyTable::load_from_catalog(
            &metadata.location,
            self.client.clone(),
            &database,
            &table_name,
            None,
            None,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn table_exists(&self, database: String, table_name: String) -> PyResult<bool> {
        TOKIO_RUNTIME
            .block_on(async { self.client.table_exists(&database, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }
}
