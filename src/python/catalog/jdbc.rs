// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use pyo3::prelude::*;
use std::sync::Arc;

use super::super::helpers::{extract_schema, TOKIO_RUNTIME};
use super::super::table::PyTable;
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
    pub(crate) fn new(
        uri: String,
        warehouse: Option<String>,
        catalog_name: Option<String>,
    ) -> PyResult<Self> {
        let catalog_name = catalog_name.unwrap_or_else(|| "default".to_string());
        let client = TOKIO_RUNTIME
            .block_on(async { JdbcCatalogClient::new(uri, warehouse, catalog_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))?;

        Ok(PyJdbcCatalog {
            client: Arc::new(client),
        })
    }

    fn create_namespace(&self, namespace: String) -> PyResult<()> {
        TOKIO_RUNTIME
            .block_on(async { self.client.create_namespace(&namespace).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn namespace_exists(&self, namespace: String) -> PyResult<bool> {
        TOKIO_RUNTIME
            .block_on(async { self.client.namespace_exists(&namespace).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn drop_namespace(&self, namespace: String) -> PyResult<()> {
        TOKIO_RUNTIME
            .block_on(async { self.client.drop_namespace(&namespace).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    /// Create a new table
    #[pyo3(signature = (namespace, table_name, schema, location=None))]
    fn create_table(
        &self,
        namespace: String,
        table_name: String,
        schema: Bound<'_, PyAny>,
        location: Option<String>,
    ) -> PyResult<()> {
        let arrow_schema = extract_schema(schema)?;
        TOKIO_RUNTIME
            .block_on(async {
                self.client
                    .create_table(&namespace, &table_name, arrow_schema, location.as_deref())
                    .await
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn drop_table(&self, namespace: String, table_name: String) -> PyResult<()> {
        TOKIO_RUNTIME
            .block_on(async { self.client.drop_table(&namespace, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn list_tables(&self, namespace: String) -> PyResult<Vec<String>> {
        TOKIO_RUNTIME
            .block_on(async { self.client.list_tables(&namespace).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    /// Load a table (returns catalog-backed PyTable)
    fn load_table(&self, namespace: String, table_name: String) -> PyResult<PyTable> {
        let metadata = TOKIO_RUNTIME
            .block_on(async { self.client.load_table(&namespace, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))?;

        PyTable::load_from_catalog(
            &metadata.location,
            self.client.clone(),
            &namespace,
            &table_name,
            None,
            None,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn table_exists(&self, namespace: String, table_name: String) -> PyResult<bool> {
        TOKIO_RUNTIME
            .block_on(async { self.client.table_exists(&namespace, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }
}
