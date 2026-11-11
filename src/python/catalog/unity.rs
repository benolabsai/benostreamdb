// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use pyo3::prelude::*;
use std::sync::Arc;

use super::super::helpers::{extract_schema, TOKIO_RUNTIME};
use super::super::table::PyTable;
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
    #[pyo3(signature = (url, token, mode=None, warehouse=None, prefix=None))]
    pub(crate) fn new(
        url: String,
        token: String,
        mode: Option<String>,
        warehouse: Option<String>,
        prefix: Option<String>,
    ) -> PyResult<Self> {
        let trimmed = url.trim_end_matches('/');
        let is_iceberg_rest = mode.as_deref().is_some_and(|m| {
            m.eq_ignore_ascii_case("iceberg-rest") || m.eq_ignore_ascii_case("rest")
        }) || trimmed.ends_with("/iceberg-rest")
            || trimmed.ends_with("/iceberg");

        let client = if is_iceberg_rest {
            UnityCatalogClient::new_iceberg_rest(url, token, warehouse, prefix)
        } else {
            let mut c = UnityCatalogClient::new(url, token);
            if let Some(wh) = warehouse.filter(|w| !w.trim().is_empty()) {
                c = c.with_default_catalog(wh);
            }
            c
        };

        Ok(PyUnityCatalog {
            client: Arc::new(client),
        })
    }

    fn create_namespace(&self, catalog: String) -> PyResult<()> {
        TOKIO_RUNTIME
            .block_on(async { self.client.create_namespace(&catalog).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn namespace_exists(&self, catalog: String) -> PyResult<bool> {
        TOKIO_RUNTIME
            .block_on(async { self.client.namespace_exists(&catalog).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn drop_namespace(&self, catalog: String) -> PyResult<()> {
        TOKIO_RUNTIME
            .block_on(async { self.client.drop_namespace(&catalog).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    /// Create a new table
    #[pyo3(signature = (catalog, table_name, schema, location=None))]
    fn create_table(
        &self,
        catalog: String,
        table_name: String,
        schema: Bound<'_, PyAny>,
        location: Option<String>,
    ) -> PyResult<()> {
        let arrow_schema = extract_schema(schema)?;
        TOKIO_RUNTIME
            .block_on(async {
                self.client
                    .create_table(&catalog, &table_name, arrow_schema, location.as_deref())
                    .await
            })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn drop_table(&self, catalog: String, table_name: String) -> PyResult<()> {
        TOKIO_RUNTIME
            .block_on(async { self.client.drop_table(&catalog, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn list_tables(&self, catalog: String) -> PyResult<Vec<String>> {
        TOKIO_RUNTIME
            .block_on(async { self.client.list_tables(&catalog).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    /// Load a table (returns catalog-backed PyTable)
    fn load_table(&self, catalog: String, table_name: String) -> PyResult<PyTable> {
        let metadata = TOKIO_RUNTIME
            .block_on(async { self.client.load_table(&catalog, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))?;

        PyTable::load_from_catalog(
            &metadata.location,
            self.client.clone(),
            &catalog,
            &table_name,
            None,
            None,
        )
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn table_exists(&self, catalog: String, table_name: String) -> PyResult<bool> {
        TOKIO_RUNTIME
            .block_on(async { self.client.table_exists(&catalog, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }
}
