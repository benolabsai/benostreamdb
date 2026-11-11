// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use pyo3::prelude::*;
use std::sync::Arc;

use super::super::helpers::{extract_schema, TOKIO_RUNTIME};
use super::super::table::PyTable;
use crate::core::catalog::glue::GlueCatalogClient;
use crate::core::catalog::rest::RestCatalogClient;
use crate::core::catalog::{is_valid_aws_region, Catalog};

/// Python wrapper for AWS Glue Catalog (native SDK or Iceberg REST mode)
#[pyclass]
pub struct PyGlueCatalog {
    client: Arc<dyn Catalog>,
}

#[pymethods]
impl PyGlueCatalog {
    #[new]
    #[pyo3(signature = (catalog_id=None, region=None, mode=None, url=None, warehouse=None, prefix=None))]
    pub(crate) fn new(
        catalog_id: Option<String>,
        region: Option<String>,
        mode: Option<String>,
        url: Option<String>,
        warehouse: Option<String>,
        prefix: Option<String>,
    ) -> PyResult<Self> {
        let mode_rest = mode
            .as_deref()
            .is_some_and(|m| m.eq_ignore_ascii_case("rest"));

        if mode_rest {
            let resolved_region = region
                .map(|r| r.trim().to_string())
                .filter(|r| !r.is_empty())
                .or_else(|| {
                    std::env::var("AWS_REGION")
                        .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
                        .ok()
                        .map(|r| r.trim().to_string())
                        .filter(|r| !r.is_empty())
                })
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'region' (or AWS_REGION) for Glue Iceberg REST mode",
                    )
                })?;

            if !is_valid_aws_region(&resolved_region) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid AWS region '{resolved_region}' for Glue Iceberg REST mode"
                )));
            }

            let endpoint_url = url
                .filter(|u| !u.trim().is_empty())
                .unwrap_or_else(|| format!("https://glue.{resolved_region}.amazonaws.com/iceberg"));
            let wh = warehouse
                .or(catalog_id)
                .map(|w| w.trim().to_string())
                .filter(|w| !w.is_empty());
            let discover = wh.is_some() && prefix.as_deref().is_none_or(|p| p.trim().is_empty());

            let rest_client =
                RestCatalogClient::with_sigv4(endpoint_url, prefix, resolved_region, "glue")
                    .with_warehouse(wh, discover);

            Ok(PyGlueCatalog {
                client: Arc::new(rest_client),
            })
        } else {
            let client = TOKIO_RUNTIME
                .block_on(async { GlueCatalogClient::with_region(catalog_id, region).await })
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))?;

            Ok(PyGlueCatalog {
                client: Arc::new(client),
            })
        }
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

    /// Load a table (returns catalog-backed PyTable)
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
