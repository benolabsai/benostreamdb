// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use pyo3::prelude::*;
use std::sync::Arc;

use super::super::helpers::{extract_schema, TOKIO_RUNTIME};
use super::super::table::PyTable;
use crate::core::catalog::rest::RestCatalogClient;
use crate::core::catalog::{extract_s3tables_region_from_arn, is_valid_aws_region, Catalog};

/// Python wrapper for Amazon S3 Tables Iceberg REST Catalog (`SigV4` signed).
#[pyclass]
pub struct PyS3TablesCatalog {
    client: Arc<RestCatalogClient>,
}

#[pymethods]
impl PyS3TablesCatalog {
    #[new]
    #[pyo3(signature = (warehouse=None, region=None, url=None, prefix=None, discover_config=None))]
    pub(crate) fn new(
        warehouse: Option<String>,
        region: Option<String>,
        url: Option<String>,
        prefix: Option<String>,
        discover_config: Option<bool>,
    ) -> PyResult<Self> {
        let warehouse_clean = warehouse
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty());
        let url_clean = url.map(|s| s.trim().to_string()).filter(|s| !s.is_empty());

        if warehouse_clean.is_none() && url_clean.is_none() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Missing 'warehouse' (S3 Tables bucket ARN) or 'url' for S3Tables catalog",
            ));
        }

        let resolved_region = region
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .or_else(|| {
                warehouse_clean
                    .as_deref()
                    .and_then(extract_s3tables_region_from_arn)
            })
            .or_else(|| {
                std::env::var("AWS_REGION")
                    .or_else(|_| std::env::var("AWS_DEFAULT_REGION"))
                    .ok()
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
            })
            .ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Missing 'region' (or regional 'warehouse' ARN / AWS_REGION) for S3Tables catalog",
                )
            })?;

        if !is_valid_aws_region(&resolved_region) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Invalid AWS region '{resolved_region}' for S3Tables catalog"
            )));
        }

        let endpoint_url = url_clean
            .unwrap_or_else(|| format!("https://s3tables.{resolved_region}.amazonaws.com/iceberg"));
        let discover = discover_config.unwrap_or_else(|| {
            warehouse_clean.is_some() && prefix.as_deref().is_none_or(|p| p.trim().is_empty())
        });

        let client =
            RestCatalogClient::with_sigv4(endpoint_url, prefix, resolved_region, "s3tables")
                .with_warehouse(warehouse_clean, discover);

        Ok(PyS3TablesCatalog {
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
