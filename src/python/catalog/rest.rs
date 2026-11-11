// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use pyo3::prelude::*;
use std::sync::Arc;

use super::super::helpers::{extract_schema, TOKIO_RUNTIME};
use super::super::table::PyTable;
use crate::core::catalog::rest::{RestCatalogAuth, RestCatalogClient};
use crate::core::catalog::{is_valid_aws_region, Catalog};

/// Python wrapper for REST Catalog (Iceberg-compatible)
#[pyclass]
pub struct PyRestCatalog {
    client: Arc<RestCatalogClient>,
}

#[pymethods]
impl PyRestCatalog {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        url,
        prefix=None,
        token=None,
        client_id=None,
        client_secret=None,
        credential=None,
        oauth2_server_uri=None,
        scope=None,
        region=None,
        service=None,
        warehouse=None,
        discover_config=None
    ))]
    pub(crate) fn new(
        url: String,
        prefix: Option<String>,
        token: Option<String>,
        client_id: Option<String>,
        client_secret: Option<String>,
        credential: Option<String>,
        oauth2_server_uri: Option<String>,
        scope: Option<String>,
        region: Option<String>,
        service: Option<String>,
        warehouse: Option<String>,
        discover_config: Option<bool>,
    ) -> PyResult<Self> {
        let region_clean = region
            .map(|r| r.trim().to_string())
            .filter(|r| !r.is_empty());

        let auth = if let Some(reg) = region_clean {
            if !is_valid_aws_region(&reg) {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid AWS region '{reg}' for SigV4 REST catalog"
                )));
            }
            let svc = service
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "execute-api".to_string());
            Some(RestCatalogAuth::AwsSigV4 {
                region: reg,
                service: svc,
            })
        } else if let Some(tok) = token.filter(|t| !t.trim().is_empty()) {
            Some(RestCatalogAuth::BearerToken(tok))
        } else if let Some(cred) = credential.filter(|c| !c.trim().is_empty()) {
            let parts: Vec<&str> = cred.splitn(2, ':').collect();
            if parts.len() != 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Invalid 'credential' format. Expected <client_id>:<client_secret>",
                ));
            }
            let endpoint = oauth2_server_uri
                .unwrap_or_else(|| format!("{}/v1/oauth/tokens", url.trim_end_matches('/')));
            Some(RestCatalogAuth::OAuth2 {
                token_endpoint: endpoint,
                client_id: parts[0].to_string(),
                client_secret: parts[1].to_string(),
                scope,
            })
        } else if let (Some(cid), Some(csec)) = (client_id, client_secret) {
            let endpoint = oauth2_server_uri
                .unwrap_or_else(|| format!("{}/v1/oauth/tokens", url.trim_end_matches('/')));
            Some(RestCatalogAuth::OAuth2 {
                token_endpoint: endpoint,
                client_id: cid,
                client_secret: csec,
                scope,
            })
        } else {
            None
        };

        let wh_clean = warehouse
            .map(|w| w.trim().to_string())
            .filter(|w| !w.is_empty());
        let discover = discover_config.unwrap_or_else(|| {
            wh_clean.is_some() && prefix.as_deref().is_none_or(|p| p.trim().is_empty())
        });

        let client =
            RestCatalogClient::with_auth(url, prefix, auth).with_warehouse(wh_clean, discover);
        Ok(PyRestCatalog {
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
