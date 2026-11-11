// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

pub mod glue;
pub mod hive;
pub mod jdbc;
pub mod memory;
pub mod nessie;
pub mod rest;
pub mod s3tables;
pub mod unity;

pub use glue::PyGlueCatalog;
pub use hive::PyHiveCatalog;
pub use jdbc::PyJdbcCatalog;
pub use memory::PyMemoryCatalog;
pub use nessie::PyNessieCatalog;
pub use rest::PyRestCatalog;
pub use s3tables::PyS3TablesCatalog;
pub use unity::PyUnityCatalog;

use pyo3::prelude::*;
use std::collections::HashMap;
use std::str::FromStr;

use super::table::PyTable;
use crate::core::catalog::{resolve_aws_region_from_config, CatalogConfig, CatalogType};

fn parse_bool_opt(config: &HashMap<String, String>, key: &str) -> Option<bool> {
    config
        .get(key)
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "true" | "1" | "yes"))
}

fn build_python_catalog(
    py: Python<'_>,
    catalog_type: CatalogType,
    config: &HashMap<String, String>,
) -> PyResult<Py<PyAny>> {
    match catalog_type {
        CatalogType::Memory => {
            let uri = config
                .get("uri")
                .or_else(|| config.get("url"))
                .or_else(|| config.get("name"))
                .cloned();
            let warehouse = config.get("warehouse").cloned();
            let catalog = PyMemoryCatalog::new(uri, warehouse)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Nessie => {
            let url = config
                .get("url")
                .or_else(|| config.get("uri"))
                .cloned()
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'url' for Nessie catalog",
                    )
                })?;
            let token = config.get("token").cloned();
            let catalog = PyNessieCatalog::new(url, token)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Rest => {
            let url = config
                .get("url")
                .or_else(|| config.get("uri"))
                .cloned()
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'url' or 'uri' for Rest catalog",
                    )
                })?;
            let prefix = config.get("prefix").cloned();
            let warehouse = config.get("warehouse").cloned();
            let token = config.get("token").cloned();
            let credential = config.get("credential").cloned();
            let client_id = config
                .get("client_id")
                .or_else(|| config.get("client-id"))
                .cloned();
            let client_secret = config
                .get("client_secret")
                .or_else(|| config.get("client-secret"))
                .cloned();
            let oauth2_server_uri = config
                .get("oauth2-server-uri")
                .or_else(|| config.get("oauth2_server_uri"))
                .cloned();
            let scope = config.get("scope").cloned();
            let discover_config = parse_bool_opt(config, "discover_config");

            let sigv4_requested = config.get("auth").is_some_and(|a| {
                a.eq_ignore_ascii_case("sigv4") || a.eq_ignore_ascii_case("aws_sigv4")
            }) || parse_bool_opt(config, "rest.sigv4-enabled")
                .unwrap_or(false);

            let region = if sigv4_requested {
                Some(resolve_aws_region_from_config(config).ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing AWS 'region' (or 'rest.signing-region' / AWS_REGION) for SigV4 REST catalog",
                    )
                })?)
            } else if token.is_none() && credential.is_none() && client_id.is_none() {
                config
                    .get("region")
                    .or_else(|| config.get("rest.signing-region"))
                    .cloned()
            } else {
                None
            };
            let service = config
                .get("service")
                .or_else(|| config.get("rest.signing-name"))
                .cloned();

            let catalog = PyRestCatalog::new(
                url,
                prefix,
                token,
                client_id,
                client_secret,
                credential,
                oauth2_server_uri,
                scope,
                region,
                service,
                warehouse,
                discover_config,
            )?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::S3Tables => {
            let warehouse = config.get("warehouse").cloned();
            let region = config
                .get("region")
                .or_else(|| config.get("rest.signing-region"))
                .cloned();
            let url = config.get("url").or_else(|| config.get("uri")).cloned();
            let prefix = config.get("prefix").cloned();
            let discover_config = parse_bool_opt(config, "discover_config");
            let catalog = PyS3TablesCatalog::new(warehouse, region, url, prefix, discover_config)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Glue => {
            let catalog_id = config.get("catalog_id").cloned();
            let region = config
                .get("region")
                .or_else(|| config.get("rest.signing-region"))
                .cloned();
            let mode = config.get("mode").cloned();
            let url = config.get("url").or_else(|| config.get("uri")).cloned();
            let warehouse = config.get("warehouse").cloned();
            let prefix = config.get("prefix").cloned();
            let catalog = PyGlueCatalog::new(catalog_id, region, mode, url, warehouse, prefix)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Hive => {
            let url = config
                .get("url")
                .or_else(|| config.get("uri"))
                .cloned()
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'url' for Hive catalog",
                    )
                })?;
            let catalog = PyHiveCatalog::new(url)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Unity => {
            let url = config
                .get("url")
                .or_else(|| config.get("uri"))
                .cloned()
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'url' for Unity catalog",
                    )
                })?;
            let token = config.get("token").cloned().ok_or_else(|| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'token' for Unity catalog")
            })?;
            let mode = config.get("mode").cloned();
            let warehouse = config
                .get("warehouse")
                .or_else(|| config.get("catalog"))
                .cloned();
            let prefix = config.get("prefix").cloned();
            let catalog = PyUnityCatalog::new(url, token, mode, warehouse, prefix)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Jdbc => {
            let uri = config
                .get("uri")
                .or_else(|| config.get("url"))
                .cloned()
                .ok_or_else(|| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Missing 'uri' for JDBC catalog",
                    )
                })?;
            let warehouse = config.get("warehouse").cloned();
            let catalog_name = config.get("catalog_name").cloned();
            let catalog = PyJdbcCatalog::new(uri, warehouse, catalog_name)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
    }
}

#[pyfunction]
#[pyo3(signature = (catalog_type, config))]
pub fn create_catalog(
    py: Python<'_>,
    catalog_type: String,
    config: HashMap<String, String>,
) -> PyResult<Py<PyAny>> {
    let parsed_type = CatalogType::from_str(&catalog_type)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
    build_python_catalog(py, parsed_type, &config)
}

#[pyfunction]
pub fn create_catalog_from_config(py: Python<'_>, path: String) -> PyResult<Py<PyAny>> {
    let config = CatalogConfig::load_from_file(&path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))?;
    build_python_catalog(py, config.catalog_type, &config.config)
}

#[pyfunction]
pub fn load_default_catalog(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let config = CatalogConfig::load_default()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))?;
    build_python_catalog(py, config.catalog_type, &config.config)
}

#[pyfunction]
pub fn open_table(_py: Python<'_>, uri: &str) -> PyResult<PyTable> {
    PyTable::new_internal(uri, None, None, None)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
}
