// Copyright (c) 2026 Richard Albright. All rights reserved.

pub mod nessie;
pub mod rest;
pub mod glue;
pub mod hive;
pub mod unity;
pub mod jdbc;

pub use nessie::PyNessieCatalog;
pub use rest::PyRestCatalog;
pub use glue::PyGlueCatalog;
pub use hive::PyHiveCatalog;
pub use unity::PyUnityCatalog;
pub use jdbc::PyJdbcCatalog;

use pyo3::prelude::*;
use pyo3::create_exception;

use super::helpers::TOKIO_RUNTIME;
use super::table::PyTable;
use crate::core::catalog::{CatalogConfig, CatalogType};

#[pyfunction]
#[pyo3(signature = (catalog_type, config))]
pub fn create_catalog(py: Python<'_>, catalog_type: String, config: std::collections::HashMap<String, String>) -> PyResult<Py<PyAny>> {
    match catalog_type.to_lowercase().as_str() {
        "nessie" => {
            let url = config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Nessie catalog"))?;
            let catalog = PyNessieCatalog::new(url)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        "rest" => {
             let url = config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Rest catalog"))?;
             let prefix = config.get("prefix").cloned();
             let catalog = PyRestCatalog::new(url, prefix)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        "glue" => {
             let catalog_id = config.get("catalog_id").cloned();
             let catalog = PyGlueCatalog::new(catalog_id)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        "hive" => {
             let url = config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Hive catalog"))?;
             let catalog = PyHiveCatalog::new(url)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        "unity" => {
            let url = config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Unity catalog"))?;
            let token = config.get("token").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'token' for Unity catalog"))?;
            let catalog = PyUnityCatalog::new(url, token)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        "jdbc" => {
            let uri = config.get("uri").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'uri' for JDBC catalog"))?;
            let warehouse = config.get("warehouse").cloned();
            let catalog_name = config.get("catalog_name").cloned();
            let catalog = PyJdbcCatalog::new(uri, warehouse, catalog_name)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Unknown catalog type: {}", catalog_type))),
    }
}

#[pyfunction]
pub fn create_catalog_from_config(py: Python<'_>, path: String) -> PyResult<Py<PyAny>> {
    let config = CatalogConfig::load_from_file(&path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;

    match config.catalog_type {
        CatalogType::Nessie => {
            let url = config.config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Nessie catalog"))?;
            let catalog = PyNessieCatalog::new(url)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Rest => {
             let url = config.config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Rest catalog"))?;
             let prefix = config.config.get("prefix").cloned();
             let catalog = PyRestCatalog::new(url, prefix)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Glue => {
             let catalog_id = config.config.get("catalog_id").cloned();
             let catalog = PyGlueCatalog::new(catalog_id)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Hive => {
             let url = config.config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Hive catalog"))?;
             let catalog = PyHiveCatalog::new(url)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Unity => {
            let url = config.config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Unity catalog"))?;
            let token = config.config.get("token").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'token' for Unity catalog"))?;
            let catalog = PyUnityCatalog::new(url, token)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Jdbc => {
            let uri = config.config.get("uri").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'uri' for JDBC catalog"))?;
            let warehouse = config.config.get("warehouse").cloned();
            let catalog_name = config.config.get("catalog_name").cloned();
            let catalog = PyJdbcCatalog::new(uri, warehouse, catalog_name)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
    }
}

#[pyfunction]
pub fn load_default_catalog(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let config = CatalogConfig::load_default()
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))?;

    match config.catalog_type {
        CatalogType::Nessie => {
            let url = config.config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Nessie catalog"))?;
            let catalog = PyNessieCatalog::new(url)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Rest => {
             let url = config.config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Rest catalog"))?;
             let prefix = config.config.get("prefix").cloned();
             let catalog = PyRestCatalog::new(url, prefix)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Glue => {
             let catalog_id = config.config.get("catalog_id").cloned();
             let catalog = PyGlueCatalog::new(catalog_id)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Hive => {
             let url = config.config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Hive catalog"))?;
             let catalog = PyHiveCatalog::new(url)?;
             Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Unity => {
            let url = config.config.get("url").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'url' for Unity catalog"))?;
            let token = config.config.get("token").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'token' for Unity catalog"))?;
            let catalog = PyUnityCatalog::new(url, token)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
        CatalogType::Jdbc => {
            let uri = config.config.get("uri").cloned()
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyValueError, _>("Missing 'uri' for JDBC catalog"))?;
            let warehouse = config.config.get("warehouse").cloned();
            let catalog_name = config.config.get("catalog_name").cloned();
            let catalog = PyJdbcCatalog::new(uri, warehouse, catalog_name)?;
            Ok(Py::new(py, catalog)?.into_any())
        }
    }
}

#[pyfunction]
pub fn open_table(_py: Python<'_>, uri: &str) -> PyResult<PyTable> {
    PyTable::new_internal(uri, None).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(), )))
}
