// Copyright (c) 2026 Richard Albright. All rights reserved.

use pyo3::prelude::*;
use std::sync::Arc;

use super::super::helpers::TOKIO_RUNTIME;
use super::super::schema::PySchema;
use super::super::table::PyTable;
use crate::core::catalog::hive::HiveMetastoreClient;
use crate::core::catalog::Catalog;

/// Python wrapper for Hive Metastore Catalog
///
/// Delegates to `HiveMetastoreClient` for table lifecycle operations.
/// For production use, consider:
/// - AWS Glue (Hive-compatible)
/// - Iceberg REST Catalog
/// - Nessie Catalog
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

    /// Load a table
    fn load_table(&self, database: String, table_name: String) -> PyResult<PyTable> {
        let metadata = TOKIO_RUNTIME
            .block_on(async { self.client.load_table(&database, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))?;

        PyTable::new_internal(&metadata.location, None)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }

    fn table_exists(&self, database: String, table_name: String) -> PyResult<bool> {
        TOKIO_RUNTIME
            .block_on(async { self.client.table_exists(&database, &table_name).await })
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err((e.to_string(),)))
    }
}
