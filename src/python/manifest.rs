// Copyright (c) 2026 Richard Albright. All rights reserved.

use pyo3::prelude::*;

#[pyclass(name = "ManifestEntry")]
#[derive(Clone, Debug)]
pub struct PyManifestEntry {
    #[pyo3(get)]
    pub file_path: String,
    #[pyo3(get)]
    pub file_size_bytes: i64,
    #[pyo3(get)]
    pub record_count: i64,
    #[pyo3(get)]
    pub index_files_count: usize,
    #[pyo3(get)]
    pub delete_files_count: usize,
}

#[pymethods]
impl PyManifestEntry {
    fn __repr__(&self) -> String {
        format!(
            "ManifestEntry(path={}, rows={})",
            self.file_path, self.record_count
        )
    }
}

#[pyclass(name = "Manifest")]
#[derive(Clone, Debug)]
pub struct PyManifest {
    #[pyo3(get)]
    pub version: u64,
    #[pyo3(get)]
    pub timestamp_ms: i64,
    #[pyo3(get)]
    pub current_schema_id: i32,
    #[pyo3(get)]
    pub partition_spec_id: i32,
    #[pyo3(get)]
    pub entries: Vec<PyManifestEntry>,
    #[pyo3(get)]
    pub properties: std::collections::HashMap<String, String>,
}

#[pymethods]
impl PyManifest {
    fn __repr__(&self) -> String {
        format!(
            "Manifest(version={}, entries={})",
            self.version,
            self.entries.len()
        )
    }

    /// Compatibility alias for 'entries'
    #[getter]
    fn files(&self) -> Vec<PyManifestEntry> {
        self.entries.clone()
    }
}
