// Copyright (c) 2026 Richard Albright. All rights reserved.

use pyo3::prelude::*;

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyMergeMode {
    MergeOnRead,
    MergeOnWrite,
}

impl From<PyMergeMode> for crate::core::table::MergeMode {
    fn from(mode: PyMergeMode) -> Self {
        match mode {
            PyMergeMode::MergeOnRead => crate::core::table::MergeMode::MergeOnRead,
            PyMergeMode::MergeOnWrite => crate::core::table::MergeMode::MergeOnWrite,
        }
    }
}

#[pyclass]
pub struct PyDataFileInfo {
    #[pyo3(get)]
    pub file_path: String,
    #[pyo3(get)]
    pub row_count: u64,
    #[pyo3(get)]
    pub file_size_bytes: u64,
    #[pyo3(get)]
    pub min_values: std::collections::HashMap<String, String>,
    #[pyo3(get)]
    pub max_values: std::collections::HashMap<String, String>,
    #[pyo3(get)]
    pub has_scalar_indexes: bool,
    #[pyo3(get)]
    pub has_vector_indexes: bool,
    #[pyo3(get)]
    pub indexed_columns: Vec<String>,
}

#[pyclass]
pub struct PySplit {
    #[pyo3(get)]
    pub file_path: String,
    #[pyo3(get)]
    pub start_offset: u64,
    #[pyo3(get)]
    pub length: u64,
    #[pyo3(get)]
    pub row_group_ids: Vec<usize>,
    #[pyo3(get)]
    pub index_file_path: Option<String>,
    #[pyo3(get)]
    pub can_use_indexes: bool,
}

#[pymethods]
impl PySplit {
    #[new]
    #[pyo3(signature = (file_path, start_offset, length, row_group_ids, index_file_path=None, can_use_indexes=false))]
    fn new(
        file_path: String,
        start_offset: u64,
        length: u64,
        row_group_ids: Vec<usize>,
        index_file_path: Option<String>,
        can_use_indexes: bool,
    ) -> Self {
        Self {
            file_path,
            start_offset,
            length,
            row_group_ids,
            index_file_path,
            can_use_indexes,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub struct PyIndexCoverage {
    #[pyo3(get)]
    pub scalar_indexed_columns: Vec<String>,
    #[pyo3(get)]
    pub vector_indexed_columns: Vec<String>,
    #[pyo3(get)]
    pub inverted_indexed_columns: Vec<String>,
    #[pyo3(get)]
    pub total_index_size_bytes: u64,
}

#[pyclass]
pub struct PyTableStatistics {
    #[pyo3(get)]
    pub row_count: u64,
    #[pyo3(get)]
    pub file_count: usize,
    #[pyo3(get)]
    pub total_size_bytes: u64,
    #[pyo3(get)]
    pub index_coverage: PyIndexCoverage,
}
