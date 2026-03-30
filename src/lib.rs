// Copyright (c) 2026 Richard Albright. All rights reserved.

use std::sync::Arc;
pub mod core;

pub mod enterprise;

pub mod telemetry;

#[cfg(feature = "python")]
pub mod python_binding;

#[cfg(feature = "python")]
pub mod python_gpu_context;

#[cfg(feature = "python")]
pub mod python_distance;

// Re-export main types for convenience
pub use crate::core::table::{Table, VectorSearchParams};
pub use crate::core::index::VectorMetric;
pub use crate::core::catalog::{CatalogType, create_catalog, create_catalog_async, Catalog};



#[cfg(feature = "python")]
use pyo3::prelude::*;

#[cfg(feature = "python")]
#[pymodule]
fn hyperstreamdb(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(python_binding::create_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(python_binding::create_catalog_from_config, m)?)?;
    m.add_function(wrap_pyfunction!(python_binding::load_default_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(python_binding::open_table, m)?)?;
    m.add_class::<python_binding::PyTable>()?;
    m.add_class::<python_binding::PyMergeMode>()?;
    m.add_class::<python_binding::PyNessieCatalog>()?;
    m.add_class::<python_binding::PyRestCatalog>()?;
    m.add_class::<python_binding::PyGlueCatalog>()?;
    m.add_class::<python_binding::PyHiveCatalog>()?;
    m.add_class::<python_binding::PyUnityCatalog>()?;
    m.add_class::<python_binding::PySession>()?;
    
    m.add_class::<python_binding::PyDataFileInfo>()?;
    m.add_class::<python_binding::PySplit>()?;
    m.add_class::<python_binding::PyTableStatistics>()?;
    m.add_class::<python_binding::PyIndexCoverage>()?;
    
    m.add_class::<python_binding::PyDataType>()?;
    m.add_class::<python_binding::PyField>()?;
    m.add_class::<python_binding::PySchema>()?;
    
    // GPU Context API
    m.add_class::<python_gpu_context::PyComputeContext>()?;
    
    // Distance API - Single-pair functions
    m.add_function(wrap_pyfunction!(python_distance::py_l2, m)?)?;
    m.add_function(wrap_pyfunction!(python_distance::py_cosine, m)?)?;
    m.add_function(wrap_pyfunction!(python_distance::py_inner_product, m)?)?;
    m.add_function(wrap_pyfunction!(python_distance::py_l1, m)?)?;
    m.add_function(wrap_pyfunction!(python_distance::py_hamming, m)?)?;
    m.add_function(wrap_pyfunction!(python_distance::py_jaccard, m)?)?;
    
    // Distance API - Batch functions
    m.add_function(wrap_pyfunction!(python_distance::py_l2_batch, m)?)?;
    m.add_function(wrap_pyfunction!(python_distance::py_cosine_batch, m)?)?;
    m.add_function(wrap_pyfunction!(python_distance::py_inner_product_batch, m)?)?;
    m.add_function(wrap_pyfunction!(python_distance::py_l1_batch, m)?)?;
    m.add_function(wrap_pyfunction!(python_distance::py_hamming_batch, m)?)?;
    m.add_function(wrap_pyfunction!(python_distance::py_jaccard_batch, m)?)?;
    
    // Sparse Vector API
    m.add_class::<python_distance::PySparseVector>()?;
    m.add_function(wrap_pyfunction!(python_distance::py_l2_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(python_distance::py_cosine_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(python_distance::py_inner_product_sparse, m)?)?;
    
    // Binary Vector API
    m.add_function(wrap_pyfunction!(python_distance::py_hamming_packed, m)?)?;
    m.add_function(wrap_pyfunction!(python_distance::py_jaccard_packed, m)?)?;
    m.add_function(wrap_pyfunction!(python_distance::py_hamming_auto, m)?)?;
    m.add_function(wrap_pyfunction!(python_distance::py_jaccard_auto, m)?)?;
    
    Ok(())
}

/// A HyperStream Segment is a self-contained unit of data and aligned indexes.
#[derive(Clone)]
pub struct SegmentConfig {
    pub base_path: String,
    pub segment_id: String,
    /// Explicit path to the data file (optional, used for external tables)
    pub parquet_path: Option<String>,
    /// Optional separate store for data files (e.g. for external Iceberg tables)
    pub data_store: Option<Arc<dyn object_store::ObjectStore>>,
    pub delete_files: Vec<crate::core::manifest::DeleteFile>,
    pub index_files: Vec<crate::core::manifest::IndexFile>,
    pub file_size: Option<u64>,
    /// Build indexes for ALL columns (overrides columns_to_index if true)
    pub index_all: bool,
    /// Columns to build indexes for. If None or empty, no indexes are built.
    pub columns_to_index: Option<Vec<String>>,
    /// Partition values for this segment
    pub partition_values: std::collections::HashMap<String, serde_json::Value>,
    /// Per-column device override (e.g. "cpu", "gpu", "mps")
    pub column_devices: std::collections::HashMap<String, String>,
    pub default_device: Option<String>,
}

impl SegmentConfig {
    pub fn new(base_path: &str, segment_id: &str) -> Self {
        Self {
            base_path: base_path.to_string(),
            segment_id: segment_id.to_string(),
            parquet_path: None,
            data_store: None,
            delete_files: Vec::new(),
            index_files: Vec::new(),
            file_size: None,
            index_all: false,
            columns_to_index: None,
            partition_values: std::collections::HashMap::new(),
            column_devices: std::collections::HashMap::new(),
            default_device: None,
        }
    }

    pub fn with_parquet_path(mut self, path: String) -> Self {
        self.parquet_path = Some(path);
        self
    }

    pub fn with_data_store(mut self, store: Arc<dyn object_store::ObjectStore>) -> Self {
        self.data_store = Some(store);
        self
    }

    pub fn with_delete_files(mut self, delete_files: Vec<crate::core::manifest::DeleteFile>) -> Self {
        self.delete_files = delete_files;
        self
    }
    
    pub fn with_index_files(mut self, index_files: Vec<crate::core::manifest::IndexFile>) -> Self {
        self.index_files = index_files;
        self
    }

    pub fn with_partition_values(mut self, partition_values: std::collections::HashMap<String, serde_json::Value>) -> Self {
        self.partition_values = partition_values;
        self
    }
    
    pub fn with_file_size(mut self, size: u64) -> Self {
        self.file_size = Some(size);
        self
    }

    pub fn with_index_all(mut self, index_all: bool) -> Self {
        self.index_all = index_all;
        self
    }

    pub fn with_default_device(mut self, device: Option<String>) -> Self {
        self.default_device = device;
        self
    }

    pub fn with_column_devices(mut self, column_devices: std::collections::HashMap<String, String>) -> Self {
        self.column_devices = column_devices;
        self
    }

    pub fn with_columns_to_index(mut self, cols: Vec<String>) -> Self {
        // If cols is empty, set to None instead of Some([]) to avoid triggering indexing path
        if cols.is_empty() {
            self.columns_to_index = None;
        } else {
            self.columns_to_index = Some(cols);
        }
        self
    }
}
