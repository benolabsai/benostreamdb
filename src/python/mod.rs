// Copyright (c) 2026 Richard Albright. All rights reserved.

pub mod catalog;
pub mod helpers;
pub mod manifest;
pub mod schema;
pub mod session;
pub mod stats;
pub mod table;

pub use catalog::*;
pub use helpers::*;
pub use manifest::*;
pub use schema::*;
pub use session::*;
pub use stats::*;
pub use table::*;

use pyo3::prelude::*;

/// Register all HyperStreamDB Python bindings into the given module.
pub fn migrate_hyperstreamdb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Helper functions
    m.add_function(wrap_pyfunction!(helpers::init_logging, m)?)?;

    // Catalog factory functions
    m.add_function(wrap_pyfunction!(catalog::create_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::create_catalog_from_config, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::load_default_catalog, m)?)?;
    m.add_function(wrap_pyfunction!(catalog::open_table, m)?)?;

    // Core classes
    m.add_class::<table::PyTable>()?;
    m.add_class::<stats::PyMergeMode>()?;
    m.add_class::<session::PySession>()?;

    // Catalog classes
    m.add_class::<catalog::PyNessieCatalog>()?;
    m.add_class::<catalog::PyRestCatalog>()?;
    m.add_class::<catalog::PyGlueCatalog>()?;
    m.add_class::<catalog::PyHiveCatalog>()?;
    m.add_class::<catalog::PyUnityCatalog>()?;
    m.add_class::<catalog::PyJdbcCatalog>()?;

    // Stats classes
    m.add_class::<stats::PyDataFileInfo>()?;
    m.add_class::<stats::PySplit>()?;
    m.add_class::<stats::PyTableStatistics>()?;
    m.add_class::<stats::PyIndexCoverage>()?;

    // Schema classes
    m.add_class::<schema::PyDataType>()?;
    m.add_class::<schema::PyField>()?;
    m.add_class::<schema::PyPartitionField>()?;
    m.add_class::<schema::PySchema>()?;

    // Manifest classes
    m.add_class::<manifest::PyManifest>()?;
    m.add_class::<manifest::PyManifestEntry>()?;

    Ok(())
}
