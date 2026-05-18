// Copyright (c) 2026 Richard Albright. All rights reserved.

/// Table state management: indexing state, catalog identity, and configuration accessors.
///
/// Contains:
/// - `TableIndexState` / `TableCatalogState` structs
/// - `ColumnIndexConfig` / `LabelPattern` types
/// - State accessor/mutator methods on `Table`
use std::collections::HashMap;
use std::sync::Arc;
use serde::{Serialize, Deserialize};
use crate::SegmentConfig;
use crate::core::index::memory::InMemoryVectorIndex;

use super::Table;

// ---------------------------------------------------------------------------
// Public type definitions (re-exported from mod.rs)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
#[derive(Default)]
pub struct ColumnIndexConfig {
    pub device: Option<String>,
    pub tokenizer: Option<String>,
    pub enabled: bool,
    pub algorithms: Vec<crate::core::manifest::IndexAlgorithm>,
}

/// Strategy for labeling unnamed or numerically named columns during first write.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LabelPattern {
    /// Excel style: A, B, C... AA, AB... (Default for premium UX)
    ExcelAlpha,
    /// Polars style: column_1, column_2...
    Polars,
    /// Pandas style: 0, 1, 2...
    Pandas,
}

impl Default for LabelPattern {
    fn default() -> Self {
        Self::ExcelAlpha
    }
}

/// Indexing state and configuration for a Table
#[derive(Clone)]
pub(crate) struct TableIndexState {
    pub index_all: bool,
    pub index_columns: Arc<parking_lot::RwLock<Vec<String>>>,
    pub index_configs: Arc<parking_lot::RwLock<HashMap<String, ColumnIndexConfig>>>,
    pub default_device: Arc<parking_lot::RwLock<Option<String>>>,
    pub memory_index: Arc<parking_lot::RwLock<Option<InMemoryVectorIndex>>>,
}

/// Catalog identity and synchronization state for a Table
#[derive(Clone)]
pub(crate) struct TableCatalogState {
    pub catalog: Option<Arc<dyn crate::core::catalog::Catalog>>,
    pub namespace: Option<String>,
    pub table_name: Option<String>,
}

// ---------------------------------------------------------------------------
// impl Table — state accessors / mutators
// ---------------------------------------------------------------------------

impl Table {
    /// Get the current SegmentConfig derived from the table's state.
    pub fn get_config(&self) -> SegmentConfig {
        SegmentConfig::new(&self.uri, "")
            .with_index_all(self.indexing.index_all)
            .with_columns_to_index(self.indexing.index_columns.read().clone())
    }

    pub fn set_index_all(&mut self, enabled: bool) {
        self.indexing.index_all = enabled;
    }

    pub fn get_index_all(&self) -> bool {
        self.indexing.index_all
    }

    pub fn set_autocommit(&self, enabled: bool) {
        self.autocommit.store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn get_autocommit(&self) -> bool {
        self.autocommit.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Check if the table currently has an active in-memory vector index.
    pub fn has_memory_index(&self) -> bool {
        self.indexing.memory_index.read().is_some()
    }

    /// Get the number of rows currently in the write buffer (not yet flushed).
    pub fn write_buffer_row_count(&self) -> usize {
        self.write_buffer.read().iter().map(|b| b.num_rows()).sum()
    }

    /// Get the list of currently indexed column names.
    pub fn get_index_columns(&self) -> Vec<String> {
        self.indexing.index_columns.read().clone()
    }

    pub fn set_default_device(&mut self, device: Option<String>) {
        let mut d = self.indexing.default_device.write();
        *d = device;
    }

    pub fn get_default_device(&self) -> Option<String> {
        self.indexing.default_device.read().clone()
    }

    pub fn remove_index_columns(&mut self, columns: Vec<String>) {
        let mut index_cols = self.indexing.index_columns.write();
        index_cols.retain(|c| !columns.contains(c));
    }

    pub fn remove_all_index_columns(&mut self) {
        let mut index_cols = self.indexing.index_columns.write();
        index_cols.clear();
        self.indexing.index_all = false;
    }
}
