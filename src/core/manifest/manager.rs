// Copyright (c) 2026 Richard Albright. All rights reserved.

//! ManifestManager: coordinates loading, committing, schema evolution, and partition management.
//!
//! Submodules:
//! - [`load`] — loading manifests, entries, and walking history
//! - [`commit`] — optimistic concurrency commits, rollback, vacuum, import
//! - [`schema`] — schema updates, identifier fields, index specs
//! - [`partition`] — partition spec updates and batch partitioning

use std::collections::HashMap;
use std::sync::Arc;
use object_store::{path::Path, ObjectStore};
use anyhow::Result;
use futures::StreamExt;

use super::types::*;

// ── Submodules ───────────────────────────────────────────────────────────────
mod load;
mod commit;
mod schema;
mod partition;

// ── Public types ─────────────────────────────────────────────────────────────
#[allow(unused_imports)]
pub(crate) use load::*;
#[allow(unused_imports)]
pub(crate) use commit::*;
#[allow(unused_imports)]
pub(crate) use schema::*;
#[allow(unused_imports)]
pub(crate) use partition::*;

// ── ManifestManager struct ──────────────────────────────────────────────────
#[derive(Clone)]
pub struct ManifestManager {
    pub(crate) store: Arc<dyn ObjectStore>,
    pub(crate) manifest_dir: Path,
    pub(crate) root_uri: String,
}

#[derive(Debug, Default, Clone)]
pub struct CommitMetadata {
    pub updated_schemas: Option<Vec<Schema>>,
    pub updated_schema_id: Option<i32>,
    pub updated_partition_specs: Option<Vec<PartitionSpec>>,
    pub updated_default_spec_id: Option<i32>,
    pub updated_properties: Option<HashMap<String, String>>,
    pub removed_properties: Option<Vec<String>>,
    pub updated_sort_orders: Option<Vec<SortOrder>>,
    pub updated_default_sort_order_id: Option<i32>,
    pub updated_last_column_id: Option<i32>,
    pub is_fast_append: bool,
}

impl ManifestManager {
    pub fn new(store: Arc<dyn ObjectStore>, base_path: &str, root_uri: &str) -> Self {
        // Manifest directory is typically `_manifest/` under the table root
        let manifest_dir = if base_path.is_empty() {
             Path::from("_manifest/")
        } else {
             Path::from(format!("{}/_manifest/", base_path))
        };

        Self {
            store,
            manifest_dir,
            root_uri: root_uri.trim_end_matches('/').to_string(),
        }
    }

    /// Construct a full object store path for a file within the manifest directory.
    pub fn manifest_path(&self, filename: &str) -> Path {
        self.manifest_dir.child(filename)
    }

    fn get_cache_key(&self, path: &Path) -> String {
        format!("{}/{}", self.root_uri, path)
    }

    fn get_dir_cache_key(&self) -> String {
        let mut key = format!("{}/{}", self.root_uri, self.manifest_dir);
        while key.ends_with('/') {
            key.pop();
        }
        key
    }

    /// Check if any manifests exist in the directory
    pub async fn exists(&self) -> Result<bool> {
        let mut stream = self.store.list(Some(&self.manifest_dir));
        if let Some(meta) = stream.next().await {
            let _n = meta?.location.as_ref().len();
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

// ── Global commit locks ─────────────────────────────────────────────────────
// COMMIT_LOCKS is defined in types.rs via lazy_static! macro.
// Re-export for use by commit submodule.
#[allow(unused_imports, dead_code)]
pub(crate) use super::types::COMMIT_LOCKS;
