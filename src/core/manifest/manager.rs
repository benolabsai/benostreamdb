// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! ManifestManager: coordinates loading, committing, schema evolution, and partition management.
//!
//! Submodules:
//! - `load` — loading manifests, entries, and walking history
//! - `commit` — optimistic concurrency commits, rollback, vacuum, import
//! - `schema` — schema updates, identifier fields, index specs
//! - `partition` — partition spec updates and batch partitioning

use anyhow::Result;
use futures::StreamExt;
use object_store::{path::Path, ObjectStore};
use std::collections::HashMap;
use std::sync::Arc;

use super::types::*;

// ── Submodules ───────────────────────────────────────────────────────────────
mod commit;
mod load;
mod partition;
mod schema;

// ── Public types ─────────────────────────────────────────────────────────────
#[allow(unused_imports)]
pub(crate) use commit::*;
#[allow(unused_imports)]
pub(crate) use load::*;
#[allow(unused_imports)]
pub(crate) use partition::*;
#[allow(unused_imports)]
pub(crate) use schema::*;

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
    /// Iceberg table format version to set on the committed manifest. `None`
    /// preserves the current version. Used to upgrade a table to v3 (row lineage).
    pub format_version: Option<i32>,
    pub is_fast_append: bool,
    /// If true, verifies that all remove_paths still exist in the current snapshot version.
    /// Used by compaction to prevent removing files concurrently replaced or deleted.
    pub require_remove_paths_exist: bool,
    /// MVCC rebase policy: when true, a `remove_path` that no longer exists in
    /// the current snapshot is *skipped* rather than aborting the commit.
    ///
    /// This is what makes concurrent writers safe: if two compactions race on
    /// the same candidate file, the loser rebases onto the winner's snapshot
    /// and simply drops the already-removed path instead of failing hard. Only
    /// meaningful together with `require_remove_paths_exist`; when that is
    /// false, missing paths are already ignored.
    pub skip_missing_remove_paths: bool,
    /// Compaction rebase policy: when true, if *any* `remove_path` is already
    /// gone from the current snapshot, the whole commit is a no-op.
    ///
    /// This is required for compaction. A concurrent compaction that already
    /// removed one of our inputs has written its own replacement containing
    /// those rows; adding our replacement too would duplicate the data, and
    /// removing our remaining inputs without adding our replacement would lose
    /// rows. The only correct outcome is to discard this compaction entirely.
    pub discard_add_on_missing_remove: bool,
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

    /// Invalidate the manifest caches for this table.
    ///
    /// Called after `DROP TABLE` deletes every object: without this, a
    /// subsequent `getTableSchema`/`getTableHandle` would still see the deleted
    /// manifest through `LATEST_VERSION_CACHE`/`MANIFEST_CACHE` (they have short
    /// TTLs, but a dropped table would look like it still exists until they
    /// expire). Mirrors the invalidation `commit()` performs.
    pub async fn invalidate_caches(&self) {
        let dir_key = self.get_dir_cache_key();
        crate::core::cache::LATEST_VERSION_CACHE
            .invalidate(&dir_key)
            .await;
        crate::core::cache::MANIFEST_CACHE.invalidate_all();
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

    /// Mirror the manifest's table properties into the official Iceberg
    /// metadata file (`metadata/v{N}.metadata.json`).
    ///
    /// The manifest (`_manifest/vN.json`) is this engine's source of truth, but
    /// external readers (Spark, Trino, PyIceberg, and the Iceberg REST catalog)
    /// only read `TableMetadata`. Without this mirror, `SET TBLPROPERTIES`
    /// values — including the MCP's `benostream.embedding.*` provenance — stay
    /// invisible outside the engine.
    ///
    /// Failures are returned so callers can log them, but they must not roll
    /// back the already-durable manifest commit: the next data commit re-syncs
    /// properties from the manifest in `write.rs`.
    pub(crate) async fn sync_metadata_properties(&self, manifest: &Manifest) -> Result<()> {
        use crate::core::metadata::TableMetadata;

        let mut metadata = match TableMetadata::load_latest(self.store.as_ref()).await {
            Ok(metadata) => metadata,
            Err(err) => {
                // Metadata is normally created by the first data commit. For a
                // properties-only table (e.g. `CREATE TABLE` + `SET
                // TBLPROPERTIES`), synthesize a skeleton from the manifest so
                // external readers can see the properties immediately.
                let Some(schema) = manifest.schemas.last().cloned() else {
                    tracing::debug!(
                        error = %err,
                        "no table schema available; skipping Iceberg metadata property sync"
                    );
                    return Ok(());
                };
                let mut metadata = TableMetadata::new(
                    manifest.format_version,
                    uuid::Uuid::new_v4().to_string(),
                    self.root_uri.clone(),
                    schema,
                    manifest.partition_spec.clone(),
                    manifest.sort_orders.first().cloned().unwrap_or_default(),
                );
                metadata.current_schema_id = manifest.current_schema_id;
                metadata.default_spec_id = manifest.default_spec_id;
                metadata.default_sort_order_id = manifest.default_sort_order_id;
                metadata.last_column_id = manifest.last_column_id;
                metadata.next_row_id = if manifest.format_version >= 3 {
                    Some(0)
                } else {
                    None
                };
                metadata
            }
        };

        metadata.properties = manifest.properties.clone();
        metadata.format_version = manifest.format_version;
        metadata.last_updated_ms = manifest.timestamp_ms;
        metadata
            .save_to_store(self.store.as_ref(), manifest.version as i32)
            .await?;
        Ok(())
    }
}
