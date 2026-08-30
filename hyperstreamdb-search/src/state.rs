// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Shared server state: storage root and the open table cache.

use arrow::datatypes::SchemaRef;
use hyperstreamdb::{HyperstreamError, Table};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};

/// Root storage URI for all search indexes.
///
/// Each index `<name>` is a HyperStreamDB table at `{storage_root}/{name}`.
#[derive(Clone)]
pub struct AppState {
    pub storage_root: String,
    /// Stable cluster identifier reported by `GET /`.
    pub cluster_uuid: String,
    /// Open tables by index name. Handlers share one `Arc<Table>` per index
    /// so writes land in a single write buffer / WAL.
    pub tables: Arc<RwLock<HashMap<String, Arc<Table>>>>,
    /// Serializes open/create so concurrent first-use requests for the same
    /// index share one `Table` instance (no forked write buffers / WALs).
    open_gate: Arc<Mutex<()>>,
}

impl AppState {
    pub fn new(storage_root: String, cluster_uuid: String) -> Self {
        Self {
            storage_root,
            cluster_uuid,
            tables: Arc::new(RwLock::new(HashMap::new())),
            open_gate: Arc::new(Mutex::new(())),
        }
    }

    /// Table URI for a named search index: `{storage_root}/{index}`.
    pub fn index_uri(&self, index: &str) -> String {
        format!("{}/{}", self.storage_root.trim_end_matches('/'), index)
    }

    /// Look up an already-open table, or open/create one.
    ///
    /// `schema` is only consulted when the table does not exist yet
    /// (schema-on-write auto-creation from the first document).
    pub async fn open_or_create(
        &self,
        index: &str,
        schema: &Option<SchemaRef>,
    ) -> Result<Arc<Table>, HyperstreamError> {
        // 1. Fast path: already open in this process.
        {
            let tables = self.tables.read().await;
            if let Some(t) = tables.get(index) {
                return Ok(Arc::clone(t));
            }
        }

        // 2. Serialize the open/create decision for this index.
        let _gate = self.open_gate.lock().await;
        let uri = self.index_uri(index);

        // Open (or create) with indexing enabled. The default builder
        // config (`index_all = false`) would leave segments without the
        // BM25/HNSW indexes that `_search` relies on; `Table::builder`
        // works for existing tables, while new ones still need
        // `create_async` for the manifest/Iceberg init.
        let mut table = if table_exists(&uri).await {
            Table::builder(uri).with_index_all(true).build_async().await.map_err(|e| {
                HyperstreamError::internal(format!("failed to open index '{index}': {e}"))
            })?
        } else {
            let schema = schema.clone().unwrap_or_else(empty_schema);
            match Table::create_async(uri.clone(), schema).await {
                Ok(_) => {}
                // Lost a create race with another request: re-open instead.
                Err(e) if e.to_string().contains("already exists") => {}
                Err(e) => {
                    return Err(HyperstreamError::internal(format!(
                        "failed to create index '{index}': {e}"
                    )))
                }
            }
            Table::builder(uri).with_index_all(true).build_async().await.map_err(|e| {
                HyperstreamError::internal(format!("failed to open index '{index}': {e}"))
            })?
        };

        // Backfill BM25/HNSW indexes on segments committed before this
        // table instance was opened with indexing enabled (a no-op for
        // fresh tables). The call also pins `index_all = true` on this
        // instance so its commits keep building the indexes in the
        // background.
        table
            .index_all_columns_async()
            .await
            .map_err(|e| {
                HyperstreamError::internal(format!(
                    "failed to build search indexes for '{index}': {e}"
                ))
            })?;

        // 3. Publish (first instance wins) and hand back the shared handle.
        let mut tables = self.tables.write().await;
        let entry = tables
            .entry(index.to_string())
            .or_insert_with(|| Arc::new(table));
        Ok(Arc::clone(entry))
    }
}

/// Default storage root: `file://~/.hyperstreamdb/search`
pub fn default_storage_uri() -> String {
    match std::env::var("HOME") {
        Ok(home) => format!("file://{home}/.hyperstreamdb/search"),
        Err(_) => "file:///tmp/.hyperstreamdb/search".to_string(),
    }
}

/// Resolve `HYPERSEARCH_STORAGE_URI` (defaulting to `file://~/.hyperstreamdb/search`),
/// expanding a leading `~` into `$HOME`.
pub fn resolve_storage_uri() -> String {
    let raw = std::env::var("HYPERSEARCH_STORAGE_URI").unwrap_or_else(|_| default_storage_uri());
    let trimmed = raw.trim_end_matches('/');
    if trimmed == "~" || trimmed.starts_with("~/") {
        let home = std::env::var("HOME").unwrap_or_default();
        format!("{home}{}", &trimmed[1..])
    } else {
        trimmed.to_string()
    }
}

/// Probe for an initialized table: the Iceberg metadata version hint under
/// the table root. The object store is prefix-scoped to the table directory,
/// so the path is relative.
pub(crate) async fn table_exists(uri: &str) -> bool {
    match hyperstreamdb::core::storage::create_object_store(uri) {
        Ok(store) => store
            .head(&object_store::path::Path::from(
                "metadata/version-hint.text",
            ))
            .await
            .is_ok(),
        Err(_) => false,
    }
}

fn empty_schema() -> SchemaRef {
    Arc::new(arrow::datatypes::Schema::empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn index_uri_joins_root_and_name() {
        let state = AppState::new("/data/search".to_string(), "uuid".to_string());
        assert_eq!(state.index_uri("people"), "/data/search/people");
    }

    #[test]
    fn index_uri_trims_trailing_slash_on_root() {
        let state = AppState::new("/data/search/".to_string(), "uuid".to_string());
        assert_eq!(state.index_uri("people"), "/data/search/people");
    }

    #[test]
    fn resolve_storage_uri_expands_tilde() {
        std::env::set_var("HYPERSEARCH_STORAGE_URI", "~/searches");
        let home = std::env::var("HOME").unwrap_or_default();
        assert_eq!(resolve_storage_uri(), format!("{home}/searches"));
        std::env::remove_var("HYPERSEARCH_STORAGE_URI");
    }
}
