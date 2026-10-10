// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Index CRUD endpoints: `PUT /{index}`, `GET /{index}`, `DELETE /{index}`.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use axum::extract::{Path, State};
use axum::response::Response;
use axum::Json;
use benostreamdb::{BenoStreamError, Table};
use serde_json::{Map, Value};

use crate::handlers::mapping::{get_mapping_core, schema_from_mapping};
use crate::state::AppState;

use super::es_response;

/// `PUT /{index}` — create an index, optionally with an explicit mapping.
/// A pre-existing index is a 400 `resource_already_exists_exception`.
pub async fn create_index(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    body: Option<Json<Value>>,
) -> Response {
    es_response(create_index_core(&state, &index, body.as_deref()).await)
}

pub(crate) async fn create_index_core(
    state: &AppState,
    index: &str,
    body: Option<&Value>,
) -> Result<Value, BenoStreamError> {
    AppState::validate_index_name(index)?;

    if state.index_exists(index).await {
        return Err(BenoStreamError::PrimaryKeyViolation {
            key: format!("index '{index}' already exists"),
        });
    }

    // Build the initial schema from an optional `mappings.properties`.
    let properties = body
        .and_then(|b| b.get("mappings"))
        .and_then(|m| m.get("properties"))
        .and_then(Value::as_object);
    let schema: SchemaRef = schema_from_mapping(properties)?;

    let uri = state.resolve_table_uri(index).await;
    Table::create_async(uri.clone(), schema.clone())
        .await
        .map_err(|e| {
            // Lost a create race with a concurrent request.
            if e.to_string().contains("already exists") {
                return BenoStreamError::PrimaryKeyViolation {
                    key: format!("index '{index}' already exists"),
                };
            }
            BenoStreamError::internal(format!("failed to create index '{index}': {e}"))
        })?;

    // Register in the external catalog if configured
    if let Some(catalog) = &state.catalog {
        if let Ok(false) = catalog.table_exists(&state.catalog_namespace, index).await {
            if let Err(e) = catalog
                .create_table(&state.catalog_namespace, index, schema, Some(&uri))
                .await
            {
                tracing::warn!(
                    index = %index,
                    namespace = %state.catalog_namespace,
                    error = %e,
                    "Failed to register index in external catalog on PUT"
                );
            } else {
                tracing::info!(
                    index = %index,
                    namespace = %state.catalog_namespace,
                    "Registered new index in external Iceberg catalog on PUT"
                );
            }
        }
    }

    // Open the shared, indexing-enabled handle so subsequent writes/searches
    // reuse one Table instance.
    state.open_or_create(index, &None).await?;

    Ok(Value::Object({
        let mut m = Map::new();
        m.insert("acknowledged".into(), Value::Bool(true));
        m.insert("shards_acknowledged".into(), Value::Bool(true));
        m.insert("index".into(), Value::String(index.to_string()));
        m
    }))
}

/// `GET /{index}` — index metadata (aliases, mappings, settings).
pub async fn get_index(State(state): State<Arc<AppState>>, Path(index): Path<String>) -> Response {
    es_response(get_index_core(&state, &index).await)
}

pub(crate) async fn get_index_core(
    state: &AppState,
    index: &str,
) -> Result<Value, BenoStreamError> {
    if !state.index_exists(index).await {
        return Err(BenoStreamError::TableNotFound {
            namespace: String::new(),
            name: index.to_string(),
        });
    }
    // Reuse the mapping renderer, then wrap it in the full index envelope.
    let mapping_root = get_mapping_core(state, index).await?;
    let idx_meta = mapping_root
        .get(index)
        .cloned()
        .ok_or_else(|| BenoStreamError::internal("missing index metadata"))?;

    let mut idx_meta = idx_meta.as_object().cloned().unwrap_or_default();
    let resolved = state.resolve_alias(index).await;
    let mut aliases_obj = Map::new();
    let aliases = state.aliases.read().await;
    for (alias_name, target) in aliases.iter() {
        if target == &resolved || target == index {
            aliases_obj.insert(alias_name.clone(), Value::Object(Map::new()));
        }
    }
    idx_meta.insert("aliases".into(), Value::Object(aliases_obj));

    let mut settings = Map::new();
    let mut index_settings = Map::new();
    index_settings.insert("number_of_shards".into(), Value::String("1".into()));
    index_settings.insert("number_of_replicas".into(), Value::String("0".into()));
    settings.insert("index".into(), Value::Object(index_settings));
    idx_meta.insert("settings".into(), Value::Object(settings));

    let mut root = Map::new();
    root.insert(index.to_string(), Value::Object(idx_meta));
    Ok(Value::Object(root))
}

/// `DELETE /{index}` — hard-delete the index (all store objects removed).
pub async fn delete_index(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
) -> Response {
    es_response(delete_index_core(&state, &index).await)
}

pub(crate) async fn delete_index_core(
    state: &AppState,
    index: &str,
) -> Result<Value, BenoStreamError> {
    AppState::validate_index_name(index)?;

    if !state.index_exists(index).await {
        return Err(BenoStreamError::TableNotFound {
            namespace: String::new(),
            name: index.to_string(),
        });
    }
    state.delete_index(index).await?;
    Ok(Value::Object({
        let mut m = Map::new();
        m.insert("acknowledged".into(), Value::Bool(true));
        m
    }))
}

/// `POST /_aliases` — batch add/remove index aliases.
pub async fn post_aliases(
    State(state): State<Arc<AppState>>,
    axum::Json(body): axum::Json<Value>,
) -> Response {
    es_response(post_aliases_core(&state, body).await)
}

pub(crate) async fn post_aliases_core(
    state: &AppState,
    body: Value,
) -> Result<Value, BenoStreamError> {
    let actions = body
        .get("actions")
        .and_then(Value::as_array)
        .ok_or_else(|| BenoStreamError::SchemaIncompatible {
            reason: "missing 'actions' array in _aliases request".into(),
        })?;

    let mut aliases = state.aliases.write().await;
    for act in actions {
        if let Some(add) = act.get("add") {
            let index = add.get("index").and_then(Value::as_str).ok_or_else(|| {
                BenoStreamError::SchemaIncompatible {
                    reason: "add action missing 'index'".into(),
                }
            })?;
            let alias = add.get("alias").and_then(Value::as_str).ok_or_else(|| {
                BenoStreamError::SchemaIncompatible {
                    reason: "add action missing 'alias'".into(),
                }
            })?;
            AppState::validate_index_name(alias)?;
            aliases.insert(alias.to_string(), index.to_string());
        } else if let Some(remove) = act.get("remove") {
            let alias = remove.get("alias").and_then(Value::as_str).ok_or_else(|| {
                BenoStreamError::SchemaIncompatible {
                    reason: "remove action missing 'alias'".into(),
                }
            })?;
            aliases.remove(alias);
        }
    }

    Ok(serde_json::json!({ "acknowledged": true }))
}

/// `GET /_alias` — get all registered aliases.
pub async fn get_all_aliases(State(state): State<Arc<AppState>>) -> Response {
    es_response(get_all_aliases_core(&state).await)
}

pub(crate) async fn get_all_aliases_core(state: &AppState) -> Result<Value, BenoStreamError> {
    let aliases = state.aliases.read().await;
    let mut out = Map::new();
    for (alias, index) in aliases.iter() {
        let entry = out
            .entry(index.clone())
            .or_insert_with(|| serde_json::json!({ "aliases": {} }));
        if let Some(m) = entry.get_mut("aliases").and_then(Value::as_object_mut) {
            m.insert(alias.clone(), serde_json::json!({}));
        }
    }
    Ok(Value::Object(out))
}

/// `GET /{index}/_alias` — get aliases for a specific index.
pub async fn get_index_aliases(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
) -> Response {
    es_response(get_index_aliases_core(&state, &index).await)
}

pub(crate) async fn get_index_aliases_core(
    state: &AppState,
    index: &str,
) -> Result<Value, BenoStreamError> {
    let resolved = state.resolve_alias(index).await;
    if !state.index_exists(&resolved).await {
        return Err(BenoStreamError::TableNotFound {
            namespace: String::new(),
            name: index.to_string(),
        });
    }

    let aliases = state.aliases.read().await;
    let mut alias_map = Map::new();
    for (alias, target) in aliases.iter() {
        if target == &resolved || target == index {
            alias_map.insert(alias.clone(), serde_json::json!({}));
        }
    }

    let mut out = Map::new();
    out.insert(
        resolved,
        serde_json::json!({ "aliases": Value::Object(alias_map) }),
    );
    Ok(Value::Object(out))
}

/// `PUT /{index}/_alias/{alias}` — create a single alias.
pub async fn put_single_alias(
    State(state): State<Arc<AppState>>,
    Path((index, alias)): Path<(String, String)>,
) -> Response {
    es_response(put_single_alias_core(&state, &index, &alias).await)
}

pub(crate) async fn put_single_alias_core(
    state: &AppState,
    index: &str,
    alias: &str,
) -> Result<Value, BenoStreamError> {
    AppState::validate_index_name(alias)?;

    if !state.index_exists(index).await {
        return Err(BenoStreamError::TableNotFound {
            namespace: String::new(),
            name: index.to_string(),
        });
    }
    state
        .aliases
        .write()
        .await
        .insert(alias.to_string(), index.to_string());
    Ok(serde_json::json!({ "acknowledged": true }))
}

/// `DELETE /{index}/_alias/{alias}` — delete a single alias.
pub async fn delete_single_alias(
    State(state): State<Arc<AppState>>,
    Path((_index, alias)): Path<(String, String)>,
) -> Response {
    state.aliases.write().await.remove(&alias);
    es_response(Ok(serde_json::json!({ "acknowledged": true })))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use benostreamdb::Catalog;
    use serde_json::json;

    #[tokio::test]
    async fn test_alias_lifecycle_and_resolution() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());

        // 1. Create index
        let create_res = create_index_core(&state, "logs-v1", None).await.unwrap();
        assert_eq!(create_res["acknowledged"], true);

        // 2. Put single alias "logs" -> "logs-v1"
        let alias_res = put_single_alias_core(&state, "logs-v1", "logs")
            .await
            .unwrap();
        assert_eq!(alias_res["acknowledged"], true);

        // Verify resolution
        assert_eq!(state.resolve_alias("logs").await, "logs-v1");
        assert_eq!(state.resolve_alias("logs-v1").await, "logs-v1");

        // 3. Get index metadata includes alias
        let idx_meta = get_index_core(&state, "logs-v1").await.unwrap();
        let aliases_obj = idx_meta["logs-v1"]["aliases"].as_object().unwrap();
        assert!(aliases_obj.contains_key("logs"));

        // 4. Get index aliases
        let idx_aliases = get_index_aliases_core(&state, "logs-v1").await.unwrap();
        assert!(idx_aliases["logs-v1"]["aliases"]["logs"].is_object());

        // 5. Get all aliases
        let all_aliases = get_all_aliases_core(&state).await.unwrap();
        assert!(all_aliases["logs-v1"]["aliases"]["logs"].is_object());

        // 6. Post aliases: atomic swap logs -> logs-v2
        create_index_core(&state, "logs-v2", None).await.unwrap();
        let swap_body = json!({
            "actions": [
                { "remove": { "index": "logs-v1", "alias": "logs" } },
                { "add": { "index": "logs-v2", "alias": "logs" } }
            ]
        });
        let post_res = post_aliases_core(&state, swap_body).await.unwrap();
        assert_eq!(post_res["acknowledged"], true);

        assert_eq!(state.resolve_alias("logs").await, "logs-v2");

        // 7. Delete index cleans up aliases
        delete_index_core(&state, "logs-v2").await.unwrap();
        assert_eq!(state.resolve_alias("logs").await, "logs");
    }

    #[derive(Default)]
    struct MockCatalog {
        tables: std::sync::RwLock<std::collections::HashMap<String, Vec<String>>>,
        dropped: std::sync::RwLock<Vec<String>>,
    }

    #[async_trait]
    impl benostreamdb::core::catalog::Catalog for MockCatalog {
        async fn create_table(
            &self,
            namespace: &str,
            table_name: &str,
            _schema: SchemaRef,
            _location: Option<&str>,
        ) -> anyhow::Result<()> {
            self.tables
                .write()
                .unwrap()
                .entry(namespace.to_string())
                .or_default()
                .push(table_name.to_string());
            Ok(())
        }

        async fn drop_table(&self, namespace: &str, table_name: &str) -> anyhow::Result<()> {
            self.dropped.write().unwrap().push(table_name.to_string());
            if let Some(list) = self.tables.write().unwrap().get_mut(namespace) {
                list.retain(|t| t != table_name);
            }
            Ok(())
        }

        async fn load_table(
            &self,
            _namespace: &str,
            _table_name: &str,
        ) -> anyhow::Result<benostreamdb::core::metadata::TableMetadata> {
            Err(anyhow::anyhow!("not found"))
        }

        async fn create_branch(&self, _branch: &str, _source: Option<&str>) -> anyhow::Result<()> {
            Ok(())
        }

        async fn table_exists(&self, namespace: &str, table_name: &str) -> anyhow::Result<bool> {
            let tables = self.tables.read().unwrap();
            Ok(tables
                .get(namespace)
                .map(|list| list.contains(&table_name.to_string()))
                .unwrap_or(false))
        }

        async fn list_tables(&self, namespace: &str) -> anyhow::Result<Vec<String>> {
            let tables = self.tables.read().unwrap();
            Ok(tables.get(namespace).cloned().unwrap_or_default())
        }

        async fn commit_table(
            &self,
            _namespace: &str,
            _table_name: &str,
            _updates: Vec<serde_json::Value>,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_catalog_integration_create_list_get_delete() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let mock_cat = Arc::new(MockCatalog::default());
        let state = AppState::with_catalog(
            root,
            "test-cluster".into(),
            benostreamdb::core::index::gpu::ComputeContext::auto_detect(),
            Some(mock_cat.clone()),
            "analytics".to_string(),
        );

        // 1. Create index via PUT /{index} -> registers in catalog
        let create_res = create_index_core(&state, "catalog_idx", None)
            .await
            .unwrap();
        assert_eq!(create_res["acknowledged"], true);

        // Verify registered in MockCatalog
        assert!(mock_cat
            .table_exists("analytics", "catalog_idx")
            .await
            .unwrap());

        // 2. Index exists returns true
        assert!(state.index_exists("catalog_idx").await);

        // 3. List indexes includes catalog index
        let indexes = state.list_indexes().await.unwrap();
        assert!(indexes.contains(&"catalog_idx".to_string()));

        // 4. GET index metadata succeeds
        let get_res = get_index_core(&state, "catalog_idx").await.unwrap();
        assert!(get_res.get("catalog_idx").is_some());

        // 5. Index and retrieve a doc (POST / GET)
        let doc = json!({ "title": "BenoStreamDB search with Iceberg Catalog" });
        let write_res =
            crate::handlers::docs::index_document_core(&state, "catalog_idx", Some("doc-1"), doc)
                .await
                .unwrap();
        assert_eq!(write_res.result, "updated");

        let doc_get = crate::handlers::docs::get_document_core(&state, "catalog_idx", "doc-1")
            .await
            .unwrap();
        assert!(doc_get.found);

        // 6. Delete doc (DELETE doc)
        let doc_del = crate::handlers::docs::delete_document_core(&state, "catalog_idx", "doc-1")
            .await
            .unwrap();
        assert_eq!(doc_del.result, "deleted");

        // 7. Delete index via DELETE /{index} -> drops from catalog
        let del_res = delete_index_core(&state, "catalog_idx").await.unwrap();
        assert_eq!(del_res["acknowledged"], true);

        // Verify dropped from MockCatalog
        assert_eq!(
            mock_cat.dropped.read().unwrap().as_slice(),
            &["catalog_idx"]
        );
        assert!(!mock_cat
            .table_exists("analytics", "catalog_idx")
            .await
            .unwrap());
        assert!(!state.index_exists("catalog_idx").await);
    }
}
