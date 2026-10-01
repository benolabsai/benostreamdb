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
use crate::state::{table_exists, AppState};

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
    let uri = state.index_uri(index);
    if table_exists(&uri).await {
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

    Table::create_async(uri.clone(), schema)
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
    if !table_exists(&state.index_uri(index)).await {
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
    if !table_exists(&state.index_uri(index)).await {
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
    if !table_exists(&state.index_uri(&resolved)).await {
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
    if !table_exists(&state.index_uri(index)).await {
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
}
