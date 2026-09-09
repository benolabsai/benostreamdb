// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Index CRUD endpoints: `PUT /{index}`, `GET /{index}`, `DELETE /{index}`.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use axum::extract::{Path, State};
use axum::response::Response;
use axum::Json;
use hyperstreamdb::{HyperstreamError, Table};
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
) -> Result<Value, HyperstreamError> {
    let uri = state.index_uri(index);
    if table_exists(&uri).await {
        return Err(HyperstreamError::PrimaryKeyViolation {
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
                return HyperstreamError::PrimaryKeyViolation {
                    key: format!("index '{index}' already exists"),
                };
            }
            HyperstreamError::internal(format!("failed to create index '{index}': {e}"))
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
) -> Result<Value, HyperstreamError> {
    if !table_exists(&state.index_uri(index)).await {
        return Err(HyperstreamError::TableNotFound {
            namespace: String::new(),
            name: index.to_string(),
        });
    }
    // Reuse the mapping renderer, then wrap it in the full index envelope.
    let mapping_root = get_mapping_core(state, index).await?;
    let idx_meta = mapping_root
        .get(index)
        .cloned()
        .ok_or_else(|| HyperstreamError::internal("missing index metadata"))?;

    let mut idx_meta = idx_meta.as_object().cloned().unwrap_or_default();
    idx_meta.insert("aliases".into(), Value::Object(Map::new()));
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
) -> Result<Value, HyperstreamError> {
    if !table_exists(&state.index_uri(index)).await {
        return Err(HyperstreamError::TableNotFound {
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
