// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Graph-scoped search endpoint: `POST /{index}/_graph_search`.
//!
//! Delegates graph neighborhood expansion to core `Table::graph_neighborhood()`,
//! then passes the resulting ID set into the standard `search_core()` pipeline
//! with an injected `terms` filter.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::response::Response;
use axum::Json;
use serde_json::Value;

use crate::handlers::{es_response, search};
use crate::state::AppState;
use benostreamdb::{BenoStreamError, GraphNeighborhoodOptions};

/// `POST /{index}/_graph_search` — graph-neighborhood-scoped search.
///
/// Computes the k-hop neighborhood of `seed_ids` in `edge_index` using
/// the core `Table::graph_neighborhood()` operator, then runs the standard
/// `_search` pipeline restricted to those IDs.
pub async fn graph_search(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    es_response(graph_search_core(&state, &index, &body).await)
}

async fn graph_search_core(
    state: &AppState,
    index: &str,
    body: &Value,
) -> Result<crate::es_types::SearchResponse, BenoStreamError> {
    let obj = body
        .as_object()
        .ok_or_else(|| BenoStreamError::SchemaIncompatible {
            reason: "request body must be a JSON object".into(),
        })?;

    // ── Extract graph parameters ──────────────────────────────────────
    let edge_index = obj
        .get("edge_index")
        .and_then(Value::as_str)
        .ok_or_else(|| BenoStreamError::SchemaIncompatible {
            reason: "graph_search: 'edge_index' (string) is required".into(),
        })?;

    let seed_ids: Vec<u64> = obj
        .get("seed_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| BenoStreamError::SchemaIncompatible {
            reason: "graph_search: 'seed_ids' (array of integers) is required".into(),
        })?
        .iter()
        .filter_map(Value::as_u64)
        .collect();

    if seed_ids.is_empty() {
        return Err(BenoStreamError::SchemaIncompatible {
            reason: "graph_search: 'seed_ids' must contain at least one integer".into(),
        });
    }

    let hops = obj.get("hops").and_then(Value::as_u64).unwrap_or(1) as u32;
    let directed = obj
        .get("directed")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let allowed_relations: Option<Vec<String>> = obj
        .get("allowed_relations")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(Value::as_str)
                .map(String::from)
                .collect()
        });
    let id_field = obj.get("id_field").and_then(Value::as_str).unwrap_or("_id");

    // ── Resolve graph neighborhood via core Table operator ────────────
    if !state.index_exists(edge_index).await {
        return Err(BenoStreamError::TableNotFound {
            namespace: String::new(),
            name: edge_index.to_string(),
        });
    }
    let edge_table = state.open_or_create(edge_index, &None).await?;

    let graph_options = GraphNeighborhoodOptions {
        seeds: seed_ids,
        hops,
        directed,
        allowed_relations,
        ..Default::default()
    };

    let visited = edge_table
        .graph_neighborhood(&graph_options)
        .await
        .map_err(|e| BenoStreamError::internal(e.to_string()))?;

    if visited.is_empty() {
        // No neighbors found — return empty search response.
        return Ok(crate::es_types::SearchResponse {
            scroll_id: None,
            took: 0,
            timed_out: false,
            hits: crate::es_types::SearchHits {
                total: crate::es_types::TotalHits {
                    value: 0,
                    relation: "eq".to_string(),
                },
                max_score: None,
                hits: vec![],
            },
            aggregations: None,
        });
    }

    // ── Inject terms filter into the search body ──────────────────────
    let neighbor_ids: Vec<Value> = visited
        .iter()
        .map(|id| Value::String(id.to_string()))
        .collect();

    let terms_filter = serde_json::json!({
        "terms": { id_field: neighbor_ids }
    });

    // Clone the body and inject/merge the graph filter.
    let mut search_body = body.clone();
    if let Some(obj) = search_body.as_object_mut() {
        // Remove graph-specific keys that search_core doesn't understand.
        obj.remove("edge_index");
        obj.remove("seed_ids");
        obj.remove("hops");
        obj.remove("directed");
        obj.remove("allowed_relations");
        obj.remove("id_field");

        // Merge the terms filter with any existing filter using bool/must.
        if let Some(existing_filter) = obj.remove("filter") {
            obj.insert(
                "filter".to_string(),
                serde_json::json!({
                    "bool": {
                        "must": [existing_filter, terms_filter]
                    }
                }),
            );
        } else {
            obj.insert("filter".to_string(), terms_filter);
        }
    }

    // ── Delegate to standard search pipeline ──────────────────────────
    search::search_core(state, index, &search_body).await
}
