// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Graph-scoped search endpoint: `POST /{index}/_graph_search`.
//!
//! Pre-computes a graph neighborhood from an edge table using BFS, then
//! delegates to the standard `search_core()` pipeline with an injected
//! `terms` filter restricting results to the neighbor set.

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use arrow::array::{Array, UInt64Array};
use axum::extract::{Path, State};
use axum::response::Response;
use axum::Json;
use serde_json::Value;

use crate::handlers::{es_response, search};
use crate::state::{table_exists, AppState};
use hyperstreamdb::HyperstreamError;

/// `POST /{index}/_graph_search` — graph-neighborhood-scoped search.
///
/// Computes the k-hop neighborhood of `seed_ids` in `edge_index`, then
/// runs the standard `_search` pipeline restricted to those IDs.
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
) -> Result<crate::es_types::SearchResponse, HyperstreamError> {
    let obj = body
        .as_object()
        .ok_or_else(|| HyperstreamError::SchemaIncompatible {
            reason: "request body must be a JSON object".into(),
        })?;

    // ── Extract graph parameters ──────────────────────────────────────
    let edge_index = obj
        .get("edge_index")
        .and_then(Value::as_str)
        .ok_or_else(|| HyperstreamError::SchemaIncompatible {
            reason: "graph_search: 'edge_index' (string) is required".into(),
        })?;

    let seed_ids: Vec<u64> = obj
        .get("seed_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| HyperstreamError::SchemaIncompatible {
            reason: "graph_search: 'seed_ids' (array of integers) is required".into(),
        })?
        .iter()
        .filter_map(Value::as_u64)
        .collect();

    if seed_ids.is_empty() {
        return Err(HyperstreamError::SchemaIncompatible {
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

    // ── Resolve graph neighborhood ────────────────────────────────────
    let edge_uri = state.index_uri(edge_index);
    if !table_exists(&edge_uri).await {
        return Err(HyperstreamError::TableNotFound {
            namespace: String::new(),
            name: edge_index.to_string(),
        });
    }
    let edge_table = state.open_or_create(edge_index, &None).await?;

    // Build SQL filter for allowed_relations if specified.
    let rel_filter = if let Some(ref rels) = allowed_relations {
        let schema = edge_table.arrow_schema();
        let rel_col = ["relation", "predicate", "type", "rel", "edge_type"]
            .iter()
            .find(|c| schema.column_with_name(c).is_some())
            .map(|s| s.to_string());
        if let Some(col) = rel_col {
            let vals: Vec<String> = rels
                .iter()
                .map(|r| format!("'{}'", r.replace('\'', "''")))
                .collect();
            Some(format!("{col} IN ({})", vals.join(", ")))
        } else {
            None
        }
    } else {
        None
    };

    // Read edges from the edge table (with optional relation filter).
    let batches = edge_table
        .read_async(rel_filter.as_deref(), None, None)
        .await
        .map_err(|e| HyperstreamError::internal(e.to_string()))?;

    // Build adjacency list from source/target columns.
    let mut adj: std::collections::HashMap<u64, Vec<u64>> = std::collections::HashMap::new();
    for batch in &batches {
        let schema = batch.schema();
        let src_idx = schema.index_of("source").ok();
        let tgt_idx = schema.index_of("target").ok();
        let (src_idx, tgt_idx) = match (src_idx, tgt_idx) {
            (Some(s), Some(t)) => (s, t),
            _ => continue,
        };
        let src_col = batch.column(src_idx);
        let tgt_col = batch.column(tgt_idx);

        // Try UInt64 first, then Int64 with cast.
        let sources = extract_u64_values(src_col.as_ref());
        let targets = extract_u64_values(tgt_col.as_ref());

        for i in 0..batch.num_rows() {
            if let (Some(Some(s)), Some(Some(t))) =
                (sources.get(i).copied(), targets.get(i).copied())
            {
                adj.entry(s).or_default().push(t);
                if !directed {
                    adj.entry(t).or_default().push(s);
                }
            }
        }
    }

    // BFS from seed nodes up to `hops` depth.
    let mut visited: HashSet<u64> = HashSet::new();
    let mut queue: VecDeque<(u64, u32)> = VecDeque::new();
    for &seed in &seed_ids {
        if visited.insert(seed) {
            queue.push_back((seed, 0));
        }
    }
    while let Some((node, depth)) = queue.pop_front() {
        if depth >= hops {
            continue;
        }
        if let Some(neighbors) = adj.get(&node) {
            for &nbr in neighbors {
                if visited.insert(nbr) {
                    queue.push_back((nbr, depth + 1));
                }
            }
        }
    }

    if visited.is_empty() {
        // No neighbors found — return empty search response.
        return Ok(crate::es_types::SearchResponse {
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

/// Extract u64 values from an Arrow array (supports UInt64, Int64, UInt32, Int32).
fn extract_u64_values(col: &dyn Array) -> Vec<Option<u64>> {
    let n = col.len();
    let mut out = Vec::with_capacity(n);

    if let Some(arr) = col.as_any().downcast_ref::<UInt64Array>() {
        for i in 0..n {
            out.push(if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i))
            });
        }
    } else if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int64Array>() {
        for i in 0..n {
            out.push(if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i) as u64)
            });
        }
    } else if let Some(arr) = col.as_any().downcast_ref::<arrow::array::UInt32Array>() {
        for i in 0..n {
            out.push(if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i) as u64)
            });
        }
    } else if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int32Array>() {
        for i in 0..n {
            out.push(if arr.is_null(i) {
                None
            } else {
                Some(arr.value(i) as u64)
            });
        }
    } else {
        for _ in 0..n {
            out.push(None);
        }
    }
    out
}
