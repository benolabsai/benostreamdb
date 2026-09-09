// Copyright (c) 2026 Richard Albright. All rights reserved.

//! ES `_bulk` endpoints: `POST /_bulk` and `POST /{index}/_bulk`.
//!
//! Accepts an NDJSON body of alternating action / source lines, groups the
//! writes by index, batches them through `Table::write_async`, and reports a
//! per-item status. `delete` actions are rejected per-item with a 501 (the
//! store is append-only in v1).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::RecordBatch;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::response::Response;
use hyperstreamdb::HyperstreamError;
use serde_json::{Map, Value};

use crate::handlers::docs::{build_row_batch, translate_write_error, with_id, ID_COLUMN};
use crate::infer;
use crate::state::{table_exists, AppState};

use super::es_response;

fn bad_request(reason: impl Into<String>) -> HyperstreamError {
    HyperstreamError::SchemaIncompatible {
        reason: reason.into(),
    }
}

/// One parsed bulk action.
#[derive(Clone)]
struct BulkItem {
    op: String, // "index" | "create" | "delete"
    index: String,
    id: String,
    doc: Option<Value>,
}

/// Parse an NDJSON bulk body into a list of [`BulkItem`]s.
fn parse_bulk(body: &str, default_index: Option<&str>) -> Result<Vec<BulkItem>, HyperstreamError> {
    let lines: Vec<&str> = body
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();
    let mut items = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let action: Value = serde_json::from_str(lines[i])
            .map_err(|e| bad_request(format!("bulk line {}: invalid action JSON: {e}", i + 1)))?;
        let (op, meta) = match action.as_object().and_then(|o| o.iter().next()) {
            Some((op, meta)) => (op.clone(), meta.clone()),
            None => {
                return Err(bad_request(format!(
                    "bulk line {}: expected a single action object",
                    i + 1
                )))
            }
        };
        if !matches!(op.as_str(), "index" | "create" | "delete") {
            return Err(bad_request(format!(
                "bulk line {}: unsupported action '{op}' (supported: index, create, delete)",
                i + 1
            )));
        }
        let meta_obj = meta.as_object().ok_or_else(|| {
            bad_request(format!(
                "bulk line {}: action meta must be an object",
                i + 1
            ))
        })?;
        let index = meta_obj
            .get("_index")
            .and_then(Value::as_str)
            .or(default_index)
            .ok_or_else(|| {
                bad_request(format!(
                    "bulk line {}: missing '_index' (use POST /<index>/_bulk or set _index)",
                    i + 1
                ))
            })?
            .to_string();
        let id = meta_obj
            .get("_id")
            .and_then(Value::as_str)
            .map(str::to_string)
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        let doc = if op == "delete" {
            // delete has no source line.
            None
        } else {
            i += 1;
            if i >= lines.len() {
                return Err(bad_request(format!(
                    "bulk: action on line {} is missing its source document",
                    i
                )));
            }
            let source: Value = serde_json::from_str(lines[i]).map_err(|e| {
                bad_request(format!("bulk line {}: invalid source JSON: {e}", i + 1))
            })?;
            Some(source)
        };

        items.push(BulkItem { op, index, id, doc });
        i += 1;
    }
    Ok(items)
}

/// A per-item bulk outcome.
struct ItemResult {
    op: String,
    index: String,
    id: String,
    status: u16,
    result: String,
    error: Option<String>,
}

impl ItemResult {
    fn to_value(&self) -> Value {
        let mut obj = Map::new();
        obj.insert("_index".into(), Value::String(self.index.clone()));
        obj.insert("_id".into(), Value::String(self.id.clone()));
        obj.insert("status".into(), Value::from(self.status));
        obj.insert("result".into(), Value::String(self.result.clone()));
        if let Some(e) = &self.error {
            obj.insert("error".into(), Value::String(e.clone()));
        }
        let mut outer = Map::new();
        outer.insert(self.op.clone(), Value::Object(obj));
        Value::Object(outer)
    }
}

/// Write the documents for one index in a single batched `write_async`,
/// falling back to per-document writes on a primary-key violation so the
/// offending item can be reported individually. Per-item schema/build errors
/// are reported without aborting the rest of the batch.
async fn write_index_docs(
    state: &AppState,
    index: &str,
    items: &[(usize, &BulkItem)],
    existed_before: bool,
) -> Vec<(usize, ItemResult)> {
    let err_result = |pos: usize, item: &BulkItem, status: u16, error: String| {
        (
            pos,
            ItemResult {
                op: item.op.clone(),
                index: index.to_string(),
                id: item.id.clone(),
                status,
                result: "error".into(),
                error: Some(error),
            },
        )
    };
    let ok_result = |pos: usize, item: &BulkItem| {
        (
            pos,
            ItemResult {
                op: item.op.clone(),
                index: index.to_string(),
                id: item.id.clone(),
                status: if existed_before { 200 } else { 201 },
                result: if existed_before { "updated" } else { "created" }.into(),
                error: None,
            },
        )
    };

    // 1. Infer each document's schema, collecting per-item errors.
    let mut doc_schemas: Vec<(usize, &BulkItem, arrow::datatypes::SchemaRef)> = Vec::new();
    let mut results: Vec<(usize, ItemResult)> = Vec::new();
    for (pos, item) in items {
        match infer::infer_schema(item.doc.as_ref().unwrap()) {
            Ok(s) => doc_schemas.push((*pos, item, s)),
            Err(e) => results.push(err_result(*pos, item, 400, e.to_string())),
        }
    }
    if doc_schemas.is_empty() {
        return results;
    }

    // 2. Merge the schemas of all documents.
    let mut merged = doc_schemas[0].2.clone();
    for (_pos, _item, s) in &doc_schemas[1..] {
        match infer::merge_schemas(merged.as_ref(), s.as_ref()) {
            Ok(m) => merged = Arc::new(m),
            Err(e) => {
                for (pos, item, _s) in doc_schemas {
                    results.push(err_result(pos, item, 400, e.to_string()));
                }
                return results;
            }
        }
    }

    // 3. Open/create the table.
    let table = match state.open_or_create(index, &Some(merged.clone())).await {
        Ok(t) => t,
        Err(e) => {
            for (pos, item, _s) in doc_schemas {
                results.push(err_result(pos, item, 500, e.to_string()));
            }
            return results;
        }
    };

    // A table created by this request gets `_id` as its primary key.
    if !existed_before && table.get_primary_key().is_empty() {
        if let Err(e) = table
            .set_primary_key_async(vec![ID_COLUMN.to_string()])
            .await
        {
            tracing::warn!(error = %e, "failed to set _id primary key on new index '{index}'");
        }
    }

    // 4. Merge the table's current schema with the document schema.
    let target = match infer::merge_schemas(table.arrow_schema().as_ref(), merged.as_ref()) {
        Ok(t) => t,
        Err(e) => {
            for (pos, item, _s) in doc_schemas {
                results.push(err_result(pos, item, 400, e.to_string()));
            }
            return results;
        }
    };

    // 5. Build one row batch per document (with `_id` injected).
    let mut batches: Vec<(usize, &BulkItem, RecordBatch)> = Vec::new();
    for (pos, item, _s) in doc_schemas {
        let mut doc = item.doc.clone().unwrap();
        if let Err(e) = with_id(&mut doc, &item.id) {
            results.push(err_result(pos, item, 400, e.to_string()));
            continue;
        }
        match build_row_batch(&target, &doc) {
            Ok(batch) => batches.push((pos, item, batch)),
            Err(e) => results.push(err_result(pos, item, 400, e.to_string())),
        }
    }
    if batches.is_empty() {
        return results;
    }

    // 6. Batch write; on failure retry per-document to identify the cause.
    let batch_vec: Vec<RecordBatch> = batches.iter().map(|(_, _, b)| b.clone()).collect();
    match table.write_async(batch_vec).await {
        Ok(()) => {
            for (pos, item, _b) in batches {
                results.push(ok_result(pos, item));
            }
        }
        Err(_e) => {
            for (pos, item, batch) in batches {
                match table.write_async(vec![batch]).await {
                    Ok(()) => results.push(ok_result(pos, item)),
                    Err(e2) => {
                        let t2 = translate_write_error(e2);
                        results.push(err_result(pos, item, es_status(&t2), t2.to_string()));
                    }
                }
            }
        }
    }
    results
}

/// Map a typed write error to its ES status code.
fn es_status(err: &HyperstreamError) -> u16 {
    match err {
        HyperstreamError::PrimaryKeyViolation { .. } => 400,
        HyperstreamError::NullConstraintViolation { .. } => 400,
        HyperstreamError::SchemaIncompatible { .. } => 400,
        _ => 500,
    }
}

/// Core bulk dispatch shared by `POST /_bulk` and `POST /{index}/_bulk`.
pub async fn bulk_core(
    state: &AppState,
    default_index: Option<&str>,
    body: &str,
) -> Result<Value, HyperstreamError> {
    let started = Instant::now();
    let items = parse_bulk(body, default_index)?;

    // Group by index, preserving original positions.
    let mut by_index: BTreeMap<String, Vec<(usize, BulkItem)>> = BTreeMap::new();
    for (pos, item) in items.iter().enumerate() {
        by_index
            .entry(item.index.clone())
            .or_default()
            .push((pos, item.clone()));
    }

    let mut results: Vec<Option<ItemResult>> = (0..items.len()).map(|_| None).collect();

    for (index, group) in by_index {
        let existed_before = table_exists(&state.index_uri(&index)).await;
        // Split off deletes (no source) — they are 501 in v1.
        let writes: Vec<(usize, &BulkItem)> = group
            .iter()
            .filter(|(_, it)| it.op != "delete")
            .map(|(pos, it)| (*pos, it))
            .collect();
        for (pos, it) in group.iter().filter(|(_, it)| it.op == "delete") {
            results[*pos] = Some(ItemResult {
                op: it.op.clone(),
                index: index.clone(),
                id: it.id.clone(),
                status: 501,
                result: "error".into(),
                error: Some(
                    "per-document delete is not supported (append-only store); use DELETE /{index}"
                        .into(),
                ),
            });
        }
        if writes.is_empty() {
            continue;
        }
        let write_results = write_index_docs(state, &index, &writes, existed_before).await;
        for (pos, res) in write_results {
            results[pos] = Some(res);
        }
    }

    let mut item_values = Vec::new();
    let mut errors = false;
    for (pos, res) in results.into_iter().enumerate() {
        let res = res.unwrap_or_else(|| {
            let it = &items[pos];
            ItemResult {
                op: it.op.clone(),
                index: it.index.clone(),
                id: it.id.clone(),
                status: 500,
                result: "error".into(),
                error: Some("no result produced".into()),
            }
        });
        if res.status >= 400 {
            errors = true;
        }
        state
            .metrics
            .bulk_items_total
            .with_label_values(&[&res.status.to_string()])
            .inc();
        item_values.push(res.to_value());
    }

    Ok(Value::Object({
        let mut m = Map::new();
        m.insert(
            "took".into(),
            Value::from(started.elapsed().as_millis() as u64),
        );
        m.insert("errors".into(), Value::Bool(errors));
        m.insert("items".into(), Value::Array(item_values));
        m
    }))
}

/// `POST /_bulk` — multi-index bulk.
pub async fn bulk(State(state): State<Arc<AppState>>, body: Bytes) -> Response {
    let text = String::from_utf8_lossy(&body);
    es_response(bulk_core(&state, None, &text).await)
}

/// `POST /{index}/_bulk` — single-index bulk (index is the default).
pub async fn bulk_indexed(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    body: Bytes,
) -> Response {
    let text = String::from_utf8_lossy(&body);
    es_response(bulk_core(&state, Some(&index), &text).await)
}
