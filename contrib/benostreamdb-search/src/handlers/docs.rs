// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! ES-style document handlers: `POST /{index}/_doc[/{id}]` and
//! `POST /{index}/_refresh`.

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use benostreamdb::BenoStreamError;
use serde_json::Value;

use crate::es_types::{
    DeleteByQueryResponse, DocGetResponse, DocWriteResponse, RefreshResponse, ReindexResponse,
    ReindexRetries, Shards,
};
use crate::infer::{self, InferError};
use crate::state::AppState;

use super::{es_response, es_response_with_status};

/// Reserved document id column.
pub const ID_COLUMN: &str = "_id";

/// Insert or overwrite the reserved `_id` field from the supplied id.
///
/// A non-object body is rejected: BenoStreamDB tables are columnar and
/// every row must carry the same reserved id column.
pub(crate) fn with_id(doc: &mut Value, id: &str) -> Result<(), BenoStreamError> {
    let obj = doc
        .as_object_mut()
        .ok_or_else(|| BenoStreamError::SchemaIncompatible {
            reason: "document body must be a JSON object".into(),
        })?;
    obj.insert(ID_COLUMN.to_string(), Value::String(id.to_string()));
    Ok(())
}

/// Map a core write failure to a typed [`BenoStreamError`].
///
/// M2 core raises typed variants (`PrimaryKeyViolation`,
/// `NullConstraintViolation`) as `anyhow::Error`; a direct downcast is
/// preferred. Errors wrapped in an anyhow context chain (or older
/// string-only failures) fall back to matching the exact Display
/// message, which is preserved by the core's Display impls.
pub(crate) fn translate_write_error(err: anyhow::Error) -> BenoStreamError {
    if let Some(he) = err.downcast_ref::<BenoStreamError>() {
        match he {
            BenoStreamError::PrimaryKeyViolation { key } => {
                return BenoStreamError::PrimaryKeyViolation { key: key.clone() };
            }
            BenoStreamError::NullConstraintViolation { column } => {
                return BenoStreamError::NullConstraintViolation {
                    column: column.clone(),
                };
            }
            _ => {}
        }
    }

    let msg = err.to_string();
    const DUP_PREFIX: &str = "Duplicate primary key error: id = ";
    if let Some(pos) = msg.find(DUP_PREFIX) {
        let key = msg[pos + DUP_PREFIX.len()..].trim();
        return BenoStreamError::PrimaryKeyViolation {
            key: key.to_string(),
        };
    }
    const NULL_PK_PREFIX: &str = "Null constraint violation: Primary key column '";
    if let Some(pos) = msg.find(NULL_PK_PREFIX) {
        let col = msg[pos + NULL_PK_PREFIX.len()..]
            .split('\'')
            .next()
            .unwrap_or_default();
        return BenoStreamError::NullConstraintViolation {
            column: col.to_string(),
        };
    }
    BenoStreamError::from(err)
}

/// Materialize one document as a single-row batch against `target_schema`.
///
/// JSON nulls (and absent fields) become Arrow nulls: values are
/// pre-filtered before [`infer::value_to_array`] because that function
/// only treats `None` as a null, and a leaked `Value::Null` would be
/// misread by the numeric arms.
pub(crate) fn build_row_batch(
    target_schema: &Schema,
    doc: &Value,
) -> Result<RecordBatch, BenoStreamError> {
    let obj = doc
        .as_object()
        .ok_or_else(|| BenoStreamError::SchemaIncompatible {
            reason: "document body must be a JSON object".into(),
        })?;
    let mut cols = Vec::with_capacity(target_schema.fields().len());
    for field in target_schema.fields() {
        let values = vec![obj.get(field.name()).filter(|v| !v.is_null()).cloned()];
        let col = infer::value_to_array(field.name(), field.data_type(), &values).map_err(
            |e: InferError| BenoStreamError::SchemaIncompatible {
                reason: e.to_string(),
            },
        )?;
        cols.push(col);
    }
    RecordBatch::try_new(Arc::new(target_schema.clone()), cols).map_err(|e| {
        BenoStreamError::SchemaIncompatible {
            reason: format!("failed to build row batch: {e}"),
        }
    })
}

pub(crate) fn build_multi_row_batch(
    target_schema: &Schema,
    docs: &[Value],
) -> Result<RecordBatch, BenoStreamError> {
    let mut cols = Vec::with_capacity(target_schema.fields().len());
    for field in target_schema.fields() {
        let values: Vec<Option<Value>> = docs
            .iter()
            .map(|doc| {
                doc.as_object()
                    .and_then(|obj| obj.get(field.name()))
                    .filter(|v| !v.is_null())
                    .cloned()
            })
            .collect();
        let col = infer::value_to_array(field.name(), field.data_type(), &values).map_err(
            |e: InferError| BenoStreamError::SchemaIncompatible {
                reason: e.to_string(),
            },
        )?;
        cols.push(col);
    }
    RecordBatch::try_new(Arc::new(target_schema.clone()), cols).map_err(|e| {
        BenoStreamError::SchemaIncompatible {
            reason: format!("failed to build multi-row batch: {e}"),
        }
    })
}

/// Index a document: generate or accept an id, infer/merge the schema,
/// and write one row through the table's write buffer.
///
/// The `result` field mirrors ES doc-write semantics: `"created"` (201)
/// when this request created the index, `"updated"` (200) when it was
/// written into an existing one. A duplicate `_id` is a 400
/// `resource_already_exists_exception`.
pub(crate) async fn index_document_core(
    state: &AppState,
    index: &str,
    path_id: Option<&str>,
    mut doc: Value,
) -> Result<DocWriteResponse, BenoStreamError> {
    let id = match path_id {
        Some(id) => id.to_string(),
        None => uuid::Uuid::new_v4().to_string(),
    };
    with_id(&mut doc, &id)?;

    // Was the index present before this request? That decides the ES
    // result and whether we install the `_id` primary key.
    let existed_before = state.index_exists(index).await;

    let doc_schema =
        infer::infer_schema(&doc).map_err(|e: InferError| BenoStreamError::SchemaIncompatible {
            reason: e.to_string(),
        })?;

    // The table is only created when it does not exist yet; the schema is
    // cloned (not moved) because it is reused below for the merge.
    let table = state
        .open_or_create(index, &Some(Arc::clone(&doc_schema)))
        .await?;

    // A table created by this request gets `_id` as its primary key so
    // duplicate ids are rejected. The PK is committed to the Iceberg
    // manifest, so re-opened tables keep it.
    if !existed_before && table.get_primary_key().is_empty() {
        table
            .set_primary_key_async(vec![ID_COLUMN.to_string()])
            .await
            .map_err(BenoStreamError::from)?;
    }

    // Pre-merge the doc schema against the table's current schema so
    // numeric promotion (e.g. Int64 + Float64 -> Float64) happens once,
    // into a single coherent Arrow type.
    let target = infer::merge_schemas(table.arrow_schema().as_ref(), &doc_schema).map_err(
        |e: InferError| BenoStreamError::SchemaIncompatible {
            reason: e.to_string(),
        },
    )?;

    let batch = build_row_batch(&target, &doc)?;
    table
        .write_async(vec![batch])
        .await
        .map_err(translate_write_error)?;

    Ok(DocWriteResponse {
        index: index.to_string(),
        id,
        version: 1,
        result: if existed_before { "updated" } else { "created" }.to_string(),
        shards: Shards {
            total: 1,
            successful: 1,
            failed: 0,
        },
    })
}

/// Record one doc-write outcome in the ingestion counter (plan 5.2.2).
fn record_ingest(state: &AppState, result: &Result<DocWriteResponse, BenoStreamError>) {
    let outcome = match result {
        Ok(resp) => resp.result.as_str(),
        Err(_) => "error",
    };
    state
        .metrics
        .docs_indexed_total
        .with_label_values(&[outcome])
        .inc();
}

/// `POST /{index}/_doc` — index a document with a server-generated id.
pub async fn index_document(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    Json(doc): Json<Value>,
) -> Response {
    let result = index_document_core(&state, &index, None, doc).await;
    record_ingest(&state, &result);
    let status = match &result {
        Ok(resp) if resp.result == "created" => StatusCode::CREATED,
        _ => StatusCode::OK,
    };
    es_response_with_status(status, result)
}

/// `POST /{index}/_doc/{id}` — index a document with a client-supplied id.
pub async fn index_document_id(
    State(state): State<Arc<AppState>>,
    Path((index, id)): Path<(String, String)>,
    Json(doc): Json<Value>,
) -> Response {
    let result = index_document_core(&state, &index, Some(&id), doc).await;
    record_ingest(&state, &result);
    let status = match &result {
        Ok(resp) if resp.result == "created" => StatusCode::CREATED,
        _ => StatusCode::OK,
    };
    es_response_with_status(status, result)
}

/// `POST /{index}/_refresh` — flush the index's write buffer to storage
/// so newly indexed documents become visible to reads.
pub async fn refresh(State(state): State<Arc<AppState>>, Path(index): Path<String>) -> Response {
    es_response_with_status(StatusCode::OK, refresh_core(&state, &index).await)
}

/// `POST /_refresh` — flush every index's write buffer to storage.
pub async fn refresh_all(State(state): State<Arc<AppState>>) -> Response {
    let indexes = state.list_indexes().await.unwrap_or_default();
    for index in indexes {
        if let Err(e) = refresh_core(&state, &index).await {
            tracing::warn!(index, error = %e, "global refresh failed for index");
        }
    }
    es_response_with_status(
        StatusCode::OK,
        Ok(RefreshResponse {
            shards: Shards {
                total: 1,
                successful: 1,
                failed: 0,
            },
        }),
    )
}

/// `GET /{index}/_doc/{id}` — retrieve a document by client id.
pub async fn get_document(
    State(state): State<Arc<AppState>>,
    Path((index, id)): Path<(String, String)>,
) -> Response {
    let result = get_document_core(&state, &index, &id).await;
    match result {
        Ok(resp) if resp.found => (StatusCode::OK, axum::Json(resp)).into_response(),
        Ok(resp) => (StatusCode::NOT_FOUND, axum::Json(resp)).into_response(),
        Err(BenoStreamError::TableNotFound { .. }) => {
            let es = DocGetResponse {
                index,
                id,
                version: None,
                seq_no: None,
                primary_term: None,
                found: false,
                source: None,
            };
            (StatusCode::NOT_FOUND, axum::Json(es)).into_response()
        }
        Err(e) => {
            es_response_with_status::<DocGetResponse>(StatusCode::INTERNAL_SERVER_ERROR, Err(e))
        }
    }
}

pub async fn get_document_core(
    state: &AppState,
    index: &str,
    id: &str,
) -> Result<DocGetResponse, BenoStreamError> {
    if !state.index_exists(index).await {
        return Err(BenoStreamError::TableNotFound {
            namespace: String::new(),
            name: index.to_string(),
        });
    }

    let table = state.open_or_create(index, &None).await?;
    let safe_id = id.replace('\'', "''");
    let filter = format!("{ID_COLUMN} = '{safe_id}'");
    let batches = table
        .read_async(Some(&filter), None, None)
        .await
        .map_err(BenoStreamError::from)?;

    if batches.is_empty() || batches[0].num_rows() == 0 {
        return Ok(DocGetResponse {
            index: index.to_string(),
            id: id.to_string(),
            version: None,
            seq_no: None,
            primary_term: None,
            found: false,
            source: None,
        });
    }

    let batch = &batches[0];
    let schema = batch.schema();
    let mut source_obj = serde_json::Map::new();

    for (col_idx, field) in schema.fields().iter().enumerate() {
        if field.name() == ID_COLUMN {
            continue;
        }
        let col = batch.column(col_idx);
        let val = crate::handlers::search::value_to_json(col.as_ref(), 0);
        source_obj.insert(field.name().clone(), val);
    }

    Ok(DocGetResponse {
        index: index.to_string(),
        id: id.to_string(),
        version: Some(1),
        seq_no: Some(0),
        primary_term: Some(1),
        found: true,
        source: Some(Value::Object(source_obj)),
    })
}

/// `DELETE /{index}/_doc/{id}` — delete a document by client id.
pub async fn delete_document(
    State(state): State<Arc<AppState>>,
    Path((index, id)): Path<(String, String)>,
) -> Response {
    let result = delete_document_core(&state, &index, &id).await;
    match result {
        Ok(resp) => {
            let status = if resp.result == "deleted" {
                StatusCode::OK
            } else {
                StatusCode::NOT_FOUND
            };
            (status, axum::Json(resp)).into_response()
        }
        Err(e) => {
            es_response_with_status::<DocWriteResponse>(StatusCode::INTERNAL_SERVER_ERROR, Err(e))
        }
    }
}

pub async fn delete_document_core(
    state: &AppState,
    index: &str,
    id: &str,
) -> Result<DocWriteResponse, BenoStreamError> {
    if !state.index_exists(index).await {
        return Err(BenoStreamError::TableNotFound {
            namespace: String::new(),
            name: index.to_string(),
        });
    }

    let table = state.open_or_create(index, &None).await?;
    let safe_id = id.replace('\'', "''");
    let filter = format!("{ID_COLUMN} = '{safe_id}'");

    // Check if doc exists before delete
    let existing = table
        .read_async(Some(&filter), None, None)
        .await
        .map_err(BenoStreamError::from)?;
    let found = !existing.is_empty() && existing[0].num_rows() > 0;

    if found {
        table
            .delete_async(&filter)
            .await
            .map_err(BenoStreamError::from)?;
    }

    Ok(DocWriteResponse {
        index: index.to_string(),
        id: id.to_string(),
        version: if found { 2 } else { 1 },
        result: if found {
            "deleted".into()
        } else {
            "not_found".into()
        },
        shards: Shards {
            total: 1,
            successful: 1,
            failed: 0,
        },
    })
}

/// `POST /{index}/_update/{id}` — partial document update.
pub async fn update_document(
    State(state): State<Arc<AppState>>,
    Path((index, id)): Path<(String, String)>,
    Json(body): Json<Value>,
) -> Response {
    let result = update_document_core(&state, &index, &id, body).await;
    match result {
        Ok(resp) => {
            let status = if resp.result == "created" {
                StatusCode::CREATED
            } else {
                StatusCode::OK
            };
            (status, axum::Json(resp)).into_response()
        }
        Err(BenoStreamError::TableNotFound { .. }) => {
            let es = crate::es_types::EsError {
                error: crate::es_types::EsErrorBody {
                    error_type: "document_missing_exception".into(),
                    reason: format!("[_doc][{id}]: document missing"),
                },
                status: 404,
            };
            (StatusCode::NOT_FOUND, axum::Json(es)).into_response()
        }
        Err(e) if e.to_string().contains("document missing") => {
            let es = crate::es_types::EsError {
                error: crate::es_types::EsErrorBody {
                    error_type: "document_missing_exception".into(),
                    reason: format!("[_doc][{id}]: document missing"),
                },
                status: 404,
            };
            (StatusCode::NOT_FOUND, axum::Json(es)).into_response()
        }
        Err(e) => es_response_with_status::<DocWriteResponse>(StatusCode::BAD_REQUEST, Err(e)),
    }
}

pub async fn update_document_core(
    state: &AppState,
    index: &str,
    id: &str,
    body: Value,
) -> Result<DocWriteResponse, BenoStreamError> {
    let doc_patch = body.get("doc").and_then(Value::as_object).ok_or_else(|| {
        BenoStreamError::SchemaIncompatible {
            reason: "update: missing or invalid 'doc' object (scripts not supported)".into(),
        }
    })?;
    let doc_as_upsert = body
        .get("doc_as_upsert")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    let resolved = state.resolve_alias(index).await;
    if !state.index_exists(&resolved).await {
        if doc_as_upsert {
            let new_doc = Value::Object(doc_patch.clone());
            return index_document_core(state, &resolved, Some(id), new_doc).await;
        } else {
            return Err(BenoStreamError::TableNotFound {
                namespace: String::new(),
                name: index.to_string(),
            });
        }
    }

    let existing = get_document_core(state, &resolved, id).await?;
    if !existing.found || existing.source.is_none() {
        if doc_as_upsert {
            let new_doc = Value::Object(doc_patch.clone());
            return index_document_core(state, &resolved, Some(id), new_doc).await;
        } else {
            return Err(BenoStreamError::internal(format!(
                "document missing: [_doc][{id}]"
            )));
        }
    }

    let mut source = existing
        .source
        .ok_or_else(|| BenoStreamError::SchemaIncompatible {
            reason: "corrupted document: missing _source".into(),
        })?;
    let source_obj = source
        .as_object_mut()
        .ok_or_else(|| BenoStreamError::SchemaIncompatible {
            reason: "corrupted document source".into(),
        })?;

    for (k, v) in doc_patch {
        source_obj.insert(k.clone(), v.clone());
    }

    let table = state.open_or_create(&resolved, &None).await?;
    let safe_id = id.replace('\'', "''");
    let filter = format!("{ID_COLUMN} = '{safe_id}'");
    table
        .delete_async(&filter)
        .await
        .map_err(BenoStreamError::from)?;

    let write_res = index_document_core(state, &resolved, Some(id), source).await?;

    Ok(DocWriteResponse {
        index: index.to_string(),
        id: id.to_string(),
        version: existing.version.unwrap_or(1) + 1,
        result: "updated".to_string(),
        shards: write_res.shards,
    })
}

/// `POST /{index}/_delete_by_query` — delete documents matching a query DSL.
pub async fn delete_by_query(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let result = delete_by_query_core(&state, &index, body).await;
    es_response_with_status(StatusCode::OK, result)
}

pub async fn delete_by_query_core(
    state: &AppState,
    index: &str,
    body: Value,
) -> Result<DeleteByQueryResponse, BenoStreamError> {
    let started = std::time::Instant::now();
    if !state.index_exists(index).await {
        return Err(BenoStreamError::TableNotFound {
            namespace: String::new(),
            name: index.to_string(),
        });
    }

    let query_clause = body
        .get("query")
        .ok_or_else(|| BenoStreamError::SchemaIncompatible {
            reason: "missing 'query' in _delete_by_query request body".into(),
        })?;

    let filter_sql = crate::handlers::search::clause_to_sql(query_clause, "_delete_by_query")?;
    let table = state.open_or_create(index, &None).await?;

    let batches = table
        .read_async(Some(&filter_sql), None, None)
        .await
        .map_err(BenoStreamError::from)?;
    let count: usize = batches.iter().map(|b| b.num_rows()).sum();

    if count > 0 {
        table
            .delete_async(&filter_sql)
            .await
            .map_err(BenoStreamError::from)?;
    }

    Ok(DeleteByQueryResponse {
        took: started.elapsed().as_millis() as u64,
        timed_out: false,
        total: count as u64,
        deleted: count as u64,
        batches: 1,
        version_conflicts: 0,
        noops: 0,
        retries: crate::es_types::RetryStats { bulk: 0, search: 0 },
        throttled_millis: 0,
        requests_per_second: -1.0,
        throttled_until_millis: 0,
        failures: vec![],
    })
}

pub async fn reindex(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    es_response(reindex_core(&state, &body).await)
}

pub async fn reindex_core(
    state: &AppState,
    body: &Value,
) -> Result<ReindexResponse, BenoStreamError> {
    let start = std::time::Instant::now();

    let source = body
        .get("source")
        .and_then(Value::as_object)
        .ok_or_else(|| BenoStreamError::SchemaIncompatible {
            reason: "reindex: missing 'source' object".into(),
        })?;
    let source_index = source.get("index").and_then(Value::as_str).ok_or_else(|| {
        BenoStreamError::SchemaIncompatible {
            reason: "reindex: 'source.index' must be a string".into(),
        }
    })?;

    let dest = body.get("dest").and_then(Value::as_object).ok_or_else(|| {
        BenoStreamError::SchemaIncompatible {
            reason: "reindex: missing 'dest' object".into(),
        }
    })?;
    let dest_index = dest.get("index").and_then(Value::as_str).ok_or_else(|| {
        BenoStreamError::SchemaIncompatible {
            reason: "reindex: 'dest.index' must be a string".into(),
        }
    })?;

    let op_type = dest
        .get("op_type")
        .and_then(Value::as_str)
        .unwrap_or("index");

    let max_docs = body.get("max_docs").and_then(Value::as_u64);
    let batch_size = source.get("size").and_then(Value::as_u64).unwrap_or(1000) as usize;

    let resolved_source = state.resolve_alias(source_index).await;
    if !state.index_exists(&resolved_source).await {
        return Err(BenoStreamError::TableNotFound {
            namespace: String::new(),
            name: source_index.to_string(),
        });
    }

    let resolved_dest = state.resolve_alias(dest_index).await;

    let dest_existed_before = state.index_exists(&resolved_dest).await;

    let source_query = source.get("query");
    let source_filter = source.get("_source");

    let mut last_id: Option<String> = None;
    let mut total_docs: u64 = 0;
    let mut created_count: u64 = 0;
    let mut updated_count: u64 = 0;
    let mut version_conflicts: u64 = 0;
    let mut batch_count: u64 = 0;
    let mut failures: Vec<Value> = Vec::new();

    loop {
        let current_batch_size = if let Some(max) = max_docs {
            let remaining = max.saturating_sub(total_docs) as usize;
            if remaining == 0 {
                break;
            }
            batch_size.min(remaining)
        } else {
            batch_size
        };

        let mut search_body = serde_json::Map::new();
        search_body.insert("size".to_string(), serde_json::json!(current_batch_size));
        search_body.insert("sort".to_string(), serde_json::json!(["_id:asc"]));
        if let Some(lid) = &last_id {
            search_body.insert("search_after".to_string(), serde_json::json!([lid]));
        }
        if let Some(q) = source_query {
            search_body.insert("query".to_string(), q.clone());
        }
        if let Some(sf) = source_filter {
            search_body.insert("_source".to_string(), sf.clone());
        }

        let resp = super::search::search_core(state, &resolved_source, &Value::Object(search_body))
            .await?;
        if resp.hits.hits.is_empty() {
            break;
        }

        batch_count += 1;

        for hit in resp.hits.hits {
            last_id = Some(hit.id.clone());

            let mut doc = hit.source.clone();
            if let Some(obj) = doc.as_object_mut() {
                obj.remove(ID_COLUMN);
            }

            let doc_already_in_dest = if dest_existed_before {
                get_document_core(state, &resolved_dest, &hit.id)
                    .await
                    .map(|d| d.found)
                    .unwrap_or(false)
            } else {
                false
            };

            if doc_already_in_dest && op_type == "create" {
                version_conflicts += 1;
                failures.push(serde_json::json!({
                    "index": dest_index,
                    "type": "_doc",
                    "id": hit.id,
                    "status": 409,
                    "cause": {
                        "type": "version_conflict_engine_exception",
                        "reason": format!("[{}]: version conflict, document already exists (op_type=create)", hit.id)
                    }
                }));
                continue;
            }

            match index_document_core(state, &resolved_dest, Some(&hit.id), doc).await {
                Ok(_) => {
                    total_docs += 1;
                    if doc_already_in_dest {
                        updated_count += 1;
                    } else {
                        created_count += 1;
                    }
                }
                Err(BenoStreamError::PrimaryKeyViolation { key }) => {
                    version_conflicts += 1;
                    if op_type == "create" {
                        failures.push(serde_json::json!({
                            "index": dest_index,
                            "type": "_doc",
                            "id": key,
                            "status": 409,
                            "cause": {
                                "type": "version_conflict_engine_exception",
                                "reason": format!("[{key}]: version conflict, document already exists (op_type=create)")
                            }
                        }));
                    }
                }
                Err(err) => {
                    failures.push(serde_json::json!({
                        "index": dest_index,
                        "type": "_doc",
                        "id": hit.id,
                        "status": 500,
                        "cause": {
                            "type": "exception",
                            "reason": err.to_string()
                        }
                    }));
                }
            }
        }
    }

    if total_docs > 0 {
        let _ = refresh_core(state, &resolved_dest).await;
    }

    Ok(ReindexResponse {
        took: start.elapsed().as_millis() as u64,
        timed_out: false,
        total: total_docs,
        updated: updated_count,
        created: created_count,
        deleted: 0,
        batches: batch_count,
        version_conflicts,
        noops: 0,
        retries: ReindexRetries { bulk: 0, search: 0 },
        throttled_millis: 0,
        requests_per_second: -1.0,
        throttled_until_millis: 0,
        failures,
    })
}

pub async fn refresh_core(
    state: &AppState,
    index: &str,
) -> Result<RefreshResponse, BenoStreamError> {
    if !state.index_exists(index).await {
        return Err(BenoStreamError::TableNotFound {
            namespace: String::new(),
            name: index.to_string(),
        });
    }
    let started = std::time::Instant::now();
    let table = state.open_or_create(index, &None).await?;
    table.commit_async().await.map_err(translate_write_error)?;
    // Wait for the background index-building tasks spawned by the commit
    // so the new segment's BM25/HNSW indexes are attached to the manifest
    // before `_search` can observe the data.
    table
        .wait_for_background_tasks_async()
        .await
        .map_err(translate_write_error)?;
    state
        .metrics
        .refresh_seconds
        .observe(started.elapsed().as_secs_f64());
    Ok(RefreshResponse {
        shards: Shards {
            total: 1,
            successful: 1,
            failed: 0,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field};

    #[test]
    fn with_id_inserts_and_overrides() {
        let mut doc = serde_json::json!({"name": "alice"});
        with_id(&mut doc, "1").unwrap();
        assert_eq!(doc[ID_COLUMN], Value::String("1".into()));

        // The path id wins over any `_id` in the body.
        let mut doc = serde_json::json!({ID_COLUMN: "body-id", "n": 2});
        with_id(&mut doc, "path-id").unwrap();
        assert_eq!(doc[ID_COLUMN], Value::String("path-id".into()));

        let mut bad = serde_json::json!([1, 2]);
        assert!(matches!(
            with_id(&mut bad, "1"),
            Err(BenoStreamError::SchemaIncompatible { .. })
        ));
    }

    #[test]
    fn translate_write_error_maps_core_strings() {
        // A leading anyhow context chain must not break the match.
        let err = anyhow::anyhow!("write failed: Duplicate primary key error: id = doc-42");
        assert!(matches!(
            translate_write_error(err),
            BenoStreamError::PrimaryKeyViolation { ref key } if key == "doc-42"
        ));

        let err = anyhow::anyhow!(
            "Null constraint violation: Primary key column '_id' cannot contain null values"
        );
        assert!(matches!(
            translate_write_error(err),
            BenoStreamError::NullConstraintViolation { ref column } if column == "_id"
        ));

        let err = anyhow::anyhow!("something exploded");
        assert!(matches!(
            translate_write_error(err),
            BenoStreamError::Internal { .. }
        ));
    }

    #[test]
    fn translate_write_error_maps_typed_core_errors() {
        let err = anyhow::Error::from(BenoStreamError::PrimaryKeyViolation {
            key: "doc-42".into(),
        });
        assert!(matches!(
            translate_write_error(err),
            BenoStreamError::PrimaryKeyViolation { ref key } if key == "doc-42"
        ));

        let err = anyhow::Error::from(BenoStreamError::NullConstraintViolation {
            column: "_id".into(),
        });
        assert!(matches!(
            translate_write_error(err),
            BenoStreamError::NullConstraintViolation { ref column } if column == "_id"
        ));
    }

    #[test]
    fn build_row_batch_treats_json_nulls_as_arrow_nulls() {
        let schema = Schema::new(vec![
            Field::new("name", DataType::Utf8, true),
            Field::new("age", DataType::Int64, true),
        ]);

        let batch =
            build_row_batch(&schema, &serde_json::json!({"name": null, "age": null})).unwrap();
        let name = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert!(name.is_null(0));
        let age = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert!(age.is_null(0));

        let batch =
            build_row_batch(&schema, &serde_json::json!({"name": "alice", "age": 30})).unwrap();
        let name = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(name.value(0), "alice");
        let age = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(age.value(0), 30);

        // An absent field becomes a null, not a default.
        let batch = build_row_batch(&schema, &serde_json::json!({"name": "bob"})).unwrap();
        let age = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert!(age.is_null(0));
    }

    #[tokio::test]
    async fn index_document_end_to_end() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());

        // The local object store does not create parent directories.
        std::fs::create_dir_all(tmp.path().join("people")).unwrap();

        // The first document creates the index with `_id` as its PK.
        let resp = index_document_core(
            &state,
            "people",
            None,
            serde_json::json!({"name": "alice", "age": 30}),
        )
        .await
        .unwrap();
        assert_eq!(resp.result, "created");
        assert!(!resp.id.is_empty());

        // A client-supplied id indexes fine the first time...
        let resp = index_document_core(
            &state,
            "people",
            Some("bob-1"),
            serde_json::json!({"name": "bob"}),
        )
        .await
        .unwrap();
        assert_eq!(resp.result, "updated");

        // ...but a duplicate id is a PK violation.
        let err = index_document_core(
            &state,
            "people",
            Some("bob-1"),
            serde_json::json!({"name": "bob again"}),
        )
        .await
        .unwrap_err();
        assert!(
            matches!(
                &err,
                BenoStreamError::PrimaryKeyViolation { key } if key == "bob-1"
            ),
            "{err}"
        );

        // Refresh flushes the write buffer; the rows are readable.
        let resp = refresh_core(&state, "people").await.unwrap();
        assert_eq!(resp.shards.successful, 1);

        let table = state.open_or_create("people", &None).await.unwrap();
        let batches = table.read_async(None, None, None).await.unwrap();
        let rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(rows, 2);

        // Refreshing an unknown index is a 404-shaped error.
        let err = refresh_core(&state, "ghost").await.unwrap_err();
        assert!(
            matches!(&err, BenoStreamError::TableNotFound { .. }),
            "{err}"
        );
    }

    #[tokio::test]
    async fn get_and_delete_document_lifecycle() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("items")).unwrap();

        // Index a doc
        let resp = index_document_core(
            &state,
            "items",
            Some("item-42"),
            serde_json::json!({"title": "Rust Book", "price": 45}),
        )
        .await
        .unwrap();
        assert_eq!(resp.result, "created");
        refresh_core(&state, "items").await.unwrap();

        // GET doc
        let doc = get_document_core(&state, "items", "item-42").await.unwrap();
        assert!(doc.found);
        assert_eq!(doc.id, "item-42");
        let src = doc.source.expect("has source");
        assert_eq!(src["title"], "Rust Book");
        assert_eq!(src["price"], 45);

        // GET nonexistent doc
        let doc_none = get_document_core(&state, "items", "nonexistent")
            .await
            .unwrap();
        assert!(!doc_none.found);
        assert!(doc_none.source.is_none());

        // DELETE doc
        let del = delete_document_core(&state, "items", "item-42")
            .await
            .unwrap();
        assert_eq!(del.result, "deleted");
        refresh_core(&state, "items").await.unwrap();

        // GET after delete
        let doc_after = get_document_core(&state, "items", "item-42").await.unwrap();
        assert!(!doc_after.found);

        // DELETE nonexistent doc
        let del_none = delete_document_core(&state, "items", "item-42")
            .await
            .unwrap();
        assert_eq!(del_none.result, "not_found");
    }

    #[tokio::test]
    async fn delete_by_query_lifecycle() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("products")).unwrap();

        for i in 1..=5 {
            index_document_core(
                &state,
                "products",
                Some(&format!("p-{}", i)),
                serde_json::json!({"tag": if i <= 3 { "old" } else { "new" }, "val": i}),
            )
            .await
            .unwrap();
        }
        refresh_core(&state, "products").await.unwrap();

        // Delete where tag = 'old'
        let del_resp = delete_by_query_core(
            &state,
            "products",
            serde_json::json!({
                "query": {
                    "term": { "tag": "old" }
                }
            }),
        )
        .await
        .unwrap();

        assert_eq!(del_resp.deleted, 3);
        refresh_core(&state, "products").await.unwrap();

        let table = state.open_or_create("products", &None).await.unwrap();
        let batches = table.read_async(None, None, None).await.unwrap();
        let remaining_rows: usize = batches.iter().map(|b| b.num_rows()).sum();
        assert_eq!(remaining_rows, 2);
    }

    #[tokio::test]
    async fn update_document_lifecycle() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("users")).unwrap();

        // Initial write
        index_document_core(
            &state,
            "users",
            Some("u-1"),
            serde_json::json!({"name": "alice", "age": 30, "city": "Seattle"}),
        )
        .await
        .unwrap();
        refresh_core(&state, "users").await.unwrap();

        // 1. Partial update: change age, add occupation
        let update_res = update_document_core(
            &state,
            "users",
            "u-1",
            serde_json::json!({
                "doc": {
                    "age": 31,
                    "occupation": "Engineer"
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(update_res.result, "updated");
        assert_eq!(update_res.version, 2);
        refresh_core(&state, "users").await.unwrap();

        // Verify merged doc
        let get_res = get_document_core(&state, "users", "u-1").await.unwrap();
        assert!(get_res.found);
        let src = get_res.source.unwrap();
        assert_eq!(src["name"], "alice");
        assert_eq!(src["city"], "Seattle");
        assert_eq!(src["age"], 31);
        assert_eq!(src["occupation"], "Engineer");

        // 2. Update nonexistent without upsert fails with document missing
        let err = update_document_core(
            &state,
            "users",
            "u-2",
            serde_json::json!({"doc": {"name": "bob"}}),
        )
        .await
        .unwrap_err();
        assert!(err.to_string().contains("document missing"));

        // 3. Update nonexistent with doc_as_upsert succeeds
        let upsert_res = update_document_core(
            &state,
            "users",
            "u-2",
            serde_json::json!({
                "doc": {"name": "bob", "age": 25},
                "doc_as_upsert": true
            }),
        )
        .await
        .unwrap();
        assert_eq!(upsert_res.id, "u-2");
        refresh_core(&state, "users").await.unwrap();

        let get_u2 = get_document_core(&state, "users", "u-2").await.unwrap();
        assert!(get_u2.found);
        assert_eq!(get_u2.source.unwrap()["name"], "bob");
    }

    #[tokio::test]
    async fn reindex_lifecycle() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());

        // Populate source index
        index_document_core(
            &state,
            "src",
            Some("d1"),
            serde_json::json!({"name": "alice", "age": 25, "role": "eng"}),
        )
        .await
        .unwrap();
        index_document_core(
            &state,
            "src",
            Some("d2"),
            serde_json::json!({"name": "bob", "age": 35, "role": "eng"}),
        )
        .await
        .unwrap();
        index_document_core(
            &state,
            "src",
            Some("d3"),
            serde_json::json!({"name": "charlie", "age": 45, "role": "mgr"}),
        )
        .await
        .unwrap();
        refresh_core(&state, "src").await.unwrap();

        // 1. Full reindex into new index "dst1"
        let res1 = reindex_core(
            &state,
            &serde_json::json!({
                "source": { "index": "src" },
                "dest": { "index": "dst1" }
            }),
        )
        .await
        .unwrap();

        assert_eq!(res1.total, 3);
        assert_eq!(res1.created, 3);
        assert_eq!(res1.failures.len(), 0);

        let d1 = get_document_core(&state, "dst1", "d1").await.unwrap();
        assert!(d1.found);
        assert_eq!(d1.source.unwrap()["name"], "alice");

        let d3 = get_document_core(&state, "dst1", "d3").await.unwrap();
        assert!(d3.found);
        assert_eq!(d3.source.unwrap()["name"], "charlie");

        // 2. Filtered reindex with query
        let res2 = reindex_core(
            &state,
            &serde_json::json!({
                "source": {
                    "index": "src",
                    "query": {
                        "range": {
                            "age": { "gte": 30 }
                        }
                    }
                },
                "dest": { "index": "dst_seniors" }
            }),
        )
        .await
        .unwrap();

        assert_eq!(res2.total, 2);
        assert_eq!(res2.created, 2);

        let d2 = get_document_core(&state, "dst_seniors", "d2")
            .await
            .unwrap();
        assert!(d2.found);
        let d1_absent = get_document_core(&state, "dst_seniors", "d1")
            .await
            .unwrap();
        assert!(!d1_absent.found);

        // 3. Reindex with max_docs
        let res3 = reindex_core(
            &state,
            &serde_json::json!({
                "source": { "index": "src" },
                "dest": { "index": "dst_limit" },
                "max_docs": 1
            }),
        )
        .await
        .unwrap();
        assert_eq!(res3.total, 1);

        // 4. Nonexistent source errors
        let err = reindex_core(
            &state,
            &serde_json::json!({
                "source": { "index": "no_such_src" },
                "dest": { "index": "dst_err" }
            }),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, BenoStreamError::TableNotFound { .. }));
    }
}
