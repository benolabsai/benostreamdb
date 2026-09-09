// Copyright (c) 2026 Richard Albright. All rights reserved.

//! ES-style document handlers: `POST /{index}/_doc[/{id}]` and
//! `POST /{index}/_refresh`.

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::Schema;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use hyperstreamdb::HyperstreamError;
use serde_json::Value;

use crate::es_types::{DocWriteResponse, RefreshResponse, Shards};
use crate::infer::{self, InferError};
use crate::state::{table_exists, AppState};

use super::es_response_with_status;

/// Reserved document id column.
pub const ID_COLUMN: &str = "_id";

/// Insert or overwrite the reserved `_id` field from the supplied id.
///
/// A non-object body is rejected: HyperStreamDB tables are columnar and
/// every row must carry the same reserved id column.
pub(crate) fn with_id(doc: &mut Value, id: &str) -> Result<(), HyperstreamError> {
    let obj = doc
        .as_object_mut()
        .ok_or_else(|| HyperstreamError::SchemaIncompatible {
            reason: "document body must be a JSON object".into(),
        })?;
    obj.insert(ID_COLUMN.to_string(), Value::String(id.to_string()));
    Ok(())
}

/// Map a core write failure to a typed [`HyperstreamError`].
///
/// M2 core raises typed variants (`PrimaryKeyViolation`,
/// `NullConstraintViolation`) as `anyhow::Error`; a direct downcast is
/// preferred. Errors wrapped in an anyhow context chain (or older
/// string-only failures) fall back to matching the exact Display
/// message, which is preserved by the core's Display impls.
pub(crate) fn translate_write_error(err: anyhow::Error) -> HyperstreamError {
    if let Some(he) = err.downcast_ref::<HyperstreamError>() {
        match he {
            HyperstreamError::PrimaryKeyViolation { key } => {
                return HyperstreamError::PrimaryKeyViolation { key: key.clone() };
            }
            HyperstreamError::NullConstraintViolation { column } => {
                return HyperstreamError::NullConstraintViolation {
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
        return HyperstreamError::PrimaryKeyViolation {
            key: key.to_string(),
        };
    }
    const NULL_PK_PREFIX: &str = "Null constraint violation: Primary key column '";
    if let Some(pos) = msg.find(NULL_PK_PREFIX) {
        let col = msg[pos + NULL_PK_PREFIX.len()..]
            .split('\'')
            .next()
            .unwrap_or_default();
        return HyperstreamError::NullConstraintViolation {
            column: col.to_string(),
        };
    }
    HyperstreamError::from(err)
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
) -> Result<RecordBatch, HyperstreamError> {
    let obj = doc
        .as_object()
        .ok_or_else(|| HyperstreamError::SchemaIncompatible {
            reason: "document body must be a JSON object".into(),
        })?;
    let mut cols = Vec::with_capacity(target_schema.fields().len());
    for field in target_schema.fields() {
        let values = vec![obj.get(field.name()).filter(|v| !v.is_null()).cloned()];
        let col = infer::value_to_array(field.name(), field.data_type(), &values).map_err(
            |e: InferError| HyperstreamError::SchemaIncompatible {
                reason: e.to_string(),
            },
        )?;
        cols.push(col);
    }
    RecordBatch::try_new(Arc::new(target_schema.clone()), cols).map_err(|e| {
        HyperstreamError::SchemaIncompatible {
            reason: format!("failed to build row batch: {e}"),
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
) -> Result<DocWriteResponse, HyperstreamError> {
    let id = match path_id {
        Some(id) => id.to_string(),
        None => uuid::Uuid::new_v4().to_string(),
    };
    with_id(&mut doc, &id)?;

    // Was the index present before this request? That decides the ES
    // result and whether we install the `_id` primary key.
    let existed_before = table_exists(&state.index_uri(index)).await;

    let doc_schema = infer::infer_schema(&doc).map_err(|e: InferError| {
        HyperstreamError::SchemaIncompatible {
            reason: e.to_string(),
        }
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
            .map_err(HyperstreamError::from)?;
    }

    // Pre-merge the doc schema against the table's current schema so
    // numeric promotion (e.g. Int64 + Float64 -> Float64) happens once,
    // into a single coherent Arrow type.
    let target = infer::merge_schemas(table.arrow_schema().as_ref(), &doc_schema).map_err(
        |e: InferError| HyperstreamError::SchemaIncompatible {
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
fn record_ingest(state: &AppState, result: &Result<DocWriteResponse, HyperstreamError>) {
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

/// `DELETE /{index}/_doc/{id}` — unsupported in v1 (append-only store).
/// Returns a 501 with a clear ES-style error.
pub async fn delete_document(
    State(_state): State<Arc<AppState>>,
    Path((index, _id)): Path<(String, String)>,
) -> Response {
    let es = crate::es_types::EsError {
        error: crate::es_types::EsErrorBody {
            error_type: "unsupported_operation".to_string(),
            reason: format!(
                "per-document delete is not supported on index '{index}' (append-only store); use DELETE /{index} to drop the index"
            ),
        },
        status: 501,
    };
    let status = StatusCode::from_u16(501).unwrap();
    (status, axum::Json(es)).into_response()
}

pub async fn refresh_core(
    state: &AppState,
    index: &str,
) -> Result<RefreshResponse, HyperstreamError> {
    if !table_exists(&state.index_uri(index)).await {
        return Err(HyperstreamError::TableNotFound {
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
            Err(HyperstreamError::SchemaIncompatible { .. })
        ));
    }

    #[test]
    fn translate_write_error_maps_core_strings() {
        // A leading anyhow context chain must not break the match.
        let err = anyhow::anyhow!("write failed: Duplicate primary key error: id = doc-42");
        assert!(matches!(
            translate_write_error(err),
            HyperstreamError::PrimaryKeyViolation { ref key } if key == "doc-42"
        ));

        let err = anyhow::anyhow!(
            "Null constraint violation: Primary key column '_id' cannot contain null values"
        );
        assert!(matches!(
            translate_write_error(err),
            HyperstreamError::NullConstraintViolation { ref column } if column == "_id"
        ));

        let err = anyhow::anyhow!("something exploded");
        assert!(matches!(
            translate_write_error(err),
            HyperstreamError::Internal { .. }
        ));
    }

    #[test]
    fn translate_write_error_maps_typed_core_errors() {
        let err = anyhow::Error::from(HyperstreamError::PrimaryKeyViolation {
            key: "doc-42".into(),
        });
        assert!(matches!(
            translate_write_error(err),
            HyperstreamError::PrimaryKeyViolation { ref key } if key == "doc-42"
        ));

        let err = anyhow::Error::from(HyperstreamError::NullConstraintViolation {
            column: "_id".into(),
        });
        assert!(matches!(
            translate_write_error(err),
            HyperstreamError::NullConstraintViolation { ref column } if column == "_id"
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
                HyperstreamError::PrimaryKeyViolation { key } if key == "bob-1"
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
            matches!(&err, HyperstreamError::TableNotFound { .. }),
            "{err}"
        );
    }
}
