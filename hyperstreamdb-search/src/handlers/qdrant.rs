// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Qdrant REST API handlers.
//!
//! Emulates the subset of the Qdrant REST API used by Zoo Code's codebase
//! indexing, backed by the shared [`AppState`] / `Table` infrastructure.

use std::sync::Arc;

use arrow::array::Array;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use hyperstreamdb::core::index::VectorValue;
use serde_json::Value;

use crate::handlers::docs::{build_row_batch, translate_write_error};
use crate::infer;
use crate::qdrant_types::*;
use crate::state::{table_exists, AppState};

/// Build a Qdrant error response from an arbitrary error message.
fn to_qdrant_err(msg: String, status: StatusCode) -> Response {
    let q_err = QdrantErrorResponse {
        status: QdrantErrorStatus { error: msg },
        time: 0.0,
    };
    (status, Json(q_err)).into_response()
}

fn generic_err(msg: &str, status: StatusCode) -> Response {
    to_qdrant_err(msg.to_string(), status)
}

/// `GET /collections/:name` — collection info (points count + vector size).
pub async fn get_collection(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
) -> Response {
    let uri = state.index_uri(&collection_name);
    if !table_exists(&uri).await {
        return generic_err("Collection not found", StatusCode::NOT_FOUND);
    }

    let table = match state.open_or_create(&collection_name, &None).await {
        Ok(t) => t,
        Err(e) => return to_qdrant_err(e.to_string(), StatusCode::INTERNAL_SERVER_ERROR),
    };

    // Extract the vector dimension from the `vector` column when present.
    let mut vector_size = 1536;
    let schema = table.arrow_schema();
    if let Ok(field) = schema.field_with_name("vector") {
        if let arrow::datatypes::DataType::FixedSizeList(_, size) = field.data_type() {
            vector_size = *size as usize;
        }
    }

    let stats = match table.get_table_statistics_async().await {
        Ok(s) => s,
        Err(e) => return to_qdrant_err(e.to_string(), StatusCode::INTERNAL_SERVER_ERROR),
    };

    let info = CollectionInfoResult {
        status: "green".to_string(),
        vectors_count: stats.row_count as usize,
        points_count: stats.row_count as usize,
        config: CollectionConfig {
            params: CollectionParams {
                vectors: VectorsConfig {
                    size: vector_size,
                    distance: "Cosine".to_string(),
                    on_disk: Some(true),
                },
            },
        },
    };

    Json(QdrantResponse::ok(info, 0.0)).into_response()
}

/// `PUT /collections/:name` — create a collection (materialized on first upsert).
pub async fn create_collection(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(_req): Json<CreateCollectionRequest>,
) -> Response {
    let uri = state.index_uri(&collection_name);
    if table_exists(&uri).await {
        return generic_err("Collection already exists", StatusCode::BAD_REQUEST);
    }
    // The table is materialized lazily on the first upsert.
    Json(QdrantResponse::ok(true, 0.0)).into_response()
}

/// `DELETE /collections/:name` — drop the collection from storage.
pub async fn delete_collection(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
) -> Response {
    let uri = state.index_uri(&collection_name);
    if !table_exists(&uri).await {
        return generic_err("Collection not found", StatusCode::NOT_FOUND);
    }
    // Unload and drop via state
    if let Err(e) = state.delete_index(&collection_name).await {
        return to_qdrant_err(e.to_string(), StatusCode::INTERNAL_SERVER_ERROR);
    }
    Json(QdrantResponse::ok(true, 0.0)).into_response()
}

/// `PUT /collections/:name/index` — payload index creation (no-op; columns are
/// inferred and indexed dynamically on write).
pub async fn create_payload_index(
    Path(_collection_name): Path<String>,
    Json(_req): Json<CreatePayloadIndexRequest>,
) -> Response {
    Json(QdrantResponse::ok(
        UpdateResult {
            operation_id: 0,
            status: "completed".to_string(),
        },
        0.0,
    ))
    .into_response()
}

/// `PUT /collections/:name/points` — upsert points (transformed into Arrow rows).
pub async fn upsert_points(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<UpsertPointsRequest>,
) -> Response {
    let uri = state.index_uri(&collection_name);
    let existed_before = table_exists(&uri).await;

    // Convert Qdrant points into flat JSON documents for HyperStreamDB inference.
    let mut docs = Vec::with_capacity(req.points.len());
    for point in req.points {
        let mut doc_map = serde_json::Map::new();
        doc_map.insert("_id".to_string(), Value::String(point.id.as_string()));
        doc_map.insert(
            "vector".to_string(),
            Value::Array(
                point
                    .vector
                    .iter()
                    .map(|f| Value::from(*f as f64))
                    .collect(),
            ),
        );
        for (k, v) in point.payload {
            doc_map.insert(k, v);
        }
        docs.push(Value::Object(doc_map));
    }

    if docs.is_empty() {
        return Json(QdrantResponse::ok(
            UpdateResult {
                operation_id: 0,
                status: "completed".to_string(),
            },
            0.0,
        ))
        .into_response();
    }

    // Infer a merged schema across all points.
    let mut merged: Option<arrow::datatypes::SchemaRef> = None;
    for doc in &docs {
        let s = match infer::infer_schema(doc) {
            Ok(s) => s,
            Err(e) => {
                return generic_err(
                    &format!("Schema inference failed: {e}"),
                    StatusCode::BAD_REQUEST,
                )
            }
        };
        merged = Some(match merged {
            None => s,
            Some(prev) => match infer::merge_schemas(prev.as_ref(), s.as_ref()) {
                Ok(m) => Arc::new(m),
                Err(e) => {
                    return generic_err(
                        &format!("Schema merge failed: {e}"),
                        StatusCode::BAD_REQUEST,
                    )
                }
            },
        });
    }
    let doc_schema = merged.unwrap();

    // Open (or create) the table with the inferred schema.
    let table = match state
        .open_or_create(&collection_name, &Some(doc_schema.clone()))
        .await
    {
        Ok(t) => t,
        Err(e) => return to_qdrant_err(e.to_string(), StatusCode::INTERNAL_SERVER_ERROR),
    };

    // A table created by this request gets `_id` as its primary key.
    if !existed_before && table.get_primary_key().is_empty() {
        let _ = table.set_primary_key_async(vec!["_id".to_string()]).await;
    }

    // Merge the table's current schema with the document schema.
    let target = match infer::merge_schemas(table.arrow_schema().as_ref(), doc_schema.as_ref()) {
        Ok(t) => t,
        Err(e) => {
            return generic_err(
                &format!("Schema merge failed: {e}"),
                StatusCode::BAD_REQUEST,
            )
        }
    };

    // Build and write the batches in a single call.
    let mut batches = Vec::with_capacity(docs.len());
    for doc in &docs {
        match build_row_batch(&target, doc) {
            Ok(b) => batches.push(b),
            Err(e) => return to_qdrant_err(e.to_string(), StatusCode::BAD_REQUEST),
        }
    }
    if let Err(e) = table.write_async(batches).await {
        return to_qdrant_err(
            translate_write_error(e).to_string(),
            StatusCode::INTERNAL_SERVER_ERROR,
        );
    }

    Json(QdrantResponse::ok(
        UpdateResult {
            operation_id: 0,
            status: "completed".to_string(),
        },
        0.0,
    ))
    .into_response()
}

/// `POST /collections/:name/points` — retrieve points by exact id.
pub async fn retrieve_points(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<RetrievePointsRequest>,
) -> Response {
    let table = match state.open_or_create(&collection_name, &None).await {
        Ok(t) => t,
        Err(_) => return generic_err("Collection not found", StatusCode::NOT_FOUND),
    };

    let batches = match table.read_async(None, None, None).await {
        Ok(b) => b,
        Err(e) => return to_qdrant_err(e.to_string(), StatusCode::INTERNAL_SERVER_ERROR),
    };

    let requested_ids: std::collections::HashSet<String> =
        req.ids.iter().map(|id| id.as_string()).collect();
    let mut retrieved = Vec::new();

    for batch in &batches {
        let schema = batch.schema();
        let id_col_idx = match schema.index_of("_id") {
            Ok(idx) => idx,
            Err(_) => continue,
        };
        let id_arr = match batch
            .column(id_col_idx)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
        {
            Some(a) => a,
            None => continue,
        };

        for row_idx in 0..batch.num_rows() {
            if id_arr.is_null(row_idx) {
                continue;
            }
            let id_val = id_arr.value(row_idx).to_string();
            if !requested_ids.contains(&id_val) {
                continue;
            }

            let mut payload = std::collections::HashMap::new();
            for (col_idx, field) in schema.fields().iter().enumerate() {
                let name = field.name();
                if name == "_id" || name == "vector" {
                    continue;
                }
                let col = batch.column(col_idx);
                if col.is_null(row_idx) {
                    continue;
                }
                if let Some(str_arr) = col.as_any().downcast_ref::<arrow::array::StringArray>() {
                    payload.insert(
                        name.clone(),
                        Value::String(str_arr.value(row_idx).to_string()),
                    );
                } else if let Some(bool_arr) =
                    col.as_any().downcast_ref::<arrow::array::BooleanArray>()
                {
                    payload.insert(name.clone(), Value::Bool(bool_arr.value(row_idx)));
                } else if let Some(int_arr) =
                    col.as_any().downcast_ref::<arrow::array::Int64Array>()
                {
                    payload.insert(
                        name.clone(),
                        Value::Number(serde_json::Number::from(int_arr.value(row_idx))),
                    );
                } else if let Some(float_arr) =
                    col.as_any().downcast_ref::<arrow::array::Float64Array>()
                {
                    if let Some(n) = serde_json::Number::from_f64(float_arr.value(row_idx)) {
                        payload.insert(name.clone(), Value::Number(n));
                    }
                }
            }

            retrieved.push(RetrievedPoint {
                id: PointId::Uuid(id_val),
                payload,
            });
        }
    }

    Json(QdrantResponse::ok(retrieved, 0.0)).into_response()
}

/// `POST /collections/:name/points/search` — vector similarity search.
pub async fn query_points(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<QueryPointsRequest>,
) -> Response {
    let table = match state.open_or_create(&collection_name, &None).await {
        Ok(t) => t,
        Err(e) => return to_qdrant_err(e.to_string(), StatusCode::INTERNAL_SERVER_ERROR),
    };

    let limit = req.limit.unwrap_or(10);
    let q = table
        .query()
        .vector_search("vector", VectorValue::Float32(req.query), limit);
    if let Some(_params) = req.params {
        // EF search could be passed through config if needed, skipping for simple emulate
    }

    let batches = match q.to_batches().await {
        Ok(res) => res,
        Err(e) => return to_qdrant_err(e.to_string(), StatusCode::INTERNAL_SERVER_ERROR),
    };

    let mut scored_points = Vec::new();
    for batch in &batches {
        let schema = batch.schema();
        let id_col_idx = match schema.index_of("_id") {
            Ok(idx) => idx,
            Err(_) => continue,
        };
        let id_arr = match batch
            .column(id_col_idx)
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
        {
            Some(a) => a,
            None => continue,
        };

        // Vector search adds a "distance" column at the end
        let dist_col_idx = batch.num_columns() - 1;
        let dist_arr = batch
            .column(dist_col_idx)
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>();

        for row_idx in 0..batch.num_rows() {
            if id_arr.is_null(row_idx) {
                continue;
            }
            let id_val = id_arr.value(row_idx).to_string();
            let score = dist_arr.map(|a| a.value(row_idx)).unwrap_or(0.0);

            let mut payload = std::collections::HashMap::new();
            for (col_idx, field) in schema.fields().iter().enumerate() {
                let name = field.name();
                if name == "_id" || name == "vector" || name == "distance" {
                    continue;
                }
                let col = batch.column(col_idx);
                if col.is_null(row_idx) {
                    continue;
                }
                if let Some(str_arr) = col.as_any().downcast_ref::<arrow::array::StringArray>() {
                    payload.insert(
                        name.clone(),
                        Value::String(str_arr.value(row_idx).to_string()),
                    );
                } else if let Some(bool_arr) =
                    col.as_any().downcast_ref::<arrow::array::BooleanArray>()
                {
                    payload.insert(name.clone(), Value::Bool(bool_arr.value(row_idx)));
                } else if let Some(int_arr) =
                    col.as_any().downcast_ref::<arrow::array::Int64Array>()
                {
                    payload.insert(
                        name.clone(),
                        Value::Number(serde_json::Number::from(int_arr.value(row_idx))),
                    );
                } else if let Some(float_arr) =
                    col.as_any().downcast_ref::<arrow::array::Float64Array>()
                {
                    if let Some(n) = serde_json::Number::from_f64(float_arr.value(row_idx)) {
                        payload.insert(name.clone(), Value::Number(n));
                    }
                }
            }

            scored_points.push(ScoredPoint {
                id: PointId::Uuid(id_val),
                version: 0,
                score,
                payload: Some(payload),
            });
        }
    }

    #[derive(serde::Serialize)]
    struct QueryResult {
        points: Vec<ScoredPoint>,
    }

    Json(QdrantResponse::ok(
        QueryResult {
            points: scored_points,
        },
        0.0,
    ))
    .into_response()
}

/// `POST /collections/:name/points/delete` — delete points (no-op in v1; the
/// store is append-only).
pub async fn delete_points(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(_req): Json<DeletePointsRequest>,
) -> Response {
    let uri = state.index_uri(&collection_name);
    if !table_exists(&uri).await {
        return generic_err("Collection not found", StatusCode::NOT_FOUND);
    }
    Json(QdrantResponse::ok(
        UpdateResult {
            operation_id: 0,
            status: "completed".to_string(),
        },
        0.0,
    ))
    .into_response()
}
