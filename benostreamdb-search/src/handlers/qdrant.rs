// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Qdrant REST API handlers.
//!
//! Implements the Qdrant v1.x REST surface (collections, points, payloads,
//! vectors, aliases, and service endpoints) on top of the shared
//! [`AppState`] / [`Table`] infrastructure. See `docs/QDRANT_COMPATIBILITY.md`
//! for the exact support matrix.
//!
//! ## Storage model
//!
//! A Qdrant collection is a BenoStreamDB table with a reserved `_id` (Utf8)
//! column and a `vector` (`FixedSizeList<Float32>`) column; payload fields are
//! inferred as additional Arrow columns. Collection parameters that the engine
//! has no native home for (the distance metric, HNSW config) are persisted in
//! a small sidecar object `_qdrant_collection.json` under the collection root.
//!
//! ## Write semantics
//!
//! Upserts are implemented as *flush → delete-by-id → append*: the write
//! buffer is committed, the previous rows for the affected ids are marked
//! deleted with Iceberg position-delete files, and the replacement rows are
//! appended. Reads apply the delete files, so the effective row set is the
//! latest write per `_id` (merge-on-read semantics). Payload and vector edits
//! are read-modify-write upserts built on the same primitive.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int32Array, Int64Array,
    StringArray, StructArray,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use benostreamdb::core::index::{VectorMetric, VectorValue};
use benostreamdb::core::planner::VectorSearchParams;
use benostreamdb::Table;
use serde::Serialize;
use serde_json::Value;

use crate::handlers::docs::{build_row_batch, translate_write_error};
use crate::infer;
use crate::qdrant_types::*;
use crate::state::{table_exists, AppState};

// ==========================================
// Error + envelope helpers
// ==========================================

/// A Qdrant-shaped error with an HTTP status.
#[derive(Debug)]
struct QErr {
    msg: String,
    status: StatusCode,
}

impl QErr {
    fn not_found(m: impl Into<String>) -> Self {
        Self {
            msg: m.into(),
            status: StatusCode::NOT_FOUND,
        }
    }
    fn bad(m: impl Into<String>) -> Self {
        Self {
            msg: m.into(),
            status: StatusCode::BAD_REQUEST,
        }
    }
    fn internal(m: impl Into<String>) -> Self {
        Self {
            msg: m.into(),
            status: StatusCode::INTERNAL_SERVER_ERROR,
        }
    }
}

fn ok<T: Serialize>(result: T, start: Instant) -> Response {
    Json(QdrantResponse::ok(result, start.elapsed().as_secs_f64())).into_response()
}

fn err(msg: impl Into<String>, status: StatusCode, start: Instant) -> Response {
    let body = QdrantErrorResponse {
        status: QdrantErrorStatus { error: msg.into() },
        time: start.elapsed().as_secs_f64(),
    };
    (status, Json(body)).into_response()
}

fn qerr(e: QErr, start: Instant) -> Response {
    err(e.msg, e.status, start)
}

fn process_start() -> Instant {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    *START.get_or_init(Instant::now)
}

// ==========================================
// Collection metadata sidecar
// ==========================================

const META_FILE: &str = "_qdrant_collection.json";

#[derive(Debug, Serialize, serde::Deserialize, Clone)]
struct CollectionMeta {
    size: usize,
    distance: String,
    #[serde(default)]
    on_disk: bool,
    #[serde(default)]
    hnsw_config: Option<HnswConfig>,
}

impl Default for CollectionMeta {
    fn default() -> Self {
        Self {
            size: 1536,
            distance: "Cosine".to_string(),
            on_disk: true,
            hnsw_config: None,
        }
    }
}

fn meta_path() -> object_store::path::Path {
    object_store::path::Path::from(META_FILE)
}

async fn read_meta(state: &AppState, name: &str) -> Option<CollectionMeta> {
    let uri = state.index_uri(name);
    let store = benostreamdb::core::storage::create_object_store(&uri).ok()?;
    let bytes = store.get(&meta_path()).await.ok()?.bytes().await.ok()?;
    serde_json::from_slice(&bytes).ok()
}

async fn write_meta(state: &AppState, name: &str, meta: &CollectionMeta) -> Result<(), QErr> {
    let uri = state.index_uri(name);
    let store = benostreamdb::core::storage::create_object_store(&uri)
        .map_err(|e| QErr::internal(e.to_string()))?;
    let bytes = serde_json::to_vec(meta).map_err(|e| QErr::internal(e.to_string()))?;
    store
        .put(&meta_path(), bytes.into())
        .await
        .map_err(|e| QErr::internal(e.to_string()))?;
    Ok(())
}

async fn load_meta_or_default(state: &AppState, name: &str) -> CollectionMeta {
    read_meta(state, name).await.unwrap_or_default()
}

/// Resolve an alias to its target collection name (identity when not an alias).
async fn resolve(state: &AppState, name: &str) -> String {
    let aliases = state.aliases.read().await;
    aliases
        .get(name)
        .cloned()
        .unwrap_or_else(|| name.to_string())
}

async fn collection_exists(state: &AppState, name: &str) -> bool {
    let name = resolve(state, name).await;
    table_exists(&state.index_uri(&name)).await || read_meta(state, &name).await.is_some()
}

/// Open an existing collection, or fail with a 404-shaped error.
async fn open_existing(state: &AppState, name: &str) -> Result<Arc<Table>, QErr> {
    let name = resolve(state, name).await;
    if !table_exists(&state.index_uri(&name)).await {
        return Err(QErr::not_found("Collection not found"));
    }
    state
        .open_or_create_qdrant(&name, &None)
        .await
        .map_err(|e| QErr::internal(e.to_string()))
}

// ==========================================
// Distance / score semantics
// ==========================================

fn metric_for(distance: &str) -> VectorMetric {
    match distance.to_ascii_lowercase().as_str() {
        "cosine" => VectorMetric::Cosine,
        "dot" | "ip" | "inner_product" => VectorMetric::InnerProduct,
        "manhattan" | "l1" => VectorMetric::L1,
        _ => VectorMetric::L2,
    }
}

/// Whether a higher score is better for this distance (Qdrant similarity
/// metrics) as opposed to a lower distance being better.
fn higher_is_better(distance: &str) -> bool {
    matches!(
        distance.to_ascii_lowercase().as_str(),
        "cosine" | "dot" | "ip" | "inner_product"
    )
}

/// Convert the engine's trailing `distance` column into a Qdrant score.
///
/// The engine returns a *distance* (lower is better) for every metric:
/// squared L2 for `L2`, `1 - cos` for `Cosine`, `-dot` for `InnerProduct`,
/// and the raw L1 distance for `L1`. Qdrant returns a *similarity* (higher is
/// better) for Cosine/Dot and a true Euclidean distance for Euclid.
fn to_qdrant_score(engine_distance: f32, distance: &str) -> f32 {
    match distance.to_ascii_lowercase().as_str() {
        "cosine" => 1.0 - engine_distance,
        "dot" | "ip" | "inner_product" => -engine_distance,
        "manhattan" | "l1" => engine_distance,
        // Euclid / L2: the engine returns squared L2; Qdrant returns the
        // Euclidean distance.
        _ => engine_distance.max(0.0).sqrt(),
    }
}

// ==========================================
// Filter → SQL translation
// ==========================================

fn valid_field(field: &str) -> bool {
    !field.is_empty()
        && field.chars().enumerate().all(|(i, c)| {
            if i == 0 {
                c == '_' || c.is_ascii_alphabetic()
            } else {
                c.is_ascii_alphanumeric() || c == '_'
            }
        })
}

fn sql_string(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

fn sql_literal(v: &Value) -> String {
    match v {
        Value::String(s) => sql_string(s),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::Null => "NULL".to_string(),
        other => sql_string(&other.to_string()),
    }
}

fn ids_filter(ids: &[PointId]) -> String {
    let lits: Vec<String> = ids.iter().map(|i| sql_string(&i.as_string())).collect();
    format!("_id IN ({})", lits.join(", "))
}

fn filter_to_sql(filter: &Filter) -> Result<Option<String>, QErr> {
    let mut parts: Vec<String> = Vec::new();
    if let Some(must) = &filter.must {
        for c in must.clone().into_vec() {
            parts.push(condition_to_sql(&c)?);
        }
    }
    if let Some(should) = &filter.should {
        let conds = should.clone().into_vec();
        if !conds.is_empty() {
            let mut ors = Vec::with_capacity(conds.len());
            for c in conds {
                ors.push(condition_to_sql(&c)?);
            }
            parts.push(format!("({})", ors.join(" OR ")));
        }
    }
    if let Some(mn) = &filter.must_not {
        for c in mn.clone().into_vec() {
            parts.push(format!("NOT ({})", condition_to_sql(&c)?));
        }
    }
    if parts.is_empty() {
        Ok(None)
    } else {
        Ok(Some(parts.join(" AND ")))
    }
}

fn condition_to_sql(cond: &Condition) -> Result<String, QErr> {
    match cond {
        Condition::Field(fc) => field_condition_to_sql(fc),
        Condition::HasId { has_id } => Ok(ids_filter(has_id)),
        Condition::Filter(f) => Ok(filter_to_sql(f)?.unwrap_or_else(|| "true".to_string())),
    }
}

fn field_condition_to_sql(fc: &FieldCondition) -> Result<String, QErr> {
    let key = &fc.key;
    if !valid_field(key) {
        return Err(QErr::bad(format!("invalid field name '{key}'")));
    }
    let mut parts: Vec<String> = Vec::new();
    if let Some(m) = &fc.match_ {
        match m {
            MatchCondition::Value { value } => {
                parts.push(format!("{key} = {}", sql_literal(value)));
            }
            MatchCondition::Any { any } => {
                let lits: Vec<String> = any.iter().map(sql_literal).collect();
                parts.push(format!("{key} IN ({})", lits.join(", ")));
            }
            MatchCondition::Text { text } => {
                parts.push(format!("{key} = {}", sql_string(text)));
            }
        }
    }
    if let Some(r) = &fc.range {
        if let Some(v) = r.gt {
            parts.push(format!("{key} > {v}"));
        }
        if let Some(v) = r.gte {
            parts.push(format!("{key} >= {v}"));
        }
        if let Some(v) = r.lt {
            parts.push(format!("{key} < {v}"));
        }
        if let Some(v) = r.lte {
            parts.push(format!("{key} <= {v}"));
        }
    }
    if parts.is_empty() {
        return Err(QErr::bad(format!(
            "condition on '{key}' has neither 'match' nor 'range'"
        )));
    }
    Ok(parts.join(" AND "))
}

// ==========================================
// Arrow → JSON helpers
// ==========================================

fn arrow_value_to_json(array: &dyn Array, row: usize) -> Option<Value> {
    if array.is_null(row) {
        return None;
    }
    match array.data_type() {
        DataType::Utf8 => array
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|a| Value::String(a.value(row).to_string())),
        DataType::Boolean => array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .map(|a| Value::Bool(a.value(row))),
        DataType::Int64 => array
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|a| Value::Number(a.value(row).into())),
        DataType::Int32 => array
            .as_any()
            .downcast_ref::<Int32Array>()
            .map(|a| Value::Number((a.value(row) as i64).into())),
        DataType::Float64 => array
            .as_any()
            .downcast_ref::<Float64Array>()
            .and_then(|a| serde_json::Number::from_f64(a.value(row)).map(Value::Number)),
        DataType::Float32 => array
            .as_any()
            .downcast_ref::<Float32Array>()
            .and_then(|a| serde_json::Number::from_f64(a.value(row) as f64).map(Value::Number)),
        DataType::FixedSizeList(_, _) => {
            let a = array.as_any().downcast_ref::<FixedSizeListArray>()?;
            let vals = a.value(row);
            let mut out = Vec::with_capacity(vals.len());
            for i in 0..vals.len() {
                out.push(arrow_value_to_json(vals.as_ref(), i).unwrap_or(Value::Null));
            }
            Some(Value::Array(out))
        }
        DataType::Struct(fields) => {
            let a = array.as_any().downcast_ref::<StructArray>()?;
            let mut map = serde_json::Map::new();
            for (i, f) in fields.iter().enumerate() {
                if let Some(v) = arrow_value_to_json(a.column(i).as_ref(), row) {
                    map.insert(f.name().clone(), v);
                }
            }
            Some(Value::Object(map))
        }
        _ => None,
    }
}

fn vector_from_array(array: &dyn Array, row: usize) -> Option<Vec<f32>> {
    let a = array.as_any().downcast_ref::<FixedSizeListArray>()?;
    if a.is_null(row) {
        return None;
    }
    let vals = a.value(row);
    let f = vals.as_any().downcast_ref::<Float32Array>()?;
    Some(f.values().to_vec())
}

fn parse_point_id(s: &str) -> PointId {
    match s.parse::<u64>() {
        Ok(n) => PointId::Num(n),
        Err(_) => PointId::Uuid(s.to_string()),
    }
}

fn payload_from_batch(
    batch: &RecordBatch,
    row: usize,
    with_payload: &WithPayload,
) -> Option<HashMap<String, Value>> {
    if !with_payload.enabled() {
        return None;
    }
    let schema = batch.schema();
    let mut map = HashMap::new();
    for (i, f) in schema.fields().iter().enumerate() {
        let name = f.name();
        if name == "_id" || name == "vector" || name == "distance" {
            continue;
        }
        if !with_payload.keep(name) {
            continue;
        }
        if let Some(v) = arrow_value_to_json(batch.column(i).as_ref(), row) {
            map.insert(name.clone(), v);
        }
    }
    Some(map)
}

fn batch_to_retrieved(
    batch: &RecordBatch,
    with_payload: &WithPayload,
    with_vector: bool,
) -> Vec<RetrievedPoint> {
    let schema = batch.schema();
    let id_idx = match schema.index_of("_id") {
        Ok(i) => i,
        Err(_) => return Vec::new(),
    };
    let id_arr = batch.column(id_idx).as_any().downcast_ref::<StringArray>();
    let vec_idx = schema.index_of("vector").ok();
    let mut out = Vec::new();
    for row in 0..batch.num_rows() {
        let id = match id_arr {
            Some(a) if !a.is_null(row) => a.value(row).to_string(),
            _ => continue,
        };
        let payload = payload_from_batch(batch, row, with_payload);
        let vector = if with_vector {
            vec_idx.and_then(|i| vector_from_array(batch.column(i).as_ref(), row))
        } else {
            None
        };
        out.push(RetrievedPoint {
            id: parse_point_id(&id),
            payload,
            vector,
        });
    }
    out
}

fn batch_to_scored(
    batch: &RecordBatch,
    distance: &str,
    with_payload: &WithPayload,
    with_vector: bool,
) -> Vec<ScoredPoint> {
    let schema = batch.schema();
    let id_idx = match schema.index_of("_id") {
        Ok(i) => i,
        Err(_) => return Vec::new(),
    };
    let id_arr = batch.column(id_idx).as_any().downcast_ref::<StringArray>();
    let vec_idx = schema.index_of("vector").ok();
    let dist_idx = schema.index_of("distance").ok();
    let mut out = Vec::new();
    for row in 0..batch.num_rows() {
        let id = match id_arr {
            Some(a) if !a.is_null(row) => a.value(row).to_string(),
            _ => continue,
        };
        let raw = dist_idx
            .and_then(|i| {
                batch
                    .column(i)
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .map(|a| a.value(row))
            })
            .unwrap_or(0.0);
        let payload = payload_from_batch(batch, row, with_payload);
        let vector = if with_vector {
            vec_idx.and_then(|i| vector_from_array(batch.column(i).as_ref(), row))
        } else {
            None
        };
        out.push(ScoredPoint {
            id: parse_point_id(&id),
            version: 0,
            score: to_qdrant_score(raw, distance),
            payload,
            vector,
        });
    }
    out
}

// ==========================================
// Read / write cores
// ==========================================

async fn read_batches(
    state: &AppState,
    name: &str,
    filter_sql: Option<&str>,
    columns: Option<&[&str]>,
) -> Result<Vec<RecordBatch>, QErr> {
    let table = open_existing(state, name).await?;
    table
        .read_async(filter_sql, None, columns)
        .await
        .map_err(|e| QErr::internal(e.to_string()))
}

async fn count_points(
    state: &AppState,
    name: &str,
    filter_sql: Option<&str>,
) -> Result<usize, QErr> {
    // Read every column: the engine evaluates the filter against the scanned
    // batch, so projecting to `_id` would drop the filter column.
    let batches = read_batches(state, name, filter_sql, None).await?;
    Ok(batches.iter().map(|b| b.num_rows()).sum())
}

/// Build the flat JSON documents the engine infers a schema from.
fn points_to_docs(points: &[PointStruct]) -> Vec<Value> {
    let mut docs = Vec::with_capacity(points.len());
    for p in points {
        let mut doc = serde_json::Map::new();
        doc.insert("_id".to_string(), Value::String(p.id.as_string()));
        if let Some(v) = p.vector.as_ref().and_then(|v| v.primary()) {
            doc.insert(
                "vector".to_string(),
                Value::Array(v.iter().map(|f| Value::from(*f as f64)).collect()),
            );
        }
        if let Some(payload) = &p.payload {
            for (k, v) in payload {
                if k == "_id" || k == "vector" {
                    continue;
                }
                doc.insert(k.clone(), v.clone());
            }
        }
        docs.push(Value::Object(doc));
    }
    docs
}

/// Upsert points: flush → delete-by-id → append (merge-on-read semantics).
async fn upsert_core(state: &AppState, name: &str, points: Vec<PointStruct>) -> Result<(), QErr> {
    if points.is_empty() {
        return Ok(());
    }
    let name = resolve(state, name).await;
    let uri = state.index_uri(&name);
    let existed_before = table_exists(&uri).await;

    let docs = points_to_docs(&points);
    let ids: Vec<String> = points.iter().map(|p| p.id.as_string()).collect();

    // Infer a merged schema across all points.
    let mut merged: Option<SchemaRef> = None;
    for doc in &docs {
        let s = infer::infer_schema(doc)
            .map_err(|e| QErr::bad(format!("Schema inference failed: {e}")))?;
        merged = Some(match merged {
            None => s,
            Some(prev) => Arc::new(
                infer::merge_schemas(prev.as_ref(), s.as_ref())
                    .map_err(|e| QErr::bad(format!("Schema merge failed: {e}")))?,
            ),
        });
    }
    let doc_schema = merged.ok_or_else(|| QErr::bad("No points supplied"))?;

    let table = state
        .open_or_create_qdrant(&name, &Some(doc_schema.clone()))
        .await
        .map_err(|e| QErr::internal(e.to_string()))?;

    let target = infer::merge_schemas(table.arrow_schema().as_ref(), doc_schema.as_ref())
        .map_err(|e| QErr::bad(format!("Schema merge failed: {e}")))?;

    let mut batches = Vec::with_capacity(docs.len());
    for doc in &docs {
        batches.push(build_row_batch(&target, doc).map_err(|e| QErr::bad(e.to_string()))?);
    }

    if existed_before {
        // Flush the write buffer so the delete sees every previously written
        // row, then mark the old rows deleted before appending replacements.
        table
            .commit_async()
            .await
            .map_err(|e| QErr::internal(e.to_string()))?;
        let filter = ids_filter(
            &ids.iter()
                .map(|i| PointId::Uuid(i.clone()))
                .collect::<Vec<_>>(),
        );
        table
            .delete_async(&filter)
            .await
            .map_err(|e| QErr::internal(e.to_string()))?;
    }

    table
        .write_async(batches)
        .await
        .map_err(|e| QErr::internal(translate_write_error(e).to_string()))?;
    Ok(())
}

struct PointRow {
    id: String,
    vector: Option<Vec<f32>>,
    payload: HashMap<String, Value>,
}

async fn read_point_rows(
    state: &AppState,
    name: &str,
    filter_sql: Option<&str>,
) -> Result<Vec<PointRow>, QErr> {
    let batches = read_batches(state, name, filter_sql, None).await?;
    let mut rows = Vec::new();
    for batch in &batches {
        let schema = batch.schema();
        let id_idx = match schema.index_of("_id") {
            Ok(i) => i,
            Err(_) => continue,
        };
        let id_arr = batch.column(id_idx).as_any().downcast_ref::<StringArray>();
        let vec_idx = schema.index_of("vector").ok();
        for row in 0..batch.num_rows() {
            let id = match id_arr {
                Some(a) if !a.is_null(row) => a.value(row).to_string(),
                _ => continue,
            };
            let vector = vec_idx.and_then(|i| vector_from_array(batch.column(i).as_ref(), row));
            let mut payload = HashMap::new();
            for (i, f) in schema.fields().iter().enumerate() {
                let n = f.name();
                if n == "_id" || n == "vector" || n == "distance" {
                    continue;
                }
                if let Some(v) = arrow_value_to_json(batch.column(i).as_ref(), row) {
                    payload.insert(n.clone(), v);
                }
            }
            rows.push(PointRow {
                id,
                vector,
                payload,
            });
        }
    }
    Ok(rows)
}

enum PayloadOp {
    Set(HashMap<String, Value>),
    Overwrite(HashMap<String, Value>),
    DeleteKeys(Vec<String>),
    Clear,
}

fn selector_filter(
    points: &Option<Vec<PointId>>,
    filter: &Option<Filter>,
) -> Result<Option<String>, QErr> {
    match (points, filter) {
        (Some(ids), _) if !ids.is_empty() => Ok(Some(ids_filter(ids))),
        (_, Some(f)) => filter_to_sql(f),
        _ => Err(QErr::bad(
            "either 'points' or 'filter' must be provided".to_string(),
        )),
    }
}

async fn apply_payload_op(
    state: &AppState,
    name: &str,
    points: Option<Vec<PointId>>,
    filter: Option<Filter>,
    op: PayloadOp,
) -> Result<(), QErr> {
    let filter_sql = selector_filter(&points, &filter)?;
    let rows = read_point_rows(state, name, filter_sql.as_deref()).await?;
    if rows.is_empty() {
        return Ok(());
    }
    let mut new_points = Vec::with_capacity(rows.len());
    for r in rows {
        let mut payload = r.payload;
        match &op {
            PayloadOp::Set(new) => {
                for (k, v) in new {
                    payload.insert(k.clone(), v.clone());
                }
            }
            PayloadOp::Overwrite(new) => {
                payload = new.clone();
            }
            PayloadOp::DeleteKeys(keys) => {
                for k in keys {
                    payload.remove(k);
                }
            }
            PayloadOp::Clear => payload.clear(),
        }
        new_points.push(PointStruct {
            id: parse_point_id(&r.id),
            vector: r.vector.map(VectorInput::Single),
            payload: Some(payload),
        });
    }
    upsert_core(state, name, new_points).await
}

async fn update_vectors_core(
    state: &AppState,
    name: &str,
    points: Vec<PointVectors>,
) -> Result<(), QErr> {
    if points.is_empty() {
        return Ok(());
    }
    let ids: Vec<PointId> = points.iter().map(|p| p.id.clone()).collect();
    let filter_sql = ids_filter(&ids);
    let rows = read_point_rows(state, name, Some(&filter_sql)).await?;
    let mut by_id: HashMap<String, PointRow> =
        rows.into_iter().map(|r| (r.id.clone(), r)).collect();
    let mut new_points = Vec::with_capacity(points.len());
    for p in points {
        let id = p.id.as_string();
        let vector = p.vector.primary().cloned();
        let payload = by_id.remove(&id).map(|r| r.payload).unwrap_or_default();
        new_points.push(PointStruct {
            id: p.id,
            vector: vector.map(VectorInput::Single),
            payload: Some(payload),
        });
    }
    upsert_core(state, name, new_points).await
}

async fn delete_points_core(
    state: &AppState,
    name: &str,
    req: &DeletePointsRequest,
) -> Result<(), QErr> {
    let filter_sql = selector_filter(&req.points, &req.filter)?;
    let Some(filter_sql) = filter_sql else {
        return Ok(());
    };
    let table = open_existing(state, name).await?;
    table
        .commit_async()
        .await
        .map_err(|e| QErr::internal(e.to_string()))?;
    table
        .delete_async(&filter_sql)
        .await
        .map_err(|e| QErr::internal(e.to_string()))?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn search_core(
    state: &AppState,
    name: &str,
    vector: Vec<f32>,
    filter: Option<&Filter>,
    limit: Option<usize>,
    offset: Option<usize>,
    with_payload: Option<&WithPayload>,
    with_vector: Option<&WithVector>,
    score_threshold: Option<f32>,
) -> Result<Vec<ScoredPoint>, QErr> {
    let meta = load_meta_or_default(state, name).await;
    let table = open_existing(state, name).await?;
    let filter_sql = match filter {
        Some(f) => filter_to_sql(f)?,
        None => None,
    };
    let limit = limit.unwrap_or(10).max(1);
    let offset = offset.unwrap_or(0);
    let k = limit + offset;
    let params = VectorSearchParams::new("vector", VectorValue::Float32(vector), k)
        .with_metric(metric_for(&meta.distance));
    let batches = table
        .read_async(filter_sql.as_deref(), Some(vec![params]), None)
        .await
        .map_err(|e| QErr::internal(e.to_string()))?;

    let wp = with_payload.cloned().unwrap_or(WithPayload::Bool(true));
    let wv = with_vector.map(|w| w.enabled()).unwrap_or(false);
    let mut points = Vec::new();
    for b in &batches {
        points.extend(batch_to_scored(b, &meta.distance, &wp, wv));
    }

    let hib = higher_is_better(&meta.distance);
    points.sort_by(|a, b| {
        let ord = if hib {
            b.score.partial_cmp(&a.score)
        } else {
            a.score.partial_cmp(&b.score)
        };
        ord.unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.as_string().cmp(&b.id.as_string()))
    });

    if let Some(t) = score_threshold {
        points.retain(|p| if hib { p.score >= t } else { p.score <= t });
    }

    Ok(points.into_iter().skip(offset).take(limit).collect())
}

async fn scroll_core(
    state: &AppState,
    name: &str,
    filter: Option<&Filter>,
    limit: Option<usize>,
    offset: Option<PointId>,
    with_payload: Option<&WithPayload>,
    with_vector: Option<&WithVector>,
) -> Result<ScrollResult, QErr> {
    let filter_sql = match filter {
        Some(f) => filter_to_sql(f)?,
        None => None,
    };
    let batches = read_batches(state, name, filter_sql.as_deref(), None).await?;
    let wp = with_payload.cloned().unwrap_or(WithPayload::Bool(true));
    let wv = with_vector.map(|w| w.enabled()).unwrap_or(false);
    let mut points = Vec::new();
    for b in &batches {
        points.extend(batch_to_retrieved(b, &wp, wv));
    }
    points.sort_by_key(|a| a.id.as_string());

    let start_idx = match offset {
        Some(o) => {
            let oid = o.as_string();
            points
                .iter()
                .position(|p| p.id.as_string() == oid)
                .map(|i| i + 1)
                .unwrap_or(0)
        }
        None => 0,
    };
    let limit = limit.unwrap_or(10).max(1);
    let page: Vec<RetrievedPoint> = points.into_iter().skip(start_idx).take(limit).collect();
    let next_page_offset = if page.len() == limit {
        page.last().map(|p| p.id.clone())
    } else {
        None
    };
    Ok(ScrollResult {
        points: page,
        next_page_offset,
    })
}

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

// ==========================================
// Collection handlers
// ==========================================

/// `GET /collections` — list collections.
pub async fn list_collections(State(state): State<Arc<AppState>>) -> Response {
    let start = Instant::now();
    match state.list_indexes().await {
        Ok(names) => ok(
            CollectionsListResult {
                collections: names
                    .into_iter()
                    .map(|name| CollectionDescription { name })
                    .collect(),
            },
            start,
        ),
        Err(e) => qerr(QErr::internal(e.to_string()), start),
    }
}

/// `GET /collections/:name/exists` — collection existence.
pub async fn collection_exists_handler(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
) -> Response {
    let start = Instant::now();
    let exists = collection_exists(&state, &collection_name).await;
    ok(CollectionExistsResult { exists }, start)
}

/// `GET /collections/:name` — collection info (real point count + vector size).
pub async fn get_collection(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
) -> Response {
    let start = Instant::now();
    if !collection_exists(&state, &collection_name).await {
        return qerr(QErr::not_found("Collection not found"), start);
    }
    let meta = load_meta_or_default(&state, &collection_name).await;
    let table = match open_existing(&state, &collection_name).await {
        Ok(t) => t,
        Err(e) => return qerr(e, start),
    };

    let mut vector_size = meta.size;
    let schema = table.arrow_schema();
    if let Ok(field) = schema.field_with_name("vector") {
        if let DataType::FixedSizeList(_, size) = field.data_type() {
            vector_size = *size as usize;
        }
    }

    let count = match count_points(&state, &collection_name, None).await {
        Ok(c) => c,
        Err(e) => return qerr(e, start),
    };

    let info = CollectionInfoResult {
        status: "green".to_string(),
        optimizer_status: "ok".to_string(),
        vectors_count: count,
        indexed_vectors_count: count,
        points_count: count,
        segments_count: 1,
        config: CollectionConfig {
            params: CollectionParams {
                vectors: VectorsConfig {
                    size: vector_size,
                    distance: meta.distance.clone(),
                    on_disk: Some(meta.on_disk),
                },
            },
            hnsw_config: meta.hnsw_config.clone(),
        },
        payload_schema: HashMap::new(),
    };
    ok(info, start)
}

/// `PUT /collections/:name` — create a collection (eagerly materialized).
pub async fn create_collection(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<CreateCollectionRequest>,
) -> Response {
    let start = Instant::now();
    if collection_exists(&state, &collection_name).await {
        return qerr(QErr::bad("Collection already exists"), start);
    }
    let size = req.vectors.size;
    if size == 0 {
        return qerr(QErr::bad("vectors.size must be greater than 0"), start);
    }
    let schema = Arc::new(Schema::new(vec![
        Field::new("_id", DataType::Utf8, true),
        Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                size as i32,
            ),
            true,
        ),
    ]));
    if let Err(e) = state
        .open_or_create_qdrant(&collection_name, &Some(schema))
        .await
    {
        return qerr(QErr::internal(e.to_string()), start);
    }
    let meta = CollectionMeta {
        size,
        distance: req.vectors.distance.clone(),
        on_disk: req.vectors.on_disk.unwrap_or(true),
        hnsw_config: req.hnsw_config.clone(),
    };
    if let Err(e) = write_meta(&state, &collection_name, &meta).await {
        return qerr(e, start);
    }
    ok(true, start)
}

/// `PATCH /collections/:name` — update collection parameters.
pub async fn update_collection(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<UpdateCollectionRequest>,
) -> Response {
    let start = Instant::now();
    if !collection_exists(&state, &collection_name).await {
        return qerr(QErr::not_found("Collection not found"), start);
    }
    let mut meta = load_meta_or_default(&state, &collection_name).await;
    if let Some(hnsw) = req.hnsw_config {
        meta.hnsw_config = Some(hnsw);
    }
    if let Err(e) = write_meta(&state, &collection_name, &meta).await {
        return qerr(e, start);
    }
    ok(true, start)
}

/// `DELETE /collections/:name` — drop the collection from storage.
pub async fn delete_collection(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
) -> Response {
    let start = Instant::now();
    let name = resolve(&state, &collection_name).await;
    if !collection_exists(&state, &name).await {
        return qerr(QErr::not_found("Collection not found"), start);
    }
    if let Err(e) = state.delete_index(&name).await {
        return qerr(QErr::internal(e.to_string()), start);
    }
    // Drop any aliases pointing at the deleted collection.
    {
        let mut aliases = state.aliases.write().await;
        aliases.retain(|_, target| target != &name);
    }
    ok(true, start)
}

/// `PUT /collections/:name/index` — payload index creation (no-op; columns
/// are inferred and indexed dynamically on write).
pub async fn create_payload_index(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(_req): Json<CreatePayloadIndexRequest>,
) -> Response {
    let start = Instant::now();
    if !collection_exists(&state, &collection_name).await {
        return qerr(QErr::not_found("Collection not found"), start);
    }
    ok(UpdateResult::completed(), start)
}

/// `DELETE /collections/:name/index/:field_name` — payload index deletion.
pub async fn delete_payload_index(
    State(state): State<Arc<AppState>>,
    Path((collection_name, _field_name)): Path<(String, String)>,
) -> Response {
    let start = Instant::now();
    if !collection_exists(&state, &collection_name).await {
        return qerr(QErr::not_found("Collection not found"), start);
    }
    ok(UpdateResult::completed(), start)
}

// ==========================================
// Point handlers
// ==========================================

/// `PUT /collections/:name/points` — upsert points.
pub async fn upsert_points(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<UpsertPointsRequest>,
) -> Response {
    let start = Instant::now();
    match upsert_core(&state, &collection_name, req.points).await {
        Ok(()) => ok(UpdateResult::completed(), start),
        Err(e) => qerr(e, start),
    }
}

/// `GET|POST /collections/:name/points` — retrieve points by id.
pub async fn retrieve_points(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    body: Option<Json<RetrievePointsRequest>>,
) -> Response {
    let start = Instant::now();
    let req = body.map(|Json(r)| r);
    let (ids, with_payload, with_vector) = match req {
        Some(r) => (r.ids, r.with_payload, r.with_vector),
        None => (Vec::new(), None, None),
    };
    if !collection_exists(&state, &collection_name).await {
        return qerr(QErr::not_found("Collection not found"), start);
    }
    let filter_sql = if ids.is_empty() {
        None
    } else {
        Some(ids_filter(&ids))
    };
    let batches = match read_batches(&state, &collection_name, filter_sql.as_deref(), None).await {
        Ok(b) => b,
        Err(e) => return qerr(e, start),
    };
    let wp = with_payload.unwrap_or(WithPayload::Bool(true));
    let wv = with_vector.map(|w| w.enabled()).unwrap_or(false);
    let mut points = Vec::new();
    for b in &batches {
        points.extend(batch_to_retrieved(b, &wp, wv));
    }
    if !ids.is_empty() {
        let wanted: HashSet<String> = ids.iter().map(|i| i.as_string()).collect();
        points.retain(|p| wanted.contains(&p.id.as_string()));
    }
    ok(points, start)
}

/// `GET /collections/:name/points/:id` — single point (404 when missing).
pub async fn get_point(
    State(state): State<Arc<AppState>>,
    Path((collection_name, id)): Path<(String, String)>,
) -> Response {
    let start = Instant::now();
    if !collection_exists(&state, &collection_name).await {
        return qerr(QErr::not_found("Collection not found"), start);
    }
    let filter_sql = format!("_id = {}", sql_string(&id));
    let batches = match read_batches(&state, &collection_name, Some(&filter_sql), None).await {
        Ok(b) => b,
        Err(e) => return qerr(e, start),
    };
    let wp = WithPayload::Bool(true);
    let mut points = Vec::new();
    for b in &batches {
        points.extend(batch_to_retrieved(b, &wp, true));
    }
    match points.into_iter().next() {
        Some(p) => ok(p, start),
        None => qerr(QErr::not_found("Point not found"), start),
    }
}

/// `POST /collections/:name/points/search` — legacy vector search.
pub async fn search_points(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<SearchPointsRequest>,
) -> Response {
    let start = Instant::now();
    match search_core(
        &state,
        &collection_name,
        req.vector,
        req.filter.as_ref(),
        req.limit,
        req.offset,
        req.with_payload.as_ref(),
        req.with_vector.as_ref(),
        req.score_threshold,
    )
    .await
    {
        Ok(points) => ok(points, start),
        Err(e) => qerr(e, start),
    }
}

#[derive(Serialize)]
struct QueryResult {
    points: Vec<ScoredPoint>,
}

/// `POST /collections/:name/points/query` — universal query API.
pub async fn query_points(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<QueryPointsRequest>,
) -> Response {
    let start = Instant::now();
    let result = match &req.query {
        Some(q) => {
            search_core(
                &state,
                &collection_name,
                q.vector().clone(),
                req.filter.as_ref(),
                req.limit,
                req.offset,
                req.with_payload.as_ref(),
                req.with_vector.as_ref(),
                req.score_threshold,
            )
            .await
        }
        None => {
            // Scroll-like: no query vector, return points with score 0.
            scroll_core(
                &state,
                &collection_name,
                req.filter.as_ref(),
                req.limit,
                None,
                req.with_payload.as_ref(),
                req.with_vector.as_ref(),
            )
            .await
            .map(|s| {
                s.points
                    .into_iter()
                    .map(|p| ScoredPoint {
                        id: p.id,
                        version: 0,
                        score: 0.0,
                        payload: p.payload,
                        vector: p.vector,
                    })
                    .collect()
            })
        }
    };
    match result {
        Ok(points) => ok(QueryResult { points }, start),
        Err(e) => qerr(e, start),
    }
}

/// `POST /collections/:name/points/scroll` — paginated point listing.
pub async fn scroll_points(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<ScrollRequest>,
) -> Response {
    let start = Instant::now();
    match scroll_core(
        &state,
        &collection_name,
        req.filter.as_ref(),
        req.limit,
        req.offset,
        req.with_payload.as_ref(),
        req.with_vector.as_ref(),
    )
    .await
    {
        Ok(r) => ok(r, start),
        Err(e) => qerr(e, start),
    }
}

/// `POST /collections/:name/points/count` — count points (optionally filtered).
pub async fn count_points_handler(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<CountRequest>,
) -> Response {
    let start = Instant::now();
    let filter_sql = match req.filter.as_ref() {
        Some(f) => match filter_to_sql(f) {
            Ok(s) => s,
            Err(e) => return qerr(e, start),
        },
        None => None,
    };
    match count_points(&state, &collection_name, filter_sql.as_deref()).await {
        Ok(count) => ok(CountResult { count }, start),
        Err(e) => qerr(e, start),
    }
}

/// `POST /collections/:name/points/recommend` — recommendation by example
/// vectors (average-vector strategy).
pub async fn recommend_points(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<RecommendRequest>,
) -> Response {
    let start = Instant::now();
    if req.positive.is_empty() {
        return qerr(
            QErr::bad("'positive' must contain at least one vector"),
            start,
        );
    }
    let dim = req.positive[0].len();
    let mut query = vec![0.0f32; dim];
    for v in &req.positive {
        if v.len() != dim {
            return qerr(QErr::bad("inconsistent vector dimensions"), start);
        }
        for (i, x) in v.iter().enumerate() {
            query[i] += x;
        }
    }
    for x in query.iter_mut() {
        *x /= req.positive.len() as f32;
    }
    if let Some(neg) = &req.negative {
        if !neg.is_empty() {
            let mut nq = vec![0.0f32; dim];
            for v in neg {
                if v.len() != dim {
                    return qerr(QErr::bad("inconsistent vector dimensions"), start);
                }
                for (i, x) in v.iter().enumerate() {
                    nq[i] += x;
                }
            }
            for x in nq.iter_mut() {
                *x /= neg.len() as f32;
            }
            for i in 0..dim {
                query[i] -= nq[i];
            }
        }
    }
    match search_core(
        &state,
        &collection_name,
        query,
        req.filter.as_ref(),
        req.limit,
        None,
        req.with_payload.as_ref(),
        req.with_vector.as_ref(),
        None,
    )
    .await
    {
        Ok(points) => ok(points, start),
        Err(e) => qerr(e, start),
    }
}

/// `POST /collections/:name/points/discover` — discovery with context pairs.
pub async fn discover_points(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<DiscoverRequest>,
) -> Response {
    let start = Instant::now();
    let limit = req.limit.unwrap_or(10).max(1);
    let want_vector = req
        .with_vector
        .as_ref()
        .map(|w| w.enabled())
        .unwrap_or(false);
    let force_vector = WithVector::Bool(true);
    let mut results = match search_core(
        &state,
        &collection_name,
        req.target.clone(),
        req.filter.as_ref(),
        Some(limit * 4),
        None,
        req.with_payload.as_ref(),
        Some(&force_vector),
        None,
    )
    .await
    {
        Ok(r) => r,
        Err(e) => return qerr(e, start),
    };
    if let Some(ctx) = &req.context {
        if !ctx.is_empty() {
            results.retain(|p| {
                let Some(v) = &p.vector else {
                    return true;
                };
                ctx.iter()
                    .all(|c| l2_sq(v, &c.positive) <= l2_sq(v, &c.negative))
            });
        }
    }
    results.truncate(limit);
    if !want_vector {
        for p in results.iter_mut() {
            p.vector = None;
        }
    }
    ok(results, start)
}

/// `POST /collections/:name/points/batch` — batched write operations.
pub async fn batch_points(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<BatchRequest>,
) -> Response {
    let start = Instant::now();
    let mut results = Vec::with_capacity(req.operations.len());
    for op in req.operations {
        let res = match op {
            BatchOperation::Upsert { upsert } => {
                upsert_core(&state, &collection_name, upsert.points).await
            }
            BatchOperation::Delete { delete } => {
                delete_points_core(&state, &collection_name, &delete).await
            }
            BatchOperation::SetPayload { set_payload } => {
                apply_payload_op(
                    &state,
                    &collection_name,
                    set_payload.points,
                    set_payload.filter,
                    PayloadOp::Set(set_payload.payload),
                )
                .await
            }
            BatchOperation::OverwritePayload { overwrite_payload } => {
                apply_payload_op(
                    &state,
                    &collection_name,
                    overwrite_payload.points,
                    overwrite_payload.filter,
                    PayloadOp::Overwrite(overwrite_payload.payload),
                )
                .await
            }
            BatchOperation::DeletePayload { delete_payload } => {
                apply_payload_op(
                    &state,
                    &collection_name,
                    delete_payload.points,
                    delete_payload.filter,
                    PayloadOp::DeleteKeys(delete_payload.keys),
                )
                .await
            }
            BatchOperation::ClearPayload { clear_payload } => {
                apply_payload_op(
                    &state,
                    &collection_name,
                    clear_payload.points,
                    clear_payload.filter,
                    PayloadOp::Clear,
                )
                .await
            }
            BatchOperation::UpdateVectors { update_vectors } => {
                update_vectors_core(&state, &collection_name, update_vectors.points).await
            }
        };
        match res {
            Ok(()) => results.push(UpdateResult::completed()),
            Err(e) => return qerr(e, start),
        }
    }
    ok(results, start)
}

/// `POST /collections/:name/points/payload` — set (merge) payload.
pub async fn set_payload(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<SetPayloadRequest>,
) -> Response {
    let start = Instant::now();
    match apply_payload_op(
        &state,
        &collection_name,
        req.points,
        req.filter,
        PayloadOp::Set(req.payload),
    )
    .await
    {
        Ok(()) => ok(UpdateResult::completed(), start),
        Err(e) => qerr(e, start),
    }
}

/// `PUT /collections/:name/points/payload` — overwrite payload.
pub async fn overwrite_payload(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<SetPayloadRequest>,
) -> Response {
    let start = Instant::now();
    match apply_payload_op(
        &state,
        &collection_name,
        req.points,
        req.filter,
        PayloadOp::Overwrite(req.payload),
    )
    .await
    {
        Ok(()) => ok(UpdateResult::completed(), start),
        Err(e) => qerr(e, start),
    }
}

/// `POST /collections/:name/points/payload/delete` — delete payload keys.
pub async fn delete_payload(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<DeletePayloadRequest>,
) -> Response {
    let start = Instant::now();
    match apply_payload_op(
        &state,
        &collection_name,
        req.points,
        req.filter,
        PayloadOp::DeleteKeys(req.keys),
    )
    .await
    {
        Ok(()) => ok(UpdateResult::completed(), start),
        Err(e) => qerr(e, start),
    }
}

/// `POST /collections/:name/points/payload/clear` — clear payload.
pub async fn clear_payload(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<ClearPayloadRequest>,
) -> Response {
    let start = Instant::now();
    match apply_payload_op(
        &state,
        &collection_name,
        req.points,
        req.filter,
        PayloadOp::Clear,
    )
    .await
    {
        Ok(()) => ok(UpdateResult::completed(), start),
        Err(e) => qerr(e, start),
    }
}

/// `PUT /collections/:name/points/vectors` — update point vectors.
pub async fn update_vectors(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<UpdateVectorsRequest>,
) -> Response {
    let start = Instant::now();
    match update_vectors_core(&state, &collection_name, req.points).await {
        Ok(()) => ok(UpdateResult::completed(), start),
        Err(e) => qerr(e, start),
    }
}

/// `POST /collections/:name/points/delete` — delete by ids and/or filter.
pub async fn delete_points(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<DeletePointsRequest>,
) -> Response {
    let start = Instant::now();
    match delete_points_core(&state, &collection_name, &req).await {
        Ok(()) => ok(UpdateResult::completed(), start),
        Err(e) => qerr(e, start),
    }
}

// ==========================================
// Alias handlers
// ==========================================

/// `GET /collections/aliases` — list aliases.
pub async fn list_aliases(State(state): State<Arc<AppState>>) -> Response {
    let start = Instant::now();
    let aliases = state.aliases.read().await;
    let mut list: Vec<AliasDescription> = aliases
        .iter()
        .map(|(alias_name, collection_name)| AliasDescription {
            alias_name: alias_name.clone(),
            collection_name: collection_name.clone(),
        })
        .collect();
    list.sort_by_key(|a| a.alias_name.clone());
    ok(AliasesListResult { aliases: list }, start)
}

/// `POST /collections/aliases` — create/delete/rename aliases.
pub async fn update_aliases(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AliasOperationsRequest>,
) -> Response {
    let start = Instant::now();
    let mut aliases = state.aliases.write().await;
    for action in req.actions {
        match action {
            AliasAction::Create { create_alias } => {
                aliases.insert(create_alias.alias_name, create_alias.collection_name);
            }
            AliasAction::Delete { delete_alias } => {
                aliases.remove(&delete_alias.alias_name);
            }
            AliasAction::Rename { rename_alias } => {
                if let Some(target) = aliases.remove(&rename_alias.old_alias_name) {
                    aliases.insert(rename_alias.new_alias_name, target);
                }
            }
        }
    }
    ok(true, start)
}

// ==========================================
// Service handlers
// ==========================================

/// `GET /` — service/version info.
pub async fn service_info() -> Response {
    let start = Instant::now();
    ok(
        ServiceInfo {
            title: "benostreamdb".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            commit: String::new(),
        },
        start,
    )
}

/// `GET /healthz`, `/livez`, `/readyz` — liveness/readiness probe.
pub async fn healthz(State(state): State<Arc<AppState>>) -> Response {
    match state.list_indexes().await {
        Ok(_) => (StatusCode::OK, "healthz check passed").into_response(),
        Err(e) => (
            StatusCode::SERVICE_UNAVAILABLE,
            format!("healthz check failed: {e}"),
        )
            .into_response(),
    }
}

/// `GET /telemetry` — basic telemetry.
pub async fn telemetry(State(state): State<Arc<AppState>>) -> Response {
    let start = Instant::now();
    let collections = state.list_indexes().await.map(|v| v.len()).unwrap_or(0);
    ok(
        TelemetryResult {
            title: "benostreamdb".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            commit: String::new(),
            collections,
            uptime_seconds: process_start().elapsed().as_secs_f64(),
        },
        start,
    )
}

// ==========================================
// Router
// ==========================================

/// Build the Qdrant-compatible router. `/collections/aliases` is registered
/// before `/collections/:collection_name` so the static path wins.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/", get(service_info))
        .route("/healthz", get(healthz))
        .route("/livez", get(healthz))
        .route("/readyz", get(healthz))
        .route("/telemetry", get(telemetry))
        .route("/collections", get(list_collections))
        .route(
            "/collections/aliases",
            get(list_aliases).post(update_aliases),
        )
        .route(
            "/collections/:collection_name",
            get(get_collection)
                .put(create_collection)
                .patch(update_collection)
                .delete(delete_collection),
        )
        .route(
            "/collections/:collection_name/exists",
            get(collection_exists_handler),
        )
        .route(
            "/collections/:collection_name/index",
            put(create_payload_index),
        )
        .route(
            "/collections/:collection_name/index/:field_name",
            axum::routing::delete(delete_payload_index),
        )
        .route(
            "/collections/:collection_name/points",
            put(upsert_points)
                .get(retrieve_points)
                .post(retrieve_points),
        )
        .route(
            "/collections/:collection_name/points/search",
            post(search_points),
        )
        .route(
            "/collections/:collection_name/points/query",
            post(query_points),
        )
        .route(
            "/collections/:collection_name/points/scroll",
            post(scroll_points),
        )
        .route(
            "/collections/:collection_name/points/count",
            post(count_points_handler),
        )
        .route(
            "/collections/:collection_name/points/recommend",
            post(recommend_points),
        )
        .route(
            "/collections/:collection_name/points/discover",
            post(discover_points),
        )
        .route(
            "/collections/:collection_name/points/batch",
            post(batch_points),
        )
        .route(
            "/collections/:collection_name/points/payload",
            post(set_payload).put(overwrite_payload),
        )
        .route(
            "/collections/:collection_name/points/payload/delete",
            post(delete_payload),
        )
        .route(
            "/collections/:collection_name/points/payload/clear",
            post(clear_payload),
        )
        .route(
            "/collections/:collection_name/points/vectors",
            put(update_vectors),
        )
        .route(
            "/collections/:collection_name/points/delete",
            post(delete_points),
        )
        .route("/collections/:collection_name/points/:id", get(get_point))
        .with_state(state)
}
