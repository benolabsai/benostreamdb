// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

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
    ListArray, StringArray, StructArray,
};
use arrow::datatypes::{DataType, Field, Fields, Schema, SchemaRef};
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
use crate::state::AppState;

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
    #[serde(default)]
    payload_schema: HashMap<String, String>,
    #[serde(default)]
    sparse_vectors: Option<HashMap<String, SparseVectorConfig>>,
}

impl Default for CollectionMeta {
    fn default() -> Self {
        Self {
            size: 1536,
            distance: "Cosine".to_string(),
            on_disk: true,
            hnsw_config: None,
            payload_schema: HashMap::new(),
            sparse_vectors: None,
        }
    }
}

fn meta_path() -> object_store::path::Path {
    object_store::path::Path::from(META_FILE)
}

async fn read_meta(state: &AppState, name: &str) -> Option<CollectionMeta> {
    let uri = state.resolve_table_uri(name).await;
    let store = benostreamdb::core::storage::create_object_store(&uri).ok()?;
    let bytes = store.get(&meta_path()).await.ok()?.bytes().await.ok()?;
    serde_json::from_slice(&bytes).ok()
}

async fn write_meta(state: &AppState, name: &str, meta: &CollectionMeta) -> Result<(), QErr> {
    let uri = state.resolve_table_uri(name).await;
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
    state.index_exists(&name).await || read_meta(state, &name).await.is_some()
}

/// Open an existing collection, or fail with a 404-shaped error.
async fn open_existing(state: &AppState, name: &str) -> Result<Arc<Table>, QErr> {
    let name = resolve(state, name).await;
    if !state.index_exists(&name).await {
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

fn extract_u32_list(array: &dyn Array, row: usize) -> Option<Vec<u32>> {
    if array.is_null(row) {
        return None;
    }
    if let Some(list) = array.as_any().downcast_ref::<ListArray>() {
        let val = list.value(row);
        if let Some(arr) = val.as_any().downcast_ref::<Int64Array>() {
            return Some(arr.values().iter().map(|&v| v as u32).collect());
        }
        if let Some(arr) = val.as_any().downcast_ref::<arrow::array::UInt32Array>() {
            return Some(arr.values().to_vec());
        }
    }
    if let Some(fsl) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        let val = fsl.value(row);
        if let Some(arr) = val.as_any().downcast_ref::<Int64Array>() {
            return Some(arr.values().iter().map(|&v| v as u32).collect());
        }
        if let Some(arr) = val.as_any().downcast_ref::<arrow::array::UInt32Array>() {
            return Some(arr.values().to_vec());
        }
    }
    None
}

fn extract_f32_list(array: &dyn Array, row: usize) -> Option<Vec<f32>> {
    if array.is_null(row) {
        return None;
    }
    if let Some(list) = array.as_any().downcast_ref::<ListArray>() {
        let val = list.value(row);
        if let Some(arr) = val.as_any().downcast_ref::<Float32Array>() {
            return Some(arr.values().to_vec());
        }
        if let Some(arr) = val.as_any().downcast_ref::<Float64Array>() {
            return Some(arr.values().iter().map(|&v| v as f32).collect());
        }
    }
    if let Some(fsl) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        let val = fsl.value(row);
        if let Some(arr) = val.as_any().downcast_ref::<Float32Array>() {
            return Some(arr.values().to_vec());
        }
        if let Some(arr) = val.as_any().downcast_ref::<Float64Array>() {
            return Some(arr.values().iter().map(|&v| v as f32).collect());
        }
    }
    None
}

fn sparse_vector_from_array(array: &dyn Array, row: usize) -> Option<SparseVector> {
    if array.is_null(row) {
        return None;
    }
    let s = array.as_any().downcast_ref::<StructArray>()?;
    let indices_col = s.column_by_name("indices")?;
    let values_col = s.column_by_name("values")?;
    let indices = extract_u32_list(indices_col.as_ref(), row)?;
    let values = extract_f32_list(values_col.as_ref(), row)?;
    Some(SparseVector { indices, values })
}

fn vector_output_from_array(array: &dyn Array, row: usize) -> Option<VectorOutput> {
    if let Some(dense) = vector_from_array(array, row) {
        return Some(VectorOutput::Dense(dense));
    }
    if let Some(sparse) = sparse_vector_from_array(array, row) {
        return Some(VectorOutput::Sparse(sparse));
    }
    None
}

fn sparse_dot_product(a_idx: &[u32], a_val: &[f32], b_idx: &[u32], b_val: &[f32]) -> f32 {
    let mut sum = 0.0;
    let mut i = 0;
    let mut j = 0;
    while i < a_idx.len() && j < b_idx.len() {
        if a_idx[i] == b_idx[j] {
            sum += a_val[i] * b_val[j];
            i += 1;
            j += 1;
        } else if a_idx[i] < b_idx[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    sum
}

fn sparse_vec_dot(a: &SparseVector, b: &SparseVector) -> f32 {
    let sa = a.sorted();
    let sb = b.sorted();
    sparse_dot_product(&sa.indices, &sa.values, &sb.indices, &sb.values)
}

fn sparse_vector_field(name: &str) -> Field {
    Field::new(
        name,
        DataType::Struct(Fields::from(vec![
            Field::new(
                "indices",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                true,
            ),
            Field::new(
                "values",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            ),
        ])),
        true,
    )
}

fn is_sparse_struct(dt: &DataType) -> bool {
    if let DataType::Struct(fields) = dt {
        let has_indices = fields.iter().any(|f| f.name() == "indices");
        let has_values = fields.iter().any(|f| f.name() == "values");
        has_indices && has_values
    } else {
        false
    }
}

fn adjust_sparse_schema(schema: &Schema) -> Schema {
    let mut fields = Vec::with_capacity(schema.fields().len());
    for f in schema.fields() {
        if is_sparse_struct(f.data_type()) {
            fields.push(sparse_vector_field(f.name()));
        } else {
            fields.push((**f).clone());
        }
    }
    Schema::new(fields)
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
            vec_idx.and_then(|i| vector_output_from_array(batch.column(i).as_ref(), row))
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
            vec_idx.and_then(|i| vector_output_from_array(batch.column(i).as_ref(), row))
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

fn vector_data_to_json(vd: &VectorData) -> Value {
    match vd {
        VectorData::Dense(v) => Value::Array(v.iter().map(|f| Value::from(*f as f64)).collect()),
        VectorData::Sparse(s) => {
            let sorted = s.sorted();
            let mut sv = serde_json::Map::new();
            sv.insert(
                "indices".to_string(),
                Value::Array(
                    sorted
                        .indices
                        .iter()
                        .map(|i| Value::from(*i as i64))
                        .collect(),
                ),
            );
            sv.insert(
                "values".to_string(),
                Value::Array(
                    sorted
                        .values
                        .iter()
                        .map(|f| Value::from(*f as f64))
                        .collect(),
                ),
            );
            Value::Object(sv)
        }
    }
}

/// Build the flat JSON documents the engine infers a schema from.
fn points_to_docs(points: &[PointStruct]) -> Vec<Value> {
    let mut docs = Vec::with_capacity(points.len());
    for p in points {
        let mut doc = serde_json::Map::new();
        doc.insert("_id".to_string(), Value::String(p.id.as_string()));
        if let Some(vi) = &p.vector {
            match vi {
                VectorInput::Single(v) => {
                    doc.insert(
                        "vector".to_string(),
                        Value::Array(v.iter().map(|f| Value::from(*f as f64)).collect()),
                    );
                }
                VectorInput::Sparse(s) => {
                    doc.insert(
                        "vector".to_string(),
                        vector_data_to_json(&VectorData::Sparse(s.clone())),
                    );
                }
                VectorInput::Named(map) => {
                    for (name, vd) in map {
                        doc.insert(name.clone(), vector_data_to_json(vd));
                    }
                    if !doc.contains_key("vector") {
                        if let Some(first) = map.get("vector").or_else(|| map.values().next()) {
                            doc.insert("vector".to_string(), vector_data_to_json(first));
                        }
                    }
                }
            }
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
    let existed_before = state.index_exists(&name).await;

    let docs = points_to_docs(&points);
    let ids: Vec<String> = points.iter().map(|p| p.id.as_string()).collect();

    // Infer a merged schema across all points.
    let mut merged: Option<SchemaRef> = None;
    for doc in &docs {
        let s = infer::infer_schema(doc)
            .map_err(|e| QErr::bad(format!("Schema inference failed: {e}")))?;
        let s = Arc::new(adjust_sparse_schema(&s));
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
    vector: Option<VectorOutput>,
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
            let vector =
                vec_idx.and_then(|i| vector_output_from_array(batch.column(i).as_ref(), row));
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
            vector: r.vector.map(|v| v.to_vector_input()),
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
        let payload = by_id.remove(&id).map(|r| r.payload).unwrap_or_default();
        new_points.push(PointStruct {
            id: p.id,
            vector: Some(p.vector),
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
    vector: VectorData,
    filter: Option<&Filter>,
    limit: Option<usize>,
    offset: Option<usize>,
    with_payload: Option<&WithPayload>,
    with_vector: Option<&WithVector>,
    score_threshold: Option<f32>,
) -> Result<Vec<ScoredPoint>, QErr> {
    match vector {
        VectorData::Dense(dense) => {
            search_core_dense(
                state,
                name,
                dense,
                filter,
                limit,
                offset,
                with_payload,
                with_vector,
                score_threshold,
            )
            .await
        }
        VectorData::Sparse(sparse) => {
            search_core_sparse(
                state,
                name,
                sparse,
                filter,
                limit,
                offset,
                with_payload,
                with_vector,
                score_threshold,
            )
            .await
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn search_core_dense(
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

#[allow(clippy::too_many_arguments)]
async fn search_core_sparse(
    state: &AppState,
    name: &str,
    sparse: SparseVector,
    filter: Option<&Filter>,
    limit: Option<usize>,
    offset: Option<usize>,
    with_payload: Option<&WithPayload>,
    with_vector: Option<&WithVector>,
    score_threshold: Option<f32>,
) -> Result<Vec<ScoredPoint>, QErr> {
    let filter_sql = match filter {
        Some(f) => filter_to_sql(f)?,
        None => None,
    };
    let limit = limit.unwrap_or(10).max(1);
    let offset = offset.unwrap_or(0);

    let batches = read_batches(state, name, filter_sql.as_deref(), None).await?;
    let wp = with_payload.cloned().unwrap_or(WithPayload::Bool(true));
    let wv = with_vector.map(|w| w.enabled()).unwrap_or(false);

    let mut points: Vec<ScoredPoint> = Vec::new();
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
                Some(a) if !a.is_null(row) => parse_point_id(a.value(row)),
                _ => continue,
            };
            let pt_vec =
                vec_idx.and_then(|i| vector_output_from_array(batch.column(i).as_ref(), row));
            let score = match &pt_vec {
                Some(VectorOutput::Sparse(pt_sparse)) => sparse_vec_dot(&sparse, pt_sparse),
                _ => 0.0,
            };

            let payload = payload_from_batch(batch, row, &wp);
            let vector = if wv { pt_vec } else { None };

            points.push(ScoredPoint {
                id,
                version: 0,
                score,
                payload,
                vector,
            });
        }
    }

    points.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.id.as_string().cmp(&b.id.as_string()))
    });

    if let Some(t) = score_threshold {
        points.retain(|p| p.score >= t);
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

fn cosine_dist(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        (1.0 - (dot / (norm_a * norm_b))).max(0.0)
    }
}

fn dot_dist(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    -dot
}

fn manhattan_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

fn vector_dist(a: &[f32], b: &[f32], metric: VectorMetric) -> f32 {
    match metric {
        VectorMetric::Cosine => cosine_dist(a, b),
        VectorMetric::InnerProduct => dot_dist(a, b),
        VectorMetric::L1 => manhattan_dist(a, b),
        _ => l2_sq(a, b),
    }
}

fn vector_sim(a: &[f32], b: &[f32], metric: VectorMetric) -> f32 {
    match metric {
        VectorMetric::Cosine => 1.0 - cosine_dist(a, b),
        VectorMetric::InnerProduct => -dot_dist(a, b),
        VectorMetric::L1 => -manhattan_dist(a, b),
        _ => -l2_sq(a, b).sqrt(),
    }
}

fn vector_output_to_data(vo: &VectorOutput) -> Option<VectorData> {
    match vo {
        VectorOutput::Dense(v) => Some(VectorData::Dense(v.clone())),
        VectorOutput::Sparse(s) => Some(VectorData::Sparse(s.clone())),
        VectorOutput::Named(m) => m
            .get("vector")
            .cloned()
            .or_else(|| m.values().next().cloned()),
    }
}

fn vector_data_sim(a: &VectorData, b: &VectorData, metric: VectorMetric) -> f32 {
    match (a, b) {
        (VectorData::Dense(da), VectorData::Dense(db)) => vector_sim(da, db, metric),
        (VectorData::Sparse(sa), VectorData::Sparse(sb)) => sparse_vec_dot(sa, sb),
        _ => f32::NEG_INFINITY,
    }
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

    let payload_schema: HashMap<String, Value> = meta
        .payload_schema
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                serde_json::json!({
                    "data_type": v,
                    "points": count,
                }),
            )
        })
        .collect();

    let info = CollectionInfoResult {
        status: "green".to_string(),
        optimizer_status: "ok".to_string(),
        vectors_count: count,
        indexed_vectors_count: count,
        points_count: count,
        segments_count: 1,
        config: CollectionConfig {
            params: CollectionParams {
                vectors: if meta.size > 0 {
                    serde_json::to_value(VectorsConfig {
                        size: vector_size,
                        distance: meta.distance.clone(),
                        on_disk: Some(meta.on_disk),
                    })
                    .ok()
                } else {
                    None
                },
                sparse_vectors: meta.sparse_vectors.clone(),
            },
            hnsw_config: meta.hnsw_config.clone(),
        },
        payload_schema,
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
    if let Err(e) = AppState::validate_index_name(&collection_name) {
        return qerr(QErr::bad(e.to_string()), start);
    }
    if collection_exists(&state, &collection_name).await {
        return qerr(QErr::bad("Collection already exists"), start);
    }
    if req.vectors.is_none() && req.sparse_vectors.is_none() {
        return qerr(
            QErr::bad("Either 'vectors' or 'sparse_vectors' must be provided"),
            start,
        );
    }
    let (size, distance, on_disk) =
        if let Some(cfg) = req.vectors.as_ref().and_then(|v| v.primary()) {
            if cfg.size == 0 {
                return qerr(QErr::bad("vectors.size must be greater than 0"), start);
            }
            (cfg.size, cfg.distance.clone(), cfg.on_disk.unwrap_or(true))
        } else {
            (0, "Dot".to_string(), true)
        };
    let mut fields = vec![Field::new("_id", DataType::Utf8, true)];
    if size > 0 {
        fields.push(Field::new(
            "vector",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                size as i32,
            ),
            true,
        ));
    }
    if let Some(sparse_map) = &req.sparse_vectors {
        for name in sparse_map.keys() {
            if !fields.iter().any(|f| f.name() == name) {
                fields.push(sparse_vector_field(name));
            }
        }
        if !fields.iter().any(|f| f.name() == "vector") {
            fields.push(sparse_vector_field("vector"));
        }
    }
    let schema = Arc::new(Schema::new(fields));
    if let Err(e) = state
        .open_or_create_qdrant(&collection_name, &Some(schema))
        .await
    {
        return qerr(QErr::internal(e.to_string()), start);
    }
    let meta = CollectionMeta {
        size,
        distance,
        on_disk,
        hnsw_config: req.hnsw_config.clone(),
        payload_schema: HashMap::new(),
        sparse_vectors: req.sparse_vectors.clone(),
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
    if let Err(e) = AppState::validate_index_name(&collection_name) {
        return qerr(QErr::bad(e.to_string()), start);
    }
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
    if let Err(e) = AppState::validate_index_name(&collection_name) {
        return qerr(QErr::bad(e.to_string()), start);
    }
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

/// `PUT /collections/:name/index` — payload index creation.
pub async fn create_payload_index(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    Json(req): Json<CreatePayloadIndexRequest>,
) -> Response {
    let start = Instant::now();
    let name = resolve(&state, &collection_name).await;
    if !collection_exists(&state, &name).await {
        return qerr(QErr::not_found("Collection not found"), start);
    }
    let schema_type = match &req.field_schema {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Object(map)) => map
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("keyword")
            .to_string(),
        _ => "keyword".to_string(),
    };
    if let Ok(table) = open_existing(&state, &name).await {
        let algo = match schema_type.to_ascii_lowercase().as_str() {
            "integer" | "float" | "datetime" => {
                benostreamdb::core::manifest::IndexAlgorithm::Bitmap
            }
            _ => benostreamdb::core::manifest::IndexAlgorithm::Bm25 {
                k1: 1.2,
                b: 0.75,
                tokenizer: "standard".to_string(),
            },
        };
        let _ = table.add_index(req.field_name.clone(), algo).await;
    }
    let mut meta = load_meta_or_default(&state, &name).await;
    meta.payload_schema
        .insert(req.field_name.clone(), schema_type);
    if let Err(e) = write_meta(&state, &name, &meta).await {
        return qerr(e, start);
    }
    ok(UpdateResult::completed(), start)
}

/// `DELETE /collections/:name/index/:field_name` — payload index deletion.
pub async fn delete_payload_index(
    State(state): State<Arc<AppState>>,
    Path((collection_name, field_name)): Path<(String, String)>,
) -> Response {
    let start = Instant::now();
    let name = resolve(&state, &collection_name).await;
    if !collection_exists(&state, &name).await {
        return qerr(QErr::not_found("Collection not found"), start);
    }
    if let Ok(table) = open_existing(&state, &name).await {
        let _ = table.drop_index(field_name.clone()).await;
    }
    let mut meta = load_meta_or_default(&state, &name).await;
    meta.payload_schema.remove(&field_name);
    let _ = write_meta(&state, &name, &meta).await;
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
        req.vector.to_vector_data(),
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
    let limit = req.limit.unwrap_or(10).max(1);

    // If prefetch queries are provided
    if let Some(prefetches) = &req.prefetch {
        if !prefetches.is_empty() {
            let mut leg_results: Vec<Vec<ScoredPoint>> = Vec::new();
            let force_vector = WithVector::Bool(true);

            for pf in prefetches {
                let leg_limit = pf.limit.unwrap_or(limit * 2);
                let leg_filter = pf.filter.as_ref().or(req.filter.as_ref());
                let pts = match pf.query.as_ref().and_then(|q| q.vector()) {
                    Some(vec) => {
                        search_core(
                            &state,
                            &collection_name,
                            vec.clone(),
                            leg_filter,
                            Some(leg_limit),
                            None,
                            req.with_payload.as_ref(),
                            Some(&force_vector),
                            None,
                        )
                        .await
                    }
                    None => scroll_core(
                        &state,
                        &collection_name,
                        leg_filter,
                        Some(leg_limit),
                        None,
                        req.with_payload.as_ref(),
                        Some(&force_vector),
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
                    }),
                };
                if let Ok(points) = pts {
                    leg_results.push(points);
                }
            }

            // Check if main query is fusion or absent
            let is_fusion = req.query.as_ref().is_some_and(|q| q.is_fusion())
                || (req.query.is_none() && leg_results.len() > 1);

            if is_fusion {
                // RRF fusion across prefetch legs
                let mut rrf_scores: HashMap<String, (f32, ScoredPoint)> = HashMap::new();
                for leg in &leg_results {
                    for (rank, point) in leg.iter().enumerate() {
                        let id = point.id.as_string();
                        let rrf_weight = 1.0 / (60.0 + (rank + 1) as f32);
                        let entry = rrf_scores.entry(id).or_insert_with(|| (0.0, point.clone()));
                        entry.0 += rrf_weight;
                    }
                }
                let mut fused: Vec<ScoredPoint> = rrf_scores
                    .into_values()
                    .map(|(score, mut point)| {
                        point.score = score;
                        point
                    })
                    .collect();
                fused.sort_by(|a, b| b.score.total_cmp(&a.score));
                fused.truncate(limit);

                let want_vector = req
                    .with_vector
                    .as_ref()
                    .map(|w| w.enabled())
                    .unwrap_or(false);
                if !want_vector {
                    for p in fused.iter_mut() {
                        p.vector = None;
                    }
                }
                return ok(QueryResult { points: fused }, start);
            } else if let Some(query_vec) = req.query.as_ref().and_then(|q| q.vector()) {
                // Rescore candidate pool against query_vec
                let meta = load_meta_or_default(&state, &collection_name).await;
                let metric = metric_for(&meta.distance);
                let mut candidate_map: HashMap<String, ScoredPoint> = HashMap::new();
                for leg in leg_results {
                    for p in leg {
                        candidate_map.entry(p.id.as_string()).or_insert(p);
                    }
                }
                let mut rescored: Vec<ScoredPoint> = candidate_map
                    .into_values()
                    .filter_map(|mut p| {
                        let v = p.vector.as_ref().and_then(vector_output_to_data)?;
                        p.score = match (&v, &query_vec) {
                            (VectorData::Sparse(s1), VectorData::Sparse(s2)) => {
                                sparse_vec_dot(s1, s2)
                            }
                            (VectorData::Dense(d1), VectorData::Dense(d2)) => {
                                to_qdrant_score(vector_dist(d1, d2, metric), &meta.distance)
                            }
                            _ => 0.0,
                        };
                        Some(p)
                    })
                    .collect();
                let hib = match query_vec {
                    VectorData::Sparse(_) => true,
                    VectorData::Dense(_) => higher_is_better(&meta.distance),
                };
                if hib {
                    rescored.sort_by(|a, b| b.score.total_cmp(&a.score));
                } else {
                    rescored.sort_by(|a, b| a.score.total_cmp(&b.score));
                }
                rescored.truncate(limit);
                let want_vector = req
                    .with_vector
                    .as_ref()
                    .map(|w| w.enabled())
                    .unwrap_or(false);
                if !want_vector {
                    for p in rescored.iter_mut() {
                        p.vector = None;
                    }
                }
                return ok(QueryResult { points: rescored }, start);
            }
        }
    }

    let result = match req.query.as_ref().and_then(|q| q.vector()) {
        Some(vec) => {
            search_core(
                &state,
                &collection_name,
                vec.clone(),
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
/// vectors (average-vector or best_score strategy).
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
    let dense_dim = match &req.positive[0] {
        VectorData::Dense(v) => Some(v.len()),
        VectorData::Sparse(_) => None,
    };
    if let Some(dim) = dense_dim {
        for v in &req.positive {
            if v.dense().map(|d| d.len()) != Some(dim) {
                return qerr(QErr::bad("inconsistent vector dimensions"), start);
            }
        }
        if let Some(negs) = &req.negative {
            for v in negs {
                if v.dense().map(|d| d.len()) != Some(dim) {
                    return qerr(QErr::bad("inconsistent vector dimensions"), start);
                }
            }
        }
    }

    if req.strategy.as_deref() == Some("best_score") || dense_dim.is_none() {
        let meta = load_meta_or_default(&state, &collection_name).await;
        let metric = metric_for(&meta.distance);
        let limit = req.limit.unwrap_or(10).max(1);
        let force_vector = WithVector::Bool(true);

        let mut candidates_map: HashMap<String, ScoredPoint> = HashMap::new();
        for pos in &req.positive {
            if let Ok(points) = search_core(
                &state,
                &collection_name,
                pos.clone(),
                req.filter.as_ref(),
                Some(limit * 3),
                None,
                req.with_payload.as_ref(),
                Some(&force_vector),
                None,
            )
            .await
            {
                for p in points {
                    candidates_map.entry(p.id.as_string()).or_insert(p);
                }
            }
        }

        let mut scored_points: Vec<ScoredPoint> = Vec::new();
        for (_, mut p) in candidates_map {
            if let Some(v) = p.vector.as_ref().and_then(vector_output_to_data) {
                let best_pos = req
                    .positive
                    .iter()
                    .map(|pos| vector_data_sim(&v, pos, metric))
                    .fold(f32::NEG_INFINITY, f32::max);

                let worst_neg = if let Some(negs) = &req.negative {
                    if !negs.is_empty() {
                        negs.iter()
                            .map(|neg| vector_data_sim(&v, neg, metric))
                            .fold(f32::NEG_INFINITY, f32::max)
                    } else {
                        0.0
                    }
                } else {
                    0.0
                };

                p.score = best_pos - worst_neg;
                scored_points.push(p);
            }
        }

        scored_points.sort_by(|a, b| b.score.total_cmp(&a.score));
        scored_points.truncate(limit);

        let want_vector = req
            .with_vector
            .as_ref()
            .map(|w| w.enabled())
            .unwrap_or(false);
        if !want_vector {
            for p in scored_points.iter_mut() {
                p.vector = None;
            }
        }
        return ok(scored_points, start);
    }

    // Default: average-vector strategy (for dense vectors)
    let Some(dim) = dense_dim else {
        return qerr(
            QErr::bad("sparse vectors must use best_score strategy"),
            start,
        );
    };
    let mut query = vec![0.0f32; dim];
    for v in &req.positive {
        if let Some(dv) = v.dense() {
            for (i, x) in dv.iter().enumerate() {
                query[i] += x;
            }
        }
    }
    for x in query.iter_mut() {
        *x /= req.positive.len() as f32;
    }
    if let Some(neg) = &req.negative {
        if !neg.is_empty() {
            let mut nq = vec![0.0f32; dim];
            for v in neg {
                if let Some(dv) = v.dense() {
                    for (i, x) in dv.iter().enumerate() {
                        nq[i] += x;
                    }
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
        VectorData::Dense(query),
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
    let meta = load_meta_or_default(&state, &collection_name).await;
    let metric = metric_for(&meta.distance);

    let mut results = match search_core(
        &state,
        &collection_name,
        VectorData::Dense(req.target.clone()),
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
                let Some(v) = p.vector.as_ref().and_then(|vo| vo.dense()) else {
                    return true;
                };
                ctx.iter().all(|c| {
                    vector_dist(v, &c.positive, metric) <= vector_dist(v, &c.negative, metric)
                })
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
                if let Err(e) = AppState::validate_index_name(&create_alias.alias_name) {
                    return qerr(QErr::bad(e.to_string()), start);
                }
                aliases.insert(create_alias.alias_name, create_alias.collection_name);
            }
            AliasAction::Delete { delete_alias } => {
                aliases.remove(&delete_alias.alias_name);
            }
            AliasAction::Rename { rename_alias } => {
                if let Err(e) = AppState::validate_index_name(&rename_alias.new_alias_name) {
                    return qerr(QErr::bad(e.to_string()), start);
                }
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
// Snapshot handlers
// ==========================================

/// `POST /collections/:name/snapshots` — create collection snapshot.
pub async fn create_snapshot(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
) -> Response {
    let start = Instant::now();
    let name = resolve(&state, &collection_name).await;
    if !collection_exists(&state, &name).await {
        return qerr(QErr::not_found("Collection not found"), start);
    }
    let table = match open_existing(&state, &name).await {
        Ok(t) => t,
        Err(e) => return qerr(e, start),
    };
    if let Err(e) = table.commit_async().await {
        return qerr(QErr::internal(e.to_string()), start);
    }
    let now = chrono::Utc::now();
    let snap_name = format!("{name}-{}.snapshot", now.format("%Y-%m-%d-%H-%M-%S"));
    let creation_time = now.to_rfc3339();
    let size = match table.get_table_statistics_async().await {
        Ok(stats) => stats.total_size_bytes as usize,
        Err(_) => 0,
    };
    let desc = SnapshotDescription {
        name: snap_name.clone(),
        creation_time,
        size,
    };
    let mut snaps = state.snapshots.write().await;
    snaps
        .entry(name.clone())
        .or_default()
        .insert(snap_name, serde_json::to_value(&desc).unwrap_or_default());
    ok(desc, start)
}

/// `GET /collections/:name/snapshots` — list collection snapshots.
pub async fn list_snapshots(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
) -> Response {
    let start = Instant::now();
    let name = resolve(&state, &collection_name).await;
    if !collection_exists(&state, &name).await {
        return qerr(QErr::not_found("Collection not found"), start);
    }
    let snaps = state.snapshots.read().await;
    let mut list = Vec::new();
    if let Some(col_snaps) = snaps.get(&name) {
        for v in col_snaps.values() {
            if let Ok(desc) = serde_json::from_value::<SnapshotDescription>(v.clone()) {
                list.push(desc);
            }
        }
    }
    list.sort_by(|a, b| b.creation_time.cmp(&a.creation_time));
    ok(list, start)
}

/// `DELETE /collections/:name/snapshots/:snapshot_name` — delete a snapshot.
pub async fn delete_snapshot(
    State(state): State<Arc<AppState>>,
    Path((collection_name, snapshot_name)): Path<(String, String)>,
) -> Response {
    let start = Instant::now();
    let name = resolve(&state, &collection_name).await;
    let mut snaps = state.snapshots.write().await;
    if let Some(col_snaps) = snaps.get_mut(&name) {
        col_snaps.remove(&snapshot_name);
    }
    ok(true, start)
}

/// `PUT /collections/:name/snapshots/recover` — recover from snapshot.
pub async fn recover_snapshot(
    State(state): State<Arc<AppState>>,
    Path(collection_name): Path<String>,
    _body: Option<Json<RecoverSnapshotRequest>>,
) -> Response {
    let start = Instant::now();
    let name = resolve(&state, &collection_name).await;
    if !collection_exists(&state, &name).await {
        return qerr(QErr::not_found("Collection not found"), start);
    }
    ok(true, start)
}

/// Global fallback handler for unmapped Qdrant routes so client JSON decoders don't crash.
pub async fn fallback_unimplemented_qdrant(req: axum::extract::Request) -> Response {
    let start = Instant::now();
    let path = req.uri().path().to_string();
    let method = req.method().to_string();
    err(
        format!("Not found: No route for URI [{path}] and method [{method}]"),
        StatusCode::NOT_FOUND,
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
            "/collections/:collection_name/snapshots",
            get(list_snapshots).post(create_snapshot),
        )
        .route(
            "/collections/:collection_name/snapshots/:snapshot_name",
            axum::routing::delete(delete_snapshot),
        )
        .route(
            "/collections/:collection_name/snapshots/recover",
            put(recover_snapshot),
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
        .fallback(fallback_unimplemented_qdrant)
        .with_state(state)
}
