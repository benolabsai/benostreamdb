// Copyright (c) 2026 Richard Albright. All rights reserved.

//! ES-compatible search endpoint: `POST /{index}/_search`.
//!
//! Supports `match` (BM25), `knn` (HNSW), hybrid (BM25 + HNSW fused with RRF),
//! `match_all`, and a top-level `filter` (term/range/exists/bool) translated
//! to a SQL predicate evaluated with DataFusion.

use std::cmp::Ordering;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{
    Array, BooleanArray, Date32Array, Date64Array, FixedSizeListArray, Float32Array, Float64Array,
    Int8Array, Int16Array, Int32Array, Int64Array, LargeStringArray, ListArray, RecordBatch,
    StringArray, StructArray, TimestampMicrosecondArray, UInt8Array, UInt16Array, UInt32Array,
    UInt64Array,
};
use arrow::datatypes::DataType;
use axum::extract::{Path, State};
use axum::response::Response;
use axum::Json;
use chrono::{DateTime, NaiveDate, SecondsFormat};
use hyperstreamdb::core::index::VectorValue;
use hyperstreamdb::core::planner::{FilterExpr, QueryPlanner};
use hyperstreamdb::core::search::{HybridSearchCoordinator, KeywordSearchParams};
use hyperstreamdb::{HyperstreamError, VectorSearchParams};
use serde_json::{Map, Value};

use crate::es_types::{SearchHit, SearchHits, SearchResponse, TotalHits};
use crate::handlers::docs::ID_COLUMN;
use crate::state::{table_exists, AppState};

use super::es_response;

pub async fn search(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    es_response(search_core(&state, &index, &body).await)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScoreKind {
    /// Trailing float column is the final relevance score (higher is better).
    Relevance,
    /// Trailing float column is a distance (lower is better).
    Distance,
    /// No score column (match_all); every hit scores 1.0.
    None,
}

struct Hit {
    id: String,
    source: Value,
    score: f32,
}

#[derive(Debug)]
struct SearchRequest {
    keyword: Option<KeywordSearchParams>,
    vector: Option<VectorSearchParams>,
    /// SQL `WHERE` clause translated from the top-level ES `filter`.
    filter: Option<String>,
    size: usize,
    from: usize,
}

impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            keyword: None,
            vector: None,
            filter: None,
            size: 10,
            from: 0,
        }
    }
}

fn bad_request(reason: impl Into<String>) -> HyperstreamError {
    HyperstreamError::SchemaIncompatible {
        reason: reason.into(),
    }
}

fn json_type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn parse_request(body: &Value) -> Result<SearchRequest, HyperstreamError> {
    let obj = body
        .as_object()
        .ok_or_else(|| bad_request("request body must be a JSON object"))?;

    let mut req = SearchRequest::default();

    if let Some(size) = obj.get("size").and_then(Value::as_u64) {
        req.size = size as usize;
    }
    if let Some(from) = obj.get("from").and_then(Value::as_u64) {
        req.from = from as usize;
    }
    if let Some(filter) = obj.get("filter") {
        req.filter = Some(clause_to_sql(filter, "filter")?);
    }

    if let Some(query) = obj.get("query") {
        match query {
            Value::Object(m) => {
                for (key, spec) in m {
                    match key.as_str() {
                        "match_all" => {
                            if !spec.is_null() && !spec.is_object() {
                                return Err(bad_request(
                                    "match_all: expected an object or null",
                                ));
                            }
                        }
                        "match" => req.keyword = Some(parse_match(spec)?),
                        "knn" => req.vector = Some(parse_knn(spec)?),
                        other => {
                            return Err(bad_request(format!(
                                "unsupported query clause '{other}' (supported: match, match_all, knn)"
                            )));
                        }
                    }
                }
            }
            other => {
                return Err(bad_request(format!(
                    "query: expected an object, got {}",
                    json_type_name(other)
                )));
            }
        }
    }
    // ES 8-style top-level `knn`; a `query` object containing `knn` wins.
    if req.vector.is_none() {
        if let Some(knn) = obj.get("knn") {
            req.vector = Some(parse_knn(knn)?);
        }
    }

    Ok(req)
}

fn parse_match(spec: &Value) -> Result<KeywordSearchParams, HyperstreamError> {
    let m = spec.as_object().ok_or_else(|| {
        bad_request("match: expected {\"field\": \"text\"} or {\"field\": {\"query\": \"text\"}}")
    })?;
    // BTreeMap iteration is alphabetical, so this deterministically takes the
    // first field when multiple are given.
    let (field, v) = m.iter().next().ok_or_else(|| {
        bad_request("match: expected {\"field\": \"text\"} or {\"field\": {\"query\": \"text\"}}")
    })?;
    let field = valid_field(field)?;
    let text = match v {
        Value::String(s) => s.clone(),
        Value::Object(o) => o
            .get("query")
            .or_else(|| o.get("value"))
            .and_then(Value::as_str)
            .map(str::to_string)
            .ok_or_else(|| {
                bad_request("match: expected {\"field\": \"text\"} or {\"field\": {\"query\": \"text\"}}")
            })?,
        other => {
            return Err(bad_request(format!(
                "match: expected a string or object for field '{field}', got {}",
                json_type_name(other)
            )));
        }
    };
    Ok(KeywordSearchParams::new(field, text))
}

fn parse_knn(spec: &Value) -> Result<VectorSearchParams, HyperstreamError> {
    let m = spec.as_object().ok_or_else(|| bad_request("knn: expected an object"))?;
    let field = m
        .get("field")
        .and_then(Value::as_str)
        .ok_or_else(|| bad_request("knn: 'field' must be a string"))?
        .to_string();
    let field = valid_field(&field)?;
    let vec_src = m.get("vector").or_else(|| m.get("query_vector"));
    let values = vec_src.and_then(as_f32_list).ok_or_else(|| {
        bad_request("knn: 'vector' (or 'query_vector') must be an array of numbers")
    })?;
    let k = m.get("k").and_then(Value::as_u64).unwrap_or(10) as usize;
    if k == 0 {
        return Err(bad_request("knn: 'k' must be greater than 0"));
    }
    Ok(VectorSearchParams::new(&field, VectorValue::Float32(values), k))
}

fn as_f32_list(v: &Value) -> Option<Vec<f32>> {
    let arr = v.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for x in arr {
        out.push(x.as_f64()? as f32);
    }
    Some(out)
}

/// Field names are inlined into SQL predicates, so validate them strictly.
fn valid_field(field: &str) -> Result<String, HyperstreamError> {
    let valid = !field.is_empty()
        && field.chars().enumerate().all(|(i, c)| {
            if i == 0 {
                c == '_' || c.is_ascii_alphabetic()
            } else {
                c.is_ascii_alphanumeric() || c == '_'
            }
        });
    if valid {
        Ok(field.to_string())
    } else {
        Err(bad_request(format!("invalid field name '{field}'")))
    }
}

fn clause_to_sql(clause: &Value, ctx: &str) -> Result<String, HyperstreamError> {
    match clause {
        Value::Array(items) => {
            if items.is_empty() {
                return Ok("true".to_string());
            }
            let parts = items
                .iter()
                .map(|c| clause_to_sql(c, ctx))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(parts.join(" AND "))
        }
        Value::Object(m) if m.len() == 1 => {
            let (key, value) = m.iter().next().unwrap();
            match key.as_str() {
                "term" => term_to_sql(value, ctx),
                "range" => range_to_sql(value, ctx),
                "exists" => exists_to_sql(value, ctx),
                "bool" => bool_to_sql(value, ctx),
                other => Err(bad_request(format!(
                    "unsupported {ctx} clause '{other}' (supported: term, range, exists, bool)"
                ))),
            }
        }
        other => Err(bad_request(format!(
            "{ctx}: expected a filter clause object, got {}",
            json_type_name(other)
        ))),
    }
}

fn term_to_sql(value: &Value, _ctx: &str) -> Result<String, HyperstreamError> {
    let m = value
        .as_object()
        .ok_or_else(|| bad_request("term: expected {\"field\": value}"))?;
    if m.len() != 1 {
        return Err(bad_request("term: expected exactly one field"));
    }
    let (field, v) = m.iter().next().unwrap();
    let field = valid_field(field)?;
    // Accept both bare `{"field": value}` and wrapped `{"field": {"value": ...}}`.
    let leaf = if v.as_object().map_or(false, |o| o.len() == 1 && o.contains_key("value")) {
        v.get("value").unwrap()
    } else {
        v
    };
    let lit = sql_literal(leaf)?;
    Ok(format!("{field} = {lit}"))
}

fn range_to_sql(value: &Value, _ctx: &str) -> Result<String, HyperstreamError> {
    let m = value
        .as_object()
        .ok_or_else(|| bad_request("range: expected {\"field\": {\"gte\": ...}}"))?;
    if m.len() != 1 {
        return Err(bad_request("range: expected exactly one field"));
    }
    let (field, bounds) = m.iter().next().unwrap();
    let field = valid_field(field)?;
    let bounds = bounds
        .as_object()
        .ok_or_else(|| bad_request("range: expected a bounds object"))?;
    const OPS: [(&str, &str); 8] = [
        ("gte", ">="),
        (">=", ">="),
        ("gt", ">"),
        (">", ">"),
        ("lte", "<="),
        ("<=", "<="),
        ("lt", "<"),
        ("<", "<"),
    ];
    let mut parts = Vec::new();
    for (key, op) in OPS {
        if let Some(v) = bounds.get(key) {
            let lit = sql_literal(v)?;
            parts.push(format!("{field} {op} {lit}"));
        }
    }
    if parts.is_empty() {
        return Err(bad_request(
            "range: no recognized bounds (gte, gt, lte, lt, >=, >, <=, <)",
        ));
    }
    Ok(parts.join(" AND "))
}

fn exists_to_sql(value: &Value, _ctx: &str) -> Result<String, HyperstreamError> {
    let m = value
        .as_object()
        .ok_or_else(|| bad_request("exists: expected {\"field\": {}}"))?;
    if m.len() != 1 {
        return Err(bad_request("exists: expected exactly one field"));
    }
    let (field, _) = m.iter().next().unwrap();
    let field = valid_field(field)?;
    Ok(format!("{field} IS NOT NULL"))
}

fn bool_to_sql(value: &Value, ctx: &str) -> Result<String, HyperstreamError> {
    let m = value.as_object().ok_or_else(|| bad_request("bool: expected an object"))?;
    let mut parts = Vec::new();
    for key in ["must", "filter"] {
        if let Some(arr) = m.get(key).and_then(Value::as_array) {
            for clause in arr {
                parts.push(clause_to_sql(clause, ctx)?);
            }
        }
    }
    if parts.is_empty() {
        return Ok("true".to_string());
    }
    Ok(parts.join(" AND "))
}

fn sql_literal(v: &Value) -> Result<String, HyperstreamError> {
    match v {
        Value::String(s) => Ok(format!("'{}'", s.replace('\'', "''"))),
        Value::Number(n) => Ok(n.to_string()),
        Value::Bool(b) => Ok(b.to_string()),
        other => Err(bad_request(format!(
            "unsupported filter value: {}",
            json_type_name(other)
        ))),
    }
}

/// Core search dispatch. Returns the full ES-shaped response so tests can
/// assert on it without going through the axum layer.
pub async fn search_core(
    state: &AppState,
    index: &str,
    body: &Value,
) -> Result<SearchResponse, HyperstreamError> {
    let start = Instant::now();

    if !table_exists(&state.index_uri(index)).await {
        return Err(HyperstreamError::TableNotFound {
            namespace: String::new(),
            name: index.to_string(),
        });
    }

    let req = parse_request(body)?;
    let table = state.open_or_create(index, &None).await?;

    let (batches, kind, knn_k) = match (&req.keyword, &req.vector) {
        (Some(kp), Some(vp)) => {
            let scored = HybridSearchCoordinator::new()
                .execute_hybrid(&table, None, Some(vp.clone()), Some(kp.clone()), 1000, None)
                .await
                .map_err(|e| HyperstreamError::internal(e.to_string()))?;
            let batches = table
                .fetch_results_by_id(scored, None)
                .await
                .map_err(|e| HyperstreamError::internal(e.to_string()))?;
            (batches, ScoreKind::Relevance, None)
        }
        (Some(kp), None) => {
            let scored = table
                .execute_keyword_search_as_scored(kp.clone())
                .await
                .map_err(|e| HyperstreamError::internal(e.to_string()))?;
            let batches = table
                .fetch_results_by_id(scored, None)
                .await
                .map_err(|e| HyperstreamError::internal(e.to_string()))?;
            (batches, ScoreKind::Relevance, None)
        }
        (None, Some(vp)) => {
            if req.filter.is_some() {
                // Core pre-filters inside the scan and appends a distance
                // column. Note: the core's smart hybrid trigger may rewrite
                // this scan into a hybrid path when the filter column has its
                // own BM25 index; the result shape (rows + trailing distance
                // column) is preserved, which is acceptable for v1.
                let batches = table
                    .read_async(req.filter.as_deref(), Some(vp.clone()), None)
                    .await
                    .map_err(|e| HyperstreamError::internal(e.to_string()))?;
                (batches, ScoreKind::Distance, Some(vp.k))
            } else {
                let scored = table
                    .execute_vector_search_as_scored(vp.clone())
                    .await
                    .map_err(|e| HyperstreamError::internal(e.to_string()))?;
                let batches = table
                    .fetch_results_by_id(scored, None)
                    .await
                    .map_err(|e| HyperstreamError::internal(e.to_string()))?;
                (batches, ScoreKind::Distance, Some(vp.k))
            }
        }
        (None, None) => {
            let batches = table
                .read_async(req.filter.as_deref(), None, None)
                .await
                .map_err(|e| HyperstreamError::internal(e.to_string()))?;
            (batches, ScoreKind::None, None)
        }
    };

    // Post-filter scanned batches unless the core already applied the filter
    // (the knn+filter `read_async` path pre-filters inside the scan).
    let batches = if req.filter.is_some() && kind != ScoreKind::Distance {
        let sql = req
            .filter
            .as_ref()
            .expect("filter present when kind != Distance");
        let expr = FilterExpr::parse_sql(sql, table.arrow_schema())
            .await
            .map_err(|e| HyperstreamError::SchemaIncompatible {
                reason: format!("invalid filter: {e}"),
            })?;
        let planner = QueryPlanner::new();
        batches
            .into_iter()
            .map(|b| planner.filter_expr(&b, &expr))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| HyperstreamError::SchemaIncompatible {
                reason: format!("filter evaluation failed: {e}"),
            })?
            .into_iter()
            .filter(|b| b.num_rows() > 0)
            .collect()
    } else {
        batches
    };

    let mut hits: Vec<Hit> = batches.iter().flat_map(|b| flatten_batch(b, kind)).collect();

    // Equal scores (match_all, ties) are ordered by `_id` so `from`/`size`
    // pagination is stable; ES itself makes no ordering guarantee for ties.
    match kind {
        ScoreKind::Relevance => {
            // Core already returns a globally score-DESC list; defensive.
            hits.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.id.cmp(&b.id))
            });
        }
        ScoreKind::Distance => {
            hits.sort_by(|a, b| {
                a.score
                    .partial_cmp(&b.score)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| a.id.cmp(&b.id))
            });
        }
        ScoreKind::None => {
            hits.sort_by(|a, b| a.id.cmp(&b.id));
        }
    }

    // Underlying BM25/HNSW candidate lists are capped per segment, so this
    // total is a best-effort approximation reported with `relation: "eq"`.
    let total = hits.len() as u64;

    if let Some(k) = knn_k {
        hits.truncate(k);
    }

    let len = hits.len();
    let from = req.from.min(len);
    let end = (from + req.size).min(len);
    let page: Vec<SearchHit> = hits[from..end]
        .iter()
        .map(|h| SearchHit {
            index: index.to_string(),
            id: h.id.clone(),
            score: Some(final_score(h, kind)),
            source: h.source.clone(),
        })
        .collect();

    let max_score = page.first().and_then(|h| h.score);

    Ok(SearchResponse {
        took: start.elapsed().as_millis() as u64,
        timed_out: false,
        hits: SearchHits {
            total: TotalHits {
                value: total,
                relation: "eq".to_string(),
            },
            max_score,
            hits: page,
        },
    })
}

fn final_score(hit: &Hit, kind: ScoreKind) -> f32 {
    match kind {
        ScoreKind::Relevance => hit.score,
        // Map a distance to an ES-style relevance score in (0, 1].
        ScoreKind::Distance => 1.0 / (1.0 + hit.score),
        ScoreKind::None => 1.0,
    }
}

fn flatten_batch(batch: &RecordBatch, kind: ScoreKind) -> Vec<Hit> {
    let n = batch.num_rows();
    let mut hits = Vec::with_capacity(n);
    let id_col = batch.column_by_name(ID_COLUMN);
    for i in 0..n {
        let id = match id_col {
            Some(c) if !c.is_null(i) => match c.data_type() {
                DataType::Utf8 => c
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .map(|a| a.value(i).to_string())
                    .unwrap_or_default(),
                DataType::LargeUtf8 => c
                    .as_any()
                    .downcast_ref::<LargeStringArray>()
                    .map(|a| a.value(i).to_string())
                    .unwrap_or_default(),
                _ => String::new(),
            },
            _ => String::new(),
        };
        let id = if id.is_empty() {
            format!("row-{i}")
        } else {
            id
        };

        let score = match kind {
            ScoreKind::None => 1.0,
            _ => {
                let last = batch.column(batch.num_columns() - 1);
                match last.as_any().downcast_ref::<Float32Array>() {
                    Some(a) if !a.is_null(i) => a.value(i),
                    _ => 0.0,
                }
            }
        };

        hits.push(Hit {
            id,
            source: row_to_json(batch, i, kind),
            score,
        });
    }
    hits
}

fn row_to_json(batch: &RecordBatch, i: usize, kind: ScoreKind) -> Value {
    let num = batch.num_columns();
    // The trailing score/distance column is synthetic; hide it from `_source`
    // only when it is the recognized distance column (a user column actually
    // named "distance" in a scored search is an accepted v1 edge case).
    let hide_trailing = matches!(kind, ScoreKind::Relevance | ScoreKind::Distance)
        && num > 0
        && batch.schema().field(num - 1).name() == "distance";
    let mut obj = Map::new();
    for c in 0..num {
        let schema = batch.schema();
        let name = schema.field(c).name();
        if name == ID_COLUMN || (c == num - 1 && hide_trailing) {
            continue;
        }
        let col = batch.column(c);
        obj.insert(name.clone(), value_to_json(col, i));
    }
    Value::Object(obj)
}

fn value_to_json(col: &dyn Array, i: usize) -> Value {
    if col.is_null(i) {
        return Value::Null;
    }
    match col.data_type() {
        DataType::Utf8 => {
            let a = col.as_any().downcast_ref::<StringArray>().unwrap();
            Value::String(a.value(i).to_string())
        }
        DataType::LargeUtf8 => {
            let a = col.as_any().downcast_ref::<LargeStringArray>().unwrap();
            Value::String(a.value(i).to_string())
        }
        DataType::Boolean => {
            let a = col.as_any().downcast_ref::<BooleanArray>().unwrap();
            Value::Bool(a.value(i))
        }
        DataType::Int8 => {
            let a = col.as_any().downcast_ref::<Int8Array>().unwrap();
            Value::from(a.value(i) as i64)
        }
        DataType::Int16 => {
            let a = col.as_any().downcast_ref::<Int16Array>().unwrap();
            Value::from(a.value(i) as i64)
        }
        DataType::Int32 => {
            let a = col.as_any().downcast_ref::<Int32Array>().unwrap();
            Value::from(a.value(i) as i64)
        }
        DataType::Int64 => {
            let a = col.as_any().downcast_ref::<Int64Array>().unwrap();
            Value::from(a.value(i))
        }
        DataType::UInt8 => {
            let a = col.as_any().downcast_ref::<UInt8Array>().unwrap();
            Value::from(a.value(i) as u64)
        }
        DataType::UInt16 => {
            let a = col.as_any().downcast_ref::<UInt16Array>().unwrap();
            Value::from(a.value(i) as u64)
        }
        DataType::UInt32 => {
            let a = col.as_any().downcast_ref::<UInt32Array>().unwrap();
            Value::from(a.value(i) as u64)
        }
        DataType::UInt64 => {
            let a = col.as_any().downcast_ref::<UInt64Array>().unwrap();
            Value::from(a.value(i))
        }
        DataType::Float32 => {
            let a = col.as_any().downcast_ref::<Float32Array>().unwrap();
            Value::from(a.value(i) as f64)
        }
        DataType::Float64 => {
            let a = col.as_any().downcast_ref::<Float64Array>().unwrap();
            Value::from(a.value(i))
        }
        DataType::Date32 => {
            let a = col.as_any().downcast_ref::<Date32Array>().unwrap();
            let days = a.value(i);
            NaiveDate::from_num_days_from_ce_opt(719163 + days)
                .map(|d| Value::String(d.to_string()))
                .unwrap_or(Value::Null)
        }
        DataType::Date64 => {
            let a = col.as_any().downcast_ref::<Date64Array>().unwrap();
            let dt = DateTime::from_timestamp_millis(a.value(i)).map(|d| d.to_rfc3339_opts(SecondsFormat::Millis, true));
            dt.map(Value::String).unwrap_or(Value::Null)
        }
        DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, _) => {
            let a = col
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            let dt = DateTime::from_timestamp_micros(a.value(i)).map(|d| {
                d.to_rfc3339_opts(SecondsFormat::Millis, true)
            });
            dt.map(Value::String).unwrap_or(Value::Null)
        }
        DataType::FixedSizeList(_, _) => {
            let a = col.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            let dim = a.value_length() as usize;
            let vals = a.values();
            let slice = vals
                .as_any()
                .downcast_ref::<Float32Array>()
                .map(|flat| {
                    flat.values()
                        [i * dim..(i + 1) * dim]
                        .to_vec()
                        .into_iter()
                        .map(|x| Value::from(x as f64))
                        .collect::<Vec<_>>()
                });
            slice.map(Value::Array).unwrap_or(Value::Null)
        }
        DataType::List(_) => {
            let a = col.as_any().downcast_ref::<ListArray>().unwrap();
            let off = a.value_offsets();
            let len = off[i + 1] - off[i];
            let vals = a
                .values()
                .as_any()
                .downcast_ref::<Float32Array>()
                .map(|flat| {
                    let s = &flat.values()[off[i] as usize..(off[i] + len) as usize];
                    s.iter().map(|x| Value::from(*x as f64)).collect::<Vec<_>>()
                });
            vals.map(Value::Array).unwrap_or(Value::Null)
        }
        DataType::Struct(_) => {
            let a = col.as_any().downcast_ref::<StructArray>().unwrap();
            let mut obj = Map::new();
            for (j, f) in a.fields().iter().enumerate() {
                obj.insert(f.name().clone(), value_to_json(&a.column(j), i));
            }
            Value::Object(obj)
        }
        dt => {
            tracing::debug!(?dt, "unmapped arrow type in _source");
            Value::Null
        }
    }
}

#[cfg(test)]
mod tests {
    use hyperstreamdb::HyperstreamError;
    use serde_json::json;

    use crate::es_types::EsError;
    use crate::handlers::docs::{index_document_core, refresh_core};
    use crate::state::AppState;

    use super::*;

    async fn index_docs(state: &AppState, index: &str, docs: &[Value]) {
        for (i, doc) in docs.iter().enumerate() {
            index_document_core(
                state,
                index,
                Some(&format!("{index}-doc-{i}")),
                doc.clone(),
            )
            .await
            .unwrap();
        }
        refresh_core(state, index)
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn search_match_bm25() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("docs")).unwrap();

        index_docs(&state, "docs", &[
            json!({"title": "alpha", "body": "quick brown fox", "category": "animal", "age": 10}),
            json!({"title": "beta", "body": "lazy dog sleeps", "category": "animal", "age": 20}),
            json!({"title": "gamma", "body": "the cat purred", "category": "animal", "age": 30}),
            json!({"title": "delta", "body": "a fish swims", "category": "seafood", "age": 40}),
        ])
        .await;

        let resp = search_core(
            &state,
            "docs",
            &json!({"query": {"match": {"body": "cat"}}}),
        )
        .await
        .unwrap();

        // BM25 scores only rows whose inverted index matched "cat".
        assert_eq!(resp.hits.total.value, 1);
        assert!(resp.hits.max_score.unwrap() > 0.0);
        assert!(!resp.timed_out);

        // Only "gamma" mentions "cat".
        let ids: Vec<&str> = resp.hits.hits.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids, vec!["docs-doc-2"]);
        let hit = &resp.hits.hits[0];
        assert_eq!(&hit.index, "docs");
        assert_eq!(hit.source["title"], "gamma");
        assert_eq!(hit.source["category"], "animal");
        assert_eq!(hit.source["age"], 30);
        // _id is addressed, never embedded in _source; no synthetic columns leak.
        assert!(!hit.source.as_object().unwrap().contains_key("_id"));
        assert!(!hit.source.as_object().unwrap().contains_key("distance"));
    }

    #[tokio::test]
    async fn search_knn_nearest_first() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("vecs")).unwrap();

        index_docs(&state, "vecs", &[
            json!({"name": "a", "vec": [1.0, 0.0]}),
            json!({"name": "b", "vec": [0.0, 1.0]}),
            json!({"name": "c", "vec": [0.1, 0.1]}),
            json!({"name": "d", "vec": [0.9, 0.1]}),
        ])
        .await;

        let resp = search_core(
            &state,
            "vecs",
            &json!({"knn": {"field": "vec", "vector": [1.0, 0.0], "k": 2}}),
        )
        .await
        .unwrap();

        assert_eq!(resp.hits.total.value, 2);
        // The exact-match doc has distance 0 → ES-style score 1/(1+0) == 1.0.
        let first = &resp.hits.hits[0];
        assert_eq!(first.id, "vecs-doc-0");
        assert!((first.score.unwrap() - 1.0).abs() < f32::EPSILON);
        // ES-style relevance is monotonic non-increasing.
        let scores: Vec<f32> = resp.hits.hits.iter().map(|h| h.score.unwrap()).collect();
        assert!(scores.windows(2).all(|w| w[0] >= w[1]));
        assert_eq!(resp.hits.hits.len(), 2);
    }

    #[tokio::test]
    async fn search_hybrid_rrf() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("hyb")).unwrap();

        index_docs(&state, "hyb", &[
            json!({"body": "hello world", "vec": [1.0, 0.0]}),
            json!({"body": "goodbye moon", "vec": [0.0, 1.0]}),
            json!({"body": "hello moon", "vec": [0.5, 0.5]}),
            json!({"body": "world moon", "vec": [1.0, 1.0]}),
        ])
        .await;

        let resp = search_core(
            &state,
            "hyb",
            &json!({
                "query": {
                    "match": {"body": "hello"},
                    "knn": {"field": "vec", "vector": [1.0, 0.0], "k": 2},
                }
            }),
        )
        .await
        .unwrap();

        assert!(!resp.hits.hits.is_empty());
        for h in &resp.hits.hits {
            let s = h.score.unwrap();
            assert!(s > 0.0 && s < 1.0, "RRF score {s} outside (0,1)");
        }
    }

    #[tokio::test]
    async fn search_filter_narrows_results() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("f")).unwrap();

        index_docs(&state, "f", &[
            json!({"title": "t1", "body": "quick brown fox", "category": "animal", "age": 10}),
            json!({"title": "t2", "body": "lazy dog sleeps", "category": "animal", "age": 45}),
            json!({"title": "t3", "body": "the cat purred", "category": "animal", "age": 30}),
            json!({"title": "t4", "body": "a fish swims", "category": "seafood", "age": 40}),
        ])
        .await;

        let resp = search_core(
            &state,
            "f",
            &json!({
                "query": {"match": {"body": "quick"}},
                "filter": {"term": {"category": "seafood"}},
            }),
        )
        .await
        .unwrap();

        // The only "seafood" row does not match "quick" → empty result.
        assert_eq!(resp.hits.total.value, 0);
        assert!(resp.hits.hits.is_empty());
        assert!(resp.hits.max_score.is_none());

        // A filter that actually intersects: animal + (quick | lazy | cat).
        let resp = search_core(
            &state,
            "f",
            &json!({
                "query": {"match": {"body": "quick"}},
                "filter": {"term": {"category": "animal"}},
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp.hits.total.value, 1);
        assert_eq!(resp.hits.hits[0].id, "f-doc-0");

        // Range filter.
        let resp = search_core(
            &state,
            "f",
            &json!({
                "query": {"match_all": {}},
                "filter": {"range": {"age": {"gte": 30}}},
            }),
        )
        .await
        .unwrap();
        let ids: Vec<&str> = resp.hits.hits.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(resp.hits.total.value, 3);
        assert_eq!(ids, vec!["f-doc-1", "f-doc-2", "f-doc-3"]);
        for h in &resp.hits.hits {
            assert_eq!(h.score, Some(1.0));
        }
    }

    #[tokio::test]
    async fn search_match_all_and_pagination() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("ma")).unwrap();

        index_docs(
            &state,
            "ma",
            &[
                json!({"n": 1}),
                json!({"n": 2}),
                json!({"n": 3}),
                json!({"n": 4}),
                json!({"n": 5}),
            ],
        )
        .await;

        let resp = search_core(
            &state,
            "ma",
            &json!({"query": {"match_all": {}}, "size": 2, "from": 1}),
        )
        .await
        .unwrap();
        assert_eq!(resp.hits.total.value, 5);
        assert_eq!(resp.hits.hits.len(), 2);
        assert_eq!(resp.hits.hits[0].id, "ma-doc-1");
        assert_eq!(resp.hits.hits[1].id, "ma-doc-2");
        for h in &resp.hits.hits {
            assert_eq!(h.score, Some(1.0));
        }
    }

    #[tokio::test]
    async fn search_unknown_index_404() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());

        let err = search_core(
            &state,
            "missing",
            &json!({"query": {"match_all": {}}}),
        )
        .await
        .unwrap_err();
        assert!(matches!(err, HyperstreamError::TableNotFound { .. }));
        let es: EsError = err.into();
        assert_eq!(es.status, 404);
    }

    #[tokio::test]
    async fn search_request_errors_are_400() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("e")).unwrap();
        index_docs(&state, "e", &[json!({"body": "hello"})])
            .await;

        let cases = vec![
            // match value as an array
            json!({"query": {"match": {"body": ["a", "b"]}}}),
            // unsupported filter clause
            json!({"filter": {"match_phrase": {"body": "a"}}}),
            // knn without a vector
            json!({"knn": {"field": "body", "k": 3}}),
            // invalid field name (SQL-injection guard)
            json!({"filter": {"term": {"bad;drop table": "x"}}}),
        ];
        for body in cases {
            let err = search_core(&state, "e", &body)
                .await
                .expect_err("expected parse error");
            assert!(
                matches!(err, HyperstreamError::SchemaIncompatible { .. }),
                "expected SchemaIncompatible, got {err:?}"
            );
            let es: EsError = err.into();
            assert_eq!(es.status, 400, "body {body}");
        }
    }
}
