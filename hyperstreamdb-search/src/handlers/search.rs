// Copyright (c) 2026 Richard Albright. All rights reserved.

//! ES-compatible search endpoint: `POST /{index}/_search`.
//!
//! Supports `match` (BM25), `knn` (HNSW), hybrid (BM25 + HNSW fused with RRF),
//! `match_all`, and a top-level `filter` (term/range/exists/bool) translated
//! to a SQL predicate evaluated with DataFusion.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use arrow::array::{
    Array, BooleanArray, Date32Array, Date64Array, FixedSizeListArray, Float32Array, Float64Array,
    Int16Array, Int32Array, Int64Array, Int8Array, LargeStringArray, ListArray, RecordBatch,
    StringArray, StructArray, TimestampMicrosecondArray, UInt16Array, UInt32Array, UInt64Array,
    UInt8Array,
};
use arrow::datatypes::DataType;
use axum::extract::{Path, State};
use axum::response::Response;
use axum::Json;
use chrono::{DateTime, NaiveDate, SecondsFormat};
use hyperstreamdb::core::index::VectorValue;
use hyperstreamdb::core::planner::{FilterExpr, QueryPlanner};
use hyperstreamdb::core::search::{HybridSearchCoordinator, KeywordSearchParams, ScoredResult};
use hyperstreamdb::{HyperstreamError, Table, VectorSearchParams};
use serde_json::{Map, Value};

use crate::es_types::{CountResponse, SearchHit, SearchHits, SearchResponse, TotalHits};
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

/// `GET /{index}/_search?q=` — a Lucene-style query string mapped to a
/// multi-field `match` over every string column (v1 approximation of ES's
/// default `_all` field). Also honours `size` and `from` query params.
pub async fn search_get(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Response {
    let q = params
        .get("q")
        .map(|s| s.as_str())
        .unwrap_or("")
        .to_string();
    let mut body = if q.is_empty() {
        serde_json::json!({ "query": { "match_all": {} } })
    } else {
        // Expand `q` into a multi-field match over every string column.
        let fields = if table_exists(&state.index_uri(&index)).await {
            state
                .open_or_create(&index, &None)
                .await
                .map(|t| {
                    let schema = t.arrow_schema();
                    let mut fields = serde_json::Map::new();
                    for f in schema.fields() {
                        if matches!(f.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
                            fields.insert(f.name().clone(), serde_json::Value::String(q.clone()));
                        }
                    }
                    fields
                })
                .unwrap_or_default()
        } else {
            serde_json::Map::new()
        };
        if fields.is_empty() {
            // No string columns (or index missing): fall back to match_all so
            // the 404 path is handled uniformly by search_core.
            serde_json::json!({ "query": { "match_all": {} } })
        } else {
            serde_json::json!({ "query": { "match": fields } })
        }
    };
    if let Some(size) = params.get("size").and_then(|s| s.parse::<u64>().ok()) {
        body["size"] = serde_json::json!(size);
    }
    if let Some(from) = params.get("from").and_then(|s| s.parse::<u64>().ok()) {
        body["from"] = serde_json::json!(from);
    }
    es_response(search_core(&state, &index, &body).await)
}

/// `GET /{index}/_count` — document count, optionally filtered by a
/// `filter` clause or a single-clause `query`.
pub async fn count(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    body: Option<Json<Value>>,
) -> Response {
    es_response(count_core(&state, &index, body.as_deref()).await)
}

pub(crate) async fn count_core(
    state: &AppState,
    index: &str,
    body: Option<&Value>,
) -> Result<CountResponse, HyperstreamError> {
    if !table_exists(&state.index_uri(index)).await {
        return Err(HyperstreamError::TableNotFound {
            namespace: String::new(),
            name: index.to_string(),
        });
    }
    let table = state.open_or_create(index, &None).await?;

    // Translate an optional filter/query into a SQL predicate.
    let filter_sql = match body {
        None => None,
        Some(b) => {
            if let Some(f) = b.get("filter") {
                Some(clause_to_sql(f, "filter")?)
            } else if let Some(q) = b.get("query") {
                match q {
                    Value::Object(m) if m.len() == 1 => {
                        let (key, val) = m.iter().next().unwrap();
                        match key.as_str() {
                            "match_all" => None,
                            other => {
                                let wrapped = serde_json::json!({ other: val });
                                Some(clause_to_sql(&wrapped, "query")?)
                            }
                        }
                    }
                    _ => None,
                }
            } else {
                None
            }
        }
    };

    let count = match filter_sql {
        None => {
            table
                .get_table_statistics_async()
                .await
                .map_err(translate_search_error)?
                .row_count
        }
        Some(sql) => {
            let batches = table
                .read_async(Some(&sql), None, None)
                .await
                .map_err(translate_search_error)?;
            batches.iter().map(|b| b.num_rows() as u64).sum()
        }
    };

    Ok(CountResponse { count })
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

/// `_source` filtering: which fields to include/exclude in hit sources.
#[derive(Debug, Clone, Default)]
struct SourceFilter {
    includes: Vec<String>,
    excludes: Vec<String>,
}

impl SourceFilter {
    /// Whether a top-level field survives the filter. Dot-prefixed includes
    /// (e.g. `"user"`) also keep nested fields (`"user.name"`), mirroring ES.
    fn keep(&self, field: &str) -> bool {
        if !self.includes.is_empty() {
            return self
                .includes
                .iter()
                .any(|i| field == i || field.starts_with(&format!("{i}.")));
        }
        !self
            .excludes
            .iter()
            .any(|e| field == e || field.starts_with(&format!("{e}.")))
    }
}

#[derive(Debug)]
struct SearchRequest {
    /// One keyword search per matched field (multi-field `match` / `q`).
    keyword: Option<Vec<KeywordSearchParams>>,
    vector: Option<VectorSearchParams>,
    /// SQL `WHERE` clause translated from the top-level ES `filter`.
    filter: Option<String>,
    size: usize,
    from: usize,
    source: Option<SourceFilter>,
    /// RRF fusion constant override (request-level; falls back to
    /// `HYPERSEARCH_RRF_K`, then the core default of 60).
    rrf_k: Option<f32>,
}

impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            keyword: None,
            vector: None,
            filter: None,
            size: 10,
            from: 0,
            source: None,
            rrf_k: None,
        }
    }
}

fn bad_request(reason: impl Into<String>) -> HyperstreamError {
    HyperstreamError::SchemaIncompatible {
        reason: reason.into(),
    }
}

/// Translate a core `anyhow::Error` from search dispatch into a typed
/// [`HyperstreamError`] so the ES error mapping returns the right status
/// (e.g. a missing filter column is a 400 `illegal_argument_exception`,
/// not a 500). Mirrors `translate_write_error` in `docs.rs`: typed variants
/// are recovered by downcast (reconstructed — `HyperstreamError` is not
/// `Clone`), DataFusion "No field named" schema errors become
/// `ColumnNotFound`, everything else stays `internal`.
fn translate_search_error(err: anyhow::Error) -> HyperstreamError {
    if let Some(he) = err.downcast_ref::<HyperstreamError>() {
        return match he {
            HyperstreamError::TableNotFound { namespace, name } => {
                HyperstreamError::TableNotFound {
                    namespace: namespace.clone(),
                    name: name.clone(),
                }
            }
            HyperstreamError::ColumnNotFound { column, table } => {
                HyperstreamError::ColumnNotFound {
                    column: column.clone(),
                    table: table.clone(),
                }
            }
            HyperstreamError::InvalidUri { uri, reason } => HyperstreamError::InvalidUri {
                uri: uri.clone(),
                reason: reason.clone(),
            },
            HyperstreamError::NullConstraintViolation { column } => {
                HyperstreamError::NullConstraintViolation {
                    column: column.clone(),
                }
            }
            HyperstreamError::SchemaIncompatible { reason } => {
                HyperstreamError::SchemaIncompatible {
                    reason: reason.clone(),
                }
            }
            _ => HyperstreamError::internal(he.to_string()),
        };
    }

    // DataFusion schema errors for a missing filter column arrive untyped
    // (anyhow): "Schema error: No field named <col>. Valid fields are …".
    // `.find` also matches the typed "DataFusion error: Schema error: …"
    // form.
    let msg = err.to_string();
    if let Some(pos) = msg.find("No field named ") {
        let rest = &msg[pos + "No field named ".len()..];
        if let Some((col, _)) = rest.split_once('.') {
            return HyperstreamError::ColumnNotFound {
                column: col.trim().to_string(),
                table: None,
            };
        }
    }
    HyperstreamError::internal(msg)
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
    if let Some(source) = obj.get("_source").or_else(|| obj.get("source")) {
        req.source = Some(parse_source(source)?);
    }
    if let Some(k) = obj.get("rrf_k").and_then(Value::as_f64) {
        if k > 0.0 {
            req.rrf_k = Some(k as f32);
        }
    }

    if let Some(query) = obj.get("query") {
        match query {
            Value::Object(m) => {
                for (key, spec) in m {
                    match key.as_str() {
                        "match_all" => {
                            if !spec.is_null() && !spec.is_object() {
                                return Err(bad_request("match_all: expected an object or null"));
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

fn parse_match(spec: &Value) -> Result<Vec<KeywordSearchParams>, HyperstreamError> {
    let m = spec.as_object().ok_or_else(|| {
        bad_request("match: expected {\"field\": \"text\"} or {\"field\": {\"query\": \"text\"}}")
    })?;
    if m.is_empty() {
        return Err(bad_request(
            "match: expected {\"field\": \"text\"} or {\"field\": {\"query\": \"text\"}}",
        ));
    }
    // BTreeMap iteration is alphabetical, giving a deterministic field order.
    // Multi-field matches are OR-merged at dispatch time.
    let mut out = Vec::with_capacity(m.len());
    for (field, v) in m {
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
        out.push(KeywordSearchParams::new(field, text));
    }
    Ok(out)
}

/// Parse `_source` / `source` into a [`SourceFilter`].
fn parse_source(v: &Value) -> Result<SourceFilter, HyperstreamError> {
    let mut f = SourceFilter::default();
    match v {
        Value::Object(m) => {
            if let Some(incl) = m.get("includes").or_else(|| m.get("include")) {
                f.includes = string_list(incl, "source.includes")?;
            }
            if let Some(excl) = m.get("excludes").or_else(|| m.get("exclude")) {
                f.excludes = string_list(excl, "source.excludes")?;
            }
        }
        Value::String(s) => {
            // A bare string is treated as a single include.
            f.includes.push(s.clone());
        }
        other => {
            return Err(bad_request(format!(
                "_source: expected an object or string, got {}",
                json_type_name(other)
            )));
        }
    }
    Ok(f)
}

fn string_list(v: &Value, ctx: &str) -> Result<Vec<String>, HyperstreamError> {
    let arr = v
        .as_array()
        .ok_or_else(|| bad_request(format!("{ctx}: expected an array of field names")))?;
    let mut out = Vec::with_capacity(arr.len());
    for x in arr {
        let s = x
            .as_str()
            .ok_or_else(|| bad_request(format!("{ctx}: entries must be strings")))?;
        out.push(s.to_string());
    }
    Ok(out)
}

fn parse_knn(spec: &Value) -> Result<VectorSearchParams, HyperstreamError> {
    let m = spec
        .as_object()
        .ok_or_else(|| bad_request("knn: expected an object"))?;
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
    let mut params = VectorSearchParams::new(&field, VectorValue::Float32(values), k);
    if let Some(nc) = m.get("num_candidates").and_then(Value::as_u64) {
        if nc > 0 {
            params = params.with_ef_search(nc as usize);
        }
    }
    Ok(params)
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
                "terms" => terms_to_sql(value, ctx),
                "range" => range_to_sql(value, ctx),
                "exists" => exists_to_sql(value, ctx),
                "bool" => bool_to_sql(value, ctx),
                other => Err(bad_request(format!(
                    "unsupported {ctx} clause '{other}' (supported: term, terms, range, exists, bool)"
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
    let leaf = if v
        .as_object()
        .is_some_and(|o| o.len() == 1 && o.contains_key("value"))
    {
        v.get("value").unwrap()
    } else {
        v
    };
    let lit = sql_literal(leaf)?;
    Ok(format!("{field} = {lit}"))
}

fn terms_to_sql(value: &Value, _ctx: &str) -> Result<String, HyperstreamError> {
    let m = value
        .as_object()
        .ok_or_else(|| bad_request("terms: expected {\"field\": [v1, v2, ...]}"))?;
    if m.len() != 1 {
        return Err(bad_request("terms: expected exactly one field"));
    }
    let (field, v) = m.iter().next().unwrap();
    let field = valid_field(field)?;
    let arr = v
        .as_array()
        .ok_or_else(|| bad_request("terms: expected an array of values"))?;
    if arr.is_empty() {
        return Err(bad_request("terms: value array must not be empty"));
    }
    let lits: Vec<String> = arr.iter().map(sql_literal).collect::<Result<_, _>>()?;
    Ok(format!("{field} IN ({})", lits.join(", ")))
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
    // ES wire format: {"exists": {"field": "<name>"}} — unlike term/range the
    // key is the literal "field" and the value is the field name.
    let m = value
        .as_object()
        .ok_or_else(|| bad_request("exists: expected {\"field\": \"<name>\"}"))?;
    if m.len() != 1 {
        return Err(bad_request("exists: expected a single \"field\" key"));
    }
    let v = m
        .get("field")
        .ok_or_else(|| bad_request("exists: expected {\"field\": \"<name>\"}"))?;
    let field = v
        .as_str()
        .ok_or_else(|| bad_request("exists: \"field\" must be a string field name"))?;
    let field = valid_field(field)?;
    Ok(format!("{field} IS NOT NULL"))
}

fn bool_to_sql(value: &Value, ctx: &str) -> Result<String, HyperstreamError> {
    let m = value
        .as_object()
        .ok_or_else(|| bad_request("bool: expected an object"))?;
    let mut parts = Vec::new();
    for key in ["must", "filter"] {
        if let Some(arr) = m.get(key).and_then(Value::as_array) {
            for clause in arr {
                parts.push(clause_to_sql(clause, ctx)?);
            }
        }
    }
    if let Some(arr) = m.get("must_not").and_then(Value::as_array) {
        for clause in arr {
            parts.push(format!("NOT ({})", clause_to_sql(clause, ctx)?));
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

    // RRF fusion constant: request-level `rrf_k` wins, then the
    // `HYPERSEARCH_RRF_K` env var, then the core default (60).
    let rrf_k = req.rrf_k.or_else(|| {
        std::env::var("HYPERSEARCH_RRF_K")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|k| *k > 0.0)
    });

    let (batches, kind, knn_k) = match (&req.keyword, &req.vector) {
        (Some(kp), Some(vp)) => {
            // Hybrid: BM25 + HNSW fused with RRF. Multi-field matches use the
            // first field for the keyword leg (v1); pure multi-field matches
            // are OR-merged in the keyword-only path below.
            let scored = HybridSearchCoordinator::new()
                .execute_hybrid(
                    &table,
                    None,
                    Some(vp.clone()),
                    Some(kp[0].clone()),
                    1000,
                    rrf_k,
                )
                .await
                .map_err(translate_search_error)?;
            let batches = table
                .fetch_results_by_id(scored, None)
                .await
                .map_err(translate_search_error)?;
            (batches, ScoreKind::Relevance, None)
        }
        (Some(kp), None) => {
            let scored = if kp.len() == 1 {
                table
                    .execute_keyword_search_as_scored(kp[0].clone())
                    .await
                    .map_err(translate_search_error)?
            } else {
                // Multi-field match: OR-merge per-field BM25 results, keeping
                // the best score per document.
                merge_keyword_results(table.as_ref(), kp).await?
            };
            let batches = table
                .fetch_results_by_id(scored, None)
                .await
                .map_err(translate_search_error)?;
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
                    .map_err(translate_search_error)?;
                (batches, ScoreKind::Distance, Some(vp.k))
            } else {
                let scored = table
                    .execute_vector_search_as_scored(vp.clone())
                    .await
                    .map_err(translate_search_error)?;
                let batches = table
                    .fetch_results_by_id(scored, None)
                    .await
                    .map_err(translate_search_error)?;
                (batches, ScoreKind::Distance, Some(vp.k))
            }
        }
        (None, None) => {
            let batches = table
                .read_async(req.filter.as_deref(), None, None)
                .await
                .map_err(translate_search_error)?;
            (batches, ScoreKind::None, None)
        }
    };

    // Post-filter scanned batches unless the core already applied the filter
    // (the knn+filter `read_async` path pre-filters inside the scan).
    let batches = if kind != ScoreKind::Distance {
        match &req.filter {
            Some(sql) => {
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
            }
            None => batches,
        }
    } else {
        batches
    };

    let mut hits: Vec<Hit> = batches
        .iter()
        .flat_map(|b| flatten_batch(b, kind, &req.source))
        .collect();

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

    // Query-latency histogram by operation class (plan 5.2.2).
    let op = match (&req.keyword, &req.vector) {
        (Some(_), Some(_)) => "hybrid",
        (Some(_), None) => "match",
        (None, Some(_)) => "knn",
        (None, None) => "filter",
    };
    state
        .metrics
        .query_seconds
        .with_label_values(&[op])
        .observe(start.elapsed().as_secs_f64());

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

/// OR-merge per-field BM25 results for a multi-field `match`, keeping the
/// best (highest) score per document and returning a single score-desc list.
async fn merge_keyword_results(
    table: &Table,
    params: &[KeywordSearchParams],
) -> Result<Vec<ScoredResult>, HyperstreamError> {
    let mut merged: HashMap<(String, u32), f32> = HashMap::new();
    for kp in params {
        let scored = table
            .execute_keyword_search_as_scored(kp.clone())
            .await
            .map_err(translate_search_error)?;
        for r in scored {
            let key = (r.segment_id.clone(), r.row_id);
            let entry = merged.entry(key).or_insert(0.0);
            if r.score > *entry {
                *entry = r.score;
            }
        }
    }
    let mut results: Vec<ScoredResult> = merged
        .into_iter()
        .map(|((segment_id, row_id), score)| ScoredResult {
            segment_id,
            row_id,
            score,
        })
        .collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
    Ok(results)
}

fn final_score(hit: &Hit, kind: ScoreKind) -> f32 {
    match kind {
        ScoreKind::Relevance => hit.score,
        // Map a distance to an ES-style relevance score in (0, 1].
        ScoreKind::Distance => 1.0 / (1.0 + hit.score),
        ScoreKind::None => 1.0,
    }
}

fn flatten_batch(batch: &RecordBatch, kind: ScoreKind, source: &Option<SourceFilter>) -> Vec<Hit> {
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
            source: row_to_json(batch, i, kind, source),
            score,
        });
    }
    hits
}

fn row_to_json(
    batch: &RecordBatch,
    i: usize,
    kind: ScoreKind,
    source: &Option<SourceFilter>,
) -> Value {
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
        if let Some(sf) = source {
            if !sf.keep(name) {
                continue;
            }
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
            let dt = DateTime::from_timestamp_millis(a.value(i))
                .map(|d| d.to_rfc3339_opts(SecondsFormat::Millis, true));
            dt.map(Value::String).unwrap_or(Value::Null)
        }
        DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, _) => {
            let a = col
                .as_any()
                .downcast_ref::<TimestampMicrosecondArray>()
                .unwrap();
            let dt = DateTime::from_timestamp_micros(a.value(i))
                .map(|d| d.to_rfc3339_opts(SecondsFormat::Millis, true));
            dt.map(Value::String).unwrap_or(Value::Null)
        }
        DataType::FixedSizeList(_, _) => {
            let a = col.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
            let dim = a.value_length() as usize;
            let vals = a.values();
            let slice = vals.as_any().downcast_ref::<Float32Array>().map(|flat| {
                flat.values()[i * dim..(i + 1) * dim]
                    .iter()
                    .copied()
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
            index_document_core(state, index, Some(&format!("{index}-doc-{i}")), doc.clone())
                .await
                .unwrap();
        }
        refresh_core(state, index).await.unwrap();
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

        index_docs(
            &state,
            "vecs",
            &[
                json!({"name": "a", "vec": [1.0, 0.0]}),
                json!({"name": "b", "vec": [0.0, 1.0]}),
                json!({"name": "c", "vec": [0.1, 0.1]}),
                json!({"name": "d", "vec": [0.9, 0.1]}),
            ],
        )
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

        index_docs(
            &state,
            "hyb",
            &[
                json!({"body": "hello world", "vec": [1.0, 0.0]}),
                json!({"body": "goodbye moon", "vec": [0.0, 1.0]}),
                json!({"body": "hello moon", "vec": [0.5, 0.5]}),
                json!({"body": "world moon", "vec": [1.0, 1.0]}),
            ],
        )
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

        index_docs(
            &state,
            "f",
            &[
                json!({"title": "t1", "body": "quick brown fox", "category": "animal", "age": 10}),
                json!({"title": "t2", "body": "lazy dog sleeps", "category": "animal", "age": 45}),
                json!({"title": "t3", "body": "the cat purred", "category": "animal", "age": 30}),
                json!({"title": "t4", "body": "a fish swims", "category": "seafood", "age": 40}),
            ],
        )
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

        let err = search_core(&state, "missing", &json!({"query": {"match_all": {}}}))
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
        index_docs(&state, "e", &[json!({"body": "hello"})]).await;

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

    #[tokio::test]
    async fn exists_filter_es_wire_format() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("ex")).unwrap();

        index_docs(
            &state,
            "ex",
            &[
                json!({"title": "t1", "body": "quick brown fox"}),
                json!({"title": "t2", "body": "lazy dog sleeps", "extra": "x"}),
                json!({"title": "t3", "body": "the cat purred"}),
                json!({"title": "t4", "body": "a fish swims", "extra": "y"}),
            ],
        )
        .await;

        let resp = search_core(
            &state,
            "ex",
            &json!({"query": {"match_all": {}}, "filter": {"exists": {"field": "extra"}}}),
        )
        .await
        .unwrap();
        let ids: Vec<&str> = resp.hits.hits.iter().map(|h| h.id.as_str()).collect();
        assert_eq!(ids, vec!["ex-doc-1", "ex-doc-3"]);
    }

    #[test]
    fn exists_filter_malformed_shapes_are_400() {
        let cases = [
            json!({"filter": {"exists": {"other": "x"}}}),
            json!({"filter": {"exists": {"field": 5}}}),
            json!({"filter": {"exists": {"field": "a", "x": "b"}}}),
            json!({"filter": {"exists": {"field": "bad;drop"}}}),
        ];
        for body in cases {
            let err = clause_to_sql(&body["filter"], "filter").unwrap_err();
            assert!(
                matches!(err, HyperstreamError::SchemaIncompatible { .. }),
                "expected SchemaIncompatible, got {err:?}"
            );
            let es: EsError = err.into();
            assert_eq!(es.status, 400, "body {body}");
        }
    }

    #[tokio::test]
    async fn unknown_filter_column_is_400_not_500() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("nf")).unwrap();

        index_docs(&state, "nf", &[json!({"body": "hello", "age": 30})]).await;

        let err = search_core(
            &state,
            "nf",
            &json!({"query": {"match_all": {}}, "filter": {"term": {"nope": "x"}}}),
        )
        .await
        .expect_err("expected column error");
        match &err {
            HyperstreamError::ColumnNotFound { column, .. } => assert_eq!(column, "nope"),
            other => panic!("expected ColumnNotFound, got {other:?}"),
        }
        let es: EsError = err.into();
        assert_eq!(es.status, 400);
    }
}
