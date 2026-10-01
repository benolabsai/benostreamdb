// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

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
use axum::response::{IntoResponse, Response};
use axum::Json;
use benostreamdb::core::index::VectorValue;
use benostreamdb::core::planner::{FilterExpr, QueryPlanner};
use benostreamdb::core::search::{HybridSearchCoordinator, KeywordSearchParams, ScoredResult};
use benostreamdb::{BenoStreamError, GraphNeighborhoodOptions, Table, VectorSearchParams};
use chrono::{DateTime, NaiveDate, SecondsFormat};
use serde_json::{Map, Value};

use crate::es_types::{
    ClearScrollResponse, CountResponse, SearchHit, SearchHits, SearchResponse, TotalHits,
};
use crate::handlers::docs::ID_COLUMN;
use crate::state::AppState;
use base64::engine::general_purpose::{STANDARD, URL_SAFE, URL_SAFE_NO_PAD};
use base64::Engine;
use serde::{Deserialize, Serialize};

use super::es_response;

pub async fn search(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
    Json(mut body): Json<Value>,
) -> Response {
    if let Some(scroll) = params.get("scroll") {
        if let Some(obj) = body.as_object_mut() {
            if !obj.contains_key("scroll") {
                obj.insert("scroll".to_string(), Value::String(scroll.clone()));
            }
        }
    }
    es_response(search_core(&state, &index, &body).await)
}

pub async fn scroll_post(State(state): State<Arc<AppState>>, Json(body): Json<Value>) -> Response {
    let scroll_id = body.get("scroll_id").and_then(Value::as_str).unwrap_or("");
    let scroll_override = body.get("scroll").and_then(Value::as_str);
    es_response(execute_scroll(&state, scroll_id, scroll_override).await)
}

pub async fn scroll_get(
    State(state): State<Arc<AppState>>,
    axum::extract::Query(params): axum::extract::Query<HashMap<String, String>>,
) -> Response {
    let scroll_id = params.get("scroll_id").map(|s| s.as_str()).unwrap_or("");
    let scroll_override = params.get("scroll").map(|s| s.as_str());
    es_response(execute_scroll(&state, scroll_id, scroll_override).await)
}

pub async fn clear_scroll(body: Option<Json<Value>>) -> Response {
    let _ = body;
    (
        axum::http::StatusCode::OK,
        Json(ClearScrollResponse {
            succeeded: true,
            num_freed: 1,
        }),
    )
        .into_response()
}

pub async fn clear_scroll_path(Path(_scroll_id): Path<String>) -> Response {
    (
        axum::http::StatusCode::OK,
        Json(ClearScrollResponse {
            succeeded: true,
            num_freed: 1,
        }),
    )
        .into_response()
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
        let string_fields = if state.index_exists(&index).await {
            state
                .open_or_create(&index, &None)
                .await
                .map(|t| {
                    let schema = t.arrow_schema();
                    let mut fields = Vec::new();
                    for f in schema.fields() {
                        if matches!(f.data_type(), DataType::Utf8 | DataType::LargeUtf8) {
                            fields.push(f.name().clone());
                        }
                    }
                    fields
                })
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        let opts = crate::handlers::query_string::QueryStringOptions {
            fields: string_fields,
            ..Default::default()
        };
        match crate::handlers::query_string::parse_query_string(&q, &opts, false) {
            Ok(query_ast) => serde_json::json!({ "query": query_ast }),
            Err(_) => serde_json::json!({ "query": { "match_all": {} } }),
        }
    };
    if let Some(size) = params.get("size").and_then(|s| s.parse::<u64>().ok()) {
        body["size"] = serde_json::json!(size);
    }
    if let Some(from) = params.get("from").and_then(|s| s.parse::<u64>().ok()) {
        body["from"] = serde_json::json!(from);
    }
    if let Some(scroll) = params.get("scroll") {
        body["scroll"] = serde_json::json!(scroll);
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
) -> Result<CountResponse, BenoStreamError> {
    if !state.index_exists(index).await {
        return Err(BenoStreamError::TableNotFound {
            namespace: String::new(),
            name: index.to_string(),
        });
    }
    let table = state.open_or_create(index, &None).await?;

    let resolved_body = match body {
        Some(b) => Some(resolve_relation_queries(state, index, b).await?),
        None => None,
    };
    let body = resolved_body.as_ref();

    // Translate an optional filter/query into a SQL predicate.
    let filter_sql = match body {
        None => None,
        Some(b) => {
            if let Some(f) = b.get("filter") {
                Some(clause_to_sql(f, "filter")?)
            } else if let Some(q) = b.get("query") {
                match q {
                    Value::Object(m) => match m.iter().next() {
                        Some((key, val)) if m.len() == 1 => match key.as_str() {
                            "match_all" => None,
                            other => {
                                let wrapped = serde_json::json!({ other: val });
                                Some(clause_to_sql(&wrapped, "query")?)
                            }
                        },
                        _ => None,
                    },
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

#[derive(Debug, Clone)]
pub struct PhraseSearchParams {
    pub column: String,
    pub phrase: String,
    pub slop: usize,
}

#[derive(Debug, Clone)]
pub struct SortClause {
    pub field: String,
    pub descending: bool,
}

#[derive(Debug, Clone, Default)]
pub struct HighlightSpec {
    pub fields: Vec<String>,
    pub pre_tags: Vec<String>,
    pub post_tags: Vec<String>,
}

#[derive(Debug)]
struct SearchRequest {
    /// One keyword search per matched field (multi-field `match` / `q`).
    keyword: Option<Vec<KeywordSearchParams>>,
    phrase: Option<PhraseSearchParams>,
    vector: Option<VectorSearchParams>,
    /// SQL `WHERE` clause translated from the top-level ES `filter`.
    filter: Option<String>,
    size: usize,
    from: usize,
    source: Option<SourceFilter>,
    /// RRF fusion constant override (request-level; falls back to
    /// `BENOSEARCH_RRF_K`, then the core default of 60).
    rrf_k: Option<f32>,
    /// ES `aggs` / `aggregations` object, computed over the top-level filter.
    aggs: Option<Value>,
    /// Sort criteria
    sort: Option<Vec<SortClause>>,
    search_after: Option<Vec<Value>>,
    highlight: Option<HighlightSpec>,
    scroll: Option<String>,
}

impl Default for SearchRequest {
    fn default() -> Self {
        Self {
            keyword: None,
            phrase: None,
            vector: None,
            filter: None,
            size: 10,
            from: 0,
            source: None,
            rrf_k: None,
            aggs: None,
            sort: None,
            search_after: None,
            highlight: None,
            scroll: None,
        }
    }
}

fn bad_request(reason: impl Into<String>) -> BenoStreamError {
    BenoStreamError::SchemaIncompatible {
        reason: reason.into(),
    }
}

/// Translate a core `anyhow::Error` from search dispatch into a typed
/// [`BenoStreamError`] so the ES error mapping returns the right status
/// (e.g. a missing filter column is a 400 `illegal_argument_exception`,
/// not a 500). Mirrors `translate_write_error` in `docs.rs`: typed variants
/// are recovered by downcast (reconstructed — `BenoStreamError` is not
/// `Clone`), DataFusion "No field named" schema errors become
/// `ColumnNotFound`, everything else stays `internal`.
fn translate_search_error(err: anyhow::Error) -> BenoStreamError {
    if let Some(he) = err.downcast_ref::<BenoStreamError>() {
        return match he {
            BenoStreamError::TableNotFound { namespace, name } => BenoStreamError::TableNotFound {
                namespace: namespace.clone(),
                name: name.clone(),
            },
            BenoStreamError::ColumnNotFound { column, table } => BenoStreamError::ColumnNotFound {
                column: column.clone(),
                table: table.clone(),
            },
            BenoStreamError::InvalidUri { uri, reason } => BenoStreamError::InvalidUri {
                uri: uri.clone(),
                reason: reason.clone(),
            },
            BenoStreamError::NullConstraintViolation { column } => {
                BenoStreamError::NullConstraintViolation {
                    column: column.clone(),
                }
            }
            BenoStreamError::SchemaIncompatible { reason } => BenoStreamError::SchemaIncompatible {
                reason: reason.clone(),
            },
            _ => BenoStreamError::internal(he.to_string()),
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
            return BenoStreamError::ColumnNotFound {
                column: col.trim().to_string(),
                table: None,
            };
        }
    }
    BenoStreamError::internal(msg)
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

fn parse_request(body: &Value) -> Result<SearchRequest, BenoStreamError> {
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
    if let Some(aggs) = obj.get("aggs").or_else(|| obj.get("aggregations")) {
        if !aggs.is_object() {
            return Err(bad_request("aggs: expected an object"));
        }
        req.aggs = Some(aggs.clone());
    }
    if let Some(sort_val) = obj.get("sort") {
        req.sort = Some(parse_sort(sort_val)?);
    }
    if let Some(sa) = obj.get("search_after").and_then(Value::as_array) {
        req.search_after = Some(sa.clone());
    }
    if let Some(hl) = obj.get("highlight").and_then(Value::as_object) {
        let mut fields = Vec::new();
        if let Some(f_map) = hl.get("fields").and_then(Value::as_object) {
            for k in f_map.keys() {
                fields.push(k.clone());
            }
        }
        let pre_tags = hl
            .get("pre_tags")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_else(|| vec!["<em>".to_string()]);
        let post_tags = hl
            .get("post_tags")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(Value::as_str)
                    .map(String::from)
                    .collect()
            })
            .unwrap_or_else(|| vec!["</em>".to_string()]);
        req.highlight = Some(HighlightSpec {
            fields,
            pre_tags,
            post_tags,
        });
    }

    if let Some(scroll) = obj.get("scroll").and_then(Value::as_str) {
        req.scroll = Some(scroll.to_string());
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
                        "multi_match" => req.keyword = Some(parse_multi_match(spec)?),
                        "match_phrase" => req.phrase = Some(parse_match_phrase(spec)?),
                        "knn" => req.vector = Some(parse_knn(spec)?),
                        "bool" => {
                            let sql = bool_to_sql(spec, "query.bool")?;
                            req.filter = match req.filter.take() {
                                Some(curr) => Some(format!("({curr}) AND ({sql})")),
                                None => Some(sql),
                            };
                        }
                        "nested" => {
                            let sql = nested_to_sql(spec, "query.nested")?;
                            req.filter = match req.filter.take() {
                                Some(curr) => Some(format!("({curr}) AND ({sql})")),
                                None => Some(sql),
                            };
                        }
                        "term" | "terms" | "range" | "exists" | "prefix" | "wildcard"
                        | "regexp" | "ids" | "fuzzy" => {
                            let mut wrap = serde_json::Map::new();
                            wrap.insert(key.clone(), spec.clone());
                            let sql = clause_to_sql(&Value::Object(wrap), "query")?;
                            req.filter = match req.filter.take() {
                                Some(curr) => Some(format!("({curr}) AND ({sql})")),
                                None => Some(sql),
                            };
                        }
                        "query_string" | "simple_query_string" => {
                            let is_simple = key.as_str() == "simple_query_string";
                            let (q, opts) =
                                crate::handlers::query_string::QueryStringOptions::from_value(
                                    spec,
                                )?;
                            let ast = crate::handlers::query_string::parse_query_string(
                                &q, &opts, is_simple,
                            )?;
                            let mut wrap = serde_json::Map::new();
                            wrap.insert("query".to_string(), ast);
                            return parse_request(&Value::Object(wrap));
                        }
                        other => {
                            return Err(bad_request(format!(
                                "unsupported query clause '{other}' (supported: match, multi_match, match_phrase, match_all, knn, bool, term, terms, range, exists, prefix, wildcard, ids, nested, has_child, has_parent, query_string, simple_query_string)"
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

    if req.scroll.is_some() && req.sort.is_none() {
        req.sort = Some(vec![SortClause {
            field: "_id".to_string(),
            descending: false,
        }]);
    }

    Ok(req)
}

fn parse_match(spec: &Value) -> Result<Vec<KeywordSearchParams>, BenoStreamError> {
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

fn parse_multi_match(spec: &Value) -> Result<Vec<KeywordSearchParams>, BenoStreamError> {
    let m = spec
        .as_object()
        .ok_or_else(|| bad_request("multi_match: expected an object with query and fields"))?;
    let query_str = m
        .get("query")
        .and_then(Value::as_str)
        .ok_or_else(|| bad_request("multi_match: missing 'query' string"))?;
    let fields_arr = m
        .get("fields")
        .and_then(Value::as_array)
        .ok_or_else(|| bad_request("multi_match: missing 'fields' array"))?;
    let mut out = Vec::with_capacity(fields_arr.len());
    for f in fields_arr {
        let field_str = f
            .as_str()
            .ok_or_else(|| bad_request("multi_match: fields array elements must be strings"))?;
        let field = valid_field(field_str)?;
        out.push(KeywordSearchParams::new(field, query_str.to_string()));
    }
    Ok(out)
}

fn parse_match_phrase(spec: &Value) -> Result<PhraseSearchParams, BenoStreamError> {
    let m = spec.as_object().ok_or_else(|| {
        bad_request(
            "match_phrase: expected {\"field\": \"phrase\"} or {\"field\": {\"query\": \"phrase\"}}",
        )
    })?;
    if m.is_empty() {
        return Err(bad_request("match_phrase: expected at least one field"));
    }
    let (field, v) = m
        .iter()
        .next()
        .ok_or_else(|| bad_request("match_phrase: expected at least one field"))?;
    let field = valid_field(field)?;
    let (phrase, slop) = match v {
        Value::String(s) => (s.clone(), 0),
        Value::Object(o) => {
            let phrase = o
                .get("query")
                .or_else(|| o.get("value"))
                .and_then(Value::as_str)
                .ok_or_else(|| bad_request("match_phrase: missing query string"))?
                .to_string();
            let slop = o.get("slop").and_then(Value::as_u64).unwrap_or(0) as usize;
            (phrase, slop)
        }
        _ => return Err(bad_request("match_phrase: expected string or object")),
    };
    Ok(PhraseSearchParams {
        column: field,
        phrase,
        slop,
    })
}

fn parse_sort(v: &Value) -> Result<Vec<SortClause>, BenoStreamError> {
    let mut clauses = Vec::new();
    let arr = match v {
        Value::Array(a) => a.clone(),
        Value::String(s) => {
            let mut parts = s.split(':');
            let field = parts.next().unwrap_or("").trim();
            let desc = parts
                .next()
                .map(|o| o.eq_ignore_ascii_case("desc"))
                .unwrap_or(false);
            if !field.is_empty() {
                clauses.push(SortClause {
                    field: field.to_string(),
                    descending: desc,
                });
            }
            return Ok(clauses);
        }
        Value::Object(_) => vec![v.clone()],
        _ => return Err(bad_request("sort: expected array, string, or object")),
    };

    for item in arr {
        match item {
            Value::String(s) => {
                let mut parts = s.split(':');
                let field = parts.next().unwrap_or("").trim();
                let desc = parts
                    .next()
                    .map(|o| o.eq_ignore_ascii_case("desc"))
                    .unwrap_or(false);
                if !field.is_empty() {
                    clauses.push(SortClause {
                        field: field.to_string(),
                        descending: desc,
                    });
                }
            }
            Value::Object(m) => {
                for (field, val) in m {
                    let desc = match val {
                        Value::String(s) => s.eq_ignore_ascii_case("desc"),
                        Value::Object(o) => o
                            .get("order")
                            .and_then(Value::as_str)
                            .map(|s| s.eq_ignore_ascii_case("desc"))
                            .unwrap_or(false),
                        _ => false,
                    };
                    clauses.push(SortClause {
                        field,
                        descending: desc,
                    });
                }
            }
            _ => {}
        }
    }
    Ok(clauses)
}

fn compare_json_values(a: Option<&Value>, b: Option<&Value>) -> Ordering {
    match (a, b) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater, // nulls sort last
        (Some(_), None) => Ordering::Less,
        (Some(va), Some(vb)) => match (va, vb) {
            (Value::Null, Value::Null) => Ordering::Equal,
            (Value::Null, _) => Ordering::Greater,
            (_, Value::Null) => Ordering::Less,
            (Value::Bool(b1), Value::Bool(b2)) => b1.cmp(b2),
            (Value::Number(n1), Value::Number(n2)) => {
                let f1 = n1.as_f64().unwrap_or(0.0);
                let f2 = n2.as_f64().unwrap_or(0.0);
                f1.partial_cmp(&f2).unwrap_or(Ordering::Equal)
            }
            (Value::String(s1), Value::String(s2)) => s1.cmp(s2),
            _ => Ordering::Equal,
        },
    }
}

fn get_json_path<'a>(mut val: &'a Value, path: &str) -> Option<&'a Value> {
    for part in path.split('.') {
        val = val.as_object()?.get(part)?;
    }
    Some(val)
}

fn extract_sort_values(hit: &Hit, sort_clauses: &[SortClause]) -> Vec<Value> {
    sort_clauses
        .iter()
        .map(|sc| {
            if sc.field == "_score" {
                Value::from(hit.score)
            } else if sc.field == "_id" {
                Value::String(hit.id.clone())
            } else {
                get_json_path(&hit.source, &sc.field)
                    .cloned()
                    .unwrap_or(Value::Null)
            }
        })
        .collect()
}

fn is_hit_after_cursor(hit_vals: &[Value], cursor: &[Value], sort_clauses: &[SortClause]) -> bool {
    for (i, sc) in sort_clauses.iter().enumerate() {
        let hv = hit_vals.get(i);
        let cv = cursor.get(i);
        let cmp = compare_json_values(hv, cv);
        let directed = if sc.descending { cmp.reverse() } else { cmp };
        match directed {
            Ordering::Greater => return true,
            Ordering::Less => return false,
            Ordering::Equal => continue,
        }
    }
    false
}

fn extract_highlights(
    hit: &Hit,
    hl: &HighlightSpec,
    req: &SearchRequest,
) -> Option<HashMap<String, Vec<String>>> {
    let mut terms: Vec<String> = Vec::new();
    if let Some(kps) = &req.keyword {
        for kp in kps {
            for token in kp.query.split_whitespace() {
                let cleaned: String = token.chars().filter(|c| c.is_alphanumeric()).collect();
                if !cleaned.is_empty() {
                    terms.push(cleaned.to_lowercase());
                }
            }
        }
    }
    if let Some(pp) = &req.phrase {
        for token in pp.phrase.split_whitespace() {
            let cleaned: String = token.chars().filter(|c| c.is_alphanumeric()).collect();
            if !cleaned.is_empty() {
                terms.push(cleaned.to_lowercase());
            }
        }
    }
    if terms.is_empty() {
        return None;
    }
    let pre = hl.pre_tags.first().map(String::as_str).unwrap_or("<em>");
    let post = hl.post_tags.first().map(String::as_str).unwrap_or("</em>");

    let mut result_map = HashMap::new();
    for field in &hl.fields {
        if let Some(val) = get_json_path(&hit.source, field).and_then(Value::as_str) {
            let mut highlighted = val.to_string();
            let mut matched = false;
            for t in &terms {
                let lower = highlighted.to_lowercase();
                if let Some(idx) = lower.find(t) {
                    matched = true;
                    let orig = &highlighted[idx..idx + t.len()];
                    highlighted = format!(
                        "{}{}{}{}{}",
                        &highlighted[..idx],
                        pre,
                        orig,
                        post,
                        &highlighted[idx + t.len()..]
                    );
                }
            }
            if matched {
                result_map.insert(field.clone(), vec![highlighted]);
            }
        }
    }
    if result_map.is_empty() {
        None
    } else {
        Some(result_map)
    }
}

/// Parse `_source` / `source` into a [`SourceFilter`].
fn parse_source(v: &Value) -> Result<SourceFilter, BenoStreamError> {
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

fn string_list(v: &Value, ctx: &str) -> Result<Vec<String>, BenoStreamError> {
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

fn parse_knn(spec: &Value) -> Result<VectorSearchParams, BenoStreamError> {
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
/// Supports standard identifiers and nested dot notation (e.g. "user.name" -> "user['name']").
fn valid_field(field: &str) -> Result<String, BenoStreamError> {
    if field.is_empty() {
        return Err(bad_request(format!("invalid field name '{field}'")));
    }
    let parts: Vec<&str> = field.split('.').collect();
    let all_valid = parts.iter().all(|part| {
        !part.is_empty()
            && part.chars().enumerate().all(|(i, c)| {
                if i == 0 {
                    c == '_' || c.is_ascii_alphabetic()
                } else {
                    c.is_ascii_alphanumeric() || c == '_'
                }
            })
    });
    if !all_valid {
        return Err(bad_request(format!("invalid field name '{field}'")));
    }
    if parts.len() == 1 {
        Ok(field.to_string())
    } else {
        let mut out = parts[0].to_string();
        for p in &parts[1..] {
            out.push_str(&format!("['{p}']"));
        }
        Ok(out)
    }
}

pub(crate) fn clause_to_sql(clause: &Value, ctx: &str) -> Result<String, BenoStreamError> {
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
        Value::Object(m) => match m.iter().next() {
            Some((key, value)) if m.len() == 1 => match key.as_str() {
                "term" => term_to_sql(value, ctx),
                "terms" => terms_to_sql(value, ctx),
                "range" => range_to_sql(value, ctx),
                "exists" => exists_to_sql(value, ctx),
                "prefix" => prefix_to_sql(value, ctx),
                "wildcard" => wildcard_to_sql(value, ctx),
                "regexp" => regexp_to_sql(value, ctx),
                "ids" => ids_to_sql(value, ctx),
                "bool" => bool_to_sql(value, ctx),
                "nested" => nested_to_sql(value, ctx),
                "match" => match_to_sql(value, ctx),
                "match_phrase" => match_phrase_to_sql(value, ctx),
                "multi_match" => multi_match_to_sql(value, ctx),
                "fuzzy" => fuzzy_to_sql(value, ctx),
                other => Err(bad_request(format!(
                    "unsupported {ctx} clause '{other}' (supported: term, terms, range, exists, prefix, wildcard, regexp, ids, bool, nested, match, match_phrase, multi_match, fuzzy)"
                ))),
            },
            _ => Err(bad_request(format!(
                "{ctx}: expected a single-key filter clause object, got {}",
                json_type_name(clause)
            ))),
        },
        other => Err(bad_request(format!(
            "{ctx}: expected a filter clause object, got {}",
            json_type_name(other)
        ))),
    }
}

fn qualify_nested_paths(val: &Value, path: &str) -> Value {
    match val {
        Value::Object(m) => {
            let mut new_m = Map::new();
            for (k, v) in m {
                let new_v = qualify_nested_paths(v, path);
                let is_query_keyword = matches!(
                    k.as_str(),
                    "query"
                        | "bool"
                        | "must"
                        | "filter"
                        | "should"
                        | "must_not"
                        | "term"
                        | "terms"
                        | "range"
                        | "match"
                        | "match_phrase"
                        | "prefix"
                        | "wildcard"
                        | "fuzzy"
                        | "regexp"
                        | "ids"
                        | "nested"
                        | "exists"
                        | "path"
                );
                if !is_query_keyword && !k.starts_with(path) && !k.starts_with('_') {
                    new_m.insert(format!("{path}.{k}"), new_v);
                } else {
                    new_m.insert(k.clone(), new_v);
                }
            }
            Value::Object(new_m)
        }
        Value::Array(arr) => Value::Array(
            arr.iter()
                .map(|item| qualify_nested_paths(item, path))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn unqualify_nested_paths(val: &Value, path: &str) -> Value {
    let prefix = format!("{path}.");
    match val {
        Value::Object(m) => {
            let mut new_m = Map::new();
            for (k, v) in m {
                let new_v = unqualify_nested_paths(v, path);
                if let Some(stripped) = k.strip_prefix(&prefix) {
                    new_m.insert(stripped.to_string(), new_v);
                } else {
                    new_m.insert(k.clone(), new_v);
                }
            }
            Value::Object(new_m)
        }
        Value::Array(arr) => Value::Array(
            arr.iter()
                .map(|item| unqualify_nested_paths(item, path))
                .collect(),
        ),
        other => other.clone(),
    }
}

fn nested_to_sql(value: &Value, ctx: &str) -> Result<String, BenoStreamError> {
    let obj = value
        .as_object()
        .ok_or_else(|| bad_request("nested: expected an object with 'path' and 'query'"))?;
    let path = obj.get("path").and_then(Value::as_str).unwrap_or("");
    let inner_query = obj
        .get("query")
        .ok_or_else(|| bad_request("nested: missing 'query'"))?;
    let qualified = if !path.is_empty() {
        qualify_nested_paths(inner_query, path)
    } else {
        inner_query.clone()
    };
    clause_to_sql(&qualified, ctx)
}

fn match_to_sql(value: &Value, _ctx: &str) -> Result<String, BenoStreamError> {
    let m = value
        .as_object()
        .ok_or_else(|| bad_request("match: expected {\"field\": \"text\"}"))?;
    if m.is_empty() {
        return Err(bad_request("match: expected at least one field"));
    }
    let mut parts = Vec::new();
    for (field, v) in m {
        let field = valid_field(field)?;
        let query_str = match v {
            Value::String(s) => s.as_str(),
            Value::Object(o) => o
                .get("query")
                .and_then(Value::as_str)
                .ok_or_else(|| bad_request("match: missing query string"))?,
            _ => return Err(bad_request("match: expected string or object")),
        };
        let words: Vec<&str> = query_str.split_whitespace().collect();
        if words.is_empty() {
            parts.push("true".to_string());
        } else {
            let word_preds: Vec<String> = words
                .into_iter()
                .map(|w| {
                    let escaped = w.replace('\'', "''");
                    format!("lower({field}) LIKE lower('%{escaped}%')")
                })
                .collect();
            parts.push(format!("({})", word_preds.join(" OR ")));
        }
    }
    Ok(parts.join(" AND "))
}

fn match_phrase_to_sql(value: &Value, _ctx: &str) -> Result<String, BenoStreamError> {
    let m = value
        .as_object()
        .ok_or_else(|| bad_request("match_phrase: expected {\"field\": \"phrase\"}"))?;
    if m.is_empty() {
        return Err(bad_request("match_phrase: expected at least one field"));
    }
    let mut parts = Vec::new();
    for (field, v) in m {
        let field = valid_field(field)?;
        let phrase_str = match v {
            Value::String(s) => s.as_str(),
            Value::Object(o) => o
                .get("query")
                .or_else(|| o.get("phrase"))
                .and_then(Value::as_str)
                .ok_or_else(|| bad_request("match_phrase: missing query string"))?,
            _ => return Err(bad_request("match_phrase: expected string or object")),
        };
        let escaped = phrase_str.replace('\'', "''");
        parts.push(format!("lower({field}) LIKE lower('%{escaped}%')"));
    }
    Ok(parts.join(" AND "))
}

fn multi_match_to_sql(value: &Value, _ctx: &str) -> Result<String, BenoStreamError> {
    let obj = value
        .as_object()
        .ok_or_else(|| bad_request("multi_match: expected an object"))?;
    let query_str = obj
        .get("query")
        .and_then(Value::as_str)
        .ok_or_else(|| bad_request("multi_match: missing 'query'"))?;
    let fields = obj
        .get("fields")
        .and_then(Value::as_array)
        .ok_or_else(|| bad_request("multi_match: missing 'fields'"))?;

    let words: Vec<&str> = query_str.split_whitespace().collect();
    let mut field_preds = Vec::new();
    for f in fields {
        if let Some(fname) = f.as_str() {
            let clean_name = fname.split('^').next().unwrap_or(fname);
            let valid = valid_field(clean_name)?;
            if words.is_empty() {
                field_preds.push("true".to_string());
            } else {
                let word_preds: Vec<String> = words
                    .iter()
                    .map(|w| {
                        let escaped = w.replace('\'', "''");
                        format!("lower({valid}) LIKE lower('%{escaped}%')")
                    })
                    .collect();
                field_preds.push(format!("({})", word_preds.join(" OR ")));
            }
        }
    }
    if field_preds.is_empty() {
        Ok("true".to_string())
    } else {
        Ok(format!("({})", field_preds.join(" OR ")))
    }
}

fn term_to_sql(value: &Value, _ctx: &str) -> Result<String, BenoStreamError> {
    let m = value
        .as_object()
        .ok_or_else(|| bad_request("term: expected {\"field\": value}"))?;
    if m.len() != 1 {
        return Err(bad_request("term: expected exactly one field"));
    }
    let (field, v) = m
        .iter()
        .next()
        .ok_or_else(|| bad_request("term: expected exactly one field"))?;
    let field = valid_field(field)?;
    // Accept both bare `{"field": value}` and wrapped `{"field": {"value": ...}}`.
    let leaf = v
        .as_object()
        .filter(|o| o.len() == 1)
        .and_then(|o| o.get("value"))
        .unwrap_or(v);
    let lit = sql_literal(leaf)?;
    Ok(format!("{field} = {lit}"))
}

fn terms_to_sql(value: &Value, _ctx: &str) -> Result<String, BenoStreamError> {
    let m = value
        .as_object()
        .ok_or_else(|| bad_request("terms: expected {\"field\": [v1, v2, ...]}"))?;
    if m.len() != 1 {
        return Err(bad_request("terms: expected exactly one field"));
    }
    let (field, v) = m
        .iter()
        .next()
        .ok_or_else(|| bad_request("terms: expected exactly one field"))?;
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

fn range_to_sql(value: &Value, _ctx: &str) -> Result<String, BenoStreamError> {
    let m = value
        .as_object()
        .ok_or_else(|| bad_request("range: expected {\"field\": {\"gte\": ...}}"))?;
    if m.len() != 1 {
        return Err(bad_request("range: expected exactly one field"));
    }
    let (field, bounds) = m
        .iter()
        .next()
        .ok_or_else(|| bad_request("range: expected exactly one field"))?;
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

fn exists_to_sql(value: &Value, _ctx: &str) -> Result<String, BenoStreamError> {
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

fn prefix_to_sql(value: &Value, _ctx: &str) -> Result<String, BenoStreamError> {
    let m = value
        .as_object()
        .ok_or_else(|| bad_request("prefix: expected {\"field\": \"value\"}"))?;
    if m.len() != 1 {
        return Err(bad_request("prefix: expected exactly one field"));
    }
    let (field, v) = m
        .iter()
        .next()
        .ok_or_else(|| bad_request("prefix: expected exactly one field"))?;
    let field = valid_field(field)?;
    let prefix_str = v
        .as_str()
        .or_else(|| {
            v.as_object()
                .and_then(|o| o.get("value"))
                .and_then(Value::as_str)
        })
        .ok_or_else(|| bad_request("prefix: expected string value"))?;
    let safe_prefix = prefix_str
        .replace('\'', "''")
        .replace('%', "\\%")
        .replace('_', "\\_");
    Ok(format!("{field} LIKE '{safe_prefix}%'"))
}

fn wildcard_to_sql(value: &Value, _ctx: &str) -> Result<String, BenoStreamError> {
    let m = value
        .as_object()
        .ok_or_else(|| bad_request("wildcard: expected {\"field\": \"value\"}"))?;
    if m.len() != 1 {
        return Err(bad_request("wildcard: expected exactly one field"));
    }
    let (field, v) = m
        .iter()
        .next()
        .ok_or_else(|| bad_request("wildcard: expected exactly one field"))?;
    let field = valid_field(field)?;
    let pattern_str = v
        .as_str()
        .or_else(|| {
            v.as_object()
                .and_then(|o| o.get("value"))
                .and_then(Value::as_str)
        })
        .ok_or_else(|| bad_request("wildcard: expected string pattern"))?;
    let mut sql_pattern = String::new();
    for c in pattern_str.chars() {
        match c {
            '*' => sql_pattern.push('%'),
            '?' => sql_pattern.push('_'),
            '%' => sql_pattern.push_str("\\%"),
            '_' => sql_pattern.push_str("\\_"),
            '\'' => sql_pattern.push_str("''"),
            other => sql_pattern.push(other),
        }
    }
    Ok(format!("{field} LIKE '{sql_pattern}'"))
}

fn regexp_to_sql(value: &Value, _ctx: &str) -> Result<String, BenoStreamError> {
    let m = value
        .as_object()
        .ok_or_else(|| bad_request("regexp: expected {\"field\": \"pattern\"}"))?;
    if m.len() != 1 {
        return Err(bad_request("regexp: expected exactly one field"));
    }
    let (field, v) = m
        .iter()
        .next()
        .ok_or_else(|| bad_request("regexp: expected exactly one field"))?;
    let field = valid_field(field)?;
    let pattern_str = v
        .as_str()
        .or_else(|| {
            v.as_object()
                .and_then(|o| o.get("value"))
                .and_then(Value::as_str)
        })
        .ok_or_else(|| bad_request("regexp: expected string pattern"))?;
    let safe_pattern = pattern_str.replace('\'', "''");
    Ok(format!("regexp_like({field}, '{safe_pattern}')"))
}

fn fuzzy_to_sql(value: &Value, _ctx: &str) -> Result<String, BenoStreamError> {
    let m = value.as_object().ok_or_else(|| {
        bad_request("fuzzy: expected {\"field\": \"term\"} or {\"field\": {\"value\": \"term\"}}")
    })?;
    if m.len() != 1 {
        return Err(bad_request("fuzzy: expected exactly one field"));
    }
    let (field, v) = m
        .iter()
        .next()
        .ok_or_else(|| bad_request("fuzzy: expected exactly one field"))?;
    let field = valid_field(field)?;

    let (term, fuzziness, prefix_len) = match v {
        Value::String(s) => (s.as_str(), 2usize, 0usize),
        Value::Object(opts) => {
            let term = opts
                .get("value")
                .and_then(Value::as_str)
                .ok_or_else(|| bad_request("fuzzy: missing 'value' string"))?;
            let prefix_len = opts
                .get("prefix_length")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize;
            let fuzziness = match opts.get("fuzziness") {
                Some(Value::Number(n)) => n.as_u64().unwrap_or(2) as usize,
                Some(Value::String(s)) => {
                    if s.eq_ignore_ascii_case("AUTO") {
                        if term.len() <= 2 {
                            0
                        } else if term.len() <= 5 {
                            1
                        } else {
                            2
                        }
                    } else {
                        s.parse::<usize>().unwrap_or(2)
                    }
                }
                _ => {
                    if term.len() <= 2 {
                        0
                    } else if term.len() <= 5 {
                        1
                    } else {
                        2
                    }
                }
            };
            (term, fuzziness, prefix_len)
        }
        _ => return Err(bad_request("fuzzy: expected string or object with 'value'")),
    };

    let safe_term = term.replace('\'', "''");
    let mut parts = Vec::new();
    if prefix_len > 0 && prefix_len <= term.len() {
        let prefix = &term[..prefix_len];
        let safe_prefix = prefix
            .replace('\'', "''")
            .replace('%', "\\%")
            .replace('_', "\\_");
        parts.push(format!("{field} LIKE '{safe_prefix}%'"));
    }

    if fuzziness == 0 {
        parts.push(format!("{field} = '{safe_term}'"));
    } else {
        parts.push(format!(
            "levenshtein({field}, '{safe_term}') <= {fuzziness}"
        ));
    }

    Ok(parts.join(" AND "))
}

fn ids_to_sql(value: &Value, _ctx: &str) -> Result<String, BenoStreamError> {
    let values_arr = match value {
        Value::Object(m) => m.get("values").and_then(Value::as_array),
        Value::Array(a) => Some(a),
        _ => None,
    }
    .ok_or_else(|| bad_request("ids: expected {\"values\": [\"id1\", \"id2\"]}"))?;

    if values_arr.is_empty() {
        return Ok("false".to_string());
    }
    let mut id_lits = Vec::new();
    for v in values_arr {
        let id_str = match v {
            Value::String(s) => s.as_str(),
            Value::Number(n) => &n.to_string(),
            _ => return Err(bad_request("ids: values must be strings or numbers")),
        };
        id_lits.push(format!("'{}'", id_str.replace('\'', "''")));
    }
    Ok(format!("{ID_COLUMN} IN ({})", id_lits.join(", ")))
}

fn bool_to_sql(value: &Value, ctx: &str) -> Result<String, BenoStreamError> {
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
    if let Some(arr) = m.get("should").and_then(Value::as_array) {
        if !arr.is_empty() {
            let mut should_parts = Vec::new();
            for clause in arr {
                should_parts.push(clause_to_sql(clause, ctx)?);
            }
            if !should_parts.is_empty() {
                parts.push(format!("({})", should_parts.join(" OR ")));
            }
        }
    }
    if parts.is_empty() {
        return Ok("true".to_string());
    }
    Ok(parts.join(" AND "))
}

fn sql_literal(v: &Value) -> Result<String, BenoStreamError> {
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

fn json_val_to_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

async fn resolve_nested_array_query(
    state: &AppState,
    target_index: &str,
    path: &str,
    inner_query: &Value,
) -> Result<Option<Value>, BenoStreamError> {
    let resolved_index = state.resolve_alias(target_index).await;
    if path.is_empty() || !state.index_exists(&resolved_index).await {
        return Ok(None);
    }

    let table = state.open_or_create(&resolved_index, &None).await?;
    let schema = table.arrow_schema();
    let Ok(field) = schema.field_with_name(path) else {
        return Ok(None);
    };

    let inner_struct_fields = match field.data_type() {
        DataType::List(item) | DataType::LargeList(item) => match item.data_type() {
            DataType::Struct(subfields) => subfields.clone(),
            _ => return Ok(None),
        },
        _ => return Ok(None),
    };

    let unqualified = unqualify_nested_paths(inner_query, path);
    let resolved_inner = resolve_relation_queries(state, target_index, &unqualified).await?;
    let sql = clause_to_sql(&resolved_inner, "nested")?;
    let struct_schema = Arc::new(arrow::datatypes::Schema::new(inner_struct_fields.to_vec()));
    let filter_expr = FilterExpr::parse_sql(&sql, struct_schema.clone())
        .await
        .map_err(|e| BenoStreamError::SchemaIncompatible {
            reason: format!("nested query parsing failed: {e}"),
        })?;
    let planner = QueryPlanner::new();

    let batches = table
        .read_async(None, None, Some(&[ID_COLUMN, path]))
        .await
        .map_err(|e| {
            BenoStreamError::internal(format!("failed to read table for nested query: {e}"))
        })?;

    let mut matching_ids = Vec::new();
    for batch in &batches {
        let id_col = batch
            .column_by_name(ID_COLUMN)
            .and_then(|c| c.as_any().downcast_ref::<StringArray>());
        let list_col = batch
            .column_by_name(path)
            .and_then(|c| c.as_any().downcast_ref::<ListArray>());
        if let Some(list_col) = list_col {
            for row_idx in 0..batch.num_rows() {
                if list_col.is_null(row_idx) || list_col.value_length(row_idx) == 0 {
                    continue;
                }
                let sub_arr = list_col.value(row_idx);
                if let Some(struct_arr) = sub_arr.as_any().downcast_ref::<StructArray>() {
                    if struct_arr.len() == 0 {
                        continue;
                    }
                    if let Ok(mini_batch) =
                        RecordBatch::try_new(struct_schema.clone(), struct_arr.columns().to_vec())
                    {
                        if let Ok(mask) = planner.evaluate_expr(&mini_batch, &filter_expr) {
                            if mask.true_count() > 0 {
                                let doc_id = if let Some(id_col) = id_col {
                                    id_col.value(row_idx).to_string()
                                } else {
                                    format!("0:{row_idx}")
                                };
                                matching_ids.push(Value::String(doc_id));
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(Some(if matching_ids.is_empty() {
        serde_json::json!({
            "term": { "_id": "__bsdb_no_match__" }
        })
    } else {
        serde_json::json!({
            "terms": { "_id": matching_ids }
        })
    }))
}

pub(crate) fn resolve_relation_queries<'a>(
    state: &'a AppState,
    target_index: &'a str,
    val: &'a Value,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, BenoStreamError>> + Send + 'a>>
{
    Box::pin(resolve_relation_queries_inner(state, target_index, val))
}

async fn resolve_relation_queries_inner(
    state: &AppState,
    target_index: &str,
    val: &Value,
) -> Result<Value, BenoStreamError> {
    match val {
        Value::Object(map) => {
            if let Some(spec) = map.get("query_string") {
                let (query, opts) =
                    crate::handlers::query_string::QueryStringOptions::from_value(spec)?;
                let expanded =
                    crate::handlers::query_string::parse_query_string(&query, &opts, false)?;
                return resolve_relation_queries(state, target_index, &expanded).await;
            }

            if let Some(spec) = map.get("simple_query_string") {
                let (query, opts) =
                    crate::handlers::query_string::QueryStringOptions::from_value(spec)?;
                let expanded =
                    crate::handlers::query_string::parse_query_string(&query, &opts, true)?;
                return resolve_relation_queries(state, target_index, &expanded).await;
            }

            if let Some(spec) = map.get("nested") {
                let spec_obj =
                    spec.as_object()
                        .ok_or_else(|| BenoStreamError::SchemaIncompatible {
                            reason: "nested: expected an object with 'path' and 'query'".into(),
                        })?;
                let path = spec_obj.get("path").and_then(Value::as_str).unwrap_or("");
                let inner_query =
                    spec_obj
                        .get("query")
                        .ok_or_else(|| BenoStreamError::SchemaIncompatible {
                            reason: "nested: missing 'query'".into(),
                        })?;

                if let Some(resolved) =
                    resolve_nested_array_query(state, target_index, path, inner_query).await?
                {
                    return Ok(resolved);
                }
            }

            if let Some(spec) = map.get("has_child") {
                let spec_obj =
                    spec.as_object()
                        .ok_or_else(|| BenoStreamError::SchemaIncompatible {
                            reason: "has_child: expected an object".into(),
                        })?;
                let child_type = spec_obj
                    .get("type")
                    .and_then(Value::as_str)
                    .ok_or_else(|| BenoStreamError::SchemaIncompatible {
                        reason: "has_child: 'type' (string) is required".into(),
                    })?;
                let subquery =
                    spec_obj
                        .get("query")
                        .ok_or_else(|| BenoStreamError::SchemaIncompatible {
                            reason: "has_child: 'query' (object) is required".into(),
                        })?;

                let min_children = spec_obj
                    .get("min_children")
                    .and_then(Value::as_u64)
                    .unwrap_or(1) as usize;
                let max_children = spec_obj
                    .get("max_children")
                    .and_then(Value::as_u64)
                    .map(|m| m as usize);

                let edge_index = spec_obj.get("edge_index").and_then(Value::as_str);
                let parent_field_opt = spec_obj.get("parent_field").and_then(Value::as_str);

                let resolved_subquery =
                    resolve_relation_queries(state, child_type, subquery).await?;
                let child_search_req = serde_json::json!({
                    "query": resolved_subquery,
                    "size": 10000,
                });

                let child_resp = search_core(state, child_type, &child_search_req).await?;

                let mut parent_counts: HashMap<String, usize> = HashMap::new();

                if let Some(edge_idx) = edge_index {
                    if state.index_exists(edge_idx).await {
                        let edge_table = state.open_or_create(edge_idx, &None).await?;
                        let child_seeds: Vec<u64> = child_resp
                            .hits
                            .hits
                            .iter()
                            .filter_map(|h| h.id.parse::<u64>().ok())
                            .collect();

                        if !child_seeds.is_empty() {
                            let options = GraphNeighborhoodOptions {
                                seeds: child_seeds,
                                hops: 1,
                                directed: false,
                                ..Default::default()
                            };
                            if let Ok(parents) = edge_table.graph_neighborhood(&options).await {
                                for p in parents {
                                    *parent_counts.entry(p.to_string()).or_insert(0) += 1;
                                }
                            }
                        }
                    }
                } else {
                    for hit in &child_resp.hits.hits {
                        let pid = if let Some(pf) = parent_field_opt {
                            hit.source.get(pf).and_then(json_val_to_string)
                        } else {
                            hit.source
                                .get("parent_id")
                                .or_else(|| hit.source.get("_parent"))
                                .or_else(|| hit.source.get("parent"))
                                .or_else(|| hit.source.get(format!("{target_index}_id").as_str()))
                                .or_else(|| {
                                    if let Some(singular) = target_index.strip_suffix('s') {
                                        hit.source.get(format!("{singular}_id").as_str())
                                    } else {
                                        None
                                    }
                                })
                                .and_then(json_val_to_string)
                        };
                        if let Some(pid_str) = pid {
                            *parent_counts.entry(pid_str).or_insert(0) += 1;
                        }
                    }
                }

                let matching_parents: Vec<Value> = parent_counts
                    .into_iter()
                    .filter(|(_, count)| {
                        *count >= min_children && max_children.is_none_or(|max| *count <= max)
                    })
                    .map(|(pid, _)| Value::String(pid))
                    .collect();

                return Ok(if matching_parents.is_empty() {
                    serde_json::json!({
                        "term": { "_id": "__bsdb_no_match__" }
                    })
                } else {
                    serde_json::json!({
                        "terms": { "_id": matching_parents }
                    })
                });
            }

            if let Some(spec) = map.get("has_parent") {
                let spec_obj =
                    spec.as_object()
                        .ok_or_else(|| BenoStreamError::SchemaIncompatible {
                            reason: "has_parent: expected an object".into(),
                        })?;
                let parent_type = spec_obj
                    .get("parent_type")
                    .and_then(Value::as_str)
                    .ok_or_else(|| BenoStreamError::SchemaIncompatible {
                        reason: "has_parent: 'parent_type' (string) is required".into(),
                    })?;
                let subquery =
                    spec_obj
                        .get("query")
                        .ok_or_else(|| BenoStreamError::SchemaIncompatible {
                            reason: "has_parent: 'query' (object) is required".into(),
                        })?;

                let edge_index = spec_obj.get("edge_index").and_then(Value::as_str);

                let resolved_subquery =
                    resolve_relation_queries(state, parent_type, subquery).await?;
                let parent_search_req = serde_json::json!({
                    "query": resolved_subquery,
                    "size": 10000,
                });

                let parent_resp = search_core(state, parent_type, &parent_search_req).await?;
                let parent_ids: Vec<String> =
                    parent_resp.hits.hits.iter().map(|h| h.id.clone()).collect();

                if parent_ids.is_empty() {
                    return Ok(serde_json::json!({
                        "term": { "_id": "__bsdb_no_match__" }
                    }));
                }

                if let Some(edge_idx) = edge_index {
                    if state.index_exists(edge_idx).await {
                        let edge_table = state.open_or_create(edge_idx, &None).await?;
                        let parent_seeds: Vec<u64> = parent_ids
                            .iter()
                            .filter_map(|s| s.parse::<u64>().ok())
                            .collect();
                        if !parent_seeds.is_empty() {
                            let options = GraphNeighborhoodOptions {
                                seeds: parent_seeds,
                                hops: 1,
                                directed: true,
                                ..Default::default()
                            };
                            let children = edge_table
                                .graph_neighborhood(&options)
                                .await
                                .unwrap_or_default();
                            let child_ids_json: Vec<Value> = children
                                .into_iter()
                                .map(|c| Value::String(c.to_string()))
                                .collect();
                            return Ok(if child_ids_json.is_empty() {
                                serde_json::json!({
                                    "term": { "_id": "__bsdb_no_match__" }
                                })
                            } else {
                                serde_json::json!({
                                    "terms": { "_id": child_ids_json }
                                })
                            });
                        }
                    }
                }

                let parent_field =
                    if let Some(pf) = spec_obj.get("parent_field").and_then(Value::as_str) {
                        pf.to_string()
                    } else if state.index_exists(target_index).await {
                        if let Ok(target_table) = state.open_or_create(target_index, &None).await {
                            let schema = target_table.arrow_schema();
                            let singular = parent_type.strip_suffix('s').unwrap_or(parent_type);
                            let candidate_fields = [
                                "parent_id",
                                "_parent",
                                "parent",
                                &format!("{parent_type}_id"),
                                &format!("{singular}_id"),
                            ];
                            candidate_fields
                                .iter()
                                .find(|&&f| schema.field_with_name(f).is_ok())
                                .unwrap_or(&"parent_id")
                                .to_string()
                        } else {
                            "parent_id".to_string()
                        }
                    } else {
                        "parent_id".to_string()
                    };

                let parent_ids_json: Vec<Value> =
                    parent_ids.into_iter().map(Value::String).collect();
                return Ok(serde_json::json!({
                    "terms": { parent_field: parent_ids_json }
                }));
            }

            let mut out = Map::new();
            for (k, v) in map {
                out.insert(
                    k.clone(),
                    resolve_relation_queries(state, target_index, v).await?,
                );
            }
            Ok(Value::Object(out))
        }
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                out.push(resolve_relation_queries(state, target_index, v).await?);
            }
            Ok(Value::Array(out))
        }
        other => Ok(other.clone()),
    }
}

/// Core search dispatch. Returns the full ES-shaped response so tests can
/// assert on it without going through the axum layer.
pub async fn search_core(
    state: &AppState,
    index: &str,
    body: &Value,
) -> Result<SearchResponse, BenoStreamError> {
    let start = Instant::now();

    if !state.index_exists(index).await {
        return Err(BenoStreamError::TableNotFound {
            namespace: String::new(),
            name: index.to_string(),
        });
    }

    let resolved_body = resolve_relation_queries(state, index, body).await?;
    let req = parse_request(&resolved_body)?;
    let table = state.open_or_create(index, &None).await?;

    // RRF fusion constant: request-level `rrf_k` wins, then the
    // `BENOSEARCH_RRF_K` env var, then the core default (60).
    let rrf_k = req.rrf_k.or_else(|| {
        std::env::var("BENOSEARCH_RRF_K")
            .ok()
            .and_then(|v| v.parse::<f32>().ok())
            .filter(|k| *k > 0.0)
    });

    let (batches, kind, knn_k) = match (&req.keyword, &req.vector, &req.phrase) {
        (Some(kp), Some(vp), _) => {
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
        (Some(kp), None, _) => {
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
        (None, None, Some(pp)) => {
            let scored = table
                .execute_phrase_search_as_scored(&pp.column, &pp.phrase, pp.slop, None)
                .await
                .map_err(translate_search_error)?;
            let batches = table
                .fetch_results_by_id(scored, None)
                .await
                .map_err(translate_search_error)?;
            (batches, ScoreKind::Relevance, None)
        }
        (None, Some(vp), _) => {
            if req.filter.is_some() {
                // Core pre-filters inside the scan and appends a distance
                // column. Note: the core's smart hybrid trigger may rewrite
                // this scan into a hybrid path when the filter column has its
                // own BM25 index; the result shape (rows + trailing distance
                // column) is preserved, which is acceptable for v1.
                let batches = table
                    .read_async(req.filter.as_deref(), Some(vec![vp.clone()]), None)
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
        (None, None, None) => {
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
                    .map_err(|e| BenoStreamError::SchemaIncompatible {
                        reason: format!("invalid filter: {e}"),
                    })?;
                let planner = QueryPlanner::new();
                batches
                    .into_iter()
                    .map(|b| planner.filter_expr(&b, &expr))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| BenoStreamError::SchemaIncompatible {
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

    // Custom sort criteria overrides relevance/distance sort order.
    if let Some(sort_clauses) = &req.sort {
        if !sort_clauses.is_empty() {
            hits.sort_by(|a, b| {
                for sc in sort_clauses {
                    let cmp = if sc.field == "_score" {
                        let sa = a.score;
                        let sb = b.score;
                        if sc.descending {
                            sb.partial_cmp(&sa).unwrap_or(Ordering::Equal)
                        } else {
                            sa.partial_cmp(&sb).unwrap_or(Ordering::Equal)
                        }
                    } else if sc.field == "_id" {
                        if sc.descending {
                            b.id.cmp(&a.id)
                        } else {
                            a.id.cmp(&b.id)
                        }
                    } else {
                        let va = get_json_path(&a.source, &sc.field);
                        let vb = get_json_path(&b.source, &sc.field);
                        let c = compare_json_values(va, vb);
                        if sc.descending {
                            c.reverse()
                        } else {
                            c
                        }
                    };
                    if cmp != Ordering::Equal {
                        return cmp;
                    }
                }
                a.id.cmp(&b.id)
            });
        }
    } else {
        // Default sort by relevance / distance / id
        match kind {
            ScoreKind::Relevance => {
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
                hits.sort_by_key(|a| a.id.clone());
            }
        }
    }

    if let (Some(sort_clauses), Some(cursor)) = (&req.sort, &req.search_after) {
        hits.retain(|h| {
            let hit_vals = extract_sort_values(h, sort_clauses);
            is_hit_after_cursor(&hit_vals, cursor, sort_clauses)
        });
    }

    // Underlying BM25/HNSW candidate lists are capped per segment, so this
    // total is a best-effort approximation reported with `relation: "eq"`.
    let total = hits.len() as u64;

    if let Some(k) = knn_k {
        hits.truncate(k);
    }

    let len = hits.len();
    let from = if req.search_after.is_some() {
        0
    } else {
        req.from.min(len)
    };
    let end = (from + req.size).min(len);
    let page: Vec<SearchHit> = hits[from..end]
        .iter()
        .map(|h| {
            let sort = req.sort.as_ref().map(|scs| extract_sort_values(h, scs));
            let highlight = req
                .highlight
                .as_ref()
                .and_then(|hl| extract_highlights(h, hl, &req));
            SearchHit {
                index: index.to_string(),
                id: h.id.clone(),
                score: Some(final_score(h, kind)),
                source: h.source.clone(),
                sort,
                highlight,
            }
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

    let aggregations = match &req.aggs {
        Some(a) => Some(
            crate::handlers::aggs::compute_aggregations(&table, req.filter.as_deref(), a).await?,
        ),
        None => None,
    };

    let scroll_id = if let Some(ref scroll_duration) = req.scroll {
        let next_search_after = page.last().and_then(|h| h.sort.clone());
        let ttl_secs = parse_scroll_ttl_secs(scroll_duration);
        let now_secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let mut next_body = body.clone();
        if let Some(sa) = next_search_after {
            if let Some(b_obj) = next_body.as_object_mut() {
                b_obj.insert("search_after".to_string(), Value::Array(sa));
            }
        }
        let token = ScrollToken {
            index: index.to_string(),
            body: next_body,
            scroll: scroll_duration.clone(),
            expires_at: now_secs + ttl_secs,
        };
        Some(encode_scroll_token(&token)?)
    } else {
        None
    };

    Ok(SearchResponse {
        scroll_id,
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
        aggregations,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrollToken {
    pub index: String,
    pub body: Value,
    pub scroll: String,
    pub expires_at: u64,
}

fn parse_scroll_ttl_secs(s: &str) -> u64 {
    let s = s.trim();
    if let Some(stripped) = s.strip_suffix('m') {
        stripped.parse::<u64>().unwrap_or(1) * 60
    } else if let Some(stripped) = s.strip_suffix('s') {
        stripped.parse::<u64>().unwrap_or(60)
    } else if let Some(stripped) = s.strip_suffix('h') {
        stripped.parse::<u64>().unwrap_or(1) * 3600
    } else if let Some(stripped) = s.strip_suffix('d') {
        stripped.parse::<u64>().unwrap_or(1) * 86400
    } else {
        s.parse::<u64>().unwrap_or(60)
    }
}

pub fn encode_scroll_token(token: &ScrollToken) -> Result<String, BenoStreamError> {
    let json_bytes =
        serde_json::to_vec(token).map_err(|e| BenoStreamError::internal(e.to_string()))?;
    Ok(URL_SAFE_NO_PAD.encode(json_bytes))
}

pub fn decode_scroll_token(s: &str) -> Result<ScrollToken, BenoStreamError> {
    let s = s.trim();
    let json_bytes = URL_SAFE_NO_PAD
        .decode(s)
        .or_else(|_| URL_SAFE.decode(s))
        .or_else(|_| STANDARD.decode(s))
        .map_err(|_| BenoStreamError::SchemaIncompatible {
            reason: "search_context_missing_exception: Invalid scroll_id base64".into(),
        })?;
    serde_json::from_slice(&json_bytes).map_err(|_| BenoStreamError::SchemaIncompatible {
        reason: "search_context_missing_exception: Malformed scroll token payload".into(),
    })
}

pub async fn execute_scroll(
    state: &AppState,
    scroll_id: &str,
    scroll_override: Option<&str>,
) -> Result<SearchResponse, BenoStreamError> {
    if scroll_id.is_empty() {
        return Err(BenoStreamError::SchemaIncompatible {
            reason: "scroll_id is required".into(),
        });
    }

    let token = decode_scroll_token(scroll_id)?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    if now > token.expires_at {
        return Err(BenoStreamError::SchemaIncompatible {
            reason: "search_context_missing_exception: Cannot execute scroll, context has expired"
                .into(),
        });
    }

    let mut body = token.body.clone();
    if let Some(s) = scroll_override {
        if let Some(obj) = body.as_object_mut() {
            obj.insert("scroll".to_string(), Value::String(s.to_string()));
        }
    }

    search_core(state, &token.index, &body).await
}

/// OR-merge per-field BM25 results for a multi-field `match`, keeping the
/// best (highest) score per document and returning a single score-desc list.
async fn merge_keyword_results(
    table: &Table,
    params: &[KeywordSearchParams],
) -> Result<Vec<ScoredResult>, BenoStreamError> {
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

pub(crate) fn value_to_json(col: &dyn Array, i: usize) -> Value {
    if col.is_null(i) {
        return Value::Null;
    }

    // Arrow guarantees that an array's concrete type matches its `data_type()`,
    // so these downcasts cannot fail in practice. The generic helper keeps the
    // function total: a violated invariant degrades to `null` instead of
    // panicking the request handler.
    fn downcast<T: 'static>(col: &dyn Array) -> Option<&T> {
        col.as_any().downcast_ref::<T>()
    }

    match col.data_type() {
        DataType::Utf8 => downcast::<StringArray>(col)
            .map(|a| Value::String(a.value(i).to_string()))
            .unwrap_or(Value::Null),
        DataType::LargeUtf8 => downcast::<LargeStringArray>(col)
            .map(|a| Value::String(a.value(i).to_string()))
            .unwrap_or(Value::Null),
        DataType::Boolean => downcast::<BooleanArray>(col)
            .map(|a| Value::Bool(a.value(i)))
            .unwrap_or(Value::Null),
        DataType::Int8 => downcast::<Int8Array>(col)
            .map(|a| Value::from(a.value(i) as i64))
            .unwrap_or(Value::Null),
        DataType::Int16 => downcast::<Int16Array>(col)
            .map(|a| Value::from(a.value(i) as i64))
            .unwrap_or(Value::Null),
        DataType::Int32 => downcast::<Int32Array>(col)
            .map(|a| Value::from(a.value(i) as i64))
            .unwrap_or(Value::Null),
        DataType::Int64 => downcast::<Int64Array>(col)
            .map(|a| Value::from(a.value(i)))
            .unwrap_or(Value::Null),
        DataType::UInt8 => downcast::<UInt8Array>(col)
            .map(|a| Value::from(a.value(i) as u64))
            .unwrap_or(Value::Null),
        DataType::UInt16 => downcast::<UInt16Array>(col)
            .map(|a| Value::from(a.value(i) as u64))
            .unwrap_or(Value::Null),
        DataType::UInt32 => downcast::<UInt32Array>(col)
            .map(|a| Value::from(a.value(i) as u64))
            .unwrap_or(Value::Null),
        DataType::UInt64 => downcast::<UInt64Array>(col)
            .map(|a| Value::from(a.value(i)))
            .unwrap_or(Value::Null),
        DataType::Float32 => downcast::<Float32Array>(col)
            .map(|a| Value::from(a.value(i) as f64))
            .unwrap_or(Value::Null),
        DataType::Float64 => downcast::<Float64Array>(col)
            .map(|a| Value::from(a.value(i)))
            .unwrap_or(Value::Null),
        DataType::Date32 => downcast::<Date32Array>(col)
            .map(|a| {
                let days = a.value(i);
                NaiveDate::from_num_days_from_ce_opt(719163 + days)
                    .map(|d| Value::String(d.to_string()))
                    .unwrap_or(Value::Null)
            })
            .unwrap_or(Value::Null),
        DataType::Date64 => downcast::<Date64Array>(col)
            .map(|a| {
                DateTime::from_timestamp_millis(a.value(i))
                    .map(|d| d.to_rfc3339_opts(SecondsFormat::Millis, true))
                    .map(Value::String)
                    .unwrap_or(Value::Null)
            })
            .unwrap_or(Value::Null),
        DataType::Timestamp(arrow::datatypes::TimeUnit::Microsecond, _) => {
            downcast::<TimestampMicrosecondArray>(col)
                .map(|a| {
                    DateTime::from_timestamp_micros(a.value(i))
                        .map(|d| d.to_rfc3339_opts(SecondsFormat::Millis, true))
                        .map(Value::String)
                        .unwrap_or(Value::Null)
                })
                .unwrap_or(Value::Null)
        }
        DataType::FixedSizeList(_, _) => downcast::<FixedSizeListArray>(col)
            .map(|a| {
                let dim = a.value_length() as usize;
                a.values()
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .map(|flat| {
                        flat.values()[i * dim..(i + 1) * dim]
                            .iter()
                            .copied()
                            .map(|x| Value::from(x as f64))
                            .collect::<Vec<_>>()
                    })
                    .map(Value::Array)
                    .unwrap_or(Value::Null)
            })
            .unwrap_or(Value::Null),
        DataType::List(_) | DataType::LargeList(_) => downcast::<ListArray>(col)
            .map(|a| {
                let off = a.value_offsets();
                let start = off[i] as usize;
                let end = off[i + 1] as usize;
                let values = a.values();
                let mut items = Vec::with_capacity(end - start);
                for idx in start..end {
                    items.push(value_to_json(values.as_ref(), idx));
                }
                Value::Array(items)
            })
            .unwrap_or(Value::Null),
        DataType::Struct(_) => downcast::<StructArray>(col)
            .map(|a| {
                let mut obj = Map::new();
                for (j, f) in a.fields().iter().enumerate() {
                    obj.insert(f.name().clone(), value_to_json(&a.column(j), i));
                }
                Value::Object(obj)
            })
            .unwrap_or(Value::Null),
        dt => {
            tracing::debug!(?dt, "unmapped arrow type in _source");
            Value::Null
        }
    }
}

#[cfg(test)]
mod tests {
    use benostreamdb::BenoStreamError;
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
        let _ = tracing_subscriber::fmt()
            .with_test_writer()
            .with_max_level(tracing::Level::DEBUG)
            .try_init();
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("docs")).unwrap();

        crate::handlers::indices::create_index_core(&state, "docs", None)
            .await
            .unwrap();
        crate::handlers::mapping::put_mapping_core(
            &state,
            "docs",
            &json!({"properties": {"body": {"type": "text"}}, "indexes": {"body": "bm25"}}),
        )
        .await
        .unwrap();

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

        crate::handlers::indices::create_index_core(&state, "f", None)
            .await
            .unwrap();
        crate::handlers::mapping::put_mapping_core(
            &state,
            "f",
            &json!({"properties": {"body": {"type": "text"}}, "indexes": {"body": "bm25"}}),
        )
        .await
        .unwrap();

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
        assert!(matches!(err, BenoStreamError::TableNotFound { .. }));
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
            json!({"filter": {"unsupported_clause": {"body": "a"}}}),
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
                matches!(err, BenoStreamError::SchemaIncompatible { .. }),
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
                matches!(err, BenoStreamError::SchemaIncompatible { .. }),
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
            BenoStreamError::ColumnNotFound { column, .. } => assert_eq!(column, "nope"),
            other => panic!("expected ColumnNotFound, got {other:?}"),
        }
        let es: EsError = err.into();
        assert_eq!(es.status, 400);
    }

    #[tokio::test]
    async fn aggregations_terms_stats_histogram_range() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("agg")).unwrap();

        index_docs(
            &state,
            "agg",
            &[
                json!({"category": "a", "age": 10}),
                json!({"category": "a", "age": 20}),
                json!({"category": "b", "age": 30}),
                json!({"category": "b", "age": 40}),
            ],
        )
        .await;

        let resp = search_core(
            &state,
            "agg",
            &json!({
                "query": {"match_all": {}},
                "size": 0,
                "aggs": {
                    "by_cat": {"terms": {"field": "category"}},
                    "age_stats": {"stats": {"field": "age"}},
                    "age_hist": {"histogram": {"field": "age", "interval": 20}},
                    "age_ranges": {"range": {"field": "age", "ranges": [{"to": 25}, {"from": 25}]}}
                }
            }),
        )
        .await
        .unwrap();

        let aggs = resp.aggregations.expect("aggregations present");
        let buckets = aggs["by_cat"]["buckets"].as_array().unwrap();
        assert_eq!(buckets.len(), 2);
        // Both categories have doc_count 2, so the tie order is arbitrary.
        let bucket_a = buckets.iter().find(|b| b["key"] == "a").expect("bucket a");
        assert_eq!(bucket_a["doc_count"], 2);
        assert_eq!(aggs["age_stats"]["count"], 4.0);
        assert_eq!(aggs["age_stats"]["sum"], 100.0);
        assert_eq!(aggs["age_stats"]["avg"], 25.0);
        assert_eq!(aggs["age_hist"]["buckets"].as_array().unwrap().len(), 3);
        let ranges = aggs["age_ranges"]["buckets"].as_array().unwrap();
        assert_eq!(ranges[0]["doc_count"], 2);
        assert_eq!(ranges[1]["doc_count"], 2);
    }

    #[tokio::test]
    async fn aggregations_percentiles() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("pct")).unwrap();

        index_docs(
            &state,
            "pct",
            &[
                json!({"latency": 10.0}),
                json!({"latency": 20.0}),
                json!({"latency": 30.0}),
                json!({"latency": 40.0}),
                json!({"latency": 50.0}),
                json!({"latency": 60.0}),
                json!({"latency": 70.0}),
                json!({"latency": 80.0}),
                json!({"latency": 90.0}),
                json!({"latency": 100.0}),
            ],
        )
        .await;

        let resp = search_core(
            &state,
            "pct",
            &json!({
                "size": 0,
                "aggs": {
                    "load_pct": {
                        "percentiles": {
                            "field": "latency",
                            "percents": [25.0, 50.0, 75.0, 99.0]
                        }
                    },
                    "default_pct": {
                        "percentiles": {
                            "field": "latency"
                        }
                    },
                    "unkeyed_pct": {
                        "percentiles": {
                            "field": "latency",
                            "percents": [50.0],
                            "keyed": false
                        }
                    }
                }
            }),
        )
        .await
        .unwrap();

        let aggs = resp.aggregations.expect("aggregations present");
        let load_values = &aggs["load_pct"]["values"];
        assert!(load_values["25.0"].as_f64().is_some());
        assert!(load_values["50.0"].as_f64().is_some());
        assert!(load_values["75.0"].as_f64().is_some());
        assert!(load_values["99.0"].as_f64().is_some());

        let default_values = &aggs["default_pct"]["values"];
        assert!(default_values["1.0"].as_f64().is_some());
        assert!(default_values["50.0"].as_f64().is_some());
        assert!(default_values["99.0"].as_f64().is_some());

        let unkeyed_values = aggs["unkeyed_pct"]["values"].as_array().expect("array");
        assert_eq!(unkeyed_values.len(), 1);
        assert_eq!(unkeyed_values[0]["key"], 50.0);
        assert!(unkeyed_values[0]["value"].as_f64().is_some());
    }

    #[tokio::test]
    async fn aggregations_composite() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("comp_idx")).unwrap();

        index_docs(
            &state,
            "comp_idx",
            &[
                json!({"category": "apparel", "product": "shirt", "price": 25.0}),
                json!({"category": "apparel", "product": "shoes", "price": 50.0}),
                json!({"category": "apparel", "product": "shoes", "price": 60.0}),
                json!({"category": "electronics", "product": "laptop", "price": 1500.0}),
                json!({"category": "electronics", "product": "phone", "price": 800.0}),
                json!({"category": "electronics", "product": "tv", "price": 1200.0}),
            ],
        )
        .await;

        // 1. Basic pagination across composite buckets (size: 2)
        let page1 = search_core(
            &state,
            "comp_idx",
            &json!({
                "size": 0,
                "aggs": {
                    "my_comp": {
                        "composite": {
                            "size": 2,
                            "sources": [
                                { "cat": { "terms": { "field": "category", "order": "asc" } } },
                                { "prod": { "terms": { "field": "product", "order": "asc" } } }
                            ]
                        },
                        "aggs": {
                            "avg_price": { "avg": { "field": "price" } }
                        }
                    }
                }
            }),
        )
        .await
        .unwrap();

        let aggs1 = page1.aggregations.expect("aggs present");
        let buckets1 = aggs1["my_comp"]["buckets"].as_array().expect("buckets");
        assert_eq!(buckets1.len(), 2);
        assert_eq!(buckets1[0]["key"]["cat"], "apparel");
        assert_eq!(buckets1[0]["key"]["prod"], "shirt");
        assert_eq!(buckets1[0]["doc_count"], 1);
        assert_eq!(buckets1[0]["aggs"]["avg_price"]["value"], 25.0);

        assert_eq!(buckets1[1]["key"]["cat"], "apparel");
        assert_eq!(buckets1[1]["key"]["prod"], "shoes");
        assert_eq!(buckets1[1]["doc_count"], 2);
        assert_eq!(buckets1[1]["aggs"]["avg_price"]["value"], 55.0);

        let after1 = aggs1["my_comp"]["after_key"].clone();
        assert_eq!(after1, buckets1[1]["key"]);

        // 2. Fetch page 2 using after_key
        let page2 = search_core(
            &state,
            "comp_idx",
            &json!({
                "size": 0,
                "aggs": {
                    "my_comp": {
                        "composite": {
                            "size": 2,
                            "sources": [
                                { "cat": { "terms": { "field": "category", "order": "asc" } } },
                                { "prod": { "terms": { "field": "product", "order": "asc" } } }
                            ],
                            "after": after1
                        }
                    }
                }
            }),
        )
        .await
        .unwrap();

        let aggs2 = page2.aggregations.expect("aggs present");
        let buckets2 = aggs2["my_comp"]["buckets"].as_array().expect("buckets");
        assert_eq!(buckets2.len(), 2);
        assert_eq!(buckets2[0]["key"]["cat"], "electronics");
        assert_eq!(buckets2[0]["key"]["prod"], "laptop");

        assert_eq!(buckets2[1]["key"]["cat"], "electronics");
        assert_eq!(buckets2[1]["key"]["prod"], "phone");

        let after2 = aggs2["my_comp"]["after_key"].clone();
        assert_eq!(after2, buckets2[1]["key"]);

        // 3. Fetch page 3
        let page3 = search_core(
            &state,
            "comp_idx",
            &json!({
                "size": 0,
                "aggs": {
                    "my_comp": {
                        "composite": {
                            "size": 2,
                            "sources": [
                                { "cat": { "terms": { "field": "category", "order": "asc" } } },
                                { "prod": { "terms": { "field": "product", "order": "asc" } } }
                            ],
                            "after": after2
                        }
                    }
                }
            }),
        )
        .await
        .unwrap();

        let aggs3 = page3.aggregations.expect("aggs present");
        let buckets3 = aggs3["my_comp"]["buckets"].as_array().expect("buckets");
        assert_eq!(buckets3.len(), 1);
        assert_eq!(buckets3[0]["key"]["cat"], "electronics");
        assert_eq!(buckets3[0]["key"]["prod"], "tv");

        // 4. Test histogram source in composite
        let hist_resp = search_core(
            &state,
            "comp_idx",
            &json!({
                "size": 0,
                "aggs": {
                    "by_bracket": {
                        "composite": {
                            "size": 10,
                            "sources": [
                                { "bracket": { "histogram": { "field": "price", "interval": 500.0, "order": "desc" } } }
                            ]
                        }
                    }
                }
            }),
        )
        .await
        .unwrap();

        let h_aggs = hist_resp.aggregations.expect("aggs present");
        let h_buckets = h_aggs["by_bracket"]["buckets"].as_array().expect("buckets");
        assert!(!h_buckets.is_empty());
        // Ordered desc: 1500, 1000, 500, 0
        assert_eq!(h_buckets[0]["key"]["bracket"], 1500.0);

        // 5. Error handling
        let empty_sources = search_core(
            &state,
            "comp_idx",
            &json!({
                "aggs": {
                    "comp": { "composite": { "sources": [] } }
                }
            }),
        )
        .await;
        assert!(empty_sources.is_err());
    }

    #[tokio::test]
    async fn aggregations_significant_terms() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("sig_idx")).unwrap();

        index_docs(
            &state,
            "sig_idx",
            &[
                json!({"city": "Seattle", "category": "boat_theft", "loss": 5000.0}),
                json!({"city": "Seattle", "category": "boat_theft", "loss": 8000.0}),
                json!({"city": "Seattle", "category": "boat_theft", "loss": 6000.0}),
                json!({"city": "Seattle", "category": "burglary", "loss": 1000.0}),
                json!({"city": "Denver", "category": "burglary", "loss": 2000.0}),
                json!({"city": "Denver", "category": "burglary", "loss": 3000.0}),
                json!({"city": "Denver", "category": "burglary", "loss": 2500.0}),
                json!({"city": "Denver", "category": "burglary", "loss": 1500.0}),
                json!({"city": "Denver", "category": "burglary", "loss": 4000.0}),
                json!({"city": "Denver", "category": "boat_theft", "loss": 7000.0}),
            ],
        )
        .await;

        let resp = search_core(
            &state,
            "sig_idx",
            &json!({
                "size": 0,
                "query": {
                    "term": { "city": "Seattle" }
                },
                "aggs": {
                    "unusual_crime": {
                        "significant_terms": {
                            "field": "category",
                            "size": 5
                        },
                        "aggs": {
                            "avg_loss": { "avg": { "field": "loss" } }
                        }
                    }
                }
            }),
        )
        .await
        .unwrap();

        let aggs = resp.aggregations.expect("aggs present");
        let sig = &aggs["unusual_crime"];
        assert_eq!(sig["doc_count"], 4);
        assert_eq!(sig["bg_count"], 10);

        let buckets = sig["buckets"].as_array().expect("buckets");
        assert!(!buckets.is_empty());
        assert_eq!(buckets[0]["key"], "boat_theft");
        assert_eq!(buckets[0]["doc_count"], 3);
        assert_eq!(buckets[0]["bg_count"], 4);
        assert!(buckets[0]["score"].as_f64().unwrap() > 0.0);
        assert_eq!(buckets[0]["aggs"]["avg_loss"]["value"], 6333.333333333333);
    }

    #[tokio::test]
    async fn search_fuzzy_query() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("fuzzy_idx")).unwrap();

        index_docs(
            &state,
            "fuzzy_idx",
            &[
                json!({"name": "kitten", "category": "pet"}),
                json!({"name": "sitting", "category": "action"}),
                json!({"name": "kitchen", "category": "room"}),
                json!({"name": "apple", "category": "fruit"}),
            ],
        )
        .await;

        // 1. Basic fuzzy string matching with distance 1
        let resp = search_core(
            &state,
            "fuzzy_idx",
            &json!({
                "query": {
                    "fuzzy": {
                        "name": "kitton"
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp.hits.total.value, 1);
        assert_eq!(resp.hits.hits[0].source["name"], "kitten");

        // 2. Fuzzy with options: value, fuzziness: 1, prefix_length: 3
        let resp2 = search_core(
            &state,
            "fuzzy_idx",
            &json!({
                "query": {
                    "fuzzy": {
                        "name": {
                            "value": "kitton",
                            "fuzziness": 1,
                            "prefix_length": 3
                        }
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp2.hits.total.value, 1);
        assert_eq!(resp2.hits.hits[0].source["name"], "kitten");

        // Prefix length 3: prefix "sit" matches "sitting" with distance 2
        let resp3 = search_core(
            &state,
            "fuzzy_idx",
            &json!({
                "query": {
                    "fuzzy": {
                        "name": {
                            "value": "sitton",
                            "fuzziness": 2,
                            "prefix_length": 3
                        }
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp3.hits.total.value, 1);
        assert_eq!(resp3.hits.hits[0].source["name"], "sitting");

        // Prefix length 3: prefix "mut" does not match any document
        let resp3_none = search_core(
            &state,
            "fuzzy_idx",
            &json!({
                "query": {
                    "fuzzy": {
                        "name": {
                            "value": "mutton",
                            "fuzziness": 2,
                            "prefix_length": 3
                        }
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp3_none.hits.total.value, 0);

        // 3. Fuzzy inside bool filter
        let resp4 = search_core(
            &state,
            "fuzzy_idx",
            &json!({
                "query": {
                    "bool": {
                        "must": [
                            {
                                "fuzzy": {
                                    "name": {
                                        "value": "kitcen",
                                        "fuzziness": 1
                                    }
                                }
                            }
                        ],
                        "filter": [
                            {
                                "term": {
                                    "category": "room"
                                }
                            }
                        ]
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp4.hits.total.value, 1);
        assert_eq!(resp4.hits.hits[0].source["name"], "kitchen");
    }

    #[tokio::test]
    async fn search_match_phrase_and_multi_match() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("phrase_idx")).unwrap();

        crate::handlers::indices::create_index_core(&state, "phrase_idx", None)
            .await
            .unwrap();
        crate::handlers::mapping::put_mapping_core(
            &state,
            "phrase_idx",
            &json!({
                "properties": {
                    "body": {"type": "text"},
                    "title": {"type": "text"}
                },
                "indexes": {
                    "body": "bm25",
                    "title": "bm25"
                }
            }),
        )
        .await
        .unwrap();

        index_docs(
            &state,
            "phrase_idx",
            &[
                json!({"title": "alpha", "body": "the quick brown fox jumps"}),
                json!({"title": "beta", "body": "the quick red fox jumps"}),
                json!({"title": "gamma fox", "body": "lazy brown dog"}),
            ],
        )
        .await;

        // Exact phrase search: "quick brown fox" matches alpha
        let resp = search_core(
            &state,
            "phrase_idx",
            &json!({"query": {"match_phrase": {"body": "quick brown fox"}}}),
        )
        .await
        .unwrap();
        assert_eq!(resp.hits.total.value, 1);
        assert_eq!(resp.hits.hits[0].id, "phrase_idx-doc-0");

        // Phrase with slop: "quick fox" with slop=1 matches alpha ("quick brown fox") and beta ("quick red fox")
        let resp_slop = search_core(
            &state,
            "phrase_idx",
            &json!({
                "query": {
                    "match_phrase": {
                        "body": {
                            "query": "quick fox",
                            "slop": 1
                        }
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_slop.hits.total.value, 2);

        // multi_match across body and title for "fox"
        let resp_multi = search_core(
            &state,
            "phrase_idx",
            &json!({
                "query": {
                    "multi_match": {
                        "query": "fox",
                        "fields": ["body", "title"]
                    }
                }
            }),
        )
        .await
        .unwrap();
        // alpha, beta have "fox" in body, gamma has "fox" in title
        assert_eq!(resp_multi.hits.total.value, 3);
    }

    #[tokio::test]
    async fn search_bool_should_wildcard_prefix_ids_and_sort() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("ops")).unwrap();

        crate::handlers::indices::create_index_core(&state, "ops", None)
            .await
            .unwrap();
        crate::handlers::mapping::put_mapping_core(
            &state,
            "ops",
            &json!({
                "properties": {
                    "title": {"type": "text"},
                    "tag": {"type": "keyword"},
                    "score_val": {"type": "long"}
                },
                "indexes": {
                    "title": "bm25"
                }
            }),
        )
        .await
        .unwrap();

        index_docs(
            &state,
            "ops",
            &[
                json!({"title": "alice in wonderland", "tag": "novel", "score_val": 10}),
                json!({"title": "bob the builder", "tag": "animation", "score_val": 40}),
                json!({"title": "charlie and chocolate factory", "tag": "novel", "score_val": 25}),
                json!({"title": "aladin and the lamp", "tag": "fairy_tale", "score_val": 50}),
            ],
        )
        .await;

        // 1. bool.should: tag = 'animation' OR score_val = 10
        let resp_should = search_core(
            &state,
            "ops",
            &json!({
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"tag": "animation"}},
                            {"term": {"score_val": 10}}
                        ]
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_should.hits.total.value, 2);

        // 2. prefix query: title starts with "al"
        let resp_prefix = search_core(
            &state,
            "ops",
            &json!({
                "query": {
                    "prefix": {
                        "title": "al"
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_prefix.hits.total.value, 2); // alice, aladin

        // 3. wildcard query: title contains "chocolate"
        let resp_wildcard = search_core(
            &state,
            "ops",
            &json!({
                "query": {
                    "wildcard": {
                        "title": "*chocolate*"
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_wildcard.hits.total.value, 1);
        assert_eq!(resp_wildcard.hits.hits[0].id, "ops-doc-2");

        // 3b. regexp query: title matches ".*choc.*"
        let resp_regexp = search_core(
            &state,
            "ops",
            &json!({
                "query": {
                    "regexp": {
                        "title": ".*choc.*"
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_regexp.hits.total.value, 1);
        assert_eq!(resp_regexp.hits.hits[0].id, "ops-doc-2");

        // 4. ids query
        let resp_ids = search_core(
            &state,
            "ops",
            &json!({
                "query": {
                    "ids": {
                        "values": ["ops-doc-1", "ops-doc-3"]
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_ids.hits.total.value, 2);

        // 5. custom sort by score_val descending
        let resp_sorted = search_core(
            &state,
            "ops",
            &json!({
                "query": {"match_all": {}},
                "sort": [
                    {"score_val": {"order": "desc"}}
                ]
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_sorted.hits.hits.len(), 4);
        assert_eq!(resp_sorted.hits.hits[0].source["score_val"], 50);
        assert_eq!(resp_sorted.hits.hits[1].source["score_val"], 40);
        assert_eq!(resp_sorted.hits.hits[2].source["score_val"], 25);
        assert_eq!(resp_sorted.hits.hits[3].source["score_val"], 10);

        // 6. search_after pagination
        let page1 = search_core(
            &state,
            "ops",
            &json!({
                "query": {"match_all": {}},
                "sort": [{"score_val": {"order": "desc"}}],
                "size": 2
            }),
        )
        .await
        .unwrap();
        assert_eq!(page1.hits.hits.len(), 2);
        assert_eq!(page1.hits.hits[0].source["score_val"], 50);
        assert_eq!(page1.hits.hits[1].source["score_val"], 40);
        assert_eq!(page1.hits.hits[0].sort, Some(vec![json!(50)]));
        assert_eq!(page1.hits.hits[1].sort, Some(vec![json!(40)]));

        let last_sort = page1.hits.hits[1].sort.as_ref().unwrap();
        let page2 = search_core(
            &state,
            "ops",
            &json!({
                "query": {"match_all": {}},
                "sort": [{"score_val": {"order": "desc"}}],
                "size": 2,
                "search_after": last_sort
            }),
        )
        .await
        .unwrap();
        assert_eq!(page2.hits.hits.len(), 2);
        assert_eq!(page2.hits.hits[0].source["score_val"], 25);
        assert_eq!(page2.hits.hits[1].source["score_val"], 10);
        assert_eq!(page2.hits.hits[0].sort, Some(vec![json!(25)]));
        assert_eq!(page2.hits.hits[1].sort, Some(vec![json!(10)]));

        // 7. highlight
        let resp_hl = search_core(
            &state,
            "ops",
            &json!({
                "query": {
                    "match": {
                        "title": "chocolate"
                    }
                },
                "highlight": {
                    "fields": {
                        "title": {}
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_hl.hits.hits.len(), 1);
        let hl_map = resp_hl.hits.hits[0].highlight.as_ref().unwrap();
        let snippets = hl_map.get("title").unwrap();
        assert!(snippets[0].contains("<em>chocolate</em>"));
    }

    #[tokio::test]
    async fn search_nested_query_and_dot_path() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("nested_idx")).unwrap();

        index_docs(
            &state,
            "nested_idx",
            &[
                json!({
                    "user": {
                        "name": "alice",
                        "age": 30
                    },
                    "status": "active"
                }),
                json!({
                    "user": {
                        "name": "bob",
                        "age": 25
                    },
                    "status": "pending"
                }),
            ],
        )
        .await;

        // Nested query with relative field path ("name") inside "user"
        let resp_rel = search_core(
            &state,
            "nested_idx",
            &json!({
                "query": {
                    "nested": {
                        "path": "user",
                        "query": {
                            "term": {
                                "name": "alice"
                            }
                        }
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_rel.hits.total.value, 1);
        assert_eq!(resp_rel.hits.hits[0].source["user"]["name"], "alice");

        // Nested query with fully-qualified field path ("user.name")
        let resp_fq = search_core(
            &state,
            "nested_idx",
            &json!({
                "query": {
                    "nested": {
                        "path": "user",
                        "query": {
                            "term": {
                                "user.name": "bob"
                            }
                        }
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_fq.hits.total.value, 1);
        assert_eq!(resp_fq.hits.hits[0].source["user"]["name"], "bob");

        // Sort by nested dot path "user.age" descending
        let resp_sorted = search_core(
            &state,
            "nested_idx",
            &json!({
                "query": {"match_all": {}},
                "sort": [
                    {"user.age": {"order": "desc"}}
                ]
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_sorted.hits.hits.len(), 2);
        assert_eq!(resp_sorted.hits.hits[0].source["user"]["name"], "alice");
        assert_eq!(resp_sorted.hits.hits[1].source["user"]["name"], "bob");
    }

    #[tokio::test]
    async fn search_nested_array_of_objects_query() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("nested_arr")).unwrap();

        // Index documents with array of objects
        index_docs(
            &state,
            "nested_arr",
            &[
                // Doc 0: alice has 5 stars, bob has 1 star
                json!({
                    "title": "Post 1",
                    "comments": [
                        { "author": "alice", "stars": 5 },
                        { "author": "bob", "stars": 1 }
                    ]
                }),
                // Doc 1: alice has 1 star
                json!({
                    "title": "Post 2",
                    "comments": [
                        { "author": "alice", "stars": 1 }
                    ]
                }),
            ],
        )
        .await;

        // Query: nested comments where author == "alice" AND stars == 1
        // In standard non-nested object semantics, Doc 0 would falsely match because it has alice (stars 5) and bob (stars 1).
        // Under nested semantics, matching must happen on the same array element.
        // Therefore, ONLY Doc 1 must match!
        let resp = search_core(
            &state,
            "nested_arr",
            &json!({
                "query": {
                    "nested": {
                        "path": "comments",
                        "query": {
                            "bool": {
                                "must": [
                                    { "term": { "comments.author": "alice" } },
                                    { "term": { "comments.stars": 1 } }
                                ]
                            }
                        }
                    }
                }
            }),
        )
        .await
        .unwrap();

        assert_eq!(resp.hits.total.value, 1);
        assert_eq!(resp.hits.hits[0].id, "nested_arr-doc-1");
        assert_eq!(resp.hits.hits[0].source["title"], "Post 2");
        assert_eq!(resp.hits.hits[0].source["comments"][0]["author"], "alice");
        assert_eq!(resp.hits.hits[0].source["comments"][0]["stars"], 1);

        // Query with relative field paths inside nested: "author" and "stars" (unqualified)
        let resp_rel = search_core(
            &state,
            "nested_arr",
            &json!({
                "query": {
                    "nested": {
                        "path": "comments",
                        "query": {
                            "bool": {
                                "must": [
                                    { "term": { "author": "alice" } },
                                    { "term": { "stars": 5 } }
                                ]
                            }
                        }
                    }
                }
            }),
        )
        .await
        .unwrap();

        assert_eq!(resp_rel.hits.total.value, 1);
        assert_eq!(resp_rel.hits.hits[0].id, "nested_arr-doc-0");

        // Query with impossible combination: bob with 5 stars
        let resp_none = search_core(
            &state,
            "nested_arr",
            &json!({
                "query": {
                    "nested": {
                        "path": "comments",
                        "query": {
                            "bool": {
                                "must": [
                                    { "term": { "author": "bob" } },
                                    { "term": { "stars": 5 } }
                                ]
                            }
                        }
                    }
                }
            }),
        )
        .await
        .unwrap();

        assert_eq!(resp_none.hits.total.value, 0);
        assert!(resp_none.hits.hits.is_empty());
    }

    #[tokio::test]
    async fn search_has_child_and_has_parent() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("posts")).unwrap();
        std::fs::create_dir_all(tmp.path().join("comments")).unwrap();

        // 1. Index posts
        crate::handlers::indices::create_index_core(&state, "posts", None)
            .await
            .unwrap();
        crate::handlers::docs::index_document_core(
            &state,
            "posts",
            Some("post1"),
            json!({"title": "Introduction to Rust", "tag": "tech"}),
        )
        .await
        .unwrap();
        crate::handlers::docs::index_document_core(
            &state,
            "posts",
            Some("post2"),
            json!({"title": "Cooking Italian Pasta", "tag": "food"}),
        )
        .await
        .unwrap();
        crate::handlers::docs::index_document_core(
            &state,
            "posts",
            Some("post3"),
            json!({"title": "Advanced Data Systems", "tag": "tech"}),
        )
        .await
        .unwrap();

        // 2. Index comments
        crate::handlers::indices::create_index_core(&state, "comments", None)
            .await
            .unwrap();
        crate::handlers::docs::index_document_core(
            &state,
            "comments",
            Some("c1"),
            json!({"post_id": "post1", "text": "Great tutorial", "author": "alice"}),
        )
        .await
        .unwrap();
        crate::handlers::docs::index_document_core(
            &state,
            "comments",
            Some("c2"),
            json!({"post_id": "post1", "text": "Loved it", "author": "bob"}),
        )
        .await
        .unwrap();
        crate::handlers::docs::index_document_core(
            &state,
            "comments",
            Some("c3"),
            json!({"post_id": "post2", "text": "Delicious recipe", "author": "charlie"}),
        )
        .await
        .unwrap();

        // Refresh tables so writes are committed and searchable
        refresh_core(&state, "posts").await.unwrap();
        refresh_core(&state, "comments").await.unwrap();

        // Test 1: has_child (find post with comments by alice)
        let resp_child = search_core(
            &state,
            "posts",
            &json!({
                "query": {
                    "has_child": {
                        "type": "comments",
                        "query": {
                            "term": {"author": "alice"}
                        }
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_child.hits.total.value, 1);
        assert_eq!(resp_child.hits.hits[0].id, "post1");

        // Test 2: has_child with min_children: 2 (only post1 has 2 comments)
        let resp_min = search_core(
            &state,
            "posts",
            &json!({
                "query": {
                    "has_child": {
                        "type": "comments",
                        "query": {"match_all": {}},
                        "min_children": 2
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_min.hits.total.value, 1);
        assert_eq!(resp_min.hits.hits[0].id, "post1");

        // Test 3: has_parent (find comments whose post tag is food)
        let resp_parent = search_core(
            &state,
            "comments",
            &json!({
                "query": {
                    "has_parent": {
                        "parent_type": "posts",
                        "query": {
                            "term": {"tag": "food"}
                        }
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_parent.hits.total.value, 1);
        assert_eq!(resp_parent.hits.hits[0].id, "c3");

        // Test 4: has_child inside bool.must
        let resp_bool = search_core(
            &state,
            "posts",
            &json!({
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"tag": "tech"}},
                            {
                                "has_child": {
                                    "type": "comments",
                                    "query": {"term": {"author": "bob"}}
                                }
                            }
                        ]
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_bool.hits.total.value, 1);
        assert_eq!(resp_bool.hits.hits[0].id, "post1");

        // Test 5: has_child with 0 matching child docs
        let resp_none = search_core(
            &state,
            "posts",
            &json!({
                "query": {
                    "has_child": {
                        "type": "comments",
                        "query": {"term": {"author": "nobody"}}
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_none.hits.total.value, 0);

        // Test 6: count_core with has_child
        let count_resp = count_core(
            &state,
            "posts",
            Some(&json!({
                "query": {
                    "has_child": {
                        "type": "comments",
                        "query": {"term": {"author": "charlie"}}
                    }
                }
            })),
        )
        .await
        .unwrap();
        assert_eq!(count_resp.count, 1);

        // Test 7: graph-backed edge_index with has_child
        std::fs::create_dir_all(tmp.path().join("g_posts")).unwrap();
        std::fs::create_dir_all(tmp.path().join("g_comments")).unwrap();
        std::fs::create_dir_all(tmp.path().join("g_edges")).unwrap();

        crate::handlers::indices::create_index_core(&state, "g_posts", None)
            .await
            .unwrap();
        crate::handlers::indices::create_index_core(&state, "g_comments", None)
            .await
            .unwrap();
        crate::handlers::indices::create_index_core(&state, "g_edges", None)
            .await
            .unwrap();

        crate::handlers::docs::index_document_core(
            &state,
            "g_posts",
            Some("100"),
            json!({"title": "Graph Parent Post"}),
        )
        .await
        .unwrap();
        crate::handlers::docs::index_document_core(
            &state,
            "g_comments",
            Some("200"),
            json!({"text": "Graph Child Comment"}),
        )
        .await
        .unwrap();
        crate::handlers::docs::index_document_core(
            &state,
            "g_edges",
            Some("e1"),
            json!({"source": 100, "target": 200}),
        )
        .await
        .unwrap();

        refresh_core(&state, "g_posts").await.unwrap();
        refresh_core(&state, "g_comments").await.unwrap();
        refresh_core(&state, "g_edges").await.unwrap();

        let resp_graph_child = search_core(
            &state,
            "g_posts",
            &json!({
                "query": {
                    "has_child": {
                        "type": "g_comments",
                        "query": {"match_all": {}},
                        "edge_index": "g_edges"
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_graph_child.hits.total.value, 1);
        assert_eq!(resp_graph_child.hits.hits[0].id, "100");

        let resp_graph_parent = search_core(
            &state,
            "g_comments",
            &json!({
                "query": {
                    "has_parent": {
                        "parent_type": "g_posts",
                        "query": {"match_all": {}},
                        "edge_index": "g_edges"
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_graph_parent.hits.total.value, 1);
        assert_eq!(resp_graph_parent.hits.hits[0].id, "200");
    }

    #[tokio::test]
    async fn search_query_string_and_simple_query_string() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("qs_idx")).unwrap();

        crate::handlers::indices::create_index_core(&state, "qs_idx", None)
            .await
            .unwrap();
        crate::handlers::mapping::put_mapping_core(
            &state,
            "qs_idx",
            &json!({
                "properties": {
                    "title": {"type": "text"},
                    "tag": {"type": "keyword"},
                    "score_val": {"type": "long"}
                },
                "indexes": {
                    "title": "bm25"
                }
            }),
        )
        .await
        .unwrap();

        index_docs(
            &state,
            "qs_idx",
            &[
                json!({"title": "Introduction to Rust programming", "tag": "tech", "score_val": 10}),
                json!({"title": "Advanced Rust concurrency and systems", "tag": "tech", "score_val": 40}),
                json!({"title": "Italian cooking pasta and pizza", "tag": "food", "score_val": 25}),
                json!({"title": "Desserts and baking cookies", "tag": "food", "score_val": 50}),
            ],
        )
        .await;

        // 1. query_string with AND and field prefix
        let resp_and = search_core(
            &state,
            "qs_idx",
            &json!({
                "query": {
                    "query_string": {
                        "query": "title:Rust AND tag:tech",
                        "default_operator": "AND"
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_and.hits.total.value, 2);

        // 2. query_string with phrase search
        let resp_phrase = search_core(
            &state,
            "qs_idx",
            &json!({
                "query": {
                    "query_string": {
                        "query": "\"Italian cooking\"",
                        "default_field": "title"
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_phrase.hits.total.value, 1);
        assert!(resp_phrase.hits.hits[0].source["title"]
            .as_str()
            .unwrap()
            .contains("Italian"));

        // 3. query_string with range and wildcard
        let resp_range = search_core(
            &state,
            "qs_idx",
            &json!({
                "query": {
                    "query_string": {
                        "query": "score_val:[20 TO 45] AND tag:tech"
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_range.hits.total.value, 1);
        assert_eq!(resp_range.hits.hits[0].source["score_val"], 40);

        // 4. simple_query_string with + and -
        let resp_simple = search_core(
            &state,
            "qs_idx",
            &json!({
                "query": {
                    "simple_query_string": {
                        "query": "tag:food -baking",
                        "default_field": "title"
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_simple.hits.total.value, 1);
        assert!(resp_simple.hits.hits[0].source["title"]
            .as_str()
            .unwrap()
            .contains("pasta"));

        // 5. query_string inside bool.must
        let resp_bool = search_core(
            &state,
            "qs_idx",
            &json!({
                "query": {
                    "bool": {
                        "must": [
                            {"term": {"tag": "tech"}},
                            {
                                "query_string": {
                                    "query": "title:concurrency"
                                }
                            }
                        ]
                    }
                }
            }),
        )
        .await
        .unwrap();
        assert_eq!(resp_bool.hits.total.value, 1);
        assert_eq!(resp_bool.hits.hits[0].source["score_val"], 40);

        // 6. search_get endpoint with Lucene query
        let state_arc = Arc::new(state);
        let mut q_params = HashMap::new();
        q_params.insert("q".to_string(), "title:Rust AND tag:tech".to_string());
        let get_resp = search_get(
            State(state_arc),
            Path("qs_idx".to_string()),
            axum::extract::Query(q_params),
        )
        .await;
        assert_eq!(get_resp.status(), axum::http::StatusCode::OK);
    }

    #[tokio::test]
    async fn search_scroll_lifecycle() {
        let tmp = tempfile::tempdir().unwrap();
        let root = format!("file://{}", tmp.path().display());
        let state = AppState::new(root, "test-cluster".into());
        std::fs::create_dir_all(tmp.path().join("scroll_idx")).unwrap();

        let docs: Vec<Value> = (0..5)
            .map(|i| json!({ "title": format!("Document {i}"), "order_num": i }))
            .collect();
        index_docs(&state, "scroll_idx", &docs).await;

        // 1. Initial search with scroll=1m and size=2
        let resp1 = search_core(
            &state,
            "scroll_idx",
            &json!({
                "size": 2,
                "scroll": "1m",
                "query": { "match_all": {} }
            }),
        )
        .await
        .unwrap();

        assert_eq!(resp1.hits.total.value, 5);
        assert_eq!(resp1.hits.hits.len(), 2);
        let scroll_id1 = resp1.scroll_id.expect("scroll_id present on resp1");
        assert!(!scroll_id1.is_empty());

        let mut all_ids = Vec::new();
        for h in resp1.hits.hits {
            all_ids.push(h.id);
        }

        // 2. Fetch page 2
        let resp2 = execute_scroll(&state, &scroll_id1, None).await.unwrap();
        assert_eq!(resp2.hits.hits.len(), 2);
        let scroll_id2 = resp2.scroll_id.expect("scroll_id present on resp2");
        for h in resp2.hits.hits {
            all_ids.push(h.id);
        }

        // 3. Fetch page 3 (last doc)
        let resp3 = execute_scroll(&state, &scroll_id2, None).await.unwrap();
        assert_eq!(resp3.hits.hits.len(), 1);
        let scroll_id3 = resp3.scroll_id.expect("scroll_id present on resp3");
        for h in resp3.hits.hits {
            all_ids.push(h.id);
        }

        // 4. Fetch page 4 (exhausted -> empty hits)
        let resp4 = execute_scroll(&state, &scroll_id3, None).await.unwrap();
        assert_eq!(resp4.hits.hits.len(), 0);

        // Verify all 5 documents retrieved in order without duplicates
        assert_eq!(all_ids.len(), 5);
        let mut sorted_ids = all_ids.clone();
        sorted_ids.sort();
        sorted_ids.dedup();
        assert_eq!(sorted_ids.len(), 5);

        // 5. Test expired token returns context error
        let mut expired_token = decode_scroll_token(&scroll_id3).unwrap();
        expired_token.expires_at = 0; // past
        let expired_id = encode_scroll_token(&expired_token).unwrap();
        let err = execute_scroll(&state, &expired_id, None).await.unwrap_err();
        assert!(err.to_string().contains("search_context_missing_exception"));
        let es_err = crate::es_types::EsError::from(err);
        assert_eq!(es_err.status, 404);
        assert_eq!(es_err.error.error_type, "search_context_missing_exception");

        // 6. Test clear_scroll endpoint
        let clear_resp = clear_scroll(None).await;
        assert_eq!(clear_resp.status(), axum::http::StatusCode::OK);
    }
}
