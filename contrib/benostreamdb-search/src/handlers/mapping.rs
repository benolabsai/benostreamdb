// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! ES `_mapping` endpoints: `GET /{index}/_mapping` and `PUT /{index}/_mapping`.
//!
//! GET renders the table's Arrow schema (plus per-column index algorithms from
//! the manifest) as ES 7.10 mapping properties. PUT adds new properties via
//! `Table::add_column` and optionally registers index algorithms via
//! `Table::add_index`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, TimeUnit};
use axum::body::Bytes;
use axum::extract::{Path, Query, State};
use axum::response::Response;
use axum::Json;
use benostreamdb::core::manifest::IndexAlgorithm;
use benostreamdb::{BenoStreamError, Table};
use serde_json::{Map, Value};

use crate::state::AppState;

use super::es_response;

fn bad_request(reason: impl Into<String>) -> BenoStreamError {
    BenoStreamError::SchemaIncompatible {
        reason: reason.into(),
    }
}

/// Map an Arrow data type to its ES 7.10 property type and (for
/// `dense_vector`) the dimensionality.
fn arrow_to_es_type(dt: &DataType) -> (String, Option<usize>) {
    match dt {
        DataType::Utf8 | DataType::LargeUtf8 => ("text".into(), None),
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => ("long".into(), None),
        DataType::Float32 => ("float".into(), None),
        DataType::Float64 => ("double".into(), None),
        DataType::Boolean => ("boolean".into(), None),
        DataType::Date32 | DataType::Date64 | DataType::Timestamp(_, _) => ("date".into(), None),
        DataType::FixedSizeList(field, dim) => {
            if field.data_type() == &DataType::Float32 {
                ("dense_vector".into(), Some(*dim as usize))
            } else {
                ("object".into(), None)
            }
        }
        DataType::List(field) | DataType::LargeList(field) => {
            if matches!(field.data_type(), DataType::Struct(_)) {
                ("nested".into(), None)
            } else {
                arrow_to_es_type(field.data_type())
            }
        }
        DataType::Struct(_) => ("object".into(), None),
        _ => ("object".into(), None),
    }
}

/// Render a single Arrow field (and its nested struct fields, if any) as an
/// ES mapping property object.
fn field_to_property(dt: &DataType) -> Value {
    match dt {
        DataType::Struct(fields) => {
            let mut props = Map::new();
            for f in fields {
                props.insert(f.name().clone(), field_to_property(f.data_type()));
            }
            let mut obj = Map::new();
            obj.insert("type".into(), Value::String("object".into()));
            obj.insert("properties".into(), Value::Object(props));
            Value::Object(obj)
        }
        DataType::List(sub) | DataType::LargeList(sub) => {
            if let DataType::Struct(fields) = sub.data_type() {
                let mut props = Map::new();
                for f in fields {
                    props.insert(f.name().clone(), field_to_property(f.data_type()));
                }
                let mut obj = Map::new();
                obj.insert("type".into(), Value::String("nested".into()));
                obj.insert("properties".into(), Value::Object(props));
                Value::Object(obj)
            } else {
                let (ty, dims) = arrow_to_es_type(sub.data_type());
                let mut obj = Map::new();
                obj.insert("type".into(), Value::String(ty));
                if let Some(d) = dims {
                    obj.insert("dims".into(), Value::from(d));
                }
                Value::Object(obj)
            }
        }
        _ => {
            let (ty, dims) = arrow_to_es_type(dt);
            let mut obj = Map::new();
            obj.insert("type".into(), Value::String(ty));
            if let Some(d) = dims {
                obj.insert("dims".into(), Value::from(d));
            }
            Value::Object(obj)
        }
    }
}

/// Build the ES `properties` object for a table, annotating text fields that
/// carry a BM25 index with their analyzer and vector fields with `index: true`.
async fn table_properties(table: &Table) -> Result<Value, BenoStreamError> {
    let schema = table.arrow_schema();

    // Per-column index algorithms from the current manifest schema.
    let manifest = table
        .manifest()
        .await
        .map_err(|e| BenoStreamError::internal(format!("failed to read manifest: {e}")))?;
    let index_by_col: std::collections::HashMap<String, Vec<IndexAlgorithm>> = manifest
        .schemas
        .iter()
        .find(|s| s.schema_id == manifest.current_schema_id)
        .map(|s| {
            s.fields
                .iter()
                .filter(|f| !f.indexes.is_empty())
                .map(|f| (f.name.clone(), f.indexes.clone()))
                .collect()
        })
        .unwrap_or_default();

    let mut props = Map::new();
    for f in schema.fields() {
        let mut prop = field_to_property(f.data_type())
            .as_object()
            .cloned()
            .unwrap_or_default();
        if let Some(algs) = index_by_col.get(f.name()) {
            if algs
                .iter()
                .any(|a| matches!(a, IndexAlgorithm::Bm25 { .. }))
            {
                let analyzer = algs
                    .iter()
                    .find_map(|a| match a {
                        IndexAlgorithm::Bm25 { tokenizer, .. } if !tokenizer.is_empty() => {
                            Some(tokenizer.clone())
                        }
                        _ => None,
                    })
                    .unwrap_or_else(|| "english".to_string());
                prop.insert("analyzer".into(), Value::String(analyzer));
            }
            if algs.iter().any(|a| {
                matches!(
                    a,
                    IndexAlgorithm::Hnsw { .. } | IndexAlgorithm::HnswTq8 { .. }
                )
            }) {
                prop.insert("index".into(), Value::Bool(true));
            }
        }
        props.insert(f.name().clone(), Value::Object(prop));
    }
    Ok(Value::Object(props))
}

/// `GET /{index}/_mapping` — ES 7.10 mapping for the index.
pub async fn get_mapping(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
) -> Response {
    es_response(get_mapping_core(&state, &index).await)
}

pub(crate) async fn get_mapping_core(
    state: &AppState,
    index: &str,
) -> Result<Value, BenoStreamError> {
    if !state.index_exists(index).await {
        return Err(BenoStreamError::TableNotFound {
            namespace: String::new(),
            name: index.to_string(),
        });
    }
    let table = state.open_or_create(index, &None).await?;
    let properties = table_properties(&table).await?;

    let mut mappings = Map::new();
    mappings.insert("properties".into(), properties);
    let mut idx = Map::new();
    idx.insert("mappings".into(), Value::Object(mappings));
    let mut root = Map::new();
    root.insert(index.to_string(), Value::Object(idx));
    Ok(Value::Object(root))
}

/// Map an ES mapping property spec to an Arrow data type.
pub(crate) fn es_to_arrow_type(spec: &Value) -> Result<DataType, BenoStreamError> {
    let ty = spec
        .get("type")
        .and_then(Value::as_str)
        .ok_or_else(|| bad_request("mapping property: 'type' is required"))?;
    match ty {
        "text" | "keyword" => Ok(DataType::Utf8),
        "long" | "integer" | "short" | "byte" => Ok(DataType::Int64),
        "double" | "float" => Ok(DataType::Float64),
        "boolean" => Ok(DataType::Boolean),
        "date" => Ok(DataType::Timestamp(TimeUnit::Microsecond, None)),
        "dense_vector" => {
            let dims = spec
                .get("dims")
                .and_then(Value::as_u64)
                .ok_or_else(|| bad_request("dense_vector requires a positive 'dims'"))? as i32;
            if dims <= 0 {
                return Err(bad_request("dense_vector 'dims' must be positive"));
            }
            Ok(DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, false)),
                dims,
            ))
        }
        "nested" => {
            let props = spec
                .get("properties")
                .and_then(Value::as_object)
                .ok_or_else(|| bad_request("nested type requires a 'properties' object"))?;
            let mut struct_fields = Vec::new();
            for (name, child_spec) in props {
                let dt = es_to_arrow_type(child_spec)?;
                struct_fields.push(Field::new(name, dt, true));
            }
            Ok(DataType::List(Arc::new(Field::new(
                "item",
                DataType::Struct(struct_fields.into()),
                true,
            ))))
        }
        other => Err(bad_request(format!(
            "unsupported mapping type '{other}' (supported: text, keyword, long, integer, double, float, boolean, date, dense_vector, nested)"
        ))),
    }
}

/// Build an Arrow schema from an ES `mappings.properties` object. Returns an
/// empty schema when no properties are supplied.
pub(crate) fn schema_from_mapping(
    properties: Option<&Map<String, Value>>,
) -> Result<arrow::datatypes::SchemaRef, BenoStreamError> {
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;
    let mut fields = Vec::new();
    if let Some(props) = properties {
        for (name, spec) in props {
            let dt = es_to_arrow_type(spec)?;
            fields.push(Field::new(name, dt, true));
        }
    }
    Ok(Arc::new(Schema::new(fields)))
}

/// `PUT /{index}/_mapping` — add properties (columns) and optionally register
/// index algorithms. Existing columns with a matching type are a no-op; a
/// type mismatch is a 400.
pub async fn put_mapping(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    es_response(put_mapping_core(&state, &index, &body).await)
}

pub(crate) async fn put_mapping_core(
    state: &AppState,
    index: &str,
    body: &Value,
) -> Result<Value, BenoStreamError> {
    if !state.index_exists(index).await {
        return Err(BenoStreamError::TableNotFound {
            namespace: String::new(),
            name: index.to_string(),
        });
    }
    let table = state.open_or_create(index, &None).await?;

    let properties = body
        .get("properties")
        .and_then(Value::as_object)
        .ok_or_else(|| bad_request("PUT _mapping: expected a 'properties' object"))?;

    let current = table.arrow_schema();
    for (name, spec) in properties {
        let dt = es_to_arrow_type(spec)?;
        match current.field_with_name(name) {
            Ok(existing) => {
                if existing.data_type() != &dt {
                    return Err(bad_request(format!(
                        "column '{name}' already exists with type {}; cannot re-map to {}",
                        existing.data_type(),
                        dt
                    )));
                }
                // Same type: no-op.
            }
            Err(_) => {
                table.add_column(name, dt).await.map_err(|e| {
                    BenoStreamError::SchemaIncompatible {
                        reason: format!("failed to add column '{name}': {e}"),
                    }
                })?;
            }
        }
    }

    // Optional index registration: {"indexes": {"field": "bm25" | "hnsw" | ...}}
    if let Some(indexes) = body.get("indexes").and_then(Value::as_object) {
        for (col, alg_name) in indexes {
            let algorithm = match alg_name.as_str().unwrap_or("") {
                "bm25" => IndexAlgorithm::Bm25 {
                    k1: 0.0,
                    b: 0.0,
                    tokenizer: String::new(),
                },
                "hnsw" => IndexAlgorithm::Hnsw {
                    metric: "l2".into(),
                    complexity: 16,
                    quality: 100,
                    build_device: None,
                    search_device: None,
                },
                "hnsw_tq8" => IndexAlgorithm::HnswTq8 {
                    metric: "l2".into(),
                    complexity: 16,
                    quality: 100,
                },
                "bloom" => IndexAlgorithm::Bloom { fpr: 0.05 },
                "bitmap" => IndexAlgorithm::Bitmap,
                other => {
                    return Err(bad_request(format!(
                        "unsupported index algorithm '{other}' (supported: bm25, hnsw, hnsw_tq8, bloom, bitmap)"
                    )))
                }
            };
            table.add_index(col.clone(), algorithm).await.map_err(|e| {
                BenoStreamError::SchemaIncompatible {
                    reason: format!("failed to add index on '{col}': {e}"),
                }
            })?;
        }
    }

    Ok(Value::Object({
        let mut m = Map::new();
        m.insert("acknowledged".into(), Value::Bool(true));
        m
    }))
}

/// Glob matcher supporting `*` (zero or more chars) and `?` (single char).
fn glob_match(pattern: &str, text: &str) -> bool {
    let p_bytes = pattern.as_bytes();
    let t_bytes = text.as_bytes();
    let mut p_idx = 0;
    let mut t_idx = 0;
    let mut star_idx = None;
    let mut match_idx = 0;

    while t_idx < t_bytes.len() {
        if p_idx < p_bytes.len() && (p_bytes[p_idx] == b'?' || p_bytes[p_idx] == t_bytes[t_idx]) {
            p_idx += 1;
            t_idx += 1;
        } else if p_idx < p_bytes.len() && p_bytes[p_idx] == b'*' {
            star_idx = Some(p_idx);
            p_idx += 1;
            match_idx = t_idx;
        } else if let Some(star) = star_idx {
            p_idx = star + 1;
            match_idx += 1;
            t_idx = match_idx;
        } else {
            return false;
        }
    }

    while p_idx < p_bytes.len() && p_bytes[p_idx] == b'*' {
        p_idx += 1;
    }

    p_idx == p_bytes.len()
}

fn is_searchable(es_type: &str) -> bool {
    !matches!(es_type, "object" | "nested")
}

fn is_aggregatable(es_type: &str) -> bool {
    matches!(
        es_type,
        "long" | "integer" | "short" | "byte" | "double" | "float" | "boolean" | "date" | "keyword"
    )
}

fn collect_field_types(prefix: &str, field: &Field, out: &mut Vec<(String, String)>) {
    let full_name = if prefix.is_empty() {
        field.name().clone()
    } else {
        format!("{}.{}", prefix, field.name())
    };

    if full_name == "_id" || full_name == "_index" {
        out.push((full_name, "keyword".to_string()));
        return;
    }

    match field.data_type() {
        DataType::Struct(subfields) => {
            out.push((full_name.clone(), "object".to_string()));
            for sub in subfields {
                collect_field_types(&full_name, sub, out);
            }
        }
        DataType::List(sub) | DataType::LargeList(sub) => {
            if let DataType::Struct(subfields) = sub.data_type() {
                out.push((full_name.clone(), "nested".to_string()));
                for sf in subfields {
                    collect_field_types(&full_name, sf, out);
                }
            } else {
                let (ty, _) = arrow_to_es_type(sub.data_type());
                out.push((full_name, ty));
            }
        }
        dt => {
            let (ty, _) = arrow_to_es_type(dt);
            out.push((full_name, ty));
        }
    }
}

fn extract_requested_fields(query_fields: Option<&str>, body_bytes: &[u8]) -> Vec<String> {
    let mut fields = Vec::new();
    if let Some(qf) = query_fields {
        for part in qf.split(',') {
            let trimmed = part.trim();
            if !trimmed.is_empty() {
                fields.push(trimmed.to_string());
            }
        }
    }
    if !body_bytes.is_empty() {
        if let Ok(val) = serde_json::from_slice::<Value>(body_bytes) {
            if let Some(f_val) = val.get("fields") {
                if let Some(arr) = f_val.as_array() {
                    for item in arr {
                        if let Some(s) = item.as_str() {
                            let trimmed = s.trim();
                            if !trimmed.is_empty() {
                                fields.push(trimmed.to_string());
                            }
                        }
                    }
                } else if let Some(s) = f_val.as_str() {
                    for part in s.split(',') {
                        let trimmed = part.trim();
                        if !trimmed.is_empty() {
                            fields.push(trimmed.to_string());
                        }
                    }
                }
            }
        }
    }
    fields
}

/// `GET /{index}/_field_caps` and `POST /{index}/_field_caps`
pub async fn field_caps(
    State(state): State<Arc<AppState>>,
    Path(index): Path<String>,
    Query(params): Query<HashMap<String, String>>,
    body: Bytes,
) -> Response {
    let fields = extract_requested_fields(params.get("fields").map(|s| s.as_str()), &body);
    es_response(field_caps_core(&state, Some(&index), &fields).await)
}

/// `GET /_field_caps` and `POST /_field_caps`
pub async fn field_caps_all(
    State(state): State<Arc<AppState>>,
    Query(params): Query<HashMap<String, String>>,
    body: Bytes,
) -> Response {
    let fields = extract_requested_fields(params.get("fields").map(|s| s.as_str()), &body);
    es_response(field_caps_core(&state, None, &fields).await)
}

pub(crate) async fn field_caps_core(
    state: &AppState,
    index_param: Option<&str>,
    requested_fields: &[String],
) -> Result<Value, BenoStreamError> {
    let target_indices = match index_param {
        Some(idx_str) if idx_str != "_all" && idx_str != "*" => {
            let mut resolved_indices = Vec::new();
            for part in idx_str.split(',') {
                let name = part.trim();
                if name.is_empty() {
                    continue;
                }
                let resolved = state.resolve_alias(name).await;
                if !state.index_exists(&resolved).await {
                    return Err(BenoStreamError::TableNotFound {
                        namespace: String::new(),
                        name: name.to_string(),
                    });
                }
                resolved_indices.push(resolved);
            }
            resolved_indices
        }
        _ => state.list_indexes().await?,
    };

    let mut target_indices = target_indices;
    target_indices.sort();
    target_indices.dedup();

    let mut field_type_indices: BTreeMap<String, BTreeMap<String, Vec<String>>> = BTreeMap::new();

    for idx_name in &target_indices {
        let table = state.open_or_create(idx_name, &None).await?;
        let schema = table.arrow_schema();
        let mut field_types = Vec::new();

        for f in schema.fields() {
            collect_field_types("", f, &mut field_types);
        }

        field_types.push(("_id".to_string(), "keyword".to_string()));
        field_types.push(("_index".to_string(), "keyword".to_string()));

        field_types.sort();
        field_types.dedup();

        for (field_name, es_type) in field_types {
            field_type_indices
                .entry(field_name)
                .or_default()
                .entry(es_type)
                .or_default()
                .push(idx_name.clone());
        }
    }

    let match_all = requested_fields.is_empty() || requested_fields.iter().any(|f| f == "*");

    let mut fields_out = Map::new();
    for (field_name, type_map) in field_type_indices {
        if !match_all
            && !requested_fields
                .iter()
                .any(|pat| glob_match(pat, &field_name))
        {
            continue;
        }

        let mut type_caps = Map::new();
        for (es_type, indices_with_type) in type_map {
            let mut cap = Map::new();
            cap.insert("type".to_string(), Value::String(es_type.clone()));
            let is_meta = field_name == "_id" || field_name == "_index";
            cap.insert("metadata_field".to_string(), Value::Bool(is_meta));
            cap.insert(
                "searchable".to_string(),
                Value::Bool(is_searchable(&es_type)),
            );
            cap.insert(
                "aggregatable".to_string(),
                Value::Bool(is_aggregatable(&es_type)),
            );

            if indices_with_type.len() < target_indices.len() {
                cap.insert(
                    "indices".to_string(),
                    Value::Array(indices_with_type.into_iter().map(Value::String).collect()),
                );
            }
            type_caps.insert(es_type, Value::Object(cap));
        }
        fields_out.insert(field_name, Value::Object(type_caps));
    }

    let mut root = Map::new();
    root.insert(
        "indices".to_string(),
        Value::Array(target_indices.into_iter().map(Value::String).collect()),
    );
    root.insert("fields".to_string(), Value::Object(fields_out));
    Ok(Value::Object(root))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn test_state(dir: &tempfile::TempDir) -> AppState {
        let root = format!("file://{}", dir.path().display());
        AppState::new(root, "test-cluster".into())
    }

    #[tokio::test]
    async fn test_field_caps_lifecycle() {
        let tmp = tempfile::tempdir().unwrap();
        let state = test_state(&tmp);

        // Create index 1 with title (text), rating (long), and score (double)
        crate::handlers::indices::create_index_core(
            &state,
            "cap_idx1",
            Some(&json!({
                "mappings": {
                    "properties": {
                        "title": { "type": "text" },
                        "rating": { "type": "long" },
                        "score": { "type": "double" }
                    }
                }
            })),
        )
        .await
        .unwrap();

        // Create index 2 with title (text), rating (double) -> type conflict
        crate::handlers::indices::create_index_core(
            &state,
            "cap_idx2",
            Some(&json!({
                "mappings": {
                    "properties": {
                        "title": { "type": "text" },
                        "rating": { "type": "double" },
                        "active": { "type": "boolean" }
                    }
                }
            })),
        )
        .await
        .unwrap();

        // 1. Single index field caps
        let caps1 = field_caps_core(&state, Some("cap_idx1"), &[])
            .await
            .unwrap();
        assert_eq!(caps1["indices"], json!(["cap_idx1"]));
        let fields1 = caps1["fields"].as_object().unwrap();
        assert_eq!(fields1["title"]["text"]["aggregatable"], false);
        assert_eq!(fields1["title"]["text"]["searchable"], true);
        assert_eq!(fields1["rating"]["long"]["aggregatable"], true);
        assert_eq!(fields1["score"]["double"]["aggregatable"], true);
        assert_eq!(fields1["_id"]["keyword"]["metadata_field"], true);

        // 2. Filter by specific field pattern
        let caps_filtered = field_caps_core(&state, Some("cap_idx1"), &["rat*".to_string()])
            .await
            .unwrap();
        let f_filtered = caps_filtered["fields"].as_object().unwrap();
        assert!(f_filtered.contains_key("rating"));
        assert!(!f_filtered.contains_key("title"));
        assert!(!f_filtered.contains_key("score"));

        // 3. Multi-index field caps with conflicting type for rating
        let caps_both = field_caps_core(&state, Some("cap_idx1,cap_idx2"), &[])
            .await
            .unwrap();
        assert_eq!(caps_both["indices"], json!(["cap_idx1", "cap_idx2"]));
        let fields_both = caps_both["fields"].as_object().unwrap();
        // title is text on both, so indices omitted
        assert!(fields_both["title"]["text"].get("indices").is_none());
        // rating has long on cap_idx1, double on cap_idx2
        assert_eq!(
            fields_both["rating"]["long"]["indices"],
            json!(["cap_idx1"])
        );
        assert_eq!(
            fields_both["rating"]["double"]["indices"],
            json!(["cap_idx2"])
        );

        // 4. Cluster-wide _field_caps (no index param)
        let caps_all = field_caps_core(&state, None, &[]).await.unwrap();
        let all_indices = caps_all["indices"].as_array().unwrap();
        assert!(all_indices.contains(&json!("cap_idx1")));
        assert!(all_indices.contains(&json!("cap_idx2")));

        // 5. Non-existent index -> 404 TableNotFound
        let err = field_caps_core(&state, Some("missing_idx"), &[]).await;
        assert!(matches!(err, Err(BenoStreamError::TableNotFound { .. })));
    }
}
