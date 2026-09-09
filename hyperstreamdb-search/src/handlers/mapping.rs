// Copyright (c) 2026 Richard Albright. All rights reserved.

//! ES `_mapping` endpoints: `GET /{index}/_mapping` and `PUT /{index}/_mapping`.
//!
//! GET renders the table's Arrow schema (plus per-column index algorithms from
//! the manifest) as ES 7.10 mapping properties. PUT adds new properties via
//! `Table::add_column` and optionally registers index algorithms via
//! `Table::add_index`.

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, TimeUnit};
use axum::extract::{Path, State};
use axum::response::Response;
use axum::Json;
use hyperstreamdb::core::manifest::IndexAlgorithm;
use hyperstreamdb::{HyperstreamError, Table};
use serde_json::{Map, Value};

use crate::state::{table_exists, AppState};

use super::es_response;

fn bad_request(reason: impl Into<String>) -> HyperstreamError {
    HyperstreamError::SchemaIncompatible {
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
async fn table_properties(table: &Table) -> Result<Value, HyperstreamError> {
    let schema = table.arrow_schema();

    // Per-column index algorithms from the current manifest schema.
    let manifest = table
        .manifest()
        .await
        .map_err(|e| HyperstreamError::internal(format!("failed to read manifest: {e}")))?;
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
) -> Result<Value, HyperstreamError> {
    if !table_exists(&state.index_uri(index)).await {
        return Err(HyperstreamError::TableNotFound {
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
pub(crate) fn es_to_arrow_type(spec: &Value) -> Result<DataType, HyperstreamError> {
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
        other => Err(bad_request(format!(
            "unsupported mapping type '{other}' (supported: text, keyword, long, integer, double, float, boolean, date, dense_vector)"
        ))),
    }
}

/// Build an Arrow schema from an ES `mappings.properties` object. Returns an
/// empty schema when no properties are supplied.
pub(crate) fn schema_from_mapping(
    properties: Option<&Map<String, Value>>,
) -> Result<arrow::datatypes::SchemaRef, HyperstreamError> {
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
) -> Result<Value, HyperstreamError> {
    if !table_exists(&state.index_uri(index)).await {
        return Err(HyperstreamError::TableNotFound {
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
                    HyperstreamError::SchemaIncompatible {
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
                HyperstreamError::SchemaIncompatible {
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
