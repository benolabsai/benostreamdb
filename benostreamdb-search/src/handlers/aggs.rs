// Copyright (c) 2026 Richard Albright. All rights reserved.

//! ES-compatible aggregations for `POST /{index}/_search`.
//!
//! Aggregations are compiled to SQL and executed with DataFusion through
//! [`Table::sql`] (the table is registered as `t`). Supported:
//!
//! - Bucket: `terms`, `histogram`, `date_histogram`, `range`, `filter`,
//!   `missing`
//! - Metric: `avg`, `sum`, `min`, `max`, `value_count`, `cardinality`,
//!   `stats`, `extended_stats`
//! - Nested `aggs` on bucket aggregations
//!
//! Aggregations run over the top-level `filter` (the query clause is not used
//! to scope aggregations in v1). See `docs/OPENSEARCH_COMPATIBILITY.md`.

use arrow::array::{
    Array, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    LargeStringArray, StringArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use benostreamdb::{BenoStreamError, Table};
use serde_json::{json, Map, Value};

fn bad(reason: impl Into<String>) -> BenoStreamError {
    BenoStreamError::SchemaIncompatible {
        reason: reason.into(),
    }
}

/// Field names are inlined into SQL, so validate them strictly.
fn valid_field(field: &str) -> Result<String, BenoStreamError> {
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
        Err(bad(format!("invalid field name '{field}'")))
    }
}

fn sql_string(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

fn where_clause(filter: Option<&str>) -> String {
    match filter {
        Some(f) if !f.trim().is_empty() => format!(" WHERE ({f})"),
        _ => String::new(),
    }
}

fn and_filter(filter: Option<&str>, extra: &str) -> String {
    match filter {
        Some(f) if !f.trim().is_empty() => format!("({f}) AND ({extra})"),
        _ => extra.to_string(),
    }
}

/// Convert one Arrow cell to JSON.
fn cell_to_json(array: &dyn Array, row: usize) -> Value {
    if array.is_null(row) {
        return Value::Null;
    }
    match array.data_type() {
        DataType::Utf8 => array
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|a| Value::String(a.value(row).to_string()))
            .unwrap_or(Value::Null),
        DataType::LargeUtf8 => array
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .map(|a| Value::String(a.value(row).to_string()))
            .unwrap_or(Value::Null),
        DataType::Boolean => array
            .as_any()
            .downcast_ref::<BooleanArray>()
            .map(|a| Value::Bool(a.value(row)))
            .unwrap_or(Value::Null),
        DataType::Int8 => array
            .as_any()
            .downcast_ref::<Int8Array>()
            .map(|a| Value::Number((a.value(row) as i64).into()))
            .unwrap_or(Value::Null),
        DataType::Int16 => array
            .as_any()
            .downcast_ref::<Int16Array>()
            .map(|a| Value::Number((a.value(row) as i64).into()))
            .unwrap_or(Value::Null),
        DataType::Int32 => array
            .as_any()
            .downcast_ref::<Int32Array>()
            .map(|a| Value::Number((a.value(row) as i64).into()))
            .unwrap_or(Value::Null),
        DataType::Int64 => array
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|a| Value::Number(a.value(row).into()))
            .unwrap_or(Value::Null),
        DataType::UInt8 => array
            .as_any()
            .downcast_ref::<UInt8Array>()
            .map(|a| Value::Number((a.value(row) as i64).into()))
            .unwrap_or(Value::Null),
        DataType::UInt16 => array
            .as_any()
            .downcast_ref::<UInt16Array>()
            .map(|a| Value::Number((a.value(row) as i64).into()))
            .unwrap_or(Value::Null),
        DataType::UInt32 => array
            .as_any()
            .downcast_ref::<UInt32Array>()
            .map(|a| Value::Number((a.value(row) as i64).into()))
            .unwrap_or(Value::Null),
        DataType::UInt64 => array
            .as_any()
            .downcast_ref::<UInt64Array>()
            .map(|a| Value::Number((a.value(row) as i64).into()))
            .unwrap_or(Value::Null),
        DataType::Float32 => array
            .as_any()
            .downcast_ref::<Float32Array>()
            .and_then(|a| serde_json::Number::from_f64(a.value(row) as f64).map(Value::Number))
            .unwrap_or(Value::Null),
        DataType::Float64 => array
            .as_any()
            .downcast_ref::<Float64Array>()
            .and_then(|a| serde_json::Number::from_f64(a.value(row)).map(Value::Number))
            .unwrap_or(Value::Null),
        _ => Value::Null,
    }
}

fn batch_rows(batch: &RecordBatch) -> Vec<Map<String, Value>> {
    let schema = batch.schema();
    let mut out = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        let mut m = Map::new();
        for (i, f) in schema.fields().iter().enumerate() {
            m.insert(
                f.name().clone(),
                cell_to_json(batch.column(i).as_ref(), row),
            );
        }
        out.push(m);
    }
    out
}

async fn run_sql(table: &Table, sql: &str) -> Result<Vec<RecordBatch>, BenoStreamError> {
    table.sql(sql).await.map_err(|e| bad(e.to_string()))
}

/// Run a single-row, single-column scalar query.
async fn scalar(table: &Table, sql: &str) -> Result<Value, BenoStreamError> {
    let batches = run_sql(table, sql).await?;
    for b in &batches {
        if b.num_rows() > 0 && b.num_columns() > 0 {
            return Ok(cell_to_json(b.column(0).as_ref(), 0));
        }
    }
    Ok(Value::Null)
}

fn as_f64(v: &Value) -> f64 {
    v.as_f64().unwrap_or(0.0)
}

/// Compute the `aggregations` object for a search request.
pub fn compute_aggregations<'a>(
    table: &'a Table,
    filter: Option<&'a str>,
    aggs: &'a Value,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value, BenoStreamError>> + Send + 'a>>
{
    // Boxed so the recursive bucket → nested-aggs cycle has a bounded future
    // size.
    Box::pin(compute_aggregations_inner(table, filter, aggs))
}

async fn compute_aggregations_inner(
    table: &Table,
    filter: Option<&str>,
    aggs: &Value,
) -> Result<Value, BenoStreamError> {
    let obj = aggs
        .as_object()
        .ok_or_else(|| bad("aggregations: expected an object"))?;
    let mut out = Map::new();
    for (name, spec) in obj {
        out.insert(name.clone(), compute_one(table, filter, spec).await?);
    }
    Ok(Value::Object(out))
}

async fn compute_one(
    table: &Table,
    filter: Option<&str>,
    spec: &Value,
) -> Result<Value, BenoStreamError> {
    let obj = spec
        .as_object()
        .ok_or_else(|| bad("aggregation: expected an object"))?;
    let nested = obj.get("aggs").or_else(|| obj.get("aggregations"));

    // Find the single aggregation type key (ignoring `aggs`/`meta`).
    let (kind, body) = obj
        .iter()
        .find(|(k, _)| k.as_str() != "aggs" && k.as_str() != "aggregations" && k.as_str() != "meta")
        .ok_or_else(|| bad("aggregation: missing type"))?;

    match kind.as_str() {
        "terms" => terms(table, filter, body, nested).await,
        "histogram" => histogram(table, filter, body, nested).await,
        "date_histogram" => date_histogram(table, filter, body, nested).await,
        "range" => range_agg(table, filter, body, nested).await,
        "filter" => filter_agg(table, filter, body, nested).await,
        "missing" => missing(table, filter, body, nested).await,
        "avg" | "sum" | "min" | "max" | "value_count" | "cardinality" => {
            metric(table, filter, kind.as_str(), body).await
        }
        "stats" | "extended_stats" => stats(table, filter, body).await,
        other => Err(bad(format!("unsupported aggregation type '{other}'"))),
    }
}

fn field_of(body: &Value) -> Result<String, BenoStreamError> {
    let f = body
        .get("field")
        .and_then(Value::as_str)
        .ok_or_else(|| bad("aggregation: 'field' must be a string"))?;
    valid_field(f)
}

async fn terms(
    table: &Table,
    filter: Option<&str>,
    body: &Value,
    nested: Option<&Value>,
) -> Result<Value, BenoStreamError> {
    let field = field_of(body)?;
    let size = body.get("size").and_then(Value::as_u64).unwrap_or(10) as usize;
    let sql = format!(
        "SELECT {field} AS key, COUNT(*) AS doc_count FROM t{} GROUP BY {field} ORDER BY doc_count DESC LIMIT {size}",
        where_clause(filter)
    );
    let batches = run_sql(table, &sql).await?;
    let mut buckets = Vec::new();
    for b in &batches {
        for row in batch_rows(b) {
            let key = row.get("key").cloned().unwrap_or(Value::Null);
            let doc_count = row.get("doc_count").cloned().unwrap_or(json!(0));
            let mut bucket = Map::new();
            bucket.insert("key".to_string(), key.clone());
            bucket.insert("doc_count".to_string(), doc_count);
            if let Some(n) = nested {
                let sub_filter = and_filter(filter, &format!("{field} = {}", sql_literal(&key)));
                bucket.insert(
                    "aggs".to_string(),
                    compute_aggregations(table, Some(&sub_filter), n).await?,
                );
            }
            buckets.push(Value::Object(bucket));
        }
    }
    Ok(json!({
        "doc_count_error_upper_bound": 0,
        "sum_other_doc_count": 0,
        "buckets": buckets,
    }))
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

async fn histogram(
    table: &Table,
    filter: Option<&str>,
    body: &Value,
    nested: Option<&Value>,
) -> Result<Value, BenoStreamError> {
    let field = field_of(body)?;
    let interval = body
        .get("interval")
        .and_then(Value::as_f64)
        .ok_or_else(|| bad("histogram: 'interval' must be a number"))?;
    if interval <= 0.0 {
        return Err(bad("histogram: 'interval' must be > 0"));
    }
    let sql = format!(
        "SELECT floor({field} / {interval}) * {interval} AS key, COUNT(*) AS doc_count FROM t{} GROUP BY key ORDER BY key",
        where_clause(filter)
    );
    let batches = run_sql(table, &sql).await?;
    let mut buckets = Vec::new();
    for b in &batches {
        for row in batch_rows(b) {
            let key = row.get("key").cloned().unwrap_or(Value::Null);
            let doc_count = row.get("doc_count").cloned().unwrap_or(json!(0));
            let mut bucket = Map::new();
            bucket.insert("key".to_string(), key.clone());
            bucket.insert("doc_count".to_string(), doc_count);
            if let Some(n) = nested {
                let sub_filter = and_filter(
                    filter,
                    &format!(
                        "{field} >= {} AND {field} < {}",
                        as_f64(&key),
                        as_f64(&key) + interval
                    ),
                );
                bucket.insert(
                    "aggs".to_string(),
                    compute_aggregations(table, Some(&sub_filter), n).await?,
                );
            }
            buckets.push(Value::Object(bucket));
        }
    }
    Ok(json!({ "buckets": buckets }))
}

async fn date_histogram(
    table: &Table,
    filter: Option<&str>,
    body: &Value,
    nested: Option<&Value>,
) -> Result<Value, BenoStreamError> {
    let field = field_of(body)?;
    let unit = body
        .get("calendar_interval")
        .or_else(|| body.get("fixed_interval"))
        .and_then(Value::as_str)
        .unwrap_or("day");
    let unit = match unit {
        "minute" | "1m" => "minute",
        "hour" | "1h" => "hour",
        "week" | "1w" => "week",
        "month" | "1M" => "month",
        "quarter" | "1q" => "quarter",
        "year" | "1y" => "year",
        _ => "day",
    };
    let sql = format!(
        "SELECT date_trunc('{unit}', {field}) AS key, COUNT(*) AS doc_count FROM t{} GROUP BY key ORDER BY key",
        where_clause(filter)
    );
    let batches = run_sql(table, &sql).await?;
    let mut buckets = Vec::new();
    for b in &batches {
        for row in batch_rows(b) {
            let mut bucket = Map::new();
            bucket.insert(
                "key".to_string(),
                row.get("key").cloned().unwrap_or(Value::Null),
            );
            bucket.insert(
                "doc_count".to_string(),
                row.get("doc_count").cloned().unwrap_or(json!(0)),
            );
            if let Some(n) = nested {
                // Nested aggs on a date bucket are not scoped (v1).
                bucket.insert(
                    "aggs".to_string(),
                    compute_aggregations(table, filter, n).await?,
                );
            }
            buckets.push(Value::Object(bucket));
        }
    }
    Ok(json!({ "buckets": buckets }))
}

async fn range_agg(
    table: &Table,
    filter: Option<&str>,
    body: &Value,
    nested: Option<&Value>,
) -> Result<Value, BenoStreamError> {
    let field = field_of(body)?;
    let ranges = body
        .get("ranges")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("range: 'ranges' must be an array"))?;
    let mut buckets = Vec::new();
    for r in ranges {
        let from = r.get("from").and_then(Value::as_f64);
        let to = r.get("to").and_then(Value::as_f64);
        let mut conds = Vec::new();
        if let Some(f) = from {
            conds.push(format!("{field} >= {f}"));
        }
        if let Some(t) = to {
            conds.push(format!("{field} < {t}"));
        }
        let cond = if conds.is_empty() {
            "true".to_string()
        } else {
            conds.join(" AND ")
        };
        let sub_filter = and_filter(filter, &cond);
        let count = scalar(
            table,
            &format!("SELECT COUNT(*) FROM t WHERE ({sub_filter})"),
        )
        .await?;
        let mut bucket = Map::new();
        if let Some(k) = r.get("key").and_then(Value::as_str) {
            bucket.insert("key".to_string(), Value::String(k.to_string()));
        } else {
            let key = match (from, to) {
                (Some(f), Some(t)) => format!("{f}-{t}"),
                (Some(f), None) => format!("{f}-*"),
                (None, Some(t)) => format!("*-{t}"),
                (None, None) => "*-*".to_string(),
            };
            bucket.insert("key".to_string(), Value::String(key));
        }
        if let Some(f) = from {
            bucket.insert("from".to_string(), json!(f));
        }
        if let Some(t) = to {
            bucket.insert("to".to_string(), json!(t));
        }
        bucket.insert("doc_count".to_string(), count);
        if let Some(n) = nested {
            bucket.insert(
                "aggs".to_string(),
                compute_aggregations(table, Some(&sub_filter), n).await?,
            );
        }
        buckets.push(Value::Object(bucket));
    }
    Ok(json!({ "buckets": buckets }))
}

async fn filter_agg(
    table: &Table,
    filter: Option<&str>,
    body: &Value,
    nested: Option<&Value>,
) -> Result<Value, BenoStreamError> {
    // `filter` agg body is itself a filter clause; translate it to SQL.
    let sub = crate::handlers::search::clause_to_sql(body, "filter")?;
    let sub_filter = and_filter(filter, &sub);
    let count = scalar(
        table,
        &format!("SELECT COUNT(*) FROM t WHERE ({sub_filter})"),
    )
    .await?;
    let mut out = Map::new();
    out.insert("doc_count".to_string(), count);
    if let Some(n) = nested {
        out.insert(
            "aggs".to_string(),
            compute_aggregations(table, Some(&sub_filter), n).await?,
        );
    }
    Ok(Value::Object(out))
}

async fn missing(
    table: &Table,
    filter: Option<&str>,
    body: &Value,
    nested: Option<&Value>,
) -> Result<Value, BenoStreamError> {
    let field = field_of(body)?;
    let sub_filter = and_filter(filter, &format!("{field} IS NULL"));
    let count = scalar(
        table,
        &format!("SELECT COUNT(*) FROM t WHERE ({sub_filter})"),
    )
    .await?;
    let mut out = Map::new();
    out.insert("doc_count".to_string(), count);
    if let Some(n) = nested {
        out.insert(
            "aggs".to_string(),
            compute_aggregations(table, Some(&sub_filter), n).await?,
        );
    }
    Ok(Value::Object(out))
}

async fn metric(
    table: &Table,
    filter: Option<&str>,
    kind: &str,
    body: &Value,
) -> Result<Value, BenoStreamError> {
    let field = field_of(body)?;
    let expr = match kind {
        "avg" => format!("AVG({field})"),
        "sum" => format!("SUM({field})"),
        "min" => format!("MIN({field})"),
        "max" => format!("MAX({field})"),
        "value_count" => format!("COUNT({field})"),
        "cardinality" => format!("COUNT(DISTINCT {field})"),
        _ => return Err(bad(format!("unsupported metric '{kind}'"))),
    };
    let v = scalar(
        table,
        &format!("SELECT {expr} AS value FROM t{}", where_clause(filter)),
    )
    .await?;
    Ok(json!({ "value": v }))
}

async fn stats(
    table: &Table,
    filter: Option<&str>,
    body: &Value,
) -> Result<Value, BenoStreamError> {
    let field = field_of(body)?;
    let sql = format!(
        "SELECT COUNT({field}) AS count, MIN({field}) AS min, MAX({field}) AS max, AVG({field}) AS avg, SUM({field}) AS sum, SUM({field} * {field}) AS sum_sq FROM t{}",
        where_clause(filter)
    );
    let batches = run_sql(table, &sql).await?;
    let row = batches
        .iter()
        .flat_map(batch_rows)
        .next()
        .unwrap_or_default();
    let count = as_f64(row.get("count").unwrap_or(&json!(0)));
    let sum = as_f64(row.get("sum").unwrap_or(&json!(0)));
    let sum_sq = as_f64(row.get("sum_sq").unwrap_or(&json!(0)));
    let variance = if count > 0.0 {
        (sum_sq / count) - (sum / count).powi(2)
    } else {
        0.0
    };
    Ok(json!({
        "count": count,
        "min": row.get("min").cloned().unwrap_or(Value::Null),
        "max": row.get("max").cloned().unwrap_or(Value::Null),
        "avg": row.get("avg").cloned().unwrap_or(Value::Null),
        "sum": sum,
        "sum_of_squares": sum_sq,
        "variance": variance,
        "std_deviation": variance.max(0.0).sqrt(),
    }))
}
