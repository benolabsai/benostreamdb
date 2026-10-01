// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! ES-compatible aggregations for `POST /{index}/_search`.
//!
//! Aggregations are compiled to SQL and executed with DataFusion through
//! [`Table::sql`] (the table is registered as `t`). Supported:
//!
//! - Bucket: `terms`, `histogram`, `date_histogram`, `range`, `filter`,
//!   `missing`
//! - Metric: `avg`, `sum`, `min`, `max`, `value_count`, `cardinality`,
//!   `stats`, `extended_stats`, `percentiles`
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
        "percentiles" => percentiles(table, filter, body).await,
        "composite" => composite(table, filter, body, nested).await,
        "significant_terms" => significant_terms(table, filter, body, nested).await,
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

async fn percentiles(
    table: &Table,
    filter: Option<&str>,
    body: &Value,
) -> Result<Value, BenoStreamError> {
    let field = field_of(body)?;
    let keyed = body.get("keyed").and_then(Value::as_bool).unwrap_or(true);

    let default_percents = [1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0];
    let percents: Vec<f64> = match body.get("percents").and_then(Value::as_array) {
        Some(arr) => {
            let mut list = Vec::with_capacity(arr.len());
            for v in arr {
                let p = v
                    .as_f64()
                    .ok_or_else(|| bad("percentiles: 'percents' items must be numbers"))?;
                if !(0.0..=100.0).contains(&p) {
                    return Err(bad("percentiles: percent must be between 0.0 and 100.0"));
                }
                list.push(p);
            }
            list
        }
        None => default_percents.to_vec(),
    };

    if percents.is_empty() {
        return Ok(if keyed {
            json!({ "values": {} })
        } else {
            json!({ "values": [] })
        });
    }

    let cols: Vec<String> = percents
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            let frac = (p / 100.0).clamp(0.0, 1.0);
            format!("approx_percentile_cont({field}, {frac}) AS p{i}")
        })
        .collect();

    let sql = format!("SELECT {} FROM t{}", cols.join(", "), where_clause(filter));

    let batches = run_sql(table, &sql).await?;
    let row = batches
        .iter()
        .flat_map(batch_rows)
        .next()
        .unwrap_or_default();

    if keyed {
        let mut values_map = Map::new();
        for (i, &p) in percents.iter().enumerate() {
            let col_name = format!("p{i}");
            let val = row.get(&col_name).cloned().unwrap_or(Value::Null);
            let p_key = if p.fract() == 0.0 {
                format!("{p:.1}")
            } else {
                format!("{p}")
            };
            values_map.insert(p_key, val);
        }
        Ok(json!({ "values": values_map }))
    } else {
        let mut values_list = Vec::with_capacity(percents.len());
        for (i, &p) in percents.iter().enumerate() {
            let col_name = format!("p{i}");
            let val = row.get(&col_name).cloned().unwrap_or(Value::Null);
            values_list.push(json!({
                "key": p,
                "value": val,
            }));
        }
        Ok(json!({ "values": values_list }))
    }
}

struct CompositeSource {
    name: String,
    expr: String,
    is_asc: bool,
    missing_bucket: bool,
}

async fn composite(
    table: &Table,
    filter: Option<&str>,
    body: &Value,
    nested: Option<&Value>,
) -> Result<Value, BenoStreamError> {
    let sources_array = body
        .get("sources")
        .and_then(Value::as_array)
        .ok_or_else(|| bad("composite: 'sources' must be an array"))?;
    if sources_array.is_empty() {
        return Err(bad("composite: 'sources' must be a non-empty array"));
    }

    let size = body.get("size").and_then(Value::as_u64).unwrap_or(10) as usize;

    let mut sources = Vec::with_capacity(sources_array.len());
    for item in sources_array {
        let obj = item
            .as_object()
            .ok_or_else(|| bad("composite: each source must be an object"))?;
        if obj.len() != 1 {
            return Err(bad(
                "composite: each source object must have exactly one name key",
            ));
        }
        let (name, spec) = obj
            .iter()
            .next()
            .ok_or_else(|| bad("composite: each source object must have exactly one name key"))?;
        let spec_obj = spec
            .as_object()
            .ok_or_else(|| bad(format!("composite source '{name}': expected an object")))?;
        if spec_obj.len() != 1 {
            return Err(bad(format!(
                "composite source '{name}': expected exactly one source type"
            )));
        }
        let (stype, sbody) = spec_obj.iter().next().ok_or_else(|| {
            bad(format!(
                "composite source '{name}': expected exactly one source type"
            ))
        })?;
        let order_str = sbody.get("order").and_then(Value::as_str).unwrap_or("asc");
        let is_asc = match order_str.to_ascii_lowercase().as_str() {
            "asc" => true,
            "desc" => false,
            other => {
                return Err(bad(format!(
                    "composite source '{name}': unsupported order '{other}'"
                )))
            }
        };
        let missing_bucket = sbody
            .get("missing_bucket")
            .and_then(Value::as_bool)
            .unwrap_or(false);

        let expr = match stype.as_str() {
            "terms" => field_of(sbody)?,
            "histogram" => {
                let field = field_of(sbody)?;
                let interval = sbody
                    .get("interval")
                    .and_then(Value::as_f64)
                    .ok_or_else(|| {
                        bad(format!(
                            "composite source '{name}': histogram 'interval' must be a number"
                        ))
                    })?;
                if interval <= 0.0 {
                    return Err(bad(format!(
                        "composite source '{name}': histogram 'interval' must be > 0"
                    )));
                }
                format!("floor({field} / {interval}) * {interval}")
            }
            "date_histogram" => {
                let field = field_of(sbody)?;
                let unit = sbody
                    .get("calendar_interval")
                    .or_else(|| sbody.get("fixed_interval"))
                    .or_else(|| sbody.get("interval"))
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
                format!("date_trunc('{unit}', {field})")
            }
            other => {
                return Err(bad(format!(
                    "composite source '{name}': unsupported source type '{other}'"
                )))
            }
        };

        sources.push(CompositeSource {
            name: name.clone(),
            expr,
            is_asc,
            missing_bucket,
        });
    }

    let mut extra_conditions = Vec::new();

    // Enforce NOT NULL for sources that do not allow missing_bucket
    for s in &sources {
        if !s.missing_bucket {
            extra_conditions.push(format!("{} IS NOT NULL", s.expr));
        }
    }

    // Handle `after` pagination cursor if present
    if let Some(after_obj) = body.get("after").and_then(Value::as_object) {
        let mut prefix_len = 0;
        for s in &sources {
            if after_obj.contains_key(&s.name) {
                prefix_len += 1;
            } else {
                break;
            }
        }

        if prefix_len > 0 {
            let mut disjunctions = Vec::with_capacity(prefix_len);
            for i in 0..prefix_len {
                let mut conjunction = Vec::with_capacity(i + 1);
                for s_j in sources.iter().take(i) {
                    let val_j = &after_obj[&s_j.name];
                    if val_j.is_null() {
                        conjunction.push(format!("{} IS NULL", s_j.expr));
                    } else {
                        conjunction.push(format!("{} = {}", s_j.expr, sql_literal(val_j)));
                    }
                }
                let s_i = &sources[i];
                let val_i = &after_obj[&s_i.name];
                let cmp = if s_i.is_asc { ">" } else { "<" };
                if val_i.is_null() {
                    conjunction.push(format!("{} IS NOT NULL", s_i.expr));
                } else {
                    conjunction.push(format!("{} {} {}", s_i.expr, cmp, sql_literal(val_i)));
                }
                disjunctions.push(format!("({})", conjunction.join(" AND ")));
            }
            extra_conditions.push(format!("({})", disjunctions.join(" OR ")));
        }
    }

    let mut combined_filter = filter.map(|f| f.to_string());
    for cond in extra_conditions {
        combined_filter = Some(match combined_filter {
            Some(curr) if !curr.trim().is_empty() => format!("({curr}) AND ({cond})"),
            _ => cond,
        });
    }

    let select_cols: Vec<String> = sources
        .iter()
        .enumerate()
        .map(|(i, s)| format!("{} AS k_{i}", s.expr))
        .collect();

    let group_by_exprs: Vec<String> = sources.iter().map(|s| s.expr.clone()).collect();

    let order_by_cols: Vec<String> = sources
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let ord = if s.is_asc { "ASC" } else { "DESC" };
            format!("k_{i} {ord}")
        })
        .collect();

    let sql = format!(
        "SELECT {}, COUNT(*) AS doc_count FROM t{} GROUP BY {} ORDER BY {} LIMIT {size}",
        select_cols.join(", "),
        where_clause(combined_filter.as_deref()),
        group_by_exprs.join(", "),
        order_by_cols.join(", ")
    );

    let batches = run_sql(table, &sql).await?;
    let mut buckets = Vec::new();
    for b in &batches {
        for row in batch_rows(b) {
            let mut key_map = Map::new();
            let mut bucket_filter_parts = Vec::with_capacity(sources.len());
            for (i, s) in sources.iter().enumerate() {
                let k_name = format!("k_{i}");
                let k_val = row.get(&k_name).cloned().unwrap_or(Value::Null);
                if k_val.is_null() {
                    bucket_filter_parts.push(format!("{} IS NULL", s.expr));
                } else {
                    bucket_filter_parts.push(format!("{} = {}", s.expr, sql_literal(&k_val)));
                }
                key_map.insert(s.name.clone(), k_val);
            }

            let doc_count = row.get("doc_count").cloned().unwrap_or(json!(0));
            let mut bucket = Map::new();
            bucket.insert("key".to_string(), Value::Object(key_map));
            bucket.insert("doc_count".to_string(), doc_count);

            if let Some(n) = nested {
                let bucket_cond = bucket_filter_parts.join(" AND ");
                let sub_filter = and_filter(filter, &bucket_cond);
                bucket.insert(
                    "aggs".to_string(),
                    compute_aggregations(table, Some(&sub_filter), n).await?,
                );
            }
            buckets.push(Value::Object(bucket));
        }
    }

    let mut out = Map::new();
    if let Some(last) = buckets.last() {
        if let Some(key) = last.get("key") {
            out.insert("after_key".to_string(), key.clone());
        }
    }
    out.insert("buckets".to_string(), Value::Array(buckets));
    Ok(Value::Object(out))
}

async fn significant_terms(
    table: &Table,
    filter: Option<&str>,
    body: &Value,
    nested: Option<&Value>,
) -> Result<Value, BenoStreamError> {
    let field = field_of(body)?;
    let size = body.get("size").and_then(Value::as_u64).unwrap_or(10) as usize;
    let min_doc_count = body
        .get("min_doc_count")
        .and_then(Value::as_u64)
        .unwrap_or(1) as usize;

    let bg_count_val = scalar(table, "SELECT COUNT(*) FROM t").await?;
    let bg_total = bg_count_val.as_u64().unwrap_or(0);
    if bg_total == 0 {
        return Ok(json!({
            "doc_count": 0,
            "bg_count": 0,
            "buckets": []
        }));
    }

    let fg_total = if let Some(f) = filter {
        if !f.trim().is_empty() {
            let fg_val = scalar(table, &format!("SELECT COUNT(*) FROM t WHERE ({f})")).await?;
            fg_val.as_u64().unwrap_or(0)
        } else {
            bg_total
        }
    } else {
        bg_total
    };

    if fg_total == 0 {
        return Ok(json!({
            "doc_count": 0,
            "bg_count": bg_total,
            "buckets": []
        }));
    }

    let fg_sql = format!(
        "SELECT {field} AS key, COUNT(*) AS fg_count FROM t{} GROUP BY {field} HAVING COUNT(*) >= {min_doc_count}",
        where_clause(filter)
    );
    let fg_batches = run_sql(table, &fg_sql).await?;
    let mut fg_items = Vec::new();
    for b in &fg_batches {
        for row in batch_rows(b) {
            let key = row.get("key").cloned().unwrap_or(Value::Null);
            if !key.is_null() {
                let fg_count = row.get("fg_count").and_then(Value::as_u64).unwrap_or(0);
                fg_items.push((key, fg_count));
            }
        }
    }

    if fg_items.is_empty() {
        return Ok(json!({
            "doc_count": fg_total,
            "bg_count": bg_total,
            "buckets": []
        }));
    }

    let key_literals: Vec<String> = fg_items.iter().map(|(k, _)| sql_literal(k)).collect();
    let bg_terms_sql = format!(
        "SELECT {field} AS key, COUNT(*) AS bg_count FROM t WHERE {field} IN ({}) GROUP BY {field}",
        key_literals.join(", ")
    );
    let bg_batches = run_sql(table, &bg_terms_sql).await?;
    let mut bg_map = std::collections::HashMap::new();
    for b in &bg_batches {
        for row in batch_rows(b) {
            let key = row.get("key").cloned().unwrap_or(Value::Null);
            let bg_count = row.get("bg_count").and_then(Value::as_u64).unwrap_or(0);
            bg_map.insert(key.to_string(), bg_count);
        }
    }

    struct ScoredTerm {
        key: Value,
        fg_count: u64,
        bg_count: u64,
        score: f64,
    }

    let mut scored = Vec::with_capacity(fg_items.len());
    let fg_total_f = fg_total as f64;
    let bg_total_f = bg_total as f64;

    for (key, fg_count) in fg_items {
        let bg_count = bg_map.get(&key.to_string()).copied().unwrap_or(fg_count);
        let p_fg = (fg_count as f64) / fg_total_f;
        let p_bg = (bg_count as f64) / bg_total_f;

        let score = if p_fg > p_bg && p_bg > 0.0 && p_bg < 1.0 {
            ((p_fg - p_bg) / (1.0 - p_bg)) * (p_fg / p_bg)
        } else {
            0.0
        };

        scored.push(ScoredTerm {
            key,
            fg_count,
            bg_count,
            score,
        });
    }

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(size);

    let mut buckets = Vec::with_capacity(scored.len());
    for item in scored {
        let mut bucket = Map::new();
        bucket.insert("key".to_string(), item.key.clone());
        bucket.insert("doc_count".to_string(), json!(item.fg_count));
        bucket.insert("score".to_string(), json!(item.score));
        bucket.insert("bg_count".to_string(), json!(item.bg_count));

        if let Some(n) = nested {
            let sub_filter = and_filter(filter, &format!("{field} = {}", sql_literal(&item.key)));
            bucket.insert(
                "aggs".to_string(),
                compute_aggregations(table, Some(&sub_filter), n).await?,
            );
        }
        buckets.push(Value::Object(bucket));
    }

    Ok(json!({
        "doc_count": fg_total,
        "bg_count": bg_total,
        "buckets": buckets
    }))
}
