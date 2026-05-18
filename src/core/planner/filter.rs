// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Query filter types, SQL parsing, and column extraction.

use crate::core::manifest::{ManifestEntry, IndexFile};
use serde_json::Value;
use std::cmp::Ordering;
use datafusion::logical_expr::Expr;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::prelude::SessionContext;

/// Represents a filter predicate.
/// For MVP, we support simple Range filters on a single column.
/// e.g. "year = 2022" or "year >= 2020 AND year < 2023"
#[derive(Debug, Clone)]
pub struct QueryFilter {
    pub column: String,
    pub min: Option<Value>,
    pub min_inclusive: bool,
    pub max: Option<Value>,
    pub max_inclusive: bool,
    /// Exact match against a list of values (e.g. IN clause)
    pub values: Option<Vec<Value>>,
    /// Whether the condition is negated (e.g. NOT IN, NOT EQUAL)
    pub negated: bool,
}

#[derive(Debug, Clone)]
pub enum FilterExpr {
    DataFusion(Expr),
}

impl FilterExpr {
    /// Parse a SQL WHERE clause into a filter expression.
    ///
    /// Uses DataFusion's SQL parser and analyzer to handle type coercion.
    ///
    /// # Errors
    /// Returns an error if the SQL is malformed or type coercion fails.
    pub async fn parse_sql(filter: &str, schema: SchemaRef) -> anyhow::Result<Self> {
        use datafusion::sql::TableReference;

        let sql = format!("SELECT * FROM t WHERE {}", filter);

        // Normalize schema to use standard Utf8 for string columns to avoid Utf8/LargeUtf8 confusion
        let normalized_fields: Vec<arrow::datatypes::Field> = schema.fields().iter().map(|f| {
            if let arrow::datatypes::DataType::LargeUtf8 = f.data_type() {
                let mut nf = f.as_ref().clone();
                nf.set_data_type(arrow::datatypes::DataType::Utf8);
                nf
            } else {
                f.as_ref().clone()
            }
        }).collect();
        let normalized_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(normalized_fields));

        let mut ctx = SessionContext::new();
        let _ = crate::core::sql::vector_operators::register_vector_operators(&mut ctx);
        let table = datafusion::datasource::empty::EmptyTable::new(normalized_schema);
        ctx.register_table(TableReference::bare("t"), std::sync::Arc::new(table))?;
        let df = ctx.sql(&sql).await?;
        let plan = df.logical_plan();

        // Apply type coercion via the analyzer (handles Int32 vs Int64 mismatches, etc.)
        // but don't run the full optimizer (which pushes filters into TableScan and breaks evaluate_expr)
        let state = ctx.state();
        let analyzed_plan = state.analyzer().execute_and_check(plan.clone(), state.config_options(), |_, _| {})?;

        use datafusion::logical_expr::LogicalPlan;

        fn find_filter(plan: &LogicalPlan) -> Option<datafusion::logical_expr::Expr> {
            match plan {
                LogicalPlan::Filter(f) => Some(f.predicate.clone()),
                LogicalPlan::Projection(p) => find_filter(&p.input),
                _ => {
                    for input in plan.inputs() {
                        if let Some(f) = find_filter(input) {
                            return Some(f);
                        }
                    }
                    None
                }
            }
        }

        if let Some(expr) = find_filter(&analyzed_plan) {
            return Ok(FilterExpr::DataFusion(expr));
        }

        Err(anyhow::anyhow!("Failed to parse filter expression: '{}'", filter))
    }

    /// Extract all columns referenced in the expression
    pub fn required_columns(&self) -> Vec<String> {
        match self {
            FilterExpr::DataFusion(expr) => {
                let mut cols = Vec::new();
                fn extract(expr: &datafusion::logical_expr::Expr, cols: &mut Vec<String>) {
                    match expr {
                        datafusion::logical_expr::Expr::Column(c) => {
                            if !cols.contains(&c.name) {
                                cols.push(c.name.clone());
                            }
                        }
                        datafusion::logical_expr::Expr::BinaryExpr(b) => {
                            extract(&b.left, cols);
                            extract(&b.right, cols);
                        }
                        datafusion::logical_expr::Expr::Not(e) => extract(e, cols),
                        datafusion::logical_expr::Expr::IsNotNull(e) => extract(e, cols),
                        datafusion::logical_expr::Expr::IsNull(e) => extract(e, cols),
                        datafusion::logical_expr::Expr::InList(l) => {
                            extract(&l.expr, cols);
                        }
                        // Add more as needed
                        _ => {}
                    }
                }
                extract(expr, &mut cols);
                cols
            }
        }
    }

    /// Convert legacy `Vec<QueryFilter>` to FilterExpr.
    ///
    /// Combines multiple filters with AND logic. Returns `None` if the input is empty.
    pub fn from_filters(filters: Vec<QueryFilter>) -> Option<Self> {
        if filters.is_empty() { return None; }

        let mut expr = filters[0].to_expr();
        for f in filters.into_iter().skip(1) {
            expr = expr.and(f.to_expr());
        }
        Some(FilterExpr::DataFusion(expr))
    }

    /// Extract a flat list of AND-ed conditions if possible (for clustering/fine-pruning)
    /// This is used to maintain our custom index/clustering logic.
    pub fn extract_and_conditions(&self) -> Vec<QueryFilter> {
        match self {
            FilterExpr::DataFusion(expr) => {
                let mut filters = Vec::new();
                extract_filters_from_expr(expr, &mut filters);
                filters
            }
        }
    }

    /// Recursively find all columns referenced in the filter
    pub fn get_referenced_columns(&self) -> std::collections::HashSet<String> {
        match self {
            FilterExpr::DataFusion(expr) => {
                let mut cols = std::collections::HashSet::new();
                find_column_names(expr, &mut cols);
                cols
            }
        }
    }
}

/// Internal helper to recursively find column names in an Expr
fn find_column_names(expr: &Expr, cols: &mut std::collections::HashSet<String>) {
    match expr {
        Expr::Column(c) => { cols.insert(c.name.clone()); },
        Expr::BinaryExpr(b) => {
            find_column_names(&b.left, cols);
            find_column_names(&b.right, cols);
        },
        Expr::Not(e) => find_column_names(e, cols),
        Expr::IsNotNull(e) => find_column_names(e, cols),
        Expr::IsNull(e) => find_column_names(e, cols),
        Expr::Cast(c) => find_column_names(&c.expr, cols),
        Expr::TryCast(c) => find_column_names(&c.expr, cols),
        Expr::InList(in_list) => {
            find_column_names(&in_list.expr, cols);
            for e in &in_list.list {
                find_column_names(e, cols);
            }
        },
        _ => {}
    }
}

impl QueryFilter {
    /// Parse a simple filter string into a QueryFilter.
    ///
    /// Expected format: `"column op value"` where `op` is one of `=`, `==`, `>`, `>=`, `<`, `<=`.
    /// Returns `None` if the format is invalid.
    ///
    /// # Examples
    /// ```ignore
    /// let filter = QueryFilter::parse("temperature > 100").unwrap();
    /// ```
    pub fn parse(filter: &str) -> Option<Self> {
        // Very simple/naive parser for legacy support where needed.
        // Format expected: "column op value"
        let parts: Vec<&str> = filter.split_whitespace().collect();
        if parts.len() == 3 {
             let col = parts[0].to_string();
             let op = parts[1];
             let val_str = parts[2];

             let val = if let Ok(i) = val_str.parse::<i64>() {
                 Value::Number(i.into())
             } else if let Ok(f) = val_str.parse::<f64>() {
                 Value::from(f)
             } else {
                 Value::String(val_str.trim_matches('\'').trim_matches('"').to_string())
             };

             match op {
                 "=" | "==" => Some(QueryFilter {
                     column: col,
                     min: Some(val.clone()),
                     min_inclusive: true,
                     max: Some(val),
                     max_inclusive: true,
                     values: None,
                     negated: false,
                 }),
                 ">" => Some(QueryFilter {
                     column: col,
                     min: Some(val),
                     min_inclusive: false,
                     max: None,
                     max_inclusive: true,
                     values: None,
                     negated: false,
                 }),
                 ">=" => Some(QueryFilter {
                     column: col,
                     min: Some(val),
                     min_inclusive: true,
                     max: None,
                     max_inclusive: true,
                     values: None,
                     negated: false,
                 }),
                 "<" => Some(QueryFilter {
                     column: col,
                     min: None,
                     min_inclusive: true,
                     max: Some(val),
                     max_inclusive: false,
                     values: None,
                     negated: false,
                 }),
                 "<=" => Some(QueryFilter {
                     column: col,
                     min: None,
                     min_inclusive: true,
                     max: Some(val),
                     max_inclusive: true,
                     values: None,
                     negated: false,
                 }),
                 _ => None,
             }
        } else {
            None
        }
    }

    pub fn parse_multi(filter: &str) -> Vec<Self> {
        // Handle "A = 1 AND B = 2"
        filter.split(" AND ")
            .filter_map(|s| Self::parse(s.trim()))
            .collect()
    }

    pub fn to_expr(&self) -> Expr {
        use datafusion::prelude::*;

        let col_expr = col(&self.column);

        let expr = if let Some(values) = &self.values {
             if values.len() == 1 {
                 if self.negated {
                     col_expr.not_eq(json_to_scalar(&values[0]))
                 } else {
                     col_expr.eq(json_to_scalar(&values[0]))
                 }
             } else {
                 let list = values.iter().map(json_to_scalar).collect();
                 if self.negated {
                     col_expr.in_list(list, true)
                 } else {
                     col_expr.in_list(list, false)
                 }
             }
        } else {
            // Range
            let mut range_expr = None;
            if let Some(min) = &self.min {
                let e = if self.min_inclusive {
                    col_expr.clone().gt_eq(json_to_scalar(min))
                } else {
                    col_expr.clone().gt(json_to_scalar(min))
                };
                range_expr = Some(e);
            }
            if let Some(max) = &self.max {
                let e = if self.max_inclusive {
                    col_expr.clone().lt_eq(json_to_scalar(max))
                } else {
                    col_expr.clone().lt(json_to_scalar(max))
                };
                if let Some(prev) = range_expr {
                    range_expr = Some(prev.and(e));
                } else {
                    range_expr = Some(e);
                }
            }

            let res = range_expr.unwrap_or(lit(true));
            if self.negated {
                res.not()
            } else {
                res
            }
        };

        expr
    }

    pub fn op_to_string(&self) -> String {
        if self.values.is_some() {
             if self.negated { "NOT IN".to_string() } else { "IN".to_string() }
        } else if self.min.is_some() && self.max.is_some() {
             if self.min == self.max {
                 if self.negated { "!=".to_string() } else { "=".to_string() }
             } else {
                 "RANGE".to_string()
             }
        } else if self.min.is_some() {
             if self.min_inclusive { ">=".to_string() } else { ">".to_string() }
        } else if self.max.is_some() {
             if self.max_inclusive { "<=".to_string() } else { "<".to_string() }
        } else {
             "TRUE".to_string()
        }
    }
}

fn json_to_scalar(v: &Value) -> Expr {
    use datafusion::prelude::lit;
    match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                lit(i)
            }
            else { lit(n.as_f64().unwrap_or(0.0)) }
        }
        Value::String(s) => lit(s.clone()),
        Value::Bool(b) => lit(*b),
        _ => lit(v.to_string()),
    }
}

fn extract_filters_from_expr(expr: &Expr, filters: &mut Vec<QueryFilter>) {
    match expr {
        Expr::BinaryExpr(binary) => {
            if binary.op == datafusion::logical_expr::Operator::And {
                extract_filters_from_expr(&binary.left, filters);
                extract_filters_from_expr(&binary.right, filters);
            } else if let Some(f) = convert_binary_expr_to_query_filter(binary) {
                filters.push(f);
            }
        }
        Expr::InList(in_list) => {
             if let Some(f) = convert_in_list_to_query_filter(in_list) {
                 filters.push(f);
             }
        }
        _ => {} // Other expressions can't be easily converted to our QueryFilter leaf
    }
}

pub(super) fn convert_binary_expr_to_query_filter(binary: &datafusion::logical_expr::BinaryExpr) -> Option<QueryFilter> {
    // Helper to strip casts and find column name
    fn get_col(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Column(c) => Some(c.name.clone()),
            Expr::Cast(cast) => get_col(&cast.expr),
            Expr::TryCast(cast) => get_col(&cast.expr),
            _ => None,
        }
    }

    fn get_lit(expr: &Expr) -> Option<Value> {
        match expr {
            Expr::Literal(scalar, _) => scalar_to_json_value(scalar),
            Expr::Cast(cast) => get_lit(&cast.expr),
            Expr::TryCast(cast) => get_lit(&cast.expr),
            _ => None,
        }
    }

    let (col, val, op) = if let Some(c) = get_col(&binary.left) {
        if let Some(v) = get_lit(&binary.right) {
            (c, v, binary.op)
        } else { return None; }
    } else if let Some(c) = get_col(&binary.right) {
        if let Some(v) = get_lit(&binary.left) {
            // Swap operator if literal is on the left
            use datafusion::logical_expr::Operator;
            let swapped_op = match binary.op {
                Operator::Eq => Operator::Eq,
                Operator::NotEq => Operator::NotEq,
                Operator::Gt => Operator::Lt,
                Operator::GtEq => Operator::LtEq,
                Operator::Lt => Operator::Gt,
                Operator::LtEq => Operator::GtEq,
                _ => return None,
            };
            (c, v, swapped_op)
        } else { return None; }
    } else {
        return None;
    };

    match op {
        datafusion::logical_expr::Operator::Eq => Some(QueryFilter {
            column: col,
            min: Some(val.clone()),
            min_inclusive: true,
            max: Some(val.clone()),
            max_inclusive: true,
            values: Some(vec![val]),
            negated: false,
        }),
        datafusion::logical_expr::Operator::NotEq => Some(QueryFilter {
            column: col,
            min: Some(val.clone()),
            min_inclusive: true,
            max: Some(val.clone()),
            max_inclusive: true,
            values: Some(vec![val]),
            negated: true,
        }),
        datafusion::logical_expr::Operator::Gt => Some(QueryFilter {
            column: col,
            min: Some(val),
            min_inclusive: false,
            max: None,
            max_inclusive: true,
            values: None,
            negated: false,
        }),
        datafusion::logical_expr::Operator::GtEq => Some(QueryFilter {
            column: col,
            min: Some(val),
            min_inclusive: true,
            max: None,
            max_inclusive: true,
            values: None,
            negated: false,
        }),
        datafusion::logical_expr::Operator::Lt => Some(QueryFilter {
            column: col,
            min: None,
            min_inclusive: true,
            max: Some(val),
            max_inclusive: false,
            values: None,
            negated: false,
        }),
        datafusion::logical_expr::Operator::LtEq => Some(QueryFilter {
            column: col,
            min: None,
            min_inclusive: true,
            max: Some(val),
            max_inclusive: true,
            values: None,
            negated: false,
        }),
        _ => None,
    }
}

pub(super) fn convert_in_list_to_query_filter(in_list: &datafusion::logical_expr::expr::InList) -> Option<QueryFilter> {
    let col = match &*in_list.expr {
        Expr::Column(c) => c.name.clone(),
        _ => return None,
    };

    let mut values = Vec::new();
    for v_expr in &in_list.list {
        if let Expr::Literal(scalar, _) = v_expr {
            if let Some(v) = scalar_to_json_value(scalar) {
                values.push(v);
            }
        }
    }

    if values.is_empty() { return None; }

    Some(QueryFilter {
        column: col,
        min: None,
        min_inclusive: false,
        max: None,
        max_inclusive: false,
        values: Some(values),
        negated: in_list.negated,
    })
}

pub(super) fn json_value_to_scalar(v: &Value, dt: &arrow::datatypes::DataType) -> anyhow::Result<datafusion::scalar::ScalarValue> {
    use datafusion::scalar::ScalarValue;
    use arrow::datatypes::DataType;

    match dt {
        DataType::Int64 => {
             let val = v.as_i64().or_else(|| v.as_f64().map(|f| f as i64));
             Ok(ScalarValue::Int64(val))
        },
        DataType::Int32 => {
             let val = v.as_i64().map(|i| i as i32).or_else(|| v.as_f64().map(|f| f as i32));
             Ok(ScalarValue::Int32(val))
        },
        DataType::Float64 => {
             Ok(ScalarValue::Float64(v.as_f64()))
        },
        DataType::Float32 => {
             Ok(ScalarValue::Float32(v.as_f64().map(|f| f as f32)))
        },
        DataType::Utf8 | DataType::LargeUtf8 => {
             Ok(ScalarValue::Utf8(v.as_str().map(|s| s.to_string())))
        },
        DataType::Boolean => {
             Ok(ScalarValue::Boolean(v.as_bool()))
        },
        _ => Err(anyhow::anyhow!("Unsupported type for filter: {:?}", dt)),
    }
}

fn scalar_to_json_value(scalar: &datafusion::scalar::ScalarValue) -> Option<Value> {
    use datafusion::scalar::ScalarValue;
    match scalar {
        ScalarValue::Int64(Some(i)) => Some(serde_json::json!(i)),
        ScalarValue::Int32(Some(i)) => Some(serde_json::json!(i)),
        ScalarValue::Int16(Some(i)) => Some(serde_json::json!(i)),
        ScalarValue::Int8(Some(i)) => Some(serde_json::json!(i)),
        ScalarValue::UInt64(Some(i)) => Some(serde_json::json!(i)),
        ScalarValue::UInt32(Some(i)) => Some(serde_json::json!(i)),
        ScalarValue::Float64(Some(f)) => Some(serde_json::json!(f)),
        ScalarValue::Float32(Some(f)) => Some(serde_json::json!(f)),
        ScalarValue::Utf8(Some(s)) => Some(serde_json::json!(s)),
        ScalarValue::Boolean(Some(b)) => Some(serde_json::json!(b)),
        _ => None,
    }
}
