// Copyright (c) 2026 Richard Albright. All rights reserved.

use crate::core::manifest::{ManifestEntry, IndexFile};
use serde_json::Value;
use std::cmp::Ordering;
use std::sync::Arc;
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
        let normalized_schema = Arc::new(arrow::datatypes::Schema::new(normalized_fields));

        let ctx = SessionContext::new();
        let table = datafusion::datasource::empty::EmptyTable::new(normalized_schema);
        ctx.register_table(TableReference::bare("t"), Arc::new(table))?;
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

    /// Convert legacy Vec<QueryFilter> to FilterExpr
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
}

impl QueryFilter {
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
}

fn json_to_scalar(v: &Value) -> Expr {
    use datafusion::prelude::lit;
    match v {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() { 
                if i >= i32::MIN as i64 && i <= i32::MAX as i64 {
                    lit(i as i32)
                } else {
                    lit(i)
                }
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

fn convert_binary_expr_to_query_filter(binary: &datafusion::logical_expr::BinaryExpr) -> Option<QueryFilter> {
    let col = match &*binary.left {
        Expr::Column(c) => c.name.clone(),
        _ => return None,
    };

    let val = match &*binary.right {
        Expr::Literal(scalar, _) => scalar_to_json_value(scalar)?,
        _ => return None,
    };

    match binary.op {
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

fn convert_in_list_to_query_filter(in_list: &datafusion::logical_expr::expr::InList) -> Option<QueryFilter> {
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

fn json_value_to_scalar(v: &Value, dt: &arrow::datatypes::DataType) -> anyhow::Result<datafusion::scalar::ScalarValue> {
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

pub struct QueryPlanner {}

impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryPlanner {
    pub fn new() -> Self {
        Self {}
    }

    /// Evaluate filter on a RecordBatch and return resulting RecordBatch
    pub fn filter_batch(&self, batch: &arrow::record_batch::RecordBatch, filter: &QueryFilter) -> anyhow::Result<arrow::record_batch::RecordBatch> {
        let mask = self.evaluate_condition(batch, filter)?;
        let filtered = arrow::compute::filter_record_batch(batch, &mask)?;
        Ok(filtered)
    }

    /// Evaluate filter expression on a RecordBatch and return the resulting filtered RecordBatch
    pub fn filter_expr(&self, batch: &arrow::record_batch::RecordBatch, expr: &FilterExpr) -> anyhow::Result<arrow::record_batch::RecordBatch> {
        let mask = self.evaluate_expr(batch, expr)?;
        let filtered = arrow::compute::filter_record_batch(batch, &mask)?;
        Ok(filtered)
    }

    pub fn evaluate_expr(&self, batch: &arrow::record_batch::RecordBatch, expr: &FilterExpr) -> anyhow::Result<arrow::array::BooleanArray> {
        let FilterExpr::DataFusion(df_expr) = expr;

        use datafusion::physical_expr::create_physical_expr;
        use datafusion::prelude::SessionContext;

        let ctx = SessionContext::new();
        let state = ctx.state();
        
        // Type Coercion: DataFusion sometimes struggles with LargeUtf8 vs Utf8 in direct physical expr evaluation.
        // We ensure the batch schema matches what's expected or coerce it.
        let mut coerced_batch = batch.clone();
        let mut coerced_fields = Vec::new();
        let mut coerced_columns = Vec::new();
        let mut changed = false;

        for (i, field) in batch.schema().fields().iter().enumerate() {
            if let arrow::datatypes::DataType::LargeUtf8 = field.data_type() {
                let casted = arrow::compute::cast(batch.column(i), &arrow::datatypes::DataType::Utf8)?;
                coerced_columns.push(casted);
                let mut new_field = field.as_ref().clone();
                new_field.set_data_type(arrow::datatypes::DataType::Utf8);
                coerced_fields.push(Arc::new(new_field));
                changed = true;
            } else if let arrow::datatypes::DataType::LargeBinary = field.data_type() {
                let casted = arrow::compute::cast(batch.column(i), &arrow::datatypes::DataType::Binary)?;
                coerced_columns.push(casted);
                let mut new_field = field.as_ref().clone();
                new_field.set_data_type(arrow::datatypes::DataType::Binary);
                coerced_fields.push(Arc::new(new_field));
                changed = true;
            } else {
                coerced_columns.push(batch.column(i).clone());
                coerced_fields.push(field.clone());
            }
        }

        if changed {
            let new_schema = Arc::new(arrow::datatypes::Schema::new(coerced_fields));
            coerced_batch = arrow::record_batch::RecordBatch::try_new(new_schema, coerced_columns)?;
        }

        let arrow_schema = coerced_batch.schema();
        use datafusion::common::DFSchema;
        let df_schema = DFSchema::try_from_qualified_schema("t", &arrow_schema)?;
        
        let phys_expr = create_physical_expr(
            df_expr,
            &df_schema,
            state.execution_props(),
        ).map_err(|e| anyhow::anyhow!("Failed to create physical expression: {}. Expression: {:?}, Schema: {:?}", e, df_expr, df_schema))?;

        let result = phys_expr.evaluate(&coerced_batch)?;
        let array = result.into_array(coerced_batch.num_rows())?;
        
        let mask = array.as_any().downcast_ref::<arrow::array::BooleanArray>()
            .ok_or_else(|| anyhow::anyhow!("Filter expression did not return a BooleanArray"))?;

        Ok(mask.clone())
    }

    /// Evaluate filter on a RecordBatch and return a BooleanArray mask
    pub fn evaluate_condition(&self, batch: &arrow::record_batch::RecordBatch, filter: &QueryFilter) -> anyhow::Result<arrow::array::BooleanArray> {
        use arrow::compute::kernels::cmp;
        use arrow::compute::kernels::boolean;

        let array = batch.column_by_name(&filter.column)
            .ok_or_else(|| anyhow::anyhow!("Column '{}' not found in batch", filter.column))?;
            
        let num_rows = batch.num_rows();
        let mut mask = arrow::array::BooleanArray::from(vec![true; num_rows]);
        
        if let Some(min_val) = &filter.min {

             let scalar = json_value_to_scalar(min_val, array.data_type())?;
             let scalar_array = scalar.to_array_of_size(num_rows)?;
             let res = if filter.min_inclusive {
                 cmp::gt_eq(array, &scalar_array)?
             } else {
                 cmp::gt(array, &scalar_array)?
             };
             mask = boolean::and(&mask, &res)?;
        }
        
        if let Some(max_val) = &filter.max {

             let scalar = json_value_to_scalar(max_val, array.data_type())?;
             let scalar_array = scalar.to_array_of_size(num_rows)?;
             let res = if filter.max_inclusive {
                 cmp::lt_eq(array, &scalar_array)?
             } else {
                 cmp::lt(array, &scalar_array)?
             };
             mask = boolean::and(&mask, &res)?;
        }

        if let Some(values) = &filter.values {
            let mut or_mask = arrow::array::BooleanArray::from(vec![false; num_rows]);
            for v in values {
                let scalar = json_value_to_scalar(v, array.data_type())?;
                let scalar_array = scalar.to_array_of_size(num_rows)?;
                let eq = cmp::eq(array, &scalar_array)?;
                or_mask = boolean::or(&or_mask, &eq)?;
            }
            mask = boolean::and(&mask, &or_mask)?;
        }

        if filter.negated {
            mask = boolean::not(&mask)?;
        }
        
        Ok(mask)
    }


    /// Evaluate multiple filters on a RecordBatch and return a BooleanArray mask
    pub fn evaluate_filters(&self, batch: &arrow::record_batch::RecordBatch, filters: &[QueryFilter]) -> anyhow::Result<arrow::array::BooleanArray> {
        use arrow::compute::kernels::boolean;
        
        let num_rows = batch.num_rows();
        let mut mask = arrow::array::BooleanArray::from(vec![true; num_rows]);
        
        for filter in filters {
            let filter_mask = self.evaluate_condition(batch, filter)?;
            mask = boolean::and(&mask, &filter_mask)?;
        }
        
        Ok(mask)
    }
    /// Prune manifest entries based on the filters.
    /// Returns a list of (Entry, Option<IndexFile>) tuples.
    pub fn prune_entries(&self, entries: &[ManifestEntry], expr: Option<&FilterExpr>) -> Vec<(ManifestEntry, Option<IndexFile>)> {
        let mut candidates = Vec::new();

        for entry in entries {
            let matches = if let Some(e) = expr {
                self.might_match_expr(entry, e)
            } else {
                true
            };
            
            if matches {
                // Select an index if possible. 
                // We'll extract flat AND conditions to look for candidates.
                let mut selected_index = None;
                if let Some(e) = expr {
                    let and_filters = e.extract_and_conditions();
                    for filter in and_filters {
                        if let Some(idx) = self.select_index(entry, &filter) {
                            selected_index = Some(idx);
                            break;
                        }
                    }
                }
                candidates.push((entry.clone(), selected_index));
            }
        }
        
        candidates
    }

    pub fn might_match_expr(&self, entry: &ManifestEntry, expr: &FilterExpr) -> bool {
        let FilterExpr::DataFusion(df_expr) = expr;
        self.might_match_df_expr(entry, df_expr)
    }

    fn might_match_df_expr(&self, entry: &ManifestEntry, expr: &Expr) -> bool {
        match expr {
            Expr::BinaryExpr(binary) => {
                match binary.op {
                    datafusion::logical_expr::Operator::And => {
                        self.might_match_df_expr(entry, &binary.left) && self.might_match_df_expr(entry, &binary.right)
                    }
                    datafusion::logical_expr::Operator::Or => {
                        self.might_match_df_expr(entry, &binary.left) || self.might_match_df_expr(entry, &binary.right)
                    }
                    _ => {
                        if let Some(filter) = convert_binary_expr_to_query_filter(binary) {
                             self.might_match_condition(entry, &filter)
                        } else {
                             true
                        }
                    }
                }
            }
            Expr::Not(_inner) => {
                // Negotiating stats is complex, coarse-grained match
                true
            }
            Expr::InList(in_list) => {
                if let Some(filter) = convert_in_list_to_query_filter(in_list) {
                    self.might_match_condition(entry, &filter)
                } else {
                    true
                }
            }
            _ => true,
        }
    }

    pub fn might_match_condition(&self, entry: &ManifestEntry, filter: &QueryFilter) -> bool {
        if filter.negated {
            // Pruning negated conditions is coarse for now.
            return true;
        }
        // 1. Partition-level Pruning (Coarse-grained)
        // If the query column is a partition column, we can prune entire files instantly.
        if let Some(entry_val) = entry.partition_values.get(&filter.column) {

            if let Some(min_val) = &filter.min {
                let res = if filter.min_inclusive {
                    self.compare_lt(entry_val, min_val) // if part < min -> NO match
                } else {
                    let ord = self.compare_values(entry_val, min_val);
                    ord == Some(std::cmp::Ordering::Less) || ord == Some(std::cmp::Ordering::Equal)
                };
                if res { 
                    return false; 
                }
            }

            if let Some(max_val) = &filter.max {
                 let res = if filter.max_inclusive {
                     self.compare_gt(entry_val, max_val) // if part > max -> NO match
                 } else {
                     let ord = self.compare_values(entry_val, max_val);
                     ord == Some(std::cmp::Ordering::Greater) || ord == Some(std::cmp::Ordering::Equal)
                 };
                 if res { 

                     return false; 
                }
            }

            if let Some(values) = &filter.values {
                if !values.contains(entry_val) {

                    return false;
                }
            }
        }

        // 2. Statistics Pruning (Fine-grained)
        if let Some(stats) = entry.column_stats.get(&filter.column) {

            if stats.null_count == entry.record_count {
                 return false;
            }

            if let Some(entry_max) = &stats.max {

                if let Some(filter_min) = &filter.min {
                    let entry_max_val = serde_json::Value::from(entry_max);
                    let too_small = if filter.min_inclusive {
                         self.compare_lt(&entry_max_val, filter_min)
                    } else {
                         let ord = self.compare_values(&entry_max_val, filter_min);
                         ord == Some(std::cmp::Ordering::Less) || ord == Some(std::cmp::Ordering::Equal)
                    };
                    if too_small { 

                        return false; 
                    }
                }
            }

            if let Some(entry_min) = &stats.min {

                if let Some(filter_max) = &filter.max {
                    let entry_min_val = serde_json::Value::from(entry_min);
                    let too_large = if filter.max_inclusive {
                        self.compare_gt(&entry_min_val, filter_max)
                    } else {
                        let ord = self.compare_values(&entry_min_val, filter_max);
                        ord == Some(std::cmp::Ordering::Greater) || ord == Some(std::cmp::Ordering::Equal)
                    };
                    if too_large { 

                        return false; 
                    }
                }
            }
            
            if let Some(values) = &filter.values {
                 let mut possible_match = false;
                 let min_val = stats.min.as_ref();
                 let max_val = stats.max.as_ref();
                 
                 if min_val.is_none() && max_val.is_none() {
                     return true;
                 }

                 for v in values {
                     let mut in_range = true;
                     if let Some(min) = min_val {
                         let min_v = serde_json::Value::from(min);
                         if self.compare_lt(v, &min_v) { in_range = false; }
                     }
                     if let Some(max) = max_val {
                         let max_v = serde_json::Value::from(max);
                         if self.compare_gt(v, &max_v) { in_range = false; }
                     }
                     if in_range {
                         possible_match = true;
                         break;
                     }
                 }
                 
                 if !possible_match {

                     return false;
                 }
            }

            true
        } else {

            true
        }
    }

    fn select_index(&self, entry: &ManifestEntry, filter: &QueryFilter) -> Option<IndexFile> {
        // Iterate over available indexes for this segment
        // Priority:
        // 1. Exact match column index (Scalar)
        // 2. Vector index? (Not applicable for Range filter usually, but maybe for similarity)
        // For MVP: We only look for scalar index on the filtered column.
        
        for idx in &entry.index_files {
            if let Some(col) = &idx.column_name {
                if col == &filter.column {
                    // Found an index for this column!
                    // Check type?
                    if idx.index_type == "scalar" || idx.index_type == "unknown" { 
                        return Some(idx.clone());
                    }
                }
            }
        }
        
        None
    }

    fn compare_lt(&self, a: &Value, b: &Value) -> bool {
        self.compare_values(a, b) == Some(Ordering::Less)
    }

    pub fn compare_gt(&self, a: &Value, b: &Value) -> bool {
        self.compare_values(a, b) == Some(Ordering::Greater)
    }

    #[allow(dead_code)]
    fn might_match_clustering(&self, entry: &ManifestEntry, filters: &[QueryFilter]) -> bool {
        let (strategy, cols, min_s, max_s, norm_mins, norm_maxs) = match (
            &entry.clustering_strategy, 
            &entry.clustering_columns, 
            entry.min_clustering_score, 
            entry.max_clustering_score, 
            &entry.normalization_mins, 
            &entry.normalization_maxs
        ) {
            (Some(s), Some(c), Some(mi), Some(ma), Some(nm), Some(nx)) => (s, c, mi, ma, nm, nx),
            _ => return true, 
        };

        let n_cols = cols.len();
        let bits_per_col = 64 / n_cols;
        
        let mut query_mins = vec![0u64; n_cols];
        let mut query_maxs = vec![ (1u64 << bits_per_col) - 1; n_cols];

        let mut has_relevant_filter = false;
        for (i, col_name) in cols.iter().enumerate() {
            for filter in filters {
                if &filter.column == col_name {
                    has_relevant_filter = true;
                    let seg_min = &norm_mins[i];
                    let seg_max = &norm_maxs[i];
                    
                    if let Some(f_min) = &filter.min {
                         let norm_f_min = self.normalize_value_u64(f_min, seg_min, seg_max, bits_per_col);
                         query_mins[i] = query_mins[i].max(norm_f_min);
                    }
                    if let Some(f_max) = &filter.max {
                         let norm_f_max = self.normalize_value_u64(f_max, seg_min, seg_max, bits_per_col);
                         query_maxs[i] = query_maxs[i].min(norm_f_max);
                    }
                }
            }
        }

        if !has_relevant_filter {
            return true;
        }

        let query_min_score = if strategy == "zorder" {
            crate::core::clustering::compute_zorder_score(bits_per_col, &query_mins)
        } else {
            crate::core::clustering::hilbert_index(n_cols, bits_per_col, &query_mins)
        };
        
        let query_max_score = if strategy == "zorder" {
            crate::core::clustering::compute_zorder_score(bits_per_col, &query_maxs)
        } else {
            crate::core::clustering::hilbert_index(n_cols, bits_per_col, &query_maxs)
        };

        if query_min_score > max_s || query_max_score < min_s {
            return false;
        }

        true
    }

    #[allow(dead_code)]
    fn normalize_value_u64(&self, val: &Value, min: &Value, max: &Value, bits: usize) -> u64 {
        let max_range = (1u64 << bits) - 1;
        match (val, min, max) {
            (Value::Number(v), Value::Number(mi), Value::Number(ma)) => {
                let v_f = v.as_f64().unwrap_or(0.0);
                let mi_f = mi.as_f64().unwrap_or(0.0);
                let ma_f = ma.as_f64().unwrap_or(0.0);
                let range = ma_f - mi_f;
                if range > 0.0 {
                    ((v_f - mi_f) / range * max_range as f64).clamp(0.0, max_range as f64) as u64
                } else {
                    0
                }
            },
            _ => 0
        }
    }

    /// Robust comparison of serde_json::Value
    fn compare_values(&self, a: &Value, b: &Value) -> Option<Ordering> {
        match (a, b) {
            (Value::Number(n1), Value::Number(n2)) => {
                if n1.is_i64() && n2.is_i64() {
                     n1.as_i64().unwrap().partial_cmp(&n2.as_i64().unwrap())
                } else if n1.is_f64() && n2.is_f64() {
                     n1.as_f64().unwrap().partial_cmp(&n2.as_f64().unwrap())
                } else {
                     // Mixed types: try f64 fallback
                     let f1 = n1.as_f64();
                     let f2 = n2.as_f64();
                     match (f1, f2) {
                         (Some(v1), Some(v2)) => v1.partial_cmp(&v2),
                         _ => None
                     }
                }
            },
            (Value::String(s1), Value::String(s2)) => s1.partial_cmp(s2),
            (Value::Bool(b1), Value::Bool(b2)) => b1.partial_cmp(b2),
            _ => None
        }
    }
}

#[cfg(test)]
mod tests {
    
    
    


    /* DEPRECATED: These tests use the old QueryFilter API which has been replaced by FilterExpr
    #[test]
    fn test_pruning_min_max() {
        let planner = QueryPlanner::new();
        let entries = vec![
            create_entry("seg1", Some(10), Some(20), None), // Range 10-20
            create_entry("seg2", Some(30), Some(40), None), // Range 30-40
            create_entry("seg3", Some(50), Some(60), None), // Range 50-60
        ];
        let manifest = Manifest { entries, ..Default::default() };

        // Query: age >= 25 AND age <= 35
        let filter = QueryFilter {
            column: "age".to_string(),
            min: Some(serde_json::json!(25)),
            min_inclusive: true,
            max: Some(serde_json::json!(35)),
            max_inclusive: true,
            values: None,
        };

        let result = planner.prune_entries(&manifest, &[filter]);
        
        // seg1: max(20) < 25 -> Prune
        // seg2: match!
        // seg3: min(50) > 35 -> Prune
        
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0.file_path, "seg2.parquet");
    }
    */

    /* DEPRECATED: Old QueryFilter API test
    #[test]
    fn test_index_selection() {
        let planner = QueryPlanner::new();
        // Entry with index on "age"
        let entry_idx = create_entry("seg_idx", Some(10), Some(50), Some("age"));
        // Entry without index
        let entry_no_idx = create_entry("seg_raw", Some(10), Some(50), None);
        
        let manifest = Manifest { 
            entries: vec![entry_idx.clone(), entry_no_idx.clone()], 
            ..Default::default() 
        };

        let filter = QueryFilter {
            column: "age".to_string(),
            min: Some(serde_json::json!(20)),
            min_inclusive: true,
            max: None,
            max_inclusive: true,
            values: None,
        };

        let result = planner.prune_entries(&manifest, &[filter]);
        assert_eq!(result.len(), 2);

        // First one should have index
        assert!(result[0].1.is_some());
        assert_eq!(result[0].1.as_ref().unwrap().column_name.as_deref(), Some("age"));

        // Second one should NOT have index
        assert!(result[1].1.is_none());
    }

    #[test]
    fn test_pruning_clustering() {
        let planner = QueryPlanner::new();
        
        // Create an entry clustered on (age, salary)
        // Normalized bits per col = 32
        // Seg range: age=[10, 20], salary=[1000, 2000]
        let mut stats = HashMap::new();
        stats.insert("age".to_string(), ColumnStats { min: Some(serde_json::json!(10)), max: Some(serde_json::json!(20)), null_count: 0, distinct_count: None });
        stats.insert("salary".to_string(), ColumnStats { min: Some(serde_json::json!(1000)), max: Some(serde_json::json!(2000)), null_count: 0, distinct_count: None });
        
        let entry = ManifestEntry {
            file_path: "clustered.parquet".to_string(),
            file_size_bytes: 1000,
            record_count: 100,
            index_files: vec![],
            delete_files: vec![],
            column_stats: stats,
            partition_values: HashMap::new(),
            clustering_strategy: Some("zorder".to_string()),
            clustering_columns: Some(vec!["age".to_string(), "salary".to_string()]),
            min_clustering_score: Some(0), // Simplified for test: assume segment covers relative bottom-left
            max_clustering_score: Some(1000), 
            normalization_mins: Some(vec![serde_json::json!(10), serde_json::json!(1000)]),
            normalization_maxs: Some(vec![serde_json::json!(20), serde_json::json!(2000)]),
        };
        
        let manifest = Manifest { entries: vec![entry], ..Default::default() };
        */
        
        /* DEPRECATED: Old QueryFilter API test
        // Query that is OUTSIDE the segment's score range but INSIDE its bounding box on individual dims
        // e.g. age=15 (middle), salary=1500 (middle) -> Might be in the segment or not.
        // But if query is age=20, salary=2000 -> This is the top-right corner.
        // If segment only has max_score=1000, then top-right corner should be pruned.
        
        let filters = vec![
            QueryFilter { column: "age".to_string(), min: Some(serde_json::json!(19)), min_inclusive: true, max: None, max_inclusive: true, values: None },
            QueryFilter { column: "salary".to_string(), min: Some(serde_json::json!(1900)), min_inclusive: true, max: None, max_inclusive: true, values: None },
        ];
        
        let result = planner.prune_entries(&manifest, &filters);
        
        // In our setup:
        // age=19 is near max(20), salary=1900 is near max(2000).
        // Their Z-score will be high (near max bits).
        // Since entry has max_clustering_score=1000 (very low), it should be pruned!
        assert_eq!(result.len(), 0);
    }
}

// Additional tests for QueryFilter parsing
#[cfg(test)]
mod query_filter_parse_tests {
    use super::*;

    #[test]
    fn test_parse_simple_equality() {
        let filter = QueryFilter::parse("age = 30");
        assert!(filter.is_some());
        let f = filter.unwrap();
        assert_eq!(f.column, "age");
        assert!(f.min.is_some());
        assert!(f.max.is_some());
    }

    #[test]
    fn test_parse_greater_than() {
        let filter = QueryFilter::parse("price > 100");
        assert!(filter.is_some());
        let f = filter.unwrap();
        assert_eq!(f.column, "price");
        assert!(f.min.is_some());
        assert!(!f.min_inclusive);
        assert!(f.max.is_none());
    }

    #[test]
    fn test_parse_less_than_or_equal() {
        let filter = QueryFilter::parse("count <= 50");
        assert!(filter.is_some());
        let f = filter.unwrap();
        assert_eq!(f.column, "count");
        assert!(f.max.is_some());
        assert!(f.max_inclusive);
    }

    #[test]
    fn test_parse_boolean_column() {
        let filter = QueryFilter::parse("is_active");
        assert!(filter.is_some());
        let f = filter.unwrap();
        assert_eq!(f.column, "is_active");
        assert!(matches!(f.min, Some(Value::Bool(true))));
    }

    #[test]
    fn test_parse_string_value() {
        let filter = QueryFilter::parse("name = 'Alice'");
        assert!(filter.is_some());
        let f = filter.unwrap();
        assert_eq!(f.column, "name");
    }

    #[test]
    fn test_parse_in_clause() {
        let filter = QueryFilter::parse("category IN (A,B,C)");
        assert!(filter.is_some());
        let f = filter.unwrap();
        assert_eq!(f.column, "category");
        assert!(f.values.is_some());
        let values = f.values.unwrap();
        assert_eq!(values.len(), 3);
    }

    #[test]
    fn test_parse_multi() {
        let filters = QueryFilter::parse_multi("age > 30 AND price < 100 AND is_active");
        assert_eq!(filters.len(), 3);
        assert_eq!(filters[0].column, "age");
        assert_eq!(filters[1].column, "price");
        assert_eq!(filters[2].column, "is_active");
    }
    */
}
