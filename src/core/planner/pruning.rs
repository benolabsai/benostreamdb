// Copyright (c) 2026 Richard Albright. All rights reserved.

//! QueryPlanner with entry pruning logic.

use crate::core::manifest::{ManifestEntry, IndexFile};
use serde_json::Value;
use std::cmp::Ordering;
use datafusion::logical_expr::Expr;
use datafusion::prelude::SessionContext;

use super::filter::{FilterExpr, QueryFilter};

pub struct QueryPlanner {}

impl Default for QueryPlanner {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryPlanner {
    /// Create a new QueryPlanner instance.
    pub fn new() -> Self {
        Self {}
    }

    /// Evaluate a single filter condition on a RecordBatch and return the filtered batch.
    ///
    /// # Errors
    /// Returns an error if the filter column is not found in the batch schema
    /// or if type coercion fails during evaluation.
    pub fn filter_batch(&self, batch: &arrow::record_batch::RecordBatch, filter: &QueryFilter) -> anyhow::Result<arrow::record_batch::RecordBatch> {
        let mask = self.evaluate_condition(batch, filter)?;
        let filtered = arrow::compute::filter_record_batch(batch, &mask)?;
        Ok(filtered)
    }

    /// Evaluate a DataFusion filter expression on a RecordBatch.
    ///
    /// Handles automatic type coercion (LargeUtf8 → Utf8, LargeBinary → Binary)
    /// to ensure the expression evaluates correctly.
    ///
    /// # Errors
    /// Returns an error if the expression references unknown columns or types.
    pub fn filter_expr(&self, batch: &arrow::record_batch::RecordBatch, expr: &FilterExpr) -> anyhow::Result<arrow::record_batch::RecordBatch> {
        let mask = self.evaluate_expr(batch, expr)?;
        let filtered = arrow::compute::filter_record_batch(batch, &mask)?;
        Ok(filtered)
    }

    /// Evaluate a DataFusion filter expression, returning a boolean mask.
    ///
    /// Performs type coercion on the input batch (LargeUtf8 → Utf8, LargeBinary → Binary)
    /// before evaluation to avoid type mismatches.
    ///
    /// # Errors
    /// Returns an error if the expression cannot be compiled or evaluated.
    pub fn evaluate_expr(&self, batch: &arrow::record_batch::RecordBatch, expr: &FilterExpr) -> anyhow::Result<arrow::array::BooleanArray> {
        let FilterExpr::DataFusion(df_expr) = expr;

        use datafusion::physical_expr::create_physical_expr;

        let mut ctx = SessionContext::new();
        let _ = crate::core::sql::vector_operators::register_vector_operators(&mut ctx);
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
                coerced_fields.push(std::sync::Arc::new(new_field));
                changed = true;
            } else if let arrow::datatypes::DataType::LargeBinary = field.data_type() {
                let casted = arrow::compute::cast(batch.column(i), &arrow::datatypes::DataType::Binary)?;
                coerced_columns.push(casted);
                let mut new_field = field.as_ref().clone();
                new_field.set_data_type(arrow::datatypes::DataType::Binary);
                coerced_fields.push(std::sync::Arc::new(new_field));
                changed = true;
            } else {
                coerced_columns.push(batch.column(i).clone());
                coerced_fields.push(field.clone());
            }
        }

        if changed {
            let new_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(coerced_fields));
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

             let scalar = super::filter::json_value_to_scalar(min_val, array.data_type())?;
             let scalar_array = scalar.to_array_of_size(num_rows)?;
             let res = if filter.min_inclusive {
                 cmp::gt_eq(array, &scalar_array)?
             } else {
                 cmp::gt(array, &scalar_array)?
             };
             mask = boolean::and(&mask, &res)?;
        }

        if let Some(max_val) = &filter.max {

             let scalar = super::filter::json_value_to_scalar(max_val, array.data_type())?;
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
                let scalar = super::filter::json_value_to_scalar(v, array.data_type())?;
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


    /// Evaluate multiple filters on a RecordBatch, returning a combined boolean mask.
    ///
    /// Filters are combined with AND logic. Returns an all-true mask if the filter list is empty.
    ///
    /// # Errors
    /// Returns an error if any individual filter evaluation fails.
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
    /// Prune manifest entries that cannot match the given filters.
    ///
    /// Uses scalar stats (min/max) from manifest entries to eliminate segments
    /// before loading their data. Vector params are used to check vector column stats.
    ///
    /// Returns a list of (Entry, `Option<IndexFile>`) tuples for surviving candidates.
    pub fn prune_entries(&self, entries: &[ManifestEntry], expr: Option<&FilterExpr>, vector_params: Option<&super::vector_search::VectorSearchParams>) -> Vec<(ManifestEntry, Option<IndexFile>)> {
        let pruning_start = std::time::Instant::now();
        let mut candidates = Vec::new();

        for entry in entries {
            let mut matches_scalar = true;
            if let Some(f) = expr {
                if !self.might_match_expr(entry, f) {
                    matches_scalar = false;
                }
            }

            let vector_matches = if let Some(vp) = vector_params {
                self.might_match_vector(entry, vp)
            } else {
                true
            };

            if !vector_matches {
                // Future: track why it didn't match
            }

            if matches_scalar && vector_matches {
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

        metrics::histogram!("hyperstreamdb.query.segment_pruning_duration").record(pruning_start.elapsed().as_secs_f64());
        candidates
    }

    /// Check if a manifest entry might contain vectors matching the query.
    ///
    /// Uses vector stats (norm bounds, per-dimension min/max) to prune segments
    /// that are unlikely to contain relevant results. Returns `true` if no stats
    /// are available (conservative: must scan).
    pub fn might_match_vector(&self, entry: &ManifestEntry, params: &super::vector_search::VectorSearchParams) -> bool {
        let stats = if let Some(s) = entry.column_stats.get(&params.column) {
            s
        } else {
            return true; // No stats, must scan
        };

        let vs = if let Some(v) = &stats.vector_stats {
            v
        } else {
            return true; // No vector stats, must scan
        };

        // 1. Norm-based pruning for L2 Distance
        // |q - v| >= ||q| - |v||
        // If we have a rough estimate or if k=1 and we want to be aggressive.
        // For now, we'll use a very conservative heuristic:
        // if query norm is 10x larger than max_norm or 10x smaller than min_norm,
        // it MIGHT NOT be a good candidate if other segments are closer.
        // But true pruning requires a global "best distance".

        // 2. Per-dimension range pruning (Zone Maps for vectors)
        // If query point is very far from the bounding box of the segment's vectors.
        if let (Some(dim_min), Some(dim_max)) = (&vs.dim_min, &vs.dim_max) {
             if let crate::core::index::VectorValue::Float32(q_vec) = &params.query {
                 for (i, &q_val) in q_vec.iter().enumerate() {
                     if i < dim_min.len() && i < dim_max.len() {
                         if q_val < dim_min[i] {
                             let _diff_sq = (dim_min[i] - q_val).powi(2);
                             // Future optimization: accumulate diff_sq for early pruning threshold
                         } else if q_val > dim_max[i] {
                             let _diff_sq = (q_val - dim_max[i]).powi(2);
                         }
                     }
                 }
             }

             // If minimum possible distance to ANY point in this segment's box is too high, prune.
             // We need a threshold. For now, since we don't have global top-k yet, we just return true.
             // But we're ready for threshold-based pruning!
        }

        true
    }

    /// Check if a manifest entry might match a DataFusion filter expression.
    ///
    /// Recursively evaluates the expression tree against column stats (min/max)
    /// to determine if the entry can be safely pruned. Returns `true` (must scan)
    /// if stats are insufficient to prove a mismatch.
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
                        if let Some(filter) = super::filter::convert_binary_expr_to_query_filter(binary) {
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
                if let Some(filter) = super::filter::convert_in_list_to_query_filter(in_list) {
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
            tracing::debug!("Pruning Check: Column {} has partition value {:?}. Filter range: {:?} - {:?}", filter.column, entry_val, filter.min, filter.max);
            if let Some(min_val) = &filter.min {
                let res = if filter.min_inclusive {
                    self.compare_lt(entry_val, min_val) // if part < min -> NO match
                } else {
                    let ord = self.compare_values(entry_val, min_val);
                    ord == Some(std::cmp::Ordering::Less) || ord == Some(std::cmp::Ordering::Equal)
                };
                if res {
                    tracing::debug!("  -> Pruned by partition min: {} < {:?}", entry_val, min_val);
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
                     tracing::debug!("  -> Pruned by partition max: {} > {:?}", entry_val, max_val);
                     return false;
                }
            }

            if let Some(values) = &filter.values {
                if !values.contains(entry_val) {
                    tracing::debug!("  -> Pruned by partition values IN list: {:?} not in {:?}", entry_val, values);
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
            crate::core::clustering::gray_code_interleave_index(n_cols, bits_per_col, &query_mins)
        };

        let query_max_score = if strategy == "zorder" {
            crate::core::clustering::compute_zorder_score(bits_per_col, &query_maxs)
        } else {
            crate::core::clustering::gray_code_interleave_index(n_cols, bits_per_col, &query_maxs)
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
                     n1.as_i64().unwrap_or(0).partial_cmp(&n2.as_i64().unwrap_or(0))
                } else if n1.is_f64() && n2.is_f64() {
                     n1.as_f64().unwrap_or(0.0).partial_cmp(&n2.as_f64().unwrap_or(0.0))
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
