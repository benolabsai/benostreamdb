// Copyright (c) 2026 Richard Albright. All rights reserved.

pub mod graph_udf;
pub mod literal;
pub mod merge_into;
pub mod optimizer;
pub mod partition_rewriter;
pub mod pgvector_rewriter;
pub mod physical_plan;
pub mod session;
pub mod udf;
pub mod vector_literal;
pub mod vector_operators;

// Alias `udf` as `vector_udf` for backward compatibility with existing import paths
pub use udf as vector_udf;

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use roaring::RoaringBitmap;
use serde_json::Value;

use crate::core::planner::QueryFilter;

use async_trait::async_trait;
use datafusion::arrow::datatypes::{Field, Schema, SchemaRef};
use datafusion::datasource::TableProvider;
use datafusion::datasource::TableType;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_plan::ExecutionPlan;

use crate::core::sql::physical_plan::BenoStreamExec;
use crate::core::table::Table;

/// A range bound on a single column, used for OR-over-ranges pruning.
#[derive(Debug, Clone)]
struct RangeBound {
    min: Option<Value>,
    min_inclusive: bool,
    max: Option<Value>,
    max_inclusive: bool,
}

fn scalar_to_json(scalar: &datafusion::scalar::ScalarValue) -> Option<Value> {
    use datafusion::scalar::ScalarValue as SV;
    match scalar {
        SV::Int8(Some(i)) => Some(Value::from(*i as i64)),
        SV::Int16(Some(i)) => Some(Value::from(*i as i64)),
        SV::Int32(Some(i)) => Some(Value::from(*i as i64)),
        SV::Int64(Some(i)) => Some(Value::from(*i)),
        SV::UInt8(Some(i)) => Some(Value::from(*i as i64)),
        SV::UInt16(Some(i)) => Some(Value::from(*i as i64)),
        SV::UInt32(Some(i)) => Some(Value::from(*i as i64)),
        SV::UInt64(Some(i)) => Some(Value::from(*i)),
        SV::Float32(Some(f)) => Some(Value::from(*f as f64)),
        SV::Float64(Some(f)) => Some(Value::from(*f)),
        SV::Utf8(Some(s)) | SV::LargeUtf8(Some(s)) => Some(Value::from(s.clone())),
        SV::Boolean(Some(b)) => Some(Value::from(*b)),
        SV::Date32(Some(d)) => Some(Value::from(*d as i64)),
        SV::TimestampMicrosecond(Some(t), _)
        | SV::TimestampMillisecond(Some(t), _)
        | SV::TimestampSecond(Some(t), _)
        | SV::TimestampNanosecond(Some(t), _) => Some(Value::from(*t)),
        _ => None,
    }
}

fn column_of(e: &Expr) -> Option<String> {
    match e {
        Expr::Column(c) => Some(c.name.clone()),
        Expr::Cast(c) => column_of(&c.expr),
        Expr::TryCast(c) => column_of(&c.expr),
        _ => None,
    }
}

fn literal_of(e: &Expr) -> Option<Value> {
    match e {
        Expr::Literal(s, _) => scalar_to_json(s),
        Expr::Cast(c) => literal_of(&c.expr),
        Expr::TryCast(c) => literal_of(&c.expr),
        _ => None,
    }
}

/// Interpret an expression as a range on a single column.
///
/// Handles `BETWEEN` and the comparison operators, plus an `AND` of two bounds
/// on the same column (how `BETWEEN` is often lowered).
fn expr_to_range(e: &Expr) -> Option<(String, RangeBound)> {
    use datafusion::logical_expr::Operator as Op;
    match e {
        Expr::Between(b) if !b.negated => {
            let col = column_of(&b.expr)?;
            let min = literal_of(&b.low)?;
            let max = literal_of(&b.high)?;
            Some((
                col,
                RangeBound {
                    min: Some(min),
                    min_inclusive: true,
                    max: Some(max),
                    max_inclusive: true,
                },
            ))
        }
        Expr::BinaryExpr(b) if b.op == Op::And => {
            let (lc, lr) = expr_to_range(&b.left)?;
            let (rc, rr) = expr_to_range(&b.right)?;
            if lc != rc {
                return None;
            }
            let (min, min_inclusive) = match (lr.min, rr.min) {
                (Some(m), _) => (Some(m), lr.min_inclusive),
                (None, m) => (m, rr.min_inclusive),
            };
            let (max, max_inclusive) = match (lr.max, rr.max) {
                (Some(m), _) => (Some(m), lr.max_inclusive),
                (None, m) => (m, rr.max_inclusive),
            };
            Some((
                lc,
                RangeBound {
                    min,
                    min_inclusive,
                    max,
                    max_inclusive,
                },
            ))
        }
        Expr::BinaryExpr(b) => {
            let col = column_of(&b.left)?;
            let val = literal_of(&b.right)?;
            match b.op {
                Op::Gt => Some((
                    col,
                    RangeBound {
                        min: Some(val),
                        min_inclusive: false,
                        max: None,
                        max_inclusive: false,
                    },
                )),
                Op::GtEq => Some((
                    col,
                    RangeBound {
                        min: Some(val),
                        min_inclusive: true,
                        max: None,
                        max_inclusive: false,
                    },
                )),
                Op::Lt => Some((
                    col,
                    RangeBound {
                        min: None,
                        min_inclusive: false,
                        max: Some(val),
                        max_inclusive: false,
                    },
                )),
                Op::LtEq => Some((
                    col,
                    RangeBound {
                        min: None,
                        min_inclusive: false,
                        max: Some(val),
                        max_inclusive: true,
                    },
                )),
                _ => None,
            }
        }
        _ => None,
    }
}

/// Extract a same-column disjunction of ranges, e.g.
/// `(id BETWEEN 1 AND 5) OR (id BETWEEN 50 AND 55)`.
///
/// Returns `None` unless the whole expression is a disjunction of two or more
/// ranges that all target the same column — anything else is left to
/// DataFusion's own filtering.
fn extract_or_ranges(expr: &Expr) -> Option<(String, Vec<RangeBound>)> {
    use datafusion::logical_expr::Operator as Op;

    fn collect(e: &Expr, out: &mut Vec<RangeBound>) -> Option<String> {
        if let Expr::BinaryExpr(b) = e {
            if b.op == Op::Or {
                let l = collect(&b.left, out)?;
                let r = collect(&b.right, out)?;
                return if l == r { Some(l) } else { None };
            }
        }
        let (col, r) = expr_to_range(e)?;
        out.push(r);
        Some(col)
    }

    let mut out = Vec::new();
    let col = collect(expr, &mut out)?;
    if out.len() < 2 {
        return None;
    }
    Some((col, out))
}

#[derive(Debug)]
pub struct BenoStreamTableProvider {
    pub table: Arc<Table>,
}

impl BenoStreamTableProvider {
    pub fn new(table: Arc<Table>) -> Self {
        Self { table }
    }
}

fn strip_field_metadata(field: &Field) -> Field {
    let dt = field.data_type().clone();
    let dt = match dt {
        arrow::datatypes::DataType::List(f) => {
            arrow::datatypes::DataType::List(Arc::new(strip_field_metadata(f.as_ref())))
        }
        arrow::datatypes::DataType::FixedSizeList(f, size) => {
            arrow::datatypes::DataType::FixedSizeList(
                Arc::new(strip_field_metadata(f.as_ref())),
                size,
            )
        }
        arrow::datatypes::DataType::LargeList(f) => {
            arrow::datatypes::DataType::LargeList(Arc::new(strip_field_metadata(f.as_ref())))
        }
        arrow::datatypes::DataType::Struct(fields) => {
            let fields = fields
                .iter()
                .map(|f| strip_field_metadata(f.as_ref()))
                .collect();
            arrow::datatypes::DataType::Struct(fields)
        }
        arrow::datatypes::DataType::Map(f, sorted) => {
            arrow::datatypes::DataType::Map(Arc::new(strip_field_metadata(f.as_ref())), sorted)
        }
        _ => dt,
    };
    Field::new(field.name(), dt, field.is_nullable()).with_metadata(HashMap::new())
}

fn strip_metadata(schema: SchemaRef) -> SchemaRef {
    let fields: Vec<Field> = schema
        .fields()
        .iter()
        .map(|f| strip_field_metadata(f.as_ref()))
        .collect();
    Arc::new(Schema::new_with_metadata(fields, HashMap::new()))
}

#[async_trait]
impl TableProvider for BenoStreamTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        strip_metadata(self.table.arrow_schema())
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        _state: &dyn datafusion::catalog::Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::error::Result<Arc<dyn ExecutionPlan>> {
        // Fetch segments from table state
        let mut segments = self
            .table
            .get_snapshot_segments()
            .await
            .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;

        // Keep the pre-pruning segment list so we can report the partition /
        // statistics pruning breakdown in the DataFusion plan (EXPLAIN).
        let all_segments = segments.clone();

        // Row-value IN-list pushdown over the primary key (A1.7).
        //
        // When the pushed-down predicate is a tuple set like
        // `(id, region) IN ((1,'US'), (2,'CA'))` we can consult the per-column
        // inverted indexes and drop segments that cannot match, before they are
        // ever read. This is only sound when *every* PK column is indexed: a
        // partial index would under-count matches and prune real rows.
        let pk_cols = self.table.primary_key.read().clone();
        let indexed_cols = self.table.get_index_columns();
        let pk_filter = if !pk_cols.is_empty() && pk_cols.iter().all(|c| indexed_cols.contains(c)) {
            filters.iter().find_map(|f| {
                crate::core::table::primary_key::PrimaryKeyFilter::from_expr(f, &pk_cols)
                    .filter(|pf| !pf.is_empty())
            })
        } else {
            None
        };

        if let Some(pf) = &pk_filter {
            let before = segments.len();
            let mut kept = Vec::with_capacity(before);
            for segment in segments {
                let mut matched = true;
                if let Ok(reader) = self.table.segment_reader(&segment) {
                    if let Ok(Some(bm)) =
                        crate::core::table::Table::pk_match_bitmap(&reader, pf).await
                    {
                        let deleted = reader.load_merged_deletes().await.unwrap_or_default();
                        matched = !(bm - deleted).is_empty();
                    }
                }
                if matched {
                    kept.push(segment);
                }
            }
            tracing::info!(
                "SQL Scan: PK row-value pushdown ({}) pruned {}/{} segment(s)",
                pf.describe(),
                before - kept.len(),
                before
            );
            segments = kept;
        }

        // Complex range pushdown (A1.6): a disjunction of ranges on one column,
        // e.g. `(id BETWEEN 1 AND 5) OR (id BETWEEN 50 AND 55)`. DataFusion
        // already post-filters this correctly; here we additionally use the
        // column's index to skip segments whose values cannot match *any*
        // disjunct, so they are never read.
        if let Some((col, ranges)) = filters.iter().find_map(extract_or_ranges) {
            if indexed_cols.contains(&col) {
                let before = segments.len();
                let mut kept = Vec::with_capacity(before);
                for segment in segments {
                    let mut matched = true;
                    if let Ok(reader) = self.table.segment_reader(&segment) {
                        let mut union: Option<RoaringBitmap> = None;
                        let mut prunable = true;
                        for r in &ranges {
                            let qf = QueryFilter {
                                column: col.clone(),
                                min: r.min.clone(),
                                min_inclusive: r.min_inclusive,
                                max: r.max.clone(),
                                max_inclusive: r.max_inclusive,
                                values: None,
                                negated: false,
                            };
                            match reader.get_scalar_filter_bitmap(&qf).await {
                                Ok(Some(bm)) => {
                                    union = Some(match union {
                                        Some(u) => u | bm,
                                        None => bm,
                                    });
                                }
                                // No index for this column (or a read error):
                                // we cannot prune safely, so keep the segment.
                                _ => {
                                    prunable = false;
                                    break;
                                }
                            }
                        }
                        if prunable {
                            if let Some(u) = union {
                                let deleted =
                                    reader.load_merged_deletes().await.unwrap_or_default();
                                matched = !(u - deleted).is_empty();
                            }
                        }
                    }
                    if matched {
                        kept.push(segment);
                    }
                }
                tracing::info!(
                    "SQL Scan: OR-range pushdown on '{}' pruned {}/{} segment(s)",
                    col,
                    before - kept.len(),
                    before
                );
                segments = kept;
            }
        }

        // Determine parallelism
        let target_partitions = self.table.get_max_parallel_readers().unwrap_or(4);

        let mut partitions = vec![Vec::new(); target_partitions];
        for (i, segment) in segments.into_iter().enumerate() {
            partitions[i % target_partitions].push(segment);
        }
        let partitions: Vec<_> = partitions.into_iter().filter(|p| !p.is_empty()).collect();

        tracing::info!("SQL Scan: Created {} partitions", partitions.len());

        // fetch index columns to prioritize filters
        let _index_cols = self.table.get_index_columns(); // This returns Vec<String>

        // Convert DataFusion filters to BenoStream SQL-like filter string
        // Returns Option<(ColumnName, SQLString)>
        fn expr_to_sql(expr: &Expr) -> Option<(String, String)> {
            match expr {
                Expr::BinaryExpr(binary) => {
                    let (left_col, left_sql) = match &*binary.left {
                        Expr::Column(c) => (c.name.clone(), c.name.clone()),
                        _ => return None, // Left must be column
                    };

                    let right_val = match &*binary.right {
                        Expr::Literal(scalar_value, _) => match scalar_value {
                            datafusion::scalar::ScalarValue::Utf8(Some(s)) => format!("'{}'", s),
                            datafusion::scalar::ScalarValue::Int32(Some(i)) => i.to_string(),
                            datafusion::scalar::ScalarValue::Int64(Some(i)) => i.to_string(),
                            datafusion::scalar::ScalarValue::Float32(Some(f)) => f.to_string(),
                            datafusion::scalar::ScalarValue::Float64(Some(f)) => f.to_string(),
                            datafusion::scalar::ScalarValue::Boolean(Some(b)) => b.to_string(),
                            _ => scalar_value.to_string(),
                        },
                        _ => return None, // Right must be literal
                    };

                    let op = match binary.op {
                        datafusion::logical_expr::Operator::Eq => "=",
                        datafusion::logical_expr::Operator::Gt => ">",
                        datafusion::logical_expr::Operator::Lt => "<",
                        datafusion::logical_expr::Operator::GtEq => ">=",
                        datafusion::logical_expr::Operator::LtEq => "<=",
                        _ => return None,
                    };

                    Some((left_col, format!("{} {} {}", left_sql, op, right_val)))
                }
                Expr::InList(in_list) => {
                    if in_list.negated {
                        return None;
                    }
                    let col_name = match &*in_list.expr {
                        Expr::Column(c) => c.name.clone(),
                        _ => return None,
                    };

                    let mut values = Vec::new();
                    for v in &in_list.list {
                        if let Expr::Literal(scalar_value, _) = v {
                            match scalar_value {
                                datafusion::scalar::ScalarValue::Utf8(Some(s)) => {
                                    values.push(format!("'{}'", s))
                                }
                                datafusion::scalar::ScalarValue::Int32(Some(i)) => {
                                    values.push(i.to_string())
                                }
                                datafusion::scalar::ScalarValue::Int64(Some(i)) => {
                                    values.push(i.to_string())
                                }
                                _ => return None, // Complex types in IN list
                            }
                        } else {
                            return None;
                        }
                    }

                    if values.is_empty() {
                        return None;
                    }
                    let val_str = values.join(",");
                    Some((col_name.clone(), format!("{} IN ({})", col_name, val_str)))
                }
                _ => None,
            }
        }

        // Selection Logic:
        // Aggregate all pushable filters
        let mut all_filters = Vec::new();
        let mut bm25_params = None;

        for filter in filters {
            if let Some((col, sql)) = expr_to_sql(filter) {
                // Check if this is a BM25 candidate: Column has BM25 index and is an equality match
                let has_bm25 = self
                    .table
                    .indexing
                    .index_configs
                    .read()
                    .get(&col)
                    .map(|cfg| cfg.tokenizer.is_some())
                    .unwrap_or(false);

                if has_bm25 && bm25_params.is_none() {
                    if let Expr::BinaryExpr(b) = filter {
                        if b.op == datafusion::logical_expr::Operator::Eq {
                            // Extract the literal value
                            if let datafusion::logical_expr::Expr::Literal(scalar, _) = &*b.right {
                                if let datafusion::scalar::ScalarValue::Utf8(Some(query_str)) =
                                    scalar
                                {
                                    tracing::info!("SQL BM25 Pushdown: Triggering keyword search for '{}' on column '{}'", query_str, col);
                                    let params = crate::core::planner::VectorSearchParams::new(
                                        &col,
                                        crate::core::index::VectorValue::Keyword(query_str.clone()),
                                        limit.unwrap_or(100),
                                    );
                                    bm25_params = Some(params);
                                }
                            }
                        }
                    }
                }

                all_filters.push(sql);
            }
        }

        let best_filter = if all_filters.is_empty() {
            None
        } else {
            Some(all_filters.join(" AND "))
        };

        if let Some(vp) = bm25_params {
            use crate::core::sql::physical_plan::vector_merge::VectorMergeExec;
            use crate::core::sql::physical_plan::vector_scan::VectorScanExec;

            let scan_exec = VectorScanExec::new(
                self.table.clone(),
                partitions,
                projection.cloned(),
                best_filter,
                vec![vp],
                limit,
                self.schema(),
            )?;

            let merge_exec =
                VectorMergeExec::new(Arc::new(scan_exec), limit.unwrap_or(100), 0, self.schema())?;

            Ok(Arc::new(merge_exec))
        } else {
            // Surface the same partition/statistics pruning breakdown that the
            // engine's own `explain()` prints, so a single DataFusion EXPLAIN
            // shows everything. Metrics are suppressed: planning must not
            // inflate the operational pruning counters.
            let pruning_summary = {
                let planner = crate::core::planner::QueryPlanner::new();
                let mut sub_filters: Vec<crate::core::planner::QueryFilter> = Vec::new();
                for f in filters {
                    let fe = crate::core::planner::FilterExpr::DataFusion(f.clone());
                    sub_filters.extend(fe.extract_and_conditions());
                }
                if sub_filters.is_empty() {
                    None
                } else {
                    let mut reasons: std::collections::HashMap<&'static str, usize> =
                        std::collections::HashMap::new();
                    for entry in &all_segments {
                        if let Some(reason) = sub_filters
                            .iter()
                            .find_map(|f| planner.classify_condition(entry, f, false))
                        {
                            *reasons.entry(reason.label()).or_insert(0) += 1;
                        }
                    }
                    if reasons.is_empty() {
                        None
                    } else {
                        let mut pairs: Vec<(&'static str, usize)> = reasons.into_iter().collect();
                        pairs.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
                        Some(
                            pairs
                                .into_iter()
                                .map(|(label, count)| format!("{}={}", label, count))
                                .collect::<Vec<_>>()
                                .join(", "),
                        )
                    }
                }
            };

            Ok(Arc::new(
                BenoStreamExec::new(
                    self.table.clone(),
                    partitions,
                    projection.cloned(),
                    best_filter,
                    limit,
                    self.schema(),
                )?
                .with_pruning_summary(pruning_summary),
            ))
        }
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::error::Result<Vec<TableProviderFilterPushDown>> {
        let index_configs = self.table.indexing.index_configs.read();

        Ok(filters
            .iter()
            .map(|f| {
                if let Expr::BinaryExpr(b) = f {
                    if b.op == datafusion::logical_expr::Operator::Eq {
                        if let Expr::Column(c) = &*b.left {
                            let has_bm25 = index_configs
                                .get(&c.name)
                                .map(|cfg| cfg.tokenizer.is_some())
                                .unwrap_or(false);
                            if has_bm25 {
                                return TableProviderFilterPushDown::Exact;
                            }
                        }
                    }
                }
                TableProviderFilterPushDown::Inexact
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::table::Table;
    use arrow::array::Int32Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::prelude::SessionContext;
    use std::sync::Arc;

    #[tokio::test]
    async fn test_parallel_scan_planning() -> datafusion::error::Result<()> {
        // Setup a table with multiple segments
        let uri = format!(
            "file://{}",
            std::env::temp_dir()
                .join("test_parallel_scan")
                .to_string_lossy()
        );
        let _ = std::fs::remove_dir_all(uri.strip_prefix("file://").unwrap()); // Cleanup previous

        let table = Table::new_async(uri.clone()).await.unwrap();

        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

        // Write Segment 1
        let batch1 =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))])
                .unwrap();
        table.write_async(vec![batch1]).await.unwrap();
        table.commit_async().await.unwrap();

        // Write Segment 2
        let batch2 =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![2]))])
                .unwrap();
        table.write_async(vec![batch2]).await.unwrap();
        table.commit_async().await.unwrap();

        // Write Segment 3
        let batch3 =
            RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![3]))])
                .unwrap();
        table.write_async(vec![batch3]).await.unwrap();
        table.commit_async().await.unwrap();

        // Create Provider
        let provider = Arc::new(BenoStreamTableProvider::new(Arc::new(table)));

        // Create DataFusion context and plan a scan
        let ctx = SessionContext::new();
        ctx.register_table("t", provider).unwrap();

        let df = ctx.sql("SELECT * FROM t").await?;
        let logical_plan = df.logical_plan();
        let physical_plan = ctx.state().create_physical_plan(logical_plan).await?;

        // Verify Physical Plan is BenoStreamExec and has partitions
        let display = format!(
            "{}",
            datafusion::physical_plan::displayable(physical_plan.as_ref()).indent(true)
        );
        println!("Plan: {}", display);

        // We expect BenoStreamExec to be present.
        // And since we didn't set max_readers, default is 4.
        // We have 3 segments.
        // 3 segments < 4 partitions => Should result in 3 partitions (logic: i % 4, so 0, 1, 2).
        // Actually the loop is:
        // Partitions[0].push(seg1)
        // Partitions[1].push(seg2)
        // Partitions[2].push(seg3)
        // Partitions[3] is empty.
        // Filter removes empty.
        // Result: 3 partitions.

        // Assert string contains "partitions=3" (based on DisplayAs impl)
        assert!(
            display.contains("BenoStreamExec: partitions=3")
                || display.contains("VectorMergeExec: k=100"),
            "Plan did not match expected structure. Plan was: {}",
            display
        );

        Ok(())
    }

    #[tokio::test]
    async fn test_scan_plan_surfaces_pruning_breakdown() -> datafusion::error::Result<()> {
        let uri = format!(
            "file://{}",
            std::env::temp_dir()
                .join("test_scan_pruning_summary")
                .to_string_lossy()
        );
        let _ = std::fs::remove_dir_all(uri.strip_prefix("file://").unwrap());

        let table = Table::new_async(uri.clone()).await.unwrap();
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int32, false)]));

        // Three single-row segments with ids 1, 2, 3.
        for v in [1, 2, 3] {
            let batch =
                RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![v]))])
                    .unwrap();
            table.write_async(vec![batch]).await.unwrap();
            table.commit_async().await.unwrap();
        }

        let provider = Arc::new(BenoStreamTableProvider::new(Arc::new(table)));
        let ctx = SessionContext::new();
        ctx.register_table("t", provider).unwrap();

        // `id >= 100` cannot match any segment (max id is 3), so all three
        // should be pruned by statistics and the reason surfaced in the plan.
        let df = ctx.sql("SELECT * FROM t WHERE id >= 100").await?;
        let physical_plan = ctx.state().create_physical_plan(df.logical_plan()).await?;
        let display = format!(
            "{}",
            datafusion::physical_plan::displayable(physical_plan.as_ref()).indent(true)
        );
        println!("Plan: {}", display);

        assert!(
            display.contains("pruning=["),
            "expected pruning breakdown in plan, got: {}",
            display
        );
        // All three segments must be pruned. This also pins the first-commit
        // stats fix: before it, the first segment's manifest entry carried no
        // bounds, so its `column_stats` came back empty and it was not pruned.
        assert!(
            display.contains("column max < filter min=3"),
            "expected all 3 segments pruned by stats, got: {}",
            display
        );

        Ok(())
    }
}
