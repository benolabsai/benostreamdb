// Copyright (c) 2026 Richard Albright. All rights reserved.

pub mod physical_plan;
pub mod session;
pub mod optimizer;
pub mod vector_operators;
pub mod vector_udf;
pub mod vector_literal;
pub mod pgvector_rewriter;

use std::any::Any;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::datasource::TableProvider;
use datafusion::datasource::TableType;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::logical_expr::{TableProviderFilterPushDown, Expr};
use datafusion::arrow::datatypes::SchemaRef;

use crate::core::table::Table;
use crate::core::sql::physical_plan::HyperStreamExec;

#[derive(Debug)]
pub struct HyperStreamTableProvider {
    pub table: Arc<Table>,
}

impl HyperStreamTableProvider {
    pub fn new(table: Arc<Table>) -> Self {
        Self { table }
    }
}

#[async_trait]
impl TableProvider for HyperStreamTableProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.table.arrow_schema()
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
        let segments = self.table.get_snapshot_segments().await
            .map_err(|e| datafusion::error::DataFusionError::Execution(e.to_string()))?;

        // Determine parallelism
        let target_partitions = self.table.get_max_parallel_readers().unwrap_or(4); 
        
        let mut partitions = vec![Vec::new(); target_partitions];
        for (i, segment) in segments.into_iter().enumerate() {
            partitions[i % target_partitions].push(segment);
        }
        let partitions: Vec<_> = partitions.into_iter().filter(|p| !p.is_empty()).collect();

        // fetch index columns to prioritize filters
        let index_cols = self.table.get_index_columns(); // This returns Vec<String>

        // Convert DataFusion filters to HyperStream SQL-like filter string
        // Returns Option<(ColumnName, SQLString)>
        fn expr_to_sql(expr: &Expr) -> Option<(String, String)> {
             match expr {
                Expr::BinaryExpr(binary) => {
                    let (left_col, left_sql) = match &*binary.left {
                        Expr::Column(c) => (c.name.clone(), c.name.clone()),
                        _ => return None, // Left must be column
                    };

                    let right_val = match &*binary.right {
                        Expr::Literal(scalar_value, _) => {
                             match scalar_value {
                                datafusion::scalar::ScalarValue::Utf8(Some(s)) => format!("'{}'", s),
                                datafusion::scalar::ScalarValue::Int32(Some(i)) => i.to_string(),
                                datafusion::scalar::ScalarValue::Int64(Some(i)) => i.to_string(),
                                datafusion::scalar::ScalarValue::Float32(Some(f)) => f.to_string(),
                                datafusion::scalar::ScalarValue::Float64(Some(f)) => f.to_string(),
                                datafusion::scalar::ScalarValue::Boolean(Some(b)) => b.to_string(),
                                _ => scalar_value.to_string(),
                            }
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
                },
                Expr::InList(in_list) => {
                    if in_list.negated { return None; }
                    let col_name = match &*in_list.expr {
                        Expr::Column(c) => c.name.clone(),
                        _ => return None,
                    };
                    
                    let mut values = Vec::new();
                    for v in &in_list.list {
                        if let Expr::Literal(scalar_value, _) = v {
                             match scalar_value {
                                datafusion::scalar::ScalarValue::Utf8(Some(s)) => values.push(format!("'{}'", s)),
                                datafusion::scalar::ScalarValue::Int32(Some(i)) => values.push(i.to_string()),
                                datafusion::scalar::ScalarValue::Int64(Some(i)) => values.push(i.to_string()),
                                _ => return None, // Complex types in IN list
                             }
                        } else {
                            return None;
                        }
                    }
                    
                    if values.is_empty() { return None; }
                    let val_str = values.join(",");
                    Some((col_name.clone(), format!("{} IN ({})", col_name, val_str)))
                }
                _ => None, 
             }
        }

        // Selection Logic:
        // 1. Pick first filter that targets an indexed column.
        // 2. If none, pick first valid filter.
        // 3. Ignore others (DataFusion will re-apply them anyway).
        
        let mut best_filter: Option<String> = None;

        for filter in filters {
            if let Some((col, sql)) = expr_to_sql(filter) {
                let is_indexed = index_cols.contains(&col);
                
                if is_indexed {
                    best_filter = Some(sql);
                    // found_indexed = true;
                    break; // Found the golden ticket
                }
                
                if best_filter.is_none() {
                    best_filter = Some(sql);
                }
            }
        }
        
        Ok(Arc::new(HyperStreamExec::new(
            self.table.clone(),
            partitions,
            projection.cloned(),
            best_filter,
            None, // vector_params
            limit,
            self.schema(),
        )))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::error::Result<Vec<TableProviderFilterPushDown>> {
        // We support filter pushdown effectively (Inexact means we filter, but DF should verify)
        Ok(filters.iter().map(|_| TableProviderFilterPushDown::Inexact).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::SessionContext;
    use crate::core::table::Table;
    use arrow::record_batch::RecordBatch;
    use arrow::array::Int32Array;
    use arrow::datatypes::{Schema, Field, DataType};
    use std::sync::Arc;

    #[tokio::test]
    async fn test_parallel_scan_planning() -> datafusion::error::Result<()> {
        // Setup a table with multiple segments
        let uri = format!("file://{}", std::env::temp_dir().join("test_parallel_scan").to_string_lossy());
        let _ = std::fs::remove_dir_all(uri.strip_prefix("file://").unwrap()); // Cleanup previous
        
        let table = Table::new_async(uri.clone()).await.unwrap();
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
        ]));
        
        // Write Segment 1
        let batch1 = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![1]))]).unwrap();
        table.write_async(vec![batch1]).await.unwrap();
        table.commit_async().await.unwrap();
        
        // Write Segment 2
        let batch2 = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![2]))]).unwrap();
        table.write_async(vec![batch2]).await.unwrap();
        table.commit_async().await.unwrap();
        
        // Write Segment 3
        let batch3 = RecordBatch::try_new(schema.clone(), vec![Arc::new(Int32Array::from(vec![3]))]).unwrap();
        table.write_async(vec![batch3]).await.unwrap();
        table.commit_async().await.unwrap();

        // Create Provider
        let provider = Arc::new(HyperStreamTableProvider::new(Arc::new(table)));
        
        // Create DataFusion context and plan a scan
        let ctx = SessionContext::new();
        ctx.register_table("t", provider).unwrap();
        
        let df = ctx.sql("SELECT * FROM t").await?;
        let logical_plan = df.logical_plan();
        let physical_plan = ctx.state().create_physical_plan(logical_plan).await?;
        
        // Verify Physical Plan is HyperStreamExec and has partitions
        let display = format!("{}", datafusion::physical_plan::displayable(physical_plan.as_ref()).indent(true));
        println!("Plan: {}", display);
        
        // We expect HyperStreamExec to be present.
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
        assert!(display.contains("HyperStreamExec: partitions=3"), "Plan did not indicate 3 partitions. Plan was: {}", display);
        
        Ok(())
    }
}
