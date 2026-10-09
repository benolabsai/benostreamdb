// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! The `subscribe_events` SQL table function — a batch-friendly projection of
//! [`Table::subscribe`](crate::core::table::Table::subscribe).
//!
//! ```sql
//! SELECT * FROM subscribe_events('edges', 'weight > 0.5', 100, 1000);
//! ```
//!
//! It drains up to `max_events` events (or until `timeout_ms` elapses) from the
//! named table's in-process change feed and returns one row per event:
//!
//! | column       | type   | meaning                                   |
//! | :----------- | :----- | :---------------------------------------- |
//! | `event_type` | UTF8   | `batch` (committed rows) or `commit`      |
//! | `rows`       | Int64  | rows in that batch / commit               |
//!
//! Because it is a DataFusion table function it is reachable from **every** SQL
//! surface — Python `execute_sql`, dbt models, the Spark pass-through reader,
//! Trino, and Flight SQL — which is what makes the subscription primitive
//! universal. The channel is in-process: a connector only observes commits made
//! by writers in the same process (cross-process streaming uses the Flight
//! endpoint).
//!
//! Arguments are positional (DataFusion 52 drops table-function argument names):
//!   * `table`      — string literal naming a registered table.
//!   * `filter`     — optional SQL predicate; only matching rows are counted.
//!   * `max_events` — optional cap on drained events (default `100`).
//!   * `timeout_ms` — optional wait budget in milliseconds (default `1000`).

use std::any::Any;
use std::sync::Arc;
use std::time::{Duration, Instant};

use arrow::array::{ArrayRef, Int64Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use async_trait::async_trait;
use datafusion::catalog::{Session, TableFunctionImpl, TableProvider};
use datafusion::common::plan_err;
use datafusion::datasource::TableType;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::SessionState;
use datafusion::logical_expr::Expr;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::scalar::ScalarValue;

use crate::core::sql::graph_udf::graph_table_functions::resolve_table;
use crate::core::table::TableEvent;

/// The name of the subscription table function.
pub const SUBSCRIBE_EVENTS: &str = "subscribe_events";

/// The names of every subscription table function.
pub fn subscribe_table_function_names() -> Vec<String> {
    vec![SUBSCRIBE_EVENTS.to_string()]
}

/// Register the subscription table function on a DataFusion context.
pub fn register_subscribe_table_functions(ctx: &mut datafusion::prelude::SessionContext) {
    ctx.register_udtf(SUBSCRIBE_EVENTS, Arc::new(SubscribeEventsFunc));
}

/// The fixed result schema: one row per drained event.
fn event_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("event_type", DataType::Utf8, false),
        Field::new("rows", DataType::Int64, false),
    ]))
}

#[derive(Debug)]
struct SubscribeEventsFunc;

impl TableFunctionImpl for SubscribeEventsFunc {
    fn call(&self, args: &[Expr]) -> Result<Arc<dyn TableProvider>> {
        let table = args
            .first()
            .ok_or_else(|| {
                DataFusionError::Plan(format!("{SUBSCRIBE_EVENTS} requires a table name"))
            })
            .and_then(as_string)?;
        let filter = args.get(1).map(as_string).transpose()?.unwrap_or_default();
        let max_events = args.get(2).map(as_u64).transpose()?.unwrap_or(100) as usize;
        let timeout_ms = args.get(3).map(as_u64).transpose()?.unwrap_or(1000);
        Ok(Arc::new(SubscribeEventsProvider {
            table,
            filter,
            max_events,
            timeout_ms,
        }))
    }
}

#[derive(Debug)]
struct SubscribeEventsProvider {
    table: String,
    filter: String,
    max_events: usize,
    timeout_ms: u64,
}

#[async_trait]
impl TableProvider for SubscribeEventsProvider {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        event_schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let session_state = state
            .as_any()
            .downcast_ref::<SessionState>()
            .ok_or_else(|| {
                DataFusionError::Internal(
                    "subscribe_events requires a DataFusion SessionState".to_string(),
                )
            })?;

        let table = resolve_table(session_state, &self.table).await?;
        let batch = drain_events(&table, &self.filter, self.max_events, self.timeout_ms).await?;

        let mem = datafusion::datasource::memory::MemTable::try_new(
            event_schema(),
            vec![vec![batch]],
        )?;
        mem.scan(state, projection, filters, limit).await
    }
}

/// Drain up to `max_events` events (or until `timeout_ms`) into a result batch.
async fn drain_events(
    table: &Arc<crate::core::table::Table>,
    filter: &str,
    max_events: usize,
    timeout_ms: u64,
) -> Result<arrow::record_batch::RecordBatch> {
    let mut sub = if filter.trim().is_empty() {
        table.subscribe()
    } else {
        table.subscribe_filtered(filter.to_string())
    };

    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    let mut event_types = StringBuilder::new();
    let mut rows = Int64Builder::new();
    let mut drained = 0usize;

    while drained < max_events {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            break;
        }
        match tokio::time::timeout(remaining, sub.recv()).await {
            Ok(Ok(TableEvent::Batch(b))) => {
                event_types.append_value("batch");
                rows.append_value(b.num_rows() as i64);
                drained += 1;
            }
            Ok(Ok(TableEvent::Commit { rows: n })) => {
                event_types.append_value("commit");
                rows.append_value(n as i64);
                drained += 1;
            }
            // Channel closed or the wait budget elapsed: return what we have.
            Ok(Err(_)) | Err(_) => break,
        }
    }

    let columns: Vec<ArrayRef> = vec![Arc::new(event_types.finish()), Arc::new(rows.finish())];
    arrow::record_batch::RecordBatch::try_new(event_schema(), columns)
        .map_err(|e| DataFusionError::Execution(e.to_string()))
}

/// Extract a string literal argument.
fn as_string(expr: &Expr) -> Result<String> {
    match expr {
        Expr::Literal(ScalarValue::Utf8(Some(s)), _)
        | Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _)
        | Expr::Literal(ScalarValue::Utf8View(Some(s)), _) => Ok(s.clone()),
        other => plan_err!("expected a string literal, got {other:?}"),
    }
}

/// Extract an integer literal argument as `u64`.
fn as_u64(expr: &Expr) -> Result<u64> {
    match expr {
        Expr::Literal(ScalarValue::Int64(Some(n)), _) => Ok(*n as u64),
        Expr::Literal(ScalarValue::Int32(Some(n)), _) => Ok(*n as u64),
        Expr::Literal(ScalarValue::UInt64(Some(n)), _) => Ok(*n),
        Expr::Literal(ScalarValue::UInt32(Some(n)), _) => Ok(*n as u64),
        other => plan_err!("expected an integer literal, got {other:?}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_is_stable() {
        let s = event_schema();
        assert_eq!(s.fields().len(), 2);
        assert_eq!(s.field(0).name(), "event_type");
        assert_eq!(s.field(1).name(), "rows");
    }

    #[test]
    fn names_are_registered() {
        assert_eq!(subscribe_table_function_names(), vec!["subscribe_events"]);
    }
}
