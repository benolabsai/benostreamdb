// Copyright (c) 2026 Richard Albright. All rights reserved.

use std::any::Any;
use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::{Array, ArrayRef};
use arrow::compute::{concat_batches, take};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow::row::{RowConverter, SortField};
use datafusion::common::Result;
use datafusion::execution::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, RecordBatchStream,
    SendableRecordBatchStream,
};
use futures::{Future, Stream, StreamExt};

use crate::core::planner::QueryFilter;
use crate::core::table::Table;
use serde_json::Value;

#[derive(Debug)]
pub struct HyperStreamIndexJoinExec {
    pub left: Arc<dyn ExecutionPlan>,
    pub right_table: Arc<Table>,
    pub left_on: Vec<Arc<dyn PhysicalExpr>>,
    pub right_cols: Vec<String>,
    pub schema: SchemaRef,
    pub properties: PlanProperties,
}

impl HyperStreamIndexJoinExec {
    pub fn new(
        left: Arc<dyn ExecutionPlan>,
        right_table: Arc<Table>,
        left_on: Vec<Arc<dyn PhysicalExpr>>,
        right_cols: Vec<String>,
        schema: SchemaRef,
    ) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            datafusion::physical_plan::Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        );
        Self {
            left,
            right_table,
            left_on,
            right_cols,
            schema,
            properties,
        }
    }
}

impl DisplayAs for HyperStreamIndexJoinExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let left_str = self
            .left_on
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        let right_str = self.right_cols.join(", ");
        write!(
            f,
            "HyperStreamIndexJoinExec: on ({}) = ({})",
            left_str, right_str
        )
    }
}

impl ExecutionPlan for HyperStreamIndexJoinExec {
    fn name(&self) -> &str {
        "HyperStreamIndexJoinExec"
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn properties(&self) -> &PlanProperties {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.left]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(HyperStreamIndexJoinExec::new(
            children[0].clone(),
            self.right_table.clone(),
            self.left_on.clone(),
            self.right_cols.clone(),
            self.schema.clone(),
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let left_stream = self.left.execute(partition, context)?;

        Ok(Box::pin(IndexJoinStream {
            left_stream,
            right_table: self.right_table.clone(),
            left_on: self.left_on.clone(),
            right_cols: self.right_cols.clone(),
            output_schema: self.schema.clone(),
            current_future: None,
        }))
    }
}

struct IndexJoinStream {
    left_stream: SendableRecordBatchStream,
    right_table: Arc<Table>,
    left_on: Vec<Arc<dyn PhysicalExpr>>,
    right_cols: Vec<String>,
    output_schema: SchemaRef,
    current_future: Option<tokio::task::JoinHandle<Result<Option<RecordBatch>>>>,
}

impl Stream for IndexJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(mut fut) = self.current_future.take() {
                match Pin::new(&mut fut).poll(cx) {
                    Poll::Ready(Ok(res)) => {
                        return Poll::Ready(res.transpose());
                    }
                    Poll::Ready(Err(e)) => {
                        return Poll::Ready(Some(Err(
                            datafusion::error::DataFusionError::Execution(format!(
                                "Join error: {}",
                                e
                            )),
                        )));
                    }
                    Poll::Pending => {
                        self.current_future = Some(fut);
                        return Poll::Pending;
                    }
                }
            }

            match self.left_stream.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(left_batch))) => {
                    let right_table = self.right_table.clone();
                    let right_cols = self.right_cols.clone();
                    let left_on = self.left_on.clone();
                    let output_schema = self.output_schema.clone();

                    let fut = tokio::spawn(async move {
                        process_join_batch(
                            left_batch,
                            right_table,
                            left_on,
                            right_cols,
                            output_schema,
                        )
                        .await
                    });
                    self.current_future = Some(fut);
                }
                Poll::Ready(Some(Err(e))) => return Poll::Ready(Some(Err(e))),
                Poll::Ready(None) => return Poll::Ready(None),
                Poll::Pending => return Poll::Pending,
            }
        }
    }
}

impl RecordBatchStream for IndexJoinStream {
    fn schema(&self) -> SchemaRef {
        self.output_schema.clone()
    }
}

async fn process_join_batch(
    left_batch: RecordBatch,
    table: Arc<Table>,
    left_on: Vec<Arc<dyn PhysicalExpr>>,
    right_cols: Vec<String>,
    output_schema: SchemaRef,
) -> Result<Option<RecordBatch>> {
    // 1. Evaluate join keys on left batch
    let mut left_keys_arrays = Vec::with_capacity(left_on.len());
    for expr in &left_on {
        let keys = expr.evaluate(&left_batch)?;
        left_keys_arrays.push(keys.into_array(left_batch.num_rows())?);
    }

    if left_keys_arrays.is_empty() || left_batch.num_rows() == 0 {
        return Ok(None);
    }

    // 2. Extract distinct keys for query
    let mut filters = Vec::with_capacity(right_cols.len());
    for (i, right_col) in right_cols.iter().enumerate() {
        let distinct_values = extract_distinct_values(&left_keys_arrays[i])?;
        if distinct_values.is_empty() {
            return Ok(None);
        }
        filters.push(QueryFilter {
            column: right_col.clone(),
            min: None,
            min_inclusive: false,
            max: None,
            max_inclusive: false,
            values: Some(distinct_values),
            negated: false,
        });
    }

    // 3. Query Right Table
    let right_batches = table
        .read_filter_async(filters, None, None)
        .await
        .map_err(|e| {
            datafusion::error::DataFusionError::Execution(format!("HyperStream read error: {}", e))
        })?;

    if right_batches.is_empty() {
        return Ok(None);
    }

    // Concatenate all right batches into one
    let right_schema = right_batches[0].schema();
    let right_batch_concat = concat_batches(&right_schema, &right_batches)?;

    // 4. Perform In-Memory Join using RowConverter
    perform_join(
        &left_batch,
        &left_keys_arrays,
        &right_batch_concat,
        &right_cols,
        &output_schema,
    )
}

fn extract_distinct_values(array: &ArrayRef) -> Result<Vec<Value>> {
    let mut values = Vec::new();
    match array.data_type() {
        DataType::Int64 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
                .unwrap();
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    values.push(Value::Number(arr.value(i).into()));
                }
            }
        }
        DataType::Int32 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow::array::Int32Array>()
                .unwrap();
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    values.push(Value::Number(arr.value(i).into()));
                }
            }
        }
        DataType::Utf8 => {
            let arr = array
                .as_any()
                .downcast_ref::<arrow::array::StringArray>()
                .unwrap();
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    values.push(Value::String(arr.value(i).to_string()));
                }
            }
        }
        _ => {}
    }
    // Dedup
    values.sort_by(|a: &Value, b: &Value| {
        a.as_string_repr()
            .partial_cmp(&b.as_string_repr())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    values.dedup();
    Ok(values)
}

trait ValueExt {
    fn as_string_repr(&self) -> String;
}
impl ValueExt for Value {
    fn as_string_repr(&self) -> String {
        match self {
            Value::Number(n) => n.to_string(),
            Value::String(s) => s.clone(),
            _ => format!("{:?}", self),
        }
    }
}

fn perform_join(
    left: &RecordBatch,
    left_keys: &[ArrayRef],
    right: &RecordBatch,
    right_cols: &[String],
    output_schema: &SchemaRef,
) -> Result<Option<RecordBatch>> {
    // 1. Build Index on Right Batch using RowConverter
    let mut right_keys = Vec::with_capacity(right_cols.len());
    for col_name in right_cols {
        let col = right.column_by_name(col_name).ok_or_else(|| {
            datafusion::error::DataFusionError::Execution(format!(
                "Right join col {} missing",
                col_name
            ))
        })?;
        right_keys.push(col.clone());
    }

    let sort_fields = left_keys
        .iter()
        .map(|a| SortField::new(a.data_type().clone()))
        .collect::<Vec<_>>();
    let converter = RowConverter::new(sort_fields).map_err(|e| {
        datafusion::error::DataFusionError::Execution(format!("RowConverter error: {}", e))
    })?;

    let right_rows = converter.convert_columns(&right_keys).map_err(|e| {
        datafusion::error::DataFusionError::Execution(format!("RowConverter error: {}", e))
    })?;

    let mut right_map: HashMap<Vec<u8>, Vec<usize>> = HashMap::new();

    for (idx, row) in right_rows.iter().enumerate() {
        right_map
            .entry(row.as_ref().to_vec())
            .or_default()
            .push(idx);
    }

    // 2. Probe with Left Batch
    let left_rows = converter.convert_columns(left_keys).map_err(|e| {
        datafusion::error::DataFusionError::Execution(format!("RowConverter error: {}", e))
    })?;

    let mut left_indices_builder = arrow::array::UInt64Builder::new();
    let mut right_indices_builder = arrow::array::UInt64Builder::new();

    for (l_idx, row) in left_rows.iter().enumerate() {
        if let Some(r_indices) = right_map.get(row.as_ref()) {
            for &r_idx in r_indices {
                left_indices_builder.append_value(l_idx as u64);
                right_indices_builder.append_value(r_idx as u64);
            }
        }
    }

    let left_indices = left_indices_builder.finish();
    let right_indices = right_indices_builder.finish();

    if left_indices.is_empty() {
        return Ok(None);
    }

    // 3. Interleave / Take
    let mut output_columns = Vec::with_capacity(left.num_columns() + right.num_columns());

    for col in left.columns() {
        output_columns.push(take(col, &left_indices, None)?);
    }

    for col in right.columns() {
        output_columns.push(take(col, &right_indices, None)?);
    }

    if output_columns.len() != output_schema.fields().len() {
        // Schema mismatch logic handled upstream
    }

    let batch = RecordBatch::try_new(output_schema.clone(), output_columns)?;
    Ok(Some(batch))
}
