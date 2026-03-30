// Copyright (c) 2026 Richard Albright. All rights reserved.

use std::any::Any;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::collections::HashMap;

use arrow::datatypes::{SchemaRef, DataType};
use arrow::record_batch::RecordBatch;
use arrow::array::{ArrayRef, Array};
use arrow::compute::{concat_batches, take};
use futures::{Stream, StreamExt, Future};
use datafusion::physical_plan::{
    ExecutionPlan, SendableRecordBatchStream, DisplayAs, DisplayFormatType, PlanProperties, RecordBatchStream
};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::execution::TaskContext;
use datafusion::common::Result;
use datafusion::physical_expr::PhysicalExpr;

use crate::core::table::Table;
use crate::core::planner::QueryFilter;
use serde_json::Value;

#[derive(Debug)]
pub struct HyperStreamIndexJoinExec {
    pub left: Arc<dyn ExecutionPlan>,
    pub right_table: Arc<Table>,
    pub left_on: Arc<dyn PhysicalExpr>,
    pub right_col: String,
    pub schema: SchemaRef,
    pub properties: PlanProperties,
}

impl HyperStreamIndexJoinExec {
    pub fn new(
        left: Arc<dyn ExecutionPlan>,
        right_table: Arc<Table>,
        left_on: Arc<dyn PhysicalExpr>,
        right_col: String,
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
            right_col,
            schema,
            properties,
        }
    }
}

impl DisplayAs for HyperStreamIndexJoinExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "HyperStreamIndexJoinExec: on {} = {}", self.left_on, self.right_col)
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
            self.right_col.clone(),
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
            right_col: self.right_col.clone(),
            output_schema: self.schema.clone(),
            current_future: None,
        }))
    }
}

struct IndexJoinStream {
    left_stream: SendableRecordBatchStream,
    right_table: Arc<Table>,
    left_on: Arc<dyn PhysicalExpr>,
    right_col: String,
    output_schema: SchemaRef,
    current_future: Option<tokio::task::JoinHandle<Result<Option<RecordBatch>>>>,
}

impl Stream for IndexJoinStream {
    type Item = Result<RecordBatch>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            // Check pending future
            if let Some(mut fut) = self.current_future.take() {
                match Pin::new(&mut fut).poll(cx) {
                    Poll::Ready(Ok(res)) => {
                        return Poll::Ready(res.transpose());
                    }
                    Poll::Ready(Err(e)) => {
                        return Poll::Ready(Some(Err(datafusion::error::DataFusionError::Execution(format!("Join error: {}", e)))));
                    }
                    Poll::Pending => {
                        self.current_future = Some(fut);
                        return Poll::Pending;
                    }
                }
            }
            
            // Poll left stream
            match self.left_stream.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(left_batch))) => {
                    let right_table = self.right_table.clone();
                    let right_col = self.right_col.clone();
                    let left_on = self.left_on.clone();
                    let output_schema = self.output_schema.clone();

                    let fut = tokio::spawn(async move {
                        process_join_batch(left_batch, right_table, left_on, right_col, output_schema).await
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
    left_on: Arc<dyn PhysicalExpr>,
    right_col: String,
    output_schema: SchemaRef,
) -> Result<Option<RecordBatch>> {
    // 1. Evaluate join key on left batch
    let keys = left_on.evaluate(&left_batch)?;
    let keys_array = keys.into_array(left_batch.num_rows())?;

    // 2. Extract distinct keys for query
    let distinct_values = extract_distinct_values(&keys_array)?;
    
    if distinct_values.is_empty() {
        return Ok(None);
    }
    
    // 3. Query Right Table
    let filter = QueryFilter {
        column: right_col.clone(),
        min: None,
        min_inclusive: false,
        max: None,
        max_inclusive: false,
        values: Some(distinct_values),
        negated: false,
    };
    
    let right_batches = table.read_filter_async(vec![filter], None, None).await
        .map_err(|e| datafusion::error::DataFusionError::Execution(format!("HyperStream read error: {}", e)))?;
        
    if right_batches.is_empty() {
        return Ok(None);
    }
    
    // Concatenate all right batches into one
    // We assume schema is consistent
    let right_schema = right_batches[0].schema();
    let right_batch_concat = concat_batches(&right_schema, &right_batches)?;

    // 4. Perform In-Memory Join using Indices
    perform_join(&left_batch, &keys_array, &right_batch_concat, &right_col, &output_schema)
}

fn extract_distinct_values(array: &ArrayRef) -> Result<Vec<Value>> {
    let mut values = Vec::new();
    // Simplified: handle common types
    match array.data_type() {
        DataType::Int64 => {
            let arr = array.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    values.push(Value::Number(arr.value(i).into()));
                }
            }
        },
        DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
            for i in 0..arr.len() {
                 if !arr.is_null(i) {
                    values.push(Value::Number(arr.value(i).into()));
                }
            }
        },
        DataType::Utf8 => {
            let arr = array.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
            for i in 0..arr.len() {
                if !arr.is_null(i) {
                    values.push(Value::String(arr.value(i).to_string()));
                }
            }
        },
        _ => {}
    }
    // Dedup
    values.sort_by(|a: &Value, b: &Value| a.as_string_repr().partial_cmp(&b.as_string_repr()).unwrap_or(std::cmp::Ordering::Equal));
    values.dedup();
    Ok(values)
}

// Helper to sort/dedup
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
    left_keys: &ArrayRef,
    right: &RecordBatch,
    right_col: &str,
    output_schema: &SchemaRef,
) -> Result<Option<RecordBatch>> {
    // 1. Build Index on Right Batch
    let right_keys_arr = right.column_by_name(right_col)
        .ok_or_else(|| datafusion::error::DataFusionError::Execution("Right join col missing".into()))?;
        
    // MultiMap: Key -> [indices]
    // Uses String representation for key to handle mix types easily in MVP
    let mut right_map: HashMap<String, Vec<usize>> = HashMap::new();
    
    // Extract keys from right batch
    // Using string conversion is inefficient but safe generic approach for MVP
    let right_key_strings = array_to_strings(right_keys_arr);
    for (idx, key_opt) in right_key_strings.iter().enumerate() {
        if let Some(key) = key_opt {
            right_map.entry(key.clone()).or_default().push(idx);
        }
    }
    
    // 2. Probe with Left Batch
    let left_key_strings = array_to_strings(left_keys);
    
    // Builders for indices
    let mut left_indices_builder = arrow::array::UInt64Builder::new();
    let mut right_indices_builder = arrow::array::UInt64Builder::new();
    
    for (l_idx, key_opt) in left_key_strings.iter().enumerate() {
        if let Some(key) = key_opt {
             if let Some(r_indices) = right_map.get(key) {
                 for &r_idx in r_indices {
                     left_indices_builder.append_value(l_idx as u64);
                     right_indices_builder.append_value(r_idx as u64);
                 }
             }
        }
    }
    
    let left_indices = left_indices_builder.finish();
    let right_indices = right_indices_builder.finish();
    
    if left_indices.is_empty() {
        return Ok(None);
    }
    
    // 3. Interleave / Take
    // We cannot use interleave directly because we are combining two batches.
    // Use `take` on each column.
    
    // Reconstruct Left columns
    let left_indices_arr = left_indices;
    let right_indices_arr = right_indices;
    
    let mut output_columns = Vec::new();
    
    // Take from Left
    for col in left.columns() {
        let taken = take(col, &left_indices_arr, None)?;
        output_columns.push(taken);
    }
    
    // Take from Right
    for col in right.columns() {
        let taken = take(col, &right_indices_arr, None)?;
        output_columns.push(taken);
    }
    
    // Verify column count matches schema
    if output_columns.len() != output_schema.fields().len() {
        // Mismatch usually due to key duplication or schema issue in creating IndexJoinExec
        // In simplest case: schema = Left fields + Right fields.
        // We just appended them in that order.
    }
    
    let batch = RecordBatch::try_new(output_schema.clone(), output_columns)?;
    Ok(Some(batch))
}

fn array_to_strings(arr: &ArrayRef) -> Vec<Option<String>> {
    let mut res = Vec::with_capacity(arr.len());
    match arr.data_type() {
        DataType::Int64 => {
            let a = arr.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap();
            for i in 0..a.len() {
                if a.is_null(i) { res.push(None); } else { res.push(Some(a.value(i).to_string())); }
            }
        },
        DataType::Int32 => {
            let a = arr.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap();
            for i in 0..a.len() {
                if a.is_null(i) { res.push(None); } else { res.push(Some(a.value(i).to_string())); }
            }
        },
        DataType::Utf8 => {
            let a = arr.as_any().downcast_ref::<arrow::array::StringArray>().unwrap();
            for i in 0..a.len() {
                if a.is_null(i) { res.push(None); } else { res.push(Some(a.value(i).to_string())); }
            }
        },
        _ => {
             // Fallback for debug: None
            for _ in 0..arr.len() { res.push(None); }
        }
    }
    res
}
