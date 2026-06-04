// Copyright (c) 2026 Richard Albright. All rights reserved.

use futures::{StreamExt, TryStreamExt};
use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::Float32Array;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::context::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::Partitioning;
use datafusion::physical_plan::{
    DisplayAs, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};

use crate::core::query::merge_and_rerank_vector_results;

/// ExecutionPlan node that merges vector search results from multiple partitions.
#[derive(Debug)]
pub struct VectorMergeExec {
    pub input: Arc<dyn ExecutionPlan>,
    pub k: usize,
    pub offset: usize,
    schema: SchemaRef,
    properties: PlanProperties,
}

impl VectorMergeExec {
    pub fn new(
        input: Arc<dyn ExecutionPlan>,
        k: usize,
        offset: usize,
        final_schema: SchemaRef,
    ) -> DataFusionResult<Self> {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(final_schema.clone()),
            Partitioning::UnknownPartitioning(1), // Merges all partitions into 1
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Ok(Self {
            input,
            k,
            offset,
            schema: final_schema,
            properties,
        })
    }
}

impl DisplayAs for VectorMergeExec {
    fn fmt_as(
        &self,
        t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            datafusion::physical_plan::DisplayFormatType::Default
            | datafusion::physical_plan::DisplayFormatType::Verbose => {
                write!(f, "VectorMergeExec: k={}, offset={}", self.k, self.offset)
            }
            _ => Ok(()),
        }
    }
}

impl ExecutionPlan for VectorMergeExec {
    fn name(&self) -> &str {
        "VectorMergeExec"
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
        vec![&self.input]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return Err(DataFusionError::Internal(
                "VectorMergeExec requires exactly 1 child".to_string(),
            ));
        }
        Ok(Arc::new(VectorMergeExec::new(
            children[0].clone(),
            self.k,
            self.offset,
            self.schema.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(
                "VectorMergeExec only supports partition 0".to_string(),
            ));
        }

        let input_partitions = self
            .input
            .properties()
            .output_partitioning()
            .partition_count();
        let mut streams = Vec::with_capacity(input_partitions);

        for i in 0..input_partitions {
            streams.push(self.input.execute(i, context.clone())?);
        }

        let k = self.k;
        let offset = self.offset;
        let expected_schema = self.schema.clone();
        let expected_schema_inner = expected_schema.clone();

        let stream = async_stream::stream! {
            // Collect all batches from all child streams concurrently
            let mut futures = futures::stream::FuturesUnordered::new();
            for (i, mut s) in streams.into_iter().enumerate() {
                futures.push(async move {
                    let mut batches = Vec::new();
                    while let Some(batch) = s.next().await {
                        batches.push(batch?);
                    }
                    Ok::<_, DataFusionError>((i, batches))
                });
            }

            let mut all_results = Vec::new();
            while let Some(res) = futures.next().await {
                let (part_idx, batches) = res.map_err(|e| DataFusionError::Execution(e.to_string()))?;
                for batch in batches {
                    if batch.num_rows() == 0 {
                        continue;
                    }

                    // Extract distance column
                    let dist_idx = batch.schema().index_of("distance")
                        .map_err(|e| DataFusionError::Execution(format!("Missing distance column: {}", e)))?;

                    let dist_col = batch.column(dist_idx).as_any().downcast_ref::<Float32Array>()
                        .ok_or_else(|| DataFusionError::Execution("distance column is not Float32".to_string()))?;

                    let mut distances = Vec::with_capacity(batch.num_rows());
                    for i in 0..batch.num_rows() {
                        distances.push(dist_col.value(i));
                    }

                    // Drop distance column to match merge_and_rerank input shape
                    let mut cols = batch.columns().to_vec();
                    cols.remove(dist_idx);
                    let mut fields = batch.schema().fields().to_vec();
                    fields.remove(dist_idx);
                    let base_schema = Arc::new(datafusion::arrow::datatypes::Schema::new(fields));

                    let mut options = datafusion::arrow::record_batch::RecordBatchOptions::default();
                    options.row_count = Some(batch.num_rows());
                    let clean_batch = datafusion::arrow::record_batch::RecordBatch::try_new_with_options(base_schema, cols, &options)
                        .map_err(|e| DataFusionError::Execution(e.to_string()))?;

                    // Use partition index as segment ID (vector scan partitions correspond to segments)
                    let segment_id = format!("segment_{}", part_idx);
                    all_results.push((segment_id, clean_batch, distances));
                }
            }

            // Perform global merge and rerank
            match merge_and_rerank_vector_results(all_results, k, offset) {
                Ok(merged_batches) => {
                    for (_, mut batch) in merged_batches {
                        // Project to final schema
                        // merge_and_rerank_vector_results ALWAYS adds the distance column back.
                        // If final schema doesn't have it, drop it.
                        let has_distance = expected_schema_inner.column_with_name("distance").is_some();
                        let batch_has_distance = batch.schema().column_with_name("distance").is_some();

                        if !has_distance && batch_has_distance {
                            let dist_idx = batch.schema().index_of("distance").unwrap();
                            let mut cols = batch.columns().to_vec();
                            cols.remove(dist_idx);
                            let mut options = datafusion::arrow::record_batch::RecordBatchOptions::default();
                            options.row_count = Some(batch.num_rows());
                            batch = datafusion::arrow::record_batch::RecordBatch::try_new_with_options(expected_schema_inner.clone(), cols, &options)
                                .map_err(|e| DataFusionError::Execution(format!("Schema mismatch: {}", e)))?;
                        } else if has_distance && batch_has_distance {
                            // Ensure schema matches strictly
                            let mut options = datafusion::arrow::record_batch::RecordBatchOptions::default();
                            options.row_count = Some(batch.num_rows());
                            batch = datafusion::arrow::record_batch::RecordBatch::try_new_with_options(expected_schema_inner.clone(), batch.columns().to_vec(), &options)
                                .map_err(|e| DataFusionError::Execution(format!("Schema mismatch: {}", e)))?;
                        }

                        yield Ok(batch);
                    }
                },
                Err(e) => {
                    yield Err(DataFusionError::Execution(e.to_string()));
                }
            }
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            expected_schema,
            Box::pin(stream),
        )))
    }
}
