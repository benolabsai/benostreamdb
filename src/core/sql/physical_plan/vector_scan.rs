// Copyright (c) 2026 Richard Albright. All rights reserved.

use std::any::Any;
use std::sync::Arc;

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

use crate::core::manifest::ManifestEntry;
use crate::core::planner::{FilterExpr, QueryFilter};
use crate::core::query::{execute_vector_search_with_config, VectorSearchRequest};
use crate::core::table::{Table, VectorSearchParams};

/// ExecutionPlan node that performs an HNSW Vector Search across a specific partition of segments.
/// Returns a stream of RecordBatches that include a `distance` column.
#[derive(Debug)]
pub struct VectorScanExec {
    pub table: Arc<Table>,
    pub partitions: Vec<Vec<ManifestEntry>>,
    pub projection: Option<Vec<usize>>,
    pub filter: Option<String>,
    pub vector_params: VectorSearchParams,
    pub limit: Option<usize>,
    base_schema: SchemaRef,
    schema: SchemaRef,
    properties: PlanProperties,
}

impl VectorScanExec {
    pub fn new(
        table: Arc<Table>,
        partitions: Vec<Vec<ManifestEntry>>,
        projection: Option<Vec<usize>>,
        filter: Option<String>,
        vector_params: VectorSearchParams,
        limit: Option<usize>,
        base_schema: SchemaRef,
    ) -> DataFusionResult<Self> {
        let projected_schema = if let Some(ref proj) = projection {
            if proj.iter().any(|&i| i >= base_schema.fields().len()) {
                base_schema.clone()
            } else {
                Arc::new(base_schema.project(proj).map_err(DataFusionError::from)?)
            }
        } else {
            base_schema.clone()
        };

        let mut fields: Vec<datafusion::arrow::datatypes::Field> = projected_schema
            .fields()
            .iter()
            .map(|f| f.as_ref().clone())
            .collect();
        if projected_schema.column_with_name("distance").is_none() {
            fields.push(datafusion::arrow::datatypes::Field::new(
                "distance",
                datafusion::arrow::datatypes::DataType::Float32,
                false,
            ));
        }
        let scan_schema = Arc::new(datafusion::arrow::datatypes::Schema::new(fields));

        let partition_count = partitions.len().max(1);

        let properties = PlanProperties::new(
            EquivalenceProperties::new(scan_schema.clone()),
            Partitioning::UnknownPartitioning(partition_count),
            EmissionType::Final,
            Boundedness::Bounded,
        );

        Ok(Self {
            table,
            partitions,
            projection,
            filter,
            vector_params,
            limit,
            base_schema,
            schema: scan_schema,
            properties,
        })
    }
}

impl DisplayAs for VectorScanExec {
    fn fmt_as(
        &self,
        t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            datafusion::physical_plan::DisplayFormatType::Default
            | datafusion::physical_plan::DisplayFormatType::Verbose => {
                write!(
                    f,
                    "VectorScanExec: column={}, metric={:?}, partitions={}",
                    self.vector_params.column,
                    self.vector_params.metric,
                    self.partitions.len()
                )
            }
            _ => Ok(()),
        }
    }
}

impl ExecutionPlan for VectorScanExec {
    fn name(&self) -> &str {
        "VectorScanExec"
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
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(VectorScanExec::new(
            self.table.clone(),
            self.partitions.clone(),
            self.projection.clone(),
            self.filter.clone(),
            self.vector_params.clone(),
            self.limit,
            self.base_schema.clone(),
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition >= self.partitions.len() && !self.partitions.is_empty() {
            return Err(DataFusionError::Internal(format!(
                "VectorScanExec invalid partition {} (count {})",
                partition,
                self.partitions.len()
            )));
        }

        let table = self.table.clone();
        let filter = self.filter.clone();
        let vector_params = self.vector_params.clone();

        let entries = if self.partitions.is_empty() {
            Vec::new()
        } else {
            self.partitions[partition].clone()
        };

        let original_schema = table.arrow_schema();
        let column_names = self.projection.as_ref().map(|proj| proj.iter()
                    .map(|i| original_schema.field(*i).name().clone())
                    .collect::<Vec<_>>());

        let col_names_owned = column_names;
        let expected_schema = self.schema.clone();
        let expected_schema_inner = expected_schema.clone();

        let stream = async_stream::stream! {
            for entry in entries {
                let filter_expr = if let Some(ref f) = filter {
                     let filters = QueryFilter::parse_multi(f);
                     FilterExpr::from_filters(filters)
                } else {
                    None
                };

                let mut request = VectorSearchRequest::new(
                    vector_params.column.clone(),
                    vector_params.query.clone(),
                    vector_params.k,
                    vector_params.metric,
                )
                .with_filter(filter_expr)
                .with_config(table.query_config().clone())
                .with_ef_search(vector_params.ef_search);

                if let Some(ref proj_names) = col_names_owned {
                    request = request.with_columns(Some(proj_names.clone()));
                }

                // Currently executes a single segment search.
                // Wait, if it executes single segment, it uses query.rs's execute_vector_search_with_config
                match execute_vector_search_with_config(
                    vec![entry.clone()],
                    table.object_store(),
                    None,
                    &table.table_uri(),
                    request,
                ).await {
                    Ok(batches) => {
                        for (_segment_id, batch) in batches {
                            let mut b = batch;
                            // Check if distance column exists (which it should from vector search)
                            if b.num_columns() == expected_schema_inner.fields().len() {
                                let mut options = datafusion::arrow::record_batch::RecordBatchOptions::default();
                                options.row_count = Some(b.num_rows());
                                b = datafusion::arrow::record_batch::RecordBatch::try_new_with_options(expected_schema_inner.clone(), b.columns().to_vec(), &options)
                                    .map_err(|e| DataFusionError::Execution(format!("Type mismatch in Vector Search: {}. Expected {:?} got {:?}", e, expected_schema_inner, b.schema())))?;
                            } else {
                                yield Err(DataFusionError::Execution(format!("Field count mismatch in Vector Search: Expected {} fields, got {}", expected_schema_inner.fields().len(), b.schema().fields().len())));
                                return;
                            }
                            yield Ok(b);
                        }
                    },
                    Err(e) => {
                        // Graceful degradation: log warning and skip this segment
                        // rather than failing the entire query
                        tracing::warn!(
                            error = %e,
                            "Vector search failed for segment — skipping. \
                             The index layer should fall back to flat scan; if this \
                             persists, check segment data accessibility."
                        );
                    }
                }
            }
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            expected_schema,
            Box::pin(stream),
        )))
    }
}
