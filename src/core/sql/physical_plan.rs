// Copyright (c) 2026 Richard Albright. All rights reserved.

use std::any::Any;
use std::sync::Arc;
pub mod index_join;

use datafusion::physical_plan::{
    ExecutionPlan, 
    SendableRecordBatchStream, 
    DisplayAs, 
    PlanProperties
};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::execution::context::TaskContext;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::error::{Result, DataFusionError};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::Partitioning;

use crate::core::table::{Table, VectorSearchParams};
use crate::core::manifest::ManifestEntry;

#[derive(Debug)]
pub struct HyperStreamExec {
    pub table: Arc<Table>,
    // Partitions: Each partition is a list of segments to read
    pub partitions: Vec<Vec<ManifestEntry>>,
    projection: Option<Vec<usize>>,
    filter: Option<String>,
    pub vector_params: Option<VectorSearchParams>,
    limit: Option<usize>,
    base_schema: SchemaRef,  // Original table schema for projection
    schema: SchemaRef,        // Projected schema
    properties: PlanProperties,
}

impl HyperStreamExec {
    pub fn new(
        table: Arc<Table>,
        partitions: Vec<Vec<ManifestEntry>>,
        projection: Option<Vec<usize>>,
        filter: Option<String>,
        vector_params: Option<VectorSearchParams>,
        limit: Option<usize>,
        base_schema: SchemaRef,
    ) -> Result<Self> {
        // Calculate projected schema
        let projected_schema = if let Some(ref proj) = projection {
            // Validate projection indices
            if proj.iter().any(|&i| i >= base_schema.fields().len()) {
                // If projection is invalid, use base schema
                base_schema.clone()
            } else {
                Arc::new(base_schema.project(proj)
                    .map_err(|e| DataFusionError::from(e))?)
            }
        } else {
            base_schema.clone()
        };

        let partition_count = partitions.len().max(1);

        let properties = PlanProperties::new(
            EquivalenceProperties::new(projected_schema.clone()),
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
            schema: projected_schema,
            properties,
        })
    }
    pub fn projection(&self) -> Option<&Vec<usize>> {
        self.projection.as_ref()
    }

    pub fn filter_str(&self) -> Option<&str> {
        self.filter.as_deref()
    }
}

impl DisplayAs for HyperStreamExec {
    fn fmt_as(
        &self,
        t: datafusion::physical_plan::DisplayFormatType,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match t {
            datafusion::physical_plan::DisplayFormatType::Default | datafusion::physical_plan::DisplayFormatType::Verbose => {
                write!(f, "HyperStreamExec: partitions={}, filter={:?}, projection={:?}, limit={:?}", self.partitions.len(), self.filter, self.projection, self.limit)
            }
            _ => Ok(()),
        }
    }
}

impl ExecutionPlan for HyperStreamExec {
    fn name(&self) -> &str {
        "HyperStreamExec"
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
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(HyperStreamExec::new(
            self.table.clone(),
            self.partitions.clone(),
            self.projection.clone(),
            self.filter.clone(),
            self.vector_params.clone(),
            self.limit,
            self.base_schema.clone(),  // Use base schema for reprojection
        )?))
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition >= self.partitions.len() && !self.partitions.is_empty() {
            return Err(DataFusionError::Internal(format!(
                "HyperStreamExec invalid partition {} (count {})",
                partition, self.partitions.len()
            )));
        }

        let table = self.table.clone();
        let filter = self.filter.clone();
        let vector_params = self.vector_params.clone();
        
        // If no partitions (empty table), return empty stream
        let entries = if self.partitions.is_empty() {
            Vec::new()
        } else {
            self.partitions[partition].clone()
        };

        // Resolve usage of projection to column names
        let original_schema = table.arrow_schema();
        let column_names = if let Some(ref proj) = self.projection {
             let names: Vec<String> = proj.iter()
                 .map(|i| original_schema.field(*i).name().clone())
                 .collect();
             Some(names)
        } else {
             None
        };
        
        // Pre-convert column names to &str slice
        let col_names_owned = column_names;
        
        let expected_schema = self.schema.clone();
        let expected_schema_inner = expected_schema.clone();
        use crate::core::planner::{QueryFilter, FilterExpr};
        
        let stream = async_stream::stream! {            
            // For each segment in this partition
            for entry in entries {
                let col_refs: Option<Vec<&str>> = col_names_owned.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());
                let col_slice = col_refs.as_deref();

                // Apply filter parsing inside the loop or pre-parse?
                // read_segment handles parsing if we pass QueryFilter.
                // But here we have string filter.
                // Better to parse once? 
                // Table::read_segment takes `query_filter_opt: Option<&QueryFilter>`
                
                let filter_expr = if let Some(ref f) = filter {
                     let filters = QueryFilter::parse_multi(f);
                     FilterExpr::from_filters(filters)
                } else {
                    None
                };
                
                // Read Segment
                if let Some(ref vp) = vector_params {
                    // Hybrid Search: Vector + Scalar
                    use crate::core::query::{execute_vector_search_with_config, VectorSearchRequest};
                    
                    let mut request = VectorSearchRequest::new(
                        vp.column.clone(),
                        vp.query.clone(),
                        vp.k,
                        vp.metric,
                    )
                    .with_filter(filter_expr)
                    .with_config(table.query_config().clone())
                    .with_ef_search(vp.ef_search);

                    // Pass projected columns if available
                    if let Some(ref proj_names) = col_names_owned {
                        request = request.with_columns(Some(proj_names.clone()));
                    }
                    
                    match execute_vector_search_with_config(
                        vec![entry.clone()],
                        table.object_store(),
                        None, // data_store
                        &table.table_uri(),
                        request,
                    ).await {
                        Ok(batches) => {
                            for batch in batches {
                                let mut b = batch;
                                
                                // If batch has one extra column (distance) that isn't in expected schema, drop it
                                if b.num_columns() == expected_schema_inner.fields().len() + 1 && b.schema().fields().last().unwrap().name() == "distance" {
                                    let mut cols = b.columns().to_vec();
                                    cols.pop();
                                    let mut options = datafusion::arrow::record_batch::RecordBatchOptions::default();
                                    options.row_count = Some(b.num_rows());
                                    b = datafusion::arrow::record_batch::RecordBatch::try_new_with_options(expected_schema_inner.clone(), cols, &options)
                                        .map_err(|e| DataFusionError::Execution(format!("Schema mismatch in Hybrid Search: {}", e)))?;
                                } else if b.num_columns() == expected_schema_inner.fields().len() {
                                    // Soft-replace schema to ignore metadata mismatches
                                    let mut options = datafusion::arrow::record_batch::RecordBatchOptions::default();
                                    options.row_count = Some(b.num_rows());
                                    let b_new = datafusion::arrow::record_batch::RecordBatch::try_new_with_options(expected_schema_inner.clone(), b.columns().to_vec(), &options)
                                        .map_err(|e| DataFusionError::Execution(format!("Type mismatch in Hybrid Search: {}. Expected {:?} got {:?}", e, expected_schema_inner, b.schema())))?;
                                    b = b_new;
                                } else {
                                    tracing::error!("Field count mismatch: Expected {:?}, Got {:?}", expected_schema_inner.fields().iter().map(|f| f.name()).collect::<Vec<_>>(), b.schema().fields().iter().map(|f| f.name()).collect::<Vec<_>>());
                                    yield Err(DataFusionError::Execution(format!("Field count mismatch in Hybrid Search: Expected {} fields, got {}", expected_schema_inner.fields().len(), b.schema().fields().len())));
                                    return;
                                }
                                yield Ok(b);
                            }
                        },
                        Err(e) => yield Err(DataFusionError::Execution(e.to_string())),
                    }
                } else {
                    // Standard scan
                    let version = 1; 
                    let query_filter = if let Some(ref f) = filter {
                         QueryFilter::parse_multi(f).into_iter().next()
                    } else { None };
                    
                    match table.read_segment(&entry, query_filter.as_ref(), version, col_slice).await {
                        Ok(batches) => {
                            for batch in batches {
                                let mut b = batch;
                                if b.schema().fields().len() == expected_schema_inner.fields().len() {
                                    // Soft-replace schema to ignore metadata mismatches
                                    let mut options = datafusion::arrow::record_batch::RecordBatchOptions::default();
                                    options.row_count = Some(b.num_rows());
                                    let b_new = datafusion::arrow::record_batch::RecordBatch::try_new_with_options(expected_schema_inner.clone(), b.columns().to_vec(), &options)
                                        .map_err(|e| DataFusionError::Execution(format!("Type mismatch in standard scan: {}. Expected {:?} got {:?}", e, expected_schema_inner, b.schema())))?;
                                    b = b_new;
                                } else {
                                    yield Err(DataFusionError::Execution(format!("Field count mismatch in standard scan: Expected {} fields, got {}", expected_schema_inner.fields().len(), b.schema().fields().len())));
                                    return;
                                }
                                yield Ok(b);
                            }
                        },
                        Err(e) => yield Err(DataFusionError::Execution(e.to_string())),
                    }
                }
            }

            // If this is partition 0, also include in-memory write buffer data
            if partition == 0 {
                let col_refs: Option<Vec<&str>> = col_names_owned.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());
                let col_slice = col_refs.as_deref();
                
                let query_filter = if let Some(ref f) = filter {
                    use crate::core::planner::QueryFilter;
                    QueryFilter::parse(f)
                } else {
                    None
                };

                match table.read_write_buffer(query_filter.as_ref(), col_slice) {
                    Ok(batches) => {
                        for batch in batches {
                            if batch.schema() != expected_schema_inner {
                                yield Err(DataFusionError::Execution("Write buffer schema mismatch".to_string()));
                                return;
                            }
                            yield Ok(batch);
                        }
                    },
                    Err(e) => yield Err(DataFusionError::Execution(e.to_string())),
                }
            }
        };

        Ok(Box::pin(RecordBatchStreamAdapter::new(
            expected_schema,
            Box::pin(stream),
        )))
    }
}
