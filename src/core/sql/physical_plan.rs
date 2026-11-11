// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use std::any::Any;
use std::sync::Arc;
pub mod index_join;
pub mod vector_merge;
pub mod vector_scan;
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::error::{DataFusionError, Result};
use datafusion::execution::context::TaskContext;
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::Partitioning;
use datafusion::physical_plan::{
    DisplayAs, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};

use crate::core::manifest::ManifestEntry;
use crate::core::table::Table;

#[derive(Debug)]
pub struct BenoStreamExec {
    pub table: Arc<Table>,
    // Partitions: Each partition is a list of segments to read
    pub partitions: Vec<Vec<ManifestEntry>>,
    projection: Option<Vec<usize>>,
    filter: Option<String>,
    limit: Option<usize>,
    base_schema: SchemaRef, // Original table schema for projection
    schema: SchemaRef,      // Projected schema
    properties: PlanProperties,
    /// Human-readable breakdown of why segments were pruned by partition /
    /// statistics rules, surfaced in `EXPLAIN` output.
    pruning_summary: Option<String>,
    /// Uncommitted write-buffer batches captured **at plan time**, from the same
    /// [`Table::consistent_scan_view`] call that produced `partitions`.
    ///
    /// Re-reading the live buffer in `execute()` would let a flush publish its
    /// manifest between planning and execution, so a row would be returned once
    /// from the new segment and once from the still-staged batch. Pinning the
    /// captured set makes the whole query read one coherent snapshot.
    write_buffer: Arc<Vec<arrow::record_batch::RecordBatch>>,
}

impl BenoStreamExec {
    pub fn new(
        table: Arc<Table>,
        partitions: Vec<Vec<ManifestEntry>>,
        projection: Option<Vec<usize>>,
        filter: Option<String>,
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
                Arc::new(base_schema.project(proj).map_err(DataFusionError::from)?)
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
            limit,
            base_schema,
            schema: projected_schema,
            properties,
            pruning_summary: None,
            write_buffer: Arc::new(Vec::new()),
        })
    }

    /// Attach a pruning-reason breakdown to be shown in `EXPLAIN` output.
    pub fn with_pruning_summary(mut self, summary: Option<String>) -> Self {
        self.pruning_summary = summary;
        self
    }

    /// Attach the write-buffer batches captured alongside `partitions`.
    ///
    /// The batches must come from the same consistent scan snapshot that produced
    /// the partitions; `BenoStreamTableProvider::scan` is the only intended caller.
    pub fn with_write_buffer(mut self, buffer: Arc<Vec<arrow::record_batch::RecordBatch>>) -> Self {
        self.write_buffer = buffer;
        self
    }

    pub fn projection(&self) -> Option<&Vec<usize>> {
        self.projection.as_ref()
    }

    pub fn base_schema(&self) -> &SchemaRef {
        &self.base_schema
    }

    pub fn filter_str(&self) -> Option<&str> {
        self.filter.as_deref()
    }
}

fn normalize_batch_to_expected(
    b: arrow::record_batch::RecordBatch,
    expected_schema_inner: &SchemaRef,
) -> Result<arrow::record_batch::RecordBatch> {
    if !expected_schema_inner.fields().is_empty()
        && b.schema().fields().len() == expected_schema_inner.fields().len()
    {
        let mut cols: Vec<Arc<dyn arrow::array::Array>> = Vec::with_capacity(b.num_columns());
        let mut ok = true;
        for (col, field) in b
            .columns()
            .iter()
            .zip(expected_schema_inner.fields().iter())
        {
            if col.data_type() == field.data_type() {
                cols.push(col.clone());
            } else {
                match arrow::compute::cast(col.as_ref(), field.data_type()) {
                    Ok(c) => cols.push(c),
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
            }
        }
        if ok {
            arrow::record_batch::RecordBatch::try_new(expected_schema_inner.clone(), cols)
                .map_err(DataFusionError::from)
        } else {
            Ok(b)
        }
    } else {
        Ok(b)
    }
}

impl DisplayAs for BenoStreamExec {
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
                    "BenoStreamExec: partitions={}, filter={:?}, projection={:?}, limit={:?}",
                    self.partitions.len(),
                    self.filter,
                    self.projection,
                    self.limit
                )?;
                if let Some(ref summary) = self.pruning_summary {
                    write!(f, ", pruning=[{}]", summary)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

impl ExecutionPlan for BenoStreamExec {
    fn name(&self) -> &str {
        "BenoStreamExec"
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
        Ok(Arc::new(
            BenoStreamExec::new(
                self.table.clone(),
                self.partitions.clone(),
                self.projection.clone(),
                self.filter.clone(),
                self.limit,
                self.base_schema.clone(), // Use base schema for reprojection
            )?
            .with_pruning_summary(self.pruning_summary.clone())
            .with_write_buffer(self.write_buffer.clone()),
        ))
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition >= self.partitions.len() && !self.partitions.is_empty() {
            return Err(DataFusionError::Internal(format!(
                "BenoStreamExec invalid partition {} (count {})",
                partition,
                self.partitions.len()
            )));
        }

        let table = self.table.clone();
        let filter = self.filter.clone();
        // Clone the Arc so the stream owns the captured buffer rather than
        // borrowing `self`, which would tie the returned stream to `&self`.
        let write_buffer = self.write_buffer.clone();

        // If no partitions (empty table), return empty stream
        let entries = if self.partitions.is_empty() {
            Vec::new()
        } else {
            self.partitions[partition].clone()
        };

        // Resolve projection against the plan's base_schema so column indices
        // match the snapshot planned at query creation.
        let base_schema = self.base_schema.clone();
        let column_names = self.projection.as_ref().map(|proj| {
            proj.iter()
                .filter_map(|&i| base_schema.fields().get(i).map(|f| f.name().clone()))
                .collect::<Vec<_>>()
        });

        let col_names_owned = column_names;

        let expected_schema = self.schema.clone();
        let expected_schema_inner = expected_schema.clone();
        use crate::core::planner::{FilterExpr, QueryFilter};

        let stream = async_stream::stream! {
            let filter_expr = if let Some(ref f) = filter {
                match FilterExpr::parse_sql(f, base_schema.clone()).await {
                    Ok(fe) => Some(fe),
                    Err(e) => {
                        tracing::debug!(error = %e, filter = %f, "BenoStreamExec::execute parse_sql fallback");
                        FilterExpr::from_filters(QueryFilter::parse_multi(f))
                    }
                }
            } else {
                None
            };

            let manifest_opt = table.manifest().await.ok();
            let iceberg_schema = manifest_opt.as_ref().and_then(|m| {
                m.schemas
                    .iter()
                    .find(|s| s.schema_id == m.current_schema_id)
                    .or_else(|| m.schemas.last())
                    .cloned()
            });

            // For each segment in this partition
            for entry in entries {
                let col_refs: Option<Vec<&str>> = col_names_owned.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());
                let col_slice = col_refs.as_deref();

                match table
                    .read_segment_expr(
                        &entry,
                        filter_expr.as_ref(),
                        1,
                        col_slice,
                        iceberg_schema.as_ref(),
                    )
                    .await
                {
                    Ok(batches) => {
                        for b in batches {
                            match normalize_batch_to_expected(b, &expected_schema_inner) {
                                Ok(rb) => yield Ok(rb),
                                Err(e) => yield Err(e),
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to read segment {}: {}", entry.file_path, e);
                    }
                }
            }

            // If this is partition 0, also include the write-buffer batches that
            // were captured with the manifest at plan time.
            if partition == 0 {
                let col_refs: Option<Vec<&str>> = col_names_owned.as_ref().map(|v| v.iter().map(|s| s.as_str()).collect());
                let col_slice = col_refs.as_deref();

                match table.filter_write_buffer_batches(
                    write_buffer.as_ref(),
                    filter_expr.as_ref(),
                    col_slice,
                ) {
                    Ok(batches) => {
                        for batch in batches {
                            match normalize_batch_to_expected(batch, &expected_schema_inner) {
                                Ok(rb) => yield Ok(rb),
                                Err(e) => {
                                    yield Err(e);
                                    return;
                                }
                            }
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
