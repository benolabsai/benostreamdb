// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

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
use crate::core::query::{execute_multi_vector_search_with_config, VectorSearchRequest};
use crate::core::table::{Table, VectorSearchParams};

/// ExecutionPlan node that performs an HNSW Vector Search across a specific partition of segments.
/// Returns a stream of RecordBatches that include a `distance` column.
#[derive(Debug)]
pub struct VectorScanExec {
    pub table: Arc<Table>,
    pub partitions: Vec<Vec<ManifestEntry>>,
    pub projection: Option<Vec<usize>>,
    pub filter: Option<String>,
    pub vector_params: Vec<VectorSearchParams>,
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
        vector_params: Vec<VectorSearchParams>,
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
        fields.push(datafusion::arrow::datatypes::Field::new(
            "distance",
            datafusion::arrow::datatypes::DataType::Float32,
            false,
        ));
        let scan_schema = Arc::new(datafusion::arrow::datatypes::Schema::new(fields));

        let properties = PlanProperties::new(
            EquivalenceProperties::new(scan_schema.clone()),
            Partitioning::UnknownPartitioning(partitions.len().max(1)),
            EmissionType::Incremental,
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
                let cols: Vec<&str> = self
                    .vector_params
                    .iter()
                    .map(|vp| vp.column.as_str())
                    .collect();
                write!(
                    f,
                    "VectorScanExec: columns={:?}, partitions={}",
                    cols,
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
        let allowed_seg_ids: std::collections::HashSet<String> = self
            .partitions
            .iter()
            .flatten()
            .map(|e| crate::core::segment::segment_id_from_path(&e.file_path).to_string())
            .collect();

        let base_schema = self.base_schema.clone();
        let column_names = self.projection.as_ref().map(|proj| {
            proj.iter()
                .filter_map(|&i| base_schema.fields().get(i).map(|f| f.name().clone()))
                .collect::<Vec<_>>()
        });

        let col_names_owned = column_names;
        let expected_schema = self.schema.clone();
        let expected_schema_inner = expected_schema.clone();

        let stream = async_stream::stream! {
            let filter_expr = if let Some(ref f) = filter {
                match FilterExpr::parse_sql(f, base_schema.clone()).await {
                    Ok(fe) => Some(fe),
                    Err(e) => {
                        tracing::debug!(error = %e, filter = %f, "VectorScanExec::execute parse_sql fallback");
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

            // Partition-scoped global cluster routing across surviving segments.
            let mut resolved_params = vector_params.clone();
            let mut routing_filters: Vec<(
                std::collections::HashSet<String>,
                std::collections::HashSet<String>,
            )> = Vec::new();
            for vp in &mut resolved_params {
                if vp.targeted_clusters.is_none() {
                    if let crate::core::index::VectorValue::Float32(ref q_f32) = vp.query {
                        if let Ok(Some(router)) = table.ensure_vector_router(&vp.column).await {
                            let n_lists = router.len();
                            let default_p = if n_lists == 0 {
                                16
                            } else {
                                (((n_lists as f64).sqrt() as usize) * 5)
                                    .max(16)
                                    .min(n_lists)
                            };
                            let probes = vp.probes.unwrap_or(default_p);
                            let plan = router.route_allowed(q_f32, probes, Some(&allowed_seg_ids));
                            if !plan.candidate_segments.is_empty() {
                                let routed: std::collections::HashSet<String> = router
                                    .segment_indices
                                    .iter()
                                    .map(|(seg, _)| seg.clone())
                                    .filter(|seg| allowed_seg_ids.contains(seg))
                                    .collect();
                                routing_filters.push((routed, plan.candidate_segments));
                                vp.targeted_clusters = Some(Arc::new(plan.segment_clusters));
                            }
                        }
                    }
                }
            }

            let surviving_entries: Vec<ManifestEntry> = if routing_filters.is_empty() {
                entries
            } else {
                entries
                    .into_iter()
                    .filter(|e| {
                        let seg_id = crate::core::segment::segment_id_from_path(&e.file_path);
                        routing_filters.iter().any(|(routed, cands)| {
                            !routed.contains(seg_id) || cands.contains(seg_id)
                        })
                    })
                    .collect()
            };

            for entry in surviving_entries {
                let mut requests = Vec::new();
                for vp in &resolved_params {
                    let mut request = VectorSearchRequest::new(
                        vp.column.clone(),
                        vp.query.clone(),
                        vp.k,
                        vp.metric,
                        vp.use_mmap,
                    )
                    .with_filter(filter_expr.clone())
                    .with_config(table.query_config().clone())
                    .with_ef_search(vp.ef_search)
                    .with_probes(vp.probes)
                    .with_refine_factor(vp.refine_factor)
                    .with_targeted_clusters(vp.targeted_clusters.clone())
                    .with_iceberg_schema(iceberg_schema.clone());

                    if let Some(ref proj_names) = col_names_owned {
                        request = request.with_columns(Some(proj_names.clone()));
                    }
                    requests.push(request);
                }

                match execute_multi_vector_search_with_config(
                    vec![entry.clone()],
                    table.object_store(),
                    table.data_store.clone(),
                    table.table_uri(),
                    requests,
                ).await {
                    Ok(batches) => {
                        for (_segment_id, batch) in batches {
                            let mut b = batch;
                            if b.num_columns() == expected_schema_inner.fields().len() {
                                let mut cols: Vec<Arc<dyn arrow::array::Array>> =
                                    Vec::with_capacity(b.num_columns());
                                for (col, field) in
                                    b.columns().iter().zip(expected_schema_inner.fields().iter())
                                {
                                    if col.data_type() == field.data_type() {
                                        cols.push(col.clone());
                                    } else {
                                        cols.push(
                                            arrow::compute::cast(col.as_ref(), field.data_type())
                                                .unwrap_or_else(|_| col.clone()),
                                        );
                                    }
                                }
                                let mut options = datafusion::arrow::record_batch::RecordBatchOptions::default();
                                options.row_count = Some(b.num_rows());
                                b = datafusion::arrow::record_batch::RecordBatch::try_new_with_options(expected_schema_inner.clone(), cols, &options)
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
