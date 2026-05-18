// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Plan node construction for vector search optimization.
//! Builds VectorScanExec and VectorMergeExec nodes from detected patterns.

use std::sync::Arc;

use datafusion::config::ConfigOptions;
use datafusion::error::Result;
use datafusion::physical_plan::execution_plan::ExecutionPlan;
use datafusion::physical_plan::filter::FilterExec;

use crate::core::sql::physical_plan::vector_scan::VectorScanExec;
use crate::core::sql::physical_plan::vector_merge::VectorMergeExec;
use crate::core::table::VectorSearchParams;

use crate::core::sql::optimizer::config::VectorSearchConfig;
use super::plan_detection::DetectedKnnPattern;
use super::sort_expr_parser::ParsedVectorSearch;

/// Build an optimized vector search plan from a detected KNN pattern.
///
/// This function:
/// 1. Creates VectorSearchParams from the primary vector search expression
/// 2. Applies configuration from the session config
/// 3. Constructs VectorScanExec and VectorMergeExec nodes
/// 4. Re-wraps with FilterExec if a filter was present
pub fn build_optimized_plan(
    pattern: &DetectedKnnPattern,
    primary_search: &ParsedVectorSearch,
    config: &ConfigOptions,
) -> Result<Arc<dyn ExecutionPlan>> {
    // Read configuration from session config
    let search_config = VectorSearchConfig::from_session_config(config);

    // LIMIT PUSHDOWN: Push the LIMIT+OFFSET to the file scanning layer (Iceberg v0.9.0+)
    // This reduces the amount of data fetched and processed
    let k_with_offset = pattern.limit + pattern.offset;

    tracing::info!(
        "VectorSearchOptimizer: Detected KNN pattern for column '{}' with k={}, offset={}, metric={:?}",
        primary_search.column, pattern.limit, pattern.offset, primary_search.metric
    );

    if search_config.limit_pushdown {
        tracing::debug!("VectorSearchOptimizer: Pushing LIMIT+OFFSET {} to vector index layer", k_with_offset);
    }

    let mut vp = VectorSearchParams::new(
        &primary_search.column,
        primary_search.query_value.clone(),
        k_with_offset,
    ).with_metric(primary_search.metric);

    // Apply configuration parameters
    if let Some(ef) = search_config.ef_search {
        vp = vp.with_ef_search(ef);
        tracing::debug!("VectorSearchOptimizer: Using ef_search={}", ef);
    }
    if let Some(probes) = search_config.probes {
        vp = vp.with_probes(probes);
        tracing::debug!("VectorSearchOptimizer: Using probes={}", probes);
    }

    // FAST PATH OPTIMIZATION: For small result sets (limit < 100), use single-threaded execution (Iceberg v0.9.0+)
    if search_config.fast_path && pattern.limit < 100 {
        tracing::debug!("VectorSearchOptimizer: Using fast path for small result set (limit={})", pattern.limit);
    }

    // ROW GROUP SKIPPING: Statistics-based row group skipping (Iceberg v0.4.0+)
    if search_config.skip_row_groups {
        tracing::debug!("VectorSearchOptimizer: Row group skipping enabled - will skip groups outside predicate range");
    }

    // Construct optimized scan
    let scan_exec = VectorScanExec::new(
        pattern.hyperstream_exec.table.clone(),
        pattern.hyperstream_exec.partitions.clone(),
        pattern.hyperstream_exec.projection.clone(),
        pattern.hyperstream_exec.filter_str.clone(),
        vp,
        Some(k_with_offset),
        pattern.hyperstream_exec.schema.clone(),
    )?;

    let merge_exec = VectorMergeExec::new(
        Arc::new(scan_exec),
        pattern.limit,
        pattern.offset,
        pattern.hyperstream_exec.schema.clone(),
    )?;

    tracing::info!(
        "VectorSearchOptimizer: Created optimized plan with vector search parameters. \
        Index will be used if available, otherwise will fall back to sequential scan."
    );

    // If there was a filter, wrap it
    let mut result: Arc<dyn ExecutionPlan> = Arc::new(merge_exec);
    if let Some(ref f) = pattern.filter {
        result = Arc::new(FilterExec::try_new(f.clone(), result)?);
        tracing::debug!("VectorSearchOptimizer: Added filter predicate to optimized plan");
    }

    Ok(result)
}
