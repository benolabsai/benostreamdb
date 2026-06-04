// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Pattern detection for vector search optimization.
//! Detects `Limit -> Sort -> Filter -> HyperStreamExec` patterns
//! in the physical plan tree.

use std::sync::Arc;

use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::execution_plan::ExecutionPlan;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::physical_plan::sorts::sort::SortExec;

use crate::core::manifest::ManifestEntry;
use crate::core::sql::physical_plan::HyperStreamExec;

/// Detected KNN pattern from the plan tree.
#[derive(Debug)]
pub struct DetectedKnnPattern {
    /// The LIMIT value (number of results to return)
    pub limit: usize,
    /// The OFFSET value (number of results to skip)
    pub offset: usize,
    /// Sort expressions from the SortExec node
    pub sort_exprs: Vec<(
        std::sync::Arc<dyn datafusion::physical_expr::PhysicalExpr>,
        bool,
    )>,
    /// Optional filter predicate found between Sort and HyperStreamExec
    pub filter: Option<std::sync::Arc<dyn datafusion::physical_expr::PhysicalExpr>>,
    /// The HyperStreamExec node at the base of the pattern
    pub hyperstream_exec: HyperStreamExecRef,
}

/// A reference wrapper to hold the HyperStreamExec without consuming the Arc.
/// Used to pass the detected node to the plan rewriter.
#[derive(Debug, Clone)]
pub struct HyperStreamExecRef {
    /// Cloned reference to the table
    pub table: std::sync::Arc<crate::core::table::Table>,
    /// Cloned partitions
    pub partitions: Vec<Vec<ManifestEntry>>,
    /// Cloned projection
    pub projection: Option<Vec<usize>>,
    /// Cloned filter string
    pub filter_str: Option<String>,
    /// Cloned schema
    pub schema: arrow::datatypes::SchemaRef,
}

impl HyperStreamExecRef {
    fn from_exec(hs: &HyperStreamExec) -> Self {
        Self {
            table: hs.table.clone(),
            partitions: hs.partitions.clone(),
            projection: hs.projection().cloned(),
            filter_str: hs.filter_str().map(|s| s.to_string()),
            schema: hs.schema().clone(),
        }
    }
}

/// Try to detect a KNN pattern in the plan tree.
///
/// Looks for: `GlobalLimitExec -> SortExec -> FilterExec? -> HyperStreamExec`
///
/// Returns `Some(DetectedKnnPattern)` if the pattern is found, `None` otherwise.
pub fn detect_knn_pattern(plan: &dyn ExecutionPlan) -> Option<DetectedKnnPattern> {
    // Step 1: Check for GlobalLimitExec
    let limit_exec = plan.as_any().downcast_ref::<GlobalLimitExec>()?;
    let limit = limit_exec.fetch()?;
    let offset = limit_exec.skip();

    // Step 2: Check child is SortExec
    let sort_exec = limit_exec.input().as_any().downcast_ref::<SortExec>()?;
    let sort_exprs: Vec<(Arc<dyn PhysicalExpr>, bool)> = sort_exec
        .expr()
        .iter()
        .map(|se| (se.expr.clone(), !se.options.descending))
        .collect();

    if sort_exprs.is_empty() {
        return None;
    }

    // Step 3: Drill down through optional FilterExec to find HyperStreamExec
    let mut current = sort_exec.input().clone();
    let mut filter = None;

    while let Some(filter_child) = current.as_any().downcast_ref::<FilterExec>() {
        filter = Some(filter_child.predicate().clone());
        current = filter_child.input().clone();
    }

    // Step 4: Check for HyperStreamExec
    let hs_exec = current.as_any().downcast_ref::<HyperStreamExec>()?;

    Some(DetectedKnnPattern {
        limit,
        offset,
        sort_exprs,
        filter,
        hyperstream_exec: HyperStreamExecRef::from_exec(hs_exec),
    })
}
