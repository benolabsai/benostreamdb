// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Pattern detection for vector search optimization.
//! Detects `Limit -> Sort -> Filter -> BenoStreamExec` patterns
//! in the physical plan tree.

use std::sync::Arc;

use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::execution_plan::ExecutionPlan;
use datafusion::physical_plan::filter::FilterExec;
use datafusion::physical_plan::limit::GlobalLimitExec;
use datafusion::physical_plan::sorts::sort::SortExec;

use crate::core::manifest::ManifestEntry;
use crate::core::sql::physical_plan::BenoStreamExec;

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
    /// Optional filter predicate found between Sort and BenoStreamExec
    pub filter: Option<std::sync::Arc<dyn datafusion::physical_expr::PhysicalExpr>>,
    /// The BenoStreamExec node at the base of the pattern
    pub benostream_exec: BenoStreamExecRef,
}

/// A reference wrapper to hold the BenoStreamExec without consuming the Arc.
/// Used to pass the detected node to the plan rewriter.
#[derive(Debug, Clone)]
pub struct BenoStreamExecRef {
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

impl BenoStreamExecRef {
    fn from_exec(hs: &BenoStreamExec) -> Self {
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
/// Looks for: `GlobalLimitExec -> SortExec -> FilterExec? -> BenoStreamExec`
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

    // Step 3: Drill down through optional FilterExec to find BenoStreamExec
    let mut current = sort_exec.input().clone();
    let mut filter = None;

    while let Some(filter_child) = current.as_any().downcast_ref::<FilterExec>() {
        filter = Some(filter_child.predicate().clone());
        current = filter_child.input().clone();
    }

    // Step 4: Check for BenoStreamExec
    let hs_exec = current.as_any().downcast_ref::<BenoStreamExec>()?;

    Some(DetectedKnnPattern {
        limit,
        offset,
        sort_exprs,
        filter,
        benostream_exec: BenoStreamExecRef::from_exec(hs_exec),
    })
}
