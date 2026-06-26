// Copyright (c) 2026 Richard Albright. All rights reserved.
// Portions Copyright The Apache Software Foundation.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Vector search optimizer rule.
//! Detects KNN patterns (Limit -> Sort -> Filter -> HyperStreamExec) and rewrites
//! them to use vector index search via VectorScanExec and VectorMergeExec.

mod plan_detection;
mod plan_rewriter;
mod sort_expr_parser;

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::config::ConfigOptions;
use datafusion::error::Result;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::execution_plan::ExecutionPlan;

use plan_detection::detect_knn_pattern;
use plan_rewriter::build_optimized_plan;
use sort_expr_parser::parse_vector_search_exprs;

/// Physical optimizer rule that detects vector search KNN patterns and rewrites
/// the plan to use vector index search.
///
/// Detects patterns of the form:
/// `GlobalLimitExec -> SortExec -> FilterExec? -> HyperStreamExec`
///
/// When the SortExec contains a vector distance expression (UDF or operator),
/// the plan is rewritten to use `VectorScanExec` and `VectorMergeExec`
/// nodes that leverage HNSW/IVF indexes when available.
#[derive(Debug, Default)]
pub struct VectorSearchOptimizerRule {}

impl PhysicalOptimizerRule for VectorSearchOptimizerRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_down(|plan| {
            // Step 1: Detect KNN pattern
            let pattern = match detect_knn_pattern(plan.as_ref()) {
                Some(p) => p,
                None => return Ok(Transformed::no(plan)),
            };

            // Step 2: Parse sort expressions to find vector distance functions
            let vector_searches = parse_vector_search_exprs(&pattern.sort_exprs);

            // Only proceed if we found at least one vector search expression
            if vector_searches.is_empty() {
                return Ok(Transformed::no(plan));
            }

            // Use the first vector search for index optimization (primary ranking)
            // Rest are applied as tiebreakers after index results
            // Adapted from Apache Iceberg Rust LIMIT pushdown (v0.9.0+)
            let primary_search = &vector_searches[0];

            // Step 3: Build optimized plan
            let new_plan = build_optimized_plan(&pattern, primary_search, config)?;

            Ok(Transformed::yes(new_plan))
        })
        .map(|t| t.data)
    }

    fn name(&self) -> &str {
        "VectorSearchOptimizerRule"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
