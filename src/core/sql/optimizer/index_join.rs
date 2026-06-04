// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Index join optimizer rule.
//! Rewrites HashJoinExec nodes with HyperStreamExec on the right side
//! into HyperStreamIndexJoinExec for point-lookup optimization.

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::config::ConfigOptions;
use datafusion::error::Result;
use datafusion::logical_expr::JoinType;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_plan::execution_plan::ExecutionPlan;
use datafusion::physical_plan::joins::HashJoinExec;

use crate::core::sql::physical_plan::index_join::HyperStreamIndexJoinExec;
use crate::core::sql::physical_plan::HyperStreamExec;

#[derive(Debug, Default)]
pub struct IndexJoinOptimizerRule {}

impl PhysicalOptimizerRule for IndexJoinOptimizerRule {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|plan| {
            // Check if plan is HashJoinExec
            if let Some(hash_join) = plan.as_any().downcast_ref::<HashJoinExec>() {
                if hash_join.join_type() != &JoinType::Inner {
                    return Ok(Transformed::no(plan));
                }

                // Check right side
                // We simply unwrap Arc to check concrete type
                // In real world, might handle Filter/Project wrapping the scan.
                // For MVP, assume direct scan or wrapped in simple nodes?
                // Lets check direct scan compatibility first.

                let right = hash_join.right();
                if let Some(hs_exec) = right.as_any().downcast_ref::<HyperStreamExec>() {
                    // It is HyperStream Scan!

                    // Check logic: Join On keys
                    let on = hash_join.on();
                    if on.len() != 1 {
                        // MVP: Single column join
                        return Ok(Transformed::no(plan));
                    }

                    let (left_col_ast, right_col_ast) = &on[0];
                    // left_col_ast is PhysicalExpr (Column). right_col_ast is PhysicalExpr (Column).

                    // We need to verify right_col_ast refers to an indexed column in hs_exec.
                    if let Some(r_col) = right_col_ast.as_any().downcast_ref::<Column>() {
                        // We have column name/index.
                        let right_col_name = r_col.name();

                        // Check if indexed?
                        // hs_exec.table has index info in manifest.
                        // Ideally we check `hs_exec.table.indexes`.
                        // But `table.rs` encapsulates it.
                        // We can blindly assume if we are here, we trust the user optimization?
                        // Or we should verify index exists to actually get perf benefit.
                        // For MVP, we will ALWAYS convert if it's HyperStreamExec,
                        // relying on HyperStream to just scan if no index (our implementation supports that via prune_entries -> fallback).
                        // Wait, `read_filter_async` prunes entires.
                        // If no index, it prunes using min/max stats only.
                        // If values are scattered, min/max overlap implies scanning everything.
                        // So correct "Index Join" requires checking if "Point Lookup" is efficient.
                        // But correctness is preserved!
                        // So rewriting is safe.

                        // Construct Custom Node
                        let new_node = Arc::new(HyperStreamIndexJoinExec::new(
                            hash_join.left().clone(),
                            hs_exec.table.clone(), // Access internal table (needs to be pub or accessor)
                            left_col_ast.clone(),
                            right_col_name.to_string(),
                            hash_join.schema(),
                        ));

                        return Ok(Transformed::yes(new_node));
                    }
                }
            }
            Ok(Transformed::no(plan))
        })
        .map(|t| t.data)
    }

    fn name(&self) -> &str {
        "IndexJoinOptimizerRule"
    }

    fn schema_check(&self) -> bool {
        true
    }
}
