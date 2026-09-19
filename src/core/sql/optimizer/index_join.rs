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
                    if on.is_empty() {
                        return Ok(Transformed::no(plan));
                    }

                    let mut left_on = Vec::new();
                    let mut right_cols = Vec::new();

                    for (left_col_ast, right_col_ast) in on {
                        if let Some(r_col) = right_col_ast.as_any().downcast_ref::<Column>() {
                            left_on.push(left_col_ast.clone());
                            right_cols.push(r_col.name().to_string());
                        } else {
                            // If any right column is not a simple column reference, abort
                            return Ok(Transformed::no(plan));
                        }
                    }

                    // Construct Custom Node
                    let new_node = Arc::new(HyperStreamIndexJoinExec::new(
                        hash_join.left().clone(),
                        hs_exec.table.clone(), // Access internal table
                        left_on,
                        right_cols,
                        hash_join.schema(),
                    ));

                    return Ok(Transformed::yes(new_node));
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
