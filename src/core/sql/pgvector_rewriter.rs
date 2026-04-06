// Copyright (c) 2026 Richard Albright. All rights reserved.

/// Expression rewriter for pgvector-compatible syntax
/// 
/// This module provides a DataFusion expression rewriter that transforms pgvector
/// operators and vector literals into UDF calls:
/// 
/// - Distance operators: `embedding <-> query` → `dist_l2(embedding, query)`
/// - Vector literals: Handled through CAST expressions with custom logic
/// 
/// This uses DataFusion's TreeNodeRewriter for proper expression tree traversal.
use datafusion::common::tree_node::{Transformed, TreeNodeRewriter};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::Expr;
use datafusion::scalar::ScalarValue;
use std::sync::Arc;

use super::vector_literal::VectorLiteralParser;

/// Mapping of pgvector operators to UDF names
const OPERATOR_MAPPINGS: &[(&str, &str)] = &[
    ("<->", "dist_l2"),
    ("<=>", "dist_cosine"),
    ("<#>", "dist_ip"),
    ("<+>", "dist_l1"),
    ("<~>", "dist_hamming"),
    ("<%>", "dist_jaccard"),
];

/// Expression rewriter that converts pgvector syntax to UDF calls
pub struct PgVectorRewriter;

impl TreeNodeRewriter for PgVectorRewriter {
    type Node = Expr;

    fn f_down(&mut self, expr: Expr) -> Result<Transformed<Expr>> {
        match expr {
            // Handle binary expressions with custom operators (e.g. <->)
            Expr::BinaryExpr(binary) => {
                let op_str = binary.op.to_string();
                
                // Check if this is a pgvector operator
                if let Some((_, udf_name)) = OPERATOR_MAPPINGS.iter().find(|(op, _)| op == &op_str) {
                    let mut left = *binary.left;
                    let mut right = *binary.right;
                    
                    // Check if either side is a string literal starting with '['
                    // and convert it automatically to a vector literal
                    for arg in [&mut left, &mut right] {
                        let literal_text = match arg {
                            Expr::Literal(ScalarValue::Utf8(Some(s)), _) => Some(s),
                            Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _) => Some(s),
                            _ => None,
                        };
                        
                        if let Some(s) = literal_text {
                            let trimmed = s.trim();
                            if trimmed.starts_with('[') {
                                if let Ok(parsed_value) = VectorLiteralParser::parse(trimmed) {
                                    *arg = Expr::Literal(parsed_value, None);
                                }
                            }
                        }
                    }
                    
                    // Create the UDF call
                    let udf = get_distance_udf(udf_name)?;
                    let args = vec![left, right];
                    let func = datafusion::logical_expr::expr::ScalarFunction::new_udf(
                        Arc::new(udf),
                        args,
                    );
                    return Ok(Transformed::yes(Expr::ScalarFunction(func)));
                }
                
                Ok(Transformed::no(Expr::BinaryExpr(binary)))
            }
            
            // Handle scalar function calls (e.g., dist_l2)
            Expr::ScalarFunction(mut func) => {
                let name = func.name();
                if matches!(name, "dist_l2" | "dist_cosine" | "dist_ip" | "dist_l1" | "dist_hamming" | "dist_jaccard") {
                    if func.args.len() == 2 {
                        // Support both regular and large string literals for the second argument
                        let literal_text = match &func.args[1] {
                            Expr::Literal(ScalarValue::Utf8(Some(s)), _) => Some(s.as_str()),
                            Expr::Literal(ScalarValue::LargeUtf8(Some(s)), _) => Some(s.as_str()),
                            _ => None,
                        };

                        if let Some(s) = literal_text {
                            let trimmed = s.trim();
                            if trimmed.starts_with('[') {
                                match VectorLiteralParser::parse(trimmed) {
                                    Ok(parsed_value) => {
                                        func.args[1] = Expr::Literal(parsed_value, None);
                                        return Ok(Transformed::yes(Expr::ScalarFunction(func)));
                                    }
                                    Err(e) => {
                                        return Err(datafusion::error::DataFusionError::Plan(
                                            format!("Failed to parse vector literal: {}. Input starts with: {}", e, &trimmed[..trimmed.len().min(50)])
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
                Ok(Transformed::no(Expr::ScalarFunction(func)))
            }
            
            // Recurse into aliases to ensure nested distance calls are rewritten
            Expr::Alias(mut alias) => {
                use datafusion::common::tree_node::TreeNode;
                // Move the expression out of the box to rewrite it
                let expr = std::mem::replace(&mut alias.expr, Box::new(Expr::Literal(ScalarValue::Null, None)));
                let transformed = expr.rewrite(self)?;
                
                alias.expr = Box::new(transformed.data);
                if transformed.transformed {
                    return Ok(Transformed::yes(Expr::Alias(alias)));
                }
                Ok(Transformed::no(Expr::Alias(alias)))
            }
            
            _ => Ok(Transformed::no(expr)),
        }
    }
}

/// Helper function to create a new distance UDF
fn get_distance_udf(name: &str) -> Result<datafusion::logical_expr::ScalarUDF> {
    use super::vector_udf::*;
    
    let udf = match name {
        "dist_l2" => datafusion::logical_expr::ScalarUDF::new_from_impl(L2DistUDF::new()),
        "dist_cosine" => datafusion::logical_expr::ScalarUDF::new_from_impl(CosineDistUDF::new()),
        "dist_ip" => datafusion::logical_expr::ScalarUDF::new_from_impl(IPDistUDF::new()),
        "dist_l1" => datafusion::logical_expr::ScalarUDF::new_from_impl(L1DistUDF::new()),
        "dist_hamming" => datafusion::logical_expr::ScalarUDF::new_from_impl(HammingDistUDF::new()),
        "dist_jaccard" => datafusion::logical_expr::ScalarUDF::new_from_impl(JaccardDistUDF::new()),
        _ => return Err(DataFusionError::Plan(
            format!("Unknown distance UDF: {}", name)
        )),
    };
    
    Ok(udf)
}

/// Rewrite a logical plan to convert pgvector syntax
pub fn rewrite_pgvector_plan(
    plan: datafusion::logical_expr::LogicalPlan,
) -> Result<datafusion::logical_expr::LogicalPlan> {
    use datafusion::common::tree_node::TreeNode;
    
    // Use transform to visit all nodes in the logical plan
    // For each node, we map its expressions through our PgVectorRewriter
    let result = plan.transform(|node| {
        let mut rewriter = PgVectorRewriter;
        node.map_expressions(|expr| {
            expr.rewrite(&mut rewriter)
        })
    })?;
    
    Ok(result.data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::common::tree_node::TreeNode;
    use datafusion::arrow::datatypes::DataType;
    
    #[test]
    fn test_cast_vector_literal() {
        let literal = Expr::Literal(ScalarValue::Utf8(Some("[1,2,3]".to_string())), None);
        let cast_expr = Expr::Cast(datafusion::logical_expr::Cast {
            expr: Box::new(literal),
            data_type: DataType::Utf8, // Placeholder
        });
        
        let mut rewriter = PgVectorRewriter;
        let result = cast_expr.rewrite(&mut rewriter);
        
        assert!(result.is_ok());
    }
}
