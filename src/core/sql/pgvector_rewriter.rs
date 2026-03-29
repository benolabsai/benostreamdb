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
use datafusion::logical_expr::{Expr, Operator};
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
        match &expr {
            // Handle binary expressions with custom operators
            Expr::BinaryExpr(binary) => {
                let op_str = format!("{:?}", binary.op);
                
                // Check if this is a pgvector operator
                for (pg_op, udf_name) in OPERATOR_MAPPINGS {
                    if op_str.contains(pg_op) || matches_operator(&binary.op, pg_op) {
                        // Convert to UDF call
                        let udf = create_distance_udf(udf_name)?;
                        let udf_expr = Expr::ScalarFunction(datafusion::logical_expr::expr::ScalarFunction::new_udf(
                            Arc::new(udf),
                            vec![(*binary.left).clone(), (*binary.right).clone()],
                        ));
                        return Ok(Transformed::yes(udf_expr));
                    }
                }
                
                Ok(Transformed::no(expr))
            }
            
            // Handle CAST expressions for vector literals
            // Pattern: CAST('[1,2,3]' AS vector)
            Expr::Cast(cast) => {
                if let Expr::Literal(scalar_value, _metadata) = &*cast.expr {
                    if let ScalarValue::Utf8(Some(literal_str)) = scalar_value {
                        // Check if casting to a vector-like type
                        let type_str = format!("{:?}", cast.data_type);
                        if type_str.to_lowercase().contains("vector") || 
                           literal_str.starts_with('[') && literal_str.ends_with(']') {
                            // Parse the vector literal
                            match VectorLiteralParser::parse(literal_str) {
                                Ok(parsed_value) => {
                                    return Ok(Transformed::yes(Expr::Literal(parsed_value, None)));
                                }
                                Err(e) => {
                                    return Err(DataFusionError::Plan(
                                        format!("Failed to parse vector literal '{}': {}", literal_str, e)
                                    ));
                                }
                            }
                        }
                    }
                }
                
                Ok(Transformed::no(expr))
            }
            
            _ => Ok(Transformed::no(expr)),
        }
    }
}

/// Check if a DataFusion operator matches a pgvector operator string
fn matches_operator(_op: &Operator, _pg_op: &str) -> bool {
    // DataFusion operators are enums, so we need to match based on the operator type
    // For custom operators, we'd need to extend DataFusion's Operator enum
    // For now, we'll rely on string matching in the debug representation
    false
}

/// Create a distance UDF by name
fn create_distance_udf(name: &str) -> Result<datafusion::logical_expr::ScalarUDF> {
    use crate::core::sql::vector_udf::*;
    
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
    
    let mut rewriter = PgVectorRewriter;
    
    // Transform all expressions in the plan
    let result = plan.map_expressions(|expr| {
        expr.rewrite(&mut rewriter)
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
            data_type: DataType::Utf8, // Placeholder - would be custom vector type
        });
        
        let mut rewriter = PgVectorRewriter;
        let result = cast_expr.rewrite(&mut rewriter);
        
        assert!(result.is_ok());
    }
}
