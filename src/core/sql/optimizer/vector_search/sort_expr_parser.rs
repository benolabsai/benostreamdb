// Copyright (c) 2026 Richard Albright. All rights reserved.

//! Sort expression parsing for vector search optimization.
//! Extracts distance UDFs from sort expressions, detects metrics,
//! and supports multi-vector queries.

use datafusion::physical_expr::expressions::{Column, BinaryExpr, Literal};
use datafusion::physical_expr::ScalarFunctionExpr;
use datafusion::logical_expr::Operator;
use datafusion::scalar::ScalarValue;
use datafusion::physical_expr::PhysicalExpr;

use crate::core::index::{VectorMetric, VectorValue};

/// Parsed vector search information extracted from a sort expression.
#[derive(Debug, Clone)]
pub struct ParsedVectorSearch {
    /// Original index in the sort expressions list
    #[allow(dead_code)]
    pub sort_index: usize,
    /// Detected distance metric
    pub metric: VectorMetric,
    /// Name of the vector column
    pub column: String,
    /// Query vector value
    pub query_value: VectorValue,
}

/// Parse sort expressions to extract vector search information.
///
/// Iterates over all sort expressions, looking for distance UDFs or operators.
/// Returns a list of parsed vector searches (primary is first, rest are tiebreakers).
/// Adapted from Apache Iceberg Rust predicate pushdown (v0.9.0+)
pub fn parse_vector_search_exprs(
    sort_exprs: &[(std::sync::Arc<dyn PhysicalExpr>, bool)],
) -> Vec<ParsedVectorSearch> {
    let mut vector_searches = Vec::new();

    for (idx, sort_expr_wrapper) in sort_exprs.iter().enumerate() {
        let sort_expr = sort_expr_wrapper.0.as_ref();

        // Check for Distance UDF or Operator
        let result = parse_single_sort_expr(sort_expr);
        if let Some((metric, col_name, query_val)) = result {
            vector_searches.push(ParsedVectorSearch {
                sort_index: idx,
                metric,
                column: col_name,
                query_value: query_val,
            });
        }
    }

    vector_searches
}

/// Parse a single sort expression to detect vector distance functions or operators.
///
/// Returns `(metric, column_name, query_vector)` if this is a vector distance expression,
/// or `None` if it's a regular sort expression.
fn parse_single_sort_expr(sort_expr: &dyn PhysicalExpr) -> Option<(VectorMetric, String, VectorValue)> {
    // Case 1: ScalarFunctionExpr (e.g., dist_l2, dist_cosine, dist_ip, etc.)
    if let Some(udf) = sort_expr.as_any().downcast_ref::<ScalarFunctionExpr>() {
        return parse_udf_expr(udf);
    }

    // Case 2: BinaryExpr with distance operator (e.g., <->, <=>, <#>)
    if let Some(bin) = sort_expr.as_any().downcast_ref::<BinaryExpr>() {
        return parse_binary_expr(bin);
    }

    None
}

/// Parse a ScalarFunctionExpr to detect distance UDFs.
fn parse_udf_expr(udf: &ScalarFunctionExpr) -> Option<(VectorMetric, String, VectorValue)> {
    let name = udf.name();
    let metric = match name {
        "dist_l2" => Some(VectorMetric::L2),
        "dist_cosine" => Some(VectorMetric::Cosine),
        "dist_ip" => Some(VectorMetric::InnerProduct),
        "dist_l1" => Some(VectorMetric::L1),
        "dist_hamming" => Some(VectorMetric::Hamming),
        "dist_jaccard" => Some(VectorMetric::Jaccard),
        _ => None,
    };

    let Some(m) = metric else {
        return None;
    };

    let args = udf.args();
    if args.len() != 2 {
        return None;
    }

    let col = args[0].as_any().downcast_ref::<Column>()?;
    let scalar_expr = args[1].as_any().downcast_ref::<Literal>()?;

    // Extract vector from FixedSizeList
    if let ScalarValue::FixedSizeList(vec_arr) = scalar_expr.value() {
        let f32_arr = vec_arr.values().as_any().downcast_ref::<arrow::array::Float32Array>()?;
        return Some((m, col.name().to_string(), VectorValue::Float32(f32_arr.values().to_vec())));
    }

    None
}

/// Parse a BinaryExpr to detect distance operators.
fn parse_binary_expr(bin: &BinaryExpr) -> Option<(VectorMetric, String, VectorValue)> {
    let op = bin.op();
    let metric = match op {
        Operator::BitwiseXor => Some(VectorMetric::L2),
        _ => {
            let op_str = format!("{}", op);
            match op_str.as_str() {
                "<->" => Some(VectorMetric::L2),
                "<=>" => Some(VectorMetric::Cosine),
                "<#>" => Some(VectorMetric::InnerProduct),
                "<+>" => Some(VectorMetric::L1),
                "<~>" => Some(VectorMetric::Hamming),
                "<%>" => Some(VectorMetric::Jaccard),
                _ => None,
            }
        }
    };

    let Some(m) = metric else {
        return None;
    };

    let col = bin.left().as_any().downcast_ref::<Column>()?;
    let literal = bin.right().as_any().downcast_ref::<Literal>()?;

    // Case 1: Dense Float32 vector
    if let ScalarValue::FixedSizeList(vec_arr) = literal.value() {
        let f32_arr = vec_arr.values().as_any().downcast_ref::<arrow::array::Float32Array>()?;
        return Some((m, col.name().to_string(), VectorValue::Float32(f32_arr.values().to_vec())));
    }

    // Case 2: Binary (Packed) vector
    if let ScalarValue::FixedSizeBinary(_, Some(bytes)) = literal.value() {
        return Some((m, col.name().to_string(), VectorValue::Binary(bytes.clone())));
    }

    // Case 3: Sparse (Represented as Map or specialized Struct in future)
    None
}
