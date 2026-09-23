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

//! Sort expression parsing for vector search optimization.
//! Extracts distance UDFs from sort expressions, detects metrics,
//! and supports multi-vector queries.

use arrow::array::Array;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::{BinaryExpr, Column, Literal};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::ScalarFunctionExpr;
use datafusion::scalar::ScalarValue;

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
fn parse_single_sort_expr(
    sort_expr: &dyn PhysicalExpr,
) -> Option<(VectorMetric, String, VectorValue)> {
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
        let f32_arr = vec_arr
            .values()
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()?;
        return Some((
            m,
            col.name().to_string(),
            VectorValue::Float32(f32_arr.values().to_vec()),
        ));
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
        let f32_arr = vec_arr
            .values()
            .as_any()
            .downcast_ref::<arrow::array::Float32Array>()?;
        return Some((
            m,
            col.name().to_string(),
            VectorValue::Float32(f32_arr.values().to_vec()),
        ));
    }

    // Case 2: Binary (Packed) vector
    if let ScalarValue::FixedSizeBinary(_, Some(bytes)) = literal.value() {
        return Some((
            m,
            col.name().to_string(),
            VectorValue::Binary(bytes.clone()),
        ));
    }

    // Case 3: Sparse (Represented as Map or specialized Struct in future)
    if let ScalarValue::Struct(struct_array) = literal.value() {
        if struct_array.num_columns() >= 2 {
            let indices_col = struct_array.column_by_name("indices")?;
            let values_col = struct_array.column_by_name("values")?;

            if let (Some(indices_list), Some(values_list)) = (
                indices_col
                    .as_any()
                    .downcast_ref::<arrow::array::ListArray>(),
                values_col
                    .as_any()
                    .downcast_ref::<arrow::array::ListArray>(),
            ) {
                // Read the first (and only) element since this is a ScalarValue
                if !indices_list.is_empty() && !values_list.is_empty() {
                    let indices_arr = indices_list.value(0);
                    let values_arr = values_list.value(0);

                    if let (Some(indices), Some(values)) = (
                        indices_arr
                            .as_any()
                            .downcast_ref::<arrow::array::UInt32Array>(),
                        values_arr
                            .as_any()
                            .downcast_ref::<arrow::array::Float32Array>(),
                    ) {
                        let mut dim = 0;
                        if struct_array.num_columns() >= 3 {
                            if let Some(dim_col) = struct_array.column_by_name("dim") {
                                if let Some(dim_arr) =
                                    dim_col.as_any().downcast_ref::<arrow::array::UInt32Array>()
                                {
                                    if !dim_arr.is_empty() {
                                        dim = dim_arr.value(0) as usize;
                                    }
                                }
                            }
                        }

                        let sv = crate::core::index::SparseVector {
                            indices: indices.values().to_vec(),
                            values: values.values().to_vec(),
                            dim,
                        };

                        return Some((m, col.name().to_string(), VectorValue::Sparse(sv)));
                    }
                }
            }
        }
    }

    // Case 4: Sparse as a `Map<key, f32>` — e.g. `{'1': 0.5, '10': 0.3}`.
    //
    // A map has no place to carry the vector dimension, so it is inferred as
    // `max(index) + 1`. Use the Struct form (`indices`/`values`/`dim`) when the
    // true dimension matters.
    if let ScalarValue::Map(map_array) = literal.value() {
        if let Some(map) = map_array.as_any().downcast_ref::<arrow::array::MapArray>() {
            if let Some(sv) = sparse_from_map(map) {
                return Some((m, col.name().to_string(), VectorValue::Sparse(sv)));
            }
        }
    }

    None
}

/// Parse a sparse vector from a `Map<key, f32>` array element.
///
/// Keys may be integer or numeric-string typed (SQL map literals such as
/// `{'1': 0.5, '10': 0.3}` produce string keys). A map cannot carry the vector
/// dimension, so it is inferred as `max(index) + 1`; use the Struct form
/// (`indices`/`values`/`dim`) when the true dimension matters.
fn sparse_from_map(map: &arrow::array::MapArray) -> Option<crate::core::index::SparseVector> {
    use arrow::array::{
        Float32Array, Int32Array, Int64Array, StringArray, UInt32Array, UInt64Array,
    };
    use arrow::datatypes::DataType;

    if map.is_empty() {
        return None;
    }

    let keys = map.keys();
    let vals = map.values();
    let mut pairs: Vec<(u32, f32)> = Vec::new();

    for i in 0..keys.len() {
        let idx = match keys.data_type() {
            DataType::UInt32 => keys
                .as_any()
                .downcast_ref::<UInt32Array>()
                .map(|a| a.value(i)),
            DataType::Int32 => keys
                .as_any()
                .downcast_ref::<Int32Array>()
                .map(|a| a.value(i) as u32),
            DataType::Int64 => keys
                .as_any()
                .downcast_ref::<Int64Array>()
                .map(|a| a.value(i) as u32),
            DataType::UInt64 => keys
                .as_any()
                .downcast_ref::<UInt64Array>()
                .map(|a| a.value(i) as u32),
            DataType::Utf8 => keys
                .as_any()
                .downcast_ref::<StringArray>()
                .and_then(|a| a.value(i).parse::<u32>().ok()),
            _ => None,
        };
        let val = vals
            .as_any()
            .downcast_ref::<Float32Array>()
            .map(|a| a.value(i));
        if let (Some(ix), Some(v)) = (idx, val) {
            pairs.push((ix, v));
        }
    }

    if pairs.is_empty() {
        return None;
    }

    // Sparse vectors must be sorted by index.
    pairs.sort_by_key(|(i, _)| *i);
    pairs.dedup_by_key(|(i, _)| *i);
    let dim = pairs.last().map(|(i, _)| *i as usize + 1).unwrap_or(0);
    let (indices, values): (Vec<u32>, Vec<f32>) = pairs.into_iter().unzip();
    Some(crate::core::index::SparseVector {
        indices,
        values,
        dim,
    })
}

#[cfg(test)]
mod map_tests {
    use super::*;
    use arrow::array::{Float32Builder, MapBuilder, StringBuilder};

    fn build_map(entries: &[(&str, f32)]) -> arrow::array::MapArray {
        let mut b = MapBuilder::new(None, StringBuilder::new(), Float32Builder::new());
        for (k, v) in entries {
            b.keys().append_value(k);
            b.values().append_value(*v);
        }
        b.append(true).unwrap();
        b.finish()
    }

    #[test]
    fn parses_string_keyed_map() {
        let map = build_map(&[("10", 0.3), ("1", 0.5)]);
        let sv = sparse_from_map(&map).expect("map should parse");
        // Sorted by index, dimension inferred from the largest index.
        assert_eq!(sv.indices, vec![1, 10]);
        assert_eq!(sv.values, vec![0.5, 0.3]);
        assert_eq!(sv.dim, 11);
    }

    #[test]
    fn empty_map_is_none() {
        let map = build_map(&[]);
        assert!(sparse_from_map(&map).is_none());
    }

    #[test]
    fn non_numeric_keys_are_skipped() {
        let map = build_map(&[("nope", 1.0), ("3", 2.0)]);
        let sv = sparse_from_map(&map).expect("one valid entry");
        assert_eq!(sv.indices, vec![3]);
        assert_eq!(sv.values, vec![2.0]);
        assert_eq!(sv.dim, 4);
    }
}
