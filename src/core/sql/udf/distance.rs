// Copyright (c) 2026 Richard Albright. All rights reserved.

use crate::core::index::distance;
use crate::core::index::SparseVector;
#[allow(unused_imports)]
use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, Int16Array, Int32Array,
    Int64Array, Int8Array, ListArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::datatypes::DataType;
use datafusion::common::cast::as_fixed_size_list_array;
use datafusion::error::Result;
use datafusion::logical_expr::{ScalarUDFImpl, Signature, Volatility};
use datafusion::scalar::ScalarValue;
use std::sync::Arc;

/// Convert a dense vector to sparse representation
/// Only stores non-zero elements
pub fn dense_to_sparse(dense: &[f32]) -> SparseVector {
    let mut indices = Vec::new();
    let mut values = Vec::new();

    for (i, &val) in dense.iter().enumerate() {
        if val != 0.0 {
            indices.push(i as u32);
            values.push(val);
        }
    }

    SparseVector {
        indices,
        values,
        dim: dense.len(),
    }
}

/// Convert a sparse vector to dense representation
/// Expands to full array with zeros
pub fn sparse_to_dense(sparse: &SparseVector) -> Vec<f32> {
    let mut dense = vec![0.0; sparse.dim];

    for (idx, val) in sparse.indices.iter().zip(sparse.values.iter()) {
        dense[*idx as usize] = *val;
    }

    dense
}

/// Compute L2 distance for sparse vectors
/// Uses sparse-aware algorithm for efficiency
pub fn sparse_l2_distance(a: &SparseVector, b: &SparseVector) -> f32 {
    assert_eq!(a.dim, b.dim, "Sparse vectors must have same dimension");
    distance::sparse_l2_distance_squared(&a.indices, &a.values, &b.indices, &b.values).sqrt()
}

/// Compute cosine distance for sparse vectors
/// Uses sparse dot product for efficiency
pub fn sparse_cosine_distance(a: &SparseVector, b: &SparseVector) -> f32 {
    assert_eq!(a.dim, b.dim, "Sparse vectors must have same dimension");

    let dot = distance::sparse_dot_product(&a.indices, &a.values, &b.indices, &b.values);

    // Compute norms
    let norm_a = a.values.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b = b.values.iter().map(|v| v * v).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 1.0; // Maximum distance for zero vectors
    }

    1.0 - (dot / (norm_a * norm_b))
}

/// Compute inner product distance for sparse vectors
/// Uses sparse dot product for efficiency
pub fn sparse_inner_product_distance(a: &SparseVector, b: &SparseVector) -> f32 {
    assert_eq!(a.dim, b.dim, "Sparse vectors must have same dimension");
    -distance::sparse_dot_product(&a.indices, &a.values, &b.indices, &b.values)
}

/// Helper macro to implement DynEq and DynHash for UDF structs
macro_rules! impl_dyn_traits {
    ($name:ident) => {
        impl PartialEq for $name {
            fn eq(&self, _other: &Self) -> bool {
                // All instances of the same UDF type are considered equal
                true
            }
        }

        impl Eq for $name {}

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                // Hash the type name
                std::any::type_name::<Self>().hash(state);
            }
        }
    };
}

/// Helper function to compute distance with GPU acceleration when available
///
/// This function checks for a global GPU context and routes computation through
/// GPU kernels when available, falling back to CPU computation otherwise.
///
/// # Arguments
/// * `v1` - First vector
/// * `v2` - Second vector
/// * `metric` - Distance metric to use
/// * `cpu_fn` - CPU fallback function
///
/// # Returns
/// The computed distance as f32
fn compute_distance_with_gpu<F>(
    v1: &[f32],
    v2: &[f32],
    metric: crate::core::index::VectorMetric,
    cpu_fn: F,
) -> f32
where
    F: Fn(&[f32], &[f32]) -> f32,
{
    use crate::core::index::gpu;

    // Check if GPU context is available
    if let Some(_ctx) = gpu::get_global_gpu_context() {
        // Prepare vectors for batch computation (single pair)
        let dim = v1.len();

        // Call GPU compute_distance with v2 as a "batch" of 1 vector
        match gpu::compute_distance(v1, v2, dim, metric) {
            Ok(distances) => {
                // Extract the single distance result
                if let Some(&dist) = distances.first() {
                    return dist;
                }
                // If GPU computation returned empty result, fall back to CPU
            }
            Err(_) => {
                // GPU computation failed, fall back to CPU
            }
        }
    }

    // Fall back to CPU computation
    cpu_fn(v1, v2)
}

/// Generic Macro to define vector distance UDFs
macro_rules! make_vector_dist_udf {
    ($name:ident, $func_name:expr, $dist_fn:ident, $metric:expr) => {
        #[derive(Debug)]
        pub struct $name {
            signature: Signature,
        }

        impl_dyn_traits!($name);

        impl $name {
            pub fn new() -> Self {
                Self {
                    signature: Signature::any(2, Volatility::Immutable),
                }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl ScalarUDFImpl for $name {
            fn as_any(&self) -> &dyn std::any::Any {
                self
            }

            fn name(&self) -> &str {
                $func_name
            }

            fn signature(&self) -> &Signature {
                &self.signature
            }

            fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
                Ok(DataType::Float32)
            }

            fn invoke_with_args(
                &self,
                args: datafusion::logical_expr::ScalarFunctionArgs,
            ) -> Result<datafusion::logical_expr::ColumnarValue> {
                use arrow::array::{Float16Array, UInt8Array};
                use datafusion::logical_expr::ColumnarValue;

                let (lhs, rhs) = (&args.args[0], &args.args[1]);

                // Helper function to extract Float32 values from either List or FixedSizeList
                let extract_f32_vec = |arr: &ArrayRef, idx: usize| -> Result<Vec<f32>> {
                    match arr.data_type() {
                        DataType::FixedSizeList(_, _) => {
                            let list_arr = as_fixed_size_list_array(arr)?;
                            let value_arr = list_arr.value(idx);

                            if let Some(f32_arr) = value_arr.as_any().downcast_ref::<Float32Array>()
                            {
                                Ok(f32_arr.values().to_vec())
                            } else if let Some(f16_arr) =
                                value_arr.as_any().downcast_ref::<Float16Array>()
                            {
                                Ok(f16_arr
                                    .iter()
                                    .map(|f| f.unwrap_or_default().to_f32())
                                    .collect())
                            } else if let Some(u8_arr) =
                                value_arr.as_any().downcast_ref::<UInt8Array>()
                            {
                                // Binary vector case
                                Ok(u8_arr.values().iter().map(|&b| b as f32).collect())
                            } else {
                                Err(datafusion::error::DataFusionError::Execution(
                                    "Unsupported inner array type".to_string(),
                                ))
                            }
                        }
                        DataType::List(_) => {
                            let list_arr =
                                arr.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                                    datafusion::error::DataFusionError::Execution(
                                        "Failed to downcast to ListArray".to_string(),
                                    )
                                })?;
                            let value_arr = list_arr.value(idx);

                            if let Some(f32_arr) = value_arr.as_any().downcast_ref::<Float32Array>()
                            {
                                Ok(f32_arr.values().to_vec())
                            } else if let Some(f16_arr) =
                                value_arr.as_any().downcast_ref::<Float16Array>()
                            {
                                Ok(f16_arr
                                    .iter()
                                    .map(|f| f.unwrap_or_default().to_f32())
                                    .collect())
                            } else if let Some(u8_arr) =
                                value_arr.as_any().downcast_ref::<UInt8Array>()
                            {
                                // Binary vector case
                                Ok(u8_arr.values().iter().map(|&b| b as f32).collect())
                            } else {
                                Err(datafusion::error::DataFusionError::Execution(
                                    "Unsupported inner array type in List".to_string(),
                                ))
                            }
                        }
                        _ => Err(datafusion::error::DataFusionError::Execution(format!(
                            "Expected List or FixedSizeList, got {:?}",
                            arr.data_type()
                        ))),
                    }
                };

                // Helper to extract vector from scalar value
                let extract_scalar_vec = |scalar: &ScalarValue| -> Result<Vec<f32>> {
                    match scalar {
                        ScalarValue::FixedSizeList(fsl_arc) => {
                            let fsl_array = fsl_arc
                                .as_any()
                                .downcast_ref::<FixedSizeListArray>()
                                .ok_or_else(|| {
                                datafusion::error::DataFusionError::Execution(
                                    "Failed to downcast FixedSizeList scalar".to_string(),
                                )
                            })?;

                            if fsl_array.len() == 0 {
                                return Err(datafusion::error::DataFusionError::Execution(
                                    "Empty FixedSizeList scalar".to_string(),
                                ));
                            }

                            let inner_array = fsl_array.value(0);
                            if let Some(f32_arr) =
                                inner_array.as_any().downcast_ref::<Float32Array>()
                            {
                                Ok(f32_arr.values().to_vec())
                            } else {
                                Err(datafusion::error::DataFusionError::Execution(format!(
                                    "Unsupported scalar FixedSizeList inner array type: {:?}",
                                    inner_array.data_type()
                                )))
                            }
                        }
                        ScalarValue::List(list_arc) => {
                            // ScalarValue::List wraps a ListArray
                            let list_array = list_arc.as_ref();
                            if list_array.len() == 0 {
                                return Err(datafusion::error::DataFusionError::Execution(
                                    "Empty List scalar".to_string(),
                                ));
                            }

                            // Get the first (and only) element
                            let inner_array = list_array.value(0);

                            if let Some(f32_arr) =
                                inner_array.as_any().downcast_ref::<Float32Array>()
                            {
                                Ok(f32_arr.values().to_vec())
                            } else if let Some(f64_arr) = inner_array
                                .as_any()
                                .downcast_ref::<arrow::array::Float64Array>(
                            ) {
                                Ok(f64_arr.values().iter().map(|&x| x as f32).collect())
                            } else if let Some(f16_arr) =
                                inner_array.as_any().downcast_ref::<Float16Array>()
                            {
                                Ok(f16_arr
                                    .iter()
                                    .map(|f| f.unwrap_or_default().to_f32())
                                    .collect())
                            } else if let Some(i8_arr) =
                                inner_array.as_any().downcast_ref::<Int8Array>()
                            {
                                Ok(i8_arr.values().iter().map(|&x| x as f32).collect())
                            } else if let Some(i16_arr) =
                                inner_array.as_any().downcast_ref::<Int16Array>()
                            {
                                Ok(i16_arr.values().iter().map(|&x| x as f32).collect())
                            } else if let Some(i32_arr) =
                                inner_array.as_any().downcast_ref::<Int32Array>()
                            {
                                Ok(i32_arr.values().iter().map(|&x| x as f32).collect())
                            } else if let Some(i64_arr) = inner_array
                                .as_any()
                                .downcast_ref::<arrow::array::Int64Array>(
                            ) {
                                Ok(i64_arr.values().iter().map(|&x| x as f32).collect())
                            } else if let Some(u8_arr) =
                                inner_array.as_any().downcast_ref::<UInt8Array>()
                            {
                                Ok(u8_arr.values().iter().map(|&x| x as f32).collect())
                            } else if let Some(u16_arr) =
                                inner_array.as_any().downcast_ref::<UInt16Array>()
                            {
                                Ok(u16_arr.values().iter().map(|&x| x as f32).collect())
                            } else if let Some(u32_arr) =
                                inner_array.as_any().downcast_ref::<UInt32Array>()
                            {
                                Ok(u32_arr.values().iter().map(|&x| x as f32).collect())
                            } else if let Some(u64_arr) =
                                inner_array.as_any().downcast_ref::<UInt64Array>()
                            {
                                Ok(u64_arr.values().iter().map(|&x| x as f32).collect())
                            } else {
                                Err(datafusion::error::DataFusionError::Execution(format!(
                                    "Unsupported scalar List inner array type: {:?}",
                                    inner_array.data_type()
                                )))
                            }
                        }
                        _ => Err(datafusion::error::DataFusionError::Execution(format!(
                            "Expected List or FixedSizeList scalar, got {:?}",
                            scalar
                        ))),
                    }
                };

                match (lhs, rhs) {
                    // Both are arrays
                    (ColumnarValue::Array(l), ColumnarValue::Array(r)) => {
                        let mut results = Vec::with_capacity(l.len());

                        // Process each row
                        for i in 0..l.len() {
                            let v1 = extract_f32_vec(l, i)?;
                            let v2 = extract_f32_vec(r, i)?;

                            // Validate dimensions match
                            if v1.len() != v2.len() {
                                return Err(datafusion::error::DataFusionError::Execution(
                                    format!(
                                        "Vector dimension mismatch: expected {}, got {}",
                                        v1.len(),
                                        v2.len()
                                    ),
                                ));
                            }

                            // Special handling for Hamming distance on binary vectors
                            if $func_name == "dist_hamming" {
                                // Check if these are binary vectors (values are 0 or 1)
                                let is_binary = v1.iter().all(|&x| x == 0.0 || x == 1.0)
                                    && v2.iter().all(|&x| x == 0.0 || x == 1.0);

                                if is_binary {
                                    // Convert to packed bytes and use bitwise Hamming
                                    let bytes1: Vec<u8> = v1.iter().map(|&x| x as u8).collect();
                                    let bytes2: Vec<u8> = v2.iter().map(|&x| x as u8).collect();
                                    results
                                        .push(distance::hamming_distance_packed(&bytes1, &bytes2)
                                            as f32);
                                } else {
                                    results.push(compute_distance_with_gpu(
                                        &v1,
                                        &v2,
                                        $metric,
                                        distance::$dist_fn,
                                    ));
                                }
                            } else {
                                results.push(compute_distance_with_gpu(
                                    &v1,
                                    &v2,
                                    $metric,
                                    distance::$dist_fn,
                                ));
                            }
                        }

                        Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
                    }
                    // Left is array, right is scalar - broadcast scalar to all rows
                    (ColumnarValue::Array(l), ColumnarValue::Scalar(r_scalar)) => {
                        let v2 = extract_scalar_vec(r_scalar)?;
                        if l.len() == 0 {
                            return Ok(ColumnarValue::Array(Arc::new(Float32Array::from(Vec::<
                                f32,
                            >::new(
                            )))));
                        }
                        let dim = v2.len();

                        if $func_name == "dist_hamming" {
                            let mut results = Vec::with_capacity(l.len());
                            for i in 0..l.len() {
                                let v1 = extract_f32_vec(l, i)?;
                                if v1.len() != dim {
                                    return Err(datafusion::error::DataFusionError::Execution(
                                        format!("Dimension mismatch"),
                                    ));
                                }
                                let is_binary = v1.iter().all(|&x| x == 0.0 || x == 1.0)
                                    && v2.iter().all(|&x| x == 0.0 || x == 1.0);
                                if is_binary {
                                    let bytes1: Vec<u8> = v1.iter().map(|&x| x as u8).collect();
                                    let bytes2: Vec<u8> = v2.iter().map(|&x| x as u8).collect();
                                    results
                                        .push(distance::hamming_distance_packed(&bytes1, &bytes2)
                                            as f32);
                                } else {
                                    results.push(compute_distance_with_gpu(
                                        &v1,
                                        &v2,
                                        $metric,
                                        distance::$dist_fn,
                                    ));
                                }
                            }
                            Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
                        } else {
                            let mut flattened_batch = Vec::with_capacity(l.len() * dim);
                            for i in 0..l.len() {
                                let v1 = extract_f32_vec(l, i)?;
                                if v1.len() != dim {
                                    return Err(datafusion::error::DataFusionError::Execution(
                                        format!(
                                            "Dimension mismatch: expected {}, got {}",
                                            dim,
                                            v1.len()
                                        ),
                                    ));
                                }
                                flattened_batch.extend_from_slice(&v1);
                            }

                            let results = match crate::core::index::gpu::compute_distance(
                                &v2,
                                &flattened_batch,
                                dim,
                                $metric,
                            ) {
                                Ok(dist) => dist,
                                Err(_) => {
                                    let mut cpu_res = Vec::with_capacity(l.len());
                                    for i in 0..l.len() {
                                        cpu_res.push(distance::$dist_fn(
                                            &v2,
                                            &flattened_batch[i * dim..(i + 1) * dim],
                                        ));
                                    }
                                    cpu_res
                                }
                            };
                            Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
                        }
                    }
                    // Right is array, left is scalar - broadcast scalar to all rows
                    (ColumnarValue::Scalar(l_scalar), ColumnarValue::Array(r)) => {
                        let v1 = extract_scalar_vec(l_scalar)?;
                        if r.len() == 0 {
                            return Ok(ColumnarValue::Array(Arc::new(Float32Array::from(Vec::<
                                f32,
                            >::new(
                            )))));
                        }
                        let dim = v1.len();

                        if $func_name == "dist_hamming" {
                            let mut results = Vec::with_capacity(r.len());
                            for i in 0..r.len() {
                                let v2 = extract_f32_vec(r, i)?;
                                if v2.len() != dim {
                                    return Err(datafusion::error::DataFusionError::Execution(
                                        format!("Dimension mismatch"),
                                    ));
                                }
                                let is_binary = v1.iter().all(|&x| x == 0.0 || x == 1.0)
                                    && v2.iter().all(|&x| x == 0.0 || x == 1.0);
                                if is_binary {
                                    let bytes1: Vec<u8> = v1.iter().map(|&x| x as u8).collect();
                                    let bytes2: Vec<u8> = v2.iter().map(|&x| x as u8).collect();
                                    results
                                        .push(distance::hamming_distance_packed(&bytes1, &bytes2)
                                            as f32);
                                } else {
                                    results.push(compute_distance_with_gpu(
                                        &v1,
                                        &v2,
                                        $metric,
                                        distance::$dist_fn,
                                    ));
                                }
                            }
                            Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
                        } else {
                            let mut flattened_batch = Vec::with_capacity(r.len() * dim);
                            for i in 0..r.len() {
                                let v2 = extract_f32_vec(r, i)?;
                                if v2.len() != dim {
                                    return Err(datafusion::error::DataFusionError::Execution(
                                        format!(
                                            "Dimension mismatch: expected {}, got {}",
                                            dim,
                                            v2.len()
                                        ),
                                    ));
                                }
                                flattened_batch.extend_from_slice(&v2);
                            }

                            let results = match crate::core::index::gpu::compute_distance(
                                &v1,
                                &flattened_batch,
                                dim,
                                $metric,
                            ) {
                                Ok(dist) => dist,
                                Err(_) => {
                                    let mut cpu_res = Vec::with_capacity(r.len());
                                    for i in 0..r.len() {
                                        cpu_res.push(distance::$dist_fn(
                                            &v1,
                                            &flattened_batch[i * dim..(i + 1) * dim],
                                        ));
                                    }
                                    cpu_res
                                }
                            };
                            Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
                        }
                    }
                    // Both are scalars
                    (ColumnarValue::Scalar(l_scalar), ColumnarValue::Scalar(r_scalar)) => {
                        let v1 = extract_scalar_vec(l_scalar)?;
                        let v2 = extract_scalar_vec(r_scalar)?;

                        // Validate dimensions match
                        if v1.len() != v2.len() {
                            return Err(datafusion::error::DataFusionError::Execution(format!(
                                "Vector dimension mismatch: expected {}, got {}",
                                v1.len(),
                                v2.len()
                            )));
                        }

                        let result = if $func_name == "dist_hamming" {
                            let is_binary = v1.iter().all(|&x| x == 0.0 || x == 1.0)
                                && v2.iter().all(|&x| x == 0.0 || x == 1.0);

                            if is_binary {
                                let bytes1: Vec<u8> = v1.iter().map(|&x| x as u8).collect();
                                let bytes2: Vec<u8> = v2.iter().map(|&x| x as u8).collect();
                                distance::hamming_distance_packed(&bytes1, &bytes2) as f32
                            } else {
                                compute_distance_with_gpu(&v1, &v2, $metric, distance::$dist_fn)
                            }
                        } else {
                            compute_distance_with_gpu(&v1, &v2, $metric, distance::$dist_fn)
                        };

                        Ok(ColumnarValue::Scalar(ScalarValue::Float32(Some(result))))
                    }
                }
            }
        }
    };
}

// Instantiate distance UDFs via macro
make_vector_dist_udf!(
    L2DistUDF,
    "dist_l2",
    l2_distance,
    crate::core::index::VectorMetric::L2
);
make_vector_dist_udf!(
    CosineDistUDF,
    "dist_cosine",
    cosine_distance,
    crate::core::index::VectorMetric::Cosine
);
make_vector_dist_udf!(
    IPDistUDF,
    "dist_ip",
    dot_product,
    crate::core::index::VectorMetric::InnerProduct
);
make_vector_dist_udf!(
    L1DistUDF,
    "dist_l1",
    l1_distance,
    crate::core::index::VectorMetric::L1
);
make_vector_dist_udf!(
    HammingDistUDF,
    "dist_hamming",
    hamming_distance,
    crate::core::index::VectorMetric::Hamming
);
make_vector_dist_udf!(
    JaccardDistUDF,
    "dist_jaccard",
    jaccard_distance,
    crate::core::index::VectorMetric::Jaccard
);

#[cfg(test)]
mod sparse_distance_tests {
    use super::*;
    use crate::core::index::distance::{cosine_distance, dot_product, l2_distance};

    #[test]
    fn test_sparse_l2_distance_basic() {
        let a = SparseVector {
            indices: vec![0, 2],
            values: vec![1.0, 3.0],
            dim: 4,
        };
        let b = SparseVector {
            indices: vec![1, 3],
            values: vec![2.0, 4.0],
            dim: 4,
        };
        let sparse_dist = sparse_l2_distance(&a, &b);
        let dense_a = sparse_to_dense(&a);
        let dense_b = sparse_to_dense(&b);
        let dense_dist = l2_distance(&dense_a, &dense_b);
        assert!((sparse_dist - dense_dist).abs() < 1e-5);
    }

    #[test]
    fn test_sparse_cosine_distance_basic() {
        let a = SparseVector {
            indices: vec![0, 2],
            values: vec![1.0, 3.0],
            dim: 4,
        };
        let b = SparseVector {
            indices: vec![1, 3],
            values: vec![2.0, 4.0],
            dim: 4,
        };
        let sparse_dist = sparse_cosine_distance(&a, &b);
        let dense_a = sparse_to_dense(&a);
        let dense_b = sparse_to_dense(&b);
        let dense_dist = cosine_distance(&dense_a, &dense_b);
        assert!((sparse_dist - dense_dist).abs() < 1e-4);
    }

    #[test]
    fn test_sparse_inner_product_distance_basic() {
        let a = SparseVector {
            indices: vec![0, 2],
            values: vec![1.0, 3.0],
            dim: 4,
        };
        let b = SparseVector {
            indices: vec![0, 2],
            values: vec![2.0, 4.0],
            dim: 4,
        };
        let sparse_dist = sparse_inner_product_distance(&a, &b);
        let dense_a = sparse_to_dense(&a);
        let dense_b = sparse_to_dense(&b);
        let dense_dist = -dot_product(&dense_a, &dense_b);
        assert!((sparse_dist - dense_dist).abs() < 1e-5);
    }

    #[test]
    fn test_dense_to_sparse_round_trip() {
        let dense = vec![0.0, 1.5, 0.0, 2.5, 0.0];
        let sparse = dense_to_sparse(&dense);
        let recovered = sparse_to_dense(&sparse);
        assert_eq!(dense.len(), recovered.len());
        for (a, b) in dense.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
