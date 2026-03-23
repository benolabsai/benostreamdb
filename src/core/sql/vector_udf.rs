use std::any::Any;
use std::sync::Arc;
use arrow::array::{Array, ArrayRef, Float32Array, Float64Array, FixedSizeListArray, Int8Array, Int16Array, Int32Array, Int64Array, ListBuilder, Float32Builder, ListArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array};
use arrow::datatypes::DataType;
use datafusion::error::Result;
use datafusion::logical_expr::{ScalarUDF, ScalarUDFImpl, Signature, Volatility, ColumnarValue, AggregateUDF, AggregateUDFImpl};
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_functions_aggregate_common::accumulator::{AccumulatorArgs, StateFieldsArgs};
use datafusion::common::cast::as_fixed_size_list_array;
use datafusion::scalar::ScalarValue;
use crate::core::index::distance;
use crate::core::index::SparseVector;

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
            fn eq(&self, other: &Self) -> bool {
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
fn compute_distance_with_gpu<F>(v1: &[f32], v2: &[f32], metric: crate::core::index::VectorMetric, cpu_fn: F) -> f32
where
    F: Fn(&[f32], &[f32]) -> f32,
{
    use crate::core::index::gpu;
    
    // Check if GPU context is available
    if let Some(ctx) = gpu::get_global_gpu_context() {
        // Prepare vectors for batch computation (single pair)
        let dim = v1.len();
        
        // Call GPU compute_distance with v2 as a "batch" of 1 vector
        match gpu::compute_distance(v1, v2, dim, metric, &ctx) {
            Ok(distances) => {
                // Extract the single distance result
                if let Some(&dist) = distances.first() {
                    return dist;
                }
                // If GPU computation returned empty result, fall back to CPU
            }
            Err(_) => {
                // GPU computation failed, fall back to CPU
                // Error is silently handled - this is expected behavior for graceful degradation
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

            fn invoke_with_args(&self, args: datafusion::logical_expr::ScalarFunctionArgs) -> Result<datafusion::logical_expr::ColumnarValue> {
                use datafusion::logical_expr::ColumnarValue;
                use arrow::array::{Float16Array, UInt8Array};
                
                let (lhs, rhs) = (&args.args[0], &args.args[1]);
                
                // Helper function to extract Float32 values from either List or FixedSizeList
                let extract_f32_vec = |arr: &ArrayRef, idx: usize| -> Result<Vec<f32>> {
                    match arr.data_type() {
                        DataType::FixedSizeList(_, _) => {
                            let list_arr = as_fixed_size_list_array(arr)?;
                            let value_arr = list_arr.value(idx);
                            
                            if let Some(f32_arr) = value_arr.as_any().downcast_ref::<Float32Array>() {
                                Ok(f32_arr.values().to_vec())
                            } else if let Some(f16_arr) = value_arr.as_any().downcast_ref::<Float16Array>() {
                                Ok(f16_arr.iter().map(|f| f.unwrap_or_default().to_f32()).collect())
                            } else if let Some(u8_arr) = value_arr.as_any().downcast_ref::<UInt8Array>() {
                                // Binary vector case
                                Ok(u8_arr.values().iter().map(|&b| b as f32).collect())
                            } else {
                                Err(datafusion::error::DataFusionError::Execution(
                                    "Unsupported inner array type".to_string()
                                ))
                            }
                        }
                        DataType::List(_) => {
                            let list_arr = arr.as_any().downcast_ref::<ListArray>()
                                .ok_or_else(|| datafusion::error::DataFusionError::Execution(
                                    "Failed to downcast to ListArray".to_string()
                                ))?;
                            let value_arr = list_arr.value(idx);
                            
                            if let Some(f32_arr) = value_arr.as_any().downcast_ref::<Float32Array>() {
                                Ok(f32_arr.values().to_vec())
                            } else if let Some(f16_arr) = value_arr.as_any().downcast_ref::<Float16Array>() {
                                Ok(f16_arr.iter().map(|f| f.unwrap_or_default().to_f32()).collect())
                            } else if let Some(u8_arr) = value_arr.as_any().downcast_ref::<UInt8Array>() {
                                // Binary vector case
                                Ok(u8_arr.values().iter().map(|&b| b as f32).collect())
                            } else {
                                Err(datafusion::error::DataFusionError::Execution(
                                    "Unsupported inner array type in List".to_string()
                                ))
                            }
                        }
                        _ => Err(datafusion::error::DataFusionError::Execution(
                            format!("Expected List or FixedSizeList, got {:?}", arr.data_type())
                        ))
                    }
                };
                
                // Helper to extract vector from scalar value
                let extract_scalar_vec = |scalar: &ScalarValue| -> Result<Vec<f32>> {
                    match scalar {
                        ScalarValue::FixedSizeList(arr) => {
                            if let Some(f32_arr) = arr.as_any().downcast_ref::<Float32Array>() {
                                Ok(f32_arr.values().to_vec())
                            } else {
                                Err(datafusion::error::DataFusionError::Execution(
                                    "Unsupported scalar FixedSizeList inner type".to_string()
                                ))
                            }
                        }
                        ScalarValue::List(list_arc) => {
                            // ScalarValue::List wraps a ListArray
                            let list_array = list_arc.as_ref();
                            if list_array.len() == 0 {
                                return Err(datafusion::error::DataFusionError::Execution(
                                    "Empty List scalar".to_string()
                                ));
                            }
                            
                            // Get the first (and only) element
                            let inner_array = list_array.value(0);
                            
                            if let Some(f32_arr) = inner_array.as_any().downcast_ref::<Float32Array>() {
                                Ok(f32_arr.values().to_vec())
                            } else if let Some(f64_arr) = inner_array.as_any().downcast_ref::<arrow::array::Float64Array>() {
                                Ok(f64_arr.values().iter().map(|&x| x as f32).collect())
                            } else if let Some(f16_arr) = inner_array.as_any().downcast_ref::<Float16Array>() {
                                Ok(f16_arr.iter().map(|f| f.unwrap_or_default().to_f32()).collect())
                            } else if let Some(i8_arr) = inner_array.as_any().downcast_ref::<Int8Array>() {
                                Ok(i8_arr.values().iter().map(|&x| x as f32).collect())
                            } else if let Some(i16_arr) = inner_array.as_any().downcast_ref::<Int16Array>() {
                                Ok(i16_arr.values().iter().map(|&x| x as f32).collect())
                            } else if let Some(i32_arr) = inner_array.as_any().downcast_ref::<Int32Array>() {
                                Ok(i32_arr.values().iter().map(|&x| x as f32).collect())
                            } else if let Some(i64_arr) = inner_array.as_any().downcast_ref::<arrow::array::Int64Array>() {
                                Ok(i64_arr.values().iter().map(|&x| x as f32).collect())
                            } else if let Some(u8_arr) = inner_array.as_any().downcast_ref::<UInt8Array>() {
                                Ok(u8_arr.values().iter().map(|&x| x as f32).collect())
                            } else if let Some(u16_arr) = inner_array.as_any().downcast_ref::<UInt16Array>() {
                                Ok(u16_arr.values().iter().map(|&x| x as f32).collect())
                            } else if let Some(u32_arr) = inner_array.as_any().downcast_ref::<UInt32Array>() {
                                Ok(u32_arr.values().iter().map(|&x| x as f32).collect())
                            } else if let Some(u64_arr) = inner_array.as_any().downcast_ref::<UInt64Array>() {
                                Ok(u64_arr.values().iter().map(|&x| x as f32).collect())
                            } else {
                                Err(datafusion::error::DataFusionError::Execution(
                                    format!("Unsupported scalar List inner array type: {:?}", inner_array.data_type())
                                ))
                            }
                        }
                        _ => Err(datafusion::error::DataFusionError::Execution(
                            format!("Expected List or FixedSizeList scalar, got {:?}", scalar)
                        ))
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
                                    format!("Vector dimension mismatch: expected {}, got {}", v1.len(), v2.len())
                                ));
                            }
                            
                            // Special handling for Hamming distance on binary vectors
                            if $func_name == "dist_hamming" {
                                // Check if these are binary vectors (values are 0 or 1)
                                let is_binary = v1.iter().all(|&x| x == 0.0 || x == 1.0) && 
                                               v2.iter().all(|&x| x == 0.0 || x == 1.0);
                                
                                if is_binary {
                                    // Convert to packed bytes and use bitwise Hamming
                                    let bytes1: Vec<u8> = v1.iter().map(|&x| x as u8).collect();
                                    let bytes2: Vec<u8> = v2.iter().map(|&x| x as u8).collect();
                                    results.push(distance::hamming_distance_packed(&bytes1, &bytes2) as f32);
                                } else {
                                    results.push(compute_distance_with_gpu(&v1, &v2, $metric, distance::$dist_fn));
                                }
                            } else {
                                results.push(compute_distance_with_gpu(&v1, &v2, $metric, distance::$dist_fn));
                            }
                        }
                        
                        Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
                    },
                    // Left is array, right is scalar - broadcast scalar to all rows
                    (ColumnarValue::Array(l), ColumnarValue::Scalar(r_scalar)) => {
                        let mut results = Vec::with_capacity(l.len());
                        let v2 = extract_scalar_vec(r_scalar)?;
                        
                        for i in 0..l.len() {
                            let v1 = extract_f32_vec(l, i)?;
                            
                            // Validate dimensions match
                            if v1.len() != v2.len() {
                                return Err(datafusion::error::DataFusionError::Execution(
                                    format!("Vector dimension mismatch: expected {}, got {}", v1.len(), v2.len())
                                ));
                            }
                            
                            if $func_name == "dist_hamming" {
                                let is_binary = v1.iter().all(|&x| x == 0.0 || x == 1.0) && 
                                               v2.iter().all(|&x| x == 0.0 || x == 1.0);
                                
                                if is_binary {
                                    let bytes1: Vec<u8> = v1.iter().map(|&x| x as u8).collect();
                                    let bytes2: Vec<u8> = v2.iter().map(|&x| x as u8).collect();
                                    results.push(distance::hamming_distance_packed(&bytes1, &bytes2) as f32);
                                } else {
                                    results.push(compute_distance_with_gpu(&v1, &v2, $metric, distance::$dist_fn));
                                }
                            } else {
                                results.push(compute_distance_with_gpu(&v1, &v2, $metric, distance::$dist_fn));
                            }
                        }
                        
                        Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
                    },
                    // Right is array, left is scalar - broadcast scalar to all rows
                    (ColumnarValue::Scalar(l_scalar), ColumnarValue::Array(r)) => {
                        let mut results = Vec::with_capacity(r.len());
                        let v1 = extract_scalar_vec(l_scalar)?;
                        
                        for i in 0..r.len() {
                            let v2 = extract_f32_vec(r, i)?;
                            
                            // Validate dimensions match
                            if v1.len() != v2.len() {
                                return Err(datafusion::error::DataFusionError::Execution(
                                    format!("Vector dimension mismatch: expected {}, got {}", v1.len(), v2.len())
                                ));
                            }
                            
                            if $func_name == "dist_hamming" {
                                let is_binary = v1.iter().all(|&x| x == 0.0 || x == 1.0) && 
                                               v2.iter().all(|&x| x == 0.0 || x == 1.0);
                                
                                if is_binary {
                                    let bytes1: Vec<u8> = v1.iter().map(|&x| x as u8).collect();
                                    let bytes2: Vec<u8> = v2.iter().map(|&x| x as u8).collect();
                                    results.push(distance::hamming_distance_packed(&bytes1, &bytes2) as f32);
                                } else {
                                    results.push(compute_distance_with_gpu(&v1, &v2, $metric, distance::$dist_fn));
                                }
                            } else {
                                results.push(compute_distance_with_gpu(&v1, &v2, $metric, distance::$dist_fn));
                            }
                        }
                        
                        Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
                    },
                    // Both are scalars
                    (ColumnarValue::Scalar(l_scalar), ColumnarValue::Scalar(r_scalar)) => {
                        let v1 = extract_scalar_vec(l_scalar)?;
                        let v2 = extract_scalar_vec(r_scalar)?;
                        
                        // Validate dimensions match
                        if v1.len() != v2.len() {
                            return Err(datafusion::error::DataFusionError::Execution(
                                format!("Vector dimension mismatch: expected {}, got {}", v1.len(), v2.len())
                            ));
                        }
                        
                        let result = if $func_name == "dist_hamming" {
                            let is_binary = v1.iter().all(|&x| x == 0.0 || x == 1.0) && 
                                           v2.iter().all(|&x| x == 0.0 || x == 1.0);
                            
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
                    },
                }
            }
        }
    };
}

/// Generic Macro for Element-wise Binary Ops
macro_rules! create_vector_binary_op_udf {
    ($name:ident, $func_name:expr, $op_fn:ident) => {
        #[derive(Debug)]
        pub struct $name { signature: Signature }
        
        impl_dyn_traits!($name);
        
        impl $name { pub fn new() -> Self { Self { signature: Signature::exact(vec![DataType::Float32, DataType::Float32], Volatility::Immutable) } } }
        impl ScalarUDFImpl for $name {
            fn as_any(&self) -> &dyn Any { self }
            fn name(&self) -> &str { $func_name }
            fn signature(&self) -> &Signature { &self.signature }
            fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> { Ok(arg_types[0].clone()) }
            fn invoke_with_args(&self, args: datafusion::logical_expr::ScalarFunctionArgs) -> Result<ColumnarValue> {
                let (lhs, rhs) = (&args.args[0], &args.args[1]);
                match (lhs, rhs) {
                    (ColumnarValue::Array(l), ColumnarValue::Array(r)) => {
                        let l_arr = as_fixed_size_list_array(l)?;
                        let r_arr = as_fixed_size_list_array(r)?;
                        let len = l_arr.value_length();
                        let mut builder = Float32Array::builder(l_arr.len() * len as usize);
                        for i in 0..l_arr.len() {
                            let v1_array = l_arr.value(i);
                            let v2_array = r_arr.value(i);
                            let v1 = v1_array.as_any().downcast_ref::<Float32Array>().unwrap().values();
                            let v2 = v2_array.as_any().downcast_ref::<Float32Array>().unwrap().values();
                            builder.append_slice(&$op_fn(v1, v2));
                        }
                        Ok(ColumnarValue::Array(Arc::new(FixedSizeListArray::try_new(
                            Arc::new(arrow::datatypes::Field::new("item", DataType::Float32, true)),
                            len,
                            Arc::new(builder.finish()),
                            None,
                        )?)))
                    },
                    _ => return Err(datafusion::error::DataFusionError::Execution(
                        "Unsupported arguments".to_string()
                    )),
                }
            }
        }
    };
}

make_vector_dist_udf!(L2DistUDF, "dist_l2", l2_distance, crate::core::index::VectorMetric::L2);
make_vector_dist_udf!(CosineDistUDF, "dist_cosine", cosine_distance, crate::core::index::VectorMetric::Cosine);
make_vector_dist_udf!(IPDistUDF, "dist_ip", dot_product, crate::core::index::VectorMetric::InnerProduct);
make_vector_dist_udf!(L1DistUDF, "dist_l1", l1_distance, crate::core::index::VectorMetric::L1);
make_vector_dist_udf!(HammingDistUDF, "dist_hamming", hamming_distance, crate::core::index::VectorMetric::Hamming);
make_vector_dist_udf!(JaccardDistUDF, "dist_jaccard", jaccard_distance, crate::core::index::VectorMetric::Jaccard);

pub fn all_vector_udfs() -> Vec<ScalarUDF> {
    vec![
        ScalarUDF::new_from_impl(L2DistUDF::new()),
        ScalarUDF::new_from_impl(CosineDistUDF::new()),
        ScalarUDF::new_from_impl(IPDistUDF::new()),
        ScalarUDF::new_from_impl(L1DistUDF::new()),
        ScalarUDF::new_from_impl(HammingDistUDF::new()),
        ScalarUDF::new_from_impl(JaccardDistUDF::new()),
        
        // Element-wise
        ScalarUDF::new_from_impl(VectorAddUDF::new()),
        ScalarUDF::new_from_impl(VectorSubUDF::new()),
        ScalarUDF::new_from_impl(VectorMulUDF::new()),
        ScalarUDF::new_from_impl(VectorConcatUDF::new()),
        
        // Utility
        ScalarUDF::new_from_impl(VectorDimsUDF::new()),
        ScalarUDF::new_from_impl(VectorNormUDF::new()),
        ScalarUDF::new_from_impl(VectorNormalizeUDF::new()),
        ScalarUDF::new_from_impl(BinaryQuantizeUDF::new()),
        ScalarUDF::new_from_impl(SubvectorUDF::new()),
        
        // Type casting
        ScalarUDF::new_from_impl(VectorToSparseUDF::new()),
        ScalarUDF::new_from_impl(SparseToVectorUDF::new()),
        ScalarUDF::new_from_impl(VectorToBinaryUDF::new()),
    ]
}

pub fn all_vector_aggregates() -> Vec<AggregateUDF> {
    vec![
        AggregateUDF::new_from_impl(VectorSumUDF::new()),
        AggregateUDF::new_from_impl(VectorAvgUDF::new()),
    ]
}

// --- Element-wise ---

create_vector_binary_op_udf!(VectorAddUDF, "vector_add", add_vectors);
create_vector_binary_op_udf!(VectorSubUDF, "vector_sub", sub_vectors);
create_vector_binary_op_udf!(VectorMulUDF, "vector_mul", mul_vectors);

fn add_vectors(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b.iter()).map(|(x, y)| x + y).collect() }
fn sub_vectors(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b.iter()).map(|(x, y)| x - y).collect() }
fn mul_vectors(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b.iter()).map(|(x, y)| x * y).collect() }

#[derive(Debug)]
pub struct VectorConcatUDF {
    signature: Signature,
}

impl_dyn_traits!(VectorConcatUDF);

impl VectorConcatUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(vec![DataType::Float32, DataType::Float32], Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for VectorConcatUDF {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "vector_concat" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        // Technically this creates a NEW size, so FixedSizeList is tricky.
        // For MVP, return List<Float32>
        Ok(DataType::List(Arc::new(arrow::datatypes::Field::new("item", DataType::Float32, true))))
    }
    fn invoke_with_args(&self, args: datafusion::logical_expr::ScalarFunctionArgs) -> Result<ColumnarValue> {
        let (lhs, rhs) = (&args.args[0], &args.args[1]);
        match (lhs, rhs) {
            (ColumnarValue::Array(l), ColumnarValue::Array(r)) => {
                let l_arr = as_fixed_size_list_array(l)?;
                let r_arr = as_fixed_size_list_array(r)?;
                
                let mut builder = arrow::array::ListBuilder::new(arrow::array::Float32Builder::new());
                for i in 0..l_arr.len() {
                    let v1_array = l_arr.value(i);
                    let v2_array = r_arr.value(i);
                    let v1 = v1_array.as_any().downcast_ref::<Float32Array>().unwrap();
                    let v2 = v2_array.as_any().downcast_ref::<Float32Array>().unwrap();
                    
                    let concatenated: Vec<f32> = v1.values().iter().chain(v2.values().iter()).copied().collect();
                    builder.values().append_slice(&concatenated);
                    builder.append(true);
                }
                Ok(ColumnarValue::Array(Arc::new(builder.finish())))
            },
            _ => return Err(datafusion::error::DataFusionError::Execution(
                "Unsupported argument combinations for vector_concat".to_string()
            )),
        }
    }
}

// --- Utility ---

#[derive(Debug)]
pub struct VectorDimsUDF { signature: Signature }
impl_dyn_traits!(VectorDimsUDF);
impl VectorDimsUDF { pub fn new() -> Self { Self { signature: Signature::any(1, Volatility::Immutable) } } }
impl ScalarUDFImpl for VectorDimsUDF {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "vector_dims" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> { Ok(DataType::Int32) }
    fn invoke_with_args(&self, args: datafusion::logical_expr::ScalarFunctionArgs) -> Result<ColumnarValue> {
        match &args.args[0] {
            ColumnarValue::Array(arr) => {
                if let Some(fsl) = arr.as_any().downcast_ref::<FixedSizeListArray>() {
                    let len = fsl.value_length();
                    let results: Int32Array = (0..fsl.len()).map(|_| Some(len)).collect();
                    Ok(ColumnarValue::Array(Arc::new(results)))
                } else { Ok(ColumnarValue::Scalar(ScalarValue::Int32(None))) }
            },
            _ => Ok(ColumnarValue::Scalar(ScalarValue::Int32(None))),
        }
    }
}

#[derive(Debug)]
pub struct VectorNormUDF { signature: Signature }
impl_dyn_traits!(VectorNormUDF);
impl VectorNormUDF { pub fn new() -> Self { Self { signature: Signature::any(1, Volatility::Immutable) } } }
impl ScalarUDFImpl for VectorNormUDF {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "vector_norm" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> { Ok(DataType::Float32) }
    fn invoke_with_args(&self, args: datafusion::logical_expr::ScalarFunctionArgs) -> Result<ColumnarValue> {
         match &args.args[0] {
            ColumnarValue::Array(arr) => {
                let fsl = as_fixed_size_list_array(arr)?;
                let mut results = Vec::with_capacity(fsl.len());
                for i in 0..fsl.len() {
                    let value_array = fsl.value(i);
                    let v = value_array.as_any().downcast_ref::<Float32Array>().unwrap().values();
                    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    results.push(norm);
                }
                Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
            },
            _ => Ok(ColumnarValue::Scalar(ScalarValue::Float32(None))),
        }
    }
}

#[derive(Debug)]
pub struct VectorNormalizeUDF { signature: Signature }
impl_dyn_traits!(VectorNormalizeUDF);
impl VectorNormalizeUDF { pub fn new() -> Self { Self { signature: Signature::any(1, Volatility::Immutable) } } }
impl ScalarUDFImpl for VectorNormalizeUDF {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "l2_normalize" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> { Ok(arg_types[0].clone()) }
    fn invoke_with_args(&self, args: datafusion::logical_expr::ScalarFunctionArgs) -> Result<ColumnarValue> {
        match &args.args[0] {
            ColumnarValue::Array(arr) => {
                let fsl = as_fixed_size_list_array(arr)?;
                let len = fsl.value_length();
                let mut builder = Float32Array::builder(fsl.len() * len as usize);
                
                for i in 0..fsl.len() {
                    let value_array = fsl.value(i);
                    let v = value_array.as_any().downcast_ref::<Float32Array>().unwrap().values();
                    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        let normalized: Vec<f32> = v.iter().map(|x| x / norm).collect();
                        builder.append_slice(&normalized);
                    } else {
                        builder.append_slice(v);
                    }
                }
                
                Ok(ColumnarValue::Array(Arc::new(FixedSizeListArray::try_new(
                    Arc::new(arrow::datatypes::Field::new("item", DataType::Float32, true)),
                    len,
                    Arc::new(builder.finish()),
                    None,
                )?)))
            },
            _ => Ok(ColumnarValue::Scalar(ScalarValue::Null)),
        }
    }
}

#[derive(Debug)]
pub struct BinaryQuantizeUDF { signature: Signature }
impl_dyn_traits!(BinaryQuantizeUDF);
impl BinaryQuantizeUDF { pub fn new() -> Self { Self { signature: Signature::any(1, Volatility::Immutable) } } }
impl ScalarUDFImpl for BinaryQuantizeUDF {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "binary_quantize" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> { 
        // Use List instead of FixedSizeList since size is determined at runtime
        Ok(DataType::List(Arc::new(arrow::datatypes::Field::new("item", DataType::UInt8, true))))
    }
    fn invoke_with_args(&self, args: datafusion::logical_expr::ScalarFunctionArgs) -> Result<ColumnarValue> {
        match &args.args[0] {
            ColumnarValue::Array(arr) => {
                // Handle both List and FixedSizeList arrays
                let vec_data: Vec<Vec<f32>> = if let Some(list_arr) = arr.as_any().downcast_ref::<ListArray>() {
                    // Handle List type
                    (0..list_arr.len()).map(|i| {
                        let value_array = list_arr.value(i);
                        // Extract f32 values from the inner array
                        if let Some(f32_arr) = value_array.as_any().downcast_ref::<Float32Array>() {
                            f32_arr.values().to_vec()
                        } else if let Some(f64_arr) = value_array.as_any().downcast_ref::<Float64Array>() {
                            f64_arr.values().iter().map(|&x| x as f32).collect()
                        } else if let Some(i32_arr) = value_array.as_any().downcast_ref::<Int32Array>() {
                            i32_arr.values().iter().map(|&x| x as f32).collect()
                        } else if let Some(i64_arr) = value_array.as_any().downcast_ref::<Int64Array>() {
                            i64_arr.values().iter().map(|&x| x as f32).collect()
                        } else if let Some(u8_arr) = value_array.as_any().downcast_ref::<UInt8Array>() {
                            u8_arr.values().iter().map(|&x| x as f32).collect()
                        } else {
                            vec![]
                        }
                    }).collect()
                } else if let Some(fsl) = arr.as_any().downcast_ref::<FixedSizeListArray>() {
                    // Handle FixedSizeList type
                    (0..fsl.len()).map(|i| {
                        let value_array = fsl.value(i);
                        // Extract f32 values from the inner array
                        if let Some(f32_arr) = value_array.as_any().downcast_ref::<Float32Array>() {
                            f32_arr.values().to_vec()
                        } else if let Some(f64_arr) = value_array.as_any().downcast_ref::<Float64Array>() {
                            f64_arr.values().iter().map(|&x| x as f32).collect()
                        } else if let Some(i32_arr) = value_array.as_any().downcast_ref::<Int32Array>() {
                            i32_arr.values().iter().map(|&x| x as f32).collect()
                        } else if let Some(i64_arr) = value_array.as_any().downcast_ref::<Int64Array>() {
                            i64_arr.values().iter().map(|&x| x as f32).collect()
                        } else if let Some(u8_arr) = value_array.as_any().downcast_ref::<UInt8Array>() {
                            u8_arr.values().iter().map(|&x| x as f32).collect()
                        } else {
                            vec![]
                        }
                    }).collect()
                } else {
                    return Err(datafusion::error::DataFusionError::Execution(
                        "binary_quantize expects List or FixedSizeList array".to_string()
                    ));
                };
                
                let packed_len = if vec_data.is_empty() { 0 } else { (vec_data[0].len() + 7) / 8 };
                let mut list_builder = ListBuilder::new(arrow::array::UInt8Builder::new());
                
                for v in vec_data {
                    let mut packed = vec![0u8; packed_len];
                    for (j, val) in v.iter().enumerate() {
                        if *val > 0.0 {
                            packed[j / 8] |= 1 << (j % 8);
                        }
                    }
                    for b in packed { 
                        list_builder.values().append_value(b); 
                    }
                    list_builder.append(true);
                }
                
                Ok(ColumnarValue::Array(Arc::new(list_builder.finish())))
            },
            ColumnarValue::Scalar(scalar) => {
                // Handle scalar input (e.g., ARRAY[1.0, 0.0, 0.0])
                let v: Vec<f32> = match scalar {
                    ScalarValue::List(list_arc) => {
                        // ScalarValue::List wraps a ListArray
                        let list_array = list_arc.as_ref();
                        if list_array.len() == 0 {
                            return Err(datafusion::error::DataFusionError::Execution(
                                "Empty List scalar".to_string()
                            ));
                        }
                        
                        // Get the first (and only) element
                        let inner_array = list_array.value(0);
                        
                        if let Some(f32_arr) = inner_array.as_any().downcast_ref::<Float32Array>() {
                            f32_arr.values().to_vec()
                        } else if let Some(f64_arr) = inner_array.as_any().downcast_ref::<Float64Array>() {
                            f64_arr.values().iter().map(|&x| x as f32).collect()
                        } else if let Some(i32_arr) = inner_array.as_any().downcast_ref::<Int32Array>() {
                            i32_arr.values().iter().map(|&x| x as f32).collect()
                        } else if let Some(i64_arr) = inner_array.as_any().downcast_ref::<Int64Array>() {
                            i64_arr.values().iter().map(|&x| x as f32).collect()
                        } else if let Some(u8_arr) = inner_array.as_any().downcast_ref::<UInt8Array>() {
                            u8_arr.values().iter().map(|&x| x as f32).collect()
                        } else {
                            return Err(datafusion::error::DataFusionError::Execution(
                                format!("Unsupported inner array type in List scalar: {:?}", inner_array.data_type())
                            ));
                        }
                    }
                    ScalarValue::FixedSizeList(arr) => {
                        if let Some(f32_arr) = arr.as_any().downcast_ref::<Float32Array>() {
                            f32_arr.values().to_vec()
                        } else {
                            return Err(datafusion::error::DataFusionError::Execution(
                                "Unsupported scalar FixedSizeList inner type".to_string()
                            ));
                        }
                    }
                    _ => {
                        return Err(datafusion::error::DataFusionError::Execution(
                            "binary_quantize expects List or FixedSizeList scalar".to_string()
                        ));
                    }
                };
                
                let packed_len = (v.len() + 7) / 8;
                let mut packed = vec![0u8; packed_len];
                for (j, val) in v.iter().enumerate() {
                    if *val > 0.0 {
                        packed[j / 8] |= 1 << (j % 8);
                    }
                }
                Ok(ColumnarValue::Scalar(ScalarValue::List(ScalarValue::new_list_nullable(
                    &packed.iter().map(|&b| ScalarValue::UInt8(Some(b))).collect::<Vec<_>>(),
                    &DataType::UInt8,
                ))))
            },
        }
    }
}

#[derive(Debug)]
pub struct SubvectorUDF { signature: Signature }
impl_dyn_traits!(SubvectorUDF);
impl SubvectorUDF { pub fn new() -> Self { Self { signature: Signature::exact(vec![DataType::Float32, DataType::Int32, DataType::Int32], Volatility::Immutable) } } }
impl ScalarUDFImpl for SubvectorUDF {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "subvector" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> { 
        Ok(DataType::List(Arc::new(arrow::datatypes::Field::new("item", DataType::Float32, true))))
    }
    fn invoke_with_args(&self, args: datafusion::logical_expr::ScalarFunctionArgs) -> Result<ColumnarValue> {
        let (vec_arg, start_arg, count_arg) = (&args.args[0], &args.args[1], &args.args[2]);
        
        match (vec_arg, start_arg, count_arg) {
            (ColumnarValue::Array(arr), ColumnarValue::Scalar(ScalarValue::Int32(Some(start))), ColumnarValue::Scalar(ScalarValue::Int32(Some(count)))) => {
                let fsl = as_fixed_size_list_array(arr)?;
                let mut builder = arrow::array::ListBuilder::new(Float32Array::builder(0));
                
                for i in 0..fsl.len() {
                    let value_array = fsl.value(i);
                    let v = value_array.as_any().downcast_ref::<Float32Array>().unwrap().values();
                    let s = (*start as usize).min(v.len());
                    let c = (*count as usize).min(v.len() - s);
                    builder.values().append_slice(&v[s..s+c]);
                    builder.append(true);
                }
                Ok(ColumnarValue::Array(Arc::new(builder.finish())))
            },
            _ => Ok(ColumnarValue::Scalar(ScalarValue::Null)),
        }
    }
}

// --- Type Casting ---

#[derive(Debug)]
struct VectorToSparseUDF { signature: Signature }
impl_dyn_traits!(VectorToSparseUDF);
impl VectorToSparseUDF { 
    pub fn new() -> Self { 
        Self { signature: Signature::any(1, Volatility::Immutable) } 
    } 
}
impl ScalarUDFImpl for VectorToSparseUDF {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "vector_to_sparse" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> { 
        // Return a struct type representing SparseVector
        Ok(DataType::Struct(vec![
            Arc::new(arrow::datatypes::Field::new("indices", DataType::List(Arc::new(arrow::datatypes::Field::new("item", DataType::UInt32, true))), true)),
            Arc::new(arrow::datatypes::Field::new("values", DataType::List(Arc::new(arrow::datatypes::Field::new("item", DataType::Float32, true))), true)),
            Arc::new(arrow::datatypes::Field::new("dim", DataType::UInt32, true)),
        ].into()))
    }
    fn invoke_with_args(&self, args: datafusion::logical_expr::ScalarFunctionArgs) -> Result<ColumnarValue> {
        match &args.args[0] {
            ColumnarValue::Array(arr) => {
                let fixed_list = arr.as_any().downcast_ref::<FixedSizeListArray>()
                    .ok_or_else(|| datafusion::error::DataFusionError::Execution("Expected FixedSizeListArray".to_string()))?;
                
                let mut indices_builder = ListBuilder::new(arrow::array::UInt32Builder::new());
                let mut values_builder = ListBuilder::new(Float32Builder::new());
                let mut dim_builder = arrow::array::UInt32Builder::new();
                
                for i in 0..fixed_list.len() {
                    let value_array = fixed_list.value(i);
                    let dense = value_array.as_any().downcast_ref::<Float32Array>()
                        .ok_or_else(|| datafusion::error::DataFusionError::Execution("Expected Float32Array".to_string()))?;
                    
                    let sparse = dense_to_sparse(dense.values());
                    
                    // Build indices list
                    for &idx in &sparse.indices {
                        indices_builder.values().append_value(idx);
                    }
                    indices_builder.append(true);
                    
                    // Build values list
                    for &val in &sparse.values {
                        values_builder.values().append_value(val);
                    }
                    values_builder.append(true);
                    
                    // Add dimension
                    dim_builder.append_value(sparse.dim as u32);
                }
                
                let indices_array = Arc::new(indices_builder.finish()) as ArrayRef;
                let values_array = Arc::new(values_builder.finish()) as ArrayRef;
                let dim_array = Arc::new(dim_builder.finish()) as ArrayRef;
                
                let struct_array = arrow::array::StructArray::from(vec![
                    (Arc::new(arrow::datatypes::Field::new("indices", DataType::List(Arc::new(arrow::datatypes::Field::new("item", DataType::UInt32, true))), true)), indices_array),
                    (Arc::new(arrow::datatypes::Field::new("values", DataType::List(Arc::new(arrow::datatypes::Field::new("item", DataType::Float32, true))), true)), values_array),
                    (Arc::new(arrow::datatypes::Field::new("dim", DataType::UInt32, true)), dim_array),
                ]);
                
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            },
            _ => Ok(ColumnarValue::Scalar(ScalarValue::Null)),
        }
    }
}

#[derive(Debug)]
struct SparseToVectorUDF { signature: Signature }
impl_dyn_traits!(SparseToVectorUDF);
impl SparseToVectorUDF { 
    pub fn new() -> Self { 
        Self { signature: Signature::any(1, Volatility::Immutable) } 
    } 
}
impl ScalarUDFImpl for SparseToVectorUDF {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "sparse_to_vector" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> { 
        // Return variable-length list since we don't know dimension at compile time
        Ok(DataType::List(Arc::new(arrow::datatypes::Field::new("item", DataType::Float32, true))))
    }
    fn invoke_with_args(&self, args: datafusion::logical_expr::ScalarFunctionArgs) -> Result<ColumnarValue> {
        match &args.args[0] {
            ColumnarValue::Array(arr) => {
                let struct_array = arr.as_any().downcast_ref::<arrow::array::StructArray>()
                    .ok_or_else(|| datafusion::error::DataFusionError::Execution("Expected StructArray for sparse vector".to_string()))?;
                
                let indices_list = struct_array.column(0).as_any().downcast_ref::<ListArray>()
                    .ok_or_else(|| datafusion::error::DataFusionError::Execution("Expected ListArray for indices".to_string()))?;
                let values_list = struct_array.column(1).as_any().downcast_ref::<ListArray>()
                    .ok_or_else(|| datafusion::error::DataFusionError::Execution("Expected ListArray for values".to_string()))?;
                let dim_array = struct_array.column(2).as_any().downcast_ref::<arrow::array::UInt32Array>()
                    .ok_or_else(|| datafusion::error::DataFusionError::Execution("Expected UInt32Array for dim".to_string()))?;
                
                let mut builder = ListBuilder::new(Float32Builder::new());
                
                for i in 0..struct_array.len() {
                    let indices_arr = indices_list.value(i);
                    let indices = indices_arr.as_any().downcast_ref::<arrow::array::UInt32Array>()
                        .ok_or_else(|| datafusion::error::DataFusionError::Execution("Expected UInt32Array".to_string()))?;
                    
                    let values_arr = values_list.value(i);
                    let values = values_arr.as_any().downcast_ref::<Float32Array>()
                        .ok_or_else(|| datafusion::error::DataFusionError::Execution("Expected Float32Array".to_string()))?;
                    
                    let dim = dim_array.value(i) as usize;
                    
                    let sparse = SparseVector {
                        indices: indices.values().to_vec(),
                        values: values.values().to_vec(),
                        dim,
                    };
                    
                    let dense = sparse_to_dense(&sparse);
                    
                    for &val in &dense {
                        builder.values().append_value(val);
                    }
                    builder.append(true);
                }
                
                Ok(ColumnarValue::Array(Arc::new(builder.finish())))
            },
            _ => Ok(ColumnarValue::Scalar(ScalarValue::Null)),
        }
    }
}

#[derive(Debug)]
struct VectorToBinaryUDF { signature: Signature }
impl_dyn_traits!(VectorToBinaryUDF);
impl VectorToBinaryUDF { 
    pub fn new() -> Self { 
        Self { signature: Signature::any(1, Volatility::Immutable) } 
    } 
}
impl ScalarUDFImpl for VectorToBinaryUDF {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "vector_to_binary" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> { 
        // Use List instead of FixedSizeList since size is determined at runtime
        Ok(DataType::List(Arc::new(arrow::datatypes::Field::new("item", DataType::UInt8, true))))
    }
    fn invoke_with_args(&self, args: datafusion::logical_expr::ScalarFunctionArgs) -> Result<ColumnarValue> {
        // Delegate to binary_quantize implementation
        match &args.args[0] {
            ColumnarValue::Array(arr) => {
                let fsl = as_fixed_size_list_array(arr)?;
                let len = fsl.value_length();
                let packed_len = (len as usize + 7) / 8;
                let mut list_builder = ListBuilder::new(arrow::array::UInt8Builder::new());
                
                for i in 0..fsl.len() {
                    let value_array = fsl.value(i);
                    let v = value_array.as_any().downcast_ref::<Float32Array>().unwrap().values();
                    let mut packed = vec![0u8; packed_len];
                    for (j, &val) in v.iter().enumerate() {
                        if val >= 0.0 {  // Use >= 0.0 for binary quantization
                            packed[j / 8] |= 1 << (j % 8);
                        }
                    }
                    for b in packed { 
                        list_builder.values().append_value(b); 
                    }
                    list_builder.append(true);
                }
                
                Ok(ColumnarValue::Array(Arc::new(list_builder.finish())))
            },
            _ => Ok(ColumnarValue::Scalar(ScalarValue::Null)),
        }
    }
}

// --- Aggregates ---

#[derive(Debug)]
pub struct VectorSumUDF { signature: Signature }
impl_dyn_traits!(VectorSumUDF);
impl VectorSumUDF { pub fn new() -> Self { Self { signature: Signature::any(1, Volatility::Immutable) } } }
impl AggregateUDFImpl for VectorSumUDF {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "vector_sum" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> { 
        Ok(DataType::List(Arc::new(arrow::datatypes::Field::new("item", DataType::Float32, true))))
    }
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(VectorSumAccumulator::new()))
    }
    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<arrow::datatypes::Field>>> {
        Ok(vec![Arc::new(arrow::datatypes::Field::new("sum", DataType::List(Arc::new(arrow::datatypes::Field::new("item", DataType::Float32, true))), true))])
    }
}

#[derive(Debug)]
struct VectorSumAccumulator {
    sum: Option<Vec<f32>>,
}
impl VectorSumAccumulator { fn new() -> Self { Self { sum: None } } }
impl Accumulator for VectorSumAccumulator {
    fn update_batch(&mut self, values: &[arrow::array::ArrayRef]) -> Result<()> {
        let arr = &values[0];
        
        // Handle both FixedSizeList and List arrays
        if let Some(fsl) = arr.as_any().downcast_ref::<FixedSizeListArray>() {
            for i in 0..fsl.len() {
                let value_array = fsl.value(i);
                let row = value_array.as_any().downcast_ref::<Float32Array>().unwrap().values();
                
                // Dimension validation
                if let Some(ref mut s) = self.sum {
                    if s.len() != row.len() {
                        return Err(datafusion::error::DataFusionError::Execution(
                            format!("Cannot aggregate vectors of different dimensions: expected {}, got {}", s.len(), row.len())
                        ));
                    }
                    for (a, b) in s.iter_mut().zip(row.iter()) { *a += b; }
                } else {
                    self.sum = Some(row.to_vec());
                }
            }
        } else if let Some(list_arr) = arr.as_any().downcast_ref::<ListArray>() {
            for i in 0..list_arr.len() {
                let value_array = list_arr.value(i);
                let row = value_array.as_any().downcast_ref::<Float32Array>().unwrap().values();
                
                // Dimension validation
                if let Some(ref mut s) = self.sum {
                    if s.len() != row.len() {
                        return Err(datafusion::error::DataFusionError::Execution(
                            format!("Cannot aggregate vectors of different dimensions: expected {}, got {}", s.len(), row.len())
                        ));
                    }
                    for (a, b) in s.iter_mut().zip(row.iter()) { *a += b; }
                } else {
                    self.sum = Some(row.to_vec());
                }
            }
        } else {
            return Err(datafusion::error::DataFusionError::Execution(
                format!("Expected FixedSizeList or List array, got {:?}", arr.data_type())
            ));
        }
        Ok(())
    }
    fn evaluate(&mut self) -> Result<ScalarValue> {
        match &self.sum {
            Some(s) => {
                // Filter out NaN and Inf values
                let filtered: Vec<f32> = s.iter().map(|&x| {
                    if x.is_nan() || x.is_infinite() {
                        0.0
                    } else {
                        x
                    }
                }).collect();
                
                // Create a ListArray with a single element containing the sum vector
                let mut builder = ListBuilder::new(Float32Builder::new());
                builder.values().append_slice(&filtered);
                builder.append(true);
                Ok(ScalarValue::List(Arc::new(builder.finish())))
            },
            None => {
                // Return a properly typed NULL - create a list with 1 NULL element
                let mut builder = ListBuilder::new(Float32Builder::new());
                builder.append(false); // Append a NULL value
                Ok(ScalarValue::List(Arc::new(builder.finish())))
            },
        }
    }
    fn size(&self) -> usize { std::mem::size_of::<Self>() + self.sum.as_ref().map(|s| s.len() * 4).unwrap_or(0) }
    fn state(&mut self) -> Result<Vec<ScalarValue>> { Ok(vec![self.evaluate()?]) }
    fn merge_batch(&mut self, states: &[arrow::array::ArrayRef]) -> Result<()> {
        // Merge partial aggregates from different partitions
        // states[0] contains the sum vectors from other partitions
        let list_array = states[0].as_any().downcast_ref::<ListArray>()
            .ok_or_else(|| datafusion::error::DataFusionError::Execution("Expected ListArray in merge_batch".to_string()))?;
        
        for i in 0..list_array.len() {
            // Skip NULL or empty lists (empty partitions)
            if list_array.is_null(i) {
                continue;
            }
            let partial_sum_array = list_array.value(i);
            if partial_sum_array.len() == 0 {
                continue;
            }
            
            let partial_sum = partial_sum_array.as_any().downcast_ref::<Float32Array>()
                .ok_or_else(|| datafusion::error::DataFusionError::Execution("Expected Float32Array in partial sum".to_string()))?;
            
            if let Some(ref mut sum) = self.sum {
                // Add partial sum to existing sum
                for (a, b) in sum.iter_mut().zip(partial_sum.values()) {
                    *a += b;
                }
            } else {
                // Initialize sum with first partial sum
                self.sum = Some(partial_sum.values().to_vec());
            }
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct VectorAvgUDF { signature: Signature }
impl_dyn_traits!(VectorAvgUDF);
impl VectorAvgUDF { pub fn new() -> Self { Self { signature: Signature::any(1, Volatility::Immutable) } } }
impl AggregateUDFImpl for VectorAvgUDF {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "vector_avg" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> { 
        Ok(DataType::List(Arc::new(arrow::datatypes::Field::new("item", DataType::Float32, true))))
    }
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(VectorAvgAccumulator::new()))
    }
    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<arrow::datatypes::Field>>> {
        Ok(vec![
            Arc::new(arrow::datatypes::Field::new("sum", DataType::List(Arc::new(arrow::datatypes::Field::new("item", DataType::Float32, true))), true)),
            Arc::new(arrow::datatypes::Field::new("count", DataType::UInt64, true)),
        ])
    }
}

#[derive(Debug)]
struct VectorAvgAccumulator {
    sum: Option<Vec<f32>>,
    count: u64,
}
impl VectorAvgAccumulator { fn new() -> Self { Self { sum: None, count: 0 } } }
impl Accumulator for VectorAvgAccumulator {
    fn update_batch(&mut self, values: &[arrow::array::ArrayRef]) -> Result<()> {
        let arr = &values[0];
        
        // Handle both FixedSizeList and List arrays
        if let Some(fsl) = arr.as_any().downcast_ref::<FixedSizeListArray>() {
            self.count += fsl.len() as u64;
            for i in 0..fsl.len() {
                let value_array = fsl.value(i);
                let row = value_array.as_any().downcast_ref::<Float32Array>().unwrap().values();
                
                // Dimension validation
                if let Some(ref mut s) = self.sum {
                    if s.len() != row.len() {
                        return Err(datafusion::error::DataFusionError::Execution(
                            format!("Cannot aggregate vectors of different dimensions: expected {}, got {}", s.len(), row.len())
                        ));
                    }
                    for (a, b) in s.iter_mut().zip(row.iter()) { *a += b; }
                } else {
                    self.sum = Some(row.to_vec());
                }
            }
        } else if let Some(list_arr) = arr.as_any().downcast_ref::<ListArray>() {
            self.count += list_arr.len() as u64;
            for i in 0..list_arr.len() {
                let value_array = list_arr.value(i);
                let row = value_array.as_any().downcast_ref::<Float32Array>().unwrap().values();
                
                // Dimension validation
                if let Some(ref mut s) = self.sum {
                    if s.len() != row.len() {
                        return Err(datafusion::error::DataFusionError::Execution(
                            format!("Cannot aggregate vectors of different dimensions: expected {}, got {}", s.len(), row.len())
                        ));
                    }
                    for (a, b) in s.iter_mut().zip(row.iter()) { *a += b; }
                } else {
                    self.sum = Some(row.to_vec());
                }
            }
        } else {
            return Err(datafusion::error::DataFusionError::Execution(
                format!("Expected FixedSizeList or List array, got {:?}", arr.data_type())
            ));
        }
        Ok(())
    }
    fn evaluate(&mut self) -> Result<ScalarValue> {
        match &self.sum {
            Some(s) => {
                let avg: Vec<f32> = s.iter().map(|&x| {
                    let val = x / self.count as f32;
                    // Filter out NaN and Inf values
                    if val.is_nan() || val.is_infinite() {
                        0.0
                    } else {
                        val
                    }
                }).collect();
                
                // Create a ListArray with a single element containing the average vector
                let mut builder = ListBuilder::new(Float32Builder::new());
                builder.values().append_slice(&avg);
                builder.append(true);
                Ok(ScalarValue::List(Arc::new(builder.finish())))
            },
            None => {
                // Return a properly typed NULL - create a list with 1 NULL element
                let mut builder = ListBuilder::new(Float32Builder::new());
                builder.append(false); // Append a NULL value
                Ok(ScalarValue::List(Arc::new(builder.finish())))
            },
        }
    }
    fn size(&self) -> usize { std::mem::size_of::<Self>() + self.sum.as_ref().map(|s| s.len() * 4).unwrap_or(0) }
    fn state(&mut self) -> Result<Vec<ScalarValue>> { Ok(vec![
        match &self.sum {
            Some(s) => {
                let mut builder = ListBuilder::new(Float32Builder::new());
                builder.values().append_slice(s);
                builder.append(true);
                ScalarValue::List(Arc::new(builder.finish()))
            },
            None => {
                // Return a properly typed NULL
                let mut builder = ListBuilder::new(Float32Builder::new());
                builder.append(false);
                ScalarValue::List(Arc::new(builder.finish()))
            },
        },
        ScalarValue::UInt64(Some(self.count))
    ]) }
    fn merge_batch(&mut self, states: &[arrow::array::ArrayRef]) -> Result<()> {
        // Merge partial aggregates from different partitions
        // states[0] contains the sum vectors, states[1] contains the counts
        let sum_array = states[0].as_any().downcast_ref::<ListArray>()
            .ok_or_else(|| datafusion::error::DataFusionError::Execution("Expected ListArray for sum in merge_batch".to_string()))?;
        let count_array = states[1].as_any().downcast_ref::<UInt64Array>()
            .ok_or_else(|| datafusion::error::DataFusionError::Execution("Expected UInt64Array for count in merge_batch".to_string()))?;
        
        for i in 0..sum_array.len() {
            // Skip NULL or empty lists (empty partitions)
            if sum_array.is_null(i) {
                continue;
            }
            let partial_sum_array = sum_array.value(i);
            if partial_sum_array.len() == 0 {
                continue;
            }
            
            let partial_sum = partial_sum_array.as_any().downcast_ref::<Float32Array>()
                .ok_or_else(|| datafusion::error::DataFusionError::Execution("Expected Float32Array in partial sum".to_string()))?;
            let partial_count = count_array.value(i);
            
            // Merge count
            self.count += partial_count;
            
            // Merge sum
            if let Some(ref mut sum) = self.sum {
                // Add partial sum to existing sum
                for (a, b) in sum.iter_mut().zip(partial_sum.values()) {
                    *a += b;
                }
            } else {
                // Initialize sum with first partial sum
                self.sum = Some(partial_sum.values().to_vec());
            }
        }
        Ok(())
    }
}


#[cfg(test)]
mod sparse_distance_property_tests {
    use super::*;
    use proptest::prelude::*;
    use crate::core::index::distance::{l2_distance, cosine_distance, dot_product};

    // Feature: pgvector-sql-support, Property 12: Sparse Distance Equivalence
    // **Validates: Requirements 5.3**
    //
    // Property: For any sparse vector, computing distance using sparse-aware algorithms
    // should produce the same result as converting to dense and computing distance on
    // the dense representation (within floating-point precision).
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_sparse_l2_distance_equivalence(
            dim in 10usize..200,
            sparsity in 0.1f32..0.9f32,
        ) {
            // Generate a random sparse vector
            let num_nonzero = ((dim as f32) * (1.0 - sparsity)) as usize;
            let num_nonzero = num_nonzero.max(1).min(dim);
            
            // Generate unique indices
            let mut indices: Vec<u32> = (0..num_nonzero)
                .map(|i| {
                    let step = dim / num_nonzero;
                    (i * step) as u32
                })
                .collect();
            
            // Generate random values
            let values: Vec<f32> = (0..num_nonzero)
                .map(|i| ((i as f32 + 1.0) * 0.1) % 1.0)
                .collect();
            
            let sparse_a = SparseVector {
                indices: indices.clone(),
                values: values.clone(),
                dim,
            };
            
            // Generate another sparse vector
            let indices_b: Vec<u32> = (0..num_nonzero)
                .map(|i| {
                    let step = dim / num_nonzero;
                    (((i * step) + 1) as u32).min((dim - 1) as u32)
                })
                .collect();
            
            let values_b: Vec<f32> = (0..num_nonzero)
                .map(|i| ((i as f32 + 2.0) * 0.15) % 1.0)
                .collect();
            
            let sparse_b = SparseVector {
                indices: indices_b,
                values: values_b,
                dim,
            };
            
            // Compute sparse distance
            let sparse_dist = sparse_l2_distance(&sparse_a, &sparse_b);
            
            // Convert to dense and compute distance
            let dense_a = sparse_to_dense(&sparse_a);
            let dense_b = sparse_to_dense(&sparse_b);
            let dense_dist = l2_distance(&dense_a, &dense_b);
            
            // Verify equivalence within floating-point precision
            let diff = (sparse_dist - dense_dist).abs();
            let tolerance = dense_dist.abs() * 1e-5 + 1e-6;
            prop_assert!(diff <= tolerance,
                "L2 distance mismatch: sparse={}, dense={}, diff={}",
                sparse_dist, dense_dist, diff);
        }

        #[test]
        fn test_sparse_cosine_distance_equivalence(
            dim in 10usize..200,
            sparsity in 0.1f32..0.9f32,
        ) {
            // Generate a random sparse vector
            let num_nonzero = ((dim as f32) * (1.0 - sparsity)) as usize;
            let num_nonzero = num_nonzero.max(1).min(dim);
            
            // Generate unique indices
            let indices_a: Vec<u32> = (0..num_nonzero)
                .map(|i| {
                    let step = dim / num_nonzero;
                    (i * step) as u32
                })
                .collect();
            
            // Generate random non-zero values
            let values_a: Vec<f32> = (0..num_nonzero)
                .map(|i| ((i as f32 + 1.0) * 0.1) % 1.0 + 0.1)
                .collect();
            
            let sparse_a = SparseVector {
                indices: indices_a,
                values: values_a,
                dim,
            };
            
            // Generate another sparse vector
            let indices_b: Vec<u32> = (0..num_nonzero)
                .map(|i| {
                    let step = dim / num_nonzero;
                    (((i * step) + 1) as u32).min((dim - 1) as u32)
                })
                .collect();
            
            let values_b: Vec<f32> = (0..num_nonzero)
                .map(|i| ((i as f32 + 2.0) * 0.15) % 1.0 + 0.1)
                .collect();
            
            let sparse_b = SparseVector {
                indices: indices_b,
                values: values_b,
                dim,
            };
            
            // Compute sparse distance
            let sparse_dist = sparse_cosine_distance(&sparse_a, &sparse_b);
            
            // Convert to dense and compute distance
            let dense_a = sparse_to_dense(&sparse_a);
            let dense_b = sparse_to_dense(&sparse_b);
            let dense_dist = cosine_distance(&dense_a, &dense_b);
            
            // Verify equivalence within floating-point precision
            let diff = (sparse_dist - dense_dist).abs();
            let tolerance = dense_dist.abs() * 1e-4 + 1e-5;
            prop_assert!(diff <= tolerance,
                "Cosine distance mismatch: sparse={}, dense={}, diff={}",
                sparse_dist, dense_dist, diff);
        }

        #[test]
        fn test_sparse_inner_product_distance_equivalence(
            dim in 10usize..200,
            sparsity in 0.1f32..0.9f32,
        ) {
            // Generate a random sparse vector
            let num_nonzero = ((dim as f32) * (1.0 - sparsity)) as usize;
            let num_nonzero = num_nonzero.max(1).min(dim);
            
            // Generate unique indices
            let indices_a: Vec<u32> = (0..num_nonzero)
                .map(|i| {
                    let step = dim / num_nonzero;
                    (i * step) as u32
                })
                .collect();
            
            // Generate random values
            let values_a: Vec<f32> = (0..num_nonzero)
                .map(|i| ((i as f32 + 1.0) * 0.1) % 1.0)
                .collect();
            
            let sparse_a = SparseVector {
                indices: indices_a,
                values: values_a,
                dim,
            };
            
            // Generate another sparse vector
            let indices_b: Vec<u32> = (0..num_nonzero)
                .map(|i| {
                    let step = dim / num_nonzero;
                    (((i * step) + 1) as u32).min((dim - 1) as u32)
                })
                .collect();
            
            let values_b: Vec<f32> = (0..num_nonzero)
                .map(|i| ((i as f32 + 2.0) * 0.15) % 1.0)
                .collect();
            
            let sparse_b = SparseVector {
                indices: indices_b,
                values: values_b,
                dim,
            };
            
            // Compute sparse distance (negative dot product)
            let sparse_dist = sparse_inner_product_distance(&sparse_a, &sparse_b);
            
            // Convert to dense and compute distance
            let dense_a = sparse_to_dense(&sparse_a);
            let dense_b = sparse_to_dense(&sparse_b);
            let dense_dist = -dot_product(&dense_a, &dense_b);
            
            // Verify equivalence within floating-point precision
            let diff = (sparse_dist - dense_dist).abs();
            let tolerance = dense_dist.abs() * 1e-5 + 1e-6;
            prop_assert!(diff <= tolerance,
                "Inner product distance mismatch: sparse={}, dense={}, diff={}",
                sparse_dist, dense_dist, diff);
        }

        #[test]
        fn test_dense_to_sparse_to_dense_round_trip(
            dim in 1usize..100,
            sparsity in 0.1f32..0.9f32,
        ) {
            // Generate a random dense vector with some zeros
            let num_nonzero = ((dim as f32) * (1.0 - sparsity)) as usize;
            let num_nonzero = num_nonzero.max(1).min(dim);
            
            let mut dense = vec![0.0; dim];
            for i in 0..num_nonzero {
                let idx = (i * (dim / num_nonzero)).min(dim - 1);
                dense[idx] = ((i as f32 + 1.0) * 0.1) % 1.0;
            }
            
            // Convert to sparse and back to dense
            let sparse = dense_to_sparse(&dense);
            let dense_recovered = sparse_to_dense(&sparse);
            
            // Verify round-trip equivalence
            prop_assert_eq!(dense.len(), dense_recovered.len(),
                "Dimension mismatch after round-trip");
            
            for (i, (original, recovered)) in dense.iter().zip(dense_recovered.iter()).enumerate() {
                let diff = (original - recovered).abs();
                prop_assert!(diff < 1e-6,
                    "Value mismatch at index {}: original={}, recovered={}, diff={}",
                    i, original, recovered, diff);
            }
        }

        #[test]
        fn test_sparse_distance_with_no_overlap(
            dim in 10usize..100,
        ) {
            // Create two sparse vectors with no overlapping indices
            let sparse_a = SparseVector {
                indices: vec![0, 2, 4],
                values: vec![1.0, 2.0, 3.0],
                dim,
            };
            
            let sparse_b = SparseVector {
                indices: vec![1, 3, 5],
                values: vec![1.0, 2.0, 3.0],
                dim,
            };
            
            // Compute sparse distances
            let sparse_l2 = sparse_l2_distance(&sparse_a, &sparse_b);
            let sparse_cosine = sparse_cosine_distance(&sparse_a, &sparse_b);
            let sparse_ip = sparse_inner_product_distance(&sparse_a, &sparse_b);
            
            // Convert to dense and compute distances
            let dense_a = sparse_to_dense(&sparse_a);
            let dense_b = sparse_to_dense(&sparse_b);
            let dense_l2 = l2_distance(&dense_a, &dense_b);
            let dense_cosine = cosine_distance(&dense_a, &dense_b);
            let dense_ip = -dot_product(&dense_a, &dense_b);
            
            // Verify equivalence
            prop_assert!((sparse_l2 - dense_l2).abs() < 1e-5,
                "L2 distance mismatch with no overlap");
            prop_assert!((sparse_cosine - dense_cosine).abs() < 1e-5,
                "Cosine distance mismatch with no overlap");
            prop_assert!((sparse_ip - dense_ip).abs() < 1e-5,
                "Inner product distance mismatch with no overlap");
        }

        #[test]
        fn test_sparse_distance_with_complete_overlap(
            dim in 10usize..100,
            num_nonzero in 1usize..20,
        ) {
            let actual_nonzero = num_nonzero.min(dim);
            
            // Create two sparse vectors with complete overlap
            let indices: Vec<u32> = (0..actual_nonzero)
                .map(|i| {
                    let step = dim / actual_nonzero;
                    (i * step) as u32
                })
                .collect();
            
            let values_a: Vec<f32> = (0..actual_nonzero)
                .map(|i| ((i as f32 + 1.0) * 0.1) % 1.0)
                .collect();
            
            let values_b: Vec<f32> = (0..actual_nonzero)
                .map(|i| ((i as f32 + 2.0) * 0.15) % 1.0)
                .collect();
            
            let sparse_a = SparseVector {
                indices: indices.clone(),
                values: values_a,
                dim,
            };
            
            let sparse_b = SparseVector {
                indices: indices.clone(),
                values: values_b,
                dim,
            };
            
            // Compute sparse distances
            let sparse_l2 = sparse_l2_distance(&sparse_a, &sparse_b);
            let sparse_cosine = sparse_cosine_distance(&sparse_a, &sparse_b);
            let sparse_ip = sparse_inner_product_distance(&sparse_a, &sparse_b);
            
            // Convert to dense and compute distances
            let dense_a = sparse_to_dense(&sparse_a);
            let dense_b = sparse_to_dense(&sparse_b);
            let dense_l2 = l2_distance(&dense_a, &dense_b);
            let dense_cosine = cosine_distance(&dense_a, &dense_b);
            let dense_ip = -dot_product(&dense_a, &dense_b);
            
            // Verify equivalence
            let l2_diff = (sparse_l2 - dense_l2).abs();
            let l2_tolerance = dense_l2.abs() * 1e-5 + 1e-6;
            prop_assert!(l2_diff <= l2_tolerance,
                "L2 distance mismatch with complete overlap: sparse={}, dense={}, diff={}",
                sparse_l2, dense_l2, l2_diff);
            
            let cosine_diff = (sparse_cosine - dense_cosine).abs();
            let cosine_tolerance = dense_cosine.abs() * 1e-4 + 1e-5;
            prop_assert!(cosine_diff <= cosine_tolerance,
                "Cosine distance mismatch with complete overlap: sparse={}, dense={}, diff={}",
                sparse_cosine, dense_cosine, cosine_diff);
            
            let ip_diff = (sparse_ip - dense_ip).abs();
            let ip_tolerance = dense_ip.abs() * 1e-5 + 1e-6;
            prop_assert!(ip_diff <= ip_tolerance,
                "Inner product distance mismatch with complete overlap: sparse={}, dense={}, diff={}",
                sparse_ip, dense_ip, ip_diff);
        }

        #[test]
        fn test_sparse_distance_with_high_sparsity(
            dim in 100usize..1000,
        ) {
            // Very sparse vectors (< 1% non-zero)
            let num_nonzero = (dim / 100).max(1);
            
            let indices_a: Vec<u32> = (0..num_nonzero)
                .map(|i| (i * 100) as u32)
                .collect();
            
            let values_a: Vec<f32> = (0..num_nonzero)
                .map(|i| (i as f32 + 1.0) * 0.1)
                .collect();
            
            let sparse_a = SparseVector {
                indices: indices_a,
                values: values_a,
                dim,
            };
            
            let indices_b: Vec<u32> = (0..num_nonzero)
                .map(|i| (i * 100 + 50) as u32)
                .collect();
            
            let values_b: Vec<f32> = (0..num_nonzero)
                .map(|i| (i as f32 + 2.0) * 0.15)
                .collect();
            
            let sparse_b = SparseVector {
                indices: indices_b,
                values: values_b,
                dim,
            };
            
            // Compute sparse distances
            let sparse_l2 = sparse_l2_distance(&sparse_a, &sparse_b);
            
            // Convert to dense and compute distance
            let dense_a = sparse_to_dense(&sparse_a);
            let dense_b = sparse_to_dense(&sparse_b);
            let dense_l2 = l2_distance(&dense_a, &dense_b);
            
            // Verify equivalence
            let diff = (sparse_l2 - dense_l2).abs();
            let tolerance = dense_l2.abs() * 1e-5 + 1e-6;
            prop_assert!(diff <= tolerance,
                "L2 distance mismatch with high sparsity: sparse={}, dense={}, diff={}",
                sparse_l2, dense_l2, diff);
        }
    }
}


/// Get the dimensionality of a sparse vector
pub fn sparsevec_dims(sparse: &SparseVector) -> usize {
    sparse.dim
}

/// Get the number of non-zero elements in a sparse vector
pub fn sparsevec_nnz(sparse: &SparseVector) -> usize {
    sparse.indices.len()
}

#[cfg(test)]
mod sparse_utility_tests {
    use super::*;

    #[test]
    fn test_sparsevec_dims() {
        let sparse = SparseVector {
            indices: vec![1, 10, 100],
            values: vec![0.5, 0.3, 0.8],
            dim: 1000,
        };
        
        assert_eq!(sparsevec_dims(&sparse), 1000);
    }

    #[test]
    fn test_sparsevec_nnz() {
        let sparse = SparseVector {
            indices: vec![1, 10, 100],
            values: vec![0.5, 0.3, 0.8],
            dim: 1000,
        };
        
        assert_eq!(sparsevec_nnz(&sparse), 3);
    }

    #[test]
    fn test_sparsevec_nnz_empty() {
        let sparse = SparseVector {
            indices: vec![],
            values: vec![],
            dim: 1000,
        };
        
        assert_eq!(sparsevec_nnz(&sparse), 0);
    }

    #[test]
    fn test_sparsevec_dims_various_sizes() {
        for dim in [10, 100, 1000, 10000] {
            let sparse = SparseVector {
                indices: vec![0, 1, 2],
                values: vec![1.0, 2.0, 3.0],
                dim,
            };
            
            assert_eq!(sparsevec_dims(&sparse), dim);
        }
    }

    #[test]
    fn test_sparsevec_nnz_various_sparsity() {
        for nnz in [0, 1, 10, 100, 1000] {
            let indices: Vec<u32> = (0..nnz).map(|i| i as u32).collect();
            let values: Vec<f32> = (0..nnz).map(|i| i as f32).collect();
            
            let sparse = SparseVector {
                indices,
                values,
                dim: 10000,
            };
            
            assert_eq!(sparsevec_nnz(&sparse), nnz);
        }
    }
}

#[cfg(test)]
mod aggregation_property_tests {
    use super::*;
    use proptest::prelude::*;
    use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array};
    use arrow::datatypes::{DataType, Field};
    use std::sync::Arc;

    // Feature: pgvector-sql-support, Property 14: Vector Aggregation Correctness
    // **Validates: Requirements 6.1, 6.2**
    //
    // For any set of vectors with the same dimensionality, vector_sum should return 
    // the element-wise sum and vector_avg should return the element-wise average.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        
        #[test]
        fn test_vector_sum_correctness(
            dim in 1usize..128,
            num_vectors in 1usize..50,
            seed in any::<u64>()
        ) {
            use rand::SeedableRng;
            use rand::Rng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            
            // Generate random vectors
            let mut vectors: Vec<Vec<f32>> = Vec::new();
            for _ in 0..num_vectors {
                let vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-100.0..100.0)).collect();
                vectors.push(vec);
            }
            
            // Compute expected sum manually
            let mut expected_sum = vec![0.0f32; dim];
            for vec in &vectors {
                for (i, &val) in vec.iter().enumerate() {
                    expected_sum[i] += val;
                }
            }
            
            // Create FixedSizeListArray from vectors
            let mut builder = arrow::array::FixedSizeListBuilder::new(
                Float32Builder::new(),
                dim as i32
            );
            for vec in &vectors {
                let values_builder = builder.values();
                for &val in vec {
                    values_builder.append_value(val);
                }
                builder.append(true);
            }
            let array = builder.finish();
            let array_ref: ArrayRef = Arc::new(array);
            
            // Test VectorSumAccumulator
            let mut accumulator = VectorSumAccumulator::new();
            accumulator.update_batch(&[array_ref]).unwrap();
            let result = accumulator.evaluate().unwrap();
            
            // Extract result vector
            if let ScalarValue::List(list_arc) = result {
                let list_array = list_arc.as_ref();
                assert_eq!(list_array.len(), 1);
                let result_array = list_array.value(0);
                let result_vec = result_array.as_any().downcast_ref::<Float32Array>().unwrap();
                
                // Compare with expected sum (with floating point tolerance)
                assert_eq!(result_vec.len(), dim);
                for i in 0..dim {
                    let diff = (result_vec.value(i) - expected_sum[i]).abs();
                    assert!(diff < 0.001 || diff / expected_sum[i].abs() < 0.0001, 
                        "Sum mismatch at index {}: expected {}, got {}", i, expected_sum[i], result_vec.value(i));
                }
            } else {
                panic!("Expected List result, got {:?}", result);
            }
        }
        
        #[test]
        fn test_vector_avg_correctness(
            dim in 1usize..128,
            num_vectors in 1usize..50,
            seed in any::<u64>()
        ) {
            use rand::SeedableRng;
            use rand::Rng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            
            // Generate random vectors
            let mut vectors: Vec<Vec<f32>> = Vec::new();
            for _ in 0..num_vectors {
                let vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-100.0..100.0)).collect();
                vectors.push(vec);
            }
            
            // Compute expected average manually
            let mut expected_sum = vec![0.0f32; dim];
            for vec in &vectors {
                for (i, &val) in vec.iter().enumerate() {
                    expected_sum[i] += val;
                }
            }
            let expected_avg: Vec<f32> = expected_sum.iter().map(|&s| s / num_vectors as f32).collect();
            
            // Create FixedSizeListArray from vectors
            let mut builder = arrow::array::FixedSizeListBuilder::new(
                Float32Builder::new(),
                dim as i32
            );
            for vec in &vectors {
                let values_builder = builder.values();
                for &val in vec {
                    values_builder.append_value(val);
                }
                builder.append(true);
            }
            let array = builder.finish();
            let array_ref: ArrayRef = Arc::new(array);
            
            // Test VectorAvgAccumulator
            let mut accumulator = VectorAvgAccumulator::new();
            accumulator.update_batch(&[array_ref]).unwrap();
            let result = accumulator.evaluate().unwrap();
            
            // Extract result vector
            if let ScalarValue::List(list_arc) = result {
                let list_array = list_arc.as_ref();
                assert_eq!(list_array.len(), 1);
                let result_array = list_array.value(0);
                let result_vec = result_array.as_any().downcast_ref::<Float32Array>().unwrap();
                
                // Compare with expected average (with floating point tolerance)
                assert_eq!(result_vec.len(), dim);
                for i in 0..dim {
                    let diff = (result_vec.value(i) - expected_avg[i]).abs();
                    assert!(diff < 0.001 || diff / expected_avg[i].abs().max(0.001) < 0.0001, 
                        "Average mismatch at index {}: expected {}, got {}", i, expected_avg[i], result_vec.value(i));
                }
            } else {
                panic!("Expected List result, got {:?}", result);
            }
        }
        
        #[test]
        fn test_vector_sum_merge_batch(
            dim in 1usize..64,
            num_partitions in 2usize..10,
            vectors_per_partition in 1usize..20,
            seed in any::<u64>()
        ) {
            use rand::SeedableRng;
            use rand::Rng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            
            // Generate random vectors for each partition
            let mut all_vectors: Vec<Vec<f32>> = Vec::new();
            let mut partition_sums: Vec<Vec<f32>> = Vec::new();
            
            for _ in 0..num_partitions {
                let mut partition_sum = vec![0.0f32; dim];
                for _ in 0..vectors_per_partition {
                    let vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-100.0..100.0)).collect();
                    for (i, &val) in vec.iter().enumerate() {
                        partition_sum[i] += val;
                    }
                    all_vectors.push(vec);
                }
                partition_sums.push(partition_sum);
            }
            
            // Compute expected total sum
            let mut expected_sum = vec![0.0f32; dim];
            for vec in &all_vectors {
                for (i, &val) in vec.iter().enumerate() {
                    expected_sum[i] += val;
                }
            }
            
            // Create partial sum states (as if from different partitions)
            let mut list_builder = ListBuilder::new(Float32Builder::new());
            for partition_sum in &partition_sums {
                list_builder.values().append_slice(partition_sum);
                list_builder.append(true);
            }
            let states_array = list_builder.finish();
            let states_ref: ArrayRef = Arc::new(states_array);
            
            // Test merge_batch
            let mut accumulator = VectorSumAccumulator::new();
            accumulator.merge_batch(&[states_ref]).unwrap();
            let result = accumulator.evaluate().unwrap();
            
            // Extract result vector
            if let ScalarValue::List(list_arc) = result {
                let list_array = list_arc.as_ref();
                assert_eq!(list_array.len(), 1);
                let result_array = list_array.value(0);
                let result_vec = result_array.as_any().downcast_ref::<Float32Array>().unwrap();
                
                // Compare with expected sum
                assert_eq!(result_vec.len(), dim);
                for i in 0..dim {
                    let diff = (result_vec.value(i) - expected_sum[i]).abs();
                    assert!(diff < 0.001 || diff / expected_sum[i].abs().max(0.001) < 0.0001, 
                        "Merged sum mismatch at index {}: expected {}, got {}", i, expected_sum[i], result_vec.value(i));
                }
            } else {
                panic!("Expected List result, got {:?}", result);
            }
        }
        
        #[test]
        fn test_vector_avg_merge_batch(
            dim in 1usize..64,
            num_partitions in 2usize..10,
            vectors_per_partition in 1usize..20,
            seed in any::<u64>()
        ) {
            use rand::SeedableRng;
            use rand::Rng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            
            // Generate random vectors for each partition
            let mut all_vectors: Vec<Vec<f32>> = Vec::new();
            let mut partition_states: Vec<(Vec<f32>, u64)> = Vec::new();
            
            for _ in 0..num_partitions {
                let mut partition_sum = vec![0.0f32; dim];
                let count = vectors_per_partition as u64;
                for _ in 0..vectors_per_partition {
                    let vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-100.0..100.0)).collect();
                    for (i, &val) in vec.iter().enumerate() {
                        partition_sum[i] += val;
                    }
                    all_vectors.push(vec);
                }
                partition_states.push((partition_sum, count));
            }
            
            // Compute expected average
            let mut expected_sum = vec![0.0f32; dim];
            let total_count = all_vectors.len() as f32;
            for vec in &all_vectors {
                for (i, &val) in vec.iter().enumerate() {
                    expected_sum[i] += val;
                }
            }
            let expected_avg: Vec<f32> = expected_sum.iter().map(|&s| s / total_count).collect();
            
            // Create partial states (sum and count arrays)
            let mut sum_builder = ListBuilder::new(Float32Builder::new());
            let mut count_builder = arrow::array::UInt64Builder::new();
            
            for (partition_sum, count) in &partition_states {
                sum_builder.values().append_slice(partition_sum);
                sum_builder.append(true);
                count_builder.append_value(*count);
            }
            
            let sum_array = sum_builder.finish();
            let count_array = count_builder.finish();
            let sum_ref: ArrayRef = Arc::new(sum_array);
            let count_ref: ArrayRef = Arc::new(count_array);
            
            // Test merge_batch
            let mut accumulator = VectorAvgAccumulator::new();
            accumulator.merge_batch(&[sum_ref, count_ref]).unwrap();
            let result = accumulator.evaluate().unwrap();
            
            // Extract result vector
            if let ScalarValue::List(list_arc) = result {
                let list_array = list_arc.as_ref();
                assert_eq!(list_array.len(), 1);
                let result_array = list_array.value(0);
                let result_vec = result_array.as_any().downcast_ref::<Float32Array>().unwrap();
                
                // Compare with expected average
                assert_eq!(result_vec.len(), dim);
                for i in 0..dim {
                    let diff = (result_vec.value(i) - expected_avg[i]).abs();
                    assert!(diff < 0.001 || diff / expected_avg[i].abs().max(0.001) < 0.0001, 
                        "Merged average mismatch at index {}: expected {}, got {}", i, expected_avg[i], result_vec.value(i));
                }
            } else {
                panic!("Expected List result, got {:?}", result);
            }
        }
        
        #[test]
        fn test_vector_sum_empty_input(dim in 1usize..128) {
            // Create empty FixedSizeListArray
            let mut builder = arrow::array::FixedSizeListBuilder::new(
                Float32Builder::new(),
                dim as i32
            );
            let array = builder.finish();
            let array_ref: ArrayRef = Arc::new(array);
            
            // Test VectorSumAccumulator with empty input
            let mut accumulator = VectorSumAccumulator::new();
            accumulator.update_batch(&[array_ref]).unwrap();
            let result = accumulator.evaluate().unwrap();
            
            // Should return a List with a NULL element for empty input
            // (not ScalarValue::Null, but a List containing NULL)
            match result {
                ScalarValue::List(arr) => {
                    assert_eq!(arr.len(), 1, "Should have one element");
                    assert!(arr.is_null(0), "Element should be NULL");
                }
                _ => panic!("Expected List, got {:?}", result),
            }
        }
        
        #[test]
        fn test_vector_avg_empty_input(dim in 1usize..128) {
            // Create empty FixedSizeListArray
            let mut builder = arrow::array::FixedSizeListBuilder::new(
                Float32Builder::new(),
                dim as i32
            );
            let array = builder.finish();
            let array_ref: ArrayRef = Arc::new(array);
            
            // Test VectorAvgAccumulator with empty input
            let mut accumulator = VectorAvgAccumulator::new();
            accumulator.update_batch(&[array_ref]).unwrap();
            let result = accumulator.evaluate().unwrap();
            
            // Should return a List with a NULL element for empty input
            // (not ScalarValue::Null, but a List containing NULL)
            match result {
                ScalarValue::List(arr) => {
                    assert_eq!(arr.len(), 1, "Should have one element");
                    assert!(arr.is_null(0), "Element should be NULL");
                }
                _ => panic!("Expected List, got {:?}", result),
            }
        }
        
        // Feature: pgvector-sql-support, Property 16: Vector Aggregation Dimension Validation
        // **Validates: Requirements 6.5**
        //
        // For any vector aggregation receiving vectors of different dimensions, the system 
        // should return an error indicating dimension mismatch.
        #[test]
        fn test_vector_sum_dimension_mismatch(
            dim1 in 2usize..64,
            dim2 in 2usize..64,
            seed in any::<u64>()
        ) {
            prop_assume!(dim1 != dim2); // Only test when dimensions are different
            
            use rand::SeedableRng;
            use rand::Rng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            
            // Create vectors with different dimensions
            let vec1: Vec<f32> = (0..dim1).map(|_| rng.gen_range(-100.0..100.0)).collect();
            let vec2: Vec<f32> = (0..dim2).map(|_| rng.gen_range(-100.0..100.0)).collect();
            
            // Create FixedSizeListArray with first vector
            let mut builder1 = arrow::array::FixedSizeListBuilder::new(
                Float32Builder::new(),
                dim1 as i32
            );
            let values_builder = builder1.values();
            for &val in &vec1 {
                values_builder.append_value(val);
            }
            builder1.append(true);
            let array1 = builder1.finish();
            let array_ref1: ArrayRef = Arc::new(array1);
            
            // Test VectorSumAccumulator with first vector
            let mut accumulator = VectorSumAccumulator::new();
            accumulator.update_batch(&[array_ref1]).unwrap();
            
            // Create FixedSizeListArray with second vector (different dimension)
            let mut builder2 = arrow::array::FixedSizeListBuilder::new(
                Float32Builder::new(),
                dim2 as i32
            );
            let values_builder = builder2.values();
            for &val in &vec2 {
                values_builder.append_value(val);
            }
            builder2.append(true);
            let array2 = builder2.finish();
            let array_ref2: ArrayRef = Arc::new(array2);
            
            // Attempt to update with different dimension should fail
            let result = accumulator.update_batch(&[array_ref2]);
            assert!(result.is_err(), "Expected error for dimension mismatch");
            
            if let Err(e) = result {
                let error_msg = format!("{}", e);
                assert!(error_msg.contains("different dimensions") || error_msg.contains("dimension"), 
                    "Error message should mention dimensions: {}", error_msg);
            }
        }
        
        #[test]
        fn test_vector_avg_dimension_mismatch(
            dim1 in 2usize..64,
            dim2 in 2usize..64,
            seed in any::<u64>()
        ) {
            prop_assume!(dim1 != dim2); // Only test when dimensions are different
            
            use rand::SeedableRng;
            use rand::Rng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            
            // Create vectors with different dimensions
            let vec1: Vec<f32> = (0..dim1).map(|_| rng.gen_range(-100.0..100.0)).collect();
            let vec2: Vec<f32> = (0..dim2).map(|_| rng.gen_range(-100.0..100.0)).collect();
            
            // Create FixedSizeListArray with first vector
            let mut builder1 = arrow::array::FixedSizeListBuilder::new(
                Float32Builder::new(),
                dim1 as i32
            );
            let values_builder = builder1.values();
            for &val in &vec1 {
                values_builder.append_value(val);
            }
            builder1.append(true);
            let array1 = builder1.finish();
            let array_ref1: ArrayRef = Arc::new(array1);
            
            // Test VectorAvgAccumulator with first vector
            let mut accumulator = VectorAvgAccumulator::new();
            accumulator.update_batch(&[array_ref1]).unwrap();
            
            // Create FixedSizeListArray with second vector (different dimension)
            let mut builder2 = arrow::array::FixedSizeListBuilder::new(
                Float32Builder::new(),
                dim2 as i32
            );
            let values_builder = builder2.values();
            for &val in &vec2 {
                values_builder.append_value(val);
            }
            builder2.append(true);
            let array2 = builder2.finish();
            let array_ref2: ArrayRef = Arc::new(array2);
            
            // Attempt to update with different dimension should fail
            let result = accumulator.update_batch(&[array_ref2]);
            assert!(result.is_err(), "Expected error for dimension mismatch");
            
            if let Err(e) = result {
                let error_msg = format!("{}", e);
                assert!(error_msg.contains("different dimensions") || error_msg.contains("dimension"), 
                    "Error message should mention dimensions: {}", error_msg);
            }
        }
    }
}

#[cfg(test)]
mod grouped_aggregation_property_tests {
    use super::*;
    use proptest::prelude::*;
    use datafusion::prelude::*;
    use datafusion::arrow::array::{ArrayRef, Float32Array, Int32Array, FixedSizeListArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    // Feature: pgvector-sql-support, Property 15: Grouped Vector Aggregation
    // **Validates: Requirements 6.3**
    //
    // For any query using vector aggregation with GROUP BY, each group should have 
    // its aggregate computed independently, and the results should match computing 
    // the aggregate on each group separately.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        
        #[test]
        fn test_grouped_vector_sum(
            dim in 2usize..32,
            num_groups in 2usize..10,
            vectors_per_group in 2usize..20,
            seed in any::<u64>()
        ) {
            use rand::SeedableRng;
            use rand::Rng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            
            // Generate vectors for each group
            let mut group_ids = Vec::new();
            let mut vectors = Vec::new();
            let mut expected_sums: std::collections::HashMap<i32, Vec<f32>> = std::collections::HashMap::new();
            
            for group_id in 0..num_groups as i32 {
                let mut group_sum = vec![0.0f32; dim];
                for _ in 0..vectors_per_group {
                    let vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-100.0..100.0)).collect();
                    for (i, &val) in vec.iter().enumerate() {
                        group_sum[i] += val;
                    }
                    group_ids.push(group_id);
                    vectors.push(vec);
                }
                expected_sums.insert(group_id, group_sum);
            }
            
            // Create a DataFusion context and register vector UDFs
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let ctx = SessionContext::new();
                
                // Register vector aggregates
                for agg in all_vector_aggregates() {
                    ctx.register_udaf(agg);
                }
                
                // Create schema with group_id and vector columns
                let schema = Arc::new(Schema::new(vec![
                    Field::new("group_id", DataType::Int32, false),
                    Field::new("vec", DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dim as i32
                    ), false),
                ]));
                
                // Build arrays
                let group_array = Int32Array::from(group_ids.clone());
                let mut vec_builder = arrow::array::FixedSizeListBuilder::new(
                    Float32Builder::new(),
                    dim as i32
                );
                for vec in &vectors {
                    let values_builder = vec_builder.values();
                    for &val in vec {
                        values_builder.append_value(val);
                    }
                    vec_builder.append(true);
                }
                let vec_array = vec_builder.finish();
                
                // Create RecordBatch
                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(group_array) as ArrayRef, Arc::new(vec_array) as ArrayRef]
                ).unwrap();
                
                // Register as a table
                let provider = datafusion::datasource::MemTable::try_new(schema, vec![vec![batch]]).unwrap();
                ctx.register_table("test_table", Arc::new(provider)).unwrap();
                
                // Execute grouped aggregation query
                let df = ctx.sql("SELECT group_id, vector_sum(vec) as sum_vec FROM test_table GROUP BY group_id ORDER BY group_id").await.unwrap();
                let results = df.collect().await.unwrap();
                
                // Verify results
                assert!(!results.is_empty(), "Expected at least one result batch");
                let result_batch = &results[0];
                assert_eq!(result_batch.num_rows(), num_groups, "Expected {} groups", num_groups);
                
                let result_group_ids = result_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                let result_vecs = result_batch.column(1).as_any().downcast_ref::<ListArray>().unwrap();
                
                for i in 0..num_groups {
                    let group_id = result_group_ids.value(i);
                    let expected_sum = expected_sums.get(&group_id).unwrap();
                    
                    let result_vec_array = result_vecs.value(i);
                    let result_vec = result_vec_array.as_any().downcast_ref::<Float32Array>().unwrap();
                    
                    assert_eq!(result_vec.len(), dim, "Result vector dimension mismatch");
                    for j in 0..dim {
                        let diff = (result_vec.value(j) - expected_sum[j]).abs();
                        assert!(diff < 0.001 || diff / expected_sum[j].abs().max(0.001) < 0.0001,
                            "Group {} sum mismatch at index {}: expected {}, got {}", 
                            group_id, j, expected_sum[j], result_vec.value(j));
                    }
                }
            });
        }
        
        #[test]
        fn test_grouped_vector_avg(
            dim in 2usize..32,
            num_groups in 2usize..10,
            vectors_per_group in 2usize..20,
            seed in any::<u64>()
        ) {
            use rand::SeedableRng;
            use rand::Rng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            
            // Generate vectors for each group
            let mut group_ids = Vec::new();
            let mut vectors = Vec::new();
            let mut expected_avgs: std::collections::HashMap<i32, Vec<f32>> = std::collections::HashMap::new();
            
            for group_id in 0..num_groups as i32 {
                let mut group_sum = vec![0.0f32; dim];
                for _ in 0..vectors_per_group {
                    let vec: Vec<f32> = (0..dim).map(|_| rng.gen_range(-100.0..100.0)).collect();
                    for (i, &val) in vec.iter().enumerate() {
                        group_sum[i] += val;
                    }
                    group_ids.push(group_id);
                    vectors.push(vec);
                }
                let group_avg: Vec<f32> = group_sum.iter().map(|&s| s / vectors_per_group as f32).collect();
                expected_avgs.insert(group_id, group_avg);
            }
            
            // Create a DataFusion context and register vector UDFs
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let ctx = SessionContext::new();
                
                // Register vector aggregates
                for agg in all_vector_aggregates() {
                    ctx.register_udaf(agg);
                }
                
                // Create schema with group_id and vector columns
                let schema = Arc::new(Schema::new(vec![
                    Field::new("group_id", DataType::Int32, false),
                    Field::new("vec", DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dim as i32
                    ), false),
                ]));
                
                // Build arrays
                let group_array = Int32Array::from(group_ids.clone());
                let mut vec_builder = arrow::array::FixedSizeListBuilder::new(
                    Float32Builder::new(),
                    dim as i32
                );
                for vec in &vectors {
                    let values_builder = vec_builder.values();
                    for &val in vec {
                        values_builder.append_value(val);
                    }
                    vec_builder.append(true);
                }
                let vec_array = vec_builder.finish();
                
                // Create RecordBatch
                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(group_array) as ArrayRef, Arc::new(vec_array) as ArrayRef]
                ).unwrap();
                
                // Register as a table
                let provider = datafusion::datasource::MemTable::try_new(schema, vec![vec![batch]]).unwrap();
                ctx.register_table("test_table", Arc::new(provider)).unwrap();
                
                // Execute grouped aggregation query
                let df = ctx.sql("SELECT group_id, vector_avg(vec) as avg_vec FROM test_table GROUP BY group_id ORDER BY group_id").await.unwrap();
                let results = df.collect().await.unwrap();
                
                // Verify results
                assert!(!results.is_empty(), "Expected at least one result batch");
                let result_batch = &results[0];
                assert_eq!(result_batch.num_rows(), num_groups, "Expected {} groups", num_groups);
                
                let result_group_ids = result_batch.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                let result_vecs = result_batch.column(1).as_any().downcast_ref::<ListArray>().unwrap();
                
                for i in 0..num_groups {
                    let group_id = result_group_ids.value(i);
                    let expected_avg = expected_avgs.get(&group_id).unwrap();
                    
                    let result_vec_array = result_vecs.value(i);
                    let result_vec = result_vec_array.as_any().downcast_ref::<Float32Array>().unwrap();
                    
                    assert_eq!(result_vec.len(), dim, "Result vector dimension mismatch");
                    for j in 0..dim {
                        let diff = (result_vec.value(j) - expected_avg[j]).abs();
                        assert!(diff < 0.001 || diff / expected_avg[j].abs().max(0.001) < 0.0001,
                            "Group {} avg mismatch at index {}: expected {}, got {}", 
                            group_id, j, expected_avg[j], result_vec.value(j));
                    }
                }
            });
        }
    }
}

#[cfg(test)]
mod type_conversion_property_tests {
    use super::*;
    use proptest::prelude::*;
    use datafusion::prelude::*;
    use datafusion::arrow::array::{ArrayRef, Float32Array, FixedSizeListArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    // Feature: pgvector-sql-support, Property 20: Vector Type Conversion Round-Trip
    // **Validates: Requirements 9.1, 9.2**
    //
    // For any dense vector, converting to sparse and back to dense should produce 
    // an equivalent vector (within floating-point precision).
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]
        
        #[test]
        fn test_dense_sparse_dense_round_trip(
            dim in 2usize..128,
            sparsity in 0.1f32..0.9f32,
            seed in any::<u64>()
        ) {
            use rand::SeedableRng;
            use rand::Rng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            
            // Generate a vector with controlled sparsity
            let mut original: Vec<f32> = Vec::new();
            for _ in 0..dim {
                if rng.gen::<f32>() < sparsity {
                    original.push(0.0);
                } else {
                    original.push(rng.gen_range(-100.0..100.0));
                }
            }
            
            // Convert to sparse
            let sparse = dense_to_sparse(&original);
            
            // Convert back to dense
            let reconstructed = sparse_to_dense(&sparse);
            
            // Verify round-trip
            assert_eq!(reconstructed.len(), original.len(), "Dimension mismatch after round-trip");
            for i in 0..dim {
                let diff = (reconstructed[i] - original[i]).abs();
                assert!(diff < 0.0001, 
                    "Value mismatch at index {}: original {}, reconstructed {}", 
                    i, original[i], reconstructed[i]);
            }
        }
        
        #[test]
        fn test_vector_to_sparse_udf_round_trip(
            dim in 2usize..64,
            sparsity in 0.1f32..0.9f32,
            seed in any::<u64>()
        ) {
            use rand::SeedableRng;
            use rand::Rng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            
            // Generate a vector with controlled sparsity
            let mut original: Vec<f32> = Vec::new();
            for _ in 0..dim {
                if rng.gen::<f32>() < sparsity {
                    original.push(0.0);
                } else {
                    original.push(rng.gen_range(-100.0..100.0));
                }
            }
            
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let ctx = SessionContext::new();
                
                // Register vector UDFs
                for udf in all_vector_udfs() {
                    ctx.register_udf(udf);
                }
                
                // Create schema with vector column
                let schema = Arc::new(Schema::new(vec![
                    Field::new("vec", DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dim as i32
                    ), false),
                ]));
                
                // Build array
                let mut vec_builder = arrow::array::FixedSizeListBuilder::new(
                    Float32Builder::new(),
                    dim as i32
                );
                let values_builder = vec_builder.values();
                for &val in &original {
                    values_builder.append_value(val);
                }
                vec_builder.append(true);
                let vec_array = vec_builder.finish();
                
                // Create RecordBatch
                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(vec_array) as ArrayRef]
                ).unwrap();
                
                // Register as a table
                let provider = datafusion::datasource::MemTable::try_new(schema, vec![vec![batch]]).unwrap();
                ctx.register_table("test_table", Arc::new(provider)).unwrap();
                
                // Execute round-trip conversion query
                let df = ctx.sql("SELECT sparse_to_vector(vector_to_sparse(vec)) as reconstructed FROM test_table").await.unwrap();
                let results = df.collect().await.unwrap();
                
                // Verify results
                assert!(!results.is_empty(), "Expected at least one result batch");
                let result_batch = &results[0];
                assert_eq!(result_batch.num_rows(), 1, "Expected 1 row");
                
                let result_vecs = result_batch.column(0).as_any().downcast_ref::<ListArray>().unwrap();
                let result_vec_array = result_vecs.value(0);
                let result_vec = result_vec_array.as_any().downcast_ref::<Float32Array>().unwrap();
                
                assert_eq!(result_vec.len(), dim, "Result vector dimension mismatch");
                for i in 0..dim {
                    let diff = (result_vec.value(i) - original[i]).abs();
                    assert!(diff < 0.0001, 
                        "Value mismatch at index {}: original {}, reconstructed {}", 
                        i, original[i], result_vec.value(i));
                }
            });
        }
        
        // Feature: pgvector-sql-support, Property 21: Binary Quantization via Casting
        // **Validates: Requirements 9.3**
        //
        // For any vector, casting to binary type should produce the same result as 
        // calling binary_quantize.
        #[test]
        fn test_binary_quantization_via_cast(
            dim in 2usize..128,
            seed in any::<u64>()
        ) {
            use rand::SeedableRng;
            use rand::Rng;
            let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
            
            // Generate a random vector
            let original: Vec<f32> = (0..dim).map(|_| rng.gen_range(-100.0..100.0)).collect();
            
            // Apply binary quantization directly
            let mut expected_packed = vec![0u8; (dim + 7) / 8];
            for (i, &val) in original.iter().enumerate() {
                if val >= 0.0 {
                    expected_packed[i / 8] |= 1 << (i % 8);
                }
            }
            
            let rt = tokio::runtime::Runtime::new().unwrap();
            rt.block_on(async {
                let ctx = SessionContext::new();
                
                // Register vector UDFs
                for udf in all_vector_udfs() {
                    ctx.register_udf(udf);
                }
                
                // Create schema with vector column
                let schema = Arc::new(Schema::new(vec![
                    Field::new("vec", DataType::FixedSizeList(
                        Arc::new(Field::new("item", DataType::Float32, true)),
                        dim as i32
                    ), false),
                ]));
                
                // Build array
                let mut vec_builder = arrow::array::FixedSizeListBuilder::new(
                    Float32Builder::new(),
                    dim as i32
                );
                let values_builder = vec_builder.values();
                for &val in &original {
                    values_builder.append_value(val);
                }
                vec_builder.append(true);
                let vec_array = vec_builder.finish();
                
                // Create RecordBatch
                let batch = RecordBatch::try_new(
                    schema.clone(),
                    vec![Arc::new(vec_array) as ArrayRef]
                ).unwrap();
                
                // Register as a table
                let provider = datafusion::datasource::MemTable::try_new(schema, vec![vec![batch]]).unwrap();
                ctx.register_table("test_table", Arc::new(provider)).unwrap();
                
                // Test both binary_quantize and vector_to_binary produce same result
                let df1 = ctx.sql("SELECT binary_quantize(vec) as binary FROM test_table").await.unwrap();
                let results1 = df1.collect().await.unwrap();
                
                let df2 = ctx.sql("SELECT vector_to_binary(vec) as binary FROM test_table").await.unwrap();
                let results2 = df2.collect().await.unwrap();
                
                // Verify both produce the same result
                assert!(!results1.is_empty() && !results2.is_empty(), "Expected results");
                
                let result1_batch = &results1[0];
                let result2_batch = &results2[0];
                
                let binary1 = result1_batch.column(0).as_any().downcast_ref::<ListArray>().unwrap();
                let binary2 = result2_batch.column(0).as_any().downcast_ref::<ListArray>().unwrap();
                
                let binary1_arr = binary1.value(0);
                let binary1_bytes = binary1_arr.as_any().downcast_ref::<arrow::array::UInt8Array>().unwrap();
                
                let binary2_arr = binary2.value(0);
                let binary2_bytes = binary2_arr.as_any().downcast_ref::<arrow::array::UInt8Array>().unwrap();
                
                // Verify both match expected
                assert_eq!(binary1_bytes.len(), expected_packed.len(), "Binary length mismatch");
                assert_eq!(binary2_bytes.len(), expected_packed.len(), "Binary length mismatch");
                
                for i in 0..expected_packed.len() {
                    assert_eq!(binary1_bytes.value(i), expected_packed[i], 
                        "binary_quantize mismatch at byte {}", i);
                    assert_eq!(binary2_bytes.value(i), expected_packed[i], 
                        "vector_to_binary mismatch at byte {}", i);
                    assert_eq!(binary1_bytes.value(i), binary2_bytes.value(i), 
                        "Results differ at byte {}", i);
                }
            });
        }
    }
}
