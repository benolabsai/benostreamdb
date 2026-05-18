// Copyright (c) 2026 Richard Albright. All rights reserved.

use std::any::Any;
use std::sync::Arc;
use arrow::array::{Array, ArrayRef, Float32Array, FixedSizeListArray, ListBuilder, Float32Builder, ListArray};
use arrow::datatypes::DataType;
use datafusion::error::Result;
use datafusion::logical_expr::{ScalarUDFImpl, Signature, Volatility, ColumnarValue};
use datafusion::scalar::ScalarValue;
use crate::core::index::SparseVector;

/// Helper macro to implement DynEq and DynHash for UDF structs
macro_rules! impl_dyn_traits {
    ($name:ident) => {
        impl PartialEq for $name {
            fn eq(&self, _other: &Self) -> bool {
                true
            }
        }

        impl Eq for $name {}

        impl std::hash::Hash for $name {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                std::any::type_name::<Self>().hash(state);
            }
        }
    };
}

// --- VectorToSparseUDF ---

#[derive(Debug)]
pub struct VectorToSparseUDF { signature: Signature }
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

                    // Use dense_to_sparse from distance module
                    let sparse = super::distance::dense_to_sparse(dense.values());

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

// --- SparseToVectorUDF ---

#[derive(Debug)]
pub struct SparseToVectorUDF { signature: Signature }
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

                    let dense = super::distance::sparse_to_dense(&sparse);

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

// --- Sparse utility functions ---

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
