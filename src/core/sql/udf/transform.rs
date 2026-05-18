// Copyright (c) 2026 Richard Albright. All rights reserved.

use std::any::Any;
use std::sync::Arc;
use arrow::array::{Array, Float32Array, Float64Array, FixedSizeListArray, Int32Array, Int64Array, ListBuilder, ListArray, UInt8Array};
use arrow::datatypes::DataType;
use datafusion::error::Result;
use datafusion::logical_expr::{ScalarUDFImpl, Signature, Volatility, ColumnarValue};
use datafusion::common::cast::as_fixed_size_list_array;
use datafusion::scalar::ScalarValue;

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

/// Generic Macro for Element-wise Binary Ops
macro_rules! create_vector_binary_op_udf {
    ($name:ident, $func_name:expr, $op_fn:ident) => {
        #[derive(Debug)]
        pub struct $name { signature: Signature }

        impl_dyn_traits!($name);

        impl $name { pub fn new() -> Self { Self { signature: Signature::exact(vec![DataType::Float32, DataType::Float32], Volatility::Immutable) } } }
        impl Default for $name { fn default() -> Self { Self::new() } }
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

// Instantiate element-wise binary ops
create_vector_binary_op_udf!(VectorAddUDF, "vector_add", add_vectors);
create_vector_binary_op_udf!(VectorSubUDF, "vector_sub", sub_vectors);
create_vector_binary_op_udf!(VectorMulUDF, "vector_mul", mul_vectors);

fn add_vectors(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b.iter()).map(|(x, y)| x + y).collect() }
fn sub_vectors(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b.iter()).map(|(x, y)| x - y).collect() }
fn mul_vectors(a: &[f32], b: &[f32]) -> Vec<f32> { a.iter().zip(b.iter()).map(|(x, y)| x * y).collect() }

// --- VectorConcatUDF ---

#[derive(Debug)]
pub struct VectorConcatUDF {
    signature: Signature,
}

impl_dyn_traits!(VectorConcatUDF);

impl Default for VectorConcatUDF {
    fn default() -> Self {
        Self::new()
    }
}

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
            _ => Err(datafusion::error::DataFusionError::Execution(
                "Unsupported argument combinations for vector_concat".to_string()
            )),
        }
    }
}

// --- VectorDimsUDF ---

#[derive(Debug)]
pub struct VectorDimsUDF { signature: Signature }
impl_dyn_traits!(VectorDimsUDF);
impl Default for VectorDimsUDF {
    fn default() -> Self {
        Self::new()
    }
}

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

// --- VectorNormUDF ---

#[derive(Debug)]
pub struct VectorNormUDF { signature: Signature }
impl_dyn_traits!(VectorNormUDF);
impl Default for VectorNormUDF {
    fn default() -> Self {
        Self::new()
    }
}

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

// --- VectorNormalizeUDF ---

#[derive(Debug)]
pub struct VectorNormalizeUDF { signature: Signature }
impl_dyn_traits!(VectorNormalizeUDF);
impl Default for VectorNormalizeUDF {
    fn default() -> Self {
        Self::new()
    }
}

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

// --- BinaryQuantizeUDF ---

#[derive(Debug)]
pub struct BinaryQuantizeUDF { signature: Signature }
impl_dyn_traits!(BinaryQuantizeUDF);
impl Default for BinaryQuantizeUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl BinaryQuantizeUDF { pub fn new() -> Self { Self { signature: Signature::any(1, Volatility::Immutable) } } }
impl ScalarUDFImpl for BinaryQuantizeUDF {
    fn as_any(&self) -> &dyn Any { self }
    fn name(&self) -> &str { "binary_quantize" }
    fn signature(&self) -> &Signature { &self.signature }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(Arc::new(arrow::datatypes::Field::new("item", DataType::UInt8, true))))
    }
    fn invoke_with_args(&self, args: datafusion::logical_expr::ScalarFunctionArgs) -> Result<ColumnarValue> {
        match &args.args[0] {
            ColumnarValue::Array(arr) => {
                // Handle both List and FixedSizeList arrays
                let vec_data: Vec<Vec<f32>> = if let Some(list_arr) = arr.as_any().downcast_ref::<ListArray>() {
                    (0..list_arr.len()).map(|i| {
                        let value_array = list_arr.value(i);
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
                    (0..fsl.len()).map(|i| {
                        let value_array = fsl.value(i);
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

                let packed_len = if vec_data.is_empty() { 0 } else { vec_data[0].len().div_ceil(8) };
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
                let v: Vec<f32> = match scalar {
                    ScalarValue::List(list_arc) => {
                        let list_array = list_arc.as_ref();
                        if list_array.len() == 0 {
                            return Err(datafusion::error::DataFusionError::Execution(
                                "Empty List scalar".to_string()
                            ));
                        }
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

                let packed_len = v.len().div_ceil(8);
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

// --- SubvectorUDF ---

#[derive(Debug)]
pub struct SubvectorUDF { signature: Signature }
impl_dyn_traits!(SubvectorUDF);
impl Default for SubvectorUDF {
    fn default() -> Self {
        Self::new()
    }
}

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

// --- VectorToBinaryUDF ---

#[derive(Debug)]
pub struct VectorToBinaryUDF { signature: Signature }
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
        Ok(DataType::List(Arc::new(arrow::datatypes::Field::new("item", DataType::UInt8, true))))
    }
    fn invoke_with_args(&self, args: datafusion::logical_expr::ScalarFunctionArgs) -> Result<ColumnarValue> {
        match &args.args[0] {
            ColumnarValue::Array(arr) => {
                let fsl = as_fixed_size_list_array(arr)?;
                let len = fsl.value_length();
                let packed_len = (len as usize).div_ceil(8);
                let mut list_builder = ListBuilder::new(arrow::array::UInt8Builder::new());

                for i in 0..fsl.len() {
                    let value_array = fsl.value(i);
                    let v = value_array.as_any().downcast_ref::<Float32Array>().unwrap().values();
                    let mut packed = vec![0u8; packed_len];
                    for (j, &val) in v.iter().enumerate() {
                        if val >= 0.0 {
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
