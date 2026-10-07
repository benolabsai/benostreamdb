// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, Int32Array, Int64Array,
    ListArray, ListBuilder, UInt8Array,
};
use arrow::datatypes::DataType;
use datafusion::common::cast::as_fixed_size_list_array;
use datafusion::error::Result;
use datafusion::logical_expr::{ColumnarValue, ScalarUDFImpl, Signature, Volatility};
use datafusion::scalar::ScalarValue;
use std::any::Any;
use std::sync::Arc;

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

/// Read a vector column as `Vec<Vec<f32>>`, accepting both `List` and
/// `FixedSizeList` layouts (and Float32/Float64/Int element types). The
/// element-wise ops previously required `FixedSizeList`, so a `List(Float32)`
/// argument (e.g. from a `::FLOAT[]` cast) failed with a cast error.
fn as_vec_of_f32(arr: &ArrayRef) -> Result<Vec<Vec<f32>>> {
    if let Some(fsl) = arr.as_any().downcast_ref::<FixedSizeListArray>() {
        let mut out = Vec::with_capacity(fsl.len());
        for i in 0..fsl.len() {
            out.push(f32_values(&fsl.value(i))?);
        }
        return Ok(out);
    }
    if let Some(list) = arr.as_any().downcast_ref::<ListArray>() {
        let mut out = Vec::with_capacity(list.len());
        for i in 0..list.len() {
            out.push(f32_values(&list.value(i))?);
        }
        return Ok(out);
    }
    Err(datafusion::error::DataFusionError::Execution(
        "vector op: expected a list-typed argument".to_string(),
    ))
}

/// Extract the `f32` values of a vector element array.
fn f32_values(a: &ArrayRef) -> Result<Vec<f32>> {
    if let Some(v) = a.as_any().downcast_ref::<Float32Array>() {
        return Ok(v.values().to_vec());
    }
    if let Some(v) = a.as_any().downcast_ref::<Float64Array>() {
        return Ok(v.values().iter().map(|&x| x as f32).collect());
    }
    if let Some(v) = a.as_any().downcast_ref::<Int32Array>() {
        return Ok(v.values().iter().map(|&x| x as f32).collect());
    }
    if let Some(v) = a.as_any().downcast_ref::<Int64Array>() {
        return Ok(v.values().iter().map(|&x| x as f32).collect());
    }
    if let Some(v) = a.as_any().downcast_ref::<UInt8Array>() {
        return Ok(v.values().iter().map(|&x| x as f32).collect());
    }
    Err(datafusion::error::DataFusionError::Execution(
        "vector op: unsupported element type".to_string(),
    ))
}

/// Normalize a `ColumnarValue` to an array of `rows` length (expanding a
/// scalar literal, e.g. `ARRAY[1.0, 1.0]::FLOAT[]`, to match the other side).
fn as_array_of(v: &ColumnarValue, rows: usize) -> Result<ArrayRef> {
    match v {
        ColumnarValue::Array(a) => Ok(a.clone()),
        ColumnarValue::Scalar(s) => s.to_array_of_size(rows),
    }
}

/// Row count of a pair of arguments (whichever side is an array).
fn pair_rows(l: &ColumnarValue, r: &ColumnarValue) -> usize {
    match (l, r) {
        (ColumnarValue::Array(a), _) => a.len(),
        (_, ColumnarValue::Array(b)) => b.len(),
        _ => 1,
    }
}

/// Build a `List(Float32)` array from per-row vectors.
fn list_of_f32(rows: &[Vec<f32>]) -> Result<ArrayRef> {
    let mut builder = arrow::array::ListBuilder::new(Float32Array::builder(0));
    for row in rows {
        builder.values().append_slice(row);
        builder.append(true);
    }
    Ok(Arc::new(builder.finish()))
}

/// Generic Macro for Element-wise Binary Ops
macro_rules! create_vector_binary_op_udf {
    ($name:ident, $func_name:expr, $op_fn:ident) => {
        #[derive(Debug)]
        pub struct $name {
            signature: Signature,
        }

        impl_dyn_traits!($name);

        impl $name {
            pub fn new() -> Self {
                Self {
                    // Element-wise vector ops take two vector arguments. The
                    // signature must accept list types (List/FixedSizeList of
                    // Float32) — declaring scalar `Float32` made every call
                    // fail argument coercion. The implementation validates the
                    // array types and reports a clear error otherwise.
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
            fn as_any(&self) -> &dyn Any {
                self
            }
            fn name(&self) -> &str {
                $func_name
            }
            fn signature(&self) -> &Signature {
                &self.signature
            }
            fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
                Ok(arg_types[0].clone())
            }
            fn invoke_with_args(
                &self,
                args: datafusion::logical_expr::ScalarFunctionArgs,
            ) -> Result<ColumnarValue> {
                let (lhs, rhs) = (&args.args[0], &args.args[1]);
                let rows = pair_rows(lhs, rhs);
                let l = as_array_of(lhs, rows)?;
                let r = as_array_of(rhs, rows)?;
                let l_rows = as_vec_of_f32(&l)?;
                let r_rows = as_vec_of_f32(&r)?;
                if l_rows.len() != r_rows.len() {
                    return Err(datafusion::error::DataFusionError::Execution(
                        "vector binary op: argument lengths differ".to_string(),
                    ));
                }
                let mut out = Vec::with_capacity(l_rows.len());
                for (v1, v2) in l_rows.iter().zip(r_rows.iter()) {
                    if v1.len() != v2.len() {
                        return Err(datafusion::error::DataFusionError::Execution(
                            "vector binary op: vector dimensions differ".to_string(),
                        ));
                    }
                    out.push($op_fn(v1, v2));
                }
                // Emit a ListArray so the produced type matches the declared
                // return type (`arg_types[0]`, i.e. List(Float32) for list
                // inputs).
                Ok(ColumnarValue::Array(list_of_f32(&out)?))
            }
        }
    };
}

// Instantiate element-wise binary ops
create_vector_binary_op_udf!(VectorAddUDF, "vector_add", add_vectors);
create_vector_binary_op_udf!(VectorSubUDF, "vector_sub", sub_vectors);
create_vector_binary_op_udf!(VectorMulUDF, "vector_mul", mul_vectors);

fn add_vectors(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}
fn sub_vectors(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}
fn mul_vectors(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

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
            // Two vector arguments (list types), not scalars.
            signature: Signature::any(2, Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for VectorConcatUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "vector_concat"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(Arc::new(arrow::datatypes::Field::new(
            "item",
            DataType::Float32,
            true,
        ))))
    }
    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let (lhs, rhs) = (&args.args[0], &args.args[1]);
        let rows = pair_rows(lhs, rhs);
        let l = as_array_of(lhs, rows)?;
        let r = as_array_of(rhs, rows)?;
        let l_rows = as_vec_of_f32(&l)?;
        let r_rows = as_vec_of_f32(&r)?;
        if l_rows.len() != r_rows.len() {
            return Err(datafusion::error::DataFusionError::Execution(
                "vector_concat: argument lengths differ".to_string(),
            ));
        }
        let mut out = Vec::with_capacity(l_rows.len());
        for (v1, v2) in l_rows.iter().zip(r_rows.iter()) {
            let mut concatenated = Vec::with_capacity(v1.len() + v2.len());
            concatenated.extend_from_slice(v1);
            concatenated.extend_from_slice(v2);
            out.push(concatenated);
        }
        Ok(ColumnarValue::Array(list_of_f32(&out)?))
    }
}

// --- VectorDimsUDF ---

#[derive(Debug)]
pub struct VectorDimsUDF {
    signature: Signature,
}
impl_dyn_traits!(VectorDimsUDF);
impl Default for VectorDimsUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorDimsUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}
impl ScalarUDFImpl for VectorDimsUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "vector_dims"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Int32)
    }
    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        match &args.args[0] {
            ColumnarValue::Array(arr) => {
                if let Some(fsl) = arr.as_any().downcast_ref::<FixedSizeListArray>() {
                    let len = fsl.value_length();
                    let results: Int32Array = (0..fsl.len()).map(|_| Some(len)).collect();
                    Ok(ColumnarValue::Array(Arc::new(results)))
                } else {
                    Ok(ColumnarValue::Scalar(ScalarValue::Int32(None)))
                }
            }
            _ => Ok(ColumnarValue::Scalar(ScalarValue::Int32(None))),
        }
    }
}

// --- VectorNormUDF ---

#[derive(Debug)]
pub struct VectorNormUDF {
    signature: Signature,
}
impl_dyn_traits!(VectorNormUDF);
impl Default for VectorNormUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorNormUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}
impl ScalarUDFImpl for VectorNormUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "vector_norm"
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
    ) -> Result<ColumnarValue> {
        match &args.args[0] {
            ColumnarValue::Array(arr) => {
                let rows = as_vec_of_f32(arr)?;
                let results: Vec<f32> = rows
                    .iter()
                    .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
                    .collect();
                Ok(ColumnarValue::Array(Arc::new(Float32Array::from(results))))
            }
            _ => Ok(ColumnarValue::Scalar(ScalarValue::Float32(None))),
        }
    }
}

// --- VectorNormalizeUDF ---

#[derive(Debug)]
pub struct VectorNormalizeUDF {
    signature: Signature,
}
impl_dyn_traits!(VectorNormalizeUDF);
impl Default for VectorNormalizeUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorNormalizeUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}
impl ScalarUDFImpl for VectorNormalizeUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "l2_normalize"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        // The implementation always emits Float32 values, so the declared type
        // must be List(Float32) — declaring the input type (e.g. List(Float64))
        // trips DataFusion's `result_data_type == expected_type` assertion.
        Ok(DataType::List(Arc::new(arrow::datatypes::Field::new(
            "item",
            DataType::Float32,
            true,
        ))))
    }
    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        match &args.args[0] {
            ColumnarValue::Array(arr) => {
                let rows = as_vec_of_f32(arr)?;
                let mut out = Vec::with_capacity(rows.len());
                for v in rows.iter() {
                    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    if norm > 0.0 {
                        out.push(v.iter().map(|x| x / norm).collect::<Vec<f32>>());
                    } else {
                        out.push(v.clone());
                    }
                }
                // Emit a ListArray so the produced type matches the declared
                // `List(Float32)` return type.
                Ok(ColumnarValue::Array(list_of_f32(&out)?))
            }
            _ => Ok(ColumnarValue::Scalar(ScalarValue::Null)),
        }
    }
}

// --- BinaryQuantizeUDF ---

#[derive(Debug)]
pub struct BinaryQuantizeUDF {
    signature: Signature,
}
impl_dyn_traits!(BinaryQuantizeUDF);
impl Default for BinaryQuantizeUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl BinaryQuantizeUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}
impl ScalarUDFImpl for BinaryQuantizeUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "binary_quantize"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(Arc::new(arrow::datatypes::Field::new(
            "item",
            DataType::UInt8,
            true,
        ))))
    }
    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        match &args.args[0] {
            ColumnarValue::Array(arr) => {
                // Handle both List and FixedSizeList arrays
                let vec_data: Vec<Vec<f32>> =
                    if let Some(list_arr) = arr.as_any().downcast_ref::<ListArray>() {
                        (0..list_arr.len())
                            .map(|i| {
                                let value_array = list_arr.value(i);
                                if let Some(f32_arr) =
                                    value_array.as_any().downcast_ref::<Float32Array>()
                                {
                                    f32_arr.values().to_vec()
                                } else if let Some(f64_arr) =
                                    value_array.as_any().downcast_ref::<Float64Array>()
                                {
                                    f64_arr.values().iter().map(|&x| x as f32).collect()
                                } else if let Some(i32_arr) =
                                    value_array.as_any().downcast_ref::<Int32Array>()
                                {
                                    i32_arr.values().iter().map(|&x| x as f32).collect()
                                } else if let Some(i64_arr) =
                                    value_array.as_any().downcast_ref::<Int64Array>()
                                {
                                    i64_arr.values().iter().map(|&x| x as f32).collect()
                                } else if let Some(u8_arr) =
                                    value_array.as_any().downcast_ref::<UInt8Array>()
                                {
                                    u8_arr.values().iter().map(|&x| x as f32).collect()
                                } else {
                                    vec![]
                                }
                            })
                            .collect()
                    } else if let Some(fsl) = arr.as_any().downcast_ref::<FixedSizeListArray>() {
                        (0..fsl.len())
                            .map(|i| {
                                let value_array = fsl.value(i);
                                if let Some(f32_arr) =
                                    value_array.as_any().downcast_ref::<Float32Array>()
                                {
                                    f32_arr.values().to_vec()
                                } else if let Some(f64_arr) =
                                    value_array.as_any().downcast_ref::<Float64Array>()
                                {
                                    f64_arr.values().iter().map(|&x| x as f32).collect()
                                } else if let Some(i32_arr) =
                                    value_array.as_any().downcast_ref::<Int32Array>()
                                {
                                    i32_arr.values().iter().map(|&x| x as f32).collect()
                                } else if let Some(i64_arr) =
                                    value_array.as_any().downcast_ref::<Int64Array>()
                                {
                                    i64_arr.values().iter().map(|&x| x as f32).collect()
                                } else if let Some(u8_arr) =
                                    value_array.as_any().downcast_ref::<UInt8Array>()
                                {
                                    u8_arr.values().iter().map(|&x| x as f32).collect()
                                } else {
                                    vec![]
                                }
                            })
                            .collect()
                    } else {
                        return Err(datafusion::error::DataFusionError::Execution(
                            "binary_quantize expects List or FixedSizeList array".to_string(),
                        ));
                    };

                let packed_len = if vec_data.is_empty() {
                    0
                } else {
                    vec_data[0].len().div_ceil(8)
                };
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
            }
            ColumnarValue::Scalar(scalar) => {
                let v: Vec<f32> = match scalar {
                    ScalarValue::List(list_arc) => {
                        let list_array = list_arc.as_ref();
                        if list_array.len() == 0 {
                            return Err(datafusion::error::DataFusionError::Execution(
                                "Empty List scalar".to_string(),
                            ));
                        }
                        let inner_array = list_array.value(0);
                        if let Some(f32_arr) = inner_array.as_any().downcast_ref::<Float32Array>() {
                            f32_arr.values().to_vec()
                        } else if let Some(f64_arr) =
                            inner_array.as_any().downcast_ref::<Float64Array>()
                        {
                            f64_arr.values().iter().map(|&x| x as f32).collect()
                        } else if let Some(i32_arr) =
                            inner_array.as_any().downcast_ref::<Int32Array>()
                        {
                            i32_arr.values().iter().map(|&x| x as f32).collect()
                        } else if let Some(i64_arr) =
                            inner_array.as_any().downcast_ref::<Int64Array>()
                        {
                            i64_arr.values().iter().map(|&x| x as f32).collect()
                        } else if let Some(u8_arr) =
                            inner_array.as_any().downcast_ref::<UInt8Array>()
                        {
                            u8_arr.values().iter().map(|&x| x as f32).collect()
                        } else {
                            return Err(datafusion::error::DataFusionError::Execution(format!(
                                "Unsupported inner array type in List scalar: {:?}",
                                inner_array.data_type()
                            )));
                        }
                    }
                    ScalarValue::FixedSizeList(arr) => {
                        if let Some(f32_arr) = arr.as_any().downcast_ref::<Float32Array>() {
                            f32_arr.values().to_vec()
                        } else {
                            return Err(datafusion::error::DataFusionError::Execution(
                                "Unsupported scalar FixedSizeList inner type".to_string(),
                            ));
                        }
                    }
                    _ => {
                        return Err(datafusion::error::DataFusionError::Execution(
                            "binary_quantize expects List or FixedSizeList scalar".to_string(),
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
                Ok(ColumnarValue::Scalar(ScalarValue::List(
                    ScalarValue::new_list_nullable(
                        &packed
                            .iter()
                            .map(|&b| ScalarValue::UInt8(Some(b)))
                            .collect::<Vec<_>>(),
                        &DataType::UInt8,
                    ),
                )))
            }
        }
    }
}

// --- SubvectorUDF ---

#[derive(Debug)]
pub struct SubvectorUDF {
    signature: Signature,
}
impl_dyn_traits!(SubvectorUDF);
impl Default for SubvectorUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl SubvectorUDF {
    pub fn new() -> Self {
        Self {
            // (vector, start, length) — the vector argument is a list type, so
            // an exact scalar signature rejected every call.
            signature: Signature::any(3, Volatility::Immutable),
        }
    }
}
impl ScalarUDFImpl for SubvectorUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "subvector"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(Arc::new(arrow::datatypes::Field::new(
            "item",
            DataType::Float32,
            true,
        ))))
    }
    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        let (vec_arg, start_arg, count_arg) = (&args.args[0], &args.args[1], &args.args[2]);

        match (vec_arg, start_arg, count_arg) {
            (
                ColumnarValue::Array(arr),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(start))),
                ColumnarValue::Scalar(ScalarValue::Int32(Some(count))),
            ) => {
                let rows = as_vec_of_f32(arr)?;
                let mut out = Vec::with_capacity(rows.len());
                for v in rows.iter() {
                    let s = (*start as usize).min(v.len());
                    let c = (*count as usize).min(v.len() - s);
                    out.push(v[s..s + c].to_vec());
                }
                Ok(ColumnarValue::Array(list_of_f32(&out)?))
            }
            _ => Ok(ColumnarValue::Scalar(ScalarValue::Null)),
        }
    }
}

// --- VectorToBinaryUDF ---

#[derive(Debug)]
pub struct VectorToBinaryUDF {
    signature: Signature,
}
impl_dyn_traits!(VectorToBinaryUDF);
impl Default for VectorToBinaryUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorToBinaryUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}
impl ScalarUDFImpl for VectorToBinaryUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "vector_to_binary"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(Arc::new(arrow::datatypes::Field::new(
            "item",
            DataType::UInt8,
            true,
        ))))
    }
    fn invoke_with_args(
        &self,
        args: datafusion::logical_expr::ScalarFunctionArgs,
    ) -> Result<ColumnarValue> {
        match &args.args[0] {
            ColumnarValue::Array(arr) => {
                let fsl = as_fixed_size_list_array(arr)?;
                let len = fsl.value_length();
                let packed_len = (len as usize).div_ceil(8);
                let mut list_builder = ListBuilder::new(arrow::array::UInt8Builder::new());

                for i in 0..fsl.len() {
                    let value_array = fsl.value(i);
                    let v = value_array
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or_else(|| {
                            datafusion::error::DataFusionError::Execution(
                                "vector transform: expected Float32Array values".to_string(),
                            )
                        })?
                        .values();
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
            }
            _ => Ok(ColumnarValue::Scalar(ScalarValue::Null)),
        }
    }
}
