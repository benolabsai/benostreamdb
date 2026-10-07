// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Element-wise vector aggregates: `centroid`, `vector_min`, `vector_max`,
//! `vector_stddev`, `vector_median`.
//!
//! These are the natural companions to `vector_sum` / `vector_avg`. Each
//! operates element-wise across the vectors in a group and returns a single
//! `List<Float32>`.

use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float32Builder, ListArray, ListBuilder,
};
use arrow::datatypes::DataType;
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{AggregateUDFImpl, Signature, Volatility};
use datafusion::scalar::ScalarValue;
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_functions_aggregate_common::accumulator::{AccumulatorArgs, StateFieldsArgs};
use std::any::Any;
use std::sync::Arc;

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

/// Extract the `f32` values of row `i` from a `FixedSizeList`/`List` array.
///
/// Accepts `Float32`, `Float64`, and `Float16` inner arrays (SQL `ARRAY[...]`
/// literals are `Float64`).
fn row_values(arr: &ArrayRef, i: usize) -> Result<Vec<f32>> {
    let value_array = if let Some(fsl) = arr.as_any().downcast_ref::<FixedSizeListArray>() {
        fsl.value(i)
    } else if let Some(list) = arr.as_any().downcast_ref::<ListArray>() {
        list.value(i)
    } else {
        return Err(DataFusionError::Execution(format!(
            "vector aggregate: expected FixedSizeList or List, got {:?}",
            arr.data_type()
        )));
    };
    if let Some(v) = value_array.as_any().downcast_ref::<Float32Array>() {
        return Ok(v.values().to_vec());
    }
    if let Some(v) = value_array
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
    {
        return Ok(v.values().iter().map(|x| *x as f32).collect());
    }
    if let Some(v) = value_array
        .as_any()
        .downcast_ref::<arrow::array::Float16Array>()
    {
        return Ok(v.iter().map(|x| x.unwrap_or_default().to_f32()).collect());
    }
    Err(DataFusionError::Execution(format!(
        "vector aggregate: expected Float32/Float64/Float16 values, got {:?}",
        value_array.data_type()
    )))
}

/// Build a single-element `List<Float32>` scalar from `values`.
/// Build a `List(Float32)` scalar, optionally carrying the input field's
/// metadata (e.g. Iceberg column ids). DataFusion compares the aggregate's
/// declared result type against the produced value, so both must agree on the
/// field metadata or the state coalescing panics inside arrow's `coalesce`.
fn list_scalar(values: Option<&[f32]>) -> ScalarValue {
    list_scalar_with(values, None)
}

fn list_scalar_with(
    values: Option<&[f32]>,
    field: Option<&Arc<arrow::datatypes::Field>>,
) -> ScalarValue {
    let f = field
        .cloned()
        .unwrap_or_else(|| Arc::new(arrow::datatypes::Field::new("item", DataType::Float32, true)));
    let mut builder = ListBuilder::new(Float32Builder::new()).with_field(f);
    match values {
        Some(v) => {
            builder.values().append_slice(v);
            builder.append(true);
        }
        None => {
            builder.append(false);
        }
    }
    ScalarValue::List(Arc::new(builder.finish()))
}

fn list_type() -> DataType {
    DataType::List(Arc::new(arrow::datatypes::Field::new(
        "item",
        DataType::Float32,
        true,
    )))
}

fn state_field(name: &str) -> Arc<arrow::datatypes::Field> {
    Arc::new(arrow::datatypes::Field::new(name, list_type(), true))
}

/// State field for the row counter — `UInt64`, matching the `count` scalar the
/// accumulators emit in `state()`. Declaring it as a list (as `state_field`
/// does) makes DataFusion reject the state array with a type mismatch.
fn count_state_field(name: &str) -> Arc<arrow::datatypes::Field> {
    Arc::new(arrow::datatypes::Field::new(
        name,
        DataType::UInt64,
        true,
    ))
}

/// Read a partial `List<Float32>` from a merge state array.
fn merge_values(states: &[ArrayRef], idx: usize) -> Result<Option<Vec<f32>>> {
    let list_array = states[idx]
        .as_any()
        .downcast_ref::<ListArray>()
        .ok_or_else(|| {
            DataFusionError::Execution("expected ListArray in merge_batch".to_string())
        })?;
    if list_array.is_null(0) {
        return Ok(None);
    }
    let inner = list_array.value(0);
    if inner.is_empty() {
        return Ok(None);
    }
    let values = inner
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| DataFusionError::Execution("expected Float32Array".to_string()))?;
    Ok(Some(values.values().to_vec()))
}

fn check_dims(current: usize, incoming: usize) -> Result<()> {
    if current != incoming {
        return Err(DataFusionError::Execution(format!(
            "Cannot aggregate vectors of different dimensions: expected {}, got {}",
            current, incoming
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// vector_min / vector_max
// ---------------------------------------------------------------------------

macro_rules! elementwise_extreme_udf {
    ($udf:ident, $acc:ident, $name:expr, $pick:expr) => {
        #[derive(Debug)]
        pub struct $udf {
            signature: Signature,
        }
        impl_dyn_traits!($udf);
        impl Default for $udf {
            fn default() -> Self {
                Self::new()
            }
        }
        impl $udf {
            pub fn new() -> Self {
                Self {
                    signature: Signature::any(1, Volatility::Immutable),
                }
            }
        }
        impl AggregateUDFImpl for $udf {
            fn as_any(&self) -> &dyn Any {
                self
            }
            fn name(&self) -> &str {
                $name
            }
            fn signature(&self) -> &Signature {
                &self.signature
            }
            fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
                Ok(list_type())
            }
            fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
                Ok(Box::new($acc::new()))
            }
            fn state_fields(
                &self,
                _args: StateFieldsArgs,
            ) -> Result<Vec<Arc<arrow::datatypes::Field>>> {
                Ok(vec![state_field("value")])
            }
        }

        #[derive(Debug)]
        pub struct $acc {
            value: Option<Vec<f32>>,
                }
        impl $acc {
            fn new() -> Self {
                Self {
                    value: None,
                        }
            }
        }
        impl Accumulator for $acc {
            fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
                let arr = &values[0];
                for i in 0..arr.len() {
                    if arr.is_null(i) {
                        continue;
                    }
                    let row = row_values(arr, i)?;
                    match &mut self.value {
                        Some(v) => {
                            check_dims(v.len(), row.len())?;
                            for (a, b) in v.iter_mut().zip(row.iter()) {
                                *a = $pick(*a, *b);
                            }
                        }
                        None => self.value = Some(row),
                    }
                }
                Ok(())
            }
            fn evaluate(&mut self) -> Result<ScalarValue> {
                Ok(list_scalar(self.value.as_deref()))
            }
            fn size(&self) -> usize {
                std::mem::size_of::<Self>() + self.value.as_ref().map(|v| v.len() * 4).unwrap_or(0)
            }
            fn state(&mut self) -> Result<Vec<ScalarValue>> {
                Ok(vec![self.evaluate()?])
            }
            fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
                if let Some(partial) = merge_values(states, 0)? {
                    match &mut self.value {
                        Some(v) => {
                            check_dims(v.len(), partial.len())?;
                            for (a, b) in v.iter_mut().zip(partial.iter()) {
                                *a = $pick(*a, *b);
                            }
                        }
                        None => self.value = Some(partial),
                    }
                }
                Ok(())
            }
        }
    };
}

elementwise_extreme_udf!(
    VectorMinUDF,
    VectorMinAccumulator,
    "vector_min",
    |a: f32, b: f32| a.min(b)
);
elementwise_extreme_udf!(
    VectorMaxUDF,
    VectorMaxAccumulator,
    "vector_max",
    |a: f32, b: f32| a.max(b)
);

// ---------------------------------------------------------------------------
// centroid (alias for vector_avg)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct CentroidUDF {
    signature: Signature,
}
impl_dyn_traits!(CentroidUDF);
impl Default for CentroidUDF {
    fn default() -> Self {
        Self::new()
    }
}
impl CentroidUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}
impl AggregateUDFImpl for CentroidUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "centroid"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(list_type())
    }
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(CentroidAccumulator::new()))
    }
    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<arrow::datatypes::Field>>> {
        Ok(vec![state_field("sum"), count_state_field("count")])
    }
}

#[derive(Debug)]
pub struct CentroidAccumulator {
    sum: Option<Vec<f32>>,
    count: u64,
}
impl CentroidAccumulator {
    fn new() -> Self {
        Self {
            sum: None,
            count: 0,
        }
    }
}
impl Accumulator for CentroidAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let arr = &values[0];
        for i in 0..arr.len() {
            if arr.is_null(i) {
                continue;
            }
            let row = row_values(arr, i)?;
            match &mut self.sum {
                Some(s) => {
                    check_dims(s.len(), row.len())?;
                    for (a, b) in s.iter_mut().zip(row.iter()) {
                        *a += b;
                    }
                }
                None => self.sum = Some(row),
            }
            self.count += 1;
        }
        Ok(())
    }
    fn evaluate(&mut self) -> Result<ScalarValue> {
        match &self.sum {
            Some(s) if self.count > 0 => {
                let n = self.count as f32;
                let mean: Vec<f32> = s.iter().map(|x| x / n).collect();
                Ok(list_scalar(Some(&mean)))
            }
            _ => Ok(list_scalar(None)),
        }
    }
    fn size(&self) -> usize {
        std::mem::size_of::<Self>() + self.sum.as_ref().map(|v| v.len() * 4).unwrap_or(0)
    }
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let sum = list_scalar(self.sum.as_deref());
        let count = ScalarValue::UInt64(Some(self.count));
        Ok(vec![sum, count])
    }
    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if let Some(partial) = merge_values(states, 0)? {
            match &mut self.sum {
                Some(s) => {
                    check_dims(s.len(), partial.len())?;
                    for (a, b) in s.iter_mut().zip(partial.iter()) {
                        *a += b;
                    }
                }
                None => self.sum = Some(partial),
            }
        }
        let counts = states[1]
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .ok_or_else(|| DataFusionError::Execution("expected UInt64Array count".to_string()))?;
        if !counts.is_null(0) {
            self.count += counts.value(0);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// vector_stddev (population)
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct VectorStddevUDF {
    signature: Signature,
}
impl_dyn_traits!(VectorStddevUDF);
impl Default for VectorStddevUDF {
    fn default() -> Self {
        Self::new()
    }
}
impl VectorStddevUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}
impl AggregateUDFImpl for VectorStddevUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "vector_stddev"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(list_type())
    }
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(VectorStddevAccumulator::new()))
    }
    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<arrow::datatypes::Field>>> {
        Ok(vec![
            state_field("sum"),
            state_field("sum_sq"),
            count_state_field("count"),
        ])
    }
}

#[derive(Debug)]
pub struct VectorStddevAccumulator {
    sum: Option<Vec<f32>>,
    sum_sq: Option<Vec<f32>>,
    count: u64,
}
impl VectorStddevAccumulator {
    fn new() -> Self {
        Self {
            sum: None,
            sum_sq: None,
            count: 0,
        }
    }
}
impl Accumulator for VectorStddevAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let arr = &values[0];
        for i in 0..arr.len() {
            if arr.is_null(i) {
                continue;
            }
            let row = row_values(arr, i)?;
            match (&mut self.sum, &mut self.sum_sq) {
                (Some(s), Some(sq)) => {
                    check_dims(s.len(), row.len())?;
                    for ((a, b), x) in s.iter_mut().zip(sq.iter_mut()).zip(row.iter()) {
                        *a += x;
                        *b += x * x;
                    }
                }
                _ => {
                    self.sum = Some(row.clone());
                    self.sum_sq = Some(row.iter().map(|x| x * x).collect());
                }
            }
            self.count += 1;
        }
        Ok(())
    }
    fn evaluate(&mut self) -> Result<ScalarValue> {
        match (&self.sum, &self.sum_sq) {
            (Some(s), Some(sq)) if self.count > 0 => {
                let n = self.count as f32;
                let stddev: Vec<f32> = s
                    .iter()
                    .zip(sq.iter())
                    .map(|(sum, sum_sq)| {
                        let mean = sum / n;
                        let var = (sum_sq / n) - mean * mean;
                        var.max(0.0).sqrt()
                    })
                    .collect();
                Ok(list_scalar(Some(&stddev)))
            }
            _ => Ok(list_scalar(None)),
        }
    }
    fn size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.sum.as_ref().map(|v| v.len() * 4).unwrap_or(0)
            + self.sum_sq.as_ref().map(|v| v.len() * 4).unwrap_or(0)
    }
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![
            list_scalar(self.sum.as_deref()),
            list_scalar(self.sum_sq.as_deref()),
            ScalarValue::UInt64(Some(self.count)),
        ])
    }
    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if let Some(partial) = merge_values(states, 0)? {
            match &mut self.sum {
                Some(s) => {
                    check_dims(s.len(), partial.len())?;
                    for (a, b) in s.iter_mut().zip(partial.iter()) {
                        *a += b;
                    }
                }
                None => self.sum = Some(partial),
            }
        }
        if let Some(partial) = merge_values(states, 1)? {
            match &mut self.sum_sq {
                Some(s) => {
                    check_dims(s.len(), partial.len())?;
                    for (a, b) in s.iter_mut().zip(partial.iter()) {
                        *a += b;
                    }
                }
                None => self.sum_sq = Some(partial),
            }
        }
        let counts = states[2]
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .ok_or_else(|| DataFusionError::Execution("expected UInt64Array count".to_string()))?;
        if !counts.is_null(0) {
            self.count += counts.value(0);
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// vector_median
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct VectorMedianUDF {
    signature: Signature,
}
impl_dyn_traits!(VectorMedianUDF);
impl Default for VectorMedianUDF {
    fn default() -> Self {
        Self::new()
    }
}
impl VectorMedianUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}
impl AggregateUDFImpl for VectorMedianUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "vector_median"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(list_type())
    }
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(VectorMedianAccumulator::new()))
    }
    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<arrow::datatypes::Field>>> {
        Ok(vec![state_field("values")])
    }
}

#[derive(Debug)]
pub struct VectorMedianAccumulator {
    rows: Vec<Vec<f32>>,
}
impl VectorMedianAccumulator {
    fn new() -> Self {
        Self {
            rows: Vec::new(),
        }
    }
}

impl Accumulator for VectorMedianAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let arr = &values[0];
        for i in 0..arr.len() {
            if arr.is_null(i) {
                continue;
            }
            let row = row_values(arr, i)?;
            if let Some(first) = self.rows.first() {
                check_dims(first.len(), row.len())?;
            }
            self.rows.push(row);
        }
        Ok(())
    }
    fn evaluate(&mut self) -> Result<ScalarValue> {
        if self.rows.is_empty() {
            return Ok(list_scalar(None));
        }
        let dim = self.rows[0].len();
        let mut median = Vec::with_capacity(dim);
        for d in 0..dim {
            let mut col: Vec<f32> = self.rows.iter().map(|r| r[d]).collect();
            col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = col.len();
            let m = if n % 2 == 1 {
                col[n / 2]
            } else {
                (col[n / 2 - 1] + col[n / 2]) / 2.0
            };
            median.push(m);
        }
        Ok(list_scalar(Some(&median)))
    }
    fn size(&self) -> usize {
        std::mem::size_of::<Self>() + self.rows.iter().map(|r| r.len() * 4).sum::<usize>()
    }
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        // Flatten all rows into a single list for the merge state.
        let flat: Vec<f32> = self.rows.iter().flatten().copied().collect();
        Ok(vec![list_scalar(Some(&flat))])
    }
    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if let Some(flat) = merge_values(states, 0)? {
            if let Some(first) = self.rows.first() {
                if !flat.is_empty() {
                    check_dims(first.len(), flat.len())?;
                }
            }
            self.rows.push(flat);
        }
        Ok(())
    }
}
