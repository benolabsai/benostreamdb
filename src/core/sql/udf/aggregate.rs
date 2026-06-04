// Copyright (c) 2026 Richard Albright. All rights reserved.

use arrow::array::{
    Array, FixedSizeListArray, Float32Array, Float32Builder, ListArray, ListBuilder, UInt64Array,
};
use arrow::datatypes::DataType;
use datafusion::error::Result;
use datafusion::logical_expr::{AggregateUDFImpl, Signature, Volatility};
use datafusion::scalar::ScalarValue;
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_functions_aggregate_common::accumulator::{AccumulatorArgs, StateFieldsArgs};
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

// --- VectorSumUDF ---

#[derive(Debug)]
pub struct VectorSumUDF {
    signature: Signature,
}
impl_dyn_traits!(VectorSumUDF);
impl Default for VectorSumUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorSumUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}
impl AggregateUDFImpl for VectorSumUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "vector_sum"
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
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(VectorSumAccumulator::new()))
    }
    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<arrow::datatypes::Field>>> {
        Ok(vec![Arc::new(arrow::datatypes::Field::new(
            "sum",
            DataType::List(Arc::new(arrow::datatypes::Field::new(
                "item",
                DataType::Float32,
                true,
            ))),
            true,
        ))])
    }
}

#[derive(Debug)]
pub struct VectorSumAccumulator {
    sum: Option<Vec<f32>>,
}
impl VectorSumAccumulator {
    fn new() -> Self {
        Self { sum: None }
    }
}
impl Accumulator for VectorSumAccumulator {
    fn update_batch(&mut self, values: &[arrow::array::ArrayRef]) -> Result<()> {
        let arr = &values[0];

        // Handle both FixedSizeList and List arrays
        if let Some(fsl) = arr.as_any().downcast_ref::<FixedSizeListArray>() {
            for i in 0..fsl.len() {
                let value_array = fsl.value(i);
                let row = value_array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .values();

                // Dimension validation
                if let Some(ref mut s) = self.sum {
                    if s.len() != row.len() {
                        return Err(datafusion::error::DataFusionError::Execution(format!(
                            "Cannot aggregate vectors of different dimensions: expected {}, got {}",
                            s.len(),
                            row.len()
                        )));
                    }
                    for (a, b) in s.iter_mut().zip(row.iter()) {
                        *a += b;
                    }
                } else {
                    self.sum = Some(row.to_vec());
                }
            }
        } else if let Some(list_arr) = arr.as_any().downcast_ref::<ListArray>() {
            for i in 0..list_arr.len() {
                let value_array = list_arr.value(i);
                let row = value_array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .values();

                // Dimension validation
                if let Some(ref mut s) = self.sum {
                    if s.len() != row.len() {
                        return Err(datafusion::error::DataFusionError::Execution(format!(
                            "Cannot aggregate vectors of different dimensions: expected {}, got {}",
                            s.len(),
                            row.len()
                        )));
                    }
                    for (a, b) in s.iter_mut().zip(row.iter()) {
                        *a += b;
                    }
                } else {
                    self.sum = Some(row.to_vec());
                }
            }
        } else {
            return Err(datafusion::error::DataFusionError::Execution(format!(
                "Expected FixedSizeList or List array, got {:?}",
                arr.data_type()
            )));
        }
        Ok(())
    }
    fn evaluate(&mut self) -> Result<ScalarValue> {
        match &self.sum {
            Some(s) => {
                // Filter out NaN and Inf values
                let filtered: Vec<f32> = s
                    .iter()
                    .map(|&x| {
                        if x.is_nan() || x.is_infinite() {
                            0.0
                        } else {
                            x
                        }
                    })
                    .collect();

                // Create a ListArray with a single element containing the sum vector
                let mut builder = ListBuilder::new(Float32Builder::new());
                builder.values().append_slice(&filtered);
                builder.append(true);
                Ok(ScalarValue::List(Arc::new(builder.finish())))
            }
            None => {
                // Return a properly typed NULL - create a list with 1 NULL element
                let mut builder = ListBuilder::new(Float32Builder::new());
                builder.append(false); // Append a NULL value
                Ok(ScalarValue::List(Arc::new(builder.finish())))
            }
        }
    }
    fn size(&self) -> usize {
        std::mem::size_of::<Self>() + self.sum.as_ref().map(|s| s.len() * 4).unwrap_or(0)
    }
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.evaluate()?])
    }
    fn merge_batch(&mut self, states: &[arrow::array::ArrayRef]) -> Result<()> {
        // Merge partial aggregates from different partitions
        // states[0] contains the sum vectors from other partitions
        let list_array = states[0]
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Execution(
                    "Expected ListArray in merge_batch".to_string(),
                )
            })?;

        for i in 0..list_array.len() {
            // Skip NULL or empty lists (empty partitions)
            if list_array.is_null(i) {
                continue;
            }
            let partial_sum_array = list_array.value(i);
            if partial_sum_array.is_empty() {
                continue;
            }

            let partial_sum = partial_sum_array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    datafusion::error::DataFusionError::Execution(
                        "Expected Float32Array in partial sum".to_string(),
                    )
                })?;

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

// --- VectorAvgUDF ---

#[derive(Debug)]
pub struct VectorAvgUDF {
    signature: Signature,
}
impl_dyn_traits!(VectorAvgUDF);
impl Default for VectorAvgUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorAvgUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::any(1, Volatility::Immutable),
        }
    }
}
impl AggregateUDFImpl for VectorAvgUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "vector_avg"
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
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(VectorAvgAccumulator::new()))
    }
    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<arrow::datatypes::Field>>> {
        Ok(vec![
            Arc::new(arrow::datatypes::Field::new(
                "sum",
                DataType::List(Arc::new(arrow::datatypes::Field::new(
                    "item",
                    DataType::Float32,
                    true,
                ))),
                true,
            )),
            Arc::new(arrow::datatypes::Field::new(
                "count",
                DataType::UInt64,
                true,
            )),
        ])
    }
}

#[derive(Debug)]
pub struct VectorAvgAccumulator {
    sum: Option<Vec<f32>>,
    count: u64,
}
impl VectorAvgAccumulator {
    fn new() -> Self {
        Self {
            sum: None,
            count: 0,
        }
    }
}
impl Accumulator for VectorAvgAccumulator {
    fn update_batch(&mut self, values: &[arrow::array::ArrayRef]) -> Result<()> {
        let arr = &values[0];

        // Handle both FixedSizeList and List arrays
        if let Some(fsl) = arr.as_any().downcast_ref::<FixedSizeListArray>() {
            self.count += fsl.len() as u64;
            for i in 0..fsl.len() {
                let value_array = fsl.value(i);
                let row = value_array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .values();

                // Dimension validation
                if let Some(ref mut s) = self.sum {
                    if s.len() != row.len() {
                        return Err(datafusion::error::DataFusionError::Execution(format!(
                            "Cannot aggregate vectors of different dimensions: expected {}, got {}",
                            s.len(),
                            row.len()
                        )));
                    }
                    for (a, b) in s.iter_mut().zip(row.iter()) {
                        *a += b;
                    }
                } else {
                    self.sum = Some(row.to_vec());
                }
            }
        } else if let Some(list_arr) = arr.as_any().downcast_ref::<ListArray>() {
            self.count += list_arr.len() as u64;
            for i in 0..list_arr.len() {
                let value_array = list_arr.value(i);
                let row = value_array
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .unwrap()
                    .values();

                // Dimension validation
                if let Some(ref mut s) = self.sum {
                    if s.len() != row.len() {
                        return Err(datafusion::error::DataFusionError::Execution(format!(
                            "Cannot aggregate vectors of different dimensions: expected {}, got {}",
                            s.len(),
                            row.len()
                        )));
                    }
                    for (a, b) in s.iter_mut().zip(row.iter()) {
                        *a += b;
                    }
                } else {
                    self.sum = Some(row.to_vec());
                }
            }
        } else {
            return Err(datafusion::error::DataFusionError::Execution(format!(
                "Expected FixedSizeList or List array, got {:?}",
                arr.data_type()
            )));
        }
        Ok(())
    }
    fn evaluate(&mut self) -> Result<ScalarValue> {
        match &self.sum {
            Some(s) => {
                let avg: Vec<f32> = s
                    .iter()
                    .map(|&x| {
                        let val = x / self.count as f32;
                        // Filter out NaN and Inf values
                        if val.is_nan() || val.is_infinite() {
                            0.0
                        } else {
                            val
                        }
                    })
                    .collect();

                // Create a ListArray with a single element containing the average vector
                let mut builder = ListBuilder::new(Float32Builder::new());
                builder.values().append_slice(&avg);
                builder.append(true);
                Ok(ScalarValue::List(Arc::new(builder.finish())))
            }
            None => {
                // Return a properly typed NULL - create a list with 1 NULL element
                let mut builder = ListBuilder::new(Float32Builder::new());
                builder.append(false); // Append a NULL value
                Ok(ScalarValue::List(Arc::new(builder.finish())))
            }
        }
    }
    fn size(&self) -> usize {
        std::mem::size_of::<Self>() + self.sum.as_ref().map(|s| s.len() * 4).unwrap_or(0)
    }
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![
            match &self.sum {
                Some(s) => {
                    let mut builder = ListBuilder::new(Float32Builder::new());
                    builder.values().append_slice(s);
                    builder.append(true);
                    ScalarValue::List(Arc::new(builder.finish()))
                }
                None => {
                    // Return a properly typed NULL
                    let mut builder = ListBuilder::new(Float32Builder::new());
                    builder.append(false);
                    ScalarValue::List(Arc::new(builder.finish()))
                }
            },
            ScalarValue::UInt64(Some(self.count)),
        ])
    }
    fn merge_batch(&mut self, states: &[arrow::array::ArrayRef]) -> Result<()> {
        // Merge partial aggregates from different partitions
        // states[0] contains the sum vectors, states[1] contains the counts
        let sum_array = states[0]
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Execution(
                    "Expected ListArray for sum in merge_batch".to_string(),
                )
            })?;
        let count_array = states[1]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Execution(
                    "Expected UInt64Array for count in merge_batch".to_string(),
                )
            })?;

        for i in 0..sum_array.len() {
            // Skip NULL or empty lists (empty partitions)
            if sum_array.is_null(i) {
                continue;
            }
            let partial_sum_array = sum_array.value(i);
            if partial_sum_array.is_empty() {
                continue;
            }

            let partial_sum = partial_sum_array
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| {
                    datafusion::error::DataFusionError::Execution(
                        "Expected Float32Array in partial sum".to_string(),
                    )
                })?;
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
