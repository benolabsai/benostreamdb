// Copyright (c) 2026 Richard Albright. All rights reserved.

use arrow::array::{Array, ArrayRef, UInt64Array};
use arrow::datatypes::{DataType, Field};
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

#[derive(Debug, Clone)]
pub struct PreferentialAttachmentUDF {
    signature: Signature,
}
impl_dyn_traits!(PreferentialAttachmentUDF);

impl Default for PreferentialAttachmentUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl PreferentialAttachmentUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![
                    DataType::UInt64,
                    DataType::UInt64,
                    DataType::UInt64,
                    DataType::UInt64,
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for PreferentialAttachmentUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "preferential_attachment"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(PreferentialAttachmentAccumulator::new()))
    }

    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        Ok(vec![
            Arc::new(Field::new(
                "sources",
                DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                true,
            )),
            Arc::new(Field::new(
                "targets",
                DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                true,
            )),
            Arc::new(Field::new("node1", DataType::UInt64, true)),
            Arc::new(Field::new("node2", DataType::UInt64, true)),
        ])
    }
}

#[derive(Debug)]
pub struct PreferentialAttachmentAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
    node1: u64,
    node2: u64,
}

impl PreferentialAttachmentAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
            node1: 0,
            node2: 0,
        }
    }
}

impl Accumulator for PreferentialAttachmentAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut sources_builder =
            arrow::array::ListBuilder::new(arrow::array::UInt64Builder::new());
        sources_builder.values().append_slice(&self.sources);
        sources_builder.append(true);

        let mut targets_builder =
            arrow::array::ListBuilder::new(arrow::array::UInt64Builder::new());
        targets_builder.values().append_slice(&self.targets);
        targets_builder.append(true);

        Ok(vec![
            ScalarValue::List(Arc::new(sources_builder.finish())),
            ScalarValue::List(Arc::new(targets_builder.finish())),
            ScalarValue::UInt64(Some(self.node1)),
            ScalarValue::UInt64(Some(self.node2)),
        ])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let sources_list = states[0]
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .unwrap();
        let targets_list = states[1]
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .unwrap();
        let node1_arr = states[2]
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap();
        let node2_arr = states[3]
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap();

        for i in 0..sources_list.len() {
            if sources_list.is_valid(i) {
                let s_arr = sources_list.value(i);
                if let Some(s) = s_arr.as_any().downcast_ref::<arrow::array::UInt64Array>() {
                    self.sources.extend_from_slice(s.values());
                }
            }
            if targets_list.is_valid(i) {
                let t_arr = targets_list.value(i);
                if let Some(t) = t_arr.as_any().downcast_ref::<arrow::array::UInt64Array>() {
                    self.targets.extend_from_slice(t.values());
                }
            }
        }

        if !node1_arr.is_empty() && node1_arr.is_valid(0) {
            self.node1 = node1_arr.value(0);
        }
        if !node2_arr.is_empty() && node2_arr.is_valid(0) {
            self.node2 = node2_arr.value(0);
        }

        Ok(())
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() != 4 {
            return Err(DataFusionError::Execution(
                "preferential_attachment expects 4 arguments".to_string(),
            ));
        }

        let sources_arr = values[0]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected UInt64Array for sources".to_string())
            })?;
        let targets_arr = values[1]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected UInt64Array for targets".to_string())
            })?;
        let n1_arr = values[2]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected UInt64Array for node1".to_string())
            })?;
        let n2_arr = values[3]
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected UInt64Array for node2".to_string())
            })?;

        if !n1_arr.is_empty() && n1_arr.is_valid(0) {
            self.node1 = n1_arr.value(0);
        }
        if !n2_arr.is_empty() && n2_arr.is_valid(0) {
            self.node2 = n2_arr.value(0);
        }

        let len = sources_arr.len();
        for i in 0..len {
            if sources_arr.is_valid(i) && targets_arr.is_valid(i) {
                self.sources.push(sources_arr.value(i));
                self.targets.push(targets_arr.value(i));
            }
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut deg1 = 0;
        let mut deg2 = 0;

        for i in 0..self.sources.len() {
            let u = self.sources[i];
            let v = self.targets[i];

            if u == self.node1 {
                deg1 += 1;
            }
            if v == self.node1 {
                deg1 += 1;
            }
            if u == self.node2 {
                deg2 += 1;
            }
            if v == self.node2 {
                deg2 += 1;
            }
        }

        Ok(ScalarValue::Float64(Some((deg1 * deg2) as f64)))
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self) + self.sources.capacity() * 8 + self.targets.capacity() * 8
    }
}
