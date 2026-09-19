// Copyright (c) 2026 Richard Albright. All rights reserved.
#![allow(unused_imports, unused_mut, unused_variables, dead_code)]

use arrow::array::{
    Array, ArrayRef, Float64Array, ListBuilder, StructBuilder, UInt32Array, UInt64Array,
    UInt64Builder,
};
use arrow::datatypes::{DataType, Field, Fields};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{AggregateUDFImpl, Signature, Volatility};
use datafusion::scalar::ScalarValue;
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_functions_aggregate_common::accumulator::{AccumulatorArgs, StateFieldsArgs};
use std::any::Any;
use std::collections::{HashMap, HashSet};
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
pub struct JaccardCoefficientUDF {
    signature: Signature,
}
impl_dyn_traits!(JaccardCoefficientUDF);

impl Default for JaccardCoefficientUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl JaccardCoefficientUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![
                    DataType::UInt64, // source
                    DataType::UInt64, // target
                    DataType::UInt64, // node1
                    DataType::UInt64, // node2
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for JaccardCoefficientUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "jaccard_coefficient"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(JaccardCoefficientAccumulator::new()))
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
pub struct JaccardCoefficientAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
    node1: Option<u64>,
    node2: Option<u64>,
}

impl JaccardCoefficientAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
            node1: None,
            node2: None,
        }
    }
}

impl Accumulator for JaccardCoefficientAccumulator {
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
            ScalarValue::UInt64(self.node1),
            ScalarValue::UInt64(self.node2),
        ])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.is_empty() {
            return Ok(());
        }
        let sources_list = states[0]
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .unwrap();
        let targets_list = states[1]
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
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

        if states.len() > 2 {
            if let Some(a_arr) = states[2].as_any().downcast_ref::<UInt64Array>() {
                if a_arr.is_valid(0) {
                    self.node1 = Some(a_arr.value(0));
                }
            }
        }
        if states.len() > 3 {
            if let Some(b_arr) = states[3].as_any().downcast_ref::<UInt64Array>() {
                if b_arr.is_valid(0) {
                    self.node2 = Some(b_arr.value(0));
                }
            }
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        // Jaccard similarity of the (undirected) neighborhoods of node1 and node2.
        let score = match (self.node1, self.node2) {
            (Some(a), Some(b)) => {
                let mut nbr: HashMap<u64, HashSet<u64>> = HashMap::new();
                for i in 0..self.sources.len().min(self.targets.len()) {
                    let (u, v) = (self.sources[i], self.targets[i]);
                    nbr.entry(u).or_default().insert(v);
                    nbr.entry(v).or_default().insert(u);
                }
                let na = nbr.get(&a).cloned().unwrap_or_default();
                let nb = nbr.get(&b).cloned().unwrap_or_default();
                let inter = na.intersection(&nb).count();
                let union = na.union(&nb).count();
                if union == 0 {
                    0.0
                } else {
                    inter as f64 / union as f64
                }
            }
            _ => 0.0,
        };
        Ok(ScalarValue::Float64(Some(score)))
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        let sources = values[0].as_any().downcast_ref::<UInt64Array>().unwrap();
        let targets = values[1].as_any().downcast_ref::<UInt64Array>().unwrap();

        self.sources.extend(sources.iter().flatten());
        self.targets.extend(targets.iter().flatten());

        if values.len() > 2 && !values[2].is_empty() {
            if let Some(a_arr) = values[2].as_any().downcast_ref::<UInt64Array>() {
                if a_arr.is_valid(0) {
                    self.node1 = Some(a_arr.value(0));
                }
            }
        }
        if values.len() > 3 && !values[3].is_empty() {
            if let Some(b_arr) = values[3].as_any().downcast_ref::<UInt64Array>() {
                if b_arr.is_valid(0) {
                    self.node2 = Some(b_arr.value(0));
                }
            }
        }

        Ok(())
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self) + self.sources.capacity() * 8 + self.targets.capacity() * 8
    }
}
