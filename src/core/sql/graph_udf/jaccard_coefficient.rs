// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
#![allow(unused_imports, unused_mut, unused_variables, dead_code)]

use crate::core::sql::graph_udf::graph_view::GraphAccumulatorBase;
use ahash::{AHashMap as HashMap, AHashSet as HashSet};
use arrow::array::{Array, ArrayRef, Float64Array, UInt32Array, UInt64Array, UInt64Builder};
use arrow::datatypes::{DataType, Field, Fields};
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
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for JaccardCoefficientUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "graph_jaccard_coefficient"
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
        let mut fields = GraphAccumulatorBase::state_fields();
        fields.push(Arc::new(Field::new("node1", DataType::UInt64, true)));
        fields.push(Arc::new(Field::new("node2", DataType::UInt64, true)));
        Ok(fields)
    }
}

#[derive(Debug)]
pub struct JaccardCoefficientAccumulator {
    base: GraphAccumulatorBase,
    node1: Option<u64>,
    node2: Option<u64>,
}

impl JaccardCoefficientAccumulator {
    fn new() -> Self {
        Self {
            base: GraphAccumulatorBase::new(),
            node1: None,
            node2: None,
        }
    }
}

impl Accumulator for JaccardCoefficientAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut state = self.base.edge_state()?;
        state.push(ScalarValue::UInt64(self.node1));
        state.push(ScalarValue::UInt64(self.node2));
        Ok(state)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.base.merge_edge_state(states, Some(2), Some(3))?;
        let node1_idx = states.len().saturating_sub(2).max(4);
        let node2_idx = states.len().saturating_sub(1).max(5);

        if states.len() > node1_idx {
            if let Some(a_arr) = states[node1_idx]
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
            {
                if a_arr.is_valid(0) {
                    self.node1 = Some(a_arr.value(0));
                }
            }
        }
        if states.len() > node2_idx {
            if let Some(b_arr) = states[node2_idx]
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
            {
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
                let graph = self.base.resolve_graph(&[], 0)?;
                let na: HashSet<u64> = graph.get_neighbors(a).into_iter().collect();
                let nb: HashSet<u64> = graph.get_neighbors(b).into_iter().collect();

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
        self.base.update_edge_batch(values, Some(4), Some(5))?;
        if values.len() > 2 && !values[2].is_empty() {
            if let Some(a_arr) = values[2]
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
            {
                if a_arr.is_valid(0) {
                    self.node1 = Some(a_arr.value(0));
                }
            }
        }
        if values.len() > 3 && !values[3].is_empty() {
            if let Some(b_arr) = values[3]
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
            {
                if b_arr.is_valid(0) {
                    self.node2 = Some(b_arr.value(0));
                }
            }
        }

        Ok(())
    }

    fn size(&self) -> usize {
        self.base.size() + std::mem::size_of::<Option<u64>>() * 2
    }
}
