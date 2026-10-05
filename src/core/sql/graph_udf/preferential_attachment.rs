// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::sql::graph_udf::graph_view::GraphAccumulatorBase;
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
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for PreferentialAttachmentUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "graph_preferential_attachment"
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
        let mut fields = GraphAccumulatorBase::state_fields();
        fields.push(Arc::new(Field::new("node1", DataType::UInt64, true)));
        fields.push(Arc::new(Field::new("node2", DataType::UInt64, true)));
        Ok(fields)
    }
}

#[derive(Debug)]
pub struct PreferentialAttachmentAccumulator {
    base: GraphAccumulatorBase,
    node1: u64,
    node2: u64,
}

impl PreferentialAttachmentAccumulator {
    fn new() -> Self {
        Self {
            base: GraphAccumulatorBase::new(),
            node1: 0,
            node2: 0,
        }
    }
}

impl Accumulator for PreferentialAttachmentAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut state = self.base.edge_state()?;
        state.push(ScalarValue::UInt64(Some(self.node1)));
        state.push(ScalarValue::UInt64(Some(self.node2)));
        Ok(state)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.is_empty() {
            return Ok(());
        }
        self.base.merge_edge_state(states, Some(2), Some(3))?;
        // An empty input partition emits default scalar args; adopting them
        // would make the result depend on merge order.
        if !GraphAccumulatorBase::state_has_edges(states) {
            return Ok(());
        }
        if states.len() <= 4 {
            return Ok(());
        }

        let node1_arr = states[4]
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Execution(
                    "preferential_attachment: expected UInt64Array for node1".to_string(),
                )
            })?;
        let node2_arr = states[5]
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Execution(
                    "preferential_attachment: expected UInt64Array for node2".to_string(),
                )
            })?;

        if !node1_arr.is_empty() && node1_arr.is_valid(0) {
            self.node1 = node1_arr.value(0);
        }
        if !node2_arr.is_empty() && node2_arr.is_valid(0) {
            self.node2 = node2_arr.value(0);
        }

        Ok(())
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() < 4 {
            return Err(DataFusionError::Execution(
                "preferential_attachment expects at least 4 arguments".to_string(),
            ));
        }

        self.base.update_edge_batch(values, Some(4), Some(5))?;

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

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        // Preferential attachment is defined on the *undirected* degree, so
        // count both endpoints of every edge. Both branches must use the same
        // definition or the result would depend on whether the accumulator
        // retained edge rows.
        let mut deg1 = 0;
        let mut deg2 = 0;

        if !self.base.is_empty() {
            for (_, u, v) in self.base.edges() {
                if u == self.node1 || v == self.node1 {
                    deg1 += 1;
                }
                if u == self.node2 || v == self.node2 {
                    deg2 += 1;
                }
            }
        } else {
            let graph = self.base.resolve_graph(&[self.node1, self.node2], 1)?;
            for (u, v) in graph.all_edges() {
                if u == self.node1 || v == self.node1 {
                    deg1 += 1;
                }
                if u == self.node2 || v == self.node2 {
                    deg2 += 1;
                }
            }
        }

        Ok(ScalarValue::Float64(Some((deg1 * deg2) as f64)))
    }

    fn size(&self) -> usize {
        self.base.size()
    }
}
