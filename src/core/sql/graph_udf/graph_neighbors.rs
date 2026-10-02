// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
#![allow(unused_imports, unused_mut, unused_variables, dead_code)]

use crate::core::sql::graph_udf::graph_view::GraphAccumulatorBase;
use arrow::array::{Array, ArrayRef, Float64Array, UInt32Array, UInt64Array};
use arrow::datatypes::{DataType, Field, Fields};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{AggregateUDFImpl, Signature, Volatility};
use datafusion::scalar::ScalarValue;
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_functions_aggregate_common::accumulator::{AccumulatorArgs, StateFieldsArgs};
use std::any::Any;
use std::collections::{HashMap, HashSet, VecDeque};
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
pub struct GraphNeighborsUDF {
    signature: Signature,
}
impl_dyn_traits!(GraphNeighborsUDF);

impl Default for GraphNeighborsUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphNeighborsUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for GraphNeighborsUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "graph_neighbors"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::UInt64,
            true,
        ))))
    }

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(GraphNeighborsAccumulator::new()))
    }

    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        let mut fields = GraphAccumulatorBase::state_fields();
        fields.push(Arc::new(Field::new("node", DataType::UInt64, true)));
        fields.push(Arc::new(Field::new("hops", DataType::UInt32, true)));
        Ok(fields)
    }
}

#[derive(Debug)]
pub struct GraphNeighborsAccumulator {
    base: GraphAccumulatorBase,
    node: Option<u64>,
    hops: Option<u32>,
}

impl GraphNeighborsAccumulator {
    fn new() -> Self {
        Self {
            base: GraphAccumulatorBase::new(),
            node: None,
            hops: None,
        }
    }
}

impl Accumulator for GraphNeighborsAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut state = self.base.edge_state()?;
        state.push(ScalarValue::UInt64(self.node));
        state.push(ScalarValue::UInt32(self.hops));
        Ok(state)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.base.merge_edge_state(states, Some(2), Some(3))?;
        let node_idx = 4;
        let hops_idx = 5;

        if states.len() > node_idx {
            if let Some(node_arr) = states[node_idx]
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
            {
                if node_arr.is_valid(0) {
                    self.node = Some(node_arr.value(0));
                }
            }
        }
        if states.len() > hops_idx {
            if let Some(hops_arr) = states[hops_idx]
                .as_any()
                .downcast_ref::<arrow::array::UInt32Array>()
            {
                if hops_arr.is_valid(0) && self.hops.is_none() {
                    self.hops = Some(hops_arr.value(0));
                }
            }
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut builder = arrow::array::ListBuilder::new(arrow::array::UInt64Builder::new());

        if let Some(node) = self.node {
            let graph = self.base.resolve_graph(&[], 0)?;
            let hops = self.hops.unwrap_or(1);

            let mut visited: HashSet<u64> = HashSet::new();
            let mut q = VecDeque::new();
            visited.insert(node);
            q.push_back((node, 0u32));
            let mut neighbors: Vec<u64> = Vec::new();
            let mut scratch: Vec<u64> = Vec::new();

            while let Some((curr, dist)) = q.pop_front() {
                if dist >= hops {
                    continue;
                }
                scratch.clear();
                graph.get_neighbors_into(curr, &mut scratch);
                for &n in &scratch {
                    if visited.insert(n) {
                        if n != node {
                            neighbors.push(n);
                        }
                        q.push_back((n, dist + 1));
                    }
                }
            }

            neighbors.sort_unstable();
            neighbors.dedup();
            builder.values().append_slice(&neighbors);
            builder.append(true);
        } else {
            builder.append(false);
        }

        Ok(ScalarValue::List(Arc::new(builder.finish())))
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.base.update_edge_batch(values, Some(4), Some(5))?;
        if values.len() > 2 && !values[2].is_empty() {
            if let Some(node_arr) = values[2]
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
            {
                if node_arr.is_valid(0) {
                    self.node = Some(node_arr.value(0));
                }
            }
        }
        if values.len() > 3 && !values[3].is_empty() {
            if let Some(hops_arr) = values[3]
                .as_any()
                .downcast_ref::<arrow::array::UInt32Array>()
            {
                if hops_arr.is_valid(0) {
                    self.hops = Some(hops_arr.value(0));
                }
            }
        }

        Ok(())
    }

    fn size(&self) -> usize {
        self.base.size() + std::mem::size_of::<Option<u64>>() + std::mem::size_of::<Option<u32>>()
    }
}
