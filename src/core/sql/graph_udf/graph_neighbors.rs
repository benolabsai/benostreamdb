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
            signature: Signature::exact(
                vec![
                    DataType::UInt64, // source
                    DataType::UInt64, // target
                    DataType::UInt64, // node
                    DataType::UInt32, // hops
                ],
                Volatility::Immutable,
            ),
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
            Arc::new(Field::new("node", DataType::UInt64, true)),
            Arc::new(Field::new("hops", DataType::UInt32, true)),
        ])
    }
}

#[derive(Debug)]
pub struct GraphNeighborsAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
    node: Option<u64>,
    hops: Option<u32>,
}

impl GraphNeighborsAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
            node: None,
            hops: None,
        }
    }
}

impl Accumulator for GraphNeighborsAccumulator {
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
            ScalarValue::UInt64(self.node),
            ScalarValue::UInt32(self.hops),
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
            if let Some(node_arr) = states[2].as_any().downcast_ref::<UInt64Array>() {
                if node_arr.is_valid(0) {
                    self.node = Some(node_arr.value(0));
                }
            }
        }
        if states.len() > 3 {
            if let Some(hops_arr) = states[3].as_any().downcast_ref::<UInt32Array>() {
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
            // BFS from `node` over the directed edge set, up to `hops` steps.
            let hops = self.hops.unwrap_or(1);
            let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
            for i in 0..self.sources.len().min(self.targets.len()) {
                adj.entry(self.sources[i])
                    .or_default()
                    .push(self.targets[i]);
            }

            let mut visited: HashSet<u64> = HashSet::new();
            let mut q = VecDeque::new();
            visited.insert(node);
            q.push_back((node, 0u32));
            let mut neighbors: Vec<u64> = Vec::new();

            while let Some((curr, dist)) = q.pop_front() {
                if dist >= hops {
                    continue;
                }
                if let Some(ns) = adj.get(&curr) {
                    for &n in ns {
                        if visited.insert(n) {
                            if n != node {
                                neighbors.push(n);
                            }
                            q.push_back((n, dist + 1));
                        }
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
        if values.is_empty() {
            return Ok(());
        }
        let sources = values[0].as_any().downcast_ref::<UInt64Array>().unwrap();
        let targets = values[1].as_any().downcast_ref::<UInt64Array>().unwrap();

        self.sources.extend(sources.iter().flatten());
        self.targets.extend(targets.iter().flatten());

        if values.len() > 2 && !values[2].is_empty() {
            if let Some(node_arr) = values[2].as_any().downcast_ref::<UInt64Array>() {
                if node_arr.is_valid(0) {
                    self.node = Some(node_arr.value(0));
                }
            }
        }
        if values.len() > 3 && !values[3].is_empty() {
            if let Some(hops_arr) = values[3].as_any().downcast_ref::<UInt32Array>() {
                if hops_arr.is_valid(0) {
                    self.hops = Some(hops_arr.value(0));
                }
            }
        }

        Ok(())
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self) + self.sources.capacity() * 8 + self.targets.capacity() * 8
    }
}
