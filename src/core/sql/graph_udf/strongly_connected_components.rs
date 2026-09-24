// Copyright (c) 2026 Richard Albright. All rights reserved.

use arrow::array::{Array, ArrayRef, ListBuilder, StructBuilder, UInt64Array, UInt64Builder};
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
pub struct StronglyConnectedComponentsUDF {
    signature: Signature,
}
impl_dyn_traits!(StronglyConnectedComponentsUDF);

impl Default for StronglyConnectedComponentsUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl StronglyConnectedComponentsUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::UInt64, DataType::UInt64],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for StronglyConnectedComponentsUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "strongly_connected_components"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        let struct_fields = vec![
            Field::new("node", DataType::UInt64, false),
            Field::new("scc_id", DataType::UInt64, false),
        ];
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(Fields::from(struct_fields)),
            true,
        ))))
    }

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(StronglyConnectedComponentsAccumulator::new()))
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
        ])
    }
}

#[derive(Debug)]
pub struct StronglyConnectedComponentsAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
}

impl StronglyConnectedComponentsAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
        }
    }
}

impl Accumulator for StronglyConnectedComponentsAccumulator {
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
        ])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let sources_list = states[0]
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Execution(
                    "strongly_connected_components: expected ListArray for sources".to_string(),
                )
            })?;
        let targets_list = states[1]
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Execution(
                    "strongly_connected_components: expected ListArray for targets".to_string(),
                )
            })?;

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
        Ok(())
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() != 2 {
            return Err(DataFusionError::Execution(
                "strongly_connected_components expects 2 arguments".to_string(),
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
        if self.sources.is_empty() {
            return Ok(ScalarValue::List(Arc::new(
                arrow::array::ListArray::from_iter_primitive::<arrow::datatypes::UInt64Type, _, _>(
                    vec![None::<Vec<Option<u64>>>],
                ),
            )));
        }

        let mut adjacency: HashMap<u64, Vec<u64>> = HashMap::new();
        let mut nodes: HashSet<u64> = HashSet::new();
        for i in 0..self.sources.len() {
            let u = self.sources[i];
            let v = self.targets[i];
            adjacency.entry(u).or_default().push(v);
            nodes.insert(u);
            nodes.insert(v);
        }

        // Tarjan's algorithm
        let mut index: HashMap<u64, usize> = HashMap::new();
        let mut lowlink: HashMap<u64, usize> = HashMap::new();
        let mut on_stack: HashSet<u64> = HashSet::new();
        let mut stack: Vec<u64> = Vec::new();
        let mut current_index = 0;
        let mut components: HashMap<u64, u64> = HashMap::new();
        let mut current_scc_id = 1;

        // Recursive DFS can blow stack, use iterative or careful recursion
        // Since nodes can be many, let's use iterative approach
        struct State {
            v: u64,
            neighbors: std::vec::IntoIter<u64>,
            has_pushed_children: bool,
        }

        let mut call_stack: Vec<State> = Vec::new();

        for &start_v in &nodes {
            if !index.contains_key(&start_v) {
                call_stack.push(State {
                    v: start_v,
                    neighbors: adjacency
                        .get(&start_v)
                        .cloned()
                        .unwrap_or_default()
                        .into_iter(),
                    has_pushed_children: false,
                });

                while let Some(mut state) = call_stack.pop() {
                    let state_v = state.v;
                    if !state.has_pushed_children {
                        index.insert(state_v, current_index);
                        lowlink.insert(state_v, current_index);
                        current_index += 1;
                        stack.push(state_v);
                        on_stack.insert(state_v);
                        state.has_pushed_children = true;
                    }

                    let mut pushed_child = false;
                    while let Some(w) = state.neighbors.next() {
                        if !index.contains_key(&w) {
                            // push new state
                            call_stack.push(State {
                                v: state_v,
                                neighbors: state.neighbors, // Move out of state
                                has_pushed_children: true,
                            });
                            call_stack.push(State {
                                v: w,
                                neighbors: adjacency
                                    .get(&w)
                                    .cloned()
                                    .unwrap_or_default()
                                    .into_iter(),
                                has_pushed_children: false,
                            });
                            pushed_child = true;
                            break;
                        } else if on_stack.contains(&w) {
                            let min_low = std::cmp::min(lowlink[&state_v], index[&w]);
                            lowlink.insert(state_v, min_low);
                        }
                    }

                    if pushed_child {
                        continue;
                    }

                    // Post-process state.v
                    if let Some(parent_state) = call_stack.last_mut() {
                        let min_low = std::cmp::min(lowlink[&parent_state.v], lowlink[&state_v]);
                        lowlink.insert(parent_state.v, min_low);
                    }

                    if lowlink[&state_v] == index[&state_v] {
                        let mut scc_nodes = Vec::new();
                        while let Some(w) = stack.pop() {
                            on_stack.remove(&w);
                            scc_nodes.push(w);
                            if w == state_v {
                                break;
                            }
                        }

                        let scc_id = current_scc_id;
                        current_scc_id += 1;
                        for w in scc_nodes {
                            components.insert(w, scc_id);
                        }
                    }
                }
            }
        }

        let struct_fields = Fields::from(vec![
            Field::new("node", DataType::UInt64, false),
            Field::new("scc_id", DataType::UInt64, false),
        ]);

        let mut node_builder = UInt64Builder::new();
        let mut scc_id_builder = UInt64Builder::new();

        for (&node, &scc_id) in &components {
            node_builder.append_value(node);
            scc_id_builder.append_value(scc_id);
        }

        let mut struct_builder = StructBuilder::new(
            struct_fields.clone(),
            vec![Box::new(node_builder), Box::new(scc_id_builder)],
        );

        for _ in 0..components.len() {
            struct_builder.append(true);
        }

        let mut list_builder = ListBuilder::new(struct_builder);
        list_builder.append(true);

        let list_array = list_builder.finish();
        Ok(ScalarValue::List(Arc::new(list_array)))
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self) + self.sources.capacity() * 8 + self.targets.capacity() * 8
    }
}
