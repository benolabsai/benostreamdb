// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::sql::graph_udf::graph_view::GraphAccumulatorBase;
use ahash::{AHashMap, AHashSet};
use arrow::array::{ArrayRef, ListBuilder, StructBuilder, UInt64Builder};
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
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for StronglyConnectedComponentsUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "graph_strongly_connected_components"
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
        Ok(GraphAccumulatorBase::state_fields())
    }
}

#[derive(Debug)]
pub struct StronglyConnectedComponentsAccumulator {
    base: GraphAccumulatorBase,
}

impl StronglyConnectedComponentsAccumulator {
    fn new() -> Self {
        Self {
            base: GraphAccumulatorBase::new(),
        }
    }
}

impl Accumulator for StronglyConnectedComponentsAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        self.base.edge_state()
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.is_empty() {
            return Ok(());
        }
        self.base.merge_edge_state(states, Some(2), Some(3))
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() < 2 {
            return Err(DataFusionError::Execution(
                "strongly_connected_components expects at least 2 arguments".to_string(),
            ));
        }

        self.base.update_edge_batch(values, Some(2), Some(3))
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let graph = self.base.resolve_graph(&[], 1)?;
        let nodes = graph.all_nodes();

        // Tarjan's algorithm.
        //
        // The per-node maps/sets are keyed by integer node ids and are probed
        // several times per edge (index/lowlink/on_stack). The default
        // SipHash `std` hasher is needlessly slow (and DoS-hardened, which is
        // irrelevant for these internal integer keys), so use ahash.
        let mut index: AHashMap<u64, usize> = AHashMap::new();
        let mut lowlink: AHashMap<u64, usize> = AHashMap::new();
        let mut on_stack: AHashSet<u64> = AHashSet::new();
        let mut stack: Vec<u64> = Vec::new();
        let mut current_index = 0;
        let mut components: AHashMap<u64, u64> = AHashMap::new();
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
                    neighbors: graph.get_neighbors(start_v).into_iter(),
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
                                neighbors: graph.get_neighbors(w).into_iter(),
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

        let mut sorted_nodes: Vec<_> = components.keys().copied().collect();
        sorted_nodes.sort_unstable();

        for node in sorted_nodes {
            let scc_id = components[&node];
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
        self.base.size()
    }
}
