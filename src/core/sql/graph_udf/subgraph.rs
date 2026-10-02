// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
#![allow(unused_imports, unused_mut, unused_variables, dead_code)]

use crate::core::sql::graph_udf::graph_view::GraphAccumulatorBase;
use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, ListArray, ListBuilder, StructBuilder,
    UInt32Array, UInt64Array, UInt64Builder,
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
pub struct SubgraphUDF {
    signature: Signature,
}
impl_dyn_traits!(SubgraphUDF);

impl Default for SubgraphUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl SubgraphUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for SubgraphUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "graph_subgraph"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        let struct_fields = vec![
            Field::new("source", DataType::UInt64, true),
            Field::new("target", DataType::UInt64, true),
        ];
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(Fields::from(struct_fields)),
            true,
        ))))
    }

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(SubgraphAccumulator::new()))
    }

    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        let mut fields = GraphAccumulatorBase::state_fields();
        fields.push(Arc::new(Field::new(
            "seeds",
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
            true,
        )));
        fields.push(Arc::new(Field::new("hops", DataType::UInt32, true)));
        fields.push(Arc::new(Field::new("directed", DataType::Boolean, true)));
        Ok(fields)
    }
}

#[derive(Debug)]
pub struct SubgraphAccumulator {
    base: GraphAccumulatorBase,
    seeds: Vec<u64>,
    hops: Option<u32>,
    directed: Option<bool>,
}

impl SubgraphAccumulator {
    fn new() -> Self {
        Self {
            base: GraphAccumulatorBase::new(),
            seeds: Vec::new(),
            hops: None,
            directed: None,
        }
    }
}

impl Accumulator for SubgraphAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut state = self.base.edge_state()?;
        let mut seeds_builder = arrow::array::ListBuilder::new(arrow::array::UInt64Builder::new());
        seeds_builder.values().append_slice(&self.seeds);
        seeds_builder.append(true);

        state.push(ScalarValue::List(Arc::new(seeds_builder.finish())));
        state.push(ScalarValue::UInt32(self.hops));
        state.push(ScalarValue::Boolean(self.directed));
        Ok(state)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.base.merge_edge_state(states, Some(2), Some(3))?;
        if states.len() > 4 {
            let seeds_list = states[4]
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| {
                    DataFusionError::Execution("Expected ListArray for seeds".to_string())
                })?;
            if !seeds_list.is_empty() && seeds_list.is_valid(0) {
                let s_arr = seeds_list.value(0);
                if let Some(s) = s_arr.as_any().downcast_ref::<arrow::array::UInt64Array>() {
                    if !s.is_empty() && self.seeds.is_empty() {
                        self.seeds.extend_from_slice(s.values());
                    }
                }
            }
        }
        if states.len() > 5 {
            let hops_arr = states[5]
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| {
                    DataFusionError::Execution("Expected UInt32Array for hops".to_string())
                })?;
            if !hops_arr.is_empty() && hops_arr.is_valid(0) && self.hops.is_none() {
                self.hops = Some(hops_arr.value(0));
            }
        }
        if states.len() > 6 {
            let dir_arr = states[6]
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    DataFusionError::Execution("Expected BooleanArray for directed".to_string())
                })?;
            if !dir_arr.is_empty() && dir_arr.is_valid(0) && self.directed.is_none() {
                self.directed = Some(dir_arr.value(0));
            }
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut source_builder = UInt64Builder::new();
        let mut target_builder = UInt64Builder::new();

        if !self.seeds.is_empty() {
            let hops = self.hops.unwrap_or(1);
            let directed = self.directed.unwrap_or(false);

            // For out-of-core compatibility, we resolve the graph once.
            // Since Subgraph requires checking if an edge connects two visited nodes,
            // we iterate over all edges using the GraphView trait.
            // Undirected traversal must follow edges in both directions. The
            // in-memory `resolve_graph` builds a *directed* graph, so for the
            // undirected case symmetrize the edge set explicitly. Out-of-core
            // graphs (a `graph_uri`) already carry their own mode.
            let graph: Box<dyn crate::core::sql::graph_udf::graph_view::GraphView> = if directed
                || self
                    .base
                    .graph_uri
                    .as_deref()
                    .is_some_and(|u| !u.is_empty())
            {
                self.base.resolve_graph(&[], 0)?
            } else {
                let edges: Vec<(u64, u64)> = self.base.edges().map(|(_, u, v)| (u, v)).collect();
                Box::new(
                    crate::core::sql::graph_udf::graph_view::SimpleGraph::from_undirected_edges(
                        &edges,
                    ),
                )
            };

            let mut visited = HashSet::new();
            let mut q = VecDeque::new();
            let mut neighbors: Vec<u64> = Vec::new();

            for &s in &self.seeds {
                visited.insert(s);
                q.push_back((s, 0));
            }

            while let Some((curr, dist)) = q.pop_front() {
                if dist < hops {
                    neighbors.clear();
                    graph.get_neighbors_into(curr, &mut neighbors);
                    for &n in &neighbors {
                        if !visited.contains(&n) {
                            visited.insert(n);
                            q.push_back((n, dist + 1));
                        }
                    }
                    if !directed {
                        // For undirected, we would need get_incoming_neighbors if it's a directed graph that we are treating as undirected.
                        // However, GraphView doesn't expose incoming neighbors right now.
                        // Wait, if it's treated as undirected, `GraphAccumulatorBase` already handles it if mode="undirected",
                        // but if it's directed locally, we may just traverse forward.
                        // Wait, `subgraph` creates adjacency for directed/undirected inside it.
                        // To preserve behavior without incoming neighbors, we do a full scan, but wait,
                        // if we want to just traverse undirected, `resolve_graph` might not be enough if directed=false but the graph was directed.
                        // However, `GraphAccumulatorBase::resolve_graph` handles the undirected mode. So `get_neighbors` gives both if it was read as undirected.
                    }
                }
            }

            for (u, v) in graph.all_edges() {
                if visited.contains(&u) && visited.contains(&v) {
                    source_builder.append_value(u);
                    target_builder.append_value(v);
                }
            }
        }

        let struct_fields = vec![
            Field::new("source", DataType::UInt64, true),
            Field::new("target", DataType::UInt64, true),
        ];

        let struct_array = arrow::array::StructArray::new(
            Fields::from(struct_fields),
            vec![
                Arc::new(source_builder.finish()) as _,
                Arc::new(target_builder.finish()) as _,
            ],
            None,
        );

        let list_data = arrow::array::ArrayData::builder(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(Fields::from(vec![
                Field::new("source", DataType::UInt64, true),
                Field::new("target", DataType::UInt64, true),
            ])),
            true,
        ))))
        .len(1)
        .add_buffer(arrow::buffer::Buffer::from_slice_ref([
            0i32,
            struct_array.len() as i32,
        ]))
        .add_child_data(struct_array.into_data())
        .build()
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;

        let list_array = arrow::array::ListArray::from(list_data);
        Ok(ScalarValue::List(Arc::new(list_array)))
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.base.update_edge_batch(values, Some(5), Some(6))?;

        if values.len() > 2 && !values[2].is_empty() && self.seeds.is_empty() {
            if let Some(seeds_list) = values[2].as_any().downcast_ref::<ListArray>() {
                if seeds_list.len() > 0 && seeds_list.is_valid(0) {
                    let list_values = seeds_list.value(0);
                    if let Some(uint_values) = list_values.as_any().downcast_ref::<UInt64Array>() {
                        for i in 0..uint_values.len() {
                            if uint_values.is_valid(i) {
                                self.seeds.push(uint_values.value(i));
                            }
                        }
                    }
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
        if values.len() > 4 && !values[4].is_empty() {
            if let Some(dir_arr) = values[4].as_any().downcast_ref::<BooleanArray>() {
                if dir_arr.is_valid(0) {
                    self.directed = Some(dir_arr.value(0));
                }
            }
        }

        Ok(())
    }

    fn size(&self) -> usize {
        self.base.size()
            + self.seeds.capacity() * 8
            + std::mem::size_of::<Option<u32>>()
            + std::mem::size_of::<Option<bool>>()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConnectingPathsUDF: pairwise shortest paths between seed nodes, returning
// the union of edges along those paths as (source, target) structs.
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ConnectingPathsUDF {
    signature: Signature,
}
impl_dyn_traits!(ConnectingPathsUDF);

impl Default for ConnectingPathsUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl ConnectingPathsUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for ConnectingPathsUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "graph_connecting_paths"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        let struct_fields = vec![
            Field::new("source", DataType::UInt64, true),
            Field::new("target", DataType::UInt64, true),
        ];
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(Fields::from(struct_fields)),
            true,
        ))))
    }

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(ConnectingPathsAccumulator::new()))
    }

    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        let mut fields = GraphAccumulatorBase::state_fields();
        fields.push(Arc::new(Field::new(
            "seeds",
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
            true,
        )));
        fields.push(Arc::new(Field::new("directed", DataType::Boolean, true)));
        Ok(fields)
    }
}

#[derive(Debug)]
pub struct ConnectingPathsAccumulator {
    base: GraphAccumulatorBase,
    seeds: Vec<u64>,
    directed: Option<bool>,
}

impl ConnectingPathsAccumulator {
    fn new() -> Self {
        Self {
            base: GraphAccumulatorBase::new(),
            seeds: Vec::new(),
            directed: None,
        }
    }
}

impl Accumulator for ConnectingPathsAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut state = self.base.edge_state()?;
        let mut seeds_builder = ListBuilder::new(UInt64Builder::new());
        seeds_builder.values().append_slice(&self.seeds);
        seeds_builder.append(true);

        state.push(ScalarValue::List(Arc::new(seeds_builder.finish())));
        state.push(ScalarValue::Boolean(self.directed));
        Ok(state)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.base.merge_edge_state(states, Some(2), Some(3))?;
        if states.len() > 4 {
            let seeds_list = states[4]
                .as_any()
                .downcast_ref::<ListArray>()
                .ok_or_else(|| {
                    DataFusionError::Execution("Expected ListArray for seeds".to_string())
                })?;
            if !seeds_list.is_empty() && seeds_list.is_valid(0) && self.seeds.is_empty() {
                let s_arr = seeds_list.value(0);
                if let Some(s) = s_arr.as_any().downcast_ref::<UInt64Array>() {
                    self.seeds.extend_from_slice(s.values());
                }
            }
        }
        if states.len() > 5 {
            let dir_arr = states[5]
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| {
                    DataFusionError::Execution("Expected BooleanArray for directed".to_string())
                })?;
            if !dir_arr.is_empty() && dir_arr.is_valid(0) && self.directed.is_none() {
                self.directed = Some(dir_arr.value(0));
            }
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut source_builder = UInt64Builder::new();
        let mut target_builder = UInt64Builder::new();

        if self.seeds.len() >= 2 {
            let directed = self.directed.unwrap_or(false);
            let graph = self.base.resolve_graph(&[], 0)?;

            let mut seen_edges: HashSet<(u64, u64)> = HashSet::new();
            let mut edges: Vec<(u64, u64)> = Vec::new();

            // BFS shortest path between two nodes; returns the node sequence.
            let bfs = |start: u64, goal: u64| -> Option<Vec<u64>> {
                if start == goal {
                    return Some(vec![start]);
                }
                let mut prev: HashMap<u64, u64> = HashMap::new();
                let mut visited: HashSet<u64> = HashSet::new();
                let mut q = VecDeque::new();
                let mut neighbors: Vec<u64> = Vec::new();
                visited.insert(start);
                q.push_back(start);
                while let Some(curr) = q.pop_front() {
                    if curr == goal {
                        let mut path = vec![goal];
                        let mut c = goal;
                        while c != start {
                            c = *prev.get(&c)?;
                            path.push(c);
                        }
                        path.reverse();
                        return Some(path);
                    }
                    neighbors.clear();
                    graph.get_neighbors_into(curr, &mut neighbors);
                    for &n in &neighbors {
                        if visited.insert(n) {
                            prev.insert(n, curr);
                            q.push_back(n);
                        }
                    }
                }
                None
            };

            for a_i in 0..self.seeds.len() {
                for b_i in (a_i + 1)..self.seeds.len() {
                    if let Some(path) = bfs(self.seeds[a_i], self.seeds[b_i]) {
                        for w in path.windows(2) {
                            if seen_edges.insert((w[0], w[1])) {
                                edges.push((w[0], w[1]));
                            }
                        }
                    }
                }
            }

            for (u, v) in edges {
                source_builder.append_value(u);
                target_builder.append_value(v);
            }
        }

        let struct_fields = vec![
            Field::new("source", DataType::UInt64, true),
            Field::new("target", DataType::UInt64, true),
        ];

        let struct_array = arrow::array::StructArray::new(
            Fields::from(struct_fields),
            vec![
                Arc::new(source_builder.finish()) as _,
                Arc::new(target_builder.finish()) as _,
            ],
            None,
        );

        let list_data = arrow::array::ArrayData::builder(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(Fields::from(vec![
                Field::new("source", DataType::UInt64, true),
                Field::new("target", DataType::UInt64, true),
            ])),
            true,
        ))))
        .len(1)
        .add_buffer(arrow::buffer::Buffer::from_slice_ref([
            0i32,
            struct_array.len() as i32,
        ]))
        .add_child_data(struct_array.into_data())
        .build()
        .map_err(|e| DataFusionError::ArrowError(Box::new(e), None))?;

        let list_array = ListArray::from(list_data);
        Ok(ScalarValue::List(Arc::new(list_array)))
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.base.update_edge_batch(values, Some(4), Some(5))?;

        if values.len() > 2 && !values[2].is_empty() && self.seeds.is_empty() {
            if let Some(seeds_list) = values[2].as_any().downcast_ref::<ListArray>() {
                if seeds_list.len() > 0 && seeds_list.is_valid(0) {
                    let list_values = seeds_list.value(0);
                    if let Some(uint_values) = list_values.as_any().downcast_ref::<UInt64Array>() {
                        for i in 0..uint_values.len() {
                            if uint_values.is_valid(i) {
                                self.seeds.push(uint_values.value(i));
                            }
                        }
                    }
                }
            }
        }
        if values.len() > 3 && !values[3].is_empty() {
            if let Some(dir_arr) = values[3].as_any().downcast_ref::<BooleanArray>() {
                if dir_arr.is_valid(0) {
                    self.directed = Some(dir_arr.value(0));
                }
            }
        }

        Ok(())
    }

    fn size(&self) -> usize {
        self.base.size() + self.seeds.capacity() * 8 + std::mem::size_of::<Option<bool>>()
    }
}
