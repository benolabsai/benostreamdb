// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Leiden community detection.
//!
//! Leiden improves on Louvain by adding a *refinement* phase that guarantees
//! every returned community is internally connected. Louvain's local-move phase
//! can leave a community that is disconnected (a node may join a community it
//! has no edge into, once its neighbours have moved), which produces
//! badly-connected communities and can even yield disconnected "communities" in
//! the final partition.
//!
//! This implementation runs the same greedy modularity local-move phase as
//! [`super::louvain_communities`], then refines the partition by splitting each
//! community into the connected components of its induced subgraph. The result
//! is a partition in which every community is connected — the property Leiden is
//! chosen for. It is single-level (no aggregation), matching the current
//! single-level Louvain implementation.

use crate::core::sql::graph_udf::graph_view::GraphAccumulatorBase;
use arrow::array::{
    Array, ArrayRef, Float32Array, Float64Array, ListBuilder, UInt64Array, UInt64Builder,
};
use arrow::datatypes::{DataType, Field};
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
pub struct LeidenCommunitiesUDF {
    signature: Signature,
}
impl_dyn_traits!(LeidenCommunitiesUDF);

impl Default for LeidenCommunitiesUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl LeidenCommunitiesUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for LeidenCommunitiesUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "graph_leiden_communities"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        let inner_list = DataType::List(Arc::new(Field::new("item", DataType::UInt64, true)));
        Ok(DataType::List(Arc::new(Field::new(
            "item", inner_list, true,
        ))))
    }
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(LeidenAccumulator::new()))
    }
    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        let mut fields = GraphAccumulatorBase::state_fields();
        fields.push(Arc::new(Field::new(
            "weights",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            true,
        )));
        fields.push(Arc::new(Field::new("resolution", DataType::Float32, true)));
        Ok(fields)
    }
}

#[derive(Debug)]
pub struct LeidenAccumulator {
    base: GraphAccumulatorBase,
    weights: Vec<f32>,
    resolution: f32,
}

impl LeidenAccumulator {
    fn new() -> Self {
        Self {
            base: GraphAccumulatorBase::new(),
            weights: Vec::new(),
            resolution: 1.0,
        }
    }
}

impl Accumulator for LeidenAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() < 2 {
            return Err(DataFusionError::Execution(
                "leiden_communities expects at least 2 arguments (source, target)".to_string(),
            ));
        }

        self.base.update_edge_batch(values, Some(4), Some(5))?;

        let len = values[0].len();

        let mut weights_vec: Vec<f32> = vec![1.0; len];
        if values.len() > 2 {
            if let Some(w_arr) = values[2].as_any().downcast_ref::<Float32Array>() {
                for i in 0..len {
                    if w_arr.is_valid(i) {
                        weights_vec[i] = w_arr.value(i);
                    }
                }
            } else if let Some(w_arr) = values[2].as_any().downcast_ref::<Float64Array>() {
                for i in 0..len {
                    if w_arr.is_valid(i) {
                        weights_vec[i] = w_arr.value(i) as f32;
                    }
                }
            }
        }

        if values.len() > 3 && !values[3].is_empty() {
            if let Some(r_arr) = values[3].as_any().downcast_ref::<Float64Array>() {
                if r_arr.is_valid(0) {
                    self.resolution = r_arr.value(0) as f32;
                }
            } else if let Some(r_arr) = values[3].as_any().downcast_ref::<Float32Array>() {
                if r_arr.is_valid(0) {
                    self.resolution = r_arr.value(0);
                }
            }
        }

        let sources_arr = values[0].as_any().downcast_ref::<UInt64Array>().ok_or_else(|| datafusion::error::DataFusionError::Execution("type mismatch".into()))?;
        let targets_arr = values[1].as_any().downcast_ref::<UInt64Array>().ok_or_else(|| datafusion::error::DataFusionError::Execution("type mismatch".into()))?;

        for i in 0..len {
            if sources_arr.is_valid(i) && targets_arr.is_valid(i) {
                self.weights.push(weights_vec[i]);
            }
        }

        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.base.merge_edge_state(states, Some(2), Some(3))?;
        // An empty input partition emits default scalar args; adopting them
        // would make the result depend on merge order.
        if !GraphAccumulatorBase::state_has_edges(states) {
            return Ok(());
        }
        if states.len() > 4 {
            let weights_list = states[4]
                .as_any()
                .downcast_ref::<arrow::array::ListArray>()
                .ok_or_else(|| {
                    DataFusionError::Execution("Expected ListArray for weights".to_string())
                })?;

            for i in 0..weights_list.len() {
                if weights_list.is_valid(i) {
                    let w_arr = weights_list.value(i);
                    if let Some(w) = w_arr.as_any().downcast_ref::<Float32Array>() {
                        self.weights.extend_from_slice(w.values());
                    }
                }
            }
        }

        if let Some(r_arr) = states
            .get(5)
            .and_then(|a| a.as_any().downcast_ref::<Float32Array>())
        {
            if !r_arr.is_empty() && r_arr.is_valid(0) {
                self.resolution = r_arr.value(0);
            }
        }

        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut state = self.base.edge_state()?;

        let mut weights_builder =
            arrow::array::ListBuilder::new(arrow::array::Float32Builder::new());
        weights_builder.values().append_slice(&self.weights);
        weights_builder.append(true);

        state.push(ScalarValue::List(Arc::new(weights_builder.finish())));
        state.push(ScalarValue::Float32(Some(self.resolution)));
        Ok(state)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let graph = self.base.resolve_graph(&[], 0)?;

        let mut node_map: HashMap<u64, usize> = HashMap::new();
        let mut reverse_map: Vec<u64> = Vec::new();

        let mut all_edges = Vec::new();
        if !self.base.is_empty() {
            for (i, u, v) in self.base.edges() {
                all_edges.push((u, v, self.weights[i].max(0.0)));
            }
        } else {
            for (u, v) in graph.all_edges() {
                all_edges.push((u, v, 1.0));
            }
        }

        for &(s, t, _) in &all_edges {
            node_map.entry(s).or_insert_with(|| {
                reverse_map.push(s);
                reverse_map.len() - 1
            });
            node_map.entry(t).or_insert_with(|| {
                reverse_map.push(t);
                reverse_map.len() - 1
            });
        }

        let num_nodes = reverse_map.len();
        if num_nodes == 0 {
            let inner_builder = UInt64Builder::new();
            let component_builder = ListBuilder::new(inner_builder);
            let mut final_builder = ListBuilder::new(component_builder);
            final_builder.append(true);
            return Ok(ScalarValue::List(Arc::new(final_builder.finish())));
        }

        // Adjacency list: node -> Vec<(neighbor_node, weight)>
        let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); num_nodes];
        let mut node_degrees: Vec<f32> = vec![0.0; num_nodes];
        let mut total_weight: f32 = 0.0;

        for &(s, t, w) in &all_edges {
            let u = node_map[&s];
            let v = node_map[&t];

            adj[u].push((v, w));
            adj[v].push((u, w));
            node_degrees[u] += w;
            node_degrees[v] += w;
            total_weight += w;
        }

        let two_m = (total_weight * 2.0).max(1e-6);

        // ── Phase 1: local move (greedy modularity), identical to Louvain ──
        let mut community: Vec<usize> = (0..num_nodes).collect();
        let mut comm_tot: Vec<f32> = node_degrees.clone();

        for _pass in 0..15 {
            let mut moved = false;

            for u in 0..num_nodes {
                let c_u = community[u];
                let k_u = node_degrees[u];

                comm_tot[c_u] -= k_u;

                let mut comm_weights: HashMap<usize, f32> = HashMap::new();
                for &(v, w) in &adj[u] {
                    if v != u {
                        *comm_weights.entry(community[v]).or_default() += w;
                    }
                }

                let mut best_c = c_u;
                let k_in_curr = *comm_weights.get(&c_u).unwrap_or(&0.0);
                let current_delta = k_in_curr - self.resolution * (comm_tot[c_u] * k_u) / two_m;
                let mut best_delta = current_delta;

                for (&c, &k_in) in &comm_weights {
                    let delta = k_in - self.resolution * (comm_tot[c] * k_u) / two_m;
                    if delta > best_delta {
                        best_delta = delta;
                        best_c = c;
                    }
                }

                if best_c != c_u {
                    moved = true;
                }
                community[u] = best_c;
                comm_tot[best_c] += k_u;
            }

            if !moved {
                break;
            }
        }

        // ── Phase 2: refinement — split each community into connected
        // components of its induced subgraph. This is the Leiden guarantee:
        // every returned community is internally connected. ──
        let mut members: HashMap<usize, Vec<usize>> = HashMap::new();
        for (u, &c) in community.iter().enumerate() {
            members.entry(c).or_default().push(u);
        }

        let mut refined: Vec<usize> = vec![0; num_nodes];
        let mut seen: Vec<bool> = vec![false; num_nodes];
        let mut next_id = 0usize;

        for nodes in members.values() {
            let node_set: HashSet<usize> = nodes.iter().copied().collect();
            for &start in nodes {
                if seen[start] {
                    continue;
                }
                // Iterative DFS over the induced subgraph.
                let mut stack = vec![start];
                seen[start] = true;
                while let Some(x) = stack.pop() {
                    refined[x] = next_id;
                    for &(v, _w) in &adj[x] {
                        if node_set.contains(&v) && !seen[v] {
                            seen[v] = true;
                            stack.push(v);
                        }
                    }
                }
                next_id += 1;
            }
        }

        // ── Emit the refined partition ──
        let mut comm_groups: HashMap<usize, Vec<u64>> = HashMap::new();
        for (u, &c) in refined.iter().enumerate() {
            comm_groups.entry(c).or_default().push(reverse_map[u]);
        }

        let inner_builder = UInt64Builder::new();
        let component_builder = ListBuilder::new(inner_builder);
        let mut final_builder = ListBuilder::new(component_builder);

        for (_, mut group) in comm_groups {
            group.sort();
            for node in group {
                final_builder.values().values().append_value(node);
            }
            final_builder.values().append(true);
        }
        final_builder.append(true);

        Ok(ScalarValue::List(Arc::new(final_builder.finish())))
    }

    fn size(&self) -> usize {
        self.base.size() + self.weights.capacity() * 4
    }
}
