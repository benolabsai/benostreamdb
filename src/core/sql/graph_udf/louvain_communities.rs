// Copyright (c) 2026 Richard Albright. All rights reserved.

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
use std::collections::HashMap;
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
pub struct LouvainCommunitiesUDF {
    signature: Signature,
}
impl_dyn_traits!(LouvainCommunitiesUDF);

impl Default for LouvainCommunitiesUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl LouvainCommunitiesUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![
                    DataType::UInt64,
                    DataType::UInt64,
                    DataType::Float32,
                    DataType::Float32,
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for LouvainCommunitiesUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "louvain_communities"
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
        Ok(Box::new(LouvainAccumulator::new()))
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
            Arc::new(Field::new(
                "weights",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            )),
            Arc::new(Field::new("resolution", DataType::Float32, true)),
        ])
    }
}

#[derive(Debug)]
pub struct LouvainAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<f32>,
    resolution: f32,
}

impl LouvainAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
            weights: Vec::new(),
            resolution: 1.0,
        }
    }
}

impl Accumulator for LouvainAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() < 2 {
            return Err(DataFusionError::Execution(
                "louvain_communities expects at least 2 arguments (source, target)".to_string(),
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

        for i in 0..len {
            if sources_arr.is_valid(i) && targets_arr.is_valid(i) {
                self.sources.push(sources_arr.value(i));
                self.targets.push(targets_arr.value(i));
                self.weights.push(weights_vec[i]);
            }
        }

        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let sources_list = states[0]
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected ListArray for sources".to_string())
            })?;
        let targets_list = states[1]
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected ListArray for targets".to_string())
            })?;
        let weights_list = states[2]
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected ListArray for weights".to_string())
            })?;

        for i in 0..sources_list.len() {
            if sources_list.is_valid(i) {
                let s_arr = sources_list.value(i);
                if let Some(s) = s_arr.as_any().downcast_ref::<UInt64Array>() {
                    self.sources.extend_from_slice(s.values());
                }
            }
            if targets_list.is_valid(i) {
                let t_arr = targets_list.value(i);
                if let Some(t) = t_arr.as_any().downcast_ref::<UInt64Array>() {
                    self.targets.extend_from_slice(t.values());
                }
            }
            if weights_list.is_valid(i) {
                let w_arr = weights_list.value(i);
                if let Some(w) = w_arr.as_any().downcast_ref::<Float32Array>() {
                    self.weights.extend_from_slice(w.values());
                }
            }
        }

        if let Some(r_arr) = states
            .get(3)
            .and_then(|a| a.as_any().downcast_ref::<Float32Array>())
        {
            if !r_arr.is_empty() && r_arr.is_valid(0) {
                self.resolution = r_arr.value(0);
            }
        }

        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut sources_builder = arrow::array::ListBuilder::new(UInt64Builder::new());
        sources_builder.values().append_slice(&self.sources);
        sources_builder.append(true);

        let mut targets_builder = arrow::array::ListBuilder::new(UInt64Builder::new());
        targets_builder.values().append_slice(&self.targets);
        targets_builder.append(true);

        let mut weights_builder =
            arrow::array::ListBuilder::new(arrow::array::Float32Builder::new());
        weights_builder.values().append_slice(&self.weights);
        weights_builder.append(true);

        Ok(vec![
            ScalarValue::List(Arc::new(sources_builder.finish())),
            ScalarValue::List(Arc::new(targets_builder.finish())),
            ScalarValue::List(Arc::new(weights_builder.finish())),
            ScalarValue::Float32(Some(self.resolution)),
        ])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut node_map: HashMap<u64, usize> = HashMap::new();
        let mut reverse_map: Vec<u64> = Vec::new();

        for i in 0..self.sources.len() {
            let s = self.sources[i];
            let t = self.targets[i];
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

        for i in 0..self.sources.len() {
            let u = node_map[&self.sources[i]];
            let v = node_map[&self.targets[i]];
            let w = self.weights[i].max(0.0);

            adj[u].push((v, w));
            adj[v].push((u, w));
            node_degrees[u] += w;
            node_degrees[v] += w;
            total_weight += w;
        }

        let two_m = (total_weight * 2.0).max(1e-6);

        // Community assignment: community[u]
        let mut community: Vec<usize> = (0..num_nodes).collect();
        let mut comm_tot: Vec<f32> = node_degrees.clone();

        // Greedy modularity optimization passes
        for _pass in 0..15 {
            let mut moved = false;

            for u in 0..num_nodes {
                let c_u = community[u];
                let k_u = node_degrees[u];

                // Remove u from its current community
                comm_tot[c_u] -= k_u;

                // Sum weights to neighboring communities
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

        let mut comm_groups: HashMap<usize, Vec<u64>> = HashMap::new();
        for (u, &c) in community.iter().enumerate() {
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
        std::mem::size_of::<Self>()
            + self.sources.capacity() * 8
            + self.targets.capacity() * 8
            + self.weights.capacity() * 4
    }
}
