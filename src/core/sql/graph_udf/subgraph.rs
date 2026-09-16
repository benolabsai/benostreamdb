// Copyright (c) 2026 Richard Albright. All rights reserved.

use arrow::array::{Array, ArrayRef, Float32Array, Float32Builder, UInt64Array, UInt64Builder};
use arrow::datatypes::{DataType, Field};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{AggregateUDFImpl, Signature, Volatility};
use datafusion::scalar::ScalarValue;
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_functions_aggregate_common::accumulator::{AccumulatorArgs, StateFieldsArgs};
use petgraph::graphmap::DiGraphMap;
use std::any::Any;
use std::collections::{HashSet, VecDeque};
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
            signature: Signature::exact(
                vec![
                    DataType::UInt64,
                    DataType::UInt64,
                    DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                    DataType::UInt32,
                    DataType::Boolean,
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for SubgraphUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "subgraph"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        let struct_fields = vec![
            Arc::new(Field::new("source", DataType::UInt64, false)),
            Arc::new(Field::new("target", DataType::UInt64, false)),
            Arc::new(Field::new("weight", DataType::Float32, false)),
        ];
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(struct_fields.into()),
            true,
        ))))
    }
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(SubgraphAccumulator::new()))
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
            Arc::new(Field::new(
                "seeds",
                DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                true,
            )),
            Arc::new(Field::new("hops", DataType::UInt32, true)),
            Arc::new(Field::new("is_directed", DataType::Boolean, true)),
        ])
    }
}

#[derive(Debug)]
pub struct SubgraphAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<f32>,
    seeds: Vec<u64>,
    hops: u32,
    is_directed: bool,
}

impl SubgraphAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
            weights: Vec::new(),
            seeds: Vec::new(),
            hops: 1,
            is_directed: false,
        }
    }
}

impl Accumulator for SubgraphAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() < 3 {
            return Err(DataFusionError::Execution(
                "subgraph expects at least 3 arguments: source, target, seeds (or source, target, weight, seeds, hops)".to_string(),
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

        // Check if values[2] is weights (Float32/Float64) or seeds (ListArray or UInt64Array)
        let mut seeds_idx = 2;
        let mut hops_idx = 3;
        let mut directed_idx = 4;

        if let Some(w_arr) = values[2].as_any().downcast_ref::<Float32Array>() {
            seeds_idx = 3;
            hops_idx = 4;
            directed_idx = 5;
            for i in 0..len {
                if sources_arr.is_valid(i) && targets_arr.is_valid(i) {
                    self.sources.push(sources_arr.value(i));
                    self.targets.push(targets_arr.value(i));
                    self.weights.push(if w_arr.is_valid(i) {
                        w_arr.value(i)
                    } else {
                        1.0
                    });
                }
            }
        } else if let Some(w_arr) = values[2]
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
        {
            seeds_idx = 3;
            hops_idx = 4;
            directed_idx = 5;
            for i in 0..len {
                if sources_arr.is_valid(i) && targets_arr.is_valid(i) {
                    self.sources.push(sources_arr.value(i));
                    self.targets.push(targets_arr.value(i));
                    self.weights.push(if w_arr.is_valid(i) {
                        w_arr.value(i) as f32
                    } else {
                        1.0
                    });
                }
            }
        } else {
            for i in 0..len {
                if sources_arr.is_valid(i) && targets_arr.is_valid(i) {
                    self.sources.push(sources_arr.value(i));
                    self.targets.push(targets_arr.value(i));
                    self.weights.push(1.0);
                }
            }
        }

        // Parse seeds
        if self.seeds.is_empty() && values.len() > seeds_idx && !values[seeds_idx].is_empty() {
            if let Some(list_arr) = values[seeds_idx]
                .as_any()
                .downcast_ref::<arrow::array::ListArray>()
            {
                if list_arr.is_valid(0) {
                    let seed_values = list_arr.value(0);
                    if let Some(s_arr) = seed_values.as_any().downcast_ref::<UInt64Array>() {
                        self.seeds.extend_from_slice(s_arr.values());
                    }
                }
            } else if let Some(u_arr) = values[seeds_idx].as_any().downcast_ref::<UInt64Array>() {
                for i in 0..u_arr.len() {
                    if u_arr.is_valid(i) {
                        self.seeds.push(u_arr.value(i));
                    }
                }
            }
        }

        // Parse hops
        if values.len() > hops_idx && !values[hops_idx].is_empty() {
            if let Some(h_arr) = values[hops_idx]
                .as_any()
                .downcast_ref::<arrow::array::UInt32Array>()
            {
                if h_arr.is_valid(0) {
                    self.hops = h_arr.value(0);
                }
            } else if let Some(h_arr) = values[hops_idx]
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
            {
                if h_arr.is_valid(0) {
                    self.hops = h_arr.value(0) as u32;
                }
            }
        }

        // Parse is_directed
        if values.len() > directed_idx && !values[directed_idx].is_empty() {
            if let Some(b_arr) = values[directed_idx]
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
            {
                if b_arr.is_valid(0) {
                    self.is_directed = b_arr.value(0);
                }
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
        let seeds_list = states[3]
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected ListArray for seeds".to_string())
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

        if self.seeds.is_empty() {
            for i in 0..seeds_list.len() {
                if seeds_list.is_valid(i) {
                    let sd_arr = seeds_list.value(i);
                    if let Some(s) = sd_arr.as_any().downcast_ref::<UInt64Array>() {
                        self.seeds.extend_from_slice(s.values());
                    }
                }
            }
        }

        if let Some(h_arr) = states
            .get(4)
            .and_then(|a| a.as_any().downcast_ref::<arrow::array::UInt32Array>())
        {
            if !h_arr.is_empty() && h_arr.is_valid(0) {
                self.hops = h_arr.value(0);
            }
        }

        if let Some(b_arr) = states
            .get(5)
            .and_then(|a| a.as_any().downcast_ref::<arrow::array::BooleanArray>())
        {
            if !b_arr.is_empty() && b_arr.is_valid(0) {
                self.is_directed = b_arr.value(0);
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

        let mut weights_builder = arrow::array::ListBuilder::new(Float32Builder::new());
        weights_builder.values().append_slice(&self.weights);
        weights_builder.append(true);

        let mut seeds_builder = arrow::array::ListBuilder::new(UInt64Builder::new());
        seeds_builder.values().append_slice(&self.seeds);
        seeds_builder.append(true);

        Ok(vec![
            ScalarValue::List(Arc::new(sources_builder.finish())),
            ScalarValue::List(Arc::new(targets_builder.finish())),
            ScalarValue::List(Arc::new(weights_builder.finish())),
            ScalarValue::List(Arc::new(seeds_builder.finish())),
            ScalarValue::UInt32(Some(self.hops)),
            ScalarValue::Boolean(Some(self.is_directed)),
        ])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        // Multi-source BFS from seeds up to `hops` to discover induced vertex set V_sub
        let mut graph = DiGraphMap::<u64, ()>::new();
        for (&s, &t) in self.sources.iter().zip(self.targets.iter()) {
            graph.add_edge(s, t, ());
            if !self.is_directed {
                graph.add_edge(t, s, ());
            }
        }

        let mut visited_nodes = HashSet::new();
        let mut queue = VecDeque::new();

        for &seed in &self.seeds {
            if graph.contains_node(seed) {
                visited_nodes.insert(seed);
                queue.push_back((seed, 0));
            }
        }

        while let Some((node, depth)) = queue.pop_front() {
            if depth < self.hops {
                for neighbor in graph.neighbors(node) {
                    if visited_nodes.insert(neighbor) {
                        queue.push_back((neighbor, depth + 1));
                    }
                }
            }
        }

        // Induced subgraph: retain all original edges where both endpoints are in visited_nodes
        let mut out_sources = Vec::new();
        let mut out_targets = Vec::new();
        let mut out_weights = Vec::new();

        for i in 0..self.sources.len() {
            let s = self.sources[i];
            let t = self.targets[i];
            let w = self.weights[i];
            if visited_nodes.contains(&s) && visited_nodes.contains(&t) {
                out_sources.push(s);
                out_targets.push(t);
                out_weights.push(w);
            }
        }

        let struct_fields = vec![
            Arc::new(Field::new("source", DataType::UInt64, false)),
            Arc::new(Field::new("target", DataType::UInt64, false)),
            Arc::new(Field::new("weight", DataType::Float32, false)),
        ];

        let mut s_builder = UInt64Builder::new();
        let mut t_builder = UInt64Builder::new();
        let mut w_builder = Float32Builder::new();

        s_builder.append_slice(&out_sources);
        t_builder.append_slice(&out_targets);
        w_builder.append_slice(&out_weights);

        let num_edges = out_sources.len();
        let struct_array = arrow::array::StructArray::from(vec![
            (
                struct_fields[0].clone(),
                Arc::new(s_builder.finish()) as ArrayRef,
            ),
            (
                struct_fields[1].clone(),
                Arc::new(t_builder.finish()) as ArrayRef,
            ),
            (
                struct_fields[2].clone(),
                Arc::new(w_builder.finish()) as ArrayRef,
            ),
        ]);

        let list_fields = Arc::new(Field::new(
            "item",
            DataType::Struct(struct_fields.into()),
            true,
        ));
        let offsets = arrow::buffer::OffsetBuffer::from_lengths(vec![num_edges]);
        let list_array =
            arrow::array::ListArray::new(list_fields, offsets, Arc::new(struct_array), None);

        Ok(ScalarValue::List(Arc::new(list_array)))
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self)
            + self.sources.capacity() * std::mem::size_of::<u64>()
            + self.targets.capacity() * std::mem::size_of::<u64>()
            + self.weights.capacity() * std::mem::size_of::<f32>()
            + self.seeds.capacity() * std::mem::size_of::<u64>()
    }
}
