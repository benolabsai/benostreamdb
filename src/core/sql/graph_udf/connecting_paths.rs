// Copyright (c) 2026 Richard Albright. All rights reserved.

use arrow::array::{Array, ArrayRef, UInt32Builder, UInt64Array, UInt64Builder};
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
            signature: Signature::exact(
                vec![
                    DataType::UInt64,
                    DataType::UInt64,
                    DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                    DataType::Boolean,
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for ConnectingPathsUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "connecting_paths"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        let struct_fields = vec![
            Arc::new(Field::new("source", DataType::UInt64, false)),
            Arc::new(Field::new("target", DataType::UInt64, false)),
            Arc::new(Field::new("path_index", DataType::UInt32, false)),
        ];
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(struct_fields.into()),
            true,
        ))))
    }
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(ConnectingPathsAccumulator::new()))
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
                "seeds",
                DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                true,
            )),
            Arc::new(Field::new("is_directed", DataType::Boolean, true)),
        ])
    }
}

#[derive(Debug)]
pub struct ConnectingPathsAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
    seeds: Vec<u64>,
    is_directed: bool,
}

impl ConnectingPathsAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
            seeds: Vec::new(),
            is_directed: false,
        }
    }
}

impl Accumulator for ConnectingPathsAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() < 3 {
            return Err(DataFusionError::Execution(
                "connecting_paths expects at least 3 arguments: source, target, seeds".to_string(),
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

        // Extract seeds from values[2]
        if self.seeds.is_empty() && !values[2].is_empty() {
            if let Some(list_arr) = values[2].as_any().downcast_ref::<arrow::array::ListArray>() {
                if list_arr.is_valid(0) {
                    let seed_values = list_arr.value(0);
                    if let Some(s_arr) = seed_values.as_any().downcast_ref::<UInt64Array>() {
                        self.seeds.extend_from_slice(s_arr.values());
                    }
                }
            } else if let Some(u_arr) = values[2].as_any().downcast_ref::<UInt64Array>() {
                for i in 0..u_arr.len() {
                    if u_arr.is_valid(i) {
                        self.seeds.push(u_arr.value(i));
                    }
                }
            }
        }

        // Optional is_directed
        if values.len() > 3 && !values[3].is_empty() {
            if let Some(b_arr) = values[3]
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
            {
                if b_arr.is_valid(0) {
                    self.is_directed = b_arr.value(0);
                }
            }
        }

        for i in 0..sources_arr.len() {
            if sources_arr.is_valid(i) && targets_arr.is_valid(i) {
                self.sources.push(sources_arr.value(i));
                self.targets.push(targets_arr.value(i));
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
        let seeds_list = states[2]
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

        if let Some(b_arr) = states
            .get(3)
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

        let mut seeds_builder = arrow::array::ListBuilder::new(UInt64Builder::new());
        seeds_builder.values().append_slice(&self.seeds);
        seeds_builder.append(true);

        Ok(vec![
            ScalarValue::List(Arc::new(sources_builder.finish())),
            ScalarValue::List(Arc::new(targets_builder.finish())),
            ScalarValue::List(Arc::new(seeds_builder.finish())),
            ScalarValue::Boolean(Some(self.is_directed)),
        ])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut graph = DiGraphMap::<u64, ()>::new();
        for (&s, &t) in self.sources.iter().zip(self.targets.iter()) {
            graph.add_edge(s, t, ());
            if !self.is_directed {
                graph.add_edge(t, s, ());
            }
        }

        let mut out_sources = Vec::new();
        let mut out_targets = Vec::new();
        let mut out_path_indices = Vec::new();
        let mut seen_edges = HashSet::new();

        let unique_seeds: Vec<u64> = {
            let mut s = self.seeds.clone();
            s.dedup();
            s
        };

        let mut path_idx = 0u32;

        // Find shortest path between all seed pairs (s_i, s_j)
        for i in 0..unique_seeds.len() {
            for j in (i + 1)..unique_seeds.len() {
                let start = unique_seeds[i];
                let goal = unique_seeds[j];

                if !graph.contains_node(start) || !graph.contains_node(goal) {
                    continue;
                }

                // BFS to find unweighted shortest path from start to goal
                let mut parent_map: std::collections::HashMap<u64, u64> =
                    std::collections::HashMap::new();
                let mut queue = VecDeque::new();
                let mut visited = HashSet::new();

                queue.push_back(start);
                visited.insert(start);
                let mut found = false;

                while let Some(curr) = queue.pop_front() {
                    if curr == goal {
                        found = true;
                        break;
                    }
                    for neighbor in graph.neighbors(curr) {
                        if visited.insert(neighbor) {
                            parent_map.insert(neighbor, curr);
                            queue.push_back(neighbor);
                        }
                    }
                }

                if found {
                    // Reconstruct path from goal back to start
                    let mut curr = goal;
                    let mut path_edges = Vec::new();

                    while let Some(&prev) = parent_map.get(&curr) {
                        path_edges.push((prev, curr));
                        curr = prev;
                        if curr == start {
                            break;
                        }
                    }
                    path_edges.reverse();

                    for (u, v) in path_edges {
                        let edge_key = if self.is_directed {
                            (u, v)
                        } else {
                            (u.min(v), u.max(v))
                        };
                        if seen_edges.insert(edge_key) {
                            out_sources.push(u);
                            out_targets.push(v);
                            out_path_indices.push(path_idx);
                        }
                    }
                    path_idx += 1;
                }
            }
        }

        let struct_fields = vec![
            Arc::new(Field::new("source", DataType::UInt64, false)),
            Arc::new(Field::new("target", DataType::UInt64, false)),
            Arc::new(Field::new("path_index", DataType::UInt32, false)),
        ];

        let mut s_builder = UInt64Builder::new();
        let mut t_builder = UInt64Builder::new();
        let mut p_builder = UInt32Builder::new();

        s_builder.append_slice(&out_sources);
        t_builder.append_slice(&out_targets);
        p_builder.append_slice(&out_path_indices);

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
                Arc::new(p_builder.finish()) as ArrayRef,
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
            + self.seeds.capacity() * std::mem::size_of::<u64>()
    }
}
