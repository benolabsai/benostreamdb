// Copyright (c) 2026 Richard Albright. All rights reserved.

use arrow::array::{Array, ArrayRef, Float32Builder, UInt64Array, UInt64Builder};
use arrow::datatypes::{DataType, Field};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{AggregateUDFImpl, Signature, TypeSignature, Volatility};
use datafusion::scalar::ScalarValue;
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_functions_aggregate_common::accumulator::{AccumulatorArgs, StateFieldsArgs};
use petgraph::graphmap::DiGraphMap;
use petgraph::Direction;
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
pub struct PersonalizedPageRankUDF {
    signature: Signature,
}
impl_dyn_traits!(PersonalizedPageRankUDF);

impl Default for PersonalizedPageRankUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl PersonalizedPageRankUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![
                        DataType::UInt64,
                        DataType::UInt64,
                        DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                        DataType::Float64,
                        DataType::UInt32,
                        DataType::Boolean,
                    ]),
                    TypeSignature::Exact(vec![
                        DataType::UInt64,
                        DataType::UInt64,
                        DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
                        DataType::Float64,
                        DataType::UInt32,
                        DataType::Boolean,
                        DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
                    ]),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for PersonalizedPageRankUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "personalized_pagerank"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        let struct_fields = vec![
            Arc::new(Field::new("node", DataType::UInt64, false)),
            Arc::new(Field::new("score", DataType::Float32, false)),
        ];
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(struct_fields.into()),
            true,
        ))))
    }
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(PersonalizedPageRankAccumulator::new()))
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
            Arc::new(Field::new("damping", DataType::Float64, true)),
            Arc::new(Field::new("iterations", DataType::Int64, true)),
            Arc::new(Field::new("is_directed", DataType::Boolean, true)),
            Arc::new(Field::new(
                "seed_weights",
                DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
                true,
            )),
        ])
    }
}

#[derive(Debug)]
pub struct PersonalizedPageRankAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
    seeds: Vec<u64>,
    damping: f32,
    iterations: u32,
    is_directed: bool,
    seed_weights: Vec<f32>,
}

impl PersonalizedPageRankAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
            seeds: Vec::new(),
            damping: 0.85,
            iterations: 30,
            is_directed: false,
            seed_weights: Vec::new(),
        }
    }
}

impl Accumulator for PersonalizedPageRankAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() < 3 {
            return Err(DataFusionError::Execution(
                "personalized_pagerank expects at least 3 arguments: source, target, seeds"
                    .to_string(),
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

        if values.len() > 3 && !values[3].is_empty() {
            if let Some(d_arr) = values[3]
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
            {
                if d_arr.is_valid(0) {
                    self.damping = d_arr.value(0) as f32;
                }
            } else if let Some(d_arr) = values[3]
                .as_any()
                .downcast_ref::<arrow::array::Float32Array>()
            {
                if d_arr.is_valid(0) {
                    self.damping = d_arr.value(0);
                }
            }
        }

        if values.len() > 4 && !values[4].is_empty() {
            if let Some(i_arr) = values[4]
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
            {
                if i_arr.is_valid(0) {
                    self.iterations = i_arr.value(0) as u32;
                }
            } else if let Some(i_arr) = values[4]
                .as_any()
                .downcast_ref::<arrow::array::UInt32Array>()
            {
                if i_arr.is_valid(0) {
                    self.iterations = i_arr.value(0);
                }
            }
        }

        if values.len() > 5 && !values[5].is_empty() {
            if let Some(b_arr) = values[5]
                .as_any()
                .downcast_ref::<arrow::array::BooleanArray>()
            {
                if b_arr.is_valid(0) {
                    self.is_directed = b_arr.value(0);
                }
            }
        }

        if self.seed_weights.is_empty() && values.len() > 6 && !values[6].is_empty() {
            if let Some(list_arr) = values[6].as_any().downcast_ref::<arrow::array::ListArray>() {
                if list_arr.is_valid(0) {
                    let weight_values = list_arr.value(0);
                    if let Some(w_arr) = weight_values
                        .as_any()
                        .downcast_ref::<arrow::array::Float64Array>()
                    {
                        for i in 0..w_arr.len() {
                            if w_arr.is_valid(i) {
                                self.seed_weights.push(w_arr.value(i) as f32);
                            }
                        }
                    } else if let Some(w_arr) = weight_values
                        .as_any()
                        .downcast_ref::<arrow::array::Float32Array>()
                    {
                        for i in 0..w_arr.len() {
                            if w_arr.is_valid(i) {
                                self.seed_weights.push(w_arr.value(i));
                            }
                        }
                    }
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

        if let Some(d_arr) = states
            .get(3)
            .and_then(|a| a.as_any().downcast_ref::<arrow::array::Float64Array>())
        {
            if !d_arr.is_empty() && d_arr.is_valid(0) {
                self.damping = d_arr.value(0) as f32;
            }
        }
        if let Some(i_arr) = states
            .get(4)
            .and_then(|a| a.as_any().downcast_ref::<arrow::array::Int64Array>())
        {
            if !i_arr.is_empty() && i_arr.is_valid(0) {
                self.iterations = i_arr.value(0) as u32;
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

        if self.seed_weights.is_empty() && states.len() > 6 {
            if let Some(weights_list) = states[6].as_any().downcast_ref::<arrow::array::ListArray>()
            {
                for i in 0..weights_list.len() {
                    if weights_list.is_valid(i) {
                        let w_arr = weights_list.value(i);
                        if let Some(f_arr) =
                            w_arr.as_any().downcast_ref::<arrow::array::Float32Array>()
                        {
                            self.seed_weights.extend_from_slice(f_arr.values());
                        }
                    }
                }
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

        let mut weights_builder = arrow::array::ListBuilder::new(Float32Builder::new());
        weights_builder.values().append_slice(&self.seed_weights);
        weights_builder.append(true);

        Ok(vec![
            ScalarValue::List(Arc::new(sources_builder.finish())),
            ScalarValue::List(Arc::new(targets_builder.finish())),
            ScalarValue::List(Arc::new(seeds_builder.finish())),
            ScalarValue::Float64(Some(self.damping as f64)),
            ScalarValue::Int64(Some(self.iterations as i64)),
            ScalarValue::Boolean(Some(self.is_directed)),
            ScalarValue::List(Arc::new(weights_builder.finish())),
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

        let nodes: Vec<u64> = graph.nodes().collect();
        let num_nodes = nodes.len();

        let struct_fields = vec![
            Arc::new(Field::new("node", DataType::UInt64, false)),
            Arc::new(Field::new("score", DataType::Float32, false)),
        ];

        if num_nodes == 0 {
            let empty_struct = arrow::array::StructArray::from(vec![
                (
                    struct_fields[0].clone(),
                    Arc::new(UInt64Builder::new().finish()) as ArrayRef,
                ),
                (
                    struct_fields[1].clone(),
                    Arc::new(Float32Builder::new().finish()) as ArrayRef,
                ),
            ]);
            let offsets = arrow::buffer::OffsetBuffer::from_lengths(vec![0]);
            let list_fields = Arc::new(Field::new(
                "item",
                DataType::Struct(struct_fields.into()),
                true,
            ));
            return Ok(ScalarValue::List(Arc::new(arrow::array::ListArray::new(
                list_fields,
                offsets,
                Arc::new(empty_struct),
                None,
            ))));
        }

        let seed_set: HashSet<u64> = self.seeds.iter().copied().collect();
        let active_seeds: Vec<u64> = self
            .seeds
            .iter()
            .filter(|&&s| graph.contains_node(s))
            .copied()
            .collect();

        let p0: HashMap<u64, f32> = if active_seeds.is_empty() {
            let uniform = 1.0 / (num_nodes as f32);
            nodes.iter().map(|&n| (n, uniform)).collect()
        } else if self.seed_weights.len() == self.seeds.len() {
            let seed_weight_map: HashMap<u64, f32> = self
                .seeds
                .iter()
                .zip(self.seed_weights.iter())
                .map(|(&s, &w)| (s, w.max(0.0)))
                .collect();
            let total_active_weight: f32 = active_seeds
                .iter()
                .map(|s| seed_weight_map.get(s).copied().unwrap_or(0.0))
                .sum();
            if total_active_weight <= 0.0 {
                let seed_prob = 1.0 / (active_seeds.len() as f32);
                nodes
                    .iter()
                    .map(|&n| {
                        (
                            n,
                            if seed_set.contains(&n) {
                                seed_prob
                            } else {
                                0.0
                            },
                        )
                    })
                    .collect()
            } else {
                nodes
                    .iter()
                    .map(|&n| {
                        let prob = if seed_set.contains(&n) {
                            seed_weight_map.get(&n).copied().unwrap_or(0.0) / total_active_weight
                        } else {
                            0.0
                        };
                        (n, prob)
                    })
                    .collect()
            }
        } else {
            let seed_prob = 1.0 / (active_seeds.len() as f32);
            nodes
                .iter()
                .map(|&n| {
                    (
                        n,
                        if seed_set.contains(&n) {
                            seed_prob
                        } else {
                            0.0
                        },
                    )
                })
                .collect()
        };

        let mut scores: HashMap<u64, f32> = p0.clone();
        let mut out_degrees: HashMap<u64, usize> = HashMap::new();
        for &node in &nodes {
            out_degrees.insert(
                node,
                graph.edges_directed(node, Direction::Outgoing).count(),
            );
        }

        for _ in 0..self.iterations {
            let mut new_scores = HashMap::with_capacity(num_nodes);
            let mut dangling_mass = 0.0f32;

            for &node in &nodes {
                let out_deg = *out_degrees.get(&node).unwrap_or(&0);
                if out_deg == 0 {
                    dangling_mass += *scores.get(&node).unwrap_or(&0.0);
                }
            }

            for &node in &nodes {
                let mut sum = 0.0;
                for incoming in graph.neighbors_directed(node, Direction::Incoming) {
                    let out_deg = *out_degrees.get(&incoming).unwrap_or(&0);
                    if out_deg > 0 {
                        sum += scores.get(&incoming).unwrap_or(&0.0) / (out_deg as f32);
                    }
                }

                let restart_val = *p0.get(&node).unwrap_or(&0.0);
                let new_score = (1.0 - self.damping) * restart_val
                    + self.damping * (sum + dangling_mass * restart_val);
                new_scores.insert(node, new_score);
            }

            scores = new_scores;
        }

        let mut sorted_nodes = nodes;
        sorted_nodes.sort_by(|a, b| {
            let sa = scores.get(a).unwrap_or(&0.0);
            let sb = scores.get(b).unwrap_or(&0.0);
            sb.partial_cmp(sa).unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut node_id_builder = UInt64Builder::new();
        let mut score_builder = Float32Builder::new();

        for &node in &sorted_nodes {
            node_id_builder.append_value(node);
            score_builder.append_value(*scores.get(&node).unwrap_or(&0.0));
        }

        let node_id_array = Arc::new(node_id_builder.finish()) as ArrayRef;
        let score_array = Arc::new(score_builder.finish()) as ArrayRef;

        let struct_array = arrow::array::StructArray::from(vec![
            (struct_fields[0].clone(), node_id_array),
            (struct_fields[1].clone(), score_array),
        ]);

        let list_fields = Arc::new(Field::new(
            "item",
            DataType::Struct(struct_fields.into()),
            true,
        ));
        let offsets = arrow::buffer::OffsetBuffer::from_lengths(vec![num_nodes]);
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
