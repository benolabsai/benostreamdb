// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

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
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for PersonalizedPageRankUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "graph_personalized_pagerank"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        let struct_fields = vec![
            Field::new("node", DataType::UInt64, false),
            Field::new("score", DataType::Float64, false),
        ];
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(Fields::from(struct_fields)),
            true,
        ))))
    }

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(PersonalizedPageRankAccumulator::new()))
    }

    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        let mut fields = GraphAccumulatorBase::state_fields();
        fields.push(Arc::new(Field::new(
            "seeds",
            DataType::List(Arc::new(Field::new("item", DataType::UInt64, true))),
            true,
        )));
        fields.push(Arc::new(Field::new("damping", DataType::Float64, true)));
        fields.push(Arc::new(Field::new("iterations", DataType::UInt32, true)));
        fields.push(Arc::new(Field::new("directed", DataType::Boolean, true)));
        fields.push(Arc::new(Field::new(
            "seed_weights",
            DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
            true,
        )));
        Ok(fields)
    }
}

#[derive(Debug)]
pub struct PersonalizedPageRankAccumulator {
    base: GraphAccumulatorBase,
    seeds: Vec<u64>,
    damping: f64,
    iterations: u32,
    directed: bool,
    seed_weights: Option<Vec<f64>>,
}

impl PersonalizedPageRankAccumulator {
    fn new() -> Self {
        Self {
            base: GraphAccumulatorBase::new(),
            seeds: Vec::new(),
            damping: 0.85,
            iterations: 30,
            directed: false,
            seed_weights: None,
        }
    }
}

impl Accumulator for PersonalizedPageRankAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut state = self.base.edge_state()?;

        let mut seeds_builder = arrow::array::ListBuilder::new(arrow::array::UInt64Builder::new());
        seeds_builder.values().append_slice(&self.seeds);
        seeds_builder.append(true);

        let seed_weights_val = if let Some(weights) = &self.seed_weights {
            let mut weights_builder =
                arrow::array::ListBuilder::new(arrow::array::Float64Builder::new());
            weights_builder.values().append_slice(weights);
            weights_builder.append(true);
            ScalarValue::List(Arc::new(weights_builder.finish()))
        } else {
            let mut weights_builder =
                arrow::array::ListBuilder::new(arrow::array::Float64Builder::new());
            weights_builder.append(true);
            ScalarValue::List(Arc::new(weights_builder.finish()))
        };

        state.push(ScalarValue::List(Arc::new(seeds_builder.finish())));
        state.push(ScalarValue::Float64(Some(self.damping)));
        state.push(ScalarValue::UInt32(Some(self.iterations)));
        state.push(ScalarValue::Boolean(Some(self.directed)));
        state.push(seed_weights_val);

        Ok(state)
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() < 3 {
            return Err(DataFusionError::Execution(
                "personalized_pagerank expects at least 3 arguments".to_string(),
            ));
        }

        self.base.update_edge_batch(values, Some(7), Some(8))?;

        // seeds is a ListArray
        if let Some(seeds_list) = values[2].as_any().downcast_ref::<ListArray>() {
            if seeds_list.len() > 0 && seeds_list.is_valid(0) {
                let list_values = seeds_list.value(0);
                if let Some(uint_values) = list_values.as_any().downcast_ref::<UInt64Array>() {
                    self.seeds.clear();
                    for i in 0..uint_values.len() {
                        if uint_values.is_valid(i) {
                            self.seeds.push(uint_values.value(i));
                        }
                    }
                }
            }
        }

        if values.len() > 3 && !values[3].is_empty() {
            if let Some(arr) = values[3].as_any().downcast_ref::<Float64Array>() {
                if arr.is_valid(0) {
                    self.damping = arr.value(0);
                }
            }
        }

        if values.len() > 4 && !values[4].is_empty() {
            if let Some(arr) = values[4].as_any().downcast_ref::<UInt32Array>() {
                if arr.is_valid(0) {
                    self.iterations = arr.value(0);
                }
            }
        }

        if values.len() > 5 && !values[5].is_empty() {
            if let Some(arr) = values[5].as_any().downcast_ref::<BooleanArray>() {
                if arr.is_valid(0) {
                    self.directed = arr.value(0);
                }
            }
        }

        if values.len() > 6 && !values[6].is_empty() {
            if let Some(weights_list) = values[6].as_any().downcast_ref::<ListArray>() {
                if weights_list.len() > 0 && weights_list.is_valid(0) {
                    let list_values = weights_list.value(0);
                    if let Some(float_values) = list_values.as_any().downcast_ref::<Float64Array>()
                    {
                        let mut sw = Vec::new();
                        for i in 0..float_values.len() {
                            if float_values.is_valid(i) {
                                sw.push(float_values.value(i));
                            }
                        }
                        if sw.len() == self.seeds.len() {
                            self.seed_weights = Some(sw);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.is_empty() {
            return Ok(());
        }
        self.base.merge_edge_state(states, Some(2), Some(3))?;
        // An empty input partition emits default scalar args (damping=0.85,
        // iterations=30, directed=false); adopting them would clobber the real
        // arguments and make the result depend on merge order.
        if !GraphAccumulatorBase::state_has_edges(states) {
            return Ok(());
        }
        if states.len() <= 4 {
            return Ok(());
        }

        let seeds_list = states[4]
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Execution(
                    "personalized_pagerank: expected ListArray for seeds".to_string(),
                )
            })?;
        let damping_arr = states[5]
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Execution(
                    "personalized_pagerank: expected Float64Array for damping".to_string(),
                )
            })?;
        let iterations_arr = states[6]
            .as_any()
            .downcast_ref::<arrow::array::UInt32Array>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Execution(
                    "personalized_pagerank: expected UInt32Array for iterations".to_string(),
                )
            })?;
        let directed_arr = states[7]
            .as_any()
            .downcast_ref::<arrow::array::BooleanArray>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Execution(
                    "personalized_pagerank: expected BooleanArray for directed".to_string(),
                )
            })?;
        let seed_weights_list = states[8]
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .ok_or_else(|| {
                datafusion::error::DataFusionError::Execution(
                    "personalized_pagerank: expected ListArray for seed weights".to_string(),
                )
            })?;

        if !seeds_list.is_empty() && seeds_list.is_valid(0) {
            let s_arr = seeds_list.value(0);
            if let Some(s) = s_arr.as_any().downcast_ref::<arrow::array::UInt64Array>() {
                if !s.is_empty() && self.seeds.is_empty() {
                    self.seeds.extend_from_slice(s.values());
                }
            }
        }

        if !seed_weights_list.is_empty() && seed_weights_list.is_valid(0) {
            let w_arr = seed_weights_list.value(0);
            if let Some(w) = w_arr.as_any().downcast_ref::<arrow::array::Float64Array>() {
                if !w.is_empty() && self.seed_weights.is_none() {
                    self.seed_weights = Some(w.values().to_vec());
                }
            }
        }

        if !damping_arr.is_empty() && damping_arr.is_valid(0) {
            self.damping = damping_arr.value(0);
        }
        if !iterations_arr.is_empty() && iterations_arr.is_valid(0) {
            self.iterations = iterations_arr.value(0);
        }
        if !directed_arr.is_empty() && directed_arr.is_valid(0) {
            self.directed = directed_arr.value(0);
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let graph = self.base.resolve_graph(&self.seeds, self.iterations)?;

        let mut all_edges = Vec::new();
        if !self.base.is_empty() {
            for (_, u, v) in self.base.edges() {
                all_edges.push((u, v));
            }
        } else {
            all_edges = graph.all_edges();
        }

        if all_edges.is_empty() || self.seeds.is_empty() {
            let struct_fields = Fields::from(vec![
                Field::new("node", DataType::UInt64, false),
                Field::new("score", DataType::Float64, false),
            ]);
            let struct_builder = StructBuilder::new(
                struct_fields,
                vec![
                    Box::new(UInt64Builder::new()),
                    Box::new(arrow::array::Float64Builder::new()),
                ],
            );
            let mut list_builder = ListBuilder::new(struct_builder);
            list_builder.append(true);
            return Ok(ScalarValue::List(Arc::new(list_builder.finish())));
        }

        let mut adjacency: HashMap<u64, Vec<u64>> = HashMap::new();
        let mut nodes: Vec<u64> = Vec::new();
        for &(u, v) in &all_edges {
            adjacency.entry(u).or_default().push(v);
            if !self.directed {
                adjacency.entry(v).or_default().push(u);
            }
            nodes.push(u);
            nodes.push(v);
        }
        nodes.sort_unstable();
        nodes.dedup();

        let _num_nodes = nodes.len() as f64;
        let mut scores: HashMap<u64, f64> = HashMap::new();

        // initialize scores
        let total_weight = if let Some(sw) = &self.seed_weights {
            sw.iter().sum::<f64>()
        } else {
            self.seeds.len() as f64
        };

        for (i, &seed) in self.seeds.iter().enumerate() {
            let weight = if let Some(sw) = &self.seed_weights {
                sw[i]
            } else {
                1.0
            };
            scores.insert(seed, weight / total_weight);
        }

        for _ in 0..self.iterations {
            let mut new_scores: HashMap<u64, f64> = HashMap::new();

            // Random jump back to seeds
            for (i, &seed) in self.seeds.iter().enumerate() {
                let weight = if let Some(sw) = &self.seed_weights {
                    sw[i]
                } else {
                    1.0
                };
                *new_scores.entry(seed).or_insert(0.0) +=
                    (1.0 - self.damping) * (weight / total_weight);
            }

            for &u in &nodes {
                let current_score = *scores.get(&u).unwrap_or(&0.0);
                if current_score == 0.0 {
                    continue;
                }

                if let Some(neighbors) = adjacency.get(&u) {
                    let transfer = (self.damping * current_score) / (neighbors.len() as f64);
                    for &v in neighbors {
                        *new_scores.entry(v).or_insert(0.0) += transfer;
                    }
                } else {
                    // dangling node
                    let transfer = (self.damping * current_score) / total_weight;
                    for (i, &seed) in self.seeds.iter().enumerate() {
                        let weight = if let Some(sw) = &self.seed_weights {
                            sw[i]
                        } else {
                            1.0
                        };
                        *new_scores.entry(seed).or_insert(0.0) += transfer * weight;
                    }
                }
            }
            scores = new_scores;
        }

        let struct_fields = Fields::from(vec![
            Field::new("node", DataType::UInt64, false),
            Field::new("score", DataType::Float64, false),
        ]);

        let mut node_builder = UInt64Builder::new();
        let mut score_builder = arrow::array::Float64Builder::new();

        let mut sorted_nodes: Vec<_> = scores.keys().copied().collect();
        sorted_nodes.sort_unstable();

        for node in sorted_nodes {
            let score = scores[&node];
            node_builder.append_value(node);
            score_builder.append_value(score);
        }

        let mut struct_builder = StructBuilder::new(
            struct_fields.clone(),
            vec![Box::new(node_builder), Box::new(score_builder)],
        );

        for _ in 0..scores.len() {
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
