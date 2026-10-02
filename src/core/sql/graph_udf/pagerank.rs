// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::sql::graph_udf::graph_view::GraphAccumulatorBase;
use ahash::AHashMap as HashMap;
use arrow::array::{Array, ArrayRef, ListBuilder, StructBuilder, UInt64Builder};
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
pub struct PageRankUDF {
    signature: Signature,
}
impl_dyn_traits!(PageRankUDF);

impl Default for PageRankUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl PageRankUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for PageRankUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "graph_pagerank"
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
        Ok(Box::new(PageRankAccumulator::new()))
    }

    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        let mut fields = GraphAccumulatorBase::state_fields();
        fields.push(Arc::new(Field::new("damping", DataType::Float64, true)));
        fields.push(Arc::new(Field::new("iterations", DataType::UInt32, true)));
        Ok(fields)
    }
}

#[derive(Debug)]
pub struct PageRankAccumulator {
    base: GraphAccumulatorBase,
    damping: f64,
    iterations: u32,
}

impl PageRankAccumulator {
    fn new() -> Self {
        Self {
            base: GraphAccumulatorBase::new(),
            damping: 0.85,
            iterations: 30,
        }
    }
}

impl Accumulator for PageRankAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut state = self.base.edge_state()?;
        state.push(ScalarValue::Float64(Some(self.damping)));
        state.push(ScalarValue::UInt32(Some(self.iterations)));
        Ok(state)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.base.merge_edge_state(states, Some(2), Some(3))?;
        // For backwards compatibility or dynamic trailing args
        let damping_idx = states.len().saturating_sub(2).max(4);
        let iter_idx = states.len().saturating_sub(1).max(5);

        if states.len() > damping_idx {
            if let Some(damping_arr) = states[damping_idx]
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
            {
                if !damping_arr.is_empty() && damping_arr.is_valid(0) {
                    self.damping = damping_arr.value(0);
                }
            }
        }
        if states.len() > iter_idx {
            if let Some(iterations_arr) = states[iter_idx]
                .as_any()
                .downcast_ref::<arrow::array::UInt32Array>()
            {
                if !iterations_arr.is_empty() && iterations_arr.is_valid(0) {
                    self.iterations = iterations_arr.value(0);
                }
            }
        }

        Ok(())
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() < 2 {
            return Err(DataFusionError::Execution(
                "pagerank expects at least 2 arguments (source, target)".to_string(),
            ));
        }

        self.base.update_edge_batch(values, Some(4), Some(5))?;

        if values.len() > 2 && !values[2].is_empty() {
            if let Some(arr) = values[2]
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
            {
                if arr.is_valid(0) {
                    self.damping = arr.value(0);
                }
            }
        }

        if values.len() > 3 && !values[3].is_empty() {
            if let Some(arr) = values[3]
                .as_any()
                .downcast_ref::<arrow::array::UInt32Array>()
            {
                if arr.is_valid(0) {
                    self.iterations = arr.value(0);
                }
            }
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let graph = self.base.resolve_graph(&[], 0)?;
        // Rank every node, not just sources: NetworkX normalizes over the full
        // node set (sources ∪ sinks) and redistributes dangling mass across it.
        // `all_nodes()` returns only source nodes, so fold in the targets too.
        let mut nodes = graph.all_nodes();
        for (_, v) in graph.all_edges() {
            nodes.push(v);
        }
        nodes.sort_unstable();
        nodes.dedup();

        let num_nodes = nodes.len() as f64;
        let mut scores: HashMap<u64, f64> = nodes.iter().map(|&n| (n, 1.0 / num_nodes)).collect();

        let mut neighbors: Vec<u64> = Vec::new();
        for _ in 0..self.iterations {
            let mut new_scores: HashMap<u64, f64> = nodes
                .iter()
                .map(|&n| (n, (1.0 - self.damping) / num_nodes))
                .collect();

            for &u in &nodes {
                let current_score = scores[&u];
                neighbors.clear();
                graph.get_neighbors_into(u, &mut neighbors);
                if !neighbors.is_empty() {
                    let transfer = (self.damping * current_score) / (neighbors.len() as f64);
                    for &v in &neighbors {
                        *new_scores.entry(v).or_insert(0.0) += transfer;
                    }
                } else {
                    // dangling node
                    let transfer = (self.damping * current_score) / num_nodes;
                    for &v in &nodes {
                        *new_scores.entry(v).or_insert(0.0) += transfer;
                    }
                }
            }
            scores = new_scores;
        }

        // Return a List of Structs {node: UInt64, score: Float64}
        let struct_fields = Fields::from(vec![
            Field::new("node", DataType::UInt64, false),
            Field::new("score", DataType::Float64, false),
        ]);

        let mut node_builder = UInt64Builder::new();
        let mut score_builder = arrow::array::Float64Builder::new();

        let mut sorted_nodes: Vec<_> = scores.keys().copied().collect();
        sorted_nodes.sort_unstable();

        for &node in &sorted_nodes {
            node_builder.append_value(node);
            score_builder.append_value(scores[&node]);
        }

        let mut struct_builder = StructBuilder::new(
            struct_fields.clone(),
            vec![Box::new(node_builder), Box::new(score_builder)],
        );

        // One struct row per scored node. This is `sorted_nodes.len()`, not
        // `nodes.len()`: `all_nodes()` returns only *source* nodes, but the
        // PageRank update also scores *sink* targets reached as `new_scores`
        // entries, so `scores` can hold more nodes than `nodes`.
        for _ in 0..sorted_nodes.len() {
            struct_builder.append(true);
        }

        let mut list_builder = ListBuilder::new(struct_builder);
        // We appended nodes.len() elements to the internal struct builder.
        // We now append one list element that spans all of those.
        list_builder.append(true);

        let list_array = list_builder.finish();
        Ok(ScalarValue::List(Arc::new(list_array)))
    }

    fn size(&self) -> usize {
        self.base.size() + std::mem::size_of::<f64>() + std::mem::size_of::<u32>()
    }
}
