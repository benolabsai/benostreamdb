// Copyright (c) 2026 Richard Albright. All rights reserved.

use arrow::array::{
    Array, ArrayRef, Float64Array, ListBuilder, StructBuilder, UInt32Array, UInt64Array,
    UInt64Builder,
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
            signature: Signature::exact(
                vec![
                    DataType::UInt64,  // source
                    DataType::UInt64,  // target
                    DataType::Float64, // damping
                    DataType::UInt32,  // iterations
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for PageRankUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "pagerank"
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
            Arc::new(Field::new("damping", DataType::Float64, true)),
            Arc::new(Field::new("iterations", DataType::UInt32, true)),
        ])
    }
}

#[derive(Debug)]
pub struct PageRankAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
    damping: f64,
    iterations: u32,
}

impl PageRankAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
            damping: 0.85,
            iterations: 30,
        }
    }
}

impl Accumulator for PageRankAccumulator {
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
            ScalarValue::Float64(Some(self.damping)),
            ScalarValue::UInt32(Some(self.iterations)),
        ])
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
        let damping_arr = states[2]
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected Float64Array for damping".to_string())
            })?;
        let iterations_arr = states[3]
            .as_any()
            .downcast_ref::<arrow::array::UInt32Array>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected UInt32Array for iterations".to_string())
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

        if !damping_arr.is_empty() && damping_arr.is_valid(0) {
            self.damping = damping_arr.value(0);
        }
        if !iterations_arr.is_empty() && iterations_arr.is_valid(0) {
            self.iterations = iterations_arr.value(0);
        }

        Ok(())
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() < 2 {
            return Err(DataFusionError::Execution(
                "pagerank expects at least 2 arguments (source, target)".to_string(),
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

        if values.len() > 2 && !values[2].is_empty() {
            if let Some(arr) = values[2].as_any().downcast_ref::<Float64Array>() {
                if arr.is_valid(0) {
                    self.damping = arr.value(0);
                }
            }
        }

        if values.len() > 3 && !values[3].is_empty() {
            if let Some(arr) = values[3].as_any().downcast_ref::<UInt32Array>() {
                if arr.is_valid(0) {
                    self.iterations = arr.value(0);
                }
            }
        }

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
            ))); // returning empty basically
        }

        let mut adjacency: HashMap<u64, Vec<u64>> = HashMap::new();
        let mut nodes: Vec<u64> = Vec::new();
        for i in 0..self.sources.len() {
            let u = self.sources[i];
            let v = self.targets[i];
            adjacency.entry(u).or_default().push(v);
            nodes.push(u);
            nodes.push(v);
        }
        nodes.sort_unstable();
        nodes.dedup();

        let num_nodes = nodes.len() as f64;
        let mut scores: HashMap<u64, f64> = nodes.iter().map(|&n| (n, 1.0 / num_nodes)).collect();

        for _ in 0..self.iterations {
            let mut new_scores: HashMap<u64, f64> = nodes
                .iter()
                .map(|&n| (n, (1.0 - self.damping) / num_nodes))
                .collect();

            for &u in &nodes {
                let current_score = scores[&u];
                if let Some(neighbors) = adjacency.get(&u) {
                    let transfer = (self.damping * current_score) / (neighbors.len() as f64);
                    for &v in neighbors {
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

        for (node, score) in scores {
            node_builder.append_value(node);
            score_builder.append_value(score);
        }

        let mut struct_builder = StructBuilder::new(
            struct_fields.clone(),
            vec![Box::new(node_builder), Box::new(score_builder)],
        );

        for _ in 0..nodes.len() {
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
        std::mem::size_of_val(self) + self.sources.capacity() * 8 + self.targets.capacity() * 8
    }
}
