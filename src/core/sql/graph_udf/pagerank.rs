// Copyright (c) 2026 Richard Albright. All rights reserved.

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float32Builder, UInt64Array, UInt64Builder};
use arrow::datatypes::{DataType, Field};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{AggregateUDFImpl, Signature, Volatility};
use datafusion::scalar::ScalarValue;
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_functions_aggregate_common::accumulator::{AccumulatorArgs, StateFieldsArgs};

use petgraph::graphmap::DiGraphMap;
use petgraph::Direction;

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

#[derive(Debug)]
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
        // Arguments: source_id (u64), target_id (u64), damping (f32), iterations (u32)
        Self {
            signature: Signature::exact(
                vec![
                    DataType::UInt64,
                    DataType::UInt64,
                    DataType::Float64,
                    DataType::Int64,
                ],
                Volatility::Immutable,
            ),
        }
    }

    fn return_type_struct() -> DataType {
        DataType::Struct(
            vec![
                Field::new("node", DataType::UInt64, false),
                Field::new("score", DataType::Float32, false),
            ]
            .into(),
        )
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
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            Self::return_type_struct(),
            true,
        ))))
    }
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(PageRankAccumulator::new()))
    }
    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        // State is kept in memory during a single partition run, but for distribution
        // we'd serialize the edges. Here we just store source/target arrays.
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
            Arc::new(Field::new("iterations", DataType::Int64, true)),
        ])
    }
}

#[derive(Debug)]
pub struct PageRankAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
    damping: f32,
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
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() != 4 {
            return Err(DataFusionError::Execution(
                "pagerank expects 4 arguments".to_string(),
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

        // Scalar values for damping and iterations (they are repeated in the array if parsed from columns,
        // or just take the first valid value if passed as literals)
        if !values[2].is_empty() {
            if let Some(d_arr) = values[2]
                .as_any()
                .downcast_ref::<arrow::array::Float64Array>()
            {
                if d_arr.is_valid(0) {
                    self.damping = d_arr.value(0) as f32;
                }
            }
        }

        if !values[3].is_empty() {
            if let Some(i_arr) = values[3]
                .as_any()
                .downcast_ref::<arrow::array::Int64Array>()
            {
                if i_arr.is_valid(0) {
                    self.iterations = i_arr.value(0) as u32;
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

        if let Some(damping_arr) = states
            .get(2)
            .and_then(|a| a.as_any().downcast_ref::<arrow::array::Float64Array>())
        {
            if !damping_arr.is_empty() && damping_arr.is_valid(0) {
                self.damping = damping_arr.value(0) as f32;
            }
        }

        if let Some(iter_arr) = states
            .get(3)
            .and_then(|a| a.as_any().downcast_ref::<arrow::array::Int64Array>())
        {
            if !iter_arr.is_empty() && iter_arr.is_valid(0) {
                self.iterations = iter_arr.value(0) as u32;
            }
        }

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
        Ok(())
    }

    fn state(&mut self) -> Result<Vec<datafusion::scalar::ScalarValue>> {
        let mut sources_builder = arrow::array::ListBuilder::new(UInt64Builder::new());
        sources_builder.values().append_slice(&self.sources);
        sources_builder.append(true);
        let sources_list =
            datafusion::scalar::ScalarValue::List(Arc::new(sources_builder.finish()));

        let mut targets_builder = arrow::array::ListBuilder::new(UInt64Builder::new());
        targets_builder.values().append_slice(&self.targets);
        targets_builder.append(true);
        let targets_list =
            datafusion::scalar::ScalarValue::List(Arc::new(targets_builder.finish()));

        Ok(vec![
            sources_list,
            targets_list,
            datafusion::scalar::ScalarValue::Float64(Some(self.damping as f64)),
            datafusion::scalar::ScalarValue::Int64(Some(self.iterations as i64)),
        ])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut graph = DiGraphMap::<u64, ()>::new();

        for (s, t) in self.sources.iter().zip(self.targets.iter()) {
            graph.add_edge(*s, *t, ());
        }

        let nodes: Vec<u64> = graph.nodes().collect();
        let num_nodes = nodes.len();

        if num_nodes == 0 {
            // Return empty list
            let fields = vec![
                Field::new("node", DataType::UInt64, false),
                Field::new("score", DataType::Float32, false),
            ];
            let struct_type = DataType::Struct(fields.into());
            return Ok(ScalarValue::List(ScalarValue::new_list(
                &[],
                &struct_type,
                true,
            )));
        }

        let initial_score = 1.0 / (num_nodes as f32);
        let mut scores: HashMap<u64, f32> = nodes.iter().map(|&n| (n, initial_score)).collect();
        let mut out_degrees: HashMap<u64, usize> = HashMap::new();

        for &node in &nodes {
            out_degrees.insert(
                node,
                graph.edges_directed(node, Direction::Outgoing).count(),
            );
        }

        for _ in 0..self.iterations {
            let mut new_scores = HashMap::with_capacity(num_nodes);

            for &node in &nodes {
                let mut sum = 0.0;
                for incoming in graph.neighbors_directed(node, Direction::Incoming) {
                    let out_deg = out_degrees.get(&incoming).unwrap_or(&0);
                    if *out_deg > 0 {
                        sum += scores.get(&incoming).unwrap_or(&0.0) / (*out_deg as f32);
                    }
                }

                let new_score = (1.0 - self.damping) / (num_nodes as f32) + self.damping * sum;
                new_scores.insert(node, new_score);
            }

            scores = new_scores;
        }

        // Build the result as a StructArray inside a ListArray
        let mut node_id_builder = UInt64Builder::new();
        let mut score_builder = Float32Builder::new();

        for &node in &nodes {
            node_id_builder.append_value(node);
            score_builder.append_value(*scores.get(&node).unwrap_or(&0.0));
        }

        let node_id_array = Arc::new(node_id_builder.finish()) as ArrayRef;
        let score_array = Arc::new(score_builder.finish()) as ArrayRef;

        let struct_fields = vec![
            Arc::new(Field::new("node", DataType::UInt64, false)),
            Arc::new(Field::new("score", DataType::Float32, false)),
        ];

        let struct_array = arrow::array::StructArray::from(vec![
            (struct_fields[0].clone(), node_id_array),
            (struct_fields[1].clone(), score_array),
        ]);

        let list_fields = Arc::new(Field::new(
            "item",
            DataType::Struct(struct_fields.into()),
            true,
        ));

        // Wrap the struct array in a list array of length 1
        let offsets = arrow::buffer::OffsetBuffer::from_lengths(vec![num_nodes]);
        let list_array =
            arrow::array::ListArray::new(list_fields, offsets, Arc::new(struct_array), None);

        Ok(ScalarValue::List(Arc::new(list_array)))
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self)
            + self.sources.capacity() * std::mem::size_of::<u64>()
            + self.targets.capacity() * std::mem::size_of::<u64>()
    }
}
