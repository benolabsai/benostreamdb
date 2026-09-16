use arrow::array::{Array, ArrayRef, Float32Array, ListBuilder, UInt64Array, UInt64Builder};
use arrow::datatypes::{DataType, Field};
use datafusion::logical_expr::{AggregateUDFImpl, Signature, TypeSignature, Volatility};
use datafusion::scalar::ScalarValue;
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_functions_aggregate_common::accumulator::{AccumulatorArgs, StateFieldsArgs};

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

use datafusion::error::{DataFusionError, Result};
// use petgraph::algo::astar;
use petgraph::graphmap::DiGraphMap;
use std::any::Any;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ShortestPathUDF {
    signature: Signature,
}
impl_dyn_traits!(ShortestPathUDF);

impl Default for ShortestPathUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl ShortestPathUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    TypeSignature::Exact(vec![
                        DataType::UInt64,
                        DataType::UInt64,
                        DataType::UInt64,
                        DataType::UInt64,
                    ]),
                    TypeSignature::Exact(vec![
                        DataType::UInt64,
                        DataType::UInt64,
                        DataType::Float32,
                        DataType::UInt64,
                        DataType::UInt64,
                    ]),
                    TypeSignature::Exact(vec![
                        DataType::UInt64,
                        DataType::UInt64,
                        DataType::Float64,
                        DataType::UInt64,
                        DataType::UInt64,
                    ]),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for ShortestPathUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "shortest_path"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::UInt64,
            true,
        ))))
    }
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(ShortestPathAccumulator::new()))
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
            Arc::new(Field::new("start_node", DataType::UInt64, true)),
            Arc::new(Field::new("end_node", DataType::UInt64, true)),
        ])
    }
}

#[derive(Debug)]
pub struct ShortestPathAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
    weights: Vec<f32>,
    start_node: Option<u64>,
    end_node: Option<u64>,
}

impl ShortestPathAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
            weights: Vec::new(),
            start_node: None,
            end_node: None,
        }
    }
}

impl Accumulator for ShortestPathAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() == 4 {
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

            if self.start_node.is_none() && !values[2].is_empty() {
                if let Some(s_arr) = values[2].as_any().downcast_ref::<UInt64Array>() {
                    if s_arr.is_valid(0) {
                        self.start_node = Some(s_arr.value(0));
                    }
                }
            }
            if self.end_node.is_none() && !values[3].is_empty() {
                if let Some(e_arr) = values[3].as_any().downcast_ref::<UInt64Array>() {
                    if e_arr.is_valid(0) {
                        self.end_node = Some(e_arr.value(0));
                    }
                }
            }

            for i in 0..sources_arr.len() {
                if sources_arr.is_valid(i) && targets_arr.is_valid(i) {
                    self.sources.push(sources_arr.value(i));
                    self.targets.push(targets_arr.value(i));
                    self.weights.push(1.0);
                }
            }
            Ok(())
        } else if values.len() == 5 {
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

            if self.start_node.is_none() && !values[3].is_empty() {
                if let Some(s_arr) = values[3].as_any().downcast_ref::<UInt64Array>() {
                    if s_arr.is_valid(0) {
                        self.start_node = Some(s_arr.value(0));
                    }
                }
            }
            if self.end_node.is_none() && !values[4].is_empty() {
                if let Some(e_arr) = values[4].as_any().downcast_ref::<UInt64Array>() {
                    if e_arr.is_valid(0) {
                        self.end_node = Some(e_arr.value(0));
                    }
                }
            }

            for i in 0..sources_arr.len() {
                if sources_arr.is_valid(i) && targets_arr.is_valid(i) {
                    self.sources.push(sources_arr.value(i));
                    self.targets.push(targets_arr.value(i));
                    let w = if let Some(w_f32) = values[2].as_any().downcast_ref::<Float32Array>() {
                        if w_f32.is_valid(i) {
                            w_f32.value(i)
                        } else {
                            1.0
                        }
                    } else if let Some(w_f64) = values[2]
                        .as_any()
                        .downcast_ref::<arrow::array::Float64Array>()
                    {
                        if w_f64.is_valid(i) {
                            w_f64.value(i) as f32
                        } else {
                            1.0
                        }
                    } else {
                        1.0
                    };
                    self.weights.push(w);
                }
            }
            Ok(())
        } else {
            Err(DataFusionError::Execution(
                "shortest_path expects 4 or 5 arguments".to_string(),
            ))
        }
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

        if let Some(s_arr) = states
            .get(3)
            .and_then(|a| a.as_any().downcast_ref::<UInt64Array>())
        {
            for i in 0..s_arr.len() {
                if s_arr.is_valid(i) {
                    self.start_node = Some(s_arr.value(i));
                    break;
                }
            }
        }

        if let Some(e_arr) = states
            .get(4)
            .and_then(|a| a.as_any().downcast_ref::<UInt64Array>())
        {
            for i in 0..e_arr.len() {
                if e_arr.is_valid(i) {
                    self.end_node = Some(e_arr.value(i));
                    break;
                }
            }
        }

        if let Some(e_arr) = states
            .get(4)
            .and_then(|a| a.as_any().downcast_ref::<UInt64Array>())
        {
            for i in 0..e_arr.len() {
                if e_arr.is_valid(i) {
                    self.end_node = Some(e_arr.value(i));
                    break;
                }
            }
        }

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
        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut sources_builder = arrow::array::ListBuilder::new(UInt64Builder::new());
        sources_builder.values().append_slice(&self.sources);
        sources_builder.append(true);
        let sources_list = ScalarValue::List(Arc::new(sources_builder.finish()));

        let mut targets_builder = arrow::array::ListBuilder::new(UInt64Builder::new());
        targets_builder.values().append_slice(&self.targets);
        targets_builder.append(true);
        let targets_list = ScalarValue::List(Arc::new(targets_builder.finish()));

        let mut weights_builder =
            arrow::array::ListBuilder::new(arrow::array::Float32Builder::new());
        weights_builder.values().append_slice(&self.weights);
        weights_builder.append(true);
        let weights_list = ScalarValue::List(Arc::new(weights_builder.finish()));

        Ok(vec![
            sources_list,
            targets_list,
            weights_list,
            ScalarValue::UInt64(self.start_node),
            ScalarValue::UInt64(self.end_node),
        ])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let start_node = self.start_node.unwrap_or(0);
        let end_node = self.end_node.unwrap_or(0);
        let mut graph = DiGraphMap::<u64, f32>::new();

        for i in 0..self.sources.len() {
            graph.add_edge(self.sources[i], self.targets[i], self.weights[i]);
        }

        let path = petgraph::algo::astar(
            &graph,
            start_node,
            |finish| finish == end_node,
            |(_, _, weight)| *weight as f64,
            |_| 0.0,
        );
        if let Some(res) = path {
            let mut builder = ListBuilder::new(UInt64Builder::new());
            builder.values().append_slice(&res.1);
            builder.append(true);
            Ok(ScalarValue::List(Arc::new(builder.finish())))
        } else {
            // Return empty list if no path
            let mut builder = ListBuilder::new(UInt64Builder::new());
            builder.append(true);
            Ok(ScalarValue::List(Arc::new(builder.finish())))
        }
    }

    fn size(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.sources.capacity() * 8
            + self.targets.capacity() * 8
            + self.weights.capacity() * 4
    }
}
