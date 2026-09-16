use arrow::array::{Array, ArrayRef, ListBuilder, UInt64Array, UInt64Builder};
use arrow::datatypes::{DataType, Field};
use datafusion::logical_expr::{AggregateUDFImpl, Signature, Volatility};
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
use petgraph::graphmap::DiGraphMap;

use std::any::Any;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct GraphNeighborsUDF {
    signature: Signature,
}
impl_dyn_traits!(GraphNeighborsUDF);

impl Default for GraphNeighborsUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphNeighborsUDF {
    pub fn new() -> Self {
        // Arguments: source_col, target_col, entity_id (u64)

        Self {
            signature: Signature::exact(
                vec![
                    DataType::UInt64,
                    DataType::UInt64,
                    DataType::UInt64,
                    DataType::UInt32,
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for GraphNeighborsUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "graph_neighbors"
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
        Ok(Box::new(GraphNeighborsAccumulator::new()))
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
            Arc::new(Field::new("entity_id", DataType::UInt64, true)),
            Arc::new(Field::new("depth", DataType::UInt32, true)),
        ])
    }
}

#[derive(Debug)]
pub struct GraphNeighborsAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
    entity_id: Option<u64>,
    depth: Option<u32>,
}

impl GraphNeighborsAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
            entity_id: None,
            depth: None,
        }
    }
}

impl Accumulator for GraphNeighborsAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() < 3 {
            return Err(DataFusionError::Execution(
                "graph_neighbors expects at least 3 arguments".to_string(),
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

        if values.len() > 2 && !values[2].is_empty() {
            if let Some(e_arr) = values[2].as_any().downcast_ref::<UInt64Array>() {
                if e_arr.is_valid(0) {
                    self.entity_id = Some(e_arr.value(0));
                }
            }
        }

        if values.len() > 3 && !values[3].is_empty() {
            if let Some(d_arr) = values[3]
                .as_any()
                .downcast_ref::<arrow::array::UInt32Array>()
            {
                if d_arr.is_valid(0) {
                    self.depth = Some(d_arr.value(0));
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

        if let Some(e_arr) = states
            .get(2)
            .and_then(|a| a.as_any().downcast_ref::<UInt64Array>())
        {
            for i in 0..e_arr.len() {
                if e_arr.is_valid(i) {
                    self.entity_id = Some(e_arr.value(i));
                    break;
                }
            }
        }

        if let Some(d_arr) = states
            .get(3)
            .and_then(|a| a.as_any().downcast_ref::<arrow::array::UInt32Array>())
        {
            for i in 0..d_arr.len() {
                if d_arr.is_valid(i) {
                    self.depth = Some(d_arr.value(i));
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

        Ok(vec![
            sources_list,
            targets_list,
            ScalarValue::UInt64(self.entity_id),
            ScalarValue::UInt32(self.depth),
        ])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut graph = DiGraphMap::<u64, ()>::new();

        for i in 0..self.sources.len() {
            graph.add_edge(self.sources[i], self.targets[i], ());
        }

        let max_depth = self.depth.unwrap_or(u32::MAX);
        let start_node = self.entity_id.unwrap_or(0);

        // Use a standard BFS queue to track depth
        let mut neighbors = Vec::new();
        let mut queue = std::collections::VecDeque::new();
        let mut visited = std::collections::HashSet::new();

        queue.push_back((start_node, 0));
        visited.insert(start_node);

        while let Some((node, d)) = queue.pop_front() {
            if d > 0 {
                neighbors.push(node);
            }
            if d < max_depth {
                for neighbor in graph.neighbors(node) {
                    if visited.insert(neighbor) {
                        queue.push_back((neighbor, d + 1));
                    }
                }
            }
        }

        let mut builder = ListBuilder::new(UInt64Builder::new());

        builder.values().append_slice(&neighbors);
        builder.append(true);
        Ok(ScalarValue::List(Arc::new(builder.finish())))
    }

    fn size(&self) -> usize {
        std::mem::size_of::<Self>() + self.sources.capacity() * 8 + self.targets.capacity() * 8
    }
}
