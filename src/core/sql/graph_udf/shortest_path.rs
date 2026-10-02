// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
#![allow(unused_imports, unused_mut, unused_variables, dead_code)]

use crate::core::sql::graph_udf::graph_view::GraphAccumulatorBase;
use ahash::{AHashMap as HashMap, AHashSet as HashSet};
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
use std::collections::VecDeque;
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
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for ShortestPathUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "graph_shortest_path"
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

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(ShortestPathAccumulator::new()))
    }
    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        let mut fields = GraphAccumulatorBase::state_fields();
        fields.extend(vec![
            Arc::new(Field::new("start", DataType::UInt64, true)),
            Arc::new(Field::new("end", DataType::UInt64, true)),
        ]);
        Ok(fields)
    }
}

#[derive(Debug)]
pub struct ShortestPathAccumulator {
    base: GraphAccumulatorBase,
    start: Option<u64>,
    end: Option<u64>,
}

impl ShortestPathAccumulator {
    fn new() -> Self {
        Self {
            base: GraphAccumulatorBase::new(),
            start: None,
            end: None,
        }
    }
}

impl Accumulator for ShortestPathAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut state = self.base.edge_state()?;
        state.push(ScalarValue::UInt64(self.start));
        state.push(ScalarValue::UInt64(self.end));
        Ok(state)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.base.merge_edge_state(states, Some(2), Some(3))?;
        let start_idx = states.len().saturating_sub(2).max(4);
        let end_idx = states.len().saturating_sub(1).max(5);

        if states.len() > start_idx {
            if let Some(start_arr) = states[start_idx].as_any().downcast_ref::<UInt64Array>() {
                if start_arr.is_valid(0) {
                    self.start = Some(start_arr.value(0));
                }
            }
        }
        if states.len() > end_idx {
            if let Some(end_arr) = states[end_idx].as_any().downcast_ref::<UInt64Array>() {
                if end_arr.is_valid(0) {
                    self.end = Some(end_arr.value(0));
                }
            }
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut builder = arrow::array::ListBuilder::new(arrow::array::UInt64Builder::new());

        if let (Some(start), Some(end)) = (self.start, self.end) {
            let graph = self.base.resolve_graph(&[], 0)?;
            let path = if start == end {
                Some(vec![start])
            } else {
                let mut prev: HashMap<u64, u64> = HashMap::new();
                let mut visited: HashSet<u64> = HashSet::new();
                let mut q = VecDeque::new();
                visited.insert(start);
                q.push_back(start);
                let mut found = false;
                // One scratch buffer for the whole BFS, refilled each hop.
                let mut neighbors: Vec<u64> = Vec::new();

                while let Some(curr) = q.pop_front() {
                    if curr == end {
                        found = true;
                        break;
                    }
                    neighbors.clear();
                    graph.get_neighbors_into(curr, &mut neighbors);
                    for &n in &neighbors {
                        if visited.insert(n) {
                            prev.insert(n, curr);
                            q.push_back(n);
                        }
                    }
                }

                if found {
                    let mut path = vec![end];
                    let mut curr = end;
                    while curr != start {
                        curr = *prev.get(&curr).ok_or_else(|| {
                            DataFusionError::Execution(
                                "shortest_path: broken predecessor chain".to_string(),
                            )
                        })?;
                        path.push(curr);
                    }
                    path.reverse();
                    Some(path)
                } else {
                    None
                }
            };

            if let Some(path) = path {
                builder.values().append_slice(&path);
                builder.append(true);
            } else {
                builder.append(false); // no path found
            }
        } else {
            builder.append(false); // missing start/end arguments
        }

        Ok(ScalarValue::List(Arc::new(builder.finish())))
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.base.update_edge_batch(values, Some(4), Some(5))?;
        if values.len() > 2 && !values[2].is_empty() {
            if let Some(arr) = values[2]
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
            {
                if arr.is_valid(0) {
                    self.start = Some(arr.value(0));
                }
            }
        }
        if values.len() > 3 && !values[3].is_empty() {
            if let Some(arr) = values[3]
                .as_any()
                .downcast_ref::<arrow::array::UInt64Array>()
            {
                if arr.is_valid(0) {
                    self.end = Some(arr.value(0));
                }
            }
        }

        Ok(())
    }

    fn size(&self) -> usize {
        self.base.size() + std::mem::size_of::<Option<u64>>() * 2
    }
}
