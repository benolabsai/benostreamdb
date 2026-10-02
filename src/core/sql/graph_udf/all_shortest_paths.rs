// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::sql::graph_udf::graph_view::GraphAccumulatorBase;
use ahash::AHashMap as HashMap;
use arrow::array::{Array, ArrayRef, ListBuilder, UInt64Array, UInt64Builder};
use arrow::datatypes::{DataType, Field};
use datafusion::error::Result;
use datafusion::logical_expr::{AggregateUDFImpl, Signature, Volatility};
use datafusion::scalar::ScalarValue;
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_functions_aggregate_common::accumulator::{AccumulatorArgs, StateFieldsArgs};
use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct AllShortestPathsUDF {
    signature: Signature,
}

impl PartialEq for AllShortestPathsUDF {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for AllShortestPathsUDF {}
impl std::hash::Hash for AllShortestPathsUDF {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::any::type_name::<Self>().hash(state);
    }
}

impl Default for AllShortestPathsUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl AllShortestPathsUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for AllShortestPathsUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "graph_all_shortest_paths"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        let inner = Arc::new(Field::new("item", DataType::UInt64, true));
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::List(inner),
            true,
        ))))
    }
    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(AllShortestPathsAccumulator::new()))
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
pub struct AllShortestPathsAccumulator {
    base: GraphAccumulatorBase,
    start: Option<u64>,
    end: Option<u64>,
}

impl AllShortestPathsAccumulator {
    fn new() -> Self {
        Self {
            base: GraphAccumulatorBase::new(),
            start: None,
            end: None,
        }
    }
}

impl Accumulator for AllShortestPathsAccumulator {
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

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.base.update_edge_batch(values, Some(4), Some(5))?;
        if values.len() > 2 && !values[2].is_empty() {
            if let Some(arr) = values[2].as_any().downcast_ref::<UInt64Array>() {
                if arr.is_valid(0) {
                    self.start = Some(arr.value(0));
                }
            }
        }
        if values.len() > 3 && !values[3].is_empty() {
            if let Some(arr) = values[3].as_any().downcast_ref::<UInt64Array>() {
                if arr.is_valid(0) {
                    self.end = Some(arr.value(0));
                }
            }
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut outer_builder = ListBuilder::new(ListBuilder::new(UInt64Builder::new()));

        if let (Some(start), Some(end)) = (self.start, self.end) {
            let graph = self.base.resolve_graph(&[], 0)?;
            let mut paths: Vec<Vec<u64>> = vec![];

            if start == end {
                paths.push(vec![start]);
            } else {
                let mut prev: HashMap<u64, Vec<u64>> = HashMap::new();
                let mut dist: HashMap<u64, usize> = HashMap::new();
                let mut q = VecDeque::new();
                let mut neighbors: Vec<u64> = Vec::new();

                dist.insert(start, 0);
                q.push_back(start);

                let mut shortest_len = usize::MAX;

                while let Some(curr) = q.pop_front() {
                    let d = dist[&curr];
                    if d > shortest_len {
                        continue;
                    }
                    if curr == end {
                        shortest_len = d;
                        continue;
                    }

                    neighbors.clear();
                    graph.get_neighbors_into(curr, &mut neighbors);
                    for &n in &neighbors {
                        let alt = d + 1;
                        if alt > shortest_len {
                            continue;
                        }
                        let mut do_push = false;
                        if !dist.contains_key(&n) {
                            dist.insert(n, alt);
                            prev.insert(n, vec![curr]);
                            do_push = true;
                        } else if dist[&n] == alt {
                            prev.get_mut(&n).unwrap().push(curr);
                        }
                        if do_push {
                            q.push_back(n);
                        }
                    }
                }

                if shortest_len != usize::MAX {
                    // Backtrack to find all paths
                    let mut current_path = vec![end];
                    let mut all_paths = vec![];

                    fn backtrack(
                        u: u64,
                        start: u64,
                        prev: &HashMap<u64, Vec<u64>>,
                        current_path: &mut Vec<u64>,
                        all_paths: &mut Vec<Vec<u64>>,
                    ) {
                        if u == start {
                            let mut p = current_path.clone();
                            p.reverse();
                            all_paths.push(p);
                            return;
                        }
                        if let Some(parents) = prev.get(&u) {
                            for &p in parents {
                                current_path.push(p);
                                backtrack(p, start, prev, current_path, all_paths);
                                current_path.pop();
                            }
                        }
                    }

                    backtrack(end, start, &prev, &mut current_path, &mut all_paths);
                    paths = all_paths;
                }
            }

            if paths.is_empty() {
                outer_builder.append(false);
            } else {
                let inner_builder = outer_builder.values();
                for path in paths {
                    inner_builder.values().append_slice(&path);
                    inner_builder.append(true);
                }
                outer_builder.append(true);
            }
        } else {
            outer_builder.append(false);
        }

        Ok(ScalarValue::List(Arc::new(outer_builder.finish())))
    }

    fn size(&self) -> usize {
        self.base.size() + std::mem::size_of::<Option<u64>>() * 2
    }
}
