// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::sql::graph_udf::graph_view::GraphAccumulatorBase;
use ahash::{AHashMap, AHashSet as HashSet};
use arrow::array::ArrayRef;
use arrow::datatypes::{DataType, Field};
use datafusion::error::Result;
use datafusion::logical_expr::{Accumulator, Signature, Volatility};
use datafusion::scalar::ScalarValue;
use std::any::Any;
use std::sync::Arc;

#[derive(Debug)]
pub struct TriangleCountUDF {
    signature: Signature,
}

impl PartialEq for TriangleCountUDF {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for TriangleCountUDF {}
impl std::hash::Hash for TriangleCountUDF {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::any::type_name::<Self>().hash(state);
    }
}

impl Default for TriangleCountUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl TriangleCountUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl datafusion::logical_expr::AggregateUDFImpl for TriangleCountUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "graph_triangle_count"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::UInt64)
    }
    fn state_fields(
        &self,
        _args: datafusion_functions_aggregate_common::accumulator::StateFieldsArgs,
    ) -> Result<Vec<Arc<Field>>> {
        let fields = GraphAccumulatorBase::state_fields();
        Ok(fields)
    }
    fn accumulator(
        &self,
        _acc_args: datafusion_functions_aggregate_common::accumulator::AccumulatorArgs,
    ) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(TriangleCountAccumulator {
            base: GraphAccumulatorBase::new(),
        }))
    }
}

#[derive(Debug)]
struct TriangleCountAccumulator {
    base: GraphAccumulatorBase,
}

impl Accumulator for TriangleCountAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let uri_idx = if values.len() > 2 { Some(2) } else { None };
        let mode_idx = if values.len() > 3 { Some(3) } else { None };
        self.base.update_edge_batch(values, uri_idx, mode_idx)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.base.merge_edge_state(states, None, None)
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        self.base.edge_state()
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let graph = self.base.resolve_graph(&[], 0)?;

        let mut count = 0u64;
        let mut neighbor_sets = AHashMap::new();

        let all_nodes = graph.all_nodes();

        // Pre-compute neighbor sets for fast lookup
        let mut scratch: Vec<u64> = Vec::new();
        for &u in &all_nodes {
            scratch.clear();
            graph.get_neighbors_into(u, &mut scratch);
            let mut set = HashSet::new();
            for &v in &scratch {
                if u != v {
                    set.insert(v);
                }
            }
            neighbor_sets.insert(u, set);
        }

        for &u in &all_nodes {
            let u_neighbors = match neighbor_sets.get(&u) {
                Some(set) => set,
                None => continue,
            };

            for &v in u_neighbors.iter() {
                if v <= u {
                    continue; // count each triangle once
                }

                if let Some(v_neighbors) = neighbor_sets.get(&v) {
                    for &w in v_neighbors.iter() {
                        if w <= v {
                            continue;
                        }
                        if u_neighbors.contains(&w) {
                            count += 1;
                        }
                    }
                }
            }
        }

        Ok(ScalarValue::UInt64(Some(count)))
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self) + self.base.size()
    }
}
