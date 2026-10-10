// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::sql::graph_udf::graph_view::GraphAccumulatorBase;
use ahash::AHashSet;
use arrow::array::{ArrayRef, Float64Builder, ListBuilder, StructBuilder, UInt64Builder};
use arrow::datatypes::{DataType, Field, Fields};
use datafusion::error::Result;
use datafusion::logical_expr::{Accumulator, Signature, Volatility};
use datafusion::scalar::ScalarValue;
use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Debug)]
pub struct ClosenessCentralityUDF {
    signature: Signature,
}

impl PartialEq for ClosenessCentralityUDF {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for ClosenessCentralityUDF {}
impl std::hash::Hash for ClosenessCentralityUDF {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::any::type_name::<Self>().hash(state);
    }
}

impl Default for ClosenessCentralityUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl ClosenessCentralityUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl datafusion::logical_expr::AggregateUDFImpl for ClosenessCentralityUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "graph_closeness_centrality"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        let fields = Fields::from(vec![
            Field::new("node", DataType::UInt64, true),
            Field::new("score", DataType::Float64, true),
        ]);
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(fields),
            true,
        ))))
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
        Ok(Box::new(ClosenessCentralityAccumulator {
            base: GraphAccumulatorBase::new(),
        }))
    }
}

#[derive(Debug)]
struct ClosenessCentralityAccumulator {
    base: GraphAccumulatorBase,
}

impl Accumulator for ClosenessCentralityAccumulator {
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
        let all_nodes = graph.all_nodes();

        let fields = Fields::from(vec![
            Field::new("node", DataType::UInt64, true),
            Field::new("score", DataType::Float64, true),
        ]);
        let struct_builder = StructBuilder::new(
            fields.clone(),
            vec![
                Box::new(UInt64Builder::new()),
                Box::new(Float64Builder::new()),
            ],
        );
        let mut list_builder = ListBuilder::new(struct_builder);

        for &u in &all_nodes {
            let mut sum_dist = 0;
            let mut reachable = 0;
            let mut visited = AHashSet::new();
            let mut q = VecDeque::new();
            let mut neighbors: Vec<u64> = Vec::new();

            visited.insert(u);
            q.push_back((u, 0));

            while let Some((curr, dist)) = q.pop_front() {
                sum_dist += dist;
                reachable += 1;

                neighbors.clear();
                graph.get_neighbors_into(curr, &mut neighbors);
                for &n in &neighbors {
                    if visited.insert(n) {
                        q.push_back((n, dist + 1));
                    }
                }
            }

            let score = if sum_dist > 0 && reachable > 1 {
                ((reachable - 1) as f64) / (sum_dist as f64)
            } else {
                0.0
            };

            let sb = list_builder.values();
            sb.field_builder::<UInt64Builder>(0)
                .ok_or_else(|| datafusion::error::DataFusionError::Execution("missing field 0".into()))?
                .append_value(u);
            sb.field_builder::<Float64Builder>(1)
                .ok_or_else(|| datafusion::error::DataFusionError::Execution("missing field 1".into()))?
                .append_value(score);
            sb.append(true);
        }

        list_builder.append(true);
        Ok(ScalarValue::List(Arc::new(list_builder.finish())))
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self) + self.base.size()
    }
}
