// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
#![allow(unused_imports, unused_mut, unused_variables, dead_code)]

use crate::core::sql::graph_udf::graph_view::GraphAccumulatorBase;
use ahash::{AHashMap as HashMap, AHashSet as HashSet};
use arrow::array::{
    Array, ArrayRef, Float64Array, Float64Builder, ListBuilder, StructBuilder, UInt32Array,
    UInt64Array, UInt64Builder,
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
pub struct BetweennessCentralityUDF {
    signature: Signature,
}
impl_dyn_traits!(BetweennessCentralityUDF);

impl Default for BetweennessCentralityUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl BetweennessCentralityUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for BetweennessCentralityUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "graph_betweenness_centrality"
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
        Ok(Box::new(BetweennessCentralityAccumulator::new()))
    }
    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        let fields = GraphAccumulatorBase::state_fields();
        Ok(fields)
    }
}

#[derive(Debug)]
pub struct BetweennessCentralityAccumulator {
    base: GraphAccumulatorBase,
}

impl BetweennessCentralityAccumulator {
    fn new() -> Self {
        Self {
            base: GraphAccumulatorBase::new(),
        }
    }
}

impl Accumulator for BetweennessCentralityAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        self.base.edge_state()
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.base.merge_edge_state(states, None, None)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let graph = self.base.resolve_graph(&[], 0)?;
        let all_nodes = graph.all_nodes();

        let mut cb: HashMap<u64, f64> = HashMap::new();
        for &v in &all_nodes {
            cb.insert(v, 0.0);
        }

        // Brandes algorithm
        for &s in &all_nodes {
            let mut s_stack = Vec::new();
            let mut p: HashMap<u64, Vec<u64>> = HashMap::new();
            for &v in &all_nodes {
                p.insert(v, Vec::new());
            }

            let mut sigma: HashMap<u64, f64> = HashMap::new();
            for &v in &all_nodes {
                sigma.insert(v, 0.0);
            }
            sigma.insert(s, 1.0);

            let mut d: HashMap<u64, i64> = HashMap::new();
            for &v in &all_nodes {
                d.insert(v, -1);
            }
            d.insert(s, 0);

            let mut q = VecDeque::new();
            q.push_back(s);
            let mut neighbors: Vec<u64> = Vec::new();

            while let Some(v) = q.pop_front() {
                s_stack.push(v);
                neighbors.clear();
                graph.get_neighbors_into(v, &mut neighbors);
                for &w in &neighbors {
                    if d[&w] < 0 {
                        q.push_back(w);
                        d.insert(w, d[&v] + 1);
                    }
                    if d[&w] == d[&v] + 1 {
                        let new_sigma = sigma[&w] + sigma[&v];
                        sigma.insert(w, new_sigma);
                        p.get_mut(&w).unwrap().push(v);
                    }
                }
            }

            let mut delta: HashMap<u64, f64> = HashMap::new();
            for &v in &all_nodes {
                delta.insert(v, 0.0);
            }

            while let Some(w) = s_stack.pop() {
                if let Some(parents) = p.get(&w) {
                    for &v in parents {
                        let delta_v = delta[&v] + (sigma[&v] / sigma[&w]) * (1.0 + delta[&w]);
                        delta.insert(v, delta_v);
                    }
                }
                if w != s {
                    let new_cb = cb[&w] + delta[&w];
                    cb.insert(w, new_cb);
                }
            }
        }

        let fields = Fields::from(vec![
            Field::new("node", DataType::UInt64, false),
            Field::new("score", DataType::Float64, false),
        ]);
        let mut struct_builder = StructBuilder::new(
            fields,
            vec![
                Box::new(UInt64Builder::new()),
                Box::new(Float64Builder::new()),
            ],
        );
        let mut list_builder = ListBuilder::new(struct_builder);

        let sb = list_builder.values();

        for &node in &all_nodes {
            sb.field_builder::<UInt64Builder>(0)
                .unwrap()
                .append_value(node);
            // Nodes that never appear as an intermediate vertex have no entry
            // in `cb`; indexing the map directly panicked with "no entry found
            // for key" (NO_PANIC_POLICY violation).
            sb.field_builder::<Float64Builder>(1)
                .unwrap()
                .append_value(cb.get(&node).copied().unwrap_or(0.0));
            sb.append(true);
        }

        if !all_nodes.is_empty() {
            list_builder.append(true);
        } else {
            list_builder.append(false); // empty list
        }

        Ok(ScalarValue::List(Arc::new(list_builder.finish())))
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let uri_idx = if values.len() > 2 { Some(2) } else { None };
        let mode_idx = if values.len() > 3 { Some(3) } else { None };
        self.base.update_edge_batch(values, uri_idx, mode_idx)
    }

    fn size(&self) -> usize {
        self.base.size()
    }
}
