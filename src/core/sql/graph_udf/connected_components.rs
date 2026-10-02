// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
#![allow(unused_imports, unused_mut, unused_variables, dead_code)]

use crate::core::sql::graph_udf::graph_view::GraphAccumulatorBase;
use ahash::AHashMap as HashMap;
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
pub struct ConnectedComponentsUDF {
    signature: Signature,
}
impl_dyn_traits!(ConnectedComponentsUDF);

impl Default for ConnectedComponentsUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl ConnectedComponentsUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for ConnectedComponentsUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "graph_connected_components"
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
        Ok(Box::new(ConnectedComponentsAccumulator::new()))
    }

    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        Ok(GraphAccumulatorBase::state_fields())
    }
}

#[derive(Debug, Default)]
pub struct ConnectedComponentsAccumulator {
    base: GraphAccumulatorBase,
}

impl ConnectedComponentsAccumulator {
    fn new() -> Self {
        Self {
            base: GraphAccumulatorBase::new(),
        }
    }
}

impl Accumulator for ConnectedComponentsAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        self.base.edge_state()
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.base.merge_edge_state(states, Some(2), Some(3))
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let graph = self.base.resolve_graph(&[], 0)?;
        let mut parent: HashMap<u64, u64> = HashMap::new();

        fn find(parent: &mut HashMap<u64, u64>, x: u64) -> u64 {
            let mut root = x;
            while parent[&root] != root {
                root = parent[&root];
            }
            let mut cur = x;
            while parent[&cur] != root {
                let next = parent[&cur];
                parent.insert(cur, root);
                cur = next;
            }
            root
        }

        for (u, v) in graph.all_edges() {
            parent.entry(u).or_insert(u);
            parent.entry(v).or_insert(v);
            let ru = find(&mut parent, u);
            let rv = find(&mut parent, v);
            if ru != rv {
                let (keep, drop) = (ru.min(rv), ru.max(rv));
                parent.insert(drop, keep);
            }
        }

        let mut builder = arrow::array::ListBuilder::new(arrow::array::UInt64Builder::new());
        let mut nodes: Vec<u64> = parent.keys().cloned().collect();
        nodes.sort_unstable();
        let mut out = Vec::with_capacity(nodes.len() * 2);
        for n in nodes {
            let root = find(&mut parent, n);
            out.push(n);
            out.push(root);
        }
        builder.values().append_slice(&out);
        builder.append(true);
        Ok(ScalarValue::List(Arc::new(builder.finish())))
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.base.update_edge_batch(values, Some(2), Some(3))
    }

    fn size(&self) -> usize {
        self.base.size()
    }
}
