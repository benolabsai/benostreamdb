// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
#![allow(unused_imports, unused_mut, unused_variables, dead_code)]

use crate::core::sql::graph_udf::graph_view::GraphAccumulatorBase;
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
pub struct DegreeCentralityUDF {
    signature: Signature,
}
impl_dyn_traits!(DegreeCentralityUDF);

impl Default for DegreeCentralityUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl DegreeCentralityUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for DegreeCentralityUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "graph_degree_centrality"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        let struct_fields = vec![
            Field::new("node", DataType::UInt64, true),
            Field::new("degree", DataType::UInt64, true),
        ];
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(Fields::from(struct_fields)),
            true,
        ))))
    }

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(DegreeCentralityAccumulator::new()))
    }

    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        Ok(GraphAccumulatorBase::state_fields())
    }
}

#[derive(Debug)]
pub struct DegreeCentralityAccumulator {
    base: GraphAccumulatorBase,
}

impl DegreeCentralityAccumulator {
    fn new() -> Self {
        Self {
            base: GraphAccumulatorBase::new(),
        }
    }
}

impl Accumulator for DegreeCentralityAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        self.base.edge_state()
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.base.merge_edge_state(states, None, None)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let graph = self.base.resolve_graph(&[], 0)?;
        let nodes = graph.all_nodes();

        let mut node_builder = arrow::array::UInt64Builder::new();
        let mut degree_builder = arrow::array::UInt64Builder::new();

        for n in &nodes {
            node_builder.append_value(*n);
            // Use the trait's O(1) degree accessor rather than allocating and
            // discarding a fresh neighbor `Vec` per node just to take `.len()`.
            let deg = graph.get_degree(*n) as u64;
            degree_builder.append_value(deg);
        }

        let struct_fields = vec![
            Field::new("node", DataType::UInt64, true),
            Field::new("degree", DataType::UInt64, true),
        ];

        let struct_array = arrow::array::StructArray::new(
            Fields::from(struct_fields.clone()),
            vec![
                Arc::new(node_builder.finish()) as _,
                Arc::new(degree_builder.finish()) as _,
            ],
            None,
        );

        let list_data = arrow::array::ArrayData::builder(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(Fields::from(struct_fields)),
            true,
        ))))
        .len(1)
        .add_buffer(arrow::buffer::Buffer::from_slice_ref([
            0i32,
            nodes.len() as i32,
        ]))
        .add_child_data(struct_array.into_data())
        .build()
        .map_err(|e| datafusion::error::DataFusionError::ArrowError(Box::new(e), None))?;

        let list_array = arrow::array::ListArray::from(list_data);
        Ok(ScalarValue::List(Arc::new(list_array)))
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        self.base.update_edge_batch(values, None, None)
    }

    fn size(&self) -> usize {
        self.base.size()
    }
}
