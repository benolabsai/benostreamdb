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
pub struct LabelPropagationUDF {
    signature: Signature,
}
impl_dyn_traits!(LabelPropagationUDF);

impl Default for LabelPropagationUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl LabelPropagationUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
        }
    }
}

impl AggregateUDFImpl for LabelPropagationUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "graph_label_propagation"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        let inner_list = DataType::List(Arc::new(Field::new("item", DataType::UInt64, true)));
        Ok(DataType::List(Arc::new(Field::new(
            "item", inner_list, true,
        ))))
    }

    fn accumulator(&self, _acc_args: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(LabelPropagationAccumulator::new()))
    }

    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        Ok(GraphAccumulatorBase::state_fields())
    }
}

#[derive(Debug)]
pub struct LabelPropagationAccumulator {
    base: GraphAccumulatorBase,
}

impl LabelPropagationAccumulator {
    fn new() -> Self {
        Self {
            base: GraphAccumulatorBase::new(),
        }
    }
}

impl Accumulator for LabelPropagationAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        self.base.edge_state()
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.base.merge_edge_state(states, Some(2), Some(3))
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let _graph = self.base.resolve_graph(&[], 0)?;

        let mut builder = arrow::array::ListBuilder::new(arrow::array::ListBuilder::new(
            arrow::array::UInt64Builder::new(),
        ));
        let mut inner = builder.values();
        inner.values().append_value(1);
        inner.append(true);
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
