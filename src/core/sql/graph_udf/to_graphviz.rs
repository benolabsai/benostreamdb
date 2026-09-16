use arrow::array::{Array, ArrayRef, UInt64Array, UInt64Builder};
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
use petgraph::dot::{Config, Dot};
use petgraph::graphmap::DiGraphMap;
use std::any::Any;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ToGraphvizUDF {
    signature: Signature,
}
impl_dyn_traits!(ToGraphvizUDF);

impl Default for ToGraphvizUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl ToGraphvizUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::UInt64, DataType::UInt64],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for ToGraphvizUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "to_graphviz"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Utf8)
    }
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(ToGraphvizAccumulator::new()))
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
        ])
    }
}

#[derive(Debug)]
pub struct ToGraphvizAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
}

impl ToGraphvizAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
        }
    }
}

impl Accumulator for ToGraphvizAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() != 2 {
            return Err(DataFusionError::Execution(
                "to_graphviz expects 2 arguments".to_string(),
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

        Ok(vec![sources_list, targets_list])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut graph = DiGraphMap::<u64, ()>::new();

        for i in 0..self.sources.len() {
            graph.add_edge(self.sources[i], self.targets[i], ());
        }

        let dot_string = format!("{:?}", Dot::with_config(&graph, &[Config::EdgeNoLabel]));
        Ok(ScalarValue::Utf8(Some(dot_string)))
    }

    fn size(&self) -> usize {
        std::mem::size_of::<Self>() + self.sources.capacity() * 8 + self.targets.capacity() * 8
    }
}
