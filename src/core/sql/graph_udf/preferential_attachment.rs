use arrow::array::{Array, ArrayRef, UInt64Array};
use arrow::datatypes::{DataType, Field};
use datafusion::error::Result;
use datafusion::logical_expr::{AggregateUDFImpl, Signature, Volatility};
use datafusion::scalar::ScalarValue;
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_functions_aggregate_common::accumulator::{AccumulatorArgs, StateFieldsArgs};
use petgraph::graph::Graph;
use petgraph::Undirected;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct PreferentialAttachmentUDF {
    signature: Signature,
}

impl PartialEq for PreferentialAttachmentUDF {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for PreferentialAttachmentUDF {}
impl std::hash::Hash for PreferentialAttachmentUDF {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::any::type_name::<Self>().hash(state);
    }
}

impl Default for PreferentialAttachmentUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl PreferentialAttachmentUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![
                    DataType::UInt64,
                    DataType::UInt64,
                    DataType::UInt64,
                    DataType::UInt64,
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for PreferentialAttachmentUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "preferential_attachment"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(PrefAttachAccumulator::new()))
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
            Arc::new(Field::new("start_node", DataType::UInt64, true)),
            Arc::new(Field::new("end_node", DataType::UInt64, true)),
        ])
    }
}

#[derive(Debug)]
pub struct PrefAttachAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
    start_node: Option<u64>,
    end_node: Option<u64>,
}

impl PrefAttachAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
            start_node: None,
            end_node: None,
        }
    }
}

impl Accumulator for PrefAttachAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let sources_arr = values[0].as_any().downcast_ref::<UInt64Array>().unwrap();
        let targets_arr = values[1].as_any().downcast_ref::<UInt64Array>().unwrap();
        let start_arr = values[2].as_any().downcast_ref::<UInt64Array>().unwrap();
        let end_arr = values[3].as_any().downcast_ref::<UInt64Array>().unwrap();

        if self.start_node.is_none() && !start_arr.is_empty() && start_arr.is_valid(0) {
            self.start_node = Some(start_arr.value(0));
        }
        if self.end_node.is_none() && !end_arr.is_empty() && end_arr.is_valid(0) {
            self.end_node = Some(end_arr.value(0));
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
            .unwrap();
        let targets_list = states[1]
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .unwrap();
        let start_arr = states[2].as_any().downcast_ref::<UInt64Array>().unwrap();
        let end_arr = states[3].as_any().downcast_ref::<UInt64Array>().unwrap();

        if self.start_node.is_none() && !start_arr.is_empty() && start_arr.is_valid(0) {
            self.start_node = Some(start_arr.value(0));
        }
        if self.end_node.is_none() && !end_arr.is_empty() && end_arr.is_valid(0) {
            self.end_node = Some(end_arr.value(0));
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
        let mut sources_builder =
            arrow::array::ListBuilder::new(arrow::array::UInt64Builder::new());
        sources_builder.values().append_slice(&self.sources);
        sources_builder.append(true);
        let mut targets_builder =
            arrow::array::ListBuilder::new(arrow::array::UInt64Builder::new());
        targets_builder.values().append_slice(&self.targets);
        targets_builder.append(true);

        Ok(vec![
            ScalarValue::List(Arc::new(sources_builder.finish())),
            ScalarValue::List(Arc::new(targets_builder.finish())),
            ScalarValue::UInt64(self.start_node),
            ScalarValue::UInt64(self.end_node),
        ])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let start_node = self.start_node.unwrap_or(0);
        let end_node = self.end_node.unwrap_or(0);

        let mut graph = Graph::<u64, (), Undirected>::new_undirected();
        let mut node_map = HashMap::new();

        for i in 0..self.sources.len() {
            let s = self.sources[i];
            let t = self.targets[i];

            let s_idx = *node_map.entry(s).or_insert_with(|| graph.add_node(s));
            let t_idx = *node_map.entry(t).or_insert_with(|| graph.add_node(t));
            graph.add_edge(s_idx, t_idx, ());
        }

        let start_idx = node_map.get(&start_node);
        let end_idx = node_map.get(&end_node);

        if start_idx.is_none() || end_idx.is_none() {
            return Ok(ScalarValue::Float64(Some(0.0)));
        }

        let start_deg = graph.neighbors(*start_idx.unwrap()).count() as f64;
        let end_deg = graph.neighbors(*end_idx.unwrap()).count() as f64;

        Ok(ScalarValue::Float64(Some(start_deg * end_deg)))
    }

    fn size(&self) -> usize {
        std::mem::size_of::<Self>() + self.sources.capacity() * 8 + self.targets.capacity() * 8
    }
}
