use arrow::array::{Array, ArrayRef, ListBuilder, UInt64Array, UInt64Builder};
use arrow::datatypes::{DataType, Field};
use datafusion::error::Result;
use datafusion::logical_expr::{AggregateUDFImpl, Signature, Volatility};
use datafusion::scalar::ScalarValue;
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_functions_aggregate_common::accumulator::{AccumulatorArgs, StateFieldsArgs};
use petgraph::graph::Graph;
use petgraph::Undirected;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct LabelPropagationUDF {
    name: String,
    signature: Signature,
}

impl PartialEq for LabelPropagationUDF {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for LabelPropagationUDF {}
impl std::hash::Hash for LabelPropagationUDF {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::any::type_name::<Self>().hash(state);
    }
}

impl Default for LabelPropagationUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl LabelPropagationUDF {
    pub fn new() -> Self {
        Self::with_name("label_propagation_communities")
    }

    pub fn new_alias() -> Self {
        Self::with_name("label_propagation")
    }

    pub fn with_name(name: &str) -> Self {
        Self {
            name: name.to_string(),
            signature: Signature::exact(
                vec![DataType::UInt64, DataType::UInt64],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for LabelPropagationUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        &self.name
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
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(LPAAccumulator::new()))
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
pub struct LPAAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
}

impl LPAAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
        }
    }
}

impl Accumulator for LPAAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let sources_arr = values[0].as_any().downcast_ref::<UInt64Array>().unwrap();
        let targets_arr = values[1].as_any().downcast_ref::<UInt64Array>().unwrap();

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
        let mut targets_builder = arrow::array::ListBuilder::new(UInt64Builder::new());
        targets_builder.values().append_slice(&self.targets);
        targets_builder.append(true);

        Ok(vec![
            ScalarValue::List(Arc::new(sources_builder.finish())),
            ScalarValue::List(Arc::new(targets_builder.finish())),
        ])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut graph = Graph::<u64, (), Undirected>::new_undirected();
        let mut node_map = HashMap::new();
        let mut reverse_map = Vec::new();

        for i in 0..self.sources.len() {
            let s = self.sources[i];
            let t = self.targets[i];

            let s_idx = *node_map.entry(s).or_insert_with(|| {
                reverse_map.push(s);
                graph.add_node(s)
            });
            let t_idx = *node_map.entry(t).or_insert_with(|| {
                reverse_map.push(t);
                graph.add_node(t)
            });
            graph.add_edge(s_idx, t_idx, ());
        }

        let num_nodes = graph.node_count();
        if num_nodes == 0 {
            let mut empty_builder = ListBuilder::new(ListBuilder::new(UInt64Builder::new()));
            return Ok(ScalarValue::List(Arc::new(empty_builder.finish())));
        }

        let mut labels: Vec<usize> = (0..num_nodes).collect();
        let mut nodes_order: Vec<usize> = (0..num_nodes).collect();
        let mut rng = thread_rng();

        let max_iter = 100;
        for _ in 0..max_iter {
            nodes_order.shuffle(&mut rng);
            let mut changed = false;

            for &n in &nodes_order {
                let n_idx = petgraph::graph::NodeIndex::new(n);
                let mut label_counts = HashMap::new();

                for neighbor in graph.neighbors(n_idx) {
                    let neighbor_label = labels[neighbor.index()];
                    *label_counts.entry(neighbor_label).or_insert(0) += 1;
                }

                if label_counts.is_empty() {
                    continue;
                }

                let max_count = label_counts.values().max().unwrap();
                let mut max_labels: Vec<usize> = label_counts
                    .iter()
                    .filter(|(_, &v)| v == *max_count)
                    .map(|(&k, _)| k)
                    .collect();

                max_labels.shuffle(&mut rng);
                let new_label = max_labels[0];

                if labels[n] != new_label {
                    labels[n] = new_label;
                    changed = true;
                }
            }

            if !changed {
                break;
            }
        }

        let mut comm_groups: HashMap<usize, Vec<u64>> = HashMap::new();
        for (i, &label) in labels.iter().enumerate() {
            comm_groups.entry(label).or_default().push(reverse_map[i]);
        }

        let inner_builder = UInt64Builder::new();
        let component_builder = ListBuilder::new(inner_builder);
        let mut final_builder = ListBuilder::new(component_builder);

        for (_, group) in comm_groups {
            for node in group {
                final_builder.values().values().append_value(node);
            }
            final_builder.values().append(true);
        }
        final_builder.append(true);

        Ok(ScalarValue::List(Arc::new(final_builder.finish())))
    }

    fn size(&self) -> usize {
        std::mem::size_of::<Self>() + self.sources.capacity() * 8 + self.targets.capacity() * 8
    }
}
