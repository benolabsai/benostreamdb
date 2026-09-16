use std::any::Any;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Builder, UInt64Array, UInt64Builder};
use arrow::datatypes::{DataType, Field};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{AggregateUDFImpl, Signature, Volatility};
use datafusion::scalar::ScalarValue;
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_functions_aggregate_common::accumulator::{AccumulatorArgs, StateFieldsArgs};

use petgraph::graphmap::UnGraphMap;

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

#[derive(Debug)]
pub struct ClusteringCoefficientUDF {
    signature: Signature,
}
impl_dyn_traits!(ClusteringCoefficientUDF);

impl Default for ClusteringCoefficientUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl ClusteringCoefficientUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![DataType::UInt64, DataType::UInt64],
                Volatility::Immutable,
            ),
        }
    }

    fn return_type_struct() -> DataType {
        DataType::Struct(
            vec![
                Field::new("node", DataType::UInt64, false),
                Field::new("clustering_coefficient", DataType::Float64, false),
            ]
            .into(),
        )
    }
}

impl AggregateUDFImpl for ClusteringCoefficientUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "clustering_coefficient"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::List(Arc::new(Field::new(
            "item",
            Self::return_type_struct(),
            true,
        ))))
    }
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(ClusteringCoefficientAccumulator::new()))
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
pub struct ClusteringCoefficientAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
}

impl ClusteringCoefficientAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
        }
    }
}

impl Accumulator for ClusteringCoefficientAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() != 2 {
            return Err(DataFusionError::Execution(
                "clustering_coefficient expects 2 arguments".to_string(),
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
                if let Some(s) = s_arr.as_any().downcast_ref::<arrow::array::UInt64Array>() {
                    self.sources.extend_from_slice(s.values());
                }
            }
            if targets_list.is_valid(i) {
                let t_arr = targets_list.value(i);
                if let Some(t) = t_arr.as_any().downcast_ref::<arrow::array::UInt64Array>() {
                    self.targets.extend_from_slice(t.values());
                }
            }
        }
        Ok(())
    }

    fn state(&mut self) -> Result<Vec<datafusion::scalar::ScalarValue>> {
        let mut sources_builder = arrow::array::ListBuilder::new(UInt64Builder::new());
        sources_builder.values().append_slice(&self.sources);
        sources_builder.append(true);
        let sources_list =
            datafusion::scalar::ScalarValue::List(Arc::new(sources_builder.finish()));

        let mut targets_builder = arrow::array::ListBuilder::new(UInt64Builder::new());
        targets_builder.values().append_slice(&self.targets);
        targets_builder.append(true);
        let targets_list =
            datafusion::scalar::ScalarValue::List(Arc::new(targets_builder.finish()));

        Ok(vec![sources_list, targets_list])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut graph = UnGraphMap::<u64, ()>::new();

        for (s, t) in self.sources.iter().zip(self.targets.iter()) {
            graph.add_edge(*s, *t, ());
        }

        let nodes: Vec<u64> = graph.nodes().collect();
        let num_nodes = nodes.len();

        if num_nodes == 0 {
            let fields = vec![
                Field::new("node", DataType::UInt64, false),
                Field::new("clustering_coefficient", DataType::Float64, false),
            ];
            let struct_type = DataType::Struct(fields.into());
            return Ok(ScalarValue::List(ScalarValue::new_list(
                &[],
                &struct_type,
                true,
            )));
        }

        let mut node_id_builder = UInt64Builder::new();
        let mut cc_builder = Float64Builder::new();

        for &node in &nodes {
            let neighbors: Vec<u64> = graph.neighbors(node).collect();
            let degree = neighbors.len() as f64;
            let mut cc = 0.0;

            if degree >= 2.0 {
                let mut edges_between = 0.0;
                for i in 0..neighbors.len() {
                    for j in (i + 1)..neighbors.len() {
                        if graph.contains_edge(neighbors[i], neighbors[j]) {
                            edges_between += 1.0;
                        }
                    }
                }
                cc = (2.0 * edges_between) / (degree * (degree - 1.0));
            }

            node_id_builder.append_value(node);
            cc_builder.append_value(cc);
        }

        let node_id_array = Arc::new(node_id_builder.finish()) as ArrayRef;
        let cc_array = Arc::new(cc_builder.finish()) as ArrayRef;

        let struct_fields = vec![
            Arc::new(Field::new("node", DataType::UInt64, false)),
            Arc::new(Field::new(
                "clustering_coefficient",
                DataType::Float64,
                false,
            )),
        ];

        let struct_array = arrow::array::StructArray::from(vec![
            (struct_fields[0].clone(), node_id_array),
            (struct_fields[1].clone(), cc_array),
        ]);

        let list_fields = Arc::new(Field::new(
            "item",
            DataType::Struct(struct_fields.into()),
            true,
        ));

        let offsets = arrow::buffer::OffsetBuffer::from_lengths(vec![num_nodes]);
        let list_array =
            arrow::array::ListArray::new(list_fields, offsets, Arc::new(struct_array), None);

        Ok(ScalarValue::List(Arc::new(list_array)))
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self)
            + self.sources.capacity() * std::mem::size_of::<u64>()
            + self.targets.capacity() * std::mem::size_of::<u64>()
    }
}
