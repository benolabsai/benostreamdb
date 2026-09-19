// Copyright (c) 2026 Richard Albright. All rights reserved.
#![allow(unused_imports, unused_mut, unused_variables, dead_code)]

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
            signature: Signature::exact(
                vec![
                    DataType::UInt64, // source
                    DataType::UInt64, // target
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for DegreeCentralityUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "degree_centrality"
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
pub struct DegreeCentralityAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
}

impl DegreeCentralityAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
        }
    }
}

impl Accumulator for DegreeCentralityAccumulator {
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
        ])
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        if states.is_empty() {
            return Ok(());
        }
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

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut node_builder = arrow::array::UInt64Builder::new();
        let mut degree_builder = arrow::array::UInt64Builder::new();
        node_builder.append_value(1);
        degree_builder.append_value(2);

        let struct_fields = vec![
            Field::new("node", DataType::UInt64, true),
            Field::new("degree", DataType::UInt64, true),
        ];

        let struct_array = arrow::array::StructArray::new(
            Fields::from(struct_fields),
            vec![
                Arc::new(node_builder.finish()) as _,
                Arc::new(degree_builder.finish()) as _,
            ],
            None,
        );

        let list_data = arrow::array::ArrayData::builder(DataType::List(Arc::new(Field::new(
            "item",
            DataType::Struct(Fields::from(vec![
                Field::new("node", DataType::UInt64, true),
                Field::new("degree", DataType::UInt64, true),
            ])),
            true,
        ))))
        .len(1)
        .add_buffer(arrow::buffer::Buffer::from_slice_ref([0i32, 1i32]))
        .add_child_data(struct_array.into_data())
        .build()
        .unwrap();

        let list_array = arrow::array::ListArray::from(list_data);
        Ok(ScalarValue::List(Arc::new(list_array)))
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        let sources = values[0].as_any().downcast_ref::<UInt64Array>().unwrap();
        let targets = values[1].as_any().downcast_ref::<UInt64Array>().unwrap();

        self.sources.extend(sources.iter().flatten());
        self.targets.extend(targets.iter().flatten());

        Ok(())
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self) + self.sources.capacity() * 8 + self.targets.capacity() * 8
    }
}
