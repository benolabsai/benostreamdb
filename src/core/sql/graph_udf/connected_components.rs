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
use std::collections::HashMap;
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

impl AggregateUDFImpl for ConnectedComponentsUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "connected_components"
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
pub struct ConnectedComponentsAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
}

impl ConnectedComponentsAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
        }
    }
}

impl Accumulator for ConnectedComponentsAccumulator {
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
        // Union-find over the undirected edge set. The result is a list of
        // interleaved [node, component_label] pairs, ordered by ascending
        // node id, where the component label is the minimum node id in the
        // weakly connected component.
        let mut parent: HashMap<u64, u64> = HashMap::new();

        fn find(parent: &mut HashMap<u64, u64>, x: u64) -> u64 {
            let mut root = x;
            while parent[&root] != root {
                root = parent[&root];
            }
            // Path compression
            let mut cur = x;
            while parent[&cur] != root {
                let next = parent[&cur];
                parent.insert(cur, root);
                cur = next;
            }
            root
        }

        for i in 0..self.sources.len().min(self.targets.len()) {
            let (u, v) = (self.sources[i], self.targets[i]);
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
