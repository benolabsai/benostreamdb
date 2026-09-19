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
use std::collections::{HashMap, HashSet, VecDeque};
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
pub struct ShortestPathUDF {
    signature: Signature,
}
impl_dyn_traits!(ShortestPathUDF);

impl Default for ShortestPathUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl ShortestPathUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::exact(
                vec![
                    DataType::UInt64, // source
                    DataType::UInt64, // target
                    DataType::UInt64, // start
                    DataType::UInt64, // end
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for ShortestPathUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "shortest_path"
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
        Ok(Box::new(ShortestPathAccumulator::new()))
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
            Arc::new(Field::new("start", DataType::UInt64, true)),
            Arc::new(Field::new("end", DataType::UInt64, true)),
        ])
    }
}

#[derive(Debug)]
pub struct ShortestPathAccumulator {
    sources: Vec<u64>,
    targets: Vec<u64>,
    start: Option<u64>,
    end: Option<u64>,
}

impl ShortestPathAccumulator {
    fn new() -> Self {
        Self {
            sources: Vec::new(),
            targets: Vec::new(),
            start: None,
            end: None,
        }
    }
}

impl Accumulator for ShortestPathAccumulator {
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
            ScalarValue::UInt64(self.start),
            ScalarValue::UInt64(self.end),
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

        if states.len() > 2 {
            if let Some(start_arr) = states[2].as_any().downcast_ref::<UInt64Array>() {
                if start_arr.is_valid(0) {
                    self.start = Some(start_arr.value(0));
                }
            }
        }
        if states.len() > 3 {
            if let Some(end_arr) = states[3].as_any().downcast_ref::<UInt64Array>() {
                if end_arr.is_valid(0) {
                    self.end = Some(end_arr.value(0));
                }
            }
        }

        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        let mut builder = arrow::array::ListBuilder::new(arrow::array::UInt64Builder::new());

        if let (Some(start), Some(end)) = (self.start, self.end) {
            // BFS over the directed edge set.
            let mut adj: HashMap<u64, Vec<u64>> = HashMap::new();
            for i in 0..self.sources.len().min(self.targets.len()) {
                adj.entry(self.sources[i])
                    .or_default()
                    .push(self.targets[i]);
            }

            let path = if start == end {
                Some(vec![start])
            } else {
                let mut prev: HashMap<u64, u64> = HashMap::new();
                let mut visited: HashSet<u64> = HashSet::new();
                let mut q = VecDeque::new();
                visited.insert(start);
                q.push_back(start);
                let mut found = false;

                while let Some(curr) = q.pop_front() {
                    if curr == end {
                        found = true;
                        break;
                    }
                    if let Some(neighbors) = adj.get(&curr) {
                        for &n in neighbors {
                            if visited.insert(n) {
                                prev.insert(n, curr);
                                q.push_back(n);
                            }
                        }
                    }
                }

                if found {
                    let mut path = vec![end];
                    let mut curr = end;
                    while curr != start {
                        curr = *prev.get(&curr).ok_or_else(|| {
                            DataFusionError::Execution(
                                "shortest_path: broken predecessor chain".to_string(),
                            )
                        })?;
                        path.push(curr);
                    }
                    path.reverse();
                    Some(path)
                } else {
                    None
                }
            };

            if let Some(path) = path {
                builder.values().append_slice(&path);
                builder.append(true);
            } else {
                builder.append(false); // no path found
            }
        } else {
            builder.append(false); // missing start/end arguments
        }

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

        if values.len() > 2 && !values[2].is_empty() {
            if let Some(start_arr) = values[2].as_any().downcast_ref::<UInt64Array>() {
                if start_arr.is_valid(0) {
                    self.start = Some(start_arr.value(0));
                }
            }
        }
        if values.len() > 3 && !values[3].is_empty() {
            if let Some(end_arr) = values[3].as_any().downcast_ref::<UInt64Array>() {
                if end_arr.is_valid(0) {
                    self.end = Some(end_arr.value(0));
                }
            }
        }

        Ok(())
    }

    fn size(&self) -> usize {
        std::mem::size_of_val(self) + self.sources.capacity() * 8 + self.targets.capacity() * 8
    }
}
