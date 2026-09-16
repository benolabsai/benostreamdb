use arrow::array::{Array, ArrayRef, Float32Array, Float64Array, Int64Array, UInt64Array};
use arrow::datatypes::{DataType, Field};
use datafusion::error::{DataFusionError, Result};
use datafusion::logical_expr::{AggregateUDFImpl, Signature, TypeSignature, Volatility};
use datafusion::scalar::ScalarValue;
use datafusion_expr_common::accumulator::Accumulator;
use datafusion_functions_aggregate_common::accumulator::{AccumulatorArgs, StateFieldsArgs};
use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ModularityUDF {
    signature: Signature,
}

impl PartialEq for ModularityUDF {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}
impl Eq for ModularityUDF {}
impl std::hash::Hash for ModularityUDF {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::any::type_name::<Self>().hash(state);
    }
}

impl Default for ModularityUDF {
    fn default() -> Self {
        Self::new()
    }
}

impl ModularityUDF {
    pub fn new() -> Self {
        Self {
            signature: Signature::one_of(
                vec![
                    // 4 args: source, target, source_community, target_community
                    TypeSignature::Exact(vec![
                        DataType::UInt64,
                        DataType::UInt64,
                        DataType::UInt64,
                        DataType::UInt64,
                    ]),
                    TypeSignature::Exact(vec![
                        DataType::Int64,
                        DataType::Int64,
                        DataType::Int64,
                        DataType::Int64,
                    ]),
                    // 5 args: source, target, weight, source_community, target_community
                    TypeSignature::Exact(vec![
                        DataType::UInt64,
                        DataType::UInt64,
                        DataType::Float64,
                        DataType::UInt64,
                        DataType::UInt64,
                    ]),
                    TypeSignature::Exact(vec![
                        DataType::UInt64,
                        DataType::UInt64,
                        DataType::Float32,
                        DataType::UInt64,
                        DataType::UInt64,
                    ]),
                    TypeSignature::Exact(vec![
                        DataType::Int64,
                        DataType::Int64,
                        DataType::Float64,
                        DataType::Int64,
                        DataType::Int64,
                    ]),
                    TypeSignature::Exact(vec![
                        DataType::Int64,
                        DataType::Int64,
                        DataType::Float32,
                        DataType::Int64,
                        DataType::Int64,
                    ]),
                    // 3 args: weight, source_community, target_community
                    TypeSignature::Exact(vec![DataType::Float32, DataType::Int64, DataType::Int64]),
                    TypeSignature::Exact(vec![DataType::Float64, DataType::Int64, DataType::Int64]),
                    TypeSignature::Exact(vec![
                        DataType::Float32,
                        DataType::UInt64,
                        DataType::UInt64,
                    ]),
                    TypeSignature::Exact(vec![
                        DataType::Float64,
                        DataType::UInt64,
                        DataType::UInt64,
                    ]),
                ],
                Volatility::Immutable,
            ),
        }
    }
}

impl AggregateUDFImpl for ModularityUDF {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn name(&self) -> &str {
        "modularity"
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        Ok(DataType::Float64)
    }
    fn accumulator(&self, _arg: AccumulatorArgs) -> Result<Box<dyn Accumulator>> {
        Ok(Box::new(ModularityAccumulator::new()))
    }
    fn state_fields(&self, _args: StateFieldsArgs) -> Result<Vec<Arc<Field>>> {
        // We only need to store the aggregated stats per community
        // But for distributed aggregation, we can store JSON or Lists.
        // For simplicity in a single node, we can store Lists of (community, a_c, e_c).
        // To make it simple for now, we'll store lists of communities, a_c, e_c and total_w.
        Ok(vec![
            Arc::new(Field::new(
                "communities",
                DataType::List(Arc::new(Field::new("item", DataType::Int64, true))),
                true,
            )),
            Arc::new(Field::new(
                "a_c",
                DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
                true,
            )),
            Arc::new(Field::new(
                "e_c",
                DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
                true,
            )),
            Arc::new(Field::new("total_w", DataType::Float64, true)),
        ])
    }
}

#[derive(Debug)]
pub struct ModularityAccumulator {
    a_c: HashMap<i64, f64>,
    e_c: HashMap<i64, f64>,
    total_w: f64,
}

impl ModularityAccumulator {
    fn new() -> Self {
        Self {
            a_c: HashMap::new(),
            e_c: HashMap::new(),
            total_w: 0.0,
        }
    }
}

impl Accumulator for ModularityAccumulator {
    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let (weights_opt, src_comm_arr, tgt_comm_arr) = if values.len() == 3 {
            (Some(&values[0]), &values[1], &values[2])
        } else if values.len() == 4 {
            (None, &values[2], &values[3])
        } else if values.len() == 5 {
            (Some(&values[2]), &values[3], &values[4])
        } else {
            return Err(DataFusionError::Execution(
                "modularity expects 3, 4, or 5 arguments".to_string(),
            ));
        };

        let get_comm = |arr: &ArrayRef, idx: usize| -> Option<i64> {
            if !arr.is_valid(idx) {
                return None;
            }
            if let Some(a) = arr.as_any().downcast_ref::<Int64Array>() {
                Some(a.value(idx))
            } else if let Some(a) = arr.as_any().downcast_ref::<UInt64Array>() {
                Some(a.value(idx) as i64)
            } else if let Some(a) = arr.as_any().downcast_ref::<arrow::array::Int32Array>() {
                Some(a.value(idx) as i64)
            } else {
                arr.as_any()
                    .downcast_ref::<arrow::array::UInt32Array>()
                    .map(|a| a.value(idx) as i64)
            }
        };

        let get_weight = |arr_opt: Option<&ArrayRef>, idx: usize| -> f64 {
            match arr_opt {
                None => 1.0,
                Some(arr) => {
                    if !arr.is_valid(idx) {
                        return 1.0;
                    }
                    if let Some(a) = arr.as_any().downcast_ref::<Float64Array>() {
                        a.value(idx)
                    } else if let Some(a) = arr.as_any().downcast_ref::<Float32Array>() {
                        a.value(idx) as f64
                    } else {
                        1.0
                    }
                }
            }
        };

        for i in 0..src_comm_arr.len() {
            if let (Some(sc), Some(tc)) = (get_comm(src_comm_arr, i), get_comm(tgt_comm_arr, i)) {
                let w = get_weight(weights_opt, i);
                self.total_w += w;
                *self.a_c.entry(sc).or_insert(0.0) += w;
                *self.a_c.entry(tc).or_insert(0.0) += w;

                if sc == tc {
                    *self.e_c.entry(sc).or_insert(0.0) += w;
                }
            }
        }

        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let comm_list = states[0]
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected ListArray for communities".to_string())
            })?;
        let ac_list = states[1]
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .ok_or_else(|| DataFusionError::Execution("Expected ListArray for a_c".to_string()))?;
        let ec_list = states[2]
            .as_any()
            .downcast_ref::<arrow::array::ListArray>()
            .ok_or_else(|| DataFusionError::Execution("Expected ListArray for e_c".to_string()))?;
        let tw_arr = states[3]
            .as_any()
            .downcast_ref::<arrow::array::Float64Array>()
            .ok_or_else(|| {
                DataFusionError::Execution("Expected Float64Array for total_w".to_string())
            })?;

        for i in 0..comm_list.len() {
            if comm_list.is_valid(i) {
                let c_arr = comm_list.value(i);
                let a_arr = ac_list.value(i);
                let e_arr = ec_list.value(i);

                let c_vals = c_arr.as_any().downcast_ref::<Int64Array>().unwrap();
                let a_vals = a_arr
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .unwrap();
                let e_vals = e_arr
                    .as_any()
                    .downcast_ref::<arrow::array::Float64Array>()
                    .unwrap();

                for j in 0..c_vals.len() {
                    let c = c_vals.value(j);
                    *self.a_c.entry(c).or_insert(0.0) += a_vals.value(j);
                    *self.e_c.entry(c).or_insert(0.0) += e_vals.value(j);
                }

                self.total_w += tw_arr.value(i);
            }
        }
        Ok(())
    }

    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        let mut comm_builder = arrow::array::ListBuilder::new(arrow::array::Int64Builder::new());
        let mut a_builder = arrow::array::ListBuilder::new(arrow::array::Float64Builder::new());
        let mut e_builder = arrow::array::ListBuilder::new(arrow::array::Float64Builder::new());

        for (&c, &a) in &self.a_c {
            let e = self.e_c.get(&c).copied().unwrap_or(0.0);
            comm_builder.values().append_value(c);
            a_builder.values().append_value(a);
            e_builder.values().append_value(e);
        }
        comm_builder.append(true);
        a_builder.append(true);
        e_builder.append(true);

        Ok(vec![
            ScalarValue::List(Arc::new(comm_builder.finish())),
            ScalarValue::List(Arc::new(a_builder.finish())),
            ScalarValue::List(Arc::new(e_builder.finish())),
            ScalarValue::Float64(Some(self.total_w)),
        ])
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        if self.total_w == 0.0 {
            return Ok(ScalarValue::Float64(Some(0.0)));
        }

        let mut q = 0.0;
        let two_m = 2.0 * self.total_w;

        for (&c, &e) in &self.e_c {
            let a = self.a_c.get(&c).copied().unwrap_or(0.0);
            let ec_over_m = e / self.total_w;
            let ac_over_2m = a / two_m;
            q += ec_over_m - (ac_over_2m * ac_over_2m);
        }

        Ok(ScalarValue::Float64(Some(q)))
    }

    fn size(&self) -> usize {
        std::mem::size_of::<Self>() + self.a_c.capacity() * 16 + self.e_c.capacity() * 16
    }
}
