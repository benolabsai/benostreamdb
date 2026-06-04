// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array, UInt64Array};
use arrow::record_batch::RecordBatch;
use serde_json::Value;
use std::sync::Arc;

/// Apply Z-Order clustering to a RecordBatch
pub fn apply_zorder(batch: &RecordBatch, columns: &[String]) -> Result<RecordBatch> {
    if columns.is_empty() {
        return Ok(batch.clone());
    }

    let (scores, _, _) = compute_zorder_scores(batch, columns)?;

    // Sort batch by scores
    let indices = arrow::compute::sort_to_indices(&scores, None, None)?;
    let columns: Vec<Arc<dyn Array>> = batch
        .columns()
        .iter()
        .map(|c| arrow::compute::take(c.as_ref(), &indices, None).unwrap())
        .collect();

    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

pub fn compute_zorder_scores(
    batch: &RecordBatch,
    columns: &[String],
) -> Result<(UInt64Array, Vec<Value>, Vec<Value>)> {
    if columns.is_empty() {
        return Ok((UInt64Array::from(vec![0; batch.num_rows()]), vec![], vec![]));
    }

    let n_cols = columns.len();
    let bits_per_col = 64 / n_cols;

    let mut normalized_cols = Vec::with_capacity(n_cols);
    let mut mins = Vec::with_capacity(n_cols);
    let mut maxs = Vec::with_capacity(n_cols);

    for col_name in columns {
        let col = batch.column(batch.schema().index_of(col_name)?);
        let (norm, min, max) = normalize_to_u64(col, bits_per_col)?;
        normalized_cols.push(norm);
        mins.push(min);
        maxs.push(max);
    }

    let num_rows = batch.num_rows();
    let mut scores = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let mut row_coords = Vec::with_capacity(n_cols);
        for col in &normalized_cols {
            row_coords.push(col.value(i));
        }
        scores.push(compute_zorder_score(bits_per_col, &row_coords));
    }

    Ok((UInt64Array::from(scores), mins, maxs))
}

pub fn compute_zorder_score(bits_per_col: usize, coords: &[u64]) -> u64 {
    let mut interleaved: u64 = 0;
    for bit in 0..bits_per_col {
        for val in coords {
            let bit_val = (val >> (bits_per_col - 1 - bit)) & 1;
            interleaved = (interleaved << 1) | bit_val;
        }
    }
    interleaved
}

/// Apply Hilbert clustering to a RecordBatch
pub fn apply_hilbert(batch: &RecordBatch, columns: &[String]) -> Result<RecordBatch> {
    if columns.is_empty() {
        return Ok(batch.clone());
    }

    let (scores, _, _) = compute_hilbert_scores(batch, columns)?;

    // Sort batch by scores
    let indices = arrow::compute::sort_to_indices(&scores, None, None)?;
    let columns: Vec<Arc<dyn Array>> = batch
        .columns()
        .iter()
        .map(|c| arrow::compute::take(c.as_ref(), &indices, None).unwrap())
        .collect();

    Ok(RecordBatch::try_new(batch.schema(), columns)?)
}

pub fn compute_hilbert_scores(
    batch: &RecordBatch,
    columns: &[String],
) -> Result<(UInt64Array, Vec<Value>, Vec<Value>)> {
    if columns.is_empty() {
        return Ok((UInt64Array::from(vec![0; batch.num_rows()]), vec![], vec![]));
    }

    let n_cols = columns.len();
    let bits_per_col = 64 / n_cols;

    let mut normalized_cols = Vec::with_capacity(n_cols);
    let mut mins = Vec::with_capacity(n_cols);
    let mut maxs = Vec::with_capacity(n_cols);

    for col_name in columns {
        let col = batch.column(batch.schema().index_of(col_name)?);
        let (norm, min, max) = normalize_to_u64(col, bits_per_col)?;
        normalized_cols.push(norm);
        mins.push(min);
        maxs.push(max);
    }

    let num_rows = batch.num_rows();
    let mut scores = Vec::with_capacity(num_rows);

    for i in 0..num_rows {
        let mut coords = Vec::with_capacity(n_cols);
        for col in &normalized_cols {
            coords.push(col.value(i));
        }
        scores.push(gray_code_interleave_index(n_cols, bits_per_col, &coords));
    }

    Ok((UInt64Array::from(scores), mins, maxs))
}

/// N-dimensional Z-order curve with Gray-code interleaving.
/// This is a common approximation of a Hilbert curve.
pub fn gray_code_interleave_index(n: usize, bits: usize, x: &[u64]) -> u64 {
    let x_vec = x.to_vec();
    let mut m: u64 = 1 << (bits - 1);
    let mut q: u64;
    let mut p: u64;
    let mut h: u64 = 0;

    for _j in 0..bits {
        q = 0;
        for (i, x) in x_vec.iter().enumerate().take(n) {
            if (x & m) != 0 {
                q |= 1 << i;
            }
        }

        p = q ^ (q >> 1);
        h = (h << n) | p;

        m >>= 1;
    }
    h
}

fn normalize_to_u64(array: &Arc<dyn Array>, bits: usize) -> Result<(UInt64Array, Value, Value)> {
    let max_val = (1u64 << bits) - 1;

    macro_rules! normalize_primitive {
        ($array_type:ty, $native_type:ty) => {{
            let arr = array.as_any().downcast_ref::<$array_type>().unwrap();
            let mut min = <$native_type>::MAX;
            let mut max = <$native_type>::MIN;
            for i in 0..arr.len() {
                if arr.is_valid(i) {
                    let v = arr.value(i);
                    if v < min {
                        min = v;
                    }
                    if v > max {
                        max = v;
                    }
                }
            }
            let range = (max as f64 - min as f64);
            let mut normalized = Vec::with_capacity(arr.len());
            for i in 0..arr.len() {
                if arr.is_valid(i) {
                    let v = arr.value(i);
                    let norm = if range > 0.0 {
                        (((v as f64 - min as f64) / range) * max_val as f64) as u64
                    } else {
                        0
                    };
                    normalized.push(norm);
                } else {
                    normalized.push(0);
                }
            }
            Ok((
                UInt64Array::from(normalized),
                Value::from(min),
                Value::from(max),
            ))
        }};
    }

    match array.data_type() {
        arrow::datatypes::DataType::Int32 => normalize_primitive!(Int32Array, i32),
        arrow::datatypes::DataType::Float32 => normalize_primitive!(Float32Array, f32),
        arrow::datatypes::DataType::Int64 => normalize_primitive!(Int64Array, i64),
        arrow::datatypes::DataType::Float64 => normalize_primitive!(Float64Array, f64),
        _ => Err(anyhow::anyhow!(
            "Unsupported type for clustering: {:?}",
            array.data_type()
        )),
    }
}
