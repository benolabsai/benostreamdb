use arrow::record_batch::RecordBatch;
use arrow::array::{Array, UInt64Array, Float32Array, Int32Array, Int64Array, Float64Array};
use anyhow::Result;
use std::sync::Arc;
use serde_json::Value;

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

pub fn compute_zorder_scores(batch: &RecordBatch, columns: &[String]) -> Result<(UInt64Array, Vec<Value>, Vec<Value>)> {
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

pub fn compute_hilbert_scores(batch: &RecordBatch, columns: &[String]) -> Result<(UInt64Array, Vec<Value>, Vec<Value>)> {
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
        scores.push(hilbert_index(n_cols, bits_per_col, &coords));
    }

    Ok((UInt64Array::from(scores), mins, maxs))
}

/// N-dimensional Hilbert Curve indexing
pub fn hilbert_index(n: usize, bits: usize, x: &[u64]) -> u64 {
    let x_vec = x.to_vec();
    let mut m: u64 = 1 << (bits - 1);
    let mut q: u64;
    let mut p: u64;
    let mut h: u64 = 0;

    for _j in 0..bits {
        q = 0;
        for i in 0..n {
            if (x_vec[i] & m) != 0 {
                q |= 1 << i;
            }
        }
        
        p = q ^ (q >> 1);
        h = (h << n) | p as u64;
        
        m >>= 1;
    }
    h
}

fn normalize_to_u64(array: &Arc<dyn Array>, bits: usize) -> Result<(UInt64Array, Value, Value)> {
    let max_val = (1u64 << bits) - 1;
    
    match array.data_type() {
        arrow::datatypes::DataType::Int32 => {
            let arr = array.as_any().downcast_ref::<Int32Array>().unwrap();
            let mut min = i32::MAX;
            let mut max = i32::MIN;
            for i in 0..arr.len() {
                if arr.is_valid(i) {
                    let v = arr.value(i);
                    if v < min { min = v; }
                    if v > max { max = v; }
                }
            }
            let range = (max as i64 - min as i64) as f64;
            let mut normalized = Vec::with_capacity(arr.len());
            for i in 0..arr.len() {
                if arr.is_valid(i) {
                    let v = arr.value(i);
                    let norm = if range > 0.0 {
                        ((v as i64 - min as i64) as f64 / range * max_val as f64) as u64
                    } else {
                        0
                    };
                    normalized.push(norm);
                } else {
                    normalized.push(0);
                }
            }
            Ok((UInt64Array::from(normalized), Value::from(min), Value::from(max)))
        }
        arrow::datatypes::DataType::Float32 => {
            let arr = array.as_any().downcast_ref::<Float32Array>().unwrap();
            let mut min = f32::MAX;
            let mut max = f32::MIN;
            for i in 0..arr.len() {
                if arr.is_valid(i) {
                    let v = arr.value(i);
                    if v < min { min = v; }
                    if v > max { max = v; }
                }
            }
            let range = max - min;
            let mut normalized = Vec::with_capacity(arr.len());
            for i in 0..arr.len() {
                if arr.is_valid(i) {
                    let v = arr.value(i);
                    let norm = if range > 0.0 {
                        ((v - min) / range * max_val as f32) as u64
                    } else {
                        0
                    };
                    normalized.push(norm);
                } else {
                    normalized.push(0);
                }
            }
            Ok((UInt64Array::from(normalized), Value::from(min), Value::from(max)))
        }
        arrow::datatypes::DataType::Int64 => {
             let arr = array.as_any().downcast_ref::<Int64Array>().unwrap();
             let mut min = i64::MAX;
             let mut max = i64::MIN;
             for i in 0..arr.len() {
                 if arr.is_valid(i) {
                     let v = arr.value(i);
                     if v < min { min = v; }
                     if v > max { max = v; }
                 }
             }
             let range = max as f64 - min as f64;
             let mut normalized = Vec::with_capacity(arr.len());
             for i in 0..arr.len() {
                 if arr.is_valid(i) {
                     let v = arr.value(i);
                     let norm = if range > 0.0 {
                         ((v - min) as f64 / range * max_val as f64) as u64
                     } else {
                         0
                     };
                     normalized.push(norm);
                 } else {
                     normalized.push(0);
                 }
             }
             Ok((UInt64Array::from(normalized), Value::from(min), Value::from(max)))
        }
        arrow::datatypes::DataType::Float64 => {
            let arr = array.as_any().downcast_ref::<Float64Array>().unwrap();
            let mut min = f64::MAX;
            let mut max = f64::MIN;
            for i in 0..arr.len() {
                if arr.is_valid(i) {
                    let v = arr.value(i);
                    if v < min { min = v; }
                    if v > max { max = v; }
                }
            }
            let range = max - min;
            let mut normalized = Vec::with_capacity(arr.len());
            for i in 0..arr.len() {
                if arr.is_valid(i) {
                    let v = arr.value(i);
                    let norm = if range > 0.0 {
                        ((v - min) / range * max_val as f64) as u64
                    } else {
                        0
                    };
                    normalized.push(norm);
                } else {
                    normalized.push(0);
                }
            }
            Ok((UInt64Array::from(normalized), Value::from(min), Value::from(max)))
        }
        _ => Err(anyhow::anyhow!("Unsupported type for clustering: {:?}", array.data_type())),
    }
}
