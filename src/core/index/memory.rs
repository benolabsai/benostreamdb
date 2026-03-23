use anyhow::{Result, Context};
use arrow::record_batch::RecordBatch;
use arrow::array::{Array, Float32Array, FixedSizeListArray, ListArray};
// use std::sync::Arc; // Unused

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// Fallback scalar L2 distance
#[inline(always)]
fn l2_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn l2_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut i = 0;
    let mut sum_vec = _mm256_setzero_ps();

    while i + 8 <= n {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
        i += 8;
    }

    let mut sums = [0.0f32; 8];
    _mm256_storeu_ps(sums.as_mut_ptr(), sum_vec);
    let mut total = sums.iter().sum::<f32>();

    // Tail
    while i < n {
        let diff = a[i] - b[i];
        total += diff * diff;
        i += 1;
    }
    total
}

#[cfg(target_arch = "aarch64")]
unsafe fn l2_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    let mut i = 0;
    let mut sum_vec = vdupq_n_f32(0.0);

    while i + 4 <= n {
        let va = vld1q_f32(a.as_ptr().add(i));
        let vb = vld1q_f32(b.as_ptr().add(i));
        let diff = vsubq_f32(va, vb);
        sum_vec = vfmaq_f32(sum_vec, diff, diff);
        i += 4;
    }

    let total = vaddvq_f32(sum_vec);

    // Tail
    let mut tail_sum = 0.0;
    while i < n {
        let diff = a[i] - b[i];
        tail_sum += diff * diff;
        i += 1;
    }
    total + tail_sum
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { l2_distance_avx2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { l2_distance_neon(a, b) };
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        l2_distance_scalar(a, b)
    }

    #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
    {
        // Fallback for M-series if neon is disabled or x86 without AVX2
        l2_distance_scalar(a, b)
    }
}

/// Flat in-memory storage for buffering writes with brute-force search
pub struct InMemoryVectorIndex {
    vectors: Vec<f32>,
    pub count: usize,
    dim: usize,
}

impl std::fmt::Debug for InMemoryVectorIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryVectorIndex")
            .field("count", &self.count)
            .field("dim", &self.dim)
            .finish()
    }
}

impl InMemoryVectorIndex {
    pub fn new(dim: usize) -> Self {
        Self { 
            vectors: Vec::with_capacity(100_000 * dim),
            count: 0, 
            dim 
        }
    }

    /// Insert vectors from a batch.
    pub fn insert_batch(&mut self, batch: &RecordBatch, column_name: &str, _start_row_id: usize) -> Result<()> {
        let col = batch.column_by_name(column_name)
            .context(format!("Column {} not found", column_name))?;

        if let Some(fsl) = col.as_any().downcast_ref::<FixedSizeListArray>() {
            let values = fsl.values().as_any().downcast_ref::<Float32Array>()
                .context("Expected Float32Array values in FixedSizeListArray")?;
            
            // Bulk extend for high performance
            self.vectors.extend_from_slice(values.values());
            self.count += fsl.len();
        } else if let Some(list) = col.as_any().downcast_ref::<ListArray>() {
            for i in 0..list.len() {
                if list.is_null(i) {
                     // Fill with zeros to maintain alignment or handle nulls
                     self.vectors.extend(std::iter::repeat(0.0).take(self.dim));
                } else {
                    let vector_array = list.value(i);
                    if let Some(vector) = vector_array.as_any().downcast_ref::<Float32Array>() {
                        if vector.len() == self.dim {
                            self.vectors.extend_from_slice(vector.values());
                        } else {
                            self.vectors.extend(std::iter::repeat(0.0).take(self.dim));
                        }
                    }
                }
            }
            self.count += list.len();
        }
        
        Ok(())
    }

    pub fn search(&self, query: &crate::core::index::VectorValue, k: usize, filter: Option<&roaring::RoaringBitmap>) -> Vec<(usize, f32)> {
        if self.count == 0 { return vec![]; }
        
        let query_f32 = match query {
            crate::core::index::VectorValue::Float32(v) => v,
            crate::core::index::VectorValue::Float16(v) => v,
            _ => return vec![], // For now, only f32/f16 supported in memory buffer
        };
        
        use rayon::prelude::*;
        
        // Brute-force L2 search using Rayon for parallelism and SIMD for distance
        let mut results: Vec<(usize, f32)> = (0..self.count)
            .into_par_iter()
            .filter(|&i| {
                if let Some(f) = filter {
                    f.contains(i as u32)
                } else {
                    true
                }
            })
            .map(|i| {
                let start = i * self.dim;
                let end = start + self.dim;
                let vec = &self.vectors[start..end];
                
                let dist = l2_distance(vec, query_f32);
                (i, dist)
            })
            .collect();

        // Sort by distance and take top K
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    pub fn len(&self) -> usize {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{Field, Schema, DataType, Float32Type};
    use std::sync::Arc;

    #[test]
    fn test_l2_distance_consistency() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];
        
        let scalar = l2_distance_scalar(&a, &b);
        let simd = l2_distance(&a, &b);
        
        assert!((scalar - simd).abs() < 1e-5);
    }

    #[test]
    fn test_memory_index_search() {
        let dim = 4;
        let mut index = InMemoryVectorIndex::new(dim);
        
        let schema = Arc::new(Schema::new(vec![
            Field::new("vec", DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim as i32
            ), true),
        ]));

        let v1 = vec![Some(1.0), Some(0.0), Some(0.0), Some(0.0)];
        let v2 = vec![Some(0.0), Some(1.0), Some(0.0), Some(0.0)];
        let v3 = vec![Some(0.0), Some(0.0), Some(1.0), Some(0.0)];

        let array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![Some(v1), Some(v2), Some(v3)],
            dim as i32
        );

        let batch = RecordBatch::try_new(schema, vec![Arc::new(array)]).unwrap();
        index.insert_batch(&batch, "vec", 0).unwrap();

        let query = crate::core::index::VectorValue::Float32(vec![1.0, 0.1, 0.0, 0.0]);
        let results = index.search(&query, 2, None);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // v1 is closest
        assert_eq!(results[1].0, 1); // v2 is second closest
    }
}
