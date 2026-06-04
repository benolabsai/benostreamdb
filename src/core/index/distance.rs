// Copyright (c) 2026 Richard Albright. All rights reserved.

/// Optimized distance functions for vector search
///
/// This module provides SIMD-accelerated distance metrics.
/// We use explicit loop unrolling and suggest the compiler use AVX2/NEON.
#[inline(always)]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    l2_distance_squared(a, b).sqrt()
}

#[inline(always)]
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { l2_distance_squared_avx2(a, b) };
        }
    }

    // Fallback to portable unrolled implementation
    l2_distance_squared_portable(a, b)
}

/// Portable, manually unrolled L2 distance implementation (works on all CPUs)
#[inline(always)]
fn l2_distance_squared_portable(a: &[f32], b: &[f32]) -> f32 {
    let _n = a.len();
    let mut sum = 0.0;

    let chunks = a.chunks_exact(16);
    let b_chunks = b.chunks_exact(16);
    let rem_a = chunks.remainder();
    let rem_b = b_chunks.remainder();

    for (a_chunk, b_chunk) in chunks.zip(b_chunks) {
        let mut local_sum = 0.0;
        for i in 0..16 {
            let diff = a_chunk[i] - b_chunk[i];
            local_sum += diff * diff;
        }
        sum += local_sum;
    }

    for (x, y) in rem_a.iter().zip(rem_b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn l2_distance_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    while i + 31 < n {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff0 = _mm256_sub_ps(a0, b0);
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);

        let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        let diff1 = _mm256_sub_ps(a1, b1);
        sum1 = _mm256_fmadd_ps(diff1, diff1, sum1);

        let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
        let b2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
        let diff2 = _mm256_sub_ps(a2, b2);
        sum2 = _mm256_fmadd_ps(diff2, diff2, sum2);

        let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
        let b3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));
        let diff3 = _mm256_sub_ps(a3, b3);
        sum3 = _mm256_fmadd_ps(diff3, diff3, sum3);

        i += 32;
    }

    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    while i + 7 < n {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff0 = _mm256_sub_ps(a0, b0);
        sum0 = _mm256_fmadd_ps(diff0, diff0, sum0);
        i += 8;
    }

    let mut sum_array = [0.0f32; 8];
    _mm256_storeu_ps(sum_array.as_mut_ptr(), sum0);
    let mut total = sum_array.iter().sum::<f32>();

    while i < n {
        let diff = a[i] - b[i];
        total += diff * diff;
        i += 1;
    }

    total
}

#[inline(always)]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

#[inline(always)]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = dot_product(a, a).sqrt();
    let norm_b = dot_product(b, b).sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[inline(always)]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_product_avx2(a, b) };
        }
    }
    dot_product_portable(a, b)
}

#[inline(always)]
fn dot_product_portable(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    assert_eq!(n, b.len(), "Vectors must have the same length");

    let mut sum = 0.0;
    let chunks = a.chunks_exact(16);
    let b_chunks = b.chunks_exact(16);
    let rem_a = chunks.remainder();
    let rem_b = b_chunks.remainder();

    for (a_chunk, b_chunk) in chunks.zip(b_chunks) {
        let mut local_sum = 0.0;
        for i in 0..16 {
            local_sum += a_chunk[i] * b_chunk[i];
        }
        sum += local_sum;
    }

    for (x, y) in rem_a.iter().zip(rem_b.iter()) {
        sum += x * y;
    }

    sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;

    let mut sum0 = _mm256_setzero_ps();
    let mut sum1 = _mm256_setzero_ps();
    let mut sum2 = _mm256_setzero_ps();
    let mut sum3 = _mm256_setzero_ps();

    while i + 31 < n {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);

        let a1 = _mm256_loadu_ps(a.as_ptr().add(i + 8));
        let b1 = _mm256_loadu_ps(b.as_ptr().add(i + 8));
        sum1 = _mm256_fmadd_ps(a1, b1, sum1);

        let a2 = _mm256_loadu_ps(a.as_ptr().add(i + 16));
        let b2 = _mm256_loadu_ps(b.as_ptr().add(i + 16));
        sum2 = _mm256_fmadd_ps(a2, b2, sum2);

        let a3 = _mm256_loadu_ps(a.as_ptr().add(i + 24));
        let b3 = _mm256_loadu_ps(b.as_ptr().add(i + 24));
        sum3 = _mm256_fmadd_ps(a3, b3, sum3);

        i += 32;
    }

    sum0 = _mm256_add_ps(sum0, sum1);
    sum2 = _mm256_add_ps(sum2, sum3);
    sum0 = _mm256_add_ps(sum0, sum2);

    while i + 7 < n {
        let a0 = _mm256_loadu_ps(a.as_ptr().add(i));
        let b0 = _mm256_loadu_ps(b.as_ptr().add(i));
        sum0 = _mm256_fmadd_ps(a0, b0, sum0);
        i += 8;
    }

    let mut sum_array = [0.0f32; 8];
    _mm256_storeu_ps(sum_array.as_mut_ptr(), sum0);
    let mut total = sum_array.iter().sum::<f32>();

    while i < n {
        total += a[i] * b[i];
        i += 1;
    }

    total
}

/// Vectorized batch L2 distance calculation
pub fn l2_distance_batch<V: AsRef<[f32]> + Sync>(query: &[f32], vectors: &[V]) -> Vec<f32> {
    use rayon::prelude::*;
    vectors
        .par_iter()
        .map(|vec| l2_distance(query, vec.as_ref()))
        .collect()
}

/// Vectorized batch dot product calculation
pub fn dot_product_batch<V: AsRef<[f32]> + Sync>(query: &[f32], vectors: &[V]) -> Vec<f32> {
    use rayon::prelude::*;
    vectors
        .par_iter()
        .map(|vec| dot_product(query, vec.as_ref()))
        .collect()
}

/// Vectorized batch cosine similarity calculation
pub fn cosine_similarity_batch<V: AsRef<[f32]> + Sync>(query: &[f32], vectors: &[V]) -> Vec<f32> {
    use rayon::prelude::*;
    let norm_q = dot_product(query, query).sqrt();
    if norm_q == 0.0 {
        return vec![0.0; vectors.len()];
    }

    vectors
        .par_iter()
        .map(|vec| {
            let v = vec.as_ref();
            let dot = dot_product(query, v);
            let norm_v = dot_product(v, v).sqrt();
            if norm_v == 0.0 {
                0.0
            } else {
                dot / (norm_q * norm_v)
            }
        })
        .collect()
}

#[inline(always)]
pub fn l1_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    assert_eq!(n, b.len(), "Vectors must have the same length");

    let mut sum = 0.0;
    let chunks = a.chunks_exact(16);
    let b_chunks = b.chunks_exact(16);
    let rem_a = chunks.remainder();
    let rem_b = b_chunks.remainder();

    for (a_chunk, b_chunk) in chunks.zip(b_chunks) {
        let mut local_sum = 0.0;
        for i in 0..16 {
            local_sum += (a_chunk[i] - b_chunk[i]).abs();
        }
        sum += local_sum;
    }

    for (x, y) in rem_a.iter().zip(rem_b.iter()) {
        sum += (x - y).abs();
    }
    sum
}

#[inline(always)]
pub fn hamming_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    assert_eq!(n, b.len(), "Vectors must have the same length");

    let mut count = 0;
    let chunks = a.chunks_exact(16);
    let b_chunks = b.chunks_exact(16);
    let rem_a = chunks.remainder();
    let rem_b = b_chunks.remainder();

    for (a_chunk, b_chunk) in chunks.zip(b_chunks) {
        let mut local_count = 0;
        for i in 0..16 {
            if a_chunk[i] != b_chunk[i] {
                local_count += 1;
            }
        }
        count += local_count;
    }

    for (x, y) in rem_a.iter().zip(rem_b.iter()) {
        if x != y {
            count += 1;
        }
    }
    count as f32
}

/// Jaccard distance for sets.
///
/// NOTE: This implementation expects vectors of binary indicators (0.0 or 1.0).
/// While it handles non-binary floats, the semantics of intersection/union
/// implemented here are tailored for set membership. For general float
/// similarity, consider Cosine or L2 distance.
pub fn jaccard_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    assert_eq!(n, b.len(), "Vectors must have the same length");

    let mut intersection = 0.0;
    let mut union = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        if *x > 0.0 || *y > 0.0 {
            if *x == *y && *x > 0.0 {
                intersection += 1.0;
            }
            union += 1.0;
        }
    }

    if union == 0.0 {
        return 0.0;
    }

    1.0 - (intersection / union)
}

/// Bit-optimized Hamming distance for packed binary vectors (e.g. 1 bit per element)
pub fn hamming_distance_packed(a: &[u8], b: &[u8]) -> u32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("popcnt") {
            return unsafe { hamming_distance_packed_popcnt(a, b) };
        }
    }
    hamming_distance_packed_portable(a, b)
}

#[inline(always)]
fn hamming_distance_packed_portable(a: &[u8], b: &[u8]) -> u32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x ^ y).count_ones())
        .sum()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "popcnt")]
unsafe fn hamming_distance_packed_popcnt(a: &[u8], b: &[u8]) -> u32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;
    let mut sum = 0;

    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    #[cfg(target_arch = "x86_64")]
    {
        while i + 7 < n {
            let a_val = std::ptr::read_unaligned(a_ptr.add(i) as *const u64);
            let b_val = std::ptr::read_unaligned(b_ptr.add(i) as *const u64);
            sum += _popcnt64((a_val ^ b_val) as i64) as u32;
            i += 8;
        }
    }

    while i < n {
        sum += (a[i] ^ b[i]).count_ones();
        i += 1;
    }

    sum
}

/// Sparse dot product: intersection of two sorted index/value pairs
pub fn sparse_dot_product(
    a_indices: &[u32],
    a_values: &[f32],
    b_indices: &[u32],
    b_values: &[f32],
) -> f32 {
    let mut sum = 0.0;
    let mut i = 0;
    let mut j = 0;

    while i < a_indices.len() && j < b_indices.len() {
        if a_indices[i] == b_indices[j] {
            sum += a_values[i] * b_values[j];
            i += 1;
            j += 1;
        } else if a_indices[i] < b_indices[j] {
            i += 1;
        } else {
            j += 1;
        }
    }
    sum
}

/// L2 distance for sparse vectors
pub fn sparse_l2_distance_squared(
    a_indices: &[u32],
    a_values: &[f32],
    b_indices: &[u32],
    b_values: &[f32],
) -> f32 {
    let mut sum = 0.0;
    let mut i = 0;
    let mut j = 0;

    while i < a_indices.len() && j < b_indices.len() {
        if a_indices[i] == b_indices[j] {
            let diff = a_values[i] - b_values[j];
            sum += diff * diff;
            i += 1;
            j += 1;
        } else if a_indices[i] < b_indices[j] {
            sum += a_values[i] * a_values[i];
            i += 1;
        } else {
            sum += b_values[j] * b_values[j];
            j += 1;
        }
    }

    // Add remaining squared values
    while i < a_indices.len() {
        sum += a_values[i] * a_values[i];
        i += 1;
    }
    while j < b_indices.len() {
        sum += b_values[j] * b_values[j];
        j += 1;
    }

    sum
}

/// Optimized L2 distance for quantized u8 vectors
#[inline(always)]
pub fn l2_distance_u8(a: &[u8], b: &[u8]) -> f32 {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { l2_distance_u8_avx2(a, b) };
        }
    }
    l2_distance_u8_portable(a, b)
}

#[inline(always)]
fn l2_distance_u8_portable(a: &[u8], b: &[u8]) -> f32 {
    let mut sum = 0;
    let chunks_a = a.chunks_exact(16);
    let chunks_b = b.chunks_exact(16);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();

    for (ca, cb) in chunks_a.zip(chunks_b) {
        for i in 0..16 {
            let diff = (ca[i] as i32) - (cb[i] as i32);
            sum += diff * diff;
        }
    }

    for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
        let diff = (x as i32) - (y as i32);
        sum += diff * diff;
    }
    sum as f32
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn l2_distance_u8_avx2(a: &[u8], b: &[u8]) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let n = a.len();
    let mut i = 0;
    let mut sum_vec = _mm256_setzero_si256();

    while i + 31 < n {
        let a_vec = _mm256_loadu_si256(a.as_ptr().add(i) as *const __m256i);
        let b_vec = _mm256_loadu_si256(b.as_ptr().add(i) as *const __m256i);

        let zero = _mm256_setzero_si256();

        let a_lo = _mm256_unpacklo_epi8(a_vec, zero);
        let a_hi = _mm256_unpackhi_epi8(a_vec, zero);

        let b_lo = _mm256_unpacklo_epi8(b_vec, zero);
        let b_hi = _mm256_unpackhi_epi8(b_vec, zero);

        let diff_lo = _mm256_sub_epi16(a_lo, b_lo);
        let diff_hi = _mm256_sub_epi16(a_hi, b_hi);

        let sq_lo = _mm256_madd_epi16(diff_lo, diff_lo);
        let sq_hi = _mm256_madd_epi16(diff_hi, diff_hi);

        sum_vec = _mm256_add_epi32(sum_vec, sq_lo);
        sum_vec = _mm256_add_epi32(sum_vec, sq_hi);

        i += 32;
    }

    let mut sum_array = [0i32; 8];
    _mm256_storeu_si256(sum_array.as_mut_ptr() as *mut __m256i, sum_vec);
    let mut total = sum_array.iter().sum::<i32>();

    while i < n {
        let diff = (a[i] as i32) - (b[i] as i32);
        total += diff * diff;
        i += 1;
    }

    total as f32
}

/// Asymmetric Distance Calculation (ADC) for quantized vectors.
/// Calculates L2 distance between a float32 query and a quantized u8 vector.
#[inline(always)]
pub fn l2_distance_adc(query: &[f32], encoded: &[u8], offset: f32, scale: f32) -> f32 {
    let mut sum = 0.0;
    let inv_scale = 1.0 / scale;

    // Unrolled for performance
    let chunks_q = query.chunks_exact(8);
    let chunks_e = encoded.chunks_exact(8);
    let rem_q = chunks_q.remainder();
    let rem_e = chunks_e.remainder();

    for (q_chunk, e_chunk) in chunks_q.zip(chunks_e) {
        for i in 0..8 {
            let decoded = (e_chunk[i] as f32 * inv_scale) + offset;
            let diff = q_chunk[i] - decoded;
            sum += diff * diff;
        }
    }

    for (q, e) in rem_q.iter().zip(rem_e.iter()) {
        let decoded = (*e as f32 * inv_scale) + offset;
        let diff = *q - decoded;
        sum += diff * diff;
    }
    sum
}

/// Optimized L2 distance for packed 4-bit quantized vectors (u4)
#[inline(always)]
pub fn l2_distance_u4(a: &[u8], b: &[u8]) -> f32 {
    let mut sum = 0;
    let chunks_a = a.chunks_exact(16);
    let chunks_b = b.chunks_exact(16);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();

    for (ca, cb) in chunks_a.zip(chunks_b) {
        for i in 0..16 {
            let x = ca[i];
            let y = cb[i];
            // Low nibbles
            let diff_low = ((x & 0x0F) as i32) - ((y & 0x0F) as i32);
            sum += diff_low * diff_low;
            // High nibbles
            let diff_high = ((x >> 4) as i32) - ((y >> 4) as i32);
            sum += diff_high * diff_high;
        }
    }

    for (&x, &y) in rem_a.iter().zip(rem_b.iter()) {
        let diff_low = ((x & 0x0F) as i32) - ((y & 0x0F) as i32);
        sum += diff_low * diff_low;
        let diff_high = ((x >> 4) as i32) - ((y >> 4) as i32);
        sum += diff_high * diff_high;
    }
    sum as f32
}

#[derive(Debug, Clone, Copy)]
pub struct DistL2u8;

impl DistL2u8 {
    #[inline(always)]
    pub fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        l2_distance_u8(a, b)
    }
}

impl super::hnsw_rs::dist::Distance<u8> for DistL2u8 {
    fn eval(&self, va: &[u8], vb: &[u8]) -> f32 {
        self.distance(va, vb)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DistL2u4;

impl DistL2u4 {
    #[inline(always)]
    pub fn distance(&self, a: &[u8], b: &[u8]) -> f32 {
        l2_distance_u4(a, b)
    }
}

impl super::hnsw_rs::dist::Distance<u8> for DistL2u4 {
    fn eval(&self, va: &[u8], vb: &[u8]) -> f32 {
        self.distance(va, vb)
    }
}
