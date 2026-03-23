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
    let n = a.len();
    assert_eq!(n, b.len(), "Vectors must have the same length");

    // Optimization: Standard iterator with manual unrolling for common dimensions
    // The compiler can usually vectorize this well-structured loop.
    let mut sum = 0.0;
    
    // Chunked for better vectorization
    let chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);
    let rem_a = chunks.remainder();
    let rem_b = b_chunks.remainder();

    for (a_chunk, b_chunk) in chunks.zip(b_chunks) {
        let mut local_sum = 0.0;
        for i in 0..8 {
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
    let n = a.len();
    assert_eq!(n, b.len(), "Vectors must have the same length");

    let mut sum = 0.0;
    let chunks = a.chunks_exact(8);
    let b_chunks = b.chunks_exact(8);
    let rem_a = chunks.remainder();
    let rem_b = b_chunks.remainder();

    for (a_chunk, b_chunk) in chunks.zip(b_chunks) {
        for i in 0..8 {
            sum += a_chunk[i] * b_chunk[i];
        }
    }

    for (x, y) in rem_a.iter().zip(rem_b.iter()) {
        sum += x * y;
    }

    sum
}

#[inline(always)]
pub fn l1_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    assert_eq!(n, b.len(), "Vectors must have the same length");

    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += (x - y).abs();
    }
    sum
}

#[inline(always)]
pub fn hamming_distance(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    assert_eq!(n, b.len(), "Vectors must have the same length");

    let mut count = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        if x != y {
            count += 1;
        }
    }
    count as f32
}

#[inline(always)]
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
    a.iter().zip(b.iter()).map(|(&x, &y)| (x ^ y).count_ones()).sum()
}

/// Sparse dot product: intersection of two sorted index/value pairs
pub fn sparse_dot_product(
    a_indices: &[u32], a_values: &[f32],
    b_indices: &[u32], b_values: &[f32]
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
    a_indices: &[u32], a_values: &[f32],
    b_indices: &[u32], b_values: &[f32]
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
