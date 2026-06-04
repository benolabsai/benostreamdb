// Copyright (c) 2026 Richard Albright. All rights reserved.

use super::distance::l2_distance_squared;
use super::ivf::simple_kmeans;
/// Product Quantization (PQ) Implementation
///
/// PQ compresses high-dimensional vectors by splitting them into 'm' sub-vectors
/// and quantizing each sub-vector space into a small codebook (usually 256 centroids).
///
/// This allows:
/// 1. Massive memory reduction (e.g., 1536 floats -> 64 bytes = 96x reduction)
/// 2. Fast search using ADC (Asymmetric Distance Calculation) with lookup tables.
use anyhow::Result;
use tracing;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PqConfig {
    /// Number of sub-vectors (m)
    pub m: usize,
    /// Number of centroids per codebook (usually 256 for 8-bit)
    pub k: usize,
    /// Vector dimensionality
    pub dim: usize,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PqEncoder {
    pub config: PqConfig,
    /// Codebooks: m x k x (dim/m)
    pub codebooks: Vec<Vec<Vec<f32>>>,
    /// SDC Lookup Table: Flat 1D array of size m * 256 * 256 for fast SIMD gathering
    #[serde(skip)]
    pub sdc_lut: Vec<f32>,
}

impl PqEncoder {
    pub fn train(vectors: &[Vec<f32>], config: PqConfig) -> Result<Self> {
        let sub_dim = config.dim / config.m;
        tracing::info!(
            "Training PQ: m={}, k={}, dim={}, sub_dim={}",
            config.m,
            config.k,
            config.dim,
            sub_dim
        );
        use rayon::prelude::*;

        let codebooks: Result<Vec<Vec<Vec<f32>>>> = (0..config.m)
            .into_par_iter()
            .map(|i| {
                let start = i * sub_dim;
                let end = (i + 1) * sub_dim;

                // Extract sub-vectors for this subspace (Still a copy, but parallel)
                let sub_vectors: Vec<Vec<f32>> =
                    vectors.iter().map(|v| v[start..end].to_vec()).collect();

                // Train codebook for this subspace using ultra-fast mini-batch K-Means
                let (centroids, _) = simple_kmeans(&sub_vectors, config.k, 5)?;
                Ok(centroids)
            })
            .collect();

        let mut encoder = Self {
            config,
            codebooks: codebooks?,
            sdc_lut: Vec::new(),
        };
        encoder.init_lut();
        Ok(encoder)
    }

    /// Precompute the Symmetric Distance Computation (SDC) lookup table
    pub fn init_lut(&mut self) {
        if !self.sdc_lut.is_empty() {
            return;
        }
        tracing::debug!(
            "Initializing PQ SDC LUT (m={}, k={})",
            self.config.m,
            self.config.k
        );
        let mut sdc_lut = Vec::with_capacity(self.config.m * self.config.k * self.config.k);
        for i in 0..self.config.m {
            for c1 in 0..self.config.k {
                for c2 in 0..self.config.k {
                    let dist = l2_distance_squared(&self.codebooks[i][c1], &self.codebooks[i][c2]);
                    sdc_lut.push(dist);
                }
            }
        }
        self.sdc_lut = sdc_lut;
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn encode_avx2(&self, vector: &[f32]) -> Vec<u8> {
        use std::arch::x86_64::*;
        let mut encoded = Vec::with_capacity(self.config.m);
        let sub_dim = self.config.dim / self.config.m;

        for i in 0..self.config.m {
            let sub_vec = &vector[i * sub_dim..(i + 1) * sub_dim];

            let mut min_dist = f32::MAX;
            let mut best_idx = 0;

            // If sub_dim is a multiple of 8, we can use 256-bit AVX registers
            if sub_dim % 8 == 0 {
                for (j, centroid) in self.codebooks[i].iter().enumerate() {
                    let mut sum = _mm256_setzero_ps();
                    let mut k_idx = 0;
                    while k_idx < sub_dim {
                        let a_val = _mm256_loadu_ps(sub_vec.as_ptr().add(k_idx));
                        let b_val = _mm256_loadu_ps(centroid.as_ptr().add(k_idx));
                        let diff = _mm256_sub_ps(a_val, b_val);
                        let sq = _mm256_mul_ps(diff, diff);
                        sum = _mm256_add_ps(sum, sq);
                        k_idx += 8;
                    }
                    let mut res = [0.0f32; 8];
                    _mm256_storeu_ps(res.as_mut_ptr(), sum);
                    let dist: f32 = res.iter().sum();

                    if dist < min_dist {
                        min_dist = dist;
                        best_idx = j as u8;
                    }
                }
            } else {
                for (j, centroid) in self.codebooks[i].iter().enumerate() {
                    let dist = l2_distance_squared(sub_vec, centroid);
                    if dist < min_dist {
                        min_dist = dist;
                        best_idx = j as u8;
                    }
                }
            }
            encoded.push(best_idx);
        }
        encoded
    }

    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.encode_avx2(vector) };
            }
        }

        let mut encoded = Vec::with_capacity(self.config.m);
        let sub_dim = self.config.dim / self.config.m;

        for i in 0..self.config.m {
            let sub_vec = &vector[i * sub_dim..(i + 1) * sub_dim];

            let mut min_dist = f32::MAX;
            let mut best_idx = 0;

            for (j, centroid) in self.codebooks[i].iter().enumerate() {
                let dist = l2_distance_squared(sub_vec, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_idx = j as u8;
                }
            }
            encoded.push(best_idx);
        }
        encoded
    }

    /// Compute ADC (Asymmetric Distance Calculation) lookup table
    pub fn compute_lut(&self, query: &[f32]) -> Vec<Vec<f32>> {
        let sub_dim = self.config.dim / self.config.m;
        let mut lut = Vec::with_capacity(self.config.m);

        for i in 0..self.config.m {
            let start = i * sub_dim;
            let end = (i + 1) * sub_dim;
            let sub_query = &query[start..end];

            let mut sub_lut = Vec::with_capacity(self.config.k);
            for centroid in &self.codebooks[i] {
                sub_lut.push(l2_distance_squared(sub_query, centroid));
            }
            lut.push(sub_lut);
        }

        lut
    }

    /// Compute distance using LUT (very fast)
    pub fn distance_from_lut(&self, lut: &[Vec<f32>], encoded: &[u8]) -> f32 {
        let mut dist = 0.0;
        for (i, &code) in encoded.iter().enumerate() {
            dist += lut[i][code as usize];
        }
        dist
    }
}

impl crate::core::index::Quantizer for PqEncoder {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        self.encode(vector)
    }

    fn decode(&self, bytes: &[u8]) -> Vec<f32> {
        let mut vector = Vec::with_capacity(self.config.dim);
        for i in 0..self.config.m {
            let code = bytes[i] as usize;
            vector.extend_from_slice(&self.codebooks[i][code]);
        }
        vector
    }

    fn distance_adc(&self, query: &[f32], encoded: &[u8]) -> f32 {
        let sub_dim = self.config.dim / self.config.m;
        let mut dist = 0.0;
        for i in 0..self.config.m {
            let start = i * sub_dim;
            let end = (i + 1) * sub_dim;
            let sub_query = &query[start..end];
            let code = encoded[i] as usize;
            dist += l2_distance_squared(sub_query, &self.codebooks[i][code]);
        }
        dist
    }

    fn name(&self) -> String {
        format!("pq_{}", self.config.m)
    }

    fn bits(&self) -> usize {
        8
    }

    fn dim(&self) -> usize {
        self.config.dim
    }
}

use crate::core::index::hnsw_rs::dist::Distance;
use std::sync::Arc;

/// Fast Symmetric Distance Computation (SDC) for PQ using precomputed LUT
#[derive(Clone, Debug)]
pub struct DistPqSdc {
    pub pq: Arc<PqEncoder>,
}

impl Distance<u8> for DistPqSdc {
    fn eval(&self, a: &[u8], b: &[u8]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { pq_sdc_avx2(a, b, &self.pq.sdc_lut, self.pq.config.m) };
            }
        }

        let mut dist = 0.0;
        let lut = &self.pq.sdc_lut;
        for i in 0..self.pq.config.m {
            let idx = i * 65536 + (a[i] as usize) * 256 + (b[i] as usize);
            dist += lut[idx];
        }
        dist
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn pq_sdc_avx2(a: &[u8], b: &[u8], lut: &[f32], m: usize) -> f32 {
    use std::arch::x86_64::*;
    let mut sum = _mm256_setzero_ps();

    // Base offsets for 8 dimensions: [0*65536, 1*65536, 2*65536, ..., 7*65536]
    let base_offsets = _mm256_set_epi32(
        7 * 65536,
        6 * 65536,
        5 * 65536,
        4 * 65536,
        3 * 65536,
        2 * 65536,
        1 * 65536,
        0 * 65536,
    );

    let mut i = 0;
    while i + 8 <= m {
        let a_ptr = a.as_ptr().add(i) as *const i64;
        let b_ptr = b.as_ptr().add(i) as *const i64;

        let a_8 = _mm_loadl_epi64(a_ptr as *const __m128i);
        let b_8 = _mm_loadl_epi64(b_ptr as *const __m128i);

        let a_32 = _mm256_cvtepu8_epi32(a_8);
        let b_32 = _mm256_cvtepu8_epi32(b_8);

        let a_shifted = _mm256_slli_epi32(a_32, 8);
        let combined = _mm256_or_si256(a_shifted, b_32);

        let final_offsets = _mm256_add_epi32(combined, base_offsets);

        // Advance the LUT pointer by 8 dimensions * 65536 entries
        let current_lut = lut.as_ptr().add(i * 65536);
        let vals = _mm256_i32gather_ps::<4>(current_lut, final_offsets);
        sum = _mm256_add_ps(sum, vals);

        i += 8;
    }

    let mut res = [0.0f32; 8];
    _mm256_storeu_ps(res.as_mut_ptr(), sum);

    // Add remaining dimensions if m is not a multiple of 8
    let mut total = res.iter().sum::<f32>();
    while i < m {
        let offset = i * 65536 + (a[i] as usize) * 256 + (b[i] as usize);
        total += lut[offset];
        i += 1;
    }

    total
}
