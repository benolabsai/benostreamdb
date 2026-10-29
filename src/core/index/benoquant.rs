// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::index::{Quantizer, VectorMetric};
use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Deterministic, stateless Rademacher sign function.
/// Maps dimension index `i` deterministically to +1.0 or -1.0 using SplitMix64.
/// Eliminates the DC energy spike on non-negative feature spaces (e.g., GIST, MNIST).
#[inline(always)]
pub fn rademacher_sign(i: usize) -> f32 {
    let mut x = (i as u64).wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    if (x & 1) == 0 { 1.0 } else { -1.0 }
}

/// BenoQuant™ (Randomized Walsh-Hadamard Transform + Scalar Quantization)
///
/// High-performance vector quantization engine that leverages deterministic
/// Rademacher sign modulation combined with the Fast Walsh-Hadamard Transform (FWHT)
/// to achieve outlier-robust, isotropic scalar quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenoQuantEncoder {
    pub dim: usize,
    pub bits: usize,
    /// Global or segment-level min value for scaling
    pub offset: f32,
    /// Global or segment-level scale factor
    pub scale: f32,
    #[serde(default)]
    pub metric: VectorMetric,
}

/// Backward compatibility alias for the legacy TurboQuant name.
pub type TurboQuantEncoder = BenoQuantEncoder;

impl BenoQuantEncoder {
    pub fn new(dim: usize, bits: usize, offset: f32, scale: f32) -> Self {
        Self {
            dim,
            bits,
            offset,
            scale,
            metric: VectorMetric::L2,
        }
    }

    pub fn with_metric(mut self, metric: VectorMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Train the quantizer by finding the min/max values across a sample of rotated vectors.
    pub fn train(vectors: &[Vec<f32>], bits: usize, metric: VectorMetric) -> Result<Self> {
        let dim = vectors[0].len();
        let mut global_min = f32::MAX;
        let mut global_max = f32::MIN;

        // Sample up to 10k vectors for training
        let sample_size = vectors.len().min(10000);
        for i in 0..sample_size {
            let mut rotated = vectors[i].clone();
            if metric == VectorMetric::Cosine {
                let norm = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-12 {
                    let inv = 1.0 / norm;
                    for x in &mut rotated {
                        *x *= inv;
                    }
                } else if !rotated.is_empty() {
                    let inv_sqrt_d = 1.0 / (rotated.len() as f32).sqrt();
                    rotated.fill(inv_sqrt_d);
                }
            }
            // Pad to power of 2 for FWHT if necessary
            if !rotated.len().is_power_of_two() {
                let next_pow2 = rotated.len().next_power_of_two();
                rotated.resize(next_pow2, 0.0);
            }
            // Deterministic Rademacher sign modulation: eliminates DC spike on non-negative domains
            for (j, val) in rotated.iter_mut().enumerate() {
                *val *= rademacher_sign(j);
            }
            fwht(&mut rotated);

            for &val in &rotated {
                if val < global_min {
                    global_min = val;
                }
                if val > global_max {
                    global_max = val;
                }
            }
        }

        let scale = if (global_max - global_min).abs() < 1e-10 {
            1.0
        } else {
            ((1 << bits) - 1) as f32 / (global_max - global_min)
        };

        Ok(Self {
            dim,
            bits,
            offset: global_min,
            scale,
            metric,
        })
    }
}

impl Quantizer for BenoQuantEncoder {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut rotated = vector.to_vec();
        if self.metric == VectorMetric::Cosine {
            let norm = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                let inv = 1.0 / norm;
                for x in &mut rotated {
                    *x *= inv;
                }
            } else if !rotated.is_empty() {
                let inv_sqrt_d = 1.0 / (rotated.len() as f32).sqrt();
                rotated.fill(inv_sqrt_d);
            }
        }
        // Pad to power of 2
        if !rotated.len().is_power_of_two() {
            let next_pow2 = rotated.len().next_power_of_two();
            rotated.resize(next_pow2, 0.0);
        }
        // Deterministic Rademacher sign modulation: eliminates DC spike on non-negative domains
        for (j, val) in rotated.iter_mut().enumerate() {
            *val *= rademacher_sign(j);
        }
        fwht(&mut rotated);

        let max_val = ((1 << self.bits) - 1) as f32;
        let quantized: Vec<u8> = rotated
            .iter()
            .map(|&val| {
                let q = ((val - self.offset) * self.scale).round();
                q.clamp(0.0, max_val) as u8
            })
            .collect();

        if self.bits == 4 {
            // Pack two 4-bit nibbles into one byte
            let mut packed = Vec::with_capacity(quantized.len().div_ceil(2));
            for chunk in quantized.chunks(2) {
                let low = chunk[0] & 0x0F;
                let high = if chunk.len() > 1 { chunk[1] & 0x0F } else { 0 };
                packed.push(low | (high << 4));
            }
            packed
        } else {
            quantized
        }
    }

    fn decode(&self, bytes: &[u8]) -> Vec<f32> {
        let decoded_bytes = if self.bits == 4 {
            // Unpack nibbles
            let mut unpacked = Vec::with_capacity(bytes.len() * 2);
            for &b in bytes {
                unpacked.push(b & 0x0F);
                unpacked.push((b >> 4) & 0x0F);
            }
            unpacked
        } else {
            bytes.to_vec()
        };

        let mut rotated: Vec<f32> = decoded_bytes
            .iter()
            .map(|&b| (b as f32 / self.scale) + self.offset)
            .collect();

        // Invert FWHT (FWHT is self-inverse up to scaling by 1/N)
        fwht(&mut rotated);
        let n = rotated.len() as f32;
        rotated.iter_mut().for_each(|x| *x /= n);

        // Invert Rademacher sign modulation: s_j * s_j = 1
        for (j, val) in rotated.iter_mut().enumerate() {
            *val *= rademacher_sign(j);
        }

        // Truncate to original dimension
        rotated.truncate(self.dim);
        rotated
    }

    fn distance_adc(&self, query_f32: &[f32], encoded: &[u8]) -> f32 {
        // Asymmetric Distance Calculation:
        // Rotate the query once, then compare with quantized bytes in the rotated space.
        let mut rotated_query = query_f32.to_vec();
        if self.metric == VectorMetric::Cosine {
            let norm = rotated_query.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                let inv = 1.0 / norm;
                for x in &mut rotated_query {
                    *x *= inv;
                }
            } else if !rotated_query.is_empty() {
                let inv_sqrt_d = 1.0 / (rotated_query.len() as f32).sqrt();
                rotated_query.fill(inv_sqrt_d);
            }
        }
        if !rotated_query.len().is_power_of_two() {
            let next_pow2 = rotated_query.len().next_power_of_two();
            rotated_query.resize(next_pow2, 0.0);
        }
        for (j, val) in rotated_query.iter_mut().enumerate() {
            *val *= rademacher_sign(j);
        }
        fwht(&mut rotated_query);

        // Calculate distance in rotated space (L2)
        let mut dist_sq = 0.0;
        let inv_scale = 1.0 / self.scale;

        if self.bits == 4 {
            for (i, &b) in encoded.iter().enumerate() {
                // Low nibble
                let decoded_low = ((b & 0x0F) as f32 * inv_scale) + self.offset;
                if i * 2 < rotated_query.len() {
                    let diff = rotated_query[i * 2] - decoded_low;
                    dist_sq += diff * diff;
                }
                // High nibble
                let decoded_high = (((b >> 4) & 0x0F) as f32 * inv_scale) + self.offset;
                if i * 2 + 1 < rotated_query.len() {
                    let diff = rotated_query[i * 2 + 1] - decoded_high;
                    dist_sq += diff * diff;
                }
            }
        } else {
            for (q, &e) in rotated_query.iter().zip(encoded.iter()) {
                let decoded = (e as f32 * inv_scale) + self.offset;
                let diff = q - decoded;
                dist_sq += diff * diff;
            }
        }
        dist_sq
    }

    fn name(&self) -> String {
        format!("benoquant_{}bit", self.bits)
    }

    fn bits(&self) -> usize {
        self.bits
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// Fast Walsh-Hadamard Transform (FWHT) - Portable Implementation
pub fn fwht(data: &mut [f32]) {
    let n = data.len();
    if !n.is_power_of_two() {
        return;
    }

    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let x = data[j];
                let y = data[j + h];
                data[j] = x + y;
                data[j + h] = x - y;
            }
        }
        h <<= 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rademacher_sign_balance() {
        let count = 1024;
        let pos = (0..count).filter(|&i| rademacher_sign(i) > 0.0).count();
        let neg = count - pos;
        let diff = (pos as isize - neg as isize).abs();
        // Over 1024 dimensions, difference between +1 and -1 should be less than 80 (~5%)
        assert!(
            diff < 80,
            "Rademacher signs should be approximately balanced: pos={}, neg={}",
            pos,
            neg
        );
    }

    #[test]
    fn test_dc_spike_elimination() {
        // Create a non-negative domain vector (e.g. all 1.0s, dimension 1024)
        let dim = 1024;
        let non_negative = vec![1.0f32; dim];

        // 1. Unmodulated FWHT produces massive coordinate-0 DC spike
        let mut unmodulated = non_negative.clone();
        fwht(&mut unmodulated);
        assert_eq!(unmodulated[0], 1024.0);
        for &v in &unmodulated[1..] {
            assert_eq!(v, 0.0);
        }

        // 2. Rademacher-modulated FWHT distributes energy isotropically
        let mut modulated = non_negative.clone();
        for (j, v) in modulated.iter_mut().enumerate() {
            *v *= rademacher_sign(j);
        }
        fwht(&mut modulated);

        // Modulated coordinate 0 should no longer be 1024.0! It should be scaled O(sqrt(d))
        assert!(
            modulated[0].abs() < 100.0,
            "Modulated coordinate 0 should be bounded by O(sqrt(d)), got {}",
            modulated[0]
        );
    }

    #[test]
    fn test_benoquant_cosine_monotonicity() {
        let dim = 64;
        let mut rng_state: u64 = 42;
        let mut rand_f32 = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u32 << 31) as f32) - 1.0
        };

        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for _ in 0..200 {
            let scale = 1.0 + rand_f32().abs() * 10.0;
            let v: Vec<f32> = (0..dim).map(|_| rand_f32() * scale).collect();
            vectors.push(v);
        }

        let encoder = BenoQuantEncoder::train(&vectors, 8, VectorMetric::Cosine).unwrap();
        assert_eq!(encoder.metric, VectorMetric::Cosine);

        let encoded: Vec<Vec<u8>> = vectors.iter().map(|v| encoder.encode(v)).collect();

        let q: Vec<f32> = (0..dim).map(|_| rand_f32() * 5.0).collect();

        let mut exact_dists: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, crate::core::index::distance::cosine_distance(&q, v)))
            .collect();
        exact_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let exact_top10: std::collections::HashSet<usize> =
            exact_dists.iter().take(10).map(|(i, _)| *i).collect();

        let q_encoded = encoder.encode(&q);
        let dist_fn = crate::core::index::distance::DistL2u8;
        let mut bq_dists: Vec<(usize, f32)> = encoded
            .iter()
            .enumerate()
            .map(|(i, v)| (i, dist_fn.distance(&q_encoded, v)))
            .collect();
        bq_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let bq_top10: std::collections::HashSet<usize> =
            bq_dists.iter().take(10).map(|(i, _)| *i).collect();

        let overlap = exact_top10.intersection(&bq_top10).count();
        assert!(
            overlap >= 8,
            "BenoQuant Cosine top-10 overlap should be >= 8, got {}",
            overlap
        );
    }
}
