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
    if (x & 1) == 0 {
        1.0
    } else {
        -1.0
    }
}

/// Converts an f32 scalar into a 2-byte IEEE 754-2008 binary16 (f16) in little-endian order.
#[inline(always)]
pub fn f32_to_f16_bytes(val: f32) -> [u8; 2] {
    let bits = val.to_bits();
    let sign = (bits >> 31) & 0x1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    let h_bits = if exp == 255 {
        // Inf / NaN
        ((sign as u16) << 15) | 0x7C00 | ((mant >> 13) as u16)
    } else if exp > 142 {
        // Overflow -> Clamp to finite max (~65504.0)
        ((sign as u16) << 15) | 0x7BFF
    } else if exp < 103 {
        // Underflow -> 0
        (sign as u16) << 15
    } else if exp < 113 {
        // Subnormal
        let m = (mant | 0x800000) >> (113 - exp);
        ((sign as u16) << 15) | ((m >> 13) as u16)
    } else {
        // Normal
        let new_exp = (exp - 127 + 15) as u16;
        let new_mant = (mant >> 13) as u16;
        ((sign as u16) << 15) | (new_exp << 10) | new_mant
    };
    h_bits.to_le_bytes()
}

/// Converts a 2-byte IEEE 754-2008 binary16 (f16) into an f32.
#[inline(always)]
pub fn f16_bytes_to_f32(bytes: [u8; 2]) -> f32 {
    let h_bits = u16::from_le_bytes(bytes);
    let sign = ((h_bits >> 15) & 0x1) as u32;
    let exp = ((h_bits >> 10) & 0x1F) as u32;
    let mant = (h_bits & 0x3FF) as u32;

    let f_bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal
            let mut m = mant;
            let mut e = 113;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3FF;
            (sign << 31) | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        // Inf / NaN
        (sign << 31) | 0x7F800000 | (mant << 13)
    } else {
        // Normal
        let new_exp = exp + 127 - 15;
        (sign << 31) | (new_exp << 23) | (mant << 13)
    };
    f32::from_bits(f_bits)
}

fn default_true() -> bool {
    true
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
    #[serde(default = "default_true")]
    pub use_rademacher: bool,
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
            use_rademacher: true,
        }
    }

    pub fn with_metric(mut self, metric: VectorMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Train the quantizer by finding the min/max values across a sample of rotated vectors.
    /// BenoQuant ALWAYS uses deterministic Rademacher sign modulation ($H \cdot D$)
    /// to eliminate DC energy spikes on non-negative domains.
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
            use_rademacher: true,
        })
    }

    /// Precompute the rotated, un-clamped query state once for asymmetric
    /// distance evaluation across encoded vectors.
    pub fn prepare_query(
        &self,
        query_f32: &[f32],
        metric: VectorMetric,
    ) -> PreparedBenoQuantQuery<'_> {
        if self.dim == 0 {
            return PreparedBenoQuantQuery {
                encoder: self,
                rotated_query: Vec::new(),
                q_norm_sq: 0.0,
                inv_scale: 1.0,
                inv_n: 1.0,
                metric,
            };
        }

        let mut rotated_query = vec![0.0f32; self.dim];
        let copy_len = query_f32.len().min(self.dim);
        rotated_query[..copy_len].copy_from_slice(&query_f32[..copy_len]);

        if metric == VectorMetric::Cosine || self.metric == VectorMetric::Cosine {
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

        let n = self.dim.next_power_of_two();
        rotated_query.resize(n, 0.0);
        for (j, val) in rotated_query.iter_mut().enumerate() {
            *val *= rademacher_sign(j);
        }
        fwht(&mut rotated_query);

        let q_norm_sq = rotated_query.iter().map(|x| x * x).sum::<f32>();
        let inv_scale = if self.scale.abs() < 1e-10 {
            1.0
        } else {
            1.0 / self.scale
        };
        let inv_n = 1.0 / (n.max(1) as f32);

        PreparedBenoQuantQuery {
            encoder: self,
            rotated_query,
            q_norm_sq,
            inv_scale,
            inv_n,
            metric,
        }
    }
}

/// Pre-rotated, un-clamped query state for Asymmetric Distance Calculation (ADC)
/// against [`BenoQuantEncoder`] codes.
///
/// Computing `distance` directly in the Walsh-Hadamard rotated space avoids
/// per-candidate inverse FWHT transforms while returning calibrated Euclidean L2
/// (`sqrt(dist_sq / N)`) or Cosine (`1 - cos`) distances comparable across
/// segments with independent quantization ranges.
#[derive(Debug, Clone)]
pub struct PreparedBenoQuantQuery<'a> {
    encoder: &'a BenoQuantEncoder,
    rotated_query: Vec<f32>,
    q_norm_sq: f32,
    inv_scale: f32,
    inv_n: f32,
    metric: VectorMetric,
}

impl<'a> PreparedBenoQuantQuery<'a> {
    /// Evaluate calibrated asymmetric distance against a stored BenoQuant code.
    /// Guaranteed finite and non-negative.
    #[inline]
    pub fn distance(&self, encoded: &[u8]) -> f32 {
        let n = self.rotated_query.len();
        if n == 0 {
            return 0.0;
        }

        let dist = if self.encoder.bits == 4 {
            let packed_len = n.div_ceil(2);
            let (nibble_bytes, alpha) = if encoded.len() >= packed_len + 2 {
                let tail = encoded.len();
                let alpha_bytes = [encoded[tail - 2], encoded[tail - 1]];
                (&encoded[..packed_len], f16_bytes_to_f32(alpha_bytes))
            } else if encoded.len() >= 2 && (encoded.len() - 2) * 2 >= self.encoder.dim {
                let tail = encoded.len();
                let alpha_bytes = [encoded[tail - 2], encoded[tail - 1]];
                (&encoded[..tail - 2], f16_bytes_to_f32(alpha_bytes))
            } else {
                (encoded, 1.0f32)
            };
            let alpha = if alpha.is_finite() { alpha } else { 1.0 };
            let step = alpha * self.inv_scale;
            let base = alpha * self.encoder.offset;

            match self.metric {
                VectorMetric::Cosine | VectorMetric::InnerProduct => {
                    let mut dot = 0.0f32;
                    let mut v_norm_sq = 0.0f32;
                    let mut idx = 0usize;
                    for &b in nibble_bytes {
                        if idx >= n {
                            break;
                        }
                        let decoded_low = ((b & 0x0F) as f32) * step + base;
                        dot += self.rotated_query[idx] * decoded_low;
                        v_norm_sq += decoded_low * decoded_low;
                        idx += 1;
                        if idx < n {
                            let decoded_high = (((b >> 4) & 0x0F) as f32) * step + base;
                            dot += self.rotated_query[idx] * decoded_high;
                            v_norm_sq += decoded_high * decoded_high;
                            idx += 1;
                        }
                    }
                    let denom = (self.q_norm_sq * v_norm_sq).sqrt();
                    if denom > 1e-12 {
                        (1.0 - (dot / denom).clamp(-1.0, 1.0)).max(0.0)
                    } else {
                        1.0
                    }
                }
                _ => {
                    let mut dist_sq = 0.0f32;
                    let mut idx = 0usize;
                    for &b in nibble_bytes {
                        if idx >= n {
                            break;
                        }
                        let decoded_low = ((b & 0x0F) as f32) * step + base;
                        let diff_low = self.rotated_query[idx] - decoded_low;
                        dist_sq += diff_low * diff_low;
                        idx += 1;
                        if idx < n {
                            let decoded_high = (((b >> 4) & 0x0F) as f32) * step + base;
                            let diff_high = self.rotated_query[idx] - decoded_high;
                            dist_sq += diff_high * diff_high;
                            idx += 1;
                        }
                    }
                    (dist_sq * self.inv_n).max(0.0).sqrt()
                }
            }
        } else {
            let count = n.min(encoded.len());
            match self.metric {
                VectorMetric::Cosine | VectorMetric::InnerProduct => {
                    let mut dot = 0.0f32;
                    let mut v_norm_sq = 0.0f32;
                    for (&q, &e) in self.rotated_query[..count].iter().zip(&encoded[..count]) {
                        let decoded = (e as f32) * self.inv_scale + self.encoder.offset;
                        dot += q * decoded;
                        v_norm_sq += decoded * decoded;
                    }
                    let denom = (self.q_norm_sq * v_norm_sq).sqrt();
                    if denom > 1e-12 {
                        (1.0 - (dot / denom).clamp(-1.0, 1.0)).max(0.0)
                    } else {
                        1.0
                    }
                }
                _ => {
                    let mut dist_sq = 0.0f32;
                    for (&q, &e) in self.rotated_query[..count].iter().zip(&encoded[..count]) {
                        let decoded = (e as f32) * self.inv_scale + self.encoder.offset;
                        let diff = q - decoded;
                        dist_sq += diff * diff;
                    }
                    (dist_sq * self.inv_n).max(0.0).sqrt()
                }
            }
        };

        if dist.is_finite() {
            dist.max(0.0)
        } else {
            f32::MAX
        }
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
            // Compute optimal 2-byte least-squares length renormalization scalar:
            // alpha = <y, y_hat> / ||y_hat||^2
            let mut dot = 0.0f32;
            let mut norm_sq = 0.0f32;
            let inv_scale = 1.0 / self.scale;
            for (&y, &q) in rotated.iter().zip(quantized.iter()) {
                let y_hat = (q as f32 * inv_scale) + self.offset;
                dot += y * y_hat;
                norm_sq += y_hat * y_hat;
            }
            let alpha = if norm_sq > 1e-12 {
                (dot / norm_sq).clamp(0.01, 100.0)
            } else {
                1.0
            };
            let alpha_bytes = f32_to_f16_bytes(alpha);

            // Pack two 4-bit nibbles into one byte
            let mut packed = Vec::with_capacity(quantized.len().div_ceil(2) + 2);
            for chunk in quantized.chunks(2) {
                let low = chunk[0] & 0x0F;
                let high = if chunk.len() > 1 { chunk[1] & 0x0F } else { 0 };
                packed.push(low | (high << 4));
            }
            // Append 2-byte float16 scalar
            packed.extend_from_slice(&alpha_bytes);
            packed
        } else {
            quantized
        }
    }

    fn decode(&self, bytes: &[u8]) -> Vec<f32> {
        let (decoded_bytes, alpha) = if self.bits == 4 {
            // Check for trailing 2-byte f16 scalar
            let (nibble_bytes, alpha) = if bytes.len() >= 2 && (bytes.len() - 2) * 2 >= self.dim {
                let alpha_bytes = [bytes[bytes.len() - 2], bytes[bytes.len() - 1]];
                (&bytes[..bytes.len() - 2], f16_bytes_to_f32(alpha_bytes))
            } else {
                (bytes, 1.0f32)
            };
            let mut unpacked = Vec::with_capacity(nibble_bytes.len() * 2);
            for &b in nibble_bytes {
                unpacked.push(b & 0x0F);
                unpacked.push((b >> 4) & 0x0F);
            }
            (unpacked, alpha)
        } else {
            (bytes.to_vec(), 1.0f32)
        };

        let mut rotated: Vec<f32> = decoded_bytes
            .iter()
            .map(|&b| ((b as f32 / self.scale) + self.offset) * alpha)
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
        self.prepare_query(query_f32, self.metric).distance(encoded)
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

    #[test]
    fn test_benoquant_4bit_recall_monotonicity() {
        let dim = 64;
        let mut rng_state: u64 = 12345;
        let mut rand_f32 = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u32 << 31) as f32) - 1.0
        };

        let mut vectors: Vec<Vec<f32>> = Vec::new();
        for _ in 0..200 {
            let scale = 1.0 + rand_f32().abs() * 5.0;
            let v: Vec<f32> = (0..dim).map(|_| rand_f32() * scale).collect();
            vectors.push(v);
        }

        let encoder = BenoQuantEncoder::train(&vectors, 4, VectorMetric::L2).unwrap();
        assert_eq!(encoder.bits, 4);

        let encoded: Vec<Vec<u8>> = vectors.iter().map(|v| encoder.encode(v)).collect();
        // 64 dimensions / 2 nibbles per byte + 2 bytes f16 scalar = 34 bytes
        assert_eq!(encoded[0].len(), dim / 2 + 2);

        let q: Vec<f32> = (0..dim).map(|_| rand_f32() * 3.0).collect();

        let mut exact_dists: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, crate::core::index::distance::l2_distance(&q, v)))
            .collect();
        exact_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let exact_top10: std::collections::HashSet<usize> =
            exact_dists.iter().take(10).map(|(i, _)| *i).collect();

        let q_encoded = encoder.encode(&q);
        assert_eq!(q_encoded.len(), dim / 2 + 2);

        let dist_fn = crate::core::index::distance::DistL2u4;
        let self_dist = dist_fn.distance(&q_encoded, &q_encoded);
        assert!(
            self_dist.abs() < 1e-4,
            "Self distance should be ~0, got {self_dist}"
        );

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
            overlap >= 6,
            "BenoQuant 4-bit L2 top-10 overlap should be >= 6 (got {})",
            overlap
        );
    }

    #[test]
    fn test_benoquant_adc_calibrated_l2_and_out_of_range_query() {
        let dim = 8;
        let seg1: Vec<Vec<f32>> = vec![vec![0.0; dim], vec![25.0; dim], vec![50.0; dim]];
        let seg2: Vec<Vec<f32>> = vec![vec![194.0; dim], vec![197.0; dim], vec![200.0; dim]];
        let query = vec![197.0f32; dim];

        for bits in [8usize, 4usize] {
            let enc1 = BenoQuantEncoder::train(&seg1, bits, VectorMetric::L2).unwrap();
            let enc2 = BenoQuantEncoder::train(&seg2, bits, VectorMetric::L2).unwrap();

            let code_0 = enc1.encode(&seg1[0]);
            let code_50 = enc1.encode(&seg1[2]);
            let code_197 = enc2.encode(&seg2[1]);

            let prep1 = enc1.prepare_query(&query, VectorMetric::L2);
            let d_0 = prep1.distance(&code_0);
            let d_50 = prep1.distance(&code_50);
            let d_197 = enc2
                .prepare_query(&query, VectorMetric::L2)
                .distance(&code_197);

            let expected_50 = (197.0 - 50.0) * (dim as f32).sqrt();
            let tol_far = if bits == 8 { 5.0 } else { 35.0 };
            let max_exact = if bits == 8 { 3.0 } else { 45.0 };
            assert!(
                (d_50 - expected_50).abs() < tol_far,
                "bits={bits}: out-of-range query to [50; 8] must stay calibrated near {expected_50}, got {d_50}"
            );
            assert!(
                d_50 < d_0,
                "bits={bits}: out-of-range ordering [50; 8] ({d_50}) must beat [0; 8] ({d_0})"
            );
            assert!(
                d_197 < max_exact,
                "bits={bits}: exact match [197; 8] must score < {max_exact}, got {d_197}"
            );
            assert!(
                d_197 < d_50,
                "bits={bits}: exact match {d_197} must beat out-of-range neighbor {d_50}"
            );
        }
    }

    #[test]
    fn test_benoquant_adc_cosine() {
        let dim = 16;
        let mut rng_state: u64 = 99;
        let mut rand_f32 = || {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng_state >> 33) as f32 / (1u32 << 31) as f32) - 1.0
        };
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..dim).map(|_| rand_f32() * 5.0).collect())
            .collect();
        let enc = BenoQuantEncoder::train(&vectors, 8, VectorMetric::Cosine).unwrap();
        let q = vectors[7].clone();
        let prep = enc.prepare_query(&q, VectorMetric::Cosine);
        let self_dist = prep.distance(&enc.encode(&q));
        assert!(self_dist < 0.02, "self cosine distance got {self_dist}");

        let opp: Vec<f32> = q.iter().map(|x| -x).collect();
        let opp_dist = prep.distance(&enc.encode(&opp));
        assert!(
            opp_dist > 1.8,
            "opposite cosine distance should be ~2.0, got {opp_dist}"
        );
    }
}
