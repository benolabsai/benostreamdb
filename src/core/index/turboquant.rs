// Copyright (c) 2026 Richard Albright. All rights reserved.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use crate::core::index::Quantizer;

/// TurboQuant Implementation (FWHT + Scalar Quantization)
/// 
/// This is the high-performance quantization engine that leverages the 
/// Fast Walsh-Hadamard Transform to provide outlier-robust scalar quantization.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboQuantEncoder {
    pub dim: usize,
    pub bits: usize,
    /// Global or segment-level min value for scaling
    pub offset: f32,
    /// Global or segment-level scale factor
    pub scale: f32,
}

impl TurboQuantEncoder {
    pub fn new(dim: usize, bits: usize, offset: f32, scale: f32) -> Self {
        Self { dim, bits, offset, scale }
    }

    /// Train the quantizer by finding the min/max values across a sample of rotated vectors.
    pub fn train(vectors: &[Vec<f32>], bits: usize) -> Result<Self> {
        let dim = vectors[0].len();
        let mut global_min = f32::MAX;
        let mut global_max = f32::MIN;

        // Sample up to 10k vectors for training
        let sample_size = vectors.len().min(10000);
        for i in 0..sample_size {
            let mut rotated = vectors[i].clone();
            // Pad to power of 2 for FWHT if necessary
            if !rotated.len().is_power_of_two() {
                let next_pow2 = rotated.len().next_power_of_two();
                rotated.resize(next_pow2, 0.0);
            }
            fwht(&mut rotated);
            
            for &val in &rotated {
                if val < global_min { global_min = val; }
                if val > global_max { global_max = val; }
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
        })
    }
}

impl Quantizer for TurboQuantEncoder {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut rotated = vector.to_vec();
        // Pad to power of 2
        if !rotated.len().is_power_of_two() {
            let next_pow2 = rotated.len().next_power_of_two();
            rotated.resize(next_pow2, 0.0);
        }
        fwht(&mut rotated);

        let max_val = ((1 << self.bits) - 1) as f32;
        let quantized: Vec<u8> = rotated.iter().map(|&val| {
            let q = ((val - self.offset) * self.scale).round();
            q.clamp(0.0, max_val) as u8
        }).collect();

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

        let mut rotated: Vec<f32> = decoded_bytes.iter().map(|&b| {
            (b as f32 / self.scale) + self.offset
        }).collect();
        
        // Invert FWHT (FWHT is self-inverse up to scaling by 1/N)
        fwht(&mut rotated);
        let n = rotated.len() as f32;
        rotated.iter_mut().for_each(|x| *x /= n);
        
        // Truncate to original dimension
        rotated.truncate(self.dim);
        rotated
    }

    fn distance_adc(&self, query_f32: &[f32], encoded: &[u8]) -> f32 {
        // Asymmetric Distance Calculation:
        // Rotate the query once, then compare with quantized bytes in the rotated space.
        let mut rotated_query = query_f32.to_vec();
        if !rotated_query.len().is_power_of_two() {
            let next_pow2 = rotated_query.len().next_power_of_two();
            rotated_query.resize(next_pow2, 0.0);
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
        format!("turboquant_{}bit", self.bits)
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
