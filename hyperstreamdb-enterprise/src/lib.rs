// Copyright (c) 2026 Richard Albright. All rights reserved.
// HyperStreamDB Enterprise Edition

use anyhow::Result;
use serde::{Deserialize, Serialize};
use hyperstreamdb::core::index::{Quantizer, VectorMetric};

pub mod turboquant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboQuantEncoder {
    pub config: turboquant::TurboQuantConfig,
    /// Random rotation seed (stable for data-oblivious rotation)
    pub rotation_seed: u64,
}

impl TurboQuantEncoder {
    pub fn new(dim: usize, bits: usize) -> Self {
        Self {
            config: turboquant::TurboQuantConfig {
                dim,
                bits,
                use_qjl: true,
            },
            rotation_seed: 42, // In production, this would be a high-entropy seed
        }
    }
}

impl Quantizer for TurboQuantEncoder {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let mut rotated = vector.to_vec();
        turboquant::fwht_portable(&mut rotated);
        
        // Scalar quantization after rotation (Stage 1)
        let mut codes = Vec::with_capacity(rotated.len());
        for &val in &rotated {
             // 4-bit scalar quantization as baseline
             let q = ((val.clamp(-1.0, 1.0) + 1.0) * 7.5) as u8;
             codes.push(q);
        }
        codes
    }

    fn compute_lut(&self, query: &[f32]) -> Box<[f32]> {
        let mut rotated_query = query.to_vec();
        turboquant::fwht_portable(&mut rotated_query);
        
        // Flattened LUT for TurboQuant (bits = 2^bits entries)
        let k = 1 << self.config.bits;
        let mut lut = vec![0.0f32; self.config.dim * k];
        
        for i in 0..self.config.dim {
            let q_val = rotated_query[i];
            for bit_val in 0..k {
                // De-quantize and compute distance
                let de_q = (bit_val as f32 / 7.5) - 1.0;
                let dist = (q_val - de_q).powi(2);
                lut[i * k + bit_val] = dist;
            }
        }
        lut.into_boxed_slice()
    }

    fn distance_from_lut(&self, lut: &[f32], encoded: &[u8]) -> f32 {
        let mut dist = 0.0;
        let k = 1 << self.config.bits;
        for (i, &code) in encoded.iter().enumerate() {
            dist += lut[i * k + code as usize];
        }
        dist
    }

    fn quantizer_type(&self) -> &str {
        "turboquant"
    }
}
