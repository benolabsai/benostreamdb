// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::index::hnsw_rs::dist::Distance;
use crate::core::index::{Quantizer, VectorMetric};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Standard Uniform Scalar Quantization (SQ8)
///
/// Implements standard coordinate-wise uniform scalar quantization (matching FAISS SQ8
/// and Qdrant int8) without any orthogonal rotations. Used for baseline comparisons
/// and ablation studies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SqEncoder {
    pub dim: usize,
    pub bits: usize,
    /// Per-dimension minimum offsets: length == dim
    pub offsets: Vec<f32>,
    /// Per-dimension scaling factors: length == dim
    pub scales: Vec<f32>,
    /// Per-dimension inverse scaling factors: length == dim
    pub inv_scales: Vec<f32>,
    #[serde(default)]
    pub metric: VectorMetric,
}

impl SqEncoder {
    pub fn new(dim: usize, bits: usize, offsets: Vec<f32>, scales: Vec<f32>) -> Self {
        let inv_scales = scales
            .iter()
            .map(|&s| if s.abs() < 1e-10 { 1.0 } else { 1.0 / s })
            .collect();
        Self {
            dim,
            bits,
            offsets,
            scales,
            inv_scales,
            metric: VectorMetric::L2,
        }
    }

    pub fn with_metric(mut self, metric: VectorMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Train SQ8 by computing per-dimension min and max across sample vectors.
    pub fn train(vectors: &[Vec<f32>], bits: usize, metric: VectorMetric) -> Result<Self> {
        if vectors.is_empty() {
            anyhow::bail!("Cannot train SqEncoder with empty vector set");
        }
        let dim = vectors[0].len();
        let mut min_vals = vec![f32::MAX; dim];
        let mut max_vals = vec![f32::MIN; dim];

        let sample_size = vectors.len().min(10000);
        for i in 0..sample_size {
            // Cosine is norm-invariant. Normalize before measuring the
            // per-dimension range so the quantized codes live on the unit
            // sphere; the HNSW graph ranks SQ codes with byte-space L2, which
            // is then cosine-equivalent. Without this, L2 on the raw vectors is
            // sensitive to the ~5% norm variation and scrambles the top-k on
            // angular datasets (e.g. nytimes-256-angular).
            let mut v = vectors[i].clone();
            if metric == VectorMetric::Cosine {
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-12 {
                    let inv = 1.0 / norm;
                    for x in &mut v {
                        *x *= inv;
                    }
                }
            }
            for j in 0..dim {
                let val = v[j];
                if val < min_vals[j] {
                    min_vals[j] = val;
                }
                if val > max_vals[j] {
                    max_vals[j] = val;
                }
            }
        }

        let max_quant = ((1 << bits) - 1) as f32;
        let mut scales = Vec::with_capacity(dim);
        let mut inv_scales = Vec::with_capacity(dim);

        for j in 0..dim {
            let span = max_vals[j] - min_vals[j];
            let scale = if span.abs() < 1e-10 {
                1.0
            } else {
                max_quant / span
            };
            scales.push(scale);
            inv_scales.push(if scale.abs() < 1e-10 {
                1.0
            } else {
                1.0 / scale
            });
        }

        Ok(Self {
            dim,
            bits,
            offsets: min_vals,
            scales,
            inv_scales,
            metric,
        })
    }
}

impl Quantizer for SqEncoder {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let max_quant = ((1 << self.bits) - 1) as f32;
        let mut encoded = Vec::with_capacity(self.dim);
        // Cosine is norm-invariant: normalize before quantizing so the stored
        // codes and the query codes both live on the unit sphere. The graph
        // ranks with byte-space L2, which is then cosine-equivalent.
        let normalized: Vec<f32>;
        let v: &[f32] = if self.metric == VectorMetric::Cosine {
            let norm = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                let inv = 1.0 / norm;
                normalized = vector.iter().map(|x| x * inv).collect();
                &normalized
            } else {
                vector
            }
        } else {
            vector
        };
        for j in 0..self.dim.min(v.len()) {
            let q = ((v[j] - self.offsets[j]) * self.scales[j]).round();
            encoded.push(q.clamp(0.0, max_quant) as u8);
        }
        encoded
    }

    fn decode(&self, bytes: &[u8]) -> Vec<f32> {
        let mut decoded = Vec::with_capacity(self.dim);
        for j in 0..self.dim.min(bytes.len()) {
            decoded.push((bytes[j] as f32 * self.inv_scales[j]) + self.offsets[j]);
        }
        decoded
    }

    fn distance_adc(&self, query: &[f32], encoded: &[u8]) -> f32 {
        let mut dist_sq = 0.0;
        let n = self.dim.min(query.len()).min(encoded.len());
        for j in 0..n {
            let decoded = (encoded[j] as f32 * self.inv_scales[j]) + self.offsets[j];
            let diff = query[j] - decoded;
            dist_sq += diff * diff;
        }
        dist_sq
    }

    fn name(&self) -> String {
        format!("sq_{}bit", self.bits)
    }

    fn bits(&self) -> usize {
        self.bits
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

/// Distance over SQ8 codes that decodes each dimension with its own scale and
/// then applies the index metric.
///
/// The HNSW graph stores `u8` codes, so a plain byte-space L2 (`DistL2u8`) is
/// only correct for isotropic quantization. SQ8 uses per-dimension scales, and
/// for cosine the byte-space L2 is dominated by the vector norm: an all-zero
/// vector sits at distance ~1.0 from every query, closer than the ~1.4 between
/// real unit vectors, so it ranks first instead of last. Decoding restores the
/// metric (cosine for angular datasets, L2 otherwise).
#[derive(Clone)]
pub struct DistSq8 {
    pub encoder: Arc<SqEncoder>,
}

impl Distance<u8> for DistSq8 {
    fn eval(&self, va: &[u8], vb: &[u8]) -> f32 {
        let n = va.len().min(vb.len()).min(self.encoder.dim);
        match self.encoder.metric {
            VectorMetric::Cosine => {
                let mut dot = 0.0f32;
                let mut na = 0.0f32;
                let mut nb = 0.0f32;
                for j in 0..n {
                    let a = va[j] as f32 * self.encoder.inv_scales[j] + self.encoder.offsets[j];
                    let b = vb[j] as f32 * self.encoder.inv_scales[j] + self.encoder.offsets[j];
                    dot += a * b;
                    na += a * a;
                    nb += b * b;
                }
                let denom = (na * nb).sqrt();
                if denom > 1e-12 {
                    1.0 - dot / denom
                } else {
                    1.0
                }
            }
            _ => {
                let mut dist_sq = 0.0f32;
                for j in 0..n {
                    let a = va[j] as f32 * self.encoder.inv_scales[j] + self.encoder.offsets[j];
                    let b = vb[j] as f32 * self.encoder.inv_scales[j] + self.encoder.offsets[j];
                    let diff = a - b;
                    dist_sq += diff * diff;
                }
                dist_sq
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sq8_encode_decode_roundtrip() {
        let dim = 16;
        let vectors = vec![
            (0..dim).map(|i| i as f32 * 1.5).collect::<Vec<f32>>(),
            (0..dim).map(|i| -(i as f32) * 2.0).collect::<Vec<f32>>(),
        ];
        let encoder = SqEncoder::train(&vectors, 8, VectorMetric::L2).unwrap();
        assert_eq!(encoder.dim, dim);
        assert_eq!(encoder.bits, 8);

        let test_vec = (0..dim).map(|i| (i as f32) * 0.5).collect::<Vec<f32>>();
        let encoded = encoder.encode(&test_vec);
        assert_eq!(encoded.len(), dim);
        let decoded = encoder.decode(&encoded);
        for j in 0..dim {
            assert!((test_vec[j] - decoded[j]).abs() < 0.5);
        }
    }
}
