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

    /// Precompute the un-clamped query state once for asymmetric distance
    /// evaluation against SQ8 codes.
    pub fn prepare_query(&self, query_f32: &[f32], metric: VectorMetric) -> PreparedSqQuery<'_> {
        let mut query = vec![0.0f32; self.dim];
        let copy_len = query_f32.len().min(self.dim);
        query[..copy_len].copy_from_slice(&query_f32[..copy_len]);

        if metric == VectorMetric::Cosine || self.metric == VectorMetric::Cosine {
            let norm = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                let inv = 1.0 / norm;
                for x in &mut query {
                    *x *= inv;
                }
            }
        }

        let q_norm_sq = query.iter().map(|x| x * x).sum::<f32>();
        PreparedSqQuery {
            encoder: self,
            query,
            q_norm_sq,
            metric,
        }
    }
}

/// Pre-processed, un-clamped query state for Asymmetric Distance Calculation
/// (ADC) against [`SqEncoder`] codes.
#[derive(Debug, Clone)]
pub struct PreparedSqQuery<'a> {
    encoder: &'a SqEncoder,
    query: Vec<f32>,
    q_norm_sq: f32,
    metric: VectorMetric,
}

impl<'a> PreparedSqQuery<'a> {
    /// Evaluate calibrated asymmetric distance against a stored SQ8 code.
    /// Guaranteed finite and non-negative.
    #[inline]
    pub fn distance(&self, encoded: &[u8]) -> f32 {
        let n = self
            .encoder
            .dim
            .min(self.query.len())
            .min(encoded.len())
            .min(self.encoder.offsets.len())
            .min(self.encoder.inv_scales.len());

        let dist = match self.metric {
            VectorMetric::Cosine | VectorMetric::InnerProduct => {
                let mut dot = 0.0f32;
                let mut v_norm_sq = 0.0f32;
                for j in 0..n {
                    let decoded =
                        (encoded[j] as f32 * self.encoder.inv_scales[j]) + self.encoder.offsets[j];
                    dot += self.query[j] * decoded;
                    v_norm_sq += decoded * decoded;
                }
                let denom = (self.q_norm_sq * v_norm_sq).sqrt();
                if denom > 1e-12 {
                    (1.0 - (dot / denom).clamp(-1.0, 1.0)).max(0.0)
                } else {
                    1.0
                }
            }
            VectorMetric::L1 => {
                let mut dist = 0.0f32;
                for j in 0..n {
                    let decoded =
                        (encoded[j] as f32 * self.encoder.inv_scales[j]) + self.encoder.offsets[j];
                    dist += (self.query[j] - decoded).abs();
                }
                dist
            }
            _ => {
                let mut dist_sq = 0.0f32;
                for j in 0..n {
                    let decoded =
                        (encoded[j] as f32 * self.encoder.inv_scales[j]) + self.encoder.offsets[j];
                    let diff = self.query[j] - decoded;
                    dist_sq += diff * diff;
                }
                dist_sq.max(0.0).sqrt()
            }
        };

        if dist.is_finite() {
            dist.max(0.0)
        } else {
            f32::MAX
        }
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
        self.prepare_query(query, self.metric).distance(encoded)
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
                    (1.0 - (dot / denom).clamp(-1.0, 1.0)).max(0.0)
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

    #[test]
    fn test_sq8_adc_calibrated_l2_and_out_of_range_query() {
        let dim = 8;
        let seg1: Vec<Vec<f32>> = vec![vec![194.0; dim], vec![195.0; dim], vec![196.0; dim]];
        let seg2: Vec<Vec<f32>> = vec![vec![197.0; dim]];
        let query = vec![197.0f32; dim];

        let enc1 = SqEncoder::train(&seg1, 8, VectorMetric::L2).unwrap();
        let enc2 = SqEncoder::train(&seg2, 8, VectorMetric::L2).unwrap();

        let d_196 = enc1
            .prepare_query(&query, VectorMetric::L2)
            .distance(&enc1.encode(&seg1[2]));
        let d_197 = enc2
            .prepare_query(&query, VectorMetric::L2)
            .distance(&enc2.encode(&seg2[0]));

        let expected_196 = (dim as f32).sqrt();
        assert!(
            (d_196 - expected_196).abs() < 0.2,
            "SQ8 out-of-range query must not clamp to 0; got {d_196}, expected ~{expected_196}"
        );
        assert!(
            d_197 < 0.05,
            "SQ8 exact single-row match must score ~0, got {d_197}"
        );
    }
}
