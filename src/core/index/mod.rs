// Copyright (c) 2026 Richard Albright. All rights reserved.

pub mod tokenizer;
pub mod ivf;
pub mod hnsw_ivf;
pub mod turboquant;
pub mod distance;
pub mod pq;
pub mod memory;
pub mod gpu;
pub mod hnsw_rs;

use anyhow::Result;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

/// Pluggable compression engine for vector indexing.
/// This trait allows us to unify TurboQuant, PQ, and other scalar methods.
pub trait Quantizer: std::fmt::Debug + Send + Sync {
    /// Encode a single f32 vector into quantized bytes.
    fn encode(&self, vector: &[f32]) -> Vec<u8>;
    
    /// Batch encode multiple vectors.
    fn encode_batch(&self, vectors: &[Vec<f32>]) -> Vec<Vec<u8>> {
        vectors.iter().map(|v| self.encode(v)).collect()
    }

    /// Decode quantized bytes back into an f32 vector (approximate).
    fn decode(&self, bytes: &[u8]) -> Vec<f32>;

    /// Calculate Asymmetric Distance (ADC) between a high-precision query and quantized bytes.
    /// This is the "fast path" for search.
    fn distance_adc(&self, query: &[f32], encoded: &[u8]) -> f32;

    /// Unique identifier for the quantizer (e.g., "turboquant_8bit").
    fn name(&self) -> String;

    /// Bit-depth of the quantization.
    fn bits(&self) -> usize;

    /// Dimension of the vectors handled by this quantizer.
    fn dim(&self) -> usize;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantizerImpl {
    TurboQuant(turboquant::TurboQuantEncoder),
    Pq(pq::PqEncoder),
}

impl Quantizer for QuantizerImpl {
    fn encode(&self, vector: &[f32]) -> Vec<u8> {
        match self {
            QuantizerImpl::TurboQuant(q) => q.encode(vector),
            QuantizerImpl::Pq(q) => q.encode(vector),
        }
    }

    fn decode(&self, bytes: &[u8]) -> Vec<f32> {
        match self {
            QuantizerImpl::TurboQuant(q) => q.decode(bytes),
            QuantizerImpl::Pq(q) => q.decode(bytes),
        }
    }

    fn distance_adc(&self, query: &[f32], encoded: &[u8]) -> f32 {
        match self {
            QuantizerImpl::TurboQuant(q) => q.distance_adc(query, encoded),
            QuantizerImpl::Pq(q) => q.distance_adc(query, encoded),
        }
    }

    fn name(&self) -> String {
        match self {
            QuantizerImpl::TurboQuant(q) => q.name(),
            QuantizerImpl::Pq(q) => q.name(),
        }
    }

    fn bits(&self) -> usize {
        match self {
            QuantizerImpl::TurboQuant(q) => q.bits(),
            QuantizerImpl::Pq(q) => q.bits(),
        }
    }

    fn dim(&self) -> usize {
        match self {
            QuantizerImpl::TurboQuant(q) => q.dim(),
            QuantizerImpl::Pq(q) => q.dim(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[derive(Default)]
pub enum VectorMetric {
    #[default]
    L2,
    Cosine,
    InnerProduct,
    L1,
    Hamming,
    Jaccard,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VectorType {
    Float32,
    Float16,
    Binary,
    Sparse,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseVector {
    pub indices: Vec<u32>,
    pub values: Vec<f32>,
    pub dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VectorValue {
    Float32(Vec<f32>),
    Float16(Vec<f32>), // Stored as f16, computed as f32
    Binary(Vec<u8>),
    Sparse(SparseVector),
    Keyword(String),
}


pub trait VectorIndex {
    fn search(&self, query: &VectorValue, k: usize, filter: Option<&RoaringBitmap>) -> Result<Vec<(u32, f32)>>;
}
