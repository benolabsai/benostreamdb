// Copyright (c) 2026 Richard Albright. All rights reserved.

pub mod tokenizer;
pub mod ivf;
pub mod hnsw_ivf;
pub mod distance;
pub mod pq;
pub mod memory;
pub mod gpu;

use anyhow::Result;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};

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
}


pub trait VectorIndex {
    fn search(&self, query: &VectorValue, k: usize, filter: Option<&RoaringBitmap>) -> Result<Vec<(u32, f32)>>;
}
