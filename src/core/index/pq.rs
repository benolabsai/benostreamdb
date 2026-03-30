// Copyright (c) 2026 Richard Albright. All rights reserved.

/// Product Quantization (PQ) Implementation
/// 
/// PQ compresses high-dimensional vectors by splitting them into 'm' sub-vectors
/// and quantizing each sub-vector space into a small codebook (usually 256 centroids).
/// 
/// This allows:
/// 1. Massive memory reduction (e.g., 1536 floats -> 64 bytes = 96x reduction)
/// 2. Fast search using ADC (Asymmetric Distance Calculation) with lookup tables.
use anyhow::Result;
use super::distance::{l2_distance_squared};
use super::ivf::simple_kmeans;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct PqConfig {
    /// Number of sub-vectors (m)
    pub m: usize,
    /// Number of centroids per codebook (usually 256 for 8-bit)
    pub k: usize,
    /// Vector dimensionality
    pub dim: usize,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct PqEncoder {
    pub config: PqConfig,
    /// Codebooks: m x k x (dim/m)
    pub codebooks: Vec<Vec<Vec<f32>>>,
}

impl PqEncoder {
    pub fn train(vectors: &[Vec<f32>], config: PqConfig) -> Result<Self> {
        let sub_dim = config.dim / config.m;
        let mut codebooks = Vec::with_capacity(config.m);

        println!("Training PQ: m={}, k={}, dim={}, sub_dim={}", config.m, config.k, config.dim, sub_dim);

        for i in 0..config.m {
            let start = i * sub_dim;
            let end = (i + 1) * sub_dim;
            
            // Extract sub-vectors for this subspace
            let sub_vectors: Vec<Vec<f32>> = vectors.iter()
                .map(|v| v[start..end].to_vec())
                .collect();
            
            // Train codebook for this subspace
            // Optimization: Use fewer iterations for PQ training speed
            let (centroids, _) = simple_kmeans(&sub_vectors, config.k, 5)?;
            codebooks.push(centroids);
            println!("  - Trained codebook for subspace {}/{}", i + 1, config.m);
        }

        Ok(Self { config, codebooks })
    }

    pub fn encode(&self, vector: &[f32]) -> Vec<u8> {
        let sub_dim = self.config.dim / self.config.m;
        let mut codes = Vec::with_capacity(self.config.m);

        for i in 0..self.config.m {
            let start = i * sub_dim;
            let end = (i + 1) * sub_dim;
            let sub_vec = &vector[start..end];
            
            // Find nearest centroid in this subspace
            let mut min_dist = f32::MAX;
            let mut best_idx = 0;
            
            for (idx, centroid) in self.codebooks[i].iter().enumerate() {
                let dist = l2_distance_squared(sub_vec, centroid);
                if dist < min_dist {
                    min_dist = dist;
                    best_idx = idx;
                }
            }
            codes.push(best_idx as u8);
        }

        codes
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
