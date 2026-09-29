// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

use anyhow::Result;
use rayon::prelude::*;
use crate::metric::Metric;
use super::GpuBackend;

pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuBackend for CpuBackend {
    fn name(&self) -> &str {
        "CPU (Rayon/SIMD)"
    }

    fn compute_distance(
        &self,
        query: &[f32],
        vectors: &[f32],
        dim: usize,
        metric: Metric,
    ) -> Result<Vec<f32>> {
        let n_vectors = vectors.len() / dim;
        let distances: Vec<f32> = (0..n_vectors)
            .into_par_iter()
            .map(|i| {
                let v = &vectors[i * dim..(i + 1) * dim];
                match metric {
                    Metric::L2 => {
                        let sum: f32 = query
                            .iter()
                            .zip(v.iter())
                            .map(|(a, b)| {
                                let d = a - b;
                                d * d
                            })
                            .sum();
                        sum.sqrt()
                    }
                    Metric::Cosine => {
                        let mut dot = 0.0f32;
                        let mut norm_q = 0.0f32;
                        let mut norm_v = 0.0f32;
                        for (&a, &b) in query.iter().zip(v.iter()) {
                            dot += a * b;
                            norm_q += a * a;
                            norm_v += b * b;
                        }
                        if norm_q == 0.0 || norm_v == 0.0 {
                            1.0
                        } else {
                            1.0 - (dot / (norm_q.sqrt() * norm_v.sqrt()))
                        }
                    }
                    Metric::InnerProduct => {
                        let dot: f32 = query.iter().zip(v.iter()).map(|(a, b)| a * b).sum();
                        -dot
                    }
                    Metric::L1 => query
                        .iter()
                        .zip(v.iter())
                        .map(|(a, b)| (a - b).abs())
                        .sum(),
                    Metric::Hamming => {
                        let mut dist = 0.0f32;
                        for (&a, &b) in query.iter().zip(v.iter()) {
                            if a != b {
                                dist += 1.0;
                            }
                        }
                        dist
                    }
                    Metric::Jaccard => {
                        let mut intersection = 0.0f32;
                        let mut union_count = 0.0f32;
                        for (&a, &b) in query.iter().zip(v.iter()) {
                            if a > 0.0 || b > 0.0 {
                                if a == b && a > 0.0 {
                                    intersection += 1.0;
                                }
                                union_count += 1.0;
                            }
                        }
                        if union_count == 0.0 {
                            0.0
                        } else {
                            1.0 - (intersection / union_count)
                        }
                    }
                }
            })
            .collect();

        Ok(distances)
    }

    fn compute_kmeans_assignment(
        &self,
        vectors: &[f32],
        centroids: &[f32],
        dim: usize,
    ) -> Result<Vec<u32>> {
        let n_vectors = vectors.len() / dim;
        let n_clusters = centroids.len() / dim;

        let assignments: Vec<u32> = (0..n_vectors)
            .into_par_iter()
            .map(|i| {
                let v = &vectors[i * dim..(i + 1) * dim];
                let mut best_dist = f32::MAX;
                let mut best_c = 0u32;

                for c in 0..n_clusters {
                    let cent = &centroids[c * dim..(c + 1) * dim];
                    let dist: f32 = v
                        .iter()
                        .zip(cent.iter())
                        .map(|(a, b)| {
                            let diff = a - b;
                            diff * diff
                        })
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_c = c as u32;
                    }
                }
                best_c
            })
            .collect();

        Ok(assignments)
    }
}
