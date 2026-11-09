// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
//
// Portions of this algorithm, lifecycle orchestration, and empirical scaling law parameters
// are adapted from:
//
//   "e-means: outperforming k-means in one-tenth the epochs"
//   Swann Bessa, Pierre-Emmanuel Mazaré, Gergely Szilvasy, Hugo Touvron, David Picard,
//   Matthijs Douze, Hervé Jégou
//   Meta FAIR (2026), arXiv:2611.09501 (https://arxiv.org/abs/2611.09501)
//
// Reference implementation in Faiss (contrib/e_means):
//   Copyright (c) Meta Platforms, Inc. and affiliates.
//   Licensed under the MIT License (https://opensource.org/licenses/MIT)

//! High-Throughput Coarse Centroid Quantizer using E-Means.
//!
//! Replaces full-batch Lloyd's algorithm for IVF and segment partition clustering
//! with stochastic Exponential Moving Average (EMA) centroid accumulation.
//!
//! Reaches lower mean squared error (MSE) than 25 iterations of Lloyd's algorithm
//! in 1.2 to 2.5 epochs, reducing disk and memory passes across Parquet vector files
//! by ~10x.

use anyhow::Result;

// Re-export core algorithmic mechanisms and scaling laws from benostream-gpu-ann
pub use benostream_gpu_ann::e_means::{
    is_emeans_eligible, resolve_alpha0, resolve_batch_size, round_power_of_two, scheduled_alpha,
    BatchLaw, BatchLawCoefficients, EMeansConfig, StepLawCoefficients, ALPHA_LAW, ALPHA_MIN_FACTOR,
    AUTO_QUALITY_MAX_D, BATCH_LAW_QUALITY, BATCH_LAW_THROUGHPUT, CHECK_UNUSED_EVERY, INIT_USAGE,
    NO_REVIVAL_MAX_D, SPLIT_EPSILON, UNUSED_THRESHOLD,
};

/// Canonical deterministic seed used throughout BenoStreamDB.
pub const DEFAULT_EMEANS_SEED: u64 = 0x00C0_FFEE_1337;

/// Train centroids using hardware-accelerated e-means clustering.
///
/// Dispatches mini-batch assignments across active GPU context (CUDA/Metal/WGPU)
/// or multi-core CPU with Rayon.
///
/// Returns `(centroids, assignments)` on success.
pub fn train_emeans(
    vectors: &[f32],
    dim: usize,
    config: &EMeansConfig,
) -> Result<(Vec<f32>, Vec<u32>)> {
    let gpu_ctx = crate::core::index::gpu::get_thread_gpu_context()
        .unwrap_or_else(crate::core::index::gpu::ComputeContext::auto_detect);
    let ann_ctx = gpu_ctx.to_gpu_ann_context();
    benostream_gpu_ann::e_means::train_emeans(vectors, dim, config, &ann_ctx)
}

/// Train centroids using e-means on a slice of vector references (`Vec<Vec<f32>>` interface).
///
/// Flattens the input vector buffers, delegates to `train_emeans`, and reconstitutes
/// centroid vectors into `Vec<Vec<f32>>`.
pub fn train_emeans_vectors(
    vectors: &[Vec<f32>],
    config: &EMeansConfig,
) -> Result<(Vec<Vec<f32>>, Vec<usize>)> {
    let n = vectors.len();
    anyhow::ensure!(n > 0, "Vector buffer must not be empty");
    let dim = vectors[0].len();
    anyhow::ensure!(dim > 0, "Vector dimension must be greater than 0");

    let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
    let (centroids_flat, labels) = train_emeans(&flat, dim, config)?;

    let centroids: Vec<Vec<f32>> = centroids_flat.chunks(dim).map(|c| c.to_vec()).collect();
    let labels_usize: Vec<usize> = labels.into_iter().map(|l| l as usize).collect();

    Ok((centroids, labels_usize))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_of_two_rounding() {
        assert_eq!(round_power_of_two(1), 1);
        assert_eq!(round_power_of_two(3), 4);
        assert_eq!(round_power_of_two(6), 8);
        assert_eq!(round_power_of_two(12), 16);
        assert_eq!(round_power_of_two(500), 512);
        assert_eq!(round_power_of_two(1000), 1024);
    }

    #[test]
    fn test_scaling_law_resolution() {
        let n = 1_000_000;
        let k = 1024;
        let dim = 128;
        let b = resolve_batch_size(n, k, 5, dim, BatchLaw::Auto);
        assert!(
            (4096..=65536).contains(&b),
            "Batch size {} out of expected range",
            b
        );

        let alpha0 = resolve_alpha0(n, k, 5, b);
        assert!(
            (0.01..=1.0).contains(&alpha0),
            "Alpha0 {} out of bounds",
            alpha0
        );
    }

    #[test]
    fn test_scheduled_alpha_decay() {
        let alpha0 = 0.5;
        let total_steps = 100;
        let start = scheduled_alpha(alpha0, 0, total_steps);
        let mid = scheduled_alpha(alpha0, 50, total_steps);
        let end = scheduled_alpha(alpha0, 100, total_steps);

        assert!((start - 0.5).abs() < 1e-4);
        assert!(mid < start && mid > end);
        assert!((end - (0.5 * ALPHA_MIN_FACTOR)).abs() < 1e-4);
    }

    #[test]
    fn test_eligibility_criteria() {
        assert!(!is_emeans_eligible(500, 16, 128)); // n too small
        assert!(!is_emeans_eligible(50_000, 2048, 128)); // n/k < 30
        assert!(!is_emeans_eligible(100_000, 1024, 2)); // dim <= 2
        assert!(is_emeans_eligible(100_000, 1024, 128)); // eligible
    }

    #[test]
    fn test_train_emeans_vectors_synthetic() {
        let k = 4;
        let dim = 8;
        let n_per_cluster = 100;

        let mut vectors = Vec::with_capacity(k * n_per_cluster);
        for c in 0..k {
            let offset = (c as f32) * 10.0;
            for _ in 0..n_per_cluster {
                let mut v = Vec::with_capacity(dim);
                for _ in 0..dim {
                    v.push(offset + 0.1);
                }
                vectors.push(v);
            }
        }

        let config = EMeansConfig {
            k,
            nepochs: 3,
            batch_law: BatchLaw::Fixed(64),
            fixed_alpha0: Some(0.5),
            unused_threshold: Some(0.1),
            seed: DEFAULT_EMEANS_SEED,
        };

        let (centroids, assignments) =
            train_emeans_vectors(&vectors, &config).expect("train_emeans_vectors failed");

        assert_eq!(centroids.len(), k);
        assert_eq!(assignments.len(), k * n_per_cluster);

        let mut unique_labels = std::collections::HashSet::new();
        for &l in &assignments {
            unique_labels.insert(l);
        }
        assert_eq!(unique_labels.len(), k);
    }
}
