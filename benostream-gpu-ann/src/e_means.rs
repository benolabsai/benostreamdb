// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.
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
//! in 1.2 to 2.5 epochs, reducing memory bus sweeps and vector scans by ~10x.

use crate::backend::{ComputeContext, GpuBackend};
use anyhow::Result;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

/// Minimum step factor for the cosine schedule relative to alpha0 (Equation 5).
pub const ALPHA_MIN_FACTOR: f32 = 0.1;

/// Batch update interval between dead-cluster revival checks (Algorithm 1, line 9).
pub const CHECK_UNUSED_EVERY: usize = 5;

/// Initial EMA mass N_j (Algorithm 1, line 1); c_j represents S_j / N_j.
pub const INIT_USAGE: f32 = 1.0 / 65536.0; // 2.0^-16

/// Coordinate perturbation delta for donor splits (Appendix J, Eq. 8).
pub const SPLIT_EPSILON: f32 = 1e-4;

/// Revival threshold tau: revive centroids with N_j < tau * mean(N).
pub const UNUSED_THRESHOLD: f32 = 0.1;

/// Minimum dimension threshold below which dead-cluster revival is disabled
/// to prevent dimensional collapse.
pub const NO_REVIVAL_MAX_D: usize = 2;

/// Dimension threshold below which the quality batch law is automatically chosen.
pub const AUTO_QUALITY_MAX_D: usize = 16;

/// Power-law coefficients for batch size derivation: B = C_B * k^b_k * n^b_n * T^b_T.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatchLawCoefficients {
    pub c_b: f64,
    pub b_k: f64,
    pub b_n: f64,
    pub b_t: f64,
}

/// Shipped throughput scaling law parameters from Meta FAIR.
pub const BATCH_LAW_THROUGHPUT: BatchLawCoefficients = BatchLawCoefficients {
    c_b: 0.035134,
    b_k: 0.134749,
    b_n: 0.876473,
    b_t: 0.813344,
};

/// Shipped quality scaling law parameters from Meta FAIR (1/8th throughput coefficient).
pub const BATCH_LAW_QUALITY: BatchLawCoefficients = BatchLawCoefficients {
    c_b: 0.00439175,
    b_k: 0.134749,
    b_n: 0.876473,
    b_t: 0.813344,
};

/// Power-law coefficients for initial EMA step size: alpha0 = C_a * k^a_k * B^a_B * n^a_n * T^a_T.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepLawCoefficients {
    pub c_alpha: f64,
    pub a_k: f64,
    pub a_b: f64,
    pub a_n: f64,
    pub a_t: f64,
}

pub const ALPHA_LAW: StepLawCoefficients = StepLawCoefficients {
    c_alpha: 0.8244,
    a_k: -0.4617,
    a_b: 0.9003,
    a_n: -0.4498,
    a_t: -0.1057,
};

/// Batch size selection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BatchLaw {
    #[default]
    Auto,
    Throughput,
    Quality,
    Fixed(usize),
}

/// Configuration parameters for e-means clustering.
#[derive(Debug, Clone)]
pub struct EMeansConfig {
    pub k: usize,
    pub nepochs: usize,
    pub batch_law: BatchLaw,
    pub fixed_alpha0: Option<f32>,
    pub unused_threshold: Option<f32>,
    pub seed: u64,
}

impl Default for EMeansConfig {
    fn default() -> Self {
        Self {
            k: 256,
            nepochs: 3,
            batch_law: BatchLaw::Auto,
            fixed_alpha0: None,
            unused_threshold: None,
            seed: 42,
        }
    }
}

/// Round a positive value to the nearest power of two.
pub fn round_power_of_two(value: usize) -> usize {
    if value <= 1 {
        return 1;
    }
    let val_f = value as f64;
    let low = 1usize << (val_f.log2().floor() as usize);
    let high = low * 2;
    if val_f / (low as f64) < (high as f64) / val_f {
        low
    } else {
        high
    }
}

/// Derive initial EMA step size alpha_0 from dataset parameters via power-law scaling.
pub fn resolve_alpha0(n: usize, k: usize, nepochs: usize, batch_size: usize) -> f32 {
    let alpha = ALPHA_LAW.c_alpha
        * (k as f64).powf(ALPHA_LAW.a_k)
        * (batch_size as f64).powf(ALPHA_LAW.a_b)
        * (n as f64).powf(ALPHA_LAW.a_n)
        * (nepochs as f64).powf(ALPHA_LAW.a_t);
    (alpha as f32).clamp(1e-4, 1.0)
}

/// Derive mini-batch size B from dataset parameters via power-law scaling.
pub fn resolve_batch_size(n: usize, k: usize, nepochs: usize, dim: usize, law: BatchLaw) -> usize {
    let selected_law = match law {
        BatchLaw::Fixed(b) => return b.min(n).max(1),
        BatchLaw::Throughput => BATCH_LAW_THROUGHPUT,
        BatchLaw::Quality => BATCH_LAW_QUALITY,
        BatchLaw::Auto => {
            if dim <= AUTO_QUALITY_MAX_D {
                BATCH_LAW_QUALITY
            } else {
                BATCH_LAW_THROUGHPUT
            }
        }
    };

    let raw_batch = selected_law.c_b
        * (k as f64).powf(selected_law.b_k)
        * (n as f64).powf(selected_law.b_n)
        * (nepochs as f64).powf(selected_law.b_t);

    let rounded = round_power_of_two(raw_batch.round() as usize);
    let max_batch = round_power_of_two(n / 2).max(1);
    rounded.min(max_batch).min(n).max(1)
}

/// Compute cosine-annealed EMA step size at a given global step index.
pub fn scheduled_alpha(alpha0: f32, step: usize, total_steps: usize) -> f32 {
    if total_steps == 0 {
        return alpha0;
    }
    let fraction = (step.min(total_steps) as f32) / (total_steps as f32);
    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * fraction).cos());
    alpha0 * (ALPHA_MIN_FACTOR + (1.0 - ALPHA_MIN_FACTOR) * cosine)
}

/// Check if a problem configuration satisfies the minimum eligibility criteria
/// for high-throughput e-means coarse quantization.
pub fn is_emeans_eligible(n: usize, k: usize, dim: usize) -> bool {
    n >= 10_000 && k >= 64 && (n / k) >= 30 && dim > NO_REVIVAL_MAX_D
}

/// Train centroids using hardware-accelerated e-means clustering.
///
/// Returns `(centroids, assignments)` on success.
pub fn train_emeans(
    vectors: &[f32],
    dim: usize,
    config: &EMeansConfig,
    ctx: &ComputeContext,
) -> Result<(Vec<f32>, Vec<u32>)> {
    let n = vectors.len() / dim;
    anyhow::ensure!(n > 0, "Vector buffer must not be empty");
    anyhow::ensure!(
        n >= config.k,
        "Vector count {} must exceed cluster count {}",
        n,
        config.k
    );

    let batch_size = resolve_batch_size(n, config.k, config.nepochs, dim, config.batch_law);
    let alpha0 = config
        .fixed_alpha0
        .unwrap_or_else(|| resolve_alpha0(n, config.k, config.nepochs, batch_size));

    let total_steps = (n * config.nepochs).div_ceil(batch_size);

    // Initial centroids sampled uniformly via seed
    let (mut centroids, mut usage) = initialize_centroids(vectors, dim, config.k, config.seed)?;

    // S_j accumulator: S_{j, d} = N_j * c_{j, d}
    let mut sum_accum = vec![0.0f32; config.k * dim];
    for (j, &u) in usage.iter().enumerate().take(config.k) {
        let offset = j * dim;
        for d in 0..dim {
            sum_accum[offset + d] = u * centroids[offset + d];
        }
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(config.seed);
    let mut global_step = 0usize;
    let mut batch_buf = vec![0.0f32; batch_size * dim];
    let mut indices: Vec<usize> = (0..n).collect();

    for _epoch in 0..config.nepochs {
        indices.shuffle(&mut rng);

        for batch_indices in indices.chunks(batch_size) {
            let cur_batch_len = batch_indices.len();
            if cur_batch_len == 0 {
                continue;
            }

            for (b_i, &v_idx) in batch_indices.iter().enumerate() {
                let src = &vectors[v_idx * dim..(v_idx + 1) * dim];
                batch_buf[b_i * dim..(b_i + 1) * dim].copy_from_slice(src);
            }
            let cur_batch = &batch_buf[0..cur_batch_len * dim];

            global_step += 1;
            let alpha = scheduled_alpha(alpha0, global_step, total_steps);

            let assignments = ctx
                .compute_kmeans_assignment(cur_batch, &centroids, dim)
                .unwrap_or_else(|_| {
                    crate::backend::cpu::CpuBackend::new()
                        .compute_kmeans_assignment(cur_batch, &centroids, dim)
                        .unwrap_or_default()
                });

            let mut batch_counts = vec![0.0f32; config.k];
            let mut batch_sums = vec![0.0f32; config.k * dim];

            for (b_i, &cluster_id) in assignments.iter().enumerate() {
                let c = cluster_id as usize;
                if c < config.k {
                    batch_counts[c] += 1.0;
                    let v_slice = &cur_batch[b_i * dim..(b_i + 1) * dim];
                    let offset = c * dim;
                    for d in 0..dim {
                        batch_sums[offset + d] += v_slice[d];
                    }
                }
            }

            let one_minus_alpha = 1.0 - alpha;
            for c in 0..config.k {
                let n_batch = batch_counts[c];
                let offset = c * dim;
                if n_batch > 0.0 {
                    let old_n = usage[c];
                    let new_n = (one_minus_alpha * old_n + alpha * n_batch).max(1e-20);
                    usage[c] = new_n;
                    let inv_new_n = 1.0 / new_n;
                    for d in 0..dim {
                        let old_s = sum_accum[offset + d];
                        let s_batch = batch_sums[offset + d];
                        let new_s = one_minus_alpha * old_s + alpha * s_batch;
                        sum_accum[offset + d] = new_s;
                        centroids[offset + d] = new_s * inv_new_n;
                    }
                } else {
                    usage[c] = (one_minus_alpha * usage[c]).max(1e-20);
                    for d in 0..dim {
                        sum_accum[offset + d] *= one_minus_alpha;
                    }
                }
            }

            // Dead cluster revival every CHECK_UNUSED_EVERY batches
            if dim > NO_REVIVAL_MAX_D && global_step.is_multiple_of(CHECK_UNUSED_EVERY) {
                let avg_usage = usage.iter().sum::<f32>() / (config.k as f32);
                let threshold = config.unused_threshold.unwrap_or(UNUSED_THRESHOLD) * avg_usage;

                let dead_clusters: Vec<usize> =
                    (0..config.k).filter(|&c| usage[c] < threshold).collect();

                if !dead_clusters.is_empty() {
                    revive_dead_clusters(
                        &mut centroids,
                        &mut usage,
                        &mut sum_accum,
                        &dead_clusters,
                        threshold,
                        dim,
                        config.k,
                        &mut rng,
                    );
                }
            }
        }
    }

    // Exact mean finalization pass
    let final_assignments = ctx
        .compute_kmeans_assignment(vectors, &centroids, dim)
        .unwrap_or_else(|_| {
            crate::backend::cpu::CpuBackend::new()
                .compute_kmeans_assignment(vectors, &centroids, dim)
                .unwrap_or_default()
        });

    let mut final_counts = vec![0usize; config.k];
    let mut final_sums = vec![0.0f32; config.k * dim];

    for (i, &cluster_id) in final_assignments.iter().enumerate() {
        let c = cluster_id as usize;
        if c < config.k {
            final_counts[c] += 1;
            let v_slice = &vectors[i * dim..(i + 1) * dim];
            let offset = c * dim;
            for d in 0..dim {
                final_sums[offset + d] += v_slice[d];
            }
        }
    }

    for (c, &count) in final_counts.iter().enumerate().take(config.k) {
        if count > 0 {
            let inv_count = 1.0 / (count as f32);
            let offset = c * dim;
            for d in 0..dim {
                centroids[offset + d] = final_sums[offset + d] * inv_count;
            }
        }
    }

    let final_labels = ctx
        .compute_kmeans_assignment(vectors, &centroids, dim)
        .unwrap_or_else(|_| {
            crate::backend::cpu::CpuBackend::new()
                .compute_kmeans_assignment(vectors, &centroids, dim)
                .unwrap_or_default()
        });

    Ok((centroids, final_labels))
}

#[inline(always)]
fn l2_dist_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn initialize_centroids(
    vectors: &[f32],
    dim: usize,
    k: usize,
    seed: u64,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let n = vectors.len() / dim;

    // Sub-sample candidate indices for fast D^2 (k-means++) seeding
    let sample_size = (k * 8).max(256).min(n);
    let mut sample_indices: Vec<usize> = (0..n).collect();
    sample_indices.shuffle(&mut rng);
    sample_indices.truncate(sample_size);

    let mut centroids = vec![0.0f32; k * dim];
    let first = sample_indices[rng.gen_range(0..sample_indices.len())];
    centroids[0..dim].copy_from_slice(&vectors[first * dim..(first + 1) * dim]);

    let mut min_d2: Vec<f32> = sample_indices
        .iter()
        .map(|&idx| {
            let v = &vectors[idx * dim..(idx + 1) * dim];
            l2_dist_sq(v, &centroids[0..dim])
        })
        .collect();

    for j in 1..k {
        let total: f64 = min_d2
            .iter()
            .filter(|d| d.is_finite())
            .map(|&d| d as f64)
            .sum();

        let chosen_idx = if !total.is_finite() || total <= 0.0 {
            sample_indices[rng.gen_range(0..sample_indices.len())]
        } else {
            let mut target = rng.gen::<f64>() * total;
            let mut picked = sample_indices[sample_indices.len() - 1];
            for (i, &d) in min_d2.iter().enumerate() {
                target -= d as f64;
                if target <= 0.0 {
                    picked = sample_indices[i];
                    break;
                }
            }
            picked
        };

        let new_c = &vectors[chosen_idx * dim..(chosen_idx + 1) * dim];
        centroids[j * dim..(j + 1) * dim].copy_from_slice(new_c);

        for (i, &s_idx) in sample_indices.iter().enumerate() {
            let v = &vectors[s_idx * dim..(s_idx + 1) * dim];
            let d = l2_dist_sq(v, new_c);
            if d < min_d2[i] {
                min_d2[i] = d;
            }
        }
    }

    let usage = vec![INIT_USAGE; k];
    Ok((centroids, usage))
}

fn revive_dead_clusters(
    centroids: &mut [f32],
    usage: &mut [f32],
    sum_accum: &mut [f32],
    dead_clusters: &[usize],
    threshold: f32,
    dim: usize,
    k: usize,
    rng: &mut rand::rngs::StdRng,
) {
    for &dead in dead_clusters {
        let mut candidates: Vec<usize> = (0..k).filter(|&c| usage[c] >= 2.0 * threshold).collect();
        if candidates.is_empty() {
            candidates = (0..k).filter(|&c| usage[c] > threshold).collect();
        }
        if candidates.is_empty() {
            if let Some((best_c, _)) = usage
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            {
                if usage[best_c] > 0.0 {
                    candidates.push(best_c);
                }
            }
        }
        if candidates.is_empty() {
            break;
        }

        let total_mass: f64 = candidates.iter().map(|&c| usage[c] as f64).sum();
        let donor = if total_mass > 0.0 {
            let mut target = rng.gen::<f64>() * total_mass;
            let mut picked = candidates[0];
            for &c in &candidates {
                target -= usage[c] as f64;
                if target <= 0.0 {
                    picked = c;
                    break;
                }
            }
            picked
        } else {
            candidates[rng.gen_range(0..candidates.len())]
        };

        let n_half = usage[donor] * 0.5;
        usage[donor] = n_half;
        usage[dead] = n_half;

        let donor_offset = donor * dim;
        let dead_offset = dead * dim;

        for d in 0..dim {
            let sign = if d % 2 == 0 { 1.0f32 } else { -1.0f32 };
            let c_orig = centroids[donor_offset + d];
            let c_donor = c_orig * (1.0 - SPLIT_EPSILON * sign);
            let c_dead = c_orig * (1.0 + SPLIT_EPSILON * sign);

            centroids[donor_offset + d] = c_donor;
            centroids[dead_offset + d] = c_dead;

            sum_accum[donor_offset + d] = n_half * c_donor;
            sum_accum[dead_offset + d] = n_half * c_dead;
        }
    }
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
    fn test_train_emeans_synthetic() {
        let ctx = ComputeContext::cpu();
        let k = 4;
        let dim = 8;
        let n_per_cluster = 100;
        let n = k * n_per_cluster;

        let mut vectors = Vec::with_capacity(n * dim);
        for c in 0..k {
            let offset = (c as f32) * 10.0;
            for _ in 0..n_per_cluster {
                for _ in 0..dim {
                    vectors.push(offset + 0.1);
                }
            }
        }

        let config = EMeansConfig {
            k,
            nepochs: 3,
            batch_law: BatchLaw::Fixed(64),
            fixed_alpha0: Some(0.5),
            unused_threshold: Some(0.1),
            seed: 12345,
        };

        let (centroids, assignments) =
            train_emeans(&vectors, dim, &config, &ctx).expect("train_emeans failed");

        assert_eq!(centroids.len(), k * dim);
        assert_eq!(assignments.len(), n);

        let mut unique_labels = std::collections::HashSet::new();
        for &l in &assignments {
            unique_labels.insert(l);
        }
        assert_eq!(unique_labels.len(), k);
    }
}
