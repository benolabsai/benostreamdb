// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

use crate::backend::{ComputeContext, GpuBackend};
use anyhow::Result;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};

/// Compute L2 squared distance between two vectors.
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

/// Train k-means centroids with hardware acceleration (CUDA/Metal/WGPU/CPU).
pub fn train_kmeans(
    vectors: &[f32],
    dim: usize,
    k: usize,
    max_iters: usize,
    ctx: &ComputeContext,
) -> Result<Vec<f32>> {
    let n = vectors.len() / dim;
    if n == 0 {
        anyhow::bail!("Cannot cluster empty vector set");
    }
    if k == 0 {
        anyhow::bail!("k must be > 0");
    }
    let k = k.min(n);

    let sample_size = (n / 10).max(1000).min(n);
    let mut rng = thread_rng();
    let training_indices: Vec<usize> = (0..n)
        .collect::<Vec<_>>()
        .choose_multiple(&mut rng, sample_size)
        .cloned()
        .collect();

    let flat_training_set: Vec<f32> = training_indices
        .iter()
        .flat_map(|&idx| &vectors[idx * dim..(idx + 1) * dim])
        .cloned()
        .collect();

    // k-means++ seeding
    let mut centroids_flat: Vec<f32> = Vec::with_capacity(k * dim);
    {
        let first_idx = training_indices[rng.gen_range(0..training_indices.len())];
        centroids_flat.extend_from_slice(&vectors[first_idx * dim..(first_idx + 1) * dim]);

        let mut min_d2: Vec<f32> = training_indices
            .iter()
            .map(|&idx| {
                l2_dist_sq(
                    &vectors[idx * dim..(idx + 1) * dim],
                    &centroids_flat[0..dim],
                )
            })
            .collect();

        while centroids_flat.len() / dim < k {
            let total: f64 = min_d2
                .iter()
                .filter(|d| d.is_finite())
                .map(|&d| d as f64)
                .sum();
            let next_idx = if !total.is_finite() || total <= 0.0 {
                training_indices[rng.gen_range(0..training_indices.len())]
            } else {
                let threshold = rng.gen_range(0.0..total);
                let mut acc = 0.0;
                let mut picked = training_indices[0];
                for (d, &orig_idx) in min_d2.iter().zip(training_indices.iter()) {
                    acc += *d as f64;
                    if acc >= threshold {
                        picked = orig_idx;
                        break;
                    }
                }
                picked
            };

            let next_centroid = &vectors[next_idx * dim..(next_idx + 1) * dim];
            centroids_flat.extend_from_slice(next_centroid);

            for (d, &orig_idx) in min_d2.iter_mut().zip(training_indices.iter()) {
                let d2 = l2_dist_sq(
                    &vectors[orig_idx * dim..(orig_idx + 1) * dim],
                    next_centroid,
                );
                if d2 < *d {
                    *d = d2;
                }
            }
        }
    }

    // Iterative centroid refinement (assignment dispatched to GPU)
    for _iter in 0..max_iters {
        let assignments = ctx
            .compute_kmeans_assignment(&flat_training_set, &centroids_flat, dim)
            .unwrap_or_else(|_| {
                crate::backend::cpu::CpuBackend::new()
                    .compute_kmeans_assignment(&flat_training_set, &centroids_flat, dim)
                    .unwrap_or_default()
            });

        let mut new_centroids = vec![0.0f32; k * dim];
        let mut counts = vec![0usize; k];

        for (i, &cluster_id) in assignments.iter().enumerate() {
            let c = cluster_id as usize;
            if c < k {
                counts[c] += 1;
                let v = &flat_training_set[i * dim..(i + 1) * dim];
                for (sum, &val) in new_centroids[c * dim..(c + 1) * dim]
                    .iter_mut()
                    .zip(v.iter())
                {
                    *sum += val;
                }
            }
        }

        let mut changed = false;
        for c in 0..k {
            let count = counts[c];
            if count > 0 {
                let inv = 1.0 / count as f32;
                for (target, orig) in new_centroids[c * dim..(c + 1) * dim]
                    .iter_mut()
                    .zip(&centroids_flat[c * dim..(c + 1) * dim])
                {
                    *target *= inv;
                    if (*target - *orig).abs() > 1e-4 {
                        changed = true;
                    }
                }
            } else {
                // Keep previous centroid if no vectors assigned
                new_centroids[c * dim..(c + 1) * dim]
                    .copy_from_slice(&centroids_flat[c * dim..(c + 1) * dim]);
            }
        }

        centroids_flat = new_centroids;
        if !changed {
            break;
        }
    }

    Ok(centroids_flat)
}
