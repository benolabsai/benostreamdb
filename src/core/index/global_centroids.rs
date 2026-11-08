// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

//! Table-level (global) centroid training.
//!
//! By default each segment trains its own k-means over its own rows
//! ([`super::hnsw_ivf::HnswIvfIndex::build_with_seed`], `n_lists = sqrt(n)/5`).
//! That makes cluster ids segment-local: a query's true nearest neighbours are
//! spread across every segment's independent clustering, so covering them needs
//! `segments × n_probe` probes.
//!
//! Training one centroid set over a sample spanning all segments makes a cluster
//! id mean the same thing everywhere. Combined with
//! [`super::hnsw_ivf::HnswIvfIndex::build_with_centroids`], every segment then
//! assigns against the same centroids.
//!
//! Note that this alone does **not** reduce probe cost: a global cluster is still
//! split across segments, so a probe still touches many files. It is the
//! prerequisite for a cluster-keyed artifact, and it is worth measuring on its
//! own precisely because the expected result is "no improvement".

use anyhow::Result;

use super::ivf::simple_kmeans_with_seed;

/// Default number of global centroids.
///
/// Deliberately smaller than the ~3,150 centroids the per-segment rule produces
/// for a 10M-row table: fewer, larger clusters cost less to train and the probe
/// count needed for a given recall is what matters, not the cluster count.
pub const DEFAULT_GLOBAL_CLUSTERS: usize = 1024;

/// Default Lloyd iterations for global training.
pub const DEFAULT_GLOBAL_ITERS: usize = 10;

/// Train a global centroid set over a sample of vectors.
///
/// `samples` should span all segments (see [`sample_vectors`]); training on one
/// segment's rows reproduces the per-segment behaviour this exists to replace.
pub fn train_global(samples: &[Vec<f32>], k: usize, seed: Option<u64>) -> Result<Vec<Vec<f32>>> {
    if samples.is_empty() {
        anyhow::bail!("train_global requires at least one sample vector");
    }
    if k == 0 {
        anyhow::bail!("train_global requires k >= 1");
    }
    let dim = samples[0].len();
    if let Some(bad) = samples.iter().find(|v| v.len() != dim) {
        anyhow::bail!(
            "sample dimension mismatch: expected {}-d, found {}-d",
            dim,
            bad.len()
        );
    }

    // k-means cannot produce more clusters than points; clamp rather than fail so
    // a small table still gets a usable centroid set.
    let k = k.min(samples.len()).max(1);
    let (centroids, _labels) = simple_kmeans_with_seed(samples, k, DEFAULT_GLOBAL_ITERS, seed)?;
    Ok(centroids)
}

/// Reduce an existing centroid set to `k` centroids by clustering the centroids
/// themselves.
///
/// This is the warm start: concatenating the per-segment centroids gives a set
/// that already covers the data distribution, so clustering it down to `k`
/// converges in far fewer Lloyd iterations than random initialisation and is
/// nearly free relative to a full pass over the corpus.
pub fn reduce_centroids(
    centroids: &[Vec<f32>],
    k: usize,
    seed: Option<u64>,
) -> Result<Vec<Vec<f32>>> {
    if centroids.is_empty() {
        anyhow::bail!("reduce_centroids requires at least one centroid");
    }
    if k == 0 {
        anyhow::bail!("reduce_centroids requires k >= 1");
    }
    if centroids.len() <= k {
        return Ok(centroids.to_vec());
    }
    let (reduced, _labels) = simple_kmeans_with_seed(centroids, k, DEFAULT_GLOBAL_ITERS, seed)?;
    Ok(reduced)
}

/// Deterministically subsample `vectors` down to at most `max_samples`.
///
/// Uses a fixed-stride walk rather than a PRNG so the same input always yields
/// the same sample, which keeps index builds reproducible. A stride walk also
/// spreads the sample across the whole input, which matters when the input is
/// concatenated segment-by-segment: a prefix sample would be one segment.
pub fn sample_vectors(vectors: &[Vec<f32>], max_samples: usize) -> Vec<Vec<f32>> {
    if max_samples == 0 || vectors.is_empty() {
        return Vec::new();
    }
    if vectors.len() <= max_samples {
        return vectors.to_vec();
    }
    let stride = vectors.len() as f64 / max_samples as f64;
    (0..max_samples)
        .map(|i| vectors[(i as f64 * stride) as usize].clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob(n: usize, dim: usize, offset: f32) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| offset + (i as f32) * 0.01 + (d as f32) * 0.001)
                    .collect()
            })
            .collect()
    }

    #[test]
    fn train_global_returns_requested_cluster_count() {
        let samples = blob(500, 8, 0.0);
        let centroids = train_global(&samples, 16, Some(42)).unwrap();
        assert_eq!(centroids.len(), 16);
        assert!(centroids.iter().all(|c| c.len() == 8));
    }

    #[test]
    fn train_global_clamps_k_to_sample_count() {
        let samples = blob(5, 4, 0.0);
        let centroids = train_global(&samples, 100, Some(1)).unwrap();
        assert_eq!(centroids.len(), 5);
    }

    #[test]
    fn train_global_rejects_empty_and_zero_k() {
        assert!(train_global(&[], 4, None).is_err());
        assert!(train_global(&blob(10, 4, 0.0), 0, None).is_err());
    }

    #[test]
    fn train_global_rejects_dimension_mismatch() {
        let mut samples = blob(10, 4, 0.0);
        samples.push(vec![0.0; 5]);
        assert!(train_global(&samples, 2, None).is_err());
    }

    #[test]
    fn reduce_centroids_is_identity_when_already_small() {
        let centroids = blob(4, 4, 0.0);
        let reduced = reduce_centroids(&centroids, 8, None).unwrap();
        assert_eq!(reduced.len(), 4);
    }

    #[test]
    fn reduce_centroids_shrinks_to_k() {
        let centroids = blob(200, 6, 0.0);
        let reduced = reduce_centroids(&centroids, 10, Some(7)).unwrap();
        assert_eq!(reduced.len(), 10);
    }

    #[test]
    fn sample_vectors_is_deterministic_and_spans_input() {
        let vectors = blob(1000, 3, 0.0);
        let a = sample_vectors(&vectors, 100);
        let b = sample_vectors(&vectors, 100);
        assert_eq!(a.len(), 100);
        assert_eq!(a, b, "sampling must be deterministic");

        // A stride walk must reach the tail; a prefix sample would not.
        let first = a.first().unwrap()[0];
        let last = a.last().unwrap()[0];
        assert!(last > first + 5.0, "sample should span the input range");
    }

    #[test]
    fn sample_vectors_returns_all_when_under_budget() {
        let vectors = blob(10, 3, 0.0);
        assert_eq!(sample_vectors(&vectors, 100).len(), 10);
        assert!(sample_vectors(&vectors, 0).is_empty());
    }
}
