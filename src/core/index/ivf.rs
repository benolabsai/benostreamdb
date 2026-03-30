// Copyright (c) 2026 Richard Albright. All rights reserved.

/// IVF (Inverted File) Index Implementation
/// 
/// This is a simplified IVF index without Product Quantization.
/// It provides memory-efficient vector search for large datasets (>100M vectors).
/// 
/// How it works:
/// 1. Cluster vectors into N lists using k-means
/// 2. Store vectors in their assigned cluster
/// 3. At search time, find nearest clusters and search only those lists
/// 
/// Memory savings: ~10x vs HNSW (no graph structure, just cluster assignments)
/// Search speed: Slower than HNSW but scales to billions of vectors
use anyhow::Result;
use std::collections::HashMap;
use rayon::prelude::*;
use super::distance::{l2_distance, l2_distance_squared};
use super::gpu::{get_global_gpu_context, compute_kmeans_assignment, ComputeContext, ComputeBackend};

/// IVF Index structure
#[allow(dead_code)]
pub struct IvfIndex {
    /// Cluster centroids (n_lists × dim)
    centroids: Vec<Vec<f32>>,
    /// Inverted lists: cluster_id -> list of (vector, row_id)
    inverted_lists: HashMap<usize, Vec<(Vec<f32>, usize)>>,
    /// Number of clusters
    n_lists: usize,
    /// Vector dimensionality
    dim: usize,
}

impl IvfIndex {
    /// Build IVF index from vectors
    pub fn build(vectors: Vec<Vec<f32>>, n_lists: Option<usize>) -> Result<Self> {
        if vectors.is_empty() {
            anyhow::bail!("Cannot build IVF index from empty vector set");
        }

        let dim = vectors[0].len();
        let n_vectors = vectors.len();
        
        // Default: sqrt(N) clusters
        let n_lists = n_lists.unwrap_or((n_vectors as f64).sqrt() as usize).max(1).min(n_vectors);
        
        println!("Building IVF index: {} vectors, {} clusters, {} dims", 
                 n_vectors, n_lists, dim);

        // Simple k-means clustering
        let (centroids, labels) = simple_kmeans(&vectors, n_lists, 10)?;

        // Assign vectors to clusters
        let mut inverted_lists: HashMap<usize, Vec<(Vec<f32>, usize)>> = HashMap::new();
        for (row_id, (vec, &cluster_id)) in vectors.into_iter().zip(labels.iter()).enumerate() {
            inverted_lists
                .entry(cluster_id)
                .or_default()
                .push((vec, row_id));
        }

        println!("IVF index built: {} non-empty clusters", inverted_lists.len());

        Ok(IvfIndex {
            centroids,
            inverted_lists,
            n_lists,
            dim,
        })
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &crate::core::index::VectorValue, k: usize, n_probe: usize, filter: Option<&roaring::RoaringBitmap>) -> Vec<(usize, f32)> {
        let query_f32 = match query {
            crate::core::index::VectorValue::Float32(v) => v,
            crate::core::index::VectorValue::Float16(v) => v,
            _ => return vec![], // For now, only f32/f16 supported in IVF
        };
        
        // 1. Find n_probe nearest clusters
        let mut cluster_distances: Vec<(usize, f32)> = self.centroids
            .iter()
            .enumerate()
            .map(|(i, centroid)| (i, l2_distance(query_f32, centroid)))
            .collect();
        cluster_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        let clusters_to_search: Vec<usize> = cluster_distances
            .iter()
            .take(n_probe)
            .map(|(i, _)| *i)
            .collect();

        // 2. Search vectors in selected clusters in parallel
        let mut candidates: Vec<(usize, f32)> = clusters_to_search
            .into_par_iter()
            .flat_map(|cluster_id| {
                if let Some(vectors) = self.inverted_lists.get(&cluster_id) {
                    vectors.iter()
                        .filter(|(_, row_id)| {
                            if let Some(f) = filter {
                                f.contains(*row_id as u32)
                            } else {
                                true
                            }
                        })
                        .map(|(vec, row_id)| {
                            let dist = l2_distance(query_f32, vec);
                            (*row_id, dist)
                        }).collect::<Vec<_>>()
                } else {
                    vec![]
                }
            })
            .collect();

        // 3. Return top-k
        candidates.par_sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(k);
        candidates
    }

    /// Serialize to bytes (for saving to file)
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        use std::io::Write;
        let mut buf = Vec::new();
        
        // Header: n_lists, dim
        buf.write_all(&(self.n_lists as u64).to_le_bytes())?;
        buf.write_all(&(self.dim as u64).to_le_bytes())?;
        
        // Centroids: n_lists × dim floats
        for centroid in &self.centroids {
            for &val in centroid {
                buf.write_all(&val.to_le_bytes())?;
            }
        }
        
        // Inverted lists: count of non-empty lists, then each list
        let non_empty_count = self.inverted_lists.len() as u64;
        buf.write_all(&non_empty_count.to_le_bytes())?;
        
        for (&cluster_id, vectors) in &self.inverted_lists {
            buf.write_all(&(cluster_id as u64).to_le_bytes())?;
            buf.write_all(&(vectors.len() as u64).to_le_bytes())?;
            for (vec, row_id) in vectors {
                buf.write_all(&(*row_id as u64).to_le_bytes())?;
                for &val in vec {
                    buf.write_all(&val.to_le_bytes())?;
                }
            }
        }
        
        Ok(buf)
    }

    /// Deserialize from bytes
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        use std::io::{Cursor, Read};
        let mut cursor = Cursor::new(bytes);
        
        let mut buf8 = [0u8; 8];
        let mut buf4 = [0u8; 4];
        
        // Header
        cursor.read_exact(&mut buf8)?;
        let n_lists = u64::from_le_bytes(buf8) as usize;
        cursor.read_exact(&mut buf8)?;
        let dim = u64::from_le_bytes(buf8) as usize;
        
        // Centroids
        let mut centroids = Vec::with_capacity(n_lists);
        for _ in 0..n_lists {
            let mut centroid = Vec::with_capacity(dim);
            for _ in 0..dim {
                cursor.read_exact(&mut buf4)?;
                centroid.push(f32::from_le_bytes(buf4));
            }
            centroids.push(centroid);
        }
        
        // Inverted lists
        cursor.read_exact(&mut buf8)?;
        let non_empty_count = u64::from_le_bytes(buf8) as usize;
        
        let mut inverted_lists = HashMap::with_capacity(non_empty_count);
        for _ in 0..non_empty_count {
            cursor.read_exact(&mut buf8)?;
            let cluster_id = u64::from_le_bytes(buf8) as usize;
            cursor.read_exact(&mut buf8)?;
            let vec_count = u64::from_le_bytes(buf8) as usize;
            
            let mut vectors = Vec::with_capacity(vec_count);
            for _ in 0..vec_count {
                cursor.read_exact(&mut buf8)?;
                let row_id = u64::from_le_bytes(buf8) as usize;
                let mut vec = Vec::with_capacity(dim);
                for _ in 0..dim {
                    cursor.read_exact(&mut buf4)?;
                    vec.push(f32::from_le_bytes(buf4));
                }
                vectors.push((vec, row_id));
            }
            inverted_lists.insert(cluster_id, vectors);
        }
        
        Ok(IvfIndex {
            centroids,
            inverted_lists,
            n_lists,
            dim,
        })
    }
}

/// Simple k-means clustering implementation
pub fn simple_kmeans(vectors: &[Vec<f32>], k: usize, max_iters: usize) -> Result<(Vec<Vec<f32>>, Vec<usize>)> {
    use rand::seq::SliceRandom;
    use rand::thread_rng;
    
    let n = vectors.len();
    let dim = vectors[0].len();
    
    // Initialize centroids randomly
    let mut rng = thread_rng();
    let mut centroids: Vec<Vec<f32>> = vectors
        .choose_multiple(&mut rng, k).cloned()
        .collect();
    
    let mut labels = vec![0; n];
    
    for iter in 0..max_iters {
        // Assign each vector to nearest centroid
        let context = get_global_gpu_context().unwrap_or_else(ComputeContext::auto_detect);
        
        let new_labels: Vec<usize> = if context.backend != ComputeBackend::Cpu {
            // Flatten vectors and centroids for GPU
            let flattened_vectors: Vec<f32> = vectors.iter().flatten().cloned().collect();
            let flattened_centroids: Vec<f32> = centroids.iter().flatten().cloned().collect();
            
            match compute_kmeans_assignment(&flattened_vectors, &flattened_centroids, dim, &context) {
                Ok(gpu_labels) => gpu_labels.into_iter().map(|l| l as usize).collect(),
                Err(e) => {
                    eprintln!("GPU K-Means assignment failed, falling back to CPU: {}", e);
                    // Fallback to CPU parallel implementation
                    vectors.par_chunks(1000)
                        .flat_map_iter(|chunk| {
                            chunk.iter().map(|vec| {
                                centroids.iter().enumerate()
                                    .map(|(i, centroid)| (i, l2_distance_squared(vec, centroid)))
                                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                                    .map(|(i, _)| i).unwrap()
                            })
                        }).collect()
                }
            }
        } else {
            // Parallel CPU implementation
            vectors.par_chunks(1000)
                .flat_map_iter(|chunk| {
                    chunk.iter().map(|vec| {
                        centroids.iter().enumerate()
                            .map(|(i, centroid)| (i, l2_distance_squared(vec, centroid)))
                            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                            .map(|(i, _)| i).unwrap()
                    })
                }).collect()
        };
        
        // Check convergence (early exit optimization)
        let changed = new_labels.iter().zip(labels.iter()).any(|(a, b)| a != b);
        if !changed {
            if iter > 0 {
                // Only log if we converged early
                println!("K-means converged early at iteration {}/{}", iter + 1, max_iters);
            }
            break;
        }
        labels = new_labels;
        
        // Update centroids
        // Update centroids in parallel
        // Partition vectors and labels or use a reduction
        // For simplicity: Parallelize the summation by chunks
        
        let (new_centroids_sum, new_counts) = vectors.par_iter().zip(labels.par_iter())
            .fold(
                || (vec![vec![0.0; dim]; k], vec![0usize; k]),
                |(mut local_centroids, mut local_counts), (vec, &cluster_id)| {
                    for (i, &val) in vec.iter().enumerate() {
                        local_centroids[cluster_id][i] += val;
                    }
                    local_counts[cluster_id] += 1;
                    (local_centroids, local_counts)
                }
            )
            .reduce(
                || (vec![vec![0.0; dim]; k], vec![0usize; k]),
                |(mut centroids_a, mut counts_a), (centroids_b, counts_b)| {
                    for (cluster_id, count) in counts_b.iter().enumerate() {
                        if *count > 0 {
                            counts_a[cluster_id] += count;
                            for (i, val) in centroids_b[cluster_id].iter().enumerate() {
                                centroids_a[cluster_id][i] += val;
                            }
                        }
                    }
                    (centroids_a, counts_a)
                }
            );

        let mut new_centroids = new_centroids_sum;
        for (cluster_id, count) in new_counts.iter().enumerate() {
             if *count > 0 {
                let div = *count as f32;
                for val in new_centroids[cluster_id].iter_mut() {
                    *val /= div;
                }
             } else {
                 // Empty cluster: Keep old centroid or re-initialize?
                 // Simple strategy: Keep old centroid
                 new_centroids[cluster_id] = centroids[cluster_id].clone();
             }
        }
        
        centroids = new_centroids;
    }
    
    Ok((centroids, labels))
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivf_basic() {
        // Create simple test vectors
        let vectors = vec![
            vec![1.0, 0.0],
            vec![1.1, 0.1],
            vec![0.0, 1.0],
            vec![0.1, 1.1],
        ];

        let index = IvfIndex::build(vectors, Some(2)).unwrap();
        
        // Search for vector similar to [1.0, 0.0]
        let query = crate::core::index::VectorValue::Float32(vec![1.0, 0.0]);
        let results = index.search(&query, 2, 1, None);
        
        assert_eq!(results.len(), 2);
        // Should find row 0 or 1 (closest to [1.0, 0.0])
        assert!(results[0].0 <= 1);
    }
}
