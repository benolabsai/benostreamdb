// Copyright (c) 2026 Richard Albright. All rights reserved.
// Modified by Richard Albright / HyperStreamDB on 2026-03-29 to add pre-filtering support and better integration with Iceberg manifests. 
// This file contains derivative work from the Apache 2.0 licensed project(s).

/// HNSW-IVF Hybrid Index Implementation
/// 
/// Combines IVF coarse quantization with HNSW fine search for optimal performance.
/// This is the industry-standard approach used by FAISS, Milvus, and Weaviate.
/// 
/// Architecture:
/// 1. Coarse quantization: Cluster vectors into N lists using k-means (IVF)
/// 2. Fine search: Build small HNSW graphs within each cluster
/// 3. Search: Find nearest clusters, then search HNSW graphs in those clusters
///
/// Attribution: Underlying HNSW graph logic relies on the vendored `hnsw_rs` library (MIT/Apache 2.0).
/// Copyright Jean-Pierre Both and hnsw_rs contributors. Vendored and patched to support exact pre-filtering.
use anyhow::{Result, Context};
use std::collections::HashMap;
use std::sync::Arc;
use tracing;

use crate::core::index::hnsw_rs::prelude::*;
use super::ivf::simple_kmeans;
use super::pq::{PqEncoder, PqConfig};
use super::{VectorMetric, VectorValue};
use rayon::prelude::*;
use arrow::record_batch::RecordBatch;
use arrow::array::{Array, AsArray};
use object_store::{path::Path, ObjectStore};
use futures::{StreamExt, TryStreamExt};
use std::io::Cursor;
use crate::core::puffin::PuffinReader;
use parquet::file::reader::FileReader;

#[derive(Clone, Copy, Debug, Default)]
pub struct DistL1;
impl Distance<f32> for DistL1 {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        crate::core::index::distance::l1_distance(va, vb)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DistHamming;
impl Distance<f32> for DistHamming {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        crate::core::index::distance::hamming_distance(va, vb)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DistJaccard;
impl Distance<f32> for DistJaccard {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        crate::core::index::distance::jaccard_distance(va, vb)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DistHammingPacked;
impl Distance<u8> for DistHammingPacked {
    fn eval(&self, va: &[u8], vb: &[u8]) -> f32 {
        crate::core::index::distance::hamming_distance_packed(va, vb) as f32
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct DistSparseDot;
impl Distance<crate::core::index::SparseVector> for DistSparseDot {
    fn eval(&self, va: &[crate::core::index::SparseVector], vb: &[crate::core::index::SparseVector]) -> f32 {
        // HNSW-rs uses &[T]. For sparse vectors, T is SparseVector, so we compare the first elements.
        1.0 - crate::core::index::distance::sparse_dot_product(&va[0].indices, &va[0].values, &vb[0].indices, &vb[0].values)
    }
}

/// Internal wrapper for different HNSW distance metrics
pub enum HnswGraph {
    L2(Hnsw<f32, DistL2>),
    Cosine(Hnsw<f32, DistCosine>),
    Dot(Hnsw<f32, DistDot>),
    L1(Hnsw<f32, DistL1>),
    Hamming(Hnsw<f32, DistHamming>),
    Jaccard(Hnsw<f32, DistJaccard>),
    BinaryHamming(Hnsw<u8, DistHammingPacked>),
    SparseDot(Hnsw<crate::core::index::SparseVector, DistSparseDot>),
}

impl HnswGraph {
    pub fn search(&self, query: &VectorValue, k: usize, ef: usize, filter: Option<&roaring::RoaringBitmap>) -> Vec<Neighbour> {
        match (self, query) {
            (HnswGraph::L2(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            (HnswGraph::Cosine(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            (HnswGraph::Dot(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            (HnswGraph::L1(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            (HnswGraph::Hamming(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            (HnswGraph::Jaccard(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            
            (HnswGraph::BinaryHamming(h), VectorValue::Binary(q)) => h.search(q, k, ef, filter),
            (HnswGraph::SparseDot(h), VectorValue::Sparse(q)) => h.search(std::slice::from_ref(q), k, ef, filter),
            
            // Casting paths
            (HnswGraph::L2(h), VectorValue::Float16(q)) => h.search(q, k, ef, filter),
            
            _ => {
                tracing::error!("HnswGraph / query type mismatch");
                vec![]
            }
        }
    }

    pub fn insert(&self, data: (VectorValue, usize)) {
        let (val, local_id) = data;
        match (self, val) {
            (HnswGraph::L2(h), VectorValue::Float32(v)) => h.insert_slice((&v, local_id)),
            (HnswGraph::Cosine(h), VectorValue::Float32(v)) => h.insert_slice((&v, local_id)),
            (HnswGraph::Dot(h), VectorValue::Float32(v)) => h.insert_slice((&v, local_id)),
            (HnswGraph::L1(h), VectorValue::Float32(v)) => h.insert_slice((&v, local_id)),
            (HnswGraph::Hamming(h), VectorValue::Float32(v)) => h.insert_slice((&v, local_id)),
            (HnswGraph::Jaccard(h), VectorValue::Float32(v)) => h.insert_slice((&v, local_id)),
            
            (HnswGraph::BinaryHamming(h), VectorValue::Binary(v)) => h.insert_slice((&v, local_id)),
            (HnswGraph::SparseDot(h), VectorValue::Sparse(v)) => h.insert_slice((&vec![v], local_id)),
            
            (HnswGraph::L2(h), VectorValue::Float16(v)) => h.insert_slice((&v, local_id)),
            
            _ => tracing::error!("HnswGraph / insert value type mismatch"),
        }
    }

    pub fn parallel_insert(&self, data: Vec<(&[f32], usize)>) {
        if data.is_empty() { return; }
        
        match self {
            HnswGraph::L2(h) => h.parallel_insert_slice(&data),
            HnswGraph::Cosine(h) => h.parallel_insert_slice(&data),
            HnswGraph::Dot(h) => h.parallel_insert_slice(&data),
            HnswGraph::L1(h) => h.parallel_insert_slice(&data),
            _ => {
                // Fallback to sequential for other types
                for (v, id) in data {

                    let val = VectorValue::Float32(v.to_vec());
                    self.insert((val, id));
                }
            }
        }
    }

    pub fn file_dump(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let path_string = path.to_string();
        match self {
            HnswGraph::L2(h) => h.file_dump(&path_string).map(|_| ()).map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::Cosine(h) => h.file_dump(&path_string).map(|_| ()).map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::Dot(h) => h.file_dump(&path_string).map(|_| ()).map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::L1(h) => h.file_dump(&path_string).map(|_| ()).map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::Hamming(h) => h.file_dump(&path_string).map(|_| ()).map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::Jaccard(h) => h.file_dump(&path_string).map(|_| ()).map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::BinaryHamming(h) => h.file_dump(&path_string).map(|_| ()).map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::SparseDot(h) => h.file_dump(&path_string).map(|_| ()).map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
        }
    }
}


/// HNSW-IVF Hybrid Index
pub struct HnswIvfIndex {
    /// Cluster centroids for coarse quantization
    centroids: Vec<Vec<f32>>,
    /// Distance metric used for search
    metric: VectorMetric,
    /// Per-cluster HNSW graphs: cluster_id -> (hnsw_index, row_id_mapping)
    cluster_graphs: HashMap<usize, (HnswGraph, Vec<usize>)>,
    /// Number of clusters (for metadata/scaling)
    _n_lists: usize,
    /// Vector dimensionality
    dim: usize,
    /// Optional PQ encoder for compression (Enterprise feature)
    _pq_encoder: Option<PqEncoder>,
    /// Hardware acceleration context (Open Source)
    _compute_context: crate::core::index::gpu::ComputeContext,
}

impl HnswIvfIndex {
    /// Build HNSW-IVF index from vectors
    pub fn build(
        vectors: Vec<Vec<f32>>,
        metric: VectorMetric,
        n_lists: Option<usize>,
        hnsw_m: Option<usize>,
        use_pq: bool,
    ) -> Result<Self> {
        if vectors.is_empty() {
            anyhow::bail!("Cannot build HNSW-IVF index from empty vector set");
        }

        let dim = vectors[0].len();
        let n_vectors = vectors.len();
        
        // Auto-detect optimal cluster count based on dataset size and CPU cores
        let num_cpus = num_cpus::get();
        let n_lists = n_lists.unwrap_or_else(|| {
            if n_vectors < 1000 {
                num_cpus.min(4) // Very small
            } else if n_vectors < 10_000 {
                num_cpus.max(16) // Small: match core count
            } else if n_vectors < 100_000 {
                (num_cpus * 2).max((n_vectors as f64).sqrt() as usize / 10).max(32)
            } else {
                ((n_vectors as f64).sqrt() / 5.0) as usize
            }
        }).max(1).min(n_vectors / 10).max(1);
        
        let hnsw_m = hnsw_m.unwrap_or(16);
        
        tracing::info!("Building HNSW-IVF index: {} vectors, {} clusters, {} dims, M={}, use_pq={}", 
                 n_vectors, n_lists, dim, hnsw_m, use_pq);

        let start = std::time::Instant::now();

        // Step 1: Cluster vectors using k-means (IVF)
        let t0 = std::time::Instant::now();
        let max_iters = if n_vectors < 100_000 {
            5  // Small: 5 iterations sufficient
        } else if n_vectors < 1_000_000 {
            8  // Medium: 8 iterations for better quality
        } else {
            10 // Large: 10 iterations for very large datasets
        };
        
        let (centroids, labels) = simple_kmeans(&vectors, n_lists, max_iters)?;

        
        // Train PQ encoder if requested
        let pq_encoder = if use_pq {
            // Use 8-bit quantization with m sub-vectors (e.g., 16 or 32)
            let m = if dim >= 32 && dim.is_multiple_of(16) { 16 } else if dim >= 8 && dim.is_multiple_of(8) { 8 } else { 1 };
            tracing::debug!("  - Training PQ encoder with m={} subspaces...", m);
            let config = PqConfig { m, k: 256, dim };
            Some(PqEncoder::train(&vectors, config)?)
        } else {
            None
        };

        tracing::debug!("  - K-Means took: {:.2?} ({} iterations, {} clusters)", t0.elapsed(), max_iters, n_lists);

        // Step 2: Group vectors by cluster in parallel
        let t1 = std::time::Instant::now();
        let cluster_vectors: HashMap<usize, Vec<(Vec<f32>, usize)>> = vectors
            .into_par_iter()
            .zip(labels.into_par_iter())
            .enumerate()
            .fold(
                || HashMap::<usize, Vec<(Vec<f32>, usize)>>::new(),
                |mut acc, (row_id, (vec, cluster_id))| {
                    acc.entry(cluster_id as usize).or_default().push((vec, row_id));
                    acc
                }
            )
            .reduce(
                || HashMap::new(),
                |mut a, b| {
                    for (k, v) in b {
                        a.entry(k).or_default().extend(v);
                    }
                    a
                }
            );
        tracing::debug!("  - Grouping vectors took: {:.2?}", t1.elapsed());

        // Step 3: Build HNSW graph for each cluster in parallel
        let t2 = std::time::Instant::now();
        let cluster_graphs: HashMap<usize, (HnswGraph, Vec<usize>)> = cluster_vectors
            .into_par_iter()
            .map(|(cluster_id, vecs): (usize, Vec<(Vec<f32>, usize)>)| {
                let max_layers = 16;
                let ef_construction = (hnsw_m * 2).max(40);
                
                let hnsw = match metric {
                    VectorMetric::L2 => HnswGraph::L2(Hnsw::new(hnsw_m, vecs.len(), max_layers, ef_construction, DistL2)),
                    VectorMetric::Cosine => HnswGraph::Cosine(Hnsw::new(hnsw_m, vecs.len(), max_layers, ef_construction, DistCosine)),
                    VectorMetric::InnerProduct => HnswGraph::Dot(Hnsw::new(hnsw_m, vecs.len(), max_layers, ef_construction, DistDot)),
                    VectorMetric::L1 => HnswGraph::L1(Hnsw::new(hnsw_m, vecs.len(), max_layers, ef_construction, DistL1)),
                    VectorMetric::Hamming => HnswGraph::Hamming(Hnsw::new(hnsw_m, vecs.len(), max_layers, ef_construction, DistHamming)),
                    VectorMetric::Jaccard => HnswGraph::Jaccard(Hnsw::new(hnsw_m, vecs.len(), max_layers, ef_construction, DistJaccard)),
                };
                
                // Build locally sequential to avoid Rayon nested parallelism overhead
                // Since Step 3 is already in parallel (one cluster per core), 
                // this saturates the CPU optimally.
                for (local_id, (vec, _)) in vecs.iter().enumerate() {
                    hnsw.insert((crate::core::index::VectorValue::Float32(vec.clone()), local_id));
                }
                
                let row_id_mapping: Vec<usize> = vecs.iter().map(|(_, row_id)| *row_id).collect();
                (cluster_id, (hnsw, row_id_mapping))
            })
            .collect();
        tracing::debug!("  - Building HNSW graphs took: {:.2?} ({} clusters)", t2.elapsed(), cluster_graphs.len());

        tracing::info!("Hnsw-IVF index built in {:.2?}: {} non-empty clusters", start.elapsed(), cluster_graphs.len());

        Ok(HnswIvfIndex {
            centroids,
            metric,
            cluster_graphs,
            _n_lists: n_lists,
            dim,
            _pq_encoder: pq_encoder,
            _compute_context: crate::core::index::gpu::ComputeContext::auto_detect(),
        })
    }

    /// Search for k nearest neighbors
    pub fn search(&self, query: &VectorValue, k: usize, n_probe: usize, filter: Option<&roaring::RoaringBitmap>) -> Vec<(usize, f32)> {
        let query_f32 = match query {
            VectorValue::Float32(v) => v,
            VectorValue::Float16(v) => v,
            VectorValue::Binary(_) => return vec![], // TODO: implement proper coarse search for non-float types
            VectorValue::Sparse(_) => return vec![], // TODO: implement
        };

        // Step 1: Find n_probe nearest clusters (coarse search)
        let all_centroids_flat: Vec<f32> = self.centroids.iter().flatten().cloned().collect();

        let distances = crate::core::index::gpu::compute_distance(query_f32, &all_centroids_flat, self.dim, self.metric)
            .unwrap_or_else(|_| {
                match self.metric {
                    VectorMetric::L2 => crate::core::index::distance::l2_distance_batch(query_f32, &self.centroids),
                    VectorMetric::Cosine => crate::core::index::distance::cosine_similarity_batch(query_f32, &self.centroids),
                    VectorMetric::InnerProduct => crate::core::index::distance::dot_product_batch(query_f32, &self.centroids),
                    VectorMetric::L1 => self.centroids.par_iter().map(|c| crate::core::index::distance::l1_distance(query_f32, c)).collect(),
                    VectorMetric::Hamming => self.centroids.par_iter().map(|c| crate::core::index::distance::hamming_distance(query_f32, c)).collect(),
                    VectorMetric::Jaccard => self.centroids.par_iter().map(|c| crate::core::index::distance::jaccard_distance(query_f32, c)).collect(),
                }
            });

        let mut cluster_distances: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();
        cluster_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        
        let clusters_to_search: Vec<usize> = cluster_distances.iter().take(n_probe).map(|(i, _)| *i).collect();

        // Step 2: Search HNSW graphs in selected clusters in parallel (fine search)
        let mut candidates: Vec<(usize, f32)> = clusters_to_search
            .into_par_iter()
            .flat_map(|cluster_id| {
                if let Some((hnsw, row_id_mapping)) = self.cluster_graphs.get(&cluster_id) {
                    let cluster_k = if filter.is_some() { k * 4 } else { k * 2 };
                    let ef_search = if filter.is_some() { 100 } else { 40 };
                    
                    let mut local_filter_opt: Option<roaring::RoaringBitmap> = None;
                    if let Some(f) = filter {
                        let mut bm = roaring::RoaringBitmap::new();
                        for (local_id, &global_id) in row_id_mapping.iter().enumerate() {
                            if f.contains(global_id as u32) {
                                bm.insert(local_id as u32);
                            }
                        }
                        local_filter_opt = Some(bm);
                    }
                    
                    let results = hnsw.search(query, cluster_k, ef_search, local_filter_opt.as_ref());
                    
                    // Map local indices back to global row IDs and apply filter
                    results.into_iter().filter_map(|neighbor| {
                        let local_id = neighbor.d_id;
                        if local_id < row_id_mapping.len() {
                            let global_id = row_id_mapping[local_id];
                            if let Some(f) = filter {
                                if f.contains(global_id as u32) {
                                    Some((global_id, neighbor.distance))
                                } else {

                                    None
                                }
                            } else {
                                Some((global_id, neighbor.distance))
                            }
                        } else {
                            None
                        }
                    }).collect::<Vec<_>>()
                } else {
                    vec![]
                }
            })
            .collect();

        // Step 3: Merge and return top-k
        candidates.par_sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.dedup_by_key(|x| x.0);  // Remove duplicates
        candidates.truncate(k);
        candidates
    }

    /// Save HNSW-IVF index to disk
    pub fn save(&self, local_path: &str) -> Result<Vec<String>> {
        use arrow::array::{UInt32Array, ListBuilder, Float32Builder};
        use arrow::datatypes::{Schema, Field, DataType};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::fs::File;
        use std::sync::Arc;

        let mut saved_files = Vec::new();

        // 1. Save centroids
        let centroids_path = format!("{}.centroids.parquet", local_path);
        let centroids_tmp = format!("{}.tmp", centroids_path);
        {
            let mut centroid_builder = ListBuilder::new(Float32Builder::new());
            for centroid in &self.centroids {
                for &val in centroid {
                    centroid_builder.values().append_value(val);
                }
                centroid_builder.append(true);
            }
            let centroid_array = Arc::new(centroid_builder.finish());

            let schema = Arc::new(Schema::new(vec![
                Field::new("centroid", DataType::List(
                    Arc::new(Field::new("item", DataType::Float32, true))
                ), false),
            ]));

            let batch = RecordBatch::try_new(schema.clone(), vec![centroid_array])?;
            
            let file = File::create(&centroids_tmp)?;
            let props = WriterProperties::builder()
                .set_key_value_metadata(Some(vec![
                    parquet::file::metadata::KeyValue {
                        key: "vector_metric".to_string(),
                        value: Some(match self.metric {
                            VectorMetric::L2 => "l2".to_string(),
                            VectorMetric::Cosine => "cosine".to_string(),
                            VectorMetric::InnerProduct => "ip".to_string(),
                            VectorMetric::L1 => "l1".to_string(),
                            VectorMetric::Hamming => "hamming".to_string(),
                            VectorMetric::Jaccard => "jaccard".to_string(),
                        }),
                    }
                ]))
                .build();
            let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
            writer.write(&batch)?;
            writer.close()?;
            
            std::fs::rename(&centroids_tmp, &centroids_path)?;
            saved_files.push(centroids_path.clone());
        }

        // 2. Save per-cluster HNSW graphs
        for (cluster_id, (hnsw, row_id_mapping)) in &self.cluster_graphs {
            let cluster_base = format!("{}.cluster_{}", local_path, cluster_id);
            let cluster_tmp = format!("{}.tmp", cluster_base);
            
            hnsw.file_dump(&cluster_tmp).map_err(|e| anyhow::anyhow!("HNSW dump failed: {}", e))?;
            
            let graph_orig = format!("{}.hnsw.graph", cluster_tmp);
            let data_orig = format!("{}.hnsw.data", cluster_tmp);
            let graph_dest = format!("{}.hnsw.graph", cluster_base);
            let data_dest = format!("{}.hnsw.data", cluster_base);
            
            if !std::path::Path::new(&graph_orig).exists() {
                 anyhow::bail!("HNSW dump did not create expected file: {}", graph_orig);
            }
            if !std::path::Path::new(&data_orig).exists() {
                 anyhow::bail!("HNSW dump did not create expected file: {}", data_orig);
            }
            
            std::fs::rename(&graph_orig, &graph_dest)?;
            std::fs::rename(&data_orig, &data_dest)?;

            saved_files.push(graph_dest);
            saved_files.push(data_dest);
            
            let mapping_path = format!("{}.cluster_{}.mapping.parquet", local_path, cluster_id);
            let mapping_tmp = format!("{}.tmp", mapping_path);
            
            {
                let row_ids: Vec<u32> = row_id_mapping.iter().map(|&x| x as u32).collect();
                let row_id_array = Arc::new(UInt32Array::from(row_ids));

                let schema = Arc::new(Schema::new(vec![
                    Field::new("row_id", DataType::UInt32, false),
                ]));

                let batch = RecordBatch::try_new(schema.clone(), vec![row_id_array])?;
                
                let file = File::create(&mapping_tmp)?;
                let props = WriterProperties::builder().build();
                let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
                writer.write(&batch)?;
                writer.close()?;
                
                std::fs::rename(&mapping_tmp, &mapping_path)?;
                saved_files.push(mapping_path);
            }
        }

        Ok(saved_files)
    }

    pub async fn load_puffin_async(store: Arc<dyn ObjectStore>, path: &str) -> Result<Arc<Self>> {
        let path_obj = Path::from(path);
        let bytes = store.get(&path_obj).await?.bytes().await?;
        let mut reader = PuffinReader::new(Cursor::new(bytes))?;
        
        let mut centroids = Vec::new();
        let mut cluster_graphs = HashMap::new();
        let mut dim = 0;
        let mut metric = VectorMetric::L2;
        
        // Temporary storage for building clusters
        let mut graphs_data: HashMap<usize, Vec<u8>> = HashMap::new();
        let mut graphs_graph: HashMap<usize, Vec<u8>> = HashMap::new();
        let mut graphs_mapping: HashMap<usize, Vec<u8>> = HashMap::new();

        let blobs = reader.footer().blobs.clone();
        for (i, blob) in blobs.iter().enumerate() {
            match blob.r#type.as_str() {
                "hnsw-ivf-centroids" => {
                    let data = reader.read_blob(i)?;
                    // Decode centroids from parquet or raw? For now let's assume it's the centroids parquet bytes
                    // BUT: We could just store them as raw F32 blobs for speed.
                    // Let's stick to the current Parquet centroids for now to reuse logic.
                    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
                    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(data))?;
                    let reader = builder.build()?;
                    let batches: Vec<RecordBatch> = reader.map(|r| r.map_err(anyhow::Error::from)).collect::<Result<Vec<_>>>()?;
                    let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches)?;
                    let centroid_list = batch.column(0).as_list::<i32>();
                    for j in 0..centroid_list.len() {
                        let values = centroid_list.value(j);
                        let float_array = values.as_primitive::<arrow::datatypes::Float32Type>();
                        centroids.push(float_array.values().to_vec());
                    }
                    if !centroids.is_empty() {
                        dim = centroids[0].len();
                    }
                    
                    if let Some(m_str) = blob.properties.get("vector-metric") {
                        metric = match m_str.as_str() {
                            "l2" => VectorMetric::L2,
                            "cosine" => VectorMetric::Cosine,
                            "ip" => VectorMetric::InnerProduct,
                            _ => VectorMetric::L2,
                        };
                    }
                },
                "hnsw-cluster-graph" => {
                    if let Some(cid_str) = blob.properties.get("cluster-id") {
                        if let Ok(cid) = cid_str.parse::<usize>() {
                            graphs_graph.insert(cid, reader.read_blob(i)?);
                        }
                    }
                },
                "hnsw-cluster-data" => {
                    if let Some(cid_str) = blob.properties.get("cluster-id") {
                        if let Ok(cid) = cid_str.parse::<usize>() {
                            graphs_data.insert(cid, reader.read_blob(i)?);
                        }
                    }
                },
                "hnsw-cluster-mapping" => {
                    if let Some(cid_str) = blob.properties.get("cluster-id") {
                        if let Ok(cid) = cid_str.parse::<usize>() {
                            graphs_mapping.insert(cid, reader.read_blob(i)?);
                        }
                    }
                },
                _ => {}
            }
        }
        
        // Assemble clusters
        for (cid, graph_bytes) in graphs_graph {
            let data_bytes = graphs_data.remove(&cid).context("Missing cluster data")?;
            let mapping_bytes = graphs_mapping.remove(&cid).context("Missing cluster mapping")?;
            
            let mut graph_reader = std::io::BufReader::new(Cursor::new(graph_bytes));
            let mut data_reader = std::io::BufReader::new(Cursor::new(data_bytes));
            
            let description = crate::core::index::hnsw_rs::hnswio::load_description(&mut graph_reader)
                .map_err(|e| anyhow::anyhow!("Failed to load HNSW description: {}", e))?;
            
            let hnsw = match metric {
                VectorMetric::L2 => HnswGraph::L2(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistL2, &mut data_reader)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
                VectorMetric::Cosine => HnswGraph::Cosine(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistCosine, &mut data_reader)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
                VectorMetric::InnerProduct => HnswGraph::Dot(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistDot, &mut data_reader)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
                VectorMetric::L1 => HnswGraph::L1(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistL1, &mut data_reader)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
                VectorMetric::Hamming => HnswGraph::Hamming(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistHamming, &mut data_reader)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
                VectorMetric::Jaccard => HnswGraph::Jaccard(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistJaccard, &mut data_reader)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
            };
            
            use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
            let map_builder = ParquetRecordBatchReaderBuilder::try_new(bytes::Bytes::from(mapping_bytes))?;
            let map_reader = map_builder.build()?;
            let batches: Vec<RecordBatch> = map_reader.map(|r| r.map_err(anyhow::Error::from)).collect::<Result<Vec<_>>>()?;
            let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches)?;
            let row_id_array = batch.column(0).as_primitive::<arrow::datatypes::UInt32Type>();
            let row_id_mapping: Vec<usize> = row_id_array.values().iter().map(|&x| x as usize).collect();
            
            cluster_graphs.insert(cid, (hnsw, row_id_mapping));
        }
        
        let index = HnswIvfIndex {
            centroids,
            metric,
            cluster_graphs,
            _n_lists: 0, // Not strictly used for search
            dim,
            _pq_encoder: None,
            _compute_context: crate::core::index::gpu::ComputeContext::auto_detect(),
        };
        
        let index_arc = Arc::new(index);
        // HNSW_IVF_CACHE.insert(path.to_string(), index_arc.clone()).await;
        
        Ok(index_arc)
    }

    pub async fn load_async(store: Arc<dyn ObjectStore>, base_path: &str) -> Result<Arc<Self>> {
        Self::load_async_with_cache_key(store, base_path, base_path).await
    }

    pub async fn load_async_with_cache_key(
        store: Arc<dyn ObjectStore>, 
        base_path: &str,
        cache_key: &str,
    ) -> Result<Arc<Self>> {
        use crate::core::cache::{HNSW_IVF_CACHE, DiskCache};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let cache_key_str = cache_key.to_string();
        if let Some(cached) = HNSW_IVF_CACHE.get(&cache_key_str).await {
            return Ok(cached);
        }

        let disk_cache = DiskCache::new(store.clone());

        let root_path = if base_path.contains("://") {
             if let Ok(url) = url::Url::parse(base_path) {
                 url.path().trim_start_matches('/').to_string()
             } else {
                 base_path.to_string()
             }
        } else {
            base_path.to_string()
        };

        let centroids_path = format!("{}.centroids.parquet", root_path);
        let centroids_bytes = disk_cache.get_bytes(&centroids_path).await?;
        
        let builder = ParquetRecordBatchReaderBuilder::try_new(centroids_bytes)?;
        let reader = builder.build()?;
        
        let batches: Vec<RecordBatch> = reader.map(|r| r.map_err(anyhow::Error::from)).collect::<Result<Vec<_>>>()?;
        if batches.is_empty() {
             anyhow::bail!("No data in centroids file");
        }
        let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches)?;

        let centroid_list = batch.column(0).as_list::<i32>();
        
        let mut centroids = Vec::new();
        for i in 0..centroid_list.len() {
            let values = centroid_list.value(i);
            let float_array = values.as_primitive::<arrow::datatypes::Float32Type>();
            let centroid: Vec<f32> = float_array.values().to_vec();
            centroids.push(centroid);
        }
        
        let n_lists = centroids.len();
        let dim = centroids[0].len();
        
        // Load metric from parquet metadata
        let metric = {
            let centroids_bytes_for_meta = disk_cache.get_bytes(&centroids_path).await?;
            let parquet_reader = parquet::file::reader::SerializedFileReader::new(centroids_bytes_for_meta)?;
            let file_metadata = parquet_reader.metadata().file_metadata();
            let mut loaded_metric = VectorMetric::L2;
            if let Some(kv_list) = file_metadata.key_value_metadata() {
                for kv in kv_list {
                    if kv.key == "vector_metric" {
                        if let Some(ref val) = kv.value {
                            loaded_metric = match val.as_str() {
                                "cosine" => VectorMetric::Cosine,
                                "ip" => VectorMetric::InnerProduct,
                                _ => VectorMetric::L2,
                            };
                        }
                        break;
                    }
                }
            }
            loaded_metric
        };

        let list_stream = store.list(None);
        let files: Vec<object_store::ObjectMeta> = list_stream.try_collect().await?;
        
        let mut cluster_ids = Vec::new();
        let prefix_str = root_path.clone();
        for file in &files {
            let path_str = file.location.to_string();
            if !path_str.starts_with(&prefix_str) {
                continue;
            }
            if path_str.ends_with(".hnsw.graph") {
                if let Some(start) = path_str.rfind(".cluster_") {
                    if let Some(end) = path_str.rfind(".hnsw.graph") {
                        if let Ok(id) = path_str[start + 9..end].parse::<usize>() {
                            cluster_ids.push(id);
                        }
                    }
                }
            }
        }
        
        let fetch_concurrency = 16; 
        let root_path_clone = root_path.clone();
        
        let bodies = futures::stream::iter(cluster_ids)
            .map(move |cluster_id| {
                let root_path = root_path_clone.clone();
                let dc = disk_cache.clone();
                async move {
                    let hnsw_key = format!("{}.cluster_{}.hnsw.graph", root_path, cluster_id);
                    let data_key = format!("{}.cluster_{}.hnsw.data", root_path, cluster_id);
                    let mapping_key = format!("{}.cluster_{}.mapping.parquet", root_path, cluster_id);
                
                    let res_graph = dc.get_bytes(&hnsw_key).await?;
                    let res_data = dc.get_bytes(&data_key).await?;
                    let res_mapping = dc.get_bytes(&mapping_key).await?;
                    
                    let graph_cursor = Cursor::new(res_graph);
                    let data_cursor = Cursor::new(res_data);
                    let mut graph_reader = std::io::BufReader::new(graph_cursor);
                    let mut data_reader = std::io::BufReader::new(data_cursor);
                    
                    let description = crate::core::index::hnsw_rs::hnswio::load_description(&mut graph_reader)
                        .map_err(|e| anyhow::anyhow!("Failed to load HNSW description: {}", e))?;
                    
                    let metric = metric; // Use the metric loaded from centroids metadata
                    // For now we assume L2 if not specified or derive from path?
                    // Better: HnswIvfIndex should save metric in centroids parquet metadata.
                    
                    let hnsw = match metric {
                        VectorMetric::L2 => HnswGraph::L2(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistL2, &mut data_reader)
                            .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
                        VectorMetric::Cosine => HnswGraph::Cosine(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistCosine, &mut data_reader)
                            .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
                        VectorMetric::InnerProduct => HnswGraph::Dot(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistDot, &mut data_reader)
                            .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
                        VectorMetric::L1 => HnswGraph::L1(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistL1, &mut data_reader)
                            .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
                        VectorMetric::Hamming => HnswGraph::Hamming(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistHamming, &mut data_reader)
                            .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
                        VectorMetric::Jaccard => HnswGraph::Jaccard(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistJaccard, &mut data_reader)
                            .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
                    };
                        
                    let map_builder = ParquetRecordBatchReaderBuilder::try_new(res_mapping)?;
                    let map_reader = map_builder.build()?;
                    let batches: Vec<RecordBatch> = map_reader.map(|r| r.map_err(anyhow::Error::from)).collect::<Result<Vec<_>>>()?;
                    if batches.is_empty() {
                         return Err(anyhow::anyhow!("No data in mapping file"));
                    }
                    let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches)?;
                    let row_id_array = batch.column(0).as_primitive::<arrow::datatypes::UInt32Type>();
                    let row_id_mapping: Vec<usize> = row_id_array.values().iter().map(|&x| x as usize).collect();
                    
                    Ok((cluster_id, (hnsw, row_id_mapping)))
                }
            })
            .buffer_unordered(fetch_concurrency);
            
        type ClusterResult = (usize, (HnswGraph, Vec<usize>));
        let results: Vec<Result<ClusterResult>> = bodies.collect().await;
        
        let mut cluster_graphs = HashMap::new();
        for res in results {
            let (cid, val) = res?;
            cluster_graphs.insert(cid, val);
        }

        println!("Loaded {} cluster graphs (Async)", cluster_graphs.len());

        let index = HnswIvfIndex {
            centroids,
            metric,
            cluster_graphs,
            _n_lists: n_lists,
            dim,
            _pq_encoder: None,
            _compute_context: crate::core::index::gpu::ComputeContext::auto_detect(),
        };
        
        let index_arc = Arc::new(index);
        HNSW_IVF_CACHE.insert(cache_key_str, index_arc.clone()).await;
        
        Ok(index_arc)
    }

    pub fn load(base_path: &str) -> Result<Self> {
        use arrow::array::{AsArray, Array};
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs::File;

        let centroids_path = format!("{}.centroids.parquet", base_path);
        let file = File::open(&centroids_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;
        
        let batches: Vec<RecordBatch> = reader.collect::<std::result::Result<_, _>>()?;
        if batches.is_empty() {
             anyhow::bail!("No data in centroids file");
        }
        let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches)?;

        let centroid_list = batch.column(0).as_list::<i32>();
        
        let mut centroids = Vec::new();
        for i in 0..centroid_list.len() {
            let values = centroid_list.value(i);
            let float_array = values.as_primitive::<arrow::datatypes::Float32Type>();
            let centroid: Vec<f32> = float_array.values().to_vec();
            centroids.push(centroid);
        }
        
        let n_lists = centroids.len();
        let dim = centroids[0].len();
        
        // Load metric from parquet metadata
        let metric = {
            let file = File::open(&centroids_path)?;
            let parquet_reader = parquet::file::reader::SerializedFileReader::new(file)?;
            let file_metadata = parquet_reader.metadata().file_metadata();
            let mut loaded_metric = VectorMetric::L2;
            if let Some(kv_list) = file_metadata.key_value_metadata() {
                for kv in kv_list {
                    if kv.key == "vector_metric" {
                        if let Some(ref val) = kv.value {
                            loaded_metric = match val.as_str() {
                                "cosine" => VectorMetric::Cosine,
                                "ip" => VectorMetric::InnerProduct,
                                _ => VectorMetric::L2,
                            };
                        }
                        break;
                    }
                }
            }
            loaded_metric
        };

        println!("Loaded {} centroids of dimension {}", n_lists, dim);

        let mut cluster_graphs = HashMap::new();
        for cluster_id in 0..n_lists {
            let hnsw_path = format!("{}.cluster_{}", base_path, cluster_id);
            let mapping_path = format!("{}.cluster_{}.mapping.parquet", base_path, cluster_id);
            
            if !std::path::Path::new(&format!("{}.hnsw.graph", hnsw_path)).exists() {
                continue;
            }

            use std::io::BufReader;
            let graph_file = File::open(format!("{}.hnsw.graph", hnsw_path))?;
            let data_file = File::open(format!("{}.hnsw.data", hnsw_path))?;
            let mut graph_reader = BufReader::new(graph_file);
            let mut data_reader = BufReader::new(data_file);
            
            let description = crate::core::index::hnsw_rs::hnswio::load_description(&mut graph_reader)
                .map_err(|e| anyhow::anyhow!("Failed to load HNSW description: {}", e))?;
            
            // Use the metric loaded from centroids metadata
            let hnsw = match metric {
                VectorMetric::L2 => HnswGraph::L2(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistL2, &mut data_reader)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
                VectorMetric::Cosine => HnswGraph::Cosine(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistCosine, &mut data_reader)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
                VectorMetric::InnerProduct => HnswGraph::Dot(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistDot, &mut data_reader)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
                VectorMetric::L1 => HnswGraph::L1(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistL1, &mut data_reader)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
                VectorMetric::Hamming => HnswGraph::Hamming(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistHamming, &mut data_reader)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
                VectorMetric::Jaccard => HnswGraph::Jaccard(crate::core::index::hnsw_rs::hnswio::load_hnsw_with_dist(&mut graph_reader, &description, DistJaccard, &mut data_reader)
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?),
            };
            
            let file = File::open(&mapping_path)?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            let reader = builder.build()?;
            
            let batches: Vec<RecordBatch> = reader.collect::<std::result::Result<_, _>>()?;
            if batches.is_empty() {
                 anyhow::bail!("No data in mapping file");
            }
            let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches)?;
            
            let row_id_array = batch.column(0).as_primitive::<arrow::datatypes::UInt32Type>();
            let row_id_mapping: Vec<usize> = row_id_array.values().iter().map(|&x| x as usize).collect();
            
            cluster_graphs.insert(cluster_id, (hnsw, row_id_mapping));
        }

        println!("Loaded {} cluster graphs", cluster_graphs.len());

        Ok(HnswIvfIndex {
            centroids,
            metric,
            cluster_graphs,
            _n_lists: n_lists,
            dim,
            _pq_encoder: None,
            _compute_context: crate::core::index::gpu::ComputeContext::auto_detect(),
        })
    }
    
    pub fn size_in_bytes(&self) -> usize {
        let mut size = 0;
        for c in &self.centroids {
            size += c.len() * 4;
        }
        for (_hnsw, mapping) in self.cluster_graphs.values() {
            size += mapping.len() * 8;
            let data_count = mapping.len();
            size += data_count * self.dim * 4;
            size += data_count * 16 * 4; 
        }
        size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use roaring::RoaringBitmap;

    #[test]
    fn test_hnsw_ivf_pre_filtering() {
        let mut vectors = Vec::new();
        for i in 0..100 {
            let mut vec = vec![0.0; 4];
            vec[0] = (i as f32) / 100.0;
            vectors.push(vec);
        }

        let index = HnswIvfIndex::build(vectors.clone(), VectorMetric::L2, Some(2), Some(16), false).unwrap();
        let query = VectorValue::Float32(vectors[50].clone());

        // Unfiltered search (should find ID 50 easily since it's an exact match)
        let results_unfiltered = index.search(&query, 5, 40, None);
        assert!(!results_unfiltered.is_empty(), "Should return results");
        
        let mut found_50 = false;
        for (id, _) in &results_unfiltered {
            if *id == 50 { found_50 = true; break; }
        }
        assert!(found_50, "Unfiltered search should easily find exact match ID 50");

        // Filtered search (exclude ID 50, only allow 51, 52, 53)
        let mut filter = RoaringBitmap::new();
        filter.insert(51);
        filter.insert(52);
        filter.insert(53);

        let results_filtered = index.search(&query, 3, 40, Some(&filter));
        assert!(!results_filtered.is_empty(), "Filtered search should return results");

        let mut found_50_filtered = false;
        let mut found_51_filtered = false;
        for (id, _) in &results_filtered {
            if *id == 50 { found_50_filtered = true; }
            if *id == 51 { found_51_filtered = true; }
        }

        assert!(!found_50_filtered, "Pre-filtering failed to exclude exact match ID 50");
        assert!(found_51_filtered, "Pre-filtering failed to find the next best allowed match ID 51");
        
        for (id, _) in &results_filtered {
            assert!(filter.contains(*id as u32), "Result ID {} was not in the filter!", id);
        }
    }
}
