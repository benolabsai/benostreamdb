// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Modified by Richard Albright / BenoStreamDB on 2026-03-29 to add pre-filtering support and better integration with Iceberg manifests.
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
use crate::core::cache::CacheExt;
use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tracing;

use super::benoquant::BenoQuantEncoder;
use super::ivf::simple_kmeans_with_seed;
use super::pq::{PqConfig, PqEncoder};
use super::{Quantizer, QuantizerImpl, VectorMetric, VectorValue};
use crate::core::index::hnsw_rs::prelude::*;
use crate::core::manifest::IndexAlgorithm;
use arrow::array::{Array, ArrayRef};
use arrow::record_batch::RecordBatch;
use futures::{StreamExt, TryStreamExt};
use object_store::ObjectStore;
use parquet::file::reader::FileReader;
use rayon::prelude::*;

/// Capacity of the per-index cluster LRU.
///
/// Each segment holds one HNSW graph per Voronoi cluster, and a miss
/// re-deserializes that cluster's Arrow IPC payload. The old cap of 64 against
/// 256 clusters meant a 25% hit rate and continuous IPC decoding under
/// concurrent queries. 256 pins every cluster of a segment; the memory cost is
/// `clusters × vectors_per_cluster × dim × bytes_per_coord`, roughly 49 MB per
/// segment at 128 dims and 294 MB at 768 dims. Override with
/// `BSDB_CLUSTER_CACHE_CAPACITY` when the cache budget cannot absorb that.
fn cluster_cache_capacity() -> u64 {
    std::env::var("BSDB_CLUSTER_CACHE_CAPACITY")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(256)
}

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
    fn eval(
        &self,
        va: &[crate::core::index::SparseVector],
        vb: &[crate::core::index::SparseVector],
    ) -> f32 {
        // HNSW-rs uses &[T]. For sparse vectors, T is SparseVector, so we compare the first elements.
        1.0 - crate::core::index::distance::sparse_dot_product(
            &va[0].indices,
            &va[0].values,
            &vb[0].indices,
            &vb[0].values,
        )
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
    TurboQuant8(Hnsw<u8, crate::core::index::distance::DistL2u8>),
    /// SQ8 codes ranked by the decoded metric (cosine for angular datasets,
    /// L2 otherwise). Byte-space L2 is wrong for SQ8: per-dimension scales and
    /// the vector norm dominate the ranking, so an all-zero vector ranks first
    /// instead of last on cosine data.
    Sq8(Hnsw<u8, crate::core::index::sq::DistSq8>),
    TurboQuant4(Hnsw<u8, crate::core::index::distance::DistL2u4>),
    SparseDot(Hnsw<crate::core::index::SparseVector, DistSparseDot>),
    Pq(Hnsw<u8, crate::core::index::pq::DistPqSdc>),

    // Arrow IPC variants (read-only)
    L2Arrow(crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw<f32, DistL2>),
    CosineArrow(crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw<f32, DistCosine>),
    DotArrow(crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw<f32, DistDot>),
    L1Arrow(crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw<f32, DistL1>),
    HammingArrow(crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw<f32, DistHamming>),
    JaccardArrow(crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw<f32, DistJaccard>),
    BinaryHammingArrow(crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw<u8, DistHammingPacked>),
    TurboQuant8Arrow(
        crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw<
            u8,
            crate::core::index::distance::DistL2u8,
        >,
    ),
    Sq8Arrow(
        crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw<u8, crate::core::index::sq::DistSq8>,
    ),
    TurboQuant4Arrow(
        crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw<
            u8,
            crate::core::index::distance::DistL2u4,
        >,
    ),
    SparseDotArrow(
        crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw<
            crate::core::index::SparseVector,
            DistSparseDot,
        >,
    ),
    PqArrow(
        crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw<u8, crate::core::index::pq::DistPqSdc>,
    ),
}

impl HnswGraph {
    pub fn search(
        &self,
        query: &VectorValue,
        k: usize,
        ef: usize,
        filter: Option<&roaring::RoaringBitmap>,
    ) -> Vec<Neighbour> {
        match (self, query) {
            (HnswGraph::L2(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            (HnswGraph::Cosine(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            (HnswGraph::Dot(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            (HnswGraph::L1(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            (HnswGraph::Hamming(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            (HnswGraph::Jaccard(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            (HnswGraph::BinaryHamming(h), VectorValue::Binary(q)) => h.search(q, k, ef, filter),
            (HnswGraph::TurboQuant8(h), VectorValue::Binary(q)) => h.search(q, k, ef, filter),
            (HnswGraph::Sq8(h), VectorValue::Binary(q)) => h.search(q, k, ef, filter),
            (HnswGraph::TurboQuant4(h), VectorValue::Binary(q)) => h.search(q, k, ef, filter),
            (HnswGraph::Pq(h), VectorValue::Binary(q)) => h.search(q, k, ef, filter),
            (HnswGraph::SparseDot(h), VectorValue::Sparse(q)) => {
                h.search(std::slice::from_ref(q), k, ef, filter)
            }
            (HnswGraph::L2(h), VectorValue::Float16(q)) => h.search(q, k, ef, filter),

            // Arrow IPC paths
            (HnswGraph::L2Arrow(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            (HnswGraph::CosineArrow(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            (HnswGraph::DotArrow(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            (HnswGraph::L1Arrow(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            (HnswGraph::HammingArrow(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            (HnswGraph::JaccardArrow(h), VectorValue::Float32(q)) => h.search(q, k, ef, filter),
            (HnswGraph::BinaryHammingArrow(h), VectorValue::Binary(q)) => {
                h.search(q, k, ef, filter)
            }
            (HnswGraph::TurboQuant8Arrow(h), VectorValue::Binary(q)) => h.search(q, k, ef, filter),
            (HnswGraph::Sq8Arrow(h), VectorValue::Binary(q)) => h.search(q, k, ef, filter),
            (HnswGraph::TurboQuant4Arrow(h), VectorValue::Binary(q)) => h.search(q, k, ef, filter),
            (HnswGraph::PqArrow(h), VectorValue::Binary(q)) => h.search(q, k, ef, filter),
            (HnswGraph::SparseDotArrow(h), VectorValue::Sparse(q)) => {
                h.search(std::slice::from_ref(q), k, ef, filter)
            }
            (HnswGraph::L2Arrow(h), VectorValue::Float16(q)) => h.search(q, k, ef, filter),

            _ => {
                tracing::error!("HnswGraph / query type mismatch");
                vec![]
            }
        }
    }

    /// Search using a precomputed [`crate::core::index::PreparedQuantizedQuery`]
    /// for quantized graphs, falling back to [`Self::search`] for unquantized or
    /// unmatched variants.
    pub fn search_with_quantizer(
        &self,
        query_fallback: &VectorValue,
        prepared: Option<&crate::core::index::PreparedQuantizedQuery<'_>>,
        k: usize,
        ef: usize,
        filter: Option<&roaring::RoaringBitmap>,
    ) -> Vec<Neighbour> {
        use crate::core::index::PreparedQuantizedQuery;
        match (self, prepared) {
            (HnswGraph::TurboQuant8(h), Some(PreparedQuantizedQuery::BenoQuant(pq))) => {
                h.search_with_scorer(k, ef, filter, |code| pq.distance(code))
            }
            (HnswGraph::TurboQuant4(h), Some(PreparedQuantizedQuery::BenoQuant(pq))) => {
                h.search_with_scorer(k, ef, filter, |code| pq.distance(code))
            }
            (HnswGraph::TurboQuant8Arrow(h), Some(PreparedQuantizedQuery::BenoQuant(pq))) => {
                h.search_with_scorer(k, ef, filter, |code| pq.distance(code))
            }
            (HnswGraph::TurboQuant4Arrow(h), Some(PreparedQuantizedQuery::BenoQuant(pq))) => {
                h.search_with_scorer(k, ef, filter, |code| pq.distance(code))
            }
            (HnswGraph::Sq8(h), Some(PreparedQuantizedQuery::Sq(psq))) => {
                h.search_with_scorer(k, ef, filter, |code| psq.distance(code))
            }
            (HnswGraph::Sq8Arrow(h), Some(PreparedQuantizedQuery::Sq(psq))) => {
                h.search_with_scorer(k, ef, filter, |code| psq.distance(code))
            }
            (HnswGraph::Pq(h), Some(PreparedQuantizedQuery::Pq(ppq))) => {
                h.search_with_scorer(k, ef, filter, |code| ppq.distance(code))
            }
            (HnswGraph::PqArrow(h), Some(PreparedQuantizedQuery::Pq(ppq))) => {
                h.search_with_scorer(k, ef, filter, |code| ppq.distance(code))
            }
            _ => self.search(query_fallback, k, ef, filter),
        }
    }

    /// Retrieve the stored vector for a neighbour, for exact rescoring.
    ///
    /// The f32 graph variants return the raw vector. The compressed variants
    /// return the quantizer-decoded approximation (the graph stores codes, not
    /// raw vectors). Returns `None` when the graph's element type cannot be
    /// decoded to f32.
    pub fn get_vector_for_rescore(
        &self,
        neighbour: &Neighbour,
        quantizer: Option<&QuantizerImpl>,
    ) -> Option<Vec<f32>> {
        // Arrow-backed graphs store rows in layer-indexed order: `d_id` is the
        // original row id while `p_id.1` is the Arrow row index. `get_vector`
        // indexes the Arrow table, so it must use `p_id.1`, not `d_id`.
        let arrow_row = neighbour.p_id.1.max(0) as usize;
        match self {
            HnswGraph::Dot(h) => h.get_point_indexation().get_point_data(&neighbour.p_id),
            HnswGraph::DotArrow(h) => Some(h.get_vector(arrow_row)),
            HnswGraph::TurboQuant8(h) => {
                let code = h.get_point_indexation().get_point_data(&neighbour.p_id)?;
                quantizer.map(|q| q.decode(&code))
            }
            HnswGraph::Sq8(h) => {
                let code = h.get_point_indexation().get_point_data(&neighbour.p_id)?;
                quantizer.map(|q| q.decode(&code))
            }
            HnswGraph::TurboQuant4(h) => {
                let code = h.get_point_indexation().get_point_data(&neighbour.p_id)?;
                quantizer.map(|q| q.decode(&code))
            }
            HnswGraph::Pq(h) => {
                let code = h.get_point_indexation().get_point_data(&neighbour.p_id)?;
                quantizer.map(|q| q.decode(&code))
            }
            HnswGraph::TurboQuant8Arrow(h) => {
                let code = h.get_vector(arrow_row);
                quantizer.map(|q| q.decode(&code))
            }
            HnswGraph::Sq8Arrow(h) => {
                let code = h.get_vector(arrow_row);
                quantizer.map(|q| q.decode(&code))
            }
            HnswGraph::TurboQuant4Arrow(h) => {
                let code = h.get_vector(arrow_row);
                quantizer.map(|q| q.decode(&code))
            }
            HnswGraph::PqArrow(h) => {
                let code = h.get_vector(arrow_row);
                quantizer.map(|q| q.decode(&code))
            }
            _ => None,
        }
    }

    /// Approximate bytes the decoded graph occupies.
    ///
    /// The Arrow IPC variants are flat record batches served zero-copy — over an
    /// mmap of the index object, or over owned bytes — so their footprint is the
    /// backing buffer size. The legacy `hnsw_rs` variants box every point behind
    /// `Arc<RwLock<Point>>` and measured ~20 KB per point under dhat profiling.
    pub fn footprint_bytes(&self) -> usize {
        // Measured heap overhead per point for the in-memory hnsw_rs graph.
        const LEGACY_BYTES_PER_POINT: usize = 20_000;

        match self {
            HnswGraph::L2Arrow(h) => h.footprint_bytes(),
            HnswGraph::CosineArrow(h) => h.footprint_bytes(),
            HnswGraph::DotArrow(h) => h.footprint_bytes(),
            HnswGraph::L1Arrow(h) => h.footprint_bytes(),
            HnswGraph::HammingArrow(h) => h.footprint_bytes(),
            HnswGraph::JaccardArrow(h) => h.footprint_bytes(),
            HnswGraph::BinaryHammingArrow(h) => h.footprint_bytes(),
            HnswGraph::TurboQuant8Arrow(h) => h.footprint_bytes(),
            HnswGraph::Sq8Arrow(h) => h.footprint_bytes(),
            HnswGraph::TurboQuant4Arrow(h) => h.footprint_bytes(),
            HnswGraph::SparseDotArrow(h) => h.footprint_bytes(),
            HnswGraph::PqArrow(h) => h.footprint_bytes(),

            HnswGraph::L2(h) => h.get_nb_point() * LEGACY_BYTES_PER_POINT,
            HnswGraph::Cosine(h) => h.get_nb_point() * LEGACY_BYTES_PER_POINT,
            HnswGraph::Dot(h) => h.get_nb_point() * LEGACY_BYTES_PER_POINT,
            HnswGraph::L1(h) => h.get_nb_point() * LEGACY_BYTES_PER_POINT,
            HnswGraph::Hamming(h) => h.get_nb_point() * LEGACY_BYTES_PER_POINT,
            HnswGraph::Jaccard(h) => h.get_nb_point() * LEGACY_BYTES_PER_POINT,
            HnswGraph::BinaryHamming(h) => h.get_nb_point() * LEGACY_BYTES_PER_POINT,
            HnswGraph::TurboQuant8(h) => h.get_nb_point() * LEGACY_BYTES_PER_POINT,
            HnswGraph::Sq8(h) => h.get_nb_point() * LEGACY_BYTES_PER_POINT,
            HnswGraph::TurboQuant4(h) => h.get_nb_point() * LEGACY_BYTES_PER_POINT,
            HnswGraph::SparseDot(h) => h.get_nb_point() * LEGACY_BYTES_PER_POINT,
            HnswGraph::Pq(h) => h.get_nb_point() * LEGACY_BYTES_PER_POINT,
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
            (HnswGraph::TurboQuant8(h), VectorValue::Binary(v)) => h.insert_slice((&v, local_id)),
            (HnswGraph::Sq8(h), VectorValue::Binary(v)) => h.insert_slice((&v, local_id)),
            (HnswGraph::TurboQuant4(h), VectorValue::Binary(v)) => h.insert_slice((&v, local_id)),
            (HnswGraph::Pq(h), VectorValue::Binary(v)) => h.insert_slice((&v, local_id)),
            (HnswGraph::SparseDot(h), VectorValue::Sparse(v)) => h.insert_slice((&[v], local_id)),
            (HnswGraph::L2(h), VectorValue::Float16(v)) => h.insert_slice((&v, local_id)),

            // Arrow IPC variants are read-only
            (HnswGraph::L2Arrow(_), _)
            | (HnswGraph::CosineArrow(_), _)
            | (HnswGraph::DotArrow(_), _)
            | (HnswGraph::L1Arrow(_), _)
            | (HnswGraph::HammingArrow(_), _)
            | (HnswGraph::JaccardArrow(_), _)
            | (HnswGraph::BinaryHammingArrow(_), _)
            | (HnswGraph::TurboQuant8Arrow(_), _)
            | (HnswGraph::Sq8Arrow(_), _)
            | (HnswGraph::TurboQuant4Arrow(_), _)
            | (HnswGraph::SparseDotArrow(_), _)
            | (HnswGraph::PqArrow(_), _) => {
                tracing::error!("Cannot insert into read-only Arrow IPC HnswGraph");
            }

            _ => tracing::error!("HnswGraph / insert value type mismatch"),
        }
    }

    pub fn parallel_insert(&self, data: Vec<(&[f32], usize)>) {
        if data.is_empty() {
            return;
        }

        match self {
            HnswGraph::L2(h) => h.parallel_insert_slice(&data),
            HnswGraph::Cosine(h) => h.parallel_insert_slice(&data),
            HnswGraph::Dot(h) => h.parallel_insert_slice(&data),
            HnswGraph::L1(h) => h.parallel_insert_slice(&data),

            // Arrow IPC variants are read-only
            HnswGraph::L2Arrow(_)
            | HnswGraph::CosineArrow(_)
            | HnswGraph::DotArrow(_)
            | HnswGraph::L1Arrow(_)
            | HnswGraph::HammingArrow(_)
            | HnswGraph::JaccardArrow(_)
            | HnswGraph::BinaryHammingArrow(_)
            | HnswGraph::TurboQuant8Arrow(_)
            | HnswGraph::Sq8Arrow(_)
            | HnswGraph::TurboQuant4Arrow(_)
            | HnswGraph::SparseDotArrow(_)
            | HnswGraph::PqArrow(_) => {
                tracing::error!("Cannot insert into read-only Arrow IPC HnswGraph");
            }

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
            HnswGraph::L2(h) => h
                .file_dump(&path_string)
                .map(|_| ())
                .map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::Cosine(h) => h
                .file_dump(&path_string)
                .map(|_| ())
                .map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::Dot(h) => h
                .file_dump(&path_string)
                .map(|_| ())
                .map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::L1(h) => h
                .file_dump(&path_string)
                .map(|_| ())
                .map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::Hamming(h) => h
                .file_dump(&path_string)
                .map(|_| ())
                .map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::Jaccard(h) => h
                .file_dump(&path_string)
                .map(|_| ())
                .map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::BinaryHamming(h) => h
                .file_dump(&path_string)
                .map(|_| ())
                .map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::TurboQuant8(h) => h
                .file_dump(&path_string)
                .map(|_| ())
                .map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::Sq8(h) => h
                .file_dump(&path_string)
                .map(|_| ())
                .map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::TurboQuant4(h) => h
                .file_dump(&path_string)
                .map(|_| ())
                .map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::Pq(h) => h
                .file_dump(&path_string)
                .map(|_| ())
                .map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::SparseDot(h) => h
                .file_dump(&path_string)
                .map(|_| ())
                .map_err(|e| Box::new(std::io::Error::other(e)) as Box<dyn std::error::Error>),
            HnswGraph::L2Arrow(_)
            | HnswGraph::CosineArrow(_)
            | HnswGraph::DotArrow(_)
            | HnswGraph::L1Arrow(_)
            | HnswGraph::HammingArrow(_)
            | HnswGraph::JaccardArrow(_)
            | HnswGraph::BinaryHammingArrow(_)
            | HnswGraph::TurboQuant8Arrow(_)
            | HnswGraph::Sq8Arrow(_)
            | HnswGraph::TurboQuant4Arrow(_)
            | HnswGraph::SparseDotArrow(_)
            | HnswGraph::PqArrow(_) => Err(Box::new(std::io::Error::other(
                "Cannot dump read-only Arrow IPC HnswGraph",
            )) as Box<dyn std::error::Error>),
        }
    }

    pub fn set_extend_candidates(&mut self, flag: bool) {
        match self {
            HnswGraph::L2(h) => h.set_extend_candidates(flag),
            HnswGraph::Cosine(h) => h.set_extend_candidates(flag),
            HnswGraph::Dot(h) => h.set_extend_candidates(flag),
            HnswGraph::L1(h) => h.set_extend_candidates(flag),
            HnswGraph::Hamming(h) => h.set_extend_candidates(flag),
            HnswGraph::Jaccard(h) => h.set_extend_candidates(flag),
            HnswGraph::BinaryHamming(h) => h.set_extend_candidates(flag),
            HnswGraph::TurboQuant8(h) => h.set_extend_candidates(flag),
            HnswGraph::Sq8(h) => h.set_extend_candidates(flag),
            HnswGraph::TurboQuant4(h) => h.set_extend_candidates(flag),
            HnswGraph::Pq(h) => h.set_extend_candidates(flag),
            HnswGraph::SparseDot(h) => h.set_extend_candidates(flag),
            _ => {}
        }
    }

    pub fn set_keeping_pruned(&mut self, flag: bool) {
        match self {
            HnswGraph::L2(h) => h.set_keeping_pruned(flag),
            HnswGraph::Cosine(h) => h.set_keeping_pruned(flag),
            HnswGraph::Dot(h) => h.set_keeping_pruned(flag),
            HnswGraph::L1(h) => h.set_keeping_pruned(flag),
            HnswGraph::Hamming(h) => h.set_keeping_pruned(flag),
            HnswGraph::Jaccard(h) => h.set_keeping_pruned(flag),
            HnswGraph::BinaryHamming(h) => h.set_keeping_pruned(flag),
            HnswGraph::TurboQuant8(h) => h.set_keeping_pruned(flag),
            HnswGraph::Sq8(h) => h.set_keeping_pruned(flag),
            HnswGraph::TurboQuant4(h) => h.set_keeping_pruned(flag),
            HnswGraph::Pq(h) => h.set_keeping_pruned(flag),
            HnswGraph::SparseDot(h) => h.set_keeping_pruned(flag),
            _ => {}
        }
    }
}

/// Owns a memory map so `bytes::Bytes` can alias it without copying.
///
/// `Bytes::from_owner` needs `AsRef<[u8]>`, which `Arc<Mmap>` does not
/// implement; this newtype supplies it and keeps the mapping alive as long as
/// any slice derived from it is held.
struct MmapOwner(Arc<memmap2::Mmap>);

impl AsRef<[u8]> for MmapOwner {
    fn as_ref(&self) -> &[u8] {
        self.0.as_ref().as_ref()
    }
}

/// Lazy loader for per-cluster HNSW graphs and mappings packed inside a Puffin container.
///
/// Holds the memory-mapped or owned byte buffer of the `.puffin` container and the byte
/// slices `(offset, length)` of all cluster graphs and mappings. Slices are deserialized
/// strictly on demand when an IVF coarse query probes that specific cluster list.
pub struct LazyPuffinBlobs {
    container: bytes::Bytes,
    cluster_slices: HashMap<usize, ((usize, usize), (usize, usize))>,
    quantizer: Option<QuantizerImpl>,
    metric: VectorMetric,
    augment_m: f32,
}

impl LazyPuffinBlobs {
    pub fn load_cluster(&self, cluster_id: usize) -> Result<(HnswGraph, Vec<usize>)> {
        let (graph_slice, mapping_slice) =
            self.cluster_slices.get(&cluster_id).ok_or_else(|| {
                anyhow::anyhow!("Cluster {} not found in puffin container", cluster_id)
            })?;

        let graph_bytes = self
            .container
            .slice(graph_slice.0..graph_slice.0 + graph_slice.1);
        let mapping_bytes = self
            .container
            .slice(mapping_slice.0..mapping_slice.0 + mapping_slice.1);

        HnswIvfIndex::deserialize_single_cluster(
            cluster_id,
            graph_bytes,
            mapping_bytes,
            self.quantizer.as_ref(),
            self.metric,
            self.augment_m,
        )
    }

    pub fn total_clusters(&self) -> usize {
        self.cluster_slices.len()
    }
}

/// HNSW-IVF Hybrid Index
pub struct HnswIvfIndex {
    /// Cluster centroids for coarse quantization
    centroids: Vec<Vec<f32>>,
    /// Distance metric used for search
    metric: VectorMetric,
    /// Per-cluster HNSW graphs: cluster_id -> (hnsw_index, row_id_mapping)
    cluster_graphs: HashMap<usize, Arc<(HnswGraph, Vec<usize>)>>,
    /// In-memory LRU cache of deserialized probed clusters (used when lazy loading from Puffin)
    cluster_cache: moka::sync::Cache<usize, Arc<(HnswGraph, Vec<usize>)>>,
    /// Lazy blob slices inside a Puffin container for on-demand cluster deserialization
    lazy_blobs: Option<Arc<LazyPuffinBlobs>>,
    /// True when the per-cluster graphs are zero-copy views over `mmap`'d files.
    /// In that case the graph bytes live in the OS page cache rather than the
    /// process heap, so [`Self::size_in_bytes`] must not charge them against the
    /// memory cache budget (only centroids + row-id mappings are heap-resident).
    mmap_backed: bool,
    /// Number of clusters (for metadata/scaling)
    _n_lists: usize,
    /// Vector dimensionality
    dim: usize,
    /// Optional quantizer for compression (TurboQuant or PQ)
    pub quantizer: Option<QuantizerImpl>,
    /// Candidate oversampling factor used before exact InnerProduct rescoring.
    ///
    /// The per-cluster HNSW graph ranks by its own safe, non-negative distance
    /// (cosine for the `Dot` variants after the `DistDot` revert). For
    /// `InnerProduct` we retrieve `k * rescore_oversample` candidates per
    /// cluster and then re-rank them by exact raw `-dot` in f32. The default of
    /// `4` matches the pre-existing `k * 4` candidate budget, so behaviour is
    /// unchanged unless the factor is raised.
    rescore_oversample: usize,
    /// Global max-norm `m` for the asymmetric MIPS augmentation
    /// ([`DistDot::augmented`](crate::core::index::hnsw_rs::dist::DistDot::augmented)).
    /// `0.0` disables it. Persisted in the centroids Parquet metadata so every
    /// load path reconstructs the same distance.
    augment_m: f32,
    /// Hardware acceleration context (Open Source)
    _compute_context: crate::core::index::gpu::ComputeContext,
}

/// Re-rank candidates by exact raw `-dot` (MIPS) using their stored f32 vectors.
///
/// `candidates` carries `(row_id, approximate_distance, raw_vector)`. When the
/// raw vector is available and matches the query dimensionality, the distance
/// is recomputed as `-dot(query, vector)`; otherwise the approximate distance
/// from the graph is kept. The result is sorted ascending (most similar first),
/// de-duplicated by row id, and truncated to `k`.
fn rescore_inner_product(
    query: &[f32],
    candidates: Vec<(usize, f32, Option<Vec<f32>>)>,
    k: usize,
) -> Vec<(usize, f32)> {
    let mut scored: Vec<(usize, f32)> = candidates
        .into_iter()
        .map(|(row_id, approx, raw)| match raw {
            Some(v) if v.len() == query.len() => (
                row_id,
                -crate::core::index::distance::dot_product(query, &v),
            ),
            _ => (row_id, approx),
        })
        .collect();
    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    scored.dedup_by_key(|x| x.0);
    scored.truncate(k);
    scored
}

/// CPU fallback for batched k-means assignment (flat `vectors` buffer, same
/// contract as `gpu::compute_kmeans_assignment`), used when the GPU helper is
/// unavailable/fails or the metric is not L2.
fn cpu_assignments(
    vectors: &[f32],
    centroids: &[Vec<f32>],
    dim: usize,
    metric: VectorMetric,
) -> Vec<u32> {
    (0..vectors.len() / dim)
        .into_par_iter()
        .map(|i| find_closest_centroid(&vectors[i * dim..(i + 1) * dim], centroids, metric) as u32)
        .collect()
}

/// Assign each vector to its nearest centroid, returning one label per vector.
///
/// Used by [`HnswIvfIndex::build_with_centroids`] when the caller supplies a
/// table-level centroid set instead of letting each segment train its own
/// k-means. Prefers the GPU batched kernel for L2 and falls back to the CPU
/// path, which honours every metric.
pub(crate) fn assign_to_centroids(
    vectors: &[Vec<f32>],
    centroids: &[Vec<f32>],
    metric: VectorMetric,
) -> Result<Vec<usize>> {
    if vectors.is_empty() {
        return Ok(Vec::new());
    }
    let dim = vectors[0].len();
    if let Some(bad) = centroids.iter().find(|c| c.len() != dim) {
        anyhow::bail!(
            "centroid dimension mismatch: vectors are {}-d but a centroid is {}-d",
            dim,
            bad.len()
        );
    }

    let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();

    // The GPU kernel implements L2 only; other metrics must go through the CPU
    // path or they would silently rank by the wrong distance.
    if metric == VectorMetric::L2 {
        let centroids_flat: Vec<f32> = centroids.iter().flat_map(|c| c.iter().copied()).collect();
        if let Ok(labels) =
            crate::core::index::gpu::compute_kmeans_assignment(&flat, &centroids_flat, dim)
        {
            return Ok(labels.into_iter().map(|l| l as usize).collect());
        }
    }

    Ok(cpu_assignments(&flat, centroids, dim, metric)
        .into_iter()
        .map(|l| l as usize)
        .collect())
}

fn find_closest_centroid(vec: &[f32], centroids: &[Vec<f32>], metric: VectorMetric) -> usize {
    let mut best_dist = f32::MAX;
    let mut best_cluster = 0;
    for (i, c) in centroids.iter().enumerate() {
        let dist = match metric {
            VectorMetric::L2 => crate::core::index::distance::l2_distance_squared(vec, c),
            // Cosine must route by cosine distance, not L2: the engine does not
            // L2-normalize on insert, so L2 and cosine rank differently for
            // unnormalized vectors (the HNSW graph itself uses true cosine).
            VectorMetric::Cosine => crate::core::index::distance::cosine_distance(vec, c),
            VectorMetric::InnerProduct => -crate::core::index::distance::dot_product(vec, c),
            VectorMetric::L1 => crate::core::index::distance::l1_distance(vec, c),
            VectorMetric::Hamming => crate::core::index::distance::hamming_distance(vec, c),
            VectorMetric::Jaccard => crate::core::index::distance::jaccard_distance(vec, c),
        };
        if dist < best_dist {
            best_dist = dist;
            best_cluster = i;
        }
    }
    best_cluster
}

/// Magic marker for the out-of-core vector temp file written by
/// `HybridSegmentWriter::build_vector_index` and read by
/// [`HnswIvfIndex::build_from_file`].
///
/// Layout: `[magic u32][dim u32]` followed by repeated
/// `[row_id u32][dim u32][f32 * dim]` records. The vector count is derived from
/// the file length, so appending more batches stays correct without rewriting
/// the header.
pub const VEC_TMP_MAGIC: u32 = 0x4853_5631; // "HSV1"

/// Build the HNSW graph for a single IVF bucket from its on-disk cluster file.
///
/// Extracted from [`HnswIvfIndex::build_from_file`] so buckets can be built in
/// parallel (they are independent). Returns `Ok(None)` for an empty bucket.
#[allow(clippy::too_many_arguments)]
fn build_bucket_graph(
    c: usize,
    c_path: &str,
    dim: usize,
    hnsw_m: usize,
    max_layers: usize,
    ef_construction: usize,
    metric: VectorMetric,
    quantizer: Option<&QuantizerImpl>,
    augment_m: f32,
) -> Result<Option<(usize, (HnswGraph, Vec<usize>))>> {
    use std::io::Read;
    let mut f = std::fs::File::open(c_path)?;
    let mut c_vectors: Vec<Vec<f32>> = Vec::new();
    let mut c_row_ids: Vec<usize> = Vec::new();
    let mut id_buf = [0u8; 4];
    loop {
        if f.read_exact(&mut id_buf).is_err() {
            break;
        }
        let mut vec_f32 = vec![0f32; dim];
        f.read_exact(bytemuck::cast_slice_mut::<f32, u8>(&mut vec_f32))?;
        c_row_ids.push(u32::from_le_bytes(id_buf) as usize);
        c_vectors.push(vec_f32);
    }
    let _ = std::fs::remove_file(c_path);

    if c_vectors.is_empty() {
        return Ok(None);
    }

    let mut hnsw = if let Some(q) = quantizer {
        match q {
            QuantizerImpl::BenoQuant(tq) => {
                if tq.bits() == 4 {
                    HnswGraph::TurboQuant4(Hnsw::new(
                        hnsw_m,
                        c_vectors.len(),
                        max_layers,
                        ef_construction,
                        crate::core::index::distance::DistL2u4,
                    ))
                } else {
                    HnswGraph::TurboQuant8(Hnsw::new(
                        hnsw_m,
                        c_vectors.len(),
                        max_layers,
                        ef_construction,
                        crate::core::index::distance::DistL2u8,
                    ))
                }
            }
            QuantizerImpl::Sq(q) => HnswGraph::Sq8(Hnsw::new(
                hnsw_m,
                c_vectors.len(),
                max_layers,
                ef_construction,
                crate::core::index::sq::DistSq8 {
                    encoder: std::sync::Arc::new(q.clone()),
                },
            )),
            QuantizerImpl::Pq(pq) => {
                let dist = crate::core::index::pq::DistPqSdc {
                    pq: std::sync::Arc::new(pq.clone()),
                };
                HnswGraph::Pq(Hnsw::new(
                    hnsw_m,
                    c_vectors.len(),
                    max_layers,
                    ef_construction,
                    dist,
                ))
            }
        }
    } else {
        match metric {
            VectorMetric::L2 => HnswGraph::L2(Hnsw::new(
                hnsw_m,
                c_vectors.len(),
                max_layers,
                ef_construction,
                DistL2,
            )),
            VectorMetric::Cosine => HnswGraph::Cosine(Hnsw::new(
                hnsw_m,
                c_vectors.len(),
                max_layers,
                ef_construction,
                DistCosine,
            )),
            VectorMetric::InnerProduct => HnswGraph::Dot(Hnsw::new(
                hnsw_m,
                c_vectors.len(),
                max_layers,
                ef_construction,
                if augment_m > 0.0 {
                    DistDot::augmented(augment_m)
                } else {
                    DistDot::norm()
                },
            )),
            VectorMetric::L1 => HnswGraph::L1(Hnsw::new(
                hnsw_m,
                c_vectors.len(),
                max_layers,
                ef_construction,
                DistL1,
            )),
            VectorMetric::Hamming => HnswGraph::Hamming(Hnsw::new(
                hnsw_m,
                c_vectors.len(),
                max_layers,
                ef_construction,
                DistHamming,
            )),
            VectorMetric::Jaccard => HnswGraph::Jaccard(Hnsw::new(
                hnsw_m,
                c_vectors.len(),
                max_layers,
                ef_construction,
                DistJaccard,
            )),
        }
    };

    hnsw.set_extend_candidates(true);
    hnsw.set_keeping_pruned(true);

    if let Some(q) = quantizer {
        let encoded = q.encode_batch(&c_vectors);
        match &mut hnsw {
            HnswGraph::TurboQuant8(g) => {
                for (i, v) in encoded.iter().enumerate() {
                    g.insert((v, i));
                }
            }
            HnswGraph::Sq8(g) => {
                for (i, v) in encoded.iter().enumerate() {
                    g.insert((v, i));
                }
            }
            HnswGraph::TurboQuant4(g) => {
                for (i, v) in encoded.iter().enumerate() {
                    g.insert((v, i));
                }
            }
            HnswGraph::Pq(g) => {
                for (i, v) in encoded.iter().enumerate() {
                    g.insert((v, i));
                }
            }
            _ => {}
        }
    } else {
        match &mut hnsw {
            HnswGraph::L2(g) => {
                for (i, v) in c_vectors.iter().enumerate() {
                    g.insert((v, i));
                }
            }
            HnswGraph::Cosine(g) => {
                for (i, v) in c_vectors.iter().enumerate() {
                    g.insert((v, i));
                }
            }
            HnswGraph::Dot(g) => {
                for (i, v) in c_vectors.iter().enumerate() {
                    g.insert((v, i));
                }
            }
            HnswGraph::L1(g) => {
                for (i, v) in c_vectors.iter().enumerate() {
                    g.insert((v, i));
                }
            }
            HnswGraph::Hamming(g) => {
                for (i, v) in c_vectors.iter().enumerate() {
                    g.insert((v, i));
                }
            }
            HnswGraph::Jaccard(g) => {
                for (i, v) in c_vectors.iter().enumerate() {
                    g.insert((v, i));
                }
            }
            _ => {}
        }
    }

    Ok(Some((c, (hnsw, c_row_ids))))
}

impl HnswIvfIndex {
    /// Build HNSW-IVF index from an out-of-core temporary vector file and stream
    /// each cluster graph and mapping file directly to disk as Rayon finishes it.
    ///
    /// This avoids collecting hundreds of bucket graphs into memory simultaneously
    /// (which peaks at 5-10 GB of RAM for a 5M segment), dropping peak memory
    /// during index build to strictly bounded worker-thread working set (~80 MB).
    pub fn build_and_save_from_file(
        tmp_path: &str,
        output_base_path: &str,
        metric: VectorMetric,
        n_lists: Option<usize>,
        hnsw_m: Option<usize>,
        algo: &IndexAlgorithm,
    ) -> Result<Vec<String>> {
        Self::build_and_save_from_file_with_seed(
            tmp_path,
            output_base_path,
            metric,
            n_lists,
            hnsw_m,
            algo,
            None,
        )
    }

    /// Build HNSW-IVF index and stream serialized artifacts directly to disk with an optional PRNG seed.
    pub fn build_and_save_from_file_with_seed(
        tmp_path: &str,
        output_base_path: &str,
        metric: VectorMetric,
        n_lists: Option<usize>,
        hnsw_m: Option<usize>,
        algo: &IndexAlgorithm,
        seed: Option<u64>,
    ) -> Result<Vec<String>> {
        Self::build_and_save_from_file_inner(
            tmp_path,
            output_base_path,
            metric,
            n_lists,
            hnsw_m,
            algo,
            seed,
            None,
        )
    }

    /// Build a segment's index against a caller-supplied (table-level) centroid
    /// set instead of training per-segment k-means.
    ///
    /// This is the Phase-1 validation path: every segment assigns against the
    /// same centroids, so a global cluster id means the same thing everywhere.
    /// The cluster count is `centroids.len()`.
    pub fn build_and_save_from_file_with_centroids(
        tmp_path: &str,
        output_base_path: &str,
        metric: VectorMetric,
        centroids: Vec<Vec<f32>>,
        hnsw_m: Option<usize>,
        algo: &IndexAlgorithm,
    ) -> Result<Vec<String>> {
        if centroids.is_empty() {
            anyhow::bail!("build_and_save_from_file_with_centroids requires centroids");
        }
        let n_lists = Some(centroids.len());
        Self::build_and_save_from_file_inner(
            tmp_path,
            output_base_path,
            metric,
            n_lists,
            hnsw_m,
            algo,
            None,
            Some(centroids),
        )
    }

    fn sample_vectors_from_tmp_file(
        tmp_path: &str,
        max_samples: usize,
    ) -> Result<(usize, usize, Vec<Vec<f32>>)> {
        let mut file = std::fs::File::open(tmp_path)?;
        let file_len = file.metadata()?.len();
        let mut n_vectors = 0usize;
        let mut dim = 0usize;
        let mut vectors_sampled = Vec::new();

        use std::io::{Read, Seek, SeekFrom};
        let mut id_buf = [0u8; 4];
        let mut dim_buf = [0u8; 4];
        let mut magic_buf = [0u8; 4];
        let has_header = file.read_exact(&mut magic_buf).is_ok()
            && u32::from_le_bytes(magic_buf) == VEC_TMP_MAGIC
            && file.read_exact(&mut dim_buf).is_ok();

        if has_header {
            dim = u32::from_le_bytes(dim_buf) as usize;
            let record = 8 + dim * 4;
            n_vectors = if file_len >= 8 && record > 0 {
                ((file_len - 8) / record as u64) as usize
            } else {
                0
            };

            if n_vectors <= max_samples {
                // Sample all vectors sequentially so small and medium corpora
                // (<= 100k vectors) never miss tail elements during k-means or
                // quantizer range estimation.
                vectors_sampled.reserve(n_vectors);
                for _ in 0..n_vectors {
                    if file.read_exact(&mut id_buf).is_err()
                        || file.read_exact(&mut dim_buf).is_err()
                    {
                        break;
                    }
                    let current_dim = u32::from_le_bytes(dim_buf) as usize;
                    let mut vec_f32 = vec![0f32; current_dim];
                    file.read_exact(bytemuck::cast_slice_mut::<f32, u8>(&mut vec_f32))?;
                    vectors_sampled.push(vec_f32);
                }
            } else {
                // Sample **across the whole file**, not its prefix.
                //
                // Read `SAMPLE_BLOCK` consecutive vectors from evenly spaced
                // offsets spanning `[0, n_vectors - SAMPLE_BLOCK]` so head,
                // middle, and tail regions are all represented.
                const SAMPLE_BLOCK: usize = 64;
                let to_sample = max_samples;
                let n_blocks = to_sample.div_ceil(SAMPLE_BLOCK).min(n_vectors);
                let max_start = n_vectors.saturating_sub(SAMPLE_BLOCK);
                let record_u64 = record as u64;

                'sample: for b in 0..n_blocks {
                    let start_idx = if n_blocks <= 1 {
                        0
                    } else {
                        ((b as u64 * max_start as u64) / (n_blocks as u64 - 1)) as usize
                    };
                    file.seek(SeekFrom::Start(8 + start_idx as u64 * record_u64))?;
                    for _ in 0..SAMPLE_BLOCK {
                        if vectors_sampled.len() >= to_sample {
                            break 'sample;
                        }
                        if file.read_exact(&mut id_buf).is_err()
                            || file.read_exact(&mut dim_buf).is_err()
                        {
                            break 'sample;
                        }
                        let current_dim = u32::from_le_bytes(dim_buf) as usize;
                        // f32-ALIGNED buffer: Vec<u8> has alignment 1, so casting it
                        // to &[f32] panics intermittently
                        // ("TargetAlignmentGreaterAndInputNotAligned").
                        let mut vec_f32 = vec![0f32; current_dim];
                        file.read_exact(bytemuck::cast_slice_mut::<f32, u8>(&mut vec_f32))?;
                        vectors_sampled.push(vec_f32);
                    }
                }
            }
        } else {
            // Legacy file without a header: full scan.
            file.seek(SeekFrom::Start(0))?;
            loop {
                if file.read_exact(&mut id_buf).is_err() {
                    break;
                }
                if file.read_exact(&mut dim_buf).is_err() {
                    break;
                }
                let current_dim = u32::from_le_bytes(dim_buf) as usize;
                if dim == 0 {
                    dim = current_dim;
                }
                let mut vec_f32 = vec![0f32; current_dim];
                file.read_exact(bytemuck::cast_slice_mut::<f32, u8>(&mut vec_f32))?;
                n_vectors += 1;
                if vectors_sampled.len() < max_samples {
                    vectors_sampled.push(vec_f32);
                }
            }
        }

        if n_vectors == 0 {
            anyhow::bail!("No vectors found in temp file");
        }

        Ok((n_vectors, dim, vectors_sampled))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_and_save_from_file_inner(
        tmp_path: &str,
        output_base_path: &str,
        metric: VectorMetric,
        n_lists: Option<usize>,
        hnsw_m: Option<usize>,
        algo: &IndexAlgorithm,
        seed: Option<u64>,
        precomputed_centroids: Option<Vec<Vec<f32>>>,
    ) -> Result<Vec<String>> {
        let start = std::time::Instant::now();
        tracing::info!(
            "Building & streaming HNSW-IVF out-of-core from file {} to {}",
            tmp_path,
            output_base_path
        );

        use std::io::{Read, Seek, SeekFrom};
        let mut id_buf = [0u8; 4];
        let mut dim_buf = [0u8; 4];
        let max_samples = 100_000;
        let (n_vectors, dim, vectors_sampled) =
            Self::sample_vectors_from_tmp_file(tmp_path, max_samples)?;

        let num_cpus = num_cpus::get();
        let n_lists = n_lists
            .unwrap_or_else(|| {
                if n_vectors < 1000 {
                    4
                } else if n_vectors < 10_000 {
                    num_cpus.max(16)
                } else {
                    ((n_vectors as f64).sqrt() / 5.0) as usize
                }
            })
            .max(1)
            .min(n_vectors / 10)
            .max(1);

        let hnsw_m = hnsw_m
            .or(match algo {
                IndexAlgorithm::Hnsw { complexity, .. }
                | IndexAlgorithm::HnswTq8 { complexity, .. }
                | IndexAlgorithm::HnswSq8 { complexity, .. }
                | IndexAlgorithm::HnswTq4 { complexity, .. }
                | IndexAlgorithm::HnswPq { complexity, .. } => Some(*complexity),
                _ => None,
            })
            .unwrap_or(16);

        // Caller-supplied centroids bypass per-segment k-means entirely; the
        // cluster count is then fixed by the centroid set, not by `n_lists`.
        let (centroids, n_lists) = match precomputed_centroids {
            Some(centroids) => {
                let n = centroids.len();
                (centroids, n)
            }
            None => {
                let (centroids, _) = simple_kmeans_with_seed(&vectors_sampled, n_lists, 10, seed)?;
                (centroids, n_lists)
            }
        };

        let quantizer = match algo {
            IndexAlgorithm::HnswTq8 { .. } => Some(QuantizerImpl::BenoQuant(
                BenoQuantEncoder::train(&vectors_sampled, 8, metric)?,
            )),
            IndexAlgorithm::HnswSq8 { .. } => Some(QuantizerImpl::Sq(
                crate::core::index::sq::SqEncoder::train(&vectors_sampled, 8, metric)?,
            )),
            IndexAlgorithm::HnswTq4 { .. } => Some(QuantizerImpl::BenoQuant(
                BenoQuantEncoder::train(&vectors_sampled, 4, metric)?,
            )),
            IndexAlgorithm::HnswPq { compression, .. } => {
                let subspaces = dim / compression;
                let config = PqConfig {
                    m: subspaces,
                    k: 256,
                    dim,
                };
                Some(QuantizerImpl::Pq(PqEncoder::train(
                    &vectors_sampled,
                    config,
                )?))
            }
            _ => None,
        };

        // Asymmetric MIPS augmentation: `m` must bound every stored vector's
        // norm. Use the sample max with a small margin (the distance clamps at
        // 0, so an underestimate cannot panic). Only the f32 IP tier uses it;
        // the quantized tiers keep the L2-on-codes path.
        let augment_m = if metric == VectorMetric::InnerProduct && quantizer.is_none() {
            let m = vectors_sampled
                .iter()
                .map(|v| v.iter().map(|x| x * x).sum::<f32>())
                .fold(0.0f32, f32::max)
                .sqrt();
            m * 1.05
        } else {
            0.0
        };

        // 1. Immediately save centroids to disk
        let mut saved_files = Vec::new();
        let centroids_path = format!("{}.centroids.parquet", output_base_path);
        Self::save_centroids_parquet(
            &centroids,
            metric,
            quantizer.as_ref(),
            augment_m,
            &centroids_path,
        )?;
        saved_files.push(centroids_path);

        // Pass 2: Out-of-core bucketing
        let mut cluster_files = HashMap::new();
        for c in 0..n_lists {
            let c_path = format!("{}.cluster_{}.bin", tmp_path, c);
            let f = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&c_path)?;
            cluster_files.insert(c, (c_path, f));
        }

        let mut file = std::fs::File::open(tmp_path)?;
        use std::io::Write;
        {
            let mut m = [0u8; 4];
            if file.read_exact(&mut m).is_ok() && u32::from_le_bytes(m) == VEC_TMP_MAGIC {
                file.seek(SeekFrom::Start(8))?;
            } else {
                file.seek(SeekFrom::Start(0))?;
            }
        }

        let centroids_flat: Vec<f32> = centroids.iter().flatten().copied().collect();
        let use_gpu_assign = matches!(metric, VectorMetric::L2);
        const ASSIGN_BATCH: usize = 65536;
        let mut ids: Vec<u32> = Vec::with_capacity(ASSIGN_BATCH);
        let mut vecs: Vec<f32> = Vec::with_capacity(ASSIGN_BATCH * dim);
        let mut progress = 0usize;

        loop {
            ids.clear();
            vecs.clear();
            let mut eof = false;
            while ids.len() < ASSIGN_BATCH {
                if file.read_exact(&mut id_buf).is_err() {
                    eof = true;
                    break;
                }
                if file.read_exact(&mut dim_buf).is_err() {
                    eof = true;
                    break;
                }
                let mut vec_f32 = vec![0f32; dim];
                file.read_exact(bytemuck::cast_slice_mut::<f32, u8>(&mut vec_f32))?;
                ids.push(u32::from_le_bytes(id_buf));
                vecs.extend_from_slice(&vec_f32);
            }
            if ids.is_empty() {
                break;
            }

            let assignments: Vec<u32> = match use_gpu_assign {
                true => {
                    crate::core::index::gpu::compute_kmeans_assignment(&vecs, &centroids_flat, dim)
                        .unwrap_or_else(|_| cpu_assignments(&vecs, &centroids, dim, metric))
                }
                false => cpu_assignments(&vecs, &centroids, dim, metric),
            };

            for (k, &c) in assignments.iter().enumerate() {
                if let Some((_, ref mut f)) = cluster_files.get_mut(&(c as usize)) {
                    f.write_all(&ids[k].to_le_bytes())?;
                    f.write_all(bytemuck::cast_slice(&vecs[k * dim..(k + 1) * dim]))?;
                }
            }
            progress += ids.len();
            if progress / 100_000 != (progress - ids.len()) / 100_000 {
                tracing::info!("Bucketed {}/{} vectors...", progress, n_vectors);
            }
            if eof {
                break;
            }
        }

        // Pass 3: Build HNSW graphs per bucket in parallel, streaming each directly to disk!
        let ef_construction = match algo {
            IndexAlgorithm::Hnsw { quality, .. }
            | IndexAlgorithm::HnswTq8 { quality, .. }
            | IndexAlgorithm::HnswSq8 { quality, .. }
            | IndexAlgorithm::HnswTq4 { quality, .. }
            | IndexAlgorithm::HnswPq { quality, .. } => *quality,
            _ => (hnsw_m * 2).max(40),
        };
        let max_layers = 16;

        let cluster_list: Vec<(usize, String)> = cluster_files
            .into_iter()
            .map(|(c, (p, _f))| (c, p))
            .collect();

        let quantizer_ref = quantizer.as_ref();
        let base_out = output_base_path.to_string();

        let cluster_files_saved: Vec<Vec<String>> = cluster_list
            .into_par_iter()
            .map(|(c, c_path)| -> Result<Vec<String>> {
                let built = build_bucket_graph(
                    c,
                    &c_path,
                    dim,
                    hnsw_m,
                    max_layers,
                    ef_construction,
                    metric,
                    quantizer_ref,
                    augment_m,
                )?;
                if let Some((cluster_id, (hnsw, row_id_mapping))) = built {
                    let mut per_bucket = Vec::with_capacity(2);
                    let cluster_base = format!("{}.cluster_{}", base_out, cluster_id);
                    let cluster_tmp = format!("{}.tmp", cluster_base);

                    hnsw.file_dump(&cluster_tmp)
                        .map_err(|e| anyhow::anyhow!("HNSW dump failed: {}", e))?;

                    let graph_orig = format!("{}.hnsw.graph", cluster_tmp);
                    let graph_dest = format!("{}.hnsw.graph", cluster_base);

                    if !std::path::Path::new(&graph_orig).exists() {
                        anyhow::bail!("HNSW dump did not create expected file: {}", graph_orig);
                    }

                    std::fs::rename(&graph_orig, &graph_dest)?;
                    per_bucket.push(graph_dest);

                    let mapping_path =
                        format!("{}.cluster_{}.mapping.parquet", base_out, cluster_id);
                    Self::save_mapping_parquet(&row_id_mapping, &mapping_path)?;
                    per_bucket.push(mapping_path);

                    // hnsw and row_id_mapping are dropped right here, freeing RAM!
                    Ok(per_bucket)
                } else {
                    Ok(Vec::new())
                }
            })
            .collect::<Result<Vec<_>>>()?;

        for files in cluster_files_saved {
            saved_files.extend(files);
        }

        tracing::info!(
            "HNSW-IVF streaming build & save complete in {:?}",
            start.elapsed()
        );

        Ok(saved_files)
    }

    pub fn build_from_file(
        tmp_path: &str,
        metric: VectorMetric,
        n_lists: Option<usize>,
        hnsw_m: Option<usize>,
        algo: &IndexAlgorithm,
    ) -> Result<Self> {
        Self::build_from_file_with_seed(tmp_path, metric, n_lists, hnsw_m, algo, None)
    }

    pub fn build_from_file_with_seed(
        tmp_path: &str,
        metric: VectorMetric,
        n_lists: Option<usize>,
        hnsw_m: Option<usize>,
        algo: &IndexAlgorithm,
        seed: Option<u64>,
    ) -> Result<Self> {
        let start = std::time::Instant::now();
        tracing::info!("Building HNSW-IVF out-of-core from file: {}", tmp_path);

        use std::io::{Read, Seek, SeekFrom};
        let mut id_buf = [0u8; 4];
        let mut dim_buf = [0u8; 4];
        let max_samples = 100_000;
        let (n_vectors, dim, vectors_sampled) =
            Self::sample_vectors_from_tmp_file(tmp_path, max_samples)?;

        let num_cpus = num_cpus::get();
        let n_lists = n_lists
            .unwrap_or_else(|| {
                if n_vectors < 1000 {
                    4
                } else if n_vectors < 10_000 {
                    num_cpus.max(16)
                } else {
                    ((n_vectors as f64).sqrt() / 5.0) as usize
                }
            })
            .max(1)
            .min(n_vectors / 10)
            .max(1);

        let hnsw_m = hnsw_m
            .or(match algo {
                IndexAlgorithm::Hnsw { complexity, .. }
                | IndexAlgorithm::HnswTq8 { complexity, .. }
                | IndexAlgorithm::HnswSq8 { complexity, .. }
                | IndexAlgorithm::HnswTq4 { complexity, .. }
                | IndexAlgorithm::HnswPq { complexity, .. } => Some(*complexity),
                _ => None,
            })
            .unwrap_or(16);

        let (centroids, _) = simple_kmeans_with_seed(&vectors_sampled, n_lists, 10, seed)?;

        let quantizer = match algo {
            IndexAlgorithm::HnswTq8 { .. } => Some(QuantizerImpl::BenoQuant(
                BenoQuantEncoder::train(&vectors_sampled, 8, metric)?,
            )),
            IndexAlgorithm::HnswSq8 { .. } => Some(QuantizerImpl::Sq(
                crate::core::index::sq::SqEncoder::train(&vectors_sampled, 8, metric)?,
            )),
            IndexAlgorithm::HnswTq4 { .. } => Some(QuantizerImpl::BenoQuant(
                BenoQuantEncoder::train(&vectors_sampled, 4, metric)?,
            )),
            IndexAlgorithm::HnswPq { compression, .. } => {
                let subspaces = dim / compression;
                let config = PqConfig {
                    m: subspaces,
                    k: 256,
                    dim,
                };
                Some(QuantizerImpl::Pq(PqEncoder::train(
                    &vectors_sampled,
                    config,
                )?))
            }
            _ => None,
        };

        // Pass 2: Out-of-core bucketing
        let mut cluster_files = HashMap::new();
        for c in 0..n_lists {
            let c_path = format!("{}.cluster_{}.bin", tmp_path, c);
            let f = std::fs::OpenOptions::new()
                .create(true)
                .write(true)
                .truncate(true)
                .open(&c_path)?;
            cluster_files.insert(c, (c_path, f));
        }

        let mut file = std::fs::File::open(tmp_path)?;
        use std::io::Write;
        // Skip the self-describing header if present (Pass 1 already consumed
        // it on its own handle; this is a fresh open).
        {
            let mut m = [0u8; 4];
            if file.read_exact(&mut m).is_ok() && u32::from_le_bytes(m) == VEC_TMP_MAGIC {
                file.seek(SeekFrom::Start(8))?;
            } else {
                file.seek(SeekFrom::Start(0))?;
            }
        }

        // Batched bucketing. The previous per-vector loop did a CPU centroid
        // scan AND two write syscalls per vector, which dominated index build
        // time for large segments. Now a batch is assigned through the
        // GPU-dispatched k-means helper (which falls back to CPU when no GPU is
        // available) and written in bulk. Non-L2 metrics keep the CPU path, as
        // the GPU helper implements L2 assignment only.
        let centroids_flat: Vec<f32> = centroids.iter().flatten().copied().collect();
        let use_gpu_assign = matches!(metric, VectorMetric::L2);
        const ASSIGN_BATCH: usize = 65536;
        let mut ids: Vec<u32> = Vec::with_capacity(ASSIGN_BATCH);
        let mut vecs: Vec<f32> = Vec::with_capacity(ASSIGN_BATCH * dim);
        let mut progress = 0usize;

        loop {
            ids.clear();
            vecs.clear();
            let mut eof = false;
            while ids.len() < ASSIGN_BATCH {
                if file.read_exact(&mut id_buf).is_err() {
                    eof = true;
                    break;
                }
                if file.read_exact(&mut dim_buf).is_err() {
                    eof = true;
                    break;
                }
                let mut vec_f32 = vec![0f32; dim];
                file.read_exact(bytemuck::cast_slice_mut::<f32, u8>(&mut vec_f32))?;
                ids.push(u32::from_le_bytes(id_buf));
                vecs.extend_from_slice(&vec_f32);
            }
            if ids.is_empty() {
                break;
            }

            let assignments: Vec<u32> = match use_gpu_assign {
                true => {
                    crate::core::index::gpu::compute_kmeans_assignment(&vecs, &centroids_flat, dim)
                        .unwrap_or_else(|_| cpu_assignments(&vecs, &centroids, dim, metric))
                }
                false => cpu_assignments(&vecs, &centroids, dim, metric),
            };

            for (k, &c) in assignments.iter().enumerate() {
                if let Some((_, ref mut f)) = cluster_files.get_mut(&(c as usize)) {
                    f.write_all(&ids[k].to_le_bytes())?;
                    f.write_all(bytemuck::cast_slice(&vecs[k * dim..(k + 1) * dim]))?;
                }
            }
            progress += ids.len();
            if progress / 100_000 != (progress - ids.len()) / 100_000 {
                tracing::info!("Bucketed {}/{} vectors...", progress, n_vectors);
            }
            if eof {
                break;
            }
        }

        // Pass 3: Build HNSW graphs per bucket — in parallel.
        //
        // Buckets are independent and each is bounded in size
        // (~n_vectors / n_lists), so at most `num_threads` buckets are resident
        // at once. This was previously a sequential loop and dominated the
        // whole out-of-core build.
        let ef_construction = match algo {
            IndexAlgorithm::Hnsw { quality, .. }
            | IndexAlgorithm::HnswTq8 { quality, .. }
            | IndexAlgorithm::HnswSq8 { quality, .. }
            | IndexAlgorithm::HnswTq4 { quality, .. }
            | IndexAlgorithm::HnswPq { quality, .. } => *quality,
            _ => (hnsw_m * 2).max(40),
        };
        let max_layers = 16;

        let augment_m = if metric == VectorMetric::InnerProduct && quantizer.is_none() {
            let m = vectors_sampled
                .iter()
                .map(|v| v.iter().map(|x| x * x).sum::<f32>())
                .fold(0.0f32, f32::max)
                .sqrt();
            m * 1.05
        } else {
            0.0
        };

        let cluster_list: Vec<(usize, String)> = cluster_files
            .into_iter()
            .map(|(c, (p, _f))| (c, p))
            .collect();

        let quantizer_ref = quantizer.as_ref();
        let cluster_graphs: HashMap<usize, (HnswGraph, Vec<usize>)> = cluster_list
            .into_par_iter()
            .map(|(c, c_path)| {
                build_bucket_graph(
                    c,
                    &c_path,
                    dim,
                    hnsw_m,
                    max_layers,
                    ef_construction,
                    metric,
                    quantizer_ref,
                    augment_m,
                )
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        tracing::info!(
            "HNSW-IVF out-of-core build complete in {:?}",
            start.elapsed()
        );

        let cluster_graphs = cluster_graphs
            .into_iter()
            .map(|(k, v)| (k, Arc::new(v)))
            .collect();

        Ok(Self {
            centroids,
            metric,
            cluster_graphs,
            cluster_cache: moka::sync::Cache::builder()
                .max_capacity(cluster_cache_capacity())
                .build(),
            lazy_blobs: None,
            mmap_backed: false,
            _n_lists: n_lists,
            dim,
            quantizer,
            rescore_oversample: 4,
            augment_m,
            _compute_context: crate::core::index::gpu::ComputeContext::auto_detect(),
        })
    }

    /// Build HNSW-IVF index from vectors
    pub fn build(
        vectors: Vec<Vec<f32>>,
        metric: VectorMetric,
        n_lists: Option<usize>,
        hnsw_m: Option<usize>,
        algo: &IndexAlgorithm,
        offset: usize,
    ) -> Result<Self> {
        Self::build_with_seed(vectors, metric, n_lists, hnsw_m, algo, offset, None)
    }

    /// Build HNSW-IVF index from vectors with an optional PRNG seed for deterministic clustering
    pub fn build_with_seed(
        vectors: Vec<Vec<f32>>,
        metric: VectorMetric,
        n_lists: Option<usize>,
        hnsw_m: Option<usize>,
        algo: &IndexAlgorithm,
        offset: usize,
        seed: Option<u64>,
    ) -> Result<Self> {
        Self::build_inner(vectors, metric, n_lists, hnsw_m, algo, offset, seed, None)
    }

    /// Build HNSW-IVF index using caller-supplied centroids instead of running
    /// per-segment k-means.
    ///
    /// This is the entry point for table-level (global) clustering: every segment
    /// is assigned against the same centroid set, so a global cluster id means the
    /// same thing in every segment. Without it, each segment trains its own
    /// k-means and a query must probe every segment to cover its true neighbours.
    ///
    /// The cluster count is `centroids.len()`; `n_lists` is not a parameter here
    /// for that reason.
    pub fn build_with_centroids(
        vectors: Vec<Vec<f32>>,
        metric: VectorMetric,
        centroids: Vec<Vec<f32>>,
        hnsw_m: Option<usize>,
        algo: &IndexAlgorithm,
        offset: usize,
        seed: Option<u64>,
    ) -> Result<Self> {
        if centroids.is_empty() {
            anyhow::bail!("build_with_centroids requires at least one centroid");
        }
        let n_lists = Some(centroids.len());
        Self::build_inner(
            vectors,
            metric,
            n_lists,
            hnsw_m,
            algo,
            offset,
            seed,
            Some(centroids),
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_inner(
        vectors: Vec<Vec<f32>>,
        metric: VectorMetric,
        n_lists: Option<usize>,
        hnsw_m: Option<usize>,
        algo: &IndexAlgorithm,
        offset: usize,
        seed: Option<u64>,
        precomputed_centroids: Option<Vec<Vec<f32>>>,
    ) -> Result<Self> {
        let start = std::time::Instant::now();
        if vectors.is_empty() {
            anyhow::bail!("Cannot build HNSW-IVF index from empty vector set");
        }

        let dim = vectors[0].len();
        let n_vectors = vectors.len();

        // Auto-detect optimal cluster count
        let num_cpus = num_cpus::get();
        let n_lists = n_lists
            .unwrap_or_else(|| {
                if n_vectors < 1000 {
                    4
                } else if n_vectors < 10_000 {
                    num_cpus.max(16)
                } else {
                    ((n_vectors as f64).sqrt() / 5.0) as usize
                }
            })
            .max(1)
            .min(n_vectors / 10)
            .max(1);

        let hnsw_m = hnsw_m
            .or(match algo {
                IndexAlgorithm::Hnsw { complexity, .. }
                | IndexAlgorithm::HnswTq8 { complexity, .. }
                | IndexAlgorithm::HnswSq8 { complexity, .. }
                | IndexAlgorithm::HnswTq4 { complexity, .. }
                | IndexAlgorithm::HnswPq { complexity, .. } => Some(*complexity),
                _ => None,
            })
            .unwrap_or(16);

        tracing::info!(
            "Building HNSW-IVF index: {} vectors, {} clusters, algo={:?}",
            n_vectors,
            n_lists,
            algo
        );

        // Step 1: Cluster vectors. With caller-supplied centroids we only assign;
        // otherwise train per-segment k-means as before.
        let t0 = std::time::Instant::now();
        let max_iters = 10;
        let (centroids, labels) = match precomputed_centroids {
            Some(centroids) => {
                let labels = assign_to_centroids(&vectors, &centroids, metric)?;
                (centroids, labels)
            }
            None => simple_kmeans_with_seed(&vectors, n_lists, max_iters, seed)?,
        };

        // Step 2: Train Quantizer if requested
        let quantizer = match algo {
            IndexAlgorithm::HnswTq8 { .. } => Some(QuantizerImpl::BenoQuant(
                BenoQuantEncoder::train(&vectors, 8, metric)?,
            )),
            IndexAlgorithm::HnswSq8 { .. } => Some(QuantizerImpl::Sq(
                crate::core::index::sq::SqEncoder::train(&vectors, 8, metric)?,
            )),
            IndexAlgorithm::HnswTq4 { .. } => Some(QuantizerImpl::BenoQuant(
                BenoQuantEncoder::train(&vectors, 4, metric)?,
            )),
            IndexAlgorithm::HnswPq { compression, .. } => {
                let subspaces = dim / compression;
                let config = PqConfig {
                    m: subspaces,
                    k: 256,
                    dim,
                };
                Some(QuantizerImpl::Pq(PqEncoder::train(&vectors, config)?))
            }
            _ => None,
        };

        // Asymmetric MIPS augmentation bound (f32 InnerProduct only). Computed
        // before `vectors` is consumed by the clustering step below.
        let augment_m = if metric == VectorMetric::InnerProduct && quantizer.is_none() {
            let m = vectors
                .iter()
                .map(|v| v.iter().map(|x| x * x).sum::<f32>())
                .fold(0.0f32, f32::max)
                .sqrt();
            m * 1.05
        } else {
            0.0
        };

        tracing::debug!(
            "  - K-Means took: {:.2?} ({} iterations, {} clusters)",
            t0.elapsed(),
            max_iters,
            n_lists
        );

        // Step 2: Group vectors by cluster in parallel
        let t1 = std::time::Instant::now();
        let cluster_vectors: HashMap<usize, Vec<(Vec<f32>, usize)>> = vectors
            .into_par_iter()
            .zip(labels.into_par_iter())
            .enumerate()
            .fold(
                HashMap::<usize, Vec<(Vec<f32>, usize)>>::new,
                |mut acc, (row_id, (vec, cluster_id))| {
                    acc.entry(cluster_id)
                        .or_default()
                        .push((vec, row_id + offset));
                    acc
                },
            )
            .reduce(HashMap::new, |mut a, b| {
                for (k, v) in b {
                    a.entry(k).or_default().extend(v);
                }
                a
            });
        tracing::debug!("  - Grouping vectors took: {:.2?}", t1.elapsed());

        // Step 3: Build HNSW graph for each cluster in parallel
        let t2 = std::time::Instant::now();

        let q_ref = quantizer.as_ref();

        let cluster_graphs: HashMap<usize, (HnswGraph, Vec<usize>)> = cluster_vectors
            .into_par_iter()
            .map(|(cluster_id, vecs): (usize, Vec<(Vec<f32>, usize)>)| {
                let max_layers = 16;
                let ef_construction = match algo {
                    IndexAlgorithm::Hnsw { quality, .. }
                    | IndexAlgorithm::HnswTq8 { quality, .. }
                    | IndexAlgorithm::HnswSq8 { quality, .. }
                    | IndexAlgorithm::HnswTq4 { quality, .. }
                    | IndexAlgorithm::HnswPq { quality, .. } => *quality,
                    _ => (hnsw_m * 2).max(40),
                };

                // Determine HNSW node type and distance metric
                let mut hnsw = if let Some(q) = quantizer.as_ref() {
                    match q {
                        QuantizerImpl::BenoQuant(tq) => {
                            if tq.bits() == 4 {
                                HnswGraph::TurboQuant4(Hnsw::new(
                                    hnsw_m,
                                    vecs.len(),
                                    max_layers,
                                    ef_construction,
                                    crate::core::index::distance::DistL2u4,
                                ))
                            } else {
                                HnswGraph::TurboQuant8(Hnsw::new(
                                    hnsw_m,
                                    vecs.len(),
                                    max_layers,
                                    ef_construction,
                                    crate::core::index::distance::DistL2u8,
                                ))
                            }
                        }
                        QuantizerImpl::Sq(q) => HnswGraph::Sq8(Hnsw::new(
                            hnsw_m,
                            vecs.len(),
                            max_layers,
                            ef_construction,
                            crate::core::index::sq::DistSq8 {
                                encoder: std::sync::Arc::new(q.clone()),
                            },
                        )),
                        QuantizerImpl::Pq(pq) => {
                            let dist = crate::core::index::pq::DistPqSdc {
                                pq: std::sync::Arc::new(pq.clone()),
                            };
                            HnswGraph::Pq(Hnsw::new(
                                hnsw_m,
                                vecs.len(),
                                max_layers,
                                ef_construction,
                                dist,
                            ))
                        }
                    }
                } else {
                    match metric {
                        VectorMetric::L2 => HnswGraph::L2(Hnsw::new(
                            hnsw_m,
                            vecs.len(),
                            max_layers,
                            ef_construction,
                            DistL2,
                        )),
                        VectorMetric::Cosine => HnswGraph::Cosine(Hnsw::new(
                            hnsw_m,
                            vecs.len(),
                            max_layers,
                            ef_construction,
                            DistCosine,
                        )),
                        VectorMetric::InnerProduct => HnswGraph::Dot(Hnsw::new(
                            hnsw_m,
                            vecs.len(),
                            max_layers,
                            ef_construction,
                            if augment_m > 0.0 {
                                DistDot::augmented(augment_m)
                            } else {
                                DistDot::norm()
                            },
                        )),
                        VectorMetric::L1 => HnswGraph::L1(Hnsw::new(
                            hnsw_m,
                            vecs.len(),
                            max_layers,
                            ef_construction,
                            DistL1,
                        )),
                        VectorMetric::Hamming => HnswGraph::Hamming(Hnsw::new(
                            hnsw_m,
                            vecs.len(),
                            max_layers,
                            ef_construction,
                            DistHamming,
                        )),
                        VectorMetric::Jaccard => HnswGraph::Jaccard(Hnsw::new(
                            hnsw_m,
                            vecs.len(),
                            max_layers,
                            ef_construction,
                            DistJaccard,
                        )),
                    }
                };

                hnsw.set_extend_candidates(true);
                hnsw.set_keeping_pruned(true);

                // Build locally sequential
                for (local_id, (vec, _)) in vecs.iter().enumerate() {
                    if let Some(q) = q_ref {
                        let encoded = q.encode(vec);
                        hnsw.insert((crate::core::index::VectorValue::Binary(encoded), local_id));
                    } else {
                        hnsw.insert((
                            crate::core::index::VectorValue::Float32(vec.clone()),
                            local_id,
                        ));
                    }
                }

                let row_id_mapping: Vec<usize> = vecs.iter().map(|(_, row_id)| *row_id).collect();
                (cluster_id, (hnsw, row_id_mapping))
            })
            .collect();

        tracing::debug!(
            "  - Building HNSW graphs took: {:.2?} ({} clusters)",
            t2.elapsed(),
            cluster_graphs.len()
        );

        tracing::info!(
            "Hnsw-IVF index built in {:.2?}: {} non-empty clusters",
            start.elapsed(),
            cluster_graphs.len()
        );

        let cluster_graphs = cluster_graphs
            .into_iter()
            .map(|(k, v)| (k, Arc::new(v)))
            .collect();

        Ok(HnswIvfIndex {
            centroids,
            metric,
            cluster_graphs,
            cluster_cache: moka::sync::Cache::builder()
                .max_capacity(cluster_cache_capacity())
                .build(),
            lazy_blobs: None,
            mmap_backed: false,
            _n_lists: n_lists,
            dim,
            quantizer,
            rescore_oversample: 4,
            augment_m,
            _compute_context: crate::core::index::gpu::ComputeContext::auto_detect(),
        })
    }

    /// Retrieve cluster HNSW graph and row id mapping by cluster id.
    ///
    /// If the index was eagerly constructed or loaded, returns the cluster from
    /// `cluster_graphs`. If the index was lazily loaded from a Puffin container,
    /// checks `cluster_cache` (LRU) and deserializes only the requested cluster
    /// from the container on demand.
    pub fn get_cluster(&self, cluster_id: usize) -> Option<Arc<(HnswGraph, Vec<usize>)>> {
        if let Some(cg) = self.cluster_graphs.get(&cluster_id) {
            return Some(cg.clone());
        }
        if let Some(hit) = self.cluster_cache.get(&cluster_id) {
            return Some(hit);
        }
        if let Some(ref lazy) = self.lazy_blobs {
            match lazy.load_cluster(cluster_id) {
                Ok(cluster_data) => {
                    let arc_data = Arc::new(cluster_data);
                    self.cluster_cache.insert(cluster_id, arc_data.clone());
                    return Some(arc_data);
                }
                Err(e) => {
                    tracing::warn!(cluster_id, error = %e, "Failed to load lazy cluster");
                    return None;
                }
            }
        }
        None
    }

    pub fn is_lazy(&self) -> bool {
        self.lazy_blobs.is_some()
    }

    pub fn cluster_cache_count(&self) -> usize {
        self.cluster_cache.run_pending_tasks();
        self.cluster_cache.entry_count() as usize
    }

    pub fn deserialize_single_cluster(
        cluster_id: usize,
        graph_bytes: bytes::Bytes,
        mapping_bytes: bytes::Bytes,
        quantizer: Option<&QuantizerImpl>,
        metric: VectorMetric,
        augment_m: f32,
    ) -> Result<(HnswGraph, Vec<usize>)> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let hnsw = if let Some(q_impl) = quantizer {
            match q_impl {
                QuantizerImpl::BenoQuant(q) => {
                    if q.bits() == 4 {
                        HnswGraph::TurboQuant4Arrow(
                            crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes_owned(
                                graph_bytes,
                                crate::core::index::distance::DistL2u4,
                            )
                            .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?,
                        )
                    } else {
                        HnswGraph::TurboQuant8Arrow(
                            crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes_owned(
                                graph_bytes,
                                crate::core::index::distance::DistL2u8,
                            )
                            .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?,
                        )
                    }
                }
                QuantizerImpl::Sq(q) => HnswGraph::Sq8Arrow(
                    crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes_owned(
                        graph_bytes,
                        crate::core::index::sq::DistSq8 {
                            encoder: std::sync::Arc::new(q.clone()),
                        },
                    )
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?,
                ),
                QuantizerImpl::Pq(pq) => {
                    let dist = crate::core::index::pq::DistPqSdc {
                        pq: std::sync::Arc::new(pq.clone()),
                    };
                    HnswGraph::PqArrow(
                        crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes_owned(
                            graph_bytes,
                            dist,
                        )
                        .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?,
                    )
                }
            }
        } else {
            match metric {
                VectorMetric::L2 => HnswGraph::L2Arrow(
                    crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes_owned(
                        graph_bytes,
                        DistL2,
                    )
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?,
                ),
                VectorMetric::Cosine => HnswGraph::CosineArrow(
                    crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes_owned(
                        graph_bytes,
                        DistCosine,
                    )
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?,
                ),
                VectorMetric::InnerProduct => HnswGraph::DotArrow(
                    crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes_owned(
                        graph_bytes,
                        if augment_m > 0.0 {
                            DistDot::augmented(augment_m)
                        } else {
                            DistDot::norm()
                        },
                    )
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?,
                ),
                VectorMetric::L1 => HnswGraph::L1Arrow(
                    crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes_owned(
                        graph_bytes,
                        DistL1,
                    )
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?,
                ),
                VectorMetric::Hamming => HnswGraph::HammingArrow(
                    crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes_owned(
                        graph_bytes,
                        DistHamming,
                    )
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?,
                ),
                VectorMetric::Jaccard => HnswGraph::JaccardArrow(
                    crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes_owned(
                        graph_bytes,
                        DistJaccard,
                    )
                    .map_err(|e| anyhow::anyhow!("HNSW load failed: {}", e))?,
                ),
            }
        };

        let map_builder = ParquetRecordBatchReaderBuilder::try_new(mapping_bytes)?;
        let map_reader = map_builder.build()?;
        let batches: Vec<RecordBatch> = map_reader
            .map(|r| r.map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?;
        if batches.is_empty() {
            return Err(anyhow::anyhow!(
                "No data in mapping file for cluster {}",
                cluster_id
            ));
        }
        let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches)?;
        let row_id_mapping = Self::extract_row_id_mapping_from_col(batch.column(0));

        Ok((hnsw, row_id_mapping))
    }

    pub fn load_lazy_from_container(
        container: bytes::Bytes,
        puffin_path: &str,
        column_name: &str,
        use_mmap: bool,
    ) -> Result<Self> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let entries: Vec<(String, usize, usize)> = {
            let reader =
                crate::core::puffin::PuffinReader::new(std::io::Cursor::new(container.as_ref()))?;
            let mut list = Vec::new();
            for (i, b) in reader.footer().blobs.iter().enumerate() {
                let name = b
                    .properties
                    .get("filename")
                    .cloned()
                    .unwrap_or_else(|| format!("blob_{i}"));
                let offset = usize::try_from(b.offset)
                    .map_err(|_| anyhow::anyhow!("negative blob offset in {}", puffin_path))?;
                let length = usize::try_from(b.length)
                    .map_err(|_| anyhow::anyhow!("negative blob length in {}", puffin_path))?;
                if offset
                    .checked_add(length)
                    .is_none_or(|end| end > container.len())
                {
                    anyhow::bail!(
                        "blob '{}' [{offset}..{offset}+{length}] exceeds container size {}",
                        name,
                        container.len()
                    );
                }
                list.push((name, offset, length));
            }
            list
        };

        let marker = format!(".{column_name}.");
        let (base_path, centroids_offset, centroids_length) = entries
            .iter()
            .find_map(|(name, offset, length)| {
                if name.contains(&marker) && name.ends_with(".centroids.parquet") {
                    let base = name.strip_suffix(".centroids.parquet")?.to_string();
                    Some((base, *offset, *length))
                } else {
                    None
                }
            })
            .or_else(|| {
                entries.iter().find_map(|(name, offset, length)| {
                    if name.ends_with(".centroids.parquet") {
                        let base = name.strip_suffix(".centroids.parquet")?.to_string();
                        Some((base, *offset, *length))
                    } else {
                        None
                    }
                })
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Puffin bundle '{}' has no centroids blob for column '{}'",
                    puffin_path,
                    column_name
                )
            })?;

        let centroids_bytes =
            container.slice(centroids_offset..centroids_offset + centroids_length);
        let builder = ParquetRecordBatchReaderBuilder::try_new(centroids_bytes.clone())?;
        let reader = builder.build()?;
        let batches: Vec<RecordBatch> = reader
            .map(|r| r.map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?;
        if batches.is_empty() {
            anyhow::bail!("No data in centroids file");
        }
        let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches)?;
        let centroids = Self::extract_centroids_from_batch(&batch)?;
        if centroids.is_empty() {
            anyhow::bail!("No centroids extracted from centroids file");
        }
        let n_lists = centroids.len();
        let dim = centroids[0].len();

        let parquet_reader = parquet::file::reader::SerializedFileReader::new(centroids_bytes)?;
        let file_metadata = parquet_reader.metadata().file_metadata();
        let mut metric = VectorMetric::L2;
        let mut quantizer = None;
        let mut augment_m = 0.0f32;
        if let Some(kv_list) = file_metadata.key_value_metadata() {
            for kv in kv_list {
                if kv.key == "vector_metric" {
                    if let Some(ref val) = kv.value {
                        metric = val.parse::<VectorMetric>().unwrap_or(VectorMetric::L2);
                    }
                } else if kv.key == "quantizer_config" {
                    if let Some(ref val) = kv.value {
                        if let Ok(mut q) = serde_json::from_str::<QuantizerImpl>(val) {
                            if let QuantizerImpl::Pq(ref mut pq) = q {
                                pq.init_lut();
                            }
                            quantizer = Some(q);
                        }
                    }
                } else if kv.key == "augment_m" {
                    if let Some(ref val) = kv.value {
                        augment_m = val.parse::<f32>().unwrap_or(0.0);
                    }
                }
            }
        }

        let mut graph_slices: HashMap<usize, (usize, usize)> = HashMap::new();
        let mut mapping_slices: HashMap<usize, (usize, usize)> = HashMap::new();

        for (name, offset, length) in &entries {
            if !name.starts_with(&base_path) {
                continue;
            }
            if name.ends_with(".hnsw.graph") {
                if let Some(start) = name.rfind(".cluster_") {
                    if let Some(end) = name.rfind(".hnsw.graph") {
                        if start + 9 < end {
                            if let Ok(cid) = name[start + 9..end].parse::<usize>() {
                                graph_slices.insert(cid, (*offset, *length));
                            }
                        }
                    }
                }
            } else if name.ends_with(".mapping.parquet") {
                if let Some(start) = name.rfind(".cluster_") {
                    if let Some(end) = name.rfind(".mapping.parquet") {
                        if start + 9 < end {
                            if let Ok(cid) = name[start + 9..end].parse::<usize>() {
                                mapping_slices.insert(cid, (*offset, *length));
                            }
                        }
                    }
                }
            }
        }

        let mut cluster_slices: HashMap<usize, ((usize, usize), (usize, usize))> = HashMap::new();
        for (cid, g_slice) in graph_slices {
            if let Some(m_slice) = mapping_slices.get(&cid) {
                cluster_slices.insert(cid, (g_slice, *m_slice));
            }
        }

        let lazy_blobs = Arc::new(LazyPuffinBlobs {
            container,
            cluster_slices,
            quantizer: quantizer.clone(),
            metric,
            augment_m,
        });

        let cluster_cache = moka::sync::Cache::builder().max_capacity(64).build();

        Ok(HnswIvfIndex {
            centroids,
            metric,
            cluster_graphs: HashMap::new(),
            cluster_cache,
            lazy_blobs: Some(lazy_blobs),
            mmap_backed: use_mmap,
            _n_lists: n_lists,
            dim,
            quantizer,
            rescore_oversample: 4,
            augment_m,
            _compute_context: crate::core::index::gpu::ComputeContext::auto_detect(),
        })
    }

    /// Cluster centroids for coarse quantization.
    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    /// Metric used by this index.
    pub fn metric(&self) -> VectorMetric {
        self.metric
    }

    /// Dimensionality of vectors in this index.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Set the candidate oversampling factor used before exact InnerProduct
    /// rescoring. Values below `1` are clamped to `1`.
    pub fn with_rescore_oversample(mut self, factor: usize) -> Self {
        self.rescore_oversample = factor.max(1);
        self
    }

    /// Search specific targeted clusters directly without running coarse search
    /// across all local centroids. Used when a table-level vector router has already
    /// resolved the global candidate clusters for this segment.
    pub fn search_targeted_clusters(
        &self,
        query: &VectorValue,
        k: usize,
        target_clusters: &[usize],
        filter: Option<&roaring::RoaringBitmap>,
        ef_search: Option<usize>,
    ) -> Result<Vec<(usize, f32)>> {
        let t_start = std::time::Instant::now();
        let query_f32 = match query {
            VectorValue::Float32(v) => v,
            VectorValue::Float16(v) => v,
            VectorValue::Binary(_) => return Ok(vec![]),
            VectorValue::Sparse(_) => return Ok(vec![]),
            VectorValue::Keyword(_) => return Ok(vec![]),
        };

        let prepared_query = self
            .quantizer
            .as_ref()
            .map(|q| q.prepare_query(query_f32, self.metric));
        self.search_clusters_internal(
            query_f32,
            query,
            prepared_query.as_ref(),
            target_clusters,
            k,
            filter,
            ef_search,
            t_start,
        )
    }

    /// Search for k nearest neighbors
    pub fn search(
        &self,
        query: &VectorValue,
        k: usize,
        n_probe: usize,
        filter: Option<&roaring::RoaringBitmap>,
        ef_search: Option<usize>,
    ) -> Result<Vec<(usize, f32)>> {
        let t_start = std::time::Instant::now();
        let query_f32 = match query {
            VectorValue::Float32(v) => v,
            VectorValue::Float16(v) => v,
            VectorValue::Binary(_) => return Ok(vec![]),
            VectorValue::Sparse(_) => return Ok(vec![]),
            VectorValue::Keyword(_) => return Ok(vec![]),
        };

        let prepared_query = self
            .quantizer
            .as_ref()
            .map(|q| q.prepare_query(query_f32, self.metric));

        // Step 1: Find n_probe nearest clusters (coarse search)
        // (Centroids are always f32 for stability)
        let all_centroids_flat: Vec<f32> = self.centroids.iter().flatten().cloned().collect();

        let distances = crate::core::index::gpu::compute_distance(
            query_f32,
            &all_centroids_flat,
            self.dim,
            self.metric,
        )
        .unwrap_or_else(|_| match self.metric {
            VectorMetric::L2 => {
                crate::core::index::distance::l2_distance_batch(query_f32, &self.centroids)
            }
            VectorMetric::Cosine => self
                .centroids
                .par_iter()
                .map(|c| crate::core::index::distance::cosine_distance(query_f32, c))
                .collect(),
            VectorMetric::InnerProduct => self
                .centroids
                .par_iter()
                .map(|c| -crate::core::index::distance::dot_product(query_f32, c))
                .collect(),
            VectorMetric::L1 => self
                .centroids
                .par_iter()
                .map(|c| crate::core::index::distance::l1_distance(query_f32, c))
                .collect(),
            VectorMetric::Hamming => self
                .centroids
                .par_iter()
                .map(|c| crate::core::index::distance::hamming_distance(query_f32, c))
                .collect(),
            VectorMetric::Jaccard => self
                .centroids
                .par_iter()
                .map(|c| crate::core::index::distance::jaccard_distance(query_f32, c))
                .collect(),
        });

        let mut cluster_distances: Vec<(usize, f32)> = distances.into_iter().enumerate().collect();
        cluster_distances
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let total_clusters = self.centroids.len();
        let effective_n_probe = n_probe.clamp(1, total_clusters);

        let clusters_to_search: Vec<usize> = cluster_distances
            .iter()
            .take(effective_n_probe)
            .map(|(i, _)| *i)
            .collect();
        tracing::debug!("Sorted cluster distances: {:?}", cluster_distances);
        tracing::debug!("Clusters to search: {:?}", clusters_to_search);

        self.search_clusters_internal(
            query_f32,
            query,
            prepared_query.as_ref(),
            &clusters_to_search,
            k,
            filter,
            ef_search,
            t_start,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn search_clusters_internal(
        &self,
        query_f32: &[f32],
        query_fallback: &VectorValue,
        prepared_query: Option<&crate::core::index::PreparedQuantizedQuery<'_>>,
        clusters_to_search: &[usize],
        k: usize,
        filter: Option<&roaring::RoaringBitmap>,
        ef_search: Option<usize>,
        t_start: std::time::Instant,
    ) -> Result<Vec<(usize, f32)>> {
        let t_coarse = t_start.elapsed();
        let t_fine_start = std::time::Instant::now();

        // Step 2: Search HNSW graphs in selected clusters in parallel (fine search)
        let ef = ef_search.unwrap_or(if filter.is_some() { 100 } else { 40 });
        let base_cluster_k = (k * 4).max(ef.min(k * 8));
        // For InnerProduct the graph ranks by its own safe, non-negative distance
        // (cosine for the `Dot` variants). Widen the candidate pool so the exact
        // raw `-dot` rescore has enough candidates to re-rank.
        // Escape hatch for A/B benchmarking: set BSDB_DISABLE_IP_RESCORE=1 to
        // fall back to the graph's own distance ordering for InnerProduct.
        let rescore = self.metric == VectorMetric::InnerProduct
            && std::env::var_os("BSDB_DISABLE_IP_RESCORE").is_none();
        let cluster_k = if rescore {
            base_cluster_k.max(k.saturating_mul(self.rescore_oversample))
        } else {
            base_cluster_k
        };

        let candidates: Vec<(usize, f32, Option<Vec<f32>>)> = clusters_to_search
            .into_par_iter()
            .flat_map_iter(|&cluster_id| {
                if let Some(cluster_data) = self.get_cluster(cluster_id) {
                    let (hnsw, row_id_mapping) = cluster_data.as_ref();
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

                    // For smaller clusters, scale ef proportionally to avoid excess priority queue churn
                    let cluster_ef = ef.min((row_id_mapping.len() * 3) / 4).max(cluster_k);

                    let results = hnsw.search_with_quantizer(
                        query_fallback,
                        prepared_query,
                        cluster_k,
                        cluster_ef,
                        local_filter_opt.as_ref(),
                    );

                    // Map local indices back to global row IDs and apply filter.
                    // For InnerProduct, also fetch the stored vector so the
                    // candidate can be re-ranked by exact raw `-dot`.
                    results
                        .into_iter()
                        .filter_map(|neighbor| {
                            let local_id = neighbor.d_id;
                            if local_id >= row_id_mapping.len() {
                                return None;
                            }
                            let global_id = row_id_mapping[local_id];
                            if let Some(f) = filter {
                                if !f.contains(global_id as u32) {
                                    return None;
                                }
                            }
                            let raw = if rescore {
                                hnsw.get_vector_for_rescore(&neighbor, self.quantizer.as_ref())
                            } else {
                                None
                            };
                            Some((global_id, neighbor.distance, raw))
                        })
                        .collect::<Vec<_>>()
                } else {
                    vec![]
                }
            })
            .collect();

        // Step 3: Merge and return top-k. InnerProduct candidates are re-ranked
        // by exact raw `-dot`; every other metric keeps the graph distance.
        tracing::debug!("Candidates before sort/dedup: len={}", candidates.len());
        let candidates = if rescore {
            rescore_inner_product(query_f32, candidates, k)
        } else {
            let mut merged: Vec<(usize, f32)> = candidates
                .into_iter()
                .map(|(row_id, distance, _)| (row_id, distance))
                .collect();
            merged.sort_by(|a, b| {
                match a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal) {
                    std::cmp::Ordering::Equal => a.0.cmp(&b.0),
                    other => other,
                }
            });
            merged.dedup_by_key(|x| x.0); // Remove duplicates
            merged.truncate(k);
            merged
        };
        tracing::debug!("Candidates after dedup: len={}", candidates.len());
        #[cfg(feature = "trace-search")]
        {
            let loaded_count = if !self.cluster_graphs.is_empty() {
                self.cluster_graphs.len()
            } else {
                self.cluster_cache.entry_count() as usize
            };
            eprintln!(
                "[trace] hnsw_ivf.search centroids={} clusters_active={} candidates={}",
                self.centroids.len(),
                loaded_count,
                candidates.len(),
            );
        }
        let t_fine = t_fine_start.elapsed();
        tracing::debug!(
            "Search Profile -> Coarse: {:?}, Fine: {:?}",
            t_coarse,
            t_fine
        );
        Ok(candidates)
    }

    /// Helper to serialize centroids directly to a Parquet file.
    pub fn save_centroids_parquet(
        centroids: &[Vec<f32>],
        metric: VectorMetric,
        quantizer: Option<&QuantizerImpl>,
        augment_m: f32,
        target_path: &str,
    ) -> Result<String> {
        use arrow::array::{Float32Builder, ListBuilder};
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::fs::File;
        use std::sync::Arc;

        let centroids_tmp = format!("{}.tmp", target_path);
        let mut centroid_builder = ListBuilder::new(Float32Builder::new());
        for centroid in centroids {
            for &val in centroid {
                centroid_builder.values().append_value(val);
            }
            centroid_builder.append(true);
        }
        let centroid_array = Arc::new(centroid_builder.finish());

        let schema = Arc::new(Schema::new(vec![Field::new(
            "centroid",
            DataType::List(Arc::new(Field::new("item", DataType::Float32, true))),
            false,
        )]));

        let batch = RecordBatch::try_new(schema.clone(), vec![centroid_array])?;

        let file = File::create(&centroids_tmp)?;
        let props = WriterProperties::builder()
            .set_key_value_metadata(Some(vec![
                parquet::file::metadata::KeyValue {
                    key: "vector_metric".to_string(),
                    value: Some(match metric {
                        VectorMetric::L2 => "l2".to_string(),
                        VectorMetric::Cosine => "cosine".to_string(),
                        VectorMetric::InnerProduct => "ip".to_string(),
                        VectorMetric::L1 => "l1".to_string(),
                        VectorMetric::Hamming => "hamming".to_string(),
                        VectorMetric::Jaccard => "jaccard".to_string(),
                    }),
                },
                parquet::file::metadata::KeyValue {
                    key: "quantizer_config".to_string(),
                    value: quantizer.map(|q| serde_json::to_string(q).unwrap_or_default()),
                },
                parquet::file::metadata::KeyValue {
                    key: "augment_m".to_string(),
                    value: Some(format!("{augment_m}")),
                },
            ]))
            .build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        std::fs::rename(&centroids_tmp, target_path)?;
        Ok(target_path.to_string())
    }

    /// Helper to serialize cluster row-id mapping directly to a Parquet file.
    pub fn save_mapping_parquet(row_id_mapping: &[usize], target_path: &str) -> Result<String> {
        use arrow::array::UInt32Array;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatch;
        use parquet::arrow::ArrowWriter;
        use parquet::file::properties::WriterProperties;
        use std::fs::File;
        use std::sync::Arc;

        let mapping_tmp = format!("{}.tmp", target_path);
        let row_ids: Vec<u32> = row_id_mapping.iter().map(|&x| x as u32).collect();
        let row_id_array = Arc::new(UInt32Array::from(row_ids));

        let schema = Arc::new(Schema::new(vec![Field::new(
            "row_id",
            DataType::UInt32,
            false,
        )]));

        let batch = RecordBatch::try_new(schema.clone(), vec![row_id_array])?;

        let file = File::create(&mapping_tmp)?;
        let props = WriterProperties::builder().build();
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        writer.write(&batch)?;
        writer.close()?;

        std::fs::rename(&mapping_tmp, target_path)?;
        Ok(target_path.to_string())
    }

    /// Save HNSW-IVF index to disk
    pub fn save(&self, local_path: &str) -> Result<Vec<String>> {
        let mut saved_files = Vec::new();

        // 1. Save centroids
        let centroids_path = format!("{}.centroids.parquet", local_path);
        Self::save_centroids_parquet(
            &self.centroids,
            self.metric,
            self.quantizer.as_ref(),
            self.augment_m,
            &centroids_path,
        )?;
        saved_files.push(centroids_path);

        // 2. Save per-cluster HNSW graphs
        for (cluster_id, cluster_data) in &self.cluster_graphs {
            let (hnsw, row_id_mapping) = cluster_data.as_ref();
            let cluster_base = format!("{}.cluster_{}", local_path, cluster_id);
            let cluster_tmp = format!("{}.tmp", cluster_base);

            hnsw.file_dump(&cluster_tmp)
                .map_err(|e| anyhow::anyhow!("HNSW dump failed: {}", e))?;

            let graph_orig = format!("{}.hnsw.graph", cluster_tmp);
            let graph_dest = format!("{}.hnsw.graph", cluster_base);

            if !std::path::Path::new(&graph_orig).exists() {
                anyhow::bail!("HNSW dump did not create expected file: {}", graph_orig);
            }

            std::fs::rename(&graph_orig, &graph_dest)?;
            saved_files.push(graph_dest);

            let mapping_path = format!("{}.cluster_{}.mapping.parquet", local_path, cluster_id);
            Self::save_mapping_parquet(row_id_mapping, &mapping_path)?;
            saved_files.push(mapping_path);
        }

        Ok(saved_files)
    }

    fn extract_centroids_from_batch(batch: &RecordBatch) -> Result<Vec<Vec<f32>>> {
        if batch.num_columns() == 0 {
            return Ok(Vec::new());
        }
        let col = batch.column(0);
        let mut centroids = Vec::new();

        if let Some(list_arr) = col.as_any().downcast_ref::<arrow::array::ListArray>() {
            for i in 0..list_arr.len() {
                let values = list_arr.value(i);
                if let Some(f32_arr) = values.as_any().downcast_ref::<arrow::array::Float32Array>()
                {
                    centroids.push(f32_arr.values().to_vec());
                } else if let Some(f64_arr) =
                    values.as_any().downcast_ref::<arrow::array::Float64Array>()
                {
                    centroids.push(f64_arr.values().iter().map(|&v| v as f32).collect());
                }
            }
        } else if let Some(list_arr) = col.as_any().downcast_ref::<arrow::array::LargeListArray>() {
            for i in 0..list_arr.len() {
                let values = list_arr.value(i);
                if let Some(f32_arr) = values.as_any().downcast_ref::<arrow::array::Float32Array>()
                {
                    centroids.push(f32_arr.values().to_vec());
                } else if let Some(f64_arr) =
                    values.as_any().downcast_ref::<arrow::array::Float64Array>()
                {
                    centroids.push(f64_arr.values().iter().map(|&v| v as f32).collect());
                }
            }
        } else if let Some(flist_arr) = col
            .as_any()
            .downcast_ref::<arrow::array::FixedSizeListArray>()
        {
            for i in 0..flist_arr.len() {
                let values = flist_arr.value(i);
                if let Some(f32_arr) = values.as_any().downcast_ref::<arrow::array::Float32Array>()
                {
                    centroids.push(f32_arr.values().to_vec());
                } else if let Some(f64_arr) =
                    values.as_any().downcast_ref::<arrow::array::Float64Array>()
                {
                    centroids.push(f64_arr.values().iter().map(|&v| v as f32).collect());
                }
            }
        }
        Ok(centroids)
    }

    fn extract_row_id_mapping_from_col(col: &ArrayRef) -> Vec<usize> {
        if let Some(arr) = col.as_any().downcast_ref::<arrow::array::UInt32Array>() {
            (0..arr.len()).map(|i| arr.value(i) as usize).collect()
        } else if let Some(arr) = col.as_any().downcast_ref::<arrow::array::UInt64Array>() {
            (0..arr.len()).map(|i| arr.value(i) as usize).collect()
        } else if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int64Array>() {
            (0..arr.len()).map(|i| arr.value(i) as usize).collect()
        } else if let Some(arr) = col.as_any().downcast_ref::<arrow::array::Int32Array>() {
            (0..arr.len()).map(|i| arr.value(i) as usize).collect()
        } else {
            Vec::new()
        }
    }

    pub async fn load_async(
        store: Arc<dyn ObjectStore>,
        base_path: &str,
        use_mmap: bool,
    ) -> Result<Arc<Self>> {
        Self::load_async_with_cache_key(store, base_path, base_path, use_mmap).await
    }

    pub async fn load_async_with_cache_key(
        store: Arc<dyn ObjectStore>,
        base_path: &str,
        cache_key: &str,
        use_mmap: bool,
    ) -> Result<Arc<Self>> {
        let disk_cache = crate::core::cache::DiskCache::new(store.clone());
        Self::load_async_inner(store, base_path, cache_key, use_mmap, disk_cache).await
    }

    /// Loader body, taking the blob cache explicitly.
    ///
    /// In-memory blob containers (a Puffin bundle sliced into `Bytes`, see
    /// [`HnswIvfIndex::load_from_puffin`]) must not be mirrored into the
    /// on-disk cache, so the caller passes [`DiskCache::disabled`].
    pub(crate) async fn load_async_inner(
        store: Arc<dyn ObjectStore>,
        base_path: &str,
        cache_key: &str,
        use_mmap: bool,
        disk_cache: crate::core::cache::DiskCache,
    ) -> Result<Arc<Self>> {
        use crate::core::cache::HNSW_IVF_CACHE;
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

        let cache_key_str = cache_key.to_string();
        if let Some(cached) = HNSW_IVF_CACHE
            .get_with_metrics(&cache_key_str, "hnsw_ivf")
            .await
        {
            return Ok(cached);
        }

        let mut root_path = if base_path.contains("://") {
            if let Ok(url) = url::Url::parse(base_path) {
                url.path().trim_start_matches('/').to_string()
            } else {
                base_path.to_string()
            }
        } else {
            base_path.to_string()
        };

        // Robustness: If the path points to a specific cluster shard or graph file, strip it to get the base
        if let Some(idx) = root_path.find(".cluster_") {
            root_path = root_path[..idx].to_string();
        } else if let Some(idx) = root_path.find(".hnsw.graph") {
            root_path = root_path[..idx].to_string();
        }

        let centroids_path = format!("{}.centroids.parquet", root_path);
        // The disk cache is host-global, so key it by the store-qualified
        // `cache_key` (root_uri + path), not the relative object path. Two
        // stores — or two CI steps writing the same relative layout into
        // /tmp — would otherwise share entries and could load a stale
        // centroids/graph while a mapping is freshly written.
        let centroids_cache_key = format!("{}.centroids.parquet", cache_key);
        let centroids_bytes = disk_cache
            .get_bytes_with_key(&centroids_path, &centroids_cache_key)
            .await?;

        let builder = ParquetRecordBatchReaderBuilder::try_new(centroids_bytes)?;
        let reader = builder.build()?;

        let batches: Vec<RecordBatch> = reader
            .map(|r| r.map_err(anyhow::Error::from))
            .collect::<Result<Vec<_>>>()?;
        if batches.is_empty() {
            anyhow::bail!("No data in centroids file");
        }
        let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches)?;

        let centroids = Self::extract_centroids_from_batch(&batch)?;
        if centroids.is_empty() {
            anyhow::bail!("No centroids extracted from centroids file");
        }

        let n_lists = centroids.len();
        let dim = centroids[0].len();

        // Load metric, quantizer, and MIPS augmentation bound from parquet metadata
        let (metric, quantizer, augment_m) = {
            let centroids_bytes_for_meta = disk_cache
                .get_bytes_with_key(&centroids_path, &centroids_cache_key)
                .await?;
            let parquet_reader =
                parquet::file::reader::SerializedFileReader::new(centroids_bytes_for_meta)?;
            let file_metadata = parquet_reader.metadata().file_metadata();
            let mut loaded_metric = VectorMetric::L2;
            let mut loaded_quantizer = None;
            let mut loaded_augment_m = 0.0f32;

            if let Some(kv_list) = file_metadata.key_value_metadata() {
                for kv in kv_list {
                    if kv.key == "vector_metric" {
                        if let Some(ref val) = kv.value {
                            loaded_metric = val.parse::<VectorMetric>().unwrap_or(VectorMetric::L2);
                        }
                    } else if kv.key == "quantizer_config" {
                        if let Some(ref val) = kv.value {
                            if let Ok(mut q) = serde_json::from_str::<QuantizerImpl>(val) {
                                if let QuantizerImpl::Pq(ref mut pq) = q {
                                    pq.init_lut();
                                }
                                loaded_quantizer = Some(q);
                            }
                        }
                    } else if kv.key == "augment_m" {
                        if let Some(ref val) = kv.value {
                            loaded_augment_m = val.parse::<f32>().unwrap_or(0.0);
                        }
                    }
                }
            }
            (loaded_metric, loaded_quantizer, loaded_augment_m)
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

        // Phase 1: Load all cluster bytes in parallel (I/O-bound, via futures).
        enum GraphData {
            Mmap(Arc<memmap2::Mmap>),
            Bytes(bytes::Bytes),
        }

        struct ClusterBytes {
            cluster_id: usize,
            graph: GraphData,
            mapping: bytes::Bytes,
        }

        let byte_futures = futures::stream::iter(cluster_ids.clone())
            .map(|cluster_id| {
                let root_path = root_path.clone();
                let dc = disk_cache.clone();
                async move {
                    let hnsw_key = format!("{}.cluster_{}.hnsw.graph", root_path, cluster_id);
                    let mapping_key =
                        format!("{}.cluster_{}.mapping.parquet", root_path, cluster_id);
                    // Qualified keys keep graph and mapping for one cluster
                    // together across stores/runs.
                    let hnsw_cache_key = format!("{}.cluster_{}.hnsw.graph", cache_key, cluster_id);
                    let mapping_cache_key =
                        format!("{}.cluster_{}.mapping.parquet", cache_key, cluster_id);

                    let res_graph = if use_mmap {
                        // mmap when the object is a real local file; an in-memory
                        // store (Puffin container slices) cannot be mmap'd, so
                        // fall back to its zero-copy Bytes view.
                        match dc.get_mmap_with_key(&hnsw_key, &hnsw_cache_key).await {
                            Ok(m) => GraphData::Mmap(m),
                            Err(_) => GraphData::Bytes(
                                dc.get_bytes_with_key(&hnsw_key, &hnsw_cache_key).await?,
                            ),
                        }
                    } else {
                        GraphData::Bytes(dc.get_bytes_with_key(&hnsw_key, &hnsw_cache_key).await?)
                    };
                    let res_mapping = dc
                        .get_bytes_with_key(&mapping_key, &mapping_cache_key)
                        .await?;

                    Ok(ClusterBytes {
                        cluster_id,
                        graph: res_graph,
                        mapping: res_mapping,
                    })
                }
            })
            .buffer_unordered(fetch_concurrency);
        let cluster_bytes: Vec<Result<ClusterBytes>> = byte_futures.collect().await;

        // Phase 2: Deserialize all HNSW graphs in parallel (CPU-bound, via rayon).
        let quantizer_for_deser = quantizer.clone();
        let deser_results: Vec<Result<(usize, (HnswGraph, Vec<usize>))>> =
            crate::core::run_blocking(|| {
                cluster_bytes
                    .into_par_iter()
                    .map(|cb_res| {
                        let cb = cb_res?;
                        let cluster_id = cb.cluster_id;
                        let graph_bytes = match cb.graph {
                            GraphData::Mmap(m) => bytes::Bytes::from_owner(MmapOwner(m)),
                            GraphData::Bytes(b) => b,
                        };
                        let (hnsw, row_id_mapping) = Self::deserialize_single_cluster(
                            cluster_id,
                            graph_bytes,
                            cb.mapping,
                            quantizer_for_deser.as_ref(),
                            metric,
                            augment_m,
                        )?;
                        Ok((cluster_id, (hnsw, row_id_mapping)))
                    })
                    .collect()
            });

        let mut cluster_graphs = HashMap::new();
        for res in deser_results {
            let (cid, val) = res?;
            cluster_graphs.insert(cid, val);
        }

        tracing::debug!("Loaded {} cluster graphs (Async)", cluster_graphs.len());

        let cluster_graphs = cluster_graphs
            .into_iter()
            .map(|(k, v)| (k, Arc::new(v)))
            .collect();

        let index = HnswIvfIndex {
            centroids,
            metric,
            cluster_graphs,
            cluster_cache: moka::sync::Cache::builder().max_capacity(64).build(),
            lazy_blobs: None,
            mmap_backed: use_mmap,
            _n_lists: n_lists,
            dim,
            quantizer,
            rescore_oversample: 4,
            augment_m,
            _compute_context: crate::core::index::gpu::ComputeContext::auto_detect(),
        };

        let index_arc = Arc::new(index);
        HNSW_IVF_CACHE
            .insert(cache_key_str, index_arc.clone())
            .await;

        Ok(index_arc)
    }

    /// Cache key for a vector index packed in a Puffin compound bundle.
    ///
    /// One container holds **one** vector index per `(column, algorithm)`,
    /// split across many blobs: centroids, plus a graph and a mapping per IVF
    /// cluster (hundreds of blobs at 10M vectors). A cache key derived from the
    /// blob offset gave every one of those blobs its own entry, and every entry
    /// had to materialize and parse the *whole* bundle — O(blobs²) bytes of
    /// temp-file I/O. Loading an index once must produce one entry, so the key
    /// is the container plus the column/algorithm it was built for.
    pub fn puffin_vector_cache_key(
        root_uri: &str,
        container_path: &str,
        column: &str,
        algorithm: &str,
    ) -> String {
        format!(
            "{}/{}:{}:{}",
            root_uri.trim_end_matches('/'),
            container_path,
            column,
            algorithm
        )
    }

    /// Load an HNSW-IVF index packed inside a Puffin compound bundle.
    ///
    /// The container is read **once** and each blob is served as a zero-copy
    /// slice of it: `use_mmap` maps the `.puffin` file when the store is local,
    /// otherwise the container is fetched as `Bytes`. The slices are handed to
    /// the standard loader through an in-memory object store, so nothing is
    /// copied to a temp directory. The previous implementation materialized
    /// every blob into `$TMPDIR` and parsed it back — a full write + re-read of
    /// the bundle (4.3 GB per load at 10M vectors) on every cold load.
    pub async fn load_from_puffin(
        store: Arc<dyn ObjectStore>,
        puffin_path: &str,
        cache_key: &str,
        use_mmap: bool,
        column_name: &str,
    ) -> Result<Arc<Self>> {
        use crate::core::cache::{DiskCache, HNSW_IVF_CACHE};

        let cache_key_str = cache_key.to_string();
        if let Some(cached) = HNSW_IVF_CACHE
            .get_with_metrics(&cache_key_str, "hnsw_ivf")
            .await
        {
            return Ok(cached);
        }

        // One read of the container: mmap it when the store is local (the page
        // cache then serves the blobs actually touched), else fetch the bytes.
        let container: bytes::Bytes = if use_mmap {
            match DiskCache::new(store.clone()).get_mmap(puffin_path).await {
                Ok(mmap) => bytes::Bytes::from_owner(MmapOwner(mmap)),
                Err(_) => {
                    store
                        .get(&object_store::path::Path::from(puffin_path))
                        .await?
                        .bytes()
                        .await?
                }
            }
        } else {
            store
                .get(&object_store::path::Path::from(puffin_path))
                .await?
                .bytes()
                .await?
        };

        // Zero-copy lazy loading directly from the container; falls back to the
        // temp-dir materialization if the container has no usable properties.
        match Self::load_lazy_from_container(container, puffin_path, column_name, use_mmap) {
            Ok(index) => {
                let arc_index = Arc::new(index);
                HNSW_IVF_CACHE
                    .insert(cache_key_str, arc_index.clone())
                    .await;
                Ok(arc_index)
            }
            Err(e) => {
                tracing::warn!(
                    puffin = %puffin_path,
                    error = %e,
                    "load_from_puffin: lazy loading unavailable, materializing blobs"
                );
                Self::load_from_puffin_materialized(store, puffin_path, cache_key, column_name)
                    .await
            }
        }
    }

    /// Read only the centroids blob from a Puffin index container.
    ///
    /// [`Self::load_from_puffin`] deserializes the whole index — graph, vectors,
    /// quantizer — to answer a question that needs none of them. At 100M that is
    /// 42 GB of index loading to build a <10 MB routing matrix, paid again after
    /// every rebuild. The centroids are already a named Puffin blob
    /// (`org.apache.iceberg.vector.centroids`), so read that one and parse it.
    pub async fn load_centroids_from_puffin(
        store: Arc<dyn ObjectStore>,
        puffin_path: &str,
        use_mmap: bool,
    ) -> Result<(Vec<Vec<f32>>, VectorMetric)> {
        use crate::core::cache::DiskCache;
        use crate::core::puffin::{PuffinIndexReader, PUFFIN_BLOB_HNSW_IVF_CENTROIDS};
        use std::io::Cursor;

        // One read of the container: mmap it when the store is local (the page
        // cache then serves only the blob actually touched), else fetch bytes.
        let container: bytes::Bytes = if use_mmap {
            match DiskCache::new(store.clone()).get_mmap(puffin_path).await {
                Ok(mmap) => bytes::Bytes::from_owner(MmapOwner(mmap)),
                Err(_) => {
                    store
                        .get(&object_store::path::Path::from(puffin_path))
                        .await?
                        .bytes()
                        .await?
                }
            }
        } else {
            store
                .get(&object_store::path::Path::from(puffin_path))
                .await?
                .bytes()
                .await?
        };

        let mut reader = PuffinIndexReader::new(Cursor::new(container))?;
        let blob = reader
            .read_blob_by_type(PUFFIN_BLOB_HNSW_IVF_CENTROIDS)?
            .ok_or_else(|| {
                anyhow::anyhow!("Puffin bundle '{}' has no centroids blob", puffin_path)
            })?;

        let builder = parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
            bytes::Bytes::from(blob),
        )?;

        let metric = {
            let mut loaded = VectorMetric::L2;
            if let Some(kv) = builder.metadata().file_metadata().key_value_metadata() {
                for entry in kv {
                    if entry.key == "vector_metric" {
                        if let Some(val) = &entry.value {
                            loaded = val.parse::<VectorMetric>().unwrap_or(VectorMetric::L2);
                        }
                    }
                }
            }
            loaded
        };

        let mut centroids = Vec::new();
        for batch in builder.build()? {
            centroids.extend(Self::extract_centroids_from_batch(&batch?)?);
        }

        Ok((centroids, metric))
    }

    /// Legacy fallback: materialize the bundle into a temp directory and load
    /// from there. Used when the container's blobs cannot be addressed directly.
    async fn load_from_puffin_materialized(
        store: Arc<dyn ObjectStore>,
        puffin_path: &str,
        cache_key: &str,
        column_name: &str,
    ) -> Result<Arc<Self>> {
        let (temp_dir, written) =
            crate::core::puffin::materialize_puffin_blobs(&store, puffin_path).await?;

        // A Puffin bundle holds one vector index per (column, algorithm). Only
        // derive the loader base path from the centroids blob belonging to the
        // requested column — otherwise a second vector column in the same
        // segment would be loaded instead (the blob order is not stable).
        let marker = format!(".{column_name}.");
        let base_path = written
            .iter()
            .find_map(|(_, name)| {
                if name.contains(&marker) {
                    name.strip_suffix(".centroids.parquet").map(str::to_string)
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Puffin bundle '{}' has no centroids blob for column '{}'",
                    puffin_path,
                    column_name
                )
            })?;

        let local_store: Arc<dyn ObjectStore> = Arc::new(
            object_store::local::LocalFileSystem::new_with_prefix(temp_dir.path())?,
        );

        let index =
            Self::load_async_with_cache_key(local_store, &base_path, cache_key, false).await?;
        drop(temp_dir);
        Ok(index)
    }

    pub fn load(base_path: &str) -> Result<Self> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        use std::fs::File;

        let mut base_path_str = base_path.to_string();
        if let Some(idx) = base_path_str.find(".cluster_") {
            base_path_str = base_path_str[..idx].to_string();
        } else if let Some(idx) = base_path_str.find(".hnsw.graph") {
            base_path_str = base_path_str[..idx].to_string();
        }

        let centroids_path = format!("{}.centroids.parquet", base_path_str);
        let file = File::open(&centroids_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let reader = builder.build()?;

        let batches: Vec<RecordBatch> = reader.collect::<std::result::Result<_, _>>()?;
        if batches.is_empty() {
            anyhow::bail!("No data in centroids file");
        }
        let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches)?;

        let centroids = Self::extract_centroids_from_batch(&batch)?;
        if centroids.is_empty() {
            anyhow::bail!("No centroids extracted from centroids file");
        }

        let n_lists = centroids.len();
        let dim = centroids[0].len();

        // Load metric and MIPS augmentation bound from parquet metadata
        let (metric, augment_m) = {
            let file = File::open(&centroids_path)?;
            let parquet_reader = parquet::file::reader::SerializedFileReader::new(file)?;
            let file_metadata = parquet_reader.metadata().file_metadata();
            let mut loaded_metric = VectorMetric::L2;
            let mut loaded_augment_m = 0.0f32;
            if let Some(kv_list) = file_metadata.key_value_metadata() {
                for kv in kv_list {
                    if kv.key == "vector_metric" {
                        if let Some(ref val) = kv.value {
                            loaded_metric = val.parse::<VectorMetric>().unwrap_or(VectorMetric::L2);
                        }
                    } else if kv.key == "augment_m" {
                        if let Some(ref val) = kv.value {
                            loaded_augment_m = val.parse::<f32>().unwrap_or(0.0);
                        }
                    }
                }
            }
            (loaded_metric, loaded_augment_m)
        };

        tracing::debug!("Loaded {} centroids of dimension {}", n_lists, dim);

        let quantizer_path = format!("{}.quantizer", base_path_str);
        let quantizer = if std::path::Path::new(&quantizer_path).exists() {
            let q_bytes = std::fs::read(&quantizer_path)?;
            let q: crate::core::index::QuantizerImpl = serde_json::from_slice(&q_bytes)?;
            Some(q)
        } else {
            None
        };

        let mut cluster_graphs = HashMap::new();
        for cluster_id in 0..n_lists {
            let hnsw_path = format!("{}.cluster_{}", base_path, cluster_id);
            let mapping_path = format!("{}.cluster_{}.mapping.parquet", base_path, cluster_id);

            if !std::path::Path::new(&format!("{}.hnsw.graph", hnsw_path)).exists() {
                continue;
            }

            use std::io::Read;
            let mut graph_file = File::open(format!("{}.hnsw.graph", hnsw_path))?;
            let mut graph_bytes = Vec::new();
            graph_file.read_to_end(&mut graph_bytes)?;

            // Use the metric loaded from centroids metadata
            let hnsw = if let Some(q_impl) = &quantizer {
                match q_impl {
                    crate::core::index::QuantizerImpl::BenoQuant(q) => {
                        if q.bits == 4 {
                            HnswGraph::TurboQuant4Arrow(
                                crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes(
                                    &graph_bytes,
                                    crate::core::index::distance::DistL2u4,
                                )
                                .map_err(|e| anyhow::anyhow!("ArrowHnsw load failed: {}", e))?,
                            )
                        } else {
                            HnswGraph::TurboQuant8Arrow(
                                crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes(
                                    &graph_bytes,
                                    crate::core::index::distance::DistL2u8,
                                )
                                .map_err(|e| anyhow::anyhow!("ArrowHnsw load failed: {}", e))?,
                            )
                        }
                    }
                    crate::core::index::QuantizerImpl::Sq(q) => HnswGraph::Sq8Arrow(
                        crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes(
                            &graph_bytes,
                            crate::core::index::sq::DistSq8 {
                                encoder: std::sync::Arc::new(q.clone()),
                            },
                        )
                        .map_err(|e| anyhow::anyhow!("ArrowHnsw load failed: {}", e))?,
                    ),
                    _ => anyhow::bail!("Unsupported quantizer variant for legacy load"),
                }
            } else {
                match metric {
                    VectorMetric::L2 => HnswGraph::L2Arrow(
                        crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes(
                            &graph_bytes,
                            DistL2,
                        )
                        .map_err(|e| anyhow::anyhow!("ArrowHnsw load failed: {}", e))?,
                    ),
                    VectorMetric::Cosine => HnswGraph::CosineArrow(
                        crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes(
                            &graph_bytes,
                            DistCosine,
                        )
                        .map_err(|e| anyhow::anyhow!("ArrowHnsw load failed: {}", e))?,
                    ),
                    VectorMetric::InnerProduct => HnswGraph::DotArrow(
                        crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes(
                            &graph_bytes,
                            if augment_m > 0.0 {
                                DistDot::augmented(augment_m)
                            } else {
                                DistDot::norm()
                            },
                        )
                        .map_err(|e| anyhow::anyhow!("ArrowHnsw load failed: {}", e))?,
                    ),
                    VectorMetric::L1 => HnswGraph::L1Arrow(
                        crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes(
                            &graph_bytes,
                            DistL1,
                        )
                        .map_err(|e| anyhow::anyhow!("ArrowHnsw load failed: {}", e))?,
                    ),
                    VectorMetric::Hamming => HnswGraph::HammingArrow(
                        crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes(
                            &graph_bytes,
                            DistHamming,
                        )
                        .map_err(|e| anyhow::anyhow!("ArrowHnsw load failed: {}", e))?,
                    ),
                    VectorMetric::Jaccard => HnswGraph::JaccardArrow(
                        crate::core::index::hnsw_rs::arrow_hnsw::ArrowHnsw::load_from_bytes(
                            &graph_bytes,
                            DistJaccard,
                        )
                        .map_err(|e| anyhow::anyhow!("ArrowHnsw load failed: {}", e))?,
                    ),
                }
            };

            let file = File::open(&mapping_path)?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            let reader = builder.build()?;

            let batches: Vec<RecordBatch> = reader.collect::<std::result::Result<_, _>>()?;
            if batches.is_empty() {
                anyhow::bail!("No data in mapping file");
            }
            let batch = arrow::compute::concat_batches(&batches[0].schema(), &batches)?;

            let row_id_mapping = Self::extract_row_id_mapping_from_col(batch.column(0));

            cluster_graphs.insert(cluster_id, (hnsw, row_id_mapping));
        }

        tracing::debug!("Loaded {} cluster graphs", cluster_graphs.len());

        let cluster_graphs = cluster_graphs
            .into_iter()
            .map(|(k, v)| (k, Arc::new(v)))
            .collect();

        Ok(HnswIvfIndex {
            centroids,
            metric,
            cluster_graphs,
            cluster_cache: moka::sync::Cache::builder().max_capacity(64).build(),
            lazy_blobs: None,
            mmap_backed: false,
            _n_lists: n_lists,
            dim,
            quantizer,
            rescore_oversample: 4,
            augment_m,
            _compute_context: crate::core::index::gpu::ComputeContext::auto_detect(),
        })
    }

    /// Bytes this index occupies, used as the weight of [`HNSW_IVF_CACHE`](crate::core::cache::HNSW_IVF_CACHE).
    ///
    /// This must describe the **loaded** representation, not the builder's. The
    /// old estimate charged 20 KB/point plus `dim * 4` bytes per vector, which
    /// put a 10M x 128 index at ~200 GB. That weight can never fit a realistic
    /// cache budget, so moka rejected every insert and each query re-loaded the
    /// whole index from the object store — the per-query reload storm that
    /// hammered the disk for an hour.
    pub fn size_in_bytes(&self) -> usize {
        let mut size = 0;
        for c in &self.centroids {
            size += c.len() * std::mem::size_of::<f32>();
        }
        for cluster_data in self.cluster_graphs.values() {
            let (hnsw, mapping) = cluster_data.as_ref();
            // Row-id mapping is one usize per point.
            size += mapping.len() * std::mem::size_of::<usize>();
            // Graph + codes. mmap-backed graphs alias the file (page-cache
            // backed, not heap), so they are not charged against the moka
            // budget; owned-byte graphs (remote store / materialized) are.
            if !self.mmap_backed {
                size += hnsw.footprint_bytes();
            }
        }
        for (_, cluster_data) in self.cluster_cache.iter() {
            let (hnsw, mapping) = cluster_data.as_ref();
            size += mapping.len() * std::mem::size_of::<usize>();
            if !self.mmap_backed {
                size += hnsw.footprint_bytes();
            }
        }
        size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use roaring::RoaringBitmap;

    #[test]
    fn rescore_inner_product_orders_by_raw_dot() {
        let query = [1.0f32, 0.0];
        // Approximate (graph) distances are deliberately wrong; the raw vectors
        // are what must decide the final order.
        let candidates = vec![
            (10usize, 0.1f32, Some(vec![0.5f32, 0.0])), // dot 0.5 -> -0.5
            (11usize, 0.2f32, Some(vec![2.0f32, 0.0])), // dot 2.0 -> -2.0 (best)
            (12usize, 0.3f32, Some(vec![1.0f32, 0.0])), // dot 1.0 -> -1.0
        ];
        let out = rescore_inner_product(&query, candidates, 3);
        assert_eq!(
            out.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![11, 12, 10]
        );
        assert!((out[0].1 + 2.0).abs() < 1e-6);
    }

    #[test]
    fn rescore_inner_product_falls_back_and_truncates() {
        let query = [1.0f32, 0.0];
        let candidates = vec![
            (1usize, 0.9f32, None),                    // no raw vector -> keep 0.9
            (2usize, 0.4f32, Some(vec![1.0f32, 0.0])), // dot 1.0 -> -1.0
            (3usize, 0.7f32, Some(vec![0.0f32, 1.0])), // dot 0.0 -> 0.0
        ];
        let out = rescore_inner_product(&query, candidates, 2);
        assert_eq!(out.len(), 2);
        // -1.0 < 0.0 < 0.9, so ids 2 and 3 survive the truncation.
        assert_eq!(
            out.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![2, 3]
        );
    }

    #[test]
    fn test_hnsw_ivf_pre_filtering() {
        let mut vectors = Vec::new();
        for i in 0..100 {
            let mut vec = vec![0.0; 4];
            vec[0] = (i as f32) / 100.0;
            vectors.push(vec);
        }

        let algo = crate::core::manifest::IndexAlgorithm::hnsw();
        let index = HnswIvfIndex::build(
            vectors.clone(),
            VectorMetric::L2,
            Some(2),
            Some(16),
            &algo,
            0,
        )
        .unwrap();
        let query = VectorValue::Float32(vectors[50].clone());

        // Unfiltered search (should find ID 50 easily since it's an exact match)
        let results_unfiltered = index.search(&query, 5, 40, None, None).unwrap();
        assert!(!results_unfiltered.is_empty(), "Should return results");

        let mut found_50 = false;
        for (id, _) in &results_unfiltered {
            if *id == 50 {
                found_50 = true;
                break;
            }
        }
        assert!(
            found_50,
            "Unfiltered search should easily find exact match ID 50"
        );

        // Filtered search (exclude ID 50, only allow 51, 52, 53)
        let mut filter = RoaringBitmap::new();
        filter.insert(51);
        filter.insert(52);
        filter.insert(53);

        let results_filtered = index.search(&query, 3, 40, Some(&filter), None).unwrap();
        assert!(
            !results_filtered.is_empty(),
            "Filtered search should return results"
        );

        let mut found_50_filtered = false;
        let mut found_51_filtered = false;
        for (id, _) in &results_filtered {
            if *id == 50 {
                found_50_filtered = true;
            }
            if *id == 51 {
                found_51_filtered = true;
            }
        }

        assert!(
            !found_50_filtered,
            "Pre-filtering failed to exclude exact match ID 50"
        );
        assert!(
            found_51_filtered,
            "Pre-filtering failed to find the next best allowed match ID 51"
        );

        for (id, _) in &results_filtered {
            assert!(
                filter.contains(*id as u32),
                "Result ID {} was not in the filter!",
                id
            );
        }
    }

    #[test]
    fn test_hnsw_ivf_path_robustness() {
        let base_path_str = std::env::temp_dir()
            .join("test_robust")
            .to_string_lossy()
            .into_owned();

        let paths = vec![
            format!("{}.cluster_0.hnsw.graph", base_path_str),
            format!("{}.cluster_99.hnsw.graph", base_path_str),
            format!("{}.hnsw.graph", base_path_str),
        ];

        for p in paths {
            let mut root = p.clone();
            if let Some(idx) = root.find(".cluster_") {
                root = root[..idx].to_string();
            } else if let Some(idx) = root.find(".hnsw.graph") {
                root = root[..idx].to_string();
            }
            assert_eq!(root, base_path_str, "Failed to strip suffix from {}", p);
        }
    }

    /// The out-of-core builder must accept both the new self-describing header
    /// and legacy header-less files, and index every vector either way.
    #[test]
    fn header_and_legacy_vector_files_build_equivalent_indexes() {
        use std::io::Write;

        fn write_vec_file(path: &std::path::Path, dim: usize, n: usize, with_header: bool) {
            let mut f = std::fs::File::create(path).unwrap();
            if with_header {
                f.write_all(&VEC_TMP_MAGIC.to_le_bytes()).unwrap();
                f.write_all(&(dim as u32).to_le_bytes()).unwrap();
            }
            for i in 0..n {
                let v: Vec<f32> = (0..dim).map(|d| (i * dim + d) as f32).collect();
                f.write_all(&(i as u32).to_le_bytes()).unwrap();
                f.write_all(&(dim as u32).to_le_bytes()).unwrap();
                f.write_all(bytemuck::cast_slice(&v)).unwrap();
            }
        }

        fn total_points(idx: &HnswIvfIndex) -> usize {
            idx.cluster_graphs.values().map(|c| c.1.len()).sum()
        }

        let dir = tempfile::tempdir().unwrap();
        let dim = 8;
        let n = 500;
        let algo = crate::core::manifest::IndexAlgorithm::hnsw();

        let p1 = dir.path().join("with_header.bin");
        write_vec_file(&p1, dim, n, true);
        let idx1 = HnswIvfIndex::build_from_file(
            p1.to_str().unwrap(),
            VectorMetric::L2,
            None,
            None,
            &algo,
        )
        .unwrap();
        assert_eq!(total_points(&idx1), n, "header path dropped vectors");

        let p2 = dir.path().join("legacy.bin");
        write_vec_file(&p2, dim, n, false);
        let idx2 = HnswIvfIndex::build_from_file(
            p2.to_str().unwrap(),
            VectorMetric::L2,
            None,
            None,
            &algo,
        )
        .unwrap();
        assert_eq!(total_points(&idx2), n, "legacy path dropped vectors");
    }

    #[test]
    fn out_of_core_sampling_includes_tail_vectors() {
        use std::io::Write;

        let dir = tempfile::tempdir().unwrap();
        let dim = 8;
        let n = 119usize; // Previously: 119 / 64 == 1 block -> only rows 0..64 were sampled.
        let path = dir.path().join("tail_sample.bin");

        {
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&VEC_TMP_MAGIC.to_le_bytes()).unwrap();
            f.write_all(&(dim as u32).to_le_bytes()).unwrap();
            for i in 0..n {
                // Rows 0..64 live near 0..63; tail rows 64..119 jump up to 1000..1054.
                let val = if i < 64 {
                    i as f32
                } else {
                    1000.0 + (i - 64) as f32
                };
                let v = vec![val; dim];
                f.write_all(&(i as u32).to_le_bytes()).unwrap();
                f.write_all(&(dim as u32).to_le_bytes()).unwrap();
                f.write_all(bytemuck::cast_slice(&v)).unwrap();
            }
        }

        let (count, parsed_dim, samples) =
            HnswIvfIndex::sample_vectors_from_tmp_file(path.to_str().unwrap(), 100_000).unwrap();
        assert_eq!(count, n);
        assert_eq!(parsed_dim, dim);
        assert_eq!(samples.len(), n);
        assert!(samples.iter().any(|v| (v[0] - 1054.0).abs() < 1e-4));

        // Build an out-of-core HnswTq8 index over the 119 vectors and verify
        // exact tail retrieval without cross-row clamping collapse.
        let algo = crate::core::manifest::IndexAlgorithm::hnsw_tq8();
        let idx = HnswIvfIndex::build_from_file(
            path.to_str().unwrap(),
            VectorMetric::L2,
            None,
            None,
            &algo,
        )
        .unwrap();
        let hits = idx
            .search(&VectorValue::Float32(vec![1054.0; dim]), 5, 16, None, None)
            .unwrap();
        assert!(
            hits.iter().any(|(id, _)| *id == n - 1),
            "expected tail row {} in top-5, got {:?}",
            n - 1,
            hits
        );
        assert!(
            hits[0].0 >= 110 && hits[0].1 < 25.0,
            "expected top hit in near-tail with calibrated L2 distance, got {:?}",
            hits
        );
    }

    #[test]
    fn multi_segment_quantized_l2_distances_remain_calibrated_on_out_of_range_queries() {
        let dim = 8;
        let seg1_vecs: Vec<Vec<f32>> = (0..16).map(|i| vec![(i * 4) as f32; dim]).collect();
        let seg2_vecs: Vec<Vec<f32>> = (190..206).map(|i| vec![i as f32; dim]).collect();
        let query = VectorValue::Float32(vec![197.0; dim]);

        for algo in [
            crate::core::manifest::IndexAlgorithm::hnsw_tq8(),
            crate::core::manifest::IndexAlgorithm::hnsw_tq4(),
            crate::core::manifest::IndexAlgorithm::hnsw_sq8(),
        ] {
            let idx1 =
                HnswIvfIndex::build(seg1_vecs.clone(), VectorMetric::L2, Some(1), None, &algo, 0)
                    .unwrap();
            let idx2 = HnswIvfIndex::build(
                seg2_vecs.clone(),
                VectorMetric::L2,
                Some(1),
                None,
                &algo,
                100,
            )
            .unwrap();

            let h1 = idx1.search(&query, 3, 1, None, None).unwrap();
            let h2 = idx2.search(&query, 3, 1, None, None).unwrap();

            assert!(!h1.is_empty() && !h2.is_empty());
            let d1_best = h1[0].1;
            let d2_best = h2[0].1;
            let expected_seg1 = (197.0 - 60.0) * (dim as f32).sqrt();
            assert!(
                (d1_best - expected_seg1).abs() < 45.0,
                "algo={:?}: seg1 best distance must be calibrated (~{expected_seg1}), got {d1_best}",
                algo
            );
            assert!(
                d2_best < 45.0 && d2_best < d1_best,
                "algo={:?}: exact match in seg2 ({d2_best}) must beat out-of-range seg1 ({d1_best})",
                algo
            );
            if !matches!(algo, crate::core::manifest::IndexAlgorithm::HnswTq4 { .. }) {
                assert!(
                    h2.iter().any(|(id, _)| *id == 107),
                    "algo={:?}: expected row 107 ([197; 8]) in top-3, got {:?}",
                    algo,
                    h2
                );
            }
        }
    }
}
