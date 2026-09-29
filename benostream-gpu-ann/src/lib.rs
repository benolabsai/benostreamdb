// Copyright (c) 2026 BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

//! # BenoStream GPU ANN
//!
//! Universal hardware-accelerated approximate nearest neighbor (ANN) vector search for Rust.
//!
//! Provides ultra-fast vector indexing and search across:
//! - **NVIDIA GPUs** via CUDA (`cudarc` + runtime `nvrtc`)
//! - **Apple Silicon** via Metal (`mps`)
//! - **AMD / Intel / Cross-platform** via WGPU / Vulkan
//! - **Multi-core CPU** fallback with SIMD and Rayon
//!
//! ## Three Algorithm Stages
//!
//! 1. **IVF-Flat** (`Algorithm::IvfFlat`): Coarse k-means Voronoi clustering on GPU. Near-zero index construction time; flat GPU matrix scan at query time.
//! 2. **HNSW** (`Algorithm::Hnsw`): Hierarchical Navigable Small World graph with GPU-batched candidate frontier evaluation.
//! 3. **CAGRA** (`Algorithm::Cagra`): GPU-native fixed-degree anisotropic graph. Built entirely on GPU via Voronoi blocks and NN-Descent, enabling coalesced warp traversal.
//!
//! ## Quickstart
//!
//! ```no_run
//! use benostream_gpu_ann::{IndexBuilder, Metric, Algorithm};
//!
//! let dim = 128;
//! let n_vectors = 10_000;
//! let vectors = vec![0.0f32; n_vectors * dim];
//!
//! // Build Stage 3: GPU-native CAGRA graph
//! let cagra = IndexBuilder::new(dim, Metric::Cosine)
//!     .algorithm(Algorithm::Cagra { graph_degree: 32, intermediate_degree: 64 })
//!     .build_cagra(&vectors, None)
//!     .expect("Failed to build CAGRA index");
//!
//! let query = vec![0.0f32; dim];
//! let results = cagra.search(&query, 10, 40, None).expect("Search failed");
//! println!("Found {} nearest neighbors on {}", results.len(), cagra.backend_name());
//! ```

pub mod backend;
pub mod cagra;
pub mod hnsw;
pub mod ivf_flat;
pub mod kmeans;
pub mod metric;

pub use backend::{ComputeContext, GpuBackend};
pub use cagra::CagraIndex;
pub use hnsw::HnswIndex;
pub use ivf_flat::{IvfFlatIndex, SearchResult};
pub use metric::Metric;

/// Indexing algorithm family.
#[derive(Debug, Clone)]
pub enum Algorithm {
    /// IVF-Flat coarse Voronoi index (Stage 1: instant build, zero graph overhead)
    IvfFlat { n_lists: Option<usize> },
    /// GPU-accelerated Hierarchical Navigable Small World graph (Stage 2: batched frontier evaluation)
    Hnsw { m: usize, ef_construction: usize },
    /// GPU-Native Anisotropic Graph (Stage 3: CAGRA fixed-degree graph built entirely on GPU)
    Cagra { graph_degree: usize, intermediate_degree: usize },
}

impl Default for Algorithm {
    fn default() -> Self {
        Algorithm::IvfFlat { n_lists: None }
    }
}

/// Unified Vector Index container supporting multiple index types.
pub enum VectorIndex {
    IvfFlat(IvfFlatIndex),
    Hnsw(HnswIndex),
    Cagra(CagraIndex),
}

impl VectorIndex {
    /// Search top-k nearest neighbors across any index type.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        beam_or_probe: usize,
        filter: Option<&roaring::RoaringBitmap>,
    ) -> anyhow::Result<Vec<SearchResult>> {
        match self {
            VectorIndex::IvfFlat(idx) => idx.search(query, k, beam_or_probe, filter),
            VectorIndex::Hnsw(idx) => idx.search(query, k, beam_or_probe, filter),
            VectorIndex::Cagra(idx) => idx.search(query, k, beam_or_probe, filter),
        }
    }

    /// Dimensionality of vectors in index.
    pub fn dim(&self) -> usize {
        match self {
            VectorIndex::IvfFlat(idx) => idx.dim(),
            VectorIndex::Hnsw(idx) => idx.dim(),
            VectorIndex::Cagra(idx) => idx.dim(),
        }
    }

    /// Total number of vectors indexed.
    pub fn len(&self) -> usize {
        match self {
            VectorIndex::IvfFlat(idx) => idx.len(),
            VectorIndex::Hnsw(idx) => idx.len(),
            VectorIndex::Cagra(idx) => idx.len(),
        }
    }

    /// Check if index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Active hardware backend name.
    pub fn backend_name(&self) -> &str {
        match self {
            VectorIndex::IvfFlat(idx) => idx.backend_name(),
            VectorIndex::Hnsw(idx) => idx.backend_name(),
            VectorIndex::Cagra(idx) => idx.backend_name(),
        }
    }
}

/// Index builder to configure and instantiate GPU vector indexes.
pub struct IndexBuilder {
    dim: usize,
    metric: Metric,
    algorithm: Algorithm,
    ctx: Option<ComputeContext>,
}

impl IndexBuilder {
    /// Create a new builder for vectors of dimension `dim` using distance `metric`.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            metric,
            algorithm: Algorithm::default(),
            ctx: None,
        }
    }

    /// Select index algorithm (e.g. `Algorithm::IvfFlat`, `Algorithm::Hnsw`, or `Algorithm::Cagra`).
    pub fn algorithm(mut self, algo: Algorithm) -> Self {
        self.algorithm = algo;
        self
    }

    /// Set number of IVF coarse Voronoi clusters (shorthand for `Algorithm::IvfFlat`).
    pub fn n_lists(mut self, n_lists: usize) -> Self {
        self.algorithm = Algorithm::IvfFlat {
            n_lists: Some(n_lists),
        };
        self
    }

    /// Specify an explicit hardware compute context (e.g. `ComputeContext::cuda(0)` or `ComputeContext::cpu()`).
    pub fn context(mut self, ctx: ComputeContext) -> Self {
        self.ctx = Some(ctx);
        self
    }

    /// Build a GPU IVF-Flat index specifically (Stage 1).
    pub fn build_ivf_flat(
        self,
        vectors: &[f32],
        ids: Option<&[u64]>,
    ) -> anyhow::Result<IvfFlatIndex> {
        let n_lists = match self.algorithm {
            Algorithm::IvfFlat { n_lists } => n_lists,
            _ => None,
        };
        IvfFlatIndex::build(vectors, ids, self.dim, n_lists, self.metric, self.ctx)
    }

    /// Build a GPU-accelerated HNSW index specifically (Stage 2).
    pub fn build_hnsw(self, vectors: &[f32], ids: Option<&[u64]>) -> anyhow::Result<HnswIndex> {
        let (m, ef_construction) = match self.algorithm {
            Algorithm::Hnsw { m, ef_construction } => (m, ef_construction),
            _ => (16, 100),
        };
        HnswIndex::build(
            vectors,
            ids,
            self.dim,
            self.metric,
            m,
            ef_construction,
            self.ctx,
        )
    }

    /// Build a GPU-native CAGRA fixed-degree index specifically (Stage 3).
    pub fn build_cagra(self, vectors: &[f32], ids: Option<&[u64]>) -> anyhow::Result<CagraIndex> {
        let (graph_degree, intermediate_degree) = match self.algorithm {
            Algorithm::Cagra { graph_degree, intermediate_degree } => (graph_degree, intermediate_degree),
            _ => (32, 64),
        };
        CagraIndex::build(
            vectors,
            ids,
            self.dim,
            self.metric,
            graph_degree,
            intermediate_degree,
            self.ctx,
        )
    }

    /// Build the configured index variant wrapped in `VectorIndex`.
    pub fn build(self, vectors: &[f32], ids: Option<&[u64]>) -> anyhow::Result<VectorIndex> {
        match self.algorithm {
            Algorithm::IvfFlat { .. } => Ok(VectorIndex::IvfFlat(self.build_ivf_flat(vectors, ids)?)),
            Algorithm::Hnsw { .. } => Ok(VectorIndex::Hnsw(self.build_hnsw(vectors, ids)?)),
            Algorithm::Cagra { .. } => Ok(VectorIndex::Cagra(self.build_cagra(vectors, ids)?)),
        }
    }
}
