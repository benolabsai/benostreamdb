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
//! ## Quickstart
//!
//! ```no_run
//! use benostream_gpu_ann::{IndexBuilder, Metric};
//!
//! let dim = 128;
//! let n_vectors = 10_000;
//! let vectors = vec![0.0f32; n_vectors * dim];
//!
//! // Build GPU IVF-Flat index (auto-detects CUDA / Metal / WGPU)
//! let index = IndexBuilder::new(dim, Metric::L2)
//!     .n_lists(100)
//!     .build(&vectors, None)
//!     .expect("Failed to build index");
//!
//! let query = vec![0.0f32; dim];
//! let results = index.search(&query, 10, 5, None).expect("Search failed");
//! println!("Found {} nearest neighbors on {}", results.len(), index.backend_name());
//! ```

pub mod backend;
pub mod ivf_flat;
pub mod kmeans;
pub mod metric;

pub use backend::{ComputeContext, GpuBackend};
pub use ivf_flat::{IvfFlatIndex, SearchResult};
pub use metric::Metric;

/// Index builder to configure and instantiate GPU vector indexes.
pub struct IndexBuilder {
    dim: usize,
    metric: Metric,
    n_lists: Option<usize>,
    ctx: Option<ComputeContext>,
}

impl IndexBuilder {
    /// Create a new builder for vectors of dimension `dim` using distance `metric`.
    pub fn new(dim: usize, metric: Metric) -> Self {
        Self {
            dim,
            metric,
            n_lists: None,
            ctx: None,
        }
    }

    /// Set number of IVF coarse Voronoi clusters.
    pub fn n_lists(mut self, n_lists: usize) -> Self {
        self.n_lists = Some(n_lists);
        self
    }

    /// Specify an explicit hardware compute context (e.g. `ComputeContext::cuda(0)` or `ComputeContext::cpu()`).
    pub fn context(mut self, ctx: ComputeContext) -> Self {
        self.ctx = Some(ctx);
        self
    }

    /// Build the GPU IVF-Flat index.
    pub fn build(self, vectors: &[f32], ids: Option<&[u64]>) -> anyhow::Result<IvfFlatIndex> {
        IvfFlatIndex::build(vectors, ids, self.dim, self.n_lists, self.metric, self.ctx)
    }
}
