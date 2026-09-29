// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

pub mod cpu;
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
pub mod cuda;
#[cfg(target_os = "macos")]
pub mod metal;
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
pub mod nvrtc;
#[cfg(feature = "wgpu")]
pub mod wgpu;

use anyhow::Result;
use std::sync::Arc;
use crate::metric::Metric;

/// Trait implemented by all hardware backends (CUDA, Apple Metal, WGPU, CPU).
pub trait GpuBackend: Send + Sync {
    /// Friendly display name of this backend.
    fn name(&self) -> &str;

    /// Compute distances between a single query vector and a flat array of vectors.
    fn compute_distance(
        &self,
        query: &[f32],
        vectors: &[f32],
        dim: usize,
        metric: Metric,
    ) -> Result<Vec<f32>>;

    /// Compute cluster centroid assignments (L2 distance) for a batch of vectors.
    fn compute_kmeans_assignment(
        &self,
        vectors: &[f32],
        centroids: &[f32],
        dim: usize,
    ) -> Result<Vec<u32>>;
}

/// ComputeContext encapsulates the selected hardware accelerator and provides
/// seamless fallback to CPU when necessary.
#[derive(Clone)]
pub struct ComputeContext {
    backend: Arc<dyn GpuBackend>,
}

impl ComputeContext {
    /// Create a context wrapping a specific backend.
    pub fn new(backend: Arc<dyn GpuBackend>) -> Self {
        Self { backend }
    }

    /// Automatically probe and select the fastest available hardware accelerator.
    ///
    /// Probe priority:
    /// 1. NVIDIA CUDA (via dynamic cudarc/nvrtc)
    /// 2. Apple Metal (on macOS Apple Silicon)
    /// 3. WGPU (Vulkan / AMD / Intel)
    /// 4. CPU (Rayon/SIMD fallback)
    pub fn auto_detect() -> Self {
        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        {
            if let Ok(b) = cuda::CudaBackend::new(0) {
                tracing::info!("benostream-gpu-ann: selected hardware backend: CUDA");
                return Self {
                    backend: Arc::new(b),
                };
            }
        }

        #[cfg(target_os = "macos")]
        {
            if let Ok(b) = metal::MetalBackend::new() {
                tracing::info!("benostream-gpu-ann: selected hardware backend: Apple Metal");
                return Self {
                    backend: Arc::new(b),
                };
            }
        }

        #[cfg(feature = "wgpu")]
        {
            if let Ok(b) = wgpu::WgpuBackend::new("WGPU_Default", None) {
                tracing::info!("benostream-gpu-ann: selected hardware backend: WGPU");
                return Self {
                    backend: Arc::new(b),
                };
            }
        }

        tracing::info!("benostream-gpu-ann: selected hardware backend: CPU Fallback");
        Self {
            backend: Arc::new(cpu::CpuBackend::new()),
        }
    }

    /// Force CPU execution.
    pub fn cpu() -> Self {
        Self {
            backend: Arc::new(cpu::CpuBackend::new()),
        }
    }

    /// Force CUDA execution on a specific device index.
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    pub fn cuda(device_id: usize) -> Result<Self> {
        let b = cuda::CudaBackend::new(device_id)?;
        Ok(Self {
            backend: Arc::new(b),
        })
    }

    /// Force Apple Metal execution.
    #[cfg(target_os = "macos")]
    pub fn metal() -> Result<Self> {
        let b = metal::MetalBackend::new()?;
        Ok(Self {
            backend: Arc::new(b),
        })
    }

    /// Force WGPU execution.
    #[cfg(feature = "wgpu")]
    pub fn wgpu(display_name: &str, vendor_id: Option<u32>) -> Result<Self> {
        let b = wgpu::WgpuBackend::new(display_name, vendor_id)?;
        Ok(Self {
            backend: Arc::new(b),
        })
    }

    /// Get the display name of the current backend.
    pub fn name(&self) -> &str {
        self.backend.name()
    }

    /// Compute distance between query and a flat vector buffer.
    pub fn compute_distance(
        &self,
        query: &[f32],
        vectors: &[f32],
        dim: usize,
        metric: Metric,
    ) -> Result<Vec<f32>> {
        self.backend.compute_distance(query, vectors, dim, metric)
    }

    /// Compute Voronoi centroid assignments for a batch of vectors.
    pub fn compute_kmeans_assignment(
        &self,
        vectors: &[f32],
        centroids: &[f32],
        dim: usize,
    ) -> Result<Vec<u32>> {
        self.backend.compute_kmeans_assignment(vectors, centroids, dim)
    }
}

impl Default for ComputeContext {
    fn default() -> Self {
        Self::auto_detect()
    }
}
