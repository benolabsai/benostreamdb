// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
use super::GpuBackend;
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
use crate::metric::Metric;
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
use anyhow::Result;
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
use std::sync::Arc;

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
const CUDA_SRC_L2: &str = include_str!("../kernels/cuda/l2_distance.cu");
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
const CUDA_SRC_COSINE: &str = include_str!("../kernels/cuda/cosine_distance.cu");
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
const CUDA_SRC_INNER_PRODUCT: &str = include_str!("../kernels/cuda/inner_product.cu");
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
const CUDA_SRC_L1: &str = include_str!("../kernels/cuda/l1_distance.cu");
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
const CUDA_SRC_HAMMING: &str = include_str!("../kernels/cuda/hamming_distance.cu");
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
const CUDA_SRC_JACCARD: &str = include_str!("../kernels/cuda/jaccard_distance.cu");
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
const CUDA_SRC_HAMMING_PACKED: &str = include_str!("../kernels/cuda/hamming_packed.cu");
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
const CUDA_SRC_JACCARD_PACKED: &str = include_str!("../kernels/cuda/jaccard_packed.cu");
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
const CUDA_SRC_KMEANS: &str = include_str!("../kernels/cuda/kmeans_assignment.cu");

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
pub struct CudaBackend {
    device: Arc<CudaDevice>,
}

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
impl CudaBackend {
    pub fn new(id: usize) -> Result<Self> {
        let device = CudaDevice::new(id)?;

        macro_rules! compile_and_load {
            ($device:expr, $src:expr, $mod_name:expr, $kernel_name:expr) => {
                let ptx_src = crate::backend::nvrtc::compile_ptx($src).map_err(|e| {
                    anyhow::anyhow!("nvrtc compile failed for {}: {:?}", $mod_name, e)
                })?;
                let ptx = cudarc::nvrtc::Ptx::from_src(ptx_src);
                $device.load_ptx(ptx, $mod_name, &[$kernel_name])?;
            };
        }

        compile_and_load!(device, CUDA_SRC_L2, "l2_distance", "l2_distance_kernel");
        compile_and_load!(
            device,
            CUDA_SRC_COSINE,
            "cosine_distance",
            "cosine_distance_kernel"
        );
        compile_and_load!(
            device,
            CUDA_SRC_INNER_PRODUCT,
            "inner_product",
            "inner_product_kernel"
        );
        compile_and_load!(device, CUDA_SRC_L1, "l1_distance", "l1_distance_kernel");
        compile_and_load!(
            device,
            CUDA_SRC_HAMMING,
            "hamming_distance",
            "hamming_distance_kernel"
        );
        compile_and_load!(
            device,
            CUDA_SRC_JACCARD,
            "jaccard_distance",
            "jaccard_distance_kernel"
        );
        compile_and_load!(
            device,
            CUDA_SRC_HAMMING_PACKED,
            "hamming_packed",
            "hamming_packed_kernel"
        );
        compile_and_load!(
            device,
            CUDA_SRC_JACCARD_PACKED,
            "jaccard_packed",
            "jaccard_packed_kernel"
        );
        compile_and_load!(device, CUDA_SRC_KMEANS, "kmeans", "kmeans_assignment");

        Ok(Self { device })
    }
}

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
impl GpuBackend for CudaBackend {
    fn name(&self) -> &str {
        "CUDA"
    }

    fn compute_distance(
        &self,
        query: &[f32],
        vectors: &[f32],
        dim: usize,
        metric: Metric,
    ) -> Result<Vec<f32>> {
        let (mod_name, kernel_name) = match metric {
            Metric::L2 => ("l2_distance", "l2_distance_kernel"),
            Metric::Cosine => ("cosine_distance", "cosine_distance_kernel"),
            Metric::InnerProduct => ("inner_product", "inner_product_kernel"),
            Metric::L1 => ("l1_distance", "l1_distance_kernel"),
            Metric::Hamming => ("hamming_distance", "hamming_distance_kernel"),
            Metric::Jaccard => ("jaccard_distance", "jaccard_distance_kernel"),
        };
        let n_vectors = vectors.len() / dim;
        let d_q = self.device.htod_copy(query.to_vec())?;
        let d_v = self.device.htod_copy(vectors.to_vec())?;
        let mut d_d = self.device.alloc_zeros::<f32>(n_vectors)?;
        let func = self
            .device
            .get_func(mod_name, kernel_name)
            .ok_or_else(|| anyhow::anyhow!("CUDA kernel {mod_name}::{kernel_name} not found"))?;

        const BLOCK: u32 = 256;
        let config = LaunchConfig {
            grid_dim: (n_vectors as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * std::mem::size_of::<f32>() as u32 * 2,
        };
        unsafe {
            func.launch(config, (&d_q, &d_v, &mut d_d, dim as u32, n_vectors as u32))?;
        }
        Ok(self.device.dtoh_sync_copy(&d_d)?)
    }

    fn compute_kmeans_assignment(
        &self,
        vectors: &[f32],
        centroids: &[f32],
        dim: usize,
    ) -> Result<Vec<u32>> {
        let n_vectors = vectors.len() / dim;
        let n_clusters = centroids.len() / dim;
        let d_v = self.device.htod_copy(vectors.to_vec())?;
        let d_c = self.device.htod_copy(centroids.to_vec())?;
        let mut d_out = self.device.alloc_zeros::<u32>(n_vectors)?;
        let func = self
            .device
            .get_func("kmeans", "kmeans_assignment")
            .ok_or_else(|| anyhow::anyhow!("CUDA kernel kmeans::kmeans_assignment not found"))?;
        let config = LaunchConfig::for_num_elems(n_vectors as u32);
        unsafe {
            func.launch(
                config,
                (
                    &d_v,
                    &d_c,
                    &mut d_out,
                    n_vectors as u32,
                    dim as u32,
                    n_clusters as u32,
                ),
            )?;
        }
        Ok(self.device.dtoh_sync_copy(&d_out)?)
    }
}
