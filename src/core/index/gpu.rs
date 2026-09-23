// Copyright (c) 2026 Richard Albright. All rights reserved.

use super::VectorMetric;
/// Hardware Acceleration Module for HyperStreamDB
///
/// This module provides support for various GPU backends:
/// - NVIDIA CUDA
/// - AMD ROCm
/// - Apple MPS (Metal Performance Shaders)
/// - Intel oneAPI / Level Zero
use anyhow::Result;
use once_cell::sync::Lazy;
use std::sync::Arc;

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
use cudarc::driver::{LaunchAsync, LaunchConfig};

// The global context to ensure PTX modules and device memory are not duplicated per thread.
// cudarc is Send+Sync and safely manages CUDA contexts internally.

// Keep a global fallback for legacy code paths that don't support thread-local contexts
static GLOBAL_GPU_CONTEXT: Lazy<parking_lot::RwLock<Option<ComputeContext>>> =
    Lazy::new(|| parking_lot::RwLock::new(None));

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ComputeBackend {
    #[default]
    Cpu,
    Cuda,
    Rocm,
    Mps,
    Intel,
}

pub trait GpuBackend: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &str;
    fn compute_distance(
        &self,
        query: &[f32],
        vectors: &[f32],
        dim: usize,
        metric: VectorMetric,
    ) -> Result<Vec<f32>>;
    fn compute_kmeans_assignment(
        &self,
        vectors: &[f32],
        centroids: &[f32],
        dim: usize,
    ) -> Result<Vec<u32>>;

    /// Batched packed-binary distance: one packed query vs N packed vectors.
    ///
    /// `query` is `dim_bytes` long; `vectors` is `n * dim_bytes`. Returns one
    /// distance per vector. Backends that don't implement it return an error so
    /// the caller falls back to CPU — this is how Metal is gated until its
    /// packed kernels land.
    fn compute_binary_distance(
        &self,
        _query: &[u8],
        _vectors: &[u8],
        _dim_bytes: usize,
        _metric: VectorMetric,
    ) -> Result<Vec<f32>> {
        anyhow::bail!("{} does not implement packed-binary distance", self.name())
    }
}

#[derive(Debug, Clone)]
pub struct ComputeContext {
    pub backend: ComputeBackend,
    pub device_id: i32,
    pub implementation: Option<Arc<dyn GpuBackend>>,
}

impl Default for ComputeContext {
    fn default() -> Self {
        Self {
            backend: ComputeBackend::Cpu,
            device_id: -1,
            implementation: Some(Arc::new(CpuBackend)),
        }
    }
}

// Resource Imports (Kernels)
// ============================================================================

#[cfg(target_os = "macos")]
static MSL_KMEANS: &str = include_str!("mps/kmeans_assignment.metal");
#[cfg(target_os = "macos")]
static MSL_L2: &str = include_str!("mps/l2_distance.metal");
#[cfg(target_os = "macos")]
static MSL_COSINE: &str = include_str!("mps/cosine_distance.metal");
#[cfg(target_os = "macos")]
static MSL_INNER_PRODUCT: &str = include_str!("mps/inner_product.metal");
#[cfg(target_os = "macos")]
static MSL_L1: &str = include_str!("mps/l1_distance.metal");
#[cfg(target_os = "macos")]
static MSL_HAMMING: &str = include_str!("mps/hamming_distance.metal");
#[cfg(target_os = "macos")]
static MSL_JACCARD: &str = include_str!("mps/jaccard_distance.metal");
#[cfg(target_os = "macos")]
static MSL_HAMMING_PACKED: &str = include_str!("mps/hamming_packed.metal");
#[cfg(target_os = "macos")]
static MSL_JACCARD_PACKED: &str = include_str!("mps/jaccard_packed.metal");

// CUDA kernels: embed .cu source at compile-time, JIT-compile at runtime via nvrtc.
// This eliminates the need for nvcc at build time — only libcuda.so is required at runtime.
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
static CUDA_SRC_KMEANS: &str = include_str!("cuda/kmeans_assignment.cu");
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
static CUDA_SRC_L2: &str = include_str!("cuda/l2_distance.cu");
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
static CUDA_SRC_COSINE: &str = include_str!("cuda/cosine_distance.cu");
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
static CUDA_SRC_INNER_PRODUCT: &str = include_str!("cuda/inner_product.cu");
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
static CUDA_SRC_L1: &str = include_str!("cuda/l1_distance.cu");
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
static CUDA_SRC_HAMMING: &str = include_str!("cuda/hamming_distance.cu");
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
static CUDA_SRC_JACCARD: &str = include_str!("cuda/jaccard_distance.cu");
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
static CUDA_SRC_HAMMING_PACKED: &str = include_str!("cuda/hamming_packed.cu");
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
static CUDA_SRC_JACCARD_PACKED: &str = include_str!("cuda/jaccard_packed.cu");

// Backend Implementations
// ============================================================================

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
#[derive(Debug)]
pub struct CudaBackend {
    device: Arc<cudarc::driver::CudaDevice>,
}

#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
impl CudaBackend {
    pub fn new(id: usize) -> Result<Self> {
        let device = cudarc::driver::CudaDevice::new(id)?;

        // JIT-compile .cu source to PTX at runtime via nvrtc (no nvcc needed at
        // build time). We compile with our own version-agnostic resolver
        // (`core::index::nvrtc`) because cudarc's loader probes a fixed
        // candidate list that predates CUDA 13, then hand the PTX to cudarc.
        macro_rules! compile_and_load {
            ($device:expr, $src:expr, $mod_name:expr, $kernel_name:expr) => {
                let ptx_src = crate::core::index::nvrtc::compile_ptx($src).map_err(|e| {
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
        metric: VectorMetric,
    ) -> Result<Vec<f32>> {
        let (mod_name, kernel_name) = match metric {
            VectorMetric::L2 => ("l2_distance", "l2_distance_kernel"),
            VectorMetric::Cosine => ("cosine_distance", "cosine_distance_kernel"),
            VectorMetric::InnerProduct => ("inner_product", "inner_product_kernel"),
            VectorMetric::L1 => ("l1_distance", "l1_distance_kernel"),
            VectorMetric::Hamming => ("hamming_distance", "hamming_distance_kernel"),
            VectorMetric::Jaccard => ("jaccard_distance", "jaccard_distance_kernel"),
        };
        let n_vectors = vectors.len() / dim;
        let d_q = self.device.htod_copy(query.to_vec())?;
        let d_v = self.device.htod_copy(vectors.to_vec())?;
        let mut d_d = self.device.alloc_zeros::<f32>(n_vectors)?;
        let func = self.device.get_func(mod_name, kernel_name).unwrap();
        // The kernels use one block per row with a shared-memory reduction, so
        // the grid must be `n_vectors` blocks (not `for_num_elems`, which packs
        // rows into 1024-thread blocks) and the shared memory must be sized for
        // the block — 2x for Jaccard's interleaved intersection/union slots.
        // Getting this wrong is an illegal memory access, not a wrong answer.
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
        let k = centroids.len() / dim;
        let d_v = self.device.htod_copy(vectors.to_vec())?;
        let d_c = self.device.htod_copy(centroids.to_vec())?;
        let mut d_l = self.device.alloc_zeros::<u32>(n_vectors)?;
        let func = self.device.get_func("kmeans", "kmeans_assignment").unwrap();
        let config = LaunchConfig::for_num_elems(n_vectors as u32);
        unsafe {
            func.launch(
                config,
                (&d_v, &d_c, &mut d_l, n_vectors as u32, k as u32, dim as u32),
            )?;
        }
        Ok(self.device.dtoh_sync_copy(&d_l)?)
    }
    fn compute_binary_distance(
        &self,
        query: &[u8],
        vectors: &[u8],
        dim_bytes: usize,
        metric: VectorMetric,
    ) -> Result<Vec<f32>> {
        let (mod_name, kernel_name) = match metric {
            VectorMetric::Hamming => ("hamming_packed", "hamming_packed_kernel"),
            VectorMetric::Jaccard => ("jaccard_packed", "jaccard_packed_kernel"),
            other => anyhow::bail!("CUDA packed-binary supports Hamming/Jaccard, not {other:?}"),
        };
        let n_vectors = vectors.len() / dim_bytes;
        let d_q = self.device.htod_copy(query.to_vec())?;
        let d_v = self.device.htod_copy(vectors.to_vec())?;
        let mut d_d = self.device.alloc_zeros::<f32>(n_vectors)?;
        let func = self.device.get_func(mod_name, kernel_name).unwrap();
        // Same shape as the dense kernels: one block per row, shared-memory
        // reduction (2x slots for Jaccard's interleaved intersection/union).
        const BLOCK: u32 = 256;
        let config = LaunchConfig {
            grid_dim: (n_vectors as u32, 1, 1),
            block_dim: (BLOCK, 1, 1),
            shared_mem_bytes: BLOCK * std::mem::size_of::<f32>() as u32 * 2,
        };
        unsafe {
            func.launch(
                config,
                (&d_q, &d_v, &mut d_d, dim_bytes as u32, n_vectors as u32),
            )?;
        }
        Ok(self.device.dtoh_sync_copy(&d_d)?)
    }
}

#[cfg(target_os = "macos")]
#[derive(Debug)]
pub struct MetalBackend {
    device: metal::Device,
    command_queue: metal::CommandQueue,
}

#[cfg(target_os = "macos")]
impl MetalBackend {
    pub fn new() -> Result<Self> {
        let device =
            metal::Device::system_default().ok_or_else(|| anyhow::anyhow!("No Metal device"))?;
        let command_queue = device.new_command_queue();
        Ok(Self {
            device,
            command_queue,
        })
    }
}

#[cfg(target_os = "macos")]
impl GpuBackend for MetalBackend {
    fn name(&self) -> &str {
        "Metal (MPS)"
    }
    fn compute_distance(
        &self,
        query: &[f32],
        vectors: &[f32],
        dim: usize,
        metric: VectorMetric,
    ) -> Result<Vec<f32>> {
        use metal::*;
        let (src, name) = match metric {
            VectorMetric::L2 => (MSL_L2, "l2_distance_kernel"),
            VectorMetric::Cosine => (MSL_COSINE, "cosine_distance_kernel"),
            VectorMetric::InnerProduct => (MSL_INNER_PRODUCT, "inner_product_kernel"),
            VectorMetric::L1 => (MSL_L1, "l1_distance_kernel"),
            VectorMetric::Hamming => (MSL_HAMMING, "hamming_distance_kernel"),
            VectorMetric::Jaccard => (MSL_JACCARD, "jaccard_distance_kernel"),
        };
        let n_vectors = vectors.len() / dim;
        let lib = self
            .device
            .new_library_with_source(src, &CompileOptions::new())
            .map_err(|e| anyhow::anyhow!(e))?;
        let func = lib
            .get_function(name, None)
            .map_err(|e| anyhow::anyhow!(e))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!(e))?;

        let q_buf = self.device.new_buffer_with_data(
            query.as_ptr() as *const _,
            (query.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let v_buf = self.device.new_buffer_with_data(
            vectors.as_ptr() as *const _,
            (vectors.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let o_buf = self.device.new_buffer(
            (n_vectors * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&q_buf), 0);
        enc.set_buffer(1, Some(&v_buf), 0);
        enc.set_buffer(2, Some(&o_buf), 0);
        enc.set_bytes(3, 4, &(dim as u32) as *const _ as *const _);
        // The dispatch rounds the thread count up to a multiple of the
        // threadgroup size; the kernel guards the tail against `n_vectors`.
        enc.set_bytes(4, 4, &(n_vectors as u32) as *const _ as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((n_vectors as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        unsafe {
            Ok(std::slice::from_raw_parts(o_buf.contents() as *const f32, n_vectors).to_vec())
        }
    }
    fn compute_kmeans_assignment(
        &self,
        vectors: &[f32],
        centroids: &[f32],
        dim: usize,
    ) -> Result<Vec<u32>> {
        use metal::*;
        let n_vectors = vectors.len() / dim;
        let k = centroids.len() / dim;
        let lib = self
            .device
            .new_library_with_source(MSL_KMEANS, &CompileOptions::new())
            .map_err(|e| anyhow::anyhow!(e))?;
        let func = lib
            .get_function("kmeans_assignment", None)
            .map_err(|e| anyhow::anyhow!(e))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!(e))?;

        let v_buf = self.device.new_buffer_with_data(
            vectors.as_ptr() as *const _,
            (vectors.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let c_buf = self.device.new_buffer_with_data(
            centroids.as_ptr() as *const _,
            (centroids.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let l_buf = self.device.new_buffer(
            (n_vectors * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&v_buf), 0);
        enc.set_buffer(1, Some(&c_buf), 0);
        enc.set_buffer(2, Some(&l_buf), 0);
        enc.set_bytes(3, 4, &(n_vectors as u32) as *const _ as *const _);
        enc.set_bytes(4, 4, &(k as u32) as *const _ as *const _);
        enc.set_bytes(5, 4, &(dim as u32) as *const _ as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((n_vectors as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        unsafe {
            Ok(std::slice::from_raw_parts(l_buf.contents() as *const u32, n_vectors).to_vec())
        }
    }

    fn compute_binary_distance(
        &self,
        query: &[u8],
        vectors: &[u8],
        dim_bytes: usize,
        metric: VectorMetric,
    ) -> Result<Vec<f32>> {
        use metal::*;

        let (src, name) = match metric {
            VectorMetric::Hamming => (MSL_HAMMING_PACKED, "hamming_packed_kernel"),
            VectorMetric::Jaccard => (MSL_JACCARD_PACKED, "jaccard_packed_kernel"),
            other => anyhow::bail!("Metal packed-binary supports Hamming/Jaccard, not {other:?}"),
        };
        let n_vectors = vectors.len() / dim_bytes;
        let lib = self
            .device
            .new_library_with_source(src, &CompileOptions::new())
            .map_err(|e| anyhow::anyhow!(e))?;
        let func = lib
            .get_function(name, None)
            .map_err(|e| anyhow::anyhow!(e))?;
        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow::anyhow!(e))?;

        let q_buf = self.device.new_buffer_with_data(
            query.as_ptr() as *const _,
            query.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let v_buf = self.device.new_buffer_with_data(
            vectors.as_ptr() as *const _,
            vectors.len() as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let o_buf = self.device.new_buffer(
            (n_vectors * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&q_buf), 0);
        enc.set_buffer(1, Some(&v_buf), 0);
        enc.set_buffer(2, Some(&o_buf), 0);
        enc.set_bytes(3, 4, &(dim_bytes as u32) as *const _ as *const _);
        // The dispatch rounds the thread count up to a multiple of the
        // threadgroup size; the kernel guards the tail against `n_vectors`.
        enc.set_bytes(4, 4, &(n_vectors as u32) as *const _ as *const _);
        enc.dispatch_thread_groups(
            MTLSize::new((n_vectors as u64 + 255) / 256, 1, 1),
            MTLSize::new(256, 1, 1),
        );
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        unsafe {
            Ok(std::slice::from_raw_parts(o_buf.contents() as *const f32, n_vectors).to_vec())
        }
    }
}

// WGPU Backend
// ============================================================================

#[cfg(all(target_os = "linux", feature = "wgpu"))]
#[derive(Debug)]
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    name: String,
}

#[cfg(all(target_os = "linux", feature = "wgpu"))]
impl WgpuBackend {
    pub fn new(display_name: &str, vendor_id: Option<u32>) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            ..Default::default()
        });

        let adapter = if let Some(vid) = vendor_id {
            instance
                .enumerate_adapters(wgpu::Backends::VULKAN)
                .into_iter()
                .find(|a| a.get_info().vendor == vid)
                .ok_or_else(|| {
                    anyhow::anyhow!("Failed to find WGPU adapter for vendor 0x{:04x}", vid)
                })?
        } else {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            }))
            .ok_or_else(|| anyhow::anyhow!("Failed to find WGPU adapter on Vulkan"))?
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Compute"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))?;

        let shader_src = include_str!("wgpu_kernel.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Distance Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(shader_src)),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Distance Compute Pipeline"),
            layout: None,
            module: &shader,
            entry_point: "main",
            compilation_options: Default::default(),
        });

        Ok(Self {
            device,
            queue,
            pipeline,
            name: display_name.to_string(),
        })
    }
}

#[cfg(all(target_os = "linux", feature = "wgpu"))]
impl GpuBackend for WgpuBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn compute_distance(
        &self,
        query: &[f32],
        vectors: &[f32],
        dim: usize,
        metric: VectorMetric,
    ) -> Result<Vec<f32>> {
        use wgpu::util::DeviceExt;

        fn as_u8_slice<T>(data: &[T]) -> &[u8] {
            unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
            }
        }

        let num_vectors = (vectors.len() / dim) as u32;
        let metric_type: u32 = match metric {
            VectorMetric::L2 => 0,
            VectorMetric::InnerProduct => 1,
            VectorMetric::Cosine => 2,
            VectorMetric::L1 => 3,
            VectorMetric::Hamming => 4,
            VectorMetric::Jaccard => 5,
        };

        let query_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Query Buffer"),
                contents: as_u8_slice(query),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let vectors_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vectors Buffer"),
                contents: as_u8_slice(vectors),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size =
            (num_vectors as usize * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let config_data = [dim as u32, num_vectors, metric_type, 0];
        let config_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Config Buffer"),
                contents: as_u8_slice(&config_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout = self.pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vectors_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let workgroups = num_vectors.div_ceil(64);
            cpass.dispatch_workgroups(workgroups, 1, 1);
        }

        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

        self.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            let result = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const f32, num_vectors as usize)
                    .to_vec()
            };
            drop(data);
            staging_buffer.unmap();
            Ok(result)
        } else {
            Err(anyhow::anyhow!("Failed to read WGPU output"))
        }
    }

    fn compute_kmeans_assignment(&self, _v: &[f32], _c: &[f32], _d: usize) -> Result<Vec<u32>> {
        super::ivf::simple_kmeans_assignment(_v, _c, _d)
    }

    fn compute_binary_distance(
        &self,
        query: &[u8],
        vectors: &[u8],
        dim_bytes: usize,
        metric: VectorMetric,
    ) -> Result<Vec<f32>> {
        use wgpu::util::DeviceExt;

        fn as_u8_slice<T>(data: &[T]) -> &[u8] {
            unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
            }
        }

        let metric_type: u32 = match metric {
            VectorMetric::Hamming => 0,
            VectorMetric::Jaccard => 1,
            other => anyhow::bail!("WGPU packed-binary supports Hamming/Jaccard, not {other:?}"),
        };

        // Pack bytes into little-endian u32 words, zero-padded to a 4-byte
        // multiple (WGSL storage buffers are u32-indexed).
        let dim_words = dim_bytes.div_ceil(4);
        let num_vectors = vectors.len() / dim_bytes;
        let mut q_words = vec![0u32; dim_words];
        for (i, &b) in query.iter().enumerate() {
            q_words[i / 4] |= (b as u32) << ((i % 4) * 8);
        }
        let mut v_words = vec![0u32; num_vectors * dim_words];
        for row in 0..num_vectors {
            for i in 0..dim_bytes {
                let b = vectors[row * dim_bytes + i];
                v_words[row * dim_words + i / 4] |= (b as u32) << ((i % 4) * 8);
            }
        }

        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Packed Binary Shader"),
                source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                    "wgpu_binary_kernel.wgsl"
                ))),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Packed Binary Pipeline"),
                layout: None,
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
            });

        let query_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Query Words"),
                contents: as_u8_slice(&q_words),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let vectors_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vector Words"),
                contents: as_u8_slice(&v_words),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_size = (num_vectors * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Output Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging Buffer"),
            size: output_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let config_data = [dim_words as u32, num_vectors as u32, metric_type, 0];
        let config_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Config Buffer"),
                contents: as_u8_slice(&config_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: vectors_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: config_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            cpass.set_pipeline(&pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups((num_vectors as u32).div_ceil(64), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_size);
        self.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
        self.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            let result = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const f32, num_vectors).to_vec()
            };
            drop(data);
            staging_buffer.unmap();
            Ok(result)
        } else {
            Err(anyhow::anyhow!("Failed to read WGPU packed-binary output"))
        }
    }
}

// ComputeContext & Dispatch
// ============================================================================

impl ComputeContext {
    pub fn from_backend(backend: ComputeBackend) -> Result<Self> {
        Self::from_backend_with_device(backend, 0)
    }

    pub fn from_backend_with_device(backend: ComputeBackend, device_id: usize) -> Result<Self> {
        let imp: Option<std::sync::Arc<dyn GpuBackend>> = match backend {
            ComputeBackend::Cpu => Some(std::sync::Arc::new(CpuBackend)),
            ComputeBackend::Cuda => {
                #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
                {
                    let b = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        CudaBackend::new(device_id)
                    }))
                    .map_err(|_| {
                        anyhow::anyhow!(
                            "CUDA backend panicked during initialization (missing library?)"
                        )
                    })?
                    .map_err(|e| anyhow::anyhow!("CUDA error: {}", e))?;
                    Some(std::sync::Arc::new(b))
                }
                #[cfg(not(all(not(target_os = "macos"), feature = "cuda")))]
                {
                    anyhow::bail!("CUDA not enabled (enable the 'cuda' feature)")
                }
            }
            ComputeBackend::Mps => {
                #[cfg(target_os = "macos")]
                {
                    Some(std::sync::Arc::new(MetalBackend::new()?))
                }
                #[cfg(not(target_os = "macos"))]
                {
                    anyhow::bail!("MPS not enabled")
                }
            }
            ComputeBackend::Rocm => {
                #[cfg(all(target_os = "linux", feature = "wgpu"))]
                {
                    Some(std::sync::Arc::new(WgpuBackend::new(
                        "WGPU_ROCm",
                        Some(0x1002),
                    )?))
                }
                #[cfg(not(all(target_os = "linux", feature = "wgpu")))]
                {
                    anyhow::bail!("ROCm not enabled on this platform (enable the 'wgpu' feature)")
                }
            }
            ComputeBackend::Intel => {
                #[cfg(all(target_os = "linux", feature = "wgpu"))]
                {
                    Some(std::sync::Arc::new(WgpuBackend::new(
                        "WGPU_Intel_XPU",
                        Some(0x8086),
                    )?))
                }
                #[cfg(not(all(target_os = "linux", feature = "wgpu")))]
                {
                    anyhow::bail!("Intel not enabled on this platform (enable the 'wgpu' feature)")
                }
            }
        };
        Ok(Self {
            backend,
            device_id: if backend == ComputeBackend::Cpu {
                -1
            } else {
                device_id as i32
            },
            implementation: imp,
        })
    }

    pub fn auto_detect() -> Self {
        {
            let read = GLOBAL_GPU_CONTEXT.read();
            if let Some(ctx) = &*read {
                return ctx.clone();
            }
        }

        let mut write = GLOBAL_GPU_CONTEXT.write();
        // Check again after acquiring lock
        if let Some(ctx) = &*write {
            return ctx.clone();
        }

        let ctx = Self::do_auto_detect();
        *write = Some(ctx.clone());
        ctx
    }

    fn do_auto_detect() -> Self {
        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        if let Ok(Ok(b)) =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| CudaBackend::new(0)))
        {
            return Self {
                backend: ComputeBackend::Cuda,
                device_id: 0,
                implementation: Some(Arc::new(b)),
            };
        }
        #[cfg(target_os = "macos")]
        if let Ok(b) = MetalBackend::new() {
            return Self {
                backend: ComputeBackend::Mps,
                device_id: 0,
                implementation: Some(Arc::new(b)),
            };
        }
        #[cfg(all(target_os = "linux", feature = "wgpu"))]
        if let Ok(b) = WgpuBackend::new("WGPU_ROCm", Some(0x1002)) {
            return Self {
                backend: ComputeBackend::Rocm,
                device_id: 0,
                implementation: Some(Arc::new(b)),
            };
        }
        #[cfg(all(target_os = "linux", feature = "wgpu"))]
        if let Ok(b) = WgpuBackend::new("WGPU_Intel_XPU", Some(0x8086)) {
            return Self {
                backend: ComputeBackend::Intel,
                device_id: 0,
                implementation: Some(Arc::new(b)),
            };
        }
        Self {
            backend: ComputeBackend::Cpu,
            device_id: -1,
            implementation: Some(Arc::new(CpuBackend)),
        }
    }

    pub fn from_device_str(device: &str) -> Result<Self> {
        let lower = device.to_lowercase();
        let trimmed = lower.trim();
        match trimmed {
            "cpu" => Ok(Self {
                backend: ComputeBackend::Cpu,
                device_id: -1,
                implementation: Some(Arc::new(CpuBackend)),
            }),
            "gpu" | "auto" => Ok(Self::auto_detect()),
            "cuda" => Self::from_backend(ComputeBackend::Cuda),
            _ if trimmed.starts_with("cuda:") => {
                let id = trimmed
                    .strip_prefix("cuda:")
                    .unwrap()
                    .parse::<usize>()
                    .unwrap_or(0);
                Self::from_backend_with_device(ComputeBackend::Cuda, id)
            }
            "mps" => Self::from_backend(ComputeBackend::Mps),
            _ if trimmed.starts_with("mps:") => Self::from_backend(ComputeBackend::Mps),
            "rocm" => Self::from_backend(ComputeBackend::Rocm),
            _ if trimmed.starts_with("rocm:") => {
                let id = trimmed
                    .strip_prefix("rocm:")
                    .unwrap()
                    .parse::<usize>()
                    .unwrap_or(0);
                Self::from_backend_with_device(ComputeBackend::Rocm, id)
            }
            "intel" => Self::from_backend(ComputeBackend::Intel),
            _ if trimmed.starts_with("intel:") => {
                let id = trimmed
                    .strip_prefix("intel:")
                    .unwrap()
                    .parse::<usize>()
                    .unwrap_or(0);
                Self::from_backend_with_device(ComputeBackend::Intel, id)
            }
            _ => anyhow::bail!("Unsupported device: {}", device),
        }
    }

    pub fn backend_name(&self) -> &'static str {
        match self.backend {
            ComputeBackend::Cpu => "cpu",
            ComputeBackend::Cuda => "cuda",
            ComputeBackend::Rocm => "rocm",
            ComputeBackend::Mps => "mps",
            ComputeBackend::Intel => "intel",
        }
    }

    pub fn is_gpu(&self) -> bool {
        self.backend != ComputeBackend::Cpu
    }

    pub fn is_available(&self) -> bool {
        match self.backend {
            ComputeBackend::Cpu => true,
            ComputeBackend::Cuda => {
                #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
                {
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        CudaBackend::new(self.device_id as usize).is_ok()
                            && cudarc::driver::CudaDevice::count()
                                .map(|c| c > 0)
                                .unwrap_or(false)
                    }))
                    .unwrap_or(false)
                }
                #[cfg(not(all(not(target_os = "macos"), feature = "cuda")))]
                {
                    false
                }
            }
            ComputeBackend::Mps => {
                #[cfg(target_os = "macos")]
                {
                    MetalBackend::new().is_ok()
                }
                #[cfg(not(target_os = "macos"))]
                {
                    false
                }
            }
            ComputeBackend::Rocm => {
                #[cfg(all(target_os = "linux", feature = "wgpu"))]
                {
                    WgpuBackend::new("Test", Some(0x1002)).is_ok()
                }
                #[cfg(not(all(target_os = "linux", feature = "wgpu")))]
                {
                    false
                }
            }
            ComputeBackend::Intel => {
                #[cfg(all(target_os = "linux", feature = "wgpu"))]
                {
                    WgpuBackend::new("Test", Some(0x8086)).is_ok()
                }
                #[cfg(not(all(target_os = "linux", feature = "wgpu")))]
                {
                    false
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct CpuBackend;
impl GpuBackend for CpuBackend {
    fn name(&self) -> &str {
        "CPU (SIMD)"
    }
    fn compute_distance(
        &self,
        q: &[f32],
        v: &[f32],
        d: usize,
        m: VectorMetric,
    ) -> Result<Vec<f32>> {
        compute_cpu(q, v, d, m)
    }
    fn compute_kmeans_assignment(&self, v: &[f32], c: &[f32], d: usize) -> Result<Vec<u32>> {
        super::ivf::simple_kmeans_assignment(v, c, d)
    }
}

pub const GPU_DISPATCH_THRESHOLD: usize = 50_000;

/// Compute pairwise distances between a query vector and a batch of vectors.
///
/// Automatically dispatches to the GPU backend (CUDA, ROCm, MPS) if the number
/// of vectors exceeds [`GPU_DISPATCH_THRESHOLD`] and a GPU context is available.
/// Otherwise falls back to CPU computation.
///
/// # Arguments
/// * `query` - The query vector
/// * `vectors` - Flat buffer of vectors to compare against (concatenated)
/// * `dim` - Dimensionality of each vector
/// * `metric` - Distance metric (L2, Cosine, InnerProduct, etc.)
///
/// # Errors
/// Returns an error if the GPU backend fails to execute or if buffer sizes are
/// inconsistent (i.e., `vectors.len()` is not a multiple of `dim`).
///
/// # Panics
/// Panics if `dim == 0` and `vectors` is non-empty.
pub fn compute_distance(
    query: &[f32],
    vectors: &[f32],
    dim: usize,
    metric: VectorMetric,
) -> Result<Vec<f32>> {
    let context = get_thread_gpu_context().unwrap_or_else(ComputeContext::auto_detect);
    let n = vectors.len().checked_div(dim).unwrap_or(0);
    if n < GPU_DISPATCH_THRESHOLD && context.backend != ComputeBackend::Cpu {
        return compute_cpu(query, vectors, dim, metric);
    }
    if let Some(imp) = &context.implementation {
        return imp.compute_distance(query, vectors, dim, metric);
    }
    compute_cpu(query, vectors, dim, metric)
}

/// Assign each vector to its nearest centroid using k-means.
///
/// Dispatches to the GPU backend if available and the vector count is large
/// enough. Otherwise falls back to a simple CPU implementation.
///
/// # Arguments
/// * `vectors` - Flat buffer of vectors
/// * `centroids` - Flat buffer of centroid vectors
/// * `dim` - Dimensionality of each vector
///
/// # Errors
/// Returns an error if the GPU backend fails or if buffer sizes are inconsistent.
pub fn compute_kmeans_assignment(
    vectors: &[f32],
    centroids: &[f32],
    dim: usize,
) -> Result<Vec<u32>> {
    let context = get_thread_gpu_context().unwrap_or_else(ComputeContext::auto_detect);
    if let Some(imp) = &context.implementation {
        return imp.compute_kmeans_assignment(vectors, centroids, dim);
    }
    super::ivf::simple_kmeans_assignment(vectors, centroids, dim)
}

fn compute_cpu(q: &[f32], v: &[f32], d: usize, m: VectorMetric) -> Result<Vec<f32>> {
    let n = v.len().checked_div(d).unwrap_or(0);
    let mut dists = Vec::with_capacity(n);
    for i in 0..n {
        let span = &v[i * d..(i + 1) * d];
        dists.push(match m {
            VectorMetric::L2 => crate::core::index::distance::l2_distance(q, span),
            VectorMetric::Cosine => crate::core::index::distance::cosine_distance(q, span),
            VectorMetric::InnerProduct => crate::core::index::distance::dot_product(q, span),
            VectorMetric::L1 => crate::core::index::distance::l1_distance(q, span),
            VectorMetric::Hamming => crate::core::index::distance::hamming_distance(q, span),
            VectorMetric::Jaccard => crate::core::index::distance::jaccard_distance(q, span),
        });
    }
    Ok(dists)
}

/// Batched packed-binary distance (Hamming/Jaccard) with GPU dispatch.
///
/// One packed `query` (`dim_bytes` long) against `n` packed `vectors`
/// (`n * dim_bytes`). Backends without a packed kernel (Metal, until its kernels
/// land) error out and are transparently replaced by the CPU reference.
pub fn compute_binary_distance(
    query: &[u8],
    vectors: &[u8],
    dim_bytes: usize,
    metric: VectorMetric,
) -> Result<Vec<f32>> {
    let context = get_thread_gpu_context().unwrap_or_else(ComputeContext::auto_detect);
    let n = vectors.len().checked_div(dim_bytes).unwrap_or(0);
    if n < GPU_DISPATCH_THRESHOLD && context.backend != ComputeBackend::Cpu {
        return compute_binary_cpu(query, vectors, dim_bytes, metric);
    }
    if let Some(imp) = &context.implementation {
        if let Ok(out) = imp.compute_binary_distance(query, vectors, dim_bytes, metric) {
            return Ok(out);
        }
        // Backend has no packed kernel -> fall through to CPU.
    }
    compute_binary_cpu(query, vectors, dim_bytes, metric)
}

fn compute_binary_cpu(
    query: &[u8],
    vectors: &[u8],
    dim_bytes: usize,
    metric: VectorMetric,
) -> Result<Vec<f32>> {
    let n = vectors.len().checked_div(dim_bytes).unwrap_or(0);
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let v = &vectors[i * dim_bytes..(i + 1) * dim_bytes];
        out.push(match metric {
            VectorMetric::Hamming => {
                crate::core::index::distance::hamming_distance_packed(query, v) as f32
            }
            VectorMetric::Jaccard => {
                crate::core::index::distance::jaccard_distance_packed(query, v)
            }
            other => {
                anyhow::bail!("packed-binary distance supports Hamming/Jaccard, not {other:?}")
            }
        });
    }
    Ok(out)
}

/// Set the global GPU context.
pub fn set_thread_gpu_context(ctx: Option<ComputeContext>) {
    *GLOBAL_GPU_CONTEXT.write() = ctx;
}

/// Retrieve the current GPU context.
pub fn get_thread_gpu_context() -> Option<ComputeContext> {
    let lock = GLOBAL_GPU_CONTEXT.read();
    lock.clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu() {
        let q = vec![1.0, 0.0];
        let v = vec![1.0, 0.0, 0.0, 1.0];
        let d = compute_distance(&q, &v, 2, VectorMetric::L2).unwrap();
        assert_eq!(d[0], 0.0);
    }

    #[test]
    fn test_from_device_str_cpu() {
        let ctx = ComputeContext::from_device_str("cpu").unwrap();
        assert_eq!(ctx.backend, ComputeBackend::Cpu);
        assert_eq!(ctx.device_id, -1);
        assert_eq!(ctx.backend_name(), "cpu");
        assert!(!ctx.is_gpu());
        assert!(ctx.is_available());
    }

    #[test]
    fn test_from_device_str_auto() {
        let ctx = ComputeContext::from_device_str("auto").unwrap();
        // auto should resolve to CPU or an active GPU
        assert!(!ctx.backend_name().is_empty());
    }

    #[test]
    fn test_from_device_str_invalid() {
        assert!(ComputeContext::from_device_str("nonexistent_device").is_err());
    }

    #[test]
    fn test_backend_name_and_is_gpu() {
        let cpu = ComputeContext {
            backend: ComputeBackend::Cpu,
            device_id: -1,
            implementation: Some(Arc::new(CpuBackend)),
        };
        assert_eq!(cpu.backend_name(), "cpu");
        assert!(!cpu.is_gpu());
    }

    /// Regression test for the CUDA 13 nvrtc discovery bug: on a machine with a
    /// CUDA device, the JIT must actually compile (cudarc's fixed candidate list
    /// missed `libnvrtc.so.13`, so this used to panic into a CPU fallback).
    #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
    #[test]
    fn cuda_backend_jit_compiles_when_a_device_is_present() {
        // Skip on machines without a CUDA device (e.g. CI).
        let has_device = cudarc::driver::CudaDevice::count()
            .map(|c| c > 0)
            .unwrap_or(false);
        if !has_device {
            return;
        }
        // Make nvrtc discoverable: the environment, or a repo-local venv (local
        // dev). Skip cleanly if neither is available.
        if crate::core::index::nvrtc::resolve_nvrtc().is_none() {
            if let Some(p) = crate::core::index::nvrtc::dev_repo_venv_nvrtc() {
                std::env::set_var("HDB_NVRTC_PATH", &p);
            }
        }
        if crate::core::index::nvrtc::resolve_nvrtc().is_none() {
            return;
        }
        // A *usable* device is required. CI installs CUDA stubs (`libcuda.so`
        // without a GPU): they report a device count but cannot create a
        // context, so treat an init failure as a skip, not a test failure. On a
        // real GPU this asserts the JIT path compiles.
        let Ok(backend) = CudaBackend::new(0) else {
            eprintln!("skipping: no usable CUDA device (stub driver?)");
            return;
        };
        assert_eq!(backend.name(), "CUDA");
    }

    // ========================================================================
    // Cross-backend correctness harness
    // ========================================================================
    //
    // CPU is the gold source: every available backend must agree with the CPU
    // reference within tolerance. Backends absent from this machine are skipped,
    // so the same test runs everywhere and validates whatever hardware exists:
    //   - CUDA  -> NVIDIA (local / self-hosted runner)
    //   - WGPU  -> any Vulkan adapter, *including NVIDIA*, so the portable WGSL
    //              kernel is validated even on an NVIDIA box (different driver
    //              path from CUDA)
    //   - Metal -> macOS (GitHub's macos-14 runners, or a local Mac)
    //   - ROCm/Intel native -> self-hosted AMD/Intel runners; the WGPU test
    //              covers the same WGSL kernel
    //
    // Run it (the `--nocapture` line prints which backends were exercised):
    //
    //   cargo test --lib --features cuda,wgpu,pollster cross_backend -- --nocapture
    //
    // `cuda` needs an NVIDIA GPU, `wgpu`/`pollster` a Vulkan adapter; drop the
    // features you don't have — the harness skips what's absent.

    /// Backends available on this machine, CPU first (the gold source).
    fn available_backends() -> Vec<(&'static str, Arc<dyn GpuBackend>)> {
        #[allow(unused_mut)]
        let mut out: Vec<(&'static str, Arc<dyn GpuBackend>)> = vec![("cpu", Arc::new(CpuBackend))];
        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        {
            // Under `cargo test` the interpreter's site-packages isn't reported,
            // so point the resolver at a repo-local venv if one exists.
            if crate::core::index::nvrtc::resolve_nvrtc().is_none() {
                if let Some(p) = crate::core::index::nvrtc::dev_repo_venv_nvrtc() {
                    std::env::set_var("HDB_NVRTC_PATH", &p);
                }
            }
            if let Ok(b) = CudaBackend::new(0) {
                out.push(("cuda", Arc::new(b)));
            }
        }
        #[cfg(target_os = "macos")]
        if let Ok(b) = MetalBackend::new() {
            out.push(("mps", Arc::new(b)));
        }
        // Vendor-agnostic WGPU: exercises the portable WGSL kernel on whatever
        // Vulkan adapter exists (NVIDIA included).
        #[cfg(all(target_os = "linux", feature = "wgpu"))]
        if let Ok(b) = WgpuBackend::new("wgpu", None) {
            out.push(("wgpu", Arc::new(b)));
        }
        out
    }

    fn random_vectors(n: usize, dim: usize, seed: u64, binary: bool) -> (Vec<f32>, Vec<f32>) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let gen = |rng: &mut rand::rngs::StdRng| {
            if binary {
                if rng.gen_bool(0.5) {
                    1.0
                } else {
                    0.0
                }
            } else {
                rng.gen_range(-1.0f32..1.0f32)
            }
        };
        let query: Vec<f32> = (0..dim).map(|_| gen(&mut rng)).collect();
        let vectors: Vec<f32> = (0..n * dim).map(|_| gen(&mut rng)).collect();
        (query, vectors)
    }

    /// Assert every available GPU backend agrees with the CPU reference.
    fn assert_backend_matches_cpu(metric: VectorMetric, dim: usize, n: usize) {
        let backends = available_backends();
        let binary = matches!(metric, VectorMetric::Hamming | VectorMetric::Jaccard);
        let (query, vectors) = random_vectors(n, dim, 0xC0FFEE, binary);
        let gold = compute_cpu(&query, &vectors, dim, metric).expect("cpu gold");

        for (name, backend) in &backends {
            if *name == "cpu" {
                continue;
            }
            let got = backend
                .compute_distance(&query, &vectors, dim, metric)
                .unwrap_or_else(|e| panic!("{name} {metric:?} failed: {e}"));
            assert_eq!(got.len(), gold.len(), "{name} {metric:?}: length mismatch");
            for (i, (g, c)) in gold.iter().zip(got.iter()).enumerate() {
                let tol = 1e-3 * g.abs().max(1.0);
                assert!(
                    (g - c).abs() <= tol,
                    "{name} {metric:?} dim={dim} n={n} idx={i}: cpu={g} gpu={c}"
                );
            }
        }
    }

    #[test]
    fn cross_backend_matches_cpu_all_metrics() {
        for metric in [
            VectorMetric::L2,
            VectorMetric::Cosine,
            VectorMetric::InnerProduct,
            VectorMetric::L1,
            VectorMetric::Hamming,
            VectorMetric::Jaccard,
        ] {
            assert_backend_matches_cpu(metric, 128, 1_000);
        }
    }

    fn random_packed(n: usize, dim_bytes: usize, seed: u64) -> (Vec<u8>, Vec<u8>) {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let query: Vec<u8> = (0..dim_bytes).map(|_| rng.gen()).collect();
        let vectors: Vec<u8> = (0..n * dim_bytes).map(|_| rng.gen()).collect();
        (query, vectors)
    }

    /// Assert every backend with a packed kernel agrees with the CPU reference.
    /// Backends without one (Metal, until its kernels land) error and are
    /// skipped — that is the gate.
    fn assert_binary_backend_matches_cpu(metric: VectorMetric, dim_bytes: usize, n: usize) {
        let backends = available_backends();
        let (query, vectors) = random_packed(n, dim_bytes, 0xBEEF);
        let gold = compute_binary_cpu(&query, &vectors, dim_bytes, metric).expect("cpu gold");

        for (name, backend) in &backends {
            if *name == "cpu" {
                continue;
            }
            let Ok(got) = backend.compute_binary_distance(&query, &vectors, dim_bytes, metric)
            else {
                continue; // no packed kernel on this backend
            };
            assert_eq!(
                got.len(),
                gold.len(),
                "{name} {metric:?} packed: length mismatch"
            );
            for (i, (g, c)) in gold.iter().zip(got.iter()).enumerate() {
                let tol = 1e-3 * g.abs().max(1.0);
                assert!(
                    (g - c).abs() <= tol,
                    "{name} {metric:?} packed dim_bytes={dim_bytes} n={n} idx={i}: cpu={g} gpu={c}"
                );
            }
        }
    }

    #[test]
    fn cross_backend_binary_matches_cpu() {
        for metric in [VectorMetric::Hamming, VectorMetric::Jaccard] {
            // 16 bytes (word-aligned) and 13 bytes (needs zero-padding).
            assert_binary_backend_matches_cpu(metric, 16, 1_000);
            assert_binary_backend_matches_cpu(metric, 13, 1_000);
        }
    }

    /// Reports which backends this machine exercised (visible with `--nocapture`).
    #[test]
    fn cross_backend_reports_available_backends() {
        let names: Vec<&str> = available_backends().iter().map(|(n, _)| *n).collect();
        eprintln!("cross-backend harness: available backends = {names:?}");
        assert!(names.contains(&"cpu"), "cpu must always be available");
    }
}
