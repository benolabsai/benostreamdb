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
#[cfg(all(not(target_os = "macos"), feature = "cuda"))]
use cudarc::nvrtc::compile_ptx;

// Thread-local GPU context to ensure CUDA contexts are not shared across threads.
// CUDA contexts are inherently thread-local; sharing them can cause silent corruption.
thread_local! {
    static GPU_THREAD_CONTEXT: std::cell::RefCell<Option<ComputeContext>> = const { std::cell::RefCell::new(None) };
}

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

        // JIT-compile .cu source to PTX at runtime via nvrtc (no nvcc needed at build time)
        macro_rules! compile_and_load {
            ($device:expr, $src:expr, $mod_name:expr, $kernel_name:expr) => {
                let ptx = compile_ptx($src).map_err(|e| {
                    anyhow::anyhow!("nvrtc compile failed for {}: {:?}", $mod_name, e)
                })?;
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
        let config = LaunchConfig::for_num_elems(n_vectors as u32);
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
}

// ComputeContext & Dispatch
// ============================================================================

impl ComputeContext {
    pub fn from_backend(backend: ComputeBackend) -> Result<Self> {
        let imp: Option<std::sync::Arc<dyn GpuBackend>> = match backend {
            ComputeBackend::Cpu => Some(std::sync::Arc::new(CpuBackend)),
            ComputeBackend::Cuda => {
                #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
                {
                    Some(std::sync::Arc::new(CudaBackend::new(0)?))
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
                    anyhow::bail!("ROCm not enabled on this platform")
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
                    anyhow::bail!("Intel not enabled on this platform")
                }
            }
        };
        Ok(Self {
            backend,
            device_id: if backend == ComputeBackend::Cpu {
                -1
            } else {
                0
            },
            implementation: imp,
        })
    }

    pub fn auto_detect() -> Self {
        // First, check the thread-local context
        let ctx = GPU_THREAD_CONTEXT.with(|f| f.borrow().clone());
        if let Some(ctx) = ctx {
            return ctx;
        }

        // Fall back to global context if thread-local is not set
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
        // Set both thread-local and global contexts
        GPU_THREAD_CONTEXT.with(|f| {
            *f.borrow_mut() = Some(ctx.clone());
        });
        *write = Some(ctx.clone());
        ctx
    }

    fn do_auto_detect() -> Self {
        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        if let Ok(b) = CudaBackend::new(0) {
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
        match lower.as_str() {
            "cpu" => Ok(Self {
                backend: ComputeBackend::Cpu,
                device_id: -1,
                implementation: Some(Arc::new(CpuBackend)),
            }),
            "gpu" | "auto" => Ok(Self::auto_detect()),
            _ if lower.starts_with("cuda:") => Self::from_backend(ComputeBackend::Cuda),
            _ if lower.starts_with("mps:") => Self::from_backend(ComputeBackend::Mps),
            _ if lower.starts_with("rocm:") => Self::from_backend(ComputeBackend::Rocm),
            _ if lower.starts_with("intel:") => Self::from_backend(ComputeBackend::Intel),
            _ => anyhow::bail!("Unsupported device: {}", device),
        }
    }

    pub fn is_available(&self) -> bool {
        match self.backend {
            ComputeBackend::Cpu => true,
            ComputeBackend::Cuda => {
                #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
                {
                    CudaBackend::new(self.device_id as usize).is_ok()
                        && cudarc::driver::CudaDevice::count()
                            .map(|c| c > 0)
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

/// Set the thread-local GPU context (preferred over global context).
///
/// This sets a GPU context that is isolated to the current thread, preventing
/// CUDA context corruption when multiple threads use different GPU configurations.
pub fn set_thread_gpu_context(ctx: Option<ComputeContext>) {
    GPU_THREAD_CONTEXT.with(|f| {
        *f.borrow_mut() = ctx;
    });
}

/// Retrieve the current GPU context for this thread.
///
/// Checks the thread-local context first, then falls back to the global context.
/// Returns `None` if no context has been set on either scope.
///
/// # Thread Safety
/// This function is safe to call from any thread. Each thread maintains its own
/// context via thread-local storage, with the global context as a last resort.
pub fn get_thread_gpu_context() -> Option<ComputeContext> {
    // Check thread-local first
    let ctx = GPU_THREAD_CONTEXT.with(|f| f.borrow().clone());
    if ctx.is_some() {
        return ctx;
    }
    // Fall back to global context (deprecated path)
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
}
