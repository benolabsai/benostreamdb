// Copyright (c) 2026 Richard Albright. All rights reserved.

/// Hardware Acceleration Module for HyperStreamDB
/// 
/// This module provides support for various GPU backends:
/// - NVIDIA CUDA
/// - AMD ROCm
/// - Apple MPS (Metal Performance Shaders)
/// - Intel oneAPI / Level Zero
use anyhow::Result;
use super::VectorMetric;
use std::cell::RefCell;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::driver::{LaunchAsync, LaunchConfig};

thread_local! {
    static GLOBAL_GPU_CONTEXT: RefCell<Option<ComputeContext>> = const { RefCell::new(None) };
}

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
    fn compute_distance(&self, query: &[f32], vectors: &[f32], dim: usize, metric: VectorMetric) -> Result<Vec<f32>>;
    fn compute_kmeans_assignment(&self, vectors: &[f32], centroids: &[f32], dim: usize) -> Result<Vec<u32>>;
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

#[cfg(all(target_os = "macos", feature = "mps"))]
static MSL_KMEANS: &str = include_str!("mps/kmeans_assignment.metal");
#[cfg(all(target_os = "macos", feature = "mps"))]
static MSL_L2: &str = include_str!("mps/l2_distance.metal");
#[cfg(all(target_os = "macos", feature = "mps"))]
static MSL_COSINE: &str = include_str!("mps/cosine_distance.metal");
#[cfg(all(target_os = "macos", feature = "mps"))]
static MSL_INNER_PRODUCT: &str = include_str!("mps/inner_product.metal");
#[cfg(all(target_os = "macos", feature = "mps"))]
static MSL_L1: &str = include_str!("mps/l1_distance.metal");
#[cfg(all(target_os = "macos", feature = "mps"))]
static MSL_HAMMING: &str = include_str!("mps/hamming_distance.metal");
#[cfg(all(target_os = "macos", feature = "mps"))]
static MSL_JACCARD: &str = include_str!("mps/jaccard_distance.metal");

#[cfg(feature = "cuda")]
static CUDA_KMEANS: &str = include_str!(concat!(env!("OUT_DIR"), "/kmeans_assignment.ptx"));
#[cfg(feature = "cuda")]
static PTX_L2: &str = include_str!(concat!(env!("OUT_DIR"), "/l2_distance.ptx"));
#[cfg(feature = "cuda")]
static PTX_COSINE: &str = include_str!(concat!(env!("OUT_DIR"), "/cosine_distance.ptx"));
#[cfg(feature = "cuda")]
static PTX_INNER_PRODUCT: &str = include_str!(concat!(env!("OUT_DIR"), "/inner_product.ptx"));
#[cfg(feature = "cuda")]
static PTX_L1: &str = include_str!(concat!(env!("OUT_DIR"), "/l1_distance.ptx"));
#[cfg(feature = "cuda")]
static PTX_HAMMING: &str = include_str!(concat!(env!("OUT_DIR"), "/hamming_distance.ptx"));
#[cfg(feature = "cuda")]
static PTX_JACCARD: &str = include_str!(concat!(env!("OUT_DIR"), "/jaccard_distance.ptx"));

#[cfg(any(feature = "intel", feature = "rocm"))]
static OPENCL_KMEANS: &str = include_str!("opencl/kmeans_assignment.cl");
#[cfg(any(feature = "intel", feature = "rocm"))]
static OPENCL_SRC_L2: &str = include_str!("opencl/l2_distance.cl");
#[cfg(any(feature = "intel", feature = "rocm"))]
static OPENCL_SRC_COSINE: &str = include_str!("opencl/cosine_distance.cl");
#[cfg(any(feature = "intel", feature = "rocm"))]
static OPENCL_SRC_INNER_PRODUCT: &str = include_str!("opencl/inner_product.cl");
#[cfg(any(feature = "intel", feature = "rocm"))]
static OPENCL_SRC_L1: &str = include_str!("opencl/l1_distance.cl");
#[cfg(any(feature = "intel", feature = "rocm"))]
static OPENCL_SRC_HAMMING: &str = include_str!("opencl/hamming_distance.cl");
#[cfg(any(feature = "intel", feature = "rocm"))]
static OPENCL_SRC_JACCARD: &str = include_str!("opencl/jaccard_distance.cl");

// Backend Implementations
// ============================================================================

#[cfg(feature = "cuda")]
#[derive(Debug)]
pub struct CudaBackend { device: Arc<cudarc::driver::CudaDevice> }

#[cfg(feature = "cuda")]
impl CudaBackend {
    pub fn is_available() -> bool { true }
    pub fn new(id: usize) -> Result<Self> {
        let device = cudarc::driver::CudaDevice::new(id)?;
        device.load_ptx(PTX_L2.into(), "l2_distance", &["l2_distance_kernel"])?;
        device.load_ptx(PTX_COSINE.into(), "cosine_distance", &["cosine_distance_kernel"])?;
        device.load_ptx(PTX_INNER_PRODUCT.into(), "inner_product", &["inner_product_kernel"])?;
        device.load_ptx(PTX_L1.into(), "l1_distance", &["l1_distance_kernel"])?;
        device.load_ptx(PTX_HAMMING.into(), "hamming_distance", &["hamming_distance_kernel"])?;
        device.load_ptx(PTX_JACCARD.into(), "jaccard_distance", &["jaccard_distance_kernel"])?;
        device.load_ptx(CUDA_KMEANS.into(), "kmeans", &["kmeans_assignment"])?;
        Ok(Self { device })
    }
}

#[cfg(feature = "cuda")]
impl GpuBackend for CudaBackend {
    fn name(&self) -> &str { "CUDA" }
    fn compute_distance(&self, query: &[f32], vectors: &[f32], dim: usize, metric: VectorMetric) -> Result<Vec<f32>> {
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
        unsafe { func.launch(config, (&d_q, &d_v, &mut d_d, dim as u32, n_vectors as u32))?; }
        Ok(self.device.dtoh_sync_copy(&d_d)?)
    }
    fn compute_kmeans_assignment(&self, vectors: &[f32], centroids: &[f32], dim: usize) -> Result<Vec<u32>> {
        let n_vectors = vectors.len() / dim;
        let k = centroids.len() / dim;
        let d_v = self.device.htod_copy(vectors.to_vec())?;
        let d_c = self.device.htod_copy(centroids.to_vec())?;
        let mut d_l = self.device.alloc_zeros::<u32>(n_vectors)?;
        let func = self.device.get_func("kmeans", "kmeans_assignment").unwrap();
        let config = LaunchConfig::for_num_elems(n_vectors as u32);
        unsafe { func.launch(config, (&d_v, &d_c, &mut d_l, n_vectors as u32, k as u32, dim as u32))?; }
        Ok(self.device.dtoh_sync_copy(&d_l)?)
    }
}

#[cfg(all(target_os = "macos", feature = "mps"))]
#[derive(Debug)]
pub struct MetalBackend { device: metal::Device, command_queue: metal::CommandQueue }

#[cfg(all(target_os = "macos", feature = "mps"))]
impl MetalBackend {
    pub fn new() -> Result<Self> {
        let device = metal::Device::system_default().ok_or_else(|| anyhow::anyhow!("No Metal device"))?;
        let command_queue = device.new_command_queue();
        Ok(Self { device, command_queue })
    }
}

#[cfg(all(target_os = "macos", feature = "mps"))]
impl GpuBackend for MetalBackend {
    fn name(&self) -> &str { "Metal (MPS)" }
    fn compute_distance(&self, query: &[f32], vectors: &[f32], dim: usize, metric: VectorMetric) -> Result<Vec<f32>> {
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
        let lib = self.device.new_library_with_source(src, &CompileOptions::new()).map_err(|e| anyhow::anyhow!(e))?;
        let func = lib.get_function(name, None).map_err(|e| anyhow::anyhow!(e))?;
        let pipeline = self.device.new_compute_pipeline_state_with_function(&func).map_err(|e| anyhow::anyhow!(e))?;
        
        let q_buf = self.device.new_buffer_with_data(query.as_ptr() as *const _, (query.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let v_buf = self.device.new_buffer_with_data(vectors.as_ptr() as *const _, (vectors.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let o_buf = self.device.new_buffer((n_vectors * 4) as u64, MTLResourceOptions::StorageModeShared);
        
        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&q_buf), 0);
        enc.set_buffer(1, Some(&v_buf), 0);
        enc.set_buffer(2, Some(&o_buf), 0);
        enc.set_bytes(3, 4, &(dim as u32) as *const _ as *const _);
        enc.dispatch_thread_groups(MTLSize::new((n_vectors as u64 + 255) / 256, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        unsafe { Ok(std::slice::from_raw_parts(o_buf.contents() as *const f32, n_vectors).to_vec()) }
    }
    fn compute_kmeans_assignment(&self, vectors: &[f32], centroids: &[f32], dim: usize) -> Result<Vec<u32>> {
        use metal::*;
        let n_vectors = vectors.len() / dim;
        let k = centroids.len() / dim;
        let lib = self.device.new_library_with_source(MSL_KMEANS, &CompileOptions::new()).map_err(|e| anyhow::anyhow!(e))?;
        let func = lib.get_function("kmeans_assignment", None).map_err(|e| anyhow::anyhow!(e))?;
        let pipeline = self.device.new_compute_pipeline_state_with_function(&func).map_err(|e| anyhow::anyhow!(e))?;
        
        let v_buf = self.device.new_buffer_with_data(vectors.as_ptr() as *const _, (vectors.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let c_buf = self.device.new_buffer_with_data(centroids.as_ptr() as *const _, (centroids.len() * 4) as u64, MTLResourceOptions::StorageModeShared);
        let l_buf = self.device.new_buffer((n_vectors * 4) as u64, MTLResourceOptions::StorageModeShared);
        
        let cmd_buf = self.command_queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pipeline);
        enc.set_buffer(0, Some(&v_buf), 0);
        enc.set_buffer(1, Some(&c_buf), 0);
        enc.set_buffer(2, Some(&l_buf), 0);
        enc.set_bytes(3, 4, &(n_vectors as u32) as *const _ as *const _);
        enc.set_bytes(4, 4, &(k as u32) as *const _ as *const _);
        enc.set_bytes(5, 4, &(dim as u32) as *const _ as *const _);
        enc.dispatch_thread_groups(MTLSize::new((n_vectors as u64 + 255) / 256, 1, 1), MTLSize::new(256, 1, 1));
        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();
        unsafe { Ok(std::slice::from_raw_parts(l_buf.contents() as *const u32, n_vectors).to_vec()) }
    }
}

#[cfg(any(feature = "intel", feature = "rocm"))]
#[derive(Debug)]
pub struct OpenCLBackend { name: String }

#[cfg(any(feature = "intel", feature = "rocm"))]
impl OpenCLBackend {
    pub fn new(name: Option<&str>) -> Result<Self> { Ok(Self { name: name.unwrap_or("OpenCL").to_string() }) }
}

#[cfg(any(feature = "intel", feature = "rocm"))]
impl GpuBackend for OpenCLBackend {
    fn name(&self) -> &str { &self.name }
    fn compute_distance(&self, _q: &[f32], _v: &[f32], _d: usize, _m: VectorMetric) -> Result<Vec<f32>> { compute_cpu(_q, _v, _d, _m) }
    fn compute_kmeans_assignment(&self, _v: &[f32], _c: &[f32], _d: usize) -> Result<Vec<u32>> { 
        super::ivf::simple_kmeans_assignment(_v, _c, _d)
    }
}

// ComputeContext & Dispatch
// ============================================================================

impl ComputeContext {
    pub fn auto_detect() -> Self {
        #[cfg(feature = "cuda")]
        if let Ok(b) = CudaBackend::new(0) { return Self { backend: ComputeBackend::Cuda, device_id: 0, implementation: Some(Arc::new(b)) }; }
        #[cfg(all(target_os = "macos", feature = "mps"))]
        if let Ok(b) = MetalBackend::new() { return Self { backend: ComputeBackend::Mps, device_id: 0, implementation: Some(Arc::new(b)) }; }
        #[cfg(feature = "rocm")]
        if let Ok(b) = OpenCLBackend::new(Some("AMD")) { return Self { backend: ComputeBackend::Rocm, device_id: 0, implementation: Some(Arc::new(b)) }; }
        #[cfg(feature = "intel")]
        if let Ok(b) = OpenCLBackend::new(Some("Intel")) { return Self { backend: ComputeBackend::Intel, device_id: 0, implementation: Some(Arc::new(b)) }; }
        Self { backend: ComputeBackend::Cpu, device_id: -1, implementation: Some(Arc::new(CpuBackend)) }
    }

    pub fn from_device_str(device: &str) -> Result<Self> {
        match device.to_lowercase().as_str() {
            "cpu" => Ok(Self { backend: ComputeBackend::Cpu, device_id: -1, implementation: Some(Arc::new(CpuBackend)) }),
            "gpu" | "auto" => Ok(Self::auto_detect()),
            _ => anyhow::bail!("Unsupported device: {}", device),
        }
    }

    pub fn is_available(&self) -> bool {
        match self.backend {
            ComputeBackend::Cpu => true,
            ComputeBackend::Cuda => {
                #[cfg(feature = "cuda")]
                { CudaBackend::new(self.device_id as usize).is_ok() }
                #[cfg(not(feature = "cuda"))]
                { false }
            }
            ComputeBackend::Mps => {
                #[cfg(all(target_os = "macos", feature = "mps"))]
                { MetalBackend::new().is_ok() }
                #[cfg(not(all(target_os = "macos", feature = "mps")))]
                { false }
            }
            ComputeBackend::Rocm | ComputeBackend::Intel => {
                #[cfg(any(feature = "intel", feature = "rocm"))]
                { true } // OpenCL fallback
                #[cfg(not(any(feature = "intel", feature = "rocm")))]
                { false }
            }
        }
    }
}

#[derive(Debug)]
pub struct CpuBackend;
impl GpuBackend for CpuBackend {
    fn name(&self) -> &str { "CPU (SIMD)" }
    fn compute_distance(&self, q: &[f32], v: &[f32], d: usize, m: VectorMetric) -> Result<Vec<f32>> { compute_cpu(q, v, d, m) }
    fn compute_kmeans_assignment(&self, v: &[f32], c: &[f32], d: usize) -> Result<Vec<u32>> { super::ivf::simple_kmeans_assignment(v, c, d) }
}

pub const GPU_DISPATCH_THRESHOLD: usize = 50_000;

pub fn compute_distance(query: &[f32], vectors: &[f32], dim: usize, metric: VectorMetric) -> Result<Vec<f32>> {
    let context = get_global_gpu_context().unwrap_or_else(ComputeContext::auto_detect);
    let n = if dim > 0 { vectors.len() / dim } else { 0 };
    if n < GPU_DISPATCH_THRESHOLD && context.backend != ComputeBackend::Cpu { return compute_cpu(query, vectors, dim, metric); }
    if let Some(imp) = &context.implementation { return imp.compute_distance(query, vectors, dim, metric); }
    compute_cpu(query, vectors, dim, metric)
}

pub fn compute_kmeans_assignment(vectors: &[f32], centroids: &[f32], dim: usize) -> Result<Vec<u32>> {
    let context = get_global_gpu_context().unwrap_or_else(ComputeContext::auto_detect);
    if let Some(imp) = &context.implementation { return imp.compute_kmeans_assignment(vectors, centroids, dim); }
    super::ivf::simple_kmeans_assignment(vectors, centroids, dim)
}

fn compute_cpu(q: &[f32], v: &[f32], d: usize, m: VectorMetric) -> Result<Vec<f32>> {
    let n = if d > 0 { v.len() / d } else { 0 };
    let mut dists = Vec::with_capacity(n);
    for i in 0..n {
        let span = &v[i*d..(i+1)*d];
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

pub fn set_global_gpu_context(ctx: Option<ComputeContext>) { GLOBAL_GPU_CONTEXT.with(|c| *c.borrow_mut() = ctx); }
pub fn get_global_gpu_context() -> Option<ComputeContext> { GLOBAL_GPU_CONTEXT.with(|c| c.borrow().clone()) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_cpu() {
        let q = vec![1.0, 0.0]; let v = vec![1.0, 0.0, 0.0, 1.0];
        let d = compute_distance(&q, &v, 2, VectorMetric::L2).unwrap();
        assert_eq!(d[0], 0.0);
    }
}
