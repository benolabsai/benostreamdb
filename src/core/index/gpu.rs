// Copyright (c) 2026 Richard Albright. All rights reserved.

/// Hardware Acceleration Module for HyperStreamDB
/// 
/// This module provides support for various GPU backends:
/// - NVIDIA CUDA
/// - AMD ROCm
/// - Apple MPS (Metal Performance Shaders)
/// - Intel oneAPI / Level Zero
///
/// These features are part of the HyperStreamDB core to ensure high-performance
/// vector search is available to everyone.
use anyhow::Result;
use super::VectorMetric;
use std::cell::RefCell;

thread_local! {
    /// Thread-local storage for GPU compute context
    /// 
    /// This enables SQL UDFs and other operations to access the GPU context
    /// configured from Python without explicitly passing it through all layers.
    /// Each thread maintains its own context, ensuring thread-safety.
    static GLOBAL_GPU_CONTEXT: RefCell<Option<ComputeContext>> = const { RefCell::new(None) };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeBackend {
    Cpu,   // Standard SIMD-accelerated CPU (AVX2, NEON, etc.)
    Cuda,  // NVIDIA
    Rocm,  // AMD
    Mps,   // Apple Silicon
    Intel, // Intel GPU (oneAPI)
}

#[derive(Debug, Clone, Copy)]
pub struct ComputeContext {
    pub backend: ComputeBackend,
    pub device_id: i32,
}

impl ComputeContext {
    /// Check if the specified backend and device are actually available on this system
    pub fn is_available(&self) -> bool {
        match self.backend {
            ComputeBackend::Cpu => true,
            
            ComputeBackend::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    return cust::init(cust::CudaFlags::empty()).is_ok();
                }
                #[cfg(not(feature = "cuda"))]
                false
            },
            
            ComputeBackend::Mps => {
                #[cfg(all(target_os = "macos", feature = "mps"))]
                return true; 
                #[cfg(not(all(target_os = "macos", feature = "mps")))]
                false
            },

            ComputeBackend::Intel => {
                #[cfg(feature = "intel")]
                {
                    use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
                    if let Ok(devices) = get_all_devices(CL_DEVICE_TYPE_GPU) {
                        for id in devices {
                            let device = Device::new(id);
                            let v = device.vendor().unwrap_or_default().to_lowercase();
                            let n = device.name().unwrap_or_default().to_lowercase();
                            if v.contains("intel") || n.contains("intel") || n.contains("graphics") || n.contains("uhd") || n.contains("iris") || n.contains("arc") {
                                return true;
                            }
                        }
                    }
                    // Diagnostic: No Intel GPU found in OpenCL devices
                }
                false
            },

            ComputeBackend::Rocm => {
                #[cfg(feature = "rocm")]
                {
                    use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
                    if let Ok(devices) = get_all_devices(CL_DEVICE_TYPE_GPU) {
                        for id in devices {
                            let device = Device::new(id);
                            if let Ok(vendor) = device.vendor() {
                                let v = vendor.to_lowercase();
                                if v.contains("amd") || v.contains("advanced micro devices") || v.contains("rocm") {
                                    return true;
                                }
                            }
                        }
                    }
                }
                false
            },
        }
    }

    pub fn auto_detect() -> Self {
        // Priority order: Cuda > Mps > Intel > Rocm > Cpu
        
        #[cfg(feature = "cuda")]
        {
            let ctx = Self { backend: ComputeBackend::Cuda, device_id: 0 };
            if ctx.is_available() { return ctx; }
        }
        
        #[cfg(all(target_os = "macos", feature = "mps"))]
        {
            let ctx = Self { backend: ComputeBackend::Mps, device_id: 0 };
            if ctx.is_available() { return ctx; }
        }

        #[cfg(feature = "intel")]
        {
            let ctx = Self { backend: ComputeBackend::Intel, device_id: 0 };
            if ctx.is_available() { return ctx; }
        }

        #[cfg(feature = "rocm")]
        {
            let ctx = Self { backend: ComputeBackend::Rocm, device_id: 0 };
            if ctx.is_available() { return ctx; }
        }

        Self { backend: ComputeBackend::Cpu, device_id: -1 }
    }

    pub fn from_device_str(device: &str) -> Result<Self> {
        let ctx = match device.to_lowercase().as_str() {
            "cpu" => Self { backend: ComputeBackend::Cpu, device_id: -1 },
            "gpu" | "auto" => Self::auto_detect(),
            "cuda" => Self { backend: ComputeBackend::Cuda, device_id: 0 },
            "mps" | "metal" => Self { backend: ComputeBackend::Mps, device_id: 0 },
            "rocm" => Self { backend: ComputeBackend::Rocm, device_id: 0 },
            "intel" | "opencl" => Self { backend: ComputeBackend::Intel, device_id: 0 },
            _ => anyhow::bail!("Unsupported device: {}", device),
        };
        
        // Final sanity check - if the user explicitly requested a backend, make sure it's there!
        if !ctx.is_available() && ctx.backend != ComputeBackend::Cpu {
             anyhow::bail!("Requested backend '{:?}' is not available on this system (probed hardware returned false). Try 'cpu' if hardware acceleration is not set up.", ctx.backend);
        }
        
        Ok(ctx)
    }
}

/// Set the global GPU context for the current thread
/// 
/// This function stores a GPU context in thread-local storage, making it accessible
/// to SQL UDFs and other operations without explicitly passing it through all layers.
/// 
/// # Arguments
/// * `context` - The compute context to set, or None to clear the context
/// 
/// # Thread Safety
/// Each thread maintains its own context, so this is thread-safe for concurrent access.
/// 
/// # Example
/// ```
/// use hyperstreamdb::core::index::gpu::{ComputeContext, ComputeBackend, set_global_gpu_context};
/// 
/// let ctx = ComputeContext { backend: ComputeBackend::Cuda, device_id: 0 };
/// set_global_gpu_context(Some(ctx));
/// ```
pub fn set_global_gpu_context(context: Option<ComputeContext>) {
    GLOBAL_GPU_CONTEXT.with(|ctx| {
        *ctx.borrow_mut() = context;
    });
}

/// Get the global GPU context for the current thread
/// 
/// This function retrieves the GPU context from thread-local storage if one has been set.
/// Returns None if no context has been configured for the current thread.
/// 
/// # Returns
/// The current thread's compute context, or None if not set
/// 
/// # Thread Safety
/// Each thread maintains its own context, so this is thread-safe for concurrent access.
/// 
/// # Example
/// ```
/// use hyperstreamdb::core::index::gpu::get_global_gpu_context;
/// 
/// if let Some(ctx) = get_global_gpu_context() {
///     println!("Using GPU backend: {:?}", ctx.backend);
/// } else {
///     println!("No GPU context configured");
/// }
/// ```
pub fn get_global_gpu_context() -> Option<ComputeContext> {
    GLOBAL_GPU_CONTEXT.with(|ctx| {
        *ctx.borrow()
    })
}

/// Generic accelerated distance calculation
pub fn compute_distance(
    _query: &[f32],
    _vectors: &[f32],
    _dim: usize,
    _metric: VectorMetric,
    _context: &ComputeContext,
) -> Result<Vec<f32>> {
    match _context.backend {
        ComputeBackend::Cpu => compute_cpu(_query, _vectors, _dim, _metric),
        ComputeBackend::Cuda => compute_cuda(_query, _vectors, _dim, _metric),
        ComputeBackend::Rocm => compute_rocm(_query, _vectors, _dim, _metric),
        ComputeBackend::Mps => compute_mps(_query, _vectors, _dim, _metric),
        ComputeBackend::Intel => compute_intel(_query, _vectors, _dim, _metric),
    }
}

/// Batch distance computation with memory transfer optimization
/// 
/// This function computes distances between a single query vector and multiple database vectors.
/// It optimizes memory transfers for large batches by:
/// - Transferring the query vector once and reusing it
/// - Processing vectors in chunks if they don't fit in GPU memory
/// - Minimizing host-device synchronization points
///
/// # Arguments
/// * `query` - The query vector (dimension: dim)
/// * `vectors` - Flattened array of database vectors (dimension: n_vectors * dim)
/// * `dim` - Dimension of each vector
/// * `metric` - Distance metric to use
/// * `context` - Compute context specifying backend and device
///
/// # Returns
/// Vector of distances, one per database vector
///
/// # Memory Transfer Strategy
/// For large batches that exceed GPU memory:
/// - Splits vectors into chunks that fit in available GPU memory
/// - Transfers query once, reuses across all chunks
/// - Processes chunks sequentially, accumulating results
pub fn compute_distance_batch(
    query: &[f32],
    vectors: &[f32],
    dim: usize,
    metric: VectorMetric,
    context: &ComputeContext,
) -> Result<Vec<f32>> {
    // Validate inputs
    if query.len() != dim {
        anyhow::bail!("Query vector length {} does not match dimension {}", query.len(), dim);
    }
    
    if !vectors.len().is_multiple_of(dim) {
        anyhow::bail!("Vectors array length {} is not a multiple of dimension {}", vectors.len(), dim);
    }
    
    let _n_vectors = vectors.len() / dim;
    
    if context.backend == ComputeBackend::Cpu {
        return compute_distance(query, vectors, dim, metric, context);
    }
    
    // For large batches on GPU, implement chunked processing
    // This helps when the entire dataset doesn't fit in GPU memory
    match context.backend {
        ComputeBackend::Cpu => compute_cpu(query, vectors, dim, metric),
        ComputeBackend::Cuda => compute_cuda_batch(query, vectors, dim, metric, context),
        ComputeBackend::Rocm => compute_rocm_batch(query, vectors, dim, metric, context),
        ComputeBackend::Mps => compute_mps_batch(query, vectors, dim, metric, context),
        ComputeBackend::Intel => compute_intel_batch(query, vectors, dim, metric, context),
    }
}

fn compute_cpu(_query: &[f32], _vectors: &[f32], _dim: usize, _metric: VectorMetric) -> Result<Vec<f32>> {
    // Falls back to optimized distance.rs logic
    let mut distances = Vec::with_capacity(_vectors.len() / _dim);
    for i in 0..(_vectors.len() / _dim) {
        let start = i * _dim;
        let end = start + _dim;
        let dist = match _metric {
            VectorMetric::L2 => crate::core::index::distance::l2_distance(_query, &_vectors[start..end]),
            VectorMetric::Cosine => crate::core::index::distance::cosine_distance(_query, &_vectors[start..end]),
            VectorMetric::InnerProduct => crate::core::index::distance::dot_product(_query, &_vectors[start..end]),
            VectorMetric::L1 => crate::core::index::distance::l1_distance(_query, &_vectors[start..end]),
            VectorMetric::Hamming => crate::core::index::distance::hamming_distance(_query, &_vectors[start..end]),
            VectorMetric::Jaccard => crate::core::index::distance::jaccard_distance(_query, &_vectors[start..end]),
        };
        distances.push(dist);
    }
    Ok(distances)
}

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

#[allow(unused_variables)]
fn compute_cuda(query: &[f32], vectors: &[f32], dim: usize, metric: VectorMetric) -> Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        use cust::prelude::*;

        // Determine kernel name and PTX based on metric
        let (ptx, kernel_name) = match metric {
            VectorMetric::L2 => (PTX_L2, "l2_distance_kernel"),
            VectorMetric::Cosine => (PTX_COSINE, "cosine_distance_kernel"),
            VectorMetric::InnerProduct => (PTX_INNER_PRODUCT, "inner_product_kernel"),
            VectorMetric::L1 => (PTX_L1, "l1_distance_kernel"),
            VectorMetric::Hamming => (PTX_HAMMING, "hamming_distance_kernel"),
            VectorMetric::Jaccard => (PTX_JACCARD, "jaccard_distance_kernel"),
        };

        // Try CUDA, fall back to CPU if not available
        let cuda_result = (|| -> Result<Vec<f32>> {
            // 1. Initialize CUDA (once per process usually, but quick context here)
            // In a real app, Context would be stored in ComputeContext
            cust::init(CudaFlags::empty())?;
            
            // Select device 0 for now
            let device = Device::get_device(0)?;
            let _ctx = Context::new(device)?;

            // 2. Load Module & Stream
            let module = Module::from_ptx(ptx, &[])?;
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

            // 3. Allocate Memory
            // Query (Host -> Device)
            let d_query = DeviceBuffer::from_slice(query)?;
            
            // Vectors (Host -> Device)
            // In product quantization or optimized search, vectors might already be on GPU
            // casting &[f32] to device buffer
            let d_vectors = DeviceBuffer::from_slice(vectors)?;
            
            // Output Distances
            let n_vectors = vectors.len() / dim;
            let d_distances = DeviceBuffer::<f32>::zeroed(n_vectors)?;

            // 4. Launch Kernel
            let function = module.get_function(kernel_name)?;
            
            // Configuration: One block per vector (row)
            let block_size = 256;
            let grid_size = n_vectors as u32;

            // Calculate shared memory size based on metric
            let shared_mem_size = match metric {
                VectorMetric::Cosine => block_size * 3 * std::mem::size_of::<f32>() as u32, // Need 3x for dot, norm_a, norm_b
                VectorMetric::Jaccard => block_size * 2 * std::mem::size_of::<f32>() as u32, // Need 2x for intersection, union
                _ => block_size * std::mem::size_of::<f32>() as u32, // Standard size for other metrics
            };

            unsafe {
                launch!(
                    function<<<grid_size, block_size, shared_mem_size, stream>>>(
                        d_query.as_device_ptr(),
                        d_vectors.as_device_ptr(),
                        d_distances.as_device_ptr(),
                        dim as i32,
                        n_vectors as i32
                    )
                )?;
            }

            stream.synchronize()?;

            // 5. Copy Back
            let mut distances = vec![0.0f32; n_vectors];
            d_distances.copy_to(&mut distances)?;

            Ok(distances)
        })();

        match cuda_result {
            Ok(result) => Ok(result),
            Err(_) => {
                // Fall back to CPU if CUDA is not available or fails
                compute_cpu(query, vectors, dim, metric)
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        // Fall back to CPU if CUDA feature is not enabled
        compute_cpu(query, vectors, dim, metric)
    }
}

#[allow(unused_variables)]
fn compute_rocm(query: &[f32], vectors: &[f32], dim: usize, metric: VectorMetric) -> Result<Vec<f32>> {
    #[cfg(feature = "rocm")]
    {
        // Try ROCm, fall back to CPU if not available
        match compute_opencl(query, vectors, dim, metric, Some("AMD")) {
            Ok(result) => Ok(result),
            Err(_) => {
                // Fall back to CPU if ROCm is not available
                compute_cpu(query, vectors, dim, metric)
            }
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        // Fall back to CPU if ROCm feature is not enabled
        compute_cpu(query, vectors, dim, metric)
    }
}

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

#[cfg(feature = "cuda")]
static CUDA_KMEANS: &str = include_str!("cuda/kmeans_assignment.cu");

#[cfg(any(feature = "intel", feature = "rocm"))]
static OPENCL_KMEANS: &str = include_str!("opencl/kmeans_assignment.cl");

#[cfg(all(target_os = "macos", feature = "mps"))]
static MSL_JACCARD: &str = include_str!("mps/jaccard_distance.metal");

#[allow(unused_variables)]
fn compute_mps(query: &[f32], vectors: &[f32], dim: usize, metric: VectorMetric) -> Result<Vec<f32>> {
    #[cfg(all(target_os = "macos", feature = "mps"))]
    {
        use metal::*;
        use std::mem;

        // Determine shader source and kernel name based on metric
        let (msl_source, kernel_name) = match metric {
            VectorMetric::L2 => (MSL_L2, "l2_distance_kernel"),
            VectorMetric::Cosine => (MSL_COSINE, "cosine_distance_kernel"),
            VectorMetric::InnerProduct => (MSL_INNER_PRODUCT, "inner_product_kernel"),
            VectorMetric::L1 => (MSL_L1, "l1_distance_kernel"),
            VectorMetric::Hamming => (MSL_HAMMING, "hamming_distance_kernel"),
            VectorMetric::Jaccard => (MSL_JACCARD, "jaccard_distance_kernel"),
        };

        // Try MPS, fall back to CPU if not available
        let mps_result = (|| -> Result<Vec<f32>> {
            let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("No Metal device found"))?;
            eprintln!("DEBUG: Using MPS backend for vector search on device: {}", device.name());
            
            // 1. Compile Shader
            let library = device.new_library_with_source(msl_source, &CompileOptions::new()).map_err(|e| anyhow::anyhow!("Metal compile error: {}", e))?;
            let kernel = library.get_function(kernel_name, None).map_err(|e| anyhow::anyhow!("Kernel not found: {}", e))?;
            
            let pipeline_state = device.new_compute_pipeline_state_with_function(&kernel).map_err(|e| anyhow::anyhow!("Pipeline error: {}", e))?;
            let command_queue = device.new_command_queue();

            // 2. Prepare Buffers
            // Vectors
            let vectors_size = (vectors.len() * mem::size_of::<f32>()) as u64;
            let vectors_buffer = device.new_buffer_with_data(
                vectors.as_ptr() as *const _,
                vectors_size,
                MTLResourceOptions::StorageModeShared
            );

            // Query
            let query_size = (query.len() * mem::size_of::<f32>()) as u64;
            let query_buffer = device.new_buffer_with_data(
                query.as_ptr() as *const _,
                query_size,
                MTLResourceOptions::StorageModeShared
            );
            
            // Output
            let n_vectors = vectors.len() / dim;
            let output_size = (n_vectors * mem::size_of::<f32>()) as u64;
            let output_buffer = device.new_buffer(output_size, MTLResourceOptions::StorageModeShared);

            // Dim
            // Unused in buffer since we pass by value? No, passed as buffer in Metal arg table usually or constant
            // In MSL: constant uint& dim [[ buffer(3) ]] -> It expects a buffer.
            let dim_u32 = dim as u32;
            let dim_buffer = device.new_buffer_with_data(
                &dim_u32 as *const _ as *const _,
                mem::size_of::<u32>() as u64,
                MTLResourceOptions::StorageModeShared
            );

            // 3. Encode Command
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            
            encoder.set_compute_pipeline_state(&pipeline_state);
            encoder.set_buffer(0, Some(&query_buffer), 0);
            encoder.set_buffer(1, Some(&vectors_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_buffer(3, Some(&dim_buffer), 0);

            // Threads
            let thread_group_count = MTLSize::new(n_vectors as u64, 1, 1);
            let thread_group_size = MTLSize::new(1, 1, 1); // 1 thread per group? 
            // Better: usage of grid dispatch
            // w = pipeline_state.thread_execution_width();
            // h = pipeline_state.max_total_threads_per_threadgroup() / w;
            // Check optimize later. For now, 1D grid matching vectors.
            
            // Use dispatch_threads if supported (newer Metal) or dispatch_threadgroups
            // dispatch_threads is easier:
            // encoder.dispatch_threads(thread_group_count, thread_group_size); // Requires Metal 2??
            
            // Fallback to threadgroups
            let threads_per_group = MTLSize::new(256, 1, 1);
            let groups_width = (n_vectors as u64 + 255) / 256;
            let groups = MTLSize::new(groups_width, 1, 1);
            
            // REWRITE SHADER to handle id = group_id * group_size + thread_id
            // Actually MSL `thread_position_in_grid` handles this automatically.
            encoder.dispatch_thread_groups(groups, threads_per_group);

            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();

            // 4. Read Results
            let output_ptr = output_buffer.contents() as *mut f32;
            let mut results = Vec::with_capacity(n_vectors);
            unsafe {
                let slice = std::slice::from_raw_parts(output_ptr, n_vectors);
                results.extend_from_slice(slice);
            }

            Ok(results)
        })();

        match mps_result {
            Ok(result) => Ok(result),
            Err(_) => {
                // Fall back to CPU if MPS is not available or fails
                compute_cpu(query, vectors, dim, metric)
            }
        }
    }

    #[cfg(not(all(target_os = "macos", feature = "mps")))]
    {
        // Fallback to CPU for tests on non-macOS
        compute_cpu(query, vectors, dim, metric)
    }
}

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

#[cfg(any(feature = "intel", feature = "rocm"))]
fn compute_opencl(query: &[f32], vectors: &[f32], dim: usize, metric: VectorMetric, _platform_filter: Option<&str>) -> Result<Vec<f32>> {
    use opencl3::command_queue::{CommandQueue, CL_BLOCKING};
    use opencl3::context::Context;
    use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
    use opencl3::kernel::{ExecuteKernel, Kernel};
    use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
    use opencl3::program::Program;
    use opencl3::types::cl_float;
    use std::ptr;

    // Determine kernel source and kernel name based on metric
    let (opencl_source, kernel_name) = match metric {
        VectorMetric::L2 => (OPENCL_SRC_L2, "l2_distance_kernel"),
        VectorMetric::Cosine => (OPENCL_SRC_COSINE, "cosine_distance_kernel"),
        VectorMetric::InnerProduct => (OPENCL_SRC_INNER_PRODUCT, "inner_product_kernel"),
        VectorMetric::L1 => (OPENCL_SRC_L1, "l1_distance_kernel"),
        VectorMetric::Hamming => (OPENCL_SRC_HAMMING, "hamming_distance_kernel"),
        VectorMetric::Jaccard => (OPENCL_SRC_JACCARD, "jaccard_distance_kernel"),
    };

    // 1. Convert inputs
    let n_vectors = vectors.len() / dim;

    // 2. Find Platform/Device
    let platform_filter = _platform_filter.unwrap_or("");
    let device_id = get_all_devices(CL_DEVICE_TYPE_GPU)?
        .into_iter()
        .find(|&id| {
            if platform_filter.is_empty() {
                return true;
            }
            let device = Device::new(id);
            if let Ok(vendor) = device.vendor() {
                vendor.contains(platform_filter)
            } else {
                false
            }
        })
        .ok_or_else(|| anyhow::anyhow!("No OpenCL GPU found matching filter: '{}'", platform_filter))?;
    
    let device = Device::new(device_id);
    let context = Context::from_device(&device)?;

    // 3. Create Command Queue
    // Properties: 0 for default (in-order)
    let queue = unsafe { CommandQueue::create_with_properties(&context, device_id, 0, 0)? };

    // 4. Compile Program
    let program = Program::create_and_build_from_source(&context, opencl_source, "")
        .map_err(|e| anyhow::anyhow!("OpenCL build error: {}", e))?;
    let kernel = Kernel::create(&program, kernel_name)?;

    // 5. Create Buffers
    let mut query_buffer = unsafe { Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, query.len(), ptr::null_mut())? };
    let mut vectors_buffer = unsafe { Buffer::<cl_float>::create(&context, CL_MEM_READ_ONLY, vectors.len(), ptr::null_mut())? };
    let output_buffer = unsafe { Buffer::<cl_float>::create(&context, CL_MEM_WRITE_ONLY, n_vectors, ptr::null_mut())? };

    // 6. Write Data
    let _query_write_event = unsafe { queue.enqueue_write_buffer(&mut query_buffer, CL_BLOCKING, 0, query, &[])? };
    let _vectors_write_event = unsafe { queue.enqueue_write_buffer(&mut vectors_buffer, CL_BLOCKING, 0, vectors, &[])? };

    // 8. Execute Kernel
    let kernel_event = unsafe {
        ExecuteKernel::new(&kernel)
            .set_arg(&query_buffer)
            .set_arg(&vectors_buffer)
            .set_arg(&output_buffer)
            .set_arg(&(dim as i32))
            .set_global_work_size(n_vectors)
            .enqueue_nd_range(&queue)?
    };

    // 9. Read Results
    let mut results = vec![0.0f32; n_vectors];
    let _read_event = unsafe { queue.enqueue_read_buffer(&output_buffer, CL_BLOCKING, 0, &mut results, &[kernel_event.get()])? };

    Ok(results)
}

#[allow(unused_variables)]
fn compute_intel(query: &[f32], vectors: &[f32], dim: usize, metric: VectorMetric) -> Result<Vec<f32>> {
    #[cfg(feature = "intel")]
    {
        // Try Intel GPU, fall back to CPU if not available
        match compute_opencl(query, vectors, dim, metric, Some("Intel")) {
            Ok(result) => Ok(result),
            Err(e) => {
                // Fall back to CPU if Intel GPU is not available
                eprintln!("DEBUG: Intel GPU computation failed, falling back to CPU. Error: {}", e);
                compute_cpu(query, vectors, dim, metric)
            }
        }
    }
    #[cfg(not(feature = "intel"))]
    {
        // Fall back to CPU if Intel GPU feature is not enabled
        compute_cpu(query, vectors, dim, metric)
    }
}

// ============================================================================
// Batch-optimized implementations with memory transfer optimization
// ============================================================================

/// CUDA batch implementation with chunked processing for large datasets
#[allow(unused_variables)]
fn compute_cuda_batch(query: &[f32], vectors: &[f32], dim: usize, metric: VectorMetric, context: &ComputeContext) -> Result<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        use cust::prelude::*;
        
        // Determine kernel name and PTX based on metric
        let (ptx, kernel_name) = match metric {
            VectorMetric::L2 => (PTX_L2, "l2_distance_kernel"),
            VectorMetric::Cosine => (PTX_COSINE, "cosine_distance_kernel"),
            VectorMetric::InnerProduct => (PTX_INNER_PRODUCT, "inner_product_kernel"),
            VectorMetric::L1 => (PTX_L1, "l1_distance_kernel"),
            VectorMetric::Hamming => (PTX_HAMMING, "hamming_distance_kernel"),
            VectorMetric::Jaccard => (PTX_JACCARD, "jaccard_distance_kernel"),
        };
        
        // Try CUDA batch, fall back to CPU if not available
        let cuda_result = (|| -> Result<Vec<f32>> {
            // Initialize CUDA
            cust::init(CudaFlags::empty())?;
            let device = Device::get_device(context.device_id as u32)?;
            let _ctx = Context::new(device)?;
            
            // Get available GPU memory
            let free_mem = device.total_memory()? / 2; // Use half of available memory for safety
            
            // Calculate chunk size based on available memory
            // Memory needed per vector: dim * sizeof(f32) + sizeof(f32) for result
            let bytes_per_vector = (dim * std::mem::size_of::<f32>()) + std::mem::size_of::<f32>();
            let query_bytes = query.len() * std::mem::size_of::<f32>();
            let max_vectors_per_chunk = ((free_mem - query_bytes) / bytes_per_vector).max(1000);
            
            let n_vectors = vectors.len() / dim;
            
            // If everything fits in one chunk, use standard implementation
            if n_vectors <= max_vectors_per_chunk {
                return compute_cuda(query, vectors, dim, metric);
            }
            
            // Process in chunks
            let module = Module::from_ptx(ptx, &[])?;
            let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
            let function = module.get_function(kernel_name)?;
            
            // Allocate query buffer once (reused across chunks)
            let d_query = DeviceBuffer::from_slice(query)?;
            
            let mut all_distances = Vec::with_capacity(n_vectors);
            let block_size = 256;
            
            // Process vectors in chunks
            for chunk_start in (0..n_vectors).step_by(max_vectors_per_chunk) {
                let chunk_end = (chunk_start + max_vectors_per_chunk).min(n_vectors);
                let chunk_size = chunk_end - chunk_start;
                
                let vector_start = chunk_start * dim;
                let vector_end = chunk_end * dim;
                let chunk_vectors = &vectors[vector_start..vector_end];
                
                // Allocate buffers for this chunk
                let d_vectors = DeviceBuffer::from_slice(chunk_vectors)?;
                let d_distances = DeviceBuffer::<f32>::zeroed(chunk_size)?;
                
                // Launch kernel
                let grid_size = chunk_size as u32;
                
                // Calculate shared memory size based on metric
                let shared_mem_size = match metric {
                    VectorMetric::Cosine => block_size * 3 * std::mem::size_of::<f32>() as u32, // Need 3x for dot, norm_a, norm_b
                    VectorMetric::Jaccard => block_size * 2 * std::mem::size_of::<f32>() as u32, // Need 2x for intersection, union
                    _ => block_size * std::mem::size_of::<f32>() as u32, // Standard size for other metrics
                };
                
                unsafe {
                    launch!(
                        function<<<grid_size, block_size, shared_mem_size, stream>>>(
                            d_query.as_device_ptr(),
                            d_vectors.as_device_ptr(),
                            d_distances.as_device_ptr(),
                            dim as i32,
                            chunk_size as i32
                        )
                    )?;
                }
                
                stream.synchronize()?;
                
                // Copy results back
                let mut chunk_distances = vec![0.0f32; chunk_size];
                d_distances.copy_to(&mut chunk_distances)?;
                all_distances.extend_from_slice(&chunk_distances);
            }
            
            Ok(all_distances)
        })();

        match cuda_result {
            Ok(result) => Ok(result),
            Err(_) => {
                // Fall back to CPU if CUDA is not available or fails
                compute_cpu(query, vectors, dim, metric)
            }
        }
    }
    
    #[cfg(not(feature = "cuda"))]
    {
        // Fall back to CPU if CUDA feature is not enabled
        compute_cpu(query, vectors, dim, metric)
    }
}

/// ROCm batch implementation (uses OpenCL with chunking)
#[allow(unused_variables)]
fn compute_rocm_batch(query: &[f32], vectors: &[f32], dim: usize, metric: VectorMetric, context: &ComputeContext) -> Result<Vec<f32>> {
    #[cfg(feature = "rocm")]
    {
        // Try ROCm batch, fall back to CPU if not available
        match compute_opencl_batch(query, vectors, dim, metric, Some("AMD"), context) {
            Ok(result) => Ok(result),
            Err(_) => {
                // Fall back to CPU if ROCm is not available
                compute_cpu(query, vectors, dim, metric)
            }
        }
    }
    #[cfg(not(feature = "rocm"))]
    {
        // Fall back to CPU if ROCm feature is not enabled
        compute_cpu(query, vectors, dim, metric)
    }
}

/// MPS batch implementation with chunked processing
#[allow(unused_variables)]
fn compute_mps_batch(query: &[f32], vectors: &[f32], dim: usize, metric: VectorMetric, context: &ComputeContext) -> Result<Vec<f32>> {
    #[cfg(all(target_os = "macos", feature = "mps"))]
    {
        use metal::*;
        use std::mem;
        
        // Determine shader source and kernel name based on metric
        let (msl_source, kernel_name) = match metric {
            VectorMetric::L2 => (MSL_L2, "l2_distance_kernel"),
            VectorMetric::Cosine => (MSL_COSINE, "cosine_distance_kernel"),
            VectorMetric::InnerProduct => (MSL_INNER_PRODUCT, "inner_product_kernel"),
            VectorMetric::L1 => (MSL_L1, "l1_distance_kernel"),
            VectorMetric::Hamming => (MSL_HAMMING, "hamming_distance_kernel"),
            VectorMetric::Jaccard => (MSL_JACCARD, "jaccard_distance_kernel"),
        };
        
        let device = Device::system_default().ok_or_else(|| anyhow::anyhow!("No Metal device found"))?;
        
        // Get available GPU memory (Metal doesn't expose this directly, use heuristic)
        // Assume 4GB available for computation (conservative estimate)
        let available_mem = 4_000_000_000_u64;
        
        // Calculate chunk size
        let bytes_per_vector = (dim * mem::size_of::<f32>()) + mem::size_of::<f32>();
        let query_bytes = query.len() * mem::size_of::<f32>();
        let max_vectors_per_chunk = ((available_mem - query_bytes as u64) / bytes_per_vector as u64).max(1000) as usize;
        
        let n_vectors = vectors.len() / dim;
        
        // If everything fits in one chunk, use standard implementation
        if n_vectors <= max_vectors_per_chunk {
            return compute_mps(query, vectors, dim, metric);
        }
        
        // Compile shader once
        let library = device.new_library_with_source(msl_source, &CompileOptions::new())
            .map_err(|e| anyhow::anyhow!("Metal compile error: {}", e))?;
        let kernel = library.get_function(kernel_name, None)
            .map_err(|e| anyhow::anyhow!("Kernel not found: {}", e))?;
        let pipeline_state = device.new_compute_pipeline_state_with_function(&kernel)
            .map_err(|e| anyhow::anyhow!("Pipeline error: {}", e))?;
        let command_queue = device.new_command_queue();
        
        // Create query buffer once (reused across chunks)
        let query_size = (query.len() * mem::size_of::<f32>()) as u64;
        let query_buffer = device.new_buffer_with_data(
            query.as_ptr() as *const _,
            query_size,
            MTLResourceOptions::StorageModeShared
        );
        
        let dim_u32 = dim as u32;
        let dim_buffer = device.new_buffer_with_data(
            &dim_u32 as *const _ as *const _,
            mem::size_of::<u32>() as u64,
            MTLResourceOptions::StorageModeShared
        );
        
        let mut all_distances = Vec::with_capacity(n_vectors);
        
        // Process in chunks
        for chunk_start in (0..n_vectors).step_by(max_vectors_per_chunk) {
            let chunk_end = (chunk_start + max_vectors_per_chunk).min(n_vectors);
            let chunk_size = chunk_end - chunk_start;
            
            let vector_start = chunk_start * dim;
            let vector_end = chunk_end * dim;
            let chunk_vectors = &vectors[vector_start..vector_end];
            
            // Create buffers for this chunk
            let vectors_size = (chunk_vectors.len() * mem::size_of::<f32>()) as u64;
            let vectors_buffer = device.new_buffer_with_data(
                chunk_vectors.as_ptr() as *const _,
                vectors_size,
                MTLResourceOptions::StorageModeShared
            );
            
            let output_size = (chunk_size * mem::size_of::<f32>()) as u64;
            let output_buffer = device.new_buffer(output_size, MTLResourceOptions::StorageModeShared);
            
            // Encode and execute
            let command_buffer = command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            
            encoder.set_compute_pipeline_state(&pipeline_state);
            encoder.set_buffer(0, Some(&query_buffer), 0);
            encoder.set_buffer(1, Some(&vectors_buffer), 0);
            encoder.set_buffer(2, Some(&output_buffer), 0);
            encoder.set_buffer(3, Some(&dim_buffer), 0);
            
            let threads_per_group = MTLSize::new(256, 1, 1);
            let groups_width = (chunk_size as u64 + 255) / 256;
            let groups = MTLSize::new(groups_width, 1, 1);
            
            encoder.dispatch_thread_groups(groups, threads_per_group);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
            
            // Read results
            let output_ptr = output_buffer.contents() as *mut f32;
            unsafe {
                let slice = std::slice::from_raw_parts(output_ptr, chunk_size);
                all_distances.extend_from_slice(slice);
            }
        }
        
        Ok(all_distances)
    }
    
    #[cfg(not(all(target_os = "macos", feature = "mps")))]
    {
        // Fallback to CPU for tests on non-macOS
        compute_cpu(query, vectors, dim, metric)
    }
}

/// OpenCL batch implementation with chunked processing
#[cfg(any(feature = "intel", feature = "rocm"))]
fn compute_opencl_batch(query: &[f32], vectors: &[f32], dim: usize, metric: VectorMetric, platform_filter: Option<&str>, _context: &ComputeContext) -> Result<Vec<f32>> {
    use opencl3::command_queue::{CommandQueue, CL_BLOCKING};
    use opencl3::context::Context;
    use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
    use opencl3::kernel::{ExecuteKernel, Kernel};
    use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
    use opencl3::program::Program;
    use opencl3::types::cl_float;
    use std::ptr;
    
    // Determine kernel source and kernel name based on metric
    let (opencl_source, kernel_name) = match metric {
        VectorMetric::L2 => (OPENCL_SRC_L2, "l2_distance_kernel"),
        VectorMetric::Cosine => (OPENCL_SRC_COSINE, "cosine_distance_kernel"),
        VectorMetric::InnerProduct => (OPENCL_SRC_INNER_PRODUCT, "inner_product_kernel"),
        VectorMetric::L1 => (OPENCL_SRC_L1, "l1_distance_kernel"),
        VectorMetric::Hamming => (OPENCL_SRC_HAMMING, "hamming_distance_kernel"),
        VectorMetric::Jaccard => (OPENCL_SRC_JACCARD, "jaccard_distance_kernel"),
    };
    
    let n_vectors = vectors.len() / dim;
    
    // Find Platform/Device
    let platform_filter = platform_filter.unwrap_or("");
    let device_id = get_all_devices(CL_DEVICE_TYPE_GPU)?
        .into_iter()
        .find(|&id| {
            if platform_filter.is_empty() {
                return true;
            }
            let device = Device::new(id);
            if let Ok(vendor) = device.vendor() {
                vendor.contains(platform_filter)
            } else {
                false
            }
        })
        .ok_or_else(|| anyhow::anyhow!("No OpenCL GPU found matching filter: '{}'", platform_filter))?;
    
    let device = Device::new(device_id);
    let opencl_context = Context::from_device(&device)?;
    
    // Get available GPU memory
    let max_mem = device.max_mem_alloc_size()? as usize;
    let global_mem = device.global_mem_size()? as usize;
    let available_mem = (global_mem / 2).min(max_mem); // Use half of global memory, capped at max alloc
    
    // Calculate chunk size
    let bytes_per_vector = (dim * std::mem::size_of::<f32>()) + std::mem::size_of::<f32>();
    let query_bytes = std::mem::size_of_val(query);
    let max_vectors_per_chunk = ((available_mem - query_bytes) / bytes_per_vector).max(1000);
    
    // If everything fits in one chunk, use standard implementation
    if n_vectors <= max_vectors_per_chunk {
        return compute_opencl(query, vectors, dim, metric, Some(platform_filter));
    }
    
    // Create command queue and compile program once
    let queue = unsafe { CommandQueue::create_with_properties(&opencl_context, device_id, 0, 0)? };
    let program = Program::create_and_build_from_source(&opencl_context, opencl_source, "")
        .map_err(|e| anyhow::anyhow!("OpenCL build error: {}", e))?;
    let kernel = Kernel::create(&program, kernel_name)?;
    let num_expected_args = kernel.num_args().map_err(|e| anyhow::anyhow!("Error getting num_args: {}", e))?;
    if num_expected_args != 4 {
        panic!("DEBUG: Kernel {} expects {} args, but code provides 4. dim={}", kernel_name, num_expected_args, dim);
    }
    
    // Create query buffer once (reused across chunks)
    let mut query_buffer = unsafe { 
        Buffer::<cl_float>::create(&opencl_context, CL_MEM_READ_ONLY, query.len(), ptr::null_mut())? 
    };
    let _query_write_event = unsafe { 
        queue.enqueue_write_buffer(&mut query_buffer, CL_BLOCKING, 0, query, &[])? 
    };
    
    let mut all_distances = Vec::with_capacity(n_vectors);
    let _dim_int = dim as i32;
    
    // Process in chunks
    for chunk_start in (0..n_vectors).step_by(max_vectors_per_chunk) {
        let chunk_end = (chunk_start + max_vectors_per_chunk).min(n_vectors);
        let chunk_size = chunk_end - chunk_start;
        
        let vector_start = chunk_start * dim;
        let vector_end = chunk_end * dim;
        let chunk_vectors = &vectors[vector_start..vector_end];
        
        // Create buffers for this chunk
        let mut vectors_buffer = unsafe { 
            Buffer::<cl_float>::create(&opencl_context, CL_MEM_READ_ONLY, chunk_vectors.len(), ptr::null_mut())? 
        };
        let output_buffer = unsafe { 
            Buffer::<cl_float>::create(&opencl_context, CL_MEM_WRITE_ONLY, chunk_size, ptr::null_mut())? 
        };
        
        // Write data
        let _vectors_write_event = unsafe { 
            queue.enqueue_write_buffer(&mut vectors_buffer, CL_BLOCKING, 0, chunk_vectors, &[])? 
        };

        // Execute kernel
        let kernel_event = unsafe {
            ExecuteKernel::new(&kernel)
                .set_arg(&query_buffer)
                .set_arg(&vectors_buffer)
                .set_arg(&output_buffer)
                .set_arg(&(dim as i32))
                .set_global_work_size(chunk_size)
                .enqueue_nd_range(&queue)?
        };
        
        // Read results
        let mut chunk_distances = vec![0.0f32; chunk_size];
        let _read_event = unsafe { 
            queue.enqueue_read_buffer(&output_buffer, CL_BLOCKING, 0, &mut chunk_distances, &[kernel_event.get()])? 
        };
        
        all_distances.extend_from_slice(&chunk_distances);
    }
    
    Ok(all_distances)
}

/// Intel GPU batch implementation (uses OpenCL with chunking)
#[allow(unused_variables)]
fn compute_intel_batch(query: &[f32], vectors: &[f32], dim: usize, metric: VectorMetric, context: &ComputeContext) -> Result<Vec<f32>> {
    #[cfg(feature = "intel")]
    {
        // Try Intel GPU batch, fall back to CPU if not available
        match compute_opencl_batch(query, vectors, dim, metric, Some("Intel"), context) {
            Ok(result) => Ok(result),
            Err(_) => {
                // Fall back to CPU if Intel GPU is not available
                compute_cpu(query, vectors, dim, metric)
            }
        }
    }
    #[cfg(not(feature = "intel"))]
    {
        // Fall back to CPU if Intel GPU feature is not enabled
        compute_cpu(query, vectors, dim, metric)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_context_auto_detect() {
        let _ctx = ComputeContext::auto_detect();
        // On a standard test environment without special flags, this should match the enabled feature
        #[cfg(feature = "intel")]
        assert_eq!(_ctx.backend, ComputeBackend::Intel);
        
        #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), not(feature = "mps"), not(feature = "intel")))]
        assert_eq!(_ctx.backend, ComputeBackend::Cpu);
    }

    #[test]
    fn test_auto_detect_fallback() {
        // Test that auto_detect returns CPU backend when no real hardware is found 
        // (This happens in standard CI environments without GPUs)
        let _ctx = ComputeContext::auto_detect();
        
        #[cfg(feature = "intel")]
        assert_eq!(_ctx.backend, ComputeBackend::Intel);

        #[cfg(all(not(feature = "cuda"), not(feature = "rocm"), not(feature = "mps"), not(feature = "intel")))]
        {
            assert_eq!(_ctx.backend, ComputeBackend::Cpu);
            assert_eq!(_ctx.device_id, -1);
        }
    }

    #[test]
    fn test_compute_distance_cpu() {
        let query = vec![1.0, 2.0, 3.0];
        let vectors = vec![
            1.0, 2.0, 3.0, // Distance 0
            4.0, 5.0, 6.0, // Distance (3^2 + 3^2 + 3^2) = 27
        ];
        let dim = 3;
        let context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
        
        let distances = compute_distance(&query, &vectors, dim, VectorMetric::L2, &context).unwrap();
        assert_eq!(distances.len(), 2);
        assert_eq!(distances[0], 0.0);
        // l2_distance returns sqrt(sum((a-b)^2))
        // (3^2 + 3^2 + 3^2) = 27. sqrt(27) ≈ 5.1961524
        assert!((distances[1] - 5.196_152).abs() < 1e-6);
    }

    #[test]
    fn test_compute_distance_all_metrics_cpu() {
        let query = vec![1.0, 2.0, 3.0];
        let vectors = vec![
            1.0, 2.0, 3.0, // Identical vector
            4.0, 5.0, 6.0, // Different vector
        ];
        let dim = 3;
        let context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
        
        // Test L2 metric
        let distances = compute_distance(&query, &vectors, dim, VectorMetric::L2, &context).unwrap();
        assert_eq!(distances.len(), 2);
        assert_eq!(distances[0], 0.0);
        assert!((distances[1] - 5.196_152).abs() < 1e-6);
        
        // Test Cosine metric
        let distances = compute_distance(&query, &vectors, dim, VectorMetric::Cosine, &context).unwrap();
        assert_eq!(distances.len(), 2);
        assert!(distances[0] < 1e-6); // Should be very close to 0 for identical vectors
        
        // Test InnerProduct metric
        let distances = compute_distance(&query, &vectors, dim, VectorMetric::InnerProduct, &context).unwrap();
        assert_eq!(distances.len(), 2);
        // Inner product of [1,2,3] with itself = 1 + 4 + 9 = 14
        assert!((distances[0] - 14.0).abs() < 1e-6);
        // Inner product of [1,2,3] with [4,5,6] = 4 + 10 + 18 = 32
        assert!((distances[1] - 32.0).abs() < 1e-6);
        
        // Test L1 metric
        let distances = compute_distance(&query, &vectors, dim, VectorMetric::L1, &context).unwrap();
        assert_eq!(distances.len(), 2);
        assert_eq!(distances[0], 0.0);
        // L1 distance = |1-4| + |2-5| + |3-6| = 3 + 3 + 3 = 9
        assert!((distances[1] - 9.0).abs() < 1e-6);
        
        // Test Hamming metric
        let distances = compute_distance(&query, &vectors, dim, VectorMetric::Hamming, &context).unwrap();
        assert_eq!(distances.len(), 2);
        assert_eq!(distances[0], 0.0);
        
        // Test Jaccard metric
        let distances = compute_distance(&query, &vectors, dim, VectorMetric::Jaccard, &context).unwrap();
        assert_eq!(distances.len(), 2);
        assert_eq!(distances[0], 0.0);
    }

    #[test]
    fn test_gpu_backend_fallback_to_cpu() {
        // Test that unimplemented GPU kernels fall back to CPU
        let _query = [1.0, 2.0, 3.0];
        let _vectors = [
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let _dim = 3;
        
        // Test with CUDA backend (will fall back to CPU for non-L2 metrics)
        #[cfg(feature = "cuda")]
        {
            let context = ComputeContext { backend: ComputeBackend::Cuda, device_id: 0 };
            
            // Cosine should fall back to CPU
            let distances = compute_distance(&_query, &_vectors, _dim, VectorMetric::Cosine, &context).unwrap();
            assert_eq!(distances.len(), 2);
            
            // InnerProduct should fall back to CPU
            let distances = compute_distance(&_query, &_vectors, _dim, VectorMetric::InnerProduct, &context).unwrap();
            assert_eq!(distances.len(), 2);
        }
        
        // Test with MPS backend (will fall back to CPU for non-L2 metrics)
        #[cfg(all(target_os = "macos", feature = "mps"))]
        {
            let context = ComputeContext { backend: ComputeBackend::Mps, device_id: 0 };
            
            // Cosine should fall back to CPU
            let distances = compute_distance(&_query, &_vectors, _dim, VectorMetric::Cosine, &context).unwrap();
            assert_eq!(distances.len(), 2);
        }
    }

    #[test]
    fn test_compute_distance_batch_small() {
        // Test batch function with small dataset (should use standard path)
        let query = vec![1.0, 2.0, 3.0];
        let vectors = vec![
            1.0, 2.0, 3.0, // Distance 0
            4.0, 5.0, 6.0, // Distance sqrt(27) ≈ 5.196
            2.0, 3.0, 4.0, // Distance sqrt(3) ≈ 1.732
        ];
        let dim = 3;
        let context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
        
        let distances = compute_distance_batch(&query, &vectors, dim, VectorMetric::L2, &context).unwrap();
        assert_eq!(distances.len(), 3);
        assert_eq!(distances[0], 0.0);
        assert!((distances[1] - 5.196_152).abs() < 1e-6);
        assert!((distances[2] - 1.7320508).abs() < 1e-6);
    }

    #[test]
    fn test_compute_distance_batch_validation() {
        let query = vec![1.0, 2.0, 3.0];
        let vectors = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let dim = 3;
        let context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
        
        // Test query dimension mismatch
        let result = compute_distance_batch(&[1.0, 2.0], &vectors, dim, VectorMetric::L2, &context);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not match dimension"));
        
        // Test vectors not multiple of dimension
        let result = compute_distance_batch(&query, &vectors, dim, VectorMetric::L2, &context);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not a multiple of dimension"));
    }

    #[test]
    fn test_compute_distance_batch_large() {
        // Test batch function with larger dataset to verify chunking logic
        let dim = 128;
        let n_vectors = 2000; // Above the 1000 threshold
        let query = vec![1.0; dim];
        let mut vectors = Vec::with_capacity(n_vectors * dim);
        
        // Create test vectors
        for i in 0..n_vectors {
            for j in 0..dim {
                vectors.push((i as f32 + j as f32) / 100.0);
            }
        }
        
        let context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
        
        let distances = compute_distance_batch(&query, &vectors, dim, VectorMetric::L2, &context).unwrap();
        assert_eq!(distances.len(), n_vectors);
        
        // Verify distances are computed correctly by comparing with standard compute_distance
        let distances_standard = compute_distance(&query, &vectors, dim, VectorMetric::L2, &context).unwrap();
        assert_eq!(distances.len(), distances_standard.len());
        
        for i in 0..n_vectors {
            assert!((distances[i] - distances_standard[i]).abs() < 1e-5, 
                "Distance mismatch at index {}: {} vs {}", i, distances[i], distances_standard[i]);
        }
    }

    #[test]
    fn test_compute_distance_batch_all_metrics() {
        // Test batch function with all metrics
        let query = vec![1.0, 2.0, 3.0];
        let vectors = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
        ];
        let dim = 3;
        let context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
        
        // Test all metrics
        for metric in &[
            VectorMetric::L2,
            VectorMetric::Cosine,
            VectorMetric::InnerProduct,
            VectorMetric::L1,
            VectorMetric::Hamming,
            VectorMetric::Jaccard,
        ] {
            let distances = compute_distance_batch(&query, &vectors, dim, *metric, &context).unwrap();
            assert_eq!(distances.len(), 2, "Failed for metric {:?}", metric);
            
            // Verify results match standard compute_distance
            let distances_standard = compute_distance(&query, &vectors, dim, *metric, &context).unwrap();
            for i in 0..2 {
                assert!((distances[i] - distances_standard[i]).abs() < 1e-5,
                    "Mismatch for metric {:?} at index {}: {} vs {}", metric, i, distances[i], distances_standard[i]);
            }
        }
    }

    #[test]
    fn test_cosine_distance_cpu() {
        // Test cosine distance with specific test cases
        let query = vec![1.0, 2.0, 3.0];
        let vectors = vec![
            1.0, 2.0, 3.0,  // Identical vector - cosine distance should be ~0
            -1.0, -2.0, -3.0,  // Opposite vector - cosine distance should be ~2
            0.0, 1.0, 0.0,  // Orthogonal-ish vector
        ];
        let dim = 3;
        let context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
        
        let distances = compute_distance(&query, &vectors, dim, VectorMetric::Cosine, &context).unwrap();
        
        assert_eq!(distances.len(), 3);
        
        // First vector is identical - cosine distance should be very close to 0
        assert!(distances[0] < 1e-6, "Distance to identical vector: {}", distances[0]);
        
        // Second vector is opposite - cosine distance should be close to 2
        assert!((distances[1] - 2.0).abs() < 1e-5, "Distance to opposite vector: {}", distances[1]);
        
        // Third vector should have some distance
        assert!(distances[2] > 0.0 && distances[2] < 2.0, "Distance to orthogonal vector: {}", distances[2]);
    }

    #[test]
    fn test_set_and_get_global_gpu_context() {
        // Test setting and getting GPU context
        let ctx = ComputeContext { backend: ComputeBackend::Cuda, device_id: 0 };
        
        // Initially should be None
        assert!(get_global_gpu_context().is_none());
        
        // Set context
        set_global_gpu_context(Some(ctx));
        
        // Should now return the context
        let retrieved = get_global_gpu_context();
        assert!(retrieved.is_some());
        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.backend, ComputeBackend::Cuda);
        assert_eq!(retrieved.device_id, 0);
        
        // Clear context
        set_global_gpu_context(None);
        
        // Should be None again
        assert!(get_global_gpu_context().is_none());
    }

    #[test]
    fn test_global_gpu_context_thread_isolation() {
        use std::thread;
        use std::sync::{Arc, Barrier};
        
        // Test that each thread has its own context
        let barrier = Arc::new(Barrier::new(3));
        
        let mut handles = vec![];
        
        // Thread 1: Set CUDA context
        let barrier1 = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            let ctx = ComputeContext { backend: ComputeBackend::Cuda, device_id: 0 };
            set_global_gpu_context(Some(ctx));
            
            barrier1.wait();
            
            // Verify this thread still has CUDA context
            let retrieved = get_global_gpu_context();
            assert!(retrieved.is_some());
            assert_eq!(retrieved.unwrap().backend, ComputeBackend::Cuda);
        }));
        
        // Thread 2: Set MPS context
        let barrier2 = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            let ctx = ComputeContext { backend: ComputeBackend::Mps, device_id: 1 };
            set_global_gpu_context(Some(ctx));
            
            barrier2.wait();
            
            // Verify this thread still has MPS context
            let retrieved = get_global_gpu_context();
            assert!(retrieved.is_some());
            let retrieved = retrieved.unwrap();
            assert_eq!(retrieved.backend, ComputeBackend::Mps);
            assert_eq!(retrieved.device_id, 1);
        }));
        
        // Thread 3: Don't set any context
        let barrier3 = Arc::clone(&barrier);
        handles.push(thread::spawn(move || {
            barrier3.wait();
            
            // Verify this thread has no context
            assert!(get_global_gpu_context().is_none());
        }));
        
        // Wait for all threads to complete
        for handle in handles {
            handle.join().unwrap();
        }
    }

    #[test]
    fn test_global_gpu_context_multiple_updates() {
        // Test updating context multiple times
        let ctx1 = ComputeContext { backend: ComputeBackend::Cuda, device_id: 0 };
        let ctx2 = ComputeContext { backend: ComputeBackend::Mps, device_id: 1 };
        let ctx3 = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
        
        // Set first context
        set_global_gpu_context(Some(ctx1));
        let retrieved = get_global_gpu_context().unwrap();
        assert_eq!(retrieved.backend, ComputeBackend::Cuda);
        assert_eq!(retrieved.device_id, 0);
        
        // Update to second context
        set_global_gpu_context(Some(ctx2));
        let retrieved = get_global_gpu_context().unwrap();
        assert_eq!(retrieved.backend, ComputeBackend::Mps);
        assert_eq!(retrieved.device_id, 1);
        
        // Update to third context
        set_global_gpu_context(Some(ctx3));
        let retrieved = get_global_gpu_context().unwrap();
        assert_eq!(retrieved.backend, ComputeBackend::Cpu);
        assert_eq!(retrieved.device_id, -1);
        
        // Clear context
        set_global_gpu_context(None);
        assert!(get_global_gpu_context().is_none());
    }


}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    // Feature: python-vector-api-gpu-acceleration, Property 8: CPU Fallback
    // **Validates: Requirements 4.6**
    //
    // Property: For any distance computation where GPU kernel is unavailable or fails,
    // the system should automatically fall back to CPU computation and produce correct results.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn test_cpu_fallback_for_unimplemented_kernels(
            dim in 2usize..128,
            n_vectors in 1usize..10,
            metric_idx in 0usize..6,
        ) {
            // Generate random query vector
            let query: Vec<f32> = (0..dim)
                .map(|i| ((i as f32 + 1.0) * 0.1) % 10.0)
                .collect();
            
            // Generate random database vectors
            let mut vectors = Vec::with_capacity(n_vectors * dim);
            for v in 0..n_vectors {
                for d in 0..dim {
                    vectors.push(((v * dim + d) as f32 * 0.2) % 10.0);
                }
            }
            
            // Map index to metric
            let metric = match metric_idx {
                0 => VectorMetric::L2,
                1 => VectorMetric::Cosine,
                2 => VectorMetric::InnerProduct,
                3 => VectorMetric::L1,
                4 => VectorMetric::Hamming,
                _ => VectorMetric::Jaccard,
            };
            
            // Test CPU context (baseline)
            let cpu_context = ComputeContext { backend: ComputeBackend::Cpu, device_id: -1 };
            let cpu_distances = compute_distance(&query, &vectors, dim, metric, &cpu_context)
                .expect("CPU computation should always succeed");
            
            prop_assert_eq!(cpu_distances.len(), n_vectors,
                "CPU should return correct number of distances");
            
            // Test GPU contexts with fallback behavior
            // For unimplemented kernels (non-L2 metrics), GPU backends should fall back to CPU
            // and produce identical results
            
            #[cfg(feature = "cuda")]
            {
                let cuda_context = ComputeContext { backend: ComputeBackend::Cuda, device_id: 0 };
                
                // For non-L2 metrics, CUDA should fall back to CPU
                if metric != VectorMetric::L2 {
                    let cuda_distances = compute_distance(&query, &vectors, dim, metric, &cuda_context)
                        .expect("CUDA should fall back to CPU for unimplemented kernels");
                    
                    prop_assert_eq!(cuda_distances.len(), n_vectors,
                        "CUDA fallback should return correct number of distances");
                    
                    // Verify results match CPU computation
                    for i in 0..n_vectors {
                        let diff = (cuda_distances[i] - cpu_distances[i]).abs();
                        let tolerance = cpu_distances[i].abs() * 1e-5 + 1e-6;
                        prop_assert!(diff <= tolerance,
                            "CUDA fallback result mismatch at index {}: cuda={}, cpu={}, diff={}",
                            i, cuda_distances[i], cpu_distances[i], diff);
                    }
                }
            }
            
            #[cfg(all(target_os = "macos", feature = "mps"))]
            {
                let mps_context = ComputeContext { backend: ComputeBackend::Mps, device_id: 0 };
                
                // For non-L2 metrics, MPS should fall back to CPU
                if metric != VectorMetric::L2 {
                    let mps_distances = compute_distance(&query, &vectors, dim, metric, &mps_context)
                        .expect("MPS should fall back to CPU for unimplemented kernels");
                    
                    prop_assert_eq!(mps_distances.len(), n_vectors,
                        "MPS fallback should return correct number of distances");
                    
                    // Verify results match CPU computation
                    for i in 0..n_vectors {
                        let diff = (mps_distances[i] - cpu_distances[i]).abs();
                        let tolerance = cpu_distances[i].abs() * 1e-5 + 1e-6;
                        prop_assert!(diff <= tolerance,
                            "MPS fallback result mismatch at index {}: mps={}, cpu={}, diff={}",
                            i, mps_distances[i], cpu_distances[i], diff);
                    }
                }
            }
            
            #[cfg(feature = "rocm")]
            {
                let rocm_context = ComputeContext { backend: ComputeBackend::Rocm, device_id: 0 };
                
                // For non-L2 metrics, ROCm should fall back to CPU
                if metric != VectorMetric::L2 {
                    let rocm_distances = compute_distance(&query, &vectors, dim, metric, &rocm_context)
                        .expect("ROCm should fall back to CPU for unimplemented kernels");
                    
                    prop_assert_eq!(rocm_distances.len(), n_vectors,
                        "ROCm fallback should return correct number of distances");
                    
                    // Verify results match CPU computation
                    for i in 0..n_vectors {
                        let diff = (rocm_distances[i] - cpu_distances[i]).abs();
                        let tolerance = cpu_distances[i].abs() * 1e-5 + 1e-6;
                        prop_assert!(diff <= tolerance,
                            "ROCm fallback result mismatch at index {}: rocm={}, cpu={}, diff={}",
                            i, rocm_distances[i], cpu_distances[i], diff);
                    }
                }
            }
            
            #[cfg(feature = "intel")]
            {
                let intel_context = ComputeContext { backend: ComputeBackend::Intel, device_id: 0 };
                
                // For non-L2 metrics, Intel GPU should fall back to CPU
                if metric != VectorMetric::L2 {
                    let intel_distances = compute_distance(&query, &vectors, dim, metric, &intel_context)
                        .expect("Intel GPU should fall back to CPU for unimplemented kernels");
                    
                    prop_assert_eq!(intel_distances.len(), n_vectors,
                        "Intel GPU fallback should return correct number of distances");
                    
                    // Verify results match CPU computation
                    for i in 0..n_vectors {
                        let diff = (intel_distances[i] - cpu_distances[i]).abs();
                        let tolerance = cpu_distances[i].abs() * 1e-5 + 1e-6;
                        prop_assert!(diff <= tolerance,
                            "Intel GPU fallback result mismatch at index {}: intel={}, cpu={}, diff={}",
                            i, intel_distances[i], cpu_distances[i], diff);
                    }
                }
            }
        }
    }

}

/// Accelerated K-Means assignment for all backends
pub fn compute_kmeans_assignment(vectors: &[f32], centroids: &[f32], dim: usize, context: &ComputeContext) -> Result<Vec<u32>> {
    println!("Using {:?} backend for K-Means assignment", context.backend);
    match context.backend {
        ComputeBackend::Cpu => compute_kmeans_assignment_cpu(vectors, centroids, dim),
        ComputeBackend::Cuda => compute_kmeans_assignment_cuda(vectors, centroids, dim),
        ComputeBackend::Mps => compute_kmeans_assignment_mps(vectors, centroids, dim),
        ComputeBackend::Rocm => compute_kmeans_assignment_opencl(vectors, centroids, dim, Some("AMD")),
        ComputeBackend::Intel => compute_kmeans_assignment_opencl(vectors, centroids, dim, Some("Intel")),
    }
}

fn compute_kmeans_assignment_cpu(vectors: &[f32], centroids: &[f32], dim: usize) -> Result<Vec<u32>> {
    let n_vectors = vectors.len() / dim;
    let k = centroids.len() / dim;
    let mut labels = Vec::with_capacity(n_vectors);

    for i in 0..n_vectors {
        let vec = &vectors[i * dim..(i + 1) * dim];
        let mut min_dist = f32::MAX;
        let mut best_idx = 0u32;

        for j in 0..k {
            let centroid = &centroids[j * dim..(j + 1) * dim];
            let mut dist = 0.0f32;
            for (v1, v2) in vec.iter().zip(centroid.iter()) {
                let diff = v1 - v2;
                dist += diff * diff;
            }

            if dist < min_dist {
                min_dist = dist;
                best_idx = j as u32;
            }
        }
        labels.push(best_idx);
    }
    Ok(labels)
}

#[cfg(all(target_os = "macos", feature = "mps"))]
fn compute_kmeans_assignment_mps(vectors: &[f32], centroids: &[f32], dim: usize) -> Result<Vec<u32>> {
    use metal::*;
    use std::mem;

    let device = Device::system_default()
        .ok_or_else(|| anyhow::anyhow!("No Metal device found"))?;
    let command_queue = device.new_command_queue();
    
    let library = device.new_library_with_source(MSL_KMEANS, &CompileOptions::new())
        .map_err(|e| anyhow::anyhow!("Metal compilation error: {}", e))?;
    let kernel = library.get_function("kmeans_assignment", None)
        .map_err(|e| anyhow::anyhow!("Metal function error: {}", e))?;
    let pipeline_state = device.new_compute_pipeline_state_with_function(&kernel)
        .map_err(|e| anyhow::anyhow!("Metal pipeline error: {}", e))?;

    let n_vectors = vectors.len() / dim;
    let k = centroids.len() / dim;

    let vectors_buffer = device.new_buffer_with_data(
        vectors.as_ptr() as *const _,
        (vectors.len() * mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared
    );
    let centroids_buffer = device.new_buffer_with_data(
        centroids.as_ptr() as *const _,
        (centroids.len() * mem::size_of::<f32>()) as u64,
        MTLResourceOptions::StorageModeShared
    );
    let labels_buffer = device.new_buffer(
        (n_vectors * mem::size_of::<u32>()) as u64,
        MTLResourceOptions::StorageModeShared
    );

    let batch_size_u32 = n_vectors as u32;
    let k_u32 = k as u32;
    let dim_u32 = dim as u32;

    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();
    
    encoder.set_compute_pipeline_state(&pipeline_state);
    encoder.set_buffer(0, Some(&vectors_buffer), 0);
    encoder.set_buffer(1, Some(&centroids_buffer), 0);
    encoder.set_buffer(2, Some(&labels_buffer), 0);
    encoder.set_bytes(3, mem::size_of::<u32>() as u64, &batch_size_u32 as *const _ as *const _);
    encoder.set_bytes(4, mem::size_of::<u32>() as u64, &k_u32 as *const _ as *const _);
    encoder.set_bytes(5, mem::size_of::<u32>() as u64, &dim_u32 as *const _ as *const _);

    let threads_per_group = MTLSize::new(256, 1, 1);
    let groups = MTLSize::new((n_vectors as u64 + 255) / 256, 1, 1);
    
    encoder.dispatch_thread_groups(groups, threads_per_group);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let labels_ptr = labels_buffer.contents() as *mut u32;
    unsafe {
        Ok(std::slice::from_raw_parts(labels_ptr, n_vectors).to_vec())
    }
}

#[cfg(not(all(target_os = "macos", feature = "mps")))]
fn compute_kmeans_assignment_mps(vectors: &[f32], centroids: &[f32], dim: usize) -> Result<Vec<u32>> {
    compute_kmeans_assignment_cpu(vectors, centroids, dim)
}

#[cfg(feature = "cuda")]
fn compute_kmeans_assignment_cuda(vectors: &[f32], centroids: &[f32], dim: usize) -> Result<Vec<u32>> {
    use cust::prelude::*;
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(CUDA_KMEANS, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    let n_vectors = vectors.len() / dim;
    let k = centroids.len() / dim;

    let d_vectors = DeviceBuffer::from_slice(vectors)?;
    let d_centroids = DeviceBuffer::from_slice(centroids)?;
    let d_labels = DeviceBuffer::from_slice(&vec![0u32; n_vectors])?;

    let func = module.get_function("kmeans_assignment")?;
    let grid_size = (n_vectors as u32 + 255) / 256;
    
    unsafe {
        launch!(
            func<<<grid_size, 256, 0, stream>>>(
                d_vectors.as_device_ptr(),
                d_centroids.as_device_ptr(),
                d_labels.as_device_ptr(),
                n_vectors as u32,
                k as u32,
                dim as u32
            )
        )?;
    }

    stream.synchronize()?;
    let mut labels = vec![0u32; n_vectors];
    d_labels.copy_to(&mut labels)?;
    Ok(labels)
}

#[cfg(not(feature = "cuda"))]
fn compute_kmeans_assignment_cuda(vectors: &[f32], centroids: &[f32], dim: usize) -> Result<Vec<u32>> {
    compute_kmeans_assignment_cpu(vectors, centroids, dim)
}

#[cfg(any(feature = "intel", feature = "rocm"))]
fn compute_kmeans_assignment_opencl(vectors: &[f32], centroids: &[f32], dim: usize, platform_filter: Option<&str>) -> Result<Vec<u32>> {
    use opencl3::command_queue::{CommandQueue, CL_BLOCKING};
    use opencl3::context::Context;
    use opencl3::device::{get_all_devices, Device, CL_DEVICE_TYPE_GPU};
    use opencl3::kernel::{ExecuteKernel, Kernel};
    use opencl3::memory::{Buffer, CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
    use opencl3::program::Program;
    use std::ptr;

    let n_vectors = vectors.len() / dim;
    let k = centroids.len() / dim;

    let platform_filter = platform_filter.unwrap_or("");
    let device_id = get_all_devices(CL_DEVICE_TYPE_GPU)?
        .into_iter()
        .find(|&id| {
            if platform_filter.is_empty() { return true; }
            let device = Device::new(id);
            device.vendor().map(|v| v.contains(platform_filter)).unwrap_or(false)
        })
        .ok_or_else(|| anyhow::anyhow!("No suitable OpenCL GPU found"))?;
    
    let device = Device::new(device_id);
    let cl_context = Context::from_device(&device)?;
    let queue = unsafe { CommandQueue::create_with_properties(&cl_context, device_id, 0, 0)? };
    let program = Program::create_and_build_from_source(&cl_context, OPENCL_KMEANS, "")
        .map_err(|e| anyhow::anyhow!("OpenCL build error: {}", e))?;
    let kernel = Kernel::create(&program, "kmeans_assignment")?;

    let mut d_vectors = unsafe { Buffer::<f32>::create(&cl_context, CL_MEM_READ_ONLY, vectors.len(), ptr::null_mut())? };
    let mut d_centroids = unsafe { Buffer::<f32>::create(&cl_context, CL_MEM_READ_ONLY, centroids.len(), ptr::null_mut())? };
    let d_labels = unsafe { Buffer::<u32>::create(&cl_context, CL_MEM_WRITE_ONLY, n_vectors, ptr::null_mut())? };

    unsafe {
        queue.enqueue_write_buffer(&mut d_vectors, CL_BLOCKING, 0, vectors, &[])?;
        queue.enqueue_write_buffer(&mut d_centroids, CL_BLOCKING, 0, centroids, &[])?;

        ExecuteKernel::new(&kernel)
            .set_arg(&d_vectors)
            .set_arg(&d_centroids)
            .set_arg(&d_labels)
            .set_arg(&(n_vectors as u32))
            .set_arg(&(k as u32))
            .set_arg(&(dim as u32))
            .set_global_work_size(n_vectors)
            .enqueue_nd_range(&queue)?;
        
        let mut labels = vec![0u32; n_vectors];
        queue.enqueue_read_buffer(&d_labels, CL_BLOCKING, 0, &mut labels, &[])?;
        Ok(labels)
    }
}

#[cfg(not(any(feature = "intel", feature = "rocm")))]
fn compute_kmeans_assignment_opencl(vectors: &[f32], centroids: &[f32], dim: usize, _pf: Option<&str>) -> Result<Vec<u32>> {
    compute_kmeans_assignment_cpu(vectors, centroids, dim)
}
