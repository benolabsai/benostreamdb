// Copyright (c) 2026 Richard Albright. All rights reserved.

use pyo3::prelude::*;
use crate::core::index::gpu::{ComputeContext, ComputeBackend};
use std::sync::{Arc, Mutex};

/// Performance statistics for GPU operations
#[derive(Debug, Clone, Default)]
pub struct GPUStats {
    pub total_kernel_launches: u64,
    pub total_gpu_time_ms: f64,
    pub total_cpu_time_ms: f64,
    pub total_vectors_processed: u64,
    pub memory_transfers_mb: f64,
}

/// Python wrapper for ComputeContext with performance monitoring
#[pyclass(name = "ComputeContext")]
pub struct PyComputeContext {
    pub(crate) context: ComputeContext,
    pub(crate) stats: Arc<Mutex<GPUStats>>,
}

#[pymethods]
#[allow(deprecated)]
impl PyComputeContext {
    /// Detect and return the best available GPU backend
    /// 
    /// Automatically detects and selects the highest-priority GPU backend available
    /// on the system. Priority order: CUDA > ROCm > MPS > Intel > CPU.
    /// 
    /// Returns
    /// -------
    /// ComputeContext
    ///     Context configured with the highest-priority available backend
    /// 
    /// Examples
    /// --------
    /// >>> import hyperstreamdb as hdb
    /// >>> ctx = hdb.ComputeContext.auto_detect()
    /// >>> print(f"Using backend: {ctx.backend}")
    /// Using backend: cuda
    #[staticmethod]
    fn auto_detect() -> Self {
        let context = ComputeContext::auto_detect();
        Self {
            context,
            stats: Arc::new(Mutex::new(GPUStats::default())),
        }
    }

    /// Create context with specific backend
    /// 
    /// Creates a ComputeContext configured to use a specific GPU backend and device.
    /// This allows explicit control over which GPU hardware is used for computations.
    /// 
    /// Parameters
    /// ----------
    /// backend : str
    ///     Backend name: 'cuda', 'rocm', 'mps', 'intel', or 'cpu'
    /// device_id : int, optional
    ///     GPU device ID (default: 0). Use -1 for CPU backend.
    /// 
    /// Returns
    /// -------
    /// ComputeContext
    ///     Context configured with the specified backend
    /// 
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the requested backend is not available on this system
    /// ValueError
    ///     If an unknown backend name is provided
    /// 
    /// Examples
    /// --------
    /// >>> import hyperstreamdb as hdb
    /// >>> # Use CUDA GPU 0
    /// >>> ctx = hdb.ComputeContext('cuda', device_id=0)
    /// >>> print(ctx.backend)
    /// cuda
    /// 
    /// >>> # Force CPU computation
    /// >>> ctx_cpu = hdb.ComputeContext('cpu')
    /// >>> print(ctx_cpu.backend)
    /// cpu
    #[new]
    #[pyo3(signature = (backend, device_id=0))]
    fn new(backend: &str, device_id: i32) -> PyResult<Self> {
        let backend_enum = match backend.to_lowercase().as_str() {
            "cpu" => ComputeBackend::Cpu,
            "cuda" => {
                #[cfg(feature = "cuda")]
                {
                    ComputeBackend::Cuda
                }
                #[cfg(not(feature = "cuda"))]
                {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("Backend 'cuda' not available. Available backends: {:?}", 
                            Self::list_available_backends_internal())
                    ));
                }
            }
            "rocm" => {
                #[cfg(feature = "rocm")]
                {
                    ComputeBackend::Rocm
                }
                #[cfg(not(feature = "rocm"))]
                {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("Backend 'rocm' not available. Available backends: {:?}", 
                            Self::list_available_backends_internal())
                    ));
                }
            }
            "mps" => {
                #[cfg(feature = "mps")]
                {
                    ComputeBackend::Mps
                }
                #[cfg(not(feature = "mps"))]
                {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("Backend 'mps' not available. Available backends: {:?}", 
                            Self::list_available_backends_internal())
                    ));
                }
            }
            "intel" => {
                #[cfg(feature = "intel")]
                {
                    ComputeBackend::Intel
                }
                #[cfg(not(feature = "intel"))]
                {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        format!("Backend 'intel' not available. Available backends: {:?}", 
                            Self::list_available_backends_internal())
                    ));
                }
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unknown backend '{}'. Valid backends: 'cuda', 'rocm', 'mps', 'intel', 'cpu'", backend)
                ));
            }
        };

        Ok(Self {
            context: ComputeContext {
                backend: backend_enum,
                device_id,
            },
            stats: Arc::new(Mutex::new(GPUStats::default())),
        })
    }

    /// Get current backend name
    /// 
    /// Returns
    /// -------
    /// str
    ///     Backend name: 'cuda', 'rocm', 'mps', 'intel', or 'cpu'
    /// 
    /// Examples
    /// --------
    /// >>> import hyperstreamdb as hdb
    /// >>> ctx = hdb.ComputeContext.auto_detect()
    /// >>> print(ctx.backend)
    /// cuda
    #[getter]
    fn backend(&self) -> String {
        match self.context.backend {
            ComputeBackend::Cpu => "cpu".to_string(),
            ComputeBackend::Cuda => "cuda".to_string(),
            ComputeBackend::Rocm => "rocm".to_string(),
            ComputeBackend::Mps => "mps".to_string(),
            ComputeBackend::Intel => "intel".to_string(),
        }
    }

    /// Get current device ID
    /// 
    /// Returns
    /// -------
    /// int
    ///     Device ID (0 for first GPU, 1 for second GPU, etc., -1 for CPU)
    /// 
    /// Examples
    /// --------
    /// >>> import hyperstreamdb as hdb
    /// >>> ctx = hdb.ComputeContext('cuda', device_id=1)
    /// >>> print(ctx.device_id)
    /// 1
    #[getter]
    fn device_id(&self) -> i32 {
        self.context.device_id
    }

    /// List all available GPU backends on this system
    /// 
    /// Queries the system to determine which GPU backends are available and can be used
    /// for acceleration. The CPU backend is always available as a fallback.
    /// 
    /// Returns
    /// -------
    /// list of str
    ///     List of available backend names (e.g., ['cuda', 'cpu'])
    /// 
    /// Examples
    /// --------
    /// >>> import hyperstreamdb as hdb
    /// >>> backends = hdb.ComputeContext.list_available_backends()
    /// >>> print(backends)
    /// ['cuda', 'cpu']
    /// >>> 
    /// >>> # Check if CUDA is available
    /// >>> if 'cuda' in backends:
    /// ...     ctx = hdb.ComputeContext('cuda')
    /// ... else:
    /// ...     ctx = hdb.ComputeContext('cpu')
    #[staticmethod]
    fn list_available_backends() -> Vec<String> {
        Self::list_available_backends_internal()
    }

    /// Get performance statistics
    /// 
    /// Returns performance metrics collected during GPU operations. These statistics
    /// help monitor GPU utilization and identify performance bottlenecks.
    /// 
    /// Returns
    /// -------
    /// dict
    ///     Dictionary with performance metrics:
    ///     
    ///     - total_kernel_launches : int
    ///         Number of GPU kernel launches
    ///     - total_gpu_time_ms : float
    ///         Total GPU computation time in milliseconds
    ///     - total_cpu_time_ms : float
    ///         Total CPU computation time in milliseconds
    ///     - total_vectors_processed : int
    ///         Total number of vectors processed
    ///     - memory_transfers_mb : float
    ///         Total memory transferred to/from GPU in megabytes
    /// 
    /// Examples
    /// --------
    /// >>> import hyperstreamdb as hdb
    /// >>> import numpy as np
    /// >>> ctx = hdb.ComputeContext.auto_detect()
    /// >>> 
    /// >>> # Perform some computations
    /// >>> query = np.random.rand(128).astype(np.float32)
    /// >>> vectors = np.random.rand(10000, 128).astype(np.float32)
    /// >>> distances = hdb.l2_batch(query, vectors, context=ctx)
    /// >>> 
    /// >>> # Check performance stats
    /// >>> stats = ctx.get_stats()
    /// >>> print(f"GPU time: {stats['total_gpu_time_ms']:.2f}ms")
    /// GPU time: 5.23ms
    /// >>> print(f"Vectors processed: {stats['total_vectors_processed']}")
    /// Vectors processed: 10000
    fn get_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let stats = self.stats.lock().unwrap();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("total_kernel_launches", stats.total_kernel_launches)?;
        dict.set_item("total_gpu_time_ms", stats.total_gpu_time_ms)?;
        dict.set_item("total_cpu_time_ms", stats.total_cpu_time_ms)?;
        dict.set_item("total_vectors_processed", stats.total_vectors_processed)?;
        dict.set_item("memory_transfers_mb", stats.memory_transfers_mb)?;
        Ok(dict.into())
    }

    /// Reset performance counters
    /// 
    /// Resets all performance statistics to zero. Useful for measuring performance
    /// of specific operations or time periods.
    /// 
    /// Examples
    /// --------
    /// >>> import hyperstreamdb as hdb
    /// >>> ctx = hdb.ComputeContext.auto_detect()
    /// >>> 
    /// >>> # Perform some computations
    /// >>> # ... operations ...
    /// >>> 
    /// >>> # Reset counters before measuring specific operation
    /// >>> ctx.reset_stats()
    /// >>> # ... perform operation to measure ...
    /// >>> stats = ctx.get_stats()
    fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = GPUStats::default();
    }

    /// Activate this context for the current thread
    /// 
    /// Sets this context as the global default for the current thread.
    /// Operations like index building will automatically use this GPU context.
    fn activate(&self) {
        crate::core::index::gpu::set_global_gpu_context(Some(self.context));
    }

    /// Deactivate the global context for the current thread
    /// 
    /// Clears the global GPU context for the current thread, falling back to CPU
    /// for operations that don't specify a context.
    #[staticmethod]
    fn deactivate() {
        crate::core::index::gpu::set_global_gpu_context(None);
    }

    fn __repr__(&self) -> String {
        format!("ComputeContext(backend='{}', device_id={})", self.backend(), self.device_id())
    }

    /// Perform K-Means clustering on a set of vectors.
    /// 
    /// This method uses the GPU backend associated with this context to accelerate
    /// the K-Means assignment phase.
    /// 
    /// Parameters
    /// ----------
    /// vectors : List[List[float]]
    ///     Input vectors to cluster
    /// k : int
    ///     Number of clusters
    /// max_iters : int, optional
    ///     Maximum number of iterations (default: 10)
    /// 
    /// Returns
    /// -------
    /// tuple (centroids, labels)
    ///     centroids : List[List[float]]
    ///         The computed cluster centroids
    ///     labels : List[int]
    ///         Cluster assignment for each input vector
    #[pyo3(signature = (vectors, k, max_iters=10))]
    fn kmeans(
        &self,
        py: Python<'_>,
        vectors: Vec<Vec<f32>>,
        k: usize,
        max_iters: usize,
    ) -> PyResult<(Vec<Vec<f32>>, Vec<usize>)> {
        // Temporarily activate this context for simple_kmeans if it's not already global
        let prev_ctx = crate::core::index::gpu::get_global_gpu_context();
        crate::core::index::gpu::set_global_gpu_context(Some(self.context));
        
        let result = py.allow_threads(|| {
            crate::core::index::ivf::simple_kmeans(&vectors, k, max_iters)
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()));
            
        // Restore previous context
        crate::core::index::gpu::set_global_gpu_context(prev_ctx);
        
        result
    }
}

impl PyComputeContext {
    /// Internal helper to list available backends
    fn list_available_backends_internal() -> Vec<String> {
        let mut backends = vec!["cpu".to_string()];
        
        #[cfg(feature = "cuda")]
        backends.push("cuda".to_string());
        
        #[cfg(feature = "rocm")]
        backends.push("rocm".to_string());
        
        #[cfg(feature = "mps")]
        backends.push("mps".to_string());
        
        #[cfg(feature = "intel")]
        backends.push("intel".to_string());
        
        backends
    }

    /// Get the underlying Rust ComputeContext
    pub fn get_context(&self) -> &ComputeContext {
        &self.context
    }

    /// Get the stats tracker for recording performance metrics
    pub fn get_stats_tracker(&self) -> Arc<Mutex<GPUStats>> {
        self.stats.clone()
    }
}
