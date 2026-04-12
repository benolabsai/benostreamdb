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

/// Python wrapper for Device with performance monitoring
#[pyclass(name = "Device")]
pub struct PyDevice {
    pub(crate) context: ComputeContext,
    pub(crate) stats: Arc<Mutex<GPUStats>>,
}

#[pymethods]
impl PyDevice {
    /// Create a new Device instance.
    #[new]
    #[pyo3(signature = (device, index=None))]
    fn new(device: &str, index: Option<i32>) -> PyResult<Self> {
        let (backend_str, device_id) = if device.contains(':') {
             let parts: Vec<&str> = device.split(':').collect();
             if parts.len() != 2 {
                 return Err(pyo3::exceptions::PyValueError::new_err("Device string must be in 'type:index' format (e.g., 'cuda:0')"));
             }
             let b = parts[0];
             let i = parts[1].parse::<i32>().map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid device index"))?;
             (b, i)
        } else {
             (device, index.unwrap_or(if device == "cpu" { -1 } else { 0 }))
        };

        let backend_enum = match backend_str.to_lowercase().as_str() {
            "cpu" => ComputeBackend::Cpu,
            "cuda" => {
                #[cfg(feature = "cuda")]
                { ComputeBackend::Cuda }
                #[cfg(not(feature = "cuda"))]
                { return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "CUDA backend not available in this build. Ensure NVIDIA drivers are installed and try 'pip install hyperstreamdb'. See GPU_SETUP_GUIDE.md for detailed troubleshooting."
                )); }
            }
            "mps" | "metal" => {
                #[cfg(target_os = "macos")]
                { ComputeBackend::Mps }
                #[cfg(not(target_os = "macos"))]
                { return Err(pyo3::exceptions::PyRuntimeError::new_err("Backend 'mps' is only available on macOS.")); }
            }
            "intel" | "graphics" => {
                #[cfg(target_os = "linux")]
                { ComputeBackend::Intel }
                #[cfg(not(target_os = "linux"))]
                { return Err(pyo3::exceptions::PyRuntimeError::new_err("Backend 'intel' (XPU) is only supported on Linux.")); }
            }
            "rocm" => {
                #[cfg(target_os = "linux")]
                { ComputeBackend::Rocm }
                #[cfg(not(target_os = "linux"))]
                { return Err(pyo3::exceptions::PyRuntimeError::new_err("Backend 'rocm' is only supported on Linux.")); }
            }
            "auto" | "gpu" => {
                let context = crate::core::index::gpu::ComputeContext::auto_detect();
                return Ok(Self {
                    context,
                    stats: Arc::new(Mutex::new(GPUStats::default())),
                });
            }
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!("Unknown device type '{}'. Valid types: 'cpu', 'cuda', 'mps', 'intel', 'rocm', 'gpu', 'auto'", backend_str)));
            }
        };

        Ok(Self {
            context: ComputeContext {
                backend: backend_enum,
                device_id,
                implementation: None,
            },
            stats: Arc::new(Mutex::new(GPUStats::default())),
        })
    }

    /// Check if a specific device type is available on this system.
    #[staticmethod]
    fn is_available(device_type: &str) -> bool {
        let backend = match device_type.to_lowercase().as_str() {
            "cpu" => ComputeBackend::Cpu,
            "cuda" => ComputeBackend::Cuda,
            "mps" | "metal" => ComputeBackend::Mps,
            "intel" | "graphics" => ComputeBackend::Intel,
            "rocm" => ComputeBackend::Rocm,
            _ => return false,
        };
        ComputeContext { backend, device_id: 0, implementation: None }.is_available()
    }

    #[getter]
    fn type_name(&self) -> String {
        match self.context.backend {
            ComputeBackend::Cpu => "cpu".to_string(),
            ComputeBackend::Cuda => "cuda".to_string(),
            ComputeBackend::Rocm => "rocm".to_string(),
            ComputeBackend::Mps => "mps".to_string(),
            ComputeBackend::Intel => "intel".to_string(),
        }
    }

    #[getter]
    fn index(&self) -> i32 {
        self.context.device_id
    }

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

    fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap();
        *stats = GPUStats::default();
    }

    fn activate(&self) {
        crate::core::index::gpu::set_global_gpu_context(Some(self.context.clone()));
    }

    #[staticmethod]
    fn deactivate() {
        crate::core::index::gpu::set_global_gpu_context(None);
    }

    fn __repr__(&self) -> String {
        format!("Device(type='{}', index={})", self.type_name(), self.index())
    }

    #[pyo3(signature = (vectors, k, max_iters=10))]
    fn kmeans(
        &self,
        py: Python<'_>,
        vectors: Vec<Vec<f32>>,
        k: usize,
        max_iters: usize,
    ) -> PyResult<(Vec<Vec<f32>>, Vec<usize>)> {
        #[allow(deprecated)]
        py.allow_threads(|| {
            crate::core::index::ivf::simple_kmeans(&vectors, k, max_iters)
        }).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
    }
}

impl PyDevice {
    pub fn get_context(&self) -> &ComputeContext {
        &self.context
    }

    pub fn get_stats_tracker(&self) -> Arc<Mutex<GPUStats>> {
        self.stats.clone()
    }
}
