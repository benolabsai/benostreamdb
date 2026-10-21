// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.

use crate::core::index::gpu::{ComputeBackend, ComputeContext};
use pyo3::prelude::*;
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
    pub fn new(device: &str, index: Option<i32>) -> PyResult<Self> {
        let lower = device.trim().to_ascii_lowercase();
        let (backend_str, device_id) = if let Some((backend, idx)) = lower.split_once(':') {
            if idx.contains(':') {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Device string must be in 'type:index' format (e.g., 'cuda:0')",
                ));
            }
            let parsed = idx
                .parse::<i32>()
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("Invalid device index"))?;
            (backend, parsed)
        } else {
            let default_index = if lower == "cpu" { -1 } else { 0 };
            (lower.as_str(), index.unwrap_or(default_index))
        };

        // `auto`/`gpu`/empty keep detection semantics: these are the only
        // inputs allowed to choose a backend on the user's behalf.
        if backend_str.is_empty() || backend_str == "auto" || backend_str == "gpu" {
            return Ok(Self {
                context: ComputeContext::auto_detect(),
                stats: Arc::new(Mutex::new(GPUStats::default())),
            });
        }

        // Explicit devices resolve to exactly the requested backend + index or
        // fail. Never silently substitute the auto-detected device: doing so
        // made `device="cpu"` and `device="cuda:1"` run on GPU 0.
        let context = match backend_str {
            "cpu" => ComputeContext {
                backend: ComputeBackend::Cpu,
                device_id,
                implementation: Some(Arc::new(crate::core::index::gpu::CpuBackend)),
            },
            "cuda" => ComputeContext::cached(ComputeBackend::Cuda, device_id).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "CUDA device {} not available: {}",
                    device_id, e
                ))
            })?,
            "mps" | "metal" => ComputeContext::cached(ComputeBackend::Mps, 0).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "MPS device not available: {}",
                    e
                ))
            })?,
            "intel" | "graphics" | "xpu" => {
                ComputeContext::cached(ComputeBackend::Intel, device_id).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Intel GPU device {} not available: {}",
                        device_id, e
                    ))
                })?
            }
            "rocm" | "hip" => ComputeContext::cached(ComputeBackend::Rocm, device_id).map_err(
                |e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "ROCm device {} not available: {}",
                        device_id, e
                    ))
                },
            )?,
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Unknown device type '{}'. Valid types: 'cpu', 'cuda', 'mps', 'intel', 'rocm', 'gpu', 'auto'",
                    backend_str
                )))
            }
        };

        Ok(Self {
            context,
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
        ComputeContext {
            backend,
            device_id: 0,
            implementation: None,
        }
        .is_available()
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

    /// Whether this device resolved to a concrete backend implementation.
    ///
    /// `False` would mean the device was never initialised and any compute
    /// would have been silently redirected to another backend.
    #[getter]
    fn initialized(&self) -> bool {
        self.context.implementation.is_some()
    }

    fn get_stats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        // Recover from mutex poisoning rather than panicking: the stats struct
        // is plain data, so a previously panicked holder cannot leave it invalid.
        let stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("total_kernel_launches", stats.total_kernel_launches)?;
        dict.set_item("total_gpu_time_ms", stats.total_gpu_time_ms)?;
        dict.set_item("total_cpu_time_ms", stats.total_cpu_time_ms)?;
        dict.set_item("total_vectors_processed", stats.total_vectors_processed)?;
        dict.set_item("memory_transfers_mb", stats.memory_transfers_mb)?;
        Ok(dict.into())
    }

    fn reset_stats(&self) {
        let mut stats = self.stats.lock().unwrap_or_else(|e| e.into_inner());
        *stats = GPUStats::default();
    }

    /// Activate this device as the process-wide compute context.
    ///
    /// Benchmark and application harnesses use this to pin a run to an exact
    /// backend (e.g. `Device("cpu").activate()` or `Device("auto").activate()`),
    /// so the recorded `device` label reflects where the work actually ran.
    pub fn activate(&self) {
        crate::core::index::gpu::set_thread_gpu_context(Some(self.context.clone()));
    }

    #[staticmethod]
    pub(crate) fn deactivate() {
        crate::core::index::gpu::set_thread_gpu_context(None);
    }

    fn __repr__(&self) -> String {
        format!(
            "Device(type='{}', index={})",
            self.type_name(),
            self.index()
        )
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
        py.allow_threads(|| crate::core::index::ivf::simple_kmeans(&vectors, k, max_iters))
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
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
