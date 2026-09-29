// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

#[cfg(target_os = "macos")]
use anyhow::Result;
#[cfg(target_os = "macos")]
use crate::metric::Metric;
#[cfg(target_os = "macos")]
use super::GpuBackend;

#[cfg(target_os = "macos")]
const MSL_L2: &str = include_str!("../kernels/metal/l2_distance.metal");
#[cfg(target_os = "macos")]
const MSL_COSINE: &str = include_str!("../kernels/metal/cosine_distance.metal");
#[cfg(target_os = "macos")]
const MSL_INNER_PRODUCT: &str = include_str!("../kernels/metal/inner_product.metal");
#[cfg(target_os = "macos")]
const MSL_L1: &str = include_str!("../kernels/metal/l1_distance.metal");
#[cfg(target_os = "macos")]
const MSL_HAMMING: &str = include_str!("../kernels/metal/hamming_distance.metal");
#[cfg(target_os = "macos")]
const MSL_JACCARD: &str = include_str!("../kernels/metal/jaccard_distance.metal");
#[cfg(target_os = "macos")]
const MSL_KMEANS: &str = include_str!("../kernels/metal/kmeans_assignment.metal");

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
        "Apple Metal (MPS)"
    }

    fn compute_distance(
        &self,
        query: &[f32],
        vectors: &[f32],
        dim: usize,
        metric: Metric,
    ) -> Result<Vec<f32>> {
        use metal::*;
        let (src, name) = match metric {
            Metric::L2 => (MSL_L2, "l2_distance_kernel"),
            Metric::Cosine => (MSL_COSINE, "cosine_distance_kernel"),
            Metric::InnerProduct => (MSL_INNER_PRODUCT, "inner_product_kernel"),
            Metric::L1 => (MSL_L1, "l1_distance_kernel"),
            Metric::Hamming => (MSL_HAMMING, "hamming_distance_kernel"),
            Metric::Jaccard => (MSL_JACCARD, "jaccard_distance_kernel"),
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
}
