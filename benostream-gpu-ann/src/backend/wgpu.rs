// Copyright (c) 2026 Richard Albright and BenoStreamDB Contributors.
// Licensed under MIT OR Apache-2.0.

#[cfg(feature = "wgpu")]
use super::GpuBackend;
#[cfg(feature = "wgpu")]
use crate::metric::Metric;
#[cfg(feature = "wgpu")]
use anyhow::Result;

#[cfg(feature = "wgpu")]
#[derive(Debug)]
pub struct WgpuBackend {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    name: String,
}

#[cfg(feature = "wgpu")]
impl WgpuBackend {
    pub fn new(display_name: &str, vendor_id: Option<u32>) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = if let Some(vid) = vendor_id {
            instance
                .enumerate_adapters(wgpu::Backends::all())
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
            .ok_or_else(|| anyhow::anyhow!("Failed to find WGPU adapter"))?
        };

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("Compute"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))?;

        let shader_src = include_str!("../kernels/wgsl/wgpu_kernel.wgsl");
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

#[cfg(feature = "wgpu")]
impl GpuBackend for WgpuBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn compute_distance(
        &self,
        query: &[f32],
        vectors: &[f32],
        dim: usize,
        metric: Metric,
    ) -> Result<Vec<f32>> {
        use wgpu::util::DeviceExt;

        fn as_u8_slice<T>(data: &[T]) -> &[u8] {
            unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
            }
        }

        let num_vectors = (vectors.len() / dim) as u32;
        let metric_type: u32 = metric.shader_type_id();

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
        buffer_slice.map_async(wgpu::MapMode::Read, move |v| {
            let _ = sender.send(v);
        });

        self.device.poll(wgpu::Maintain::Wait);

        if let Ok(Ok(())) = receiver.recv() {
            let data = buffer_slice.get_mapped_range();
            let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging_buffer.unmap();
            Ok(result)
        } else {
            anyhow::bail!("Failed to read WGPU staging buffer");
        }
    }

    fn compute_kmeans_assignment(
        &self,
        vectors: &[f32],
        centroids: &[f32],
        dim: usize,
    ) -> Result<Vec<u32>> {
        // Fall back to CPU assignment for WGPU until custom k-means compute shader is added
        crate::backend::cpu::CpuBackend::new().compute_kmeans_assignment(vectors, centroids, dim)
    }
}
