//! Thin wgpu helpers shared by the GPU experiments.

use std::time::Instant;

pub struct Gpu {
  pub device: wgpu::Device,
  pub queue: wgpu::Queue,
  pub info: wgpu::AdapterInfo,
  pub limits: wgpu::Limits,
}

impl Gpu {
  /// `backend`: "vulkan", "dx12" or None (wgpu's default order).
  pub fn new(backend: Option<&str>) -> Gpu {
    Self::new_with(backend, true)
  }

  /// `indirect_validation = false` drops `InstanceFlags::VALIDATION_INDIRECT_CALL`, the extra
  /// work wgpu injects before every indirect dispatch.
  pub fn new_with(backend: Option<&str>, indirect_validation: bool) -> Gpu {
    let backends = match backend {
      Some("vulkan") => wgpu::Backends::VULKAN,
      Some("dx12") => wgpu::Backends::DX12,
      Some(other) => panic!("unknown backend {other}"),
      None => wgpu::Backends::PRIMARY,
    };
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = backends;
    if !indirect_validation {
      desc.flags.remove(wgpu::InstanceFlags::VALIDATION_INDIRECT_CALL);
    }
    let instance = wgpu::Instance::new(desc);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
      power_preference: wgpu::PowerPreference::HighPerformance,
      ..Default::default()
    }))
    .expect("no adapter");
    let info = adapter.get_info();
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
      label: Some("research"),
      required_features: wgpu::Features::empty(),
      required_limits: limits.clone(),
      ..Default::default()
    }))
    .expect("request_device failed");
    Gpu { device, queue, info, limits }
  }

  pub fn describe(&self) -> String {
    format!(
      "{} [{:?}, driver {}] max_buffer={} MiB max_storage_binding={} MiB",
      self.info.name,
      self.info.backend,
      self.info.driver_info,
      self.limits.max_buffer_size >> 20,
      self.limits.max_storage_buffer_binding_size >> 20
    )
  }

  /// Block until everything submitted so far has finished executing.
  pub fn wait(&self) {
    self.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll failed");
  }

  pub fn storage(&self, label: &str, size: u64) -> wgpu::Buffer {
    self.device.create_buffer(&wgpu::BufferDescriptor {
      label: Some(label),
      size,
      usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    })
  }

  pub fn storage_init(&self, label: &str, data: &[u8]) -> wgpu::Buffer {
    let size = (data.len() as u64 + 3) & !3;
    let buf = self.storage(label, size.max(4));
    // upload in chunks so that the staging belt never needs a giant allocation
    const CHUNK: usize = 64 << 20;
    let mut off = 0usize;
    while off < data.len() {
      let end = (off + CHUNK).min(data.len());
      let mut chunk = data[off..end].to_vec();
      while chunk.len() % 4 != 0 {
        chunk.push(0);
      }
      self.queue.write_buffer(&buf, off as u64, &chunk);
      self.queue.submit(None);
      off = end;
    }
    self.wait();
    buf
  }

  pub fn read(&self, buf: &wgpu::Buffer, offset: u64, size: u64) -> Vec<u8> {
    let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("staging"),
      size,
      usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    let mut enc = self.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, offset, &staging, 0, size);
    self.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
    self.wait();
    let data = slice.get_mapped_range().expect("mapped range").to_vec();
    staging.unmap();
    data
  }

  pub fn read_u32(&self, buf: &wgpu::Buffer, offset_words: u64, words: u64) -> Vec<u32> {
    bytemuck::pod_collect_to_vec(&self.read(buf, offset_words * 4, words * 4))
  }

  pub fn pipeline(&self, label: &str, wgsl: &str, entry: &str) -> wgpu::ComputePipeline {
    let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some(label),
      source: wgpu::ShaderSource::Wgsl(wgsl.into()),
    });
    let p = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label: Some(label),
      layout: None,
      module: &module,
      entry_point: Some(entry),
      compilation_options: Default::default(),
      cache: None,
    });
    if let Some(e) = pollster::block_on(scope.pop()) {
      panic!("shader/pipeline error in {label}:\n{e}");
    }
    p
  }

  pub fn bind(&self, pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers
      .iter()
      .enumerate()
      .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
      .collect();
    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &pipeline.get_bind_group_layout(0),
      entries: &entries,
    })
  }

  /// Run `f` to record + submit work, wait for completion, return elapsed seconds.
  pub fn timed(&self, f: impl FnOnce()) -> f64 {
    self.wait();
    let t = Instant::now();
    f();
    self.wait();
    t.elapsed().as_secs_f64()
  }
}
