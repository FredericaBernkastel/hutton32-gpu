//! The device, its queue and limits, and the helpers every engine uses: buffers, read-backs,
//! pipelines with or without naga's runtime checks, timing that keeps uploads out of the clock.
//! A UI can hand in its own `Device` and `Queue` ([`GpuContext::from_device`]) so that it draws from
//! the engine's buffers in place.

use std::{sync::Arc, time::Instant};

use crate::{Error, Result};

/// A device and its queue, cheap to clone.
#[derive(Clone)]
pub struct GpuContext {
  pub device: wgpu::Device,
  pub queue: wgpu::Queue,
  pub info: wgpu::AdapterInfo,
  pub limits: wgpu::Limits,
  /// compile kernels that passed their cell-exact tests without naga's index clamping and loop
  /// bounding (18 % on the Life shared-tile kernel)
  pub trusted: bool,
  scope: Arc<()>,
}

impl std::fmt::Debug for GpuContext {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "GpuContext({})", self.describe())
  }
}

impl GpuContext {
  /// The high-performance adapter of the primary backends, with all of its limits. wgpu's
  /// `WGPU_BACKEND` variable (`vulkan`, `dx12`, `metal`, comma-separated) restricts the backends
  /// tried: a backend whose driver or layers fail at instance creation takes the process with it, so
  /// this is how a user avoids it.
  pub fn new() -> Result<GpuContext> {
    let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
    desc.backends = wgpu::Backends::from_env().unwrap_or(wgpu::Backends::PRIMARY);
    let instance = wgpu::Instance::new(desc);
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }))
      .map_err(|e| Error::Gpu(format!("no GPU adapter: {e}")))?;
    let info = adapter.get_info();
    if info.device_type == wgpu::DeviceType::Cpu {
      return Err(Error::Gpu(format!("only a software adapter: {}", info.name)));
    }
    let limits = adapter.limits();
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { label: Some("ghl"), required_features: wgpu::Features::empty(), required_limits: limits.clone(), ..Default::default() }))
      .map_err(|e| Error::Gpu(format!("request_device failed: {e}")))?;
    Ok(GpuContext { device, queue, info, limits, trusted: true, scope: Arc::new(()) })
  }

  /// A context on a device the caller owns (shared with a UI).
  pub fn from_device(device: wgpu::Device, queue: wgpu::Queue, info: wgpu::AdapterInfo) -> GpuContext {
    let limits = device.limits();
    GpuContext { device, queue, info, limits, trusted: true, scope: Arc::new(()) }
  }

  pub fn describe(&self) -> String {
    format!("{} [{:?}] max_buffer={} MiB max_storage_binding={} MiB", self.info.name, self.info.backend, self.limits.max_buffer_size >> 20, self.limits.max_storage_buffer_binding_size >> 20)
  }

  /// The largest storage binding, in bytes (below 2 GiB on both backends).
  pub fn max_binding(&self) -> u64 {
    (self.limits.max_storage_buffer_binding_size as u64).min(self.limits.max_buffer_size).min((2u64 << 30) - 4)
  }

  /// Block until everything submitted so far has finished.
  pub fn wait(&self) {
    let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
  }

  pub fn storage(&self, label: &str, size: u64) -> wgpu::Buffer {
    self.device.create_buffer(&wgpu::BufferDescriptor {
      label: Some(label),
      size: size.max(16).next_multiple_of(4),
      usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    })
  }

  pub fn uniform(&self, label: &str, words: &[u32]) -> wgpu::Buffer {
    let b = self.device.create_buffer(&wgpu::BufferDescriptor {
      label: Some(label),
      size: (words.len() as u64 * 4).max(16).next_multiple_of(16),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    self.queue.write_buffer(&b, 0, bytemuck::cast_slice(words));
    b
  }

  /// A storage buffer with `data`, uploaded in chunks.
  pub fn storage_init(&self, label: &str, data: &[u32]) -> wgpu::Buffer {
    let buf = self.storage(label, data.len() as u64 * 4);
    self.write(&buf, 0, data);
    buf
  }

  /// Write words at a word offset, in chunks of 64 MiB.
  pub fn write(&self, buf: &wgpu::Buffer, offset_words: u64, data: &[u32]) {
    const CHUNK: usize = 16 << 20;
    let mut off = 0usize;
    while off < data.len() {
      let end = (off + CHUNK).min(data.len());
      self.queue.write_buffer(buf, (offset_words + off as u64) * 4, bytemuck::cast_slice(&data[off..end]));
      off = end;
      if data.len() > CHUNK {
        self.queue.submit(None);
      }
    }
  }

  /// Read `words` words at a word offset (blocks).
  pub fn read_u32(&self, buf: &wgpu::Buffer, offset_words: u64, words: u64) -> Vec<u32> {
    if words == 0 {
      return Vec::new();
    }
    let size = words * 4;
    let staging = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("staging"), size, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut enc = self.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(buf, offset_words * 4, &staging, 0, size);
    self.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
    self.wait();
    let data = bytemuck::pod_collect_to_vec(&slice.get_mapped_range().expect("mapped range"));
    staging.unmap();
    data
  }

  /// A compute pipeline; `trusted` drops naga's runtime checks (only for kernels whose indices are
  /// checked against the CPU).
  pub fn pipeline(&self, label: &str, wgsl: &str, entry: &str, trusted: bool) -> Result<wgpu::ComputePipeline> {
    // a shader compiler's or driver's failure is an error of this call, not a panic of the process
    let oom = self.device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let internal = self.device.push_error_scope(wgpu::ErrorFilter::Internal);
    let scope = self.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let desc = wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(wgsl.into()) };
    let module = if trusted && self.trusted {
      // SAFETY: the library's own kernels, whose every index is checked by the cell-exact tests
      unsafe { self.device.create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked()) }
    } else {
      self.device.create_shader_module(desc)
    };
    let p = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: None, module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None });
    let errors = [pollster::block_on(scope.pop()), pollster::block_on(internal.pop()), pollster::block_on(oom.pop())];
    if let Some(e) = errors.into_iter().flatten().next() {
      return Err(Error::Gpu(format!("kernel {label}: {e:?}")));
    }
    Ok(p)
  }

  pub fn bind(&self, pipeline: &wgpu::ComputePipeline, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
    self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &entries })
  }

  /// A bind group of the given bindings only (a pipeline's derived layout holds the bindings its
  /// entry point uses).
  pub fn bind_at(&self, pipeline: &wgpu::ComputePipeline, buffers: &[(u32, &wgpu::Buffer)]) -> wgpu::BindGroup {
    let entries: Vec<_> = buffers.iter().map(|(i, b)| wgpu::BindGroupEntry { binding: *i, resource: b.as_entire_binding() }).collect();
    self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &entries })
  }

  /// Run `f` (record and submit), wait, return seconds. Writes queued before are submitted before
  /// the clock starts: an upload is not GPU work.
  pub fn timed(&self, f: impl FnOnce()) -> f64 {
    self.queue.submit(None);
    self.wait();
    let t = Instant::now();
    f();
    self.wait();
    t.elapsed().as_secs_f64()
  }

  /// Whether two contexts are the same device.
  pub fn same(&self, o: &GpuContext) -> bool {
    Arc::ptr_eq(&self.scope, &o.scope)
  }
}
