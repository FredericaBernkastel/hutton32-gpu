use std::borrow::Cow;
use wgpu::util::DeviceExt;
use pollster::FutureExt as _;

pub struct WGPUKernel {
  device: wgpu::Device,
  queue: wgpu::Queue,
  global_group: wgpu::BindGroup,
  pipeline: wgpu::ComputePipeline,

  staging_tex_buffer: wgpu::Buffer,
  storage_tex_buffer: wgpu::Buffer,
  uniforms_buffer: wgpu::Buffer,

  pub external_buffers: ExternalBuffers
}

pub struct ExternalBuffers {
  pub texture: image::RgbaImage
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
  time: u32,
}

impl WGPUKernel {
  pub async fn init(buffers: ExternalBuffers) -> Option<Self> {
    // Instantiates instance of WebGPU
    let instance = wgpu::Instance::new(wgpu::Backends::all());

    // `request_adapter` instantiates the general connection to the GPU
    let adapter = instance
      .request_adapter(&wgpu::RequestAdapterOptions::default())
      .await?;

    // `request_device` instantiates the feature specific connection to the GPU, defining some parameters,
    //  `features` being the available features.
    let (device, queue) = adapter
      .request_device(
        &wgpu::DeviceDescriptor {
          label: None,
          features: wgpu::Features::empty(),
          limits: wgpu::Limits::downlevel_defaults(),
        },
        None,
      )
      .await.expect("Unable to request device");

    // Loads the shader from WGSL
    let cs_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: None,
      source: wgpu::ShaderSource::Wgsl(Cow::Owned(std::fs::read_to_string("kernel/main.wgsl").unwrap())),
    });

    let texture = &buffers.texture;
    // Gets the size in bytes of the texture buffer.
    let slice_size = texture.len() * std::mem::size_of::<u8>();
    let size = slice_size as wgpu::BufferAddress;

    // Instantiates buffer without data.
    // `usage` of buffer specifies how it can be used:
    //   `BufferUsages::MAP_READ` allows it to be read (outside the shader).
    //   `BufferUsages::COPY_DST` allows it to be the destination of the copy.
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("texture staging buf"),
      size,
      usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });

    // Instantiates buffer with data (`numbers`).
    // Usage allowing the buffer to be:
    //   A storage buffer (can be bound within a bind group and thus available to a shader).
    //   The destination of a copy.
    //   The source of a copy.
    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("Storage Buffer"),
      contents: texture.as_raw(),
      usage: wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::COPY_SRC,
    });

    let uniforms_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("scalar_buffer"),
      contents: bytemuck::cast_slice(&[Uniforms { time: 0 }]),
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // A bind group defines how buffers are accessed by shaders.
    // It is to WebGPU what a descriptor set is to Vulkan.
    // `binding` here refers to the `binding` of a buffer in the shader (`layout(set = 0, binding = 0) buffer`).

    // A pipeline specifies the operation of a shader

    // Instantiates the pipeline.
    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label: None,
      layout: None,
      module: &cs_module,
      entry_point: "main",
    });

    // Instantiates the bind group, once again specifying the binding of buffers.
    let bind_group_layout = compute_pipeline.get_bind_group_layout(0);
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &bind_group_layout,
      entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: storage_buffer.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: uniforms_buffer.as_entire_binding()  }
      ],
    });

    Some(Self {
      device,
      queue,
      global_group: bind_group,
      pipeline: compute_pipeline,
      storage_tex_buffer: storage_buffer,
      staging_tex_buffer: staging_buffer,
      uniforms_buffer,
      external_buffers: buffers
    })
  }

  pub fn iter(&mut self, time: u32) {
    // A command encoder executes one or many pipelines.
    // It is to WebGPU what a command buffer is to Vulkan.
    let mut encoder =
      self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
      let texture = &self.external_buffers.texture;
      let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
      cpass.set_pipeline(&self.pipeline);
      cpass.set_bind_group(0, &self.global_group, &[]);
      cpass.insert_debug_marker("compute texture");
      cpass.dispatch_workgroups( // Number of cells to run, the (x,y,z) size of item being processed
        texture.width(),
        texture.height(),
        1
      );
    };

    // Sets adds copy operation to command encoder.
    // Will copy data from storage buffer on GPU to staging buffer on CPU.
    encoder.copy_buffer_to_buffer(
      &self.storage_tex_buffer, 0, &self.staging_tex_buffer, 0, self.storage_tex_buffer.size()
    );

    self.queue.write_buffer(&self.uniforms_buffer, 0, bytemuck::cast_slice(&[Uniforms { time }]));

    // Submits command encoder for processing
    self.queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let staging_tex_buffer_slice = self.staging_tex_buffer.slice(..);

    // Sets the buffer up for mapping, sending over the result of the mapping back to us when it is finished.
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    staging_tex_buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    self.device.poll(wgpu::Maintain::Wait);

    // Awaits until `buffer_future` can be read from
    receiver.receive().block_on().unwrap().unwrap();

    // Gets contents of buffer
    let data = staging_tex_buffer_slice.get_mapped_range();
    // Since contents are got in bytes, this converts these bytes back to u32
    //let result = bytemuck::cast_slice(&data).to_vec();
    self.external_buffers.texture
      .copy_from_slice(data.as_ref());

    // With the current interface, we have to make sure all mapped views are
    // dropped before we unmap the buffer.
    drop(data);

    self.staging_tex_buffer.unmap();
  }
}
