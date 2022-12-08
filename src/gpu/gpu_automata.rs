use {
  std::mem,
  image::Pixel,
  wgpu::{
    BufferDescriptor, BufferUsages, CommandEncoder, ComputePipelineDescriptor, Device, PipelineLayoutDescriptor, Queue
  }
};

impl super::GPUDriver {
  pub(in super) fn initialize_ca_lut(&self, device: &Device, queue: &Queue) {
    let mut encoder =  device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
      label: Some("lut layout"),
      bind_group_layouts: &[&self.bind_group_layout],
      push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
      label: Some("lut pipeline"),
      layout: Some(&layout),
      module: &self.shader,
      entry_point: "compute_lut",
    });
    {
      let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
      cpass.set_pipeline(&pipeline);
      cpass.set_bind_group(0, &self.bind_group, &[]);
      cpass.insert_debug_marker("compute lut");
      cpass.dispatch_workgroups(2u32.pow(13), 2u32.pow(12), 1);
    }
    queue.submit(std::iter::once(encoder.finish()));
  }

  pub fn load_simulation(&mut self, device: &Device, queue: &Queue) {
    //TODO: RLE/Macrocell Pattern reader
    let path = "./doc/hutton32_squares.png";
    let mut pattern = image::open(path)
      .expect("Failed to load {path}")
      .to_rgba8();
    self.uniforms.simulation_dimm = [pattern.width(), pattern.height()];
    pattern.pixels_mut().for_each(|pixel| {
      let pixel_rgb = pixel.to_rgb().0;
      for (i, color) in hutton32_gpu::HUTTON32_COLORS.into_iter().enumerate() {
        if color == pixel_rgb {
          pixel.0[0] = i as u8;
        }
      }
    });
    self.simulation_buffer = device.create_buffer(&BufferDescriptor {
      label: Some("Simulation Buffer"),
      size: (pattern.width() * pattern.height() * mem::size_of::<u32>() as u32) as _,
      usage: BufferUsages::STORAGE
        | BufferUsages::COPY_DST
        | BufferUsages::COPY_SRC,
      mapped_at_creation: false
    });
    self.simulation_buffer_tex = device.create_buffer(&BufferDescriptor {
      label: Some("Simulation Buffer"),
      size: (pattern.width() * pattern.height() * mem::size_of::<u32>() as u32) as _,
      usage: BufferUsages::STORAGE
        | BufferUsages::COPY_DST
        | BufferUsages::COPY_SRC,
      mapped_at_creation: false
    });
    self.bind_group = Self::create_bind_group(
      &device, &self.bind_group_layout,
      &self.uniform_buffer, &self.simulation_buffer, &self.simulation_buffer_tex, &self.lut_buffer
    );
    queue.write_buffer(&self.simulation_buffer, 0, &pattern);
    queue.write_buffer(&self.simulation_buffer_tex, 0, &pattern);
  }

  pub(in super) fn simulation_advance(&mut self, encoder: &mut CommandEncoder) {
    for _ in 0..self.simulatiion_steps_per_call {
      let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
      cpass.set_pipeline(&self.compute_pipeline);
      cpass.set_bind_group(0, &self.bind_group, &[]);
      cpass.insert_debug_marker("compute simulation iter");
      cpass.dispatch_workgroups(self.uniforms.simulation_dimm[0], self.uniforms.simulation_dimm[1], 1);
      //queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));
    }
  }
}