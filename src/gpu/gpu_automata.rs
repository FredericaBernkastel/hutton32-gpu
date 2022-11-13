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
      for (i, color) in HUTTON32_COLORS.into_iter().enumerate() {
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
    self.bind_group = Self::create_bind_group(
      &device, &self.bind_group_layout,
      &self.uniform_buffer, &self.simulation_buffer, &self.lut_buffer
    );
    queue.write_buffer(&self.simulation_buffer, 0, &pattern);
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

pub(in super) const HUTTON32_COLORS: [[u8; 3]; 32] = [
  [ 0 ,   0,   0],    // 0  dark gray
  [255,   0,   0],    // 1  red
  [255, 125,   0],    // 2  orange (to match red and yellow)
  [255, 150,  25],    // 3   lighter
  [255, 175,  50],    // 4    lighter
  [255, 200,  75],    // 5     lighter
  [255, 225, 100],    // 6      lighter
  [255, 250, 125],    // 7       lighter
  [251, 255,   0],    // 8  yellow
  [ 89,  89, 255],    // 9  blue
  [106, 106, 255],    // 10  lighter
  [122, 122, 255],    // 11   lighter
  [139, 139, 255],    // 12    lighter
  [ 27, 176,  27],    // 13 green
  [ 36, 200,  36],    // 14  lighter
  [ 73, 255,  73],    // 15   lighter
  [106, 255, 106],    // 16    lighter
  [235,  36,  36],    // 17 red
  [255,  56,  56],    // 18  lighter
  [255,  73,  73],    // 19   lighter
  [255,  89,  89],    // 20    lighter
  [185,  56, 255],    // 21 purple
  [191,  73, 255],    // 22  lighter
  [197,  89, 255],    // 23   lighter
  [203, 106, 255],    // 24    lighter
  [  0, 255, 128],    // 25 light green
  [255, 128,  64],    // 26 light orange
  [255, 255, 128],    // 27 light yellow
  [ 33, 215, 215],    // 28 cyan
  [ 27, 176, 176],    // 29  darker
  [ 24, 156, 156],    // 30   darker
  [ 21, 137, 137]     // 31    darker
];