use std::{mem};
use std::sync::Arc;
use wgpu::{util::DeviceExt, TextureViewDescriptor};
use egui::plot::PlotBounds;
use image::{Pixel};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniform {
  pub display_x_range: [f32; 2],
  pub display_y_range: [f32; 2],

  pub simulation_dimm: [u32; 2],
  pub time: u64,
}

impl Default for Uniform {
  fn default() -> Self {
    Self {
      display_x_range: [0.0, 1.0],
      display_y_range: [0.0, 1.0],
      simulation_dimm: [1, 1],
      time: 0
    }}}

pub struct GPUDrawer {
  shader: wgpu::ShaderModule,
  render_pipeline: wgpu::RenderPipeline,
  compute_pipeline: wgpu::ComputePipeline,

  target_format: wgpu::TextureFormat,
  bind_group: wgpu::BindGroup,
  bind_group_layout: wgpu::BindGroupLayout,

  uniform_buffer: wgpu::Buffer,
  simulation_buffer: wgpu::Buffer,
  lut_buffer: wgpu::Buffer,

  texture: (wgpu::Texture, wgpu::TextureView),
  pub texture_size: [u32; 2],

  pub uniforms: Uniform,

  pub simulatiion_steps_per_call: u32,
}

impl GPUDrawer {
  pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, target_format: wgpu::TextureFormat) -> Self {
    let shader = device.create_shader_module(
      wgsl_preprocessor::ShaderBuilder::new("./kernel/main.wgsl")
        .expect("Failed to load ./kernel/main.wgsl")
        .put_array_definition("hutton32_colors",
          &hutton32_colors().map(|x| x.map(|c| c as i32))
            .iter().collect()
        )
        .build()
    );

    // Allocate some stand-in textures since we don't know the final width
    // and height yet.
    const DEFAULT_WIDTH: u32 = 1;
    const DEFAULT_HEIGHT: u32 = 1;
    let texture = Self::create_texture(device, target_format, 1, DEFAULT_WIDTH, DEFAULT_HEIGHT);

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
      label: Some("egui_plot_bind_group_layout"),
      entries: &[wgpu::BindGroupLayoutEntry { // uniform_buffer
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
          ty: wgpu::BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      wgpu::BindGroupLayoutEntry { // simulation_buffer
        binding: 1,
        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
          ty: wgpu::BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      wgpu::BindGroupLayoutEntry { // LUT
        binding: 2,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
          ty: wgpu::BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      }],
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("egui_plot_pipeline_layout"),
      bind_group_layouts: &[&bind_group_layout],
      ..Default::default()
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Render Pipeline"),
      layout: Some(&render_pipeline_layout),
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: "vs_main",
        buffers: &[],
      },
      depth_stencil: None,
      fragment: Some(wgpu::FragmentState {
        module: &shader,
        entry_point: "fs_main",
        targets: &[Some(wgpu::ColorTargetState {
          format: target_format,
          blend: Some(wgpu::BlendState::ALPHA_BLENDING),
          write_mask: wgpu::ColorWrites::ALL,
        })],
      }),
      primitive: Default::default(),
      multisample: Default::default(),
      multiview: None
    });

    let uniforms = Uniform {
      simulation_dimm: [1, 1],
      ..Default::default()
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("egui_plot_uniforms"),
      contents: bytemuck::cast_slice(&[uniforms]),
      usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
    });

    let simulation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("Simulation Buffer"),
      size: (/*sim_dimm[0] * sim_dimm[1]*/ 1 * mem::size_of::<u32>() as u32) as _,
      usage: wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::COPY_SRC,
      mapped_at_creation: false
    });

    let lut_buffer = device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("Simulation Buffer"),
      // c(5)-n(5)-e(5)-s(5)-w(5) = r(8)
      size: 2u32.pow(25) as _, // 32MB
      usage: wgpu::BufferUsages::STORAGE,
      mapped_at_creation: false
    });

    let bind_group = Self::create_bind_group(
      &device, &bind_group_layout,
      &uniform_buffer, &simulation_buffer, &lut_buffer
    );

    let compute_pipeline = Self::create_compute_pipeline(device, &[&bind_group_layout], &shader, None);

    let this = Self {
      shader,

      render_pipeline,
      compute_pipeline,

      target_format,
      bind_group,
      bind_group_layout,

      uniform_buffer,
      simulation_buffer,
      lut_buffer,

      texture,
      texture_size: [0, 0],
      uniforms,

      simulatiion_steps_per_call: 1
    };

    this.initialize_ca_lut(&device, &queue);
    this
  }

  fn initialize_ca_lut(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
    let mut encoder =  device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("lut layout"),
      bind_group_layouts: &[&self.bind_group_layout],
      push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
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

  fn create_texture(
    device: &wgpu::Device,
    target_format: wgpu::TextureFormat,
    sample_count: u32,
    width: u32,
    height: u32,
  ) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
      label: Some("egui_plot_texture"),
      size: wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
      },
      mip_level_count: 1,
      sample_count,
      dimension: wgpu::TextureDimension::D2,
      format: target_format,
      usage: wgpu::TextureUsages::TEXTURE_BINDING
        | wgpu::TextureUsages::RENDER_ATTACHMENT,
    });

    let view = texture.create_view(&TextureViewDescriptor::default());

    (texture, view)
  }

  pub fn create_view(&self) -> wgpu::TextureView {
    self.texture
      .0
      .create_view(&TextureViewDescriptor::default())
  }

  pub fn create_compute_pipeline(
    device: &wgpu::Device,
    bind_group_layouts: &[&wgpu::BindGroupLayout],
    shader: &wgpu::ShaderModule,
    label: Option<&str>,
  ) -> wgpu::ComputePipeline {
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label,
      bind_group_layouts,
      push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
      label,
      layout: Some(&layout),
      module: shader,
      entry_point: "compute_main",
    });
    pipeline
  }

  fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    uniform_buffer: &wgpu::Buffer,
    simulation_buffer: &wgpu::Buffer,
    lut_buffer: &wgpu::Buffer,
  ) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("egui_plot_bind_group"),
      layout,
      entries: &[wgpu::BindGroupEntry {
          binding: 0,
          resource: uniform_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 1,
          resource: simulation_buffer.as_entire_binding(),
        },
        wgpu::BindGroupEntry {
          binding: 2,
          resource: lut_buffer.as_entire_binding(),
        }
      ],
    })
  }

  pub fn prepare(
    &mut self,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    dimensions: [u32; 2],
    bounds: &PlotBounds
  ) {
    // Re-allocate the render targets if the requested dimensions have changed.
    if dimensions != self.texture_size {
      self.texture_size = dimensions;

      self.texture =
        Self::create_texture(device, self.target_format, 1, dimensions[0], dimensions[1]);
    }

    self.uniforms.display_x_range = [bounds.min()[0] as f32, bounds.max()[0] as f32];
    self.uniforms.display_y_range = [bounds.min()[1] as f32, bounds.max()[1] as f32];

    queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));
  }

  pub fn render(&self, encoder: &mut wgpu::CommandEncoder) {
    let view = self.create_view();

    // Render directly to the texture if no MSAA, or use the
    // multisampled buffer and resolve to the texture if using MSAA.
    let rpass_color_attachment = wgpu::RenderPassColorAttachment {
      view: &view,
      resolve_target: None,
      ops: wgpu::Operations {
        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
        store: true,
      },
    };

    let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
      label: None,
      color_attachments: &[Some(rpass_color_attachment)],
      depth_stencil_attachment: None,
    });

    self.render_onto_renderpass(&mut rpass);
  }

  pub fn render_onto_renderpass<'rp>(&'rp self, rpass: &mut wgpu::RenderPass<'rp>) {
    rpass.set_pipeline(&self.render_pipeline);
    rpass.set_bind_group(0, &self.bind_group, &[]);
    rpass.draw(0..6, 0..1);
  }

  pub fn load_simulation(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
    //TODO: Pattern reader
    let mut pattern = image::open("./doc/test_pattern.png")
      .expect("./doc/test_pattern.png")
      .to_rgba8();
    self.uniforms.simulation_dimm = [pattern.width(), pattern.height()];
    let color_table = hutton32_colors();
    pattern.pixels_mut().for_each(|pixel| {
      let pixel_rgb = pixel.to_rgb().0;
      for (i, color) in color_table.into_iter().enumerate() {
        if color == pixel_rgb {
          pixel.0[0] = i as u8;
        }
      }
    });
    self.simulation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("Simulation Buffer"),
      size: (pattern.width() * pattern.height() * mem::size_of::<u32>() as u32) as _,
      usage: wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::COPY_SRC,
      mapped_at_creation: false
    });
    self.bind_group = Self::create_bind_group(
      &device, &self.bind_group_layout,
      &self.uniform_buffer, &self.simulation_buffer, &self.lut_buffer
    );
    queue.write_buffer(&self.simulation_buffer, 0, &pattern);
  }

  pub fn simulation_advance(&mut self, encoder: &mut wgpu::CommandEncoder) {
    for _ in 0..self.simulatiion_steps_per_call {
      {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.insert_debug_marker("compute simulation iter");
        cpass.dispatch_workgroups(self.uniforms.simulation_dimm[0], self.uniforms.simulation_dimm[1], 1);
      }

      self.uniforms.time = self.uniforms.time.wrapping_add(1);
      //queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));
    }
  }
}

pub fn egui_wgpu_callback(
  bounds: PlotBounds,
  rect: egui::Rect,
  compute_requested: bool
) -> egui::PaintCallback {
  let cb =
    egui_wgpu::CallbackFn::new().prepare(move |device, queue, _encoder, paint_callback_resources| {
      let gpu_drawer: &mut GPUDrawer = paint_callback_resources.get_mut().unwrap();

      let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

      if compute_requested {
        gpu_drawer.simulation_advance(&mut encoder);
      }

      gpu_drawer.prepare(
        device,
        queue,
        [rect.width() as u32, rect.height() as u32],
        &bounds,
      );

      gpu_drawer.render(&mut encoder);

      vec![encoder.finish()]
    });

  egui::PaintCallback {
    rect,
    callback: Arc::new(cb),
  }
}

fn hutton32_colors() -> [[u8; 3]; 32] {
  [
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
  ]
}