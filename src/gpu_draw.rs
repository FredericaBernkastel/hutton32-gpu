use std::borrow::Cow;
use std::{fs, mem};
use std::sync::Arc;
use wgpu::{util::DeviceExt, TextureViewDescriptor};
use egui::plot::PlotBounds;

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
  render_pipeline: wgpu::RenderPipeline,
  compute_pipeline: wgpu::ComputePipeline,

  target_format: wgpu::TextureFormat,
  bind_group: wgpu::BindGroup,

  uniform_buffer: wgpu::Buffer,
  simulation_buffer: wgpu::Buffer,

  texture: (wgpu::Texture, wgpu::TextureView),
  pub texture_size: [u32; 2],

  pub uniforms: Uniform,

  pub simulatiion_steps_per_call: u32,
}

impl GPUDrawer {
  pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat, sim_dimm: [u32; 2]) -> Self {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
      label: Some("egui_display_shader"),
      source: wgpu::ShaderSource::Wgsl(Cow::Owned(fs::read_to_string("./kernel/display_shader.wgsl").unwrap())),
    });

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
      simulation_dimm: sim_dimm,
      ..Default::default()
    };

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("egui_plot_uniforms"),
      contents: bytemuck::cast_slice(&[uniforms]),
      usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
    });

    let simulation_buffer = device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("Simulation Buffer"),
      size: (sim_dimm[0] * sim_dimm[1] * mem::size_of::<u32>() as u32) as _,
      usage: wgpu::BufferUsages::STORAGE
        | wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::COPY_SRC,
      mapped_at_creation: false
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("egui_plot_bind_group"),
      layout: &bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: uniform_buffer.as_entire_binding(),
      },
      wgpu::BindGroupEntry {
        binding: 1,
        resource: simulation_buffer.as_entire_binding(),
      }],
    });

    let compute_pipeline = Self::create_compute_pipeline(device, &[&bind_group_layout], &shader, None);

    Self {
      render_pipeline,
      compute_pipeline,

      target_format,
      bind_group,

      uniform_buffer,
      simulation_buffer,

      texture,
      texture_size: [0, 0],
      uniforms,

      simulatiion_steps_per_call: 1
    }
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

  pub fn render(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
    let mut encoder =
      device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
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

    queue.submit(std::iter::once(encoder.finish()));
  }

  pub fn render_onto_renderpass<'rp>(&'rp self, rpass: &mut wgpu::RenderPass<'rp>) {
    rpass.set_pipeline(&self.render_pipeline);
    rpass.set_bind_group(0, &self.bind_group, &[]);
    rpass.draw(0..6, 0..1);
  }

  pub fn load_simulation(&self, queue: &wgpu::Queue) {
    let mut simulation_init = image::RgbaImage::new(self.uniforms.simulation_dimm[0], self.uniforms.simulation_dimm[1]);
    //let glider = [[1, 0], [2, 1], [0, 2], [1, 2], [2, 2]];
    let acorn = [[1, 0], [3, 1], [0, 2], [1, 2], [4, 2], [5, 2], [6, 2]];
    acorn.iter().for_each(|[x, y]| {
      simulation_init.get_pixel_mut(x + 128, y + 128).0[0] = 0x01;
    });
    queue.write_buffer(&self.simulation_buffer, 0, &simulation_init);
  }

  pub fn simulation_advance(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
    for _ in 0..self.simulatiion_steps_per_call {
      let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

      {
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.insert_debug_marker("compute simulation iter");
        cpass.dispatch_workgroups(self.uniforms.simulation_dimm[0], self.uniforms.simulation_dimm[1], 1);
      }

      queue.write_buffer(&self.uniform_buffer, 0, bytemuck::cast_slice(&[self.uniforms]));
      queue.submit(std::iter::once(encoder.finish()));
      self.uniforms.time = self.uniforms.time.wrapping_add(1);
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

      if compute_requested {
        gpu_drawer.simulation_advance(device, queue);
      }

      gpu_drawer.prepare(
        device,
        queue,
        [rect.width() as u32, rect.height() as u32],
        &bounds,
      );

      gpu_drawer.render(device, queue);

      vec![]
    });

  egui::PaintCallback {
    rect,
    callback: Arc::new(cb),
  }
}