use {
  std::{mem, sync::Arc},
  wgpu::{
    util::{DeviceExt, BufferInitDescriptor}, Device, Queue,
    TextureViewDescriptor, ShaderModule, ComputePipeline,
    RenderPipeline, CommandEncoder, BindGroup, BindGroupLayout, BindGroupLayoutDescriptor,
    PipelineLayoutDescriptor, RenderPipelineDescriptor, ComputePipelineDescriptor, Buffer,
    BufferDescriptor, Texture, TextureView, TextureFormat, BufferUsages, ShaderStages,
    BindingType, BufferBindingType
  },
  egui::plot::PlotBounds,
};

mod gpu_automata;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniform {
  pub display_x_range: [f32; 2],
  pub display_y_range: [f32; 2],
  pub simulation_dimm: [u32; 2]
}

impl Default for Uniform {
  fn default() -> Self {
    Self {
      display_x_range: [0.0, 1.0],
      display_y_range: [0.0, 1.0],
      simulation_dimm: [1, 1],
    }}}

pub struct GPUDriver {
  shader: ShaderModule,
  render_pipeline: RenderPipeline,
  compute_pipeline: ComputePipeline,
  texture: (Texture, TextureView),
  target_format: TextureFormat,
  bind_group: BindGroup,
  bind_group_layout: BindGroupLayout,

  uniform_buffer: Buffer,
  simulation_buffer: Buffer,
  lut_buffer: Buffer,

  pub texture_size: [u32; 2],
  pub uniforms: Uniform,
  pub simulatiion_steps_per_call: u64,
}

impl GPUDriver {
  pub fn new(device: &Device, queue: &Queue, target_format: TextureFormat) -> Self {
    let shader = device.create_shader_module(
      wgsl_preprocessor::ShaderBuilder::new("./kernel/main.wgsl")
        .expect("Failed to load ./kernel/main.wgsl")
        .put_array_definition("hutton32_colors",
          &gpu_automata::HUTTON32_COLORS.map(|x| x.map(|c| c as i32))
            .iter().collect()
        )
        .build()
    );

    // Allocate some stand-in textures since we don't know the final width
    // and height yet.
    const DEFAULT_WIDTH: u32 = 1;
    const DEFAULT_HEIGHT: u32 = 1;
    let texture = Self::create_texture(device, target_format, 1, DEFAULT_WIDTH, DEFAULT_HEIGHT);

    // We need a DSL for that...
    let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
      label: Some("egui_plot_bind_group_layout"),
      entries: &[wgpu::BindGroupLayoutEntry { // uniform_buffer
        binding: 0,
        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      wgpu::BindGroupLayoutEntry { // simulation_buffer
        binding: 1,
        visibility: ShaderStages::FRAGMENT | ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      wgpu::BindGroupLayoutEntry { // LUT
        binding: 2,
        visibility: ShaderStages::COMPUTE,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      }],
    });

    let render_pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
      label: Some("egui_plot_pipeline_layout"),
      bind_group_layouts: &[&bind_group_layout],
      ..Default::default()
    });

    let render_pipeline = device.create_render_pipeline(&RenderPipelineDescriptor {
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

    let uniform_buffer = device.create_buffer_init(&BufferInitDescriptor {
      label: Some("egui_plot_uniforms"),
      contents: bytemuck::cast_slice(&[uniforms]),
      usage: BufferUsages::COPY_DST | BufferUsages::UNIFORM,
    });

    let simulation_buffer = device.create_buffer(&BufferDescriptor {
      label: Some("Simulation Buffer"),
      size: (/*sim_dimm[0] * sim_dimm[1]*/ 1 * mem::size_of::<u32>() as u32) as _,
      usage: BufferUsages::STORAGE
        | BufferUsages::COPY_DST
        | BufferUsages::COPY_SRC,
      mapped_at_creation: false
    });

    let lut_buffer = device.create_buffer(&BufferDescriptor {
      label: Some("Simulation Buffer"),
      // c(5)-n(5)-e(5)-s(5)-w(5) = r(8)
      size: 2u32.pow(25) as _, // 32MB
      usage: BufferUsages::STORAGE,
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

  fn create_texture(
    device: &Device,
    target_format: TextureFormat,
    sample_count: u32,
    width: u32,
    height: u32,
  ) -> (Texture, TextureView) {
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

  pub fn create_view(&self) -> TextureView {
    self.texture.0.create_view(&TextureViewDescriptor::default())
  }

  fn create_compute_pipeline(
    device: &Device,
    bind_group_layouts: &[&BindGroupLayout],
    shader: &ShaderModule,
    label: Option<&str>,
  ) -> ComputePipeline {
    let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
      label,
      bind_group_layouts,
      push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
      label,
      layout: Some(&layout),
      module: shader,
      entry_point: "compute_main",
    });
    pipeline
  }

  fn create_bind_group(
    device: &Device,
    layout: &BindGroupLayout,
    uniform_buffer: &Buffer,
    simulation_buffer: &Buffer,
    lut_buffer: &Buffer,
  ) -> BindGroup {
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

  fn prepare(
    &mut self,
    device: &Device,
    queue: &Queue,
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

  fn render(&self, encoder: &mut CommandEncoder) {
    let view = self.create_view();

    // Render directly to the texture if no MSAA
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

  fn render_onto_renderpass<'rp>(&'rp self, rpass: &mut wgpu::RenderPass<'rp>) {
    rpass.set_pipeline(&self.render_pipeline);
    rpass.set_bind_group(0, &self.bind_group, &[]);
    rpass.draw(0..6, 0..1);
  }
}

pub fn egui_wgpu_callback(
  bounds: PlotBounds,
  rect: egui::Rect,
  compute_requested: bool
) -> egui::PaintCallback {
  let cb = egui_wgpu::CallbackFn::new()
    .prepare(move |device, queue, _encoder, paint_callback_resources| {
      let gpu_driver: &mut GPUDriver = paint_callback_resources.get_mut().unwrap();

      let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

      if compute_requested {
        gpu_driver.simulation_advance(&mut encoder);
      }

      gpu_driver.prepare(
        device,
        queue,
        [rect.width() as u32, rect.height() as u32],
        &bounds,
      );

      gpu_driver.render(&mut encoder);

      vec![encoder.finish()]
    });

  egui::PaintCallback {
    rect,
    callback: Arc::new(cb),
  }
}
