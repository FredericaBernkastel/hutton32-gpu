use std::borrow::Cow;
use std::fs;
use std::sync::Arc;
use wgpu::{util::DeviceExt, TextureViewDescriptor};
use egui::plot::PlotBounds;

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Uniform {
  pub x_bounds: [f32; 2],
  pub y_bounds: [f32; 2],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
  position: [f32; 2],
}

const VERTICES: &[Vertex] = &[
  Vertex { position: [ -1.0,  1.0] },
  Vertex { position: [ -1.0, -1.0] },
  Vertex { position: [  1.0,  1.0] },

  Vertex { position: [  1.0,  1.0] },
  Vertex { position: [ -1.0, -1.0] },
  Vertex { position: [  1.0, -1.0] },
];

pub struct GPUDrawer {
  render_pipeline: wgpu::RenderPipeline,
  compute_pipeline: wgpu::ComputePipeline,

  target_format: wgpu::TextureFormat,
  bind_group: wgpu::BindGroup,

  uniform_buffer: wgpu::Buffer,
  vertex_buffer: wgpu::Buffer,

  texture: (wgpu::Texture, wgpu::TextureView),
  width: u32,
  height: u32,
}

impl Vertex {
  const ATTRIBS: [wgpu::VertexAttribute; 1] =
    wgpu::vertex_attr_array![0 => Float32x2];

  fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
    wgpu::VertexBufferLayout {
      array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
      step_mode: wgpu::VertexStepMode::Vertex,
      attributes: &Self::ATTRIBS,
    }
  }
}

impl GPUDrawer {
  pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, target_format: wgpu::TextureFormat) -> Self {
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
      entries: &[wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
          ty: wgpu::BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      }],
    });

    let render_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
      label: Some("egui_plot_pipeline_layout"),
      bind_group_layouts: &[&bind_group_layout],
      push_constant_ranges: &[],
    });

    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
      label: Some("Render Pipeline"),
      layout: Some(&render_pipeline_layout),
      vertex: wgpu::VertexState {
        module: &shader,
        entry_point: "vs_main", // 1.
        buffers: &[Vertex::desc()], // 2.
      },
      fragment: Some(wgpu::FragmentState { // 3.
        module: &shader,
        entry_point: "fs_main",
        targets: &[Some(wgpu::ColorTargetState { // 4.
          format: target_format,
          blend: Some(wgpu::BlendState::ALPHA_BLENDING),
          write_mask: wgpu::ColorWrites::ALL,
        })],
      }),
      primitive: Default::default(),
      depth_stencil: None,
      multisample: wgpu::MultisampleState {
        count: 1,
        mask: !0,
        alpha_to_coverage_enabled: false,
      },
      multiview: None,
    });

    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
      label: Some("egui_plot_uniforms"),
      contents: bytemuck::cast_slice(&[Uniform {
        x_bounds: [-1.0, 1.0],
        y_bounds: [-1.0, 1.0],
      }]),
      usage: wgpu::BufferUsages::COPY_DST
        | wgpu::BufferUsages::UNIFORM,
    });

    let vertex_buffer = device.create_buffer_init(
      &wgpu::util::BufferInitDescriptor {
        label: Some("Vertex Buffer"),
        contents: bytemuck::cast_slice(VERTICES),
        usage: wgpu::BufferUsages::VERTEX,
      }
    );

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: Some("egui_plot_bind_group"),
      layout: &bind_group_layout,
      entries: &[wgpu::BindGroupEntry {
        binding: 0,
        resource: uniform_buffer.as_entire_binding(),
      }],
    });

    let compute_pipeline = Self::create_compute_pipeline(device, &[&bind_group_layout], &shader, None);

    Self {
      render_pipeline,
      compute_pipeline,

      target_format,
      bind_group,

      uniform_buffer,
      vertex_buffer,

      texture,
      width: 0,
      height: 0
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
    if dimensions[0] != self.width || dimensions[1] != self.height {
      self.width = dimensions[0];
      self.height = dimensions[1];

      self.texture =
        Self::create_texture(device, self.target_format, 1, self.width, self.height);
    }

    queue.write_buffer(
      &self.uniform_buffer,
      0,
      bytemuck::cast_slice(&[Uniform {
        x_bounds: [bounds.min()[0] as f32, bounds.max()[0] as f32],
        y_bounds: [bounds.min()[1] as f32, bounds.max()[1] as f32],
      }]),
    );
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

      {
        let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
          label: None,
          color_attachments: &[Some(rpass_color_attachment)],
          depth_stencil_attachment: None,
        });

        self.render_onto_renderpass(&mut rpass);
      }

      /*{
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None });
        cpass.set_pipeline(&self.compute_pipeline);
        cpass.set_bind_group(0, &self.bind_group, &[]);
        cpass.insert_debug_marker("compute texture");
        // Number of cells to run, the (x,y,z) size of item being processed
        cpass.dispatch_workgroups( self.width,self.height,1);
      }*/
    }

    queue.submit(std::iter::once(encoder.finish()));
  }

  pub fn render_onto_renderpass<'rp>(&'rp self, rpass: &mut wgpu::RenderPass<'rp>) {
    rpass.set_pipeline(&self.render_pipeline);
    rpass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
    rpass.set_bind_group(0, &self.bind_group, &[]);
    rpass.draw(0..VERTICES.len() as u32, 0..1);
  }
}

pub fn egui_wgpu_callback(
  bounds: PlotBounds,
  rect: egui::Rect
) -> egui::PaintCallback {
  let cb =
    egui_wgpu::CallbackFn::new().prepare(move |device, queue, _encoder, paint_callback_resources| {
      let gpu_drawer: &mut GPUDrawer = paint_callback_resources.get_mut().unwrap();

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