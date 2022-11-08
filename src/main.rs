use eframe::egui::plot::{Legend, PlotImage};
use eframe::egui::{self, plot::PlotBounds};
use eframe::emath::Vec2;
use wgpu;

mod gpu_draw;
use gpu_draw::GPUDrawer;

pub struct GpuPlot {
  dirty: bool,
  texture_id: egui::TextureId
}

impl GpuPlot {
  pub fn new<'a>(cc: &'a eframe::CreationContext<'a>) -> Option<Self> {
    let wgpu_render_state = cc.wgpu_render_state.as_ref()?;

    let device = &wgpu_render_state.device;
    let target_format = wgpu_render_state.target_format;

    let gpu_drawer = GPUDrawer::new(device, &wgpu_render_state.queue, target_format);
    let texture_id = {
      let mut renderer = wgpu_render_state.renderer.write();
      renderer.register_native_texture(device, &gpu_drawer.create_view(), wgpu::FilterMode::Linear)
    };

    wgpu_render_state
      .renderer
      .write()
      .paint_callback_resources
      .insert(gpu_drawer);

    Some(Self {
      dirty: true,
      texture_id
    })
  }
}

impl eframe::App for GpuPlot {
  fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
    egui::CentralPanel::default().show(ctx, |ui| {
      if ui.button("GPU execute").clicked() {
        self.dirty = true;
      }

      let mut bounds = PlotBounds::NOTHING;
      let resp = egui::plot::Plot::new("my_plot")
        .legend(Legend::default())
        .data_aspect(1.0)
        .view_aspect(1.0)
        // Must set margins to zero or the image and plot bounds will
        // constantly fight, expanding the plot to infinity.
        .set_margin_fraction(Vec2::new(0.0, 0.0))
        .include_x(-1.0)
        .include_x(1.0)
        .include_y(-1.0)
        .include_y(1.0)
        .show(ui, |ui| {
          bounds = ui.plot_bounds();

          // Render the plot texture filling the viewport.
          ui.image(
            PlotImage::new(
              self.texture_id,
              bounds.center(),
              [bounds.width() as f32, bounds.height() as f32],
            )
              .name("Mandelbrot (GPU)"),
          );
        });


      // Add a callback to egui to render the plot contents to
      // texture.
      ui.painter().add(gpu_draw::egui_wgpu_callback(
        bounds,
        resp.response.rect
      ));

      // Update the texture handle in egui from the previously
      // rendered texture (from the last frame).
      let wgpu_render_state = frame.wgpu_render_state().unwrap();
      let mut renderer = wgpu_render_state.renderer.write();

      let plot: &GPUDrawer = renderer.paint_callback_resources.get().unwrap();
      let texture_view = plot.create_view();

      renderer.update_egui_texture_from_wgpu_texture(
        &wgpu_render_state.device,
        &texture_view,
        wgpu::FilterMode::Linear,
        self.texture_id,
      );

      self.dirty = false;
    });
  }
}

fn main() {
  let native_options = eframe::NativeOptions {
    renderer: eframe::Renderer::Wgpu,
    initial_window_size: Some(Vec2::new(550.0, 550.0)),
    ..Default::default()
  };

  eframe::run_native(
    "GPU Accelerated Plotter",
    native_options,
    Box::new(|cc| Box::new(GpuPlot::new(cc).unwrap())),
  );
}
