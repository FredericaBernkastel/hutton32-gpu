use eframe::egui::plot::{Legend, PlotImage};
use eframe::egui::{self, plot::PlotBounds};
use eframe::emath::Vec2;
use wgpu;

mod gpu_draw;
use gpu_draw::GPUDrawer;

pub struct GpuPlot {
  compute_requested: bool,
  texture_id: egui::TextureId
}

const SIMULATION_DIMM: [u32; 2] = [256, 256];

impl GpuPlot {
  pub fn new<'a>(cc: &'a eframe::CreationContext<'a>) -> Option<Self> {
    let wgpu_render_state = cc.wgpu_render_state.as_ref()?;

    let device = &wgpu_render_state.device;
    let target_format = wgpu_render_state.target_format;

    let gpu_drawer = GPUDrawer::new(device, target_format, SIMULATION_DIMM);
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
      compute_requested: false,
      texture_id
    })
  }
}

impl eframe::App for GpuPlot {
  fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
    egui::CentralPanel::default().show(ctx, |ui| {
      ui.add_enabled_ui(!self.compute_requested , |ui| {
        if ui.button("▶ Start simulation").clicked() {
          self.compute_requested = true;
        }
      });

      let mut bounds = PlotBounds::NOTHING;
      let resp = egui::plot::Plot::new("my_plot")
        .legend(Legend::default())
        .data_aspect(1.0)
        .view_aspect(1.0)
        // Must set margins to zero or the image and plot bounds will
        // constantly fight, expanding the plot to infinity.
        .set_margin_fraction(Vec2::new(0.0, 0.0))
        .include_x(0.0)
        .include_x(1.0)
        .include_y(0.0)
        .include_y(1.0)
        .x_grid_spacer(|grid| (egui::widgets::plot::log_grid_spacer(16))(grid))
        .y_grid_spacer(|grid| (egui::widgets::plot::log_grid_spacer(16))(grid))
        .coordinates_formatter(
          egui::widgets::plot::Corner::LeftTop,
          egui::widgets::plot::CoordinatesFormatter::new(|pt, _| {
            format!(
              "x = {}\ny = {}",
              (pt.x * SIMULATION_DIMM[0] as f64) as i64,
              (pt.y * SIMULATION_DIMM[1] as f64) as i64,
            )
          })
        )
        .show_x(false)
        .show_y(false)
        .allow_scroll(false)
        .x_axis_formatter(|x, _| if x >= 0.0 && x <= 1.0 {
            ((x * SIMULATION_DIMM[0] as f64) as i64).to_string()
          } else { "".to_string() })
        .y_axis_formatter(|y, _| if y >= 0.0 && y <= 1.0 {
            ((y * SIMULATION_DIMM[1] as f64) as i64).to_string()
          } else { "".to_string() })
        .show(ui, |ui| {
          bounds = ui.plot_bounds();

          // Render the plot texture filling the viewport.
          ui.image(
            PlotImage::new(
              self.texture_id,
              bounds.center(),
              [bounds.width() as f32, bounds.height() as f32],
            ).name("Game of Life (GPU)"),
          );
        });

      // Add a callback to egui to render the plot contents to
      // texture.
      ui.painter().add(gpu_draw::egui_wgpu_callback(
        bounds,
        resp.response.rect,
        self.compute_requested
      ));

      // Update the texture handle in egui from the previously
      // rendered texture (from the last frame).
      let wgpu_render_state = frame.wgpu_render_state().unwrap();
      let mut renderer = wgpu_render_state.renderer.write();
      {
        let gpu_drawer: &GPUDrawer = renderer.paint_callback_resources.get().unwrap();
        let texture_view = gpu_drawer.create_view();

        renderer.update_egui_texture_from_wgpu_texture(
          &wgpu_render_state.device,
          &texture_view,
          wgpu::FilterMode::Linear,
          self.texture_id,
        );
      }
      ctx.request_repaint();
      //self.compute_requested = false;
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
