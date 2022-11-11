use eframe::egui::plot::{PlotImage};
use eframe::egui::{self, plot::PlotBounds};
use eframe::emath::Vec2;
use egui::{RichText, TextStyle};
use wgpu;

mod gpu_draw;
use gpu_draw::GPUDrawer;

pub struct GpuPlot {
  adapter_info: Option<wgpu::AdapterInfo>,
  compute_requested: bool,
  texture_id: egui::TextureId,

  edit_iters_frame: String,
  t0: Option<std::time::Instant>,

  debug_windows: DebugWingows
}

#[derive(Default)]
struct DebugWingows {
  ui_settings: bool,
  inspection: bool,
  memory: bool
}

//const SIMULATION_DIMM: [u32; 2] = [256, 256];

impl GpuPlot {
  pub fn new<'a>(cc: &'a eframe::CreationContext<'a>) -> Option<Self> {
    let wgpu_render_state = cc.wgpu_render_state.as_ref()?;

    let adapter_info = wgpu::Instance::new(wgpu::Backends::all())
      .enumerate_adapters(wgpu::Backends::all())
      .next().map(|a| a.get_info());

    let device = &wgpu_render_state.device;
    let target_format = wgpu_render_state.target_format;

    let mut gpu_drawer = GPUDrawer::new(device, &wgpu_render_state.queue, target_format);
    gpu_drawer.load_simulation(device, &wgpu_render_state.queue);
    let texture_id = {
      let mut renderer = wgpu_render_state.renderer.write();
      renderer.register_native_texture(device, &gpu_drawer.create_view(), wgpu::FilterMode::Linear)
    };

    wgpu_render_state
      .renderer
      .write()
      .paint_callback_resources
      .insert(gpu_drawer);

    configure_text_styles(&cc.egui_ctx);

    Some(Self {
      adapter_info,
      compute_requested: false,
      texture_id,

      edit_iters_frame: "1".to_string(),
      t0: None,
      debug_windows: DebugWingows::default()
    })
  }
}

impl eframe::App for GpuPlot {
  fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
    let render_state = frame.wgpu_render_state().unwrap();
    let queue = render_state.queue.as_ref();
    let mut renderer = render_state.renderer.write();
    let gpu_drawer = renderer.paint_callback_resources.get_mut::<GPUDrawer>().unwrap();

    /*
     * profiling for {
     *   simulation_dimm: 633x449,
     *   generations: 8192,
     *   iters/frame: 32
     * }
     *
     * hutton32, naive branching -> 7.803s
     * hutton32, LUT 32MB        -> 5.695s (memory bound)
     */
    /*if gpu_drawer.uniforms.time >= 8192 && self.compute_requested {
      self.compute_requested = false;
      self.t0.map(|t0| println!("{:.3}s", t0.elapsed().as_secs_f64()));
    }*/

    egui::SidePanel::left("left_panel")
      .default_width(180.0)
      .show(ctx, |ui| {
        ui.horizontal_wrapped(|ui| {
          let label = if !self.compute_requested { "▶ Start" } else { "⏸ Stop" };
          ui.button(label).clicked().then(|| {
            self.compute_requested = !self.compute_requested;
            self.t0 = if self.t0.is_none() {
              Some(std::time::Instant::now())
            } else {
              None
            };
          });

          ui.button("↺  Reset").clicked().then(|| {
            gpu_drawer.load_simulation(&render_state.device, queue);
            gpu_drawer.uniforms.time = 0;
            self.t0 = None;
          });
        });

        ui.add_space(10.0);
        ui.horizontal_wrapped(|ui| {
          ui.label("iters / frame: ");
          let input = ui.text_edit_singleline(&mut self.edit_iters_frame);
          if input.lost_focus() {
            match self.edit_iters_frame.parse::<u32>() {
              Ok(step_zize @ 1..=256) => {
                gpu_drawer.simulatiion_steps_per_call = step_zize;
              },
              _ => ()
            }
          }
        });

        ui.add_space(10.0);
        ui.label("\
          LMB: pan\n\
          Ctrl+Scroll: zoom\n\
          RMB: boxed zoom mode\n"
        );
        ui.separator();
        ui.add_space(10.0);
        egui::CollapsingHeader::new("Statistics")
          .default_open(true)
          .show(ui, |ui| ui.scope(|ui| {
            //ui.style_mut().wrap = Some(false);
            ui.label(RichText::new(format!("\
              device: {}\n\
              generation: {}\n\
              texture_size: {:?}\n\
              simulation_size: {:?}\n\
              T: {:.3}s",
              self.adapter_info.as_ref().map(|a| a.name.as_ref()).unwrap_or(""),
              gpu_drawer.uniforms.time,
              gpu_drawer.texture_size,
              gpu_drawer.uniforms.simulation_dimm,
              self.t0.map(|t0| t0.elapsed().as_secs_f64()).unwrap_or(0.0)
            )).text_style(TextStyle::Name("mono_small".into())))
          }));

        ui.add_space(10.0);
        ui.checkbox(&mut self.debug_windows.ui_settings, "🔧 UI Settings");
        ui.checkbox(&mut self.debug_windows.inspection, "🔍 Inspection");
        ui.checkbox(&mut self.debug_windows.memory, "📝 Memory");

        egui::Window::new("🔧 UI Settings")
          .open(&mut self.debug_windows.ui_settings)
          .vscroll(true)
          .show(ctx, |ui| {
            ctx.settings_ui(ui);
          });
        egui::Window::new("🔍 Inspection")
          .open(&mut self.debug_windows.inspection)
          .vscroll(true)
          .show(ctx, |ui| {
            ctx.inspection_ui(ui);
          });
        egui::Window::new("📝 Memory")
          .open(&mut self.debug_windows.memory)
          .resizable(false)
          .show(ctx, |ui| {
            ctx.memory_ui(ui);
          });
      });

    let simulation_dimm = gpu_drawer.uniforms.simulation_dimm;

    egui::CentralPanel::default().show(ctx, |ui| {
      let mut bounds = PlotBounds::NOTHING;
      let resp = egui::plot::Plot::new("my_plot")
        //.legend(Legend::default())
        .data_aspect(1.0)
        // Must set margins to zero or the image and plot bounds will
        // constantly fight, expanding the plot to infinity.
        .set_margin_fraction(Vec2::new(0.0, 0.0))
        .include_x(simulation_dimm[0] as f64 * -0.33)
        .include_x(simulation_dimm[0] as f64 * 1.33)
        .include_y(simulation_dimm[1] as f64 * 0.33)
        .include_y(simulation_dimm[1] as f64 * -1.33)
        .x_grid_spacer(egui::widgets::plot::log_grid_spacer(16))
        .y_grid_spacer(egui::widgets::plot::log_grid_spacer(16))
        .coordinates_formatter(
          egui::widgets::plot::Corner::LeftTop,
          egui::widgets::plot::CoordinatesFormatter::new(move |pt, _| {
            format!("x = {}\ny = {}", pt.x as i64, pt.y as i64, )
          })
        )
        .show_x(false)
        .show_y(false)
        .allow_scroll(false)
        .x_axis_formatter(move |x, _| if x >= 0.0 { x.to_string() } else { "".to_string() })
        .y_axis_formatter(move |y, _| if y <= 0.0 { (-y).to_string() } else { "".to_string() })
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
      drop(renderer); // reacquire lifetime
      let mut renderer = render_state.renderer.write();
      let gpu_drawer = renderer.paint_callback_resources.get::<GPUDrawer>().unwrap();
      let texture_view = gpu_drawer.create_view();

      renderer.update_egui_texture_from_wgpu_texture(
        &render_state.device,
        &texture_view,
        wgpu::FilterMode::Linear,
        self.texture_id,
      );

      if self.compute_requested {
        ctx.request_repaint();
      }
      //self.compute_requested = false;
    });
  }
}

fn configure_text_styles(ctx: &egui::Context) {
  use egui::{FontFamily::{Proportional, Monospace}, FontId};

  let mut style = (*ctx.style()).clone();
  style.text_styles = [
    (TextStyle::Heading, FontId::new(25.0, Proportional)),
    (TextStyle::Body, FontId::new(12.0, Proportional)),
    (TextStyle::Monospace, FontId::new(12.0, Monospace)),
    (TextStyle::Button, FontId::new(12.0, Proportional)),
    (TextStyle::Small, FontId::new(8.0, Proportional)),
    (TextStyle::Name("mono_small".into()), FontId::new(8.0, Monospace)),
  ].into();
  ctx.set_style(style);
}
