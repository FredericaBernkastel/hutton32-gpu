use {
  eframe::{
    CreationContext,
    egui::{
      self,
      plot::{self, Plot, PlotImage, PlotBounds},
      Key, KeyboardShortcut, Modifiers, RichText, TextStyle, TextureId,
      TopBottomPanel, CollapsingHeader, CentralPanel, SidePanel
    },
    emath::Vec2,
    Storage,
  },
  wgpu,
  crate::gpu::{self, GPUDriver}
};

pub struct GUI {
  adapter_info: Option<wgpu::AdapterInfo>,
  compute_requested: bool,
  is_step: bool,
  texture_id: TextureId,

  edit_iters_frame: String,
  t0: Option<std::time::Instant>,
  generation: u64,

  debug_windows: DebugWingows
}

#[derive(Default)]
struct DebugWingows {
  ui_settings: bool,
  inspection: bool,
  memory: bool
}

impl GUI {
  pub fn new<'a>(cc: &'a CreationContext<'a>) -> Option<Self> {
    let wgpu_render_state = cc.wgpu_render_state.as_ref()?;

    let adapter_info = wgpu::Instance::new(wgpu::Backends::all())
      .enumerate_adapters(wgpu::Backends::all())
      .next().map(|a| a.get_info());

    let device = &wgpu_render_state.device;
    let target_format = wgpu_render_state.target_format;

    let edit_iters_frame = cc.storage.map(|s| s.get_string("edit_iters_frame"))
      .flatten().unwrap_or("1".to_string());

    let mut gpu_driver = GPUDriver::new(device, &wgpu_render_state.queue, target_format);
    gpu_driver.load_simulation(device, &wgpu_render_state.queue);
    gpu_driver.simulatiion_steps_per_call = edit_iters_frame.parse().unwrap_or(1);

    let texture_id = {
      let mut renderer = wgpu_render_state.renderer.write();
      renderer.register_native_texture(device, &gpu_driver.create_view(), wgpu::FilterMode::Linear)
    };

    wgpu_render_state
      .renderer
      .write()
      .paint_callback_resources
      .insert(gpu_driver);

    configure_text_styles(&cc.egui_ctx);

    Some(Self {
      adapter_info,
      compute_requested: false,
      is_step: false,
      texture_id,

      edit_iters_frame,
      t0: None,
      generation: 0,
      debug_windows: DebugWingows::default()
    })
  }

  fn on_start_click(&mut self, gpu_driver: &mut GPUDriver) {
    self.on_edit_iters_frame_changed(gpu_driver);
    self.compute_requested = !self.compute_requested;
    self.t0 = if self.t0.is_none() && self.compute_requested {
      Some(std::time::Instant::now())
    } else {
      None
    };
  }

  fn on_reset_click(&mut self, gpu_driver: &mut GPUDriver, device: &wgpu::Device, queue: &wgpu::Queue) {
    gpu_driver.load_simulation(device, queue);
    self.on_edit_iters_frame_changed(gpu_driver);
    self.generation = 0;
    self.t0 = None;
  }

  fn on_step_click(&mut self, gpu_driver: &mut GPUDriver) {
    gpu_driver.simulatiion_steps_per_call = 1;
    self.compute_requested = true;
    self.is_step = true;
  }

  fn on_recomple_click(
    &mut self,
    gpu_driver: &mut GPUDriver,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    target_format: wgpu::TextureFormat
  ) {
    *gpu_driver = GPUDriver::new(device, queue, target_format);
    self.on_reset_click(gpu_driver, device, queue);
  }

  fn on_edit_iters_frame_changed(&self, gpu_driver: &mut GPUDriver) {
    match self.edit_iters_frame.parse::<u64>() {
      Ok(step_zize @ 1..=512) => {
        gpu_driver.simulatiion_steps_per_call = step_zize;
      },
      _ => ()
    }
  }
}

impl eframe::App for GUI {
  fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
    frame.set_window_title("GPU Accelerated CA");

    let render_state = frame.wgpu_render_state().unwrap();
    let mut renderer = render_state.renderer.write();
    let device = render_state.device.as_ref();
    let queue = render_state.queue.as_ref();
    let gpu_driver = renderer.paint_callback_resources.get_mut::<GPUDriver>().unwrap();

    if self.is_step && self.compute_requested {
      self.compute_requested = false;
      self.is_step = false;
      //self.t0.map(|t0| println!("{:.3}s", t0.elapsed().as_secs_f64()));
    }

    TopBottomPanel::top("control buttons").show(ctx, |ui| {
      ui.add_space(1.0);

      ui.horizontal_wrapped(|ui| {
        ui.style_mut().visuals.button_frame = false;

        (
          ui.button(if !self.compute_requested { "▶ Start" } else { "⏸ Stop" })
            .on_hover_text_at_pointer("Space")
            .clicked() || ui.input_mut().consume_shortcut(&KeyboardShortcut { modifiers: Modifiers::NONE, key: Key::Space })
        ).then(|| self.on_start_click(gpu_driver));

        ui.label("|");

        (
          ui.button("↺  Reset")
            .on_hover_text_at_pointer("R")
            .clicked() || ui.input_mut().consume_shortcut(&KeyboardShortcut { modifiers: Modifiers::NONE, key: Key::R })
        ).then(|| self.on_reset_click(gpu_driver, device, queue));

        ui.label("|");

        (
          ui.button("▶|| Step")
            .on_hover_text_at_pointer("S")
            .clicked() || ui.input_mut().consume_shortcut(&KeyboardShortcut { modifiers: Modifiers::NONE, key: Key::S })
        ).then(|| self.on_step_click(gpu_driver));

        ui.label("|");

        (
          ui.button("< / > Recompile")
            .on_hover_text_at_pointer("Ctrl+R")
            .clicked() || ui.input_mut().consume_shortcut(&KeyboardShortcut { modifiers: Modifiers::CTRL, key: Key::R })
        ).then(|| self.on_recomple_click(gpu_driver, device, queue, render_state.target_format));
      });

      ui.add_space(1.0);
    });

    SidePanel::left("left_panel")
      .default_width(180.0)
      .show(ctx, |ui| {
        ui.add_space(10.0);
        ui.horizontal_wrapped(|ui| {
          ui.label("iters / frame: ");
          ui.text_edit_singleline(&mut self.edit_iters_frame)
            .lost_focus().then(|| self.on_edit_iters_frame_changed(gpu_driver));
        });

        ui.add_space(10.0);
        ui.label("\
          LMB: pan\n\
          Ctrl+Scroll: zoom\n\
          RMB: boxed zoom mode\n"
        );
        ui.separator();
        ui.add_space(10.0);
        CollapsingHeader::new("Statistics")
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
              self.generation,
              gpu_driver.texture_size,
              gpu_driver.uniforms.simulation_dimm,
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

    let simulation_dimm = gpu_driver.uniforms.simulation_dimm;

    CentralPanel::default().show(ctx, |ui| {
      let mut bounds = PlotBounds::NOTHING;
      let resp = Plot::new("my_plot")
        //.legend(Legend::default())
        .data_aspect(1.0)
        // Must set margins to zero or the image and plot bounds will
        // constantly fight, expanding the plot to infinity.
        .set_margin_fraction(Vec2::new(0.0, 0.0))
        .include_x(simulation_dimm[0] as f64 * -0.33)
        .include_x(simulation_dimm[0] as f64 * 1.33)
        .include_y(simulation_dimm[1] as f64 * 0.33)
        .include_y(simulation_dimm[1] as f64 * -1.33)
        .x_grid_spacer(plot::log_grid_spacer(16))
        .y_grid_spacer(plot::log_grid_spacer(16))
        .coordinates_formatter(
          plot::Corner::LeftTop,
          plot::CoordinatesFormatter::new(move |pt, _|
            format!("x = {}\ny = {}", pt.x as i64, pt.y as i64, )
          ))
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
      ui.painter().add(gpu::egui_wgpu_callback(
        bounds,
        resp.response.rect,
        self.compute_requested
      ));
    });

    self.compute_requested.then(|| {
      self.generation = self.generation.wrapping_add(gpu_driver.simulatiion_steps_per_call);
    });

    // Update the texture handle in egui from the previously
    // rendered texture (from the last frame).
    let texture_view = gpu_driver.create_view();
    renderer.update_egui_texture_from_wgpu_texture(
      &render_state.device,
      &texture_view,
      wgpu::FilterMode::Linear,
      self.texture_id,
    );

    self.compute_requested.then(||
      ctx.request_repaint()
    );
  }

  // save app state on exit
  fn save(&mut self, storage: &mut dyn Storage) {
    storage.set_string("edit_iters_frame", self.edit_iters_frame.clone());
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
