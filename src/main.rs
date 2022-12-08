mod gui;
mod gpu;

// When compiling natively:
#[cfg(not(target_arch = "wasm32"))]
fn main() {
  // Log to stdout (if you run with `RUST_LOG=debug`).
  //tracing_subscriber::fmt::init();

  let native_options = eframe::NativeOptions {
    renderer: eframe::Renderer::Wgpu,
    initial_window_size: Some(egui::Vec2::new(631.0, 465.0)),
    vsync: true,
    ..Default::default()
  };

  eframe::run_native(
    "hutton32-gpu",
    native_options,
    Box::new(|cc| Box::new(gui::GUI::new(cc).unwrap())),
  );
}

// when compiling to web using trunk.
//#[cfg(target_arch = "wasm32")]
fn main() {
  // Make sure panics are logged using `console.error`.
  console_error_panic_hook::set_once();

  // Redirect tracing to console.log and friends:
  tracing_wasm::set_as_global_default();

  let web_options = eframe::WebOptions {
    wgpu_options: egui_wgpu::WgpuConfiguration {
      // WebGPU is not stable enough yet, use WebGL emulation
      backends: wgpu::Backends::GL,
      device_descriptor: wgpu::DeviceDescriptor {
        label: Some("egui wgpu device"),
        features: wgpu::Features::default(),
        limits: wgpu::Limits {
          // When using a depth buffer, we have to be able to create a texture
          // large enough for the entire surface, and we want to support 4k+ displays.
          max_texture_dimension_2d: 8192,
          max_storage_buffers_per_shader_stage: 0,
          ..wgpu::Limits::downlevel_webgl2_defaults()
        },
      },
      ..Default::default()
    },
    ..Default::default()
  };

  wasm_bindgen_futures::spawn_local(async {
    eframe::start_web(
      "the_canvas_id", // hardcode it
      web_options,
      Box::new(|cc| Box::new(gui::GUI::new(cc).unwrap())),
    )
      .await
      .expect("failed to start eframe");
  });
}