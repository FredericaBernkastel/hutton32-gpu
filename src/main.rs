use egui::Vec2;

fn main () {
  let native_options = eframe::NativeOptions {
    renderer: eframe::Renderer::Wgpu,
    initial_window_size: Some(Vec2::new(631.0, 465.0)),
    vsync: true,
    ..Default::default()
  };

  eframe::run_native(
    "GPU Accelerated CA",
    native_options,
    Box::new(|cc| Box::new(hutton32_gpu::GpuPlot::new(cc).unwrap())),
  );
}