mod gui;
mod gpu;

fn main () {
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