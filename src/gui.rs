use std::sync::{Arc, RwLock, atomic::Ordering};
use std::sync::atomic::AtomicU64;
use std::thread;
use std::time::Duration;
use eframe::egui;
use eframe::egui::{ColorImage, TextureHandle, Vec2};
use image::{Luma, Pixel, RgbaImage};
use pollster::FutureExt as _;
//use tracing::info;
use crate::gpu;

pub fn init() {
  let options = eframe::NativeOptions {
    initial_window_size: Some(Vec2::new(527.0, 569.0)),
    vsync: true,
    ..Default::default()
  };

  let texture = RgbaImage::from_pixel(512, 512, Luma([32]).to_rgba());

  let gpu_kernel = Arc::new(RwLock::new(gpu::WGPUKernel::init(
    gpu::ExternalBuffers { texture }
  ).block_on().unwrap()));

  eframe::run_native(
    "My egui App",
    options,
    Box::new(|_cc| Box::new(MyApp {
      gpu_kernel,
      gpu_profile: Arc::new(AtomicU64::new(0)),
      texture_handle: Arc::new(RwLock::new(None))
    })),
  );
}

struct MyApp {
  gpu_kernel: Arc<RwLock<gpu::WGPUKernel>>,
  texture_handle: Arc<RwLock<Option<TextureHandle>>>,
  gpu_profile: Arc<AtomicU64>
}

impl MyApp {
  fn update_texture(
    gpu_kernel: &RwLock<gpu::WGPUKernel>,
    ctx: &egui::Context,
    texture_handle: &RwLock<Option<TextureHandle>>
  ) {
    let texture = &gpu_kernel.read().unwrap().external_buffers.texture;
    let image = ctx.load_texture(
      "image",
      ColorImage::from_rgba_unmultiplied(
        [texture.width() as _, texture.height() as _],
        &texture
      ),
      egui::TextureFilter::Nearest
    );
    *texture_handle.write().unwrap() = Some(image);
  }
}

impl eframe::App for MyApp {
  fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
    egui::TopBottomPanel::top("my_panel").show(ctx, |ui| {
      if ui.button("GPU Execute").clicked() {
        let ctx = ctx.clone();
        let gpu_kernel = self.gpu_kernel.clone();
        let gpu_profile = self.gpu_profile.clone();
        let texture_handle = self.texture_handle.clone();

        thread::spawn(move || {
          for i in 0..10000 {
            let t0 = std::time::Instant::now();
            gpu_kernel.write().unwrap().iter(i);
            gpu_profile.store(t0.elapsed().as_micros() as u64, Ordering::SeqCst);

            Self::update_texture(&gpu_kernel, &ctx, &texture_handle);
            ctx.request_repaint();

            // stable dt
            let frame_delay = ((1.0 / 60.0) - t0.elapsed().as_secs_f64()).max(1e-3);
            thread::sleep(Duration::from_secs_f64(frame_delay));
          }
        });
      }
    });
    egui::CentralPanel::default().show(ctx, |ui| {
      ui.label(format!("GPU profile: {:.3}ms", self.gpu_profile.load(Ordering::Relaxed) as f64 / 1000.0));
      if let Some(tex) = self.texture_handle.read().unwrap().as_ref() {
        ui.image(tex, tex.size_vec2());
      }
    });
  }
}
