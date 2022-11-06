use std::ops::DerefMut;
use std::sync::{Arc, RwLock, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::Duration;
use eframe::egui;
use eframe::egui::{ColorImage, TextureHandle, Vec2};
use image::{Luma, Rgba, Pixel, RgbaImage};
use poll_promise::Promise;
//use tracing::info;
use crate::gpu;

pub fn init() {
  let options = eframe::NativeOptions {
    initial_window_size: Some(Vec2::new(527.0, 569.0)),
    vsync: true,
    ..Default::default()
  };

  let texture = Arc::new(RwLock::new(
    RgbaImage::from_pixel(512, 512, Luma([32]).to_rgba())
  ));

  eframe::run_native(
    "My egui App",
    options,
    Box::new(|_cc| Box::new(MyApp {
        texture,
        texture_update_queued: Arc::new(AtomicBool::new(true)),
        ..Default::default() }
    )),
  );
}

#[derive(Default)]
struct MyApp {
  texture: Arc<RwLock<RgbaImage>>,
  texture_handle: Option<TextureHandle>,
  texture_update_queued: Arc<AtomicBool>,
  gpu_promise: Option<Promise<String>>,
}

impl MyApp {
  fn update_texture(&mut self, ctx: &egui::Context) {
    if self.texture_update_queued.load(Ordering::SeqCst) {
      if let Ok(texture) = self.texture.try_read() {
        let image = ctx.load_texture(
          "image",
          ColorImage::from_rgba_unmultiplied(
            [texture.width() as _, texture.height() as _],
            &texture
          ),
          egui::TextureFilter::Nearest
        );
        self.texture_handle = Some(image);
        self.texture_update_queued.store(false, Ordering::SeqCst);
      }
    }
  }
}

impl eframe::App for MyApp {
  fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
    egui::TopBottomPanel::top("my_panel").show(ctx, |ui| {
      if ui.button("GPU Execute").clicked() {
        let texture = Arc::clone(&self.texture);
        let ctx1 = ctx.clone();
        let texture_update_queued = Arc::clone(&self.texture_update_queued);
        self.gpu_promise = Some(Promise::spawn_thread("draw thread", move || {
          for i in 0..10000 {
            texture.write().unwrap().pixels_mut().for_each(|p| *p = Rgba([0, 0, 0, 255]));

            let i = i % 200;

            [ ((256 - 5 + i, 256), (256 + 5 + i, 256)),
              ((256 + i, 256 - 5), (256 + i, 256 + 5)) ].iter()
              .for_each(|segment|
                imageproc::drawing::draw_antialiased_line_segment_mut(
                  texture.write().unwrap().deref_mut(),
                  segment.0, segment.1,
                  Rgba([255, 0, 0, 255]),
                  imageproc::pixelops::interpolate
                )
            );
            texture_update_queued.store(true, Ordering::SeqCst);
            ctx1.request_repaint();

            thread::sleep(Duration::from_millis(15));
          }
          let ret = gpu::run();
          ctx1.request_repaint();
          ret
        }));
      }
    });
    egui::CentralPanel::default().show(ctx, |ui| {
       self.gpu_promise.as_ref().map(|p| match p.ready() {
        None => (), // still loading
        Some(ret) => {
          ui.label(format!("{ret}"));
        }
      });

      self.update_texture(ctx);

      self.texture_handle.as_ref().map(|tex|
        ui.image(tex, tex.size_vec2())
      )
    });
  }
}
