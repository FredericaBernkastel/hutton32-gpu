//! The archived application's pattern format: a PNG whose colours are the 32 Hutton32 colours.

use std::path::Path;

use super::Grid;
use crate::{rule::HUTTON32_COLORS, Error, Result};

/// Load a palette PNG; a colour outside the palette is an error.
pub fn load_png(path: &Path) -> Result<Grid> {
  let img = image::open(path).map_err(|e| Error::Pattern(format!("{}: {e}", path.display())))?.to_rgb8();
  let (w, h) = (img.width() as usize, img.height() as usize);
  let mut g = Grid::new(w, h);
  for (x, y, p) in img.enumerate_pixels() {
    let st = HUTTON32_COLORS
      .iter()
      .position(|c| *c == p.0)
      .ok_or_else(|| Error::Pattern(format!("{}: pixel ({x},{y}) colour {:?} is not in the palette", path.display(), p.0)))?;
    g.set(x as usize, y as usize, st as u8);
  }
  Ok(g)
}

/// Save a grid as a palette PNG.
pub fn save_png(path: &Path, g: &Grid, palette: &[[u8; 3]]) -> Result<()> {
  let mut img = image::RgbImage::new(g.w as u32, g.h as u32);
  for y in 0..g.h {
    for x in 0..g.w {
      let c = palette.get(g.get(x, y) as usize).copied().unwrap_or([255, 0, 255]);
      img.put_pixel(x as u32, y as u32, image::Rgb(c));
    }
  }
  img.save(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))
}
