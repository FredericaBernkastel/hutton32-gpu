//! Patterns: dense grids of cells and macrocell quadtrees, read from and written to Golly's RLE
//! and macrocell formats, and the archived application's palette PNG (feature `png`).

mod macrocell;
#[cfg(feature = "png")]
mod png;
mod rle;

use std::path::Path;

pub use macrocell::{Macrocell, McNode};
#[cfg(feature = "png")]
pub use png::{load_png, save_png};
pub use rle::{read_rle, write_rle};

use crate::{Error, Result};

/// A dense rectangle of cells, row-major, y downwards (north is y - 1), one state per byte.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Grid {
  pub w: usize,
  pub h: usize,
  pub cells: Vec<u8>,
}

impl Grid {
  pub fn new(w: usize, h: usize) -> Self {
    Grid { w, h, cells: vec![0; w * h] }
  }
  #[inline]
  pub fn get(&self, x: usize, y: usize) -> u8 {
    self.cells[y * self.w + x]
  }
  #[inline]
  pub fn set(&mut self, x: usize, y: usize, v: u8) {
    self.cells[y * self.w + x] = v;
  }
  pub fn population(&self) -> usize {
    self.cells.iter().filter(|&&c| c != 0).count()
  }
  /// Copy `src` into `self` with its top-left corner at `(x0, y0)`.
  pub fn blit(&mut self, src: &Grid, x0: usize, y0: usize) {
    for y in 0..src.h {
      let d = (y0 + y) * self.w + x0;
      self.cells[d..d + src.w].copy_from_slice(&src.cells[y * src.w..(y + 1) * src.w]);
    }
  }
  /// Embed into a larger zero grid.
  pub fn padded(&self, left: usize, top: usize, w: usize, h: usize) -> Grid {
    let mut g = Grid::new(w, h);
    g.blit(self, left, top);
    g
  }
  /// Bounding box of the non-zero cells: `(x0, y0, x1, y1)`, upper bounds exclusive.
  pub fn bbox(&self) -> Option<(usize, usize, usize, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for y in 0..self.h {
      let row = &self.cells[y * self.w..(y + 1) * self.w];
      if let Some(a) = row.iter().position(|&c| c != 0) {
        let b = row.iter().rposition(|&c| c != 0).unwrap();
        x0 = x0.min(a);
        x1 = x1.max(b + 1);
        y0 = y0.min(y);
        y1 = y + 1;
      }
    }
    if x1 == 0 {
      None
    } else {
      Some((x0, y0, x1, y1))
    }
  }
  pub fn crop(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> Grid {
    let mut g = Grid::new(x1 - x0, y1 - y0);
    for y in y0..y1 {
      g.cells[(y - y0) * g.w..(y - y0 + 1) * g.w].copy_from_slice(&self.cells[y * self.w + x0..y * self.w + x1]);
    }
    g
  }
}

/// A pattern as read from a file: dense cells or a macrocell quadtree, with the rule its file names.
#[derive(Clone, Debug)]
pub enum Pattern {
  Cells(Grid),
  Macrocell(Macrocell),
}

/// The output formats of [`crate::Universe::save`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Format {
  Rle,
  Macrocell,
}

/// A pattern file and the rule named in it (empty when it names none).
pub struct Loaded {
  pub pattern: Pattern,
  pub rule: String,
}

impl Pattern {
  /// Load a pattern by extension: `.rle`, `.mc` (and `.png` with the `png` feature).
  pub fn load(path: impl AsRef<Path>) -> Result<Loaded> {
    let path = path.as_ref();
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
      "rle" => {
        let text = std::fs::read_to_string(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
        let (g, rule) = read_rle(&text)?;
        Ok(Loaded { pattern: Pattern::Cells(g), rule })
      }
      "mc" => {
        let text = std::fs::read_to_string(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
        let m = Macrocell::parse(&text)?;
        let rule = m.rule.clone();
        Ok(Loaded { pattern: Pattern::Macrocell(m), rule })
      }
      #[cfg(feature = "png")]
      "png" => Ok(Loaded { pattern: Pattern::Cells(load_png(path)?), rule: String::new() }),
      _ => Err(Error::Pattern(format!("unknown pattern extension: {}", path.display()))),
    }
  }
}

impl From<Grid> for Pattern {
  fn from(g: Grid) -> Self {
    Pattern::Cells(g)
  }
}
