//! Pattern I/O: palette PNG (as used by the archived app) and Golly extended RLE.

use std::{fmt::Write as _, fs, path::Path};

/// Copy of `HUTTON32_COLORS` from `archive/src/gpu/gpu_automata.rs`.
pub const HUTTON32_COLORS: [[u8; 3]; 32] = [
  [0, 0, 0], [255, 0, 0], [255, 125, 0], [255, 150, 25], [255, 175, 50], [255, 200, 75], [255, 225, 100],
  [255, 250, 125], [251, 255, 0], [89, 89, 255], [106, 106, 255], [122, 122, 255], [139, 139, 255],
  [27, 176, 27], [36, 200, 36], [73, 255, 73], [106, 255, 106], [235, 36, 36], [255, 56, 56], [255, 73, 73],
  [255, 89, 89], [185, 56, 255], [191, 73, 255], [197, 89, 255], [203, 106, 255], [0, 255, 128],
  [255, 128, 64], [255, 255, 128], [33, 215, 215], [27, 176, 176], [24, 156, 156], [21, 137, 137],
];

#[derive(Clone, PartialEq, Eq)]
pub struct Grid {
  pub w: usize,
  pub h: usize,
  /// Row-major, y grows downwards (north = y-1), one state per byte.
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
  /// Bounding box of non-zero cells: (x0, y0, x1, y1), exclusive upper bounds.
  pub fn bbox(&self) -> Option<(usize, usize, usize, usize)> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for y in 0..self.h {
      for x in 0..self.w {
        if self.get(x, y) != 0 {
          x0 = x0.min(x);
          y0 = y0.min(y);
          x1 = x1.max(x + 1);
          y1 = y1.max(y + 1);
        }
      }
    }
    if x1 == 0 { None } else { Some((x0, y0, x1, y1)) }
  }
  pub fn crop(&self, x0: usize, y0: usize, x1: usize, y1: usize) -> Grid {
    let mut g = Grid::new(x1 - x0, y1 - y0);
    for y in y0..y1 {
      g.cells[(y - y0) * g.w..(y - y0 + 1) * g.w].copy_from_slice(&self.cells[y * self.w + x0..y * self.w + x1]);
    }
    g
  }
}

/// Load a palette PNG exactly like `load_simulation`, but fail loudly on unknown colours.
pub fn load_png(path: &Path) -> Grid {
  let img = image::open(path).unwrap_or_else(|e| panic!("failed to load {}: {e}", path.display())).to_rgb8();
  let (w, h) = (img.width() as usize, img.height() as usize);
  let mut g = Grid::new(w, h);
  for (x, y, p) in img.enumerate_pixels() {
    let st = HUTTON32_COLORS
      .iter()
      .position(|c| *c == p.0)
      .unwrap_or_else(|| panic!("{}: pixel ({x},{y}) colour {:?} is not in the palette", path.display(), p.0));
    g.set(x as usize, y as usize, st as u8);
  }
  g
}

fn rle_state(out: &mut String, s: u8) {
  if s == 0 {
    out.push('.');
  } else {
    let s = s as u32 - 1;
    if s >= 24 {
      out.push((b'p' + (s / 24 - 1) as u8) as char);
    }
    out.push((b'A' + (s % 24) as u8) as char);
  }
}

/// Write Golly extended RLE.
pub fn write_rle(path: &Path, g: &Grid, rule: &str) {
  let mut body = String::new();
  let mut line_len = 0usize;
  let mut emit = |body: &mut String, count: usize, tok: &str| {
    let mut item = String::new();
    if count > 1 {
      write!(item, "{count}").unwrap();
    }
    item.push_str(tok);
    if line_len + item.len() > 70 {
      body.push('\n');
      line_len = 0;
    }
    line_len += item.len();
    body.push_str(&item);
  };
  let mut pending_newlines = 0usize;
  for y in 0..g.h {
    let row = &g.cells[y * g.w..(y + 1) * g.w];
    let end = row.iter().rposition(|&c| c != 0).map(|p| p + 1).unwrap_or(0);
    if end > 0 {
      if pending_newlines > 0 {
        emit(&mut body, pending_newlines, "$");
        pending_newlines = 0;
      }
      let mut x = 0;
      while x < end {
        let s = row[x];
        let mut run = 1;
        while x + run < end && row[x + run] == s {
          run += 1;
        }
        let mut tok = String::new();
        rle_state(&mut tok, s);
        emit(&mut body, run, &tok);
        x += run;
      }
    }
    pending_newlines += 1;
  }
  body.push('!');
  let text = format!("x = {}, y = {}, rule = {}\n{}\n", g.w, g.h, rule, body);
  fs::write(path, text).unwrap();
}

/// Read Golly extended RLE. Returns the grid and the rule name from the header.
pub fn read_rle(path: &Path) -> (Grid, String) {
  let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
  let mut lines = text.lines();
  let (mut w, mut h, mut rule) = (0usize, 0usize, String::new());
  let mut data = String::new();
  for line in &mut lines {
    let t = line.trim();
    if t.is_empty() || t.starts_with('#') {
      continue;
    }
    if t.starts_with('x') {
      for part in t.split(',') {
        let mut kv = part.splitn(2, '=');
        let k = kv.next().unwrap().trim();
        let v = kv.next().unwrap_or("").trim();
        match k {
          "x" => w = v.parse().unwrap(),
          "y" => h = v.parse().unwrap(),
          "rule" => rule = v.to_string(),
          _ => {}
        }
      }
      break;
    }
  }
  for line in lines {
    if !line.starts_with('#') {
      data.push_str(line.trim());
    }
  }
  let mut g = Grid::new(w, h);
  let (mut x, mut y, mut count, mut prefix) = (0usize, 0usize, 0usize, 0u32);
  for ch in data.bytes() {
    match ch {
      b'0'..=b'9' => count = count * 10 + (ch - b'0') as usize,
      b'$' => {
        y += count.max(1);
        x = 0;
        count = 0;
      }
      b'!' => break,
      b'p'..=b'y' => prefix = (ch - b'p') as u32 + 1,
      b'.' | b'b' | b'o' | b'A'..=b'X' => {
        let s = match ch {
          b'.' | b'b' => 0,
          b'o' => 1,
          _ => prefix * 24 + (ch - b'A') as u32 + 1,
        };
        prefix = 0;
        let n = count.max(1);
        if s != 0 {
          for i in 0..n {
            g.set(x + i, y, s as u8);
          }
        }
        x += n;
        count = 0;
      }
      _ => {}
    }
  }
  (g, rule)
}
