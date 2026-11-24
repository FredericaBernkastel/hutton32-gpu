//! Plain per-cell LUT stepping on the CPU (reference semantics; zero boundary).

use crate::pattern::Grid;

pub struct Sim<'a> {
  pub w: usize,
  pub h: usize,
  stride: usize,
  cur: Vec<u8>,
  nxt: Vec<u8>,
  lut: &'a [u8],
  pub generation: u64,
}

impl<'a> Sim<'a> {
  pub fn new(g: &Grid, lut: &'a [u8]) -> Self {
    let stride = g.w + 2;
    let mut cur = vec![0u8; stride * (g.h + 2)];
    for y in 0..g.h {
      cur[(y + 1) * stride + 1..(y + 1) * stride + 1 + g.w].copy_from_slice(&g.cells[y * g.w..(y + 1) * g.w]);
    }
    let nxt = vec![0u8; cur.len()];
    Sim { w: g.w, h: g.h, stride, cur, nxt, lut, generation: 0 }
  }

  /// Advance one generation; returns the number of cells that changed.
  pub fn step(&mut self) -> usize {
    let (s, w, h) = (self.stride, self.w, self.h);
    let mut changed = 0usize;
    for y in 1..=h {
      let up = &self.cur[(y - 1) * s..y * s];
      let mid = &self.cur[y * s..(y + 1) * s];
      let dn = &self.cur[(y + 1) * s..(y + 2) * s];
      let out = &mut self.nxt[y * s..(y + 1) * s];
      for x in 1..=w {
        let c = mid[x] as usize;
        let idx = (mid[x - 1] as usize) | (dn[x] as usize) << 5 | (mid[x + 1] as usize) << 10 | (up[x] as usize) << 15 | c << 20;
        let r = self.lut[idx];
        changed += (r as usize != c) as usize;
        out[x] = r;
      }
    }
    std::mem::swap(&mut self.cur, &mut self.nxt);
    self.generation += 1;
    changed
  }

  pub fn grid(&self) -> Grid {
    let mut g = Grid::new(self.w, self.h);
    for y in 0..self.h {
      g.cells[y * self.w..(y + 1) * self.w]
        .copy_from_slice(&self.cur[(y + 1) * self.stride + 1..(y + 1) * self.stride + 1 + self.w]);
    }
    g
  }

  /// 64-bit FNV-style hash of the interior.
  pub fn hash(&self) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for y in 1..=self.h {
      for &c in &self.cur[y * self.stride + 1..y * self.stride + 1 + self.w] {
        h = (h ^ c as u64).wrapping_mul(0x100000001b3);
      }
      h = h.rotate_left(17) ^ 0x9e3779b97f4a7c15;
    }
    h
  }

  /// True if any non-zero cell touches the outermost ring of the grid.
  pub fn touches_border(&self) -> bool {
    let s = self.stride;
    let row = |y: usize| self.cur[y * s + 1..y * s + 1 + self.w].iter().any(|&c| c != 0);
    row(1) || row(self.h) || (1..=self.h).any(|y| self.cur[y * s + 1] != 0 || self.cur[y * s + self.w] != 0)
  }
}

/// Advance a square window in place for `gens` generations with garbage creeping in from the
/// boundary (outside treated as zero). Only the cells at distance >= gens from the border are
/// valid afterwards. Used for memo misses. `buf`/`tmp` are `side*side`.
pub fn advance_window(lut: &[u8], buf: &mut Vec<u8>, tmp: &mut Vec<u8>, side: usize, gens: usize) {
  for g in 0..gens {
    // cells closer than g+1 to the border are already garbage; skip them
    let lo = g + 1;
    let hi = side - g - 1;
    for y in lo..hi {
      let r = y * side;
      for x in lo..hi {
        let c = buf[r + x] as usize;
        let idx = (buf[r + x - 1] as usize)
          | (buf[r + side + x] as usize) << 5
          | (buf[r + x + 1] as usize) << 10
          | (buf[r - side + x] as usize) << 15
          | c << 20;
        tmp[r + x] = lut[idx];
      }
    }
    std::mem::swap(buf, tmp);
  }
}
