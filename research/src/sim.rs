//! Plain per-cell table stepping on the CPU (reference semantics; zero boundary), for any rule of
//! `ca::Rule`: von Neumann, hexagonal or Moore neighbourhood.

use crate::ca::{Nbhd, Rule};
use crate::pattern::Grid;

pub struct Sim<'a> {
  pub w: usize,
  pub h: usize,
  stride: usize,
  cur: Vec<u8>,
  nxt: Vec<u8>,
  lut: &'a Rule,
  pub generation: u64,
}

/// Step cells `lo..hi` of one row: `up`, `mid`, `dn` are the rows above, at and below it (same
/// x coordinates), `out` receives the new row. Returns the number of cells that changed.
#[inline]
fn step_span(rule: &Rule, up: &[u8], mid: &[u8], dn: &[u8], out: &mut [u8], lo: usize, hi: usize) -> usize {
  let t = &rule.table[..];
  let b = rule.bits;
  let mut changed = 0usize;
  macro_rules! span {
    ($idx:expr) => {{
      let f = $idx;
      for x in lo..hi {
        let r = t[f(x)];
        changed += (r != mid[x]) as usize;
        out[x] = r;
      }
    }};
  }
  let u = |v: u8| v as usize;
  match rule.nbhd {
    // the Hutton32 layout, kept as a fast path
    Nbhd::VonNeumann if b == 5 => span!(|x: usize| u(mid[x - 1]) | u(dn[x]) << 5 | u(mid[x + 1]) << 10 | u(up[x]) << 15 | u(mid[x]) << 20),
    Nbhd::VonNeumann => span!(|x: usize| u(mid[x - 1]) | u(dn[x]) << b | u(mid[x + 1]) << (2 * b) | u(up[x]) << (3 * b) | u(mid[x]) << (4 * b)),
    Nbhd::Hex => span!(|x: usize| u(mid[x - 1])
      | u(dn[x]) << b
      | u(mid[x + 1]) << (2 * b)
      | u(up[x]) << (3 * b)
      | u(mid[x]) << (4 * b)
      | u(up[x - 1]) << (5 * b)
      | u(dn[x + 1]) << (6 * b)),
    // two states: the nine fields are single bits
    Nbhd::Moore if b == 1 => span!(|x: usize| u(mid[x - 1])
      | u(dn[x]) << 1
      | u(mid[x + 1]) << 2
      | u(up[x]) << 3
      | u(mid[x]) << 4
      | u(up[x - 1]) << 5
      | u(up[x + 1]) << 6
      | u(dn[x - 1]) << 7
      | u(dn[x + 1]) << 8),
    Nbhd::Moore => span!(|x: usize| u(mid[x - 1])
      | u(dn[x]) << b
      | u(mid[x + 1]) << (2 * b)
      | u(up[x]) << (3 * b)
      | u(mid[x]) << (4 * b)
      | u(up[x - 1]) << (5 * b)
      | u(up[x + 1]) << (6 * b)
      | u(dn[x - 1]) << (7 * b)
      | u(dn[x + 1]) << (8 * b)),
  }
  changed
}

impl<'a> Sim<'a> {
  pub fn new(g: &Grid, lut: &'a Rule) -> Self {
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
      changed += step_span(self.lut, up, mid, dn, out, 1, w + 1);
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
/// valid afterwards (the light cone of every radius-1 neighbourhood). Used for memo misses.
/// `buf`/`tmp` are `side*side`.
pub fn advance_window(rule: &Rule, buf: &mut Vec<u8>, tmp: &mut Vec<u8>, side: usize, gens: usize) {
  for g in 0..gens {
    // cells closer than g+1 to the border are already garbage; skip them
    let lo = g + 1;
    let hi = side - g - 1;
    for y in lo..hi {
      let r = y * side;
      let (up, mid, dn) = (&buf[r - side..r], &buf[r..r + side], &buf[r + side..r + 2 * side]);
      step_span(rule, up, mid, dn, &mut tmp[r..r + side], lo, hi);
    }
    std::mem::swap(buf, tmp);
  }
}
