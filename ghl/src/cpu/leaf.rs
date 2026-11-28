//! Brute force on the CPU: cells as bit-planes in `u64` words, stepped by the rule's network 64 cells
//! at a time (the Life-like adder tree, the compiled Hutton32b circuit, any other gate network), or
//! cell by cell through the table or tree for rules without one.
//!
//! A [`PlaneGrid`] holds `planes` planes of `h` rows of `ww` words; bit i of word j of a row is the
//! cell at x = 64 j + i. Cells outside the grid are 0. Windows that the recursion brute-forces are
//! laid side by side in one grid, an atlas: garbage from a neighbouring window travels one cell per
//! generation and cannot reach a window's centre in a quarter of its side.

use std::sync::Arc;

use super::simd::{self, circuits, Lanes, X4};
use crate::{
  rule::{hutton32_planes, hutton32b_planes, Network, NetworkKind, Neighbourhood, Pos, Rule},
  tile::TileGeom,
};

/// Cells as bit-planes, `u64` words.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct PlaneGrid {
  pub planes: usize,
  pub h: usize,
  /// words per row
  pub ww: usize,
  pub data: Vec<u64>,
}

impl PlaneGrid {
  pub fn new(planes: usize, w_cells: usize, h: usize) -> PlaneGrid {
    let ww = w_cells.div_ceil(64).max(1);
    PlaneGrid { planes, h, ww, data: vec![0; planes * h * ww] }
  }
  pub fn width(&self) -> usize {
    self.ww * 64
  }
  /// Re-shape to `planes` planes of `w_cells` by `h`, all zero, reusing the allocation.
  pub fn reset(&mut self, planes: usize, w_cells: usize, h: usize) {
    self.planes = planes;
    self.h = h;
    self.ww = w_cells.div_ceil(64).max(1);
    self.data.clear();
    self.data.resize(planes * h * self.ww, 0);
  }
  #[inline]
  fn idx(&self, k: usize, y: usize, j: usize) -> usize {
    (k * self.h + y) * self.ww + j
  }
  pub fn clear(&mut self) {
    self.data.fill(0);
  }
  #[inline]
  pub fn get(&self, x: usize, y: usize) -> u8 {
    let mut c = 0u8;
    for k in 0..self.planes {
      c |= (((self.data[self.idx(k, y, x / 64)] >> (x % 64)) & 1) as u8) << k;
    }
    c
  }
  #[inline]
  pub fn set(&mut self, x: usize, y: usize, v: u8) {
    for k in 0..self.planes {
      let i = self.idx(k, y, x / 64);
      let m = 1u64 << (x % 64);
      if (v >> k) & 1 == 1 {
        self.data[i] |= m;
      } else {
        self.data[i] &= !m;
      }
    }
  }
  /// `n <= 64` bits of plane `k`, row `y`, from cell `x0` (bit 0 = cell x0).
  #[inline]
  pub fn row_bits(&self, k: usize, y: usize, x0: usize, n: usize) -> u64 {
    let (j, o) = (x0 / 64, x0 % 64);
    let lo = self.data[self.idx(k, y, j)] >> o;
    let v = if o != 0 && o + n > 64 && j + 1 < self.ww { lo | self.data[self.idx(k, y, j + 1)] << (64 - o) } else { lo };
    if n == 64 {
      v
    } else {
      v & ((1u64 << n) - 1)
    }
  }
  /// Overwrite `n <= 64` bits of plane `k`, row `y`, from cell `x0`.
  #[inline]
  pub fn put_row_bits(&mut self, k: usize, y: usize, x0: usize, n: usize, v: u64) {
    let (j, o) = (x0 / 64, x0 % 64);
    let m = if n == 64 { !0u64 } else { (1u64 << n) - 1 };
    let v = v & m;
    let i = self.idx(k, y, j);
    self.data[i] = (self.data[i] & !(m << o)) | (v << o);
    if o != 0 && o + n > 64 {
      let i2 = self.idx(k, y, j + 1);
      let m2 = m >> (64 - o);
      self.data[i2] = (self.data[i2] & !m2) | (v >> (64 - o));
    }
  }
  /// Copy a base tile (bit-planes in `geom`'s layout) to `(x0, y0)`.
  pub fn blit_tile(&mut self, geom: &TileGeom, words: &[u32], x0: usize, y0: usize) {
    for k in 0..self.planes.min(geom.planes as usize) {
      for y in 0..geom.side {
        self.put_row_bits(k, y0 + y, x0, geom.side, geom.row(words, k, y));
      }
    }
  }
  /// Read the base tile at `(x0, y0)` into `out`.
  pub fn extract_tile(&self, geom: &TileGeom, x0: usize, y0: usize, out: &mut [u32]) {
    out.fill(0);
    for k in 0..geom.planes as usize {
      for y in 0..geom.side {
        geom.set_row(out, k, y, self.row_bits(k, y0 + y, x0, geom.side));
      }
    }
  }
  pub fn from_cells(planes: usize, w: usize, h: usize, cells: &[u8]) -> PlaneGrid {
    let mut g = PlaneGrid::new(planes, w, h);
    for y in 0..h {
      for x in 0..w {
        let c = cells[y * w + x];
        if c != 0 {
          g.set(x, y, c);
        }
      }
    }
    g
  }
  pub fn to_cells(&self, w: usize, h: usize) -> Vec<u8> {
    let mut v = vec![0u8; w * h];
    for y in 0..h {
      for x in 0..w {
        v[y * w + x] = self.get(x, y);
      }
    }
    v
  }
  pub fn is_zero(&self) -> bool {
    self.data.iter().all(|&w| w == 0)
  }
}

/// How one generation is computed.
#[derive(Clone)]
enum Kernel {
  /// outer-totalistic, two states: count the neighbours with an adder tree, decode B and S
  Life { birth: u16, survive: u16, nbhd: Neighbourhood },
  /// the compiled 367-gate circuit
  Hutton32b,
  /// the compiled 597-gate circuit of Hutton32
  Hutton32,
  Network(Arc<Network>),
  /// cell by cell through the rule's table or tree
  Cells(Rule),
}

impl Kernel {
  fn of(rule: &Rule) -> Kernel {
    if let Some((b, s)) = rule.life_like_masks() {
      if rule.neighbourhood() != Neighbourhood::OneDimensional {
        return Kernel::Life { birth: b, survive: s, nbhd: rule.neighbourhood() };
      }
    }
    match rule.network() {
      Some(n) if n.kind() == NetworkKind::Hutton32b => Kernel::Hutton32b,
      Some(n) if n.kind() == NetworkKind::Hutton32 => Kernel::Hutton32,
      Some(n) => Kernel::Network(Arc::new(n.clone())),
      None => Kernel::Cells(rule.clone()),
    }
  }
}

/// The brute-force engine of a rule on the CPU.
#[derive(Clone)]
pub struct Leaf {
  even: Kernel,
  odd: Option<Kernel>,
  planes: usize,
  nbhd: Neighbourhood,
  /// four words at a time with AVX2 (Stage 5) where the CPU has it
  simd: bool,
}

// ---- four words at a time (AVX2) --------------------------------------------------------------------

/// The neighbour counts of a Life-like rule on any word type.
#[inline(always)]
fn count_g<W: Lanes>(xs: &[W]) -> [W; 4] {
  let mut x = [W::ZERO; 8];
  x[..xs.len()].copy_from_slice(xs);
  let [a, b, c, d, e, f, g, h] = x;
  let t1 = a ^ b;
  let s1 = t1 ^ c;
  let c1 = (a & b) | (t1 & c);
  let t2 = d ^ e;
  let s2 = t2 ^ f;
  let c2 = (d & e) | (t2 & f);
  let s3 = g ^ h;
  let c3 = g & h;
  let t4 = s1 ^ s2;
  let b0 = t4 ^ s3;
  let c4 = (s1 & s2) | (t4 & s3);
  let t5 = c1 ^ c2;
  let s5 = t5 ^ c3;
  let c5 = (c1 & c2) | (t5 & c3);
  let b1 = s5 ^ c4;
  let c6 = s5 & c4;
  let b2 = c5 ^ c6;
  let b3 = c5 & c6;
  [b0, b1, b2, b3]
}

#[inline(always)]
fn in_set_g<W: Lanes>(cnt: &[W; 4], mask: u16) -> W {
  let mut acc = W::ZERO;
  let mut m = mask;
  while m != 0 {
    let k = m.trailing_zeros();
    m &= m - 1;
    let lit = |bit: u32, v: W| if k >> bit & 1 == 1 { v } else { !v };
    acc = acc | (lit(0, cnt[0]) & lit(1, cnt[1]) & lit(2, cnt[2]) & lit(3, cnt[3]));
  }
  acc
}

#[inline(always)]
fn life_g<W: Lanes>(w: &[W; 9], birth: u16, survive: u16, nbhd: Neighbourhood) -> W {
  let cnt = match nbhd {
    Neighbourhood::Moore => count_g(&[w[IN], w[IS], w[IE], w[IW], w[INE], w[INW], w[ISE], w[ISW]]),
    Neighbourhood::Hexagonal => count_g(&[w[IN], w[IE], w[ISE], w[IS], w[IW], w[INW]]),
    _ => count_g(&[w[IN], w[IE], w[IS], w[IW]]),
  };
  let c = w[IC];
  (c & in_set_g(&cnt, survive)) | (!c & in_set_g(&cnt, birth))
}

/// Four consecutive words of plane `k`, row `y`, from word `j`, as lanes per position.
#[inline(always)]
fn around4(g: &PlaneGrid, k: usize, y: usize, j: usize) -> [X4; 9] {
  let a = [around(g, k, y, j), around(g, k, y, j + 1), around(g, k, y, j + 2), around(g, k, y, j + 3)];
  std::array::from_fn(|p| X4([a[0][p], a[1][p], a[2][p], a[3][p]]))
}

#[inline(always)]
fn rows_x4(kernel: &Kernel, g: &PlaneGrid, out: &mut PlaneGrid) {
  let (h, ww) = (g.h, g.ww);
  for y in 0..h {
    let mut j = 0;
    while j + 4 <= ww {
      match kernel {
        Kernel::Life { birth, survive, nbhd } => {
          let o = life_g(&around4(g, 0, y, j), *birth, *survive, *nbhd);
          out.data[y * ww + j..y * ww + j + 4].copy_from_slice(&o.0);
        }
        Kernel::Hutton32b | Kernel::Hutton32 => {
          let (mut c, mut n, mut e, mut s, mut w) = ([X4::ZERO; 5], [X4::ZERO; 5], [X4::ZERO; 5], [X4::ZERO; 5], [X4::ZERO; 5]);
          let mut any = X4::ZERO;
          for k in 0..5 {
            let a = around4(g, k, y, j);
            c[k] = a[IC];
            n[k] = a[IN];
            e[k] = a[IE];
            s[k] = a[IS];
            w[k] = a[IW];
            any = any | a[IC] | a[IN] | a[IE] | a[IS] | a[IW];
          }
          let o = if any.is_zero() {
            [X4::ZERO; 5]
          } else if matches!(kernel, Kernel::Hutton32b) {
            circuits::hutton32b_step(c, n, e, s, w)
          } else {
            circuits::hutton32_step(c, n, e, s, w)
          };
          for k in 0..5 {
            out.data[(k * h + y) * ww + j..(k * h + y) * ww + j + 4].copy_from_slice(&o[k].0);
          }
        }
        _ => unreachable!(),
      }
      j += 4;
    }
    // the last words one at a time, on the same generic code
    for jj in j..ww {
      match kernel {
        Kernel::Life { birth, survive, nbhd } => out.data[y * ww + jj] = life_g(&around(g, 0, y, jj), *birth, *survive, *nbhd),
        _ => {
          let (mut c, mut n, mut e, mut s, mut w) = ([0u64; 5], [0u64; 5], [0u64; 5], [0u64; 5], [0u64; 5]);
          for k in 0..5 {
            let a = around(g, k, y, jj);
            c[k] = a[IC];
            n[k] = a[IN];
            e[k] = a[IE];
            s[k] = a[IS];
            w[k] = a[IW];
          }
          let o = if matches!(kernel, Kernel::Hutton32b) { circuits::hutton32b_step(c, n, e, s, w) } else { circuits::hutton32_step(c, n, e, s, w) };
          for k in 0..5 {
            out.data[(k * h + y) * ww + jj] = o[k];
          }
        }
      }
    }
  }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rows_avx2(kernel: &Kernel, g: &PlaneGrid, out: &mut PlaneGrid) {
  rows_x4(kernel, g, out)
}

/// Words of plane `k` at the nine positions around word `j` of row `y` (order of [`Pos::ALL`]).
#[inline(always)]
fn around(g: &PlaneGrid, k: usize, y: usize, j: usize) -> [u64; 9] {
  let ww = g.ww;
  let row = |yy: isize| -> (u64, u64, u64) {
    if yy < 0 || yy as usize >= g.h {
      return (0, 0, 0);
    }
    let b = (k * g.h + yy as usize) * ww;
    let c = g.data[b + j];
    let l = if j > 0 { g.data[b + j - 1] } else { 0 };
    let r = if j + 1 < ww { g.data[b + j + 1] } else { 0 };
    (l, c, r)
  };
  let (ul, u, ur) = row(y as isize - 1);
  let (ml, m, mr) = row(y as isize);
  let (dl, d, dr) = row(y as isize + 1);
  // east neighbour of cell i is cell i + 1: shift towards bit 0, pull the next word's bit 0 in on top
  let east = |c: u64, r: u64| (c >> 1) | (r << 63);
  let west = |c: u64, l: u64| (c << 1) | (l >> 63);
  [m, u, east(u, ur), east(m, mr), east(d, dr), d, west(d, dl), west(m, ml), west(u, ul)]
}

const IC: usize = 0;
const IN: usize = 1;
const INE: usize = 2;
const IE: usize = 3;
const ISE: usize = 4;
const IS: usize = 5;
const ISW: usize = 6;
const IW: usize = 7;
const INW: usize = 8;

fn pos_index(p: Pos) -> usize {
  match p {
    Pos::C => IC,
    Pos::N => IN,
    Pos::NE => INE,
    Pos::E => IE,
    Pos::SE => ISE,
    Pos::S => IS,
    Pos::SW => ISW,
    Pos::W => IW,
    Pos::NW => INW,
  }
}

/// Count `xs` one-bit inputs per lane into four bits (at most 8 inputs).
#[inline(always)]
fn count(xs: &[u64]) -> [u64; 4] {
  let mut x = [0u64; 8];
  x[..xs.len()].copy_from_slice(xs);
  let [a, b, c, d, e, f, g, h] = x;
  // carry-save adder tree, as gpu-life's
  let t1 = a ^ b;
  let s1 = t1 ^ c;
  let c1 = (a & b) | (t1 & c);
  let t2 = d ^ e;
  let s2 = t2 ^ f;
  let c2 = (d & e) | (t2 & f);
  let s3 = g ^ h;
  let c3 = g & h;
  let t4 = s1 ^ s2;
  let b0 = t4 ^ s3;
  let c4 = (s1 & s2) | (t4 & s3);
  let t5 = c1 ^ c2;
  let s5 = t5 ^ c3;
  let c5 = (c1 & c2) | (t5 & c3);
  let b1 = s5 ^ c4;
  let c6 = s5 & c4;
  let b2 = c5 ^ c6;
  let b3 = c5 & c6;
  [b0, b1, b2, b3]
}

#[inline(always)]
fn in_set(cnt: &[u64; 4], mask: u16) -> u64 {
  let mut acc = 0u64;
  let mut m = mask;
  while m != 0 {
    let k = m.trailing_zeros();
    m &= m - 1;
    let lit = |bit: u32, v: u64| if k >> bit & 1 == 1 { v } else { !v };
    acc |= lit(0, cnt[0]) & lit(1, cnt[1]) & lit(2, cnt[2]) & lit(3, cnt[3]);
  }
  acc
}

impl Leaf {
  pub fn new(rule: &Rule) -> Leaf {
    Leaf {
      even: Kernel::of(rule),
      odd: if rule.alternates() { Some(Kernel::of(rule.for_generation(1))) } else { None },
      planes: rule.planes() as usize,
      nbhd: rule.neighbourhood(),
      simd: simd::has_avx2(),
    }
  }
  /// Use (or not) the AVX2 path where the CPU has it.
  pub fn with_simd(mut self, on: bool) -> Leaf {
    self.simd = on && simd::has_avx2();
    self
  }
  pub fn simd(&self) -> bool {
    self.simd
  }
  pub fn planes(&self) -> usize {
    self.planes
  }

  /// Step `g` by `gens` generations, the first one starting at generation `start` (which picks the
  /// rule of an alternating B0 rule). `tmp` is scratch of the same shape.
  pub fn step(&self, g: &mut PlaneGrid, tmp: &mut PlaneGrid, gens: usize, start: u64) {
    if tmp.data.len() != g.data.len() || tmp.ww != g.ww || tmp.h != g.h {
      *tmp = PlaneGrid::new(g.planes, g.width(), g.h);
    }
    for i in 0..gens {
      let kernel = match (&self.odd, (start + i as u64) % 2) {
        (Some(o), 1) => o,
        _ => &self.even,
      };
      self.step_once(kernel, g, tmp);
      std::mem::swap(g, tmp);
    }
  }

  fn step_once(&self, kernel: &Kernel, g: &PlaneGrid, out: &mut PlaneGrid) {
    #[cfg(target_arch = "x86_64")]
    if self.simd && g.ww >= 4 && matches!(kernel, Kernel::Life { .. } | Kernel::Hutton32b | Kernel::Hutton32) {
      // SAFETY: `simd` is set only when the CPU reports AVX2
      unsafe { rows_avx2(kernel, g, out) };
      return;
    }
    let (h, ww, p) = (g.h, g.ww, g.planes);
    match kernel {
      Kernel::Life { birth, survive, nbhd } => {
        for y in 0..h {
          for j in 0..ww {
            let w = around(g, 0, y, j);
            let cnt = match nbhd {
              Neighbourhood::Moore => count(&[w[IN], w[IS], w[IE], w[IW], w[INE], w[INW], w[ISE], w[ISW]]),
              Neighbourhood::Hexagonal => count(&[w[IN], w[IE], w[ISE], w[IS], w[IW], w[INW]]),
              _ => count(&[w[IN], w[IE], w[IS], w[IW]]),
            };
            let c = w[IC];
            let o = (c & in_set(&cnt, *survive)) | (!c & in_set(&cnt, *birth));
            out.data[y * ww + j] = o;
          }
        }
      }
      Kernel::Hutton32b | Kernel::Hutton32 => {
        let f = if matches!(kernel, Kernel::Hutton32b) { hutton32b_planes::step_planes } else { hutton32_planes::step_planes };
        for y in 0..h {
          for j in 0..ww {
            let (mut c, mut n, mut e, mut s, mut w) = ([0u64; 5], [0u64; 5], [0u64; 5], [0u64; 5], [0u64; 5]);
            let mut any = 0u64;
            for k in 0..5 {
              let a = around(g, k, y, j);
              c[k] = a[IC];
              n[k] = a[IN];
              e[k] = a[IE];
              s[k] = a[IS];
              w[k] = a[IW];
              any |= a[IC] | a[IN] | a[IE] | a[IS] | a[IW];
            }
            // an empty neighbourhood stays empty (no B0): skip the circuit
            let o = if any == 0 { [0u64; 5] } else { f(c, n, e, s, w) };
            for k in 0..5 {
              out.data[(k * h + y) * ww + j] = o[k];
            }
          }
        }
      }
      Kernel::Network(net) => {
        let mut scratch = Vec::new();
        let mut o = vec![0u64; p];
        let mut words = vec![[0u64; 9]; p];
        for y in 0..h {
          for j in 0..ww {
            let mut any = 0u64;
            for (k, wk) in words.iter_mut().enumerate() {
              *wk = around(g, k, y, j);
              any |= wk.iter().fold(0, |a, &x| a | x);
            }
            if any == 0 {
              // an empty neighbourhood stays empty (no rule of the engines has B0)
              for k in 0..p {
                out.data[(k * h + y) * ww + j] = 0;
              }
              continue;
            }
            net.eval64(&|pos, bit| words[bit as usize][pos_index(pos)], &mut scratch, &mut o);
            for k in 0..p {
              out.data[(k * h + y) * ww + j] = o[k];
            }
          }
        }
      }
      Kernel::Cells(rule) => {
        let fields: Vec<usize> = self.nbhd.fields().iter().map(|&q| pos_index(q)).collect();
        let mut cells = vec![0u8; fields.len()];
        let mut words = vec![[0u64; 9]; p];
        for y in 0..h {
          for j in 0..ww {
            let mut any = 0u64;
            for (k, wk) in words.iter_mut().enumerate() {
              *wk = around(g, k, y, j);
              any |= wk.iter().fold(0, |a, &x| a | x);
            }
            let mut o = vec![0u64; p];
            for lane in 0..64 {
              if any == 0 {
                break;
              }
              for (fi, &pi) in fields.iter().enumerate() {
                let mut st = 0u8;
                for (k, wk) in words.iter().enumerate() {
                  st |= (((wk[pi] >> lane) & 1) as u8) << k;
                }
                cells[fi] = st;
              }
              let r = rule.next(&cells);
              for (k, ok) in o.iter_mut().enumerate() {
                *ok |= (((r >> k) & 1) as u64) << lane;
              }
            }
            for k in 0..p {
              out.data[(k * h + y) * ww + j] = o[k];
            }
          }
        }
      }
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::io::Grid;

  /// Per-cell stepping through the table, zero boundary: the oracle of every engine.
  pub fn reference_step(rule: &Rule, g: &Grid, gen: u64) -> Grid {
    let r = rule.for_generation(gen);
    let nb = r.neighbourhood();
    let mut out = Grid::new(g.w, g.h);
    let mut cells = vec![0u8; nb.cells()];
    for y in 0..g.h {
      for x in 0..g.w {
        for (i, p) in nb.fields().iter().enumerate() {
          let (dx, dy) = p.offset();
          let (xx, yy) = (x as isize + dx as isize, y as isize + dy as isize);
          cells[i] = if xx < 0 || yy < 0 || xx >= g.w as isize || yy >= g.h as isize { 0 } else { g.get(xx as usize, yy as usize) };
        }
        out.set(x, y, r.next(&cells));
      }
    }
    out
  }

  fn random_grid(w: usize, h: usize, states: u32, seed: u64) -> Grid {
    let mut g = Grid::new(w, h);
    let mut s = seed;
    for c in g.cells.iter_mut() {
      s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
      *c = ((s >> 33) % states as u64) as u8;
    }
    g
  }

  #[test]
  fn kernels_equal_the_tables() {
    let rules = [Rule::named("B3/S23").unwrap(), Rule::named("B2/S34H").unwrap(), Rule::named("B13/S0V").unwrap(), Rule::named("Hutton32b").unwrap(), Rule::named("Hutton32").unwrap()];
    for rule in rules {
      let g = random_grid(150, 37, rule.states(), 7);
      let leaf = Leaf::new(&rule);
      let mut pg = PlaneGrid::from_cells(rule.planes() as usize, g.w, g.h, &g.cells);
      let mut tmp = pg.clone();
      leaf.step(&mut pg, &mut tmp, 3, 0);
      // the plane grid is a whole number of words wide: compare on the grid padded to its width
      let mut padded = g.padded(0, 0, pg.width(), g.h);
      for gen in 0..3 {
        padded = reference_step(&rule, &padded, gen);
      }
      let got = Grid { w: pg.width(), h: pg.h, cells: pg.to_cells(pg.width(), pg.h) };
      assert_eq!(got, padded, "rule {}", rule.name());
    }
  }

  #[test]
  fn simd_equals_scalar() {
    for name in ["B3/S23", "B2/S34H", "B13/S0V", "B03/S23", "Hutton32b", "Hutton32"] {
      let rule = Rule::named(name).unwrap();
      let g = random_grid(600, 41, rule.states(), 11);
      let mut a = PlaneGrid::from_cells(rule.planes() as usize, g.w, g.h, &g.cells);
      let mut b = a.clone();
      let mut t = a.clone();
      Leaf::new(&rule).with_simd(true).step(&mut a, &mut t, 5, 1);
      Leaf::new(&rule).with_simd(false).step(&mut b, &mut t, 5, 1);
      assert_eq!(a, b, "{name}");
      let mut padded = g.padded(0, 0, a.width(), g.h);
      for gen in 1..6 {
        padded = reference_step(&rule, &padded, gen);
      }
      assert_eq!(a.to_cells(a.width(), a.h), padded.cells, "{name} against per-cell stepping");
    }
  }

  #[test]
  fn alternating_b0() {
    let rule = Rule::named("B03/S23").unwrap();
    let leaf = Leaf::new(&rule);
    let g = random_grid(64, 20, 2, 3);
    let mut pg = PlaneGrid::from_cells(1, g.w, g.h, &g.cells);
    let mut tmp = pg.clone();
    leaf.step(&mut pg, &mut tmp, 4, 0);
    let mut want = g;
    for gen in 0..4 {
      want = reference_step(&rule, &want, gen);
    }
    assert_eq!(pg.to_cells(64, 20), want.cells);
  }

  #[test]
  fn tiles_in_and_out() {
    let geom = TileGeom::new(3, 5);
    let cells: Vec<u8> = (0..64).map(|i| (i * 5 % 32) as u8).collect();
    let w = geom.pack(&cells);
    let mut g = PlaneGrid::new(5, 200, 30);
    g.blit_tile(&geom, &w, 60, 7);
    let mut back = vec![0u32; geom.words()];
    g.extract_tile(&geom, 60, 7, &mut back);
    assert_eq!(back, w);
    assert_eq!(g.get(60 + 3, 7 + 2), cells[2 * 8 + 3]);
  }
}
