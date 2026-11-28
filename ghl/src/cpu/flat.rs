//! The positional sweep on the CPU (`research/src/flat.rs`): the universe as a dense array of
//! level-`lt` tile IDs; one sweep maps every 2×2 window of IDs to the ID of the centred tile
//! `2^(lt-1)` generations later, so the lattice shifts by half a tile, and two phases (A, B)
//! alternate. It is the hit path of the GPU (`gpu::sweep`) on the host, and that path's oracle.

use super::hl::Engine;
use crate::io::Grid;

#[derive(Clone, Debug)]
pub struct Flat {
  pub lt: usize,
  pub nx: usize,
  pub ny: usize,
  /// phase A: `nx * ny` IDs, tile `(i, j)` covers cells `[i S, (i + 1) S)`
  pub a_ids: Vec<u32>,
  /// phase B: `(nx + 1) * (ny + 1)` IDs, tile `(i, j)` covers cells `[(i - 1) S + S/2, i S + S/2)`
  pub b_ids: Vec<u32>,
  pub in_b: bool,
  pub sweeps: u64,
}

/// What one sweep asked for.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct SweepStats {
  pub windows: u64,
  pub unique_windows: u64,
  pub top_misses: u64,
}

impl Flat {
  /// Positional grid from an ID array at level `lt` (`n * n`), with `margin` empty tiles around.
  pub fn from_ids(e: &mut Engine, ids: &[u32], n: usize, lt: usize, margin: usize) -> Flat {
    Flat::from_ids_rect(e, ids, n, n, lt, margin)
  }

  /// Positional grid from a `w * h` ID array at level `lt`, with `margin` empty tiles around.
  pub fn from_ids_rect(e: &mut Engine, ids: &[u32], w: usize, h: usize, lt: usize, margin: usize) -> Flat {
    let z = e.empty(lt);
    let (nx, ny) = (w + 2 * margin, h + 2 * margin);
    let mut a_ids = vec![z; nx * ny];
    for y in 0..h {
      a_ids[(y + margin) * nx + margin..(y + margin) * nx + margin + w].copy_from_slice(&ids[y * w..(y + 1) * w]);
    }
    Flat { lt, nx, ny, a_ids, b_ids: Vec::new(), in_b: false, sweeps: 0 }
  }

  /// Build from a dense grid placed with `margin` empty tiles on every side.
  pub fn from_grid(e: &mut Engine, g: &Grid, lt: usize, margin: usize) -> Flat {
    let s = 1usize << lt;
    let tx = g.w.div_ceil(s) + 2 * margin;
    let ty = g.h.div_ceil(s) + 2 * margin;
    let padded = g.padded(margin * s, margin * s, tx * s, ty * s);
    let z = e.empty(lt);
    let mut a_ids = vec![z; tx * ty];
    for j in 0..ty {
      for i in 0..tx {
        let sub = padded.crop(i * s, j * s, (i + 1) * s, (j + 1) * s);
        if sub.cells.iter().any(|&c| c != 0) {
          a_ids[j * tx + i] = if lt == e.a {
            e.tile_from_cells(&sub.cells)
          } else {
            let (id, l) = e.from_grid(&sub);
            assert_eq!(l, lt);
            id
          };
        }
      }
    }
    Flat { lt, nx: tx, ny: ty, a_ids, b_ids: Vec::new(), in_b: false, sweeps: 0 }
  }

  pub fn generation(&self) -> u64 {
    self.sweeps << (self.lt - 1)
  }

  /// The keys of the next sweep as level-`(lt + 1)` nodes, in the order the sweep writes them.
  pub fn keys(&self, e: &mut Engine) -> Vec<u32> {
    let (lt, nx, ny) = (self.lt, self.nx, self.ny);
    let z = e.empty(lt);
    if !self.in_b {
      let (bx, by) = (nx + 1, ny + 1);
      let mut keys = Vec::with_capacity(bx * by);
      let a = &self.a_ids;
      let at = |i: isize, j: isize| -> u32 {
        if i < 0 || j < 0 || i >= nx as isize || j >= ny as isize {
          z
        } else {
          a[j as usize * nx + i as usize]
        }
      };
      for j in 0..by as isize {
        for i in 0..bx as isize {
          let k = [at(i - 1, j - 1), at(i, j - 1), at(i - 1, j), at(i, j)];
          keys.push(e.intern(lt + 1, k));
        }
      }
      keys
    } else {
      let bx = nx + 1;
      let mut keys = Vec::with_capacity(nx * ny);
      for j in 0..ny {
        for i in 0..nx {
          let b = &self.b_ids;
          let k = [b[j * bx + i], b[j * bx + i + 1], b[(j + 1) * bx + i], b[(j + 1) * bx + i + 1]];
          keys.push(e.intern(lt + 1, k));
        }
      }
      keys
    }
  }

  /// One sweep = `2^(lt-1)` generations.
  pub fn sweep(&mut self, e: &mut Engine) -> SweepStats {
    let lt = self.lt;
    let before = e.levels.get(lt + 1).map(|l| l.stats.misses).unwrap_or(0);
    let keys = self.keys(e);
    let mut q = keys.clone();
    e.solve(lt + 1, &mut q);
    let out: Vec<u32> = keys.iter().map(|&k| e.result(lt + 1, k)).collect();
    let stats = SweepStats { windows: keys.len() as u64, unique_windows: q.len() as u64, top_misses: e.levels[lt + 1].stats.misses - before };
    if !self.in_b {
      self.b_ids = out;
    } else {
      self.a_ids = out;
    }
    self.in_b = !self.in_b;
    self.sweeps += 1;
    stats
  }

  /// True if the outermost ring of tiles is empty in the current phase.
  pub fn border_clear(&self, e: &mut Engine) -> bool {
    let z = e.empty(self.lt);
    let (ids, w, h) = if self.in_b { (&self.b_ids, self.nx + 1, self.ny + 1) } else { (&self.a_ids, self.nx, self.ny) };
    (0..w).all(|i| ids[i] == z && ids[(h - 1) * w + i] == z) && (0..h).all(|j| ids[j * w] == z && ids[j * w + w - 1] == z)
  }

  /// Dense expansion of phase A (panics in phase B).
  pub fn to_grid(&self, e: &Engine) -> Grid {
    assert!(!self.in_b, "expand only in phase A");
    let s = 1usize << self.lt;
    let mut g = Grid::new(self.nx * s, self.ny * s);
    for j in 0..self.ny {
      for i in 0..self.nx {
        let id = self.a_ids[j * self.nx + i];
        let t = if self.lt == e.a { Grid { w: s, h: s, cells: e.tile_cells(id) } } else { e.expand(id, self.lt) };
        g.blit(&t, i * s, j * s);
      }
    }
    g
  }
}
