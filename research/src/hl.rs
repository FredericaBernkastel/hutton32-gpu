//! Reference implementation (CPU) of the algorithm proposed in `paper.md`:
//! hash-consed tiles + memoised results, evaluated *level-synchronously* (one batch of distinct
//! nodes per level and wave) instead of depth-first. It is HashLife with a different schedule,
//! so every counter below is the size of a batch a GPU would process in one dispatch.
//!
//! Conventions
//!   * level `l` node  = square of side 2^l cells.
//!   * levels `0..a`   do not exist; a level-`a` node is a *base tile* stored as raw cells.
//!   * level `l > a`   node = four level-(l-1) ids `[nw, ne, sw, se]`.
//!   * `RESULT(node at level l)` = centred level-(l-1) node, advanced `2^min(l-2, k)` generations,
//!     where `k = step_log2` (Golly's `ngens`). Requires `k >= a`.

use crate::pattern::Grid;
use crate::sim::advance_window;
use rustc_hash::FxHashMap;

pub const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Default, Debug)]
pub struct LevelStats {
  /// find-or-insert probes into this level's node table
  pub finds: u64,
  /// probes that created a new node
  pub inserts: u64,
  /// RESULT requests after per-wave dedup (= batch elements probed for a result)
  pub requests: u64,
  /// RESULT requests that were not memoised (= batch elements that needed work)
  pub misses: u64,
  /// number of batches (waves) issued at this level
  pub waves: u64,
  /// number of waves that had at least one miss
  pub waves_with_miss: u64,
  /// largest miss batch seen
  pub max_miss_batch: u64,
}

pub struct Level {
  pub map: FxHashMap<[u32; 4], u32>,
  pub nodes: Vec<[u32; 4]>,
  pub res: Vec<u32>,
  pub stats: LevelStats,
}

impl Level {
  fn new() -> Self {
    Level { map: FxHashMap::default(), nodes: Vec::new(), res: Vec::new(), stats: LevelStats::default() }
  }
}

pub struct Engine<'a> {
  pub a: usize,
  pub step_log2: usize,
  lut: &'a crate::ca::Rule,
  pub tile_side: usize,
  tile_area: usize,
  pub tile_data: Vec<u8>,
  tile_map: FxHashMap<Box<[u8]>, u32>,
  /// `levels[l]` for `l > a`; lower indices are unused placeholders.
  pub levels: Vec<Level>,
  empty: Vec<u32>,
  pub brute_calls: u64,
  /// cell evaluations spent in brute-force windows (including the discarded halo)
  pub brute_cell_evals: u64,
  /// base-tile find-or-insert calls / new base tiles
  pub tile_finds: u64,
  pub tile_inserts: u64,
  /// base tiles gathered to expand nodes into cells for brute-forced windows
  pub expand_tiles: u64,
  /// brute-forced windows per window level, and the cell evaluations they cost when the whole
  /// window is stepped for every generation (no pyramid shrinking), as a GPU atlas would
  pub brute_win: Vec<u64>,
  pub brute_win_evals: Vec<u64>,
  /// When set, `solve` records every wave as `[level, distinct requests, misses, parent]`, where
  /// `parent` is the index of the wave whose misses asked for this one (`NONE` at the top), and
  /// the misses of every floor wave as node ids in `floor_misses` (for `overview`).
  pub wave_log: Option<Vec<[u32; 4]>>,
  pub floor_misses: Vec<Vec<u32>>,
  wave_parent: u32,
  buf: Vec<u8>,
  tmp: Vec<u8>,
}

/// What happened to one RESULT request (reported to a [`FloorPolicy`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Outcome {
  Hit,
  /// missed; resolved through thirteen requests one level down
  Descend,
  /// missed; the window was brute-forced
  Brute,
}

/// Decides, per place, how deep a missed window may be subdivided before it is brute-forced.
/// `floor(..) = f` means: a missed window of level `l <= f + 1` is brute-forced, a larger one is
/// subdivided. Coordinates are the window centre in cells.
pub trait FloorPolicy {
  fn floor(&mut self, cx: i64, cy: i64, l: usize) -> usize;
  fn record(&mut self, cx: i64, cy: i64, l: usize, o: Outcome);
}

impl<'a> Engine<'a> {
  pub fn new(lut: &'a crate::ca::Rule, a: usize, step_log2: usize) -> Self {
    assert!(a >= 1 && step_log2 >= a, "need step_log2 >= a >= 1");
    let tile_side = 1usize << a;
    let mut e = Engine {
      a,
      step_log2,
      lut,
      tile_side,
      tile_area: tile_side * tile_side,
      tile_data: Vec::new(),
      tile_map: FxHashMap::default(),
      levels: Vec::new(),
      empty: Vec::new(),
      brute_calls: 0,
      brute_cell_evals: 0,
      tile_finds: 0,
      tile_inserts: 0,
      expand_tiles: 0,
      brute_win: Vec::new(),
      brute_win_evals: Vec::new(),
      wave_log: None,
      floor_misses: Vec::new(),
      wave_parent: NONE,
      buf: vec![0; 4 * tile_side * tile_side],
      tmp: vec![0; 4 * tile_side * tile_side],
    };
    for _ in 0..=a {
      e.levels.push(Level::new());
    }
    let z = e.tile(&vec![0u8; e.tile_area]);
    e.empty = vec![NONE; a];
    e.empty.push(z);
    e
  }

  fn level(&mut self, l: usize) -> &mut Level {
    while self.levels.len() <= l {
      self.levels.push(Level::new());
    }
    &mut self.levels[l]
  }

  pub fn num_tiles(&self) -> usize {
    self.tile_data.len() / self.tile_area
  }
  pub fn tile_cells(&self, id: u32) -> &[u8] {
    &self.tile_data[id as usize * self.tile_area..(id as usize + 1) * self.tile_area]
  }

  /// Intern a base tile by content.
  pub fn tile(&mut self, cells: &[u8]) -> u32 {
    debug_assert_eq!(cells.len(), self.tile_area);
    self.tile_finds += 1;
    if let Some(&id) = self.tile_map.get(cells) {
      return id;
    }
    self.tile_inserts += 1;
    let id = self.num_tiles() as u32;
    self.tile_data.extend_from_slice(cells);
    self.tile_map.insert(cells.into(), id);
    id
  }

  /// Find-or-insert a node at level `l > a`.
  pub fn intern(&mut self, l: usize, ch: [u32; 4]) -> u32 {
    debug_assert!(l > self.a);
    let lv = self.level(l);
    lv.stats.finds += 1;
    if let Some(&id) = lv.map.get(&ch) {
      return id;
    }
    let id = lv.nodes.len() as u32;
    lv.nodes.push(ch);
    lv.res.push(NONE);
    lv.map.insert(ch, id);
    lv.stats.inserts += 1;
    id
  }

  pub fn empty(&mut self, l: usize) -> u32 {
    while self.empty.len() <= l {
      let below = *self.empty.last().unwrap();
      let lv = self.empty.len();
      let id = self.intern(lv, [below; 4]);
      self.empty.push(id);
    }
    self.empty[l]
  }

  #[inline]
  fn ch(&self, l: usize, id: u32) -> [u32; 4] {
    self.levels[l].nodes[id as usize]
  }
  #[inline]
  pub fn result(&self, l: usize, id: u32) -> u32 {
    self.levels[l].res[id as usize]
  }

  /// Brute-force the RESULT of a level-(a+1) node.
  fn brute(&mut self, id: u32) -> u32 {
    let (s, a) = (self.tile_side, self.a);
    let w = 2 * s;
    let ch = self.ch(a + 1, id);
    for (q, &t) in ch.iter().enumerate() {
      let (ox, oy) = ((q & 1) * s, (q >> 1) * s);
      let src = t as usize * self.tile_area;
      for y in 0..s {
        self.buf[(oy + y) * w + ox..(oy + y) * w + ox + s].copy_from_slice(&self.tile_data[src + y * s..src + (y + 1) * s]);
      }
    }
    let gens = 1usize << (a - 1).min(self.step_log2);
    advance_window(self.lut, &mut self.buf, &mut self.tmp, w, gens);
    self.brute_calls += 1;
    for g in 0..gens {
      let side = (w - 2 * (g + 1)) as u64;
      self.brute_cell_evals += side * side;
    }
    let mut out = vec![0u8; self.tile_area];
    for y in 0..s {
      out[y * s..(y + 1) * s].copy_from_slice(&self.buf[(s / 2 + y) * w + s / 2..(s / 2 + y) * w + s / 2 + s]);
    }
    self.tile(&out)
  }

  /// Ensure RESULT is memoised for every node id in `q` (level `l`). `q` is deduplicated in place.
  /// This is one *wave*: everything in the miss list is independent and could run in one dispatch.
  pub fn solve(&mut self, l: usize, q: &mut Vec<u32>) {
    q.sort_unstable();
    q.dedup();
    if q.is_empty() {
      return;
    }
    let miss: Vec<u32> = q.iter().copied().filter(|&id| self.levels[l].res[id as usize] == NONE).collect();
    {
      let st = &mut self.levels[l].stats;
      st.waves += 1;
      st.requests += q.len() as u64;
      st.misses += miss.len() as u64;
      st.max_miss_batch = st.max_miss_batch.max(miss.len() as u64);
      if !miss.is_empty() {
        st.waves_with_miss += 1;
      }
    }
    let parent = self.wave_parent;
    if let Some(log) = self.wave_log.as_mut() {
      log.push([l as u32, q.len() as u32, miss.len() as u32, parent]);
      self.wave_parent = log.len() as u32 - 1;
    }
    if miss.is_empty() {
      self.wave_parent = parent;
      return;
    }
    let a = self.a;
    if l == a + 1 {
      if self.wave_log.is_some() {
        self.floor_misses.push(miss.clone());
      }
      for &m in &miss {
        let r = self.brute(m);
        self.levels[l].res[m as usize] = r;
      }
      self.wave_parent = parent;
      return;
    }
    let full = l - 2 <= self.step_log2;
    if full {
      // phase 1: nine overlapping level-(l-1) nodes per miss
      let mut nine: Vec<[u32; 9]> = Vec::with_capacity(miss.len());
      let mut s1: Vec<u32> = Vec::with_capacity(miss.len() * 9);
      for &m in &miss {
        let [nw, ne, sw, se] = self.ch(l, m);
        let (cnw, cne, csw, cse) = (self.ch(l - 1, nw), self.ch(l - 1, ne), self.ch(l - 1, sw), self.ch(l - 1, se));
        let n01 = self.intern(l - 1, [cnw[1], cne[0], cnw[3], cne[2]]);
        let n10 = self.intern(l - 1, [cnw[2], cnw[3], csw[0], csw[1]]);
        let n11 = self.intern(l - 1, [cnw[3], cne[2], csw[1], cse[0]]);
        let n12 = self.intern(l - 1, [cne[2], cne[3], cse[0], cse[1]]);
        let n21 = self.intern(l - 1, [csw[1], cse[0], csw[3], cse[2]]);
        let n = [nw, n01, ne, n10, n11, n12, sw, n21, se];
        s1.extend_from_slice(&n);
        nine.push(n);
      }
      self.solve(l - 1, &mut s1);
      // phase 2: four level-(l-1) nodes built from the nine results
      let mut four: Vec<[u32; 4]> = Vec::with_capacity(miss.len());
      let mut s2: Vec<u32> = Vec::with_capacity(miss.len() * 4);
      for n in &nine {
        let r: Vec<u32> = n.iter().map(|&id| self.result(l - 1, id)).collect();
        // (for l-2 == a the results are base-tile ids; level a+1 nodes are made of 4 base tiles)
        let f = [
          self.intern(l - 1, [r[0], r[1], r[3], r[4]]),
          self.intern(l - 1, [r[1], r[2], r[4], r[5]]),
          self.intern(l - 1, [r[3], r[4], r[6], r[7]]),
          self.intern(l - 1, [r[4], r[5], r[7], r[8]]),
        ];
        s2.extend_from_slice(&f);
        four.push(f);
      }
      self.solve(l - 1, &mut s2);
      for (i, &m) in miss.iter().enumerate() {
        let f = four[i];
        let r = [self.result(l - 1, f[0]), self.result(l - 1, f[1]), self.result(l - 1, f[2]), self.result(l - 1, f[3])];
        let id = self.intern(l - 1, r);
        self.levels[l].res[m as usize] = id;
      }
    } else {
      // step is smaller than this level's natural step: nine centres (no time advance), then
      // four level-(l-1) results. Needs great-grandchildren, i.e. l-3 >= a.
      debug_assert!(l >= a + 3);
      let mut four: Vec<[u32; 4]> = Vec::with_capacity(miss.len());
      let mut s: Vec<u32> = Vec::with_capacity(miss.len() * 4);
      for &m in &miss {
        let c = self.ch(l, m);
        // 4x4 array of level-(l-2) nodes
        let mut g = [[0u32; 4]; 4];
        for q in 0..4 {
          let cc = self.ch(l - 1, c[q]);
          let (ox, oy) = ((q & 1) * 2, (q >> 1) * 2);
          g[oy][ox] = cc[0];
          g[oy][ox + 1] = cc[1];
          g[oy + 1][ox] = cc[2];
          g[oy + 1][ox + 1] = cc[3];
        }
        // 8x8 array of level-(l-3) nodes would be needed for centres; take them lazily
        let mut centre = [[0u32; 3]; 3];
        for cy in 0..3 {
          for cx in 0..3 {
            // centre of the 2x2 block of level-(l-2) nodes at (cx, cy)
            let (p, q2, r, t) = (g[cy][cx], g[cy][cx + 1], g[cy + 1][cx], g[cy + 1][cx + 1]);
            let (pc, qc, rc, tc) = (self.ch(l - 2, p), self.ch(l - 2, q2), self.ch(l - 2, r), self.ch(l - 2, t));
            centre[cy][cx] = self.intern(l - 2, [pc[3], qc[2], rc[1], tc[0]]);
          }
        }
        let f = [
          self.intern(l - 1, [centre[0][0], centre[0][1], centre[1][0], centre[1][1]]),
          self.intern(l - 1, [centre[0][1], centre[0][2], centre[1][1], centre[1][2]]),
          self.intern(l - 1, [centre[1][0], centre[1][1], centre[2][0], centre[2][1]]),
          self.intern(l - 1, [centre[1][1], centre[1][2], centre[2][1], centre[2][2]]),
        ];
        s.extend_from_slice(&f);
        four.push(f);
      }
      self.solve(l - 1, &mut s);
      for (i, &m) in miss.iter().enumerate() {
        let f = four[i];
        let r = [self.result(l - 1, f[0]), self.result(l - 1, f[1]), self.result(l - 1, f[2]), self.result(l - 1, f[3])];
        let id = self.intern(l - 1, r);
        self.levels[l].res[m as usize] = id;
      }
    }
    self.wave_parent = parent;
  }

  // ---- construction ----------------------------------------------------------------------

  /// Build a node covering `g` placed at the origin of a 2^level square (zero padded).
  /// Returns `(id, level)` with `level >= a + 1`.
  pub fn from_grid(&mut self, g: &Grid) -> (u32, usize) {
    let mut level = self.a + 1;
    while (1usize << level) < g.w.max(g.h) {
      level += 1;
    }
    let id = self.build_from_grid(g, 0, 0, level);
    (id, level)
  }

  fn build_from_grid(&mut self, g: &Grid, x0: usize, y0: usize, l: usize) -> u32 {
    let side = 1usize << l;
    if x0 >= g.w || y0 >= g.h {
      return self.empty(l);
    }
    if l == self.a {
      let mut cells = vec![0u8; self.tile_area];
      for y in 0..side.min(g.h - y0) {
        let n = side.min(g.w - x0);
        cells[y * side..y * side + n].copy_from_slice(&g.cells[(y0 + y) * g.w + x0..(y0 + y) * g.w + x0 + n]);
      }
      return self.tile(&cells);
    }
    let h = side / 2;
    let ch = [
      self.build_from_grid(g, x0, y0, l - 1),
      self.build_from_grid(g, x0 + h, y0, l - 1),
      self.build_from_grid(g, x0, y0 + h, l - 1),
      self.build_from_grid(g, x0 + h, y0 + h, l - 1),
    ];
    self.intern(l, ch)
  }

  /// Positional grid of level-`lt` ids covering the node `(id, l)`; side = 2^(l-lt).
  pub fn ids_at_level(&self, id: u32, l: usize, lt: usize) -> Vec<u32> {
    let n = 1usize << (l - lt);
    let mut out = vec![0u32; n * n];
    self.fill_ids(id, l, lt, 0, 0, n, &mut out);
    out
  }
  fn fill_ids(&self, id: u32, l: usize, lt: usize, x: usize, y: usize, n: usize, out: &mut [u32]) {
    if l == lt {
      out[y * n + x] = id;
      return;
    }
    let h = 1usize << (l - 1 - lt);
    let c = self.ch(l, id);
    self.fill_ids(c[0], l - 1, lt, x, y, n, out);
    self.fill_ids(c[1], l - 1, lt, x + h, y, n, out);
    self.fill_ids(c[2], l - 1, lt, x, y + h, n, out);
    self.fill_ids(c[3], l - 1, lt, x + h, y + h, n, out);
  }

  /// Dense expansion (small universes only).
  pub fn expand(&self, id: u32, l: usize) -> Grid {
    let side = 1usize << l;
    let mut g = Grid::new(side, side);
    self.expand_into(id, l, 0, 0, &mut g);
    g
  }
  fn expand_into(&self, id: u32, l: usize, x: usize, y: usize, g: &mut Grid) {
    if l == self.a {
      let s = self.tile_side;
      let src = self.tile_cells(id);
      for yy in 0..s {
        g.cells[(y + yy) * g.w + x..(y + yy) * g.w + x + s].copy_from_slice(&src[yy * s..(yy + 1) * s]);
      }
      return;
    }
    if self.empty.get(l) == Some(&id) {
      return;
    }
    let h = 1usize << (l - 1);
    let c = self.ch(l, id);
    self.expand_into(c[0], l - 1, x, y, g);
    self.expand_into(c[1], l - 1, x + h, y, g);
    self.expand_into(c[2], l - 1, x, y + h, g);
    self.expand_into(c[3], l - 1, x + h, y + h, g);
  }

  /// Visit every non-zero cell: `f(x, y, state)`. Skips empty subtrees.
  pub fn for_each_cell(&self, id: u32, l: usize, x: i64, y: i64, f: &mut impl FnMut(i64, i64, u8)) {
    if self.empty.get(l) == Some(&id) {
      return;
    }
    if l == self.a {
      let s = self.tile_side;
      let src = self.tile_cells(id);
      for yy in 0..s {
        for xx in 0..s {
          let c = src[yy * s + xx];
          if c != 0 {
            f(x + xx as i64, y + yy as i64, c);
          }
        }
      }
      return;
    }
    let h = 1i64 << (l - 1);
    let c = self.ch(l, id);
    self.for_each_cell(c[0], l - 1, x, y, f);
    self.for_each_cell(c[1], l - 1, x + h, y, f);
    self.for_each_cell(c[2], l - 1, x, y + h, f);
    self.for_each_cell(c[3], l - 1, x + h, y + h, f);
  }

  // ---- whole-universe stepping (Golly-style root handling) --------------------------------

  /// Embed `n` (level l) in the centre of a level l+1 node.
  pub fn pushroot(&mut self, n: u32, l: usize) -> u32 {
    let z = self.empty(l - 1);
    let c = self.ch(l, n);
    let nw = self.intern(l, [z, z, z, c[0]]);
    let ne = self.intern(l, [z, z, c[1], z]);
    let sw = self.intern(l, [z, c[2], z, z]);
    let se = self.intern(l, [c[3], z, z, z]);
    self.intern(l + 1, [nw, ne, sw, se])
  }

  /// Advance the universe `(root, level)` by `2^step_log2` generations.
  /// Returns the new `(root, level)` and the shift of the origin: the new node's top-left corner
  /// is at `old_origin - shift` (so callers can track absolute coordinates).
  pub fn step_root(&mut self, mut root: u32, mut level: usize) -> (u32, usize, i64) {
    let mut shift = 0i64;
    // pad until the pattern can neither escape nor is the level too small for the step
    let mut pads = 0;
    while pads < 2 || level < self.step_log2 + 3 || level < self.a + 3 {
      shift += 1i64 << (level - 1);
      root = self.pushroot(root, level);
      level += 1;
      pads += 1;
    }
    let mut q = vec![root];
    self.solve(level, &mut q);
    let res = self.result(level, root);
    // result is the centred level-1 node: its origin is offset by 2^(level-2) inside `root`
    shift -= 1i64 << (level - 2);
    let (r, l, s) = self.popzeros(res, level - 1);
    (r, l, shift - s)
  }

  /// Strip empty borders; returns the inward shift of the origin.
  pub fn popzeros(&mut self, mut n: u32, mut l: usize) -> (u32, usize, i64) {
    let mut shift = 0i64;
    while l > self.a + 2 {
      let z = self.empty(l - 2);
      let c = self.ch(l, n);
      let (a, b, cc, d) = (self.ch(l - 1, c[0]), self.ch(l - 1, c[1]), self.ch(l - 1, c[2]), self.ch(l - 1, c[3]));
      let ring_empty = a[0] == z && a[1] == z && a[2] == z
        && b[0] == z && b[1] == z && b[3] == z
        && cc[0] == z && cc[2] == z && cc[3] == z
        && d[1] == z && d[2] == z && d[3] == z;
      if !ring_empty {
        break;
      }
      n = self.intern(l - 1, [a[3], b[2], cc[1], d[0]]);
      shift += 1i64 << (l - 2);
      l -= 1;
    }
    (n, l, shift)
  }

  pub fn total_nodes(&self) -> usize {
    self.levels.iter().map(|l| l.nodes.len()).sum::<usize>() + self.num_tiles()
  }

  // ---- adaptive subdivision: depth-first, position-aware, per-place floors -----------------

  /// Brute-force the RESULT of a window of any level `l >= a + 1`: expand it to cells, step it
  /// `2^min(l-2, k)` generations and hash-cons the centre.
  pub fn brute_window(&mut self, l: usize, id: u32) -> u32 {
    if self.brute_win.len() <= l {
      self.brute_win.resize(l + 1, 0);
      self.brute_win_evals.resize(l + 1, 0);
    }
    let gens = 1usize << (l - 2).min(self.step_log2);
    let side = 1usize << l;
    self.brute_win[l] += 1;
    self.brute_win_evals[l] += (side * side * gens) as u64;
    if l == self.a + 1 {
      self.expand_tiles += 4;
      return self.brute(id);
    }
    self.expand_tiles += 1u64 << (2 * (l - self.a));
    let mut g = self.expand(id, l);
    let mut tmp = vec![0u8; g.cells.len()];
    advance_window(self.lut, &mut g.cells, &mut tmp, side, gens);
    let (h, q) = (side / 2, side / 4);
    let c = g.crop(q, q, q + h, q + h);
    self.build_from_grid(&c, 0, 0, l - 1)
  }

  /// RESULT of the level-`l` node `id` whose top-left cell is `(x0, y0)`, computed depth-first.
  /// On a miss the policy decides between subdividing and brute force. Same results as `solve`.
  pub fn result_at(&mut self, l: usize, id: u32, x0: i64, y0: i64, pol: &mut dyn FloorPolicy) -> u32 {
    if id == self.empty(l) {
      return self.empty(l - 1);
    }
    let half = 1i64 << (l - 1);
    let (cx, cy) = (x0 + half, y0 + half);
    self.levels[l].stats.requests += 1;
    let r = self.levels[l].res[id as usize];
    if r != NONE {
      pol.record(cx, cy, l, Outcome::Hit);
      return r;
    }
    self.levels[l].stats.misses += 1;
    let a = self.a;
    let full = l - 2 <= self.step_log2;
    let f = pol.floor(cx, cy, l);
    let res = if l == a + 1 || l <= f + 1 || (!full && l < a + 3) {
      pol.record(cx, cy, l, Outcome::Brute);
      self.brute_window(l, id)
    } else if full {
      pol.record(cx, cy, l, Outcome::Descend);
      let u = 1i64 << (l - 2);
      let [nw, ne, sw, se] = self.ch(l, id);
      let (cnw, cne, csw, cse) = (self.ch(l - 1, nw), self.ch(l - 1, ne), self.ch(l - 1, sw), self.ch(l - 1, se));
      let n01 = self.intern(l - 1, [cnw[1], cne[0], cnw[3], cne[2]]);
      let n10 = self.intern(l - 1, [cnw[2], cnw[3], csw[0], csw[1]]);
      let n11 = self.intern(l - 1, [cnw[3], cne[2], csw[1], cse[0]]);
      let n12 = self.intern(l - 1, [cne[2], cne[3], cse[0], cse[1]]);
      let n21 = self.intern(l - 1, [csw[1], cse[0], csw[3], cse[2]]);
      let nine = [nw, n01, ne, n10, n11, n12, sw, n21, se];
      let mut r = [0u32; 9];
      for i in 0..3 {
        for j in 0..3 {
          r[i * 3 + j] = self.result_at(l - 1, nine[i * 3 + j], x0 + j as i64 * u, y0 + i as i64 * u, pol);
        }
      }
      let mut s = [0u32; 4];
      for i in 0..2 {
        for j in 0..2 {
          let k = i * 3 + j;
          let node = self.intern(l - 1, [r[k], r[k + 1], r[k + 3], r[k + 4]]);
          s[i * 2 + j] = self.result_at(l - 1, node, x0 + u / 2 + j as i64 * u, y0 + u / 2 + i as i64 * u, pol);
        }
      }
      self.intern(l - 1, s)
    } else {
      // step smaller than this level's natural step: nine centres, then four windows
      pol.record(cx, cy, l, Outcome::Descend);
      let u = 1i64 << (l - 2);
      let c = self.ch(l, id);
      let mut g = [[0u32; 4]; 4];
      for q in 0..4 {
        let cc = self.ch(l - 1, c[q]);
        let (ox, oy) = ((q & 1) * 2, (q >> 1) * 2);
        g[oy][ox] = cc[0];
        g[oy][ox + 1] = cc[1];
        g[oy + 1][ox] = cc[2];
        g[oy + 1][ox + 1] = cc[3];
      }
      let mut centre = [[0u32; 3]; 3];
      for cyi in 0..3 {
        for cxi in 0..3 {
          let (p, q2, r, t) = (g[cyi][cxi], g[cyi][cxi + 1], g[cyi + 1][cxi], g[cyi + 1][cxi + 1]);
          let (pc, qc, rc, tc) = (self.ch(l - 2, p), self.ch(l - 2, q2), self.ch(l - 2, r), self.ch(l - 2, t));
          centre[cyi][cxi] = self.intern(l - 2, [pc[3], qc[2], rc[1], tc[0]]);
        }
      }
      let mut s = [0u32; 4];
      for i in 0..2 {
        for j in 0..2 {
          let node = self.intern(l - 1, [centre[i][j], centre[i][j + 1], centre[i + 1][j], centre[i + 1][j + 1]]);
          s[i * 2 + j] = self.result_at(l - 1, node, x0 + u / 2 + j as i64 * u, y0 + u / 2 + i as i64 * u, pol);
        }
      }
      self.intern(l - 1, s)
    };
    self.levels[l].res[id as usize] = res;
    res
  }

  /// Like `step_root`, but through `result_at`. `(ox, oy)` is the absolute position of the root's
  /// top-left cell; returns the new root, level and position. The root is never shrunk below
  /// `min_level`, so that nodes of level `min_level - 2` stay aligned to an absolute grid.
  pub fn step_root_at(
    &mut self,
    mut root: u32,
    mut level: usize,
    mut ox: i64,
    mut oy: i64,
    min_level: usize,
    pol: &mut dyn FloorPolicy,
  ) -> (u32, usize, i64, i64) {
    let mut pads = 0;
    while pads < 2 || level < self.step_log2 + 3 || level < self.a + 3 || level < min_level + 1 {
      let sh = 1i64 << (level - 1);
      ox -= sh;
      oy -= sh;
      root = self.pushroot(root, level);
      level += 1;
      pads += 1;
    }
    let res = self.result_at(level, root, ox, oy, pol);
    let c = 1i64 << (level - 2);
    let (mut n, mut l) = (res, level - 1);
    let (mut x, mut y) = (ox + c, oy + c);
    // strip empty borders, but keep the alignment
    while l > min_level.max(self.a + 2) {
      let z = self.empty(l - 2);
      let ch = self.ch(l, n);
      let (p, q, r, t) = (self.ch(l - 1, ch[0]), self.ch(l - 1, ch[1]), self.ch(l - 1, ch[2]), self.ch(l - 1, ch[3]));
      let ring_empty = p[0] == z && p[1] == z && p[2] == z
        && q[0] == z && q[1] == z && q[3] == z
        && r[0] == z && r[2] == z && r[3] == z
        && t[1] == z && t[2] == z && t[3] == z;
      if !ring_empty {
        break;
      }
      n = self.intern(l - 1, [p[3], q[2], r[1], t[0]]);
      let s = 1i64 << (l - 2);
      x += s;
      y += s;
      l -= 1;
    }
    (n, l, x, y)
  }

  /// Absolute top-left corners of the non-empty level-`lt` nodes under `(id, l)` at `(x0, y0)`.
  pub fn nonempty_at(&mut self, id: u32, l: usize, x0: i64, y0: i64, lt: usize, out: &mut Vec<(i64, i64)>) {
    if id == self.empty(l) {
      return;
    }
    if l <= lt {
      out.push((x0, y0));
      return;
    }
    let h = 1i64 << (l - 1);
    let c = self.ch(l, id);
    self.nonempty_at(c[0], l - 1, x0, y0, lt, out);
    self.nonempty_at(c[1], l - 1, x0 + h, y0, lt, out);
    self.nonempty_at(c[2], l - 1, x0, y0 + h, lt, out);
    self.nonempty_at(c[3], l - 1, x0 + h, y0 + h, lt, out);
  }

  /// Number of non-empty nodes of level `lt` under `(id, l)`, counted by position.
  pub fn count_nonempty(&mut self, id: u32, l: usize, lt: usize) -> u64 {
    if id == self.empty(l) {
      return 0;
    }
    if l <= lt {
      return 1;
    }
    let c = self.ch(l, id);
    c.iter().map(|&x| self.count_nonempty(x, l - 1, lt)).sum()
  }
}

/// Order-independent content hash + population + bounding box of a set of cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fingerprint {
  pub population: u64,
  pub w: i64,
  pub h: i64,
  pub hash: u64,
}

fn mix(x: i64, y: i64, s: u8) -> u64 {
  let mut v = (x as u64).wrapping_mul(0x9E3779B97F4A7C15) ^ (y as u64).wrapping_mul(0xC2B2AE3D27D4EB4F) ^ (s as u64) << 56;
  v ^= v >> 29;
  v = v.wrapping_mul(0xBF58476D1CE4E5B9);
  v ^= v >> 32;
  v = v.wrapping_mul(0x94D049BB133111EB);
  v ^ v >> 29
}

/// Fingerprint relative to the bounding box's top-left corner (two passes over the cells).
pub fn fingerprint(mut visit: impl FnMut(&mut dyn FnMut(i64, i64, u8))) -> Fingerprint {
  let (mut x0, mut y0, mut x1, mut y1, mut pop) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN, 0u64);
  visit(&mut |x, y, _| {
    x0 = x0.min(x);
    y0 = y0.min(y);
    x1 = x1.max(x);
    y1 = y1.max(y);
    pop += 1;
  });
  if pop == 0 {
    return Fingerprint { population: 0, w: 0, h: 0, hash: 0 };
  }
  let mut hash = 0u64;
  visit(&mut |x, y, s| hash = hash.wrapping_add(mix(x - x0, y - y0, s)));
  Fingerprint { population: pop, w: x1 - x0 + 1, h: y1 - y0 + 1, hash }
}
