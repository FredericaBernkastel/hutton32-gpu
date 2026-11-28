//! The host recursion: HashLife with base tiles stored as bit-planes, evaluated level-synchronously
//! (paper Algorithm 2): one batch of distinct nodes per level and wave, so every counter below is
//! the size of a batch the GPU processes in one dispatch. A port of the harness engine
//! (`research/src/hl.rs`) that interns tiles and nodes in the same order, so its IDs equal the
//! harness's, with the brute force of a wave run on an atlas by the rule's network.
//!
//! Conventions:
//!
//! * a level-`l` node is a square of side `2^l` cells; a level-`a` node is a base tile stored as
//!   bit-planes ([`TileGeom`]), a level-`l > a` node is four level-`(l-1)` IDs `[nw, ne, sw, se]`;
//!   every level has its own ID space, and tile 0 is the empty tile;
//! * `RESULT(node at level l)` is the centred level-`(l-1)` node advanced `2^min(l-2, k)`
//!   generations, where `k` is the step's log2 ([`Engine::set_step`]);
//! * a missed window of a level up to the floor (`a + 1` by default) is brute-forced.

use rustc_hash::FxHashMap;

use super::leaf::{Leaf, PlaneGrid};
use crate::{
  io::{Grid, Macrocell, McNode},
  rule::Rule,
  tile::{tile_hash, TileGeom, NONE},
};

/// Counters of one level of the recursion.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct LevelStats {
  /// RESULT requests before per-wave deduplication
  pub windows: u64,
  /// find-or-insert probes into this level's node table
  pub finds: u64,
  /// probes that created a new node
  pub inserts: u64,
  /// RESULT requests after per-wave deduplication (batch elements probed for a result)
  pub requests: u64,
  /// requests that were not memoized (batch elements that needed work)
  pub misses: u64,
  /// batches (waves) issued at this level
  pub waves: u64,
  /// waves with at least one miss
  pub waves_with_miss: u64,
  /// the largest miss batch
  pub max_miss_batch: u64,
}

impl LevelStats {
  /// `self - earlier`, with the maximum kept.
  pub fn since(&self, earlier: &LevelStats) -> LevelStats {
    LevelStats {
      windows: self.windows - earlier.windows,
      finds: self.finds - earlier.finds,
      inserts: self.inserts - earlier.inserts,
      requests: self.requests - earlier.requests,
      misses: self.misses - earlier.misses,
      waves: self.waves - earlier.waves,
      waves_with_miss: self.waves_with_miss - earlier.waves_with_miss,
      max_miss_batch: self.max_miss_batch,
    }
  }
}

/// The nodes of one level, their memoized results and populations.
#[derive(Default)]
pub struct Level {
  pub map: FxHashMap<[u32; 4], u32>,
  pub nodes: Vec<[u32; 4]>,
  pub res: Vec<u32>,
  pub pop: Vec<u64>,
  pub stats: LevelStats,
}

/// Counters of the engine as a whole.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct EngineStats {
  /// floor windows (level `a + 1`) brute-forced
  pub brute_calls: u64,
  /// cell evaluations of those windows when each generation steps only the cells still exact (the
  /// harness's count)
  pub brute_cell_evals: u64,
  pub tile_finds: u64,
  pub tile_inserts: u64,
  /// base tiles gathered to expand windows into cells
  pub expand_tiles: u64,
  /// cell-updates of the atlases stepped, whole atlases every generation
  pub atlas_cell_updates: u64,
  pub atlases: u64,
  /// region-level windows whose result came from a block of plain cells
  pub block_windows: u64,
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
  /// The same request with the window's node, for policies that track who computed what.
  fn record_node(&mut self, _l: usize, _id: u32, _cx: i64, _cy: i64, _o: Outcome) {}
  /// Regions stepped together as plain cells before the step: their cells after the step.
  fn block(&self) -> Option<&Block> {
    None
  }
}

/// Plain cells stepped as one block over a mask of regions: the cells of `grid` (corner at
/// `(x0, y0)`) after the step are exact inside the regions of `regions` (regions of side `2^rl`, the
/// ones in plain cells, which share their halos); the recursion takes the result of a window of
/// level `level` whose centre lies inside them from here instead of computing it.
#[derive(Clone, Debug)]
pub struct Block {
  pub grid: PlaneGrid,
  pub x0: i64,
  pub y0: i64,
  pub level: usize,
  pub rl: usize,
  pub regions: rustc_hash::FxHashSet<(i64, i64)>,
}

impl Block {
  /// Whether every cell of the square `[x, x + s)²` lies in a region of the block.
  pub fn covers(&self, x: i64, y: i64, s: i64) -> bool {
    let rl = self.rl;
    let (rx0, ry0, rx1, ry1) = (x >> rl, y >> rl, (x + s - 1) >> rl, (y + s - 1) >> rl);
    (ry0..=ry1).all(|ry| (rx0..=rx1).all(|rx| self.regions.contains(&(rx, ry))))
  }
}

/// A floor policy that brute-forces at level `f + 1` everywhere.
pub struct UniformFloor(pub usize);
impl FloorPolicy for UniformFloor {
  fn floor(&mut self, _: i64, _: i64, _: usize) -> usize {
    self.0
  }
  fn record(&mut self, _: i64, _: i64, _: usize, _: Outcome) {}
}

/// A side of a bounding box.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Side {
  Left,
  Right,
  Top,
  Bottom,
}

/// The brute-forced windows of one wave, kept when waves are logged.
#[derive(Clone, Debug)]
pub struct BruteWave {
  /// index of the wave in the wave log
  pub wave: u32,
  pub level: u32,
  /// node IDs of the missed windows, in the order their results were interned
  pub ids: Vec<u32>,
}

/// The host recursion.
pub struct Engine {
  pub a: usize,
  pub geom: TileGeom,
  step_log2: usize,
  /// misses of windows up to this level are brute-forced (`a + 1`: the rule's floor)
  floor: usize,
  rule: Rule,
  leaf: Leaf,
  /// the tile store: tile `t` is `tiles[t * W..(t + 1) * W]`
  tiles: Vec<u32>,
  tile_pop: Vec<u32>,
  tile_slots: Vec<u32>,
  /// `levels[l]` for `l > a`; lower indices are unused
  pub levels: Vec<Level>,
  empty: Vec<u32>,
  pub stats: EngineStats,
  /// brute-forced windows per level, and their cell-updates (whole windows every generation)
  pub brute_win: Vec<u64>,
  pub brute_win_evals: Vec<u64>,
  /// When set, `solve` records every wave as `[level, distinct requests, misses, parent]`, where
  /// `parent` is the index of the wave whose misses asked for this one (`NONE` at the top), and
  /// the windows of every brute-force wave in `brute_waves`.
  pub wave_log: Option<Vec<[u32; 4]>>,
  pub brute_waves: Vec<BruteWave>,
  wave_parent: u32,
  atlas: PlaneGrid,
  atlas_tmp: PlaneGrid,
}

/// Cells in the largest atlas stepped at once.
const ATLAS_CELLS: usize = 1 << 22;
/// The widest atlas row, in cells.
const ATLAS_WIDTH: usize = 1024;

impl Engine {
  /// An engine for `rule` with base tiles of side `2^a` and steps of `2^k` generations.
  pub fn new(rule: &Rule, a: usize, k: usize) -> Engine {
    assert!((1..=6).contains(&a), "base tiles of 2x2 to 64x64 cells");
    assert!(!rule.alternates() || a >= 2, "an alternating B0 rule needs base tiles of at least 4x4 cells");
    let geom = TileGeom::new(a as u32, rule.planes());
    let mut e = Engine {
      a,
      geom,
      step_log2: k,
      floor: a + 1,
      rule: rule.clone(),
      leaf: Leaf::new(rule),
      tiles: Vec::new(),
      tile_pop: Vec::new(),
      tile_slots: vec![NONE; 1024],
      levels: Vec::new(),
      empty: Vec::new(),
      stats: EngineStats::default(),
      brute_win: Vec::new(),
      brute_win_evals: Vec::new(),
      wave_log: None,
      brute_waves: Vec::new(),
      wave_parent: NONE,
      atlas: PlaneGrid::new(rule.planes() as usize, 64, 1),
      atlas_tmp: PlaneGrid::new(rule.planes() as usize, 64, 1),
    };
    for _ in 0..=a {
      e.levels.push(Level::default());
    }
    let z = e.tile(&vec![0u32; geom.words()]);
    debug_assert_eq!(z, 0);
    e.empty = vec![NONE; a];
    e.empty.push(z);
    e
  }

  pub fn rule(&self) -> &Rule {
    &self.rule
  }
  pub fn step_log2(&self) -> usize {
    self.step_log2
  }
  /// Change the step to `2^k` generations. Results whose meaning changes (levels above
  /// `min(old, new) + 2`) are dropped; the others stay valid.
  pub fn set_step(&mut self, k: usize) {
    if k == self.step_log2 {
      return;
    }
    let m = k.min(self.step_log2);
    for l in (m + 3).max(self.a + 1)..self.levels.len() {
      self.levels[l].res.fill(NONE);
    }
    self.step_log2 = k;
  }
  pub fn floor(&self) -> usize {
    self.floor
  }
  /// Brute-force missed windows up to level `f` (at least `a + 1`). Memoized results stay valid:
  /// the floor changes how a result is computed, not what it is.
  pub fn set_floor(&mut self, f: usize) {
    self.floor = f.max(self.a + 1);
  }
  /// Drop every memoized result.
  pub fn clear_results(&mut self) {
    for lv in self.levels.iter_mut() {
      lv.res.fill(NONE);
    }
  }
  pub fn reset_stats(&mut self) {
    for lv in self.levels.iter_mut() {
      lv.stats = LevelStats::default();
    }
    self.stats = EngineStats::default();
    self.brute_win.clear();
    self.brute_win_evals.clear();
  }

  fn level(&mut self, l: usize) -> &mut Level {
    while self.levels.len() <= l {
      self.levels.push(Level::default());
    }
    &mut self.levels[l]
  }

  // ---- tiles ----------------------------------------------------------------------------------

  pub fn num_tiles(&self) -> usize {
    self.tile_pop.len()
  }
  /// The bit-plane words of tile `id`.
  #[inline]
  pub fn tile_words(&self, id: u32) -> &[u32] {
    let w = self.geom.words();
    &self.tiles[id as usize * w..(id as usize + 1) * w]
  }
  /// The cells of tile `id`, row-major.
  pub fn tile_cells(&self, id: u32) -> Vec<u8> {
    self.geom.unpack(self.tile_words(id))
  }
  pub fn tile_population(&self, id: u32) -> u32 {
    self.tile_pop[id as usize]
  }
  /// The whole tile store, `geom.words()` words per tile.
  pub fn tile_store(&self) -> &[u32] {
    &self.tiles
  }

  /// Intern a base tile by content.
  pub fn tile(&mut self, words: &[u32]) -> u32 {
    let w = self.geom.words();
    debug_assert_eq!(words.len(), w);
    self.stats.tile_finds += 1;
    let mask = self.tile_slots.len() - 1;
    let mut h = tile_hash(words) as usize & mask;
    loop {
      let id = self.tile_slots[h];
      if id == NONE {
        break;
      }
      if &self.tiles[id as usize * w..(id as usize + 1) * w] == words {
        return id;
      }
      h = (h + 1) & mask;
    }
    let id = self.num_tiles() as u32;
    self.tiles.extend_from_slice(words);
    let wpp = self.geom.wpp;
    let mut pop = 0u32;
    for j in 0..wpp {
      let mut or = 0u32;
      for k in 0..self.geom.planes as usize {
        or |= words[k * wpp + j];
      }
      pop += or.count_ones();
    }
    self.tile_pop.push(pop);
    self.tile_slots[h] = id;
    self.stats.tile_inserts += 1;
    if self.num_tiles() * 2 > self.tile_slots.len() {
      self.rehash_tiles(self.tile_slots.len() * 2);
    }
    id
  }

  fn rehash_tiles(&mut self, slots: usize) {
    let w = self.geom.words();
    self.tile_slots = vec![NONE; slots];
    let mask = slots - 1;
    for id in 0..self.num_tiles() {
      let mut h = tile_hash(&self.tiles[id * w..(id + 1) * w]) as usize & mask;
      while self.tile_slots[h] != NONE {
        h = (h + 1) & mask;
      }
      self.tile_slots[h] = id as u32;
    }
  }

  /// Intern a base tile given as `side * side` cells.
  pub fn tile_from_cells(&mut self, cells: &[u8]) -> u32 {
    let words = self.geom.pack(cells);
    self.tile(&words)
  }

  // ---- nodes ----------------------------------------------------------------------------------

  /// Find-or-insert a node at level `l > a`.
  pub fn intern(&mut self, l: usize, ch: [u32; 4]) -> u32 {
    debug_assert!(l > self.a);
    let a = self.a;
    let pop = if l == a + 1 {
      ch.iter().map(|&t| self.tile_pop[t as usize] as u64).sum()
    } else {
      let below = &self.levels[l - 1].pop;
      ch.iter().map(|&n| below[n as usize]).sum()
    };
    let lv = self.level(l);
    lv.stats.finds += 1;
    if let Some(&id) = lv.map.get(&ch) {
      return id;
    }
    let id = lv.nodes.len() as u32;
    lv.nodes.push(ch);
    lv.res.push(NONE);
    lv.pop.push(pop);
    lv.map.insert(ch, id);
    lv.stats.inserts += 1;
    id
  }

  /// The node `(l, ch)` if it exists; never creates one.
  pub fn find(&self, l: usize, ch: [u32; 4]) -> Option<u32> {
    self.levels.get(l)?.map.get(&ch).copied()
  }

  /// The empty node of level `l` (created on first use, as the harness does).
  pub fn empty(&mut self, l: usize) -> u32 {
    while self.empty.len() <= l {
      let below = *self.empty.last().unwrap();
      let lv = self.empty.len();
      let id = self.intern(lv, [below; 4]);
      self.empty.push(id);
    }
    self.empty[l]
  }
  /// The empty node of every level created so far (`NONE` below the base tile).
  pub fn empty_ids(&self) -> &[u32] {
    &self.empty
  }
  pub(crate) fn into_tile_parts(self) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    (self.tiles, self.tile_pop, self.tile_slots)
  }
  pub(crate) fn set_tile_parts(&mut self, tiles: Vec<u32>, pop: Vec<u32>, slots: Vec<u32>) {
    self.tiles = tiles;
    self.tile_pop = pop;
    self.tile_slots = slots;
  }
  pub(crate) fn remap_empty(&mut self, levels: &[Vec<u32>]) {
    for l in self.a + 1..self.empty.len() {
      self.empty[l] = levels[l][self.empty[l] as usize];
    }
  }
  /// Is `id` the empty node of level `l`? Never creates a node.
  #[inline]
  pub fn is_empty(&self, l: usize, id: u32) -> bool {
    if l == self.a {
      return id == 0;
    }
    match self.empty.get(l) {
      Some(&z) => z == id,
      None => self.levels[l].pop[id as usize] == 0,
    }
  }
  #[inline]
  pub fn ch(&self, l: usize, id: u32) -> [u32; 4] {
    self.levels[l].nodes[id as usize]
  }
  #[inline]
  pub fn result(&self, l: usize, id: u32) -> u32 {
    self.levels[l].res[id as usize]
  }
  /// Live cells of node `id` at level `l` (a tile at `l == a`).
  #[inline]
  pub fn population(&self, l: usize, id: u32) -> u64 {
    if l == self.a {
      self.tile_pop[id as usize] as u64
    } else {
      self.levels[l].pop[id as usize]
    }
  }
  pub fn total_nodes(&self) -> usize {
    self.levels.iter().map(|l| l.nodes.len()).sum::<usize>() + self.num_tiles()
  }
  /// Bytes held by tiles, nodes, results and node tables (an estimate for the tables).
  pub fn memory_bytes(&self) -> u64 {
    let tiles = (self.tiles.len() * 4 + self.tile_pop.len() * 4 + self.tile_slots.len() * 4) as u64;
    let nodes: u64 = self.levels.iter().map(|l| (l.nodes.len() * (16 + 4 + 8)) as u64 + (l.map.capacity() * 24) as u64).sum();
    tiles + nodes
  }

  // ---- brute force on an atlas ------------------------------------------------------------------

  /// Copy the cells of node `(l, id)` into `g` with its top-left corner at `(x, y)`.
  fn blit_node(&self, g: &mut PlaneGrid, l: usize, id: u32, x: usize, y: usize) {
    if self.is_empty(l, id) {
      return;
    }
    if l == self.a {
      g.blit_tile(&self.geom, self.tile_words(id), x, y);
      return;
    }
    let h = 1usize << (l - 1);
    let c = self.ch(l, id);
    self.blit_node(g, l - 1, c[0], x, y);
    self.blit_node(g, l - 1, c[1], x + h, y);
    self.blit_node(g, l - 1, c[2], x, y + h);
    self.blit_node(g, l - 1, c[3], x + h, y + h);
  }

  /// Intern the level-`l` node whose cells lie in `g` at `(x, y)`, depth-first as the harness does.
  fn build_from_planes(&mut self, g: &PlaneGrid, x: usize, y: usize, l: usize, scratch: &mut Vec<u32>) -> u32 {
    if l == self.a {
      scratch.resize(self.geom.words(), 0);
      g.extract_tile(&self.geom, x, y, scratch);
      return self.tile(scratch);
    }
    let h = 1usize << (l - 1);
    let ch = [
      self.build_from_planes(g, x, y, l - 1, scratch),
      self.build_from_planes(g, x + h, y, l - 1, scratch),
      self.build_from_planes(g, x, y + h, l - 1, scratch),
      self.build_from_planes(g, x + h, y + h, l - 1, scratch),
    ];
    self.intern(l, ch)
  }

  /// Brute-force the RESULT of every window `(l, id)` of `ids`: expand them side by side into an
  /// atlas, step it `2^min(l-2, k)` generations with the rule's network, intern the centres in the
  /// order of `ids`. Windows share their borders: what leaks in from a neighbour cannot reach a
  /// centre in that many generations.
  pub fn brute_windows(&mut self, l: usize, ids: &[u32]) -> Vec<u32> {
    let a = self.a;
    let side = 1usize << l;
    let gens = 1usize << (l - 2).min(self.step_log2);
    let n = ids.len() as u64;
    if self.brute_win.len() <= l {
      self.brute_win.resize(l + 1, 0);
      self.brute_win_evals.resize(l + 1, 0);
    }
    self.brute_win[l] += n;
    self.brute_win_evals[l] += n * (side * side * gens) as u64;
    self.stats.expand_tiles += n << (2 * (l - a));
    if l == a + 1 {
      self.stats.brute_calls += n;
      let per: u64 = (0..gens).map(|g| ((side - 2 * (g + 1)) as u64).pow(2)).sum();
      self.stats.brute_cell_evals += n * per;
    }
    let per_atlas = (ATLAS_CELLS / (side * side)).max(1);
    let cols_max = (ATLAS_WIDTH / side).max(1);
    let mut out = Vec::with_capacity(ids.len());
    let mut atlas = std::mem::take(&mut self.atlas);
    let mut tmp = std::mem::take(&mut self.atlas_tmp);
    let mut scratch = Vec::new();
    for chunk in ids.chunks(per_atlas) {
      let cols = chunk.len().min(cols_max);
      let rows = chunk.len().div_ceil(cols);
      atlas.reset(self.geom.planes as usize, cols * side, rows * side);
      for (i, &id) in chunk.iter().enumerate() {
        self.blit_node(&mut atlas, l, id, (i % cols) * side, (i / cols) * side);
      }
      self.leaf.step(&mut atlas, &mut tmp, gens, 0);
      self.stats.atlases += 1;
      self.stats.atlas_cell_updates += (atlas.width() * atlas.h * gens) as u64;
      let q = side / 4;
      for i in 0..chunk.len() {
        let (x, y) = ((i % cols) * side + q, (i / cols) * side + q);
        out.push(self.build_from_planes(&atlas, x, y, l - 1, &mut scratch));
      }
    }
    self.atlas = atlas;
    self.atlas_tmp = tmp;
    out
  }

  // ---- the level-synchronous recursion ----------------------------------------------------------

  /// Ensure RESULT is memoized for every node ID in `q` (level `l`). `q` is deduplicated in place.
  /// This is one wave: everything in the miss list is independent and could run in one dispatch.
  pub fn solve(&mut self, l: usize, q: &mut Vec<u32>) {
    let asked = q.len() as u64;
    q.sort_unstable();
    q.dedup();
    if q.is_empty() {
      return;
    }
    self.levels[l].stats.windows += asked;
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
    let mut wave = NONE;
    if let Some(log) = self.wave_log.as_mut() {
      log.push([l as u32, q.len() as u32, miss.len() as u32, parent]);
      wave = log.len() as u32 - 1;
      self.wave_parent = wave;
    }
    if miss.is_empty() {
      self.wave_parent = parent;
      return;
    }
    let a = self.a;
    let full = l - 2 <= self.step_log2;
    if l <= self.floor || (!full && l < a + 3) {
      if self.wave_log.is_some() {
        self.brute_waves.push(BruteWave { wave, level: l as u32, ids: miss.clone() });
      }
      let r = self.brute_windows(l, &miss);
      for (&m, &r) in miss.iter().zip(&r) {
        self.levels[l].res[m as usize] = r;
      }
      self.wave_parent = parent;
      return;
    }
    if full {
      // phase 1: nine overlapping level-(l-1) nodes per miss
      let mut nine: Vec<[u32; 9]> = Vec::with_capacity(miss.len());
      let mut s1: Vec<u32> = Vec::with_capacity(miss.len() * 9);
      for &m in &miss {
        let n = self.nine(l, m);
        s1.extend_from_slice(&n);
        nine.push(n);
      }
      self.solve(l - 1, &mut s1);
      // phase 2: four level-(l-1) nodes built from the nine results
      let mut four: Vec<[u32; 4]> = Vec::with_capacity(miss.len());
      let mut s2: Vec<u32> = Vec::with_capacity(miss.len() * 4);
      for n in &nine {
        let mut r = [0u32; 9];
        for i in 0..9 {
          r[i] = self.result(l - 1, n[i]);
        }
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
      // the step is shorter than this level's natural step: nine centres (no time advance), then
      // four level-(l-1) results
      let mut four: Vec<[u32; 4]> = Vec::with_capacity(miss.len());
      let mut s: Vec<u32> = Vec::with_capacity(miss.len() * 4);
      for &m in &miss {
        let f = self.four_centres(l, m);
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

  /// The nine overlapping level-`(l-1)` nodes of a level-`l` node, row by row.
  fn nine(&mut self, l: usize, m: u32) -> [u32; 9] {
    let [nw, ne, sw, se] = self.ch(l, m);
    let (cnw, cne, csw, cse) = (self.ch(l - 1, nw), self.ch(l - 1, ne), self.ch(l - 1, sw), self.ch(l - 1, se));
    let n01 = self.intern(l - 1, [cnw[1], cne[0], cnw[3], cne[2]]);
    let n10 = self.intern(l - 1, [cnw[2], cnw[3], csw[0], csw[1]]);
    let n11 = self.intern(l - 1, [cnw[3], cne[2], csw[1], cse[0]]);
    let n12 = self.intern(l - 1, [cne[2], cne[3], cse[0], cse[1]]);
    let n21 = self.intern(l - 1, [csw[1], cse[0], csw[3], cse[2]]);
    [nw, n01, ne, n10, n11, n12, sw, n21, se]
  }

  /// The four level-`(l-1)` windows made of the nine centred level-`(l-2)` nodes of a level-`l`
  /// node, for steps shorter than the level's natural step.
  fn four_centres(&mut self, l: usize, m: u32) -> [u32; 4] {
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
    let mut centre = [[0u32; 3]; 3];
    for cy in 0..3 {
      for cx in 0..3 {
        let (p, q2, r, t) = (g[cy][cx], g[cy][cx + 1], g[cy + 1][cx], g[cy + 1][cx + 1]);
        let (pc, qc, rc, tc) = (self.ch(l - 2, p), self.ch(l - 2, q2), self.ch(l - 2, r), self.ch(l - 2, t));
        centre[cy][cx] = self.intern(l - 2, [pc[3], qc[2], rc[1], tc[0]]);
      }
    }
    [
      self.intern(l - 1, [centre[0][0], centre[0][1], centre[1][0], centre[1][1]]),
      self.intern(l - 1, [centre[0][1], centre[0][2], centre[1][1], centre[1][2]]),
      self.intern(l - 1, [centre[1][0], centre[1][1], centre[2][0], centre[2][1]]),
      self.intern(l - 1, [centre[1][1], centre[1][2], centre[2][1], centre[2][2]]),
    ]
  }

  // ---- construction -----------------------------------------------------------------------------

  /// Build a node covering `g` placed at the origin of a `2^level` square (zero padded).
  /// Returns `(id, level)` with `level >= a + 1`.
  pub fn from_grid(&mut self, g: &Grid) -> (u32, usize) {
    let mut level = self.a + 1;
    while (1usize << level) < g.w.max(g.h) {
      level += 1;
    }
    let mut cells = Vec::new();
    let id = self.build_from_grid(g, 0, 0, level, &mut cells);
    (id, level)
  }

  fn build_from_grid(&mut self, g: &Grid, x0: usize, y0: usize, l: usize, cells: &mut Vec<u8>) -> u32 {
    let side = 1usize << l;
    if x0 >= g.w || y0 >= g.h {
      return self.empty(l);
    }
    if l == self.a {
      cells.clear();
      cells.resize(side * side, 0);
      let n = side.min(g.w - x0);
      for y in 0..side.min(g.h - y0) {
        cells[y * side..y * side + n].copy_from_slice(&g.cells[(y0 + y) * g.w + x0..(y0 + y) * g.w + x0 + n]);
      }
      return self.tile_from_cells(cells);
    }
    let h = side / 2;
    let ch = [
      self.build_from_grid(g, x0, y0, l - 1, cells),
      self.build_from_grid(g, x0 + h, y0, l - 1, cells),
      self.build_from_grid(g, x0, y0 + h, l - 1, cells),
      self.build_from_grid(g, x0 + h, y0 + h, l - 1, cells),
    ];
    self.intern(l, ch)
  }

  /// Convert a macrocell quadtree. Returns `(root id, level)` with `level >= a + 1`; the root's
  /// top-left cell is the macrocell's.
  pub fn from_mc(&mut self, mc: &Macrocell) -> (u32, usize) {
    enum Conv {
      Cells(Vec<u8>),
      Id(u32),
    }
    let a = self.a;
    fn raw(e: &mut Engine, cells: Vec<u8>, d: usize) -> Conv {
      use std::cmp::Ordering::*;
      match d.cmp(&e.a) {
        Less => Conv::Cells(cells),
        Equal => Conv::Id(e.tile_from_cells(&cells)),
        Greater => {
          let side = 1usize << d;
          let h = side / 2;
          let mut ids = [0u32; 4];
          for (q, id) in ids.iter_mut().enumerate() {
            let (ox, oy) = ((q & 1) * h, (q >> 1) * h);
            let mut sub = vec![0u8; h * h];
            for y in 0..h {
              sub[y * h..(y + 1) * h].copy_from_slice(&cells[(oy + y) * side + ox..(oy + y) * side + ox + h]);
            }
            let Conv::Id(i) = raw(e, sub, d - 1) else { unreachable!() };
            *id = i;
          }
          Conv::Id(e.intern(d, ids))
        }
      }
    }
    let mut conv: Vec<Conv> = Vec::with_capacity(mc.nodes.len());
    let mut last_level = 0usize;
    for node in &mc.nodes {
      let c = match *node {
        McNode::Leaf8(bits) => {
          last_level = 3;
          raw(self, (0..64).map(|i| (bits >> i & 1) as u8).collect(), 3)
        }
        McNode::Inner(d, ch) => {
          let d = d as usize;
          last_level = d;
          if d <= a {
            let side = 1usize << d;
            let h = side / 2;
            let mut cells = vec![0u8; side * side];
            if d == 1 {
              for q in 0..4 {
                cells[(q >> 1) * 2 + (q & 1)] = ch[q] as u8;
              }
            } else {
              for q in 0..4 {
                if ch[q] == 0 {
                  continue;
                }
                let Conv::Cells(sub) = &conv[ch[q] as usize - 1] else { panic!("level mismatch in macrocell") };
                let (ox, oy) = ((q & 1) * h, (q >> 1) * h);
                for y in 0..h {
                  cells[(oy + y) * side + ox..(oy + y) * side + ox + h].copy_from_slice(&sub[y * h..(y + 1) * h]);
                }
              }
            }
            raw(self, cells, d)
          } else {
            let z = self.empty(d - 1);
            let mut ids = [z; 4];
            for q in 0..4 {
              if ch[q] != 0 {
                match conv[ch[q] as usize - 1] {
                  Conv::Id(id) => ids[q] = id,
                  Conv::Cells(_) => panic!("level mismatch in macrocell"),
                }
              }
            }
            Conv::Id(self.intern(d, ids))
          }
        }
      };
      conv.push(c);
    }
    let mut level = last_level;
    let mut id = match conv.pop().expect("empty macrocell") {
      Conv::Id(id) => id,
      Conv::Cells(cells) => {
        let side = 1usize << level;
        let s = self.geom.side;
        let mut t = vec![0u8; s * s];
        for y in 0..side {
          t[y * s..y * s + side].copy_from_slice(&cells[y * side..(y + 1) * side]);
        }
        level = a;
        self.tile_from_cells(&t)
      }
    };
    while level < a + 1 {
      let z = self.empty(level);
      id = self.intern(level + 1, [id, z, z, z]);
      level += 1;
    }
    (id, level)
  }

  /// Positional grid of level-`lt` IDs covering the node `(id, l)`; side `2^(l-lt)`.
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

  /// Dense expansion (small nodes only).
  pub fn expand(&self, id: u32, l: usize) -> Grid {
    let side = 1usize << l;
    let mut g = Grid::new(side, side);
    self.expand_into(id, l, 0, 0, &mut g);
    g
  }
  fn expand_into(&self, id: u32, l: usize, x: usize, y: usize, g: &mut Grid) {
    if self.is_empty(l, id) {
      return;
    }
    if l == self.a {
      let s = self.geom.side;
      let cells = self.tile_cells(id);
      for yy in 0..s {
        g.cells[(y + yy) * g.w + x..(y + yy) * g.w + x + s].copy_from_slice(&cells[yy * s..(yy + 1) * s]);
      }
      return;
    }
    let h = 1usize << (l - 1);
    let c = self.ch(l, id);
    self.expand_into(c[0], l - 1, x, y, g);
    self.expand_into(c[1], l - 1, x + h, y, g);
    self.expand_into(c[2], l - 1, x, y + h, g);
    self.expand_into(c[3], l - 1, x + h, y + h, g);
  }

  /// Visit every non-zero cell of a tile: `f(x, y, state)`.
  pub fn for_each_tile_cell(&self, id: u32, x: i64, y: i64, f: &mut impl FnMut(i64, i64, u8)) {
    let s = self.geom.side;
    let words = self.tile_words(id);
    for yy in 0..s {
      let mut any = 0u64;
      for k in 0..self.geom.planes as usize {
        any |= self.geom.row(words, k, yy);
      }
      while any != 0 {
        let xx = any.trailing_zeros() as usize;
        any &= any - 1;
        f(x + xx as i64, y + yy as i64, self.geom.cell(words, xx, yy));
      }
    }
  }

  /// Visit every non-zero cell: `f(x, y, state)`. Skips empty subtrees.
  pub fn for_each_cell(&self, id: u32, l: usize, x: i64, y: i64, f: &mut impl FnMut(i64, i64, u8)) {
    if self.is_empty(l, id) {
      return;
    }
    if l == self.a {
      self.for_each_tile_cell(id, x, y, f);
      return;
    }
    let h = 1i64 << (l - 1);
    let c = self.ch(l, id);
    self.for_each_cell(c[0], l - 1, x, y, f);
    self.for_each_cell(c[1], l - 1, x + h, y, f);
    self.for_each_cell(c[2], l - 1, x, y + h, f);
    self.for_each_cell(c[3], l - 1, x + h, y + h, f);
  }

  // ---- whole-universe stepping (Golly-style root handling) ----------------------------------------

  /// Embed `n` (level `l`) in the centre of a level-`(l+1)` node.
  pub fn pushroot(&mut self, n: u32, l: usize) -> u32 {
    let z = self.empty(l - 1);
    let c = self.ch(l, n);
    let nw = self.intern(l, [z, z, z, c[0]]);
    let ne = self.intern(l, [z, z, c[1], z]);
    let sw = self.intern(l, [z, c[2], z, z]);
    let se = self.intern(l, [c[3], z, z, z]);
    self.intern(l + 1, [nw, ne, sw, se])
  }

  /// Advance the universe `(root, level)` by `2^k` generations. Returns the new `(root, level)` and
  /// the shift of the origin: the new node's top-left corner is at `old_origin - shift`.
  pub fn step_root(&mut self, mut root: u32, mut level: usize) -> (u32, usize, i64) {
    let mut shift = 0i64;
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
      let ring_empty = a[0] == z && a[1] == z && a[2] == z && b[0] == z && b[1] == z && b[3] == z && cc[0] == z && cc[2] == z && cc[3] == z && d[1] == z && d[2] == z && d[3] == z;
      if !ring_empty {
        break;
      }
      n = self.intern(l - 1, [a[3], b[2], cc[1], d[0]]);
      shift += 1i64 << (l - 2);
      l -= 1;
    }
    (n, l, shift)
  }

  // ---- adaptive subdivision: depth-first, position-aware, per-place floors --------------------------

  /// Brute-force the RESULT of one window of level `l >= a + 1`.
  pub fn brute_window(&mut self, l: usize, id: u32) -> u32 {
    self.brute_windows(l, &[id])[0]
  }

  /// RESULT of the level-`l` node `id` whose top-left cell is `(x0, y0)`, computed depth-first. On
  /// a miss the policy decides between subdividing and brute force. Same results as `solve`.
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
      pol.record_node(l, id, cx, cy, Outcome::Hit);
      return r;
    }
    self.levels[l].stats.misses += 1;
    let a = self.a;
    let full = l - 2 <= self.step_log2;
    // plain cells stepped as a block: the centre comes from there
    let from_block = match pol.block() {
      Some(b) => {
        let q = 1i64 << (l - 2);
        if l == b.level && b.covers(x0 + q, y0 + q, 2 * q) {
          let (bx, by) = ((x0 + q - b.x0) as usize, (y0 + q - b.y0) as usize);
          let mut scratch = Vec::new();
          Some(self.build_from_planes(&b.grid, bx, by, l - 1, &mut scratch))
        } else {
          None
        }
      }
      None => None,
    };
    if let Some(r) = from_block {
      self.stats.block_windows += 1;
      pol.record(cx, cy, l, Outcome::Brute);
      pol.record_node(l, id, cx, cy, Outcome::Brute);
      self.levels[l].res[id as usize] = r;
      return r;
    }
    let f = pol.floor(cx, cy, l);
    let res = if l == a + 1 || l <= f + 1 || (!full && l < a + 3) {
      pol.record(cx, cy, l, Outcome::Brute);
      pol.record_node(l, id, cx, cy, Outcome::Brute);
      self.brute_window(l, id)
    } else if full {
      pol.record(cx, cy, l, Outcome::Descend);
      pol.record_node(l, id, cx, cy, Outcome::Descend);
      let u = 1i64 << (l - 2);
      let nine = self.nine(l, id);
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
      pol.record(cx, cy, l, Outcome::Descend);
      pol.record_node(l, id, cx, cy, Outcome::Descend);
      let u = 1i64 << (l - 2);
      let f4 = self.four_centres(l, id);
      let mut s = [0u32; 4];
      for i in 0..2 {
        for j in 0..2 {
          s[i * 2 + j] = self.result_at(l - 1, f4[i * 2 + j], x0 + u / 2 + j as i64 * u, y0 + u / 2 + i as i64 * u, pol);
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
  #[allow(clippy::too_many_arguments)]
  pub fn step_root_at(&mut self, mut root: u32, mut level: usize, mut ox: i64, mut oy: i64, min_level: usize, pol: &mut dyn FloorPolicy) -> (u32, usize, i64, i64) {
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
    let (n, l, x, y) = self.strip(res, level - 1, ox + c, oy + c, min_level);
    (n, l, x, y)
  }

  /// Advance `(root, level)` at `(ox, oy)` by `2^k` generations through the level-synchronous
  /// recursion; returns the new root, level and position.
  pub fn step_root_abs(&mut self, mut root: u32, mut level: usize, mut ox: i64, mut oy: i64, min_level: usize) -> (u32, usize, i64, i64) {
    let mut pads = 0;
    while pads < 2 || level < self.step_log2 + 3 || level < self.a + 3 || level < min_level + 1 {
      let sh = 1i64 << (level - 1);
      ox -= sh;
      oy -= sh;
      root = self.pushroot(root, level);
      level += 1;
      pads += 1;
    }
    let mut q = vec![root];
    self.solve(level, &mut q);
    let res = self.result(level, root);
    let c = 1i64 << (level - 2);
    self.strip(res, level - 1, ox + c, oy + c, min_level)
  }

  /// Strip empty borders down to `min_level` (and `a + 2`), keeping track of the position.
  pub fn strip(&mut self, mut n: u32, mut l: usize, mut x: i64, mut y: i64, min_level: usize) -> (u32, usize, i64, i64) {
    while l > min_level.max(self.a + 2) {
      let z = self.empty(l - 2);
      let ch = self.ch(l, n);
      let (p, q, r, t) = (self.ch(l - 1, ch[0]), self.ch(l - 1, ch[1]), self.ch(l - 1, ch[2]), self.ch(l - 1, ch[3]));
      let ring_empty = p[0] == z && p[1] == z && p[2] == z && q[0] == z && q[1] == z && q[3] == z && r[0] == z && r[2] == z && r[3] == z && t[1] == z && t[2] == z && t[3] == z;
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
    if self.is_empty(l, id) {
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
  pub fn count_nonempty(&self, id: u32, l: usize, lt: usize) -> u64 {
    if self.is_empty(l, id) {
      return 0;
    }
    if l <= lt {
      return 1;
    }
    let c = self.ch(l, id);
    c.iter().map(|&x| self.count_nonempty(x, l - 1, lt)).sum()
  }

  /// The cell at `(x, y)` of node `(id, l)`.
  pub fn cell(&self, mut id: u32, mut l: usize, mut x: u64, mut y: u64) -> u8 {
    while l > self.a {
      if self.is_empty(l, id) {
        return 0;
      }
      let h = 1u64 << (l - 1);
      let q = (x >= h) as usize | ((y >= h) as usize) << 1;
      id = self.ch(l, id)[q];
      x &= h - 1;
      y &= h - 1;
      l -= 1;
    }
    self.geom.cell(self.tile_words(id), x as usize, y as usize)
  }

  /// Fill the rectangle `[x0, x0 + w) x [y0, y0 + h)` of node `(id, l)` (node coordinates, possibly
  /// outside it) into `out` (`w * h`, row-major; cells outside the node are 0).
  #[allow(clippy::too_many_arguments)]
  pub fn read_rect(&self, id: u32, l: usize, nx: i64, ny: i64, x0: i64, y0: i64, w: usize, h: usize, out: &mut [u8]) {
    let side = 1i64 << l;
    if nx >= x0 + w as i64 || ny >= y0 + h as i64 || nx + side <= x0 || ny + side <= y0 || self.is_empty(l, id) {
      return;
    }
    if l == self.a {
      let s = self.geom.side as i64;
      let words = self.tile_words(id);
      for yy in 0..s {
        let ay = ny + yy;
        if ay < y0 || ay >= y0 + h as i64 {
          continue;
        }
        for xx in 0..s {
          let ax = nx + xx;
          if ax < x0 || ax >= x0 + w as i64 {
            continue;
          }
          out[(ay - y0) as usize * w + (ax - x0) as usize] = self.geom.cell(words, xx as usize, yy as usize);
        }
      }
      return;
    }
    let hs = side / 2;
    let c = self.ch(l, id);
    self.read_rect(c[0], l - 1, nx, ny, x0, y0, w, h, out);
    self.read_rect(c[1], l - 1, nx + hs, ny, x0, y0, w, h, out);
    self.read_rect(c[2], l - 1, nx, ny + hs, x0, y0, w, h, out);
    self.read_rect(c[3], l - 1, nx + hs, ny + hs, x0, y0, w, h, out);
  }

  /// Bounding box of the live cells of `(id, l)` with its corner at `(x, y)`: `(x0, y0, x1, y1)`,
  /// upper bounds exclusive.
  pub fn bbox(&self, id: u32, l: usize, x: i64, y: i64) -> Option<(i64, i64, i64, i64)> {
    if self.is_empty(l, id) {
      return None;
    }
    // each side on its own: descend into the children nearer to it, the farther ones only when the
    // nearer are empty, each distinct node once
    let mut memo = FxHashMap::default();
    let mut e = |s: Side| {
      memo.clear();
      self.edge(l, id, s, &mut memo).unwrap() as i64
    };
    let (x0, x1, y0, y1) = (e(Side::Left), e(Side::Right), e(Side::Top), e(Side::Bottom));
    Some((x + x0, y + y0, x + x1, y + y1))
  }

  /// The extreme live cell of node `(l, id)` towards `side`, relative to its corner: the least
  /// coordinate for left and top, one past the greatest for right and bottom.
  fn edge(&self, l: usize, id: u32, side: Side, memo: &mut FxHashMap<(usize, u32), Option<u64>>) -> Option<u64> {
    if self.is_empty(l, id) {
      return None;
    }
    if let Some(&v) = memo.get(&(l, id)) {
      return v;
    }
    let v = if l == self.a {
      self.tile_edge(id, side)
    } else {
      let h = 1u64 << (l - 1);
      let c = self.ch(l, id);
      // the nearer children, the farther ones, and the farther ones' offset along the axis
      let (near, far, min_side) = match side {
        Side::Left => ([c[0], c[2]], [c[1], c[3]], true),
        Side::Right => ([c[1], c[3]], [c[0], c[2]], false),
        Side::Top => ([c[0], c[1]], [c[2], c[3]], true),
        Side::Bottom => ([c[2], c[3]], [c[0], c[1]], false),
      };
      if min_side {
        let mut best: Option<u64> = None;
        for n in near {
          if let Some(e) = self.edge(l - 1, n, side, memo) {
            best = Some(best.map_or(e, |b| b.min(e)));
            if e == 0 {
              break;
            }
          }
        }
        best.or_else(|| far.iter().filter_map(|&n| self.edge(l - 1, n, side, memo)).min().map(|e| e + h))
      } else {
        let mut best: Option<u64> = None;
        for n in near {
          if let Some(e) = self.edge(l - 1, n, side, memo) {
            best = Some(best.map_or(e + h, |b| b.max(e + h)));
            if e == h {
              break;
            }
          }
        }
        best.or_else(|| far.iter().filter_map(|&n| self.edge(l - 1, n, side, memo)).max())
      }
    };
    memo.insert((l, id), v);
    v
  }

  fn tile_edge(&self, id: u32, side: Side) -> Option<u64> {
    let s = self.geom.side;
    let words = self.tile_words(id);
    let rows: Vec<u64> = (0..s).map(|y| (0..self.geom.planes as usize).fold(0u64, |a, k| a | self.geom.row(words, k, y))).collect();
    match side {
      Side::Top => rows.iter().position(|&r| r != 0).map(|y| y as u64),
      Side::Bottom => rows.iter().rposition(|&r| r != 0).map(|y| y as u64 + 1),
      Side::Left => rows.iter().filter(|&&r| r != 0).map(|r| r.trailing_zeros() as u64).min(),
      Side::Right => rows.iter().filter(|&&r| r != 0).map(|r| 64 - r.leading_zeros() as u64).max(),
    }
  }

  /// Replace cells of `(id, l)` at node coordinates: `cells` sorted by `(y, x)`, all inside the node.
  pub fn set_cells(&mut self, id: u32, l: usize, cells: &[((u64, u64), u8)]) -> u32 {
    if cells.is_empty() {
      return id;
    }
    if l == self.a {
      let mut words = self.tile_words(id).to_vec();
      let s = self.geom.side;
      let wpp = self.geom.wpp;
      for &((x, y), v) in cells {
        let bit = y as usize * s + x as usize;
        for k in 0..self.geom.planes as usize {
          let w = &mut words[k * wpp + bit / 32];
          if (v >> k) & 1 == 1 {
            *w |= 1 << (bit % 32);
          } else {
            *w &= !(1 << (bit % 32));
          }
        }
      }
      return self.tile(&words);
    }
    let h = 1u64 << (l - 1);
    let mut parts: [Vec<((u64, u64), u8)>; 4] = Default::default();
    for &((x, y), v) in cells {
      let q = (x >= h) as usize | ((y >= h) as usize) << 1;
      parts[q].push(((x & (h - 1), y & (h - 1)), v));
    }
    let c = self.ch(l, id);
    let mut out = c;
    for q in 0..4 {
      if !parts[q].is_empty() {
        out[q] = self.set_cells(c[q], l - 1, &parts[q]);
      }
    }
    self.intern(l, out)
  }

  /// Golly's macrocell (`[M2]`) of the node: the two-state flavour (8×8 leaves) for two-state rules,
  /// the multistate one otherwise.
  pub fn write_macrocell(&self, id: u32, l: usize, rule: &str, generation: Option<u64>) -> String {
    use std::fmt::Write as _;
    let two_state = self.rule.states() == 2;
    let mut out = String::from("[M2] (ghl)\n");
    if !rule.is_empty() {
      writeln!(out, "#R {rule}").unwrap();
    }
    if let Some(g) = generation {
      writeln!(out, "#G {g}").unwrap();
    }
    let mut w = McWriter { lines: Vec::new(), memo: FxHashMap::default(), two_state };
    let root = w.node(self, id, l);
    if root == 0 {
      // an empty universe: one empty node of the smallest level Golly accepts
      w.lines.push(if two_state { "$".into() } else { "1 0 0 0 0".into() });
    }
    for line in w.lines {
      out.push_str(&line);
      out.push('\n');
    }
    out
  }
}

/// Writes macrocell nodes depth-first, children before parents, 1-based indices.
struct McWriter {
  lines: Vec<String>,
  memo: FxHashMap<(u32, u32, u32), u64>,
  two_state: bool,
}

impl McWriter {
  /// Index of node `(id, l)`, 0 if it is empty.
  fn node(&mut self, e: &Engine, id: u32, l: usize) -> u64 {
    if e.is_empty(l, id) {
      return 0;
    }
    if l == e.a || (self.two_state && l <= 3) {
      return self.cells(e, &e.expand(id, l).cells, l);
    }
    if let Some(&i) = self.memo.get(&(l as u32, id, 0)) {
      return i;
    }
    let c = e.ch(l, id);
    let k: Vec<u64> = c.iter().map(|&x| self.node(e, x, l - 1)).collect();
    let i = self.push(format!("{} {} {} {} {}", l, k[0], k[1], k[2], k[3]));
    self.memo.insert((l as u32, id, 0), i);
    i
  }

  fn push(&mut self, s: String) -> u64 {
    self.lines.push(s);
    self.lines.len() as u64
  }

  /// A square of raw cells of side `2^d` (row-major), as nodes of the format.
  fn cells(&mut self, e: &Engine, cells: &[u8], d: usize) -> u64 {
    if cells.iter().all(|&c| c == 0) {
      return 0;
    }
    let side = 1usize << d;
    if self.two_state && d == 3 {
      let mut s = String::new();
      let mut rows: Vec<String> = Vec::new();
      for y in 0..8 {
        let row = &cells[y * 8..(y + 1) * 8];
        let end = row.iter().rposition(|&c| c != 0).map(|p| p + 1).unwrap_or(0);
        rows.push(row[..end].iter().map(|&c| if c != 0 { '*' } else { '.' }).collect());
      }
      while rows.last().is_some_and(|r| r.is_empty()) {
        rows.pop();
      }
      for r in rows {
        s.push_str(&r);
        s.push('$');
      }
      let key = (3u32, u32::MAX, hash_cells(cells));
      if let Some(&i) = self.memo.get(&key) {
        if self.lines[i as usize - 1] == s {
          return i;
        }
      }
      let i = self.push(s);
      self.memo.insert(key, i);
      return i;
    }
    if !self.two_state && d == 1 {
      let s = format!("1 {} {} {} {}", cells[0], cells[1], cells[2], cells[3]);
      let key = (1u32, u32::MAX, hash_cells(cells));
      if let Some(&i) = self.memo.get(&key) {
        if self.lines[i as usize - 1] == s {
          return i;
        }
      }
      let i = self.push(s);
      self.memo.insert(key, i);
      return i;
    }
    if d < 3 && self.two_state {
      // a two-state universe with base tiles below 8x8: pad up to one leaf
      let mut leaf = vec![0u8; 64];
      for y in 0..side {
        leaf[y * 8..y * 8 + side].copy_from_slice(&cells[y * side..(y + 1) * side]);
      }
      return self.cells(e, &leaf, 3);
    }
    let h = side / 2;
    let mut k = [0u64; 4];
    for (q, kq) in k.iter_mut().enumerate() {
      let (ox, oy) = ((q & 1) * h, (q >> 1) * h);
      let mut sub = vec![0u8; h * h];
      for y in 0..h {
        sub[y * h..(y + 1) * h].copy_from_slice(&cells[(oy + y) * side + ox..(oy + y) * side + ox + h]);
      }
      *kq = self.cells(e, &sub, d - 1);
    }
    let s = format!("{} {} {} {} {}", d, k[0], k[1], k[2], k[3]);
    let key = (d as u32, u32::MAX - 1, hash_cells(cells));
    if let Some(&i) = self.memo.get(&key) {
      if self.lines[i as usize - 1] == s {
        return i;
      }
    }
    let i = self.push(s);
    self.memo.insert(key, i);
    i
  }
}

fn hash_cells(cells: &[u8]) -> u32 {
  let mut h: u32 = 0x811C_9DC5;
  for &c in cells {
    h ^= c as u32;
    h = h.wrapping_mul(0x0100_0193);
  }
  h
}

/// Order-independent content hash, population and bounding box of a set of cells: the harness's
/// fingerprint, which the recorded Golly fixtures use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Fingerprint {
  pub population: u64,
  pub w: i64,
  pub h: i64,
  pub hash: u64,
}

#[inline]
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
    return Fingerprint::default();
  }
  let mut hash = 0u64;
  visit(&mut |x, y, s| hash = hash.wrapping_add(mix(x - x0, y - y0, s)));
  Fingerprint { population: pop, w: x1 - x0 + 1, h: y1 - y0 + 1, hash }
}

/// Fingerprint of a dense grid.
pub fn grid_fingerprint(g: &Grid) -> Fingerprint {
  fingerprint(|f| {
    for y in 0..g.h {
      for x in 0..g.w {
        let c = g.get(x, y);
        if c != 0 {
          f(x as i64, y as i64, c);
        }
      }
    }
  })
}

impl Engine {
  /// Fingerprint of node `(id, l)`.
  pub fn fingerprint(&self, id: u32, l: usize) -> Fingerprint {
    fingerprint(|f| self.for_each_cell(id, l, 0, 0, &mut |x, y, s| f(x, y, s)))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn bbox_equals_the_cells() {
    let rule = Rule::named("Hutton32b").unwrap();
    let mut s = 0x1234_5678u64;
    for trial in 0..60 {
      let (w, h) = (1 + trial * 13 % 211, 1 + trial * 29 % 173);
      let mut g = Grid::new(w, h);
      for _ in 0..(trial % 7 + 1) {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let (x, y) = ((s >> 20) as usize % w, (s >> 40) as usize % h);
        g.set(x, y, 1 + (s >> 59) as u8 % 31);
      }
      let mut e = Engine::new(&rule, 3, 3);
      let (id, l) = e.from_grid(&g);
      let want = g.bbox().map(|(a, b, c, d)| (a as i64 + 5, b as i64 - 7, c as i64 + 5, d as i64 - 7));
      assert_eq!(e.bbox(id, l, 5, -7), want, "trial {trial}");
    }
  }
}