//! The GPU engines (feature `gpu`): the plain and shared-tile kernels over dense grids
//! ([`brute`], Stage 1), the memo tables in VRAM ([`memo`]) with the hit-path sweep over ID grids
//! ([`sweep`], Stage 2) and the miss path ([`miss`], Stage 3), start-up calibration
//! ([`calibrate`]), and grids beyond one binding ([`shard`], Stage 5). Every kernel is WGSL
//! generated from the rule ([`kernels`]).

pub mod brute;
pub mod calibrate;
pub mod context;
pub mod kernels;
pub mod memo;
pub mod miss;
pub mod shard;
pub mod sweep;

use std::time::{Duration, Instant};

pub use context::GpuContext;
use rustc_hash::FxHashMap;

use crate::{
  cpu::{Engine, PlaneGrid},
  inspect::{EngineUse, Hub, LevelCounters, MemoEntry, MemoStats, Slot, SlotState},
  io::Grid,
  query::{EngineKind, Memory, Rect},
  rule::Rule,
  sched::Regime,
  tile::{TileGeom, NONE},
  universe::{Config, Core, Root},
  Error, Result,
};

use brute::{DenseGrid, DenseKernels};
use memo::Memo;
use sweep::IdGrid;

/// Where the GPU holds the universe.
pub(crate) enum State {
  None,
  Dense(DenseGrid),
  /// a dense grid beyond one binding (Stage 5)
  Sharded(shard::ShardedGrid),
  Ids(IdGrid),
}

/// The GPU's part of a universe.
pub struct GpuCore {
  pub ctx: GpuContext,
  rule: Rule,
  a: u32,
  kernels: DenseKernels,
  pub(crate) state: State,
  memo: Option<Memo>,
  miss: Option<miss::MissPath>,
  /// generations per dense epoch, adapted to the epoch's time
  dense_gens: u64,
  /// sweeps per memo burst
  burst: u32,
  overflows: u64,
  /// use the GPU's miss path (Stage 3); off: the host learns (Stage 2)
  pub gpu_miss: bool,
  epoch_target: Duration,
  last_active: f64,
  /// the last burst of the GPU's miss path, as a trace
  last_trace: Option<crate::inspect::StepTrace>,
  /// dense grids above this many words are sharded (default: one binding)
  grid_shard_words: Option<u64>,
  /// ID grids seen at epoch boundaries, by hash, for cycle detection (small grids only)
  cycle_grids: FxHashMap<u64, (u64, Vec<u32>)>,
}

/// The buffers a UI on the same device draws from, with their layouts.
#[derive(Clone, Debug)]
pub enum GpuView {
  /// a dense grid: groups of `planes` words per 32 cells, rows of `ww` groups, `h` rows, a ring of
  /// zero groups and rows; the first interior cell at `(x0, y0)`
  Cells { buffer: wgpu::Buffer, ww: u32, h: u32, planes: u32, x0: i64, y0: i64 },
  /// an ID grid of `nx * ny` tiles of side `2^lt` at `(x0, y0)` (phase A), the node table
  /// (eight words per entry) and the tile store (`tile_words` per tile, bit-planes)
  /// The node table may be split into shards of `2^shard_log2` entries (a slot's shard is its top bits).
  Ids { ids: wgpu::Buffer, nx: u32, ny: u32, lt: u32, x0: i64, y0: i64, nodes: Vec<wgpu::Buffer>, shard_log2: u32, tile_slots: wgpu::Buffer, tile_store: wgpu::Buffer, base_tile: u32, tile_words: u32 },
}

fn random_grid(w: usize, h: usize, states: u32, seed: u32) -> Grid {
  let mut g = Grid::new(w, h);
  let mut s = seed | 1;
  for c in g.cells.iter_mut() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    *c = if s % 3 == 0 { 0 } else { (s >> 8) as u8 % states.max(1) as u8 };
  }
  g
}

impl GpuCore {
  /// Compile the rule's kernels and check them cell-exact against the host's brute force.
  pub fn new(ctx: GpuContext, rule: &Rule, config: &Config) -> Result<GpuCore> {
    let mut ctx = ctx;
    ctx.trusted = config.trusted_kernels;
    let kernels = DenseKernels::new(&ctx, rule)?;
    let core = GpuCore {
      ctx,
      rule: rule.clone(),
      a: 3,
      kernels,
      state: State::None,
      memo: None,
      miss: None,
      dense_gens: 64,
      burst: 8,
      overflows: 0,
      gpu_miss: config.gpu_miss_path,
      epoch_target: config.epoch_target,
      last_active: 0.0,
      cycle_grids: FxHashMap::default(),
      last_trace: None,
      grid_shard_words: config.gpu_grid_shard_words,
    };
    core.self_test()?;
    Ok(core)
  }

  /// The plain and shared-tile kernels against `cpu::leaf` on a random grid, from both parities.
  fn self_test(&self) -> Result<()> {
    let g = random_grid(150, 70, self.rule.states(), 0x1234_5678);
    let k = &self.kernels;
    for (shared, gens, gen0) in [(false, 9u64, 0u64), (true, 2 * k.tile_shape.2 as u64 + 3, 1)] {
      let m = gens as usize + 2;
      let padded = g.padded(m, m, g.w + 2 * m, g.h + 2 * m);
      let mut dg = DenseGrid::new(&self.ctx, k, padded.w as u64, padded.h as u64, 0, 0, Some(&padded))?;
      dg.step(&self.ctx, k, gens, gen0, shared);
      let got = dg.download(&self.ctx);
      let mut pg = PlaneGrid::from_cells(self.rule.planes() as usize, padded.w, padded.h, &padded.cells);
      let mut tmp = pg.clone();
      crate::cpu::Leaf::new(&self.rule).step(&mut pg, &mut tmp, gens as usize, gen0);
      let want = pg.to_cells(padded.w, padded.h);
      let got = got.crop(0, 0, padded.w, padded.h);
      if got.cells != want {
        let bad = got.cells.iter().zip(&want).filter(|(a, b)| a != b).count();
        return Err(Error::Gpu(format!("the {} kernel differs from the host on {bad} cells ({})", if shared { "shared-tile" } else { "plain" }, k.name())));
      }
    }
    Ok(())
  }

  pub fn calibrate(&self) -> Result<calibrate::Calibrated> {
    calibrate::calibrate(&self.ctx, &self.kernels, self.rule.states(), crate::sched::Rates::paper(self.kernels.src.class))
  }
  pub fn kernel_name(&self) -> String {
    self.kernels.name()
  }
  /// The universe-wide floor changed: the miss path is rebuilt for it at the next burst.
  pub fn set_floor(&mut self, f: usize) {
    if self.miss.as_ref().map(|m| m.floor != f).unwrap_or(false) {
      self.miss = None;
    }
  }
  /// The last burst of the miss path, wave by level, from the GPU's counters.
  pub fn last_trace(&self) -> Option<crate::inspect::StepTrace> {
    self.last_trace.clone()
  }
  pub fn set_base_tile(&mut self, a: u32) {
    self.a = a;
  }
  pub fn holds_state(&self) -> bool {
    !matches!(self.state, State::None)
  }
  /// The host's state is the truth again (an edit): forget the GPU's grid, keep the memo.
  pub fn invalidate(&mut self) {
    self.state = State::None;
    self.cycle_grids.clear();
  }
  /// The host renumbered its nodes: the translation is stale (the GPU's tables stay valid).
  pub fn host_reclaimed(&mut self) {
    if let Some(m) = self.memo.as_mut() {
      m.forget_host();
    }
  }
  pub fn active_cells(&self) -> f64 {
    self.last_active
  }
  pub fn memory(&self) -> Memory {
    let mut m = Memory::default();
    match &self.state {
      State::Dense(d) => m.grids = d.bytes(),
      State::Sharded(s) => m.grids = s.bytes(),
      State::Ids(g) => m.grids = g.bytes(),
      State::None => {}
    }
    if let Some(memo) = &self.memo {
      let (t, s) = memo.bytes();
      m.tables = t;
      m.tile_store = s;
    }
    if let Some(mp) = &self.miss {
      m.atlas = mp.bytes();
    }
    m
  }
  pub fn entries(&self) -> u64 {
    self.memo.as_ref().map(|m| m.node_count).unwrap_or(0)
  }
  pub fn overflows(&self) -> u64 {
    self.overflows + self.miss.as_ref().map(|m| m.overflows).unwrap_or(0)
  }

  // ---- conversions --------------------------------------------------------------------------------

  /// The dense grid of the host's root with `margin` free cells around its live cells.
  fn upload_dense(&mut self, host: &Engine, root: Root, margin: i64) -> Result<()> {
    let (x0, y0, x1, y1) = host.bbox(root.id, root.level, root.x, root.y).unwrap_or((root.x, root.y, root.x + 1, root.y + 1));
    let (gx0, gy0) = (x0 - margin, y0 - margin);
    let (w, h) = ((x1 + margin - gx0) as usize, (y1 + margin - gy0) as usize);
    let w = w.div_ceil(32) * 32;
    let mut g = Grid::new(w, h);
    host.read_rect(root.id, root.level, root.x, root.y, gx0, gy0, w, h, &mut g.cells);
    let words = ((w as u64).div_ceil(32) + 2) * (h as u64 + 2) * self.rule.planes() as u64;
    let shard_words = self.grid_shard_words.unwrap_or((self.ctx.max_binding() / 4) & !1023).max(words.div_ceil(4));
    if words > shard_words {
      // beyond one binding: the grid in shards, one kernel spanning them
      let s = shard::ShardedGrid::new(&self.ctx, &self.kernels.src, self.kernels.extra.as_ref(), w as u64, h as u64, gx0, gy0, Some(&g), shard_words)?;
      self.last_active = s.last.bbox.map(|b| ((b.2 - b.0) * (b.3 - b.1)) as f64).unwrap_or(0.0);
      self.state = State::Sharded(s);
      return Ok(());
    }
    let mut d = DenseGrid::new(&self.ctx, &self.kernels, w as u64, h as u64, gx0, gy0, Some(&g))?;
    d.census(&self.ctx, &self.kernels);
    self.last_active = d.last.bbox.map(|b| ((b.2 - b.0) * (b.3 - b.1)) as f64).unwrap_or(0.0);
    self.state = State::Dense(d);
    Ok(())
  }

  /// The GPU's state into the host's quadtree; the GPU holds nothing afterwards.
  pub fn download(&mut self, host: &mut Engine, align: usize) -> Root {
    let state = std::mem::replace(&mut self.state, State::None);
    match state {
      State::None => Root { id: host.empty(align + 2), level: align + 2, x: 0, y: 0 },
      State::Dense(d) => dense_to_host(&self.ctx, &d, host, align),
      State::Sharded(s) => sharded_to_host(&self.ctx, &s, host, align),
      State::Ids(g) => {
        let memo = self.memo.as_mut().expect("an ID grid has tables");
        let gpu = g.read_a(&self.ctx);
        let ids = memo.download(&self.ctx, host, g.lt as usize, &gpu);
        let (id, level) = sweep::build_from_ids(host, &ids, g.nx as usize, g.ny as usize, g.lt as usize);
        Root { id, level, x: g.x0, y: g.y0 }
      }
    }
  }

  /// A copy of the GPU's state in a fresh host engine `e`.
  pub fn download_copy(&self, e: &mut Engine, align: usize) -> Root {
    match &self.state {
      State::None => Root { id: e.empty(align + 2), level: align + 2, x: 0, y: 0 },
      State::Dense(d) => dense_to_host(&self.ctx, d, e, align),
      State::Sharded(s) => sharded_to_host(&self.ctx, s, e, align),
      State::Ids(g) => {
        let memo = self.memo.as_ref().expect("an ID grid has tables");
        let gpu = g.read_a(&self.ctx);
        let ids = fresh_download(&self.ctx, memo, e, g.lt as usize, &gpu);
        let (id, level) = sweep::build_from_ids(e, &ids, g.nx as usize, g.ny as usize, g.lt as usize);
        Root { id, level, x: g.x0, y: g.y0 }
      }
    }
  }

  // ---- queries while the GPU holds the state ------------------------------------------------------

  pub fn population(&self) -> u64 {
    match &self.state {
      State::Dense(d) => d.last.population,
      State::Sharded(s) => s.last.population,
      State::Ids(g) => {
        let memo = self.memo.as_ref().unwrap();
        let ids = g.read_a(&self.ctx);
        let mut count: FxHashMap<u32, u64> = FxHashMap::default();
        for &i in &ids {
          if i != g.empty {
            *count.entry(i).or_default() += 1;
          }
        }
        let keys: Vec<u32> = count.keys().copied().collect();
        let pops = node_pops(&self.ctx, memo, g.lt as usize, &keys);
        keys.iter().zip(pops).map(|(k, p)| count[k] * p).sum()
      }
      State::None => 0,
    }
  }

  pub fn bounding_box(&self) -> Option<Rect> {
    match &self.state {
      State::Dense(d) => d.last.bbox.map(|(a, b, c, e)| Rect::from_corners(a, b, c, e)),
      State::Sharded(s) => s.last.bbox.map(|(a, b, c, e)| Rect::from_corners(a, b, c, e)),
      State::Ids(_) => {
        let mut e = Engine::new(&self.rule, self.a as usize, 1);
        let r = self.download_copy(&mut e, 10);
        e.bbox(r.id, r.level, r.x, r.y).map(|(a, b, c, d)| Rect::from_corners(a, b, c, d))
      }
      State::None => None,
    }
  }

  pub fn read_region(&self, rect: Rect, out: &mut [u8]) {
    match &self.state {
      State::Dense(d) => d.read_rect(&self.ctx, rect, out),
      State::Sharded(_) | State::Ids(_) => {
        let mut e = Engine::new(&self.rule, self.a as usize, 1);
        let r = self.download_copy(&mut e, 10);
        e.read_rect(r.id, r.level, r.x, r.y, rect.x, rect.y, rect.w as usize, rect.h as usize, out);
      }
      State::None => {}
    }
  }

  /// IDs at the sweep level, read from VRAM (`None` for other levels or a dense state).
  pub fn ids(&self, rect: Rect, level: u32) -> Option<Vec<u32>> {
    let State::Ids(g) = &self.state else { return None };
    if level != g.lt {
      return None;
    }
    let all = g.read_a(&self.ctx);
    let (tx0, ty0, nx, ny) = rect.tiles(level);
    let s = 1i64 << level;
    let mut out = vec![NONE; nx * ny];
    for j in 0..ny {
      for i in 0..nx {
        let (gx, gy) = (((tx0 + i as i64) * s - g.x0) / s, ((ty0 + j as i64) * s - g.y0) / s);
        if gx >= 0 && gy >= 0 && gx < g.nx as i64 && gy < g.ny as i64 {
          out[j * nx + i] = all[(gy * g.nx as i64 + gx) as usize];
        }
      }
    }
    Some(out)
  }

  /// A tile of the GPU's tables: the cells of a base tile (a slot of the tile store) or the four
  /// children of a node (a slot of the node table). `None` while the GPU holds no ID grid.
  pub fn tile(&self, level: u32, id: u32) -> Option<Option<crate::inspect::Tile>> {
    let State::Ids(_) = &self.state else { return None };
    let memo = self.memo.as_ref()?;
    if level as usize == memo.a {
      if id as u64 >= 1u64 << memo.tile_log2 {
        return Some(None);
      }
      let slot = self.ctx.read_u32(&memo.tile_slots, id as u64 * memo::TILE_SLOT, memo::TILE_SLOT);
      if slot[0] & memo::COMMITTED == 0 {
        return Some(None);
      }
      let words = self.ctx.read_u32(&memo.store, id as u64 * memo.w as u64, memo.w as u64);
      let geom = TileGeom::new(memo.a as u32, self.rule.planes());
      return Some(Some(crate::inspect::Tile::Cells { side: geom.side as u32, cells: geom.unpack(&words) }));
    }
    if id as u64 >= 1u64 << memo.node_log2 {
      return Some(None);
    }
    let e = memo.read_slots(&self.ctx, id as u64, 1);
    if e[5] & memo::COMMITTED == 0 || e[6] & 0xFF != level {
      return Some(None);
    }
    Some(Some(crate::inspect::Tile::Node { level, children: [e[0], e[1], e[2], e[3]], population: e[7] as u64 }))
  }

  /// The outcome of the windows of the sweep that would run now, from the GPU's tables.
  pub fn outcomes(&self, rect: Rect) -> Option<Vec<crate::inspect::Outcome>> {
    let State::Ids(g) = &self.state else { return None };
    let memo = self.memo.as_ref()?;
    let all = g.read_a(&self.ctx);
    let (tx0, ty0, nx, ny) = rect.tiles(g.lt);
    let s = 1i64 << g.lt;
    let at = |gx: i64, gy: i64| if gx >= 0 && gy >= 0 && gx < g.nx as i64 && gy < g.ny as i64 { all[(gy * g.nx as i64 + gx) as usize] } else { g.empty };
    let mut keys = Vec::new();
    for j in 0..ny {
      for i in 0..nx {
        let (gx, gy) = (((tx0 + i as i64) * s - g.x0) / s, ((ty0 + j as i64) * s - g.y0) / s);
        keys.push([at(gx - 1, gy - 1), at(gx, gy - 1), at(gx - 1, gy), at(gx, gy)]);
      }
    }
    let table = memo.read_slots(&self.ctx, 0, 1u64 << memo.node_log2);
    let mask = (1u32 << memo.node_log2) - 1;
    Some(
      keys
        .iter()
        .map(|k| {
          if k.iter().all(|&x| x == g.empty) {
            return crate::inspect::Outcome::Empty;
          }
          match probe_host(&table, *k, g.lt + 1, mask) {
            Some(s) if table[s as usize * 8 + 4] != NONE => crate::inspect::Outcome::Hit,
            _ => crate::inspect::Outcome::Miss,
          }
        })
        .collect(),
    )
  }

  /// Occupancy and probe lengths of the GPU table's entries of `level`.
  pub fn memo_stats(&self, level: u32) -> Option<MemoStats> {
    let memo = self.memo.as_ref()?;
    let table = memo.read_slots(&self.ctx, 0, 1u64 << memo.node_log2);
    let mask = (1u32 << memo.node_log2) - 1;
    let mut hist = vec![0u64; 32];
    let (mut entries, mut probes) = (0u64, 0u64);
    for s in 0..=mask {
      let b = s as usize * 8;
      if table[b + 5] & memo::COMMITTED != 0 && table[b + 6] & 0xFF == level {
        let home = memo::node_home([table[b], table[b + 1], table[b + 2], table[b + 3]], level, mask);
        let d = s.wrapping_sub(home) & mask;
        hist[(d as usize).min(31)] += 1;
        entries += 1;
        probes += d as u64 + 1;
      }
    }
    let slots = mask as u64 + 1;
    Some(MemoStats { level, entries, slots, bytes: slots * 32, load: entries as f64 / slots as f64, avg_probes: probes as f64 / entries.max(1) as f64, displacement: hist })
  }

  pub fn memo_map(&self, level: u32, slots: std::ops::Range<u64>) -> Option<Vec<Slot>> {
    let memo = self.memo.as_ref()?;
    let n = (1u64 << memo.node_log2).min(slots.end).saturating_sub(slots.start);
    let words = memo.read_slots(&self.ctx, slots.start, n);
    let mask = (1u32 << memo.node_log2) - 1;
    Some(
      words
        .chunks(8)
        .enumerate()
        .map(|(i, e)| {
          let s = slots.start as u32 + i as u32;
          let key = [e[0], e[1], e[2], e[3]];
          let state = if e[5] == 0 {
            SlotState::Empty
          } else if e[5] & memo::COMMITTED != 0 {
            SlotState::Committed
          } else {
            SlotState::Claimed(e[5] - 1)
          };
          let disp = if state == SlotState::Empty || e[6] & 0xFF != level { 0 } else { s.wrapping_sub(memo::node_home(key, level, mask)) & mask };
          Slot { state, key, value: e[4], displacement: disp }
        })
        .collect(),
    )
  }

  pub fn memo_lookup(&self, level: u32, key: [u32; 4]) -> Option<Option<MemoEntry>> {
    let memo = self.memo.as_ref()?;
    let mask = (1u32 << memo.node_log2) - 1;
    let mut slot = memo::node_home(key, level, mask);
    for _ in 0..=mask {
      let e = memo.read_slots(&self.ctx, slot as u64, 1);
      if e[5] == 0 {
        return Some(None);
      }
      if [e[0], e[1], e[2], e[3]] == key && e[6] & 0xFF == level {
        return Some(Some(MemoEntry { level, key, id: slot, result: (e[4] != NONE).then_some(e[4]), slot: slot as u64 }));
      }
      slot = (slot + 1) & mask;
    }
    Some(None)
  }

  pub fn view(&self) -> Option<GpuView> {
    match &self.state {
      State::Dense(d) => Some(GpuView::Cells { buffer: d.bufs[d.cur].clone(), ww: d.ww, h: d.h, planes: d.b, x0: d.x0, y0: d.y0 }),
      State::Sharded(_) => None,
      State::Ids(g) => {
        let m = self.memo.as_ref()?;
        Some(GpuView::Ids { ids: g.a.clone(), nx: g.nx, ny: g.ny, lt: g.lt, x0: g.x0, y0: g.y0, nodes: m.nodes.clone(), shard_log2: m.shard_log2, tile_slots: m.tile_slots.clone(), tile_store: m.store.clone(), base_tile: self.a, tile_words: m.w as u32 })
      }
      State::None => None,
    }
  }
}

fn sharded_to_host(ctx: &GpuContext, s: &shard::ShardedGrid, host: &mut Engine, align: usize) -> Root {
  let g = s.download(ctx);
  let al = 1i64 << align;
  let (ax, ay) = (s.x0.div_euclid(al) * al, s.y0.div_euclid(al) * al);
  let (px, py) = ((s.x0 - ax) as usize, (s.y0 - ay) as usize);
  let (id, level) = host.from_grid(&g.padded(px, py, g.w + px, g.h + py));
  Root { id, level, x: ax, y: ay }
}

fn dense_to_host(ctx: &GpuContext, d: &DenseGrid, host: &mut Engine, align: usize) -> Root {
  let g = d.download(ctx);
  let al = 1i64 << align;
  let (ax, ay) = (d.x0.div_euclid(al) * al, d.y0.div_euclid(al) * al);
  let (px, py) = ((d.x0 - ax) as usize, (d.y0 - ay) as usize);
  let padded = g.padded(px, py, g.w + px, g.h + py);
  let (id, level) = host.from_grid(&padded);
  Root { id, level, x: ax, y: ay }
}

/// Probe a host copy of the node table.
fn probe_host(table: &[u32], key: [u32; 4], level: u32, mask: u32) -> Option<u32> {
  let mut s = memo::node_home(key, level, mask);
  for _ in 0..=mask {
    let b = s as usize * 8;
    if table[b + 5] == 0 {
      return None;
    }
    if [table[b], table[b + 1], table[b + 2], table[b + 3]] == key && table[b + 6] & 0xFF == level {
      return Some(s);
    }
    s = (s + 1) & mask;
  }
  None
}

/// Populations of GPU nodes of level `l` (tiles at the base level).
fn node_pops(ctx: &GpuContext, memo: &Memo, l: usize, ids: &[u32]) -> Vec<u64> {
  if l == memo.a {
    return ids.iter().map(|&t| ctx.read_u32(&memo.tile_slots, t as u64 * memo::TILE_SLOT + 2, 1)[0] as u64).collect();
  }
  let entries = memo.read_entries(ctx, ids);
  // a saturated population is recounted from the children
  entries.iter().map(|e| e[7] as u64).collect()
}

/// GPU IDs to IDs of a fresh host engine, reading every entry below them.
fn fresh_download(ctx: &GpuContext, memo: &Memo, e: &mut Engine, l: usize, ids: &[u32]) -> Vec<u32> {
  let a = memo.a;
  let mut maps: Vec<FxHashMap<u32, u32>> = vec![FxHashMap::default(); l + 1];
  let mut unknown: Vec<Vec<u32>> = vec![Vec::new(); l + 1];
  let mut keys: Vec<FxHashMap<u32, [u32; 4]>> = vec![FxHashMap::default(); l + 1];
  unknown[l] = ids.to_vec();
  for lv in (a + 1..=l).rev() {
    let mut list = std::mem::take(&mut unknown[lv]);
    list.sort_unstable();
    list.dedup();
    let entries = memo.read_entries(ctx, &list);
    for (&id, en) in list.iter().zip(&entries) {
      let k = [en[0], en[1], en[2], en[3]];
      keys[lv].insert(id, k);
      unknown[lv - 1].extend_from_slice(&k);
    }
    unknown[lv] = list;
  }
  unknown[a].sort_unstable();
  unknown[a].dedup();
  for &t in &unknown[a] {
    let words = ctx.read_u32(&memo.store, t as u64 * memo.w as u64, memo.w as u64);
    let h = e.tile(&words);
    maps[a].insert(t, h);
  }
  for lv in a + 1..=l {
    for &id in &unknown[lv] {
      let ch = keys[lv][&id].map(|c| maps[lv - 1][&c]);
      let h = e.intern(lv, ch);
      maps[lv].insert(id, h);
    }
  }
  ids.iter().map(|i| maps[l][i]).collect()
}

// ---- the GPU's epochs -----------------------------------------------------------------------------

impl Core {
  /// One epoch on the GPU, or `None` when the regime does not run there.
  pub(crate) fn gpu_epoch(&mut self, regime: Regime, left: u64, budget: Option<Duration>, hub: &Hub) -> Result<Option<(u64, EngineKind, f64, Duration)>> {
    let _ = budget;
    match regime {
      Regime::PlainCells | Regime::SharedTile => self.dense_epoch(regime, left, hub).map(Some),
      Regime::Memo => self.memo_epoch(left, hub),
      Regime::HostRecursion => Ok(None),
    }
  }

  fn dense_epoch(&mut self, regime: Regime, left: u64, hub: &Hub) -> Result<(u64, EngineKind, f64, Duration)> {
    // a sharded grid grows through the host, which keeps the state if the new upload fails
    let gens_next = self.gpu.as_ref().unwrap().dense_gens.min(left).max(1) as i64;
    if matches!(&self.gpu.as_ref().unwrap().state, State::Sharded(s) if s.margin() <= gens_next + 1) {
      self.sync_host();
    }
    let needs_upload = !matches!(self.gpu.as_ref().unwrap().state, State::Dense(_) | State::Sharded(_));
    if needs_upload {
      self.sync_host();
      let (root, gens) = (self.root, self.gpu.as_ref().unwrap().dense_gens);
      let g = self.gpu.as_mut().unwrap();
      g.upload_dense(&self.host, root, (4 * gens as i64).max(256))?;
    }
    let gen0 = self.generation;
    let g = self.gpu.as_mut().unwrap();
    let gens = g.dense_gens.min(left).max(1);
    let mut shared = regime == Regime::SharedTile;
    let (bbox0, census, wall) = match &mut g.state {
      State::Dense(d) => {
        if d.margin() <= gens as i64 + 1 {
          let nd = d.grown(&g.ctx, &g.kernels, (4 * gens as i64).max(256))?;
          *d = nd;
        }
        let bbox0 = d.last.bbox;
        let t = Instant::now();
        d.step(&g.ctx, &g.kernels, gens, gen0, shared);
        let census = d.census(&g.ctx, &g.kernels);
        (bbox0, census, t.elapsed())
      }
      State::Sharded(s) => {
        // the plain kernel only (the grid grew through the host above)
        shared = false;
        let bbox0 = s.last.bbox;
        let t = Instant::now();
        s.step(&g.ctx, gens, gen0);
        let census = s.census(&g.ctx);
        (bbox0, census, t.elapsed())
      }
      _ => unreachable!(),
    };
    g.last_active = census.bbox.map(|b| ((b.2 - b.0) * (b.3 - b.1)) as f64).unwrap_or(0.0);
    let area = bbox0.or(census.bbox).map(|b| ((b.2 - b.0) * (b.3 - b.1)) as f64).unwrap_or(0.0);
    let cu = area * gens as f64;
    // epochs aim at the target time; the census read-back is paid once per epoch
    let target = g.epoch_target.as_secs_f64();
    if wall.as_secs_f64() < target / 2.0 && g.dense_gens < 1 << 16 {
      g.dense_gens *= 2;
    } else if wall.as_secs_f64() > 2.0 * target && g.dense_gens > 1 {
      g.dense_gens /= 2;
    }
    let (tw, th, k) = g.kernels.tile_shape;
    let kernel = format!("{} {}", g.kernels.name(), if shared { "shared tile" } else { "plain" });
    self.generation += gens;
    let engine = if shared { EngineKind::GpuSharedTile } else { EngineKind::GpuPlain };
    self.account(engine, gens, cu, wall);
    if self.sched.racer.current == Some(regime) {
      self.sched.racer.report(regime, gens as f64 / wall.as_secs_f64().max(1e-9));
    }
    let eu = EngineUse { engine, kernel, tile: if shared { (tw * 32, th) } else { (32, 1) }, k: if shared { k } else { 1 }, generations: gens, cell_updates: cu, micros: wall.as_secs_f64() * 1e6 };
    let s = self.sample(regime, gens, vec![eu], Vec::new(), wall);
    hub.publish(s);
    hub.serve(self);
    Ok((gens, engine, cu, wall))
  }

  /// One burst of the memo engine: sweeps at the sweep tile, misses resolved on the GPU (or by the
  /// host). `None` when fewer generations are left than one burst covers.
  fn memo_epoch(&mut self, left: u64, hub: &Hub) -> Result<Option<(u64, EngineKind, f64, Duration)>> {
    let a = self.host.a;
    let lt = (self.sched.sweep.lt as usize).max(a);
    let per_sweep = 1u64 << (lt - 1);
    let mut n = self.gpu.as_ref().unwrap().burst;
    while n > 2 && (n as u64) * per_sweep > left {
      n -= 2;
    }
    if (n as u64) * per_sweep > left {
      return Ok(None);
    }
    // the tables: created small, rebuilt twice as large past 45 % load (the memo is a cache: the
    // rebuild keeps what the current grid reaches and relearns the rest)
    let geom = TileGeom::new(a as u32, self.rule.planes());
    let (max_node_log2, max_tile_log2) = {
      let g = self.gpu.as_ref().unwrap();
      // up to eight shards of one binding each
      let n = 63 - (8 * (g.ctx.max_binding() / (memo::ENTRY * 4))).min(1 << 29).leading_zeros();
      let t = 63 - (g.ctx.max_binding() / ((geom.words() as u64 + memo::TILE_SLOT) * 4)).min(1 << 24).leading_zeros();
      (n, t)
    };
    let full = self.gpu.as_ref().unwrap().memo.as_ref().map(|m| (m.node_count as f64 > 0.45 * (1u64 << m.node_log2) as f64 && m.node_log2 < max_node_log2) || (m.tile_count as f64 > 0.45 * (1u64 << m.tile_log2) as f64 && m.tile_log2 < max_tile_log2));
    if let Some(true) = full {
      // rebuild: the state goes to the host, the tables are replaced, the grid is uploaded again
      self.sync_host();
      let g = self.gpu.as_mut().unwrap();
      let old = g.memo.take().unwrap();
      let nl = (old.node_log2 + 1).min(max_node_log2);
      let m = match self.config.gpu_table_shard_log2 {
        Some(s) => Memo::with_shards(&g.ctx, a, self.rule.planes(), nl, (old.tile_log2 + 1).min(max_tile_log2), s)?,
        None => Memo::new(&g.ctx, a, self.rule.planes(), nl, (old.tile_log2 + 1).min(max_tile_log2))?,
      };
      g.memo = Some(m);
      g.miss = None;
    }
    if self.gpu.as_ref().unwrap().memo.is_none() {
      let g = self.gpu.as_ref().unwrap();
      let m = match self.config.gpu_table_shard_log2 {
        Some(s) => Memo::with_shards(&g.ctx, a, self.rule.planes(), 20u32.min(max_node_log2), 18u32.min(max_tile_log2), s)?,
        None => Memo::new(&g.ctx, a, self.rule.planes(), 20u32.min(max_node_log2), 18u32.min(max_tile_log2))?,
      };
      self.gpu.as_mut().unwrap().memo = Some(m);
    }
    // the ID grid
    let need_grid = !matches!(&self.gpu.as_ref().unwrap().state, State::Ids(gr) if gr.lt as usize == lt);
    if need_grid {
      self.sync_host();
      // the host's results at the sweep's windows must be natural for the sweep's step
      if self.host.step_log2() < lt - 1 {
        self.host.set_step(lt - 1);
      }
      let r = self.root;
      let s = 1i64 << lt;
      let Some((x0, y0, x1, y1)) = self.host.bbox(r.id, r.level, r.x, r.y) else {
        return Ok(None);
      };
      let al = 1i64 << self.config.align_log2;
      // ten tiles: the rings that must stay clear (six) and the movement of one burst (four)
      let margin = 10 * s;
      let (gx0, gy0) = ((x0 - margin).div_euclid(al) * al, (y0 - margin).div_euclid(al) * al);
      let nx = ((x1 + margin - gx0) as u64).div_ceil(s as u64) as usize;
      let ny = ((y1 + margin - gy0) as u64).div_ceil(s as u64) as usize;
      let host_ids = sweep::host_ids(&mut self.host, (r.id, r.level, r.x, r.y), lt, gx0, gy0, nx, ny);
      let g = self.gpu.as_mut().unwrap();
      let memo = g.memo.as_mut().unwrap();
      memo.empties(&g.ctx, &mut self.host, lt + 1)?;
      // the empty windows' results, so that the sweep finds them
      let z = self.host.empty(lt + 1);
      let mut q = vec![z];
      self.host.solve(lt + 1, &mut q);
      memo.upload_results(&g.ctx, &self.host, lt + 1, &[z])?;
      let gpu_ids = memo.upload(&g.ctx, &self.host, lt, &host_ids)?;
      let empty = memo.empty[lt];
      let grid = IdGrid::new(&g.ctx, memo, lt as u32, nx as u32, ny as u32, gx0, gy0, &gpu_ids, empty)?;
      g.last_active = (nx * ny) as f64 * (s * s) as f64;
      g.state = State::Ids(grid);
    }
    // the miss path for this sweep level and the universe-wide floor
    {
      let floor = self.host.floor().min(lt + 1);
      let g = self.gpu.as_mut().unwrap();
      if g.gpu_miss && g.miss.as_ref().map(|m| m.top != lt + 1 || m.floor != floor.max(a + 1)).unwrap_or(true) {
        let geom = TileGeom::new(a as u32, self.rule.planes());
        g.miss = miss::MissPath::new(&g.ctx, g.memo.as_ref().unwrap(), &g.kernels, geom, lt, floor, 4096).ok();
      }
    }
    let t = Instant::now();
    let gens = n as u64 * per_sweep;
    let g = self.gpu.as_mut().unwrap();
    let State::Ids(grid) = &mut g.state else { unreachable!() };
    let memo = g.memo.as_mut().unwrap();
    let mut levels: Vec<LevelCounters> = Vec::new();
    let mut engine = EngineKind::GpuMemo;
    let mut resolved = false;
    if let Some(mp) = g.miss.as_mut() {
      for _ in 0..4 {
        let b = mp.burst(&g.ctx, memo, grid, &g.kernels, n, (lt - 1) as u32)?;
        if b.overflow.is_none() && b.unresolved == 0 {
          // the burst as a trace: the GPU's own counters per level
          let waves = b.levels.iter().filter(|x| x.3 > 0 || x.1 > 0).map(|&(l, misses, inserts, distinct)| crate::inspect::Wave { level: l, side: 1 << l, generations: 1 << (l.saturating_sub(2)).min(lt as u32 - 1), requests: distinct, distinct, misses, inserts, dispatches: b.dispatches, capacity: mp.cap.get(l as usize).copied().unwrap_or(0) as u64, micros: 0.0, parent: None }).collect();
          g.last_trace = Some(crate::inspect::StepTrace { generation: self.generation, step: n as u64 * per_sweep, engine: Some(EngineKind::GpuMemo), waves, atlases: Vec::new() });
          for (l, misses, inserts, distinct) in b.levels {
            if l as usize == a {
              memo.tile_count += inserts;
            } else {
              memo.node_count += inserts;
            }
            if distinct > 0 || misses > 0 {
              levels.push(LevelCounters { level: l, side: 1 << l, windows: distinct, distinct, misses, inserts, entries: 0, waves: 0 });
            }
          }
          resolved = true;
          break;
        }
        grid.restore(&g.ctx);
        g.overflows += 1;
      }
    }
    if !resolved {
      // the hit path with the host learning the misses (Stage 2)
      engine = EngineKind::GpuMemoHost;
      let b = grid.burst(&g.ctx, n);
      if b.misses > 0 {
        grid.restore(&g.ctx);
        let st = grid.learn_on_host(&g.ctx, memo, &mut self.host, n)?;
        let (w, d, m): (u64, u64, u64) = st.iter().fold((0, 0, 0), |a, s| (a.0 + s.0, a.1 + s.1, a.2 + s.2));
        levels.push(LevelCounters { level: lt as u32 + 1, side: 1 << (lt + 1), windows: w, distinct: d, misses: m, inserts: 0, entries: 0, waves: n as u64 });
      } else {
        levels.push(LevelCounters { level: lt as u32 + 1, side: 1 << (lt + 1), windows: b.windows, distinct: b.windows, misses: 0, inserts: 0, entries: 0, waves: n as u64 });
      }
    }
    let wall = t.elapsed();
    let windows = levels.iter().find(|l| l.level == lt as u32 + 1).map(|l| (l.windows, l.misses)).unwrap_or((0, 0));
    let area = g.last_active;
    // the outer rings must stay empty: grow through the host when they are not
    let ring = grid.ring_check(&g.ctx) > 0;
    // a globally periodic universe: IDs are canonical, so equal grids are equal universes
    if self.config.detect_cycles && (grid.nx as u64 * grid.ny as u64) <= 1 << 16 && self.sched.cycles.found.is_none() {
      let ids = grid.read_a(&g.ctx);
      let mut h = rustc_hash::FxHasher::default();
      std::hash::Hash::hash(&ids, &mut h);
      std::hash::Hash::hash(&(grid.x0, grid.y0, grid.lt), &mut h);
      let key = std::hash::Hasher::finish(&h);
      let now = self.generation + gens;
      match g.cycle_grids.get(&key) {
        Some((g0, old)) if *old == ids && now > *g0 => {
          self.sched.cycles.found = Some((now - g0, 0, 0));
          self.sched.cycles.at = now;
        }
        _ => {
          if g.cycle_grids.len() > 4096 {
            g.cycle_grids.clear();
          }
          g.cycle_grids.insert(key, (now, ids));
        }
      }
    }
    self.generation += gens;
    let cu = area * gens as f64;
    self.account(engine, gens, cu, wall);
    if let Some(new_lt) = self.sched.sweep.report(windows.0, windows.1) {
      let _ = new_lt;
      self.sync_host();
    } else if ring {
      self.sync_host();
    }
    let eu = EngineUse { engine, kernel: format!("sweep {0}x{0}", 1u32 << lt), tile: (1 << lt, 1 << lt), k: (1 << (lt - 1)) as u32, generations: gens, cell_updates: cu, micros: wall.as_secs_f64() * 1e6 };
    let s = self.sample(Regime::Memo, gens, vec![eu], levels, wall);
    hub.publish(s);
    hub.serve(self);
    Ok(Some((gens, engine, cu, wall)))
  }
}
