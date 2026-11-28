//! Introspection: the engine's own data structures and counters, read while it runs (paper's
//! overview figure, live). Three kinds of data at three costs:
//!
//! * **counters**, published once per epoch as an immutable [`Sample`] and kept in a ring as
//!   history: always on;
//! * **maps** (IDs, the outcome of the last sweep, the engine per region, memo slots): answered at
//!   once when the universe is not stepping, else at its next epoch boundary ([`Pending`]);
//! * **traces**: one step wave by wave ([`StepTrace`]), recorded by the engine as it steps.
//!
//! Debug data are read from the engine's own structures and never recomputed by a second
//! implementation; the one exception is the signal trace of the rule's network, a pure function of
//! the rule ([`crate::rule::Network::trace`]).

use std::{
  ops::Range,
  sync::{Arc, Condvar, Mutex, RwLock, Weak},
};

use crate::{
  query::{Memory, Rect},
  sched::{FloorRow, Rates, Regime},
  tile::TileId,
  universe::Core,
};

pub use crate::query::EngineKind;

bitflags::bitflags! {
  /// Which maps are kept current at epoch boundaries (they cost a read-back on the GPU).
  #[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
  pub struct Taps: u32 {
    const IDS = 1;
    const OUTCOMES = 2;
    const REGIONS = 4;
    const MEMO = 8;
  }
}

/// The work of one engine in an epoch.
#[derive(Clone, Debug, PartialEq)]
pub struct EngineUse {
  pub engine: EngineKind,
  /// the kernel, e.g. `life-adder-tree/shared-tile`
  pub kernel: String,
  /// cells per tile of the kernel (shared tile) or of the base tile
  pub tile: (u32, u32),
  /// generations per dispatch
  pub k: u32,
  pub generations: u64,
  pub cell_updates: f64,
  pub micros: f64,
}

/// Counters of one level in an epoch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LevelCounters {
  pub level: u32,
  pub side: u32,
  /// windows asked for, before deduplication
  pub windows: u64,
  /// distinct windows (requests after per-wave deduplication)
  pub distinct: u64,
  pub misses: u64,
  pub inserts: u64,
  /// entries of the level's table
  pub entries: u64,
  pub waves: u64,
}

/// How deep the recursion may go (Algorithm 1, lines 2–6).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Depth {
  pub active_cells: u64,
  pub a1: f64,
  pub a2: f64,
  /// the universe-wide floor: the side of the brute-forced windows, 0 for plain cells
  pub floor: u32,
  pub dispatch_us: f64,
  /// the engines raced, with their last rates in generations/s
  pub raced: Vec<(Regime, f64)>,
}

/// The counters of one epoch.
#[derive(Clone, Debug, PartialEq)]
pub struct Sample {
  pub epoch: u64,
  pub generation: u64,
  /// generations of this epoch
  pub generations: u64,
  /// the step in generations, and the region's side
  pub step: u32,
  pub region_side: u32,
  pub regime: Regime,
  pub engines: Vec<EngineUse>,
  pub levels: Vec<LevelCounters>,
  pub depth: Depth,
  pub overflows: u32,
  pub memory: Memory,
  pub population: u64,
  pub micros: f64,
}

impl Default for Sample {
  fn default() -> Self {
    Sample {
      epoch: 0,
      generation: 0,
      generations: 0,
      step: 0,
      region_side: 0,
      regime: Regime::HostRecursion,
      engines: Vec::new(),
      levels: Vec::new(),
      depth: Depth::default(),
      overflows: 0,
      memory: Memory::default(),
      population: 0,
      micros: 0.0,
    }
  }
}

/// The constants a run is scheduled with.
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
  pub rates: Rates,
  /// the brute-force kernel in use
  pub kernel: String,
  /// R_B per kernel measured, cell-updates/s
  pub r_b: Vec<(String, f64)>,
  /// R_P by table size: `(bytes, probes/s)`
  pub r_p: Vec<(u64, f64)>,
  pub rho: f64,
  /// the base tile's side and the floor window's side
  pub base_tile: u32,
  pub floor_window: u32,
  pub a1: f64,
  pub a2: f64,
}

/// A level of the universe's quadtree: its side, non-empty and distinct tiles.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LevelInfo {
  pub level: u32,
  pub side: u64,
  pub nonempty: u64,
  pub distinct: u64,
  /// nodes (or tiles) the level's table holds, and results memoized
  pub entries: u64,
  pub results: u64,
  /// the empty node's ID (the empty tile at the base level), once it exists
  pub empty: Option<TileId>,
}

/// A tile's content: cells of a base tile, or the four children of a node.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Tile {
  Cells { side: u32, cells: Vec<u8> },
  Node { level: u32, children: [TileId; 4], population: u64 },
}

/// What the last sweep did at a tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
  Hit,
  Miss,
  Empty,
}

/// The engine of a region, its prices and the time charged (Algorithm 3's grid).
#[derive(Clone, Debug, PartialEq)]
pub struct RegionInfo {
  pub x: i64,
  pub y: i64,
  pub side: u32,
  pub engine: RegionEngine,
  /// steps spent at the floor and in plain cells
  pub steps: [u32; 2],
  pub price_floor_ns: f64,
  pub price_cells_ns: f64,
  pub charged_ns: f64,
  pub switches: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegionEngine {
  /// missed windows of this side are brute-forced
  Floor(u32),
  Cells,
}

/// Occupancy and probe lengths of a memo table.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MemoStats {
  pub level: u32,
  pub entries: u64,
  pub slots: u64,
  pub bytes: u64,
  pub load: f64,
  pub avg_probes: f64,
  /// entries by displacement from their home slot (0..=31, the last bucket for 31 and more)
  pub displacement: Vec<u64>,
}

/// A slot of a memo table as it lies in memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotState {
  Empty,
  /// claimed in the current batch by element `j`
  Claimed(u32),
  Committed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
  pub state: SlotState,
  pub key: [TileId; 4],
  pub value: TileId,
  /// distance from the home slot
  pub displacement: u32,
}

/// A memo entry: four IDs give one ID.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoEntry {
  pub level: u32,
  pub key: [TileId; 4],
  pub id: TileId,
  /// the result, `None` while not computed
  pub result: Option<TileId>,
  pub slot: u64,
}

/// A word of 32 cells and its neighbour words, per bit-plane, as the bit-sliced kernel sees them.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Planes {
  /// the word's first cell
  pub x: i64,
  pub y: i64,
  /// per plane: the words at `[C, N, NE, E, SE, S, SW, W, NW]` (bit i = the neighbour of cell `x + i`)
  pub words: Vec<[u32; 9]>,
  /// the cells' states
  pub cells: Vec<u8>,
}

/// One wave of a step: a batch of distinct windows at one level.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Wave {
  pub level: u32,
  pub side: u32,
  /// generations the level's result stands for
  pub generations: u32,
  pub requests: u64,
  pub distinct: u64,
  pub misses: u64,
  pub inserts: u64,
  pub dispatches: u32,
  pub capacity: u64,
  pub micros: f32,
  /// the wave whose misses asked for this one
  pub parent: Option<u32>,
}

/// One window of an atlas: its cells before and its centre after.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AtlasWindow {
  pub id: TileId,
  pub cells: Vec<u8>,
  pub result: Vec<u8>,
}

/// The atlas of a brute-force wave: the missed windows side by side.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Atlas {
  pub wave: u32,
  pub level: u32,
  pub side: u32,
  pub generations: u32,
  pub windows: Vec<AtlasWindow>,
}

/// One step, wave by wave.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StepTrace {
  pub generation: u64,
  /// generations of the step
  pub step: u64,
  pub engine: Option<EngineKind>,
  pub waves: Vec<Wave>,
  pub atlases: Vec<Atlas>,
}

impl StepTrace {
  /// The atlas of wave `wave`, if it brute-forced windows.
  pub fn atlas(&self, wave: u32) -> Option<&Atlas> {
    self.atlases.iter().find(|a| a.wave == wave)
  }
  /// The largest atlas of the step.
  pub fn largest_atlas(&self) -> Option<&Atlas> {
    self.atlases.iter().max_by_key(|a| a.windows.len())
  }
  /// Per level: `(level, distinct requests, misses, waves)`.
  pub fn totals(&self) -> Vec<(u32, u64, u64, u64)> {
    let mut t: Vec<(u32, u64, u64, u64)> = Vec::new();
    for w in &self.waves {
      match t.iter_mut().find(|x| x.0 == w.level) {
        Some(x) => {
          x.1 += w.distinct;
          x.2 += w.misses;
          x.3 += 1;
        }
        None => t.push((w.level, w.distinct, w.misses, 1)),
      }
    }
    t.sort_by_key(|x| x.0);
    t
  }
}

// ---- pending answers --------------------------------------------------------------------------

struct Slot0<T> {
  value: Mutex<Option<T>>,
  cv: Condvar,
}

/// An answer that is filled at the next epoch boundary, or at once on a universe that is not
/// stepping. `poll` from any thread, `wait` from a thread other than the universe's.
pub struct Pending<T>(Arc<Slot0<T>>);

impl<T> Clone for Pending<T> {
  fn clone(&self) -> Self {
    Pending(self.0.clone())
  }
}

impl<T: Clone> Pending<T> {
  fn new() -> Pending<T> {
    Pending(Arc::new(Slot0 { value: Mutex::new(None), cv: Condvar::new() }))
  }
  fn ready(v: T) -> Pending<T> {
    let p = Pending::new();
    p.fill(v);
    p
  }
  pub(crate) fn fill(&self, v: T) {
    *self.0.value.lock().unwrap() = Some(v);
    self.0.cv.notify_all();
  }
  /// The answer, if it is there.
  pub fn poll(&self) -> Option<T> {
    self.0.value.lock().unwrap().clone()
  }
  pub fn is_ready(&self) -> bool {
    self.0.value.lock().unwrap().is_some()
  }
  /// Block until the answer is there (from a thread other than the universe's).
  pub fn wait(&self) -> T {
    let mut g = self.0.value.lock().unwrap();
    while g.is_none() {
      g = self.0.cv.wait(g).unwrap();
    }
    g.clone().unwrap()
  }
  /// Block at most `timeout`.
  pub fn wait_timeout(&self, timeout: std::time::Duration) -> Option<T> {
    let g = self.0.value.lock().unwrap();
    let (g, _) = self.0.cv.wait_timeout_while(g, timeout, |v| v.is_none()).unwrap();
    g.clone()
  }
}

/// A map request queued while the universe steps.
pub(crate) enum Request {
  Ids(Rect, u32, Pending<Vec<TileId>>),
  Tile(u32, TileId, Pending<Option<Tile>>),
  Outcomes(Rect, Pending<Vec<Outcome>>),
  Regions(Rect, Pending<Vec<RegionInfo>>),
  MemoMap(u32, Range<u64>, Pending<Vec<Slot>>),
  MemoLookup(u32, [TileId; 4], Pending<Option<MemoEntry>>),
  Planes(i64, i64, Pending<Planes>),
  LevelInfo(u32, Pending<LevelInfo>),
  MemoStats(u32, Pending<MemoStats>),
}

impl Request {
  /// Answer from the engine's structures.
  pub(crate) fn answer(self, core: &Core) {
    match self {
      Request::Ids(r, l, p) => p.fill(core.ids(r, l)),
      Request::Tile(l, id, p) => p.fill(core.tile(l, id)),
      Request::Outcomes(r, p) => p.fill(core.outcomes(r)),
      Request::Regions(r, p) => p.fill(core.regions(r)),
      Request::MemoMap(l, s, p) => p.fill(core.memo_map(l, s)),
      Request::MemoLookup(l, k, p) => p.fill(core.memo_lookup(l, k)),
      Request::Planes(x, y, p) => p.fill(core.read_planes(x, y)),
      Request::LevelInfo(l, p) => p.fill(core.level_info(l)),
      Request::MemoStats(l, p) => p.fill(core.memo_stats(l)),
    }
  }
}

/// History of samples: a ring, thinned to log-spaced generations when full.
pub(crate) struct History {
  samples: Vec<Arc<Sample>>,
  cap: usize,
}

impl History {
  fn push(&mut self, s: Arc<Sample>) {
    self.samples.push(s);
    if self.samples.len() > self.cap {
      // keep every other sample of the older half: older generations get sparser
      let half = self.samples.len() / 2;
      let mut kept: Vec<Arc<Sample>> = self.samples[..half].iter().step_by(2).cloned().collect();
      kept.extend_from_slice(&self.samples[half..]);
      self.samples = kept;
    }
  }
}

/// What the universe shares with its observers.
pub(crate) struct Hub {
  pub sample: RwLock<Arc<Sample>>,
  pub history: Mutex<History>,
  pub calibration: RwLock<Arc<Calibration>>,
  pub floor_table: RwLock<Vec<FloorRow>>,
  pub requests: Mutex<Vec<Request>>,
  pub level_cache: Mutex<rustc_hash::FxHashMap<u32, LevelInfo>>,
  pub memo_cache: Mutex<rustc_hash::FxHashMap<u32, MemoStats>>,
}

impl Hub {
  pub fn new(calibration: Calibration, floor_table: Vec<FloorRow>, history: usize) -> Hub {
    Hub {
      sample: RwLock::new(Arc::new(Sample::default())),
      history: Mutex::new(History { samples: Vec::new(), cap: history.max(16) }),
      calibration: RwLock::new(Arc::new(calibration)),
      floor_table: RwLock::new(floor_table),
      requests: Mutex::new(Vec::new()),
      level_cache: Mutex::new(Default::default()),
      memo_cache: Mutex::new(Default::default()),
    }
  }
  pub fn publish(&self, s: Sample) {
    let s = Arc::new(s);
    *self.sample.write().unwrap() = s.clone();
    self.history.lock().unwrap().push(s);
  }
  pub fn clear_history(&self) {
    self.history.lock().unwrap().samples.clear();
  }
  /// Answer every queued request.
  pub fn serve(&self, core: &Core) {
    let reqs: Vec<Request> = std::mem::take(&mut *self.requests.lock().unwrap());
    for r in reqs {
      match r {
        Request::LevelInfo(l, p) => {
          let v = core.level_info(l);
          self.level_cache.lock().unwrap().insert(l, v);
          p.fill(v);
        }
        Request::MemoStats(l, p) => {
          let v = core.memo_stats(l);
          self.memo_cache.lock().unwrap().insert(l, v.clone());
          p.fill(v);
        }
        r => r.answer(core),
      }
    }
  }
}

/// The handle of a UI thread: `Clone + Send + Sync`; reading never blocks stepping and stepping
/// never waits for it.
#[derive(Clone)]
pub struct Observer {
  pub(crate) hub: Arc<Hub>,
  pub(crate) core: Weak<RwLock<Core>>,
}

impl Observer {
  /// Counters of the last finished epoch.
  pub fn sample(&self) -> Arc<Sample> {
    self.hub.sample.read().unwrap().clone()
  }
  /// Every published sample, oldest first, thinned to log-spaced generations when the ring is full.
  pub fn history(&self) -> Vec<Arc<Sample>> {
    self.hub.history.lock().unwrap().samples.clone()
  }
  /// R_B per kernel, R_P by table size, R_I, latencies; `rho`, the floor and the thresholds.
  pub fn calibration(&self) -> Arc<Calibration> {
    self.hub.calibration.read().unwrap().clone()
  }
  /// The paper's Table II for this run's `rho`: window, V, V / rho, memoized or not.
  pub fn floor_table(&self) -> Vec<FloorRow> {
    self.hub.floor_table.read().unwrap().clone()
  }

  /// Answer now if the universe is not stepping, else at its next epoch boundary.
  fn ask<T: Clone>(&self, make: impl FnOnce(Pending<T>) -> Request, now: impl FnOnce(&Core) -> T) -> Pending<T> {
    let Some(core) = self.core.upgrade() else {
      // the universe is gone: the answer never comes; hand back an unfilled handle
      return Pending::new();
    };
    if let Ok(c) = core.try_read() {
      return Pending::ready(now(&c));
    }
    let p = Pending::new();
    self.hub.requests.lock().unwrap().push(make(p.clone()));
    p
  }

  /// Side, non-empty and distinct tiles of a level: computed now when the universe is not
  /// stepping, else the value of its last epoch boundary.
  pub fn level_info(&self, level: u32) -> LevelInfo {
    if let Some(core) = self.core.upgrade() {
      if let Ok(c) = core.try_read() {
        let v = c.level_info(level);
        self.hub.level_cache.lock().unwrap().insert(level, v);
        return v;
      }
      let p = Pending::new();
      self.hub.requests.lock().unwrap().push(Request::LevelInfo(level, p));
    }
    self.hub.level_cache.lock().unwrap().get(&level).copied().unwrap_or(LevelInfo { level, side: 1u64 << level, ..Default::default() })
  }
  /// IDs of the tiles of side `2^level` that the rectangle touches, row-major over
  /// [`Rect::tiles`].
  pub fn ids(&self, rect: Rect, level: u32) -> Pending<Vec<TileId>> {
    self.ask(|p| Request::Ids(rect, level, p), |c| c.ids(rect, level))
  }
  /// Cells of a base tile, or the four children of a node.
  pub fn tile(&self, level: u32, id: TileId) -> Pending<Option<Tile>> {
    self.ask(|p| Request::Tile(level, id, p), |c| c.tile(level, id))
  }
  /// Hit, Miss or Empty per window of the last sweep (or of the sweep that would run now).
  pub fn outcomes(&self, rect: Rect) -> Pending<Vec<Outcome>> {
    self.ask(|p| Request::Outcomes(rect, p), |c| c.outcomes(rect))
  }
  pub fn regions(&self, rect: Rect) -> Pending<Vec<RegionInfo>> {
    self.ask(|p| Request::Regions(rect, p), |c| c.regions(rect))
  }
  /// Entries, slots, bytes and the probe-length histogram of a level's table: computed now when
  /// the universe is not stepping, else the value of its last epoch boundary.
  pub fn memo_stats(&self, level: u32) -> MemoStats {
    if let Some(core) = self.core.upgrade() {
      if let Ok(c) = core.try_read() {
        let v = c.memo_stats(level);
        self.hub.memo_cache.lock().unwrap().insert(level, v.clone());
        return v;
      }
      let p = Pending::new();
      self.hub.requests.lock().unwrap().push(Request::MemoStats(level, p));
    }
    self.hub.memo_cache.lock().unwrap().get(&level).cloned().unwrap_or(MemoStats { level, ..Default::default() })
  }
  pub fn memo_map(&self, level: u32, slots: Range<u64>) -> Pending<Vec<Slot>> {
    let s2 = slots.clone();
    self.ask(|p| Request::MemoMap(level, s2, p), |c| c.memo_map(level, slots))
  }
  pub fn memo_lookup(&self, level: u32, key: [TileId; 4]) -> Pending<Option<MemoEntry>> {
    self.ask(|p| Request::MemoLookup(level, key, p), |c| c.memo_lookup(level, key))
  }
  /// A word of 32 cells at `(x, y)` (x rounded down to a multiple of 32) and its neighbour words.
  pub fn read_planes(&self, x: i64, y: i64) -> Pending<Planes> {
    self.ask(|p| Request::Planes(x, y, p), |c| c.read_planes(x, y))
  }
}
