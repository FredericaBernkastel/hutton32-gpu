//! What a front end asks of a universe: rectangles of cells, per-tile summaries for zoomed-out
//! views, fingerprints, and the statistics of a run.

pub use crate::cpu::Fingerprint;

/// A rectangle of cells: `[x, x + w) x [y, y + h)`, y downwards.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct Rect {
  pub x: i64,
  pub y: i64,
  pub w: u64,
  pub h: u64,
}

impl Rect {
  pub fn new(x: i64, y: i64, w: u64, h: u64) -> Rect {
    Rect { x, y, w, h }
  }
  /// The rectangle `[x0, x1) x [y0, y1)`.
  pub fn from_corners(x0: i64, y0: i64, x1: i64, y1: i64) -> Rect {
    Rect { x: x0, y: y0, w: (x1 - x0).max(0) as u64, h: (y1 - y0).max(0) as u64 }
  }
  pub fn x1(&self) -> i64 {
    self.x + self.w as i64
  }
  pub fn y1(&self) -> i64 {
    self.y + self.h as i64
  }
  pub fn area(&self) -> u64 {
    self.w * self.h
  }
  pub fn is_empty(&self) -> bool {
    self.w == 0 || self.h == 0
  }
  pub fn contains(&self, x: i64, y: i64) -> bool {
    x >= self.x && y >= self.y && x < self.x1() && y < self.y1()
  }
  pub fn intersect(&self, o: &Rect) -> Rect {
    Rect::from_corners(self.x.max(o.x), self.y.max(o.y), self.x1().min(o.x1()), self.y1().min(o.y1()))
  }
  pub fn union(&self, o: &Rect) -> Rect {
    if self.is_empty() {
      return *o;
    }
    if o.is_empty() {
      return *self;
    }
    Rect::from_corners(self.x.min(o.x), self.y.min(o.y), self.x1().max(o.x1()), self.y1().max(o.y1()))
  }
  /// The tiles of side `2^level` (on the absolute lattice) that the rectangle touches: their
  /// columns and rows, `(tx0, ty0, nx, ny)`.
  pub fn tiles(&self, level: u32) -> (i64, i64, usize, usize) {
    let s = 1i64 << level;
    let (tx0, ty0) = (self.x.div_euclid(s), self.y.div_euclid(s));
    let (tx1, ty1) = ((self.x1() - 1).div_euclid(s) + 1, (self.y1() - 1).div_euclid(s) + 1);
    if self.is_empty() {
      return (tx0, ty0, 0, 0);
    }
    (tx0, ty0, (tx1 - tx0) as usize, (ty1 - ty0) as usize)
  }
}

/// One tile of a zoomed-out view: its live cells and the mean colour of its cells (dead cells in
/// the colour of state 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct TileSummary {
  pub population: u64,
  pub color: [u8; 3],
}

/// Which engine did a piece of work.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum EngineKind {
  /// the host recursion (`cpu::hl`)
  Host,
  /// one generation at a time on the host's bit-planes (odd generations of alternating rules,
  /// edits that need a dense step)
  HostCells,
  /// the GPU plain kernel over a dense grid
  GpuPlain,
  /// the GPU shared-tile kernel, `k` generations per dispatch
  GpuSharedTile,
  /// the GPU hit path (sweep) with misses learned by the host
  GpuMemoHost,
  /// the GPU hit path with the GPU miss path
  GpuMemo,
  /// a recognised cycle: generations skipped by a lookup
  Cycle,
}

impl EngineKind {
  pub fn name(&self) -> &'static str {
    match self {
      EngineKind::Host => "host recursion",
      EngineKind::HostCells => "host cells",
      EngineKind::GpuPlain => "plain kernel",
      EngineKind::GpuSharedTile => "shared-tile kernel",
      EngineKind::GpuMemoHost => "memo, host learning",
      EngineKind::GpuMemo => "memo",
      EngineKind::Cycle => "cycle",
    }
  }
  pub fn on_gpu(&self) -> bool {
    matches!(self, EngineKind::GpuPlain | EngineKind::GpuSharedTile | EngineKind::GpuMemoHost | EngineKind::GpuMemo)
  }
}

/// The work of one engine within a call to `step`.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct EngineTime {
  pub generations: u64,
  /// cell-updates the generations stand for: the bounding box's cells times generations
  pub cell_updates: f64,
  pub wall: std::time::Duration,
}

/// What a call to `step` or `step_for` did.
#[derive(Clone, Debug, Default)]
pub struct StepReport {
  pub generations: u64,
  pub generation: u64,
  /// cell-updates over the pattern's bounding box
  pub cell_updates: f64,
  pub wall: std::time::Duration,
  pub epochs: u32,
  pub by_engine: Vec<(EngineKind, EngineTime)>,
}

impl StepReport {
  pub fn cell_updates_per_s(&self) -> f64 {
    self.cell_updates / self.wall.as_secs_f64().max(1e-12)
  }
  pub fn gens_per_s(&self) -> f64 {
    self.generations as f64 / self.wall.as_secs_f64().max(1e-12)
  }
  pub(crate) fn add(&mut self, e: EngineKind, gens: u64, cu: f64, wall: std::time::Duration) {
    self.generations += gens;
    self.cell_updates += cu;
    match self.by_engine.iter_mut().find(|x| x.0 == e) {
      Some(x) => {
        x.1.generations += gens;
        x.1.cell_updates += cu;
        x.1.wall += wall;
      }
      None => self.by_engine.push((e, EngineTime { generations: gens, cell_updates: cu, wall })),
    }
  }
}

/// Memory held by the engines, in bytes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Memory {
  /// host tiles, nodes and tables
  pub host: u64,
  /// GPU cell grids or ID grids
  pub grids: u64,
  /// GPU memo tables
  pub tables: u64,
  /// GPU tile store
  pub tile_store: u64,
  /// GPU atlas and miss-path buffers
  pub atlas: u64,
}

/// Statistics of a universe: engines, hit rates, memory, overflows.
#[derive(Clone, Debug, Default)]
pub struct Stats {
  pub generation: u64,
  pub population: u64,
  pub engine: Option<EngineKind>,
  /// generations, cell-updates and time per engine since the universe was made
  pub engines: Vec<(EngineKind, EngineTime)>,
  /// per level of the recursion: `(level, requests, misses)` since the universe was made
  pub levels: Vec<(u32, u64, u64)>,
  pub host_nodes: u64,
  pub host_tiles: u64,
  pub gpu_entries: u64,
  pub memory: Memory,
  pub overflows: u64,
  pub reclaims: u64,
  /// the step in generations and the base tile's side
  pub step: u64,
  pub base_tile: u32,
  /// a period found by cycle detection: `(period, dx, dy)`
  pub cycle: Option<(u64, i64, i64)>,
}

impl Stats {
  /// Share of requests that hit, over all levels.
  pub fn hit_rate(&self) -> f64 {
    let (r, m) = self.levels.iter().fold((0u64, 0u64), |a, l| (a.0 + l.1, a.1 + l.2));
    if r == 0 {
      0.0
    } else {
      1.0 - m as f64 / r as f64
    }
  }
}
