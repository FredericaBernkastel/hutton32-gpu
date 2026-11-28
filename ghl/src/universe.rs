//! The universe: a rule, its cells and the engines that advance them, behind the interface of
//! Section 7 of the roadmap. The state lives either in the host recursion's quadtree or on the
//! GPU; every epoch the scheduler chooses the engine (paper Algorithm 1), counters are published
//! to the observers, and queued map requests are answered.

use std::{
  ops::Range,
  sync::{Arc, RwLock},
  time::{Duration, Instant},
};

use rustc_hash::FxHashSet;

use crate::{
  cpu::{self, reclaim, BruteWave, Engine, LevelStats},
  inspect::{self, Calibration, Depth, EngineUse, Hub, LevelCounters, LevelInfo, MemoEntry, MemoStats, Observer, Planes, RegionEngine, RegionInfo, Sample, Slot, SlotState, StepTrace, Taps, Tile, Wave},
  io::{write_rle, Format, Grid, Pattern},
  query::{EngineKind, EngineTime, Fingerprint, Memory, Rect, Stats, StepReport, TileSummary},
  rule::Rule,
  sched::{cost, Controller, KernelClass, Mode, Policy, PolicyKind, Rates, Regime, Scheduler},
  tile::{self, TileId, NONE},
  Error, Result,
};

/// Which engines a universe may use.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Engines {
  /// the GPU when one is available (feature `gpu`), the host recursion always
  #[default]
  Auto,
  /// the host recursion only
  Cpu,
  /// require a GPU
  Gpu,
}

/// Settings of a universe. The defaults are the paper's choices; every threshold the scheduler
/// uses is calibrated at start-up when `calibrate` is set.
#[derive(Clone, Debug)]
pub struct Config {
  /// log2 of the base tile's side; `None`: from the rule's `rho` (8×8 cells for Hutton32b and
  /// Hutton32, 16×16 for Life-like rules)
  pub base_tile_log2: Option<u32>,
  pub engines: Engines,
  /// run one regime only; `None`: Algorithm 1 chooses per epoch
  pub regime: Option<Regime>,
  /// the memo engine's step, log2 of generations; `None`: the step rule (one base tile's side
  /// while content is new, longer where the memo carries the universe)
  pub step_log2: Option<u32>,
  /// the host recursion's step when it runs alone, log2; `None`: adapted to `epoch_target`
  pub host_step_log2: Option<u32>,
  /// the engine per region; the uniform floor by default (see [`Controller::Uniform`])
  pub controller: Controller,
  /// measure the scheduler's constants at start-up (GPU)
  pub calibrate: bool,
  /// race the host recursion and the dense kernels in the latency-bound regime
  pub race: bool,
  pub detect_cycles: bool,
  /// reclaim the host's nodes above this many bytes
  pub host_memory_limit: u64,
  /// samples kept in the observers' history
  pub history: usize,
  /// compile kernels that passed their cell-exact tests without naga's runtime checks
  pub trusted_kernels: bool,
  /// the wall time an epoch aims at
  pub epoch_target: Duration,
  /// record every host step wave by wave for `trace_step`
  pub trace: bool,
  /// log2 of the lattice the root stays aligned to (tile IDs, regions and sweep tiles up to this
  /// size sit on the absolute lattice)
  pub align_log2: u32,
  /// resolve the memo engine's misses on the GPU (Stage 3); off: the host learns them (Stage 2)
  pub gpu_miss_path: bool,
  /// split the GPU's node table into shards of this many entries (log2); `None`: as large as one
  /// binding allows
  pub gpu_table_shard_log2: Option<u32>,
  /// split dense GPU grids above this many words into shards; `None`: as large as one binding
  pub gpu_grid_shard_words: Option<u64>,
}

impl Default for Config {
  fn default() -> Self {
    Config {
      base_tile_log2: None,
      engines: Engines::Auto,
      regime: None,
      step_log2: None,
      host_step_log2: None,
      controller: Controller::Uniform,
      calibrate: true,
      race: true,
      detect_cycles: true,
      host_memory_limit: 8 << 30,
      history: 4096,
      trusted_kernels: true,
      epoch_target: Duration::from_millis(8),
      trace: true,
      align_log2: 10,
      gpu_miss_path: true,
      gpu_table_shard_log2: None,
      gpu_grid_shard_words: None,
    }
  }
}

impl Config {
  /// The host recursion only.
  pub fn cpu() -> Config {
    Config { engines: Engines::Cpu, calibrate: false, ..Config::default() }
  }
  /// The host recursion with a fixed step of `2^k` generations.
  pub fn cpu_step(k: u32) -> Config {
    Config { host_step_log2: Some(k), ..Config::cpu() }
  }
}

/// The root of the host's quadtree: the node, its level and its top-left cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Root {
  pub id: u32,
  pub level: usize,
  pub x: i64,
  pub y: i64,
}

/// The last host step, as recorded (waves and the windows of every brute-force wave).
#[derive(Clone, Debug)]
struct HostTrace {
  generation: u64,
  k: usize,
  waves: Vec<[u32; 4]>,
  brute: Vec<BruteWave>,
  inserts: Vec<u64>,
}

/// Everything behind the universe's lock.
pub struct Core {
  pub(crate) rule: Rule,
  pub(crate) config: Config,
  pub(crate) host: Engine,
  pub(crate) root: Root,
  pub(crate) generation: u64,
  pub(crate) sched: Scheduler,
  pub(crate) policy: Option<Policy>,
  pub(crate) taps: Taps,
  trace: Option<HostTrace>,
  gpu_trace: Option<StepTrace>,
  totals: Vec<(EngineKind, EngineTime)>,
  epoch: u64,
  reclaims: u64,
  overflows: u64,
  last_engine: Option<EngineKind>,
  policy_switches: u64,
  /// the current call steps a fixed number of generations (no time budget)
  bulk: bool,
  /// active cells when Algorithm 1 last looked
  active: f64,
  #[cfg(feature = "gpu")]
  pub(crate) gpu: Option<crate::gpu::GpuCore>,
}

/// A universe of one rule.
pub struct Universe {
  core: Arc<RwLock<Core>>,
  hub: Arc<Hub>,
}

fn kernel_class(rule: &Rule) -> KernelClass {
  if rule.life_like_masks().is_some() && rule.network().is_some() {
    KernelClass::LifeLike
  } else if rule.network().is_some() {
    KernelClass::Circuit
  } else if rule.table().is_some() {
    KernelClass::ByteLut
  } else {
    KernelClass::Tree
  }
}

impl Universe {
  /// A universe of `rule`, with a GPU of its own when `config` allows and one is available.
  pub fn new(rule: Rule, config: Config) -> Result<Universe> {
    #[cfg(feature = "gpu")]
    {
      if config.engines != Engines::Cpu {
        match crate::gpu::GpuContext::new() {
          Ok(ctx) => return Universe::with_gpu(rule, config, ctx),
          Err(e) if config.engines == Engines::Gpu => return Err(e),
          Err(_) => {}
        }
      }
    }
    if config.engines == Engines::Gpu {
      return Err(Error::Gpu("this build has no GPU support (feature \"gpu\")".into()));
    }
    Universe::build(rule, config, None)
  }

  /// A universe on a GPU the caller owns (a shared `Device` and `Queue`).
  #[cfg(feature = "gpu")]
  pub fn with_gpu(rule: Rule, config: Config, gpu: crate::gpu::GpuContext) -> Result<Universe> {
    Universe::build(rule, config, Some(gpu))
  }

  #[cfg(feature = "gpu")]
  fn build(rule: Rule, config: Config, gpu: Option<crate::gpu::GpuContext>) -> Result<Universe> {
    let class = kernel_class(&rule);
    let mut rates = Rates::paper(class);
    let mut calib_extra = (Vec::new(), Vec::new(), String::new());
    let gpu = match gpu {
      Some(ctx) => {
        let g = crate::gpu::GpuCore::new(ctx, &rule, &config)?;
        if config.calibrate {
          let c = g.calibrate()?;
          rates = c.rates.clone();
          calib_extra = (c.r_b.clone(), c.r_p.clone(), c.kernel.clone());
        } else {
          calib_extra.2 = g.kernel_name();
        }
        Some(g)
      }
      None => None,
    };
    let u = Universe::assemble(rule, config, rates, calib_extra, gpu.is_some())?;
    if let Some(mut g) = gpu {
      let mut core = u.core.write().unwrap();
      g.set_base_tile(core.host.a as u32);
      core.gpu = Some(g);
    }
    Ok(u)
  }

  #[cfg(not(feature = "gpu"))]
  fn build(rule: Rule, config: Config, _gpu: Option<()>) -> Result<Universe> {
    let rates = Rates::paper(kernel_class(&rule));
    Universe::assemble(rule, config, rates, (Vec::new(), Vec::new(), String::new()), false)
  }

  fn assemble(rule: Rule, config: Config, rates: Rates, extra: (Vec<(String, f64)>, Vec<(u64, f64)>, String), has_gpu: bool) -> Result<Universe> {
    let class = kernel_class(&rule);
    let mut a = config.base_tile_log2.unwrap_or_else(|| rates.base_tile_log2().clamp(3, 6));
    if rule.alternates() {
      a = a.max(2);
    }
    if !(1..=6).contains(&a) {
      return Err(Error::Config(format!("base tiles of 2x2 to 64x64 cells, not 2^{a}")));
    }
    let racing: Vec<Regime> = if has_gpu { vec![Regime::HostRecursion, Regime::SharedTile, Regime::PlainCells] } else { vec![Regime::HostRecursion] };
    let mut sched = Scheduler::new(rates.clone(), class, a, &racing, config.controller);
    if let Some(k) = config.host_step_log2 {
      sched.host_step_log2 = k;
    }
    if let Some(k) = config.step_log2 {
      sched.step.step_log2 = k;
    }
    let k0 = sched.host_step_log2 as usize;
    let host = Engine::new(&rule, a as usize, k0);
    // the empty tile as the root: nothing is interned before the first pattern, so that a pattern
    // loaded into a fresh universe gets the IDs the harness's engine gives it
    let root = Root { id: 0, level: a as usize, x: 0, y: 0 };
    let rho = rates.rho();
    let calibration = Calibration {
      kernel: if extra.2.is_empty() { format!("{class:?}") } else { extra.2.clone() },
      r_b: extra.0,
      r_p: extra.1,
      rho,
      base_tile: 1 << a,
      floor_window: 2 << a,
      a1: sched.thresholds.a1,
      a2: sched.thresholds.a2,
      rates,
    };
    let hub = Arc::new(Hub::new(calibration, cost::floor_table(rho, a), config.history));
    let policy = make_policy(&config, &sched, a as usize);
    let core = Core {
      rule,
      config,
      host,
      root,
      generation: 0,
      sched,
      policy,
      taps: Taps::empty(),
      trace: None,
      gpu_trace: None,
      totals: Vec::new(),
      epoch: 0,
      reclaims: 0,
      overflows: 0,
      last_engine: None,
      policy_switches: 0,
      bulk: true,
      active: 0.0,
      #[cfg(feature = "gpu")]
      gpu: None,
    };
    Ok(Universe { core: Arc::new(RwLock::new(core)), hub })
  }

  fn core(&self) -> std::sync::RwLockReadGuard<'_, Core> {
    self.core.read().unwrap()
  }
  fn core_mut(&mut self) -> std::sync::RwLockWriteGuard<'_, Core> {
    self.core.write().unwrap()
  }

  pub fn rule(&self) -> Rule {
    self.core().rule.clone()
  }
  pub fn config(&self) -> Config {
    self.core().config.clone()
  }
  /// The base tile's side, log2.
  pub fn base_tile_log2(&self) -> u32 {
    self.core().host.a as u32
  }
  pub fn has_gpu(&self) -> bool {
    #[cfg(feature = "gpu")]
    {
      self.core().gpu.is_some()
    }
    #[cfg(not(feature = "gpu"))]
    {
      false
    }
  }

  // ---- content ------------------------------------------------------------------------------

  /// Paste a pattern with its top-left cell at `origin`; its live cells replace those under them.
  pub fn load(&mut self, pattern: &Pattern, origin: (i64, i64)) -> Result<()> {
    let hub = self.hub.clone();
    let mut c = self.core_mut();
    c.sync_host();
    c.load(pattern, origin)?;
    c.after_edit(&hub);
    Ok(())
  }

  /// Set cells: `((x, y), state)`.
  pub fn set_cells(&mut self, cells: &[((i64, i64), u8)]) {
    let hub = self.hub.clone();
    let mut c = self.core_mut();
    c.sync_host();
    c.set_cells(cells);
    c.after_edit(&hub);
  }

  /// Remove every cell and start again at generation 0.
  pub fn clear(&mut self) {
    let hub = self.hub.clone();
    let mut c = self.core_mut();
    c.sync_host();
    c.root = Root { id: 0, level: c.host.a, x: 0, y: 0 };
    c.generation = 0;
    c.after_edit(&hub);
    hub.clear_history();
  }

  /// Write the universe as RLE or macrocell.
  pub fn save(&self, format: Format, out: &mut dyn std::io::Write) -> Result<()> {
    let text = self.core().save(format)?;
    out.write_all(text.as_bytes())?;
    Ok(())
  }

  // ---- time -----------------------------------------------------------------------------------

  /// Advance exactly `generations` generations.
  pub fn step(&mut self, generations: u64) -> StepReport {
    let hub = self.hub.clone();
    let mut c = self.core_mut();
    c.run(generations, None, &hub)
  }

  /// Advance by whole epochs until `budget` is spent; returns the generation reached.
  pub fn step_for(&mut self, budget: Duration) -> StepReport {
    let hub = self.hub.clone();
    let mut c = self.core_mut();
    c.run(u64::MAX, Some(budget), &hub)
  }

  pub fn generation(&self) -> u64 {
    self.core().generation
  }

  // ---- queries --------------------------------------------------------------------------------

  pub fn population(&self) -> u64 {
    self.core().population()
  }
  /// The bounding box of the live cells, `None` for an empty universe.
  pub fn bounding_box(&self) -> Option<Rect> {
    self.core().bounding_box()
  }
  /// One state per cell of `rect`, row-major, into `out` (`rect.w * rect.h`).
  pub fn read_region(&self, rect: Rect, out: &mut [u8]) {
    self.core().read_region(rect, out)
  }
  /// A summary per tile of side `2^level` on the absolute lattice, row-major over
  /// [`Rect::tiles`].
  pub fn tile_summary(&self, rect: Rect, level: u32, out: &mut [TileSummary]) {
    self.core().tile_summary(rect, level, out)
  }
  pub fn fingerprint(&self) -> Fingerprint {
    self.core().fingerprint()
  }
  pub fn stats(&self) -> Stats {
    self.core().stats()
  }

  // ---- introspection --------------------------------------------------------------------------

  pub fn observer(&self) -> Observer {
    Observer { hub: self.hub.clone(), core: Arc::downgrade(&self.core) }
  }
  /// Which maps are kept current at epoch boundaries; none by default.
  pub fn set_taps(&mut self, taps: Taps) {
    self.core_mut().taps = taps;
  }
  /// Answer the observers' queued map requests now (a paused UI calls this once per frame).
  pub fn serve(&self) {
    let c = self.core();
    self.hub.serve(&c);
  }
  /// The last step, wave by wave.
  pub fn trace_step(&mut self) -> StepTrace {
    self.core_mut().trace_step()
  }

  /// The GPU buffers a UI on the same device can draw from (feature `gpu`).
  #[cfg(feature = "gpu")]
  pub fn gpu_view(&self) -> Option<crate::gpu::GpuView> {
    self.core().gpu.as_ref().and_then(|g| g.view())
  }

  /// Run `f` with the host recursion and the root (read-only), syncing the state to the host first.
  pub fn with_host<R>(&mut self, f: impl FnOnce(&Engine, Root) -> R) -> R {
    let mut c = self.core_mut();
    c.sync_host();
    f(&c.host, c.root)
  }
}

fn make_policy(config: &Config, sched: &Scheduler, a: usize) -> Option<Policy> {
  let kind = match config.controller {
    Controller::Uniform => return None,
    Controller::Algorithm3 => PolicyKind::Adaptive,
    Controller::Priced => PolicyKind::Priced,
    Controller::PlainCells => PolicyKind::Fixed(Mode::Cells),
  };
  let rl = sched.step.region_log2() as usize;
  let k = sched.step.step_log2 as usize;
  let maxf = 4usize.clamp(a, rl.saturating_sub(1).max(a));
  Some(Policy::new(kind, rl, a, k, Mode::Floor(a), sched.model, maxf))
}

/// `floor(log2(n))` for `n >= 1`.
fn log2_floor(n: u64) -> u32 {
  63 - n.leading_zeros()
}

impl Core {
  fn min_level(&self) -> usize {
    (self.config.align_log2 as usize + 2).max(self.host.a + 2)
  }

  fn is_empty(&self) -> bool {
    self.host.is_empty(self.root.level, self.root.id)
  }

  /// Make the host's quadtree hold the state (the GPU's state is read back).
  pub(crate) fn sync_host(&mut self) {
    #[cfg(feature = "gpu")]
    if let Some(g) = self.gpu.as_mut() {
      if g.holds_state() {
        let r = g.download(&mut self.host, self.config.align_log2 as usize);
        self.root = self.normalise(r);
      }
    }
  }

  /// Grow a root to at least the minimum level, keeping its corner.
  fn normalise(&mut self, mut r: Root) -> Root {
    let min = self.min_level();
    while r.level < min {
      let z = self.host.empty(r.level);
      r.id = self.host.intern(r.level + 1, [r.id, z, z, z]);
      r.level += 1;
    }
    r
  }

  fn after_edit(&mut self, hub: &Hub) {
    self.sched.cycles.reset();
    self.sched.racer.reset();
    if let Some(p) = self.policy.as_mut() {
      p.reset();
    }
    self.trace = None;
    self.gpu_trace = None;
    #[cfg(feature = "gpu")]
    if let Some(g) = self.gpu.as_mut() {
      g.invalidate();
    }
    hub.serve(self);
  }

  // ---- edits --------------------------------------------------------------------------------

  /// Grow the root until it covers the rectangle `[x0, x1) x [y0, y1)`.
  fn cover(&mut self, x0: i64, y0: i64, x1: i64, y1: i64) {
    if self.root.level < self.min_level() {
      self.root = self.normalise(self.root);
    }
    loop {
      let side = 1i64 << self.root.level;
      if x0 >= self.root.x && y0 >= self.root.y && x1 <= self.root.x + side && y1 <= self.root.y + side {
        return;
      }
      let h = side / 2;
      let r = self.root;
      let id = self.host.pushroot(r.id, r.level);
      self.root = Root { id, level: r.level + 1, x: r.x - h, y: r.y - h };
    }
  }

  fn set_cells(&mut self, cells: &[((i64, i64), u8)]) {
    if cells.is_empty() {
      return;
    }
    let (mut x0, mut y0, mut x1, mut y1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
    for &((x, y), _) in cells {
      x0 = x0.min(x);
      y0 = y0.min(y);
      x1 = x1.max(x + 1);
      y1 = y1.max(y + 1);
    }
    let states = self.rule.states();
    self.cover(x0, y0, x1, y1);
    let r = self.root;
    let rel: Vec<((u64, u64), u8)> = cells.iter().map(|&((x, y), v)| (((x - r.x) as u64, (y - r.y) as u64), if (v as u32) < states { v } else { 0 })).collect();
    let id = self.host.set_cells(r.id, r.level, &rel);
    self.root.id = id;
  }

  fn load(&mut self, pattern: &Pattern, origin: (i64, i64)) -> Result<()> {
    let states = self.rule.states();
    match pattern {
      Pattern::Cells(g) => {
        if let Some(&bad) = g.cells.iter().find(|&&c| c as u32 >= states) {
          return Err(Error::Pattern(format!("the pattern has state {bad}, the rule {}", states)));
        }
        if self.is_empty() {
          // build the quadtree directly, the corner on the aligned lattice
          let al = 1i64 << self.config.align_log2;
          let (ax, ay) = (origin.0.div_euclid(al) * al, origin.1.div_euclid(al) * al);
          let (px, py) = ((origin.0 - ax) as usize, (origin.1 - ay) as usize);
          let padded = if px == 0 && py == 0 { g.clone() } else { g.padded(px, py, g.w + px, g.h + py) };
          let (id, level) = self.host.from_grid(&padded);
          self.root = self.normalise(Root { id, level, x: ax, y: ay });
        } else {
          let mut cells = Vec::new();
          for y in 0..g.h {
            for x in 0..g.w {
              let c = g.get(x, y);
              if c != 0 {
                cells.push(((origin.0 + x as i64, origin.1 + y as i64), c));
              }
            }
          }
          self.set_cells(&cells);
        }
      }
      Pattern::Macrocell(m) => {
        let (id, level) = self.host.from_mc(m);
        let al = 1i64 << self.config.align_log2;
        if self.is_empty() && origin.0 % al == 0 && origin.1 % al == 0 {
          self.root = self.normalise(Root { id, level, x: origin.0, y: origin.1 });
        } else {
          let mut cells = Vec::new();
          self.host.for_each_cell(id, level, origin.0, origin.1, &mut |x, y, s| cells.push(((x, y), s)));
          if cells.iter().any(|c| c.1 as u32 >= states) {
            return Err(Error::Pattern(format!("the macrocell has states beyond the rule's {states}")));
          }
          self.set_cells(&cells);
        }
      }
    }
    Ok(())
  }

  fn save(&self, format: Format) -> Result<String> {
    let host = self.host_view();
    let (e, r) = (host.engine(), host.root());
    match format {
      Format::Macrocell => {
        let (id, level, _, _) = (r.id, r.level, r.x, r.y);
        Ok(e.write_macrocell(id, level, self.rule.name(), Some(self.generation)))
      }
      Format::Rle => {
        let Some((x0, y0, x1, y1)) = e.bbox(r.id, r.level, r.x, r.y) else {
          return Ok(write_rle(&Grid::new(0, 0), self.rule.name(), self.rule.states() == 2));
        };
        let (w, h) = ((x1 - x0) as usize, (y1 - y0) as usize);
        if (w as u64) * (h as u64) > 1 << 30 {
          return Err(Error::Pattern(format!("{w}x{h} cells are too many for RLE; save as macrocell")));
        }
        let mut g = Grid::new(w, h);
        e.read_rect(r.id, r.level, r.x, r.y, x0, y0, w, h, &mut g.cells);
        Ok(write_rle(&g, self.rule.name(), self.rule.states() == 2))
      }
    }
  }

  /// The host quadtree to read from: the host's own, or (while the GPU holds the state) a copy
  /// built from the GPU's.
  fn host_view(&self) -> HostView<'_> {
    #[cfg(feature = "gpu")]
    if let Some(g) = self.gpu.as_ref() {
      if g.holds_state() {
        let mut e = Engine::new(&self.rule, self.host.a, 1);
        let r = g.download_copy(&mut e, self.config.align_log2 as usize);
        return HostView::Owned(Box::new(e), r);
      }
    }
    HostView::Borrowed(&self.host, self.root)
  }

  // ---- stepping -------------------------------------------------------------------------------

  fn run(&mut self, generations: u64, budget: Option<Duration>, hub: &Hub) -> StepReport {
    let t0 = Instant::now();
    self.bulk = budget.is_none();
    if self.root.level < self.min_level() {
      self.root = self.normalise(self.root);
    }
    let mut rep = StepReport::default();
    let mut left = generations;
    while left > 0 {
      if let Some(b) = budget {
        if t0.elapsed() >= b {
          break;
        }
      }
      let epoch_budget = budget.map(|b| b.saturating_sub(t0.elapsed()));
      let (gens, engine, cu, wall) = self.epoch(left, epoch_budget, hub);
      if gens == 0 {
        break;
      }
      left -= gens;
      rep.add(engine, gens, cu, wall);
      rep.epochs += 1;
    }
    rep.wall = t0.elapsed();
    rep.generation = self.generation;
    hub.serve(self);
    rep
  }

  /// Whether the next memo step must first reach an even generation (alternating B0 rules).
  fn needs_parity_step(&self) -> bool {
    self.rule.alternates() && self.generation % 2 == 1
  }

  /// One epoch of at most `left` generations. Returns the generations run, the engine, the
  /// cell-updates they stand for and the wall time.
  fn epoch(&mut self, left: u64, budget: Option<Duration>, hub: &Hub) -> (u64, EngineKind, f64, Duration) {
    let t = Instant::now();
    // a known cycle: skip whole periods
    if let Some((p, dx, dy)) = self.sched.cycles.found {
      if left >= p && !self.needs_parity_step() {
        self.sync_host();
        let n = left / p;
        self.root.x += dx * n as i64;
        self.root.y += dy * n as i64;
        self.generation += n * p;
        let cu = self.bbox_cells() * (n * p) as f64;
        let wall = t.elapsed();
        self.account(EngineKind::Cycle, n * p, cu, wall);
        self.publish_simple(hub, Regime::HostRecursion, n * p, EngineKind::Cycle, cu, wall);
        return (n * p, EngineKind::Cycle, cu, wall);
      }
    }
    if self.needs_parity_step() || (left == 1 && self.rule.alternates()) {
      self.sync_host();
      let cu = self.cells_step(1);
      let wall = t.elapsed();
      self.account(EngineKind::HostCells, 1, cu, wall);
      self.publish_simple(hub, Regime::HostRecursion, 1, EngineKind::HostCells, cu, wall);
      return (1, EngineKind::HostCells, cu, wall);
    }
    #[allow(unused_variables)]
    let regime = self.choose_regime();
    #[cfg(feature = "gpu")]
    if regime != Regime::HostRecursion && self.gpu.is_some() {
      match self.gpu_epoch(regime, left, budget, hub) {
        Ok(Some(out)) => return out,
        Ok(None) => {}
        Err(e) => {
          // a GPU failure: fall back to the host for good
          eprintln!("ghl: GPU engine disabled after an error: {e}");
          self.sync_host();
          self.gpu = None;
          self.sched.racer = crate::sched::Racer::new(&[Regime::HostRecursion]);
        }
      }
    }
    let _ = budget;
    self.host_epoch(left, hub)
  }

  fn choose_regime(&mut self) -> Regime {
    if let Some(r) = self.config.regime {
      return r;
    }
    #[cfg(feature = "gpu")]
    {
      if self.gpu.is_none() {
        return Regime::HostRecursion;
      }
      let active = self.active_cells();
      self.active = active;
      match crate::sched::epoch::floor_for(active, &self.sched.thresholds, self.host.a + 1) {
        None if self.config.race => self.sched.racer.choose(),
        None => Regime::SharedTile,
        Some(f) => {
          // the universe-wide floor: one level coarser than the rule's between A1 and A2
          if f != self.host.floor() {
            self.host.set_floor(f);
            if let Some(g) = self.gpu.as_mut() {
              g.set_floor(f);
            }
          }
          Regime::Memo
        }
      }
    }
    #[cfg(not(feature = "gpu"))]
    {
      Regime::HostRecursion
    }
  }

  /// Live cells in active blocks: the cells of the non-empty base tiles' 64×64 blocks.
  fn active_cells(&self) -> f64 {
    #[cfg(feature = "gpu")]
    if let Some(g) = self.gpu.as_ref() {
      if g.holds_state() {
        return g.active_cells();
      }
    }
    let bl = 6.max(self.host.a);
    self.host.count_nonempty(self.root.id, self.root.level, bl) as f64 * (1u64 << (2 * bl)) as f64
  }

  fn bbox_cells(&self) -> f64 {
    self.bounding_box().map(|r| r.area() as f64).unwrap_or(0.0)
  }

  /// Step the whole universe `gens` generations on the host's bit-planes (odd generations of
  /// alternating rules). Returns the cell-updates.
  fn cells_step(&mut self, gens: u64) -> f64 {
    let r = self.root;
    let Some((x0, y0, x1, y1)) = self.host.bbox(r.id, r.level, r.x, r.y) else {
      self.generation += gens;
      return 0.0;
    };
    let m = gens as i64 + 1;
    let al = 1i64 << self.config.align_log2;
    let (gx0, gy0) = ((x0 - m).div_euclid(al) * al, (y0 - m).div_euclid(al) * al);
    let (w, h) = ((x1 + m - gx0) as usize, (y1 + m - gy0) as usize);
    let mut g = Grid::new(w, h);
    self.host.read_rect(r.id, r.level, r.x, r.y, gx0, gy0, w, h, &mut g.cells);
    let mut pg = cpu::PlaneGrid::from_cells(self.rule.planes() as usize, w, h, &g.cells);
    let mut tmp = pg.clone();
    let leaf = cpu::Leaf::new(&self.rule);
    leaf.step(&mut pg, &mut tmp, gens as usize, self.generation);
    let out = Grid { w, h, cells: pg.to_cells(w, h) };
    let (id, level) = self.host.from_grid(&out);
    self.root = self.normalise(Root { id, level, x: gx0, y: gy0 });
    self.generation += gens;
    (w * h) as f64 * gens as f64
  }

  /// One epoch of the host recursion: a step of `2^k` generations from the root.
  fn host_epoch(&mut self, left: u64, hub: &Hub) -> (u64, EngineKind, f64, Duration) {
    self.sync_host();
    let t = Instant::now();
    let with_policy = self.policy.is_some();
    let k = if with_policy {
      self.sched.step.step_log2
    } else if self.bulk && self.config.host_step_log2.is_none() {
      self.sched.host_step_log2.max(10)
    } else {
      self.sched.host_step_log2
    };
    let k = k.min(log2_floor(left)) as usize;
    let gens = 1u64 << k;
    self.host.set_step(k);
    let before: Vec<LevelStats> = self.host.levels.iter().map(|l| l.stats).collect();
    let tiles_before = self.host.stats.tile_inserts;
    if self.config.trace {
      self.host.wave_log = Some(Vec::new());
      self.host.brute_waves.clear();
    }
    let bbox_cells = self.bbox_cells();
    let gen0 = self.generation;
    let r = self.root;
    let min_level = self.min_level();
    let mut nonempty: FxHashSet<(i64, i64)> = Default::default();
    let (id, level, x, y) = match self.policy.as_mut() {
      Some(p) => {
        let rl = p.rl;
        let mut cells = Vec::new();
        self.host.nonempty_at(r.id, r.level, r.x, r.y, rl, &mut cells);
        nonempty = cells.iter().map(|&(x, y)| (x >> rl, y >> rl)).collect();
        // the regions in plain cells are stepped together as one block, sharing their halos
        let regions: FxHashSet<(i64, i64)> = nonempty.iter().copied().filter(|&(x, y)| p.mode_at(x << rl, y << rl) == Mode::Cells).collect();
        let block = if regions.is_empty() || rl != k + 2 {
          None
        } else {
          let (mut x0, mut y0, mut x1, mut y1) = (i64::MAX, i64::MAX, i64::MIN, i64::MIN);
          for &(x, y) in &regions {
            x0 = x0.min(x << rl);
            y0 = y0.min(y << rl);
            x1 = x1.max((x + 1) << rl);
            y1 = y1.max((y + 1) << rl);
          }
          let t = gens as i64;
          let (gx0, gy0) = (x0 - t, y0 - t);
          let (w, h) = ((x1 - x0 + 2 * t) as usize, (y1 - y0 + 2 * t) as usize);
          let mut g = Grid::new(w, h);
          self.host.read_rect(r.id, r.level, r.x, r.y, gx0, gy0, w, h, &mut g.cells);
          let mut pg = cpu::PlaneGrid::from_cells(self.rule.planes() as usize, w, h, &g.cells);
          let mut tmp = pg.clone();
          cpu::Leaf::new(&self.rule).step(&mut pg, &mut tmp, gens as usize, gen0);
          Some(cpu::Block { grid: pg, x0: gx0, y0: gy0, level: rl + 1, rl, regions })
        };
        let mut wb = WithBlock { p, block };
        self.host.step_root_at(r.id, r.level, r.x, r.y, min_level.max(rl + 1), &mut wb)
      }
      None => self.host.step_root_abs(r.id, r.level, r.x, r.y, min_level),
    };
    if let Some(p) = self.policy.as_mut() {
      p.end_step(&nonempty);
    }
    self.root = Root { id, level, x, y };
    self.generation += gens;
    let wall = t.elapsed();
    if self.config.trace {
      let waves = self.host.wave_log.take().unwrap_or_default();
      let inserts: Vec<u64> = self.host.levels.iter().enumerate().map(|(l, lv)| lv.stats.inserts - before.get(l).map(|b| b.inserts).unwrap_or(0)).collect();
      self.trace = Some(HostTrace { generation: gen0, k, waves, brute: std::mem::take(&mut self.host.brute_waves), inserts });
      self.gpu_trace = None;
    }
    // the step and the region (equations (7)-(9)): lengthened while no region is in plain cells and
    // the hashing above the floor dominates, shortened when plain cells are entered
    if self.config.step_log2.is_none() {
      if let Some(p) = self.policy.as_ref() {
        let cells = p.regions_in(i64::MIN / 4, i64::MIN / 4, i64::MAX / 4, i64::MAX / 4).iter().filter(|r| r.mode == Mode::Cells).count();
        let entered = p.switches > self.policy_switches;
        self.policy_switches = p.switches;
        let fl = self.host.floor();
        let (mut above, mut all) = (0u64, 0u64);
        for (l, lv) in self.host.levels.iter().enumerate() {
          let d = lv.stats.since(before.get(l).unwrap_or(&LevelStats::default()));
          all += d.requests;
          if l > fl {
            above += d.requests;
          }
        }
        let share = if all == 0 { 0.0 } else { above as f64 / all as f64 };
        if self.sched.step.update(cells, entered, share) {
          self.policy = make_policy(&self.config, &self.sched, self.host.a);
          self.policy_switches = 0;
        }
      }
    }
    let cu = bbox_cells * gens as f64;
    self.account(EngineKind::Host, gens, cu, wall);
    if self.config.host_step_log2.is_none() && !with_policy {
      // within a time budget the step shrinks to fit it; a bulk run only lengthens it, since a
      // longer HashLife step costs less per generation
      let target = self.config.epoch_target.as_secs_f64();
      let floor = if self.bulk { self.sched.host_step_log2 } else { 0 };
      self.sched.adapt_host_step(wall.as_secs_f64(), target, floor);
    }
    if self.sched.racer.current == Some(Regime::HostRecursion) {
      self.sched.racer.report(Regime::HostRecursion, gens as f64 / wall.as_secs_f64().max(1e-9));
    }
    self.after_host_step();
    // counters of the epoch
    let levels: Vec<LevelCounters> = self
      .host
      .levels
      .iter()
      .enumerate()
      .skip(self.host.a + 1)
      .filter_map(|(l, lv)| {
        let d = lv.stats.since(before.get(l).unwrap_or(&LevelStats::default()));
        (d.requests > 0 || d.inserts > 0).then(|| LevelCounters {
          level: l as u32,
          side: 1 << l,
          windows: d.windows,
          distinct: d.requests,
          misses: d.misses,
          inserts: d.inserts,
          entries: lv.nodes.len() as u64,
          waves: d.waves,
        })
      })
      .collect();
    let a = self.host.a;
    let engines = vec![EngineUse {
      engine: EngineKind::Host,
      kernel: format!("host {}", leaf_name(&self.rule)),
      tile: (1 << a, 1 << a),
      k: 1 << (a - 1),
      generations: gens,
      cell_updates: cu,
      micros: wall.as_secs_f64() * 1e6,
    }];
    let _ = tiles_before;
    let s = self.sample(Regime::HostRecursion, gens, engines, levels, wall);
    hub.publish(s);
    hub.serve(self);
    (gens, EngineKind::Host, cu, wall)
  }

  /// Bookkeeping after a host step: cycle detection and reclaiming.
  fn after_host_step(&mut self) {
    if self.config.detect_cycles {
      let parity = if self.rule.alternates() { self.generation & 1 } else { 0 };
      let key = (self.root.id as u64) << 9 | (self.root.level as u64) << 1 | parity;
      let (g, x, y) = (self.generation, self.root.x, self.root.y);
      self.sched.cycles.record(key, g, x, y);
    }
    if self.host.memory_bytes() > self.config.host_memory_limit {
      self.reclaim();
    }
  }

  /// Drop the host's unreachable nodes.
  pub(crate) fn reclaim(&mut self) {
    let r = self.root;
    let (map, _) = reclaim::reclaim(&mut self.host, &[(r.level, r.id)], true);
    self.root.id = map.node(r.level, r.id);
    if self.host.memory_bytes() > self.config.host_memory_limit / 2 {
      // results kept too much alive: drop them as well
      let r = self.root;
      let (map, _) = reclaim::reclaim(&mut self.host, &[(r.level, r.id)], false);
      self.root.id = map.node(r.level, r.id);
    }
    self.reclaims += 1;
    if let Some(p) = self.policy.as_mut() {
      p.forget_nodes();
    }
    self.sched.cycles.reset();
    self.trace = None;
    #[cfg(feature = "gpu")]
    if let Some(g) = self.gpu.as_mut() {
      g.host_reclaimed();
    }
  }

  pub(crate) fn account(&mut self, e: EngineKind, gens: u64, cu: f64, wall: Duration) {
    self.last_engine = Some(e);
    self.epoch += 1;
    match self.totals.iter_mut().find(|x| x.0 == e) {
      Some(x) => {
        x.1.generations += gens;
        x.1.cell_updates += cu;
        x.1.wall += wall;
      }
      None => self.totals.push((e, EngineTime { generations: gens, cell_updates: cu, wall })),
    }
  }

  fn publish_simple(&self, hub: &Hub, regime: Regime, gens: u64, e: EngineKind, cu: f64, wall: Duration) {
    let engines = vec![EngineUse { engine: e, kernel: e.name().into(), tile: (1, 1), k: 1, generations: gens, cell_updates: cu, micros: wall.as_secs_f64() * 1e6 }];
    hub.publish(self.sample(regime, gens, engines, Vec::new(), wall));
  }

  pub(crate) fn sample(&self, regime: Regime, gens: u64, engines: Vec<EngineUse>, levels: Vec<LevelCounters>, wall: Duration) -> Sample {
    Sample {
      epoch: self.epoch,
      generation: self.generation,
      generations: gens,
      step: self.sched.step.step() as u32,
      region_side: 1 << self.sched.step.region_log2(),
      regime,
      engines,
      levels,
      depth: Depth {
        active_cells: self.active as u64,
        a1: self.sched.thresholds.a1,
        a2: self.sched.thresholds.a2,
        floor: 2 << self.host.floor().saturating_sub(1),
        dispatch_us: self.sched.rates.dispatch_s * 1e6,
        raced: self.sched.racer.rates(),
      },
      overflows: self.overflows as u32,
      memory: self.memory(),
      population: self.population(),
      micros: wall.as_secs_f64() * 1e6,
    }
  }

  fn memory(&self) -> Memory {
    #[allow(unused_mut)]
    let mut m = Memory { host: self.host.memory_bytes(), ..Default::default() };
    #[cfg(feature = "gpu")]
    if let Some(g) = self.gpu.as_ref() {
      let gm = g.memory();
      m.grids = gm.grids;
      m.tables = gm.tables;
      m.tile_store = gm.tile_store;
      m.atlas = gm.atlas;
    }
    m
  }

  // ---- queries --------------------------------------------------------------------------------

  pub(crate) fn population(&self) -> u64 {
    #[cfg(feature = "gpu")]
    if let Some(g) = self.gpu.as_ref() {
      if g.holds_state() {
        return g.population();
      }
    }
    self.host.population(self.root.level, self.root.id)
  }

  pub(crate) fn bounding_box(&self) -> Option<Rect> {
    #[cfg(feature = "gpu")]
    if let Some(g) = self.gpu.as_ref() {
      if g.holds_state() {
        return g.bounding_box();
      }
    }
    let r = self.root;
    self.host.bbox(r.id, r.level, r.x, r.y).map(|(x0, y0, x1, y1)| Rect::from_corners(x0, y0, x1, y1))
  }

  pub(crate) fn read_region(&self, rect: Rect, out: &mut [u8]) {
    let n = (rect.w * rect.h) as usize;
    out[..n].fill(0);
    #[cfg(feature = "gpu")]
    if let Some(g) = self.gpu.as_ref() {
      if g.holds_state() {
        g.read_region(rect, out);
        return;
      }
    }
    let r = self.root;
    self.host.read_rect(r.id, r.level, r.x, r.y, rect.x, rect.y, rect.w as usize, rect.h as usize, out);
  }

  pub(crate) fn tile_summary(&self, rect: Rect, level: u32, out: &mut [TileSummary]) {
    let (tx0, ty0, nx, ny) = rect.tiles(level);
    let palette = self.rule.palette().to_vec();
    let s = 1i64 << level;
    let view = self.host_view();
    let (e, r) = (view.engine(), view.root());
    let mut memo: rustc_hash::FxHashMap<(usize, u32), (u64, [u64; 3])> = Default::default();
    let dead = palette.first().copied().unwrap_or([0, 0, 0]);
    for j in 0..ny {
      for i in 0..nx {
        let (x, y) = ((tx0 + i as i64) * s, (ty0 + j as i64) * s);
        let (pop, sum) = sum_rect(e, r, x, y, s, &palette, &mut memo);
        let cells = (s * s) as u64;
        let mut color = [0u8; 3];
        for ch in 0..3 {
          let total = sum[ch] + (cells - pop) * dead[ch] as u64;
          color[ch] = ((total + cells / 2) / cells) as u8;
        }
        out[j * nx + i] = TileSummary { population: pop, color };
      }
    }
  }

  pub(crate) fn fingerprint(&self) -> Fingerprint {
    let view = self.host_view();
    let r = view.root();
    view.engine().fingerprint(r.id, r.level)
  }

  pub(crate) fn stats(&self) -> Stats {
    let levels: Vec<(u32, u64, u64)> = self.host.levels.iter().enumerate().skip(self.host.a + 1).filter(|(_, l)| l.stats.requests > 0).map(|(l, lv)| (l as u32, lv.stats.requests, lv.stats.misses)).collect();
    #[allow(unused_mut)]
    let mut s = Stats {
      generation: self.generation,
      population: self.population(),
      engine: self.last_engine,
      engines: self.totals.clone(),
      levels,
      host_nodes: self.host.total_nodes() as u64,
      host_tiles: self.host.num_tiles() as u64,
      gpu_entries: 0,
      memory: self.memory(),
      overflows: self.overflows,
      reclaims: self.reclaims,
      step: 1 << self.host.step_log2(),
      base_tile: 1 << self.host.a,
      cycle: self.sched.cycles.found,
    };
    #[cfg(feature = "gpu")]
    if let Some(g) = self.gpu.as_ref() {
      s.gpu_entries = g.entries();
      s.overflows += g.overflows();
    }
    s
  }

  // ---- maps for the inspector ---------------------------------------------------------------

  /// IDs of the level-`level` nodes covering `rect` on the absolute lattice (`NONE` outside the
  /// universe's root).
  pub(crate) fn ids(&self, rect: Rect, level: u32) -> Vec<TileId> {
    #[cfg(feature = "gpu")]
    if let Some(g) = self.gpu.as_ref() {
      if g.holds_state() {
        if let Some(v) = g.ids(rect, level) {
          return v;
        }
      }
    }
    let (tx0, ty0, nx, ny) = rect.tiles(level);
    let mut out = vec![NONE; nx * ny];
    let l = level as usize;
    if l < self.host.a || l > self.root.level {
      return out;
    }
    let s = 1i64 << level;
    for j in 0..ny {
      for i in 0..nx {
        let (x, y) = ((tx0 + i as i64) * s, (ty0 + j as i64) * s);
        out[j * nx + i] = self.node_at(x, y, l).unwrap_or(NONE);
      }
    }
    out
  }

  /// The level-`l` node with its corner at absolute `(x, y)`, if it is on the root's lattice.
  fn node_at(&self, x: i64, y: i64, l: usize) -> Option<u32> {
    let r = self.root;
    let side = 1i64 << r.level;
    let (dx, dy) = (x - r.x, y - r.y);
    if dx < 0 || dy < 0 || dx >= side || dy >= side || dx % (1 << l) != 0 || dy % (1 << l) != 0 {
      return None;
    }
    let (mut id, mut lv) = (r.id, r.level);
    let (mut px, mut py) = (dx as u64, dy as u64);
    while lv > l {
      let h = 1u64 << (lv - 1);
      let q = (px >= h) as usize | ((py >= h) as usize) << 1;
      id = self.host.ch(lv, id)[q];
      px &= h - 1;
      py &= h - 1;
      lv -= 1;
    }
    Some(id)
  }

  pub(crate) fn tile(&self, level: u32, id: TileId) -> Option<Tile> {
    #[cfg(feature = "gpu")]
    if let Some(t) = self.gpu.as_ref().and_then(|g| g.tile(level, id)) {
      return t;
    }
    let l = level as usize;
    if l == self.host.a {
      return ((id as usize) < self.host.num_tiles()).then(|| Tile::Cells { side: 1 << level, cells: self.host.tile_cells(id) });
    }
    let lv = self.host.levels.get(l)?;
    let ch = *lv.nodes.get(id as usize)?;
    Some(Tile::Node { level, children: ch, population: lv.pop[id as usize] })
  }

  /// Per tile of the sweep level touched by `rect`: whether the window of the 2×2 tiles that ends
  /// at it (the harness's phase-A key) is memoized.
  pub(crate) fn outcomes(&self, rect: Rect) -> Vec<inspect::Outcome> {
    #[cfg(feature = "gpu")]
    if let Some(g) = self.gpu.as_ref() {
      if let Some(v) = g.outcomes(rect) {
        return v;
      }
    }
    let lt = self.sched.sweep.lt as usize;
    let (tx0, ty0, nx, ny) = rect.tiles(lt as u32);
    let s = 1i64 << lt;
    let z = self.host.empty_ids().get(lt).copied().unwrap_or(NONE);
    let mut out = Vec::with_capacity(nx * ny);
    for j in 0..ny {
      for i in 0..nx {
        let (x, y) = ((tx0 + i as i64) * s, (ty0 + j as i64) * s);
        let at = |dx: i64, dy: i64| self.node_at(x + dx * s, y + dy * s, lt).unwrap_or(z);
        let key = [at(-1, -1), at(0, -1), at(-1, 0), at(0, 0)];
        out.push(if key.iter().all(|&k| k == z) {
          inspect::Outcome::Empty
        } else {
          match self.host.find(lt + 1, key) {
            Some(n) if self.host.result(lt + 1, n) != NONE => inspect::Outcome::Hit,
            _ => inspect::Outcome::Miss,
          }
        });
      }
    }
    out
  }

  pub(crate) fn regions(&self, rect: Rect) -> Vec<RegionInfo> {
    let rl = self.sched.step.region_log2();
    match self.policy.as_ref() {
      Some(p) => p
        .regions_in(rect.x, rect.y, rect.x1(), rect.y1())
        .into_iter()
        .map(|r| RegionInfo {
          x: r.x,
          y: r.y,
          side: 1 << p.rl,
          engine: match r.mode {
            Mode::Floor(f) => RegionEngine::Floor(2 << f),
            Mode::Cells => RegionEngine::Cells,
          },
          steps: r.steps,
          price_floor_ns: r.price_floor_ns,
          price_cells_ns: r.price_cells_ns,
          charged_ns: r.charged_ns,
          switches: r.switches,
        })
        .collect(),
      None => {
        let mut cells = Vec::new();
        let mut host_regions = Vec::new();
        let r = self.root;
        // the uniform floor: every region with content, at the floor
        let l = rl as usize;
        if l <= r.level {
          let mut e_cells = Vec::new();
          collect_nonempty(&self.host, r.id, r.level, r.x, r.y, l, &mut e_cells);
          cells = e_cells;
        }
        for (x, y) in cells {
          if Rect::new(x, y, 1 << rl, 1 << rl).intersect(&rect).is_empty() {
            continue;
          }
          host_regions.push(RegionInfo {
            x,
            y,
            side: 1 << rl,
            engine: RegionEngine::Floor(2 << self.host.a),
            steps: [0, 0],
            price_floor_ns: 0.0,
            price_cells_ns: 0.0,
            charged_ns: 0.0,
            switches: 0,
          });
        }
        host_regions
      }
    }
  }

  /// The level's table in the GPU's layout (entries with a memoized result).
  fn level_table(&self, level: u32) -> Option<(Vec<u32>, u32, f64, u64)> {
    let lv = self.host.levels.get(level as usize)?;
    let n = lv.res.iter().filter(|&&r| r != NONE).count();
    let (t, mask, avg) = tile::build_table(lv.nodes.iter().zip(&lv.res).filter(|(_, &r)| r != NONE).map(|(k, &r)| (*k, r)), n, 10);
    Some((t, mask, avg, n as u64))
  }

  pub(crate) fn memo_stats(&self, level: u32) -> MemoStats {
    #[cfg(feature = "gpu")]
    if let Some(g) = self.gpu.as_ref() {
      if let Some(m) = g.memo_stats(level) {
        return m;
      }
    }
    let Some((t, mask, avg, n)) = self.level_table(level) else {
      return MemoStats { level, ..Default::default() };
    };
    let mut hist = vec![0u64; 32];
    for d in tile::displacements(&t, mask).flatten() {
      hist[(d as usize).min(31)] += 1;
    }
    let slots = mask as u64 + 1;
    MemoStats { level, entries: n, slots, bytes: slots * 32, load: n as f64 / slots as f64, avg_probes: avg, displacement: hist }
  }

  pub(crate) fn memo_map(&self, level: u32, slots: Range<u64>) -> Vec<Slot> {
    #[cfg(feature = "gpu")]
    if let Some(g) = self.gpu.as_ref() {
      if let Some(m) = g.memo_map(level, slots.clone()) {
        return m;
      }
    }
    let Some((t, mask, _, _)) = self.level_table(level) else { return Vec::new() };
    let disp: Vec<Option<u32>> = tile::displacements(&t, mask).collect();
    (slots.start..slots.end.min(mask as u64 + 1))
      .map(|s| {
        let b = s as usize * 8;
        match disp[s as usize] {
          None => Slot { state: SlotState::Empty, key: [NONE; 4], value: NONE, displacement: 0 },
          Some(d) => Slot { state: SlotState::Committed, key: [t[b], t[b + 1], t[b + 2], t[b + 3]], value: t[b + 4], displacement: d },
        }
      })
      .collect()
  }

  pub(crate) fn memo_lookup(&self, level: u32, key: [TileId; 4]) -> Option<MemoEntry> {
    #[cfg(feature = "gpu")]
    if let Some(g) = self.gpu.as_ref() {
      if let Some(m) = g.memo_lookup(level, key) {
        return m;
      }
    }
    let id = self.host.find(level as usize, key)?;
    let r = self.host.result(level as usize, id);
    Some(MemoEntry { level, key, id, result: (r != NONE).then_some(r), slot: id as u64 })
  }

  pub(crate) fn read_planes(&self, x: i64, y: i64) -> Planes {
    let x0 = x.div_euclid(32) * 32;
    let mut rows = vec![0u8; 34 * 3];
    let rect = Rect::new(x0 - 1, y - 1, 34, 3);
    self.read_region(rect, &mut rows);
    let planes = self.rule.planes() as usize;
    let cell = |dx: i64, dy: i64| rows[((dy + 1) * 34 + dx + 1) as usize];
    let mut words = vec![[0u32; 9]; planes];
    for (k, w) in words.iter_mut().enumerate() {
      for (p, pos) in crate::rule::Pos::ALL.iter().enumerate() {
        let (ox, oy) = pos.offset();
        let mut v = 0u32;
        for i in 0..32i64 {
          v |= (((cell(i + ox as i64, oy as i64) >> k) & 1) as u32) << i;
        }
        w[p] = v;
      }
    }
    Planes { x: x0, y, words, cells: (0..32).map(|i| cell(i, 0)).collect() }
  }

  pub(crate) fn level_info(&self, level: u32) -> LevelInfo {
    let l = level as usize;
    let r = self.root;
    let mut info = LevelInfo { level, side: 1u64 << level, ..Default::default() };
    info.empty = self.host.empty_ids().get(l).copied().filter(|&z| z != NONE);
    if l < self.host.a || l > r.level {
      return info;
    }
    info.nonempty = self.host.count_nonempty(r.id, r.level, l);
    let mut seen: FxHashSet<u32> = Default::default();
    distinct_at(&self.host, r.id, r.level, l, &mut seen, &mut FxHashSet::default());
    info.distinct = seen.len() as u64;
    if l == self.host.a {
      info.entries = self.host.num_tiles() as u64;
    } else if let Some(lv) = self.host.levels.get(l) {
      info.entries = lv.nodes.len() as u64;
      info.results = lv.res.iter().filter(|&&x| x != NONE).count() as u64;
    }
    info
  }

  fn trace_step(&mut self) -> StepTrace {
    #[cfg(feature = "gpu")]
    if matches!(self.last_engine, Some(EngineKind::GpuMemo)) {
      if let Some(t) = self.gpu.as_ref().and_then(|g| g.last_trace()) {
        return t;
      }
    }
    if let Some(t) = self.gpu_trace.clone() {
      return t;
    }
    let Some(t) = self.trace.as_ref() else { return StepTrace::default() };
    let a = self.host.a;
    let waves: Vec<Wave> = t
      .waves
      .iter()
      .map(|w| {
        let l = w[0];
        Wave {
          level: l,
          side: 1 << l,
          generations: 1 << (l as usize - 2).min(t.k),
          requests: w[1] as u64,
          distinct: w[1] as u64,
          misses: w[2] as u64,
          inserts: 0,
          dispatches: if w[2] > 0 { 1 } else { 0 },
          capacity: w[1] as u64,
          micros: 0.0,
          parent: (w[3] != NONE).then_some(w[3]),
        }
      })
      .collect();
    let mut atlases = Vec::new();
    for b in &t.brute {
      let l = b.level as usize;
      let side = 1usize << l;
      let windows = b
        .ids
        .iter()
        .take(4096)
        .map(|&id| {
          let r = self.host.result(l, id);
          let result = if r == NONE { Vec::new() } else if l - 1 == a { self.host.tile_cells(r) } else { self.host.expand(r, l - 1).cells };
          inspect::AtlasWindow { id, cells: self.host.expand(id, l).cells, result }
        })
        .collect();
      atlases.push(inspect::Atlas { wave: b.wave, level: b.level, side: side as u32, generations: 1 << (l - 2).min(t.k), windows });
    }
    let _ = &t.inserts;
    StepTrace { generation: t.generation, step: 1 << t.k, engine: Some(EngineKind::Host), waves, atlases }
  }
}

/// The controller's policy with the block of plain-cells regions of the current step.
struct WithBlock<'a> {
  p: &'a mut Policy,
  block: Option<cpu::Block>,
}

impl cpu::FloorPolicy for WithBlock<'_> {
  fn floor(&mut self, cx: i64, cy: i64, l: usize) -> usize {
    cpu::FloorPolicy::floor(self.p, cx, cy, l)
  }
  fn record(&mut self, cx: i64, cy: i64, l: usize, o: cpu::Outcome) {
    cpu::FloorPolicy::record(self.p, cx, cy, l, o)
  }
  fn record_node(&mut self, l: usize, id: u32, cx: i64, cy: i64, o: cpu::Outcome) {
    cpu::FloorPolicy::record_node(self.p, l, id, cx, cy, o)
  }
  fn block(&self) -> Option<&cpu::Block> {
    self.block.as_ref()
  }
}

/// The host quadtree to read from: the host's own, or a copy built from the GPU's state.
enum HostView<'a> {
  Borrowed(&'a Engine, Root),
  #[allow(dead_code)]
  Owned(Box<Engine>, Root),
}

impl HostView<'_> {
  fn engine(&self) -> &Engine {
    match self {
      HostView::Borrowed(e, _) => e,
      HostView::Owned(e, _) => e,
    }
  }
  fn root(&self) -> Root {
    match self {
      HostView::Borrowed(_, r) | HostView::Owned(_, r) => *r,
    }
  }
}

fn leaf_name(rule: &Rule) -> String {
  match kernel_class(rule) {
    KernelClass::LifeLike => "adder tree".into(),
    KernelClass::Circuit => format!("circuit ({} gates)", rule.network().map(|n| n.gates().len()).unwrap_or(0)),
    KernelClass::ByteLut => "table".into(),
    KernelClass::Tree => "rule tree".into(),
  }
}

/// Population and summed palette colours of the live cells in the square `[x, x + s)²`.
fn sum_rect(e: &Engine, r: Root, x: i64, y: i64, s: i64, palette: &[[u8; 3]], memo: &mut rustc_hash::FxHashMap<(usize, u32), (u64, [u64; 3])>) -> (u64, [u64; 3]) {
  fn node_sum(e: &Engine, l: usize, id: u32, palette: &[[u8; 3]], memo: &mut rustc_hash::FxHashMap<(usize, u32), (u64, [u64; 3])>) -> (u64, [u64; 3]) {
    if e.is_empty(l, id) {
      return (0, [0; 3]);
    }
    if let Some(&v) = memo.get(&(l, id)) {
      return v;
    }
    let v = if l == e.a {
      let (mut pop, mut sum) = (0u64, [0u64; 3]);
      e.for_each_tile_cell(id, 0, 0, &mut |_, _, st| {
        pop += 1;
        let c = palette.get(st as usize).copied().unwrap_or([255, 0, 255]);
        for i in 0..3 {
          sum[i] += c[i] as u64;
        }
      });
      (pop, sum)
    } else {
      let mut acc = (0u64, [0u64; 3]);
      for c in e.ch(l, id) {
        let (p, s) = node_sum(e, l - 1, c, palette, memo);
        acc.0 += p;
        for i in 0..3 {
          acc.1[i] += s[i];
        }
      }
      acc
    };
    memo.insert((l, id), v);
    v
  }
  // a whole node on the root's lattice: from the memo; otherwise cell by cell
  let lvl = s.trailing_zeros() as usize;
  let (dx, dy) = (x - r.x, y - r.y);
  if dx >= 0 && dy >= 0 && dx % s == 0 && dy % s == 0 && lvl >= e.a && lvl <= r.level && dx < (1i64 << r.level) && dy < (1i64 << r.level) {
    let (mut id, mut lv) = (r.id, r.level);
    let (mut px, mut py) = (dx as u64, dy as u64);
    while lv > lvl {
      let h = 1u64 << (lv - 1);
      let q = (px >= h) as usize | ((py >= h) as usize) << 1;
      id = e.ch(lv, id)[q];
      px &= h - 1;
      py &= h - 1;
      lv -= 1;
    }
    return node_sum(e, lvl, id, palette, memo);
  }
  let (w, h) = (s as usize, s as usize);
  if w * h > 1 << 24 {
    return (0, [0; 3]);
  }
  let mut cells = vec![0u8; w * h];
  e.read_rect(r.id, r.level, r.x, r.y, x, y, w, h, &mut cells);
  let (mut pop, mut sum) = (0u64, [0u64; 3]);
  for &st in &cells {
    if st != 0 {
      pop += 1;
      let c = palette.get(st as usize).copied().unwrap_or([255, 0, 255]);
      for i in 0..3 {
        sum[i] += c[i] as u64;
      }
    }
  }
  (pop, sum)
}

fn collect_nonempty(e: &Engine, id: u32, l: usize, x: i64, y: i64, lt: usize, out: &mut Vec<(i64, i64)>) {
  if e.is_empty(l, id) {
    return;
  }
  if l <= lt {
    out.push((x, y));
    return;
  }
  let h = 1i64 << (l - 1);
  let c = e.ch(l, id);
  collect_nonempty(e, c[0], l - 1, x, y, lt, out);
  collect_nonempty(e, c[1], l - 1, x + h, y, lt, out);
  collect_nonempty(e, c[2], l - 1, x, y + h, lt, out);
  collect_nonempty(e, c[3], l - 1, x + h, y + h, lt, out);
}

fn distinct_at(e: &Engine, id: u32, l: usize, lt: usize, seen: &mut FxHashSet<u32>, visited: &mut FxHashSet<(usize, u32)>) {
  if e.is_empty(l, id) || !visited.insert((l, id)) {
    return;
  }
  if l == lt {
    seen.insert(id);
    return;
  }
  for c in e.ch(l, id) {
    distinct_at(e, c, l - 1, lt, seen, visited);
  }
}
