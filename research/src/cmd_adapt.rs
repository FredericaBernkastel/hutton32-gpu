//! `adapt`: adaptive subdivision and per-region engine choice, measured on real patterns.
//!
//! The universe is stepped from the root (HashLife order), so tile size is already adaptive
//! upwards: a window that hits at 4096x4096 is never opened. What this experiment varies is how
//! far a *missed* window may be subdivided before it is brute-forced. That depth is a property of
//! the 2^rl-cell *region* under the window's centre:
//!   * `Floor(f)`: missed windows of level <= f+1 (side 2^(f+1)) are brute-forced, larger ones
//!     are subdivided and memoised;
//!   * `Brute`: nothing below the region is memoised; the region is stepped as plain cells.
//!
//! Operation counts come from the reference engine. Times are a model built from the constants
//! measured on the GPU (`gpu-brute`, `gpu-probe`, `gpu-insert`).

use crate::{cmd_stats, cmd_stats::Loaded, hl, hl::{Engine, FloorPolicy, Outcome}, opt, opt_or, pattern::Grid, sim::Sim};
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

const MAXL: usize = 32;
const BRUTE_SLOT: usize = MAXL - 1;

/// GPU cost model, nanoseconds per operation (RTX 5060 Ti measurements).
#[derive(Clone, Copy)]
pub struct Model {
  /// read-only probe / find of an existing entry in a table larger than the L2 cache
  pub probe: f64,
  /// find-or-insert that has to claim a slot (CAS protocol + commit)
  pub insert: f64,
  /// one cell evaluation by the brute-force kernel
  pub cell: f64,
  /// gather of one base tile when a node is expanded into cells
  pub gather: f64,
}
impl Default for Model {
  fn default() -> Self {
    Model { probe: 1e9 / 3.7e9, insert: 1.0, cell: 1e9 / 1.86e11, gather: 1e9 / 3.7e9 }
  }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
  Floor(usize),
  Brute,
}

struct Trial {
  prev: Mode,
  left: u32,
  steps: u32,
  req: [f64; MAXL],
  miss: [f64; MAXL],
}

struct Region {
  mode: Mode,
  // this macro-step
  req: [u32; MAXL],
  miss: [u32; MAXL],
  brute: [u32; MAXL],
  needs_brute: bool,
  /// floor misses that were subdivided once more as an experiment, and what their sub-windows did
  /// (cumulative since the last mode change)
  samp: u64,
  samp_req: u64,
  samp_miss: u64,
  // smoothed evidence per macro-step; `seen[l]`: level l has been observed in the current regime
  ereq: [f64; MAXL],
  emiss: [f64; MAXL],
  seen: [bool; MAXL],
  evidence_steps: u32,
  trial: Option<Trial>,
  backoff: u32,
  since_trial: u32,
  // accounting
  cost_hash: f64,
  cost_brute: f64,
  /// macro-steps with work, per mode: index = floor, BRUTE_SLOT = plain cells
  time_in: [u32; MAXL],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
  Fixed(Mode),
  Adaptive,
}

pub struct Policy {
  pub kind: Kind,
  pub rl: usize,
  pub a: usize,
  pub k: usize,
  pub model: Model,
  regions: FxHashMap<(i64, i64), Region>,
  init: Mode,
  pub total_hash: f64,
  pub total_brute: f64,
  pub switches: u64,
  pub trials: u64,
  pub trials_kept: u64,
  /// deepest brute-force window the heuristic considers is 2^(maxf+1)
  pub maxf: usize,
  /// one floor miss in 2^sample_shift is subdivided once more to gather evidence (0 = use
  /// whole-step trials instead)
  pub sample_shift: u32,
  step: u64,
}

impl Policy {
  pub fn new(kind: Kind, rl: usize, a: usize, k: usize, init: Mode, model: Model, maxf: usize) -> Self {
    Policy {
      kind,
      rl,
      a,
      k,
      model,
      maxf,
      sample_shift: 4,
      step: 0,
      regions: Default::default(),
      init,
      total_hash: 0.0,
      total_brute: 0.0,
      switches: 0,
      trials: 0,
      trials_kept: 0,
    }
  }
  fn region(&mut self, cx: i64, cy: i64) -> &mut Region {
    let key = (cx >> self.rl, cy >> self.rl);
    let init = match self.kind {
      Kind::Fixed(m) => m,
      Kind::Adaptive => self.init,
    };
    self.regions.entry(key).or_insert_with(|| Region {
      mode: init,
      req: [0; MAXL],
      miss: [0; MAXL],
      brute: [0; MAXL],
      needs_brute: false,
      samp: 0,
      samp_req: 0,
      samp_miss: 0,
      ereq: [0.0; MAXL],
      emiss: [0.0; MAXL],
      seen: [false; MAXL],
      evidence_steps: 0,
      trial: None,
      backoff: 4,
      since_trial: 0,
      cost_hash: 0.0,
      cost_brute: 0.0,
      time_in: [0; MAXL],
    })
  }
  /// cell evaluations to brute-force one window of level l with every cell stepped every
  /// generation (what an atlas of independent windows costs)
  fn window_evals(&self, l: usize) -> f64 {
    (1u64 << (2 * l)) as f64 * (1u64 << (l - 2).min(self.k)) as f64
  }
  /// find-or-inserts needed to hash-cons the result of a brute-forced level-l window
  fn intern_count(&self, l: usize) -> f64 {
    let mut n = 0.0;
    let mut c = (1u64 << (2 * (l - 1 - self.a))) as f64;
    loop {
      n += c;
      if c <= 1.0 {
        break;
      }
      c /= 4.0;
    }
    n
  }
  /// a missed window that is subdivided: its result slot plus ten sub-window nodes (5 + 4 + 1)
  fn cost_descend(&self) -> f64 {
    self.model.insert + 10.0 * self.model.probe
  }
  /// a missed window that is brute-forced: (hashing part, cell part)
  fn cost_brute_window(&self, l: usize) -> (f64, f64) {
    let hash = self.model.insert
      + self.intern_count(l) * (self.model.probe + 0.5 * self.model.insert)
      + (1u64 << (2 * (l - self.a))) as f64 * self.model.gather;
    (hash, self.window_evals(l) * self.model.cell)
  }
  /// stepping one region as plain cells for one macro-step, without halo
  fn region_brute_nominal(&self) -> f64 {
    (1u64 << (2 * self.rl)) as f64 * (1u64 << self.k) as f64 * self.model.cell
  }
  /// cells -> ids for a whole region (needed so that neighbours and coarser levels can use it)
  fn region_convert(&self) -> f64 {
    let tiles = (1u64 << (2 * (self.rl - self.a))) as f64;
    tiles * (self.model.gather + 1.34 * (self.model.probe + 0.5 * self.model.insert))
  }

  /// Model cost per macro-step of running a region in `mode`, given per-step request and miss
  /// counts of the levels that mode would exercise. Levels above the region are common to all
  /// modes and left out.
  fn mode_cost(&self, req: &[f64; MAXL], miss: &[f64; MAXL], mode: Mode) -> f64 {
    match mode {
      Mode::Brute => 1.3 * self.region_brute_nominal() + self.region_convert(),
      Mode::Floor(c) => {
        let mut cost = 0.0;
        for l in c + 2..=self.rl {
          cost += req[l] * self.model.probe + miss[l] * self.cost_descend();
        }
        let (h, b) = self.cost_brute_window(c + 1);
        cost + req[c + 1] * self.model.probe + miss[c + 1] * (h + b)
      }
    }
  }

  /// Close a macro-step: charge costs and let adaptive regions reconsider their mode.
  /// `nonempty` holds the regions that contained live cells when the step began.
  pub fn end_step(&mut self, nonempty: &FxHashSet<(i64, i64)>) {
    let (rl, a, k) = (self.rl, self.a, self.k);
    let keys: Vec<(i64, i64)> = self.regions.keys().copied().collect();

    // ---- regions stepped as plain cells this step ----
    // a brute region is charged if it has content and something around it had to be computed
    let mut charged: FxHashSet<(i64, i64)> = Default::default();
    for &key in nonempty {
      let is_brute = self.regions.get(&key).map(|r| r.mode == Mode::Brute).unwrap_or(matches!(self.kind, Kind::Fixed(Mode::Brute)));
      if !is_brute {
        continue;
      }
      let mut any = false;
      for dy in -1..=1 {
        for dx in -1..=1 {
          any |= self.regions.get(&(key.0 + dx, key.1 + dy)).map(|r| r.needs_brute).unwrap_or(false);
        }
      }
      if any {
        charged.insert(key);
      }
    }
    let s = (1u64 << rl) as f64;
    let t = 1u64 << k;
    let mut cells_cost: FxHashMap<(i64, i64), f64> = Default::default();
    for &(x, y) in &charged {
      // own cells every generation, plus the halo that falls outside the charged set: side strips
      // of width r and corner squares r x r, for r = t-1 .. 0
      let out = |dx: i64, dy: i64| !charged.contains(&(x + dx, y + dy));
      let sides = [out(-1, 0), out(1, 0), out(0, -1), out(0, 1)].iter().filter(|&&b| b).count() as f64;
      let corner = |dx: i64, dy: i64| out(dx, 0) && out(0, dy) && out(dx, dy);
      let corners = [corner(-1, -1), corner(1, -1), corner(-1, 1), corner(1, 1)].iter().filter(|&&b| b).count() as f64;
      let mut evals = 0.0;
      for r in 0..t {
        let r = r as f64;
        evals += s * s + sides * r * s + corners * r * r;
      }
      cells_cost.insert((x, y), evals * self.model.cell);
    }

    let window_cost: Vec<(f64, f64)> = (0..MAXL).map(|l| if l > a && l < BRUTE_SLOT { self.cost_brute_window(l) } else { (0.0, 0.0) }).collect();
    let (probe, insert, descend) = (self.model.probe, self.model.insert, self.cost_descend());
    let convert = self.region_convert();
    let kind = self.kind;
    let (mut th, mut tb) = (0.0, 0.0);
    for &key in &keys {
      // ---- charge ----
      let (mode, mut ch, mut cb) = {
        let reg = &self.regions[&key];
        let (mut ch, mut cb) = (0.0, 0.0);
        for l in a + 1..BRUTE_SLOT {
          if reg.req[l] == 0 {
            continue;
          }
          ch += reg.req[l] as f64 * probe + (reg.miss[l] - reg.brute[l]) as f64 * descend;
          if reg.mode == Mode::Brute && l <= rl + 1 {
            // the engine brute-forced these windows only to stay correct; the model steps the
            // region as cells (charged below) and pays for the memo entry
            ch += reg.brute[l] as f64 * insert;
          } else {
            ch += reg.brute[l] as f64 * window_cost[l].0;
            cb += reg.brute[l] as f64 * window_cost[l].1;
          }
        }
        (reg.mode, ch, cb)
      };
      if let Some(&c) = cells_cost.get(&key) {
        cb += c;
        ch += convert;
      }
      let worked = ch + cb > 0.0;
      th += ch;
      tb += cb;

      // ---- adapt ----
      let decision = if kind == Kind::Adaptive { self.decide(key) } else { None };
      let reg = self.regions.get_mut(&key).unwrap();
      reg.cost_hash += ch;
      reg.cost_brute += cb;
      if worked {
        reg.time_in[match mode {
          Mode::Floor(f) => f,
          Mode::Brute => BRUTE_SLOT,
        }] += 1;
      }
      if let Some(m) = decision {
        if m != reg.mode {
          self.switches += 1;
          reg.samp = 0;
          reg.samp_req = 0;
          reg.samp_miss = 0;
        }
        let lowest_new = match m {
          Mode::Floor(f) => f + 1,
          Mode::Brute => rl + 1,
        };
        for l in 0..lowest_new.min(MAXL) {
          reg.seen[l] = false;
        }
        reg.mode = m;
      }
      reg.req = [0; MAXL];
      reg.miss = [0; MAXL];
      reg.brute = [0; MAXL];
      reg.needs_brute = false;
    }
    self.total_hash += th;
    self.total_brute += tb;
    self.step += 1;
  }

  /// The runtime heuristic. Returns the mode for the next macro-step, or None to keep it.
  ///
  /// * Evidence is the number of requests and misses per level, smoothed over macro-steps.
  /// * Coarser modes (and plain cells) can be costed exactly from the levels the current mode
  ///   already exercises, so moving up needs no experiment: switch when it is 15 % cheaper.
  /// * Finer modes cannot be costed without trying: every `backoff` steps run a three-step trial
  ///   one level finer and keep it only if it is 10 % cheaper than what it replaced; back off
  ///   exponentially when it is not.
  fn decide(&mut self, key: (i64, i64)) -> Option<Mode> {
    let (rl, a) = (self.rl, self.a);
    let (mode, lowest, evidence) = {
      let r = &self.regions[&key];
      let lowest = match r.mode {
        Mode::Floor(f) => f + 1,
        Mode::Brute => rl + 1,
      };
      // something had to be computed here: a region-sized window missed
      let top_missed = r.miss[rl + 1] > 0 || r.miss[rl] > 0;
      (r.mode, lowest, top_missed && r.req[lowest.min(rl + 1)] > 0)
    };
    if !evidence {
      return None;
    }
    // smooth
    {
      let r = self.regions.get_mut(&key).unwrap();
      for l in lowest..=rl + 1 {
        if r.seen[l] {
          r.ereq[l] = 0.7 * r.ereq[l] + 0.3 * r.req[l] as f64;
          r.emiss[l] = 0.7 * r.emiss[l] + 0.3 * r.miss[l] as f64;
        } else {
          r.ereq[l] = r.req[l] as f64;
          r.emiss[l] = r.miss[l] as f64;
          r.seen[l] = true;
        }
      }
      if let Some(tr) = r.trial.as_mut() {
        for l in lowest..=rl + 1 {
          tr.req[l] += r.req[l] as f64;
          tr.miss[l] += r.miss[l] as f64;
        }
        tr.steps += 1;
        tr.left -= 1;
      }
    }
    // a running trial?
    let trial_done = self.regions[&key].trial.as_ref().map(|t| t.left == 0);
    match trial_done {
      Some(false) => return None,
      Some(true) => {
        let tr = self.regions.get_mut(&key).unwrap().trial.take().unwrap();
        let n = tr.steps.max(1) as f64;
        let (mut rq, mut ms) = (tr.req, tr.miss);
        for l in 0..MAXL {
          rq[l] /= n;
          ms[l] /= n;
        }
        // everything at or above the trial's depth can be costed from the trial's statistics
        let Mode::Floor(ft) = mode else { unreachable!() };
        let mut cands: Vec<(f64, Mode)> = (ft..=self.maxf).map(|c| (self.mode_cost(&rq, &ms, Mode::Floor(c)), Mode::Floor(c))).collect();
        cands.push((self.mode_cost(&rq, &ms, Mode::Brute), Mode::Brute));
        let c_prev = cands.iter().find(|c| c.1 == tr.prev).map(|c| c.0).unwrap_or(f64::INFINITY);
        let (c_best, m_best) = cands.iter().copied().min_by(|x, y| x.0.partial_cmp(&y.0).unwrap()).unwrap();
        self.trials += 1;
        let r = self.regions.get_mut(&key).unwrap();
        r.since_trial = 0;
        r.evidence_steps = 0;
        if m_best == tr.prev || c_best >= 0.9 * c_prev {
          // not worth it: go back and wait longer before the next experiment
          r.backoff = (r.backoff.max(16) * 2).min(256);
          return Some(tr.prev);
        }
        self.trials_kept += 1;
        r.backoff = 2;
        r.ereq = rq;
        r.emiss = ms;
        return if m_best == mode { None } else { Some(m_best) };
      }
      None => {}
    }
    // steady state: is a coarser mode cheaper on the smoothed evidence?
    let (ereq, emiss, steps) = {
      let r = self.regions.get_mut(&key).unwrap();
      r.evidence_steps += 1;
      (r.ereq, r.emiss, r.evidence_steps)
    };
    if steps < 2 {
      return None;
    }
    let cur_cost = self.mode_cost(&ereq, &emiss, mode);
    let mut best = (cur_cost, mode);
    if let Mode::Floor(f) = mode {
      for c in f + 1..=self.maxf {
        let cost = self.mode_cost(&ereq, &emiss, Mode::Floor(c));
        if cost < best.0 {
          best = (cost, Mode::Floor(c));
        }
      }
      let cost = self.mode_cost(&ereq, &emiss, Mode::Brute);
      if cost < best.0 {
        best = (cost, Mode::Brute);
      }
    }
    // one level finer, from the sampled sub-descents: a floor miss would cost a subdivision plus
    // its sub-windows instead of one brute-forced window
    if self.sample_shift > 0 {
      if let Mode::Floor(f) = mode {
        let (samp, sreq, smiss) = {
          let r = &self.regions[&key];
          (r.samp as f64, r.samp_req as f64, r.samp_miss as f64)
        };
        if f > a && samp >= 24.0 && sreq > 0.0 {
          let (wh, wb) = self.cost_brute_window(f + 1);
          let (fh, fb) = self.cost_brute_window(f);
          let per_miss_finer = self.cost_descend() + (sreq / samp) * (self.model.probe + (smiss / sreq) * (fh + fb));
          let finer_cost = cur_cost + emiss[f + 1] * (per_miss_finer - (wh + wb));
          if finer_cost < 0.85 * cur_cost && finer_cost < best.0 {
            best = (finer_cost, Mode::Floor(f - 1));
          }
        }
      }
    }
    let r = self.regions.get_mut(&key).unwrap();
    if best.1 != mode && best.0 < 0.85 * cur_cost {
      r.evidence_steps = 0;
      r.since_trial = 0;
      return Some(best.1);
    }
    // otherwise, is it time for a whole-step experiment? (always for plain cells, which have no
    // windows to sample; for floors only when sampling is off)
    if self.sample_shift > 0 && mode != Mode::Brute {
      return None;
    }
    r.since_trial += 1;
    // plain cells are left lazily: memo above the region already catches whole-region
    // recurrence, so the experiment only serves content that turned regular at the fine scale
    let due = if mode == Mode::Brute { r.backoff.max(16) } else { r.backoff };
    if r.since_trial >= due {
      // from plain cells the experiment jumps straight to the default depth (16x16 windows);
      // large brute windows never pay
      let finer = match mode {
        Mode::Brute => Some(Mode::Floor(a.max(3).min(self.maxf))),
        Mode::Floor(f) if f > a => Some(Mode::Floor(f - 1)),
        _ => None,
      };
      if let Some(m) = finer {
        r.trial = Some(Trial { prev: mode, left: 2, steps: 0, req: [0.0; MAXL], miss: [0.0; MAXL] });
        return Some(m);
      }
    }
    None
  }

  /// share of worked region-steps per mode
  pub fn mode_histogram(&self) -> Vec<(String, f64)> {
    let mut h = [0u64; MAXL];
    for r in self.regions.values() {
      for (i, &c) in r.time_in.iter().enumerate() {
        h[i] += c as u64;
      }
    }
    let total: u64 = h.iter().sum();
    h.iter()
      .enumerate()
      .filter(|(_, &c)| c > 0)
      .map(|(i, &c)| (if i == BRUTE_SLOT { "cells".to_string() } else { format!("{}", 1u32 << (i + 1)) }, 100.0 * c as f64 / total.max(1) as f64))
      .collect()
  }
  pub fn region_costs(&self) -> FxHashMap<(i64, i64), f64> {
    self.regions.iter().map(|(k, r)| (*k, r.cost_hash + r.cost_brute)).collect()
  }
  /// Per region with work, sorted by position: what the controller did there.
  pub fn region_report(&self) -> Vec<RegionReport> {
    let mut v: Vec<RegionReport> = self
      .regions
      .iter()
      .filter(|(_, r)| r.time_in.iter().any(|&c| c > 0))
      .map(|(k, r)| RegionReport {
        x: k.0,
        y: k.1,
        steps: r
          .time_in
          .iter()
          .enumerate()
          .filter(|(_, &c)| c > 0)
          .map(|(i, &c)| (if i == BRUTE_SLOT { 0 } else { 1u32 << (i + 1) }, c))
          .collect(),
        hash_ns: r.cost_hash,
        cells_ns: r.cost_brute,
      })
      .collect();
    v.sort_by_key(|r| (r.y, r.x));
    v
  }
}

/// One region of the controller's grid: its position in regions, the macro-steps with work it
/// spent in each mode as `(side of the brute-forced window, steps)` with side 0 for plain cells,
/// and its model cost.
pub struct RegionReport {
  pub x: i64,
  pub y: i64,
  pub steps: Vec<(u32, u32)>,
  pub hash_ns: f64,
  pub cells_ns: f64,
}

impl FloorPolicy for Policy {
  fn floor(&mut self, cx: i64, cy: i64, l: usize) -> usize {
    let (rl, a, shift, step) = (self.rl, self.a, self.sample_shift, self.step);
    let adaptive = self.kind == Kind::Adaptive;
    let r = self.region(cx, cy);
    match r.mode {
      Mode::Floor(f) => {
        if adaptive && shift > 0 && f > a && l == f + 1 {
          let h = rnd((cx as u32).wrapping_mul(0x9E3779B1) ^ (cy as u32).wrapping_mul(0x85EBCA77) ^ (step as u32).wrapping_mul(0xC2B2AE3D));
          if h & ((1 << shift) - 1) == 0 {
            r.samp += 1;
            return f - 1; // subdivide this one: evidence about the next finer level
          }
        }
        f
      }
      // executed as brute windows up to twice the region's size; charged as plain cells
      Mode::Brute => rl,
    }
  }
  fn record(&mut self, cx: i64, cy: i64, l: usize, o: Outcome) {
    let rl = self.rl;
    let r = self.region(cx, cy);
    r.req[l] += 1;
    if let Mode::Floor(f) = r.mode {
      // requests at the floor level itself only exist under sampled sub-descents
      if l == f {
        r.samp_req += 1;
        if o != Outcome::Hit {
          r.samp_miss += 1;
        }
      }
    }
    match o {
      Outcome::Hit => {}
      Outcome::Descend => r.miss[l] += 1,
      Outcome::Brute => {
        r.miss[l] += 1;
        r.brute[l] += 1;
        if l <= rl + 1 {
          r.needs_brute = true;
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------------------------

fn rnd(x0: u32) -> u32 {
  let mut x = x0;
  x ^= x >> 16;
  x = x.wrapping_mul(0x7FEB352D);
  x ^= x >> 15;
  x = x.wrapping_mul(0x846CA68B);
  x ^= x >> 16;
  x
}

fn dense(loaded: Loaded) -> Grid {
  match loaded {
    Loaded::Dense(g) => g,
    Loaded::Mc(m) => {
      let dummy = crate::ca::Rule::dummy();
      let mut e = Engine::new(&dummy, 1, 1);
      let (root, level) = e.from_mc(&m);
      let full = e.expand(root, level);
      let (x0, y0, x1, y1) = full.bbox().expect("empty pattern");
      full.crop(x0, y0, x1, y1)
    }
  }
}

/// Build a universe from `--compose "file@x,y[+gens][~seed];..."` or a single `--pattern`.
/// `+gens` pre-runs that copy, `~seed` first overwrites eight of its cells so that copies diverge.
pub fn build_universe(args: &[String], lut_for: &dyn Fn(&str) -> crate::ca::Rule) -> (Grid, crate::ca::Rule, String) {
  if let Some(spec) = opt(args, "--compose") {
    let lut = lut_for("Hutton32b");
    let mut items = Vec::new();
    for item in spec.split(';').filter(|s| !s.trim().is_empty()) {
      let (path, rest) = item.trim().split_once('@').expect("compose item needs @x,y");
      let (pos, seed) = match rest.split_once('~') {
        Some((p, s)) => (p, Some(s.parse::<u32>().expect("seed"))),
        None => (rest, None),
      };
      let (pos, pre) = match pos.split_once('+') {
        Some((p, g)) => (p, g.parse::<u64>().expect("gens")),
        None => (pos, 0),
      };
      let (x, y) = pos.split_once(',').expect("x,y");
      let (x, y) = (x.parse::<usize>().unwrap(), y.parse::<usize>().unwrap());
      let (loaded, _) = cmd_stats::load_pattern(std::path::Path::new(path));
      let mut g = dense(loaded);
      if let Some(seed) = seed {
        let live: Vec<usize> = (0..g.cells.len()).filter(|&i| g.cells[i] != 0).collect();
        for n in 0..8u32 {
          let i = live[rnd(seed.wrapping_mul(977).wrapping_add(n)) as usize % live.len()];
          let j = live[rnd(seed.wrapping_mul(31337).wrapping_add(n * 7 + 1)) as usize % live.len()];
          g.cells[i] = g.cells[j];
        }
      }
      if pre > 0 {
        let padded = g.padded(32, 32, g.w + 64, g.h + 64);
        let mut sim = Sim::new(&padded, &lut);
        for _ in 0..pre {
          sim.step();
        }
        assert!(!sim.touches_border(), "{path}: grew out of its margin during the pre-run");
        items.push((sim.grid(), x.saturating_sub(32), y.saturating_sub(32)));
      } else {
        items.push((g, x, y));
      }
    }
    let w = items.iter().map(|(g, x, _)| x + g.w).max().unwrap();
    let h = items.iter().map(|(g, _, y)| y + g.h).max().unwrap();
    let mut canvas = Grid::new(w, h);
    for (g, x, y) in &items {
      canvas.blit(g, *x, *y);
    }
    (canvas, lut, format!("composite of {} patterns", items.len()))
  } else {
    let path: std::path::PathBuf = opt(args, "--pattern").expect("--pattern or --compose").into();
    let (loaded, file_rule) = cmd_stats::load_pattern(&path);
    let lut = lut_for(&file_rule);
    (dense(loaded), lut, path.file_name().unwrap().to_string_lossy().into_owned())
  }
}

struct RunResult {
  label: String,
  finds: u64,
  inserts: u64,
  brute_evals: u64,
  hash_ms: f64,
  brute_ms: f64,
  counter_ms: f64,
  mem_mb: f64,
  note: String,
  hist: Vec<(String, f64)>,
  fp: hl::Fingerprint,
  region_costs: FxHashMap<(i64, i64), f64>,
  regions: Vec<RegionReport>,
  cpu_s: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_policy(label: &str, kind: Kind, init: Mode, g: &Grid, lut: &crate::ca::Rule, a: usize, k: usize, rl: usize, steps: u64, model: Model, maxf: usize, nominal_out: &mut f64) -> RunResult {
  let t0 = Instant::now();
  let mut e = Engine::new(lut, a, k);
  let (mut root, mut level) = e.from_grid(g);
  // keep the origin at (0, 0) while growing the root to the region grid's alignment
  while level < rl + 1 {
    let z = e.empty(level);
    root = e.intern(level + 1, [root, z, z, z]);
    level += 1;
  }
  let (mut ox, mut oy) = (0i64, 0i64);
  for lv in e.levels.iter_mut() {
    lv.stats = Default::default();
  }
  let (tf0, ti0) = (e.tile_finds, e.tile_inserts);
  let mut pol = Policy::new(kind, rl, a, k, init, model, maxf);
  let mut nominal = 0.0;
  let mut cells: Vec<(i64, i64)> = Vec::new();
  for _ in 0..steps {
    cells.clear();
    e.nonempty_at(root, level, ox, oy, rl, &mut cells);
    let nonempty: FxHashSet<(i64, i64)> = cells.iter().map(|&(x, y)| (x >> rl, y >> rl)).collect();
    // plain-cells baseline: every non-empty region stepped for the macro-step, no halo
    nominal += nonempty.len() as f64 * pol.region_brute_nominal();
    let (r, l, x, y) = e.step_root_at(root, level, ox, oy, rl + 1, &mut pol);
    root = r;
    level = l;
    ox = x;
    oy = y;
    pol.end_step(&nonempty);
  }
  *nominal_out = nominal;
  let finds: u64 = e.levels.iter().map(|l| l.stats.finds).sum();
  let inserts: u64 = e.levels.iter().map(|l| l.stats.inserts).sum();
  let brute_evals: u64 = e.brute_win_evals.iter().sum();
  // cost straight from the engine's counters (every window brute-forced in isolation)
  let counter_ns = finds as f64 * model.probe
    + inserts as f64 * model.insert
    + (e.tile_finds - tf0) as f64 * model.probe
    + (e.tile_inserts - ti0) as f64 * model.insert
    + e.expand_tiles as f64 * model.gather
    + brute_evals as f64 * model.cell;
  let tile_bytes = (1usize << (2 * a)) as f64 * 5.0 / 8.0;
  let mem = inserts as f64 * 32.0 + (e.tile_inserts - ti0) as f64 * (tile_bytes + 8.0);
  let fp = hl::fingerprint(|v| e.for_each_cell(root, level, 0, 0, &mut |x, y, c| v(x, y, c)));
  let note = if pol.trials > 0 || pol.switches > 0 {
    format!("  [{} mode changes; {} of {} finer trials kept]", pol.switches, pol.trials_kept, pol.trials)
  } else {
    String::new()
  };
  RunResult {
    label: label.to_string(),
    finds,
    inserts,
    brute_evals,
    hash_ms: pol.total_hash / 1e6,
    brute_ms: pol.total_brute / 1e6,
    counter_ms: counter_ns / 1e6,
    mem_mb: mem / 1048576.0,
    note,
    hist: pol.mode_histogram(),
    fp,
    region_costs: pol.region_costs(),
    regions: pol.region_report(),
    cpu_s: t0.elapsed().as_secs_f64(),
  }
}

/// What `overview` shows of the controller: the adaptive policy (Algorithm 3 of the paper) on the
/// universe `g`, as `controller` runs it.
pub struct ControllerMap {
  pub regions: Vec<RegionReport>,
  /// model time of the adaptive policy and of plain cells over the non-empty regions, in ms
  pub adaptive_ms: f64,
  pub plain_ms: f64,
  /// share of worked region-steps per mode, as the `adapt` table prints it
  pub hist: Vec<(String, f64)>,
  pub note: String,
}

pub fn controller_map(g: &Grid, lut: &crate::ca::Rule, a: usize, k: usize, rl: usize, gens: u64) -> ControllerMap {
  let maxf = 4usize.clamp(a, rl - 1);
  let mut plain_ns = 0.0;
  let r = run_policy("adaptive", Kind::Adaptive, Mode::Floor(a.clamp(a, maxf)), g, lut, a, k, rl, gens >> k, Model::default(), maxf, &mut plain_ns);
  ControllerMap { adaptive_ms: r.hash_ms + r.brute_ms, plain_ms: plain_ns / 1e6, hist: r.hist, note: r.note.trim().to_string(), regions: r.regions }
}

pub fn run(args: &[String]) {
  let lut_for = |file_rule: &str| crate::load_lut(&cmd_stats::variant_for(args, file_rule));
  let (g, lut, name) = build_universe(args, &lut_for);
  let a: usize = opt_or(args, "--a", 2);
  let k: usize = opt_or(args, "--k", 6);
  let rl: usize = opt_or(args, "--rl", 8);
  let gens: u64 = opt_or(args, "--gens", 8192);
  let steps = gens >> k;
  assert!(k >= a && rl >= k + 2, "need a <= k <= rl - 2");
  println!(
    "universe: {name}, {}x{} cells, population {} | base tile {}x{}, macro-step {} gens, region {}x{} | {} gens",
    g.w,
    g.h,
    g.population(),
    1 << a,
    1 << a,
    1u64 << k,
    1u64 << rl,
    1u64 << rl,
    steps << k
  );
  let which = opt(args, "--policies").unwrap_or("all".into());
  // floors above this are dominated on every pattern tried (isolated large windows cost 4x)
  let maxf: usize = opt_or(args, "--max-floor", 4usize).clamp(a, rl - 1);
  // price of memory in ns per byte: makes every insert dearer (table rebuilds, eviction)
  let mem_price: f64 = opt_or(args, "--mem-price", 0.0);
  let mut model = Model::default();
  model.insert += 32.0 * mem_price;
  // brute kernel rate in cell-updates/s: 1.86e11 for the byte-LUT kernel (default), 2.9e11 for
  // the bit-plane kernel of cmd_bitslice.rs
  model.cell = 1e9 / opt_or(args, "--cell-rate", 1.86e11);
  let mut plan: Vec<(String, Kind, Mode)> = Vec::new();
  for f in a..=maxf {
    plan.push((format!("fixed: brute at {0}x{0}", 1u32 << (f + 1)), Kind::Fixed(Mode::Floor(f)), Mode::Floor(f)));
  }
  plan.push(("fixed: plain cells".into(), Kind::Fixed(Mode::Brute), Mode::Brute));
  plan.push(("adaptive".into(), Kind::Adaptive, Mode::Floor(opt_or(args, "--init-floor", a + 1).clamp(a, maxf))));
  let mut runs: Vec<RunResult> = Vec::new();
  let mut nominal = 0.0;
  for (label, kind, init) in plan {
    if which != "all" && !which.split(',').any(|w| label.contains(w)) {
      continue;
    }
    let r = run_policy(&label, kind, init, &g, &lut, a, k, rl, steps, model, maxf, &mut nominal);
    eprintln!("  [{:>7.1}s cpu] {}", r.cpu_s, r.label);
    runs.push(r);
  }
  for r in &runs[1..] {
    assert_eq!(r.fp, runs[0].fp, "policy '{}' produced a different universe", r.label);
  }
  println!("all {} policies end in the same state: population {} hash {:016x}", runs.len(), runs[0].fp.population, runs[0].fp.hash);
  let m = model;
  let nominal_ms = nominal / 1e6;
  println!("plain cells over non-empty regions, no halo: {:.2} ms model time ({:.3e} cell-updates)\n", nominal_ms, nominal / m.cell);
  println!(
    "{:<26} {:>11} {:>10} {:>11} {:>8} {:>9} {:>9} {:>8} {:>8}  time per mode",
    "policy", "finds", "inserts", "brute evals", "hash ms", "cells ms", "total ms", "vs cells", "mem MiB"
  );
  for r in &runs {
    let total = r.hash_ms + r.brute_ms;
    let hist: Vec<String> = r.hist.iter().map(|(n, p)| format!("{n}:{p:.0}%")).collect();
    println!(
      "{:<26} {:>11} {:>10} {:>11.3e} {:>8.2} {:>9.2} {:>9.2} {:>7.2}x {:>8.1}  {}{}",
      r.label,
      r.finds,
      r.inserts,
      r.brute_evals as f64,
      r.hash_ms,
      r.brute_ms,
      total,
      nominal_ms / total,
      r.mem_mb,
      hist.join(" "),
      r.note
    );
  }
  // best fixed mode per region, in hindsight
  let fixed: Vec<&RunResult> = runs.iter().filter(|r| r.label.starts_with("fixed")).collect();
  if fixed.len() > 1 {
    let mut keys: Vec<(i64, i64)> = fixed.iter().flat_map(|r| r.region_costs.keys().copied()).collect();
    keys.sort_unstable();
    keys.dedup();
    let mut oracle = 0.0;
    let mut choice: FxHashMap<String, u32> = Default::default();
    for key in &keys {
      let (best, lab) = fixed
        .iter()
        .map(|r| (r.region_costs.get(key).copied().unwrap_or(0.0), &r.label))
        .min_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
        .unwrap();
      oracle += best;
      if best > 0.0 {
        *choice.entry(lab.replace("fixed: ", "")).or_default() += 1;
      }
    }
    let mut ch: Vec<(String, u32)> = choice.into_iter().collect();
    ch.sort();
    println!(
      "{:<26} {:>11} {:>10} {:>11} {:>8} {:>9} {:>9.2} {:>7.2}x {:>8}  regions: {}",
      "best fixed per region",
      "",
      "",
      "",
      "",
      "",
      oracle / 1e6,
      nominal_ms / (oracle / 1e6),
      "",
      ch.iter().map(|(l, n)| format!("{l}: {n}")).collect::<Vec<_>>().join(", ")
    );
  }
  println!("\ncross-check, cost from raw engine counters (every brute window isolated, no plain-cells model):");
  println!("  {}", runs.iter().map(|r| format!("{} {:.2} ms", r.label.replace("fixed: ", ""), r.counter_ms)).collect::<Vec<_>>().join(" | "));
}
