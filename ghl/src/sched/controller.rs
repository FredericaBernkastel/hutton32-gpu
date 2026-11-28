//! The engine per region. Algorithm 3 of the paper, ported from the harness (`cmd_adapt.rs`,
//! `Policy`), decides per region of `2^rl` cells how deep a missed window may be subdivided
//! (`Mode::Floor(f)`: windows of side `2^(f+1)` are brute-forced) or whether the region is stepped
//! as plain cells (`Mode::Cells`). It is a [`FloorPolicy`] for the host recursion, and it charges
//! every macro-step to the regions with a cost model, so that the same run also prices plain cells
//! in every region (the region model: `PlainModel`, `plain_cost`) and, after a fixed-floor run, the
//! cheaper engine per region in hindsight (`hindsight`).
//!
//! Algorithm 3 enters plain cells on bursts of misses that the floor would have learned; against
//! the choice in hindsight it costs 49–55 % more on Hutton32b and it loses to the uniform floor on
//! Life (paper Table XXIX). Its replacement, [`Kind::Priced`], is a rent rule on the two prices of a
//! region (`decide_priced`). At the roadmap's three configurations it passes its acceptance test
//! (`tests/controller.rs`, model times): 1.12 and 1.19 times the choice in hindsight on Hutton32b,
//! 1.02 times on Life, no universe above the uniform floor. At every configuration of paper Tables
//! XXVII and XXVIII the sums stay within 1.17 times the choice in hindsight, but the Unit Life Cell
//! costs 1.11 and 1.34 times the floor at 16-generation steps with 64×64 and 128×128 regions, so the
//! controller ships disabled. Content that is new for about 250 generations and then recurs for far
//! longer (the Unit Life Cell at short steps, the ROM at 64-generation steps) prices, while the
//! controller has to choose, as content that never recurs (a soup, the ROM at short steps) does.

use rustc_hash::{FxHashMap, FxHashSet};

use super::cost::Model;
use crate::cpu::{FloorPolicy, Outcome};

pub const MAXL: usize = 32;
const CELLS_SLOT: usize = MAXL - 1;

/// What a region runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Mode {
  /// missed windows of side `2^(f+1)` are brute-forced, larger ones subdivided and memoized
  Floor(usize),
  /// plain cells: nothing below the region is memoized
  Cells,
}

/// Which policy runs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
  Fixed(Mode),
  /// Algorithm 3
  Adaptive,
  /// plain cells in the regions of `Policy::assign`, the initial floor elsewhere
  Assigned,
  /// the price-based controller: a rent rule on the floor's price against plain cells'
  /// (`decide_priced`)
  Priced,
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
  req: [u32; MAXL],
  miss: [u32; MAXL],
  brute: [u32; MAXL],
  needs_brute: bool,
  samp: u64,
  samp_req: u64,
  samp_miss: u64,
  ereq: [f64; MAXL],
  emiss: [f64; MAXL],
  seen: [bool; MAXL],
  evidence_steps: u32,
  trial: Option<Trial>,
  backoff: u32,
  since_trial: u32,
  cost_hash: f64,
  cost_brute: f64,
  time_in: [u32; MAXL],
  memo_below: f64,
  top_misses: u64,
  switches: u32,
  last_cells_price: f64,
  /// the price-based controller: what the floor cost beyond `PRICED_RHO` times plain cells, summed
  /// over the region's macro-steps with a region-level miss since it last had none
  excess: f64,
  /// with `log`: what other regions saved in this macro-step by hitting results this region computed
  step_credit: f64,
}

/// The price-based controller counts the floor's price beyond this many times the price of plain
/// cells as evidence of content the floor cannot learn. On the paper's universes a floor that learns
/// costs two to three times plain cells while it learns (Life's first steps at 64-generation steps,
/// the counter's bursts), content that does not recur four and a half to six times (the ROM's data,
/// a soup).
/// At shorter steps on Life the first period of recurring content costs as much as a soup's, which
/// is where the controller loses (`implementation.md`, Section 9). Chosen by simulating the rule on
/// the floor runs' price traces (`tests/fixtures/controller_sim.py`), where 2.5 and 3.5 pass the
/// acceptance criteria as well with the threshold below.
pub const PRICED_RHO: f64 = 3.0;
/// ... and enters plain cells once that evidence exceeds this many plain-cells steps of a nominal
/// region (1 to 3 pass in that simulation with the premium above).
pub const PRICED_B: f64 = 2.0;

/// What plain cells in every region would have cost in this run, whatever engine ran.
#[derive(Clone, Copy, Default, Debug)]
pub struct PlainModel {
  pub nonempty: u64,
  pub charged: u64,
  pub own_evals: f64,
  pub halo_evals: f64,
}

struct StepTrace {
  top_miss: Vec<(i64, i64)>,
  any_brute: Vec<(i64, i64)>,
  near: Vec<(i64, i64)>,
}

/// Cell evaluations of one charged region for a macro-step of `t` generations: its own cells, and
/// the halo outside the charged set.
pub fn block_evals(charged: &FxHashSet<(i64, i64)>, x: i64, y: i64, s: f64, t: u64) -> (f64, f64) {
  let out = |dx: i64, dy: i64| !charged.contains(&(x + dx, y + dy));
  let sides = [out(-1, 0), out(1, 0), out(0, -1), out(0, 1)].iter().filter(|&&b| b).count() as f64;
  let corner = |dx: i64, dy: i64| out(dx, 0) && out(0, dy) && out(dx, dy);
  let corners = [corner(-1, -1), corner(1, -1), corner(-1, 1), corner(1, 1)].iter().filter(|&&b| b).count() as f64;
  let mut halo = 0.0;
  for r in 0..t {
    let r = r as f64;
    halo += sides * r * s + corners * r * r;
  }
  (s * s * t as f64, halo)
}

fn near(set: &FxHashSet<(i64, i64)>, x: i64, y: i64) -> bool {
  (-1..=1).any(|dy| (-1..=1).any(|dx| set.contains(&(x + dx, y + dy))))
}

/// One region in one macro-step, as [`Policy::log`] records it.
#[derive(Clone, Copy, Debug)]
pub struct PriceRow {
  pub step: u64,
  pub x: i64,
  pub y: i64,
  pub mode: Mode,
  /// region-level windows that missed
  pub top_misses: u32,
  /// what the floor engine cost at and below the region level
  pub floor_ns: f64,
  /// what plain cells would have cost the region alone (`None`: they would not have stepped it)
  pub cells_ns: Option<f64>,
  /// what other regions saved by hitting results this region computed
  pub credit_ns: f64,
  /// what was charged
  pub charged_ns: f64,
  /// the part of `floor_ns` spent on brute-forced windows
  pub brute_ns: f64,
  /// requests and misses at the brute-forced level and at the levels between it and the region
  pub lo: (u32, u32),
  pub hi: (u32, u32),
}

/// The choice of [`Policy::hindsight`].
#[derive(Clone, Debug)]
pub struct Hindsight {
  pub cells: FxHashSet<(i64, i64)>,
  pub ns: f64,
  pub floor_ns: f64,
  pub regions: usize,
}

/// One region of the controller's grid: its position in regions, the macro-steps with work it
/// spent in each mode as `(side of the brute-forced window, steps)` with side 0 for plain cells, and
/// its model cost.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionReport {
  pub x: i64,
  pub y: i64,
  pub steps: Vec<(u32, u32)>,
  pub hash_ns: f64,
  pub cells_ns: f64,
}

/// A region as the inspector shows it.
#[derive(Clone, Debug, PartialEq)]
pub struct RegionState {
  pub x: i64,
  pub y: i64,
  pub mode: Mode,
  /// macro-steps with work at the floor and in plain cells
  pub steps: [u32; 2],
  /// what the floor engine cost at and below the region level, ns
  pub price_floor_ns: f64,
  /// what plain cells would have cost alone, ns
  pub price_cells_ns: f64,
  /// what was charged, ns
  pub charged_ns: f64,
  pub switches: u32,
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
  pub maxf: usize,
  pub sample_shift: u32,
  step: u64,
  pub assign: FxHashSet<(i64, i64)>,
  lvl_req: [u64; MAXL],
  lvl_miss: [u64; MAXL],
  top_miss: FxHashSet<(i64, i64)>,
  any_brute: FxHashSet<(i64, i64)>,
  pub plain: PlainModel,
  trace: Vec<StepTrace>,
  /// with `log`: which region computed a window's result, and at what cost
  computed_by: FxHashMap<(usize, u32), ((i64, i64), f64)>,
  /// keep the per-step trace that `hindsight` needs (fixed-floor runs)
  pub keep_trace: bool,
  /// every region's prices per macro-step, when `Some` (analysis of controllers)
  pub log: Option<Vec<PriceRow>>,
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
      assign: Default::default(),
      lvl_req: [0; MAXL],
      lvl_miss: [0; MAXL],
      top_miss: Default::default(),
      any_brute: Default::default(),
      plain: Default::default(),
      trace: Vec::new(),
      computed_by: FxHashMap::default(),
      keep_trace: matches!(kind, Kind::Fixed(Mode::Floor(_))),
      log: None,
      regions: Default::default(),
      init,
      total_hash: 0.0,
      total_brute: 0.0,
      switches: 0,
      trials: 0,
      trials_kept: 0,
    }
  }

  /// Macro-steps closed so far.
  pub fn steps(&self) -> u64 {
    self.step
  }

  fn region(&mut self, cx: i64, cy: i64) -> &mut Region {
    let key = (cx >> self.rl, cy >> self.rl);
    let init = match self.kind {
      Kind::Fixed(m) => m,
      Kind::Adaptive | Kind::Priced => self.init,
      Kind::Assigned => {
        if self.assign.contains(&key) {
          Mode::Cells
        } else {
          self.init
        }
      }
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
      memo_below: 0.0,
      top_misses: 0,
      switches: 0,
      last_cells_price: 0.0,
      excess: 0.0,
      step_credit: 0.0,
    })
  }

  /// The mode of the region that contains cell `(x, y)`, without creating it.
  pub fn mode_at(&self, x: i64, y: i64) -> Mode {
    let key = (x >> self.rl, y >> self.rl);
    match self.regions.get(&key) {
      Some(r) => r.mode,
      None => match self.kind {
        Kind::Fixed(m) => m,
        Kind::Assigned if self.assign.contains(&key) => Mode::Cells,
        _ => self.init,
      },
    }
  }

  fn window_evals(&self, l: usize) -> f64 {
    (1u64 << (2 * l)) as f64 * (1u64 << (l - 2).min(self.k)) as f64
  }
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
  fn cost_descend(&self) -> f64 {
    self.model.insert + 10.0 * self.model.probe
  }
  fn cost_brute_window(&self, l: usize) -> (f64, f64) {
    let hash = self.model.insert + self.intern_count(l) * (self.model.probe + 0.5 * self.model.insert) + (1u64 << (2 * (l - self.a))) as f64 * self.model.gather;
    (hash, self.window_evals(l) * self.model.cell)
  }
  /// Stepping one region as plain cells for one macro-step, without halo.
  pub fn region_brute_nominal(&self) -> f64 {
    (1u64 << (2 * self.rl)) as f64 * (1u64 << self.k) as f64 * self.model.cell
  }
  fn region_convert(&self) -> f64 {
    let tiles = (1u64 << (2 * (self.rl - self.a))) as f64;
    tiles * (self.model.gather + 1.34 * (self.model.probe + 0.5 * self.model.insert))
  }

  fn mode_cost(&self, req: &[f64; MAXL], miss: &[f64; MAXL], mode: Mode) -> f64 {
    match mode {
      Mode::Cells => 1.3 * self.region_brute_nominal() + self.region_convert(),
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

  /// Close a macro-step: charge costs and let adaptive regions reconsider their mode. `nonempty`
  /// holds the regions that contained live cells when the step began.
  pub fn end_step(&mut self, nonempty: &FxHashSet<(i64, i64)>) {
    let (rl, a, k) = (self.rl, self.a, self.k);
    let keys: Vec<(i64, i64)> = self.regions.keys().copied().collect();
    let mut charged: FxHashSet<(i64, i64)> = Default::default();
    for &key in nonempty {
      let unseen = matches!(self.kind, Kind::Fixed(Mode::Cells)) || (self.kind == Kind::Assigned && self.assign.contains(&key));
      let is_cells = self.regions.get(&key).map(|r| r.mode == Mode::Cells).unwrap_or(unseen);
      if !is_cells {
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
      let (own, halo) = block_evals(&charged, x, y, s, t);
      cells_cost.insert((x, y), (own + halo) * self.model.cell);
    }
    // the region model: the same charge with every region in plain cells
    let alone_cells: FxHashMap<(i64, i64), f64>;
    {
      let all: FxHashSet<(i64, i64)> = nonempty.iter().copied().filter(|&(x, y)| near(&self.top_miss, x, y)).collect();
      self.plain.nonempty += nonempty.len() as u64;
      self.plain.charged += all.len() as u64;
      let mut alone = FxHashMap::default();
      for &(x, y) in &all {
        let (own, halo) = block_evals(&all, x, y, s, t);
        self.plain.own_evals += own;
        self.plain.halo_evals += halo;
        alone.insert((x, y), (own + halo) * self.model.cell + self.region_convert());
      }
      alone_cells = alone;
      if self.keep_trace {
        let near_any: Vec<(i64, i64)> = nonempty.iter().copied().filter(|&(x, y)| near(&self.top_miss, x, y) || near(&self.any_brute, x, y)).collect();
        self.trace.push(StepTrace { top_miss: self.top_miss.iter().copied().collect(), any_brute: self.any_brute.iter().copied().collect(), near: near_any });
      }
      self.top_miss.clear();
      self.any_brute.clear();
    }
    let window_cost: Vec<(f64, f64)> = (0..MAXL).map(|l| if l > a && l < CELLS_SLOT { self.cost_brute_window(l) } else { (0.0, 0.0) }).collect();
    let (probe, insert, descend) = (self.model.probe, self.model.insert, self.cost_descend());
    let convert = self.region_convert();
    let kind = self.kind;
    let (mut th, mut tb) = (0.0, 0.0);
    for &key in &keys {
      let (mode, mut ch, mut cb, below) = {
        let reg = &self.regions[&key];
        let (mut ch, mut cb, mut below) = (0.0, 0.0, 0.0);
        for l in a + 1..CELLS_SLOT {
          if reg.req[l] == 0 {
            continue;
          }
          let before = ch + cb;
          ch += reg.req[l] as f64 * probe + (reg.miss[l] - reg.brute[l]) as f64 * descend;
          if reg.mode == Mode::Cells && l <= rl + 1 {
            ch += reg.brute[l] as f64 * insert;
          } else {
            ch += reg.brute[l] as f64 * window_cost[l].0;
            cb += reg.brute[l] as f64 * window_cost[l].1;
          }
          if l <= rl {
            below += ch + cb - before;
          } else if l == rl + 1 {
            below += reg.miss[l] as f64 * descend;
          }
        }
        (reg.mode, ch, cb, below)
      };
      if let Some(&c) = cells_cost.get(&key) {
        cb += c;
        ch += convert;
      }
      let worked = ch + cb > 0.0;
      th += ch;
      tb += cb;
      let decision = match kind {
        Kind::Adaptive => self.decide(key),
        Kind::Priced => self.decide_priced(key, below, alone_cells.get(&key).copied()),
        _ => None,
      };
      let reg = self.regions.get_mut(&key).unwrap();
      if let Some(log) = self.log.as_mut() {
        if worked || reg.miss[rl + 1] > 0 || reg.step_credit > 0.0 {
          let f = match mode {
            Mode::Floor(f) => f,
            Mode::Cells => rl,
          };
          let brute_ns: f64 = (a + 1..=rl).map(|l| reg.brute[l] as f64 * (window_cost[l].0 + window_cost[l].1)).sum();
          let lo = (reg.req[f + 1], reg.miss[f + 1]);
          let hi = (f + 2..=rl).fold((0, 0), |(q, m), l| (q + reg.req[l], m + reg.miss[l]));
          log.push(PriceRow { step: self.step, x: key.0, y: key.1, mode, top_misses: reg.miss[rl + 1], floor_ns: below, cells_ns: alone_cells.get(&key).copied(), credit_ns: reg.step_credit, charged_ns: ch + cb, brute_ns, lo, hi });
        }
      }
      reg.step_credit = 0.0;
      reg.cost_hash += ch;
      reg.cost_brute += cb;
      reg.memo_below += below;
      reg.top_misses += reg.miss[rl + 1] as u64;
      if let Some(&c) = alone_cells.get(&key) {
        reg.last_cells_price += c;
      }
      if worked {
        reg.time_in[match mode {
          Mode::Floor(f) => f,
          Mode::Cells => CELLS_SLOT,
        }] += 1;
      }
      if let Some(m) = decision {
        if m != reg.mode {
          self.switches += 1;
          reg.switches += 1;
          reg.samp = 0;
          reg.samp_req = 0;
          reg.samp_miss = 0;
        }
        let lowest_new = match m {
          Mode::Floor(f) => f + 1,
          Mode::Cells => rl + 1,
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

  /// Algorithm 3. Returns the mode for the next macro-step, or `None` to keep it.
  fn decide(&mut self, key: (i64, i64)) -> Option<Mode> {
    let (rl, a) = (self.rl, self.a);
    let (mode, lowest, evidence) = {
      let r = &self.regions[&key];
      let lowest = match r.mode {
        Mode::Floor(f) => f + 1,
        Mode::Cells => rl + 1,
      };
      let top_missed = r.miss[rl + 1] > 0 || r.miss[rl] > 0;
      (r.mode, lowest, top_missed && r.req[lowest.min(rl + 1)] > 0)
    };
    if !evidence {
      return None;
    }
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
        let Mode::Floor(ft) = mode else { unreachable!() };
        let mut cands: Vec<(f64, Mode)> = (ft..=self.maxf).map(|c| (self.mode_cost(&rq, &ms, Mode::Floor(c)), Mode::Floor(c))).collect();
        cands.push((self.mode_cost(&rq, &ms, Mode::Cells), Mode::Cells));
        let c_prev = cands.iter().find(|c| c.1 == tr.prev).map(|c| c.0).unwrap_or(f64::INFINITY);
        let (c_best, m_best) = cands.iter().copied().min_by(|x, y| x.0.partial_cmp(&y.0).unwrap()).unwrap();
        self.trials += 1;
        let r = self.regions.get_mut(&key).unwrap();
        r.since_trial = 0;
        r.evidence_steps = 0;
        if m_best == tr.prev || c_best >= 0.9 * c_prev {
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
      let cost = self.mode_cost(&ereq, &emiss, Mode::Cells);
      if cost < best.0 {
        best = (cost, Mode::Cells);
      }
    }
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
    if self.sample_shift > 0 && mode != Mode::Cells {
      return None;
    }
    r.since_trial += 1;
    let due = if mode == Mode::Cells { r.backoff.max(16) } else { r.backoff };
    if r.since_trial >= due {
      let finer = match mode {
        Mode::Cells => Some(Mode::Floor(a.max(3).min(self.maxf))),
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

  /// The price-based controller, a rent rule on the two prices of the region. `below` is what the
  /// floor engine cost at and below the region level in this macro-step, `cells` what plain cells
  /// would have cost the region (`None` when they would not have stepped it).
  ///
  /// While the region misses at the region level, every macro-step at the floor adds the floor's
  /// price minus [`PRICED_RHO`] times the price of plain cells to the region's excess, and the
  /// region enters plain cells once the excess exceeds [`PRICED_B`] nominal plain-cells steps of a
  /// region. The premium keeps learning out of the evidence: a floor that learns costs two to three
  /// times plain cells while it learns and repays it afterwards, content that does not recur costs
  /// it four and a half to six times plain cells for as long as it lasts. It also covers learning that other
  /// regions reuse, which a price per region misses: crediting the regions that computed the results
  /// others hit changed nothing on the paper's universes once the premium was in place, so the
  /// controller does not pay for that bookkeeping. A step in which the region has no content adds
  /// nothing: its window misses because of its neighbours' cells, plain cells would not step it, and
  /// counted at their price of one insert such steps carried regions at a growing pattern's front
  /// into plain cells just as content arrived (catacryst). The threshold is in units of a nominal
  /// stepped region for the same reason. The excess is bounded below so that a region that was cheap
  /// for long still enters within a few steps once its content stops recurring.
  ///
  /// A step without a region-level miss means the region's content recurs above the floor, where
  /// both engines cost a probe: the region returns to the floor and forgets its excess, so that the
  /// next burst of new content is priced afresh (a burst that the floor learns, as on the counter,
  /// is never charged to plain cells).
  fn decide_priced(&mut self, key: (i64, i64), below: f64, cells: Option<f64>) -> Option<Mode> {
    let rl = self.rl;
    let init = self.init;
    let insert = self.model.insert;
    let nominal = self.region_brute_nominal() + self.region_convert();
    let r = self.regions.get_mut(&key).unwrap();
    if r.miss[rl + 1] == 0 {
      r.excess = 0.0;
      return if r.mode == Mode::Cells { Some(init) } else { None };
    }
    // in plain cells nothing is learned below the region; without content (a region-level window
    // that misses because of its neighbours' cells) plain cells would not step the region, and the
    // two prices say nothing about content the floor could learn
    let (Mode::Floor(_), Some(cells)) = (r.mode, cells) else {
      return None;
    };
    let c = cells + r.miss[rl + 1] as f64 * insert;
    r.excess = (r.excess + below - PRICED_RHO * c).max(-PRICED_B * nominal);
    if r.excess > PRICED_B * nominal {
      r.excess = 0.0;
      return Some(Mode::Cells);
    }
    None
  }

  fn common_cost(&self) -> f64 {
    let req: u64 = self.lvl_req[self.rl + 1..].iter().sum();
    let descends: u64 = self.lvl_miss[self.rl + 2..].iter().sum();
    req as f64 * self.model.probe + descends as f64 * self.cost_descend()
  }

  /// Model time of plain cells in every region: (memo at and above the region, conversion, cells).
  pub fn plain_cost(&self) -> (f64, f64, f64) {
    let memo = self.common_cost() + self.lvl_miss[self.rl + 1] as f64 * self.model.insert;
    (memo, self.plain.charged as f64 * self.region_convert(), (self.plain.own_evals + self.plain.halo_evals) * self.model.cell)
  }

  fn cells_given(&self, set: &FxHashSet<(i64, i64)>) -> (f64, FxHashMap<(i64, i64), f64>) {
    let (s, t) = ((1u64 << self.rl) as f64, 1u64 << self.k);
    let (cell, convert) = (self.model.cell, self.region_convert());
    let mut total = 0.0;
    let mut alone: FxHashMap<(i64, i64), f64> = Default::default();
    for st in &self.trace {
      let tm: FxHashSet<(i64, i64)> = st.top_miss.iter().copied().collect();
      let ab: FxHashSet<(i64, i64)> = st.any_brute.iter().copied().collect();
      let needs = |n: (i64, i64), also: Option<(i64, i64)>| if set.contains(&n) || also == Some(n) { tm.contains(&n) } else { ab.contains(&n) };
      let stepped = |x: i64, y: i64, also: Option<(i64, i64)>| (-1..=1).any(|dy| (-1..=1).any(|dx| needs((x + dx, y + dy), also)));
      let charged: FxHashSet<(i64, i64)> = st.near.iter().copied().filter(|&(x, y)| set.contains(&(x, y)) && stepped(x, y, None)).collect();
      for &(x, y) in &st.near {
        let inside = charged.contains(&(x, y));
        if !inside && (set.contains(&(x, y)) || !stepped(x, y, Some((x, y)))) {
          continue;
        }
        let (own, halo) = block_evals(&charged, x, y, s, t);
        let c = (own + halo) * cell + convert;
        *alone.entry((x, y)).or_default() += c;
        if inside {
          total += c;
        }
      }
    }
    (total, alone)
  }

  /// The cheaper of the two engines per region in hindsight, from a fixed-floor run.
  pub fn hindsight(&self) -> Hindsight {
    let common = self.common_cost();
    let insert = self.model.insert;
    let total_for = |set: &FxHashSet<(i64, i64)>, cells: f64| common + cells + self.regions.iter().map(|(k, r)| if set.contains(k) { r.top_misses as f64 * insert } else { r.memo_below }).sum::<f64>();
    let none: FxHashSet<(i64, i64)> = Default::default();
    let floor_ns = total_for(&none, 0.0);
    let (mut best, mut best_set) = (floor_ns, none.clone());
    let every: FxHashSet<(i64, i64)> = self.regions.keys().copied().chain(self.trace.iter().flat_map(|st| st.near.iter().copied())).collect();
    for start in [none, every] {
      let mut set = start;
      let (cells, mut alone) = self.cells_given(&set);
      let total = total_for(&set, cells);
      if total < best {
        best = total;
        best_set = set.clone();
      }
      for _ in 0..16 {
        let next: FxHashSet<(i64, i64)> = self.regions.iter().filter(|(k, r)| alone.get(k).copied().unwrap_or(0.0) + r.top_misses as f64 * insert < r.memo_below).map(|(k, _)| *k).collect();
        if next == set {
          break;
        }
        let (cells, al) = self.cells_given(&next);
        let total = total_for(&next, cells);
        set = next;
        alone = al;
        if total < best {
          best = total;
          best_set = set.clone();
        }
      }
    }
    Hindsight { cells: best_set, ns: best, floor_ns, regions: self.regions.values().filter(|r| r.memo_below > 0.0).count() }
  }

  /// Share of worked region-steps per mode.
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
      .map(|(i, &c)| (if i == CELLS_SLOT { "cells".to_string() } else { format!("{}", 1u32 << (i + 1)) }, 100.0 * c as f64 / total.max(1) as f64))
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
        steps: r.time_in.iter().enumerate().filter(|(_, &c)| c > 0).map(|(i, &c)| (if i == CELLS_SLOT { 0 } else { 1u32 << (i + 1) }, c)).collect(),
        hash_ns: r.cost_hash,
        cells_ns: r.cost_brute,
      })
      .collect();
    v.sort_by_key(|r| (r.y, r.x));
    v
  }

  /// The regions whose cells intersect `[x0, x1) x [y0, y1)` (cells), sorted by position.
  pub fn regions_in(&self, x0: i64, y0: i64, x1: i64, y1: i64) -> Vec<RegionState> {
    let rl = self.rl;
    let (rx0, ry0, rx1, ry1) = (x0 >> rl, y0 >> rl, (x1 - 1) >> rl, (y1 - 1) >> rl);
    let mut v: Vec<RegionState> = self
      .regions
      .iter()
      .filter(|(k, _)| k.0 >= rx0 && k.0 <= rx1 && k.1 >= ry0 && k.1 <= ry1)
      .map(|(k, r)| {
        let cells_steps = r.time_in[CELLS_SLOT];
        let floor_steps: u32 = r.time_in[..CELLS_SLOT].iter().sum();
        RegionState {
          x: k.0 << rl,
          y: k.1 << rl,
          mode: r.mode,
          steps: [floor_steps, cells_steps],
          price_floor_ns: r.memo_below,
          price_cells_ns: r.last_cells_price + r.top_misses as f64 * self.model.insert,
          charged_ns: r.cost_hash + r.cost_brute,
          switches: r.switches,
        }
      })
      .collect();
    v.sort_by_key(|r| (r.y, r.x));
    v
  }

  /// Forget everything learned about the regions (after an edit, a change of step or region).
  pub fn reset(&mut self) {
    self.regions.clear();
    self.top_miss.clear();
    self.any_brute.clear();
    self.trace.clear();
    self.computed_by.clear();
  }

  /// The host renumbered its nodes: forget who computed what.
  pub fn forget_nodes(&mut self) {
    self.computed_by.clear();
  }
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
            return f - 1;
          }
        }
        f
      }
      Mode::Cells => rl,
    }
  }
  fn record_node(&mut self, l: usize, id: u32, cx: i64, cy: i64, o: Outcome) {
    // the credit is for analysis only (`log`, `PriceRow::credit_ns`)
    if self.log.is_none() || l > self.rl + 1 {
      return;
    }
    let key = (cx >> self.rl, cy >> self.rl);
    match o {
      Outcome::Hit => {
        if let Some(&(by, cost)) = self.computed_by.get(&(l, id)) {
          if by != key {
            if let Some(r) = self.regions.get_mut(&by) {
              r.step_credit += cost;
            }
          }
        }
      }
      Outcome::Brute | Outcome::Descend => {
        let cost = if o == Outcome::Brute {
          let (h, b) = self.cost_brute_window(l);
          h + b
        } else {
          self.cost_descend()
        };
        if self.computed_by.len() > 1 << 22 {
          self.computed_by.clear();
        }
        self.computed_by.insert((l, id), (key, cost));
      }
    }
  }
  fn record(&mut self, cx: i64, cy: i64, l: usize, o: Outcome) {
    let rl = self.rl;
    let key = (cx >> rl, cy >> rl);
    self.lvl_req[l] += 1;
    if o != Outcome::Hit {
      self.lvl_miss[l] += 1;
      if l == rl + 1 {
        self.top_miss.insert(key);
      }
    }
    if o == Outcome::Brute {
      self.any_brute.insert(key);
    }
    let r = self.region(cx, cy);
    r.req[l] += 1;
    if let Mode::Floor(f) = r.mode {
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

/// The harness's integer mixer for sampled sub-descents.
pub fn rnd(x0: u32) -> u32 {
  let mut x = x0;
  x ^= x >> 16;
  x = x.wrapping_mul(0x7FEB352D);
  x ^= x >> 15;
  x = x.wrapping_mul(0x846CA68B);
  x ^= x >> 16;
  x
}
