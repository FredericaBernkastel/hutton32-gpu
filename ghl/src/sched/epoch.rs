//! Algorithm 1 of the paper: per epoch, the regime from the active area, and in the latency-bound
//! regime the fastest of the host recursion, the shared-tile kernel and the plain kernel, found by
//! racing: each candidate times one epoch, the fastest is kept, the others are timed again with
//! exponential back-off. Switching costs a read-back, so it happens only at epoch boundaries.

use super::cost::DepthThresholds;

/// What runs an epoch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Regime {
  /// the host recursion (Algorithm 2 on the CPU)
  HostRecursion,
  /// the plain kernel over a dense grid
  PlainCells,
  /// the shared-tile kernel, `k` generations per dispatch
  SharedTile,
  /// the hit path with the miss path (or host learning)
  Memo,
}

impl Regime {
  pub fn name(&self) -> &'static str {
    match self {
      Regime::HostRecursion => "host recursion",
      Regime::PlainCells => "plain cells",
      Regime::SharedTile => "shared tile",
      Regime::Memo => "memo",
    }
  }
}

/// One candidate of the race.
#[derive(Clone, Debug)]
pub struct Candidate {
  pub regime: Regime,
  /// the last measured rate, generations per second
  pub rate: Option<f64>,
  /// the epoch at which it is timed again
  pub next: u64,
  pub backoff: u64,
}

/// Racing of executors (Algorithm 1, line 4).
#[derive(Clone, Debug)]
pub struct Racer {
  pub candidates: Vec<Candidate>,
  pub epoch: u64,
  pub current: Option<Regime>,
}

impl Racer {
  pub fn new(regimes: &[Regime]) -> Racer {
    Racer { candidates: regimes.iter().map(|&r| Candidate { regime: r, rate: None, next: 0, backoff: 2 }).collect(), epoch: 0, current: None }
  }
  /// The regime for the next epoch: an untimed or due candidate, else the fastest.
  pub fn choose(&mut self) -> Regime {
    let best = self.best();
    let due = self.candidates.iter().find(|c| c.rate.is_none() || (Some(c.regime) != best && c.next <= self.epoch)).map(|c| c.regime);
    let r = due.or(best).unwrap_or(self.candidates[0].regime);
    self.current = Some(r);
    r
  }
  /// The fastest candidate measured so far.
  pub fn best(&self) -> Option<Regime> {
    self.candidates.iter().filter_map(|c| c.rate.map(|r| (r, c.regime))).max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).map(|x| x.1)
  }
  /// Report the rate of an epoch run by `r`.
  pub fn report(&mut self, r: Regime, gens_per_s: f64) {
    let best_before = self.best();
    let epoch = self.epoch;
    if let Some(c) = self.candidates.iter_mut().find(|c| c.regime == r) {
      let was_timed = c.rate.is_some();
      c.rate = Some(gens_per_s);
      if was_timed && best_before != Some(r) {
        c.backoff = (c.backoff * 2).min(256);
      }
      c.next = epoch + c.backoff;
    }
    self.epoch += 1;
  }
  /// The rates of every candidate, for the inspector.
  pub fn rates(&self) -> Vec<(Regime, f64)> {
    self.candidates.iter().filter_map(|c| c.rate.map(|r| (c.regime, r))).collect()
  }
  /// Forget the measurements (the universe changed size or content class).
  pub fn reset(&mut self) {
    for c in &mut self.candidates {
      c.rate = None;
      c.next = 0;
      c.backoff = 2;
    }
  }
}

/// The universe-wide floor from the active area (Algorithm 1, lines 3 and 6): `None` for the
/// latency-bound regime, else the floor's window level.
pub fn floor_for(active_cells: f64, t: &DepthThresholds, rule_floor: usize) -> Option<usize> {
  if active_cells < t.a1 {
    None
  } else if active_cells < t.a2 {
    Some(rule_floor + 1)
  } else {
    Some(rule_floor)
  }
}

/// Detects a globally periodic universe from the canonical state at epoch boundaries: the root
/// node of the host recursion or a hash of the GPU ID grid.
#[derive(Clone, Debug, Default)]
pub struct CycleDetector {
  seen: rustc_hash::FxHashMap<u64, (u64, i64, i64)>,
  /// `(period, dx, dy)` once found
  pub found: Option<(u64, i64, i64)>,
  /// the generation at which the period was found
  pub at: u64,
}

impl CycleDetector {
  /// Record the state `key` (equal keys mean equal universes up to translation) at `generation`,
  /// with the universe's reference corner at `(x, y)`. Returns the period once found.
  pub fn record(&mut self, key: u64, generation: u64, x: i64, y: i64) -> Option<(u64, i64, i64)> {
    if self.found.is_some() {
      return self.found;
    }
    if let Some(&(g, x0, y0)) = self.seen.get(&key) {
      if generation > g {
        self.found = Some((generation - g, x - x0, y - y0));
        self.at = generation;
        return self.found;
      }
    }
    if self.seen.len() > 1 << 16 {
      self.seen.clear();
    }
    self.seen.insert(key, (generation, x, y));
    None
  }
  pub fn reset(&mut self) {
    self.seen.clear();
    self.found = None;
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn racing_keeps_the_fastest_and_backs_off() {
    let mut r = Racer::new(&[Regime::HostRecursion, Regime::SharedTile, Regime::PlainCells]);
    let rates = |g: Regime| match g {
      Regime::HostRecursion => 10.0,
      Regime::SharedTile => 30.0,
      _ => 20.0,
    };
    let mut runs = std::collections::HashMap::new();
    for _ in 0..200 {
      let g = r.choose();
      *runs.entry(g).or_insert(0) += 1;
      r.report(g, rates(g));
    }
    assert_eq!(r.best(), Some(Regime::SharedTile));
    assert!(runs[&Regime::SharedTile] > 180, "{runs:?}");
    assert!(runs[&Regime::HostRecursion] >= 2);
  }
  #[test]
  fn cycles() {
    let mut c = CycleDetector::default();
    assert!(c.record(7, 0, 0, 0).is_none());
    assert!(c.record(8, 16, 0, 0).is_none());
    assert_eq!(c.record(7, 32, 4, -4), Some((32, 4, -4)));
  }
}
