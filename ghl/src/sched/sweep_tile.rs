//! Raising and lowering the sweep tile (paper Section IV-F, equation (5)). A level that never hits
//! adds one failed probe and one insert per eight tile-steps below it; it pays for itself once its
//! hit rate exceeds `h* = (p + i) / (8p + i)`, about 40 %. GHL starts at 16×16 tiles and raises the
//! tile while the hit rate stays above that; every level lives in the same tables, so a change of
//! level is a parallel re-key of the ID grid.

use super::cost::sweep_hit_threshold;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SweepTile {
  /// log2 of the sweep tile's side
  pub lt: u32,
  pub min_lt: u32,
  pub max_lt: u32,
  /// the hit rate above which a level pays
  pub threshold: f64,
  /// smoothed hit rate of the current level and of the level above (the windows of the next sweep
  /// level are the ones probed at `lt + 1` now)
  hit: f64,
  sweeps: u32,
}

impl SweepTile {
  pub fn new(start_lt: u32, min_lt: u32, max_lt: u32, probe: f64, insert: f64) -> SweepTile {
    SweepTile { lt: start_lt.clamp(min_lt, max_lt), min_lt, max_lt, threshold: sweep_hit_threshold(probe, insert), hit: 0.0, sweeps: 0 }
  }
  /// Report one sweep's windows and misses; returns the new level if it changes.
  pub fn report(&mut self, windows: u64, misses: u64) -> Option<u32> {
    if windows == 0 {
      return None;
    }
    let h = 1.0 - misses as f64 / windows as f64;
    self.hit = if self.sweeps == 0 { h } else { 0.8 * self.hit + 0.2 * h };
    self.sweeps += 1;
    if self.sweeps < 8 {
      return None;
    }
    if self.hit > self.threshold && self.lt < self.max_lt {
      // the level pays: try one coarser; it has to learn again
      self.lt += 1;
      self.sweeps = 0;
      return Some(self.lt);
    }
    if self.hit < 0.5 * self.threshold && self.lt > self.min_lt {
      self.lt -= 1;
      self.sweeps = 0;
      return Some(self.lt);
    }
    None
  }
  pub fn hit_rate(&self) -> f64 {
    self.hit
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn raises_while_it_pays() {
    let mut s = SweepTile::new(4, 4, 8, 1.0, 4.0);
    let mut changed = None;
    for _ in 0..8 {
      changed = s.report(100, 10).or(changed);
    }
    assert_eq!(changed, Some(5));
    let mut changed = None;
    for _ in 0..8 {
      changed = s.report(100, 95).or(changed);
    }
    assert_eq!(changed, Some(4));
  }
}
