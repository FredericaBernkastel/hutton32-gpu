//! The region grid and the step (paper Section VI-G, equations (7)–(9)). A region is four steps
//! wide, the smallest a step admits, because a window yields its centre a quarter of its side
//! later. The step is one base tile's side while periodic content is new (8 generations and 32×32
//! regions for Hutton32b, 16 and 64×64 for Life) and two to eight base tiles' sides where the memo
//! engine carries the universe. The step is universe-wide and changes at epoch boundaries.
//!
//! The rule for changing it is not in the paper: start at one base tile's side, lengthen while no
//! region is in plain cells and the hashing above the floor dominates the epoch, shorten when plain
//! cells are entered.

use super::cost::{region_price, RegionPrice};

/// The step and the region it implies.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepRule {
  /// log2 of the base tile's side: the shortest step
  pub base: u32,
  /// log2 of the step in generations
  pub step_log2: u32,
  /// the longest step: eight base tiles' sides
  pub max_log2: u32,
  /// epochs since the last change, so that a change is not undone at once
  pub since_change: u32,
}

impl StepRule {
  pub fn new(base_tile_log2: u32) -> StepRule {
    StepRule { base: base_tile_log2, step_log2: base_tile_log2, max_log2: base_tile_log2 + 3, since_change: 0 }
  }
  pub fn step(&self) -> u64 {
    1u64 << self.step_log2
  }
  /// log2 of the region's side: four steps.
  pub fn region_log2(&self) -> u32 {
    self.step_log2 + 2
  }
  /// Update at an epoch boundary. `cells_regions`: regions in plain cells now; `entered_cells`:
  /// a region entered plain cells in this epoch; `above_floor_share`: the share of the epoch's
  /// hashing spent above the floor. Returns whether the step changed.
  pub fn update(&mut self, cells_regions: usize, entered_cells: bool, above_floor_share: f64) -> bool {
    self.since_change += 1;
    if entered_cells && self.step_log2 > self.base {
      self.step_log2 -= 1;
      self.since_change = 0;
      return true;
    }
    if cells_regions == 0 && above_floor_share > 0.5 && self.step_log2 < self.max_log2 && self.since_change >= 4 {
      self.step_log2 += 1;
      self.since_change = 0;
      return true;
    }
    false
  }
}

/// The price of one region stepped as plain cells for one step (equation (7)), in seconds.
pub fn plain_region_seconds(region_log2: u32, step_log2: u32, base_log2: u32, rho: f64, r_b: f64, c_m: f64, alone: bool) -> (f64, RegionPrice) {
  let s = (1u64 << region_log2) as f64;
  let t = (1u64 << step_log2) as f64;
  let b = (1u64 << base_log2) as f64;
  let p = region_price(s, t, b, rho, c_m, alone);
  (p.q * s * s * t / r_b, p)
}

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn step_rule() {
    let mut r = StepRule::new(3);
    assert_eq!((r.step(), r.region_log2()), (8, 5));
    for _ in 0..4 {
      r.update(0, false, 0.9);
    }
    assert_eq!(r.step(), 16);
    assert!(r.update(1, true, 0.9));
    assert_eq!(r.step(), 8);
  }
}
