//! The prices the scheduler works with: rates measured at start-up (`gpu::calibrate`) or the
//! paper's, the cost of one operation derived from them, and the formulas of the paper that turn
//! them into choices: the exchange rate `rho` (2) and the floor it implies (Table II), the price
//! `q` of a stepped region (7), and the active-area thresholds of the depth per universe (6).

/// The class of brute-force kernel a rule runs on; it sets `R_B`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelClass {
  /// one bit per cell, the adder tree of a Life-like rule
  LifeLike,
  /// a synthesised circuit on bit-planes (Hutton32b)
  Circuit,
  /// the rule's table, one byte per cell (Hutton32 today)
  ByteLut,
  /// the rule tree, for tables that do not fit
  Tree,
}

/// Rates of one machine, in operations per second and seconds.
#[derive(Clone, Debug, PartialEq)]
pub struct Rates {
  /// brute force, cell-updates/s
  pub r_b: f64,
  /// probes/s into a table larger than the L2 cache
  pub r_p: f64,
  /// concurrent find-or-inserts/s
  pub r_i: f64,
  /// one dispatch chained inside a compute pass
  pub dispatch_s: f64,
  /// a new compute pass
  pub pass_s: f64,
  /// a separate submit
  pub submit_s: f64,
  /// a dispatch and a 16-byte read-back
  pub readback_s: f64,
  /// a miss-path wave of 16 fixed-capacity dispatches
  pub wave_s: f64,
  /// the host's brute force (`cpu::leaf`), cell-updates/s on one thread
  pub host_r_b: f64,
  /// whether these were measured on this machine
  pub measured: bool,
}

impl Rates {
  /// The paper's constants (RTX 5060 Ti, Table I) for a kernel class.
  pub fn paper(class: KernelClass) -> Rates {
    Rates {
      r_b: match class {
        KernelClass::LifeLike => 1.42e12,
        KernelClass::Circuit => 3.0e11,
        KernelClass::ByteLut => 1.85e11,
        KernelClass::Tree => 2.0e10,
      },
      r_p: 3.7e9,
      r_i: 1.15e9,
      dispatch_s: 3.4e-6,
      pass_s: 49e-6,
      submit_s: 150e-6,
      readback_s: 0.1e-3,
      wave_s: 52e-6,
      host_r_b: match class {
        KernelClass::LifeLike => 4.0e9,
        KernelClass::Circuit => 1.0e9,
        KernelClass::ByteLut => 3.5e8,
        KernelClass::Tree => 5.0e7,
      },
      measured: false,
    }
  }
  /// `rho = R_B / R_P` (paper equation (2)): the cell-updates that cost as much as one probe.
  pub fn rho(&self) -> f64 {
    self.r_b / self.r_p
  }
  /// The rule's base tile from its `rho`: the smallest tile of side `t` whose step from a `2t`
  /// window is worth more than a probe, `t^3 / 2 > rho` (Table II). As log2 of the side, 3 to 6.
  pub fn base_tile_log2(&self) -> u32 {
    base_tile_for_rho(self.rho())
  }
}

/// The smallest tile side `2^a` (a in 2..=6) with `V = t^2 * t/2 > rho`.
pub fn base_tile_for_rho(rho: f64) -> u32 {
  for a in 2..=6u32 {
    let t = (1u64 << a) as f64;
    if t * t * t / 2.0 > rho {
      return a;
    }
  }
  6
}

/// One row of the paper's Table II for a rate: a tile, the window that yields it, what a hit
/// saves (`V`), `V / rho`, and whether the level is memoized.
#[derive(Clone, Debug, PartialEq)]
pub struct FloorRow {
  pub tile: u32,
  pub window: u32,
  pub generations: u32,
  pub v: f64,
  pub v_over_rho: f64,
  pub memoized: bool,
}

/// Table II for `rho`, tiles 4×4 to 64×64.
pub fn floor_table(rho: f64, base_tile_log2: u32) -> Vec<FloorRow> {
  (2..=6u32)
    .map(|a| {
      let t = 1u32 << a;
      let v = (t as f64).powi(3) / 2.0;
      FloorRow { tile: t, window: 2 * t, generations: t / 2, v, v_over_rho: v / rho, memoized: a >= base_tile_log2 }
    })
    .collect()
}

/// Nanoseconds per operation: the harness's GPU cost model (`cmd_adapt.rs`, `Model`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Model {
  /// read-only probe of a table larger than the L2 cache
  pub probe: f64,
  /// find-or-insert that claims a slot (claim protocol and commit)
  pub insert: f64,
  /// one cell evaluation by the brute-force kernel
  pub cell: f64,
  /// gather of one base tile when a node is expanded into cells
  pub gather: f64,
}

impl Default for Model {
  /// The harness's defaults: the probe rate of Table I, inserts at 1 ns, cells at the byte-LUT
  /// kernel's rate.
  fn default() -> Self {
    Model { probe: 1e9 / 3.7e9, insert: 1.0, cell: 1e9 / 1.86e11, gather: 1e9 / 3.7e9 }
  }
}

impl Model {
  pub fn from_rates(r: &Rates) -> Model {
    Model { probe: 1e9 / r.r_p, insert: 1e9 / r.r_i, cell: 1e9 / r.r_b, gather: 1e9 / r.r_p }
  }
}

/// The price of a region stepped as plain cells, as a multiple of its cell-updates (paper
/// equation (7)): `q = 1 + eta + gamma + epsilon`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RegionPrice {
  pub q: f64,
  /// halo: zero inside a block of stepped regions, at its bound for a region stepped alone
  pub eta: f64,
  /// conversion of the region to tile IDs once per step
  pub gamma: f64,
  /// memo at and above the region
  pub epsilon: f64,
}

/// Probes per base tile to convert a stepped region to IDs (the paper's `c_v`).
pub const C_V: f64 = 4.8;

/// `q` of a region of side `s` stepped for `t` generations on base tiles of side `b`; `c_m` is the
/// memo's probes per stepped region (13 to 68 measured); `alone`: the region is stepped without
/// stepped neighbours, so it pays the whole halo.
pub fn region_price(s: f64, t: f64, b: f64, rho: f64, c_m: f64, alone: bool) -> RegionPrice {
  let eta = if alone { 2.0 * (t - 1.0) / s * (1.0 + (2.0 * t - 1.0) / (3.0 * s)) } else { 0.0 };
  let gamma = C_V * rho / (b * b * t);
  let epsilon = c_m * rho / (s * s * t);
  RegionPrice { q: 1.0 + eta + gamma + epsilon, eta, gamma, epsilon }
}

/// The universe-wide floor from the active area (paper equation (6), Table XXXI): plain cells
/// below `a1` active cells, one level coarser than the rule's floor up to `a2`, the rule's floor
/// above.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DepthThresholds {
  pub a1: f64,
  pub a2: f64,
}

/// Dispatches per generation of each engine (Table XXXI): plain cells, the floor one level coarser
/// than the rule's, the rule's floor.
pub const DISPATCHES_PER_GEN: [f64; 3] = [1.1, 2.5, 4.0];

/// `a1` and `a2` from the dispatch time and the work per region and generation of each engine
/// (`w` in seconds: plain cells, coarser floor, rule's floor) for regions of `region_cells` cells.
pub fn depth_thresholds(dispatch_s: f64, w: [f64; 3], region_cells: f64) -> DepthThresholds {
  let l: Vec<f64> = DISPATCHES_PER_GEN.iter().map(|d| d * dispatch_s).collect();
  let a1 = if w[0] > w[1] { (l[1] - l[0]) / (w[0] - w[1]) * region_cells } else { f64::INFINITY };
  let a2 = if w[1] > w[2] { (l[2] - l[1]) / (w[1] - w[2]) * region_cells } else { f64::INFINITY };
  DepthThresholds { a1, a2: a2.max(a1) }
}

/// The paper's thresholds for Hutton32b with the counter's work per 256×256 region and generation
/// at the bit-plane rate (377, 59 and 33 ns): about 1.0 M and 1.3 · 10^7 active cells.
pub fn paper_thresholds(dispatch_s: f64) -> DepthThresholds {
  depth_thresholds(dispatch_s, [377e-9, 59e-9, 33e-9], 65536.0)
}

/// The hit rate above which a sweep level pays for itself (paper equation (5)):
/// `h* = (p + i) / (8p + i)` with `p` and `i` the prices of a probe and an insert.
pub fn sweep_hit_threshold(probe: f64, insert: f64) -> f64 {
  (probe + insert) / (8.0 * probe + insert)
}

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn floors_follow_rho() {
    assert_eq!(Rates::paper(KernelClass::Circuit).base_tile_log2(), 3);
    assert_eq!(Rates::paper(KernelClass::ByteLut).base_tile_log2(), 3);
    assert_eq!(Rates::paper(KernelClass::LifeLike).base_tile_log2(), 4);
    let t = paper_thresholds(3.4e-6);
    assert!((t.a1 / 1.0e6 - 1.0).abs() < 0.1, "{t:?}");
    assert!((t.a2 / 1.3e7 - 1.0).abs() < 0.1, "{t:?}");
    // a step of one base tile's side adds 47 % of conversion for Hutton32b at the byte-LUT rho
    let p = region_price(32.0, 8.0, 8.0, 50.0, 0.0, false);
    assert!((p.gamma - 0.47).abs() < 0.01, "{p:?}");
    assert!((sweep_hit_threshold(1.0, 4.0) - 5.0 / 12.0).abs() < 1e-12);
  }
}
