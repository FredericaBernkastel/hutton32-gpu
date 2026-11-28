//! The scheduler: which engine runs each epoch and with which parameters (paper Algorithm 1),
//! the engine per region (Algorithm 3 and its price-based replacement), the step and the region
//! (equations (7)–(9)), the sweep tile (5), the depth per universe (6), racing, cycle detection.
//! Costs come from live counters and the calibrated rates ([`cost`]).

pub mod controller;
pub mod cost;
pub mod epoch;
pub mod region;
pub mod sweep_tile;

pub use controller::{Kind as PolicyKind, Mode, Policy, PriceRow, RegionReport, RegionState};
pub use cost::{DepthThresholds, FloorRow, KernelClass, Model, Rates, RegionPrice};
pub use epoch::{CycleDetector, Racer, Regime};
pub use region::StepRule;
pub use sweep_tile::SweepTile;

/// Which controller chooses the engine per region.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Controller {
  /// every region at the universe's floor. The default: neither controller passes at every
  /// configuration of the paper's Tables XXVII and XXVIII (`tests/controller.rs`); and a controller
  /// runs in the host recursion, whose steps it holds to the step rule's (at most eight base tiles'
  /// sides, where HashLife alone lengthens its step without bound), pricing plain cells at the GPU's
  /// rate while this build steps them on the host
  #[default]
  Uniform,
  /// Algorithm 3 as built and measured (16-generation steps)
  Algorithm3,
  /// the price-based controller, a rent rule on the floor's price against plain cells'; it passes
  /// its acceptance test at the roadmap's three configurations, not at every configuration of the
  /// paper's Tables XXVII and XXVIII
  Priced,
  /// plain cells in every region, stepped as blocks under a region-level memo (the region model's
  /// other end)
  PlainCells,
}

/// The scheduler's state for one universe.
#[derive(Clone, Debug)]
pub struct Scheduler {
  pub rates: Rates,
  pub model: Model,
  pub class: KernelClass,
  pub thresholds: DepthThresholds,
  pub racer: Racer,
  pub step: StepRule,
  pub sweep: SweepTile,
  pub cycles: CycleDetector,
  /// the host recursion's step, log2 of generations, adapted to the time an epoch takes
  pub host_step_log2: u32,
  pub controller: Controller,
}

impl Scheduler {
  pub fn new(rates: Rates, class: KernelClass, base_tile_log2: u32, racing: &[Regime], controller: Controller) -> Scheduler {
    let model = Model::from_rates(&rates);
    // equation (6): the dispatch time per generation of each engine against its work per region and
    // generation; the work of plain cells is the counter's at the bit-plane rate scaled to this
    // rule's brute-force rate, the memo engines' (probe-bound) the counter's
    let thresholds = cost::depth_thresholds(rates.dispatch_s, [377e-9 * 3.0e11 / rates.r_b, 59e-9, 33e-9], 65536.0);
    Scheduler {
      sweep: SweepTile::new(4.max(base_tile_log2), base_tile_log2, 8.max(base_tile_log2), model.probe, model.insert),
      rates,
      model,
      class,
      thresholds,
      racer: Racer::new(racing),
      step: StepRule::new(base_tile_log2),
      cycles: CycleDetector::default(),
      host_step_log2: 10,
      controller,
    }
  }

  /// Adapt the host recursion's step to the time its epochs take: double it while an epoch takes
  /// less than `target / 8`, halve it when it takes more than `target`.
  pub fn adapt_host_step(&mut self, epoch_s: f64, target_s: f64, min_log2: u32) {
    if epoch_s < target_s / 8.0 && self.host_step_log2 < 30 {
      self.host_step_log2 += 1;
    } else if epoch_s > target_s && self.host_step_log2 > min_log2 {
      self.host_step_log2 -= 1;
    }
  }
}
