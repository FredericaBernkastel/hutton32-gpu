//! Start-up micro-benchmarks for the scheduler's constants (roadmap Section 5.2): chained-dispatch
//! time, compute-pass and submit costs, read-back latency, the brute-force rate of the rule's
//! kernels, the probe rate on a table larger than the L2 cache and the insert rate of the claim
//! protocol. Every threshold of the paper comes from one RTX 5060 Ti; these runs replace them with
//! the machine's own. Short runs: about half a second in all.

use std::time::Instant;

use super::{
  brute::{DenseGrid, DenseKernels},
  context::GpuContext,
  memo,
};
use crate::{io::Grid, sched::Rates, Result};

/// What calibration measured.
#[derive(Clone, Debug)]
pub struct Calibrated {
  pub rates: Rates,
  pub kernel: String,
  /// R_B per kernel, cell-updates/s
  pub r_b: Vec<(String, f64)>,
  /// R_P by table size: `(bytes, probes/s)`
  pub r_p: Vec<(u64, f64)>,
}

fn median(mut v: Vec<f64>) -> f64 {
  v.sort_by(|a, b| a.partial_cmp(b).unwrap());
  v[v.len() / 2]
}

/// Latencies: a chained dispatch, a compute pass, a submit, a dispatch with a 16-byte read-back.
pub fn latencies(ctx: &GpuContext) -> Result<(f64, f64, f64, f64)> {
  let nop = ctx.pipeline("nop", super::kernels::NOP, "main", true)?;
  let buf = ctx.storage("nop", 64);
  let bg = ctx.bind(&nop, &[&buf]);
  let run = |passes: usize, per_pass: usize, submits: usize| {
    ctx.timed(|| {
      for _ in 0..submits {
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        for _ in 0..passes {
          let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&nop);
          pass.set_bind_group(0, &bg, &[]);
          for _ in 0..per_pass {
            pass.dispatch_workgroups(1, 1, 1);
          }
        }
        ctx.queue.submit(Some(enc.finish()));
      }
    })
  };
  run(1, 64, 1);
  let dispatch = median((0..3).map(|_| run(1, 2000, 1) / 2000.0).collect());
  let pass = median((0..3).map(|_| run(200, 1, 1) / 200.0).collect()) - dispatch;
  let submit = median((0..3).map(|_| run(1, 1, 40) / 40.0).collect()) - dispatch;
  let readback = median(
    (0..9)
      .map(|_| {
        let t = Instant::now();
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        {
          let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&nop);
          pass.set_bind_group(0, &bg, &[]);
          pass.dispatch_workgroups(1, 1, 1);
        }
        ctx.queue.submit(Some(enc.finish()));
        let _ = ctx.read_u32(&buf, 0, 4);
        t.elapsed().as_secs_f64()
      })
      .collect(),
  );
  Ok((dispatch.max(1e-7), pass.max(0.0), submit.max(0.0), readback))
}

/// The brute-force rate of the plain and shared-tile kernels, cell-updates/s.
pub fn brute_rates(ctx: &GpuContext, k: &DenseKernels, states: u32) -> Result<(f64, f64)> {
  let mut seed = 0x9E37_79B9u32;
  let mut rnd = |side: usize| {
    let mut g = Grid::new(side, side);
    for c in g.cells.iter_mut() {
      seed ^= seed << 13;
      seed ^= seed >> 17;
      seed ^= seed << 5;
      *c = (seed % states.max(1)) as u8;
    }
    g
  };
  let big = rnd(2048);
  let mut g = DenseGrid::new(ctx, k, 2048, 2048, 0, 0, Some(&big))?;
  g.step(ctx, k, 4, 0, false);
  let gens = 64;
  let t = ctx.timed(|| g.step(ctx, k, gens, 0, false));
  let plain = 2048.0 * 2048.0 * gens as f64 / t;
  let small = rnd(256);
  let mut s = DenseGrid::new(ctx, k, 256, 256, 0, 0, Some(&small))?;
  let gens = 16 * k.tile_shape.2 as u64;
  s.step(ctx, k, gens, 0, true);
  let t = ctx.timed(|| s.step(ctx, k, gens, 0, true));
  let tile = 256.0 * 256.0 * gens as f64 / t;
  Ok((plain, tile))
}

/// Calibrate everything for the rule of `k`.
pub fn calibrate(ctx: &GpuContext, k: &DenseKernels, states: u32, mut base: Rates) -> Result<Calibrated> {
  let (dispatch, pass, submit, readback) = latencies(ctx)?;
  let (plain, tile) = brute_rates(ctx, k, states)?;
  let (r_p, r_i, sizes) = memo::calibrate_tables(ctx)?;
  base.r_b = plain;
  base.r_p = r_p;
  base.r_i = r_i;
  base.dispatch_s = dispatch;
  base.pass_s = pass;
  base.submit_s = submit;
  base.readback_s = readback;
  base.wave_s = 16.0 * dispatch;
  base.measured = true;
  Ok(Calibrated { rates: base, kernel: k.name(), r_b: vec![(format!("{} plain", k.name()), plain), (format!("{} shared tile", k.name()), tile)], r_p: sizes })
}
