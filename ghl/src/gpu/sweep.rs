//! The hit path (Stage 2): the universe as a grid of tile IDs in VRAM, advanced by sweeps. One
//! dispatch per sweep gathers four IDs per window, probes the memo table and writes one ID; no
//! atomics but the miss counter. A missed window writes a sentinel, which every window containing
//! it inherits.
//!
//! Misses are learned in bursts: the grid is snapshotted, a burst of sweeps runs inside one compute
//! pass, and one read-back of the counters says whether every window hit. If one missed, the
//! snapshot is restored and the burst is resolved by the miss path: the host recursion (Stage 2)
//! or the GPU's own (Stage 3, [`super::miss`]).

use super::{
  context::GpuContext,
  memo::{self, Memo},
};
use crate::{
  cpu::{Engine, Flat},
  tile::NONE,
  Result,
};

/// The sentinel of a missed window.
pub const MISS: u32 = 0xFFFF_FFFE;

fn sweep_wgsl() -> String {
  format!(
    "struct P {{ sw: u32, sh: u32, dw: u32, dh: u32, off: i32, mask: u32, empty: u32, level: u32, bw: u32, bh: u32, pad0: u32, pad1: u32 }};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> nodes: array<u32>;
@group(0) @binding(4) var<storage, read_write> counters: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> misses: array<u32>;
@group(0) @binding(6) var<storage, read_write> act: array<atomic<u32>>;
@group(0) @binding(7) var<storage, read> tick: array<u32>;
{common}{find}
fn at(x: i32, y: i32) -> u32 {{
  if (x < 0 || y < 0 || x >= i32(p.sw) || y >= i32(p.sh)) {{ return p.empty; }}
  return src[u32(y) * p.sw + u32(x)];
}}
var<workgroup> on: u32;
var<workgroup> any: atomic<u32>;
// one workgroup per block of 16x16 tiles; a block runs only while it or a neighbour holds content
// (active blocks): flags hold the last sweep a block is active for
@compute @workgroup_size(16, 16) fn main(@builtin(global_invocation_id) id: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {{
  let s = tick[0];
  let blk = wid.y * p.bw + wid.x;
  if (li == 0u) {{
    on = atomicLoad(&act[blk]);
    atomicStore(&any, 0u);
  }}
  if (workgroupUniformLoad(&on) < s) {{ return; }}
  let inside = id.x < p.dw && id.y < p.dh;
  if (inside) {{
    let x = i32(id.x) + p.off;
    let y = i32(id.y) + p.off;
    let k = vec4<u32>(at(x, y), at(x + 1, y), at(x, y + 1), at(x + 1, y + 1));
    var val = 0xFFFFFFFEu;
    if (all(k != vec4<u32>(0xFFFFFFFEu))) {{
      let slot = find(k, p.level, p.mask);
      if (slot != 0xFFFFFFFFu) {{
        let r = nodes[slot * 8u + 4u];
        if (r != 0xFFFFFFFFu) {{ val = r; }}
      }}
    }}
    if (val == 0xFFFFFFFEu) {{
      let m = atomicAdd(&counters[0], 1u);
      if (m < arrayLength(&misses)) {{ misses[m] = id.y * p.dw + id.x; }}
    }}
    dst[id.y * p.dw + id.x] = val;
    if (val != p.empty) {{ atomicOr(&any, 1u); }}
  }}
  workgroupBarrier();
  // content here: this block and its neighbours run in the next two sweeps (the second cleans the
  // other phase's grid)
  if (li == 0u && atomicLoad(&any) != 0u) {{
    for (var dy = -1; dy <= 1; dy++) {{
      for (var dx = -1; dx <= 1; dx++) {{
        let bx = i32(wid.x) + dx;
        let by = i32(wid.y) + dy;
        if (bx >= 0 && by >= 0 && bx < i32(p.bw) && by < i32(p.bh)) {{ atomicMax(&act[u32(by) * p.bw + u32(bx)], s + 2u); }}
      }}
    }}
  }}
}}
",
    common = memo::common_wgsl(),
    find = memo::find_wgsl("nodes")
  )
}

/// Non-empty tiles in the outer two rings of phase A.
const RING_WGSL: &str = "struct P { sw: u32, sh: u32, dw: u32, dh: u32, off: i32, mask: u32, empty: u32, level: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> counters: array<atomic<u32>>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x + id.y * 65535u * 64u;
  if (i >= p.sw * p.sh) { return; }
  let x = i % p.sw;
  let y = i / p.sw;
  // a burst of at most eight sweeps moves content at most four tiles: six rings stay clear
  if (x >= 6u && y >= 6u && x + 6u < p.sw && y + 6u < p.sh) { return; }
  if (src[i] != p.empty) { atomicAdd(&counters[1], 1u); }
}
";

/// Zero the sweep's counters inside a compute pass.
const RESET_WGSL: &str = "@group(0) @binding(0) var<storage, read_write> c: array<u32>;
@compute @workgroup_size(1) fn main() { c[0] = 0u; c[1] = 0u; }
";

/// Advance the sweep counter that the active-block flags compare against.
const TICK_WGSL: &str = "@group(0) @binding(0) var<storage, read_write> t: array<u32>;
@compute @workgroup_size(1) fn main() { t[0] = t[0] + 1u; }
";

/// Blocks of 16x16 tiles: flags marking those with content in or next to them (active for the
/// first two sweeps, counted from 1).
pub fn block_flags(ids: &[u32], nx: usize, ny: usize, empty: u32) -> (Vec<u32>, u32, u32) {
  let (bw, bh) = ((nx + 1).div_ceil(16), (ny + 1).div_ceil(16));
  let mut act = vec![0u32; bw * bh];
  for j in 0..ny {
    for i in 0..nx {
      if ids[j * nx + i] != empty {
        let (bx, by) = ((i / 16) as i64, (j / 16) as i64);
        for dy in -1..=1i64 {
          for dx in -1..=1i64 {
            let (x, y) = (bx + dx, by + dy);
            if x >= 0 && y >= 0 && (x as usize) < bw && (y as usize) < bh {
              act[y as usize * bw + x as usize] = 2;
            }
          }
        }
      }
    }
  }
  (act, bw as u32, bh as u32)
}

/// The ID grid and its sweep kernels.
pub struct IdGrid {
  /// log2 of the sweep tile's side
  pub lt: u32,
  pub nx: u32,
  pub ny: u32,
  /// absolute cells of tile (0, 0) of phase A
  pub x0: i64,
  pub y0: i64,
  pub a: wgpu::Buffer,
  pub b: wgpu::Buffer,
  pub snap: wgpu::Buffer,
  pub counters: wgpu::Buffer,
  pub misses: wgpu::Buffer,
  pub miss_capacity: u64,
  pub in_b: bool,
  pub sweeps: u64,
  /// the empty tile's ID at the sweep level
  pub empty: u32,
  pipe: wgpu::ComputePipeline,
  ring: wgpu::ComputePipeline,
  bg_ab: wgpu::BindGroup,
  bg_ba: wgpu::BindGroup,
  bg_ring: wgpu::BindGroup,
  reset: wgpu::ComputePipeline,
  bg_reset: wgpu::BindGroup,
  u_ab: wgpu::Buffer,
  u_ba: wgpu::Buffer,
  /// active-block flags and the sweep counter they compare against, with their snapshots
  pub act: wgpu::Buffer,
  act_snap: wgpu::Buffer,
  pub tick: wgpu::Buffer,
  tick_snap: wgpu::Buffer,
  tick_pipe: wgpu::ComputePipeline,
  bg_tick: wgpu::BindGroup,
  pub bw: u32,
  pub bh: u32,
  /// windows and misses of every sweep run, as the inspector's learning curve
  pub history: Vec<(u64, u64)>,
}

/// What a burst did.
#[derive(Clone, Copy, Debug, Default)]
pub struct Burst {
  pub sweeps: u32,
  pub windows: u64,
  pub misses: u64,
  /// non-empty tiles in the outer rings after the burst
  pub ring: u64,
}

impl IdGrid {
  /// A grid of `nx * ny` IDs (phase A) at `(x0, y0)`.
  pub fn new(ctx: &GpuContext, memo: &Memo, lt: u32, nx: u32, ny: u32, x0: i64, y0: i64, ids: &[u32], empty: u32) -> Result<IdGrid> {
    let src = memo::shard_nodes(&sweep_wgsl(), memo.nodes.len() as u32, memo.shard_log2);
    let pipe = ctx.pipeline("sweep", &src, "main", true)?;
    let ring = ctx.pipeline("sweep ring", RING_WGSL, "main", true)?;
    let a = ctx.storage_init("ids A", ids);
    let b = ctx.storage("ids B", (nx as u64 + 1) * (ny as u64 + 1) * 4);
    let snap = ctx.storage("ids snapshot", nx as u64 * ny as u64 * 4);
    let counters = ctx.storage("sweep counters", 16);
    let miss_capacity = (nx as u64 * ny as u64).clamp(1024, 1 << 22);
    let misses = ctx.storage("sweep misses", miss_capacity * 4);
    let mask = (1u32 << memo.node_log2) - 1;
    let level = lt + 1;
    let (flags, bw, bh) = block_flags(ids, nx as usize, ny as usize, empty);
    let act = ctx.storage_init("active blocks", &flags);
    let act_snap = ctx.storage("active blocks snapshot", flags.len() as u64 * 4);
    let tick = ctx.storage_init("sweep tick", &[1, 0, 0, 0]);
    let tick_snap = ctx.storage("sweep tick snapshot", 16);
    let ab = ctx.uniform("sweep A->B", &[nx, ny, nx + 1, ny + 1, (-1i32) as u32, mask, empty, level, bw, bh, 0, 0]);
    let ba = ctx.uniform("sweep B->A", &[nx + 1, ny + 1, nx, ny, 0, mask, empty, level, bw, bh, 0, 0]);
    let mut eab: Vec<(u32, &wgpu::Buffer)> = vec![(0, &ab), (1, &a), (2, &b), (4, &counters), (5, &misses), (6, &act), (7, &tick)];
    eab.extend(memo.node_entries());
    let mut eba: Vec<(u32, &wgpu::Buffer)> = vec![(0, &ba), (1, &b), (2, &a), (4, &counters), (5, &misses), (6, &act), (7, &tick)];
    eba.extend(memo.node_entries());
    let bg_ab = ctx.bind_at(&pipe, &eab);
    let bg_ba = ctx.bind_at(&pipe, &eba);
    let bg_ring = ctx.bind(&ring, &[&ab, &a, &counters]);
    let reset = ctx.pipeline("sweep reset", RESET_WGSL, "main", true)?;
    let bg_reset = ctx.bind(&reset, &[&counters]);
    let tick_pipe = ctx.pipeline("sweep tick", TICK_WGSL, "main", true)?;
    let bg_tick = ctx.bind(&tick_pipe, &[&tick]);
    Ok(IdGrid {
      lt,
      nx,
      ny,
      x0,
      y0,
      a,
      b,
      snap,
      counters,
      misses,
      miss_capacity,
      in_b: false,
      sweeps: 0,
      empty,
      pipe,
      ring,
      bg_ab,
      bg_ba,
      bg_ring,
      reset,
      bg_reset,
      u_ab: ab,
      u_ba: ba,
      act,
      act_snap,
      tick,
      tick_snap,
      tick_pipe,
      bg_tick,
      bw,
      bh,
      history: Vec::new(),
    })
  }

  /// Snapshot phase A with the active-block flags (at the start of a burst).
  pub fn encode_snapshot(&self, enc: &mut wgpu::CommandEncoder) {
    enc.copy_buffer_to_buffer(&self.a, 0, &self.snap, 0, self.nx as u64 * self.ny as u64 * 4);
    enc.copy_buffer_to_buffer(&self.act, 0, &self.act_snap, 0, self.bw as u64 * self.bh as u64 * 4);
    enc.copy_buffer_to_buffer(&self.tick, 0, &self.tick_snap, 0, 16);
  }

  /// Advance the sweep counter (inside a pass, after a sweep).
  pub fn encode_tick(&self, pass: &mut wgpu::ComputePass<'_>) {
    pass.set_pipeline(&self.tick_pipe);
    pass.set_bind_group(0, &self.bg_tick, &[]);
    pass.dispatch_workgroups(1, 1, 1);
  }

  /// New flags for a grid written from the host.
  pub fn reset_flags(&self, ctx: &GpuContext, ids: &[u32]) {
    let (flags, _, _) = block_flags(ids, self.nx as usize, self.ny as usize, self.empty);
    ctx.write(&self.act, 0, &flags);
    ctx.write(&self.tick, 0, &[1]);
    ctx.queue.submit(None);
  }

  /// Zero the sweep's counters (inside a pass).
  pub fn encode_reset(&self, pass: &mut wgpu::ComputePass<'_>) {
    pass.set_pipeline(&self.reset);
    pass.set_bind_group(0, &self.bg_reset, &[]);
    pass.dispatch_workgroups(1, 1, 1);
  }

  /// One sweep, A to B (`ab`) or B to A (inside a pass).
  pub fn encode_sweep(&self, pass: &mut wgpu::ComputePass<'_>, ab: bool) {
    pass.set_pipeline(&self.pipe);
    if ab {
      pass.set_bind_group(0, &self.bg_ab, &[]);
      pass.dispatch_workgroups((self.nx + 1).div_ceil(16), (self.ny + 1).div_ceil(16), 1);
    } else {
      pass.set_bind_group(0, &self.bg_ba, &[]);
      pass.dispatch_workgroups(self.nx.div_ceil(16), self.ny.div_ceil(16), 1);
    }
  }

  /// The source grid of a sweep and its parameters.
  pub fn source(&self, ab: bool) -> (&wgpu::Buffer, &wgpu::Buffer) {
    if ab {
      (&self.a, &self.u_ab)
    } else {
      (&self.b, &self.u_ba)
    }
  }

  pub fn bytes(&self) -> u64 {
    (2 * self.nx as u64 * self.ny as u64 + (self.nx as u64 + 1) * (self.ny as u64 + 1) + self.miss_capacity + 4) * 4
  }

  /// Generations of one sweep.
  pub fn gens_per_sweep(&self) -> u64 {
    1u64 << (self.lt - 1)
  }

  /// Run `n` sweeps (even, from phase A) from a snapshot in one compute pass; one read-back.
  pub fn burst(&mut self, ctx: &GpuContext, n: u32) -> Burst {
    debug_assert!(!self.in_b && n % 2 == 0);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    self.encode_snapshot(&mut enc);
    enc.clear_buffer(&self.counters, 0, None);
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      for i in 0..n {
        self.encode_sweep(&mut pass, i % 2 == 0);
        self.encode_tick(&mut pass);
      }
      pass.set_pipeline(&self.ring);
      pass.set_bind_group(0, &self.bg_ring, &[]);
      let (gx, gy) = memo::grid64(self.nx as u64 * self.ny as u64);
      pass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit(Some(enc.finish()));
    let c = ctx.read_u32(&self.counters, 0, 2);
    let per = ((self.nx as u64 + 1) * (self.ny as u64 + 1) + self.nx as u64 * self.ny as u64) / 2;
    let b = Burst { sweeps: n, windows: per * n as u64, misses: c[0] as u64, ring: c[1] as u64 };
    if b.misses == 0 {
      self.sweeps += n as u64;
      for i in 0..n {
        let w = if i % 2 == 0 { (self.nx as u64 + 1) * (self.ny as u64 + 1) } else { self.nx as u64 * self.ny as u64 };
        self.history.push((w, 0));
      }
    }
    b
  }

  /// Non-empty tiles in the outer two rings of phase A (one dispatch and a read-back).
  pub fn ring_check(&self, ctx: &GpuContext) -> u64 {
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.clear_buffer(&self.counters, 0, None);
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&self.ring);
      pass.set_bind_group(0, &self.bg_ring, &[]);
      let (gx, gy) = memo::grid64(self.nx as u64 * self.ny as u64);
      pass.dispatch_workgroups(gx, gy, 1);
    }
    ctx.queue.submit(Some(enc.finish()));
    ctx.read_u32(&self.counters, 1, 1)[0] as u64
  }

  /// Restore phase A from the snapshot (after a burst that missed).
  pub fn restore(&self, ctx: &GpuContext) {
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(&self.snap, 0, &self.a, 0, self.nx as u64 * self.ny as u64 * 4);
    enc.copy_buffer_to_buffer(&self.act_snap, 0, &self.act, 0, self.bw as u64 * self.bh as u64 * 4);
    enc.copy_buffer_to_buffer(&self.tick_snap, 0, &self.tick, 0, 16);
    ctx.queue.submit(Some(enc.finish()));
  }

  pub fn read_a(&self, ctx: &GpuContext) -> Vec<u32> {
    ctx.read_u32(&self.a, 0, self.nx as u64 * self.ny as u64)
  }
  pub fn write_a(&self, ctx: &GpuContext, ids: &[u32]) {
    ctx.write(&self.a, 0, ids);
    ctx.queue.submit(None);
  }

  /// The burst on the host (Stage 2's learning): the snapshot's IDs to host IDs, `n` sweeps of the
  /// host's positional sweep, the windows it resolved uploaded with their results, and the final
  /// grid written back. Returns per sweep `(windows, distinct windows, misses)`.
  pub fn learn_on_host(&mut self, ctx: &GpuContext, memo: &mut Memo, e: &mut Engine, n: u32) -> Result<Vec<(u64, u64, u64)>> {
    let lt = self.lt as usize;
    let gpu_ids = self.read_a(ctx);
    let host_ids = memo.download(ctx, e, lt, &gpu_ids);
    let mut f = Flat::from_ids_rect(e, &host_ids, self.nx as usize, self.ny as usize, lt, 0);
    let mut resolved: Vec<u32> = Vec::new();
    let mut stats = Vec::with_capacity(n as usize);
    for _ in 0..n {
      let before = e.levels.get(lt + 1).map(|l| l.stats.misses).unwrap_or(0);
      let keys = f.keys(e);
      let mut q = keys.clone();
      e.solve(lt + 1, &mut q);
      let out: Vec<u32> = keys.iter().map(|&k| e.result(lt + 1, k)).collect();
      stats.push((keys.len() as u64, q.len() as u64, e.levels[lt + 1].stats.misses - before));
      resolved.extend_from_slice(&q);
      if !f.in_b {
        f.b_ids = out;
      } else {
        f.a_ids = out;
      }
      f.in_b = !f.in_b;
      f.sweeps += 1;
    }
    resolved.sort_unstable();
    resolved.dedup();
    memo.upload_results(ctx, e, lt + 1, &resolved)?;
    let g = memo.upload(ctx, e, lt, &f.a_ids)?;
    self.write_a(ctx, &g);
    self.reset_flags(ctx, &g);
    self.sweeps += n as u64;
    for s in &stats {
      self.history.push((s.0, s.2));
    }
    Ok(stats)
  }
}

/// Host IDs of the level-`lt` tiles covering `[x0, x0 + nx 2^lt) x [y0, ...)` of the host's root
/// (empty outside it).
pub fn host_ids(e: &mut Engine, root: (u32, usize, i64, i64), lt: usize, x0: i64, y0: i64, nx: usize, ny: usize) -> Vec<u32> {
  let z = e.empty(lt);
  let (rid, rl, rx, ry) = root;
  let s = 1i64 << lt;
  let side = 1i64 << rl;
  let mut out = vec![z; nx * ny];
  for j in 0..ny {
    for i in 0..nx {
      let (x, y) = (x0 + i as i64 * s, y0 + j as i64 * s);
      let (dx, dy) = (x - rx, y - ry);
      if dx < 0 || dy < 0 || dx >= side || dy >= side {
        continue;
      }
      let (mut id, mut l) = (rid, rl);
      let (mut px, mut py) = (dx as u64, dy as u64);
      while l > lt {
        if e.is_empty(l, id) {
          id = NONE;
          break;
        }
        let h = 1u64 << (l - 1);
        let q = (px >= h) as usize | ((py >= h) as usize) << 1;
        id = e.ch(l, id)[q];
        px &= h - 1;
        py &= h - 1;
        l -= 1;
      }
      out[j * nx + i] = if id == NONE { z } else { id };
    }
  }
  out
}

/// A host quadtree from a positional grid of host IDs at level `lt` with its corner at `(x0, y0)`:
/// `(root, level)` with the grid in the root's top-left corner.
pub fn build_from_ids(e: &mut Engine, ids: &[u32], nx: usize, ny: usize, lt: usize) -> (u32, usize) {
  let mut level = lt + 1;
  while (1usize << (level - lt)) < nx.max(ny) {
    level += 1;
  }
  fn rec(e: &mut Engine, ids: &[u32], nx: usize, ny: usize, lt: usize, l: usize, i0: usize, j0: usize) -> u32 {
    if i0 >= nx || j0 >= ny {
      return e.empty(l);
    }
    if l == lt {
      return ids[j0 * nx + i0];
    }
    let h = 1usize << (l - 1 - lt);
    let c = [rec(e, ids, nx, ny, lt, l - 1, i0, j0), rec(e, ids, nx, ny, lt, l - 1, i0 + h, j0), rec(e, ids, nx, ny, lt, l - 1, i0, j0 + h), rec(e, ids, nx, ny, lt, l - 1, i0 + h, j0 + h)];
    e.intern(l, c)
  }
  let id = rec(e, ids, nx, ny, lt, level, 0, 0);
  (id, level)
}
