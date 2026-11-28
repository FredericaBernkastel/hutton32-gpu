//! Plain cells on the GPU (Stage 1): a dense grid of bit-planes stepped without memoization, by
//! the plain kernel (one invocation per 32 cells, all generations of an epoch in one compute pass)
//! or the shared-tile kernel (`k` generations per dispatch in workgroup memory, for small grids).
//! The grid grows with a margin when live cells come near its ring.

use super::{
  context::GpuContext,
  kernels::{self, RuleSource},
};
use crate::{io::Grid, query::Rect, rule::Rule, Result};

/// The compiled kernels of a rule.
pub struct DenseKernels {
  pub src: RuleSource,
  pub plain: wgpu::ComputePipeline,
  pub tile: wgpu::ComputePipeline,
  /// the shared tile: groups, rows, generations per dispatch
  pub tile_shape: (u32, u32, u32),
  pub census: wgpu::ComputePipeline,
  pub blit: wgpu::ComputePipeline,
  pub extra: Option<wgpu::Buffer>,
}

/// The shared tile's shape for `b` bit-planes: `k = 16` generations, within 256 invocations and
/// 16 KiB of workgroup memory.
pub fn tile_shape(b: u32, limits: &wgpu::Limits) -> (u32, u32, u32) {
  let inv = limits.max_compute_invocations_per_workgroup.min(256);
  let mem = limits.max_compute_workgroup_storage_size.min(16384);
  let mut k = 16u32;
  loop {
    for th in [8u32, 6, 4, 2] {
      for tw in [4u32, 3, 2, 1] {
        let (ew, eh) = (tw + 2, th + 2 * k);
        if ew * eh <= inv && 2 * ew * eh * b * 4 <= mem && eh <= limits.max_compute_workgroup_size_y && ew <= limits.max_compute_workgroup_size_x {
          return (tw, th, k);
        }
      }
    }
    k /= 2;
    if k == 0 {
      return (1, 1, 1);
    }
  }
}

impl DenseKernels {
  pub fn new(ctx: &GpuContext, rule: &Rule) -> Result<DenseKernels> {
    let src = kernels::rule_source(rule);
    let b = src.planes;
    let plain = ctx.pipeline("plain", &kernels::plain(&src), "main", true)?;
    let tile_shape = tile_shape(b, &ctx.limits);
    let tile = ctx.pipeline("shared-tile", &kernels::shared_tile(&src, tile_shape.0, tile_shape.1, tile_shape.2), "main", true)?;
    let census = ctx.pipeline("census", &kernels::census(b), "main", true)?;
    let blit = ctx.pipeline("blit", &kernels::blit(b), "main", true)?;
    let extra = src.extra.as_ref().map(|w| ctx.storage_init("rule", w));
    Ok(DenseKernels { src, plain, tile, tile_shape, census, blit, extra })
  }
  pub fn name(&self) -> String {
    self.src.name.clone()
  }
}

/// What the census of a grid found: population and the bounding box of the live cells (absolute).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
  pub population: u64,
  pub bbox: Option<(i64, i64, i64, i64)>,
}

/// A dense grid of bit-planes on the GPU.
pub struct DenseGrid {
  /// groups per row and rows, the ring included
  pub ww: u32,
  pub h: u32,
  /// absolute coordinates of the first interior cell
  pub x0: i64,
  pub y0: i64,
  pub b: u32,
  pub bufs: [wgpu::Buffer; 2],
  pub cur: usize,
  uni: [wgpu::Buffer; 2],
  census_out: wgpu::Buffer,
  /// bind groups `[source buffer][parity]` of the plain and tile kernels, and the census per buffer
  plain_bg: [[wgpu::BindGroup; 2]; 2],
  tile_bg: [[wgpu::BindGroup; 2]; 2],
  census_bg: [wgpu::BindGroup; 2],
  pub last: Census,
}

impl DenseGrid {
  /// Interior cells: width and height.
  pub fn cells(&self) -> (u64, u64) {
    ((self.ww as u64 - 2) * 32, self.h as u64 - 2)
  }
  pub fn bytes(&self) -> u64 {
    2 * self.ww as u64 * self.h as u64 * self.b as u64 * 4
  }

  fn bind(ctx: &GpuContext, k: &DenseKernels, p: &wgpu::ComputePipeline, uni: &wgpu::Buffer, a: &wgpu::Buffer, b: &wgpu::Buffer) -> wgpu::BindGroup {
    match &k.extra {
      Some(x) => ctx.bind(p, &[uni, a, b, x]),
      None => ctx.bind(p, &[uni, a, b]),
    }
  }

  /// A grid of `w x h` interior cells at `(x0, y0)` with `cells` (row-major, `w * h`) in it.
  pub fn new(ctx: &GpuContext, k: &DenseKernels, w: u64, h: u64, x0: i64, y0: i64, cells: Option<&Grid>) -> Result<DenseGrid> {
    let b = k.src.planes;
    let ww = (w.div_ceil(32) + 2) as u32;
    let hh = (h + 2) as u32;
    let words = ww as u64 * hh as u64 * b as u64;
    if words * 4 > ctx.max_binding() {
      return Err(crate::Error::Gpu(format!("a grid of {w}x{h} cells needs {} MiB, above one binding", (words * 4) >> 20)));
    }
    let mut data = vec![0u32; words as usize];
    if let Some(g) = cells {
      for y in 0..g.h {
        for x in 0..g.w {
          let c = g.cells[y * g.w + x] as u32;
          if c != 0 {
            let base = ((y + 1) * ww as usize + 1 + x / 32) * b as usize;
            for kk in 0..b as usize {
              data[base + kk] |= ((c >> kk) & 1) << (x % 32);
            }
          }
        }
      }
    }
    let a = ctx.storage_init("cells A", &data);
    let bb = ctx.storage("cells B", words * 4);
    let uni = [ctx.uniform("grid even", &[ww, hh, 0, 0]), ctx.uniform("grid odd", &[ww, hh, 1, 0])];
    let census_out = ctx.storage("census", (hh as u64) * 12);
    let bufs = [a, bb];
    let plain_bg = [
      [Self::bind(ctx, k, &k.plain, &uni[0], &bufs[0], &bufs[1]), Self::bind(ctx, k, &k.plain, &uni[1], &bufs[0], &bufs[1])],
      [Self::bind(ctx, k, &k.plain, &uni[0], &bufs[1], &bufs[0]), Self::bind(ctx, k, &k.plain, &uni[1], &bufs[1], &bufs[0])],
    ];
    let tile_bg = [
      [Self::bind(ctx, k, &k.tile, &uni[0], &bufs[0], &bufs[1]), Self::bind(ctx, k, &k.tile, &uni[1], &bufs[0], &bufs[1])],
      [Self::bind(ctx, k, &k.tile, &uni[0], &bufs[1], &bufs[0]), Self::bind(ctx, k, &k.tile, &uni[1], &bufs[1], &bufs[0])],
    ];
    let census_bg = [ctx.bind(&k.census, &[&uni[0], &bufs[0], &census_out]), ctx.bind(&k.census, &[&uni[0], &bufs[1], &census_out])];
    let mut g = DenseGrid { ww, h: hh, x0, y0, b, bufs, cur: 0, uni, census_out, plain_bg, tile_bg, census_bg, last: Census::default() };
    if cells.is_some() {
      g.last = Census { population: cells.map(|c| c.population() as u64).unwrap_or(0), bbox: None };
    }
    Ok(g)
  }

  /// Record `gens` generations from generation `gen0` into one compute pass and submit them.
  pub fn step(&mut self, ctx: &GpuContext, k: &DenseKernels, gens: u64, gen0: u64, shared: bool) {
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      let (tw, th, kk) = k.tile_shape;
      let mut g = 0u64;
      if shared {
        pass.set_pipeline(&k.tile);
        while gens - g >= kk as u64 {
          pass.set_bind_group(0, &self.tile_bg[self.cur][((gen0 + g) & 1) as usize], &[]);
          pass.dispatch_workgroups(self.ww.div_ceil(tw), (self.h - 2).div_ceil(th), 1);
          self.cur ^= 1;
          g += kk as u64;
        }
      }
      if g < gens {
        pass.set_pipeline(&k.plain);
        while g < gens {
          pass.set_bind_group(0, &self.plain_bg[self.cur][((gen0 + g) & 1) as usize], &[]);
          pass.dispatch_workgroups((self.ww - 2).div_ceil(8), (self.h - 2).div_ceil(8), 1);
          self.cur ^= 1;
          g += 1;
        }
      }
    }
    ctx.queue.submit(Some(enc.finish()));
  }

  /// Population and bounding box (one dispatch and a read-back of 12 bytes per row).
  pub fn census(&mut self, ctx: &GpuContext, k: &DenseKernels) -> Census {
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&k.census);
      pass.set_bind_group(0, &self.census_bg[self.cur], &[]);
      pass.dispatch_workgroups((self.h - 2).div_ceil(64), 1, 1);
    }
    ctx.queue.submit(Some(enc.finish()));
    let rows = ctx.read_u32(&self.census_out, 0, (self.h as u64 - 2) * 3);
    let (mut pop, mut x0, mut x1, mut y0, mut y1) = (0u64, u32::MAX, 0u32, u64::MAX, 0u64);
    for (y, r) in rows.chunks(3).enumerate() {
      pop += r[0] as u64;
      if r[0] > 0 {
        x0 = x0.min(r[1]);
        x1 = x1.max(r[2]);
        y0 = y0.min(y as u64);
        y1 = y as u64 + 1;
      }
    }
    self.last = Census { population: pop, bbox: (pop > 0).then(|| (self.x0 + x0 as i64, self.y0 + y0 as i64, self.x0 + x1 as i64, self.y0 + y1 as i64)) };
    self.last
  }

  /// The free cells between the live cells and the ring, on the nearest side (from the last census).
  pub fn margin(&self) -> i64 {
    let (w, h) = self.cells();
    match self.last.bbox {
      None => i64::MAX,
      Some((x0, y0, x1, y1)) => (x0 - self.x0).min(y0 - self.y0).min(self.x0 + w as i64 - x1).min(self.y0 + h as i64 - y1),
    }
  }

  /// A larger grid holding the same cells with `margin` free cells around the live ones
  /// (a GPU copy, no read-back).
  pub fn grown(&self, ctx: &GpuContext, k: &DenseKernels, margin: i64) -> Result<DenseGrid> {
    let Some((bx0, by0, bx1, by1)) = self.last.bbox else {
      return DenseGrid::new(ctx, k, 64, 64, self.x0, self.y0, None);
    };
    // new interior: the bounding box with the margin, its left edge on a group boundary of the old
    let left_groups = ((bx0 - margin - self.x0).div_euclid(32)).min(0);
    let nx0 = self.x0 + left_groups * 32;
    let ny0 = (by0 - margin).min(self.y0);
    let nx1 = (bx1 + margin).max(self.x0 + self.cells().0 as i64);
    let ny1 = (by1 + margin).max(self.y0 + self.cells().1 as i64);
    let mut g = DenseGrid::new(ctx, k, (nx1 - nx0) as u64, (ny1 - ny0) as u64, nx0, ny0, None)?;
    let (ox, oy) = ((-left_groups) as u32, (self.y0 - ny0) as u32);
    let q = ctx.uniform("blit", &[self.ww, self.h, g.ww, g.h, ox, oy, 0, 0]);
    let bg = ctx.bind(&k.blit, &[&q, &self.bufs[self.cur], &g.bufs[0]]);
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&k.blit);
      pass.set_bind_group(0, &bg, &[]);
      pass.dispatch_workgroups(g.ww.div_ceil(8), g.h.div_ceil(8), 1);
    }
    ctx.queue.submit(Some(enc.finish()));
    g.cur = 0;
    g.last = self.last;
    Ok(g)
  }

  /// The cells of `rect` (absolute), row-major, 0 outside the grid.
  pub fn read_rect(&self, ctx: &GpuContext, rect: Rect, out: &mut [u8]) {
    let (w, h) = self.cells();
    let inner = Rect::new(self.x0, self.y0, w, h).intersect(&rect);
    if inner.is_empty() {
      return;
    }
    let b = self.b as usize;
    let gx0 = (inner.x - self.x0) as u64 / 32;
    let gx1 = ((inner.x1() - 1 - self.x0) as u64) / 32 + 1;
    let gw = (gx1 - gx0) as usize;
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    let rows = inner.h as usize;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor { label: Some("read rect"), size: (rows * gw * b * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    for r in 0..rows {
      let y = (inner.y - self.y0) as u64 + r as u64 + 1;
      let off = (y * self.ww as u64 + 1 + gx0) * b as u64 * 4;
      enc.copy_buffer_to_buffer(&self.bufs[self.cur], off, &staging, (r * gw * b * 4) as u64, (gw * b * 4) as u64);
    }
    ctx.queue.submit(Some(enc.finish()));
    let slice = staging.slice(..);
    slice.map_async(wgpu::MapMode::Read, |r| r.expect("map failed"));
    ctx.wait();
    let words: Vec<u32> = bytemuck::pod_collect_to_vec(&slice.get_mapped_range().expect("mapped"));
    staging.unmap();
    for r in 0..rows {
      let ay = inner.y + r as i64;
      for ax in inner.x..inner.x1() {
        let cx = (ax - self.x0) as u64;
        let gi = (cx / 32 - gx0) as usize;
        let bit = cx % 32;
        let base = (r * gw + gi) * b;
        let mut c = 0u8;
        for kk in 0..b {
          c |= (((words[base + kk] >> bit) & 1) as u8) << kk;
        }
        out[((ay - rect.y) as u64 * rect.w + (ax - rect.x) as u64) as usize] = c;
      }
    }
  }

  /// The whole interior as cells.
  pub fn download(&self, ctx: &GpuContext) -> Grid {
    let (w, h) = self.cells();
    let mut g = Grid::new(w as usize, h as usize);
    self.read_rect(ctx, Rect::new(self.x0, self.y0, w, h), &mut g.cells);
    g
  }

  pub fn uniforms(&self) -> &[wgpu::Buffer; 2] {
    &self.uni
  }
}
