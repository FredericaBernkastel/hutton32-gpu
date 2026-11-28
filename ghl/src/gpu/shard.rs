//! Grids beyond one binding (Stage 5). wgpu caps a storage binding below 2 GiB on both backends,
//! while a 10^5 × 10^5 universe as plain cells is 6.3 GB per buffer. A sharded grid keeps the flat
//! word array of a dense grid in up to eight buffers of `shard_words` words each; its kernel reads
//! and writes through an accessor that picks the binding from the word index, so one dispatch
//! spans every shard. The same accessor shards a memo table by its top hash bits.

use super::{
  context::GpuContext,
  kernels::{self, RuleSource},
};
use crate::{io::Grid, Error, Result};

/// The accessor functions `ld_w(i)` and `st_w(i, v)` over `n` shard bindings `src{j}` / `dst{j}`.
fn accessors(n: u32, shard_words: u64) -> String {
  let mut s = String::new();
  s += "fn ld_w(i: u32) -> u32 {\n";
  for j in 0..n {
    let lo = j as u64 * shard_words;
    if j + 1 < n {
      s += &format!("  if (i < {}u) {{ return src{j}[i - {lo}u]; }}\n", lo + shard_words);
    } else {
      s += &format!("  return src{j}[i - {lo}u];\n");
    }
  }
  s += "}\nfn st_w(i: u32, v: u32) {\n";
  for j in 0..n {
    let lo = j as u64 * shard_words;
    if j + 1 < n {
      s += &format!("  if (i < {}u) {{ dst{j}[i - {lo}u] = v; return; }}\n", lo + shard_words);
    } else {
      s += &format!("  dst{j}[i - {lo}u] = v;\n");
    }
  }
  s += "}\n";
  s
}

/// The plain kernel over a sharded grid.
pub fn sharded_plain(r: &RuleSource, n: u32, shard_words: u64) -> String {
  let b = r.planes;
  let mut s = String::from("struct P { ww: u32, h: u32, gen: u32, pad: u32 };\n@group(0) @binding(0) var<uniform> p: P;\n");
  for j in 0..n {
    s += &format!("@group(0) @binding({}) var<storage, read> src{j}: array<u32>;\n", 1 + j);
    s += &format!("@group(0) @binding({}) var<storage, read_write> dst{j}: array<u32>;\n", 1 + n + j);
  }
  if r.extra.is_some() {
    s += &format!("@group(0) @binding({}) var<storage, read> extra: array<u32>;\n", 1 + 2 * n);
  }
  s += &accessors(n, shard_words);
  s += &r.step;
  // the plain kernel with `src[...]` and `dst[...]` replaced by the accessors
  let plain = kernels::plain(r);
  let body = plain.split("fn east(").nth(1).expect("plain kernel layout");
  let body = format!("fn east({body}");
  let mut out = String::new();
  for line in body.lines() {
    let mut l = line.to_string();
    // loads: src[b + k] -> ld_w(b + k)
    while let Some(p) = l.find("src[") {
      let end = l[p..].find(']').unwrap() + p;
      let inner = l[p + 4..end].to_string();
      l = format!("{}ld_w({}){}", &l[..p], inner, &l[end + 1..]);
    }
    // stores: dst[x] = v; -> st_w(x, v);
    if let Some(p) = l.find("dst[") {
      let end = l[p..].find(']').unwrap() + p;
      let idx = l[p + 4..end].to_string();
      let rest = l[end + 1..].trim_start().trim_start_matches('=').trim().trim_end_matches(';').to_string();
      l = format!("{}st_w({}, {});", &l[..p], idx, rest);
    }
    out += &l;
    out.push('\n');
  }
  s += &out;
  let _ = b;
  s
}

/// The census kernel (population and bounding box per row) over a sharded grid.
pub fn sharded_census(b: u32, n: u32, shard_words: u64) -> String {
  let mut s = String::from("struct P { ww: u32, h: u32, gen: u32, pad: u32 };\n@group(0) @binding(0) var<uniform> p: P;\n");
  for j in 0..n {
    s += &format!("@group(0) @binding({}) var<storage, read> src{j}: array<u32>;\n", 1 + j);
  }
  s += &format!("@group(0) @binding({}) var<storage, read_write> out: array<u32>;\n", 1 + n);
  s += "fn ld_w(i: u32) -> u32 {\n";
  for j in 0..n {
    let lo = j as u64 * shard_words;
    if j + 1 < n {
      s += &format!("  if (i < {}u) {{ return src{j}[i - {lo}u]; }}\n", lo + shard_words);
    } else {
      s += &format!("  return src{j}[i - {lo}u];\n");
    }
  }
  s += "}\n";
  let census = kernels::census(b);
  let body = census.split("@compute").nth(1).expect("census kernel layout");
  let body = format!("@compute{body}");
  let mut out = String::new();
  for line in body.lines() {
    let mut l = line.to_string();
    while let Some(p) = l.find("src[") {
      let end = l[p..].find(']').unwrap() + p;
      let inner = l[p + 4..end].to_string();
      l = format!("{}ld_w({}){}", &l[..p], inner, &l[end + 1..]);
    }
    out += &l;
    out.push('\n');
  }
  s + &out
}

/// A dense grid split across shards.
pub struct ShardedGrid {
  pub ww: u32,
  pub h: u32,
  pub b: u32,
  pub x0: i64,
  pub y0: i64,
  pub shard_words: u64,
  pub bufs: [Vec<wgpu::Buffer>; 2],
  pub cur: usize,
  pipe: wgpu::ComputePipeline,
  bgs: [[wgpu::BindGroup; 2]; 2],
  census_pipe: wgpu::ComputePipeline,
  census_out: wgpu::Buffer,
  census_bgs: [wgpu::BindGroup; 2],
  /// the last census
  pub last: super::brute::Census,
}

impl ShardedGrid {
  /// A grid of `w x h` interior cells holding `cells`, in shards of at most `shard_words` words.
  #[allow(clippy::too_many_arguments)]
  pub fn new(ctx: &GpuContext, r: &RuleSource, extra: Option<&wgpu::Buffer>, w: u64, h: u64, x0: i64, y0: i64, cells: Option<&Grid>, shard_words: u64) -> Result<ShardedGrid> {
    let b = r.planes;
    let ww = (w.div_ceil(32) + 2) as u32;
    let hh = (h + 2) as u32;
    let words = ww as u64 * hh as u64 * b as u64;
    let n = words.div_ceil(shard_words).max(1) as u32;
    if n > 4 {
      return Err(Error::Gpu(format!("{words} words need {n} shards of {shard_words}; at most four are bound")));
    }
    if words >= 1 << 32 {
      return Err(Error::Gpu("a grid of 2^32 words or more".into()));
    }
    let mut data = vec![0u32; words as usize];
    if let Some(g) = cells {
      for y in 0..g.h {
        for x in 0..g.w {
          let c = g.cells[y * g.w + x] as u32;
          if c != 0 {
            let base = ((y + 1) * ww as usize + 1 + x / 32) * b as usize;
            for k in 0..b as usize {
              data[base + k] |= ((c >> k) & 1) << (x % 32);
            }
          }
        }
      }
    }
    let mk = |label: &str| -> Vec<wgpu::Buffer> {
      (0..n)
        .map(|j| {
          let lo = (j as u64 * shard_words) as usize;
          let hi = ((j as u64 + 1) * shard_words).min(words) as usize;
          ctx.storage_init(label, &data[lo..hi.max(lo)])
        })
        .collect()
    };
    let bufs = [mk("shard A"), mk("shard B")];
    let pipe = ctx.pipeline("sharded plain", &sharded_plain(r, n, shard_words), "main", true)?;
    let uni = [ctx.uniform("grid even", &[ww, hh, 0, 0]), ctx.uniform("grid odd", &[ww, hh, 1, 0])];
    let bind = |u: &wgpu::Buffer, s: &[wgpu::Buffer], d: &[wgpu::Buffer]| {
      let mut list: Vec<&wgpu::Buffer> = vec![u];
      list.extend(s.iter());
      list.extend(d.iter());
      if let Some(x) = extra {
        list.push(x);
      }
      ctx.bind(&pipe, &list)
    };
    let bgs = [[bind(&uni[0], &bufs[0], &bufs[1]), bind(&uni[1], &bufs[0], &bufs[1])], [bind(&uni[0], &bufs[1], &bufs[0]), bind(&uni[1], &bufs[1], &bufs[0])]];
    let census_pipe = ctx.pipeline("sharded census", &sharded_census(b, n, shard_words), "main", true)?;
    let census_out = ctx.storage("census", hh as u64 * 12);
    let cb = |s: &[wgpu::Buffer]| {
      let mut list: Vec<&wgpu::Buffer> = vec![&uni[0]];
      list.extend(s.iter());
      list.push(&census_out);
      ctx.bind(&census_pipe, &list)
    };
    let census_bgs = [cb(&bufs[0]), cb(&bufs[1])];
    let mut g = ShardedGrid { ww, h: hh, b, x0, y0, shard_words, bufs, cur: 0, pipe, bgs, census_pipe, census_out, census_bgs, last: Default::default() };
    g.census(ctx);
    Ok(g)
  }

  pub fn bytes(&self) -> u64 {
    2 * self.ww as u64 * self.h as u64 * self.b as u64 * 4
  }

  /// Population and bounding box (one dispatch, a read-back of 12 bytes per row).
  pub fn census(&mut self, ctx: &GpuContext) -> super::brute::Census {
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&self.census_pipe);
      pass.set_bind_group(0, &self.census_bgs[self.cur], &[]);
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
    self.last = super::brute::Census { population: pop, bbox: (pop > 0).then(|| (self.x0 + x0 as i64, self.y0 + y0 as i64, self.x0 + x1 as i64, self.y0 + y1 as i64)) };
    self.last
  }

  /// The free cells between the live cells and the ring, on the nearest side.
  pub fn margin(&self) -> i64 {
    let (w, h) = ((self.ww as i64 - 2) * 32, self.h as i64 - 2);
    match self.last.bbox {
      None => i64::MAX,
      Some((a, b, c, d)) => (a - self.x0).min(b - self.y0).min(self.x0 + w - c).min(self.y0 + h - d),
    }
  }

  pub fn shards(&self) -> usize {
    self.bufs[0].len()
  }

  /// `gens` generations from generation `gen0`, one compute pass.
  pub fn step(&mut self, ctx: &GpuContext, gens: u64, gen0: u64) {
    let mut enc = ctx.device.create_command_encoder(&Default::default());
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&self.pipe);
      for g in 0..gens {
        pass.set_bind_group(0, &self.bgs[self.cur][((gen0 + g) & 1) as usize], &[]);
        pass.dispatch_workgroups((self.ww - 2).div_ceil(8), (self.h - 2).div_ceil(8), 1);
        self.cur ^= 1;
      }
    }
    ctx.queue.submit(Some(enc.finish()));
  }

  /// The interior as cells.
  pub fn download(&self, ctx: &GpuContext) -> Grid {
    let mut words: Vec<u32> = Vec::new();
    for buf in &self.bufs[self.cur] {
      let n = buf.size() / 4;
      words.extend(ctx.read_u32(buf, 0, n));
    }
    let b = self.b as usize;
    let (w, h) = ((self.ww as usize - 2) * 32, self.h as usize - 2);
    let mut g = Grid::new(w, h);
    for y in 0..h {
      for x in 0..w {
        let base = ((y + 1) * self.ww as usize + 1 + x / 32) * b;
        let mut c = 0u8;
        for k in 0..b {
          c |= (((words[base + k] >> (x % 32)) & 1) as u8) << k;
        }
        g.cells[y * w + x] = c;
      }
    }
    g
  }
}
