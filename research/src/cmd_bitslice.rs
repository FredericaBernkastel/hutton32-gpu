//! `gpu-bitslice`: per-cell stepping with no lookup table at all. The grid is stored as five
//! bit-planes (5 bits per cell), one invocation updates 32 cells with a straight-line Boolean
//! circuit. The circuit comes from `research/bitslice/step_planes.wgsl`:
//!
//!   fn step_planes(c: array<u32,5>, n: array<u32,5>, e: array<u32,5>, s: array<u32,5>,
//!                  w: array<u32,5>) -> array<u32,5>
//!
//! where element k is bit k of the state and bit i of a word is cell i.
//!
//! Layout: rows of `ww` word groups (one padding group on each side), `h + 2` rows; each group is
//! five consecutive `u32` (planes 0..4). Cell x lives in group `1 + x/32`, bit `x % 32`.

use crate::{cmd_stats, flag, gpu::Gpu, opt, opt_or, pattern::Grid, sim::Sim};

const KERNEL: &str = r#"
struct P { ww: u32, h: u32, a: u32, b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

fn ld(i: u32) -> array<u32, 5> {
  let b = i * 5u;
  return array<u32, 5>(src[b], src[b + 1u], src[b + 2u], src[b + 3u], src[b + 4u]);
}
@compute @workgroup_size(WGX, WGY) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x + 1u;
  let y = id.y + 1u;
  if (x >= p.ww - 1u || y >= p.h - 1u) { return; }
  let i = y * p.ww + x;
  let c = ld(i);
  let n = ld(i - p.ww);
  let s = ld(i + p.ww);
  let l = ld(i - 1u);
  let r = ld(i + 1u);
  // east neighbour of cell i is cell i+1: shift down, pull bit 0 of the next group in at the top
  let e = array<u32, 5>((c[0] >> 1u) | (r[0] << 31u), (c[1] >> 1u) | (r[1] << 31u), (c[2] >> 1u) | (r[2] << 31u),
                        (c[3] >> 1u) | (r[3] << 31u), (c[4] >> 1u) | (r[4] << 31u));
  let w = array<u32, 5>((c[0] << 1u) | (l[0] >> 31u), (c[1] << 1u) | (l[1] >> 31u), (c[2] << 1u) | (l[2] >> 31u),
                        (c[3] << 1u) | (l[3] >> 31u), (c[4] << 1u) | (l[4] >> 31u));
  let o = step_planes(c, n, e, s, w);
  let b = i * 5u;
  dst[b] = o[0]; dst[b + 1u] = o[1]; dst[b + 2u] = o[2]; dst[b + 3u] = o[3]; dst[b + 4u] = o[4];
}
"#;

const SIG: &str = "fn step_planes(c: array<u32,5>, n: array<u32,5>, e: array<u32,5>, s: array<u32,5>, w: array<u32,5>) -> array<u32,5>";

fn rnd(x0: u32) -> u32 {
  let mut x = x0;
  x ^= x >> 16;
  x = x.wrapping_mul(0x7FEB352D);
  x ^= x >> 15;
  x = x.wrapping_mul(0x846CA68B);
  x ^= x >> 16;
  x
}

/// Pack into bit-planes. Returns (words, ww, rows).
pub fn pack_planes(g: &Grid) -> (Vec<u32>, usize, usize) {
  let ww = g.w.div_ceil(32) + 2;
  let h = g.h + 2;
  let mut words = vec![0u32; ww * h * 5];
  for y in 0..g.h {
    for x in 0..g.w {
      let c = g.cells[y * g.w + x] as u32;
      if c != 0 {
        let base = ((y + 1) * ww + 1 + x / 32) * 5;
        for k in 0..5 {
          words[base + k] |= ((c >> k) & 1) << (x % 32);
        }
      }
    }
  }
  (words, ww, h)
}
pub fn unpack_planes(words: &[u32], ww: usize, w: usize, h: usize) -> Grid {
  let mut g = Grid::new(w, h);
  for y in 0..h {
    for x in 0..w {
      let base = ((y + 1) * ww + 1 + x / 32) * 5;
      let mut c = 0u32;
      for k in 0..5 {
        c |= ((words[base + k] >> (x % 32)) & 1) << k;
      }
      g.cells[y * w + x] = c as u8;
    }
  }
  g
}

struct Run<'a> {
  gpu: &'a Gpu,
  uni: wgpu::Buffer,
  ww: usize,
  h: usize,
  wg: (u32, u32),
}
impl Run<'_> {
  /// Run `gens` generations of `step` from `words`; returns (seconds, final words).
  fn go(&self, step: &str, words: &[u32], gens: u64, same_src: bool, read: bool) -> (f64, Vec<u32>) {
    let gpu = self.gpu;
    let src = format!("{}\n{}", step, KERNEL.replace("WGX", &self.wg.0.to_string()).replace("WGY", &self.wg.1.to_string()));
    let pipeline = gpu.pipeline("bitslice", &src, "main");
    let a = gpu.storage_init("a", bytemuck::cast_slice(words));
    let b = gpu.storage_init("b", bytemuck::cast_slice(&vec![0u32; words.len()]));
    let mk = |s: &wgpu::Buffer, d: &wgpu::Buffer| {
      gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
          wgpu::BindGroupEntry { binding: 0, resource: self.uni.as_entire_binding() },
          wgpu::BindGroupEntry { binding: 1, resource: s.as_entire_binding() },
          wgpu::BindGroupEntry { binding: 2, resource: d.as_entire_binding() },
        ],
      })
    };
    let (ab, ba) = (mk(&a, &b), mk(&b, &a));
    let (dx, dy) = ((self.ww as u32 - 2).div_ceil(self.wg.0), (self.h as u32 - 2).div_ceil(self.wg.1));
    let run = |n: u64| {
      gpu.timed(|| {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
          let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&pipeline);
          for i in 0..n {
            pass.set_bind_group(0, if same_src || i % 2 == 0 { &ab } else { &ba }, &[]);
            pass.dispatch_workgroups(dx, dy, 1);
          }
        }
        gpu.queue.submit(Some(enc.finish()));
      })
    };
    let secs = run(gens);
    let out = if read {
      // after an even number of ping-pong generations the result is back in `a`
      gpu.read_u32(if same_src || gens % 2 == 1 { &b } else { &a }, 0, words.len() as u64)
    } else {
      Vec::new()
    };
    (secs, out)
  }
}

pub fn run(args: &[String]) {
  let side: usize = opt_or(args, "--side", 8192);
  let gens: u64 = opt_or(args, "--gens", 256);
  let wg: (u32, u32) = (opt_or(args, "--wgx", 8), opt_or(args, "--wgy", 8));
  let same_src = flag(args, "--same-src");
  let spec = cmd_stats::variant_for(args, "hutton32b");
  let lut = crate::load_lut(&spec);
  let default_wgsl = crate::repo_root().join("research/bitslice/step_planes.wgsl");
  let wgsl_path: std::path::PathBuf = opt(args, "--wgsl").map(Into::into).unwrap_or(default_wgsl);

  // content: uniformly random states, or a tiled pattern; width is a multiple of 32 so that the
  // CPU reference sees exactly the cells the kernel updates
  let side = side.div_ceil(32) * 32;
  let (g, what) = if let Some(p) = opt(args, "--pattern") {
    let (loaded, _) = cmd_stats::load_pattern(std::path::Path::new(&p));
    let cmd_stats::Loaded::Dense(pat) = loaded else { panic!("dense pattern required") };
    let (cw, ch) = (pat.w + 8, pat.h + 8);
    let (rx, ry) = ((side / cw).max(1), (side / ch).max(1));
    let mut g = Grid::new((cw * rx).div_ceil(32) * 32, ch * ry);
    for j in 0..ry {
      for i in 0..rx {
        g.blit(&pat, i * cw + 4, j * ch + 4);
      }
    }
    (g, format!("{} tiled {rx}x{ry}", std::path::Path::new(&p).file_name().unwrap().to_string_lossy()))
  } else {
    let mut g = Grid::new(side, side);
    for (i, c) in g.cells.iter_mut().enumerate() {
      *c = (rnd(i as u32 ^ 0xABCDEF) & 31) as u8;
    }
    (g, "uniform random soup".to_string())
  };
  let gpu = Gpu::new(opt(args, "--backend").as_deref());
  println!("GPU: {}", gpu.describe());
  let (words, ww, h) = pack_planes(&g);
  let uni = gpu.device.create_buffer(&wgpu::BufferDescriptor {
    label: None,
    size: 16,
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });
  gpu.queue.write_buffer(&uni, 0, bytemuck::cast_slice(&[ww as u32, h as u32, 0, 0]));
  let r = Run { gpu: &gpu, uni, ww, h, wg };

  // ---- wiring self-test: kernels that just return one neighbour must shift the grid ----
  {
    let small = {
      let mut s = Grid::new(96, 40);
      for (i, c) in s.cells.iter_mut().enumerate() {
        *c = (rnd(i as u32 ^ 0x51DE) & 31) as u8;
      }
      s
    };
    let (sw, sww, sh) = pack_planes(&small);
    let suni = gpu.device.create_buffer(&wgpu::BufferDescriptor {
      label: None,
      size: 16,
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&suni, 0, bytemuck::cast_slice(&[sww as u32, sh as u32, 0, 0]));
    let sr = Run { gpu: &gpu, uni: suni, ww: sww, h: sh, wg };
    let at = |x: isize, y: isize| -> u8 {
      if x < 0 || y < 0 || x >= small.w as isize || y >= small.h as isize { 0 } else { small.get(x as usize, y as usize) }
    };
    for (ret, dx, dy) in [("c", 0isize, 0isize), ("n", 0, -1), ("s", 0, 1), ("e", 1, 0), ("w", -1, 0)] {
      let step = format!("{SIG} {{ return {ret}; }}");
      let (_, out) = sr.go(&step, &sw, 1, true, true);
      let got = unpack_planes(&out, sww, small.w, small.h);
      let bad = (0..small.h).flat_map(|y| (0..small.w).map(move |x| (x, y))).filter(|&(x, y)| got.get(x, y) != at(x as isize + dx, y as isize + dy)).count();
      assert_eq!(bad, 0, "wiring self-test failed for neighbour {ret}");
    }
    println!("wiring self-test: centre and the four neighbours arrive in the right planes and bits");
  }

  let step = std::fs::read_to_string(&wgsl_path).unwrap_or_else(|e| panic!("cannot read {}: {e}", wgsl_path.display()));
  let ops = step.matches(" & ").count() + step.matches(" | ").count() + step.matches(" ^ ").count();
  println!(
    "circuit: {} ({} binary operations, {} inversions in the source)",
    wgsl_path.display(),
    ops,
    step.matches('~').count()
  );
  println!(
    "rule {} | content: {what} | {}x{} = {} cells | {} | {}x{} workgroups, 32 cells per invocation",
    spec.name(),
    g.w,
    g.h,
    g.w * g.h,
    if same_src { "same source every generation" } else { "ping-pong" },
    wg.0,
    wg.1
  );
  let verify = flag(args, "--verify");
  let (secs, out) = r.go(&step, &words, gens, same_src, verify);
  let mut line = format!(
    "  bit-plane kernel: {gens} gens in {secs:.3}s = {:.1} gens/s = {:.3e} cell-updates/s ({:.0} MiB per buffer)",
    gens as f64 / secs,
    (g.w * g.h) as f64 * gens as f64 / secs,
    (words.len() * 4) as f64 / 1048576.0
  );
  if verify {
    let got = unpack_planes(&out, ww, g.w, g.h);
    let mut sim = Sim::new(&g, &lut);
    for _ in 0..if same_src { 1 } else { gens } {
      sim.step();
    }
    let bad = got.cells.iter().zip(&sim.grid().cells).filter(|(a, b)| a != b).count();
    line += &format!(" | verify vs LUT on the CPU: {bad} mismatching cells");
  }
  println!("{line}");
}

// ---------------------------------------------------------------------------------------------
// gpu-window: the miss-path evaluator as ONE dispatch. Each invocation takes a 32x32 window
// (32 rows x 5 planes), steps it 8 generations in private memory with the circuit, and writes
// the central 16x16 tile. No table, no atlas passes.
// ---------------------------------------------------------------------------------------------
const WINDOW_KERNEL: &str = r#"
struct P { n: u32, row: u32, a: u32, b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;

fn east(c: array<u32, 5>) -> array<u32, 5> { return array<u32, 5>(c[0] >> 1u, c[1] >> 1u, c[2] >> 1u, c[3] >> 1u, c[4] >> 1u); }
fn west(c: array<u32, 5>) -> array<u32, 5> { return array<u32, 5>(c[0] << 1u, c[1] << 1u, c[2] << 1u, c[3] << 1u, c[4] << 1u); }

@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let j = id.x + id.y * p.row;
  if (j >= p.n) { return; }
  var a: array<array<u32, 5>, 32>;
  var b: array<array<u32, 5>, 32>;
  let base = j * 160u;
  for (var y = 0u; y < 32u; y++) {
    let o = base + y * 5u;
    a[y] = array<u32, 5>(src[o], src[o + 1u], src[o + 2u], src[o + 3u], src[o + 4u]);
  }
  b[0] = a[0];
  b[31] = a[31];
  // eight generations: rows 0 and 31 and the outer bits go stale one step per generation, which
  // never reaches rows 8..23 / bits 8..23
  for (var g = 0u; g < 4u; g++) {
    for (var y = 1u; y < 31u; y++) { b[y] = step_planes(a[y], a[y - 1u], east(a[y]), a[y + 1u], west(a[y])); }
    for (var y = 1u; y < 31u; y++) { a[y] = step_planes(b[y], b[y - 1u], east(b[y]), b[y + 1u], west(b[y])); }
  }
  let ob = j * 80u;
  for (var y = 0u; y < 16u; y++) {
    let r = a[y + 8u];
    let o = ob + y * 5u;
    dst[o] = (r[0] >> 8u) & 0xFFFFu; dst[o + 1u] = (r[1] >> 8u) & 0xFFFFu; dst[o + 2u] = (r[2] >> 8u) & 0xFFFFu;
    dst[o + 3u] = (r[3] >> 8u) & 0xFFFFu; dst[o + 4u] = (r[4] >> 8u) & 0xFFFFu;
  }
}
"#;

pub fn window(args: &[String]) {
  let n: u32 = opt_or(args, "--windows", 1 << 18);
  let iters: u32 = opt_or(args, "--iters", 8);
  let spec = cmd_stats::variant_for(args, "hutton32b");
  let lut = crate::load_lut(&spec);
  let default_wgsl = crate::repo_root().join("research/bitslice/step_planes.wgsl");
  let wgsl_path: std::path::PathBuf = opt(args, "--wgsl").map(Into::into).unwrap_or(default_wgsl);
  let step = std::fs::read_to_string(&wgsl_path).unwrap_or_else(|e| panic!("cannot read {}: {e}", wgsl_path.display()));
  // content: random states, or windows cut from a pattern at random positions
  let pat = opt(args, "--pattern").map(|p| {
    let (loaded, _) = cmd_stats::load_pattern(std::path::Path::new(&p));
    let cmd_stats::Loaded::Dense(g) = loaded else { panic!("dense pattern required") };
    g
  });
  let cell = |win: u32, x: u32, y: u32| -> u8 {
    match &pat {
      Some(g) => {
        let (ox, oy) = (rnd(win * 2 + 1) as usize % (g.w - 32), rnd(win * 2 + 2) as usize % (g.h - 32));
        g.get(ox + x as usize, oy + y as usize)
      }
      None => (rnd(win.wrapping_mul(1024).wrapping_add(y * 32 + x) ^ 0xC0DE) & 31) as u8,
    }
  };
  let mut words = vec![0u32; n as usize * 160];
  for win in 0..n {
    for y in 0..32u32 {
      for x in 0..32u32 {
        let c = cell(win, x, y) as u32;
        if c != 0 {
          let base = (win * 160 + y * 5) as usize;
          for k in 0..5 {
            words[base + k] |= ((c >> k) & 1) << x;
          }
        }
      }
    }
  }
  let gpu = Gpu::new(opt(args, "--backend").as_deref());
  println!("GPU: {}", gpu.describe());
  let src = format!("{step}\n{WINDOW_KERNEL}");
  let pipeline = gpu.pipeline("window", &src, "main");
  let a = gpu.storage_init("windows", bytemuck::cast_slice(&words));
  let b = gpu.storage("tiles", n as u64 * 80 * 4);
  let row = 1u32 << 22;
  let uni = gpu.device.create_buffer(&wgpu::BufferDescriptor {
    label: None,
    size: 16,
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });
  gpu.queue.write_buffer(&uni, 0, bytemuck::cast_slice(&[n, row, 0, 0]));
  let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: None,
    layout: &pipeline.get_bind_group_layout(0),
    entries: &[
      wgpu::BindGroupEntry { binding: 0, resource: uni.as_entire_binding() },
      wgpu::BindGroupEntry { binding: 1, resource: a.as_entire_binding() },
      wgpu::BindGroupEntry { binding: 2, resource: b.as_entire_binding() },
    ],
  });
  let run = |k: u32| {
    gpu.timed(|| {
      let mut enc = gpu.device.create_command_encoder(&Default::default());
      {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        for _ in 0..k {
          pass.dispatch_workgroups((row / 64).min(n.div_ceil(64)), n.div_ceil(row), 1);
        }
      }
      gpu.queue.submit(Some(enc.finish()));
    })
  };
  run(1);
  let secs = run(iters);
  let rate = n as f64 * iters as f64 / secs;
  // verify a sample against the LUT
  let out = gpu.read_u32(&b, 0, n as u64 * 80);
  let sample = 2048.min(n);
  let mut bad = 0usize;
  let (mut buf, mut tmp) = (vec![0u8; 1024], vec![0u8; 1024]);
  for s in 0..sample {
    let win = (s as u64 * n as u64 / sample as u64) as u32;
    for y in 0..32u32 {
      for x in 0..32u32 {
        buf[(y * 32 + x) as usize] = cell(win, x, y);
      }
    }
    crate::sim::advance_window(&lut, &mut buf, &mut tmp, 32, 8);
    for y in 0..16u32 {
      for x in 0..16u32 {
        let base = (win * 80 + y * 5) as usize;
        let mut c = 0u32;
        for k in 0..5 {
          c |= ((out[base + k] >> x) & 1) << k;
        }
        bad += (c as u8 != buf[((y + 8) * 32 + x + 8) as usize]) as usize;
      }
    }
  }
  println!(
    "window kernel ({}): {n} windows of 32x32, 8 generations in one dispatch: {:.3e} windows/s = {:.1} ns per window = {:.3e} cell evaluations/s | verify: {bad} wrong cells in {sample} windows",
    if pat.is_some() { "pattern windows" } else { "random windows" },
    rate,
    1e9 / rate,
    rate * 8192.0
  );
}
