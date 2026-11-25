//! GPU measurements: baseline reproduction, per-cell brute force done properly, hash-probe
//! throughput, and the flat tile-CA sweep on real patterns.

use crate::{cmd_stats, cmd_stats::Loaded, flag, flat::Flat, gpu::Gpu, hl::Engine, opt, opt_or, pattern::Grid, sim::Sim};
use std::time::Instant;

fn uniform(gpu: &Gpu, words: &[u32]) -> wgpu::Buffer {
  let buf = gpu.device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("params"),
    size: (words.len() * 4) as u64,
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });
  gpu.queue.write_buffer(&buf, 0, bytemuck::cast_slice(words));
  buf
}

fn dense_pattern(args: &[String]) -> (Grid, crate::ca::Rule, String) {
  let default = crate::patterns_dir().join("presets/hutton32b_8bit_counter.rle");
  let path: std::path::PathBuf = opt(args, "--pattern").map(Into::into).unwrap_or(default);
  let (loaded, file_rule) = cmd_stats::load_pattern(&path);
  let variant = cmd_stats::variant_for(args, &file_rule);
  let lut = crate::load_lut(&variant);
  let g = match loaded {
    Loaded::Dense(g) => g,
    Loaded::Mc(m) => {
      let dummy = crate::ca::Rule::dummy();
      let mut e = Engine::new(&dummy, 1, 1);
      let (root, level) = e.from_mc(&m);
      let full = e.expand(root, level);
      let (x0, y0, x1, y1) = full.bbox().expect("empty pattern");
      full.crop(x0, y0, x1, y1)
    }
  };
  (g, lut, path.file_name().unwrap().to_string_lossy().into_owned())
}

/// Tile `g` `reps x reps` times with a `gap`-cell empty margin around every copy.
fn tiled(g: &Grid, reps: usize, gap: usize) -> Grid {
  let (cw, ch) = (g.w + 2 * gap, g.h + 2 * gap);
  let mut out = Grid::new(cw * reps, ch * reps);
  for j in 0..reps {
    for i in 0..reps {
      out.blit(g, i * cw + gap, j * ch + gap);
    }
  }
  out
}

fn lut_words(lut: &[u8]) -> Vec<u8> {
  lut.to_vec()
}

// ---------------------------------------------------------------------------------------------
// baseline: the archived kernel, verbatim structure (workgroup_size(1), u32 record per cell,
// in-place double buffering via parity bit, one compute pass per generation).
// ---------------------------------------------------------------------------------------------
const BASELINE_WGSL: &str = r#"
struct Uniforms { simulation_dimm: vec2<u32> };
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> simulation_buffer: array<u32>;
@group(0) @binding(2) var<storage, read> hutton32_lut: array<u32>;

fn get_cell(xy: vec2<u32>, parity: bool) -> u32 {
  let offset = xy.y * uniforms.simulation_dimm.x + xy.x;
  return (simulation_buffer[offset] >> (u32(!parity) * 8u)) & 0xffu;
}
fn set_cell(cell_record: u32, next_in: u32, parity: bool) -> u32 {
  let shift = u32(parity) * 8u;
  let next = next_in << shift;
  let current = cell_record & (0xffu << (8u - shift));
  return next | current;
}
@compute @workgroup_size(1) fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let xy = vec2<i32>(global_id.xy);
  let offset = u32(xy.y) * uniforms.simulation_dimm.x + u32(xy.x);
  let current_record = simulation_buffer[offset];
  let parity = bool(current_record >> 16u);
  let c = get_cell(vec2<u32>(xy), parity);
  let n = get_cell(vec2<u32>(xy + vec2( 0, -1)), parity);
  let s = get_cell(vec2<u32>(xy + vec2( 0,  1)), parity);
  let e = get_cell(vec2<u32>(xy + vec2( 1,  0)), parity);
  let w = get_cell(vec2<u32>(xy + vec2(-1,  0)), parity);
  let lut_offset = w | s << 5u | e << 10u | n << 15u | c << 20u;
  let cell = (hutton32_lut[lut_offset / 4u] >> ((lut_offset % 4u) * 8u)) & 0xFFu;
  let next_record = set_cell(current_record, cell, parity);
  simulation_buffer[offset] = next_record | (u32(!parity) << 16u);
}
"#;

pub fn baseline(args: &[String]) {
  let (pat, lut, name) = dense_pattern(args);
  let reps: usize = opt_or(args, "--reps", 1);
  // one empty row/column of padding so the original's missing boundary handling cannot matter
  let g = tiled(&pat, reps, opt_or(args, "--gap", 1));
  let gens: u64 = opt_or(args, "--gens", 8192);
  let per_submit: u64 = opt_or(args, "--per-submit", 32);
  let gpu = Gpu::new(opt(args, "--backend").as_deref());
  println!("GPU: {}", gpu.describe());
  println!("baseline (archived kernel): {name} x{reps} -> {}x{} = {} cells", g.w, g.h, g.w * g.h);
  // the app uploads RGBA pixels with R := state, so bits 16.. (blue, alpha=255) are non-zero and
  // the first step sees parity = true (current state in the low byte)
  let records: Vec<u32> = g.cells.iter().map(|&c| c as u32 | 0xFF00_0000).collect();
  let sim_buf = gpu.storage_init("sim", bytemuck::cast_slice(&records));
  let lut_buf = gpu.storage_init("lut", &lut_words(&lut));
  let uni = uniform(&gpu, &[g.w as u32, g.h as u32]);
  let pipeline = gpu.pipeline("baseline", BASELINE_WGSL, "compute_main");
  let bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: None,
    layout: &pipeline.get_bind_group_layout(0),
    entries: &[
      wgpu::BindGroupEntry { binding: 0, resource: uni.as_entire_binding() },
      wgpu::BindGroupEntry { binding: 1, resource: sim_buf.as_entire_binding() },
      wgpu::BindGroupEntry { binding: 2, resource: lut_buf.as_entire_binding() },
    ],
  });
  let mut done = 0u64;
  let secs = gpu.timed(|| {
    while done < gens {
      let n = per_submit.min(gens - done);
      let mut enc = gpu.device.create_command_encoder(&Default::default());
      for _ in 0..n {
        // one pass per generation, exactly like `simulation_advance`
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(g.w as u32, g.h as u32, 1);
      }
      gpu.queue.submit(Some(enc.finish()));
      done += n;
      if done % (per_submit * 8) == 0 {
        gpu.wait();
      }
    }
  });
  println!(
    "  {gens} gens in {secs:.3}s = {:.1} gens/s = {:.3e} cell-updates/s",
    gens as f64 / secs,
    (g.w * g.h) as f64 * gens as f64 / secs
  );
  if flag(args, "--verify") {
    let out = gpu.read_u32(&sim_buf, 0, (g.w * g.h) as u64);
    let mut sim = Sim::new(&g, &lut);
    for _ in 0..gens {
      sim.step();
    }
    let want = sim.grid();
    let bad = out
      .iter()
      .zip(&want.cells)
      .filter(|(r, &c)| {
        let parity = (**r >> 16) & 1;
        let cur = (**r >> ((1 - parity) * 8)) & 0xff;
        cur != c as u32
      })
      .count();
    println!("  verify vs CPU: {bad} mismatching cells");
  }
}

// ---------------------------------------------------------------------------------------------
// brute: per-cell LUT stepping done properly. One byte per cell packed 4 per u32, ping-pong
// buffers with a zero ring, 8x8 workgroups, each invocation updates one word (4 cells), all
// generations of a submit recorded in a single compute pass.
// ---------------------------------------------------------------------------------------------
const BRUTE_WGSL: &str = r#"
struct P { ww: u32, h: u32, a: u32, b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> lut: array<u32>;

fn lut_get(i: u32) -> u32 { return (lut[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu; }

@compute @workgroup_size(WGX, WGY) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x + 1u;
  let y = id.y + 1u;
  if (x >= p.ww - 1u || y >= p.h - 1u) { return; }
  let i = y * p.ww + x;
  let c = src[i];
  let up = src[i - p.ww];
  let dn = src[i + p.ww];
  let wv = (c << 8u) | (src[i - 1u] >> 24u);
  let ev = (c >> 8u) | (src[i + 1u] << 24u);
  var out = 0u;
  for (var k = 0u; k < 4u; k++) {
    let sh = k * 8u;
    let idx = ((wv >> sh) & 31u) | (((dn >> sh) & 31u) << 5u) | (((ev >> sh) & 31u) << 10u)
            | (((up >> sh) & 31u) << 15u) | (((c >> sh) & 31u) << 20u);
    out |= lut_get(idx) << sh;
  }
  dst[i] = out;
}
"#;

/// Pack a grid into padded words: rows `h+2`, `ww = ceil(w/4)+2` words per row.
fn pack_bytes(g: &Grid) -> (Vec<u32>, usize, usize) {
  let ww = g.w.div_ceil(4) + 2;
  let h = g.h + 2;
  let mut words = vec![0u32; ww * h];
  for y in 0..g.h {
    for x in 0..g.w {
      let c = g.cells[y * g.w + x] as u32;
      words[(y + 1) * ww + 1 + x / 4] |= c << ((x % 4) * 8);
    }
  }
  (words, ww, h)
}
fn unpack_bytes(words: &[u32], ww: usize, w: usize, h: usize) -> Grid {
  let mut g = Grid::new(w, h);
  for y in 0..h {
    for x in 0..w {
      g.cells[y * w + x] = ((words[(y + 1) * ww + 1 + x / 4] >> ((x % 4) * 8)) & 0xff) as u8;
    }
  }
  g
}

pub fn brute(args: &[String]) {
  let (pat, lut, name) = dense_pattern(args);
  let reps: usize = opt_or(args, "--reps", 1);
  let g = tiled(&pat, reps, opt_or(args, "--gap", 4));
  let gens: u64 = opt_or(args, "--gens", 1024) & !1;
  let per_submit: u64 = opt_or(args, "--per-submit", 64) & !1;
  let (wgx, wgy): (u32, u32) = (opt_or(args, "--wgx", 8), opt_or(args, "--wgy", 8));
  let gpu = Gpu::new(opt(args, "--backend").as_deref());
  println!("GPU: {}", gpu.describe());
  println!("brute (byte cells, {wgx}x{wgy} workgroups): {name} x{reps} -> {}x{} = {} cells", g.w, g.h, g.w * g.h);
  let (words, ww, h) = pack_bytes(&g);
  let a = gpu.storage_init("a", bytemuck::cast_slice(&words));
  let b = gpu.storage_init("b", bytemuck::cast_slice(&vec![0u32; words.len()]));
  let lut_buf = gpu.storage_init("lut", &lut);
  let uni = uniform(&gpu, &[ww as u32, h as u32, 0, 0]);
  let src = BRUTE_WGSL.replace("WGX", &wgx.to_string()).replace("WGY", &wgy.to_string());
  let pipeline = gpu.pipeline("brute", &src, "main");
  let mk = |s: &wgpu::Buffer, d: &wgpu::Buffer| {
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &pipeline.get_bind_group_layout(0),
      entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: uni.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: s.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: d.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: lut_buf.as_entire_binding() },
      ],
    })
  };
  let (ab, ba) = (mk(&a, &b), mk(&b, &a));
  let (dx, dy) = ((ww as u32 - 2).div_ceil(wgx), (h as u32 - 2).div_ceil(wgy));
  let all = crate::repeat_timed(args, || {
    gpu.queue.write_buffer(&a, 0, bytemuck::cast_slice(&words));
    let mut done = 0u64;
    gpu.timed(|| {
      while done < gens {
        let n = per_submit.min(gens - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
          let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&pipeline);
          for i in 0..n {
            pass.set_bind_group(0, if i % 2 == 0 { &ab } else { &ba }, &[]);
            pass.dispatch_workgroups(dx, dy, 1);
          }
        }
        gpu.queue.submit(Some(enc.finish()));
        done += n;
        gpu.wait();
      }
    })
  });
  let (secs, spread) = crate::median_and_spread(&all, gens as f64, "gens/s");
  println!(
    "  {gens} gens in {secs:.3}s = {:.1} gens/s = {:.3e} cell-updates/s{spread}",
    gens as f64 / secs,
    (g.w * g.h) as f64 * gens as f64 / secs
  );
  if flag(args, "--verify") {
    let out = gpu.read_u32(&a, 0, words.len() as u64);
    let got = unpack_bytes(&out, ww, g.w, g.h);
    let t = Instant::now();
    let mut sim = Sim::new(&g, &lut);
    for _ in 0..gens {
      sim.step();
    }
    let bad = got.cells.iter().zip(&sim.grid().cells).filter(|(a, b)| a != b).count();
    println!(
      "  verify vs CPU: {bad} mismatching cells (CPU reference: {:.3e} cell-updates/s single thread)",
      (g.w * g.h) as f64 * gens as f64 / t.elapsed().as_secs_f64()
    );
  }
}

// ---------------------------------------------------------------------------------------------
// sweep: the flat tile CA on the GPU. The memo table is learned by the CPU reference engine over
// one period of the pattern; the GPU then runs pure hit-path sweeps on `reps x reps` copies, each
// started at a different phase of the period, and the result is checked tile by tile.
// ---------------------------------------------------------------------------------------------
const SWEEP_WGSL: &str = r#"
struct P { sw: u32, sh: u32, dw: u32, dh: u32, off: i32, mask: u32, empty: u32, pad: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> table: array<u32>;
@group(0) @binding(4) var<storage, read_write> misses: array<atomic<u32>>;

fn at(x: i32, y: i32) -> u32 {
  if (x < 0 || y < 0 || x >= i32(p.sw) || y >= i32(p.sh)) { return p.empty; }
  return src[u32(y) * p.sw + u32(x)];
}
fn hash4(k: vec4<u32>) -> u32 {
  var h = k.x * 0x9E3779B1u;
  h = (h ^ (h >> 15u)) + k.y * 0x85EBCA77u;
  h = (h ^ (h >> 13u)) + k.z * 0xC2B2AE3Du;
  h = (h ^ (h >> 16u)) + k.w * 0x27D4EB2Fu;
  h = (h ^ (h >> 15u)) * 0x2C1B3C6Du;
  return h ^ (h >> 12u);
}
@compute @workgroup_size(WGX, WGY) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.dw || id.y >= p.dh) { return; }
  let x = i32(id.x) + p.off;
  let y = i32(id.y) + p.off;
  let k = vec4<u32>(at(x, y), at(x + 1, y), at(x, y + 1), at(x + 1, y + 1));
  var slot = hash4(k) & p.mask;
  var val = 0xFFFFFFFFu;
  for (var i = 0u; i < 256u; i++) {
    let base = slot * 8u;
    let e = vec4<u32>(table[base], table[base + 1u], table[base + 2u], table[base + 3u]);
    if (all(e == k)) { val = table[base + 4u]; break; }
    if (e.x == 0xFFFFFFFFu) { break; }
    slot = (slot + 1u) & p.mask;
  }
  if (val == 0xFFFFFFFFu) { atomicAdd(&misses[0], 1u); val = p.empty; }
  dst[id.y * p.dw + id.x] = val;
}
"#;

pub fn hash4(k: [u32; 4]) -> u32 {
  let mut h = k[0].wrapping_mul(0x9E3779B1);
  h = (h ^ (h >> 15)).wrapping_add(k[1].wrapping_mul(0x85EBCA77));
  h = (h ^ (h >> 13)).wrapping_add(k[2].wrapping_mul(0xC2B2AE3D));
  h = (h ^ (h >> 16)).wrapping_add(k[3].wrapping_mul(0x27D4EB2F));
  h = (h ^ (h >> 15)).wrapping_mul(0x2C1B3C6D);
  h ^ (h >> 12)
}

/// Open-addressing table, 8 words per entry: `[k0, k1, k2, k3, value, 0, 0, 0]`.
pub fn build_table(entries: impl Iterator<Item = ([u32; 4], u32)>, count: usize, min_log2: u32) -> (Vec<u32>, u32, f64) {
  let mut log2 = min_log2;
  while (1usize << log2) < count * 2 {
    log2 += 1;
  }
  let mask = (1u32 << log2) - 1;
  let mut t = vec![0u32; 8usize << log2];
  for s in 0..(1usize << log2) {
    t[s * 8] = u32::MAX;
    t[s * 8 + 1] = u32::MAX;
    t[s * 8 + 2] = u32::MAX;
    t[s * 8 + 3] = u32::MAX;
  }
  let (mut probes, mut n) = (0u64, 0u64);
  for (k, v) in entries {
    let mut slot = hash4(k) & mask;
    loop {
      probes += 1;
      let b = slot as usize * 8;
      if t[b] == u32::MAX && t[b + 1] == u32::MAX {
        t[b..b + 4].copy_from_slice(&k);
        t[b + 4] = v;
        break;
      }
      slot = (slot + 1) & mask;
    }
    n += 1;
  }
  (t, mask, probes as f64 / n.max(1) as f64)
}

pub fn sweep(args: &[String]) {
  let path: std::path::PathBuf = opt(args, "--pattern").expect("--pattern").into();
  let (loaded, file_rule) = cmd_stats::load_pattern(&path);
  let variant = cmd_stats::variant_for(args, &file_rule);
  let lut = crate::load_lut(&variant);
  let a: usize = opt_or(args, "--a", 2);
  let lt: usize = opt_or(args, "--lt", 4);
  let reps: usize = opt_or(args, "--reps", 8);
  let cycles: u64 = opt_or(args, "--cycles", 64);
  let max_learn: u64 = opt_or(args, "--max-learn-gens", 1 << 16);
  let (wgx, wgy): (u32, u32) = (opt_or(args, "--wgx", 16), opt_or(args, "--wgy", 16));
  let s = 1u64 << lt;

  // ---- learn one period on the CPU ----
  // `--alignments K` (default 1): learn the pattern at K different cell offsets inside a tile, so
  // that copies at arbitrary positions can be simulated; copy (cx, cy) uses class (cx + cy * reps) % K.
  let n_align: usize = opt_or(args, "--alignments", 1);
  let t0 = Instant::now();
  let mut e = Engine::new(&lut, a, (lt - 1).max(a));
  let (mut nx, mut ny, mut period_cycles) = (0usize, 0usize, 0u64);
  let mut classes: Vec<Vec<Vec<u32>>> = Vec::with_capacity(n_align);
  let mut offsets: Vec<(usize, usize)> = Vec::with_capacity(n_align);
  for k in 0..n_align {
    let mut f = match &loaded {
      Loaded::Dense(g) if n_align == 1 => Flat::from_grid(&mut e, g, lt, 2),
      Loaded::Dense(g) => {
        let su = s as usize;
        let (ox, oy) = if k == 0 { (0, 0) } else { ((k * 2654435761usize >> 5) % su, (k * 40503 + 7 * k * k) % su) };
        offsets.push((ox, oy));
        let mut c = crate::pattern::Grid::new(g.w + su, g.h + su);
        c.blit(g, ox, oy);
        Flat::from_grid(&mut e, &c, lt, 2)
      }
      Loaded::Mc(m) => {
        assert_eq!(n_align, 1, "--alignments needs a dense pattern");
        let (root, level) = e.from_mc(m);
        let (root, level, _) = e.popzeros(root, level);
        let ids = e.ids_at_level(root, level, lt);
        Flat::from_ids(&mut e, &ids, 1 << (level - lt), lt, 2)
      }
    };
    if k == 0 {
      (nx, ny) = (f.nx, f.ny);
    }
    assert_eq!((f.nx, f.ny), (nx, ny), "alignment classes must have the same tile grid");
    let mut seen: rustc_hash::FxHashMap<Vec<u32>, u64> = Default::default();
    let mut pc = 0u64;
    seen.insert(f.a_ids.clone(), 0);
    let mut cyc = 0u64;
    while cyc * s < max_learn {
      f.sweep(&mut e);
      f.sweep(&mut e);
      cyc += 1;
      if let Some(&prev) = seen.get(&f.a_ids) {
        pc = cyc - prev;
        break;
      }
      seen.insert(f.a_ids.clone(), cyc);
    }
    assert!(pc > 0, "pattern did not become periodic within --max-learn-gens");
    drop(seen);
    if k == 0 {
      period_cycles = pc;
    }
    assert_eq!(pc, period_cycles, "alignment classes must have the same period");
    // record the cycle: a_phase[i] = A grid after i cycles from here (f is on the cycle now)
    let mut a_phase: Vec<Vec<u32>> = Vec::with_capacity(pc as usize);
    for _ in 0..pc {
      a_phase.push(f.a_ids.clone());
      f.sweep(&mut e);
      f.sweep(&mut e);
    }
    assert_eq!(f.a_ids, a_phase[0], "cycle did not close");
    classes.push(a_phase);
  }
  if n_align > 1 {
    println!("alignment classes: {n_align}, cell offsets {offsets:?}");
  }
  let class_of = |cx: usize, cy: usize| -> usize { (cx + cy * opt_or::<usize>(args, "--reps", 8)) % n_align };
  let top = &e.levels[lt + 1];
  let n_entries = top.res.iter().filter(|&&r| r != crate::hl::NONE).count();
  println!(
    "pattern {} rule {}: tile {s}x{s}, grid {nx}x{ny} tiles, period {} gens ({} cycles), learned in {:.1}s",
    path.file_name().unwrap().to_string_lossy(),
    variant.name(),
    period_cycles * s,
    period_cycles,
    t0.elapsed().as_secs_f64()
  );
  let (table, mask, avg_probe) = build_table(
    top.nodes.iter().zip(&top.res).filter(|(_, &r)| r != crate::hl::NONE).map(|(k, &r)| (*k, r)),
    n_entries,
    10,
  );
  let n_tiles = if lt == a { e.num_tiles() } else { e.levels[lt].nodes.len() };
  println!(
    "memo: {n_entries} entries -> table {} slots ({} MiB), avg insert probes {avg_probe:.2}; dictionary {n_tiles} tiles",
    mask + 1,
    table.len() * 4 >> 20
  );
  let empty = e.empty(lt);

  // ---- big universe: reps x reps copies at pseudo-random phases ----
  let (bw, bh) = (nx * reps, ny * reps);
  let mut big = vec![empty; bw * bh];
  let phase_of = |cx: usize, cy: usize| -> u64 { ((cx as u64 * 2654435761 + cy as u64 * 40503 + 12345) >> 3) % period_cycles };
  for cy in 0..reps {
    for cx in 0..reps {
      let src = &classes[class_of(cx, cy)][phase_of(cx, cy) as usize];
      for y in 0..ny {
        big[(cy * ny + y) * bw + cx * nx..(cy * ny + y) * bw + cx * nx + nx].copy_from_slice(&src[y * nx..(y + 1) * nx]);
      }
    }
  }
  let cells = (bw as u64 * s) * (bh as u64 * s);
  let gpu = Gpu::new(opt(args, "--backend").as_deref());
  println!("GPU: {}", gpu.describe());
  println!(
    "universe: {reps}x{reps} copies = {bw}x{bh} tiles = {}x{} cells ({:.3e}); id buffers {} MiB each",
    bw as u64 * s,
    bh as u64 * s,
    cells as f64,
    (bw * bh * 4) >> 20
  );
  let buf_a = gpu.storage_init("ids_a", bytemuck::cast_slice(&big));
  let buf_b = gpu.storage("ids_b", ((bw + 1) * (bh + 1) * 4) as u64);
  let buf_t = gpu.storage_init("table", bytemuck::cast_slice(&table));
  let buf_m = gpu.storage_init("misses", &[0u8; 16]);
  let src = SWEEP_WGSL.replace("WGX", &wgx.to_string()).replace("WGY", &wgy.to_string());
  let pipeline = gpu.pipeline("sweep", &src, "main");
  // A -> B: dst (bw+1)x(bh+1), window top-left = dst - 1.   B -> A: dst bw x bh, offset 0.
  let u_ab = uniform(&gpu, &[bw as u32, bh as u32, bw as u32 + 1, bh as u32 + 1, (-1i32) as u32, mask, empty, 0]);
  let u_ba = uniform(&gpu, &[bw as u32 + 1, bh as u32 + 1, bw as u32, bh as u32, 0, mask, empty, 0]);
  let mk = |u: &wgpu::Buffer, s: &wgpu::Buffer, d: &wgpu::Buffer| {
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &pipeline.get_bind_group_layout(0),
      entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: s.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: d.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: buf_t.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: buf_m.as_entire_binding() },
      ],
    })
  };
  let (ab, ba) = (mk(&u_ab, &buf_a, &buf_b), mk(&u_ba, &buf_b, &buf_a));
  let per_submit: u64 = opt_or(args, "--per-submit", 8);
  let all = crate::repeat_timed(args, || {
    gpu.queue.write_buffer(&buf_a, 0, bytemuck::cast_slice(&big));
    gpu.queue.write_buffer(&buf_m, 0, &[0u8; 16]);
    let mut done = 0u64;
    gpu.timed(|| {
      while done < cycles {
        let n = per_submit.min(cycles - done);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
          let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&pipeline);
          for _ in 0..n {
            pass.set_bind_group(0, &ab, &[]);
            pass.dispatch_workgroups((bw as u32 + 1).div_ceil(wgx), (bh as u32 + 1).div_ceil(wgy), 1);
            pass.set_bind_group(0, &ba, &[]);
            pass.dispatch_workgroups((bw as u32).div_ceil(wgx), (bh as u32).div_ceil(wgy), 1);
          }
        }
        gpu.queue.submit(Some(enc.finish()));
        done += n;
        gpu.wait();
      }
    })
  });
  let gens = cycles * s;
  let (secs, spread) = crate::median_and_spread(&all, gens as f64, "gens/s");
  let probes = cycles as f64 * ((bw + 1) * (bh + 1) + bw * bh) as f64;
  println!(
    "  {gens} gens ({} sweeps) in {secs:.3}s = {:.1} gens/s = {:.3e} cell-updates/s | {:.3e} table probes/s{spread}",
    cycles * 2,
    gens as f64 / secs,
    cells as f64 * gens as f64 / secs,
    probes / secs
  );
  // ---- verify ----
  let misses = gpu.read_u32(&buf_m, 0, 1)[0];
  let out = gpu.read_u32(&buf_a, 0, (bw * bh) as u64);
  let mut bad = 0usize;
  for cy in 0..reps {
    for cx in 0..reps {
      let want = &classes[class_of(cx, cy)][((phase_of(cx, cy) + cycles) % period_cycles) as usize];
      for y in 0..ny {
        let row = &out[(cy * ny + y) * bw + cx * nx..(cy * ny + y) * bw + cx * nx + nx];
        bad += row.iter().zip(&want[y * nx..(y + 1) * nx]).filter(|(a, b)| a != b).count();
      }
    }
  }
  println!("  verify: {misses} table misses, {bad} mismatching tiles of {}", bw * bh);
}

// ---------------------------------------------------------------------------------------------
// probe: random 32-byte entry fetches from a table of 2^n slots spread over up to 8 bindings
// (1 GiB each). Reproduces the memory behaviour of a memo hit (hash -> one entry -> compare ->
// 4-byte result written sequentially) for tables far beyond the L2 cache.
// ---------------------------------------------------------------------------------------------
fn probe_wgsl(nb: u32) -> String {
  let mut s = String::from(
    r#"
struct P { mask: u32, shift: u32, sub_mask: u32, seed: u32, n: u32, row: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> out: array<u32>;
"#,
  );
  for b in 0..nb {
    s += &format!("@group(0) @binding({}) var<storage, read_write> t{b}: array<u32>;\n", b + 2);
  }
  s += r#"
fn rnd(x0: u32) -> u32 {
  var x = x0;
  x ^= x >> 16u; x *= 0x7FEB352Du; x ^= x >> 15u; x *= 0x846CA68Bu; x ^= x >> 16u;
  return x;
}
fn key_of(slot: u32) -> vec4<u32> {
  return vec4<u32>(slot * 0x9E3779B1u, slot ^ 0xA5A5A5A5u, slot * 7u + 1u, ~slot);
}
@compute @workgroup_size(256) fn fill(@builtin(global_invocation_id) id: vec3<u32>) {
  let slot = id.x + id.y * p.row;
  if (slot > p.mask) { return; }
  let k = key_of(slot);
  let i = (slot & p.sub_mask) * 8u;
  switch (slot >> p.shift) {
"#;
  for b in 0..nb {
    s += &format!(
      "    case {b}u: {{ t{b}[i] = k.x; t{b}[i + 1u] = k.y; t{b}[i + 2u] = k.z; t{b}[i + 3u] = k.w; t{b}[i + 4u] = rnd(slot); }}\n"
    );
  }
  s += r#"    default: {}
  }
}
@compute @workgroup_size(256) fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
  let j = id.x + id.y * p.row;
  if (j >= p.n) { return; }
  let slot = rnd(j ^ p.seed) & p.mask;
  let want = key_of(slot);
  let i = (slot & p.sub_mask) * 8u;
  var k = vec4<u32>(0u);
  var v = 0u;
  switch (slot >> p.shift) {
"#;
  for b in 0..nb {
    s += &format!("    case {b}u: {{ k = vec4<u32>(t{b}[i], t{b}[i + 1u], t{b}[i + 2u], t{b}[i + 3u]); v = t{b}[i + 4u]; }}\n");
  }
  s += r#"    default: {}
  }
  if (!all(k == want)) { v = 0xFFFFFFFFu; }
  out[j] = v;
}
"#;
  s
}

fn rnd32(x0: u32) -> u32 {
  let mut x = x0;
  x ^= x >> 16;
  x = x.wrapping_mul(0x7FEB352D);
  x ^= x >> 15;
  x = x.wrapping_mul(0x846CA68B);
  x ^= x >> 16;
  x
}

pub fn probe(args: &[String]) {
  let gpu = Gpu::new(opt(args, "--backend").as_deref());
  println!("GPU: {}", gpu.describe());
  let lookups: u32 = opt_or(args, "--lookups", 1 << 25);
  let iters: u32 = opt_or(args, "--iters", 16);
  let sizes: Vec<u32> = opt(args, "--log2-slots")
    .map(|s| s.split(',').map(|x| x.parse().unwrap()).collect())
    .unwrap_or((10..=28).step_by(2).collect());
  let out = gpu.storage("out", lookups as u64 * 4);
  println!("  slots(log2)  table      lookups/dispatch   lookups/s      GB/s(32B)   check");
  for &log2 in &sizes {
    let sub_log2 = log2.min(25); // 2^25 slots * 32 B = 1 GiB per binding
    let nb = 1u32 << (log2 - sub_log2);
    assert!(nb <= 8, "at most 8 GiB");
    let bufs: Vec<wgpu::Buffer> = (0..nb).map(|b| gpu.storage(&format!("t{b}"), 32u64 << sub_log2)).collect();
    let src = probe_wgsl(nb);
    let fill = gpu.pipeline("fill", &src, "fill");
    let probe = gpu.pipeline("probe", &src, "probe");
    let mask = ((1u64 << log2) - 1) as u32;
    let row = 1u32 << 22; // invocations per dispatch row: 16384 groups * 256
    let mk = |pl: &wgpu::ComputePipeline, seed: u32, with_out: bool| {
      let u = uniform(&gpu, &[mask, sub_log2, (1u32 << sub_log2) - 1, seed, lookups, row, 0, 0]);
      let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() }];
      if with_out {
        entries.push(wgpu::BindGroupEntry { binding: 1, resource: out.as_entire_binding() });
      }
      for (b, buf) in bufs.iter().enumerate() {
        entries.push(wgpu::BindGroupEntry { binding: b as u32 + 2, resource: buf.as_entire_binding() });
      }
      gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pl.get_bind_group_layout(0), entries: &entries })
    };
    // fill
    {
      let bg = mk(&fill, 0, false);
      let mut enc = gpu.device.create_command_encoder(&Default::default());
      {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&fill);
        pass.set_bind_group(0, &bg, &[]);
        let slots = 1u64 << log2;
        pass.dispatch_workgroups((row / 256).min((slots / 256).max(1) as u32), slots.div_ceil(row as u64) as u32, 1);
      }
      gpu.queue.submit(Some(enc.finish()));
      gpu.wait();
    }
    let groups: Vec<wgpu::BindGroup> = (0..iters).map(|i| mk(&probe, 0x1234567 + i * 0x9E3779B1u32, true)).collect();
    let run = |n: u32| {
      gpu.timed(|| {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
          let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&probe);
          for g in groups.iter().take(n as usize) {
            pass.set_bind_group(0, g, &[]);
            pass.dispatch_workgroups((row / 256).min(lookups.div_ceil(256)), lookups.div_ceil(row), 1);
          }
        }
        gpu.queue.submit(Some(enc.finish()));
      })
    };
    run(1); // warm-up (page-in, shader compile)
    let secs = run(iters);
    // check the last dispatch's results on a sample
    let seed = 0x1234567u32.wrapping_add((iters - 1).wrapping_mul(0x9E3779B1));
    let sample = gpu.read_u32(&out, 0, 4096.min(lookups) as u64);
    let ok = sample.iter().enumerate().all(|(j, &v)| v == rnd32(rnd32(j as u32 ^ seed) & mask));
    let rate = lookups as f64 * iters as f64 / secs;
    println!(
      "  {:>11}  {:>6} MiB  {:>16}  {:>10.3e}  {:>10.1}   {}",
      log2,
      (32u64 << log2) >> 20,
      lookups,
      rate,
      rate * 32.0 / 1e9,
      if ok { "ok" } else { "MISMATCH" }
    );
  }
}

// ---------------------------------------------------------------------------------------------
// latency: fixed costs that bound the level-synchronous schedule when batches are small.
// ---------------------------------------------------------------------------------------------
const TINY_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<storage, read_write> args: array<u32>;
@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  buf[id.x] = buf[id.x] * 1664525u + 1013904223u;
}
// writes the next indirect dispatch count from data, as a miss-compaction pass would
@compute @workgroup_size(1) fn count() {
  args[0] = 1u + (buf[0] & 3u); args[1] = 1u; args[2] = 1u;
}
"#;


/// Fixed-capacity dispatch: every invocation looks up the live count and leaves if it is beyond it.
const SKIP_WGSL: &str = r#"
@group(0) @binding(0) var<storage, read_write> buf: array<u32>;
@group(0) @binding(1) var<storage, read> live: array<u32>;
@compute @workgroup_size(64) fn skip(@builtin(global_invocation_id) id: vec3<u32>) {
  let j = id.x + id.y * 4194240u;
  if (j >= live[0]) { return; }
  buf[j & 4095u] = buf[j & 4095u] * 1664525u + 1013904223u;
}
"#;

/// Costs of sizing a dispatch from data computed on the GPU (what the miss path needs).
fn data_dependent(args: &[String], indirect_validation: bool) {
  let gpu = Gpu::new_with(opt(args, "--backend").as_deref(), indirect_validation);
  let n: u32 = opt_or(args, "--n", 20000);
  println!("-- indirect-dispatch validation {} --", if indirect_validation { "on (wgpu default)" } else { "off" });
  let buf = gpu.storage_init("buf", bytemuck::cast_slice(&vec![1u32; 4096]));
  let mk_ind = || {
    let b = gpu.device.create_buffer(&wgpu::BufferDescriptor {
      label: Some("indirect"),
      size: 16,
      usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&b, 0, bytemuck::cast_slice(&[1u32, 1, 1, 0]));
    b
  };
  let (ind, ind2) = (mk_ind(), mk_ind());
  let main = gpu.pipeline("tiny", TINY_WGSL, "main");
  let count = gpu.pipeline("count", TINY_WGSL, "count");
  let bg_main = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: None,
    layout: &main.get_bind_group_layout(0),
    entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
  });
  let bg_count = |i: &wgpu::Buffer| {
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &count.get_bind_group_layout(0),
      entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: i.as_entire_binding() },
      ],
    })
  };
  let (bgc1, bgc2) = (bg_count(&ind), bg_count(&ind2));
  // (a) one pass writes the count, the next consumes it with `k` indirect dispatches
  for k in [1u32, 4, 16] {
    let m = n / 8;
    let t = gpu.timed(|| {
      let mut enc = gpu.device.create_command_encoder(&Default::default());
      for _ in 0..m {
        {
          let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&count);
          pass.set_bind_group(0, &bgc1, &[]);
          pass.dispatch_workgroups(1, 1, 1);
        }
        {
          let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&main);
          pass.set_bind_group(0, &bg_main, &[]);
          for _ in 0..k {
            pass.dispatch_workgroups_indirect(&ind, 0);
          }
        }
      }
      gpu.queue.submit(Some(enc.finish()));
    });
    println!("  count pass + pass with {k:>2} indirect dispatches: {:>8.2} us", t * 1e6 / m as f64);
  }
  // (b) a miss-path wave with two dependent counts: probe -> M; claim(M) -> J; 13 x work(J)
  {
    let m = n / 8;
    let t = gpu.timed(|| {
      let mut enc = gpu.device.create_command_encoder(&Default::default());
      for _ in 0..m {
        {
          let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&main);
          pass.set_bind_group(0, &bg_main, &[]);
          pass.dispatch_workgroups(4, 1, 1); // probe
          pass.set_pipeline(&count);
          pass.set_bind_group(0, &bgc1, &[]);
          pass.dispatch_workgroups(1, 1, 1); // M
        }
        {
          let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&main);
          pass.set_bind_group(0, &bg_main, &[]);
          pass.dispatch_workgroups_indirect(&ind, 0); // claim over M
          pass.set_pipeline(&count);
          pass.set_bind_group(0, &bgc2, &[]);
          pass.dispatch_workgroups(1, 1, 1); // J
        }
        {
          let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&main);
          pass.set_bind_group(0, &bg_main, &[]);
          for _ in 0..12 {
            pass.dispatch_workgroups_indirect(&ind2, 0); // expand, evaluate x8, intern, commit, publish
          }
          pass.dispatch_workgroups_indirect(&ind, 0); // re-probe the M missed positions
        }
      }
      gpu.queue.submit(Some(enc.finish()));
    });
    println!("  whole miss-path wave, 3 passes, 16 dispatches (14 indirect): {:>8.2} us", t * 1e6 / m as f64);
  }
  // (c) the same wave with no indirect dispatch: one pass, 16 chained fixed-size dispatches
  let skip = gpu.pipeline("skip", SKIP_WGSL, "skip");
  let live = gpu.storage_init("live", bytemuck::cast_slice(&[256u32, 0, 0, 0]));
  let bg_skip = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: None,
    layout: &skip.get_bind_group_layout(0),
    entries: &[
      wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
      wgpu::BindGroupEntry { binding: 1, resource: live.as_entire_binding() },
    ],
  });
  if indirect_validation {
    for (gx, gy) in [(4u32, 1u32), (64, 1), (1024, 1), (16384, 1), (65535, 1), (65535, 16)] {
      let reps = (n as u64 * 4 / (gx as u64 * gy as u64).max(64)).clamp(8, n as u64) as u32;
      let t = gpu.timed(|| {
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
          let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&skip);
          pass.set_bind_group(0, &bg_skip, &[]);
          for _ in 0..reps {
            pass.dispatch_workgroups(gx, gy, 1);
          }
        }
        gpu.queue.submit(Some(enc.finish()));
      });
      let groups = gx as f64 * gy as f64;
      println!(
        "  fixed dispatch of {:>8} workgroups (64 threads each), 256 invocations live: {:>9.2} us = {:.2} ns per idle workgroup",
        groups,
        t * 1e6 / reps as f64,
        t * 1e9 / reps as f64 / groups
      );
    }
    // a wave of 16 such dispatches, each sized for 4096 misses (64 workgroups)
    let m = n / 8;
    let t = gpu.timed(|| {
      let mut enc = gpu.device.create_command_encoder(&Default::default());
      {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&skip);
        pass.set_bind_group(0, &bg_skip, &[]);
        for _ in 0..m {
          for _ in 0..16 {
            pass.dispatch_workgroups(64, 1, 1);
          }
        }
      }
      gpu.queue.submit(Some(enc.finish()));
    });
    println!("  whole miss-path wave, 1 pass, 16 fixed dispatches sized for 4096 misses:  {:>8.2} us", t * 1e6 / m as f64);
  }
}

pub fn latency(args: &[String]) {
  let gpu = Gpu::new(opt(args, "--backend").as_deref());
  println!("GPU: {}", gpu.describe());
  let n: u32 = opt_or(args, "--n", 20000);
  let buf = gpu.storage_init("buf", bytemuck::cast_slice(&vec![1u32; 4096]));
  let ind = gpu.device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("indirect"),
    size: 16,
    usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });
  gpu.queue.write_buffer(&ind, 0, bytemuck::cast_slice(&[1u32, 1, 1, 0]));
  let main = gpu.pipeline("tiny", TINY_WGSL, "main");
  let count = gpu.pipeline("count", TINY_WGSL, "count");
  let bg_main = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: None,
    layout: &main.get_bind_group_layout(0),
    entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }],
  });
  let bg_count = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
    label: None,
    layout: &count.get_bind_group_layout(0),
    entries: &[
      wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() },
      wgpu::BindGroupEntry { binding: 1, resource: ind.as_entire_binding() },
    ],
  });
  // (1) n tiny dispatches chained in one pass, one submit
  let t = gpu.timed(|| {
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&main);
      for _ in 0..n {
        pass.set_bind_group(0, &bg_main, &[]);
        pass.dispatch_workgroups(4, 1, 1);
      }
    }
    gpu.queue.submit(Some(enc.finish()));
  });
  println!("  chained dispatch, one pass:            {:>8.2} us each", t * 1e6 / n as f64);
  // (1b) the same commands recorded and finished on the CPU but never submitted: the host-side
  // share of (1); the GPU-side share is the difference
  let t_rec = {
    let t0 = Instant::now();
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&main);
      for _ in 0..n {
        pass.set_bind_group(0, &bg_main, &[]);
        pass.dispatch_workgroups(4, 1, 1);
      }
    }
    let cb = enc.finish();
    let dt = t0.elapsed().as_secs_f64();
    drop(cb);
    dt
  };
  println!(
    "    of which recording on the CPU:       {:>8.2} us each (GPU side {:.2} us)",
    t_rec * 1e6 / n as f64,
    (t - t_rec) * 1e6 / n as f64
  );
  // (2) one pass per dispatch, one submit
  let t = gpu.timed(|| {
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    for _ in 0..n {
      let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&main);
      pass.set_bind_group(0, &bg_main, &[]);
      pass.dispatch_workgroups(4, 1, 1);
    }
    gpu.queue.submit(Some(enc.finish()));
  });
  println!("  one pass per dispatch, one submit:     {:>8.2} us each", t * 1e6 / n as f64);
  // (3) data-dependent chain without the CPU: count pass writes the indirect args, then an
  //     indirect dispatch consumes them (each needs its own pass: the buffer changes usage)
  let m = n / 4;
  let t = gpu.timed(|| {
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    for _ in 0..m {
      {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&count);
        pass.set_bind_group(0, &bg_count, &[]);
        pass.dispatch_workgroups(1, 1, 1);
      }
      {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&main);
        pass.set_bind_group(0, &bg_main, &[]);
        pass.dispatch_workgroups_indirect(&ind, 0);
      }
    }
    gpu.queue.submit(Some(enc.finish()));
  });
  println!("  count pass + indirect dispatch pair:   {:>8.2} us per pair", t * 1e6 / m as f64);
  // (4) submit per dispatch, no readback
  let m = (n / 10).max(100);
  let t = gpu.timed(|| {
    for _ in 0..m {
      let mut enc = gpu.device.create_command_encoder(&Default::default());
      {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&main);
        pass.set_bind_group(0, &bg_main, &[]);
        pass.dispatch_workgroups(4, 1, 1);
      }
      gpu.queue.submit(Some(enc.finish()));
    }
  });
  println!("  submit per dispatch (no readback):     {:>8.2} us each", t * 1e6 / m as f64);
  // (5) full CPU round trip: dispatch, copy 16 bytes, map, read
  let staging = gpu.device.create_buffer(&wgpu::BufferDescriptor {
    label: Some("staging"),
    size: 16,
    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });
  let m = (n / 20).max(100);
  let t = gpu.timed(|| {
    for _ in 0..m {
      let mut enc = gpu.device.create_command_encoder(&Default::default());
      {
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&main);
        pass.set_bind_group(0, &bg_main, &[]);
        pass.dispatch_workgroups(4, 1, 1);
      }
      enc.copy_buffer_to_buffer(&buf, 0, &staging, 0, 16);
      gpu.queue.submit(Some(enc.finish()));
      let slice = staging.slice(..);
      slice.map_async(wgpu::MapMode::Read, |r| r.expect("map"));
      gpu.wait();
      let _v = slice.get_mapped_range().expect("range")[0];
      staging.unmap();
    }
  });
  println!("  dispatch + 16-byte readback round trip: {:>7.2} us each", t * 1e6 / m as f64);
  drop(gpu);
  data_dependent(args, true);
  data_dependent(args, false);
}
