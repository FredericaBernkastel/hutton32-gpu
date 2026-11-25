//! `gpu-sparse`: a large sparse universe (scattered, phase-shifted copies of one periodic pattern)
//! run three ways on the GPU hit path, and exported as a macrocell so that bgolly and the CPU
//! reference engine can be timed on the identical universe.
//!
//!   * flat sweep        : every tile of the universe is probed each sweep (work ~ area)
//!   * active-block sweep: only blocks that contain something are dispatched (work ~ content)

use crate::{cmd_gpu::build_table, cmd_stats, cmd_stats::Loaded, flag, flat::Flat, gpu::Gpu, hl, hl::Engine, opt, opt_or};
use rustc_hash::FxHashMap;
use std::{io::Write as _, time::Instant};

const WGSL: &str = r#"
struct P { sw: u32, sh: u32, dw: u32, dh: u32, off: i32, mask: u32, empty: u32, rw: u32, rh: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> table: array<u32>;
@group(0) @binding(4) var<storage, read_write> misses: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read> rects: array<u32>;

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
fn step_tile(dx: u32, dy: u32) {
  let x = i32(dx) + p.off;
  let y = i32(dy) + p.off;
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
  dst[dy * p.dw + dx] = val;
}
// every tile of the universe
@compute @workgroup_size(16, 16) fn flat(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.dw || id.y >= p.dh) { return; }
  step_tile(id.x, id.y);
}
// only the listed blocks: z = block index, (x, y) = tile inside the block
@compute @workgroup_size(8, 8, 1) fn blocks(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= p.rw || id.y >= p.rh) { return; }
  step_tile(rects[id.z * 2u] + id.x, rects[id.z * 2u + 1u] + id.y);
}
"#;

/// Write a Golly macrocell file for the node `(root, level)`.
pub fn write_mc(e: &mut Engine, root: u32, level: usize, rule: &str, path: &std::path::Path) -> usize {
  let empties: Vec<u32> = (0..=level).map(|l| if l >= e.a { e.empty(l) } else { hl::NONE }).collect();
  let mut lines: Vec<String> = Vec::new();
  let mut raw_memo: FxHashMap<Vec<u8>, usize> = Default::default();
  let mut node_memo: FxHashMap<(usize, u32), usize> = Default::default();
  // raw square of side 2^d (d >= 1) -> line index (0 = empty)
  fn emit_raw(cells: &[u8], side: usize, lines: &mut Vec<String>, memo: &mut FxHashMap<Vec<u8>, usize>) -> usize {
    if cells.iter().all(|&c| c == 0) {
      return 0;
    }
    if let Some(&i) = memo.get(cells) {
      return i;
    }
    let line = if side == 2 {
      format!("1 {} {} {} {}", cells[0], cells[1], cells[2], cells[3])
    } else {
      let h = side / 2;
      let mut idx = [0usize; 4];
      for q in 0..4 {
        let (ox, oy) = ((q & 1) * h, (q >> 1) * h);
        let mut sub = vec![0u8; h * h];
        for y in 0..h {
          sub[y * h..(y + 1) * h].copy_from_slice(&cells[(oy + y) * side + ox..(oy + y) * side + ox + h]);
        }
        idx[q] = emit_raw(&sub, h, lines, memo);
      }
      format!("{} {} {} {} {}", side.trailing_zeros(), idx[0], idx[1], idx[2], idx[3])
    };
    lines.push(line);
    memo.insert(cells.to_vec(), lines.len());
    lines.len()
  }
  fn emit(
    e: &Engine,
    id: u32,
    l: usize,
    empties: &[u32],
    lines: &mut Vec<String>,
    raw: &mut FxHashMap<Vec<u8>, usize>,
    memo: &mut FxHashMap<(usize, u32), usize>,
  ) -> usize {
    if id == empties[l] {
      return 0;
    }
    if let Some(&i) = memo.get(&(l, id)) {
      return i;
    }
    let i = if l == e.a {
      emit_raw(e.tile_cells(id), e.tile_side, lines, raw)
    } else {
      let ch = e.levels[l].nodes[id as usize];
      let c: Vec<usize> = ch.iter().map(|&c| emit(e, c, l - 1, empties, lines, raw, memo)).collect();
      lines.push(format!("{} {} {} {} {}", l, c[0], c[1], c[2], c[3]));
      lines.len()
    };
    memo.insert((l, id), i);
    i
  }
  emit(e, root, level, &empties, &mut lines, &mut raw_memo, &mut node_memo);
  let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
  writeln!(f, "[M2] (ca-research)\n#R {rule}").unwrap();
  for l in &lines {
    writeln!(f, "{l}").unwrap();
  }
  lines.len()
}

/// Quadtree node covering tiles `[x0, x0+2^(l-lt)) x [y0, ...)` of a positional id grid.
fn build_node(e: &mut Engine, ids: &[u32], w: usize, h: usize, x0: usize, y0: usize, l: usize, lt: usize, occupied: &dyn Fn(usize, usize, usize) -> bool) -> u32 {
  let span = 1usize << (l - lt);
  if x0 >= w || y0 >= h || !occupied(x0, y0, span) {
    return e.empty(l);
  }
  if l == lt {
    return ids[y0 * w + x0];
  }
  let hs = span / 2;
  let ch = [
    build_node(e, ids, w, h, x0, y0, l - 1, lt, occupied),
    build_node(e, ids, w, h, x0 + hs, y0, l - 1, lt, occupied),
    build_node(e, ids, w, h, x0, y0 + hs, l - 1, lt, occupied),
    build_node(e, ids, w, h, x0 + hs, y0 + hs, l - 1, lt, occupied),
  ];
  e.intern(l, ch)
}

pub fn run(args: &[String]) {
  let path: std::path::PathBuf = opt(args, "--pattern").expect("--pattern").into();
  let (loaded, file_rule) = cmd_stats::load_pattern(&path);
  let spec = cmd_stats::variant_for(args, &file_rule);
  let lut = crate::load_lut(&spec);
  let a: usize = opt_or(args, "--a", 2);
  let lt: usize = opt_or(args, "--lt", 4);
  let reps: usize = opt_or(args, "--reps", 48);
  let pitch: usize = opt_or(args, "--pitch", 8);
  let cycles: u64 = opt_or(args, "--cycles", 1024);
  let max_learn: u64 = opt_or(args, "--max-learn-gens", 1 << 20);
  let s = 1u64 << lt;

  // ---- learn one period of the single pattern on the CPU ----
  let t0 = Instant::now();
  let mut e = Engine::new(&lut, a, (lt - 1).max(a));
  let mut f = match &loaded {
    Loaded::Dense(g) => Flat::from_grid(&mut e, g, lt, 2),
    Loaded::Mc(m) => {
      let (root, level) = e.from_mc(m);
      let (root, level, _) = e.popzeros(root, level);
      let ids = e.ids_at_level(root, level, lt);
      Flat::from_ids(&mut e, &ids, 1 << (level - lt), lt, 2)
    }
  };
  let (nx, ny) = (f.nx, f.ny);
  let mut seen: FxHashMap<Vec<u32>, u64> = Default::default();
  seen.insert(f.a_ids.clone(), 0);
  let (mut cyc, mut period) = (0u64, 0u64);
  while cyc * s < max_learn {
    f.sweep(&mut e);
    f.sweep(&mut e);
    cyc += 1;
    if let Some(&prev) = seen.get(&f.a_ids) {
      period = cyc - prev;
      break;
    }
    seen.insert(f.a_ids.clone(), cyc);
  }
  assert!(period > 0, "pattern did not become periodic within --max-learn-gens");
  drop(seen);
  let mut a_phase: Vec<Vec<u32>> = Vec::with_capacity(period as usize);
  for _ in 0..period {
    a_phase.push(f.a_ids.clone());
    f.sweep(&mut e);
    f.sweep(&mut e);
  }
  assert_eq!(f.a_ids, a_phase[0]);
  let empty = e.empty(lt);
  let top = &e.levels[lt + 1];
  let n_entries = top.res.iter().filter(|&&r| r != hl::NONE).count();
  let (table, mask, _) = build_table(
    top.nodes.iter().zip(&top.res).filter(|(_, &r)| r != hl::NONE).map(|(k, &r)| (*k, r)),
    n_entries,
    10,
  );
  println!(
    "pattern {} rule {}: tile {s}x{s}, {nx}x{ny} tiles, period {} gens, learned in {:.1}s; memo {n_entries} entries ({} MiB table)",
    path.file_name().unwrap().to_string_lossy(),
    spec.name(),
    period * s,
    t0.elapsed().as_secs_f64(),
    table.len() * 4 >> 20
  );

  // ---- sparse universe: one copy per (pitch*nx) x (pitch*ny) block ----
  let (bx, by) = (pitch * nx, pitch * ny);
  let (bw, bh) = (reps * bx, reps * by);
  let phase_of = |cx: usize, cy: usize| -> u64 { ((cx as u64 * 2654435761 + cy as u64 * 40503 + 12345) >> 3) % period };
  let fill = |shift: u64| -> Vec<u32> {
    let mut big = vec![empty; bw * bh];
    for cy in 0..reps {
      for cx in 0..reps {
        let src = &a_phase[((phase_of(cx, cy) + shift) % period) as usize];
        for y in 0..ny {
          let d = (cy * by + y) * bw + cx * bx;
          big[d..d + nx].copy_from_slice(&src[y * nx..(y + 1) * nx]);
        }
      }
    }
    big
  };
  let big = fill(0);
  let cells = (bw as u64 * s) * (bh as u64 * s);
  let live_tiles = big.iter().filter(|&&t| t != empty).count();
  println!(
    "universe: {reps}x{reps} copies at pitch {pitch} = {bw}x{bh} tiles = {}x{} cells ({:.3e}); {:.2}% of tiles non-empty",
    bw as u64 * s,
    bh as u64 * s,
    cells as f64,
    100.0 * live_tiles as f64 / (bw * bh) as f64
  );

  // ---- export for bgolly / the CPU engine, with the expected final fingerprint ----
  if let Some(out) = opt(args, "--export-mc") {
    let t0 = Instant::now();
    let mut top_level = lt;
    while (1usize << (top_level - lt)) < bw.max(bh) {
      top_level += 1;
    }
    let occupied = |x0: usize, y0: usize, span: usize| -> bool {
      // a block is occupied only in its first nx x ny tiles
      let (x1, y1) = ((x0 + span).min(bw), (y0 + span).min(bh));
      let hit = |lo: usize, hi: usize, pitch: usize, len: usize| -> bool {
        // does [lo, hi) intersect any [k*pitch, k*pitch+len)?
        let k = lo / pitch;
        lo < k * pitch + len || (k + 1) * pitch < hi
      };
      hit(x0, x1, bx, nx) && hit(y0, y1, by, ny)
    };
    let root = build_node(&mut e, &big, bw, bh, 0, 0, top_level, lt, &occupied);
    let rule_name = if file_rule.is_empty() { "Hutton32b".to_string() } else { file_rule.clone() };
    let n = write_mc(&mut e, root, top_level, &rule_name, std::path::Path::new(&out));
    let fin = fill(cycles);
    let root_fin = build_node(&mut e, &fin, bw, bh, 0, 0, top_level, lt, &occupied);
    let fp0 = hl::fingerprint(|v| e.for_each_cell(root, top_level, 0, 0, &mut |x, y, c| v(x, y, c)));
    let fp1 = hl::fingerprint(|v| e.for_each_cell(root_fin, top_level, 0, 0, &mut |x, y, c| v(x, y, c)));
    println!(
      "exported {out} ({n} macrocell nodes, {:.1}s): population {} bbox {}x{}",
      t0.elapsed().as_secs_f64(),
      fp0.population,
      fp0.w,
      fp0.h
    );
    println!("expected after {} gens: population {} bbox {}x{} hash {:016x}", cycles * s, fp1.population, fp1.w, fp1.h, fp1.hash);
  }
  if flag(args, "--no-gpu") {
    return;
  }

  // ---- GPU ----
  let gpu = Gpu::new(opt(args, "--backend").as_deref());
  println!("GPU: {}", gpu.describe());
  let rects: Vec<u32> = (0..reps).flat_map(|cy| (0..reps).flat_map(move |cx| [(cx * bx) as u32, (cy * by) as u32])).collect();
  let buf_t = gpu.storage_init("table", bytemuck::cast_slice(&table));
  let buf_r = gpu.storage_init("rects", bytemuck::cast_slice(&rects));
  let p_flat = gpu.pipeline("flat", WGSL, "flat");
  let p_blocks = gpu.pipeline("blocks", WGSL, "blocks");
  let uni = |words: [u32; 12]| {
    let b = gpu.device.create_buffer(&wgpu::BufferDescriptor {
      label: None,
      size: 48,
      usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    gpu.queue.write_buffer(&b, 0, bytemuck::cast_slice(&words));
    b
  };
  for mode in ["flat", "active blocks"] {
    let sparse = mode != "flat";
    let buf_a = gpu.storage_init("ids_a", bytemuck::cast_slice(&big));
    let buf_b = gpu.storage_init("ids_b", bytemuck::cast_slice(&vec![empty; (bw + 1) * (bh + 1)]));
    let buf_m = gpu.storage_init("misses", &[0u8; 16]);
    let u_ab = uni([bw as u32, bh as u32, bw as u32 + 1, bh as u32 + 1, (-1i32) as u32, mask, empty, nx as u32 + 1, ny as u32 + 1, 0, 0, 0]);
    let u_ba = uni([bw as u32 + 1, bh as u32 + 1, bw as u32, bh as u32, 0, mask, empty, nx as u32, ny as u32, 0, 0, 0]);
    let pl = if sparse { &p_blocks } else { &p_flat };
    let mk = |u: &wgpu::Buffer, sbuf: &wgpu::Buffer, d: &wgpu::Buffer| {
      let mut entries = vec![
        wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: sbuf.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: d.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 3, resource: buf_t.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 4, resource: buf_m.as_entire_binding() },
      ];
      if sparse {
        entries.push(wgpu::BindGroupEntry { binding: 5, resource: buf_r.as_entire_binding() });
      }
      gpu.device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pl.get_bind_group_layout(0), entries: &entries })
    };
    let (ab, ba) = (mk(&u_ab, &buf_a, &buf_b), mk(&u_ba, &buf_b, &buf_a));
    let per_submit: u64 = opt_or(args, "--per-submit", if sparse { 256 } else { 8 });
    let n_rects = (reps * reps) as u32;
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
          pass.set_pipeline(pl);
          for _ in 0..n {
            pass.set_bind_group(0, &ab, &[]);
            if sparse {
              pass.dispatch_workgroups((nx as u32 + 1).div_ceil(8), (ny as u32 + 1).div_ceil(8), n_rects);
            } else {
              pass.dispatch_workgroups((bw as u32 + 1).div_ceil(16), (bh as u32 + 1).div_ceil(16), 1);
            }
            pass.set_bind_group(0, &ba, &[]);
            if sparse {
              pass.dispatch_workgroups((nx as u32).div_ceil(8), (ny as u32).div_ceil(8), n_rects);
            } else {
              pass.dispatch_workgroups((bw as u32).div_ceil(16), (bh as u32).div_ceil(16), 1);
            }
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
    let misses = gpu.read_u32(&buf_m, 0, 1)[0];
    let out = gpu.read_u32(&buf_a, 0, (bw * bh) as u64);
    let want = fill(cycles);
    let bad = out.iter().zip(&want).filter(|(x, y)| x != y).count();
    println!(
      "  {mode:<14}: {gens} gens in {secs:.3}s = {:.1} gens/s = {:.3e} cell-updates/s (universe area) | verify: {misses} misses, {bad} mismatching tiles of {}{spread}",
      gens as f64 / secs,
      cells as f64 * gens as f64 / secs,
      bw * bh
    );
  }
}
