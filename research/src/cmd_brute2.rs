//! `gpu-brute2`: stress the per-cell kernel on hostile content and test the class-compressed LUT.
//!
//! The plain kernel indexes a 32 MiB table with a 25-bit neighbourhood code. Structured patterns
//! touch a tiny part of it, so it behaves as if cached; random content does not. Each neighbour
//! position only distinguishes 15 classes of states (measured by `lut`), so the same function fits
//! a 21-bit index = 2 MiB table, which always fits the GPU's L2 cache.

use crate::{cmd_stats, flag, gpu::Gpu, opt, opt_or, pattern::Grid, rule::LUT_SIZE, sim::Sim};

/// For each neighbour position (w, s, e, n) partition the 32 states into classes that the rule
/// cannot tell apart at that position. Returns `(class[pos][state], representatives[pos])`.
pub fn neighbour_classes(lut: &[u8]) -> ([[u8; 32]; 4], [Vec<u32>; 4]) {
  let mut class = [[0u8; 32]; 4];
  let mut reps: [Vec<u32>; 4] = Default::default();
  for pos in 0..4 {
    let shift = pos as u32 * 5;
    for v in 0..32u32 {
      let mut found = None;
      'rep: for (ci, &r) in reps[pos].iter().enumerate() {
        for rest in 0..(1u32 << 20) {
          let lo = rest & ((1 << shift) - 1);
          let hi = (rest >> shift) << (shift + 5);
          if lut[(hi | lo | v << shift) as usize] != lut[(hi | lo | r << shift) as usize] {
            continue 'rep;
          }
        }
        found = Some(ci);
        break;
      }
      class[pos][v as usize] = match found {
        Some(ci) => ci as u8,
        None => {
          reps[pos].push(v);
          (reps[pos].len() - 1) as u8
        }
      };
    }
    assert!(reps[pos].len() <= 16, "position {pos} needs more than 16 classes");
  }
  (class, reps)
}

/// 2^21-entry table indexed by `cw | cs<<4 | ce<<8 | cn<<12 | c<<16`.
pub fn compact_lut(lut: &[u8], reps: &[Vec<u32>; 4]) -> Vec<u8> {
  let mut out = vec![0u8; 1 << 21];
  for c in 0..32u32 {
    for cn in 0..reps[3].len() as u32 {
      for ce in 0..reps[2].len() as u32 {
        for cs in 0..reps[1].len() as u32 {
          for cw in 0..reps[0].len() as u32 {
            let full = reps[0][cw as usize] | reps[1][cs as usize] << 5 | reps[2][ce as usize] << 10 | reps[3][cn as usize] << 15 | c << 20;
            out[(cw | cs << 4 | ce << 8 | cn << 12 | c << 16) as usize] = lut[full as usize];
          }
        }
      }
    }
  }
  out
}

fn shader(compact: bool, class: &[[u8; 32]; 4], wgx: u32, wgy: u32) -> String {
  let mut s = String::from(
    r#"
struct P { ww: u32, h: u32, a: u32, b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@group(0) @binding(3) var<storage, read> lut: array<u32>;
fn lut_get(i: u32) -> u32 { return (lut[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu; }
"#,
  );
  if compact {
    // 32 nibbles per position packed into four words
    for (pos, name) in ["CW", "CS", "CE", "CN"].iter().enumerate() {
      let words: Vec<String> = (0..4)
        .map(|w| {
          let mut v = 0u32;
          for k in 0..8 {
            v |= (class[pos][w * 8 + k] as u32) << (k * 4);
          }
          format!("{v}u")
        })
        .collect();
      s += &format!("const {name} = array<u32, 4>({});\n", words.join(", "));
    }
    s += "fn cw(x: u32) -> u32 { var t = CW; return (t[x >> 3u] >> ((x & 7u) << 2u)) & 15u; }\n";
    s += "fn cs(x: u32) -> u32 { var t = CS; return (t[x >> 3u] >> ((x & 7u) << 2u)) & 15u; }\n";
    s += "fn ce(x: u32) -> u32 { var t = CE; return (t[x >> 3u] >> ((x & 7u) << 2u)) & 15u; }\n";
    s += "fn cn(x: u32) -> u32 { var t = CN; return (t[x >> 3u] >> ((x & 7u) << 2u)) & 15u; }\n";
  }
  s += &format!("@compute @workgroup_size({wgx}, {wgy}) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{\n");
  s += r#"
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
"#;
  if compact {
    s += "    let idx = cw((wv >> sh) & 31u) | (cs((dn >> sh) & 31u) << 4u) | (ce((ev >> sh) & 31u) << 8u) | (cn((up >> sh) & 31u) << 12u) | (((c >> sh) & 31u) << 16u);\n";
  } else {
    s += "    let idx = ((wv >> sh) & 31u) | (((dn >> sh) & 31u) << 5u) | (((ev >> sh) & 31u) << 10u) | (((up >> sh) & 31u) << 15u) | (((c >> sh) & 31u) << 20u);\n";
  }
  s += "    out |= lut_get(idx) << sh;\n  }\n  dst[i] = out;\n}\n";
  s
}

fn rnd(x0: u32) -> u32 {
  let mut x = x0;
  x ^= x >> 16;
  x = x.wrapping_mul(0x7FEB352D);
  x ^= x >> 15;
  x = x.wrapping_mul(0x846CA68B);
  x ^= x >> 16;
  x
}

pub fn run(args: &[String]) {
  let side: usize = opt_or(args, "--side", 8192);
  let gens: u64 = opt_or(args, "--gens", 256) & !1;
  let (wgx, wgy): (u32, u32) = (opt_or(args, "--wgx", 8), opt_or(args, "--wgy", 8));
  let same_src = flag(args, "--same-src");
  let spec = cmd_stats::variant_for(args, "hutton32b");
  let lut = crate::load_lut(&spec);
  assert_eq!(lut.len(), LUT_SIZE);
  // content: uniformly random states, or a tiled pattern
  let (g, what) = if let Some(p) = opt(args, "--pattern") {
    let (loaded, _) = cmd_stats::load_pattern(std::path::Path::new(&p));
    let cmd_stats::Loaded::Dense(pat) = loaded else { panic!("dense pattern required") };
    let (cw, ch) = (pat.w + 8, pat.h + 8);
    let (rx, ry) = (side / cw, side / ch);
    let mut g = Grid::new(cw * rx, ch * ry);
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
  println!("rule {} | content: {what} | {}x{} = {} cells | {}", spec.name(), g.w, g.h, g.w * g.h, if same_src { "same source every generation (content never decays)" } else { "ping-pong" });
  let (class, reps) = neighbour_classes(&lut);
  println!("neighbour classes per position (w,s,e,n): {:?}", reps.iter().map(|r| r.len()).collect::<Vec<_>>());
  let small = compact_lut(&lut, &reps);
  let ww = g.w.div_ceil(4) + 2;
  let h = g.h + 2;
  let mut words = vec![0u32; ww * h];
  for y in 0..g.h {
    for x in 0..g.w {
      words[(y + 1) * ww + 1 + x / 4] |= (g.cells[y * g.w + x] as u32) << ((x % 4) * 8);
    }
  }
  let uni = gpu.device.create_buffer(&wgpu::BufferDescriptor {
    label: None,
    size: 16,
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });
  gpu.queue.write_buffer(&uni, 0, bytemuck::cast_slice(&[ww as u32, h as u32, 0, 0]));
  for compact in [false, true] {
    let a = gpu.storage_init("a", bytemuck::cast_slice(&words));
    let b = gpu.storage_init("b", bytemuck::cast_slice(&vec![0u32; words.len()]));
    let lut_buf = gpu.storage_init("lut", if compact { &small } else { &lut });
    let pipeline = gpu.pipeline("brute2", &shader(compact, &class, wgx, wgy), "main");
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
    // correctness: `gens` ping-pong generations (or one generation when same_src) vs the CPU
    let check = if flag(args, "--verify") {
      let out = gpu.read_u32(if same_src { &b } else { &a }, 0, words.len() as u64);
      let mut sim = Sim::new(&g, &lut);
      for _ in 0..if same_src { 1 } else { gens } {
        sim.step();
      }
      let want = sim.grid();
      let mut bad = 0usize;
      for y in 0..g.h {
        for x in 0..g.w {
          let got = (out[(y + 1) * ww + 1 + x / 4] >> ((x % 4) * 8)) & 0xff;
          bad += (got != want.cells[y * g.w + x] as u32) as usize;
        }
      }
      format!(" | verify: {bad} mismatching cells")
    } else {
      String::new()
    };
    println!(
      "  {:<28} {gens} gens in {secs:.3}s = {:.3e} cell-updates/s{check}",
      if compact { "class-compressed LUT (2 MiB):" } else { "full LUT (32 MiB):" },
      (g.w * g.h) as f64 * gens as f64 / secs
    );
  }
}
