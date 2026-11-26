//! `gpu-life`: the brute-force engine for two-state outer-totalistic (Life-like) rules on the
//! Moore neighbourhood. One bit per cell, 32 cells per `u32`: an invocation forms its eight
//! neighbour words with shifts, adds them with a carry-save adder tree into a four-bit count and
//! decodes the rule's B and S sets from it. This is the counterpart for low state counts of the
//! 367-gate Hutton32b circuit (`gpu-bitslice`, `gpu-tile`).
//!
//! Plain kernel (`--k 1`): one generation per dispatch. Shared tile (`--k K`, K >= 2): a
//! workgroup loads a tile of TW words x TH rows plus a halo of K rows and one word on each side
//! into workgroup memory, steps it K generations there and writes the tile once, as `gpu-tile`.
//!
//! Layout: rows of `ww` words (one zero word on each side), `h + 2` rows (one zero row above and
//! below). Cell x of a row lives in word `1 + x / 32`, bit `x % 32`; bit i+1 is the east neighbour
//! of bit i. Cells outside the grid stay 0, as in the CPU reference (`sim::Sim`).
//!
//! `life-soup`: a random square of live cells, written as RLE, for Golly and for this command.

use crate::{
  ca::{self, Nbhd},
  cmd_stats, flag,
  gpu::Gpu,
  opt, opt_or,
  pattern::{self, Grid},
  rulespec::RuleSpec,
  sim::Sim,
};

/// The rule as a WGSL function of the centre word and its eight neighbour words.
pub fn life_fn(birth: u16, survive: u16) -> String {
  let eq = |k: u32| -> String {
    let lit = |bit: u32, name: &str| if k >> bit & 1 == 1 { name.to_string() } else { format!("~{name}") };
    format!("({} & {} & {} & {})", lit(0, "b0"), lit(1, "b1"), lit(2, "b2"), lit(3, "b3"))
  };
  let set = |m: u16| -> String {
    let v: Vec<String> = (0..=8).filter(|k| m >> k & 1 == 1).map(eq).collect();
    if v.is_empty() { "0u".into() } else { v.join(" | ") }
  };
  format!(
    r#"
fn life(c: u32, n: u32, s: u32, e: u32, w: u32, ne: u32, nw: u32, se: u32, sw: u32) -> u32 {{
  // carry-save adder tree: eight one-bit inputs -> count b3 b2 b1 b0
  let t1 = n ^ s;  let s1 = t1 ^ e;  let c1 = (n & s) | (t1 & e);
  let t2 = w ^ ne; let s2 = t2 ^ nw; let c2 = (w & ne) | (t2 & nw);
  let s3 = se ^ sw; let c3 = se & sw;
  let t4 = s1 ^ s2; let b0 = t4 ^ s3; let c4 = (s1 & s2) | (t4 & s3);
  let t5 = c1 ^ c2; let s5 = t5 ^ c3; let c5 = (c1 & c2) | (t5 & c3);
  let b1 = s5 ^ c4; let c6 = s5 & c4;
  let b2 = c5 ^ c6; let b3 = c5 & c6;
  let born = {born};
  let keep = {keep};
  return (c & keep) | (~c & born);
}}
"#,
    born = set(birth),
    keep = set(survive)
  )
}

const PLAIN: &str = r#"
struct P { ww: u32, h: u32, a: u32, b: u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
@compute @workgroup_size(WGX, WGY) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let x = id.x + 1u;
  let y = id.y + 1u;
  if (x >= p.ww - 1u || y >= p.h - 1u) { return; }
  let i = y * p.ww + x;
  let c = src[i]; let l = src[i - 1u]; let r = src[i + 1u];
  let u = src[i - p.ww]; let ul = src[i - p.ww - 1u]; let ur = src[i - p.ww + 1u];
  let d = src[i + p.ww]; let dl = src[i + p.ww - 1u]; let dr = src[i + p.ww + 1u];
  dst[i] = life(c, u, d, (c >> 1u) | (r << 31u), (c << 1u) | (l >> 31u),
                (u >> 1u) | (ur << 31u), (u << 1u) | (ul >> 31u), (d >> 1u) | (dr << 31u), (d << 1u) | (dl >> 31u));
}
"#;

/// Shared-tile kernel: one invocation per word of the tile and its halo.
fn tile_kernel(tw: u32, th: u32, k: u32) -> String {
  let (ew, eh) = (tw + 2, th + 2 * k);
  let n = ew * eh;
  let mut s = format!(
    r#"
struct P {{ ww: u32, h: u32, a: u32, b: u32 }};
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<u32>;
@group(0) @binding(2) var<storage, read_write> dst: array<u32>;
var<workgroup> sa: array<u32, {n}>;
var<workgroup> sb: array<u32, {n}>;
@compute @workgroup_size({ew}, {eh}) fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
  let lx = lid.x;
  let ly = lid.y;
  // local column 1 and local row K are the tile's first word and row
  let gx = i32(wid.x * {tw}u + lx);
  let gy = i32(wid.y * {th}u + ly) + 1 - {k};
  let inside = gx >= 1 && gx < i32(p.ww) - 1 && gy >= 1 && gy < i32(p.h) - 1;
  let gi = u32(max(gy, 0)) * p.ww + u32(max(gx, 0));
  let li = ly * {ew}u + lx;
  var v = 0u;
  if (inside) {{ v = src[gi]; }}
  sa[li] = v;
  workgroupBarrier();
"#
  );
  for g in 1..=k {
    let (from, to) = if g % 2 == 1 { ("sa", "sb") } else { ("sb", "sa") };
    s += &format!(
      r#"  {{
    var o = 0u;
    // cells outside the grid stay 0; rows nearer than g to the edge of the halo are no longer valid
    if (inside && ly >= {g}u && ly + {g}u < {eh}u) {{
      let c = {from}[li]; let u = {from}[li - {ew}u]; let d = {from}[li + {ew}u];
      var l = 0u; var ul = 0u; var dl = 0u; var r = 0u; var ur = 0u; var dr = 0u;
      if (lx > 0u) {{ l = {from}[li - 1u]; ul = {from}[li - {ew}u - 1u]; dl = {from}[li + {ew}u - 1u]; }}
      if (lx + 1u < {ew}u) {{ r = {from}[li + 1u]; ur = {from}[li - {ew}u + 1u]; dr = {from}[li + {ew}u + 1u]; }}
      o = life(c, u, d, (c >> 1u) | (r << 31u), (c << 1u) | (l >> 31u),
               (u >> 1u) | (ur << 31u), (u << 1u) | (ul >> 31u), (d >> 1u) | (dr << 31u), (d << 1u) | (dl >> 31u));
    }}
    {to}[li] = o;
  }}
  workgroupBarrier();
"#
    );
  }
  let fin = if k % 2 == 1 { "sb" } else { "sa" };
  s += &format!(
    r#"  if (inside && lx >= 1u && lx <= {tw}u && ly >= {k}u && ly < {last}u) {{
    dst[gi] = {fin}[li];
  }}
}}
"#,
    last = th + k
  );
  s
}

/// Pack one bit per cell. Returns (words, ww, rows).
pub fn pack_bits(g: &Grid) -> (Vec<u32>, usize, usize) {
  let ww = g.w.div_ceil(32) + 2;
  let h = g.h + 2;
  let mut words = vec![0u32; ww * h];
  for y in 0..g.h {
    for x in 0..g.w {
      if g.cells[y * g.w + x] != 0 {
        words[(y + 1) * ww + 1 + x / 32] |= 1 << (x % 32);
      }
    }
  }
  (words, ww, h)
}
pub fn unpack_bits(words: &[u32], ww: usize, w: usize, h: usize) -> Grid {
  let mut g = Grid::new(w, h);
  for y in 0..h {
    for x in 0..w {
      g.cells[y * w + x] = ((words[(y + 1) * ww + 1 + x / 32] >> (x % 32)) & 1) as u8;
    }
  }
  g
}

fn hash32(x0: u32) -> u32 {
  let mut x = x0;
  x ^= x >> 16;
  x = x.wrapping_mul(0x7FEB352D);
  x ^= x >> 15;
  x = x.wrapping_mul(0x846CA68B);
  x ^= x >> 16;
  x
}

/// A square of side `side` with each cell alive with probability `density`, from `seed`.
pub fn soup(side: usize, density: f64, seed: u32) -> Grid {
  let mut g = Grid::new(side, side);
  let thr = (density * 4294967296.0).min(4294967295.0) as u64;
  for (i, c) in g.cells.iter_mut().enumerate() {
    *c = ((hash32(hash32(i as u32) ^ seed.wrapping_mul(0x9E3779B9)) as u64) < thr) as u8;
  }
  g
}

/// `life-soup --side S [--density D] [--seed N] --out F`
pub fn soup_cmd(args: &[String]) {
  let side: usize = opt_or(args, "--side", 2048);
  let density: f64 = opt_or(args, "--density", 0.5);
  let seed: u32 = opt_or(args, "--seed", 1);
  let out = opt(args, "--out").expect("--out");
  let g = soup(side, density, seed);
  pattern::write_rle(std::path::Path::new(&out), &g, "B3/S23");
  println!("wrote {out}: {side}x{side} soup, density {density}, seed {seed}, population {}", g.population());
}

/// What a Life run starts from: the grid, a description of it, and the rule. Shared by `gpu-life`
/// and its CubeCL port `cubecl-life`, so that the two differ only on the GPU.
pub struct Setup {
  pub g: Grid,
  pub what: String,
  pub rule_name: String,
  pub birth: u16,
  pub survive: u16,
  pub rule: ca::Rule,
}

/// `--side N | --pattern P [--tiled | --margin M]`, `--rule R`: the grid and the rule.
pub fn setup(args: &[String]) -> Setup {
  let side: usize = opt_or(args, "--side", 4096);
  // content: random soup filling the grid, a pattern tiled over it, or a pattern centred with a
  // margin (`--margin M`, for patterns that grow, e.g. soups that emit gliders)
  let mut file_rule = String::new();
  let (g, what) = if let Some(p) = opt(args, "--pattern") {
    let (loaded, fr) = cmd_stats::load_pattern(std::path::Path::new(&p));
    file_rule = fr;
    let pat = match loaded {
      cmd_stats::Loaded::Dense(g) => g,
      cmd_stats::Loaded::Mc(m) => {
        let r = ca::Rule::dummy();
        let mut e = crate::hl::Engine::new(&r, 3, 3);
        let (root, level) = e.from_mc(&m);
        let (root, level, _) = e.popzeros(root, level);
        let full = e.expand(root, level);
        let (x0, y0, x1, y1) = full.bbox().expect("empty pattern");
        full.crop(x0, y0, x1, y1)
      }
    };
    let name = std::path::Path::new(&p).file_name().unwrap().to_string_lossy().into_owned();
    if flag(args, "--tiled") {
      let (cw, ch) = (pat.w + 8, pat.h + 8);
      let (rx, ry) = ((side / cw).max(1), (side / ch).max(1));
      let mut g = Grid::new((cw * rx).div_ceil(32) * 32, ch * ry);
      for j in 0..ry {
        for i in 0..rx {
          g.blit(&pat, i * cw + 4, j * ch + 4);
        }
      }
      (g, format!("{name} tiled {rx}x{ry}"))
    } else {
      let m: usize = opt_or(args, "--margin", 64);
      let w = (pat.w + 2 * m).div_ceil(32) * 32;
      let mut g = Grid::new(w, pat.h + 2 * m);
      g.blit(&pat, (w - pat.w) / 2, m);
      (g, format!("{name} with a margin of {m} cells"))
    }
  } else {
    let side = side.div_ceil(32) * 32;
    let density: f64 = opt_or(args, "--density", 0.5);
    (soup(side, density, opt_or(args, "--seed", 1)), format!("random soup, density {density}"))
  };
  let spec = cmd_stats::variant_for(args, if file_rule.is_empty() { "B3/S23" } else { &file_rule });
  let RuleSpec::LifeLike(birth, survive, Nbhd::Moore) = spec else { panic!("gpu-life runs Life-like rules on the Moore neighbourhood, not {}", spec.name()) };
  let rule = crate::load_lut(&spec);
  Setup { g, what, rule_name: spec.name(), birth, survive, rule }
}

/// After the timed runs: the final state's fingerprint (`--fingerprint`) and, with `--verify`, its
/// comparison with the table stepped on the CPU. `words` is the final buffer, laid out as `pack_bits`.
pub fn report(args: &[String], s: &Setup, words: &[u32], ww: usize, gens: u64) {
  let g = &s.g;
  let got = unpack_bits(words, ww, g.w, g.h);
  let fp = cmd_stats::grid_fingerprint(&got);
  println!("  final state after {gens} gens: population {} bbox {}x{} hash {:016x}", fp.population, fp.w, fp.h, fp.hash);
  if got.bbox().is_some_and(|(x0, y0, x1, y1)| x0 == 0 || y0 == 0 || x1 == got.w || y1 == got.h) {
    println!("  warning: live cells reached the edge of the grid; the result is the bounded grid's, not the plane's");
  }
  if flag(args, "--verify") {
    let mut sim = Sim::new(g, &s.rule);
    for _ in 0..gens {
      sim.step();
    }
    let bad = got.cells.iter().zip(&sim.grid().cells).filter(|(a, b)| a != b).count();
    println!("  verify vs the table on the CPU: {bad} mismatching cells");
  }
}

pub fn run(args: &[String]) {
  let k: u32 = opt_or(args, "--k", 1);
  let gens: u64 = opt_or::<u64>(args, "--gens", 256) / k as u64 * k as u64;
  let (wgx, wgy): (u32, u32) = (opt_or(args, "--wgx", 8), opt_or(args, "--wgy", 8));
  let tw: u32 = opt_or(args, "--tw", 6);
  let th: u32 = opt_or(args, "--th", 32u32.saturating_sub(2 * k).max(2));
  let s = setup(args);
  let (g, what, birth, survive) = (&s.g, &s.what, s.birth, s.survive);

  let gpu = Gpu::new(opt(args, "--backend").as_deref());
  println!("GPU: {}", gpu.describe());
  let (words, ww, h) = pack_bits(g);
  let uni = gpu.device.create_buffer(&wgpu::BufferDescriptor {
    label: None,
    size: 16,
    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    mapped_at_creation: false,
  });
  gpu.queue.write_buffer(&uni, 0, bytemuck::cast_slice(&[ww as u32, h as u32, 0, 0]));
  let body = if k <= 1 {
    PLAIN.replace("WGX", &wgx.to_string()).replace("WGY", &wgy.to_string())
  } else {
    assert!(k <= 32, "the one-word halo allows at most 32 generations per dispatch");
    tile_kernel(tw, th, k)
  };
  let src = format!("{}\n{}", life_fn(birth, survive), body);
  // `--trusted`: without naga's runtime checks, as CubeCL compiles its WGSL
  let pipeline = gpu.pipeline_with("life", &src, "main", flag(args, "--trusted"));
  let a = gpu.storage_init("a", bytemuck::cast_slice(&words));
  let b = gpu.storage_init("b", bytemuck::cast_slice(&vec![0u32; words.len()]));
  let mk = |s: &wgpu::Buffer, d: &wgpu::Buffer| {
    gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
      label: None,
      layout: &pipeline.get_bind_group_layout(0),
      entries: &[
        wgpu::BindGroupEntry { binding: 0, resource: uni.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 1, resource: s.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 2, resource: d.as_entire_binding() },
      ],
    })
  };
  let (ab, ba) = (mk(&a, &b), mk(&b, &a));
  let (dx, dy) = if k <= 1 {
    ((ww as u32 - 2).div_ceil(wgx), (h as u32 - 2).div_ceil(wgy))
  } else {
    ((ww as u32 - 2).div_ceil(tw), (h as u32 - 2).div_ceil(th))
  };
  let dispatches = gens / k.max(1) as u64;
  let per_submit: u64 = opt_or(args, "--per-submit", 4096);
  println!(
    "rule {} | content: {what} | {}x{} = {} cells | {} | {} MiB per buffer",
    s.rule_name,
    g.w,
    g.h,
    g.w * g.h,
    if k <= 1 { format!("plain kernel, {wgx}x{wgy} workgroups") } else { format!("shared tile, k = {k}, tiles of {} x {th} cells", tw * 32) },
    (words.len() * 4) >> 20
  );
  let all = crate::repeat_timed(args, || {
    gpu.queue.write_buffer(&a, 0, bytemuck::cast_slice(&words));
    gpu.timed(|| {
      let mut done = 0u64;
      while done < dispatches {
        let n = (dispatches - done).min(per_submit);
        let mut enc = gpu.device.create_command_encoder(&Default::default());
        {
          let mut pass = enc.begin_compute_pass(&Default::default());
          pass.set_pipeline(&pipeline);
          for i in done..done + n {
            pass.set_bind_group(0, if i % 2 == 0 { &ab } else { &ba }, &[]);
            pass.dispatch_workgroups(dx, dy, 1);
          }
        }
        gpu.queue.submit(Some(enc.finish()));
        done += n;
      }
    })
  });
  let (secs, spread) = crate::median_and_spread(&all, gens as f64, "gens/s");
  println!(
    "  life kernel: {gens} gens in {secs:.4}s = {:.1} gens/s = {:.3e} cell-updates/s{spread}",
    gens as f64 / secs,
    (g.w * g.h) as f64 * gens as f64 / secs
  );
  if flag(args, "--verify") || flag(args, "--fingerprint") {
    let out = gpu.read_u32(if dispatches % 2 == 1 { &b } else { &a }, 0, words.len() as u64);
    report(args, &s, &out, ww, gens);
  }
}
