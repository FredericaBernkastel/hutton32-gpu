//! `cubecl-life`: `gpu-life` (`cmd_life.rs`) ported to CubeCL 0.12.0-pre.2, kernel for kernel.
//!
//! The same two kernels on the same bit layout: the plain kernel steps one generation per launch,
//! the shared-tile kernel K generations in workgroup memory. The rule's B and S sets and the tile
//! shape are comptime parameters: CubeCL compiles one kernel per rule and shape at first launch,
//! where `gpu-life` formats a WGSL string. Grid loading and verification are `cmd_life`'s.
//!
//! Build with `cargo build --release --features cubecl`. `--compiler wgsl|spirv` picks CubeCL's
//! shader compiler on wgpu (default: whichever it prefers on the device); `--checked` launches the
//! bounds-checked variant. `CUBECL_DEBUG_PLIRON=<dir>` dumps the generated shaders.

use crate::{cmd_life, flag, opt, opt_or};
use cubecl::prelude::*;
use cubecl::wgpu::{AutoCompiler, SpirvCompiler, WgpuDevice, WgpuDeviceKind, WgpuRuntime, WgslCompiler};
use cubecl_runtime::runtime::Runtime;

/// One literal of a minterm: `b` where bit `bit` of the count `k` is 1, `!b` where it is 0.
#[cube]
fn literal(b: u32, #[comptime] k: u32, #[comptime] bit: u32) -> u32 {
  if comptime!(k >> bit & 1 == 1) { b } else { !b }
}

/// The lanes whose four-bit count `b3 b2 b1 b0` is in `set` (bit k of `set`: count k), as an OR
/// of minterms. The loop and the tests run at comptime: only the counts in the set are emitted.
#[cube]
fn in_set(b0: u32, b1: u32, b2: u32, b3: u32, #[comptime] set: u32) -> u32 {
  let mut lanes = 0u32;
  #[unroll]
  for k in 0..9u32 {
    if comptime!(set >> k & 1 == 1) {
      lanes |= literal(b0, k, 0u32) & literal(b1, k, 1u32) & literal(b2, k, 2u32) & literal(b3, k, 3u32);
    }
  }
  lanes
}

/// The rule for 32 cells at once: the centre word and its eight neighbour words, already shifted.
#[cube]
#[allow(clippy::too_many_arguments)]
fn life(c: u32, n: u32, s: u32, e: u32, w: u32, ne: u32, nw: u32, se: u32, sw: u32, #[comptime] birth: u32, #[comptime] survive: u32) -> u32 {
  // carry-save adder tree: eight one-bit inputs -> count b3 b2 b1 b0
  let t1 = n ^ s;
  let s1 = t1 ^ e;
  let c1 = (n & s) | (t1 & e);
  let t2 = w ^ ne;
  let s2 = t2 ^ nw;
  let c2 = (w & ne) | (t2 & nw);
  let s3 = se ^ sw;
  let c3 = se & sw;
  let t4 = s1 ^ s2;
  let b0 = t4 ^ s3;
  let c4 = (s1 & s2) | (t4 & s3);
  let t5 = c1 ^ c2;
  let s5 = t5 ^ c3;
  let c5 = (c1 & c2) | (t5 & c3);
  let b1 = s5 ^ c4;
  let c6 = s5 & c4;
  let b2 = c5 ^ c6;
  let b3 = c5 & c6;
  (c & in_set(b0, b1, b2, b3, survive)) | (!c & in_set(b0, b1, b2, b3, birth))
}

/// The next centre word from the 3x3 words around it: bit i+1 is the east neighbour of bit i.
#[cube]
#[allow(clippy::too_many_arguments)]
fn next(c: u32, u: u32, d: u32, l: u32, r: u32, ul: u32, ur: u32, dl: u32, dr: u32, #[comptime] birth: u32, #[comptime] survive: u32) -> u32 {
  life(
    c,
    u,
    d,
    (c >> 1) | (r << 31),
    (c << 1) | (l >> 31),
    (u >> 1) | (ur << 31),
    (u << 1) | (ul >> 31),
    (d >> 1) | (dr << 31),
    (d << 1) | (dl >> 31),
    birth,
    survive,
  )
}

/// Plain kernel: one invocation per word of the grid's interior, one generation.
#[cube(launch, launch_unchecked)]
fn step(src: &[u32], dst: &mut [u32], ww: u32, h: u32, #[comptime] birth: u32, #[comptime] survive: u32) {
  let x = ABSOLUTE_POS_X + 1;
  let y = ABSOLUTE_POS_Y + 1;
  if x < ww - 1 && y < h - 1 {
    let row = ww as usize;
    let i = y as usize * row + x as usize;
    let (u, d) = (i - row, i + row);
    dst[i] = next(src[i], src[u], src[d], src[i - 1], src[i + 1], src[u - 1], src[u + 1], src[d - 1], src[d + 1], birth, survive);
  }
}

/// One generation of the shared tile, `from` to `to`. Cells outside the grid stay 0; rows nearer
/// than `g` to the edge of the halo are no longer valid after `g` generations.
#[cube]
#[allow(clippy::too_many_arguments)]
fn tile_generation(from: &[u32], to: &mut [u32], li: usize, lx: u32, ly: u32, inside: bool, #[comptime] g: u32, #[comptime] ew: u32, #[comptime] eh: u32, #[comptime] birth: u32, #[comptime] survive: u32) {
  let mut o = 0u32;
  if inside && ly >= g && ly + g < eh {
    let row = ew as usize;
    let (u, d) = (li - row, li + row);
    let mut l = 0u32;
    let mut ul = 0u32;
    let mut dl = 0u32;
    let mut r = 0u32;
    let mut ur = 0u32;
    let mut dr = 0u32;
    if lx > 0 {
      l = from[li - 1];
      ul = from[u - 1];
      dl = from[d - 1];
    }
    if lx + 1 < ew {
      r = from[li + 1];
      ur = from[u + 1];
      dr = from[d + 1];
    }
    o = next(from[li], from[u], from[d], l, r, ul, ur, dl, dr, birth, survive);
  }
  to[li] = o;
}

/// Shared-tile kernel: a cube of (tw + 2) x (th + 2k) units loads a tile of tw words x th rows with
/// a halo of one word and k rows, steps it k generations in shared memory and writes the tile.
#[cube(launch, launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn tile(src: &[u32], dst: &mut [u32], ww: u32, h: u32, #[comptime] tw: u32, #[comptime] th: u32, #[comptime] k: u32, #[comptime] birth: u32, #[comptime] survive: u32) {
  let ew = comptime!(tw + 2);
  let eh = comptime!(th + 2 * k);
  let mut sa = Shared::<[u32]>::new_slice(comptime!((ew * eh) as usize));
  let mut sb = Shared::<[u32]>::new_slice(comptime!((ew * eh) as usize));
  let (lx, ly) = (UNIT_POS_X, UNIT_POS_Y);
  // grid word gx, row gy + k: local column 1 and local row k are the tile's first word and row
  let gx = CUBE_POS_X * tw + lx;
  let gyk = CUBE_POS_Y * th + ly + 1;
  let inside = gx >= 1 && gx < ww - 1 && gyk > k && gyk < h - 1 + k;
  let gi = if inside { (gyk - k) as usize * ww as usize + gx as usize } else { 0usize };
  let li = (ly * ew + lx) as usize;
  let mut v = 0u32;
  if inside {
    v = src[gi];
  }
  sa[li] = v;
  sync_cube();
  #[unroll]
  for g in 1..comptime!(k + 1) {
    if comptime!(g % 2 == 1) {
      tile_generation(&sa, &mut sb, li, lx, ly, inside, g, ew, eh, birth, survive);
    } else {
      tile_generation(&sb, &mut sa, li, lx, ly, inside, g, ew, eh, birth, survive);
    }
    sync_cube();
  }
  if inside && lx >= 1 && lx <= tw && ly >= k && ly < th + k {
    dst[gi] = if comptime!(k % 2 == 1) { sb[li] } else { sa[li] };
  }
}

/// `cubecl-life`: the flags of `gpu-life`, plus `--compiler wgsl|spirv` and `--checked`.
pub fn run(args: &[String]) {
  let k: u32 = opt_or(args, "--k", 1);
  let gens: u64 = opt_or::<u64>(args, "--gens", 256) / k as u64 * k as u64;
  let (wgx, wgy): (u32, u32) = (opt_or(args, "--wgx", 8), opt_or(args, "--wgy", 8));
  let tw: u32 = opt_or(args, "--tw", 6);
  let th: u32 = opt_or(args, "--th", 32u32.saturating_sub(2 * k).max(2));
  let checked = flag(args, "--checked");
  let s = cmd_life::setup(args);
  let (birth, survive) = (s.birth as u32, s.survive as u32);

  let device = WgpuDevice::new(WgpuDeviceKind::DiscreteGpu(0));
  let client = match opt(args, "--compiler").as_deref() {
    Some("wgsl") => WgpuRuntime::<WgslCompiler>::client(&device),
    Some("spirv") => WgpuRuntime::<SpirvCompiler>::client(&device),
    None => WgpuRuntime::<AutoCompiler>::client(&device),
    Some(other) => panic!("unknown compiler {other}"),
  };
  println!("GPU: {} ({})", client.name(), opt(args, "--compiler").unwrap_or_else(|| "auto".into()));

  let (words, ww, h) = cmd_life::pack_bits(&s.g);
  let bytes: &[u8] = bytemuck::cast_slice(&words);
  let n = words.len();
  let (cube_dim, cube_count) = if k <= 1 {
    (CubeDim::new_2d(wgx, wgy), CubeCount::Static((ww as u32 - 2).div_ceil(wgx), (h as u32 - 2).div_ceil(wgy), 1))
  } else {
    assert!(k <= 32, "the one-word halo allows at most 32 generations per launch");
    (CubeDim::new_2d(tw + 2, th + 2 * k), CubeCount::Static((ww as u32 - 2).div_ceil(tw), (h as u32 - 2).div_ceil(th), 1))
  };
  let launches = gens / k.max(1) as u64;
  println!(
    "rule {} | content: {} | {}x{} = {} cells | {} | {} | {} MiB per buffer",
    s.rule_name,
    s.what,
    s.g.w,
    s.g.h,
    s.g.w * s.g.h,
    if k <= 1 { format!("plain kernel, {wgx}x{wgy} cubes") } else { format!("shared tile, k = {k}, tiles of {} x {th} cells", tw * 32) },
    if checked { "checked" } else { "unchecked" },
    (n * 4) >> 20
  );

  // one launch: `src` to `dst`, on the plain or the shared-tile kernel
  let launch = |src: &cubecl::server::Handle, dst: &cubecl::server::Handle| {
    let (a, b) = unsafe { (BufferArg::from_raw_parts(src.clone(), n), BufferArg::from_raw_parts(dst.clone(), n)) };
    let (cc, cd) = (cube_count.clone(), cube_dim);
    match (k <= 1, checked) {
      (true, true) => step::launch(&client, cc, cd, a, b, ww as u32, h as u32, birth, survive),
      (true, false) => unsafe { step::launch_unchecked(&client, cc, cd, a, b, ww as u32, h as u32, birth, survive) },
      (false, true) => tile::launch(&client, cc, cd, a, b, ww as u32, h as u32, tw, th, k, birth, survive),
      (false, false) => unsafe { tile::launch_unchecked(&client, cc, cd, a, b, ww as u32, h as u32, tw, th, k, birth, survive) },
    }
  };
  let mut last = None;
  let all = crate::repeat_timed(args, || {
    let (a, b) = (client.create_from_slice(bytes), client.create_from_slice(bytes));
    pollster::block_on(client.sync()).expect("sync");
    let t = std::time::Instant::now();
    for i in 0..launches {
      if i % 2 == 0 { launch(&a, &b) } else { launch(&b, &a) }
    }
    pollster::block_on(client.sync()).expect("sync");
    let secs = t.elapsed().as_secs_f64();
    last = Some(if launches % 2 == 1 { b } else { a });
    secs
  });
  let (secs, spread) = crate::median_and_spread(&all, gens as f64, "gens/s");
  println!(
    "  life kernel: {gens} gens in {secs:.4}s = {:.1} gens/s = {:.3e} cell-updates/s{spread}",
    gens as f64 / secs,
    (s.g.w * s.g.h) as f64 * gens as f64 / secs
  );
  if flag(args, "--verify") || flag(args, "--fingerprint") {
    let out = client.read_one(last.expect("a run")).expect("read");
    let out: Vec<u32> = bytemuck::pod_collect_to_vec(&out);
    cmd_life::report(args, &s, &out, ww, gens);
  }
}
