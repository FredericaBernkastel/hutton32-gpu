//! `symmetry`: what a memo would gain by sharing tiles and windows that are rotations or mirror
//! images of each other, measured on the flat sweep of real patterns.
//!
//! The sweep runs with the exact memo, as GHL's hit path does. Afterwards every memo entry is
//! transformed by the eight symmetries of the square: a base tile's cells are moved and, for the
//! 32-state rules, its directional states relabelled (an east-pointing transmission state becomes
//! a north-pointing one under a quarter turn; the horizontal and vertical crossings swap); a node
//! above it permutes its four children and transforms each. Entries that are images of each other
//! form one orbit; a memo keyed by orbit would miss once per orbit instead of once per entry.
//!
//! Sharing is sound only if the rule commutes with the symmetry. Life does; the Hutton rules do
//! not on every neighbourhood (`tools/symmetry_check.py`). `--check N` therefore solves the images
//! of N memo entries at the floor and at the top and counts those whose true result differs from
//! the transformed result: the wrong answers a shared memo would have given on this content.

use crate::{
  cmd_stats::{self, Loaded},
  flat::Flat,
  hl::{Engine, NONE},
  opt, opt_or,
  rulespec::RuleSpec,
};
use rustc_hash::{FxHashMap, FxHashSet};
use std::time::Instant;

/// The symmetries of the square: 0 identity, 1-3 quarter turns counter-clockwise (east moves
/// north), 4 mirror east-west, 5 mirror north-south, 6 transpose, 7 anti-transpose. Rows grow
/// southwards.
fn place(k: usize, n: usize, x: usize, y: usize) -> (usize, usize) {
  match k {
    0 => (x, y),
    1 => (y, n - 1 - x),
    2 => (n - 1 - x, n - 1 - y),
    3 => (n - 1 - y, x),
    4 => (n - 1 - x, y),
    5 => (x, n - 1 - y),
    6 => (y, x),
    _ => (n - 1 - y, n - 1 - x),
  }
}
/// Where a direction (east, north, west, south = 0..3) points after the symmetry.
const DIRS: [[u8; 4]; 8] = [[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2], [2, 1, 0, 3], [0, 3, 2, 1], [3, 2, 1, 0], [1, 0, 3, 2]];
/// Symmetries that exchange the axes, and with them the horizontal and vertical crossings.
const SWAPS_AXES: [bool; 8] = [false, true, false, true, false, false, true, true];

struct Sym {
  /// relabel directional states (the 32-state rules); Life's two states are isotropic
  relabel: bool,
  images: FxHashMap<(usize, u32, u8), u32>,
}

impl Sym {
  fn state(&self, k: usize, c: u8) -> u8 {
    if !self.relabel {
      return c;
    }
    match c {
      9..=24 => {
        let base = 9 + (c - 9) / 4 * 4;
        base + DIRS[k][((c - 9) % 4) as usize]
      }
      29 | 30 if SWAPS_AXES[k] => 59 - c,
      _ => c,
    }
  }

  /// The image of node `id` (level `l`; `l == a` is a base tile) under symmetry `k`, interned.
  fn image(&mut self, e: &mut Engine, l: usize, id: u32, k: usize) -> u32 {
    if k == 0 {
      return id;
    }
    if let Some(&t) = self.images.get(&(l, id, k as u8)) {
      return t;
    }
    let t = if l == e.a {
      let s = e.tile_side;
      let src = e.tile_cells(id).to_vec();
      let mut out = vec![0u8; s * s];
      for y in 0..s {
        for x in 0..s {
          let (u, v) = place(k, s, x, y);
          out[v * s + u] = self.state(k, src[y * s + x]);
        }
      }
      e.tile(&out)
    } else {
      let ch = e.levels[l].nodes[id as usize];
      let mut out = [0u32; 4];
      for (q, &c) in ch.iter().enumerate() {
        let (u, v) = place(k, 2, q & 1, q >> 1);
        out[v * 2 + u] = self.image(e, l - 1, c, k);
      }
      e.intern(l, out)
    };
    self.images.insert((l, id, k as u8), t);
    t
  }

  /// The orbit's representative: the smallest id among the images under `group`.
  fn canonical(&mut self, e: &mut Engine, l: usize, id: u32, group: &[usize]) -> u32 {
    group.iter().map(|&k| self.image(e, l, id, k)).min().unwrap()
  }
}

/// `symmetry --pattern P [--a A --lt L --gens N --margin M --check N]`
pub fn run(args: &[String]) {
  let path: std::path::PathBuf = opt(args, "--pattern").expect("--pattern").into();
  let (loaded, file_rule) = cmd_stats::load_pattern(&path);
  let spec = cmd_stats::variant_for(args, &file_rule);
  let lut = crate::load_lut(&spec);
  let relabel = !matches!(spec, RuleSpec::LifeLike(..));
  let a: usize = opt_or(args, "--a", 3);
  let lt: usize = opt_or(args, "--lt", 4);
  let gens: u64 = opt_or(args, "--gens", 4096);
  let margin: usize = opt_or(args, "--margin", 2);
  let check: usize = opt_or(args, "--check", 0);
  let mut e = Engine::new(&lut, a, (lt - 1).max(a));
  let mut f = match &loaded {
    Loaded::Dense(g) => Flat::from_grid(&mut e, g, lt, margin),
    Loaded::Mc(m) => {
      let (root, level) = e.from_mc(m);
      let (root, level, _) = e.popzeros(root, level);
      let lvl = level.max(lt);
      let ids = e.ids_at_level(root, lvl, lt);
      Flat::from_ids(&mut e, &ids, 1 << (lvl - lt), lt, margin)
    }
  };
  for lv in e.levels.iter_mut() {
    lv.stats = Default::default();
  }
  let s = 1u64 << lt;
  let t0 = Instant::now();
  let sweeps = gens / (s / 2);
  let mut probes = 0u64; // window instances the sweeps probe, before deduplication
  for _ in 0..sweeps {
    probes += f.sweep(&mut e).windows;
  }
  let run_s = t0.elapsed().as_secs_f64();
  println!(
    "pattern {} rule {} | base tiles {}x{}, sweep tiles {s}x{s} | {} gens ({sweeps} sweeps) in {run_s:.1}s",
    path.file_name().unwrap().to_string_lossy(),
    spec.name(),
    1 << a,
    1 << a,
    sweeps * (s / 2)
  );

  // the memo as the run left it: base tiles, and the entries with a result at every level above
  let n_tiles = e.num_tiles() as u32;
  let mut entries: Vec<(usize, Vec<u32>)> = vec![(a, (0..n_tiles).collect())];
  for l in a + 1..=lt + 1 {
    let ids: Vec<u32> = (0..e.levels[l].res.len() as u32).filter(|&i| e.levels[l].res[i as usize] != NONE).collect();
    entries.push((l, ids));
  }
  // requests per level: probes of the sweep at the top, the recursion's (deduplicated per wave) below
  let requests: Vec<u64> = (0..=lt + 1).map(|l| if l == lt + 1 { probes } else if l > a { e.levels[l].stats.requests } else { 0 }).collect();
  let mut sym = Sym { relabel, images: FxHashMap::default() };
  const C4: [usize; 4] = [0, 1, 2, 3];
  const D4: [usize; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
  println!("  level  window        entries   rotations (C4)       + mirrors (D4)     requests   hit rate: exact / C4 / D4");
  for (l, ids) in &entries {
    let mut c4 = FxHashSet::default();
    let mut d4 = FxHashSet::default();
    for &id in ids {
      c4.insert(sym.canonical(&mut e, *l, id, &C4));
      d4.insert(sym.canonical(&mut e, *l, id, &D4));
    }
    let n = ids.len() as f64;
    let side = 1usize << *l;
    let what = if *l == a { format!("{side}x{side} tile") } else { format!("{side}x{side}") };
    let req = requests[*l];
    let rates = if req > 0 {
      let h = |m: f64| 100.0 * (1.0 - m / req as f64);
      format!("{:>10}   {:5.1}% / {:5.1}% / {:5.1}%", req, h(n), h(c4.len() as f64), h(d4.len() as f64))
    } else {
      String::new()
    };
    println!(
      "  {l:5}  {what:12} {:10}   {:9} ({:4.2}x)   {:9} ({:4.2}x)   {rates}",
      ids.len(),
      c4.len(),
      n / c4.len().max(1) as f64,
      d4.len(),
      n / d4.len().max(1) as f64
    );
  }

  // would sharing have been right? solve the images of sampled entries and compare
  if check > 0 {
    for &l in &[a + 1, lt + 1] {
      let ids = &entries.iter().find(|(m, _)| *m == l).unwrap().1;
      let stride = (ids.len() / check).max(1);
      let sample: Vec<u32> = ids.iter().step_by(stride).copied().collect();
      let mut wrong = [0u64; 8];
      let mut tried = 0u64;
      for k in 1..8 {
        let images: Vec<u32> = sample.iter().map(|&id| sym.image(&mut e, l, id, k)).collect();
        let mut q = images.clone();
        e.solve(l, &mut q);
        for (&id, &t) in sample.iter().zip(&images) {
          let r = e.result(l, id);
          let expected = sym.image(&mut e, l - 1, r, k);
          if e.result(l, t) != expected {
            wrong[k] += 1;
          }
        }
        tried += sample.len() as u64;
      }
      let side = 1usize << l;
      println!(
        "  check at {side}x{side}: {} entries x 7 symmetries = {tried} images solved; wrong results per symmetry (rot90 rot180 rot270 mirrorEW mirrorNS transpose anti): {:?}",
        sample.len(),
        &wrong[1..]
      );
    }
  }
  println!("  analysis {:.1}s", t0.elapsed().as_secs_f64() - run_s);
}
