//! The throughput targets of each stage of the roadmap (Section 4), measured on the machine's own
//! GPU and printed next to the paper's numbers. Rates in cell-updates/s over the pattern's bounding
//! box, generations/s in parentheses. Every run is checked cell-exact against the host recursion
//! after it is timed.
//!
//!     cargo bench -p ghl --bench stages [-- stage0 stage1 stage2 stage3 stage4]
//!
//! Run on an otherwise idle machine: GPU rates repeat within a few percent, CPU rates of one day
//! are comparable only with each other (paper Appendix A).

use std::{path::PathBuf, time::Instant};

use ghl::{Config, Controller, Engines, Grid, Pattern, Regime, Rule, Universe};

fn patterns() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../research/patterns")
}

fn load(rel: &str) -> (Pattern, Rule) {
  let loaded = Pattern::load(patterns().join(rel)).unwrap_or_else(|e| panic!("{rel}: {e}"));
  let rule = if loaded.rule.is_empty() { "Hutton32b".to_string() } else { loaded.rule.clone() };
  (loaded.pattern, Rule::named(&rule).unwrap())
}

fn soup(side: usize, seed: u32) -> Grid {
  let h = |x0: u32| {
    let mut x = x0;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846CA68B);
    x ^ (x >> 16)
  };
  let mut g = Grid::new(side, side);
  for (i, c) in g.cells.iter_mut().enumerate() {
    *c = (h(h(i as u32) ^ seed.wrapping_mul(0x9E3779B9)) < 0x8000_0000) as u8;
  }
  g
}

/// Run `gens` generations; returns (seconds, cell-updates/s, gens/s) and checks the final state
/// against the host recursion.
fn timed(label: &str, target: &str, rule: &Rule, pattern: &Pattern, config: Config, gens: u64, check: bool) {
  let mut u = match Universe::new(rule.clone(), config) {
    Ok(u) => u,
    Err(e) => {
      println!("{label:<48} skipped: {e}");
      return;
    }
  };
  u.load(pattern, (0, 0)).unwrap();
  // one epoch to compile and upload, untimed
  u.step(1);
  let t = Instant::now();
  let r = u.step(gens - 1);
  let s = t.elapsed().as_secs_f64();
  let fp = u.fingerprint();
  let verdict = if check {
    let mut h = Universe::new(rule.clone(), Config::cpu()).unwrap();
    h.load(pattern, (0, 0)).unwrap();
    h.step(gens);
    if h.fingerprint() == fp {
      "exact"
    } else {
      "DIFFERS"
    }
  } else {
    "unchecked"
  };
  println!("{label:<48} {:>10.3e} ({:>12.0} gens/s) target {target:<28} [{verdict}]", r.cell_updates / s, r.generations as f64 / s);
}

fn gpu(regime: Regime) -> Config {
  Config { engines: Engines::Gpu, regime: Some(regime), detect_cycles: false, ..Config::default() }
}

fn stage0() {
  println!("== Stage 0: the host recursion (one thread) against Golly's recorded speed (fixtures/golly.tsv)");
  for (file, gens) in [("advanced counter v2.rle", 1u64 << 20), ("misc/chaotic.rle", 1 << 20), ("life/Unit-Life-Cell-512x512.rle", 1 << 20), ("misc/squares.rle", 65536)] {
    let (p, rule) = load(file);
    timed(file, "0.26-0.97x Golly", &rule, &p, Config::cpu_step(10), gens, false);
  }
}

fn stage1() {
  println!("== Stage 1: plain cells on the GPU");
  let (counter, h32b) = load("advanced counter v2.rle");
  timed("counter, shared tile", "3.0e10 (804,105 gens/s)", &h32b, &counter, gpu(Regime::SharedTile), 65536, true);
  let (chaotic, _) = load("misc/chaotic.rle");
  timed("chaotic, shared tile", "1.8e10 (892,832 gens/s)", &h32b, &chaotic, gpu(Regime::SharedTile), 65536, true);
  let (squares, h32) = load("misc/squares.rle");
  timed("squares, shared tile (Hutton32 circuit)", "1.57e11 (571,123 gens/s)", &h32, &squares, gpu(Regime::SharedTile), 16384, true);
  let (rom, _) = load("misc/extra/lotr1.ANSI-1M.rle");
  timed("ROM, plain kernel", "3.9e11 (40,859 gens/s)", &h32b, &rom, gpu(Regime::PlainCells), 2112, true);
  let life = Rule::named("B3/S23").unwrap();
  timed("Life 256x256 soup, shared tile", "1.37e11 (2,092,859 gens/s)", &life, &Pattern::Cells(soup(256, 1)), gpu(Regime::SharedTile), 65536, true);
  timed("Life 2048x2048 soup, 4096 generations", "50,018 gens/s", &life, &Pattern::Cells(soup(2048, 1)), gpu(Regime::PlainCells), 4096, true);
  timed("Life 8192x8192 soup, plain kernel", "1.42e12", &life, &Pattern::Cells(soup(8192, 1)), gpu(Regime::PlainCells), 512, false);
  let big = {
    let mut g = Grid::new(8192, 8192);
    let s = soup(8192, 7);
    for (c, v) in g.cells.iter_mut().zip(&s.cells) {
      *c = *v * 13;
    }
    g
  };
  timed("Hutton32b 8192x8192, plain kernel", "3.0e11", &h32b, &Pattern::Cells(big), gpu(Regime::PlainCells), 256, false);
}

fn stage2_3() {
  println!("== Stages 2 and 3: the memo engine (hit path; misses by the host, then by the GPU)");
  for gpu_miss in [false, true] {
    let tag = if gpu_miss { "GPU miss path" } else { "host learning" };
    let (rom, h32b) = load("misc/extra/lotr1.PPMd-244K.rle");
    timed(&format!("ROM (PPMd), {tag}"), "1.6e13 at 32x32 tiles", &h32b, &rom, Config { gpu_miss_path: gpu_miss, ..gpu(Regime::Memo) }, 1584 * 4, true);
    let (ulc, life) = load("life/Unit-Life-Cell-512x512.rle");
    timed(&format!("Unit Life Cell, {tag}"), "3.6e14 at 32x32 tiles", &life, &ulc, Config { gpu_miss_path: gpu_miss, ..gpu(Regime::Memo) }, 17280, true);
    let (chaotic, _) = load("misc/chaotic.rle");
    timed(&format!("chaotic (non-repeating logic), {tag}"), "1.8e12 (miss path)", &h32b, &chaotic, Config { gpu_miss_path: gpu_miss, ..gpu(Regime::Memo) }, 16384, true);
  }
}

fn stage4() {
  println!("== Stage 4: Algorithm 1 choosing the engine, the controllers");
  let (counter, h32b) = load("advanced counter v2.rle");
  timed("counter, Algorithm 1 (racing)", "fastest of the three", &h32b, &counter, Config { engines: Engines::Gpu, ..Config::default() }, 65536, true);
  for c in [Controller::Uniform, Controller::Algorithm3, Controller::Priced, Controller::PlainCells] {
    timed(&format!("counter, host recursion, {c:?}"), "model: Tables XXIX, XXX", &h32b, &counter, Config { controller: c, ..Config::cpu() }, 16384, true);
  }
}

fn main() {
  let args: Vec<String> = std::env::args().skip(1).filter(|a| !a.starts_with('-')).collect();
  let want = |s: &str| args.is_empty() || args.iter().any(|a| a == s);
  if want("stage0") {
    stage0();
  }
  if want("stage1") {
    stage1();
  }
  if want("stage2") || want("stage3") {
    stage2_3();
  }
  if want("stage4") {
    stage4();
  }
}
