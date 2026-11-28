//! Stage 0 of the roadmap: the rules, the host recursion and the CPU universe against their
//! oracles. The oracle of every engine is per-cell stepping; on the large grids here it is the
//! bit-sliced brute force of `cpu::leaf`, which the unit tests check against per-cell stepping.

use std::path::PathBuf;

use ghl::{
  cpu::{grid_fingerprint, Engine, Flat, Leaf, PlaneGrid},
  io::read_rle,
  Config, Controller, Format, Grid, Pattern, Rect, Rule, Universe,
};

fn patterns() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../research/patterns")
}

fn load(rel: &str) -> (Grid, Rule) {
  let text = std::fs::read_to_string(patterns().join(rel)).unwrap();
  let (g, rule) = read_rle(&text).unwrap();
  let rule = if rule.is_empty() { "Hutton32b".to_string() } else { rule };
  (g, Rule::named(&rule).unwrap())
}

/// `g` stepped `gens` generations from generation `start` by the bit-sliced brute force, with the
/// grid's border as a zero ring.
fn oracle(rule: &Rule, g: &Grid, gens: usize, start: u64) -> Grid {
  let mut pg = PlaneGrid::from_cells(rule.planes() as usize, g.w, g.h, &g.cells);
  let mut tmp = pg.clone();
  Leaf::new(rule).step(&mut pg, &mut tmp, gens, start);
  Grid { w: g.w, h: g.h, cells: pg.to_cells(g.w, g.h) }
}

#[test]
fn canonical_networks_equal_their_tables() {
  for name in ["Hutton32b", "Hutton32"] {
    let r = Rule::named(name).unwrap();
    let net = r.network().expect("a network");
    assert_eq!(net.verify(&r), 0, "{name}: the network differs from the table");
  }
}

#[test]
fn life_like_networks_equal_their_tables() {
  for name in ["B3/S23", "B36/S23", "B2/S", "B3678/S34678", "B2/S34H", "B13/S0V", "B1357/S1357", "B03/S23", "B0123478/S01234678"] {
    let r = Rule::named(name).unwrap();
    assert_eq!(r.network().unwrap().verify(&r), 0, "{name}");
    if let Some(o) = r.alternates().then(|| r.for_generation(1)) {
      assert_eq!(o.network().unwrap().verify(o), 0, "{name} (odd generations)");
    }
  }
}

/// `ca-research selftest`: 15 flat sweeps and 7 root-stepping configurations on the 8-bit counter.
#[test]
fn selftest_flat_sweeps() {
  let (pat, rule) = load("presets/hutton32b_8bit_counter.rle");
  for a in 1..=3usize {
    for lt in a..=6usize {
      let margin = 2;
      let s = 1usize << lt;
      let mut e = Engine::new(&rule, a, (lt - 1).max(a));
      let mut f = Flat::from_grid(&mut e, &pat, lt, margin);
      let padded = pat.padded(margin * s, margin * s, f.nx * s, f.ny * s);
      let cycles = (768 / s).max(2);
      let mut clear = true;
      for _ in 0..2 * cycles {
        f.sweep(&mut e);
        clear &= f.border_clear(&mut e);
      }
      let want = oracle(&rule, &padded, cycles * s, 0);
      assert!(clear, "flat a={a} lt={lt}: the pattern reached the border");
      assert!(f.to_grid(&e) == want, "flat a={a} lt={lt}: differs from per-cell stepping");
    }
  }
}

#[test]
fn selftest_root_stepping() {
  let (pat, rule) = load("presets/hutton32b_8bit_counter.rle");
  for (a, k) in [(1usize, 1usize), (1, 4), (2, 2), (2, 5), (2, 9), (3, 3), (3, 7)] {
    let mut e = Engine::new(&rule, a, k);
    let (mut root, mut level) = e.from_grid(&pat);
    let gens = 1024usize.max(2 << k);
    for _ in 0..gens >> k {
      let (r, l, _) = e.step_root(root, level);
      root = r;
      level = l;
    }
    let big = pat.padded(512, 512, pat.w + 1024, pat.h + 1024);
    let want = oracle(&rule, &big, gens, 0);
    assert_eq!(e.fingerprint(root, level), grid_fingerprint(&want), "root a={a} k={k}");
  }
}

/// The universe ends in the oracle's state under every step size, controller and origin.
#[test]
fn universe_matches_the_oracle() {
  let cases = [("presets/hutton32b_8bit_counter.rle", 1000u64), ("misc/chaotic.rle", 777), ("life/Unit-Life-Cell-512x512.rle", 300)];
  for (file, gens) in cases {
    let (pat, rule) = load(file);
    let m = gens as usize + 8;
    let want = grid_fingerprint(&oracle(&rule, &pat.padded(m, m, pat.w + 2 * m, pat.h + 2 * m), gens as usize, 0));
    let configs = [
      Config::cpu(),
      Config::cpu_step(3),
      Config::cpu_step(6),
      Config { controller: Controller::Algorithm3, ..Config::cpu() },
      Config { controller: Controller::Priced, ..Config::cpu() },
      Config { controller: Controller::PlainCells, ..Config::cpu() },
      Config { controller: Controller::PlainCells, step_log2: Some(3), ..Config::cpu() },
      Config { base_tile_log2: Some(4), ..Config::cpu() },
      Config { trace: false, detect_cycles: false, ..Config::cpu_step(10) },
    ];
    for (i, c) in configs.into_iter().enumerate() {
      let mut u = Universe::new(rule.clone(), c).unwrap();
      u.load(&Pattern::Cells(pat.clone()), (-37, 1011)).unwrap();
      assert_eq!(u.population(), pat.population() as u64);
      // in uneven pieces, so that steps of every size occur
      let mut left = gens;
      for piece in [1u64, 2, 61, 128, 300, u64::MAX] {
        let n = piece.min(left);
        u.step(n);
        left -= n;
      }
      assert_eq!(u.generation(), gens);
      assert_eq!(u.fingerprint(), want, "{file}, configuration {i}");
    }
  }
}

/// Plain cells in every region are stepped as one block whose results the recursion takes.
#[test]
fn plain_cells_regions_are_stepped_as_blocks() {
  let (pat, rule) = load("misc/chaotic.rle");
  let m = 520;
  let want = grid_fingerprint(&oracle(&rule, &pat.padded(m, m, pat.w + 2 * m, pat.h + 2 * m), 512, 0));
  let mut u = Universe::new(rule, Config { controller: Controller::PlainCells, step_log2: Some(4), ..Config::cpu() }).unwrap();
  u.load(&Pattern::Cells(pat), (0, 0)).unwrap();
  u.step(512);
  assert_eq!(u.fingerprint(), want);
  let served = u.with_host(|e, _| e.stats.block_windows);
  assert!(served > 0, "no window came from a block");
}

#[test]
fn b0_rules_alternate_as_golly_does() {
  let mut g = Grid::new(40, 40);
  let mut s = 12345u64;
  for c in g.cells.iter_mut() {
    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *c = (s >> 61 & 1) as u8;
  }
  for name in ["B03/S23", "B0123478/S01234678"] {
    let rule = Rule::named(name).unwrap();
    for gens in [1u64, 2, 7, 64, 101] {
      let m = gens as usize + 4;
      let want = grid_fingerprint(&oracle(&rule, &g.padded(m, m, g.w + 2 * m, g.h + 2 * m), gens as usize, 0));
      let mut u = Universe::new(rule.clone(), Config::cpu()).unwrap();
      u.load(&Pattern::Cells(g.clone()), (0, 0)).unwrap();
      u.step(gens);
      assert_eq!(u.fingerprint(), want, "{name} after {gens}");
    }
  }
}

#[test]
fn edits_queries_and_files() {
  let (pat, rule) = load("presets/hutton32b_8bit_counter.rle");
  let mut u = Universe::new(rule.clone(), Config::cpu()).unwrap();
  u.load(&Pattern::Cells(pat.clone()), (100, -50)).unwrap();
  let bb = u.bounding_box().unwrap();
  let (x0, y0, x1, y1) = pat.bbox().unwrap();
  assert_eq!(bb, Rect::from_corners(100 + x0 as i64, -50 + y0 as i64, 100 + x1 as i64, -50 + y1 as i64));
  let mut cells = vec![0u8; pat.w * pat.h];
  u.read_region(Rect::new(100, -50, pat.w as u64, pat.h as u64), &mut cells);
  assert_eq!(cells, pat.cells);
  // edits far away grow the root
  u.set_cells(&[((-100_000, 77), 5), ((3_000_000, -9), 1)]);
  let mut one = [0u8; 1];
  u.read_region(Rect::new(-100_000, 77, 1, 1), &mut one);
  assert_eq!(one[0], 5);
  assert_eq!(u.population(), pat.population() as u64 + 2);
  u.set_cells(&[((-100_000, 77), 0), ((3_000_000, -9), 0)]);
  assert_eq!(u.population(), pat.population() as u64);
  u.step(100);
  let fp = u.fingerprint();
  // RLE and macrocell round trips
  for format in [Format::Rle, Format::Macrocell] {
    let mut out = Vec::new();
    u.save(format, &mut out).unwrap();
    let text = String::from_utf8(out).unwrap();
    let pattern = match format {
      Format::Rle => Pattern::Cells(read_rle(&text).unwrap().0),
      Format::Macrocell => Pattern::Macrocell(ghl::Macrocell::parse(&text).unwrap()),
    };
    let mut v = Universe::new(rule.clone(), Config::cpu()).unwrap();
    v.load(&pattern, (0, 0)).unwrap();
    assert_eq!(v.fingerprint(), fp, "{format:?}");
  }
  // tile summaries add up to the population
  let bb = u.bounding_box().unwrap();
  let (_, _, nx, ny) = bb.tiles(6);
  let mut tiles = vec![ghl::TileSummary::default(); nx * ny];
  u.tile_summary(bb, 6, &mut tiles);
  assert_eq!(tiles.iter().map(|t| t.population).sum::<u64>(), u.population());
  u.clear();
  assert_eq!((u.population(), u.generation(), u.bounding_box()), (0, 0, None));
}

#[test]
fn life_macrocell_two_state_round_trip() {
  let (pat, rule) = load("life/Unit-Life-Cell-512x512.rle");
  let mut u = Universe::new(rule.clone(), Config::cpu()).unwrap();
  u.load(&Pattern::Cells(pat), (0, 0)).unwrap();
  let mut out = Vec::new();
  u.save(Format::Macrocell, &mut out).unwrap();
  let m = ghl::Macrocell::parse(std::str::from_utf8(&out).unwrap()).unwrap();
  assert!(m.nodes.iter().any(|n| matches!(n, ghl::io::McNode::Leaf8(_))), "two-state rules write 8x8 leaves");
  let mut v = Universe::new(rule, Config::cpu()).unwrap();
  v.load(&Pattern::Macrocell(m), (0, 0)).unwrap();
  assert_eq!(v.fingerprint(), u.fingerprint());
}

#[test]
fn cycles_are_skipped_exactly() {
  // a blinker: period 2, found at the second epoch, then whole periods are skipped
  let rule = Rule::named("B3/S23").unwrap();
  let mut u = Universe::new(rule.clone(), Config::cpu_step(4)).unwrap();
  u.set_cells(&[((10, 10), 1), ((11, 10), 1), ((12, 10), 1)]);
  u.step(1 << 40);
  assert_eq!(u.population(), 3);
  assert!(u.stats().cycle.is_some());
  assert_eq!(u.bounding_box(), Some(Rect::new(10, 10, 3, 1)));
  // a glider moves (1, 1) every 4 generations
  let mut g = Universe::new(rule, Config::cpu_step(5)).unwrap();
  g.set_cells(&[((1, 0), 1), ((2, 1), 1), ((0, 2), 1), ((1, 2), 1), ((2, 2), 1)]);
  g.step(4_000_000);
  assert_eq!(g.bounding_box(), Some(Rect::new(1_000_000, 1_000_000, 3, 3)));
}
