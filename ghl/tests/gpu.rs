//! The GPU engines against the host recursion: every engine ends in the same state (Stages 1–3).
//! Tests return early when the machine has no GPU. No timing is asserted here.

#![cfg(feature = "gpu")]

use std::path::PathBuf;

use ghl::{io::read_rle, Config, Engines, Grid, Pattern, Rect, Regime, Rule, Universe};

fn gpu_config(regime: Regime) -> Config {
  Config { engines: Engines::Gpu, calibrate: false, regime: Some(regime), detect_cycles: false, ..Config::default() }
}

fn has_gpu() -> bool {
  match ghl::GpuContext::new() {
    Ok(_) => true,
    Err(e) => {
      eprintln!("no GPU, skipping: {e}");
      false
    }
  }
}

fn load(rel: &str) -> (Grid, Rule) {
  let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../research/patterns").join(rel);
  let (g, rule) = read_rle(&std::fs::read_to_string(path).unwrap()).unwrap();
  (g, Rule::named(if rule.is_empty() { "Hutton32b" } else { &rule }).unwrap())
}

fn host_run(rule: &Rule, g: &Grid, origin: (i64, i64), gens: u64) -> ghl::Fingerprint {
  let mut u = Universe::new(rule.clone(), Config::cpu()).unwrap();
  u.load(&Pattern::Cells(g.clone()), origin).unwrap();
  u.step(gens);
  u.fingerprint()
}

fn random(w: usize, h: usize, states: u32, seed: u32) -> Grid {
  let mut g = Grid::new(w, h);
  let mut s = seed | 1;
  for c in g.cells.iter_mut() {
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    *c = if s % 2 == 0 { 0 } else { (s >> 9) as u8 % states as u8 };
  }
  g
}

/// Plain and shared-tile kernels of every rule class: networks (adder trees, the two Hutton
/// circuits), B0 rules, a table rule and a tree rule.
#[test]
fn dense_kernels_equal_the_host() {
  if !has_gpu() {
    return;
  }
  let table_rule = Rule::from_rule_text(
    "@RULE Wires\n@TABLE\nn_states:4\nneighborhood:Moore\nsymmetries:permute\nvar a={0,1,2,3}\nvar b={0,1,2,3}\nvar c={0,1,2,3}\nvar d={0,1,2,3}\nvar e={0,1,2,3}\nvar f={0,1,2,3}\nvar g={0,1,2,3}\nvar h={0,1,2,3}\n1,a,b,c,d,e,f,g,h,2\n2,a,b,c,d,e,f,g,h,3\n3,1,1,a,b,c,d,e,f,1\n3,1,a,b,c,d,e,f,g,3\n",
    "Wires",
    "rule",
  )
  .unwrap();
  let rules = vec![
    Rule::named("B3/S23").unwrap(),
    Rule::named("B36/S23").unwrap(),
    Rule::named("B2/S34H").unwrap(),
    Rule::named("B13/S0V").unwrap(),
    Rule::named("B03/S23").unwrap(),
    Rule::named("B0123478/S01234678").unwrap(),
    Rule::named("Hutton32b").unwrap(),
    Rule::named("Hutton32").unwrap(),
    table_rule,
  ];
  for rule in rules {
    let g = random(90, 60, rule.states(), 99);
    for (regime, gens) in [(Regime::PlainCells, 37u64), (Regime::SharedTile, 77)] {
      let want = host_run(&rule, &g, (5, -3), gens);
      let mut u = Universe::new(rule.clone(), gpu_config(regime)).unwrap();
      u.load(&Pattern::Cells(g.clone()), (5, -3)).unwrap();
      u.step(gens);
      assert_eq!(u.fingerprint(), want, "{} under {regime:?}", rule.name());
      // queries while the GPU holds the state
      assert_eq!(u.population(), want.population);
      let bb = u.bounding_box().unwrap();
      assert_eq!((bb.w as i64, bb.h as i64), (want.w, want.h));
    }
  }
}

/// A grid that grows: a glider gun under the plain kernel for long enough to regrow the grid.
#[test]
fn dense_grid_grows() {
  if !has_gpu() {
    return;
  }
  let rule = Rule::named("B3/S23").unwrap();
  let gun = ghl::io::read_rle("x = 36, y = 9, rule = B3/S23\n24bo$22bobo$12b2o6b2o12b2o$11bo3bo4b2o12b2o$2o8bo5bo3b2o$2o8bo3bob2o4bobo$10bo5bo7bo$11bo3bo$12b2o!\n").unwrap().0;
  let want = host_run(&rule, &gun, (0, 0), 3000);
  let mut u = Universe::new(rule, gpu_config(Regime::PlainCells)).unwrap();
  u.load(&Pattern::Cells(gun), (0, 0)).unwrap();
  u.step(3000);
  assert_eq!(u.fingerprint(), want);
}

/// The memo engine: the hit path with the host learning (Stage 2) and with the GPU's miss path
/// (Stage 3), on Hutton32b and Life.
#[test]
fn memo_engine_equals_the_host() {
  if !has_gpu() {
    return;
  }
  for (file, gens) in [("presets/hutton32b_8bit_counter.rle", 2048u64), ("misc/chaotic.rle", 1024), ("life/Unit-Life-Cell-512x512.rle", 512)] {
    let (g, rule) = load(file);
    let want = host_run(&rule, &g, (0, 0), gens);
    for gpu_miss in [false, true] {
      let c = Config { gpu_miss_path: gpu_miss, ..gpu_config(Regime::Memo) };
      let mut u = Universe::new(rule.clone(), c).unwrap();
      u.load(&Pattern::Cells(g.clone()), (0, 0)).unwrap();
      u.step(gens);
      assert_eq!(u.fingerprint(), want, "{file}: memo engine, GPU miss path {gpu_miss}");
      assert_eq!(u.population(), want.population);
      // the engine that ran is the one asked for
      let engines: Vec<ghl::EngineKind> = u.stats().engines.iter().map(|e| e.0).collect();
      let want_engine = if gpu_miss { ghl::EngineKind::GpuMemo } else { ghl::EngineKind::GpuMemoHost };
      assert!(engines.contains(&want_engine), "{file}: engines {engines:?}");
      if gpu_miss {
        assert!(!engines.contains(&ghl::EngineKind::GpuMemoHost), "{file}: the GPU miss path fell back to the host: {engines:?}");
      }
    }
  }
}

/// A sparse universe: copies far apart, so that most blocks of the ID grid are inactive.
#[test]
fn sparse_universe_runs_on_active_blocks() {
  if !has_gpu() {
    return;
  }
  let (g, rule) = load("misc/chaotic.rle");
  let mut cells = Vec::new();
  for (ox, oy) in [(0i64, 0i64), (3000, 200), (900, 2500)] {
    for y in 0..g.h {
      for x in 0..g.w {
        let c = g.get(x, y);
        if c != 0 {
          cells.push(((ox + x as i64, oy + y as i64), c));
        }
      }
    }
  }
  let mut h = Universe::new(rule.clone(), Config::cpu()).unwrap();
  h.set_cells(&cells);
  h.step(1024);
  for gpu_miss in [false, true] {
    let mut u = Universe::new(rule.clone(), Config { gpu_miss_path: gpu_miss, ..gpu_config(Regime::Memo) }).unwrap();
    u.set_cells(&cells);
    u.step(1024);
    assert_eq!(u.fingerprint(), h.fingerprint(), "GPU miss path {gpu_miss}");
    assert_eq!(u.population(), h.population());
  }
}

/// Algorithm 1 choosing per epoch (racing in the latency-bound regime) ends where the host does.
#[test]
fn scheduler_choice_equals_the_host() {
  if !has_gpu() {
    return;
  }
  let (g, rule) = load("presets/hutton32b_8bit_counter.rle");
  let want = host_run(&rule, &g, (100, 200), 5000);
  let c = Config { engines: Engines::Gpu, calibrate: false, ..Config::default() };
  let mut u = Universe::new(rule, c).unwrap();
  u.load(&Pattern::Cells(g), (100, 200)).unwrap();
  let mut left = 5000u64;
  while left > 0 {
    let r = u.step(left.min(333));
    left -= r.generations;
  }
  assert_eq!(u.fingerprint(), want);
  // edits between steps on a GPU-held state
  u.set_cells(&[((0, 0), 7)]);
  let mut one = [0u8];
  u.read_region(Rect::new(0, 0, 1, 1), &mut one);
  assert_eq!(one[0], 7);
}

/// A globally periodic universe on the GPU's ID grid is recognised and its periods skipped.
#[test]
fn memo_cycles_are_skipped() {
  if !has_gpu() {
    return;
  }
  let rule = Rule::named("B3/S23").unwrap();
  let mut cells = Vec::new();
  for i in 0..20i64 {
    for j in 0..20i64 {
      // a field of blinkers and blocks
      let (x, y) = (i * 40, j * 40);
      if (i + j) % 2 == 0 {
        cells.extend([((x, y), 1u8), ((x + 1, y), 1), ((x + 2, y), 1)]);
      } else {
        cells.extend([((x, y), 1u8), ((x + 1, y), 1), ((x, y + 1), 1), ((x + 1, y + 1), 1)]);
      }
    }
  }
  let c = Config { detect_cycles: true, ..gpu_config(Regime::Memo) };
  let mut u = Universe::new(rule.clone(), c).unwrap();
  u.set_cells(&cells);
  let pop = u.population();
  u.step(1 << 30);
  assert_eq!(u.population(), pop);
  assert!(u.stats().cycle.is_some(), "no cycle found");
  let mut h = Universe::new(rule, Config::cpu()).unwrap();
  h.set_cells(&cells);
  h.step(1 << 30);
  assert_eq!(u.fingerprint(), h.fingerprint());
}

/// The memo engine with its node table split across four shards (tables beyond one binding).
#[test]
fn sharded_memo_table_equals_the_host() {
  if !has_gpu() {
    return;
  }
  let (g, rule) = load("presets/hutton32b_8bit_counter.rle");
  let want = host_run(&rule, &g, (0, 0), 1024);
  for gpu_miss in [false, true] {
    let c = Config { gpu_miss_path: gpu_miss, gpu_table_shard_log2: Some(18), ..gpu_config(Regime::Memo) };
    let mut u = Universe::new(rule.clone(), c).unwrap();
    u.load(&Pattern::Cells(g.clone()), (0, 0)).unwrap();
    u.step(1024);
    assert_eq!(u.fingerprint(), want, "GPU miss path {gpu_miss}");
    assert_eq!(u.population(), want.population);
    match u.gpu_view() {
      Some(ghl::GpuView::Ids { nodes, .. }) => assert_eq!(nodes.len(), 4),
      other => panic!("expected an ID grid: {other:?}"),
    }
  }
}

/// The dense regime on a grid split across shards (forced small), growing through the host.
#[test]
fn sharded_dense_regime_equals_the_host() {
  if !has_gpu() {
    return;
  }
  let (g, rule) = load("misc/chaotic.rle");
  let want = host_run(&rule, &g, (0, 0), 700);
  let c = Config { gpu_grid_shard_words: Some(30_000), ..gpu_config(Regime::PlainCells) };
  let mut u = Universe::new(rule, c).unwrap();
  u.load(&Pattern::Cells(g), (0, 0)).unwrap();
  u.step(700);
  assert_eq!(u.population(), want.population);
  assert_eq!(u.fingerprint(), want);
}

/// Maps read from VRAM equal the host's for the same universe and generation: the IDs of the GPU's
/// grid, re-keyed through the tile contents, give the host's cells (roadmap Section 8.4).
#[test]
fn vram_maps_equal_the_host() {
  if !has_gpu() {
    return;
  }
  let (g, rule) = load("presets/hutton32b_8bit_counter.rle");
  let mut u = Universe::new(rule.clone(), gpu_config(Regime::Memo)).unwrap();
  u.load(&Pattern::Cells(g.clone()), (0, 0)).unwrap();
  u.step(512);
  let Some(ghl::GpuView::Ids { lt, x0, y0, nx, ny, .. }) = u.gpu_view() else { panic!("expected an ID grid") };
  let mut h = Universe::new(rule, Config::cpu()).unwrap();
  h.load(&Pattern::Cells(g), (0, 0)).unwrap();
  h.step(512);
  let obs = u.observer();
  let s = 1i64 << lt;
  let rect = Rect::new(x0, y0, nx as u64 * s as u64, ny as u64 * s as u64);
  let ids = obs.ids(rect, lt).wait();
  assert_eq!(ids.len(), (nx * ny) as usize);
  // expand a GPU ID through the GPU's tables, down to the base tiles' cells
  fn cells(obs: &ghl::Observer, level: u32, id: u32, out: &mut Vec<u8>, side: usize, x: usize, y: usize) {
    match obs.tile(level, id).wait().expect("the ID is in the tables") {
      ghl::inspect::Tile::Cells { side: ts, cells } => {
        for yy in 0..ts as usize {
          out[(y + yy) * side + x..(y + yy) * side + x + ts as usize].copy_from_slice(&cells[yy * ts as usize..(yy + 1) * ts as usize]);
        }
      }
      ghl::inspect::Tile::Node { children, .. } => {
        let half = 1usize << (level - 1);
        for (q, c) in children.iter().enumerate() {
          cells(obs, level - 1, *c, out, side, x + (q & 1) * half, y + (q >> 1) * half);
        }
      }
    }
  }
  let mut checked = 0;
  for (i, &id) in ids.iter().enumerate() {
    let (tx, ty) = ((i as u32 % nx) as i64, (i as u32 / nx) as i64);
    let mut got = vec![0u8; (s * s) as usize];
    cells(&obs, lt, id, &mut got, s as usize, 0, 0);
    let mut want = vec![0u8; (s * s) as usize];
    h.read_region(Rect::new(x0 + tx * s, y0 + ty * s, s as u64, s as u64), &mut want);
    assert_eq!(got, want, "tile ({tx}, {ty})");
    checked += (got.iter().any(|&c| c != 0)) as u32;
  }
  assert!(checked > 0);
}

/// A grid split across shards steps as one.
#[test]
fn sharded_grid_equals_one_binding() {
  if !has_gpu() {
    return;
  }
  let ctx = ghl::GpuContext::new().unwrap();
  for name in ["B3/S23", "Hutton32b"] {
    let rule = Rule::named(name).unwrap();
    let g = random(200, 150, rule.states(), 7);
    let src = ghl::gpu::kernels::rule_source(&rule);
    let words = (256 / 32 + 2) * (180 + 2) * rule.planes() as u64;
    let mut sh = ghl::gpu::shard::ShardedGrid::new(&ctx, &src, None, 256, 180, 0, 0, Some(&g.padded(20, 10, 256, 180)), words / 3 + 1).unwrap();
    assert_eq!(sh.shards(), 3);
    sh.step(&ctx, 25, 0);
    let got = sh.download(&ctx);
    let k = ghl::gpu::brute::DenseKernels::new(&ctx, &rule).unwrap();
    let mut one = ghl::gpu::brute::DenseGrid::new(&ctx, &k, 256, 180, 0, 0, Some(&g.padded(20, 10, 256, 180))).unwrap();
    one.step(&ctx, &k, 25, 0, false);
    assert_eq!(got, one.download(&ctx), "{name}");
  }
}
