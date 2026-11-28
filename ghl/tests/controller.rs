//! The acceptance test of the price-based controller (roadmap Stage 4): on no universe of paper
//! Tables XXVII and XXVIII more than 5 % above the uniform floor, and within 20 % of the sums of the
//! engine per region chosen in hindsight, at the configurations the roadmap names (Hutton32b:
//! 16-generation steps and 64×64 regions, 8-generation steps and 32×32 regions; Life: 64-generation
//! steps and 256×256 regions). Model times, not measurements: operation counts priced with the
//! constants of paper Table I, as the harness's `adapt` prices them, so the machine's load does not
//! enter. The choice in hindsight is the floor run's price of it (`Policy::hindsight`). Every policy
//! must also end in the same state.
//!
//!     cargo test --release -p ghl --test controller -- --ignored --nocapture
//!
//! `price_based_controller_acceptance` takes about a minute and a half with the universes in
//! parallel and passes. The controller's two constants were chosen from the price traces of these
//! universes' floor runs (`GHL_CONTROLLER_TRACES=1` writes them to
//! `research/out/controller_traces/`). `price_based_controller_tables_xxvii_xxviii` asserts the
//! floor criterion at every configuration of the two tables the criterion names, and fails on one
//! universe; `price_based_controller_step_rule_scales` reports the steps the step rule can choose
//! beyond those.

use std::path::PathBuf;

use ghl::{
  cpu::{Engine, Fingerprint},
  io::Grid,
  sched::{controller::rnd, Mode, Model, Policy, PolicyKind, PriceRow},
  Config, Pattern, Rect, Rule, Universe,
};
use rustc_hash::FxHashSet;

fn research() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../research")
}

fn dense(path: &std::path::Path) -> Grid {
  match Pattern::load(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())).pattern {
    Pattern::Cells(g) => g,
    Pattern::Macrocell(m) => {
      let rule = Rule::named("Hutton32b").unwrap();
      let mut e = Engine::new(&rule, 1, 1);
      let (root, level) = e.from_mc(&m);
      let full = e.expand(root, level);
      let (x0, y0, x1, y1) = full.bbox().expect("empty pattern");
      full.crop(x0, y0, x1, y1)
    }
  }
}

/// The harness's `--compose` (paths relative to research/).
fn compose(spec: &str, rule: &Rule) -> Grid {
  let mut items = Vec::new();
  for item in spec.split(';').filter(|s| !s.trim().is_empty()) {
    let (path, rest) = item.trim().split_once('@').unwrap();
    let (pos, seed) = match rest.split_once('~') {
      Some((p, s)) => (p, Some(s.parse::<u32>().unwrap())),
      None => (rest, None),
    };
    let (pos, pre) = match pos.split_once('+') {
      Some((p, g)) => (p, g.parse::<u64>().unwrap()),
      None => (pos, 0),
    };
    let (x, y) = pos.split_once(',').unwrap();
    let (x, y) = (x.parse::<usize>().unwrap(), y.parse::<usize>().unwrap());
    let p = if path.starts_with("patterns/") { research().join(path) } else { PathBuf::from(path) };
    let mut g = dense(&p);
    if let Some(seed) = seed {
      let live: Vec<usize> = (0..g.cells.len()).filter(|&i| g.cells[i] != 0).collect();
      for n in 0..8u32 {
        let i = live[rnd(seed.wrapping_mul(977).wrapping_add(n)) as usize % live.len()];
        let j = live[rnd(seed.wrapping_mul(31337).wrapping_add(n * 7 + 1)) as usize % live.len()];
        g.cells[i] = g.cells[j];
      }
    }
    if pre > 0 {
      let padded = g.padded(32, 32, g.w + 64, g.h + 64);
      let mut u = Universe::new(rule.clone(), Config::cpu()).unwrap();
      u.load(&Pattern::Cells(padded.clone()), (0, 0)).unwrap();
      u.step(pre);
      let mut out = Grid::new(padded.w, padded.h);
      u.read_region(Rect::new(0, 0, padded.w as u64, padded.h as u64), &mut out.cells);
      items.push((out, x.saturating_sub(32), y.saturating_sub(32)));
    } else {
      items.push((g, x, y));
    }
  }
  let w = items.iter().map(|(g, x, _)| x + g.w).max().unwrap();
  let h = items.iter().map(|(g, _, y)| y + g.h).max().unwrap();
  let mut canvas = Grid::new(w, h);
  for (g, x, y) in &items {
    canvas.blit(g, *x, *y);
  }
  canvas
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
  let thr = (0.5f64 * 4294967296.0).min(4294967295.0) as u64;
  let mut g = Grid::new(side, side);
  for (i, c) in g.cells.iter_mut().enumerate() {
    *c = ((h(h(i as u32) ^ seed.wrapping_mul(0x9E3779B9)) as u64) < thr) as u8;
  }
  g
}

struct Run {
  ns: f64,
  hindsight_ns: Option<f64>,
  hindsight_cells: FxHashSet<(i64, i64)>,
  fp: Fingerprint,
  switches: u64,
  log: Vec<PriceRow>,
}

/// The harness's `run_policy`: the universe stepped from the root through the policy.
#[allow(clippy::too_many_arguments)]
fn run_policy(g: &Grid, rule: &Rule, a: usize, k: usize, rl: usize, gens: u64, kind: PolicyKind, init: Mode, model: Model, maxf: usize, log: bool) -> Run {
  let mut e = Engine::new(rule, a, k);
  let (mut root, mut level) = e.from_grid(g);
  while level < rl + 1 {
    let z = e.empty(level);
    root = e.intern(level + 1, [root, z, z, z]);
    level += 1;
  }
  let (mut ox, mut oy) = (0i64, 0i64);
  let mut pol = Policy::new(kind, rl, a, k, init, model, maxf);
  if log {
    pol.log = Some(Vec::new());
  }
  let mut cells: Vec<(i64, i64)> = Vec::new();
  for _ in 0..gens >> k {
    cells.clear();
    e.nonempty_at(root, level, ox, oy, rl, &mut cells);
    let nonempty: FxHashSet<(i64, i64)> = cells.iter().map(|&(x, y)| (x >> rl, y >> rl)).collect();
    let (r, l, x, y) = e.step_root_at(root, level, ox, oy, rl + 1, &mut pol);
    root = r;
    level = l;
    ox = x;
    oy = y;
    pol.end_step(&nonempty);
  }
  let hind = matches!(kind, PolicyKind::Fixed(Mode::Floor(_))).then(|| pol.hindsight());
  Run {
    ns: pol.total_hash + pol.total_brute,
    hindsight_ns: hind.as_ref().map(|h| h.ns),
    hindsight_cells: hind.map(|h| h.cells).unwrap_or_default(),
    fp: e.fingerprint(root, level),
    switches: pol.switches,
    log: pol.log.take().unwrap_or_default(),
  }
}

struct Universe0 {
  name: &'static str,
  grid: Grid,
  gens: u64,
}

fn hutton_universes(rule: &Rule) -> Vec<Universe0> {
  let p = research().join("patterns");
  let spec = std::fs::read_to_string(research().join("tools/composite_spec.txt")).unwrap();
  let cha = p.join("misc/chaotic.rle").display().to_string();
  let ch8 = [(64, 64, None), (576, 64, Some(1)), (1088, 64, Some(2)), (1600, 64, Some(3)), (64, 576, Some(4)), (576, 576, Some(5)), (1088, 576, Some(6)), (1600, 576, Some(7))]
    .iter()
    .map(|(x, y, s)| match s {
      Some(s) => format!("{cha}@{x},{y}~{s}"),
      None => format!("{cha}@{x},{y}"),
    })
    .collect::<Vec<_>>()
    .join(";");
  vec![
    Universe0 { name: "mixed", grid: compose(spec.trim(), rule), gens: 4096 },
    Universe0 { name: "ROM", grid: dense(&p.join("misc/extra/lotr1.PPMd-244K.rle")), gens: 2112 },
    Universe0 { name: "counter", grid: dense(&p.join("advanced counter v2.rle")), gens: 16384 },
    Universe0 { name: "8 chaotic", grid: compose(&ch8, rule), gens: 8192 },
  ]
}

fn life_universes() -> Vec<Universe0> {
  let p = research().join("patterns");
  vec![
    Universe0 { name: "soup", grid: soup(512, 1), gens: 1024 },
    Universe0 { name: "Unit Life Cell", grid: dense(&p.join("life/Unit-Life-Cell-512x512.rle")), gens: 11520 },
    Universe0 { name: "metapixel galaxy", grid: dense(&p.join("life/metapixel-galaxy.mc")), gens: 4096 },
    Universe0 { name: "catacryst", grid: dense(&p.join("life/catacryst.mc")), gens: 4096 },
    Universe0 { name: "gotts-dots", grid: dense(&p.join("life/gotts-dots.mc")), gens: 4096 },
  ]
}

/// Every region's prices per macro-step, as CSV (`GHL_CONTROLLER_TRACES=1`).
fn write_trace(name: &str, log: &[PriceRow], hindsight_cells: &FxHashSet<(i64, i64)>) {
  let dir = research().join("out/controller_traces");
  std::fs::create_dir_all(&dir).unwrap();
  let mut s = String::from("step,x,y,mode,top_misses,floor_ns,cells_ns,credit_ns,charged_ns,hindsight_cells,brute_ns,lo_req,lo_miss,hi_req,hi_miss\n");
  for r in log {
    let mode = match r.mode {
      Mode::Floor(f) => format!("F{f}"),
      Mode::Cells => "C".to_string(),
    };
    let cells = r.cells_ns.map(|c| format!("{c:.3}")).unwrap_or_default();
    let h = hindsight_cells.contains(&(r.x, r.y)) as u8;
    s.push_str(&format!(
      "{},{},{},{},{},{:.3},{},{:.3},{:.3},{},{:.3},{},{},{},{}\n",
      r.step, r.x, r.y, mode, r.top_misses, r.floor_ns, cells, r.credit_ns, r.charged_ns, h, r.brute_ns, r.lo.0, r.lo.1, r.hi.0, r.hi.1
    ));
  }
  std::fs::write(dir.join(format!("{name}.csv")), s).unwrap();
}

/// One configuration (steps of `2^k` generations, regions of `2^rl` cells): returns (sum of the
/// floor, sum in hindsight, sum of the controller, the universes where the controller is more than
/// 5 % above the floor). The runs go in parallel.
#[allow(clippy::too_many_arguments)]
fn configuration(label: &str, rule: &Rule, universes: &[Universe0], a: usize, k: usize, rl: usize, model: Model, maxf: usize) -> (f64, f64, f64, Vec<String>) {
  let maxf = maxf.clamp(a, rl - 1);
  let init = Mode::Floor(a);
  let traces = std::env::var_os("GHL_CONTROLLER_TRACES").is_some();
  let runs: Vec<(Run, Run)> = std::thread::scope(|s| {
    let handles: Vec<_> = universes
      .iter()
      .map(|u| {
        let floor = s.spawn(move || run_policy(&u.grid, rule, a, k, rl, u.gens, PolicyKind::Fixed(init), init, model, maxf, traces));
        let priced = s.spawn(move || run_policy(&u.grid, rule, a, k, rl, u.gens, PolicyKind::Priced, init, model, maxf, traces));
        (floor, priced)
      })
      .collect();
    handles.into_iter().map(|(f, p)| (f.join().unwrap(), p.join().unwrap())).collect()
  });
  let (mut sf, mut sh, mut sp) = (0.0, 0.0, 0.0);
  let mut over = Vec::new();
  eprintln!("{label}: step {}, region {}x{}", 1 << k, 1 << rl, 1 << rl);
  eprintln!("  {:<18} {:>10} {:>10} {:>10} {:>7} {:>9}", "universe", "floor ms", "hindsight", "priced ms", "ratio", "switches");
  for (u, (floor, priced)) in universes.iter().zip(&runs) {
    assert_eq!(floor.fp, priced.fp, "{}: the controller changed the universe", u.name);
    if traces {
      let name = if rl == k + 2 { format!("{}_k{}_{}", label, k, u.name.replace(' ', "_")) } else { format!("{}_k{}r{}_{}", label, k, rl, u.name.replace(' ', "_")) };
      write_trace(&format!("{name}_floor"), &floor.log, &floor.hindsight_cells);
      write_trace(&format!("{name}_priced"), &priced.log, &floor.hindsight_cells);
    }
    let h = floor.hindsight_ns.unwrap();
    let ratio = priced.ns / floor.ns;
    eprintln!("  {:<18} {:>10.3} {:>10.3} {:>10.3} {:>7.3} {:>9}", u.name, floor.ns / 1e6, h / 1e6, priced.ns / 1e6, ratio, priced.switches);
    if ratio > 1.05 {
      over.push(format!("{} {} at step {}, region {}: {:.3}x the floor", label, u.name, 1 << k, 1 << rl, ratio));
    }
    sf += floor.ns;
    sh += h;
    sp += priced.ns;
  }
  eprintln!("  {:<18} {:>10.3} {:>10.3} {:>10.3} {:>7.3}   (priced / hindsight {:.3})", "sum", sf / 1e6, sh / 1e6, sp / 1e6, sp / sf, sp / sh);
  (sf, sh, sp, over)
}

/// Run configurations `(rule, step log2, region log2)`; `bound`: the sum's bound over the choice in
/// hindsight, if any. Returns the failures of the floor criterion and of the bound.
fn check(configs: &[(&str, usize, usize)], bound: Option<f64>) -> Vec<String> {
  let h32b = Rule::named("Hutton32b").unwrap();
  let life = Rule::named("B3/S23").unwrap();
  let hu = hutton_universes(&h32b);
  let li = life_universes();
  let life_model = Model { cell: 1e9 / 1.39e12, ..Model::default() };
  let mut failures = Vec::new();
  for &(label, k, rl) in configs {
    // the harness's settings: 8×8 base tiles and floor F16 for Hutton32b, at most F32; 16×16 and
    // F32 for Life at its kernel's rate, at most F64
    let (_, sh, sp, over) = match label {
      "Hutton32b" => configuration(label, &h32b, &hu, 3, k, rl, Model::default(), 4),
      _ => configuration(label, &life, &li, 4, k, rl, life_model, 5),
    };
    failures.extend(over);
    if let Some(bound) = bound {
      if sp > bound * sh {
        failures.push(format!("{label} at step {}, region {}: {:.3} ms against {:.3} ms in hindsight ({:.3}x)", 1 << k, 1 << rl, sp / 1e6, sh / 1e6, sp / sh));
      }
    }
  }
  failures
}

/// `GHL_CONTROLLER_CONFIGS=Hutton32b:6:8,Life:4:6` keeps only those (rule:step log2:region log2).
fn only(mut configs: Vec<(&'static str, usize, usize)>) -> Vec<(&'static str, usize, usize)> {
  if let Ok(only) = std::env::var("GHL_CONTROLLER_CONFIGS") {
    let want: Vec<String> = only.split(',').map(|s| s.trim().to_string()).collect();
    configs.retain(|(r, k, rl)| want.contains(&format!("{r}:{k}:{rl}")));
  }
  configs
}

/// Both criteria at the roadmap's three configurations: the sums within 20 % of the choice in
/// hindsight and no universe more than 5 % above the floor.
#[test]
#[ignore]
fn price_based_controller_acceptance() {
  let failures = check(&only(vec![("Hutton32b", 4, 6), ("Hutton32b", 3, 5), ("Life", 6, 8)]), Some(1.2));
  assert!(failures.is_empty(), "the price-based controller fails its acceptance test:\n{}", failures.join("\n"));
}

/// The floor criterion at every configuration of paper Tables XXVII and XXVIII, whose universes the
/// acceptance test names: 16-generation steps, regions of 64×64 to 1024×1024, both rules. It fails
/// on one universe (`implementation.md`, Section 9: Life's Unit Life Cell with 64×64 and 128×128
/// regions), which is why the controller ships disabled. About four minutes.
#[test]
#[ignore]
fn price_based_controller_tables_xxvii_xxviii() {
  let configs: Vec<_> = ["Hutton32b", "Life"].into_iter().flat_map(|r| (6..=10).map(move |rl| (r, 4, rl))).collect();
  let failures = check(&only(configs), None);
  assert!(failures.is_empty(), "the price-based controller is more than 5 % above the floor:\n{}", failures.join("\n"));
}

/// A report at the other steps the step rule can choose (one to eight base tiles' sides, regions
/// four steps wide; paper Table XXX): every universe against the floor and the choice in hindsight.
/// It asserts only that every policy ends in the same state; what it found is in `implementation.md`,
/// Section 9. About three minutes.
#[test]
#[ignore]
fn price_based_controller_step_rule_scales() {
  let failures = check(&only(vec![("Hutton32b", 5, 7), ("Hutton32b", 6, 8), ("Life", 5, 7), ("Life", 7, 9)]), None);
  eprintln!("above the floor by more than 5 %:\n{}", if failures.is_empty() { "none".to_string() } else { failures.join("\n") });
}
