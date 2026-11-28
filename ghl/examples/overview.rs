//! The data of the overview figure at the head of `paper.md`, through the library's public API
//! alone (roadmap Section 8.4): the patterns as cells, the hash-consed tiles and nodes of a universe,
//! one sweep of the hit path window by window with the memo table slot by slot, miss rates over
//! time, the waves of whole-universe steps with the atlas of a floor wave, and the controller's
//! choice per region. Writes the JSON that `research/tools/ghl_diagram.py` draws; its keys equal the
//! harness's (`ca-research overview`), which `compare_overview.py` checks key by key.
//!
//! The universes run on the host here (`Config::cpu()`): the figure's panels are the host
//! recursion's structures, which the GPU's counters equal (roadmap Section 8.1).
//!
//!     cargo run --release -p ghl --example overview -- [--out research/out/overview_ghl.json]

use std::{path::PathBuf, time::Instant};

use ghl::{
  cpu::{Engine, Flat},
  inspect::Tile,
  io::{read_rle, Grid, Macrocell},
  rule::HUTTON32_COLORS,
  sched::{controller::rnd, Mode, Model, Policy, PolicyKind},
  tile::{build_table, displacements, NONE},
  Config, Pattern, Rect, Rule, Universe,
};
use rustc_hash::{FxHashMap, FxHashSet};

const B32: &[u8; 32] = b"0123456789abcdefghijklmnopqrstuv";

fn enc(cells: &[u8]) -> String {
  cells.iter().map(|&c| B32[c as usize & 31] as char).collect()
}
fn quote(s: &str) -> String {
  format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}
fn list<T: std::fmt::Display>(v: impl IntoIterator<Item = T>) -> String {
  format!("[{}]", v.into_iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
}

#[derive(Default)]
struct Obj(Vec<String>);
impl Obj {
  fn raw(&mut self, k: &str, v: impl Into<String>) -> &mut Self {
    self.0.push(format!("{}:{}", quote(k), v.into()));
    self
  }
  fn num(&mut self, k: &str, v: impl std::fmt::Display) -> &mut Self {
    self.raw(k, v.to_string())
  }
  fn text(&mut self, k: &str, v: &str) -> &mut Self {
    self.raw(k, quote(v))
  }
  fn grid(&mut self, k: &str, g: &Grid) -> &mut Self {
    let mut o = Obj::default();
    o.num("w", g.w).num("h", g.h).num("population", g.population()).text("cells", &enc(&g.cells));
    self.raw(k, o.done())
  }
  fn done(&self) -> String {
    format!("{{{}}}", self.0.join(","))
  }
}

fn research() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../research")
}

/// A pattern as dense cells (macrocells expanded and cropped to their bounding box).
fn dense(path: &std::path::Path) -> (Grid, String) {
  let loaded = Pattern::load(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
  match loaded.pattern {
    Pattern::Cells(g) => (g, loaded.rule),
    Pattern::Macrocell(m) => (mc_cells(&m), loaded.rule),
  }
}

fn mc_cells(m: &Macrocell) -> Grid {
  let rule = Rule::named("Hutton32b").unwrap();
  let mut e = Engine::new(&rule, 1, 1);
  let (root, level) = e.from_mc(m);
  let full = e.expand(root, level);
  let (x0, y0, x1, y1) = full.bbox().expect("empty pattern");
  full.crop(x0, y0, x1, y1)
}

fn cpu_universe(rule: &Rule, a: u32) -> Universe {
  Universe::new(rule.clone(), Config { base_tile_log2: Some(a), ..Config::cpu() }).unwrap()
}

/// `g` after `gens` generations, cropped to its bounding box, and the offset of its top-left cell.
fn advance(g: &Grid, rule: &Rule, gens: u64) -> (Grid, i64, i64) {
  let mut u = cpu_universe(rule, 3);
  u.load(&Pattern::Cells(g.clone()), (0, 0)).unwrap();
  u.step(gens);
  let bb = u.bounding_box().expect("the pattern died");
  let mut out = Grid::new(bb.w as usize, bb.h as usize);
  u.read_region(bb, &mut out.cells);
  (out, bb.x, bb.y)
}

fn thumb(g: &Grid, block: usize) -> Obj {
  let (tw, th) = (g.w.div_ceil(block), g.h.div_ceil(block));
  let mut hex = String::with_capacity(tw * th * 6);
  for ty in 0..th {
    for tx in 0..tw {
      let (mut sum, mut n) = ([0u64; 3], 0u64);
      for y in ty * block..((ty + 1) * block).min(g.h) {
        for x in tx * block..((tx + 1) * block).min(g.w) {
          let c = HUTTON32_COLORS[g.get(x, y) as usize & 31];
          for i in 0..3 {
            sum[i] += c[i] as u64;
          }
          n += 1;
        }
      }
      for s in sum {
        hex.push_str(&format!("{:02x}", (s + n / 2) / n));
      }
    }
  }
  let mut o = Obj::default();
  o.num("w", tw).num("h", th).num("block", block).text("rgb", &hex);
  o
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

/// The harness's `life-soup`: each cell alive with probability `density`, from `seed`.
fn soup(side: usize, density: f64, seed: u32) -> Grid {
  let mut g = Grid::new(side, side);
  let thr = (density * 4294967296.0).min(4294967295.0) as u64;
  for (i, c) in g.cells.iter_mut().enumerate() {
    *c = ((hash32(hash32(i as u32) ^ seed.wrapping_mul(0x9E3779B9)) as u64) < thr) as u8;
  }
  g
}

// ---- the hash-consed structure of one universe, through the inspector -------------------------------

fn structures(g: &Grid, rule: &Rule, a: u32) -> String {
  let mut u = cpu_universe(rule, a);
  u.load(&Pattern::Cells(g.clone()), (0, 0)).unwrap();
  let obs = u.observer();
  let mut level = a + 1;
  while (1usize << level) < g.w.max(g.h) {
    level += 1;
  }
  let side = 1u64 << level;
  let all = Rect::new(0, 0, side, side);
  let root = obs.ids(all, level).wait()[0];
  let mut o = Obj::default();
  o.num("a", a).num("tile", 1usize << a).num("root_level", level).num("root", root);
  let mut levels = Vec::new();
  for l in a..=level {
    let ids = obs.ids(all, l).wait();
    let info = obs.level_info(l);
    let z = info.empty.expect("the empty node exists");
    let mut distinct: Vec<u32> = ids.iter().copied().filter(|&i| i != z).collect();
    let nonempty = distinct.len();
    distinct.sort_unstable();
    distinct.dedup();
    let mut lo = Obj::default();
    lo.num("level", l).num("side", 1usize << l).num("n", 1usize << (level - l)).num("empty", z);
    lo.num("nonempty", nonempty).num("distinct", distinct.len()).raw("ids", list(ids.iter()));
    if l > a {
      let nodes: Vec<String> = distinct
        .iter()
        .map(|&id| match obs.tile(l, id).wait() {
          Some(Tile::Node { children, .. }) => format!("{}:{}", quote(&id.to_string()), list(children)),
          other => panic!("level {l} node {id}: {other:?}"),
        })
        .collect();
      lo.raw("nodes", format!("{{{}}}", nodes.join(",")));
    } else {
      let mut count: FxHashMap<u32, (u32, usize)> = Default::default();
      for (pos, &id) in ids.iter().enumerate() {
        if id != z {
          let c = count.entry(id).or_insert((0, pos));
          c.0 += 1;
        }
      }
      let mut store: Vec<(u32, u32, usize)> = count.into_iter().map(|(id, (n, first))| (id, n, first)).collect();
      store.sort_by_key(|&(_, n, first)| (std::cmp::Reverse(n), first));
      let tiles: Vec<String> = store
        .iter()
        .map(|&(id, n, _)| {
          let cells = match obs.tile(a, id).wait() {
            Some(Tile::Cells { cells, .. }) => cells,
            other => panic!("tile {id}: {other:?}"),
          };
          let mut t = Obj::default();
          t.num("id", id).num("count", n).text("cells", &enc(&cells));
          t.done()
        })
        .collect();
      lo.raw("tiles", format!("[{}]", tiles.join(",")));
    }
    levels.push(lo.done());
  }
  o.raw("levels", format!("[{}]", levels.join(",")));
  o.done()
}

// ---- the positional sweep: one sweep in detail, the learning curve, the memo table ------------------

struct SweepRun {
  series: Vec<[u64; 3]>,
  snapshot: Option<String>,
  table: String,
  period: Option<u64>,
  nx: usize,
  ny: usize,
  nonempty: usize,
  w: usize,
  h: usize,
}

fn step_grid(rule: &Rule, g: &Grid) -> Grid {
  let padded = g.padded(1, 1, g.w + 2, g.h + 2);
  let mut u = cpu_universe(rule, 3);
  u.load(&Pattern::Cells(padded), (0, 0)).unwrap();
  u.step(1);
  let mut out = Grid::new(g.w, g.h);
  u.read_region(Rect::new(1, 1, g.w as u64, g.h as u64), &mut out.cells);
  out
}

fn sweep_run(g: &Grid, rule: &Rule, a: usize, lt: usize, gens: u64, snap_at: Option<u64>, with_table: bool) -> SweepRun {
  let s = 1usize << lt;
  let mut e = Engine::new(rule, a, (lt - 1).max(a));
  let mut f = Flat::from_grid(&mut e, g, lt, 2);
  let (nx, ny) = (f.nx, f.ny);
  let z = e.empty(lt);
  let nonempty = f.a_ids.iter().filter(|&&i| i != z).count();
  let sweeps = gens / (s as u64 / 2);
  let mut series = Vec::with_capacity(sweeps as usize);
  let mut snapshot = None;
  let mut seen: FxHashMap<Vec<u32>, u64> = Default::default();
  let mut period = None;
  for sw in 0..sweeps {
    if !f.in_b && period.is_none() {
      if let Some(&prev) = seen.get(&f.a_ids) {
        period = Some((sw - prev) * (s as u64 / 2));
      } else {
        seen.insert(f.a_ids.clone(), sw);
      }
    }
    let snap = snap_at == Some(sw);
    let mut o = Obj::default();
    if snap {
      assert!(!f.in_b, "the snapshot sweep must start in phase A");
      let at = |i: isize, j: isize, a_ids: &[u32]| -> u32 {
        if i < 0 || j < 0 || i >= nx as isize || j >= ny as isize {
          z
        } else {
          a_ids[j as usize * nx + i as usize]
        }
      };
      let mut miss = Vec::with_capacity((nx + 1) * (ny + 1));
      for j in 0..=ny as isize {
        for i in 0..=nx as isize {
          let key = e.intern(lt + 1, [at(i - 1, j - 1, &f.a_ids), at(i, j - 1, &f.a_ids), at(i - 1, j, &f.a_ids), at(i, j, &f.a_ids)]);
          miss.push((e.result(lt + 1, key) == NONE) as u8);
        }
      }
      o.num("generation", sw * (s as u64 / 2)).num("tile", s).num("nx", nx).num("ny", ny).num("empty", z).num("margin", 2 * s);
      let cells = f.to_grid(&e);
      let next = step_grid(rule, &cells);
      o.raw("a_ids", list(f.a_ids.iter())).raw("miss", list(miss.iter())).grid("cells", &cells).grid("cells_next", &next);
    }
    let st = f.sweep(&mut e);
    series.push([st.windows, st.unique_windows, st.top_misses]);
    if snap {
      o.raw("b_ids", list(f.b_ids.iter())).num("windows", st.windows).num("unique", st.unique_windows).num("misses", st.top_misses);
      let mut ids: Vec<u32> = f.a_ids.iter().chain(f.b_ids.iter()).copied().collect();
      ids.sort_unstable();
      ids.dedup();
      let tiles: Vec<String> = ids
        .iter()
        .map(|&id| {
          let cells = if lt == e.a { e.tile_cells(id) } else { e.expand(id, lt).cells };
          format!("{}:{}", quote(&id.to_string()), quote(&enc(&cells)))
        })
        .collect();
      o.raw("tiles", format!("{{{}}}", tiles.join(",")));
      snapshot = Some(o.done());
    }
  }
  let mut t = Obj::default();
  if with_table {
    let top = &e.levels[lt + 1];
    let n_entries = top.res.iter().filter(|&&r| r != NONE).count();
    let (table, mask, avg_probe) = build_table(top.nodes.iter().zip(&top.res).filter(|(_, &r)| r != NONE).map(|(k, &r)| (*k, r)), n_entries, 10);
    let mut hist = [0u64; 32];
    let slots: String = displacements(&table, mask)
      .map(|d| match d {
        None => '.',
        Some(d) => {
          let d = d.min(31) as usize;
          hist[d] += 1;
          B32[d] as char
        }
      })
      .collect();
    t.num("entries", n_entries).num("slots", mask as u64 + 1).num("avg_probes", format!("{avg_probe:.4}")).num("entry_bytes", 32);
    t.num("tiles", if lt == e.a { e.num_tiles() } else { e.levels[lt].nodes.len() });
    t.text("map", &slots[..slots.len().min(16384)]).raw("displacement", list(hist.iter()));
    t.num("generations", sweeps * (s as u64 / 2)).num("tile", s).num("floor_window", 2usize << a).num("base_tiles", e.num_tiles());
    t.num("windows", series.iter().map(|x| x[0]).sum::<u64>()).num("unique", series.iter().map(|x| x[1]).sum::<u64>());
    t.num("misses", series.iter().map(|x| x[2]).sum::<u64>()).num("brute_windows", e.stats.brute_calls).num("brute_cell_evals", e.stats.brute_cell_evals);
    t.num("delivered", (nx * s) as u64 * (ny * s) as u64 * sweeps * (s as u64 / 2));
  }
  SweepRun { series, snapshot, table: t.done(), period, nx, ny, nonempty, w: g.w, h: g.h }
}

fn source(o: &mut Obj, file: &str, rule: &str) {
  o.text("file", file).text("rule", rule);
}

fn series_json(r: &SweepRun, lt: usize, file: &str, rule: &str) -> String {
  let mut o = Obj::default();
  source(&mut o, file, rule);
  o.num("tile", 1usize << lt).num("gens_per_sweep", 1usize << (lt - 1)).num("nx", r.nx).num("ny", r.ny).num("nonempty_tiles", r.nonempty);
  o.num("w", r.w).num("h", r.h);
  match r.period {
    Some(p) => o.num("period", p),
    None => o.raw("period", "null"),
  };
  o.raw("windows", list(r.series.iter().map(|s| s[0])));
  o.raw("unique", list(r.series.iter().map(|s| s[1])));
  o.raw("misses", list(r.series.iter().map(|s| s[2])));
  o.done()
}

// ---- the level-synchronous recursion: waves of whole-universe steps ----------------------------------

#[allow(clippy::too_many_arguments)]
fn waves(g: &Grid, rule: &Rule, file: &str, rname: &str, a: usize, k: usize, steps: u64, logged: &[u64], atlas_max: usize) -> String {
  let mut e = Engine::new(rule, a, k);
  let (mut root, mut level) = e.from_grid(g);
  e.reset_stats();
  let side = 2usize << a;
  let mut out = Vec::new();
  for i in 1..=steps {
    let log = logged.contains(&i);
    if log {
      e.wave_log = Some(Vec::new());
      e.brute_waves.clear();
    }
    let (r, l, _) = e.step_root(root, level);
    root = r;
    level = l;
    if log {
      let w = e.wave_log.take().unwrap();
      let mut o = Obj::default();
      let rows: Vec<String> = w.iter().map(|x| list(x.iter().map(|&v| if v == NONE { -1 } else { v as i64 }))).collect();
      o.num("step", i).num("generation", (i - 1) << k).raw("waves", format!("[{}]", rows.join(",")));
      let floor: Vec<&Vec<u32>> = e.brute_waves.iter().filter(|b| b.level as usize == a + 1).map(|b| &b.ids).collect();
      if let Some(big) = floor.iter().copied().max_by_key(|m| m.len()) {
        let wins: Vec<String> = big
          .iter()
          .take(atlas_max)
          .map(|&id| {
            let cells = e.expand(id, a + 1).cells;
            let mut wo = Obj::default();
            wo.text("window", &enc(&cells)).text("result", &enc(&e.tile_cells(e.result(a + 1, id))));
            wo.done()
          })
          .collect();
        let mut ao = Obj::default();
        ao.num("side", side).num("generations", side / 4).num("misses", big.len()).raw("windows", format!("[{}]", wins.join(",")));
        o.raw("atlas", ao.done());
      }
      out.push(o.done());
    }
  }
  let mut o = Obj::default();
  source(&mut o, file, rname);
  o.num("a", a).num("k", k).num("step_gens", 1u64 << k).num("steps", steps).num("floor_window", side);
  let totals: Vec<String> = e.levels.iter().enumerate().skip(a + 1).map(|(l, lv)| list([l as u64, lv.stats.requests, lv.stats.misses, lv.stats.waves, lv.stats.max_miss_batch])).collect();
  o.raw("totals", format!("[{}]", totals.join(","))).raw("steps_logged", format!("[{}]", out.join(",")));
  o.done()
}

// ---- the controller on the composite universe ------------------------------------------------------

/// `file@x,y[+generations run before][~seed of a perturbation];...`, paths relative to research/.
fn compose(spec: &str, rule: &Rule) -> Grid {
  let mut items = Vec::new();
  for item in spec.split(';').filter(|s| !s.trim().is_empty()) {
    let (path, rest) = item.trim().split_once('@').expect("compose item needs @x,y");
    let (pos, seed) = match rest.split_once('~') {
      Some((p, s)) => (p, Some(s.parse::<u32>().expect("seed"))),
      None => (rest, None),
    };
    let (pos, pre) = match pos.split_once('+') {
      Some((p, g)) => (p, g.parse::<u64>().expect("gens")),
      None => (pos, 0),
    };
    let (x, y) = pos.split_once(',').expect("x,y");
    let (x, y) = (x.parse::<usize>().unwrap(), y.parse::<usize>().unwrap());
    let (mut g, _) = dense(&research().join(path));
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
      let mut u = cpu_universe(rule, 3);
      u.load(&Pattern::Cells(padded.clone()), (0, 0)).unwrap();
      u.step(pre);
      let bb = u.bounding_box().unwrap();
      assert!(bb.x > 0 && bb.y > 0 && bb.x1() < padded.w as i64 && bb.y1() < padded.h as i64, "{path}: grew out of its margin during the pre-run");
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

struct ControllerRun {
  regions: Vec<ghl::sched::RegionReport>,
  adaptive_ms: f64,
  plain_ms: f64,
  hist: Vec<(String, f64)>,
  note: String,
}

/// Algorithm 3 (the library's port, `sched::Policy`) on `g`, as the harness's `controller_map`.
fn controller_map(g: &Grid, rule: &Rule, a: usize, k: usize, rl: usize, gens: u64) -> ControllerRun {
  let maxf = 4usize.clamp(a, rl - 1);
  let model = Model::default();
  let mut e = Engine::new(rule, a, k);
  let (mut root, mut level) = e.from_grid(g);
  while level < rl + 1 {
    let z = e.empty(level);
    root = e.intern(level + 1, [root, z, z, z]);
    level += 1;
  }
  let (mut ox, mut oy) = (0i64, 0i64);
  e.reset_stats();
  let mut pol = Policy::new(PolicyKind::Adaptive, rl, a, k, Mode::Floor(a.clamp(a, maxf)), model, maxf);
  let mut nominal = 0.0;
  let mut cells: Vec<(i64, i64)> = Vec::new();
  for _ in 0..gens >> k {
    cells.clear();
    e.nonempty_at(root, level, ox, oy, rl, &mut cells);
    let nonempty: FxHashSet<(i64, i64)> = cells.iter().map(|&(x, y)| (x >> rl, y >> rl)).collect();
    nominal += nonempty.len() as f64 * pol.region_brute_nominal();
    let (r, l, x, y) = e.step_root_at(root, level, ox, oy, rl + 1, &mut pol);
    root = r;
    level = l;
    ox = x;
    oy = y;
    pol.end_step(&nonempty);
  }
  let note = if pol.trials > 0 || pol.switches > 0 { format!("  [{} mode changes; {} of {} finer trials kept]", pol.switches, pol.trials_kept, pol.trials) } else { String::new() };
  ControllerRun { adaptive_ms: (pol.total_hash + pol.total_brute) / 1e6, plain_ms: nominal / 1e6, hist: pol.mode_histogram(), note: note.trim().to_string(), regions: pol.region_report() }
}

fn main() {
  let args: Vec<String> = std::env::args().collect();
  let opt = |name: &str| args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned();
  let out: PathBuf = opt("--out").map(Into::into).unwrap_or(research().join("out/overview_ghl.json"));
  let only = opt("--sections");
  let want = |name: &str| only.as_deref().is_none_or(|s| s.split(',').any(|x| x == name));
  let p = research().join("patterns");
  let t_all = Instant::now();
  let mut t = Instant::now();
  let mut lap = |what: &str| {
    eprintln!("  [{:>5.1}s] {what}", t.elapsed().as_secs_f64());
    t = Instant::now();
  };
  let h32b = Rule::named("Hutton32b").unwrap();
  let h32 = Rule::named("Hutton32").unwrap();
  let life = Rule::named("B3/S23").unwrap();
  let mut o = Obj::default();
  o.raw("palette", format!("[{}]", HUTTON32_COLORS.iter().map(|c| list(c.iter())).collect::<Vec<_>>().join(",")));

  const COUNTER: &str = "advanced counter v2.rle";
  const CHAOTIC: &str = "misc/chaotic.rle";
  const ROM: &str = "misc/extra/lotr1.ANSI-1M.rle";
  const ROM_SMALL: &str = "misc/extra/lotr1.PPMd-244K.rle";
  const ULC: &str = "life/Unit-Life-Cell-512x512.rle";
  const SQUARES: &str = "misc/squares.rle";
  const SOUP: &str = "life-soup";
  let rd = |f: &str| read_rle(&std::fs::read_to_string(p.join(f)).unwrap()).unwrap().0;
  let counter = rd(COUNTER);
  let chaotic = rd(CHAOTIC);
  let rom = rd(ROM);
  let ppmd = rd(ROM_SMALL);
  let ulc = rd(ULC);
  let squares = rd(SQUARES);
  let cha_gen: u64 = opt("--chaotic-gen").map(|s| s.parse().unwrap()).unwrap_or(8192);
  let soup_gen: usize = opt("--soup-gen").map(|s| s.parse().unwrap()).unwrap_or(128);

  let mut pats = Obj::default();
  if want("patterns") {
    let mut c = Obj::default();
    c.text("file", COUNTER).text("rule", "Hutton32b").grid("gen0", &counter);
    c.grid("gen1", &step_grid(&h32b, &counter));
    pats.raw("counter", c.done());

    let mut c = Obj::default();
    c.text("file", SQUARES).text("rule", "Hutton32").grid("gen0", &squares);
    pats.raw("squares", c.done());

    let mut c = Obj::default();
    let (later, dx, dy) = advance(&chaotic, &h32b, cha_gen);
    c.text("file", CHAOTIC).text("rule", "Hutton32b").grid("gen0", &chaotic);
    c.num("later_gen", cha_gen).num("later_dx", dx).num("later_dy", dy).grid("later", &later);
    pats.raw("chaotic", c.done());

    let mut c = Obj::default();
    let (cw, ch) = (192usize, 160usize);
    let (cx, cy) = (rom.w / 2 - cw / 2, rom.h / 2 - ch / 2);
    c.text("file", ROM).text("rule", "Hutton32b").num("w", rom.w).num("h", rom.h).num("population", rom.population());
    c.num("crop_x", cx).num("crop_y", cy).grid("crop", &rom.crop(cx, cy, cx + cw, cy + ch));
    pats.raw("rom", c.done());

    let mut c = Obj::default();
    c.text("file", ULC).text("rule", "B3/S23").grid("gen0", &ulc);
    pats.raw("ulc", c.done());

    let mut c = Obj::default();
    let (side, view) = (2048usize, 256usize);
    let sp = soup(side, 0.5, 1);
    let w0 = side / 2 - view / 2 - soup_gen;
    let win = sp.crop(w0, w0, w0 + view + 2 * soup_gen, w0 + view + 2 * soup_gen);
    let mut u = cpu_universe(&life, 4);
    u.load(&Pattern::Cells(win.clone()), (0, 0)).unwrap();
    u.step(soup_gen as u64);
    let mut later = Grid::new(view, view);
    u.read_region(Rect::new(soup_gen as i64, soup_gen as i64, view as u64, view as u64), &mut later.cells);
    c.text("source", SOUP).text("rule", "B3/S23").num("side", side).num("density", 0.5).num("seed", 1).num("population", sp.population());
    c.num("view_x", side / 2 - view / 2).num("view_gen", soup_gen);
    c.grid("gen0", &sp.crop(side / 2 - view / 2, side / 2 - view / 2, side / 2 + view / 2, side / 2 + view / 2));
    c.grid("later", &later);
    pats.raw("soup", c.done());
  }
  o.raw("patterns", pats.done());
  lap("patterns");

  if want("tiles") {
    o.raw("structures", structures(&counter, &h32b, 3));
    o.raw("structures_life", structures(&ulc, &life, 4));
    lap("tiles and nodes");
  }

  let snap_gen: u64 = opt("--snapshot-gen").map(|s| s.parse().unwrap()).unwrap_or(20480);
  let sweep_gens: u64 = opt("--sweep-gens").map(|s| s.parse().unwrap()).unwrap_or(65536);
  if want("sweep") || want("learning") {
    let run = sweep_run(&squares, &h32, 3, 4, sweep_gens, Some(snap_gen / 16 * 2), true);
    o.raw("sweep", run.snapshot.clone().expect("no snapshot"));
    o.raw("table", run.table.clone());
    lap("sweep and memo table");
    if want("learning") {
      let mut learn = Obj::default();
      learn.raw("squares", series_json(&run, 4, SQUARES, "Hutton32"));
      learn.raw("counter", series_json(&sweep_run(&counter, &h32b, 3, 4, 262144, None, false), 4, COUNTER, "Hutton32b"));
      learn.raw("chaotic", series_json(&sweep_run(&chaotic, &h32b, 3, 4, 16384, None, false), 4, CHAOTIC, "Hutton32b"));
      learn.raw("rom", series_json(&sweep_run(&ppmd, &h32b, 3, 4, 1584, None, false), 4, ROM_SMALL, "Hutton32b"));
      let ulc_run = sweep_run(&ulc, &life, 4, 5, 17280, None, true);
      learn.raw("ulc", series_json(&ulc_run, 5, ULC, "B3/S23"));
      let soup512 = soup(512, 0.5, 1);
      learn.raw("soup", series_json(&sweep_run(&soup512, &life, 4, 5, 1024, None, false), 5, SOUP, "B3/S23"));
      o.raw("learning", learn.done());
      o.raw("table_life", ulc_run.table.clone());
      lap("miss rates over time");
    }
  }

  if want("waves") {
    let mut w = Obj::default();
    let at = snap_gen / 64 + 1;
    w.raw("squares", waves(&squares, &h32, SQUARES, "Hutton32", 3, 6, at, &[1, at], 400));
    w.raw("counter", waves(&counter, &h32b, COUNTER, "Hutton32b", 3, 6, 64, &[1, 2, 64], 160));
    w.raw("chaotic", waves(&chaotic, &h32b, CHAOTIC, "Hutton32b", 3, 6, 64, &[1, 2, 64], 160));
    w.raw("rom", waves(&ppmd, &h32b, ROM_SMALL, "Hutton32b", 3, 6, 1, &[1], 160));
    o.raw("waves", w.done());
    lap("waves");
  }

  if want("controller") {
    let spec = std::fs::read_to_string(research().join("tools/composite_spec.txt")).expect("tools/composite_spec.txt");
    let parts: Vec<String> = spec
      .trim()
      .split(';')
      .filter(|item| !item.trim().is_empty())
      .map(|item| {
        let (path, rest) = item.trim().split_once('@').expect("compose item needs @x,y");
        let (x, y) = rest.split(['~', '+']).next().unwrap().split_once(',').expect("x,y");
        let mut po = Obj::default();
        po.text("file", path.trim_start_matches("patterns/")).num("x", x).num("y", y);
        po.done()
      })
      .collect();
    let gens: u64 = opt("--controller-gens").map(|s| s.parse().unwrap()).unwrap_or(1024);
    let g = compose(&spec, &h32b);
    let (a, k, rl) = (3usize, 4usize, 8usize);
    let m = controller_map(&g, &h32b, a, k, rl, gens);
    let mut c = Obj::default();
    c.text("rule", "Hutton32b").raw("parts", format!("[{}]", parts.join(",")));
    c.num("w", g.w).num("h", g.h).num("population", g.population()).num("region", 1usize << rl).num("step_gens", 1usize << k).num("gens", gens);
    c.num("adaptive_ms", format!("{:.4}", m.adaptive_ms)).num("plain_ms", format!("{:.4}", m.plain_ms)).text("note", &m.note);
    c.raw("hist", format!("[{}]", m.hist.iter().map(|(n, s)| format!("[{},{s:.3}]", quote(n))).collect::<Vec<_>>().join(",")));
    c.raw("thumb", thumb(&g, 16).done());
    let regs: Vec<String> = m
      .regions
      .iter()
      .map(|r| {
        let mut ro = Obj::default();
        ro.num("x", r.x).num("y", r.y).raw("steps", format!("[{}]", r.steps.iter().map(|s| list([s.0, s.1])).collect::<Vec<_>>().join(",")));
        ro.num("hash_ns", format!("{:.1}", r.hash_ns)).num("cells_ns", format!("{:.1}", r.cells_ns));
        ro.done()
      })
      .collect();
    c.raw("regions", format!("[{}]", regs.join(",")));
    o.raw("controller", c.done());
    lap("controller");
  }

  let text = o.done();
  std::fs::write(&out, &text).unwrap_or_else(|e| panic!("failed to write {}: {e}", out.display()));
  println!("wrote {} ({} KiB) in {:.1}s", out.display(), text.len() >> 10, t_all.elapsed().as_secs_f64());
}
