//! `overview`: the data behind the overview diagram at the head of `paper.md`
//! (`tools/ghl_diagram.py` draws it). Everything here is computed on the CPU by the reference
//! engine and is deterministic: the patterns as cells, the hash-consed tiles and nodes of a
//! universe, a memo table slot by slot, one sweep of the hit path window by window, the waves of
//! the level-synchronous recursion with the windows a floor wave brute-forces, miss rates over
//! time, and the controller's choice per region. Writes one JSON file.

use crate::{
  cmd_adapt, cmd_gpu, cmd_life, cmd_stats,
  cmd_stats::Loaded,
  flat::Flat,
  hl::{Engine, NONE},
  opt, opt_or,
  pattern::{Grid, HUTTON32_COLORS},
  rulespec::RuleSpec,
  sim::Sim,
};
use std::{path::Path, time::Instant};

const B32: &[u8; 32] = b"0123456789abcdefghijklmnopqrstuv";

/// One character per cell: its state in base 32, row-major.
fn enc(cells: &[u8]) -> String {
  cells.iter().map(|&c| B32[c as usize & 31] as char).collect()
}
fn quote(s: &str) -> String {
  format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}
fn list<T: std::fmt::Display>(v: impl IntoIterator<Item = T>) -> String {
  format!("[{}]", v.into_iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
}

/// A JSON object under construction.
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

fn dense(path: &Path) -> (Grid, String) {
  match cmd_stats::load_pattern(path) {
    (Loaded::Dense(g), r) => (g, r),
    _ => panic!("{}: overview reads dense patterns", path.display()),
  }
}

/// `g` after `gens` generations on the plane, cropped to its bounding box. `(dx, dy)` is the
/// offset of the result's top-left cell from that of `g`.
fn advance(g: &Grid, lut: &crate::ca::Rule, gens: u64, margin: usize) -> (Grid, i64, i64) {
  let padded = g.padded(margin, margin, g.w + 2 * margin, g.h + 2 * margin);
  let mut sim = Sim::new(&padded, lut);
  for _ in 0..gens {
    sim.step();
  }
  assert!(!sim.touches_border(), "the pattern grew out of a margin of {margin} cells");
  let out = sim.grid();
  let (x0, y0, x1, y1) = out.bbox().expect("the pattern died");
  (out.crop(x0, y0, x1, y1), x0 as i64 - margin as i64, y0 as i64 - margin as i64)
}

/// Mean palette colour of every `block` x `block` square, as `rrggbb` per square.
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

// ---- hash-consed structure of one universe ------------------------------------------------------

/// The universe `g` as base tiles of side `2^a` and nodes above them: the positional grid of ids
/// at every level, every distinct base tile with its number of occurrences, and the children of
/// every node.
fn structures(g: &Grid, lut: &crate::ca::Rule, a: usize) -> String {
  let mut e = Engine::new(lut, a, a);
  let (root, level) = e.from_grid(g);
  let mut o = Obj::default();
  o.num("a", a).num("tile", 1usize << a).num("root_level", level).num("root", root);
  let mut levels = Vec::new();
  for l in a..=level {
    let ids = e.ids_at_level(root, level, l);
    let z = e.empty(l);
    let mut distinct: Vec<u32> = ids.iter().copied().filter(|&i| i != z).collect();
    let nonempty = distinct.len();
    distinct.sort_unstable();
    distinct.dedup();
    let mut lo = Obj::default();
    lo.num("level", l).num("side", 1usize << l).num("n", 1usize << (level - l)).num("empty", z);
    lo.num("nonempty", nonempty).num("distinct", distinct.len()).raw("ids", list(ids.iter()));
    if l > a {
      let nodes: Vec<String> = distinct.iter().map(|&id| format!("{}:{}", quote(&id.to_string()), list(e.levels[l].nodes[id as usize]))).collect();
      lo.raw("nodes", format!("{{{}}}", nodes.join(",")));
    } else {
      // the tile store: every distinct tile, most frequent first
      let mut count: rustc_hash::FxHashMap<u32, (u32, usize)> = Default::default();
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
          let mut t = Obj::default();
          t.num("id", id).num("count", n).text("cells", &enc(e.tile_cells(id)));
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

// ---- the flat tile automaton: one sweep in detail, the learning curve, the memo table ----------

struct SweepRun {
  series: Vec<[u64; 3]>,
  snapshot: Option<String>,
  table: String,
  period: Option<u64>,
  nx: usize,
  ny: usize,
  nonempty: usize,
  /// the pattern's own size in cells
  w: usize,
  h: usize,
}

/// Sweep `g` at tile level `lt` for `gens` generations. With `snap_at = Some(n)` the sweep after
/// `n` sweeps (n even, so the grid is in phase A) is recorded window by window.
fn sweep_run(g: &Grid, lut: &crate::ca::Rule, a: usize, lt: usize, gens: u64, snap_at: Option<u64>, with_table: bool) -> SweepRun {
  let s = 1usize << lt;
  let mut e = Engine::new(lut, a, (lt - 1).max(a));
  let mut f = Flat::from_grid(&mut e, g, lt, 2);
  let (nx, ny) = (f.nx, f.ny);
  let z = e.empty(lt);
  let nonempty = f.a_ids.iter().filter(|&&i| i != z).count();
  let sweeps = gens / (s as u64 / 2);
  let mut series = Vec::with_capacity(sweeps as usize);
  let mut snapshot = None;
  let mut seen: rustc_hash::FxHashMap<Vec<u32>, u64> = Default::default();
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
      // the keys of the sweep, as `Flat::sweep` forms them, and whether each is memoized
      let at = |i: isize, j: isize| -> u32 {
        if i < 0 || j < 0 || i >= nx as isize || j >= ny as isize { z } else { f.a_ids[j as usize * nx + i as usize] }
      };
      let mut miss = Vec::with_capacity((nx + 1) * (ny + 1));
      for j in 0..=ny as isize {
        for i in 0..=nx as isize {
          let key = e.intern(lt + 1, [at(i - 1, j - 1), at(i, j - 1), at(i - 1, j), at(i, j)]);
          miss.push((e.result(lt + 1, key) == NONE) as u8);
        }
      }
      // the pattern's top-left cell is `margin` cells from the grid's
      o.num("generation", sw * (s as u64 / 2)).num("tile", s).num("nx", nx).num("ny", ny).num("empty", z).num("margin", 2 * s);
      let cells = f.to_grid(&e);
      let mut sim = Sim::new(&cells, lut);
      sim.step();
      o.raw("a_ids", list(f.a_ids.iter())).raw("miss", list(miss.iter())).grid("cells", &cells).grid("cells_next", &sim.grid());
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
          let cells = if lt == e.a { e.tile_cells(id).to_vec() } else { e.expand(id, lt).cells };
          format!("{}:{}", quote(&id.to_string()), quote(&enc(&cells)))
        })
        .collect();
      o.raw("tiles", format!("{{{}}}", tiles.join(",")));
      snapshot = Some(o.done());
    }
  }
  let mut t = Obj::default();
  if with_table {
    // the memo table of the sweep level as the GPU hit path holds it (`gpu-sweep`)
    let top = &e.levels[lt + 1];
    let n_entries = top.res.iter().filter(|&&r| r != NONE).count();
    let (table, mask, avg_probe) =
      cmd_gpu::build_table(top.nodes.iter().zip(&top.res).filter(|(_, &r)| r != NONE).map(|(k, &r)| (*k, r)), n_entries, 10);
    // per slot: '.' if empty, else how far the entry sits from its home slot (capped at 31);
    // `map` is the first 16,384 slots, `displacement` counts every slot of the table
    let mut hist = [0u64; 32];
    let slots: String = (0..=mask)
      .map(|sl| {
        let b = sl as usize * 8;
        if table[b] == u32::MAX && table[b + 1] == u32::MAX {
          '.'
        } else {
          let home = cmd_gpu::hash4([table[b], table[b + 1], table[b + 2], table[b + 3]]) & mask;
          let d = (sl.wrapping_sub(home) & mask).min(31) as usize;
          hist[d] += 1;
          B32[d] as char
        }
      })
      .collect();
    t.num("entries", n_entries).num("slots", mask as u64 + 1).num("avg_probes", format!("{avg_probe:.4}")).num("entry_bytes", 32);
    t.num("tiles", if lt == e.a { e.num_tiles() } else { e.levels[lt].nodes.len() });
    t.text("map", &slots[..slots.len().min(16384)]).raw("displacement", list(hist.iter()));
    // what the run cost: probes of the sweep level, brute-forced floor windows and their cell
    // evaluations, against the cell-updates delivered (the swept grid's cells times generations)
    t.num("generations", sweeps * (s as u64 / 2)).num("tile", s).num("floor_window", 2usize << a).num("base_tiles", e.num_tiles());
    t.num("windows", series.iter().map(|x| x[0]).sum::<u64>()).num("unique", series.iter().map(|x| x[1]).sum::<u64>());
    t.num("misses", series.iter().map(|x| x[2]).sum::<u64>()).num("brute_windows", e.brute_calls).num("brute_cell_evals", e.brute_cell_evals);
    t.num("delivered", (nx * s) as u64 * (ny * s) as u64 * sweeps * (s as u64 / 2));
  }
  SweepRun { series, snapshot, table: t.done(), period, nx, ny, nonempty, w: g.w, h: g.h }
}

/// The fields `file` (relative to patterns/) and `rule` of a run, for the figure's labels.
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

// ---- the level-synchronous recursion: waves of whole-universe steps ------------------------------

/// Step `g` from the root by `2^k` generations `steps` times; record the waves of the steps in
/// `logged` (1-based) and, for those steps, the windows of the largest floor wave.
#[allow(clippy::too_many_arguments)]
fn waves(g: &Grid, lut: &crate::ca::Rule, file: &str, rule: &str, a: usize, k: usize, steps: u64, logged: &[u64], atlas_max: usize) -> String {
  let mut e = Engine::new(lut, a, k);
  let (mut root, mut level) = e.from_grid(g);
  for lv in e.levels.iter_mut() {
    lv.stats = Default::default();
  }
  let side = 2usize << a;
  let mut out = Vec::new();
  for i in 1..=steps {
    let log = logged.contains(&i);
    if log {
      e.wave_log = Some(Vec::new());
      e.floor_misses.clear();
    }
    let (r, l, _) = e.step_root(root, level);
    root = r;
    level = l;
    if log {
      let w = e.wave_log.take().unwrap();
      let mut o = Obj::default();
      // [level, distinct requests, misses, parent wave or -1]
      let rows: Vec<String> = w.iter().map(|x| list(x.iter().map(|&v| if v == NONE { -1 } else { v as i64 }))).collect();
      o.num("step", i).num("generation", (i - 1) << k).raw("waves", format!("[{}]", rows.join(",")));
      // the atlas: the missed floor windows of the largest floor wave, and their results
      if let Some(big) = e.floor_misses.iter().max_by_key(|m| m.len()) {
        let wins: Vec<String> = big
          .iter()
          .take(atlas_max)
          .map(|&id| {
            let ch = e.levels[a + 1].nodes[id as usize];
            let ts = side / 2;
            let mut cells = vec![0u8; side * side];
            for (q, &t) in ch.iter().enumerate() {
              let (ox, oy) = ((q & 1) * ts, (q >> 1) * ts);
              let src = e.tile_cells(t);
              for y in 0..ts {
                cells[(oy + y) * side + ox..(oy + y) * side + ox + ts].copy_from_slice(&src[y * ts..(y + 1) * ts]);
              }
            }
            let mut wo = Obj::default();
            wo.text("window", &enc(&cells)).text("result", &enc(e.tile_cells(e.result(a + 1, id))));
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
  source(&mut o, file, rule);
  o.num("a", a).num("k", k).num("step_gens", 1u64 << k).num("steps", steps).num("floor_window", side);
  let totals: Vec<String> = e
    .levels
    .iter()
    .enumerate()
    .skip(a + 1)
    .map(|(l, lv)| list([l as u64, lv.stats.requests, lv.stats.misses, lv.stats.waves, lv.stats.max_miss_batch]))
    .collect();
  // per level over all steps: [level, distinct requests, misses, waves, largest miss batch]
  o.raw("totals", format!("[{}]", totals.join(","))).raw("steps_logged", format!("[{}]", out.join(",")));
  o.done()
}

// ---- command --------------------------------------------------------------------------------------

pub fn run(args: &[String]) {
  let out: std::path::PathBuf = opt(args, "--out").map(Into::into).unwrap_or(crate::out_dir().join("overview.json"));
  let p = crate::patterns_dir();
  let t_all = Instant::now();
  let mut t = Instant::now();
  let mut lap = |what: &str| {
    eprintln!("  [{:>5.1}s] {what}", t.elapsed().as_secs_f64());
    t = Instant::now();
  };
  // `--sections a,b`: only those (patterns, tiles, sweep, learning, waves, controller); the figure needs all
  let only = opt(args, "--sections");
  let want = |name: &str| only.as_deref().is_none_or(|s| s.split(',').any(|x| x == name));
  let h32b = crate::load_lut(&RuleSpec::parse("hutton32b").unwrap());
  let life = crate::load_lut(&RuleSpec::parse("B3/S23").unwrap());
  let mut o = Obj::default();
  o.raw("palette", format!("[{}]", HUTTON32_COLORS.iter().map(|c| list(c.iter())).collect::<Vec<_>>().join(",")));

  // ---- the universes of the evaluation, as cells; files relative to patterns/ ----
  const COUNTER: &str = "advanced counter v2.rle";
  const CHAOTIC: &str = "misc/chaotic.rle";
  const ROM: &str = "misc/extra/lotr1.ANSI-1M.rle";
  const ROM_SMALL: &str = "misc/extra/lotr1.PPMd-244K.rle";
  const ULC: &str = "life/Unit-Life-Cell-512x512.rle";
  const SQUARES: &str = "misc/squares.rle";
  const SOUP: &str = "life-soup"; // not a file: `life-soup --side S --seed 1`
  let (counter, _) = dense(&p.join(COUNTER));
  let (chaotic, _) = dense(&p.join(CHAOTIC));
  let (rom, _) = dense(&p.join(ROM));
  let (ppmd, _) = dense(&p.join(ROM_SMALL));
  let (ulc, _) = dense(&p.join(ULC));
  let (squares, _) = dense(&p.join(SQUARES));
  let h32 = crate::load_lut(&RuleSpec::parse("hutton32").unwrap());
  let cha_gen: u64 = opt_or(args, "--chaotic-gen", 8192);
  let soup_gen: usize = opt_or(args, "--soup-gen", 128);
  let mut pats = Obj::default();
  if want("patterns") {
    let mut c = Obj::default();
    c.text("file", COUNTER).text("rule", "Hutton32b").grid("gen0", &counter);
    // one generation later, on the pattern's own bounding box: checks the netlist in the figure
    let (x0, y0) = (1usize, 1usize);
    let padded = counter.padded(x0, y0, counter.w + 2, counter.h + 2);
    let mut sim = Sim::new(&padded, &h32b);
    sim.step();
    c.grid("gen1", &sim.grid().crop(x0, y0, x0 + counter.w, y0 + counter.h));
    pats.raw("counter", c.done());

    let mut c = Obj::default();
    c.text("file", SQUARES).text("rule", "Hutton32").grid("gen0", &squares);
    pats.raw("squares", c.done());

    let mut c = Obj::default();
    let (later, dx, dy) = advance(&chaotic, &h32b, cha_gen, 64);
    c.text("file", CHAOTIC).text("rule", "Hutton32b").grid("gen0", &chaotic);
    c.num("later_gen", cha_gen).num("later_dx", dx).num("later_dy", dy).grid("later", &later);
    pats.raw("chaotic", c.done());

    let mut c = Obj::default();
    // a window of the ROM at one cell per pixel (the whole pattern is a uniform texture)
    let (cw, ch) = (192usize, 160usize);
    let (cx, cy) = (rom.w / 2 - cw / 2, rom.h / 2 - ch / 2);
    c.text("file", ROM).text("rule", "Hutton32b").num("w", rom.w).num("h", rom.h).num("population", rom.population());
    c.num("crop_x", cx).num("crop_y", cy).grid("crop", &rom.crop(cx, cy, cx + cw, cy + ch));
    pats.raw("rom", c.done());

    let mut c = Obj::default();
    c.text("file", ULC).text("rule", "B3/S23").grid("gen0", &ulc);
    pats.raw("ulc", c.done());

    // the soup of the Life targets (`life-soup --side 2048 --seed 1`): a window of it after
    // `soup_gen` generations, exact because the window keeps a margin of that many cells
    let mut c = Obj::default();
    let (side, view) = (2048usize, 256usize);
    let soup = cmd_life::soup(side, 0.5, 1);
    let w0 = side / 2 - view / 2 - soup_gen;
    let win = soup.crop(w0, w0, w0 + view + 2 * soup_gen, w0 + view + 2 * soup_gen);
    let mut sim = Sim::new(&win, &life);
    for _ in 0..soup_gen {
      sim.step();
    }
    c.text("source", SOUP).text("rule", "B3/S23").num("side", side).num("density", 0.5).num("seed", 1).num("population", soup.population());
    c.num("view_x", side / 2 - view / 2).num("view_gen", soup_gen);
    c.grid("gen0", &soup.crop(side / 2 - view / 2, side / 2 - view / 2, side / 2 + view / 2, side / 2 + view / 2));
    c.grid("later", &sim.grid().crop(soup_gen, soup_gen, soup_gen + view, soup_gen + view));
    pats.raw("soup", c.done());
  }
  o.raw("patterns", pats.done());
  lap("patterns");

  // ---- data structures: the counter as 8x8 base tiles, Hutton32b's base tile ----
  if want("tiles") {
    o.raw("structures", structures(&counter, &h32b, 3));
    o.raw("structures_life", structures(&ulc, &life, 4));
    lap("tiles and nodes");
  }

  // ---- hit path: `squares` swept at 16x16 tiles over the 65,536 generations the paper times; the
  // sweep that starts at `--snapshot-gen` window by window (at 20,480 its displays read 00025);
  // the memo table at the end ----
  let snap_gen: u64 = opt_or(args, "--snapshot-gen", 20480);
  let sweep_gens: u64 = opt_or(args, "--sweep-gens", 65536);
  if want("sweep") || want("learning") {
    let run = sweep_run(&squares, &h32, 3, 4, sweep_gens, Some(snap_gen / 16 * 2), true);
    o.raw("sweep", run.snapshot.clone().expect("no snapshot"));
    o.raw("table", run.table.clone());
    lap("sweep and memo table");

    // ---- miss rates over time at each rule's floor tile ----
    if want("learning") {
      let mut learn = Obj::default();
      learn.raw("squares", series_json(&run, 4, SQUARES, "Hutton32"));
      learn.raw("counter", series_json(&sweep_run(&counter, &h32b, 3, 4, 262144, None, false), 4, COUNTER, "Hutton32b"));
      learn.raw("chaotic", series_json(&sweep_run(&chaotic, &h32b, 3, 4, 16384, None, false), 4, CHAOTIC, "Hutton32b"));
      learn.raw("rom", series_json(&sweep_run(&ppmd, &h32b, 3, 4, 1584, None, false), 4, ROM_SMALL, "Hutton32b"));
      let ulc_run = sweep_run(&ulc, &life, 4, 5, 17280, None, true);
      learn.raw("ulc", series_json(&ulc_run, 5, ULC, "B3/S23"));
      let soup512 = cmd_life::soup(512, 0.5, 1);
      learn.raw("soup", series_json(&sweep_run(&soup512, &life, 4, 5, 1024, None, false), 5, SOUP, "B3/S23"));
      o.raw("learning", learn.done());
      o.raw("table_life", ulc_run.table.clone());
      lap("miss rates over time");
    }
  }

  // ---- miss path: waves of 64-generation steps from the root, 8x8 base tiles ----
  if want("waves") {
    let mut w = Obj::default();
    let at = snap_gen / 64 + 1; // the step that starts at the snapshot's generation
    w.raw("squares", waves(&squares, &h32, SQUARES, "Hutton32", 3, 6, at, &[1, at], 400));
    w.raw("counter", waves(&counter, &h32b, COUNTER, "Hutton32b", 3, 6, 64, &[1, 2, 64], 160));
    w.raw("chaotic", waves(&chaotic, &h32b, CHAOTIC, "Hutton32b", 3, 6, 64, &[1, 2, 64], 160));
    w.raw("rom", waves(&ppmd, &h32b, ROM_SMALL, "Hutton32b", 3, 6, 1, &[1], 160));
    o.raw("waves", w.done());
    lap("waves");
  }

  // ---- the controller on the composite universe of the `controller` target ----
  if want("controller") {
    let spec = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("tools/composite_spec.txt")).expect("tools/composite_spec.txt");
    // every item is `file@x,y[+generations run before][~seed of a perturbation]`
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
    // the spec's paths are relative to research/
    let spec: String = spec.trim().split(';').map(|item| format!("{}/{}", env!("CARGO_MANIFEST_DIR"), item)).collect::<Vec<_>>().join(";");
    let gens: u64 = opt_or(args, "--controller-gens", 1024);
    let cargs = vec!["--compose".to_string(), spec];
    let (g, lut, _) = cmd_adapt::build_universe(&cargs, &|_| crate::load_lut(&RuleSpec::parse("hutton32b").unwrap()));
    let (a, k, rl) = (3usize, 4usize, 8usize);
    let m = cmd_adapt::controller_map(&g, &lut, a, k, rl, gens);
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
