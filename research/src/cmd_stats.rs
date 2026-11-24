//! CPU experiments on real patterns: self-test, redundancy / learning curves, root stepping.

use crate::{flag, flat::Flat, hl, hl::Engine, mc, opt, opt_or, pattern, pattern::Grid, rule, rulespec::RuleSpec, sim::Sim};
use std::{path::Path, time::Instant};

pub enum Loaded {
  Dense(Grid),
  Mc(mc::Macrocell),
}

/// Load a pattern by extension. Returns the pattern and the rule named in the file (if any).
pub fn load_pattern(path: &Path) -> (Loaded, String) {
  let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
  match ext.as_str() {
    "png" => (Loaded::Dense(pattern::load_png(path)), String::new()),
    "rle" => {
      let (g, r) = pattern::read_rle(path);
      (Loaded::Dense(g), r)
    }
    "mc" => {
      let m = mc::read_mc(path);
      let r = m.rule.clone();
      (Loaded::Mc(m), r)
    }
    _ => panic!("unknown pattern extension: {}", path.display()),
  }
}

/// Pick the transition function. `--rule` overrides; otherwise it is the canonical rule named in
/// the pattern file, which is what Golly runs for it (so results can be cross-checked against
/// bgolly). Patterns without a rule header, such as the PNG presets, get Hutton32b.
pub fn variant_for(args: &[String], file_rule: &str) -> RuleSpec {
  if let Some(r) = opt(args, "--rule") {
    return RuleSpec::parse(&r).expect("bad --rule (hutton32 | hutton32b | hutton32-v1 | hutton32b-v1 | table:<path>)");
  }
  match file_rule.to_ascii_lowercase().as_str() {
    "hutton32" => RuleSpec::Wgsl(rule::Variant::Hutton32),
    "hutton32b" | "" => RuleSpec::Wgsl(rule::Variant::Hutton32b),
    other => panic!("pattern uses rule {other}, which this harness does not implement"),
  }
}

fn engine_fingerprint(e: &Engine, root: u32, level: usize) -> hl::Fingerprint {
  hl::fingerprint(|f| e.for_each_cell(root, level, 0, 0, &mut |x, y, s| f(x, y, s)))
}
fn grid_fingerprint(g: &Grid) -> hl::Fingerprint {
  hl::fingerprint(|f| {
    for y in 0..g.h {
      for x in 0..g.w {
        let c = g.get(x, y);
        if c != 0 {
          f(x as i64, y as i64, c);
        }
      }
    }
  })
}

/// `selftest`: the reference engine must agree with per-cell stepping.
pub fn selftest(args: &[String]) {
  let default = crate::repo_root().join("archive/doc/hutton32b_8bit_counter.png");
  let path = opt(args, "--pattern").map(Into::into).unwrap_or(default);
  let (loaded, file_rule) = load_pattern(&path);
  let Loaded::Dense(pat) = loaded else { panic!("selftest needs a dense pattern") };
  let variant = variant_for(args, &file_rule);
  let lut = crate::load_lut(&variant);
  println!("pattern {} ({}x{}), rule {}", path.display(), pat.w, pat.h, variant.name());

  // flat sweeps
  for a in 1..=3usize {
    for lt in a..=6usize {
      let margin = 2;
      let s = 1usize << lt;
      let mut e = Engine::new(&lut, a, (lt - 1).max(a));
      let mut f = Flat::from_grid(&mut e, &pat, lt, margin);
      let padded = pat.padded(margin * s, margin * s, f.nx * s, f.ny * s);
      let mut sim = Sim::new(&padded, &lut);
      let cycles = (768 / s).max(2);
      let mut clear = true;
      for _ in 0..2 * cycles {
        f.sweep(&mut e);
        clear &= f.border_clear(&mut e);
      }
      for _ in 0..cycles * s {
        sim.step();
      }
      let ok = f.to_grid(&e) == sim.grid();
      println!("  flat a={a} lt={lt}: {} gens, border clear={clear}, match={ok}", cycles * s);
      assert!(ok && clear);
    }
  }
  // root stepping
  for (a, k) in [(1usize, 1usize), (1, 4), (2, 2), (2, 5), (2, 9), (3, 3), (3, 7)] {
    let mut e = Engine::new(&lut, a, k);
    let (mut root, mut level) = e.from_grid(&pat);
    let gens = 1024usize.max(2 << k);
    for _ in 0..gens >> k {
      let (r, l, _) = e.step_root(root, level);
      root = r;
      level = l;
    }
    let big = pat.padded(512, 512, pat.w + 1024, pat.h + 1024);
    let mut sim = Sim::new(&big, &lut);
    for _ in 0..gens {
      sim.step();
    }
    assert!(!sim.touches_border());
    let (fa, fb) = (engine_fingerprint(&e, root, level), grid_fingerprint(&sim.grid()));
    println!("  root a={a} k={k}: {gens} gens, pop={} match={}", fa.population, fa == fb);
    assert_eq!(fa, fb);
  }
  println!("selftest OK");
}

fn print_levels(e: &Engine) {
  println!("  level  side      nodes      finds    inserts   requests     misses  waves  waves_miss  max_batch");
  println!("  {:>5}  {:>4}  {:>9}  (base tiles; brute calls {}, cell evals {})", e.a, 1 << e.a, e.num_tiles(), e.brute_calls, e.brute_cell_evals);
  for (l, lv) in e.levels.iter().enumerate().skip(e.a + 1) {
    let s = lv.stats;
    println!(
      "  {:>5}  {:>4}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>5}  {:>10}  {:>9}",
      l,
      1u64 << l,
      lv.nodes.len(),
      s.finds,
      s.inserts,
      s.requests,
      s.misses,
      s.waves,
      s.waves_with_miss,
      s.max_miss_batch
    );
  }
  // Where the work was resolved: a hit on a level-l window delivers its centre for
  // 2^min(l-2,k) generations in one probe; a floor miss delivers it by brute force.
  let mut rows: Vec<(String, f64)> = Vec::new();
  for (l, lv) in e.levels.iter().enumerate().skip(e.a + 1) {
    let v = (1u64 << (2 * (l - 1))) as f64 * (1u64 << (l - 2).min(e.step_log2)) as f64;
    let hits = lv.stats.requests.saturating_sub(lv.stats.misses) as f64;
    if hits > 0.0 {
      rows.push((format!("hit at {0}x{0}", 1u64 << l), hits * v));
    }
    if l == e.a + 1 && lv.stats.misses > 0 {
      rows.push((format!("brute at {0}x{0}", 1u64 << l), lv.stats.misses as f64 * v));
    }
  }
  let total: f64 = rows.iter().map(|r| r.1).sum();
  if total > 0.0 {
    println!("  cell-updates by the window size that resolved them (overlapping windows counted as requested):");
    println!("   {}", rows.iter().map(|(n, d)| format!("{n}: {:.1}%", 100.0 * d / total)).collect::<Vec<_>>().join(" | "));
  }
}

/// `flat`: learning curve of the flat tile CA at tile level `--lt`.
pub fn flat(args: &[String]) {
  let path: std::path::PathBuf = opt(args, "--pattern").expect("--pattern").into();
  let (loaded, file_rule) = load_pattern(&path);
  let variant = variant_for(args, &file_rule);
  let lut = crate::load_lut(&variant);
  let a: usize = opt_or(args, "--a", 2);
  let lt: usize = opt_or(args, "--lt", 4);
  let gens: u64 = opt_or(args, "--gens", 65536);
  let margin: usize = opt_or(args, "--margin", 2);
  let mut e = Engine::new(&lut, a, (lt - 1).max(a));
  let t0 = Instant::now();
  let mut f = match &loaded {
    Loaded::Dense(g) => Flat::from_grid(&mut e, g, lt, margin),
    Loaded::Mc(m) => {
      let (root, level) = e.from_mc(m);
      let (root, level, _) = e.popzeros(root, level);
      let lvl = level.max(lt);
      assert!(level >= lt, "pattern smaller than one tile");
      let ids = e.ids_at_level(root, lvl, lt);
      Flat::from_ids(&mut e, &ids, 1 << (lvl - lt), lt, margin)
    }
  };
  let s = 1u64 << lt;
  let cells = (f.nx as u64 * s) * (f.ny as u64 * s);
  let z = e.empty(lt);
  let nonempty = f.a_ids.iter().filter(|&&i| i != z).count();
  println!(
    "pattern {} rule {} | a={a} lt={lt} (tile {s}x{s}, {} gens/sweep) | grid {}x{} tiles = {} cells, {} non-empty tiles ({:.1}%) | load {:.1}s",
    path.file_name().unwrap().to_string_lossy(),
    variant.name(),
    s / 2,
    f.nx,
    f.ny,
    cells,
    nonempty,
    100.0 * nonempty as f64 / f.a_ids.len() as f64,
    t0.elapsed().as_secs_f64()
  );
  // reset counters accumulated while loading
  for lv in e.levels.iter_mut() {
    lv.stats = Default::default();
  }
  let sweeps = gens / (s / 2);
  let quiet = flag(args, "--quiet");
  let mut seen: rustc_hash::FxHashMap<u64, u64> = Default::default();
  let mut period: Option<(u64, u64)> = None;
  let (mut win, mut uniq, mut miss) = (0u64, 0u64, 0u64);
  // second half of the run = "steady state" figures for the summary line
  let (mut win2, mut uniq2, mut miss2) = (0u64, 0u64, 0u64);
  // tile-steps that produced a non-empty tile: the work plain cells would have to do
  let mut nonempty_tile_steps = 0u64;
  let (tf0, ti0) = (e.tile_finds, e.tile_inserts);
  let (mut bwin, mut buniq, mut bmiss, mut bbrute) = (0u64, 0u64, 0u64, 0u64);
  let mut next_report = 1u64;
  let mut clear = true;
  let t0 = Instant::now();
  if !quiet {
    println!("     gens   windows/sweep  uniq/sweep   miss-rate(all)  miss-rate(uniq)  brute/sweep   tiles(lt)   keys(lt+1)");
  }
  for sw in 1..=sweeps {
    let st = f.sweep(&mut e);
    clear &= f.border_clear(&mut e);
    nonempty_tile_steps += (if f.in_b { &f.b_ids } else { &f.a_ids }).iter().filter(|&&i| i != z).count() as u64;
    win += st.windows;
    uniq += st.unique_windows;
    miss += st.top_misses;
    bwin += st.windows;
    buniq += st.unique_windows;
    bmiss += st.top_misses;
    if sw > sweeps / 2 {
      win2 += st.windows;
      uniq2 += st.unique_windows;
      miss2 += st.top_misses;
    }
    if !f.in_b && period.is_none() {
      let mut h = 0xcbf29ce484222325u64;
      for &id in &f.a_ids {
        h = (h ^ id as u64).wrapping_mul(0x100000001b3);
      }
      if let Some(&prev) = seen.get(&h) {
        period = Some((prev, (sw - prev) * (s / 2)));
      } else {
        seen.insert(h, sw);
      }
    }
    if sw == next_report || sw == sweeps {
      let n = (sw - next_report / 2).max(1) as f64;
      let _ = n;
      if !quiet {
        let brute_now = e.brute_calls;
        let span = if sw == 1 { 1 } else { sw - (next_report / 2).min(sw - 1) } as f64;
        println!(
          "  {:>7}   {:>12.0}  {:>10.0}   {:>13.5}%  {:>14.5}%  {:>11.1}  {:>10}  {:>11}",
          sw * (s / 2),
          bwin as f64 / span,
          buniq as f64 / span,
          100.0 * bmiss as f64 / bwin as f64,
          100.0 * bmiss as f64 / buniq.max(1) as f64,
          (brute_now - bbrute) as f64 / span,
          if lt == a { e.num_tiles() } else { e.levels[lt].nodes.len() },
          e.levels[lt + 1].nodes.len()
        );
        bbrute = brute_now;
      }
      bwin = 0;
      buniq = 0;
      bmiss = 0;
      next_report *= 2;
    }
  }
  let secs = t0.elapsed().as_secs_f64();
  println!(
    "total: {} gens, {} window probes, {} unique, {} top-level misses ({:.4}% of probes); border clear={clear}; cpu {:.1}s",
    sweeps * (s / 2),
    win,
    uniq,
    miss,
    100.0 * miss as f64 / win as f64,
    secs
  );
  match period {
    Some((start, p)) => println!("state repeats: first seen at sweep {start}, period {p} generations"),
    None => println!("no repeat of the phase-A id grid within the run"),
  }
  print_levels(&e);
  if flag(args, "--summary") {
    // pattern, lt, a, gens, cells, tiles/sweep, miss% (whole run), miss% (2nd half), unique-window% (2nd half),
    // distinct tiles at lt, memo entries at lt+1, total finds, brute cell evals, period
    println!(
      "SUMMARY	{}	{lt}	{a}	{}	{cells}	{}	{:.4}	{:.4}	{:.2}	{}	{}	{}	{}	{}",
      path.file_name().unwrap().to_string_lossy(),
      sweeps * (s / 2),
      win / sweeps.max(1),
      100.0 * miss as f64 / win.max(1) as f64,
      100.0 * miss2 as f64 / win2.max(1) as f64,
      100.0 * uniq2 as f64 / win2.max(1) as f64,
      if lt == a { e.num_tiles() } else { e.levels[lt].nodes.len() },
      e.levels[lt + 1].nodes.len(),
      e.levels.iter().map(|l| l.stats.finds).sum::<u64>(),
      e.brute_cell_evals,
      period.map(|(_, p)| p.to_string()).unwrap_or("-".into())
    );
  }
  {
    // GPU cost model (same constants as `adapt`): the sweep level is the largest memoised tile
    let m = crate::cmd_adapt::Model::default();
    let finds: u64 = e.levels.iter().map(|l| l.stats.finds).sum();
    let inserts: u64 = e.levels.iter().map(|l| l.stats.inserts).sum();
    let (tf, ti) = (e.tile_finds - tf0, e.tile_inserts - ti0);
    let win_evals = e.brute_calls as f64 * (1u64 << (2 * (a + 1))) as f64 * (1u64 << (a - 1)) as f64;
    let hash_ms = ((finds + tf) as f64 * m.probe + (inserts + ti) as f64 * m.insert + e.brute_calls as f64 * 4.0 * m.gather) / 1e6;
    let cells_ms = win_evals * m.cell / 1e6;
    let nominal_ms = nonempty_tile_steps as f64 * (s * s * s / 2) as f64 * m.cell / 1e6;
    let mem = inserts as f64 * 32.0 + ti as f64 * ((1usize << (2 * a)) as f64 * 5.0 / 8.0 + 8.0);
    println!(
      "MODEL	{}	sweep {s}x{s}	floor {}x{}	{} gens	hash {:.2} ms	cells {:.2} ms	total {:.2} ms	plain cells {:.2} ms	{:.1}x	mem {:.1} MiB	{:.0} finds/gen",
      path.file_name().unwrap().to_string_lossy(),
      1 << (a + 1),
      1 << (a + 1),
      sweeps * (s / 2),
      hash_ms,
      cells_ms,
      hash_ms + cells_ms,
      nominal_ms,
      nominal_ms / (hash_ms + cells_ms),
      mem / 1048576.0,
      (finds + tf) as f64 / (sweeps * (s / 2)) as f64
    );
  }
  // cost-model inputs: cell updates delivered vs. operations spent
  let delivered = cells as f64 * (sweeps * (s / 2)) as f64;
  let finds: u64 = e.levels.iter().map(|l| l.stats.finds).sum();
  println!(
    "cell-updates delivered {:.3e}; hash finds {:.3e} ({:.1} cell-updates per find); brute cell evals {:.3e} ({:.2}% of delivered)",
    delivered,
    finds as f64,
    delivered / finds as f64,
    e.brute_cell_evals as f64,
    100.0 * e.brute_cell_evals as f64 / delivered
  );
}

/// `root`: Golly-style whole-universe stepping by `2^k`, level-synchronous, with batch statistics.
pub fn root(args: &[String]) {
  let path: std::path::PathBuf = opt(args, "--pattern").expect("--pattern").into();
  let (loaded, file_rule) = load_pattern(&path);
  let variant = variant_for(args, &file_rule);
  let lut = crate::load_lut(&variant);
  let a: usize = opt_or(args, "--a", 2);
  let k: usize = opt_or(args, "--k", 10);
  let gens: u64 = opt_or(args, "--gens", 1 << 16);
  let mut e = Engine::new(&lut, a, k);
  let (mut root, mut level) = match &loaded {
    Loaded::Dense(g) => e.from_grid(g),
    Loaded::Mc(m) => e.from_mc(m),
  };
  let fp0 = engine_fingerprint(&e, root, level);
  println!(
    "pattern {} rule {} | a={a} step=2^{k} | level {} | population {} bbox {}x{}",
    path.file_name().unwrap().to_string_lossy(),
    variant.name(),
    level,
    fp0.population,
    fp0.w,
    fp0.h
  );
  for lv in e.levels.iter_mut() {
    lv.stats = Default::default();
  }
  let steps = gens >> k;
  let t0 = Instant::now();
  let mut next = 1u64;
  for i in 1..=steps {
    let (r, l, _) = e.step_root(root, level);
    root = r;
    level = l;
    if i == next || i == steps {
      let finds: u64 = e.levels.iter().map(|l| l.stats.finds).sum();
      let misses: u64 = e.levels.iter().map(|l| l.stats.misses).sum();
      println!(
        "  gen {:>10}: level {:>2}, nodes {:>9}, finds {:>11}, misses {:>10}, brute {:>9}, cpu {:>7.2}s",
        i << k,
        level,
        e.total_nodes(),
        finds,
        misses,
        e.brute_calls,
        t0.elapsed().as_secs_f64()
      );
      next *= 2;
    }
  }
  print_levels(&e);
  let fp = engine_fingerprint(&e, root, level);
  println!("final: gen {} population {} bbox {}x{} hash {:016x}", steps << k, fp.population, fp.w, fp.h, fp.hash);
}

/// `fingerprint`: print the fingerprint of a pattern file (for comparing with bgolly output).
pub fn fingerprint(args: &[String]) {
  let path: std::path::PathBuf = opt(args, "--pattern").expect("--pattern").into();
  let (loaded, _) = load_pattern(&path);
  let fp = match &loaded {
    Loaded::Dense(g) => grid_fingerprint(g),
    Loaded::Mc(m) => {
      let lut = vec![0u8; 1]; // unused
      let mut e = Engine::new(&lut, 1, 1);
      let (root, level) = e.from_mc(m);
      engine_fingerprint(&e, root, level)
    }
  };
  println!("population {} bbox {}x{} hash {:016x}", fp.population, fp.w, fp.h, fp.hash);
}

/// `export`: write a pattern as RLE (optionally after `--gens` CPU generations) for bgolly.
pub fn export(args: &[String]) {
  let path: std::path::PathBuf = opt(args, "--pattern").expect("--pattern").into();
  let out: std::path::PathBuf = opt(args, "--out").expect("--out").into();
  let (loaded, file_rule) = load_pattern(&path);
  let Loaded::Dense(g) = loaded else { panic!("export needs a dense pattern") };
  let rule_name = opt(args, "--as").unwrap_or(if file_rule.is_empty() { "Hutton32b".into() } else { file_rule });
  pattern::write_rle(&out, &g, &rule_name);
  println!("wrote {} ({}x{}, rule {rule_name})", out.display(), g.w, g.h);
}

/// `ruletree`: write a Golly `.rule` file (@TREE) generated from the validated LUT.
pub fn ruletree(args: &[String]) {
  let variant = RuleSpec::parse(&opt(args, "--rule").unwrap_or("hutton32b".into())).expect("bad --rule");
  let name = opt(args, "--name").expect("--name (rule name written into the file)");
  let out: std::path::PathBuf = opt(args, "--out").map(Into::into).unwrap_or(crate::out_dir().join(format!("{name}.rule")));
  let lut = crate::load_lut(&variant);
  // Golly evaluates `b[a[a[a[a[base+n]+w]+e]+s]+c]`: parameter order n, w, e, s, c (root = n).
  let mut world: rustc_hash::FxHashMap<Vec<u32>, u32> = Default::default();
  let mut lines: Vec<String> = Vec::new();
  fn recur(
    at: usize,
    params: &mut [u32; 5],
    lut: &[u8],
    world: &mut rustc_hash::FxHashMap<Vec<u32>, u32>,
    lines: &mut Vec<String>,
  ) -> u32 {
    if at == 0 {
      let [n, w, e, s, c] = *params;
      return lut[rule::lut_index(c, n, s, e, w)] as u32;
    }
    let mut node = vec![at as u32];
    for i in 0..32 {
      params[5 - at] = i;
      node.push(recur(at - 1, params, lut, world, lines));
    }
    if let Some(&id) = world.get(&node) {
      return id;
    }
    let id = lines.len() as u32;
    lines.push(node.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" "));
    world.insert(node, id);
    id
  }
  recur(5, &mut [0; 5], &lut, &mut world, &mut lines);
  let mut text = format!("@RULE {name}\n\ngenerated from the LUT of {}\n\n@TREE\n\nnum_states=32\nnum_neighbors=4\nnum_nodes={}\n", variant.name(), lines.len());
  for l in &lines {
    text.push_str(l);
    text.push('\n');
  }
  std::fs::write(&out, text).unwrap();
  println!("wrote {} ({} tree nodes)", out.display(), lines.len());
}

/// `rulediff`: compare two rule LUTs entry by entry and summarise where they disagree.
pub fn rulediff(args: &[String]) {
  let a = RuleSpec::parse(&opt(args, "--a").expect("--a")).expect("bad --a");
  let b = RuleSpec::parse(&opt(args, "--b").expect("--b")).expect("bad --b");
  let (la, lb) = (crate::load_lut(&a), crate::load_lut(&b));
  let total = la.iter().zip(&lb).filter(|(x, y)| x != y).count();
  println!("A = {}\nB = {}\n{} of {} neighbourhoods differ", a.name(), b.name(), total, rule::LUT_SIZE);
  // group by (centre, A result, B result)
  let mut groups: std::collections::BTreeMap<(u32, u8, u8), (u64, u32)> = Default::default();
  for i in 0..rule::LUT_SIZE {
    if la[i] != lb[i] {
      let e = groups.entry(((i >> 20) as u32, la[i], lb[i])).or_insert((0, i as u32));
      e.0 += 1;
    }
  }
  println!("  centre   A->   B->      count   example (c n e s w)");
  for ((c, ra, rb), (count, ex)) in &groups {
    println!(
      "  {:>6}  {:>4}  {:>4}  {:>9}   {} {} {} {} {}",
      c, ra, rb, count, (ex >> 20) & 31, (ex >> 15) & 31, (ex >> 10) & 31, (ex >> 5) & 31, ex & 31
    );
  }
}
