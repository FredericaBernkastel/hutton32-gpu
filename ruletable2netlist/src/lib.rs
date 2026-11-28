//! From a Golly rule table to a gate network for bit-sliced kernels.
//!
//! [`compile`] reads the transitions of a table, builds a network of two-input gates that computes
//! the next state of a cell from the bits of its neighbourhood, checks it against the table on
//! every neighbourhood of the rule, makes it smaller without changing what it computes, and
//! returns a netlist that can be written as WGSL or Rust.
//!
//! ```no_run
//! let table = ruletable2netlist::Table::load(std::path::Path::new("Hutton32b.rule")).unwrap();
//! let c = ruletable2netlist::compile(&table, &ruletable2netlist::Options::default()).unwrap();
//! println!("{}", c.netlist.summary());
//! std::fs::write("step_planes.wgsl", c.netlist.wgsl("step_planes", table.nbhd.table_order())).unwrap();
//! ```

pub mod build;
pub mod check;
pub mod cover;
pub mod factor;
pub mod map;
pub mod net;
mod recut;
pub mod resub;
pub mod simplify;
pub mod symmetry;

pub use build::Style;
pub use ca_rule::{Case, Gate, Interp, Neighbourhood, Netlist, Op, Pos, Signal, Table};
pub use check::{Checker, Nb, Space};
pub use net::Net;

use std::time::Instant;

#[derive(Clone, Debug)]
pub struct Options {
  /// seconds the optimiser may run
  pub seconds: f64,
  /// rules with at most this many neighbourhoods are optimised against all of them
  pub working_limit: u64,
  /// rules with at most this many neighbourhoods have their network checked on all of them;
  /// a rule with more is not optimised, and its network as built is tested on a sample
  pub exhaustive_limit: u64,
  /// neighbourhoods drawn at random for a rule with more than that
  pub sample: usize,
  pub seed: u64,
  /// 0: as many as the machine has
  pub threads: usize,
  /// print progress to standard error
  pub verbose: bool,
  /// skip the optimiser
  pub raw: bool,
  /// simplify the transitions before building (on by default)
  pub simplify: bool,
  /// build in this style only
  pub style: Option<Style>,
  /// do not look for a rotational symmetry
  pub raw_plain: bool,
  /// give the lists one vocabulary of tested sets (on by default)
  pub unify: bool,
  /// the folded style: the centre states without orientation as one chain for the state
  /// (true) instead of a chain per bit (the default, which optimises to fewer gates)
  pub rest_whole: Option<bool>,
  /// the folded style selects a cell by the centre's orientation if at least this many sets
  /// are tested there; none: try a few values
  pub hoist: Option<usize>,
  /// of a hundred changes the optimiser makes to leave a network no sweep makes smaller, how
  /// many derive the whole network again from a cover by cuts; the others replace nodes by
  /// expressions of equal size
  pub restructure: u32,
}

impl Default for Options {
  fn default() -> Self {
    Options { seconds: 60.0, working_limit: 1 << 25, exhaustive_limit: 1 << 30, sample: 1 << 22, seed: 1, threads: 0, verbose: false, raw: false, simplify: true, style: None, raw_plain: false, unify: true, rest_whole: None, hoist: None, restructure: 60 }
  }
}

#[derive(Clone, Debug, Default)]
pub struct Report {
  pub name: String,
  pub states: usize,
  pub bits: usize,
  pub neighbourhood: String,
  pub symmetries: String,
  pub variables: usize,
  /// transitions as written
  pub lines: usize,
  /// transitions after the symmetries
  pub cases: usize,
  /// transitions the network is built from, after simplification
  pub kept: usize,
  /// the same per bit of the state, each list keeping one bit
  pub kept_bits: Vec<usize>,
  pub simplified: simplify::Stats,
  /// AND and XOR nodes of the network as built, per style tried
  pub built: Vec<(Style, usize)>,
  pub style: Option<Style>,
  /// AND and XOR nodes after optimisation
  pub nodes: usize,
  /// gates of the netlist: NOT, AND, OR, XOR
  pub gates: usize,
  pub counts: [usize; 4],
  pub depth: usize,
  /// neighbourhoods the final network was checked on
  pub checked: u64,
  /// whether those are all the neighbourhoods of the rule
  pub exhaustive: bool,
  /// whether the optimiser ran (it does not where the result could not be proven)
  pub optimised: bool,
  /// the rule's rotational symmetry, if one was found: its order, the centre states it holds
  /// for, those with an orientation, and the transitions that serve them (those of orientation 0)
  pub symmetry: Option<(usize, usize, usize, usize)>,
  pub seconds: f64,
  pub optimiser: resub::Stats,
}

pub struct Compiled {
  pub netlist: Netlist,
  pub report: Report,
}

fn threads(opts: &Options) -> usize {
  if opts.threads > 0 {
    opts.threads
  } else {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
  }
}

/// Compile a table to a checked, optimised netlist.
pub fn compile(table: &Table, opts: &Options) -> Result<Compiled, String> {
  compile_from(table, opts, None)
}

/// The same, starting from a netlist written elsewhere instead of the networks built from the
/// table: it is checked against the table and optimised.
pub fn compile_from(table: &Table, opts: &Options, start: Option<&Netlist>) -> Result<Compiled, String> {
  let t0 = Instant::now();
  resub::SEED.store(opts.seed, std::sync::atomic::Ordering::Relaxed);
  resub::RECUT.store(opts.restructure as usize, std::sync::atomic::Ordering::Relaxed);
  let cases = table.cases()?;
  let space = Space { n: table.states, k: table.cells(), bits: table.bits() };
  let interp = Interp::new(&cases, space.n, space.k);
  let nthreads = threads(opts);
  let log = |s: String| {
    if opts.verbose {
      eprintln!("[{:7.1}s] {s}", t0.elapsed().as_secs_f64());
    }
  };
  let mut report = Report {
    name: table.name.clone(),
    states: table.states,
    bits: space.bits,
    neighbourhood: table.nbhd.golly_name().to_string(),
    symmetries: table.symmetries.clone(),
    variables: table.variables,
    lines: table.lines.len(),
    cases: cases.len(),
    ..Default::default()
  };

  // the set the optimiser works against: every neighbourhood, or a sample that holds, for every
  // transition, neighbourhoods where it applies and near misses
  let total = space.total();
  let all = total.is_some_and(|t| t <= opts.working_limit);
  // Where the neighbourhoods cannot all be enumerated, a network can only be tested on a sample.
  // The optimiser's replacements are right on the sample by construction and may be wrong
  // elsewhere, so it is not run: the network as built follows the transitions one to one.
  let provable = total.is_some_and(|t| t <= opts.exhaustive_limit);
  let raw = opts.raw || !provable;
  if !provable && !opts.raw {
    log(format!("{}: more than {} neighbourhoods: not optimised, tested on a sample", table.name, opts.exhaustive_limit));
  }
  let mut rng = check::Rng(opts.seed);
  let sample = resub::Sample::new(space, &cases, &mut rng);
  let mut working = if all {
    Checker::exhaustive(space, &interp, true, nthreads)
  } else {
    let mut pts: Vec<Nb> = (0..opts.sample).map(|_| space.random(&mut rng)).collect();
    pts.extend_from_slice(&sample.pts);
    Checker::sampled(space, &interp, pts, nthreads)
  };
  log(format!("{}: {} states, {}, {} transitions ({} with symmetries {}); checking on {} neighbourhoods{}", table.name, table.states, table.nbhd.golly_name(), table.lines.len(), cases.len(), table.symmetries, working.points(), if all { " (all)" } else { "" }));

  // the transitions to build from: simplified, the function unchanged; and once per bit of the
  // state, keeping only that bit
  let mut lists: Vec<Vec<Case>> = vec![cases.clone(); 1 + space.bits];
  // what each list is for: one bit or the whole state; some centre states or all
  let mut what: Vec<(Option<usize>, Option<ca_rule::StateSet>)> = (0..1 + space.bits).map(|i| (i.checked_sub(1), None)).collect();
  // with a rotational symmetry: the table of the centre states of orientation 0, and per bit the
  // table of the centre states without orientation
  let wanted = |s: Style| opts.style.is_none_or(|o| o == s);
  let sym = if all && start.is_none() && !opts.raw_plain && (wanted(Style::Fold) || opts.style == Some(Style::Cover)) { symmetry::find(space, &interp, opts.working_limit, nthreads) } else { None };
  if let Some(sym) = &sym {
    let moved: Vec<String> = (0..space.n).filter(|&v| sym.tau[v] as usize != v).map(|v| format!("{v}>{}", sym.tau[v])).collect();
    log(format!("symmetry: a turn of order {} with the states {}; {} centre states symmetric, {} with an orientation", sym.order, moved.join(" "), sym.covariant.len(), sym.all_oriented().len()));
    let oriented = sym.all_oriented();
    let mut others = ca_rule::StateSet::default();
    for v in (0..space.n as u8).filter(|&v| !oriented.contains(v)) {
      others.insert(v);
    }
    lists.push(cases.clone());
    what.push((None, Some(sym.oriented[0])));
    for b in 0..space.bits {
      lists.push(cases.clone());
      what.push((Some(b), Some(others)));
    }
    // and the same centre states in one list for the whole state
    lists.push(cases.clone());
    what.push((None, Some(others)));
  }
  if opts.simplify && start.is_none() {
    // with `raw` nothing else is bounded by time: let the simplification finish, so that the
    // network as built does not depend on the machine's load
    let deadline = Instant::now() + std::time::Duration::from_secs_f64(if opts.raw { 3600.0 } else { (opts.seconds * 0.2).clamp(2.0, 120.0) });
    let mut stats = vec![simplify::Stats::default(); lists.len()];
    let what = &what;
    let jobs = std::sync::Mutex::new(lists.iter_mut().zip(stats.iter_mut()).enumerate());
    std::thread::scope(|s| {
      for _ in 0..nthreads {
        s.spawn(|| loop {
          let job = jobs.lock().unwrap().next();
          let Some((i, (list, st))) = job else { break };
          let (c, r) = simplify::simplify_with(space, std::mem::take(list), what[i].0, what[i].1, sym.as_ref().filter(|_| i == 1 + space.bits), deadline);
          *list = c;
          *st = r;
        });
      }
    });
    let per_bit: Vec<String> = lists[1..1 + space.bits].iter().map(|l| l.len().to_string()).collect();
    log(format!("simplified: {} transitions left of {}, {} sets widened, {} of them to every state; per bit {} transitions", lists[0].len(), cases.len(), stats[0].widened, stats[0].freed, per_bit.join(", ")));
    report.simplified = stats[0];
  }
  if opts.simplify && start.is_none() && opts.unify {
    // one vocabulary of tested sets for the lists that are built together
    let deadline = Instant::now() + std::time::Duration::from_secs_f64(if opts.raw { 3600.0 } else { (opts.seconds * 0.2).clamp(2.0, 240.0) });
    let b = space.bits;
    let before: usize = lists.iter().map(|l| l.len()).sum();
    let mut given = simplify::unify(space, &mut lists[1..1 + b], &what[1..1 + b], sym.as_ref(), deadline);
    if sym.is_some() {
      given += simplify::unify(space, &mut lists[1 + b..], &what[1 + b..], sym.as_ref(), deadline);
    }
    // what became redundant
    for (i, (list, w)) in lists.iter_mut().zip(&what).enumerate().skip(1) {
      *list = simplify::simplify_with(space, std::mem::take(list), w.0, w.1, sym.as_ref().filter(|_| i == 1 + space.bits), deadline).0;
    }
    log(format!("vocabulary: {given} sets given up; {} transitions in the lists, {before} before", lists.iter().map(|l| l.len()).sum::<usize>()));
  }
  report.kept = lists[0].len();
  report.kept_bits = lists[1..1 + space.bits].iter().map(|l| l.len()).collect();
  if !(opts.simplify && start.is_none()) {
    // the lists of the folded style still have to be cut down to their centre states
    for (i, (list, w)) in lists.iter_mut().zip(&what).enumerate().skip(1 + space.bits) {
      *list = simplify::simplify_with(space, std::mem::take(list), w.0, w.1, sym.as_ref().filter(|_| i == 1 + space.bits), Instant::now()).0;
    }
  }
  let input = |style: Style| if matches!(style, Style::Bits | Style::Factor) { &lists[1..1 + space.bits] } else { &lists[..1] };

  // the networks as built
  let mut starts: Vec<(Style, Net)> = Vec::new();
  if let Some(nl) = start {
    let net = map::to_net(nl, table.nbhd.table_order(), space.bits).map_err(|e| format!("{}: {e}", table.name))?;
    let bad = working.check(&net, 1);
    if !bad.is_empty() {
      return Err(format!("{}: the netlist given differs from the table at {:?}", table.name, &bad[0][..space.k]));
    }
    log(format!("given: {} gates, {} nodes", nl.gates.len(), net.gates()));
    report.built.push((Style::Given, net.gates()));
    starts.push((Style::Given, net));
  }
  if let Some(sym) = sym.as_ref().filter(|_| wanted(Style::Fold)) {
    let b = space.bits;
    let (canon, rest, whole) = (&lists[1 + b], &lists[2 + b..2 + 2 * b], &lists[2 + 2 * b..3 + 2 * b]);
    report.symmetry = Some((sym.order, sym.covariant.len(), sym.all_oriented().len(), canon.len()));
    log(format!("folded: {} transitions for orientation 0, {} per bit for the centre states without orientation", canon.len(), rest.iter().map(|l| l.len().to_string()).collect::<Vec<_>>().join(", ")));
    if std::env::var("R2N_DEBUG").is_ok() {
      let show = |c: &Case| {
        let sets: Vec<String> = c.sets[..space.k].iter().map(|s| if s.iter().filter(|&v| (v as usize) < space.n).count() == space.n { "*".to_string() } else { s.iter().filter(|&v| (v as usize) < space.n).map(|v| v.to_string()).collect::<Vec<_>>().join(" ") }).collect();
        format!("{} -> {:?}{}", sets.join(" | "), c.out, if c.eq.is_empty() { String::new() } else { format!(" eq {:?}", c.eq) })
      };
      eprintln!("canon:");
      for c in canon {
        eprintln!("  {}", show(c));
      }
      eprintln!("rest (whole state):");
      for c in &whole[0] {
        eprintln!("  {}", show(c));
      }
      for (b, l) in rest.iter().enumerate() {
        eprintln!("rest bit {b}:");
        for c in l {
          eprintln!("  {}", show(c));
        }
      }
    }
    let variants: Vec<(usize, &[Vec<Case>])> = opts.hoist.map_or(vec![3usize, usize::MAX], |h| vec![h]).into_iter().flat_map(|h| [(h, rest), (h, whole)]).filter(|v| opts.rest_whole.unwrap_or(false) == (v.1.len() == 1)).collect();
    for (hoist, rest) in variants {
      if let Some(b) = build::build_folded(table, sym, canon, rest, hoist) {
        let bad = working.check(&b.net, 1);
        if !bad.is_empty() {
          log(format!("folded network (hoist {hoist}) differs from the table at {:?}: not used", &bad[0][..space.k]));
          continue;
        }
        log(format!("built Fold (hoist {hoist}, the other centre states {}): {} nodes, {} terms", if rest.len() == 1 { "as one chain" } else { "per bit" }, b.net.gates(), b.terms));
        report.built.push((Style::Fold, b.net.gates()));
        starts.push((Style::Fold, b.net));
      }
    }
  }
  if all && start.is_none() && opts.style == Some(Style::Cover) {
    // the transitions made from the function itself
    let lut = cover::function(space, &interp, nthreads);
    let none = symmetry::Symmetry { order: 1, pi: (0..space.k).collect(), tau: (0..space.n as u8).collect(), covariant: ca_rule::StateSet::default(), oriented: vec![ca_rule::StateSet::default()] };
    let s = sym.as_ref().unwrap_or(&none);
    if let Some((canon, rest)) = cover::covers(space, &lut, s) {
      log(format!("covers: {} transitions for orientation 0, {} per bit for the centre states without orientation", canon.len(), rest.iter().map(|l| l.len().to_string()).collect::<Vec<_>>().join(", ")));
      for hoist in opts.hoist.map_or(vec![3usize, usize::MAX], |h| vec![h]) {
        if let Some(b) = build::build_folded(table, s, &canon, &rest, hoist) {
          let bad = working.check(&b.net, 1);
          if !bad.is_empty() {
            log(format!("network from covers (hoist {hoist}) differs from the table at {:?}: not used", &bad[0][..space.k]));
            continue;
          }
          log(format!("built Cover (hoist {hoist}): {} nodes, {} terms", b.net.gates(), b.terms));
          report.built.push((Style::Cover, b.net.gates()));
          starts.push((Style::Cover, b.net));
        }
      }
    }
  }
  for style in [Style::Factor, Style::Chain, Style::Bits, Style::Flat] {
    if start.is_some() {
      break;
    }
    if opts.style.is_some_and(|s| s != style) {
      continue;
    }
    if let Some(b) = build::build(table, input(style), style) {
      let bad = working.check(&b.net, 1);
      if !bad.is_empty() {
        return Err(format!("{}: the network built ({style:?}) differs from the table at {:?}", table.name, &bad[0][..space.k]));
      }
      log(format!("built {style:?}: {} nodes, {} terms", b.net.gates(), b.terms));
      report.built.push((style, b.net.gates()));
      starts.push((style, b.net));
    }
  }

  // optimise: each start down to a network no sweep makes smaller (at most 15 % of the time
  // each), then the smallest on from there for the rest of the time
  let mut best: Option<(Style, Net)> = None;
  if raw {
    best = starts.into_iter().min_by_key(|s| s.1.gates());
  } else {
    let mut results: Vec<(Style, Net, resub::Sample)> = Vec::new();
    starts.sort_by_key(|s| s.1.gates());
    starts.truncate(3);
    for (style, net) in starts {
      let mut s = sample.clone();
      let share = Instant::now() + std::time::Duration::from_secs_f64(opts.seconds * 0.15);
      // first with the signals kept for it to use, then without those it did not use
      let kept = net.keep.len();
      let mut net = resub::optimise(net, &mut s, &working, share, false, &mut report.optimiser, &|m| log(format!("{style:?}: {m}")));
      if kept > 0 {
        net.keep.clear();
        let net2 = net.compact();
        log(format!("{style:?}: {} nodes without the signals nothing uses", net2.gates()));
        net = resub::optimise(net2, &mut s, &working, share.max(Instant::now() + std::time::Duration::from_secs_f64(opts.seconds * 0.05)), false, &mut report.optimiser, &|m| log(format!("{style:?}: {m}")));
      }
      results.push((style, net, s));
    }
    if let Some((style, net, mut s)) = results.into_iter().min_by_key(|r| r.1.gates()) {
      let deadline = t0 + std::time::Duration::from_secs_f64(opts.seconds);
      let net = resub::optimise(net, &mut s, &working, deadline, true, &mut report.optimiser, &|m| log(format!("{style:?}: {m}")));
      best = Some((style, net));
    }
  }
  let (style, mut net) = best.ok_or_else(|| format!("{}: no network", table.name))?;

  // the final check: every neighbourhood where they can be enumerated
  let bad = working.check(&net, 1);
  if !bad.is_empty() {
    return Err(format!("{}: the optimised network differs from the table at {:?}", table.name, &bad[0][..space.k]));
  }
  report.checked = working.points();
  report.exhaustive = all;
  if !all && total.is_some_and(|t| t <= opts.exhaustive_limit) {
    log("checking every neighbourhood".to_string());
    let full = Checker::exhaustive(space, &interp, false, nthreads);
    let mut tries = 0;
    loop {
      let bad = full.check(&net, 64);
      if bad.is_empty() {
        break;
      }
      // the sample missed a difference: add what the full check found and optimise again from
      // the network as built
      tries += 1;
      log(format!("the full check found a difference ({} neighbourhoods added)", bad.len()));
      working.add(&bad);
      net = match start {
        Some(nl) => map::to_net(nl, table.nbhd.table_order(), space.bits)?,
        None => build::build(table, input(style), style).unwrap().net,
      };
      if tries <= 2 {
        let mut s = sample.clone();
        s.add(&bad);
        let deadline = Instant::now() + std::time::Duration::from_secs_f64(opts.seconds * 0.5);
        net = resub::optimise(net, &mut s, &working, deadline, true, &mut report.optimiser, &|m| log(format!("{style:?}: {m}")));
      } else if !full.check(&net, 1).is_empty() {
        return Err(format!("{}: the network built differs from the table", table.name));
      }
    }
    report.checked = full.points();
    report.exhaustive = true;
  }

  net.keep.clear();
  let netlist = map::to_netlist(&net, table.nbhd.table_order(), space.bits);
  report.style = Some(style);
  report.optimised = !raw;
  report.nodes = net.gates();
  report.gates = netlist.gates.len();
  let (n, a, o, x) = netlist.counts();
  report.counts = [n, a, o, x];
  report.depth = netlist.depth();
  report.seconds = t0.elapsed().as_secs_f64();
  log(format!("{} gates (NOT {}, AND {}, OR {}, XOR {}), depth {}", report.gates, report.counts[0], report.counts[1], report.counts[2], report.counts[3], report.depth));
  Ok(Compiled { netlist, report })
}

/// Check a netlist written elsewhere against a table, without changing it: on every neighbourhood
/// of the rule if there are at most `opts.exhaustive_limit`, else on a sample. The report holds
/// the netlist's own gate counts; `Err` names a neighbourhood on which the two differ.
pub fn verify(table: &Table, nl: &Netlist, opts: &Options) -> Result<Report, String> {
  let t0 = Instant::now();
  let cases = table.cases()?;
  let space = Space { n: table.states, k: table.cells(), bits: table.bits() };
  let interp = Interp::new(&cases, space.n, space.k);
  let net = map::to_net(nl, table.nbhd.table_order(), space.bits).map_err(|e| format!("{}: {e}", table.name))?;
  let all = space.total().is_some_and(|t| t <= opts.exhaustive_limit);
  let checker = if all {
    Checker::exhaustive(space, &interp, false, threads(opts))
  } else {
    let mut rng = check::Rng(opts.seed);
    let mut pts: Vec<Nb> = (0..opts.sample).map(|_| space.random(&mut rng)).collect();
    pts.extend_from_slice(&resub::Sample::new(space, &cases, &mut rng).pts);
    Checker::sampled(space, &interp, pts, threads(opts))
  };
  if let Some(bad) = checker.check(&net, 1).first() {
    return Err(format!("{}: the netlist differs from the table at {:?}", table.name, &bad[..space.k]));
  }
  let (n, a, o, x) = nl.counts();
  Ok(Report {
    name: table.name.clone(),
    states: table.states,
    bits: space.bits,
    neighbourhood: table.nbhd.golly_name().to_string(),
    symmetries: table.symmetries.clone(),
    variables: table.variables,
    lines: table.lines.len(),
    cases: cases.len(),
    nodes: net.gates(),
    gates: nl.gates.len(),
    counts: [n, a, o, x],
    depth: nl.depth(),
    checked: checker.points(),
    exhaustive: all,
    seconds: t0.elapsed().as_secs_f64(),
    ..Default::default()
  })
}
