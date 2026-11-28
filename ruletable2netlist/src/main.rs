//! `ruletable2netlist RULE [options]`: compile a Golly rule table to a gate network.
//!
//! ```text
//! --seconds S        time for the optimiser (default 60)
//! --out DIR          write NAME.netlist, NAME.wgsl and NAME.rs there
//! --fn NAME          name of the emitted function (default step_planes)
//! --raw              the network as built, not optimised
//! --no-simplify      build from the transitions as written
//! --style S          chain, bits or flat: build in this style only
//! --from FILE        start from this netlist instead of the table's own network
//! --check FILE       only check this netlist against the table (with --lut and --golly if given)
//! --lut FILE         compare with a table of next states, one byte per neighbourhood
//! --golly BGOLLY     run a random soup in Golly and compare
//! --limit N          check on every neighbourhood if the rule has at most N (default 2^30)
//! --threads N, --seed N, --quiet
//! ```

use ruletable2netlist::{check::Rng, compile_from, verify, Interp, Netlist, Options, Table};
use std::path::{Path, PathBuf};
use std::process::Command;

fn opt(args: &[String], key: &str) -> Option<String> {
  args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}

fn main() {
  let args: Vec<String> = std::env::args().skip(1).collect();
  if args.is_empty() || args[0].starts_with("--") {
    eprintln!("usage: ruletable2netlist RULE [--seconds S] [--out DIR] [--fn NAME] [--raw] [--no-simplify] [--from NETLIST] [--check NETLIST] [--lut FILE] [--golly BGOLLY] [--limit N] [--threads N] [--seed N] [--restructure PERCENT] [--quiet]");
    std::process::exit(2);
  }
  if let Err(e) = run(&args) {
    eprintln!("error: {e}");
    std::process::exit(1);
  }
}

fn run(args: &[String]) -> Result<(), String> {
  let path = PathBuf::from(&args[0]);
  let table = Table::load(&path)?;
  let num = |key: &str, default: f64| -> Result<f64, String> {
    match opt(args, key) {
      Some(v) => v.parse().map_err(|_| format!("bad value for {key}")),
      None => Ok(default),
    }
  };
  if let Some(v) = opt(args, "--care") {
    // diagnosis: look at some centre states only ("0-8,25"); the netlist written is not the rule's
    let mut m = 0u64;
    for part in v.split(',') {
      let (a, b) = part.split_once('-').unwrap_or((part, part));
      let bad = |_| format!("bad value for --care: {v}");
      for s in a.trim().parse::<u32>().map_err(bad)?..=b.trim().parse::<u32>().map_err(bad)? {
        m |= 1u64.checked_shl(s).ok_or_else(|| format!("bad value for --care: {v}"))?;
      }
    }
    ruletable2netlist::check::CARE.store(m, std::sync::atomic::Ordering::Relaxed);
    eprintln!("DIAGNOSIS: only the centre states {v} are looked at; the netlist is not the rule's");
  }
  let opts = Options {
    seconds: num("--seconds", 60.0)?,
    exhaustive_limit: num("--limit", (1u64 << 30) as f64)? as u64,
    threads: num("--threads", 0.0)? as usize,
    seed: num("--seed", 1.0)? as u64,
    verbose: !args.iter().any(|a| a == "--quiet"),
    raw: args.iter().any(|a| a == "--raw"),
    simplify: !args.iter().any(|a| a == "--no-simplify"),
    raw_plain: args.iter().any(|a| a == "--no-symmetry"),
    unify: !args.iter().any(|a| a == "--no-unify"),
    restructure: num("--restructure", 60.0)? as u32,
    rest_whole: match opt(args, "--rest").as_deref() {
      None => None,
      Some("whole") => Some(true),
      Some("bits") => Some(false),
      Some(s) => return Err(format!("unknown --rest {s}")),
    },
    hoist: match opt(args, "--hoist") {
      Some(h) => Some(h.parse().map_err(|_| "bad value for --hoist".to_string())?),
      None => None,
    },
    style: match opt(args, "--style").as_deref() {
      None => None,
      Some("chain") => Some(ruletable2netlist::Style::Chain),
      Some("bits") => Some(ruletable2netlist::Style::Bits),
      Some("flat") => Some(ruletable2netlist::Style::Flat),
      Some("factor") => Some(ruletable2netlist::Style::Factor),
      Some("fold") => Some(ruletable2netlist::Style::Fold),
      Some("cover") => Some(ruletable2netlist::Style::Cover),
      Some(s) => return Err(format!("unknown style {s}")),
    },
    ..Options::default()
  };
  if args.iter().any(|a| a == "--symmetry") {
    // what the compiler finds out about the rule's rotational symmetry
    let cases = table.cases()?;
    let space = ruletable2netlist::Space { n: table.states, k: table.cells(), bits: table.bits() };
    let interp = Interp::new(&cases, space.n, space.k);
    let threads = if opts.threads > 0 { opts.threads } else { 4 };
    match ruletable2netlist::symmetry::find(space, &interp, opts.working_limit, threads) {
      None => println!("symmetry {}: none found", table.name),
      Some(s) => {
        let list = |set: &ca_rule::StateSet| set.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ");
        let moved: Vec<String> = (0..space.n).filter(|&v| s.tau[v] as usize != v).map(|v| format!("{v}>{}", s.tau[v])).collect();
        println!("symmetry {}: turn of order {}, cells {:?}; states {}", table.name, s.order, s.pi, moved.join(" "));
        println!("  symmetric for the centre states {}", list(&s.covariant));
        for (k, o) in s.oriented.iter().enumerate() {
          println!("  orientation {k}: {}", list(o));
        }
      }
    }
    return Ok(());
  }
  if let Some(f) = opt(args, "--check") {
    let nl = Netlist::parse(&std::fs::read_to_string(&f).map_err(|e| format!("{f}: {e}"))?)?;
    let r = verify(&table, &nl, &opts)?;
    println!(
      "netlist {} | rule {}, {} states ({} bits), {} | {} nodes | {} gates (AND {}, OR {}, XOR {}, NOT {}), depth {} | equal to the table on {} neighbourhoods ({})",
      Path::new(&f).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or(f.clone()),
      r.name,
      r.states,
      r.bits,
      r.neighbourhood,
      r.nodes,
      r.gates,
      r.counts[1],
      r.counts[2],
      r.counts[3],
      r.counts[0],
      r.depth,
      r.checked,
      if r.exhaustive { "all" } else { "a sample" }
    );
    return extra_checks(args, &table, &path, &nl, opts.seed);
  }
  let start = match opt(args, "--from") {
    Some(f) => Some(Netlist::parse(&std::fs::read_to_string(&f).map_err(|e| format!("{f}: {e}"))?)?),
    None => None,
  };
  let c = compile_from(&table, &opts, start.as_ref())?;
  let r = &c.report;
  let built: Vec<String> = r.built.iter().map(|(s, n)| format!("{} {n}", format!("{s:?}").to_lowercase())).collect();
  println!(
    "rule {} | {} states ({} bits), {}, symmetries {} | {} transitions, {} with symmetries, {} after simplification | built: {} nodes | {}: {} nodes ({}) | {} gates (AND {}, OR {}, XOR {}, NOT {}), depth {} | checked on {} neighbourhoods ({}) | {} sweeps, {} replacements, {} checks, {} refuted | {:.1} s",
    r.name,
    r.states,
    r.bits,
    r.neighbourhood,
    r.symmetries,
    r.lines,
    r.cases,
    r.kept,
    built.join(", "),
    if r.optimised { "optimised" } else { "not optimised" },
    r.nodes,
    format!("{:?}", r.style.unwrap()).to_lowercase(),
    r.gates,
    r.counts[1],
    r.counts[2],
    r.counts[3],
    r.counts[0],
    r.depth,
    r.checked,
    if r.exhaustive { "all" } else { "a sample" },
    r.optimiser.sweeps,
    r.optimiser.accepted,
    r.optimiser.checks,
    r.optimiser.refuted,
    r.seconds
  );

  if let Some((order, symmetric, oriented, canon)) = r.symmetry {
    println!("symmetry {}: a turn of order {order} with the states relabelled; {symmetric} centre states symmetric, {oriented} with an orientation; {canon} transitions for orientation 0", r.name);
  }
  let header = format!(
    "{} ({} states, {}, {} transitions) as a bit-sliced gate network, {} by ruletable2netlist.\nChecked against the table on {} neighbourhoods ({}).",
    r.name,
    r.states,
    r.neighbourhood,
    r.lines,
    match opt(args, "--from") {
      Some(f) => format!("the netlist {} optimised", Path::new(&f).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or(f.clone())),
      None if r.optimised => "compiled from the rule table and optimised".to_string(),
      None => "compiled from the rule table, as built".to_string(),
    },
    r.checked,
    if r.exhaustive { "all of them" } else { "a sample" }
  );
  if let Some(dir) = opt(args, "--out") {
    let dir = PathBuf::from(dir);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let func = opt(args, "--fn").unwrap_or_else(|| "step_planes".into());
    let write = |ext: &str, text: String| std::fs::write(dir.join(format!("{}.{ext}", r.name)), text).map_err(|e| e.to_string());
    let order = table.nbhd.table_order();
    let lines = |mark: &str| -> String { header.lines().chain([c.netlist.summary().as_str()]).map(|l| format!("{mark} {l}\n")).collect() };
    write("netlist", format!("{}{}", lines("#"), c.netlist.to_netlist()))?;
    write("wgsl", format!("{}{}", lines("//"), c.netlist.wgsl(&func, order)))?;
    write("rs", format!("{}{}", lines("//"), c.netlist.rust(&func, order)))?;
  }

  extra_checks(args, &table, &path, &c.netlist, opts.seed)
}

/// `--lut` and `--golly`: the netlist against a table of next states and against Golly.
fn extra_checks(args: &[String], table: &Table, path: &Path, nl: &Netlist, seed: u64) -> Result<(), String> {
  if let Some(f) = opt(args, "--lut") {
    let lut = std::fs::read(&f).map_err(|e| format!("{f}: {e}"))?;
    let bad = check_lut(table, nl, &lut)?;
    println!("lut {}: {} entries, {bad} differ", Path::new(&f).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or(f.clone()), lut.len());
    if bad > 0 {
      return Err("the netlist differs from the table of next states".into());
    }
  }
  if let Some(g) = opt(args, "--golly") {
    let res = golly_soup(table, path, nl, Path::new(&g), seed)?;
    println!("golly: {res}");
  }
  Ok(())
}

/// Entry `i` of the table of next states is the neighbourhood whose cells are the digits of `i`,
/// the centre highest.
fn check_lut(table: &Table, nl: &Netlist, lut: &[u8]) -> Result<u64, String> {
  let (n, k, bits) = (table.states as u64, table.cells(), table.bits());
  if n.checked_pow(k as u32) != Some(lut.len() as u64) {
    return Err(format!("the table of next states has {} entries, the rule has {n}^{k} neighbourhoods", lut.len()));
  }
  let (mut inputs, mut out, mut scratch) = (vec![0u64; k * bits], vec![0u64; bits], Vec::new());
  let mut bad = 0u64;
  for base in (0..lut.len()).step_by(64) {
    inputs.fill(0);
    let count = 64.min(lut.len() - base);
    for j in 0..count {
      let mut idx = (base + j) as u64;
      for c in (0..k).rev() {
        let v = idx % n;
        idx /= n;
        for b in 0..bits {
          inputs[c * bits + b] |= (v >> b & 1) << j;
        }
      }
    }
    nl.eval_words(&inputs, &mut scratch, &mut out);
    for j in 0..count {
      let v: u8 = (0..bits).map(|b| ((out[b] >> j & 1) as u8) << b).sum();
      bad += (v != lut[base + j]) as u64;
    }
  }
  Ok(bad)
}

// ---- Golly ------------------------------------------------------------------------------------

const SOUP: usize = 40;
const GENS: usize = 16;

fn rle_state(v: u8, two: bool) -> String {
  match v {
    0 => if two { "b" } else { "." }.to_string(),
    _ if two => "o".to_string(),
    _ => {
      let (hi, lo) = ((v - 1) / 24, (v - 1) % 24);
      let mut s = String::new();
      if hi > 0 {
        s.push((b'p' + hi - 1) as char);
      }
      s.push((b'A' + lo) as char);
      s
    }
  }
}

fn write_rle(grid: &[Vec<u8>], rule: &str, two: bool) -> String {
  let mut s = format!("x = {}, y = {}, rule = {rule}\n", grid[0].len(), grid.len());
  for row in grid {
    let mut line = String::new();
    for &v in row {
      line.push_str(&rle_state(v, two));
    }
    s.push_str(&line);
    s.push_str("$\n");
  }
  s.push_str("!\n");
  s
}

/// The cells of an RLE, as rows.
fn read_rle(text: &str) -> Result<Vec<Vec<u8>>, String> {
  let mut rows: Vec<Vec<u8>> = vec![Vec::new()];
  let (mut count, mut prefix) = (0usize, 0u8);
  let mut body = false;
  for line in text.lines() {
    if line.starts_with('#') {
      continue;
    }
    if !body {
      body = true;
      if line.trim_start().starts_with('x') {
        continue;
      }
    }
    for ch in line.chars() {
      let run = count.max(1);
      match ch {
        '0'..='9' => {
          count = count * 10 + ch as usize - '0' as usize;
          continue;
        }
        'b' | '.' => rows.last_mut().unwrap().extend(std::iter::repeat_n(0, run)),
        'o' => rows.last_mut().unwrap().extend(std::iter::repeat_n(1, run)),
        'p'..='y' => {
          prefix = ch as u8 - b'p' + 1;
          continue;
        }
        'A'..='X' => rows.last_mut().unwrap().extend(std::iter::repeat_n(prefix * 24 + (ch as u8 - b'A') + 1, run)),
        '$' => {
          for _ in 0..run {
            rows.push(Vec::new());
          }
        }
        '!' => return Ok(rows),
        c if c.is_whitespace() => continue,
        c => return Err(format!("unexpected {c} in RLE")),
      }
      count = 0;
      prefix = 0;
    }
  }
  Ok(rows)
}

/// The live cells' bounding box.
fn crop(grid: &[Vec<u8>]) -> Vec<Vec<u8>> {
  let (mut x0, mut x1, mut y0, mut y1) = (usize::MAX, 0, usize::MAX, 0);
  for (y, row) in grid.iter().enumerate() {
    for (x, &v) in row.iter().enumerate() {
      if v != 0 {
        x0 = x0.min(x);
        x1 = x1.max(x);
        y0 = y0.min(y);
        y1 = y1.max(y);
      }
    }
  }
  if x0 == usize::MAX {
    return Vec::new();
  }
  (y0..=y1).map(|y| (x0..=x1).map(|x| grid[y].get(x).copied().unwrap_or(0)).collect()).collect()
}

/// Run a random soup for some generations three ways (the table, the netlist, Golly) and compare.
fn golly_soup(table: &Table, path: &Path, nl: &Netlist, bgolly: &Path, seed: u64) -> Result<String, String> {
  let cases = table.cases()?;
  let (n, k, bits) = (table.states, table.cells(), table.bits());
  let interp = Interp::new(&cases, n, k);
  let mut scratch = vec![0u64; interp.words()];
  if interp.next(&vec![0u8; k], &mut scratch) != 0 {
    return Ok("skipped (empty space does not stay empty)".into());
  }
  let size = SOUP + 2 * GENS + 2;
  let mut rng = Rng(seed ^ 0x50f7);
  let mut grid = vec![vec![0u8; size]; size];
  for y in 0..SOUP {
    for x in 0..SOUP {
      grid[GENS + 1 + y][GENS + 1 + x] = rng.below(n) as u8;
    }
  }
  let soup: Vec<Vec<u8>> = (0..SOUP).map(|y| grid[GENS + 1 + y][GENS + 1..GENS + 1 + SOUP].to_vec()).collect();
  let offsets: Vec<(i32, i32)> = table.nbhd.table_order().iter().map(|p| p.offset()).collect();
  let mut by_net = grid.clone();
  let (mut inputs, mut out, mut gates) = (vec![0u64; k * bits], vec![0u64; bits], Vec::new());
  for _ in 0..GENS {
    let mut next = grid.clone();
    let mut next_net = by_net.clone();
    for y in 1..size - 1 {
      for x in 1..size - 1 {
        let nb: Vec<u8> = offsets.iter().map(|&(dx, dy)| grid[(y as i32 + dy) as usize][(x as i32 + dx) as usize]).collect();
        next[y][x] = interp.next(&nb, &mut scratch);
        for (c, &(dx, dy)) in offsets.iter().enumerate() {
          let v = by_net[(y as i32 + dy) as usize][(x as i32 + dx) as usize];
          for b in 0..bits {
            inputs[c * bits + b] = (v >> b & 1) as u64;
          }
        }
        nl.eval_words(&inputs, &mut gates, &mut out);
        next_net[y][x] = (0..bits).map(|b| ((out[b] & 1) as u8) << b).sum();
      }
    }
    grid = next;
    by_net = next_net;
  }
  if grid != by_net {
    return Err("the netlist and the table differ on the soup".into());
  }

  // Golly reads the rule from a .rule file in the directory it is told to search
  let dir = std::env::temp_dir().join(format!("ruletable2netlist-{}", std::process::id()));
  std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
  let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
  let rule_text = if text.lines().any(|l| l.trim_start().starts_with("@RULE")) { text } else { format!("@RULE {}\n@TABLE\n{text}", table.name) };
  std::fs::write(dir.join(format!("{}.rule", table.name)), rule_text).map_err(|e| e.to_string())?;
  let (input, output) = (dir.join("soup.rle"), dir.join("out.rle"));
  std::fs::write(&input, write_rle(&soup, &table.name, n == 2)).map_err(|e| e.to_string())?;
  let mut search = dir.to_string_lossy().into_owned();
  search.push(std::path::MAIN_SEPARATOR);
  let run = Command::new(bgolly).args(["-a", "RuleLoader", "-s", &search, "-m", &GENS.to_string(), "-o"]).arg(&output).arg(&input).output().map_err(|e| format!("{}: {e}", bgolly.display()))?;
  let result = std::fs::read_to_string(&output).map_err(|_| format!("Golly wrote no pattern: {}", String::from_utf8_lossy(&run.stdout).trim()));
  let _ = std::fs::remove_dir_all(&dir);
  let golly = crop(&read_rle(&result?)?);
  let mine = crop(&grid);
  let pad = |g: &Vec<Vec<u8>>| -> Vec<Vec<u8>> {
    let wd = g.iter().map(|r| r.len()).max().unwrap_or(0);
    g.iter().map(|r| r.iter().copied().chain(std::iter::repeat(0)).take(wd).collect()).collect()
  };
  let (golly, mine) = (pad(&golly), pad(&mine));
  if golly != mine {
    return Err(format!("Golly's soup after {GENS} generations differs ({}x{} cells against {}x{})", golly.first().map_or(0, |r| r.len()), golly.len(), mine.first().map_or(0, |r| r.len()), mine.len()));
  }
  let live: usize = mine.iter().map(|r| r.iter().filter(|&&v| v != 0).count()).sum();
  Ok(format!("a {SOUP}x{SOUP} soup after {GENS} generations agrees ({live} live cells; table, netlist and Golly)"))
}
