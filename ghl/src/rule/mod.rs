//! Rules of Golly's HashLife family: a deterministic transition function of a radius-1
//! neighbourhood (von Neumann, Moore, hexagonal or one-dimensional) over at most 256 states.
//!
//! A rule is held in one of two forms for the CPU, and optionally as a gate network for the
//! bit-sliced engines:
//!
//! * a **table** of `2^(bits * cells)` bytes indexed by the neighbourhood, where `bits` is the
//!   number of bit-planes a state needs; this is the form of every rule whose table fits (Life-like
//!   rules: 512 bytes; Hutton32b: 32 MiB);
//! * a **rule tree** (Golly's `@TREE`) for rules whose table would not fit, a decision DAG walked
//!   neighbour by neighbour;
//! * a **network** of two-input gates over bit-planes, evaluating 32 or 64 cells per machine word:
//!   the adder tree of a Life-like rule, generated from its B and S sets, and the 367-gate circuit of
//!   Hutton32b. Rules without one run on the table or the tree.
//!
//! Index layout of a table: one field of `bits` bits per cell of the neighbourhood, in the order of
//! [`Neighbourhood::fields`] (field 0 least significant). For Hutton32 and Hutton32b this is
//! `w | s << 5 | e << 10 | n << 15 | c << 20`, the layout of the harness and of the GPU kernels.
//!
//! **B0 rules.** A Life-like rule in which an empty neighbourhood gives birth is run as Golly runs
//! it: with Smax (S8 on the Moore neighbourhood) as the complementary rule, the stored cells being the
//! complement of the true ones at every generation; without Smax as two rules alternated between
//! even and odd generations, the stored cells being the complement at odd generations. Neither
//! converted rule has B0, so the memo engines run them unchanged; [`Rule::inversion`] tells a front
//! end when to swap the colours of states 0 and 1, as Golly does.

mod bs;
pub(crate) mod hutton;
mod network;
mod table;
mod tree;
/// The Hutton32b circuit as Rust on `u64` bit-planes (`research/bitslice/step_planes.rs`).
#[path = "generated/hutton32b_planes.rs"]
#[allow(clippy::all)]
pub(crate) mod hutton32b_planes;
/// The Hutton32 circuit as Rust on `u64` bit-planes (emitted from `generated/hutton32.netlist`).
#[path = "generated/hutton32_planes.rs"]
#[allow(clippy::all)]
pub(crate) mod hutton32_planes;

pub use hutton::{HUTTON32B_NETLIST, HUTTON32B_RULE, HUTTON32_COLORS, HUTTON32_NETLIST, HUTTON32_TABLE};
pub use network::Kind as NetworkKind;

use std::{path::Path, sync::Arc};

pub use bs::parse_bs;
pub use network::{Gate, Netlist, Network, Op as GateOp, Signal};
pub use tree::RuleTree;

use crate::{Error, Result};

/// The cells of a radius-1 neighbourhood and the neighbourhoods of Golly's HashLife family, shared
/// with the other crates of the workspace.
pub use ca_rule::{Neighbourhood, Pos};

/// When the stored cells are the complement of the true ones (B0 rules, as in Golly).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Inversion {
  Never,
  /// B0 without Smax: two rules alternate and odd generations are stored inverted.
  OddGenerations,
  /// B0 with Smax: the complementary rule, stored inverted at every generation.
  Always,
}

/// Where a rule came from, for reports.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Source {
  LifeLike,
  /// One of the two canonical tables of `research/rules`, built from the transition function.
  Canonical,
  Table,
  Tree,
  Custom,
}

struct Inner {
  name: String,
  nbhd: Neighbourhood,
  states: u32,
  bits: u32,
  table: Option<Vec<u8>>,
  tree: Option<RuleTree>,
  life_like: Option<(u16, u16)>,
  network: Option<Arc<Network>>,
  /// B0 without Smax: the rule of odd generations (this one is the rule of even generations)
  odd: Option<Rule>,
  inversion: Inversion,
  palette: Vec<[u8; 3]>,
  source: Source,
}

/// A rule of Golly's HashLife family, cheap to clone.
#[derive(Clone)]
pub struct Rule(Arc<Inner>);

impl std::fmt::Debug for Rule {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("Rule")
      .field("name", &self.0.name)
      .field("neighbourhood", &self.0.nbhd)
      .field("states", &self.0.states)
      .field("form", &if self.0.table.is_some() { "table" } else { "tree" })
      .field("network", &self.0.network.as_ref().map(|n| n.gates().len()))
      .finish()
  }
}

impl PartialEq for Rule {
  fn eq(&self, other: &Self) -> bool {
    Arc::ptr_eq(&self.0, &other.0)
      || (self.0.name == other.0.name && self.0.nbhd == other.0.nbhd && self.0.states == other.0.states && self.0.table == other.0.table)
  }
}

/// The largest table built: 2^28 bytes. Larger rules become rule trees.
pub const MAX_TABLE_BITS: u32 = 28;

/// Bit-planes needed for `states` states.
pub fn planes_for(states: u32) -> u32 {
  assert!((1..=256).contains(&states), "a rule has 1 to 256 states");
  let mut b = 1;
  while (1u32 << b) < states {
    b += 1;
  }
  b
}

impl Rule {
  // ---- construction ---------------------------------------------------------------------------

  /// A rule by name: `Life`, B/S notation (`B3/S23`, `b36s23`, `B2/S34H`, `B2/SV`, `23/3`), or one of
  /// the canonical rules `Hutton32` and `Hutton32b`. Rule files are loaded with [`Rule::load`].
  pub fn named(name: &str) -> Result<Rule> {
    let t = name.trim();
    match t.to_ascii_lowercase().as_str() {
      "hutton32" => return Ok(hutton::rule(hutton::Variant::Hutton32)),
      "hutton32b" => return Ok(hutton::rule(hutton::Variant::Hutton32b)),
      _ => {}
    }
    if let Some((b, s, nbhd)) = parse_bs(t) {
      return Rule::life_like_named(t, b, s, nbhd);
    }
    Err(Error::Rule(format!("unknown rule {t:?}: not B/S notation and not a canonical rule; load rule files with Rule::load")))
  }

  /// A Life-like rule: bit k of `birth` / `survive` = born / survives with k live neighbours.
  pub fn life_like(birth: u16, survive: u16, nbhd: Neighbourhood) -> Result<Rule> {
    Rule::life_like_named(&bs::bs_name(birth, survive, nbhd), birth, survive, nbhd)
  }

  fn life_like_named(name: &str, birth: u16, survive: u16, nbhd: Neighbourhood) -> Result<Rule> {
    if nbhd == Neighbourhood::OneDimensional {
      return Err(Error::Rule("B/S notation is not defined on the one-dimensional neighbourhood".into()));
    }
    let max = nbhd.neighbours() as u16;
    let all = (1u16 << (max + 1)) - 1;
    if birth & !all != 0 || survive & !all != 0 {
      return Err(Error::Rule(format!("{name}: a neighbour count above {max}")));
    }
    let palette = vec![[48, 48, 48], [255, 255, 255]];
    if birth & 1 == 0 {
      return Ok(Rule::from_life_like(name, birth, survive, nbhd, Inversion::Never, None, palette));
    }
    // B0: Golly's emulation (liferules.cpp)
    let comp = |m: u16| -> u16 { (0..=max).filter(|&n| m >> (max - n) & 1 == 0).fold(0, |a, n| a | 1 << n) };
    let flip = |m: u16| -> u16 { (0..=max).filter(|&n| m >> (max - n) & 1 == 1).fold(0, |a, n| a | 1 << n) };
    if survive >> max & 1 == 1 {
      // h(c, n) = 1 - f(1 - c, N - n): born where N - n is no survival count, survives where N - n is no birth count
      let (b, s) = (comp(survive), comp(birth));
      Ok(Rule::from_life_like(name, b, s, nbhd, Inversion::Always, None, palette))
    } else {
      // even generations: g(c, n) = 1 - f(c, n); odd ones: g'(c, n) = f(1 - c, N - n)
      let (be, se) = (!birth & all, !survive & all);
      let (bo, so) = (flip(survive), flip(birth));
      let odd = Rule::from_life_like(&format!("{name} (odd generations)"), bo, so, nbhd, Inversion::OddGenerations, None, palette.clone());
      Ok(Rule::from_life_like(name, be, se, nbhd, Inversion::OddGenerations, Some(odd), palette))
    }
  }

  fn from_life_like(name: &str, birth: u16, survive: u16, nbhd: Neighbourhood, inversion: Inversion, odd: Option<Rule>, palette: Vec<[u8; 3]>) -> Rule {
    debug_assert!(birth & 1 == 0);
    let table = bs::life_like_table(birth, survive, nbhd);
    let network = match nbhd {
      Neighbourhood::Moore | Neighbourhood::Hexagonal | Neighbourhood::VonNeumann => Some(Arc::new(Network::life_like(birth, survive, nbhd))),
      Neighbourhood::OneDimensional => None,
    };
    Rule(Arc::new(Inner {
      name: name.to_string(),
      nbhd,
      states: 2,
      bits: 1,
      table: Some(table),
      tree: None,
      life_like: Some((birth, survive)),
      network,
      odd,
      inversion,
      palette,
      source: Source::LifeLike,
    }))
  }

  /// A rule from its table (`2^(bits * cells)` entries, index layout of [`Neighbourhood::fields`]).
  pub fn from_table(name: &str, nbhd: Neighbourhood, states: u32, table: Vec<u8>, palette: Option<Vec<[u8; 3]>>) -> Result<Rule> {
    let bits = planes_for(states);
    let want = 1usize << (bits as usize * nbhd.cells());
    if table.len() != want {
      return Err(Error::Rule(format!("{name}: table has {} entries, the neighbourhood needs {want}", table.len())));
    }
    if let Some(&bad) = table.iter().find(|&&v| v as u32 >= states) {
      return Err(Error::Rule(format!("{name}: table produces state {bad}, the rule has {states}")));
    }
    if table[0] != 0 {
      return Err(Error::Rule(format!("{name}: an empty neighbourhood gives birth (B0); HashLife cannot run table rules with B0")));
    }
    Ok(Rule(Arc::new(Inner {
      name: name.to_string(),
      nbhd,
      states,
      bits,
      table: Some(table),
      tree: None,
      life_like: None,
      network: None,
      odd: None,
      inversion: Inversion::Never,
      palette: palette.unwrap_or_else(|| default_palette(states)),
      source: Source::Custom,
    })))
  }

  /// A rule from a rule tree (Golly's `@TREE`). Its table is built as well when it fits.
  pub fn from_tree(name: &str, tree: RuleTree, palette: Option<Vec<[u8; 3]>>) -> Result<Rule> {
    let nbhd = tree.neighbourhood();
    let states = tree.states();
    let bits = planes_for(states);
    let table = if bits as usize * nbhd.cells() <= MAX_TABLE_BITS as usize { Some(tree.to_table(nbhd, bits)) } else { None };
    let zero = match &table {
      Some(t) => t[0],
      None => tree.eval_fields(&vec![0u8; nbhd.cells()]),
    };
    if zero != 0 {
      return Err(Error::Rule(format!("{name}: an empty neighbourhood gives birth (B0); HashLife cannot run tree rules with B0")));
    }
    Ok(Rule(Arc::new(Inner {
      name: name.to_string(),
      nbhd,
      states,
      bits,
      table,
      tree: Some(tree),
      life_like: None,
      network: None,
      odd: None,
      inversion: Inversion::Never,
      palette: palette.unwrap_or_else(|| default_palette(states)),
      source: Source::Tree,
    })))
  }

  /// The same rule with a gate network attached, for the bit-sliced engines. The network must
  /// compute the rule's table on every neighbourhood; [`Network::verify`] checks it exhaustively.
  pub fn with_network(&self, network: Network) -> Result<Rule> {
    if network.bits() != self.0.bits {
      return Err(Error::Rule(format!("network has {} bit-planes, the rule {}", network.bits(), self.0.bits)));
    }
    let mut inner = Inner {
      name: self.0.name.clone(),
      nbhd: self.0.nbhd,
      states: self.0.states,
      bits: self.0.bits,
      table: self.0.table.clone(),
      tree: self.0.tree.clone(),
      life_like: self.0.life_like,
      network: Some(Arc::new(network)),
      odd: self.0.odd.clone(),
      inversion: self.0.inversion,
      palette: self.0.palette.clone(),
      source: self.0.source.clone(),
    };
    inner.source = if inner.source == Source::Canonical { Source::Canonical } else { inner.source };
    Ok(Rule(Arc::new(inner)))
  }

  /// Load a Golly rule file: `.rule` (`@RULE` with `@TABLE` or `@TREE`, and `@COLORS`), a bare
  /// `.table` or `.tree` file. The rule is named after `@RULE`, or after the file.
  pub fn load(path: impl AsRef<Path>) -> Result<Rule> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    let stem = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    let ext = path.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default();
    Rule::from_rule_text(&text, &stem, &ext)
  }

  /// Parse the text of a rule file. `name` is used when there is no `@RULE` line; `ext` is the
  /// file's extension (`rule`, `table` or `tree`) and decides how a file without sections is read.
  pub fn from_rule_text(text: &str, name: &str, ext: &str) -> Result<Rule> {
    let sections = split_sections(text, ext);
    let rule_name = sections.iter().find(|s| s.0 == "@RULE").map(|s| s.1.trim().lines().next().unwrap_or("").trim().to_string());
    let name = rule_name.filter(|n| !n.is_empty()).unwrap_or_else(|| name.to_string());
    let palette = sections.iter().find(|s| s.0 == "@COLORS").map(|s| parse_colors(&s.1));
    if let Some((_, body)) = sections.iter().find(|s| s.0 == "@TABLE") {
      let t = table::parse(body).map_err(|e| Error::Rule(format!("{name}: {e}")))?;
      let palette = palette.map(|p| complete_palette(p, t.states));
      return table::build_rule(&name, &t, palette);
    }
    if let Some((_, body)) = sections.iter().find(|s| s.0 == "@TREE") {
      let tree = RuleTree::parse(body).map_err(|e| Error::Rule(format!("{name}: {e}")))?;
      let states = tree.states();
      let palette = palette.map(|p| complete_palette(p, states));
      return Rule::from_tree(&name, tree, palette);
    }
    Err(Error::Rule(format!("{name}: no @TABLE or @TREE section")))
  }

  /// Load a Golly `@TABLE` rule file and compile its table into a gate network with
  /// `ruletable2netlist` (feature `compile`), so that the bit-sliced engines run it instead of the
  /// byte-LUT or rule-tree kernels. The compiler checks its result on every neighbourhood when the
  /// rule has at most 2^30 of them, and the network is checked against the rule's table here as
  /// well. `seconds` bounds the optimiser (0 skips it); 32-state rules take minutes.
  #[cfg(feature = "compile")]
  pub fn load_compiled(path: impl AsRef<Path>, seconds: f64) -> Result<Rule> {
    let path = path.as_ref();
    let rule = Rule::load(path)?;
    let text = std::fs::read_to_string(path).map_err(|e| Error::Io(format!("{}: {e}", path.display())))?;
    let table = ca_rule::Table::parse(rule.name(), &text).map_err(|e| Error::Rule(format!("{}: {e}", path.display())))?;
    let opts = ruletable2netlist::Options { seconds, raw: seconds <= 0.0, ..Default::default() };
    let compiled = ruletable2netlist::compile(&table, &opts).map_err(|e| Error::Rule(format!("{}: {e}", rule.name())))?;
    let net = Network::from_netlist(compiled.netlist);
    if rule.table().is_some() {
      let bad = net.verify(&rule);
      if bad != 0 {
        return Err(Error::Rule(format!("{}: the compiled network differs from the table on {bad} neighbourhoods", rule.name())));
      }
    } else if !compiled.report.exhaustive {
      // exactness everywhere: a network tested on a sample only is not used
      return Err(Error::Rule(format!("{}: the compiled network was tested on a sample of the neighbourhoods, not proven on all of them", rule.name())));
    }
    rule.with_network(net)
  }

  // ---- accessors ------------------------------------------------------------------------------

  pub fn name(&self) -> &str {
    &self.0.name
  }
  pub fn neighbourhood(&self) -> Neighbourhood {
    self.0.nbhd
  }
  pub fn states(&self) -> u32 {
    self.0.states
  }
  /// Bit-planes per cell: `ceil(log2(states))`, at least 1.
  pub fn planes(&self) -> u32 {
    self.0.bits
  }
  /// The transition table, when the rule has one.
  pub fn table(&self) -> Option<&[u8]> {
    self.0.table.as_deref()
  }
  pub fn tree(&self) -> Option<&RuleTree> {
    self.0.tree.as_ref()
  }
  /// The gate network of the rule, when it has one (Life-like rules, Hutton32b).
  pub fn network(&self) -> Option<&Network> {
    self.0.network.as_deref()
  }
  /// `(birth, survive)` masks of a Life-like rule, after the B0 conversion.
  pub fn life_like_masks(&self) -> Option<(u16, u16)> {
    self.0.life_like
  }
  pub fn palette(&self) -> &[[u8; 3]] {
    &self.0.palette
  }
  pub fn inversion(&self) -> Inversion {
    self.0.inversion
  }
  /// Whether the stored cells at generation `g` are the complement of the true ones.
  pub fn inverted_at(&self, g: u64) -> bool {
    match self.0.inversion {
      Inversion::Never => false,
      Inversion::Always => true,
      Inversion::OddGenerations => g % 2 == 1,
    }
  }
  /// The rule that steps generation `g` to `g + 1`: the odd-generation rule of an alternating B0
  /// rule at odd `g`, this rule otherwise.
  pub fn for_generation(&self, g: u64) -> &Rule {
    match &self.0.odd {
      Some(o) if g % 2 == 1 => o,
      _ => self,
    }
  }
  /// Two rules alternate: memoized steps must start at even generations.
  pub fn alternates(&self) -> bool {
    self.0.odd.is_some()
  }
  pub fn source(&self) -> &Source {
    &self.0.source
  }

  /// Index of a neighbourhood in the table: `cells[i]` is the state of [`Neighbourhood::fields`]`[i]`.
  #[inline]
  pub fn index(&self, cells: &[u8]) -> usize {
    let b = self.0.bits;
    let mut i = 0usize;
    for (f, &c) in cells.iter().enumerate() {
      i |= (c as usize) << (b as usize * f);
    }
    i
  }

  /// The next state of a cell: `cells[i]` is the state of [`Neighbourhood::fields`]`[i]`.
  #[inline]
  pub fn next(&self, cells: &[u8]) -> u8 {
    match &self.0.table {
      Some(t) => t[self.index(cells)],
      None => self.0.tree.as_ref().unwrap().eval_fields(cells),
    }
  }
}

/// Golly's default colours for a rule without `@COLORS`: state 0 black, the others a gradient from
/// red to yellow.
pub fn default_palette(states: u32) -> Vec<[u8; 3]> {
  let mut p = vec![[0, 0, 0]];
  let n = states.saturating_sub(1).max(1);
  for i in 0..states.saturating_sub(1) {
    let t = if n > 1 { i as f64 / (n - 1) as f64 } else { 0.0 };
    p.push([255, (255.0 * t).round() as u8, 0]);
  }
  p
}

fn complete_palette(mut p: Vec<[u8; 3]>, states: u32) -> Vec<[u8; 3]> {
  let d = default_palette(states);
  p.resize(states as usize, [0, 0, 0]);
  // states that @COLORS leaves out keep the default
  p
    .iter()
    .zip(d)
    .map(|(c, d)| if *c == [0, 0, 0] { d } else { *c })
    .enumerate()
    .map(|(i, c)| if i == 0 { [0, 0, 0] } else { c })
    .collect()
}

fn parse_colors(body: &str) -> Vec<[u8; 3]> {
  let mut p: Vec<[u8; 3]> = Vec::new();
  for line in body.lines() {
    let line = line.split('#').next().unwrap().trim();
    let v: Vec<u32> = line.split_whitespace().filter_map(|t| t.parse().ok()).collect();
    if v.len() == 4 && v[0] < 256 {
      let s = v[0] as usize;
      if p.len() <= s {
        p.resize(s + 1, [0, 0, 0]);
      }
      p[s] = [v[1].min(255) as u8, v[2].min(255) as u8, v[3].min(255) as u8];
    }
  }
  p
}

/// `(section name, body)` pairs of a rule file. A file without sections is one `@TABLE` (`.table`)
/// or `@TREE` (`.tree`) section.
fn split_sections(text: &str, ext: &str) -> Vec<(String, String)> {
  let has_sections = text.lines().any(|l| l.trim_start().starts_with('@'));
  if !has_sections {
    let name = if ext == "tree" || text.contains("num_neighbors") { "@TREE" } else { "@TABLE" };
    return vec![(name.to_string(), text.to_string())];
  }
  let mut out: Vec<(String, String)> = Vec::new();
  for line in text.lines() {
    let t = line.trim_start();
    if t.starts_with('@') {
      let mut parts = t.splitn(2, char::is_whitespace);
      let head = parts.next().unwrap().to_string();
      let rest = parts.next().unwrap_or("").to_string();
      out.push((head, if rest.is_empty() { String::new() } else { format!("{rest}\n") }));
    } else if let Some(last) = out.last_mut() {
      last.1.push_str(line);
      last.1.push('\n');
    }
  }
  out
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn life_table_and_network_agree() {
    let r = Rule::named("B3/S23").unwrap();
    assert_eq!(r.states(), 2);
    assert_eq!(r.planes(), 1);
    let net = r.network().unwrap();
    assert_eq!(net.verify(&r), 0);
  }

  #[test]
  fn b0_conversions() {
    // B0 with S8: complementary rule, stored inverted always
    let r = Rule::named("B03/S238").unwrap();
    assert_eq!(r.inversion(), Inversion::Always);
    assert_eq!(r.life_like_masks().unwrap().0 & 1, 0);
    // B0 without S8: two alternating rules, neither with B0
    let r = Rule::named("B03/S23").unwrap();
    assert!(r.alternates());
    assert_eq!(r.life_like_masks().unwrap().0 & 1, 0);
    assert_eq!(r.for_generation(1).life_like_masks().unwrap().0 & 1, 0);
  }

  #[test]
  fn parse_sections() {
    let text = "@RULE Test\n@TABLE\nn_states:2\nneighborhood:Moore\nsymmetries:permute\n0,1,1,1,0,0,0,0,0,1\n1,1,1,0,0,0,0,0,0,1\n1,1,1,1,0,0,0,0,0,1\n1,0,0,0,0,0,0,0,0,0\n";
    let r = Rule::from_rule_text(text, "x", "rule").unwrap();
    assert_eq!(r.name(), "Test");
    // a dead cell with exactly three live neighbours anywhere is born
    let mut cells = vec![0u8; 9];
    for f in [0usize, 5, 8] {
      cells[f] = 1;
    }
    assert_eq!(r.next(&cells), 1);
  }
}
