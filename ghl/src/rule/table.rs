//! Golly's rule tables (`@TABLE`, `ruletable_algo.cpp`):
//!
//! ```text
//! n_states:32
//! neighborhood:vonNeumann        (or Moore, hexagonal, oneDimensional)
//! symmetries:none                (rotate4, rotate4reflect, rotate8, rotate8reflect, reflect_horizontal,
//!                                 rotate2, rotate3, rotate6, rotate6reflect, reflect, permute)
//! var a={0,1,2}
//! 0,a,a,1,0,1                    the centre, the neighbours clockwise from north, the new state
//! ```
//!
//! A variable that occurs more than once in a transition is bound: it takes the same value at every
//! occurrence. With `n_states` up to 10 the commas may be left out. The first matching transition
//! wins; a neighbourhood that no transition matches leaves the cell unchanged. Symmetries apply each
//! transition to the permuted neighbourhoods as well.

use rustc_hash::{FxHashMap, FxHashSet};

use super::{planes_for, Neighbourhood, Rule, RuleTree, Source, MAX_TABLE_BITS};
use crate::{Error, Result};

/// A set of states (up to 256).
pub type StateSet = [u64; 4];

fn set_has(s: &StateSet, v: u32) -> bool {
  s[(v >> 6) as usize] >> (v & 63) & 1 == 1
}
fn set_of(v: u32) -> StateSet {
  let mut s = [0u64; 4];
  s[(v >> 6) as usize] |= 1 << (v & 63);
  s
}

/// One expanded transition: a set of states per position of the Golly table order, and the output.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Pattern {
  pub sets: Vec<StateSet>,
  pub out: u8,
}

pub struct Table {
  pub states: u32,
  pub nbhd: Neighbourhood,
  #[allow(dead_code)]
  pub symmetries: String,
  pub patterns: Vec<Pattern>,
}

/// Permutations of the Golly table order (index 0, the centre, stays) for a symmetry.
fn symmetry_group(nbhd: Neighbourhood, name: &str) -> std::result::Result<Vec<Vec<usize>>, String> {
  let n = nbhd.cells();
  let m = n - 1; // neighbours, in cyclic order clockwise from north
  let rot = |k: usize| -> Vec<usize> { (0..n).map(|i| if i == 0 { 0 } else { 1 + (i - 1 + k) % m }).collect() };
  // reflection fixing the first neighbour (north; west for one-dimensional)
  let refl = |p: &Vec<usize>| -> Vec<usize> { (0..n).map(|i| if i == 0 { 0 } else { p[1 + (m - (i - 1)) % m] }).collect() };
  let rotations = |step: usize| -> Vec<Vec<usize>> { (0..m / step).map(|k| rot(k * step)).collect() };
  let with_reflections = |g: Vec<Vec<usize>>| -> Vec<Vec<usize>> {
    let mut out = g.clone();
    for p in &g {
      out.push(refl(p));
    }
    out
  };
  let identity = vec![(0..n).collect::<Vec<usize>>()];
  let g = match (nbhd, name) {
    (_, "none") => identity,
    (_, "permute") => {
      let mut perms = Vec::new();
      let mut cur: Vec<usize> = (1..n).collect();
      permutations(&mut cur, 0, &mut perms);
      perms.into_iter().map(|p| std::iter::once(0).chain(p).collect()).collect()
    }
    (Neighbourhood::VonNeumann, "rotate4") => rotations(1),
    (Neighbourhood::VonNeumann, "rotate4reflect") => with_reflections(rotations(1)),
    (Neighbourhood::VonNeumann, "reflect_horizontal") => vec![identity[0].clone(), refl(&identity[0])],
    (Neighbourhood::Moore, "rotate4") => rotations(2),
    (Neighbourhood::Moore, "rotate8") => rotations(1),
    (Neighbourhood::Moore, "rotate4reflect") => with_reflections(rotations(2)),
    (Neighbourhood::Moore, "rotate8reflect") => with_reflections(rotations(1)),
    (Neighbourhood::Moore, "reflect_horizontal") => vec![identity[0].clone(), refl(&identity[0])],
    (Neighbourhood::Hexagonal, "rotate2") => rotations(3),
    (Neighbourhood::Hexagonal, "rotate3") => rotations(2),
    (Neighbourhood::Hexagonal, "rotate6") => rotations(1),
    (Neighbourhood::Hexagonal, "rotate6reflect") => with_reflections(rotations(1)),
    (Neighbourhood::OneDimensional, "reflect") => vec![identity[0].clone(), vec![0, 2, 1]],
    _ => return Err(format!("symmetries:{name} is not defined for the {} neighbourhood", nbhd.golly_name())),
  };
  Ok(g)
}

fn permutations(v: &mut Vec<usize>, k: usize, out: &mut Vec<Vec<usize>>) {
  if k == v.len() {
    out.push(v.clone());
    return;
  }
  for i in k..v.len() {
    v.swap(k, i);
    permutations(v, k + 1, out);
    v.swap(k, i);
  }
}

/// Parse the body of an `@TABLE` section and expand its transitions.
pub fn parse(body: &str) -> std::result::Result<Table, String> {
  let mut states = 0u32;
  let mut nbhd = Neighbourhood::Moore;
  let mut symmetries = "none".to_string();
  let mut vars: FxHashMap<String, Vec<u32>> = FxHashMap::default();
  let mut patterns: Vec<Pattern> = Vec::new();
  let mut seen: FxHashSet<Pattern> = FxHashSet::default();
  let mut group: Option<Vec<Vec<usize>>> = None;
  for (no, raw) in body.lines().enumerate() {
    let line = raw.split('#').next().unwrap().trim();
    if line.is_empty() {
      continue;
    }
    let err = |m: &str| format!("line {}: {m}: {line}", no + 1);
    if let Some(v) = line.strip_prefix("n_states:") {
      states = v.trim().parse().map_err(|_| err("bad n_states"))?;
      if !(1..=256).contains(&states) {
        return Err(err("n_states must be 1 to 256"));
      }
      continue;
    }
    if let Some(v) = line.strip_prefix("neighborhood:") {
      nbhd = Neighbourhood::from_golly_name(v).ok_or_else(|| err("unknown neighborhood"))?;
      group = None;
      continue;
    }
    if let Some(v) = line.strip_prefix("symmetries:") {
      symmetries = v.trim().to_string();
      group = None;
      continue;
    }
    if let Some(rest) = line.strip_prefix("var ") {
      let (name, set) = rest.split_once('=').ok_or_else(|| err("bad var line"))?;
      let set = set.trim().trim_start_matches('{').trim_end_matches('}');
      let mut vals = Vec::new();
      for tok in set.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        match tok.parse::<u32>() {
          Ok(v) => vals.push(v),
          Err(_) => vals.extend(vars.get(tok).ok_or_else(|| err(&format!("unknown variable {tok}")))?.iter().copied()),
        }
      }
      vars.insert(name.trim().to_string(), vals);
      continue;
    }
    if states == 0 {
      return Err(err("transition before n_states"));
    }
    if group.is_none() {
      group = Some(symmetry_group(nbhd, &symmetries)?);
    }
    let n = nbhd.cells();
    let toks: Vec<String> = if line.contains(',') {
      line.split(',').map(|t| t.trim().to_string()).collect()
    } else {
      line.chars().filter(|c| !c.is_whitespace()).map(|c| c.to_string()).collect()
    };
    if toks.len() != n + 1 {
      return Err(err(&format!("expected {} entries", n + 1)));
    }
    // variables used more than once are bound
    let mut bound: Vec<&str> = Vec::new();
    for t in &toks {
      if vars.contains_key(t.as_str()) && toks.iter().filter(|x| *x == t).count() > 1 && !bound.contains(&t.as_str()) {
        bound.push(t);
      }
    }
    let mut idx = vec![0usize; bound.len()];
    loop {
      let value = |tok: &str| -> std::result::Result<Vec<u32>, String> {
        if let Some(bi) = bound.iter().position(|b| *b == tok) {
          Ok(vec![vars[tok][idx[bi]]])
        } else if let Some(v) = vars.get(tok) {
          Ok(v.clone())
        } else {
          let v: u32 = tok.parse().map_err(|_| err(&format!("bad entry {tok}")))?;
          Ok(vec![v])
        }
      };
      let mut sets = Vec::with_capacity(n);
      for t in &toks[..n] {
        let mut s = [0u64; 4];
        for v in value(t)? {
          if v >= states {
            return Err(err(&format!("state {v} out of range")));
          }
          let o = set_of(v);
          for k in 0..4 {
            s[k] |= o[k];
          }
        }
        sets.push(s);
      }
      let out = value(&toks[n])?;
      if out.len() != 1 {
        return Err(err("the new state is an unbound variable"));
      }
      if out[0] >= states {
        return Err(err("new state out of range"));
      }
      for perm in group.as_ref().unwrap() {
        let p = Pattern { sets: perm.iter().map(|&i| sets[i]).collect(), out: out[0] as u8 };
        if seen.insert(p.clone()) {
          patterns.push(p);
        }
      }
      // next combination of the bound variables
      let mut i = 0;
      while i < bound.len() {
        idx[i] += 1;
        if idx[i] < vars[bound[i]].len() {
          break;
        }
        idx[i] = 0;
        i += 1;
      }
      if i == bound.len() {
        break;
      }
    }
  }
  if states == 0 {
    return Err("no n_states".into());
  }
  Ok(Table { states, nbhd, symmetries, patterns })
}

/// Fill `table` (initialised with "unchanged") by a depth-first walk over the fields, narrowing the
/// list of candidate transitions at each field; the first candidate left at a leaf wins.
fn fill_table(t: &Table, bits: u32, table: &mut [u8]) {
  let nbhd = t.nbhd;
  let fields = nbhd.fields();
  let order = nbhd.table_order();
  // position of each field in the transition (Golly order)
  let tpos: Vec<usize> = fields.iter().map(|p| order.iter().position(|q| q == p).unwrap()).collect();
  let centre = nbhd.field_of(super::Pos::C).unwrap();
  // walk the centre first: transitions usually name it
  let mut walk: Vec<usize> = vec![centre];
  walk.extend((0..fields.len()).filter(|&f| f != centre));
  fn rec(t: &Table, tpos: &[usize], walk: &[usize], depth: usize, bits: u32, index: usize, cand: &[u32], table: &mut [u8]) {
    let f = walk[depth];
    let p = tpos[f];
    let last = depth + 1 == walk.len();
    for v in 0..t.states {
      let i = index | (v as usize) << (bits as usize * f);
      if last {
        if let Some(&c) = cand.iter().find(|&&c| set_has(&t.patterns[c as usize].sets[p], v)) {
          table[i] = t.patterns[c as usize].out;
        }
      } else {
        let next: Vec<u32> = cand.iter().copied().filter(|&c| set_has(&t.patterns[c as usize].sets[p], v)).collect();
        if !next.is_empty() {
          rec(t, tpos, walk, depth + 1, bits, i, &next, table);
        }
      }
    }
  }
  let all: Vec<u32> = (0..t.patterns.len() as u32).collect();
  rec(t, &tpos, &walk, 0, bits, 0, &all, table);
}

/// The rule of a parsed table: a table when it fits, a rule tree otherwise.
pub fn build_rule(name: &str, t: &Table, palette: Option<Vec<[u8; 3]>>) -> Result<Rule> {
  let bits = planes_for(t.states);
  let nbhd = t.nbhd;
  let cells = nbhd.cells();
  if (bits as usize) * cells <= MAX_TABLE_BITS as usize {
    let n = 1usize << (bits as usize * cells);
    let centre = nbhd.field_of(super::Pos::C).unwrap();
    let mask = (1usize << bits) - 1;
    let mut table: Vec<u8> = (0..n).map(|i| ((i >> (bits as usize * centre)) & mask) as u8).collect();
    fill_table(t, bits, &mut table);
    // codes that are no state of the rule map to 0
    if t.states < 1 << bits {
      for (i, out) in table.iter_mut().enumerate() {
        if (0..cells).any(|f| (i >> (bits as usize * f)) & mask >= t.states as usize) {
          *out = 0;
        }
      }
    }
    let mut r = Rule::from_table(name, nbhd, t.states, table, palette)?;
    let inner = std::sync::Arc::get_mut(&mut r.0).unwrap();
    inner.source = Source::Table;
    return Ok(r);
  }
  // a rule tree, built by the same walk in Golly's tree order
  let neighbors = match nbhd {
    Neighbourhood::VonNeumann => 4,
    Neighbourhood::Moore => 8,
    _ => return Err(Error::Rule(format!("{name}: {} states on the {} neighbourhood do not fit a table, and rule trees exist only for von Neumann and Moore", t.states, nbhd.golly_name()))),
  };
  let order = nbhd.table_order();
  let tree_order = RuleTree::order(neighbors);
  let tpos: Vec<usize> = tree_order.iter().map(|p| order.iter().position(|q| q == p).unwrap()).collect();
  let tree = build_tree(t, &tpos, neighbors);
  let mut r = Rule::from_tree(name, tree, palette)?;
  let inner = std::sync::Arc::get_mut(&mut r.0).unwrap();
  inner.source = Source::Table;
  Ok(r)
}

/// A rule tree from the transitions, sharing identical (depth, candidates) subtrees.
fn build_tree(t: &Table, tpos: &[usize], neighbors: u32) -> RuleTree {
  let states = t.states;
  let vars = tpos.len();
  let mut levels: Vec<u8> = Vec::new();
  let mut data: Vec<u32> = Vec::new();
  let mut nodes: FxHashMap<(u8, Vec<u32>), u32> = FxHashMap::default();
  let mut memo: FxHashMap<(usize, Vec<u32>), u32> = FxHashMap::default();
  #[allow(clippy::too_many_arguments)]
  fn rec(
    t: &Table,
    tpos: &[usize],
    depth: usize,
    vars: usize,
    cand: Vec<u32>,
    levels: &mut Vec<u8>,
    data: &mut Vec<u32>,
    nodes: &mut FxHashMap<(u8, Vec<u32>), u32>,
    memo: &mut FxHashMap<(usize, Vec<u32>), u32>,
  ) -> u32 {
    if let Some(&id) = memo.get(&(depth, cand.clone())) {
      return id;
    }
    let p = tpos[depth];
    let mut row = Vec::with_capacity(t.states as usize);
    for v in 0..t.states {
      if depth + 1 == vars {
        // the centre is the last variable: unmatched means unchanged
        let out = cand.iter().find(|&&c| set_has(&t.patterns[c as usize].sets[p], v)).map(|&c| t.patterns[c as usize].out as u32).unwrap_or(v);
        row.push(out);
      } else {
        let next: Vec<u32> = cand.iter().copied().filter(|&c| set_has(&t.patterns[c as usize].sets[p], v)).collect();
        row.push(rec(t, tpos, depth + 1, vars, next, levels, data, nodes, memo));
      }
    }
    let level = (vars - depth) as u8;
    let key = (level, row);
    let id = match nodes.get(&key) {
      Some(&id) => id,
      None => {
        let id = levels.len() as u32;
        levels.push(level);
        data.extend_from_slice(&key.1);
        nodes.insert(key, id);
        id
      }
    };
    memo.insert((depth, cand), id);
    id
  }
  let all: Vec<u32> = (0..t.patterns.len() as u32).collect();
  let _root = rec(t, tpos, 0, vars, all, &mut levels, &mut data, &mut nodes, &mut memo);
  let mut text = format!("num_states={states}\nnum_neighbors={neighbors}\nnum_nodes={}\n", levels.len());
  for (i, &l) in levels.iter().enumerate() {
    text += &l.to_string();
    for v in &data[i * states as usize..(i + 1) * states as usize] {
      text.push(' ');
      text += &v.to_string();
    }
    text.push('\n');
  }
  RuleTree::parse(&text).expect("tree built from a table")
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn symmetry_groups_have_the_right_size() {
    assert_eq!(symmetry_group(Neighbourhood::VonNeumann, "rotate4").unwrap().len(), 4);
    assert_eq!(symmetry_group(Neighbourhood::VonNeumann, "rotate4reflect").unwrap().len(), 8);
    assert_eq!(symmetry_group(Neighbourhood::Moore, "rotate8reflect").unwrap().len(), 16);
    assert_eq!(symmetry_group(Neighbourhood::Moore, "permute").unwrap().len(), 40320);
    assert_eq!(symmetry_group(Neighbourhood::Hexagonal, "rotate6reflect").unwrap().len(), 12);
    // reflect_horizontal on von Neumann swaps E and W: C,N,E,S,W -> C,N,W,S,E
    assert_eq!(symmetry_group(Neighbourhood::VonNeumann, "reflect_horizontal").unwrap()[1], vec![0, 1, 4, 3, 2]);
    // on Moore: C,N,NE,E,SE,S,SW,W,NW -> C,N,NW,W,SW,S,SE,E,NE
    assert_eq!(symmetry_group(Neighbourhood::Moore, "reflect_horizontal").unwrap()[1], vec![0, 1, 8, 7, 6, 5, 4, 3, 2]);
  }

  #[test]
  fn canonical_tables_equal_the_transition_functions() {
    for (text, v) in [(super::super::hutton::HUTTON32_TABLE, super::super::hutton::Variant::Hutton32), (super::super::hutton::HUTTON32B_RULE, super::super::hutton::Variant::Hutton32b)] {
      let r = Rule::from_rule_text(text, "x", if v == super::super::hutton::Variant::Hutton32 { "table" } else { "rule" }).unwrap();
      let canon = super::super::hutton::rule(v);
      let a = r.table().unwrap();
      let b = canon.table().unwrap();
      let diff = a.iter().zip(b).filter(|(x, y)| x != y).count();
      assert_eq!(diff, 0, "{:?}: {diff} neighbourhoods differ", v);
    }
  }
}
