//! Golly rule tables: the `@TABLE` section of a `.rule` file, or a bare `.table`.
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
//! A table gives the number of states, the neighbourhood, a symmetry group, variables (named sets
//! of states) and transitions `C,N,...,C'`. The first transition that matches a neighbourhood
//! applies; a neighbourhood that matches none keeps its state. A variable that occurs more than
//! once in a transition is bound: the cells it names hold the same state, and as the output it
//! copies the cell it names. Semantics as in Golly's `ruletable_algo.cpp`.

use crate::nbhd::Neighbourhood;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Transitions after the expansion of the symmetries, at most.
pub const MAX_CASES: usize = 20_000;

/// A set of states (at most 256).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Default, Debug)]
pub struct StateSet(pub [u64; 4]);

impl StateSet {
  pub fn single(v: u8) -> Self {
    let mut s = StateSet::default();
    s.insert(v);
    s
  }
  pub fn insert(&mut self, v: u8) {
    self.0[v as usize >> 6] |= 1 << (v & 63);
  }
  pub fn contains(&self, v: u8) -> bool {
    self.0[v as usize >> 6] >> (v & 63) & 1 == 1
  }
  pub fn len(&self) -> usize {
    self.0.iter().map(|w| w.count_ones() as usize).sum()
  }
  pub fn is_empty(&self) -> bool {
    self.0 == [0; 4]
  }
  pub fn intersects(&self, o: &StateSet) -> bool {
    (0..4).any(|i| self.0[i] & o.0[i] != 0)
  }
  pub fn iter(&self) -> impl Iterator<Item = u8> + '_ {
    (0..=255u8).filter(move |&v| self.contains(v))
  }
}

/// What a transition writes: a state, or the state of one of its input cells.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Out {
  Const(u8),
  Copy(usize),
}

/// One transition under one symmetry: the states allowed in each input cell, the groups of input
/// cells bound to one variable (equal states), and the output.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Case {
  pub sets: Vec<StateSet>,
  pub eq: Vec<Vec<usize>>,
  pub out: Out,
}

#[derive(Clone, Debug)]
pub struct Table {
  pub name: String,
  pub states: usize,
  pub nbhd: Neighbourhood,
  pub symmetries: String,
  pub variables: usize,
  /// transitions as written: one field per input cell, then the output
  pub lines: Vec<Vec<Field>>,
}

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Field {
  State(u8),
  /// index of the variable, and its states
  Var(usize, StateSet),
}

impl Table {
  pub fn load(path: &Path) -> Result<Table, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let name = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    Table::parse(&name, &text)
  }

  pub fn parse(name: &str, text: &str) -> Result<Table, String> {
    // a bare `.table` has no sections: all of it is the table
    let mut in_table = !text.lines().any(|l| l.trim_start().starts_with('@'));
    let (mut states, mut nbhd, mut symmetries) = (0usize, Neighbourhood::VonNeumann, "none".to_string());
    let mut vars: HashMap<String, (usize, StateSet)> = HashMap::new();
    let mut raw: Vec<Vec<String>> = Vec::new();
    for line in text.lines() {
      let line = line.split('#').next().unwrap().trim();
      if line.starts_with('@') {
        in_table = line.split_whitespace().next() == Some("@TABLE");
        continue;
      }
      if !in_table || line.is_empty() {
        continue;
      }
      if let Some((key, val)) = line.split_once(':') {
        let (key, val) = (key.trim(), val.trim());
        match key {
          "n_states" => {
            states = val.parse().map_err(|_| format!("{name}: bad n_states {val}"))?;
            continue;
          }
          "neighborhood" | "neighbourhood" => {
            nbhd = Neighbourhood::from_golly_name(val).ok_or_else(|| format!("{name}: neighbourhood {val} is not supported"))?;
            continue;
          }
          "symmetries" => {
            symmetries = val.to_string();
            continue;
          }
          _ => {}
        }
      }
      if let Some(rest) = line.strip_prefix("var ") {
        let (vname, set) = rest.split_once('=').ok_or_else(|| format!("{name}: bad variable: {line}"))?;
        let set = set.trim().trim_start_matches('{').trim_end_matches('}');
        let mut s = StateSet::default();
        for tok in set.split(',').map(str::trim).filter(|t| !t.is_empty()) {
          match tok.parse::<u8>() {
            Ok(v) => s.insert(v),
            Err(_) => {
              let (_, other) = vars.get(tok).ok_or_else(|| format!("{name}: unknown variable {tok} in: {line}"))?;
              for v in other.iter() {
                s.insert(v);
              }
            }
          }
        }
        let idx = vars.len();
        vars.insert(vname.trim().to_string(), (idx, s));
        continue;
      }
      // a transition; tables with single-digit states and no variables may write it as 012345
      let toks: Vec<String> = if line.contains(',') {
        line.split(',').map(|t| t.trim().to_string()).collect()
      } else {
        line.chars().filter(|c| !c.is_whitespace()).map(|c| c.to_string()).collect()
      };
      raw.push(toks);
    }
    if !(2..=256).contains(&states) {
      return Err(format!("{name}: n_states missing or out of range"));
    }
    let k = nbhd.cells();
    let mut lines = Vec::with_capacity(raw.len());
    for toks in raw {
      if toks.len() != k + 1 {
        return Err(format!("{name}: transition {} has {} fields, not {}", toks.join(","), toks.len(), k + 1));
      }
      let mut fields = Vec::with_capacity(k + 1);
      for t in &toks {
        fields.push(match vars.get(t.as_str()) {
          Some(&(i, s)) => Field::Var(i, s),
          None => {
            let v: usize = t.parse().map_err(|_| format!("{name}: bad field {t} in transition {}", toks.join(",")))?;
            if v >= states {
              return Err(format!("{name}: state {v} in transition {} is not a state", toks.join(",")));
            }
            Field::State(v as u8)
          }
        });
      }
      lines.push(fields);
    }
    Ok(Table { name: name.to_string(), states, nbhd, symmetries, variables: vars.len(), lines })
  }

  /// Bits of a state: states are coded in binary as the table numbers them.
  pub fn bits(&self) -> usize {
    crate::bits_for(self.states)
  }

  /// Cells of the neighbourhood, the centre included. Cell i of a transition, of a [`Case`] and
  /// of a neighbourhood given to [`Interp`] is `self.nbhd.table_order()[i]`.
  pub fn cells(&self) -> usize {
    self.nbhd.cells()
  }

  /// The symmetry group as permutations of the input cells (the centre, cell 0, stays), or
  /// `None` for `permute`, which is expanded by arrangements instead.
  fn permutations(&self) -> Result<Option<Vec<Vec<usize>>>, String> {
    let k = self.cells();
    let nb: Vec<usize> = (1..k).collect();
    let identity: Vec<usize> = (0..k).collect();
    let rotations = |count: usize| -> Vec<Vec<usize>> {
      let step = nb.len() / count;
      (0..count).map(|r| std::iter::once(0).chain((0..nb.len()).map(|i| nb[(i + r * step) % nb.len()])).collect()).collect()
    };
    // the mirror image keeps the first neighbour and reverses the others
    let mirror: Vec<usize> = if self.nbhd == Neighbourhood::OneDimensional { vec![0, 2, 1] } else { [0, 1].into_iter().chain((2..k).rev()).collect() };
    let sym = self.symmetries.as_str();
    Ok(Some(match sym {
      "none" => vec![identity],
      "permute" => return Ok(None),
      "reflect_horizontal" | "reflect" => vec![identity, mirror],
      _ => {
        let (count, reflect) = match sym.strip_prefix("rotate") {
          Some(rest) => match rest.strip_suffix("reflect") {
            Some(c) => (c, true),
            None => (rest, false),
          },
          None => return Err(format!("{}: symmetries {sym} are not supported", self.name)),
        };
        let count: usize = count.parse().map_err(|_| format!("{}: symmetries {sym} are not supported", self.name))?;
        if count == 0 || nb.len() % count != 0 {
          return Err(format!("{}: symmetries {sym} do not fit the {} neighbourhood", self.name, self.nbhd.golly_name()));
        }
        let mut all = rotations(count);
        if reflect {
          let more: Vec<Vec<usize>> = all.iter().map(|r| (0..k).map(|i| r[mirror[i]]).collect()).collect();
          all.extend(more);
        }
        all
      }
    }))
  }

  /// Every transition under every symmetry, in the table's order of priority, without repeats.
  pub fn cases(&self) -> Result<Vec<Case>, String> {
    let k = self.cells();
    let perms = self.permutations()?;
    let mut out: Vec<Case> = Vec::new();
    let mut seen: HashSet<Case> = HashSet::new();
    for line in &self.lines {
      let (ins, res) = line.split_at(k);
      let sets: Vec<StateSet> = ins
        .iter()
        .map(|f| match f {
          Field::State(v) => StateSet::single(*v),
          Field::Var(_, s) => *s,
        })
        .collect();
      // cells bound to the same variable
      let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
      for (i, f) in ins.iter().enumerate() {
        if let Field::Var(v, _) = f {
          match groups.iter_mut().find(|g| g.0 == *v) {
            Some(g) => g.1.push(i),
            None => groups.push((*v, vec![i])),
          }
        }
      }
      let o = match &res[0] {
        Field::State(v) => Out::Const(*v),
        Field::Var(v, _) => match groups.iter().find(|g| g.0 == *v) {
          Some(g) => Out::Copy(g.1[0]),
          None => return Err(format!("{}: the output variable of a transition is not among its inputs", self.name)),
        },
      };
      let eq: Vec<Vec<usize>> = groups.into_iter().map(|g| g.1).filter(|g| g.len() > 1).collect();
      let images: Vec<Case> = match &perms {
        Some(ps) => ps.iter().map(|p| permuted(&sets, &eq, o, p)).collect(),
        None => arrangements(&sets, &eq, o)?,
      };
      for c in images {
        if seen.insert(c.clone()) {
          out.push(c);
          if out.len() > MAX_CASES {
            return Err(format!("{}: more than {MAX_CASES} transitions after expanding symmetries {}", self.name, self.symmetries));
          }
        }
      }
    }
    Ok(out)
  }
}

/// The image of a transition under a permutation of its input cells: cell i takes field `perm[i]`.
fn permuted(sets: &[StateSet], eq: &[Vec<usize>], o: Out, perm: &[usize]) -> Case {
  let mut inv = vec![0usize; perm.len()];
  for (new, &old) in perm.iter().enumerate() {
    inv[old] = new;
  }
  let mut e: Vec<Vec<usize>> = eq
    .iter()
    .map(|g| {
      let mut g: Vec<usize> = g.iter().map(|&i| inv[i]).collect();
      g.sort_unstable();
      g
    })
    .collect();
  e.sort();
  Case {
    sets: perm.iter().map(|&p| sets[p]).collect(),
    eq: e,
    out: match o {
      Out::Copy(i) => Out::Copy(inv[i]),
      c => c,
    },
  }
}

/// `permute`: the images under every permutation of the neighbours, without repeats, as the
/// distinct arrangements of the neighbours' fields.
fn arrangements(sets: &[StateSet], eq: &[Vec<usize>], o: Out) -> Result<Vec<Case>, String> {
  let k = sets.len();
  // a neighbour's field: its set, the group it is bound in, whether the output copies it
  let label = |i: usize| eq.iter().position(|g| g.contains(&i));
  let fields: Vec<(StateSet, Option<usize>, bool)> = (1..k).map(|i| (sets[i], label(i), o == Out::Copy(i))).collect();
  let mut kinds: Vec<(StateSet, Option<usize>, bool)> = Vec::new();
  let mut count: Vec<usize> = Vec::new();
  for f in &fields {
    match kinds.iter().position(|x| x == f) {
      Some(i) => count[i] += 1,
      None => {
        kinds.push(*f);
        count.push(1);
      }
    }
  }
  let mut out = Vec::new();
  let mut cur: Vec<usize> = Vec::with_capacity(k - 1);
  fn rec(kinds: &[(StateSet, Option<usize>, bool)], count: &mut [usize], cur: &mut Vec<usize>, total: usize, visit: &mut dyn FnMut(&[usize]) -> bool) -> bool {
    if cur.len() == total {
      return visit(cur);
    }
    for i in 0..kinds.len() {
      if count[i] > 0 {
        count[i] -= 1;
        cur.push(i);
        let go = rec(kinds, count, cur, total, visit);
        cur.pop();
        count[i] += 1;
        if !go {
          return false;
        }
      }
    }
    true
  }
  let centre_group = label(0);
  let done = rec(&kinds, &mut count, &mut cur, k - 1, &mut |arr: &[usize]| {
    let mut s = vec![sets[0]];
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    if let Some(g) = centre_group {
      groups.push((g, vec![0]));
    }
    let mut out_case = o;
    for (pos, &ki) in arr.iter().enumerate() {
      let (set, grp, copied) = kinds[ki];
      s.push(set);
      if let Some(g) = grp {
        match groups.iter_mut().find(|x| x.0 == g) {
          Some(x) => x.1.push(pos + 1),
          None => groups.push((g, vec![pos + 1])),
        }
      }
      if copied {
        if let Out::Copy(_) = out_case {
          out_case = Out::Copy(pos + 1);
        }
      }
    }
    let mut e: Vec<Vec<usize>> = groups.into_iter().map(|g| g.1).filter(|g| g.len() > 1).collect();
    e.sort();
    // a copied neighbour that is bound: every member of its group holds the same state
    out.push(Case { sets: s, eq: e, out: out_case });
    out.len() <= MAX_CASES
  });
  if !done {
    return Err(format!("more than {MAX_CASES} images of one transition under permute"));
  }
  Ok(out)
}

/// The table's meaning: the next state of a neighbourhood, by the first matching case.
pub struct Interp<'a> {
  cases: &'a [Case],
  n: usize,
  k: usize,
  words: usize,
  /// for each (cell, state): the cases whose set of that cell holds the state
  rows: Vec<u64>,
}

impl<'a> Interp<'a> {
  pub fn new(cases: &'a [Case], n: usize, k: usize) -> Self {
    let words = cases.len().div_ceil(64).max(1);
    let mut rows = vec![0u64; k * n * words];
    for (ci, c) in cases.iter().enumerate() {
      for (i, s) in c.sets.iter().enumerate() {
        for v in s.iter().filter(|&v| (v as usize) < n) {
          rows[(i * n + v as usize) * words + ci / 64] |= 1 << (ci % 64);
        }
      }
    }
    Interp { cases, n, k, words, rows }
  }

  pub fn words(&self) -> usize {
    self.words
  }

  /// `nb[i]` is the state of cell i; `scratch` has `words()` entries.
  pub fn next(&self, nb: &[u8], scratch: &mut [u64]) -> u8 {
    let w = self.words;
    scratch.copy_from_slice(&self.rows[nb[0] as usize * w..nb[0] as usize * w + w]);
    for i in 1..self.k {
      let row = &self.rows[(i * self.n + nb[i] as usize) * w..(i * self.n + nb[i] as usize) * w + w];
      for (s, r) in scratch.iter_mut().zip(row) {
        *s &= r;
      }
    }
    for (wi, &word) in scratch.iter().enumerate() {
      let mut m = word;
      while m != 0 {
        let c = &self.cases[wi * 64 + m.trailing_zeros() as usize];
        if c.eq.iter().all(|g| g[1..].iter().all(|&j| nb[j] == nb[g[0]])) {
          return match c.out {
            Out::Const(v) => v,
            Out::Copy(i) => nb[i],
          };
        }
        m &= m - 1;
      }
    }
    nb[0]
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  fn next(t: &Table, nb: &[u8]) -> u8 {
    let cases = t.cases().unwrap();
    let i = Interp::new(&cases, t.states, t.cells());
    let mut scratch = vec![0u64; i.words()];
    i.next(nb, &mut scratch)
  }

  #[test]
  fn first_match_and_default() {
    let t = Table::parse("t", "n_states:3\nneighborhood:vonNeumann\nsymmetries:none\nvar a={0,1,2}\n0,1,a,a,a,2\n0,1,0,0,0,1\n").unwrap();
    assert_eq!(t.cases().unwrap().len(), 2);
    // the first line wins; three different variables would not be bound, one variable is
    assert_eq!(next(&t, &[0, 1, 0, 0, 0]), 2);
    assert_eq!(next(&t, &[0, 1, 2, 2, 2]), 2);
    assert_eq!(next(&t, &[0, 1, 0, 1, 0]), 0);
    assert_eq!(next(&t, &[1, 1, 0, 0, 0]), 1);
  }

  #[test]
  fn rotations_and_copy() {
    // a cell copies a non-zero northern neighbour, in any of the four orientations
    let t = Table::parse("t", "n_states:4\nneighborhood:vonNeumann\nsymmetries:rotate4\nvar a={1,2,3}\n0,a,0,0,0,a\n").unwrap();
    assert_eq!(t.cases().unwrap().len(), 4);
    assert_eq!(next(&t, &[0, 3, 0, 0, 0]), 3);
    assert_eq!(next(&t, &[0, 0, 2, 0, 0]), 2);
    assert_eq!(next(&t, &[0, 0, 0, 0, 1]), 1);
    assert_eq!(next(&t, &[0, 3, 2, 0, 0]), 0);
  }

  #[test]
  fn permute_and_reflect() {
    // exactly one live neighbour, anywhere among the eight
    let t = Table::parse("t", "n_states:2\nneighborhood:Moore\nsymmetries:permute\n0,1,0,0,0,0,0,0,0,1\n").unwrap();
    assert_eq!(t.cases().unwrap().len(), 8);
    assert_eq!(next(&t, &[0, 0, 0, 0, 0, 1, 0, 0, 0]), 1);
    assert_eq!(next(&t, &[0, 1, 0, 0, 0, 1, 0, 0, 0]), 0);
    let r = Table::parse("t", "n_states:2\nneighborhood:vonNeumann\nsymmetries:rotate4reflect\n0,1,1,0,0,1\n").unwrap();
    assert_eq!(r.cases().unwrap().len(), 4);
    assert_eq!(next(&r, &[0, 1, 0, 0, 1]), 1);
    assert_eq!(next(&r, &[0, 1, 0, 1, 0]), 0);
  }

  #[test]
  fn compressed_lines_and_sections() {
    let t = Table::parse("t", "@RULE t\n@TABLE\nn_states:3\nneighborhood:oneDimensional\nsymmetries:reflect\n0122\n@COLORS\n1 255 0 0\n").unwrap();
    assert_eq!((t.nbhd, t.lines.len(), t.cells(), t.bits()), (Neighbourhood::OneDimensional, 1, 3, 2));
    assert_eq!(next(&t, &[0, 1, 2]), 2);
    assert_eq!(next(&t, &[0, 2, 1]), 2);
    assert_eq!(next(&t, &[0, 1, 1]), 0);
  }
}
