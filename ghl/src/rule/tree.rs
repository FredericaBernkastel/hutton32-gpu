//! Golly's rule trees (`@TREE`): a decision DAG over the neighbourhood. A node of level `j > 1`
//! has one child per state of the next variable; a level-1 node lists the next state per state of
//! the centre. Golly's variable order (`ruletreealgo.cpp`) is N, W, E, S, C for four neighbours and
//! NW, NE, SW, SE, N, W, E, S, C for eight; the root is the last node of the file.

use super::{Neighbourhood, Pos};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RuleTree {
  states: u32,
  neighbors: u32,
  /// level of every node
  levels: Vec<u8>,
  /// `states` entries per node: child node indices (level > 1) or next states (level 1)
  data: Vec<u32>,
  root: u32,
}

impl RuleTree {
  /// The variable order of a tree with `neighbors` neighbours (the centre last).
  pub fn order(neighbors: u32) -> &'static [Pos] {
    if neighbors == 8 {
      &[Pos::NW, Pos::NE, Pos::SW, Pos::SE, Pos::N, Pos::W, Pos::E, Pos::S, Pos::C]
    } else {
      &[Pos::N, Pos::W, Pos::E, Pos::S, Pos::C]
    }
  }

  /// Parse the body of a `@TREE` section (or a `.tree` file).
  pub fn parse(body: &str) -> Result<RuleTree, String> {
    let (mut states, mut neighbors, mut num_nodes) = (0u32, 0u32, 0usize);
    let mut levels = Vec::new();
    let mut data = Vec::new();
    for raw in body.lines() {
      let line = raw.split('#').next().unwrap().trim();
      if line.is_empty() {
        continue;
      }
      if let Some((k, v)) = line.split_once('=') {
        let v: u64 = v.trim().parse().map_err(|_| format!("bad header line: {line}"))?;
        match k.trim() {
          "num_states" => states = v as u32,
          "num_neighbors" => neighbors = v as u32,
          "num_nodes" => num_nodes = v as usize,
          other => return Err(format!("unknown header {other}")),
        }
        continue;
      }
      if states == 0 || neighbors == 0 {
        return Err("node line before num_states / num_neighbors".into());
      }
      let v: Vec<u32> = line.split_whitespace().map(|t| t.parse::<u32>()).collect::<Result<_, _>>().map_err(|_| format!("bad node line: {line}"))?;
      if v.len() != states as usize + 1 {
        return Err(format!("node line with {} entries, expected {}: {line}", v.len() - 1, states));
      }
      let level = v[0];
      if level == 0 || level > neighbors + 1 {
        return Err(format!("bad level {level}"));
      }
      for &x in &v[1..] {
        if level == 1 {
          if x >= states {
            return Err(format!("next state {x} out of range"));
          }
        } else if x as usize >= levels.len() || levels[x as usize] as u32 != level - 1 {
          return Err(format!("child {x} is not a level-{} node defined before", level - 1));
        }
      }
      levels.push(level as u8);
      data.extend_from_slice(&v[1..]);
    }
    if !(neighbors == 4 || neighbors == 8) {
      return Err(format!("num_neighbors must be 4 or 8, not {neighbors}"));
    }
    if !(1..=256).contains(&states) {
      return Err(format!("num_states must be 1 to 256, not {states}"));
    }
    if levels.is_empty() || (num_nodes != 0 && num_nodes != levels.len()) {
      return Err(format!("num_nodes says {num_nodes}, the file has {}", levels.len()));
    }
    let root = levels.len() as u32 - 1;
    if levels[root as usize] as u32 != neighbors + 1 {
      return Err("the last node is not the root".into());
    }
    Ok(RuleTree { states, neighbors, levels, data, root })
  }

  /// Build a tree from a function of the neighbourhood in tree order, sharing identical nodes.
  pub fn from_fn(states: u32, neighbors: u32, f: &mut dyn FnMut(&[u8]) -> u8) -> RuleTree {
    use rustc_hash::FxHashMap;
    let mut levels: Vec<u8> = Vec::new();
    let mut data: Vec<u32> = Vec::new();
    let mut memo: FxHashMap<(u8, Vec<u32>), u32> = FxHashMap::default();
    let vars = neighbors as usize + 1;
    fn rec(
      depth: usize,
      vars: usize,
      states: u32,
      at: &mut Vec<u8>,
      f: &mut dyn FnMut(&[u8]) -> u8,
      levels: &mut Vec<u8>,
      data: &mut Vec<u32>,
      memo: &mut FxHashMap<(u8, Vec<u32>), u32>,
    ) -> u32 {
      let level = (vars - depth) as u8;
      let mut row = Vec::with_capacity(states as usize);
      for s in 0..states {
        at.push(s as u8);
        let v = if depth + 1 == vars { f(at) as u32 } else { rec(depth + 1, vars, states, at, f, levels, data, memo) };
        at.pop();
        row.push(v);
      }
      let key = (level, row);
      if let Some(&id) = memo.get(&key) {
        return id;
      }
      let id = levels.len() as u32;
      levels.push(level);
      data.extend_from_slice(&key.1);
      memo.insert(key, id);
      id
    }
    let mut at = Vec::with_capacity(vars);
    let root = rec(0, vars, states, &mut at, f, &mut levels, &mut data, &mut memo);
    // the root must come last
    let mut t = RuleTree { states, neighbors, levels, data, root };
    t.renumber_root_last();
    t
  }

  fn renumber_root_last(&mut self) {
    let n = self.levels.len();
    let r = self.root as usize;
    if r == n - 1 {
      return;
    }
    // nodes are created children first, so the root is the last node of its level; swap it with
    // the last node, which has the same top level
    let s = self.states as usize;
    let last = n - 1;
    debug_assert_eq!(self.levels[last], self.levels[r]);
    for i in 0..s {
      self.data.swap(r * s + i, last * s + i);
    }
    self.root = last as u32;
  }

  pub fn states(&self) -> u32 {
    self.states
  }
  pub fn neighbours(&self) -> u32 {
    self.neighbors
  }
  pub fn nodes(&self) -> usize {
    self.levels.len()
  }
  pub fn root(&self) -> u32 {
    self.root
  }
  /// `states` entries per node: children or next states.
  pub fn data(&self) -> &[u32] {
    &self.data
  }
  pub fn level(&self, node: u32) -> u32 {
    self.levels[node as usize] as u32
  }
  pub fn neighbourhood(&self) -> Neighbourhood {
    if self.neighbors == 8 {
      Neighbourhood::Moore
    } else {
      Neighbourhood::VonNeumann
    }
  }

  /// The next state for the variables in tree order (`order(neighbours)`).
  #[inline]
  pub fn eval(&self, vars: &[u8]) -> u8 {
    let s = self.states as usize;
    let mut node = self.root as usize;
    for &v in vars {
      node = self.data[node * s + v as usize] as usize;
    }
    node as u8
  }

  /// The next state for the cells in the field order of the tree's neighbourhood.
  #[inline]
  pub fn eval_fields(&self, cells: &[u8]) -> u8 {
    let nbhd = self.neighbourhood();
    let mut vars = [0u8; 9];
    for (i, &p) in Self::order(self.neighbors).iter().enumerate() {
      vars[i] = cells[nbhd.field_of(p).unwrap()];
    }
    self.eval(&vars[..self.neighbors as usize + 1])
  }

  /// The table in the index layout of `nbhd` with `bits` bits per field.
  pub fn to_table(&self, nbhd: Neighbourhood, bits: u32) -> Vec<u8> {
    let cells = nbhd.cells();
    let n = 1usize << (bits as usize * cells);
    let mask = (1usize << bits) - 1;
    let order = Self::order(self.neighbors);
    let field_of: Vec<usize> = order.iter().map(|&p| nbhd.field_of(p).unwrap()).collect();
    let mut t = vec![0u8; n];
    let mut vars = [0u8; 9];
    for (i, out) in t.iter_mut().enumerate() {
      let mut ok = true;
      for (j, &f) in field_of.iter().enumerate() {
        let st = (i >> (bits as usize * f)) & mask;
        if st >= self.states as usize {
          ok = false;
          break;
        }
        vars[j] = st as u8;
      }
      *out = if ok { self.eval(&vars[..order.len()]) } else { 0 };
    }
    t
  }

  /// The tree as the body of a `@TREE` section.
  pub fn to_text(&self) -> String {
    let s = self.states as usize;
    let mut out = format!("num_states={}\nnum_neighbors={}\nnum_nodes={}\n", self.states, self.neighbors, self.levels.len());
    for (i, &l) in self.levels.iter().enumerate() {
      out += &l.to_string();
      for v in &self.data[i * s..(i + 1) * s] {
        out.push(' ');
        out += &v.to_string();
      }
      out.push('\n');
    }
    out
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn life_as_tree_round_trip() {
    // Life: born with 3, survives with 2 or 3, as a tree over 8 neighbours
    let mut f = |v: &[u8]| -> u8 {
      let n: u32 = v[..8].iter().map(|&x| x as u32).sum();
      let c = v[8];
      ((c == 1 && (n == 2 || n == 3)) || (c == 0 && n == 3)) as u8
    };
    let t = RuleTree::from_fn(2, 8, &mut f);
    let t2 = RuleTree::parse(&t.to_text()).unwrap();
    assert_eq!(t, t2);
    let table = t.to_table(Neighbourhood::Moore, 1);
    let life = crate::rule::bs::life_like_table(8, 12, Neighbourhood::Moore);
    assert_eq!(table, life);
  }
}
