//! From the transitions of a table to a network that computes the next state.
//!
//! A transition matches where every cell holds a state of its set and bound cells are equal. The
//! network has one predicate per (cell, set), one comparator per bound pair and one AND chain per
//! transition; the order of the table is kept only between transitions that can match the same
//! neighbourhood with different results.

use crate::check::Space;
use crate::factor::{cubes_of, fast_extract, Sop};
use crate::net::{not, FxMap, Lit, Net, FALSE, TRUE};
use crate::symmetry::Symmetry;
use ca_rule::table::{Case, Out, StateSet, Table};

/// How the matches are combined into the next state.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Style {
  /// a transition fires where it matches and no earlier transition with another result does; a
  /// bit of the state flips where a firing transition changes it
  Flat,
  /// transitions with one result are grouped, and the groups are chained in order of priority
  Chain,
  /// a chain per bit of the state, each from the transitions simplified for that bit alone
  Bits,
  /// as `Bits`, with the conditions of the groups factored: the centre's sets as cubes over its
  /// bits, and the divisors that the products share extracted ([`crate::factor`])
  Factor,
  /// the rule's rotational symmetry used: the centre states that have an orientation share one
  /// table, written for orientation 0 and read through the centre's orientation; the other
  /// centre states as in `Factor`
  Fold,
  /// as `Fold`, with the transitions made from the rule's function ([`crate::cover`]) instead of
  /// taken from the table's text; without a symmetry every centre state counts as having no
  /// orientation
  Cover,
  /// not built: a netlist given by the caller
  Given,
}

/// What a transition does to the centre cell.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Effect {
  Keep,
  Const(u8),
  Copy(usize),
}

fn effect(c: &Case) -> Effect {
  match c.out {
    Out::Copy(0) => Effect::Keep,
    Out::Copy(i) if c.eq.iter().any(|g| g.contains(&0) && g.contains(&i)) => Effect::Keep,
    Out::Copy(i) => Effect::Copy(i),
    Out::Const(v) if c.sets[0] == StateSet::single(v) => Effect::Keep,
    Out::Const(v) => Effect::Const(v),
  }
}

/// What a transition does to one bit of the centre cell.
fn bit_effect(c: &Case, b: usize) -> Effect {
  match effect(c) {
    Effect::Const(v) => {
      let bit = v >> b & 1;
      if c.sets[0].iter().all(|s| s >> b & 1 == bit) {
        Effect::Keep
      } else {
        Effect::Const(bit)
      }
    }
    e => e,
  }
}

fn and_sets(a: &StateSet, b: &StateSet) -> StateSet {
  StateSet([a.0[0] & b.0[0], a.0[1] & b.0[1], a.0[2] & b.0[2], a.0[3] & b.0[3]])
}

/// Whether some neighbourhood can match both transitions (an over-estimate).
fn overlap(a: &Case, b: &Case) -> bool {
  if !a.sets.iter().zip(&b.sets).all(|(x, y)| x.intersects(y)) {
    return false;
  }
  for g in a.eq.iter().chain(&b.eq) {
    let mut s = StateSet([!0; 4]);
    for &m in g {
      s = and_sets(&s, &and_sets(&a.sets[m], &b.sets[m]));
    }
    if s.is_empty() {
      return false;
    }
  }
  true
}

/// Whether two overlapping transitions can give different states where both match.
fn differ(a: &Case, ea: Effect, b: &Case, eb: Effect) -> bool {
  if ea == eb {
    return false;
  }
  let centre = and_sets(&a.sets[0], &b.sets[0]);
  match (ea, eb) {
    (Effect::Keep, Effect::Const(v)) | (Effect::Const(v), Effect::Keep) => centre != StateSet::single(v),
    _ => true,
  }
}

// ---- membership predicates ---------------------------------------------------------------------

/// The codes with bit `b` set.
fn half(b: usize) -> StateSet {
  const P: [u64; 6] = [0xAAAA_AAAA_AAAA_AAAA, 0xCCCC_CCCC_CCCC_CCCC, 0xF0F0_F0F0_F0F0_F0F0, 0xFF00_FF00_FF00_FF00, 0xFFFF_0000_FFFF_0000, 0xFFFF_FFFF_0000_0000];
  match b {
    0..=5 => StateSet([P[b]; 4]),
    6 => StateSet([0, !0, 0, !0]),
    _ => StateSet([0, 0, !0, !0]),
  }
}

/// The codes of `s` with bit `b` set, with that bit cleared.
fn down(s: &StateSet, b: usize) -> StateSet {
  let h = half(b);
  let m = and_sets(s, &h).0;
  match b {
    0..=5 => StateSet([m[0] >> (1 << b), m[1] >> (1 << b), m[2] >> (1 << b), m[3] >> (1 << b)]),
    6 => StateSet([m[1], 0, m[3], 0]),
    _ => StateSet([m[2], m[3], 0, 0]),
  }
}

fn minus(a: &StateSet, b: &StateSet) -> StateSet {
  StateSet([a.0[0] & !b.0[0], a.0[1] & !b.0[1], a.0[2] & !b.0[2], a.0[3] & !b.0[3]])
}

fn union(a: &StateSet, b: &StateSet) -> StateSet {
  StateSet([a.0[0] | b.0[0], a.0[1] | b.0[1], a.0[2] | b.0[2], a.0[3] | b.0[3]])
}

/// The order of the bits that gives the fewest gates for the table's sets.
fn bit_order(bits: usize, valid: &StateSet, sets: &[StateSet]) -> Vec<usize> {
  let mut orders: Vec<Vec<usize>> = Vec::new();
  if bits <= 5 {
    // every order
    let mut p: Vec<usize> = (0..bits).collect();
    fn perms(p: &mut Vec<usize>, i: usize, out: &mut Vec<Vec<usize>>) {
      if i == p.len() {
        out.push(p.clone());
        return;
      }
      for j in i..p.len() {
        p.swap(i, j);
        perms(p, i + 1, out);
        p.swap(i, j);
      }
    }
    perms(&mut p, 0, &mut orders);
  } else {
    let mut rng = crate::check::Rng(bits as u64);
    for r in 0..bits {
      orders.push((0..bits).map(|i| (i + r) % bits).collect());
      orders.push((0..bits).rev().map(|i| (i + r) % bits).collect());
    }
    for _ in 0..64 {
      let mut p: Vec<usize> = (0..bits).collect();
      for i in (1..bits).rev() {
        p.swap(i, rng.below(i + 1));
      }
      orders.push(p);
    }
  }
  let mut best = (usize::MAX, orders[0].clone());
  for o in orders {
    let mut b = Builder::new(Space { n: 0, k: 1, bits }, *valid, o.clone());
    for s in sets {
      b.member(0, *s, *valid, 0);
    }
    if b.net.len() < best.0 {
      best = (b.net.len(), o);
    }
  }
  best.1
}

// ---- the network -------------------------------------------------------------------------------

struct Builder {
  net: Net,
  space: Space,
  /// `[cell][bit]`
  cell_bits: Vec<Vec<Lit>>,
  valid: StateSet,
  order: Vec<usize>,
  memo: Vec<FxMap<(StateSet, StateSet, u8), Lit>>,
  equal: FxMap<(usize, usize), Lit>,
}

impl Builder {
  fn new(space: Space, valid: StateSet, order: Vec<usize>) -> Builder {
    let net = Net::new(space.inputs());
    let cell_bits = (0..space.k).map(|c| (0..space.bits).map(|b| net.input(c * space.bits + b)).collect()).collect();
    Builder { net, space, cell_bits, valid, order, memo: vec![FxMap::default(); space.k], equal: FxMap::default() }
  }

  /// A function of a cell's bits that is true on `on` and false on the rest of `care`; codes
  /// outside `care` (no state has them) are free. Shannon expansion over the bits in `order`,
  /// merging the two halves where they agree on the codes both care about.
  fn member(&mut self, cell: usize, on: StateSet, care: StateSet, d: usize) -> Lit {
    if on.is_empty() {
      return FALSE;
    }
    if on == care {
      return TRUE;
    }
    if let Some(&l) = self.memo[cell].get(&(on, care, d as u8)) {
      return l;
    }
    let b = self.order[d];
    let var = self.cell_bits[cell][b];
    let h = half(b);
    let (on0, care0) = (minus(&on, &h), minus(&care, &h));
    let (on1, care1) = (down(&on, b), down(&care, b));
    let (off0, off1) = (minus(&care0, &on0), minus(&care1, &on1));
    let l = if care1.is_empty() {
      self.member(cell, on0, care0, d + 1)
    } else if care0.is_empty() {
      self.member(cell, on1, care1, d + 1)
    } else if !on0.intersects(&off1) && !on1.intersects(&off0) {
      // the bit does not matter
      self.member(cell, union(&on0, &on1), union(&care0, &care1), d + 1)
    } else if !on0.intersects(&on1) && !off0.intersects(&off1) {
      // the halves are complements
      let g = self.member(cell, union(&on0, &off1), union(&care0, &care1), d + 1);
      self.net.xor(var, g)
    } else {
      let f0 = self.member(cell, on0, care0, d + 1);
      let f1 = self.member(cell, on1, care1, d + 1);
      self.net.mux(var, f1, f0)
    };
    self.memo[cell].insert((on, care, d as u8), l);
    l
  }

  /// Where each transition matches: the AND of its predicates and its comparators.
  fn matches(&mut self, cases: &[Case]) -> Vec<Lit> {
    let bits = self.space.bits;
    let mut terms: Vec<Vec<Lit>> = Vec::with_capacity(cases.len());
    for c in cases {
      let mut t: Vec<Lit> = Vec::new();
      for (i, s) in c.sets.iter().enumerate() {
        let s = and_sets(s, &self.valid);
        if s != self.valid {
          t.push(self.member(i, s, self.valid, 0));
        }
      }
      for g in &c.eq {
        for &j in &g[1..] {
          let key = (g[0].min(j), g[0].max(j));
          let e = match self.equal.get(&key) {
            Some(&e) => e,
            None => {
              let mut e = TRUE;
              for b in 0..bits {
                let x = self.net.xor(self.cell_bits[key.0][b], self.cell_bits[key.1][b]);
                e = self.net.and(e, not(x));
              }
              self.equal.insert(key, e);
              e
            }
          };
          t.push(e);
        }
      }
      terms.push(t);
    }
    extract(&mut self.net, terms, true)
  }
}

/// Products (or sums) of literals as gates, sharing pairs: the pair of literals that the most
/// terms have in common becomes one gate, used by all of them, until no pair is shared.
fn extract(net: &mut Net, mut terms: Vec<Vec<Lit>>, and: bool) -> Vec<Lit> {
  let (unit, zero) = if and { (TRUE, FALSE) } else { (FALSE, TRUE) };
  for t in terms.iter_mut() {
    t.sort_unstable();
    t.dedup();
    t.retain(|&l| l != unit);
    if t.contains(&zero) || t.windows(2).any(|w| w[0] == not(w[1])) {
      *t = vec![zero];
    }
  }
  let mut rounds = 0;
  loop {
    let mut count: FxMap<(Lit, Lit), u32> = FxMap::default();
    for t in &terms {
      if t.len() > 12 {
        continue;
      }
      for i in 0..t.len() {
        for j in i + 1..t.len() {
          *count.entry((t[i], t[j])).or_default() += 1;
        }
      }
    }
    let Some((&(a, b), _)) = count.iter().filter(|c| *c.1 >= 2).max_by_key(|c| (*c.1, std::cmp::Reverse(*c.0))) else { break };
    let g = if and { net.and(a, b) } else { net.or(a, b) };
    for t in terms.iter_mut() {
      if t.contains(&a) && t.contains(&b) {
        t.retain(|&l| l != a && l != b);
        if g == zero {
          *t = vec![zero];
        } else if g != unit && !t.contains(&g) {
          t.push(g);
          t.sort_unstable();
        }
      }
    }
    rounds += 1;
    if rounds > 4000 {
      break;
    }
  }
  terms
    .into_iter()
    .map(|t| {
      let mut m = unit;
      for l in t {
        m = if and { net.and(m, l) } else { net.or(m, l) };
      }
      m
    })
    .collect()
}

impl Builder {
  /// Where each group of each chain applies, the groups' conditions factored together. A
  /// transition is a product of literals: the bits of the centre (its set as cubes over them),
  /// one predicate per other cell, one comparator per bound pair.
  fn factored(&mut self, cases: &[Vec<Case>], chains: &[(usize, Vec<(Effect, Vec<usize>)>)]) -> Vec<Vec<Lit>> {
    let (bits, n) = (self.space.bits, self.space.n);
    // variables: the centre's bits, then predicates and comparators as they come
    let mut leaves: Vec<Lit> = self.cell_bits[0].clone();
    let mut pred: FxMap<(usize, StateSet), u32> = FxMap::default();
    let mut cmp: FxMap<(usize, usize), u32> = FxMap::default();
    let mut centre_cubes: FxMap<StateSet, Vec<Vec<u32>>> = FxMap::default();
    let mut sops: Vec<Sop> = Vec::new();
    for (li, groups) in chains {
      for (_, members) in groups {
        let mut sop: Sop = Vec::new();
        for &i in members {
          let c = &cases[*li][i];
          let mut lits: Vec<u32> = Vec::new();
          for (cell, set) in c.sets.iter().enumerate().skip(1) {
            let set = and_sets(set, &self.valid);
            if set == self.valid {
              continue;
            }
            // a set and its complement are one predicate
            let other = minus(&self.valid, &set);
            let (key, neg) = if set.0 <= other.0 { (set, 0) } else { (other, 1) };
            let var = match pred.get(&(cell, key)) {
              Some(&v) => v,
              None => {
                let l = self.member(cell, key, self.valid, 0);
                leaves.push(l);
                pred.insert((cell, key), leaves.len() as u32 - 1);
                leaves.len() as u32 - 1
              }
            };
            lits.push(var << 1 | neg);
          }
          for g in &c.eq {
            for &j in &g[1..] {
              let key = (g[0].min(j), g[0].max(j));
              let var = match cmp.get(&key) {
                Some(&v) => v,
                None => {
                  let mut e = TRUE;
                  for b in 0..bits {
                    let x = self.net.xor(self.cell_bits[key.0][b], self.cell_bits[key.1][b]);
                    e = self.net.and(e, not(x));
                  }
                  leaves.push(e);
                  cmp.insert(key, leaves.len() as u32 - 1);
                  leaves.len() as u32 - 1
                }
              };
              lits.push(var << 1);
            }
          }
          let centre = and_sets(&c.sets[0], &self.valid);
          let cubes = centre_cubes.entry(centre).or_insert_with(|| {
            cubes_of(&centre, n, bits).iter().map(|q| (0..bits as u32).filter(|&j| q.mask >> j & 1 == 1).map(|j| j << 1 | (q.value >> j & 1 ^ 1) as u32).collect()).collect()
          });
          for q in cubes.iter() {
            sop.push(q.iter().chain(&lits).copied().collect());
          }
        }
        sops.push(sop);
      }
    }
    let vars = leaves.len() as u32;
    let nodes = fast_extract(&mut sops, vars);
    // gates: a node is the OR of its cubes, a cube the AND of its literals
    let mut made: Vec<Option<Lit>> = vec![None; nodes.len()];
    fn lit(net: &mut Net, leaves: &[Lit], nodes: &[Sop], made: &mut Vec<Option<Lit>>, l: u32) -> Lit {
      let var = (l >> 1) as usize;
      let v = if var < leaves.len() {
        leaves[var]
      } else {
        let k = var - leaves.len();
        match made[k] {
          Some(v) => v,
          None => {
            let v = sum(net, leaves, nodes, made, &nodes[k]);
            made[k] = Some(v);
            v
          }
        }
      };
      v ^ (l & 1)
    }
    fn sum(net: &mut Net, leaves: &[Lit], nodes: &[Sop], made: &mut Vec<Option<Lit>>, sop: &Sop) -> Lit {
      let mut s = FALSE;
      for cube in sop {
        let mut m = TRUE;
        for &l in cube {
          let x = lit(net, leaves, nodes, made, l);
          m = net.and(m, x);
        }
        s = net.or(s, m);
      }
      s
    }
    let mut sops = sops.iter();
    chains.iter().map(|c| c.1.iter().map(|_| sum(&mut self.net, &leaves, &nodes, &mut made, sops.next().unwrap())).collect()).collect()
  }
}

pub struct Built {
  pub net: Net,
  /// groups of the chains, or transitions that change the state
  pub terms: usize,
}

/// Build the network. `cases` holds one list of transitions for the whole state, or, for
/// `Style::Bits`, one list per bit of the state. `None` if the style would make a network too
/// large to be worth optimising.
pub fn build(table: &Table, cases: &[Vec<Case>], style: Style) -> Option<Built> {
  if style == Style::Given {
    return None;
  }
  let space = Space { n: table.states, k: table.cells(), bits: table.bits() };
  let bits = space.bits;
  let mut valid = StateSet::default();
  for v in 0..space.n {
    valid.insert(v as u8);
  }

  // the sets the predicates are for
  let mut distinct: Vec<StateSet> = Vec::new();
  for c in cases.iter().flatten() {
    for s in c.sets.iter() {
      let s = and_sets(s, &valid);
      if s != valid && !distinct.contains(&s) {
        distinct.push(s);
      }
    }
  }
  let mut b = Builder::new(space, valid, bit_order(bits, &valid, &distinct));
  let centre = b.cell_bits[0].clone();
  let mut out: Vec<Lit> = centre.clone();
  let mut terms = 0;
  if style == Style::Flat {
    let cases = &cases[0];
    let matches = b.matches(cases);
    let effects: Vec<Effect> = cases.iter().map(effect).collect();
    let mut flip = vec![FALSE; bits];
    let mut total = 0usize;
    for i in 0..cases.len() {
      if effects[i] == Effect::Keep || matches[i] == FALSE {
        continue;
      }
      let mut earlier = FALSE;
      for j in 0..i {
        if matches[j] != FALSE && overlap(&cases[i], &cases[j]) && differ(&cases[i], effects[i], &cases[j], effects[j]) {
          earlier = b.net.or(earlier, matches[j]);
          total += 1;
        }
      }
      if total > 400_000 {
        return None;
      }
      let fire = b.net.and(matches[i], not(earlier));
      terms += 1;
      for bit in 0..bits {
        // where the new bit differs from the centre's
        let d = match effects[i] {
          Effect::Const(v) => centre[bit] ^ (v as u32 >> bit & 1),
          Effect::Copy(c) => b.net.xor(b.cell_bits[c][bit], centre[bit]),
          Effect::Keep => FALSE,
        };
        let t = b.net.and(fire, d);
        flip[bit] = b.net.or(flip[bit], t);
      }
    }
    for bit in 0..bits {
      out[bit] = b.net.xor(centre[bit], flip[bit]);
    }
  } else {
    // one chain for the state, or one per bit; a transition joins the first group of its result
    // that comes after every group holding an earlier transition it must yield to
    let one = style != Style::Chain;
    let possible = |c: &Case| c.sets.iter().all(|s| !and_sets(s, &valid).is_empty());
    let mut chains: Vec<(usize, Vec<(Effect, Vec<usize>)>)> = Vec::new();
    for (li, cases) in cases.iter().enumerate() {
      let effects: Vec<Effect> = cases.iter().map(|c| if one { bit_effect(c, li) } else { effect(c) }).collect();
      let mut groups: Vec<(Effect, Vec<usize>)> = Vec::new();
      let mut group_of: Vec<usize> = vec![usize::MAX; cases.len()];
      for i in 0..cases.len() {
        if !possible(&cases[i]) {
          continue;
        }
        let mut first = 0;
        for j in 0..i {
          if group_of[j] != usize::MAX && group_of[j] >= first && effects[i] != effects[j] && overlap(&cases[i], &cases[j]) && (one || differ(&cases[i], effects[i], &cases[j], effects[j])) {
            first = group_of[j] + 1;
          }
        }
        let g = match (first..groups.len()).find(|&g| groups[g].0 == effects[i]) {
          Some(g) => g,
          None => {
            groups.push((effects[i], Vec::new()));
            groups.len() - 1
          }
        };
        groups[g].1.push(i);
        group_of[i] = g;
      }
      terms += groups.len();
      chains.push((li, groups));
    }
    // where each group applies
    let fired: Vec<Vec<Lit>> = if style == Style::Factor {
      b.factored(cases, &chains)
    } else {
      // all the transitions of all the lists at once: the lists share products
      let all: Vec<Case> = cases.iter().flatten().cloned().collect();
      let all_matches = b.matches(&all);
      let offsets: Vec<usize> = cases.iter().scan(0, |o, c| { let at = *o; *o += c.len(); Some(at) }).collect();
      let sums: Vec<Vec<Lit>> = chains.iter().flat_map(|(li, groups)| groups.iter().map(|g| g.1.iter().map(|&i| all_matches[offsets[*li] + i]).collect::<Vec<Lit>>()).collect::<Vec<_>>()).collect();
      let mut sums = extract(&mut b.net, sums, false).into_iter();
      chains.iter().map(|c| c.1.iter().map(|_| sums.next().unwrap()).collect()).collect()
    };
    for ((li, groups), fired) in chains.iter().zip(&fired) {
      for ((e, _), &m) in groups.iter().zip(fired).rev() {
        let range = if one { *li..*li + 1 } else { 0..bits };
        for bit in range {
          let t = match *e {
            Effect::Const(v) if one => v as u32,
            Effect::Const(v) => v as u32 >> bit & 1,
            Effect::Copy(c) => b.cell_bits[c][bit],
            Effect::Keep => centre[bit],
          };
          out[bit] = b.net.mux(m, t, out[bit]);
        }
      }
    }
  }
  b.net.outputs = out;
  Some(Built { net: b.net.compact(), terms })
}

/// Groups of a list of transitions: a transition joins the first group of its effect that comes
/// after every group holding an earlier transition it must yield to.
fn grouped(cases: &[Case], effects: &[Effect], valid: &StateSet, per_bit: bool) -> Vec<(Effect, Vec<usize>)> {
  let possible = |c: &Case| c.sets.iter().all(|s| !and_sets(s, valid).is_empty());
  let mut groups: Vec<(Effect, Vec<usize>)> = Vec::new();
  let mut group_of: Vec<usize> = vec![usize::MAX; cases.len()];
  for i in 0..cases.len() {
    if !possible(&cases[i]) {
      continue;
    }
    let mut first = 0;
    for j in 0..i {
      if group_of[j] != usize::MAX && group_of[j] >= first && effects[i] != effects[j] && overlap(&cases[i], &cases[j]) && (per_bit || differ(&cases[i], effects[i], &cases[j], effects[j])) {
        first = group_of[j] + 1;
      }
    }
    let g = match (first..groups.len()).find(|&g| groups[g].0 == effects[i]) {
      Some(g) => g,
      None => {
        groups.push((effects[i], Vec::new()));
        groups.len() - 1
      }
    };
    groups[g].1.push(i);
    group_of[i] = g;
  }
  groups
}

/// The variables of the sums that are factored together, and the gates they stand for.
struct Leaves {
  lits: Vec<Lit>,
  pred: FxMap<(usize, StateSet), u32>,
  cmp: FxMap<(usize, usize), u32>,
}

impl Leaves {
  /// The literal "cell `cell` holds a state of `set`"; `None` if every state is in it. A set and
  /// its complement are one variable.
  fn member(&mut self, b: &mut Builder, cell: usize, set: &StateSet) -> Option<u32> {
    let set = and_sets(set, &b.valid);
    if set == b.valid {
      return None;
    }
    let other = minus(&b.valid, &set);
    let (key, neg) = if set.0 <= other.0 { (set, 0) } else { (other, 1) };
    let var = match self.pred.get(&(cell, key)) {
      Some(&v) => v,
      None => {
        let l = b.member(cell, key, b.valid, 0);
        self.lits.push(l);
        self.pred.insert((cell, key), self.lits.len() as u32 - 1);
        self.lits.len() as u32 - 1
      }
    };
    Some(var << 1 | neg)
  }
  /// The literal "cells `x` and `y` hold the same state".
  fn equal(&mut self, b: &mut Builder, x: usize, y: usize) -> u32 {
    let key = (x.min(y), x.max(y));
    let var = match self.cmp.get(&key) {
      Some(&v) => v,
      None => {
        let mut e = TRUE;
        for bit in 0..b.space.bits {
          let d = b.net.xor(b.cell_bits[key.0][bit], b.cell_bits[key.1][bit]);
          e = b.net.and(e, not(d));
        }
        self.lits.push(e);
        self.cmp.insert(key, self.lits.len() as u32 - 1);
        self.lits.len() as u32 - 1
      }
    };
    var << 1
  }
  fn leaf(&mut self, l: Lit) -> u32 {
    self.lits.push(l);
    (self.lits.len() as u32 - 1) << 1
  }
}

/// Cubes over the centre's bits (variables `0..bits`) for a set of centre states.
fn centre_cubes(set: &StateSet, n: usize, bits: usize) -> Vec<Vec<u32>> {
  cubes_of(set, n, bits).iter().map(|q| (0..bits as u32).filter(|&j| q.mask >> j & 1 == 1).map(|j| j << 1 | (q.value >> j & 1 ^ 1) as u32).collect()).collect()
}

/// Gates for factored sums: a node is the OR of its cubes, a cube the AND of its literals.
fn realise(net: &mut Net, leaves: &[Lit], nodes: &[Sop], sops: &[Sop]) -> Vec<Lit> {
  fn lit(net: &mut Net, leaves: &[Lit], nodes: &[Sop], made: &mut Vec<Option<Lit>>, l: u32) -> Lit {
    let var = (l >> 1) as usize;
    let v = if var < leaves.len() {
      leaves[var]
    } else {
      let k = var - leaves.len();
      match made[k] {
        Some(v) => v,
        None => {
          let v = sum(net, leaves, nodes, made, &nodes[k]);
          made[k] = Some(v);
          v
        }
      }
    };
    v ^ (l & 1)
  }
  fn sum(net: &mut Net, leaves: &[Lit], nodes: &[Sop], made: &mut Vec<Option<Lit>>, sop: &Sop) -> Lit {
    let mut s = FALSE;
    for cube in sop {
      let mut m = TRUE;
      for &l in cube {
        let x = lit(net, leaves, nodes, made, l);
        m = net.and(m, x);
      }
      s = net.or(s, m);
    }
    s
  }
  let mut made: Vec<Option<Lit>> = vec![None; nodes.len()];
  sops.iter().map(|sop| sum(net, leaves, nodes, &mut made, sop)).collect()
}

/// Build with the rule's rotational symmetry. `canon` is the table of the centre states of
/// orientation 0 (its centre sets hold no other state); a centre of orientation k is served by
/// the same transitions with the cells and the states turned k times. `rest` holds, per bit of
/// the state, the transitions of the centre states that have no orientation. A cell that many
/// transitions test for sets the turn does not change (the cell an arrow points at, tested for
/// "ground" or "any special arrow") is selected once by the centre's orientation, bit by bit,
/// if at least `hoist` such sets are tested there.
pub fn build_folded(table: &Table, sym: &Symmetry, canon: &[Case], rest: &[Vec<Case>], hoist: usize) -> Option<Built> {
  let space = Space { n: table.states, k: table.cells(), bits: table.bits() };
  let (n, k, bits, order) = (space.n, space.k, space.bits, sym.order);
  let mut valid = StateSet::default();
  for v in 0..n {
    valid.insert(v as u8);
  }
  let oriented = sym.all_oriented();
  let invariant = |s: &StateSet| sym.turn_set(s, 1) == *s;

  // the sets the predicates are for, to choose the order of the bits
  let mut distinct: Vec<StateSet> = Vec::new();
  let mut note = |s: StateSet| {
    if s != valid && !distinct.contains(&s) {
      distinct.push(s);
    }
  };
  for c in rest.iter().flatten() {
    for s in &c.sets[1..] {
      note(and_sets(s, &valid));
    }
  }
  for c in canon {
    for s in &c.sets[1..] {
      for t in 0..order {
        note(and_sets(&sym.turn_set(s, t), &valid));
      }
    }
  }
  let mut b = Builder::new(space, valid, bit_order(bits, &valid, &distinct));
  let centre = b.cell_bits[0].clone();
  let mut leaves = Leaves { lits: centre.clone(), pred: FxMap::default(), cmp: FxMap::default() };

  // the centre's orientation: free on the states that have none
  let turned: Vec<u32> = (0..order).map(|t| b.member(0, sym.oriented[t], oriented, 0)).map(|l| leaves.leaf(l)).collect();
  // cells selected by the orientation
  let mut selected: Vec<Option<usize>> = vec![None; k];
  let mut uses = vec![Vec::<StateSet>::new(); k];
  for c in canon {
    for i in 1..k {
      let s = and_sets(&c.sets[i], &valid);
      if s != valid && invariant(&s) && !uses[i].contains(&s) {
        uses[i].push(s);
      }
    }
  }
  let select = |b: &mut Builder, turned: &[u32], lits: &[Lit], i: usize| -> usize {
    let cell = b.cell_bits.len();
    let mux: Vec<Lit> = (0..bits)
      .map(|bit| {
        let mut m = FALSE;
        for t in 0..order {
          let x = b.net.and(lits[(turned[t] >> 1) as usize], b.cell_bits[sym.turn_cell(i, t)][bit]);
          m = b.net.or(m, x);
        }
        m
      })
      .collect();
    b.cell_bits.push(mux);
    b.memo.push(FxMap::default());
    cell
  };
  for i in 1..k {
    if uses[i].len() >= hoist {
      selected[i] = Some(select(&mut b, &turned, &leaves.lits, i));
    }
  }

  let mut sops: Vec<Sop> = Vec::new();
  // the centre states without an orientation: a chain per bit
  let mut rest_chains: Vec<Vec<Effect>> = Vec::new();
  // (a single list for a state of several bits is one chain for the whole state instead)
  let whole = rest.len() == 1 && bits > 1;
  for (bit, cases) in rest.iter().enumerate() {
    let effects: Vec<Effect> = cases.iter().map(|c| if whole { effect(c) } else { bit_effect(c, bit) }).collect();
    let groups = grouped(cases, &effects, &valid, !whole);
    let mut chain = Vec::new();
    for (e, members) in groups {
      let mut sop: Sop = Vec::new();
      for i in members {
        let c = &cases[i];
        let mut lits: Vec<u32> = Vec::new();
        for cell in 1..k {
          lits.extend(leaves.member(&mut b, cell, &c.sets[cell]));
        }
        for g in &c.eq {
          for &j in &g[1..] {
            lits.push(leaves.equal(&mut b, g[0], j));
          }
        }
        let mut cubes = centre_cubes(&and_sets(&c.sets[0], &valid), n, bits);
        if cubes.len() > 2 {
          cubes = vec![leaves.member(&mut b, 0, &c.sets[0]).into_iter().collect()];
        }
        for q in cubes {
          sop.push(q.iter().chain(&lits).copied().collect());
        }
      }
      sops.push(sop);
      chain.push(e);
    }
    rest_chains.push(chain);
  }
  // the centre states with an orientation: one chain for the state
  let effects: Vec<Effect> = canon.iter().map(effect).collect();
  let groups = grouped(canon, &effects, &valid, false);
  // A set tested in a neighbour, as the first cell of that neighbour's cycle under the turn
  // would see it: the same for a transition's turned images. `frame[cell]` is that first cell
  // and the turns from it; `cycle[cell]` the cells of the cycle, if the turn moves it through
  // `order` different ones.
  let frame: Vec<(usize, usize)> = (0..k).map(|cell| (0..order).map(|t| ((0..k).find(|&c| sym.turn_cell(c, t) == cell).unwrap(), t)).min().unwrap()).collect();
  let cycle = |cell: usize| -> Option<Vec<usize>> {
    let c: Vec<usize> = (0..order).map(|t| sym.turn_cell(frame[cell].0, t)).collect();
    (cell != 0 && (0..order).all(|x| (0..x).all(|y| c[x] != c[y]))).then_some(c)
  };
  /// A test of one neighbour that moves with the centre's orientation: the cell in the
  /// transition as written, its cycle's first cell, the set as seen there (the smaller of the
  /// set and its complement) and whether the test is for the complement.
  #[derive(Clone, PartialEq)]
  struct Moving {
    cell: usize,
    base: usize,
    key: StateSet,
    neg: bool,
    set: StateSet,
  }
  #[derive(Clone, PartialEq)]
  struct Proto {
    cubes: Vec<Vec<u32>>,
    fixed: Vec<u32>,
    moving: Vec<Moving>,
    eq: Vec<Vec<usize>>,
  }
  // "one of the cells of a cycle holds a state of the set as that cell sees it": of all its
  // cells, or of all but the one that is at a given place for the centre's orientation
  let mut aggregates: FxMap<(usize, StateSet, usize), u32> = FxMap::default();
  let mut aggregate = |b: &mut Builder, leaves: &mut Leaves, base: usize, key: &StateSet, but: Option<usize>| -> u32 {
    if let Some(&l) = aggregates.get(&(base, *key, but.unwrap_or(0))) {
      return l;
    }
    let mut any = FALSE;
    for t in 0..order {
      let cell = sym.turn_cell(base, t);
      let mut hit = b.member(cell, sym.turn_set(key, t), valid, 0);
      if let Some(but) = but {
        // this cell is at place `but` for the orientation u with turn_cell(but, u) == cell
        let u = (0..order).find(|&u| sym.turn_cell(but, u) == cell).unwrap();
        hit = b.net.and(hit, not(leaves.lits[(turned[u] >> 1) as usize]));
      }
      any = b.net.or(any, hit);
    }
    let l = leaves.leaf(any);
    aggregates.insert((base, *key, but.unwrap_or(0)), l);
    l
  };
  // the places of a cycle that `at` leaves out, if it leaves out one at most
  let missing = |cyc: &[usize], at: &[usize]| -> Option<Option<usize>> {
    let out: Vec<usize> = cyc.iter().copied().filter(|c| !at.contains(c)).collect();
    match out.len() {
      0 => Some(None),
      1 => Some(Some(out[0])),
      _ => None,
    }
  };
  let mut canon_chain = Vec::new();
  for (e, members) in &groups {
    let mut protos: Vec<Proto> = Vec::new();
    for &i in members {
      let c = &canon[i];
      // every centre state this transition serves, whatever its orientation
      let mut all = StateSet::default();
      for t in 0..order {
        all = union(&all, &sym.turn_set(&and_sets(&c.sets[0], &sym.oriented[0]), t));
      }
      let mut cubes = centre_cubes(&all, n, bits);
      if cubes.len() > 2 {
        cubes = vec![leaves.member(&mut b, 0, &all).into_iter().collect()];
      }
      let mut fixed: Vec<u32> = Vec::new();
      let mut moving: Vec<Moving> = Vec::new();
      for i in 1..k {
        let set = and_sets(&c.sets[i], &valid);
        if set == valid {
          continue;
        }
        match selected[i] {
          Some(cell) if invariant(&set) => fixed.extend(leaves.member(&mut b, cell, &set)),
          _ => {
            let (base, t) = frame[i];
            let seen = sym.turn_set(&set, (order - t) % order);
            let other = minus(&valid, &seen);
            let neg = other.0 < seen.0;
            moving.push(Moving { cell: i, base, key: if neg { other } else { seen }, neg, set });
          }
        }
      }
      // no cell of a cycle, or none but one, holds a state of a set: one test
      let mut at = 0;
      while at < moving.len() {
        let m = moving[at].clone();
        let same: Vec<usize> = moving.iter().filter(|x| x.neg && x.base == m.base && x.key == m.key).map(|x| x.cell).collect();
        let but = cycle(m.cell).filter(|_| m.neg && !c.eq.iter().any(|g| same.iter().any(|x| g.contains(x)))).and_then(|cyc| missing(&cyc, &same));
        match but {
          Some(but) => {
            fixed.push(aggregate(&mut b, &mut leaves, m.base, &m.key, but) ^ 1);
            moving.retain(|x| !(x.neg && x.base == m.base && x.key == m.key));
          }
          None => at += 1,
        }
      }
      fixed.sort_unstable();
      protos.push(Proto { cubes, fixed, moving, eq: c.eq.clone() });
    }
    // transitions that differ in one test only, the same test at every cell of a cycle or at
    // every cell but one: one transition with one test
    let mut used = vec![false; protos.len()];
    let mut list: Vec<Proto> = Vec::new();
    for x in 0..protos.len() {
      if used[x] {
        continue;
      }
      used[x] = true;
      let mut p = protos[x].clone();
      let mut at = 0;
      while at < p.moving.len() {
        let m = p.moving[at].clone();
        let without = |q: &Proto, cell: usize| -> Proto {
          let mut q = q.clone();
          q.moving.retain(|y| y.cell != cell);
          q
        };
        let rest = without(&p, m.cell);
        let mut found: Vec<(usize, usize)> = vec![(x, m.cell)];
        if !m.neg && !p.eq.iter().any(|g| g.contains(&m.cell)) {
          for y in x + 1..protos.len() {
            if used[y] {
              continue;
            }
            let twin = protos[y].moving.iter().find(|z| !z.neg && z.base == m.base && z.key == m.key && !found.iter().any(|f| f.1 == z.cell) && without(&protos[y], z.cell) == rest);
            if let Some(z) = twin {
              found.push((y, z.cell));
            }
          }
        }
        let cells: Vec<usize> = found.iter().map(|f| f.1).collect();
        match cycle(m.cell).filter(|_| found.len() > 1).and_then(|cyc| missing(&cyc, &cells)) {
          Some(but) => {
            for f in &found {
              used[f.0] = true;
            }
            p = rest;
            p.fixed.push(aggregate(&mut b, &mut leaves, m.base, &m.key, but));
            p.fixed.sort_unstable();
          }
          None => at += 1,
        }
      }
      list.push(p);
    }
    let mut sop: Sop = Vec::new();
    for p in &list {
      if p.moving.is_empty() && p.eq.is_empty() {
        for q in &p.cubes {
          sop.push(q.iter().chain(&p.fixed).copied().collect());
        }
        continue;
      }
      for t in 0..order {
        let mut lits = p.fixed.clone();
        lits.push(turned[t]);
        for m in &p.moving {
          lits.extend(leaves.member(&mut b, sym.turn_cell(m.cell, t), &sym.turn_set(&m.set, t)));
        }
        for g in &p.eq {
          for &j in &g[1..] {
            lits.push(leaves.equal(&mut b, sym.turn_cell(g[0], t), sym.turn_cell(j, t)));
          }
        }
        for q in &p.cubes {
          sop.push(q.iter().chain(&lits).copied().collect());
        }
      }
    }
    sops.push(sop);
    canon_chain.push(*e);
  }

  if std::env::var("R2N_DEBUG").is_ok() {
    // where the cubes are: per bit, by the kind of centre
    for (bit, cases) in rest.iter().enumerate() {
      let mut by: Vec<(String, usize)> = Vec::new();
      for c in cases {
        let key = c.sets[0].iter().map(|v| v.to_string()).collect::<Vec<_>>().join(" ");
        match by.iter_mut().find(|x| x.0 == key) {
          Some(x) => x.1 += 1,
          None => by.push((key, 1)),
        }
      }
      eprintln!("  rest bit {bit}: {}", by.iter().map(|x| format!("[{}]x{}", x.0, x.1)).collect::<Vec<_>>().join(" "));
    }
    let before: usize = sops.iter().map(|s| s.iter().map(|c| c.len()).sum::<usize>()).sum();
    eprintln!("  {} sums, {} cubes, {} literals; {} leaves; nodes before sums {}", sops.len(), sops.iter().map(|s| s.len()).sum::<usize>(), before, leaves.lits.len(), b.net.len());
  }
  let nodes = fast_extract(&mut sops, leaves.lits.len() as u32);
  let fired = realise(&mut b.net, &leaves.lits, &nodes, &sops);
  if std::env::var("R2N_DEBUG").is_ok() {
    eprintln!("  after extraction: {} new nodes; network {} nodes", nodes.len(), b.net.len());
  }
  let mut fired = fired.into_iter();
  let mut out: Vec<Lit> = centre.clone();
  let mut terms = 0;
  for (bit, chain) in rest_chains.iter().enumerate() {
    let conds: Vec<Lit> = chain.iter().map(|_| fired.next().unwrap()).collect();
    terms += chain.len();
    for (e, &m) in chain.iter().zip(&conds).rev() {
      for bit in if whole { 0..bits } else { bit..bit + 1 } {
        let t = match *e {
          Effect::Const(v) if whole => v as u32 >> bit & 1,
          Effect::Const(v) => v as u32,
          Effect::Copy(c) => b.cell_bits[c][bit],
          Effect::Keep => centre[bit],
        };
        out[bit] = b.net.mux(m, t, out[bit]);
      }
    }
  }
  let conds: Vec<Lit> = canon_chain.iter().map(|_| fired.next().unwrap()).collect();
  terms += canon_chain.len();
  // the bits of each result
  let mut results: Vec<(Effect, Vec<Lit>)> = Vec::new();
  for e in &canon_chain {
    if results.iter().any(|r| r.0 == *e) {
      continue;
    }
    let mut t: Vec<Lit> = Vec::with_capacity(bits);
    for bit in 0..bits {
      t.push(match *e {
        // the state, turned as the centre is: free on the centre states without orientation
        Effect::Const(v) => {
          let mut on = StateSet::default();
          for c in oriented.iter() {
            if sym.turn(v, sym.orientation(c).unwrap()) >> bit & 1 == 1 {
              on.insert(c);
            }
          }
          b.member(0, on, oriented, 0)
        }
        Effect::Copy(i) => {
          let cell = match selected[i] {
            Some(cell) => cell,
            None => {
              let cell = select(&mut b, &turned, &leaves.lits, i);
              selected[i] = Some(cell);
              cell
            }
          };
          b.cell_bits[cell][bit]
        }
        Effect::Keep => centre[bit],
      });
    }
    results.push((*e, t));
  }
  {
    // Which result first: one signal per result, the OR of its groups where no group before
    // them holds; then a bit is the OR, over the values it can take, of the results that give
    // it that value. Many results share a value (a bit of the centre, a constant), so this is
    // far fewer gates than a selection per group and bit.
    let mut before = FALSE;
    let mut which: Vec<Lit> = vec![FALSE; results.len()];
    for (e, &m) in canon_chain.iter().zip(&conds) {
      let r = results.iter().position(|r| r.0 == *e).unwrap();
      let first = b.net.and(m, not(before));
      which[r] = b.net.or(which[r], first);
      before = b.net.or(before, m);
    }
    for bit in 0..bits {
      let mut by: Vec<(Lit, Lit)> = Vec::new();
      for (r, &w) in results.iter().zip(&which) {
        match by.iter_mut().find(|x| x.0 == r.1[bit]) {
          Some(x) => x.1 = b.net.or(x.1, w),
          None => by.push((r.1[bit], w)),
        }
      }
      let mut o = b.net.and(not(before), out[bit]);
      for (t, w) in by {
        let x = b.net.and(w, t);
        o = b.net.or(o, x);
      }
      out[bit] = o;
    }
  }
  // Signals for the optimiser to try: a set tested in one direction and its turned images in
  // the others, taken together ("from any side"), and the same without the cell the centre's
  // orientation selects ("from a side I do not point at").
  let mut families: FxMap<(usize, StateSet), Vec<(usize, Lit)>> = FxMap::default();
  let preds: Vec<((usize, StateSet), u32)> = leaves.pred.iter().map(|(key, &var)| (*key, var)).collect();
  for ((cell, set), _) in preds {
    if cell == 0 || cell >= k {
      continue;
    }
    let (base, t) = frame[cell];
    let key = (base, sym.turn_set(&set, order - t));
    if families.contains_key(&key) {
      continue;
    }
    let members: Vec<(usize, Lit)> = (0..order).map(|t| (sym.turn_cell(base, t), b.member(sym.turn_cell(base, t), sym.turn_set(&key.1, t), valid, 0))).collect();
    families.insert(key, members);
  }
  let mut bank: Vec<Lit> = Vec::new();
  let mut keys: Vec<(usize, StateSet)> = families.keys().copied().collect();
  keys.sort_by(|a, b| a.0.cmp(&b.0).then(a.1 .0.cmp(&b.1 .0)));
  for key in keys {
    let members = &families[&key];
    let mut any = FALSE;
    for &(_, l) in members {
      any = b.net.or(any, l);
    }
    bank.push(any);
    for i in (1..k).filter(|&i| selected[i].is_some()) {
      let mut others = FALSE;
      for &(cell, l) in members {
        // this cell is the selected one for the orientation t with turn_cell(i, t) == cell
        let mut hit = l;
        for t in 0..order {
          if sym.turn_cell(i, t) == cell {
            hit = b.net.and(hit, not(leaves.lits[(turned[t] >> 1) as usize]));
          }
        }
        others = b.net.or(others, hit);
      }
      bank.push(others);
    }
  }
  b.net.outputs = out;
  b.net.keep = bank;
  Some(Built { net: b.net.compact(), terms })
}
