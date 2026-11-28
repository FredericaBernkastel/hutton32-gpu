//! Making the transitions simpler before a network is built from them, without changing the
//! function they define.
//!
//! A table is written for people: a transition lists what its author had in mind, not the least
//! that tells it apart from the others. Two steps, each checked on every neighbourhood it touches:
//!
//! * **remove**: a transition goes if, wherever it is the first to match, the transitions after
//!   it (or the default, no change) give the same state;
//! * **widen**: a cell's set of states grows, to every state or to a larger set the table already
//!   uses for that cell, if on the neighbourhoods this adds the transition either is not the
//!   first to match or gives the state the table gave before. A set of every state needs no gate.

use crate::check::Space;
use crate::net::FxMap;
use crate::symmetry::Symmetry;
use ca_rule::table::{Case, Out, StateSet};
use std::time::Instant;

/// Neighbourhoods one step may enumerate.
const BUDGET: u64 = 1 << 23;

#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
  pub removed: usize,
  pub widened: usize,
  /// cell sets that became every state
  pub freed: usize,
}

/// The table with its transitions indexed by (cell, state), as `Interp`, but changeable.
struct Tab {
  cases: Vec<Case>,
  alive: Vec<bool>,
  n: usize,
  k: usize,
  words: usize,
  rows: Vec<u64>,
}

impl Tab {
  fn new(cases: Vec<Case>, n: usize, k: usize) -> Tab {
    let words = cases.len().div_ceil(64).max(1);
    let mut t = Tab { alive: vec![true; cases.len()], cases, n, k, words, rows: vec![0u64; k * n * words] };
    for i in 0..t.cases.len() {
      for c in 0..k {
        let s = t.cases[i].sets[c];
        t.mark(i, c, &s, true);
      }
    }
    t
  }

  fn mark(&mut self, i: usize, cell: usize, set: &StateSet, on: bool) {
    for v in set.iter().filter(|&v| (v as usize) < self.n) {
      let w = &mut self.rows[(cell * self.n + v as usize) * self.words + i / 64];
      if on {
        *w |= 1 << (i % 64);
      } else {
        *w &= !(1 << (i % 64));
      }
    }
  }

  /// The first transition from `from` on that matches.
  fn first(&self, nb: &[u8], from: usize, scratch: &mut [u64]) -> Option<usize> {
    let w = self.words;
    scratch.copy_from_slice(&self.rows[nb[0] as usize * w..][..w]);
    for c in 1..self.k {
      let row = &self.rows[(c * self.n + nb[c] as usize) * w..][..w];
      for (s, r) in scratch.iter_mut().zip(row) {
        *s &= r;
      }
    }
    for wi in from / 64..w {
      let mut m = scratch[wi];
      if wi == from / 64 {
        m &= !0u64 << (from % 64);
      }
      while m != 0 {
        let i = wi * 64 + m.trailing_zeros() as usize;
        if self.cases[i].eq.iter().all(|g| g[1..].iter().all(|&j| nb[j] == nb[g[0]])) {
          return Some(i);
        }
        m &= m - 1;
      }
    }
    None
  }

  fn out(&self, i: Option<usize>, nb: &[u8]) -> u8 {
    match i.map(|i| self.cases[i].out) {
      None => nb[0],
      Some(Out::Const(v)) => v,
      Some(Out::Copy(c)) => nb[c],
    }
  }
}

/// Visit the neighbourhoods of a product of sets until `f` returns false; whether it never did.
fn each(sets: &[Vec<u8>], mut f: impl FnMut(&[u8]) -> bool) -> bool {
  let k = sets.len();
  if sets.iter().any(|s| s.is_empty()) {
    return true;
  }
  let mut idx = vec![0usize; k];
  let mut nb: Vec<u8> = sets.iter().map(|s| s[0]).collect();
  loop {
    if !f(&nb) {
      return false;
    }
    let mut c = k;
    loop {
      if c == 0 {
        return true;
      }
      c -= 1;
      idx[c] += 1;
      if idx[c] < sets[c].len() {
        nb[c] = sets[c][idx[c]];
        break;
      }
      idx[c] = 0;
      nb[c] = sets[c][0];
    }
  }
}

fn product(sets: &[Vec<u8>]) -> u64 {
  sets.iter().fold(1u64, |p, s| p.saturating_mul(s.len() as u64))
}

/// Simplify the transitions; they define the same function afterwards. With `bit`, only that
/// bit of the next state is kept the same: the list then defines one bit of the function. With
/// `centres`, the list is the table for those centre states only: the others are dropped from
/// every transition, and a centre's set is widened to all of them at most.
pub fn simplify(space: Space, cases: Vec<Case>, bit: Option<usize>, centres: Option<StateSet>, deadline: Instant) -> (Vec<Case>, Stats) {
  simplify_with(space, cases, bit, centres, None, deadline)
}

/// The same. With a symmetry, a neighbour's set that cannot grow to every state is tried grown to
/// the set with all its turned images, which the turn leaves as it is: a transition that serves
/// every orientation of the centre then tests the same set whichever way the centre points.
pub fn simplify_with(space: Space, cases: Vec<Case>, bit: Option<usize>, centres: Option<StateSet>, sym: Option<&Symmetry>, deadline: Instant) -> (Vec<Case>, Stats) {
  let mask: u8 = bit.map_or(!0, |b| 1 << b);
  let (n, k) = (space.n, space.k);
  let mut valid = StateSet::default();
  for v in 0..n {
    valid.insert(v as u8);
  }
  let clip = |s: &StateSet| StateSet([s.0[0] & valid.0[0], s.0[1] & valid.0[1], s.0[2] & valid.0[2], s.0[3] & valid.0[3]]);
  let centres = centres.map_or(valid, |d| clip(&d));
  let cases: Vec<Case> = cases
    .into_iter()
    .map(|mut c| {
      for s in c.sets.iter_mut() {
        *s = clip(s);
      }
      for i in 0..4 {
        c.sets[0].0[i] &= centres.0[i];
      }
      c
    })
    .filter(|c| !c.sets[0].is_empty())
    .collect();
  let mut t = Tab::new(cases, n, k);
  let mut scratch = vec![0u64; t.words];
  let mut stats = Stats::default();
  let lists = |c: &Case| -> Vec<Vec<u8>> { c.sets.iter().map(|s| s.iter().collect()).collect() };
  let superset = |a: &StateSet, b: &StateSet| (0..4).all(|i| a.0[i] & !b.0[i] == 0) && a != b;

  let remove = |t: &mut Tab, stats: &mut Stats, scratch: &mut [u64]| {
    for i in (0..t.cases.len()).rev() {
      if !t.alive[i] || Instant::now() >= deadline {
        continue;
      }
      let sets = lists(&t.cases[i]);
      if product(&sets) > BUDGET {
        continue;
      }
      let redundant = each(&sets, |nb| t.first(nb, 0, scratch) != Some(i) || (t.out(t.first(nb, i + 1, scratch), nb) ^ t.out(Some(i), nb)) & mask == 0);
      if redundant {
        t.alive[i] = false;
        for c in 0..k {
          let s = t.cases[i].sets[c];
          t.mark(i, c, &s, false);
        }
        stats.removed += 1;
      }
    }
  };

  remove(&mut t, &mut stats, &mut scratch);
  for i in 0..t.cases.len() {
    if !t.alive[i] {
      continue;
    }
    for c in 0..k {
      if Instant::now() >= deadline {
        break;
      }
      let cur = t.cases[i].sets[c];
      let valid = if c == 0 { centres } else { valid };
      if cur == valid || t.cases[i].eq.iter().any(|g| g.contains(&c)) {
        continue;
      }
      // every state first, then the larger sets this cell has elsewhere, the most used first
      let mut wider: Vec<(usize, StateSet)> = Vec::new();
      for (j, o) in t.cases.iter().enumerate() {
        if t.alive[j] && superset(&cur, &o.sets[c]) && o.sets[c] != valid {
          match wider.iter_mut().find(|w| w.1 == o.sets[c]) {
            Some(w) => w.0 += 1,
            None => wider.push((1, o.sets[c])),
          }
        }
      }
      wider.sort_by(|a, b| b.0.cmp(&a.0));
      wider.truncate(5);
      wider.insert(0, (0, valid));
      if let Some(sym) = sym.filter(|_| c > 0) {
        let mut all = cur;
        for t in 1..sym.order {
          let s = sym.turn_set(&cur, t);
          for i in 0..4 {
            all.0[i] |= s.0[i];
          }
        }
        let all = clip(&all);
        if all != cur && all != valid {
          wider.insert(1, (0, all));
        }
      }
      for (_, new) in wider {
        let mut sets = lists(&t.cases[i]);
        sets[c] = new.iter().filter(|&v| !cur.contains(v)).collect();
        if product(&sets) > BUDGET {
          continue;
        }
        let ok = each(&sets, |nb| {
          // bound cells that differ: the transition does not match here
          if !t.cases[i].eq.iter().all(|g| g[1..].iter().all(|&j| nb[j] == nb[g[0]])) {
            return true;
          }
          let f: Option<usize> = t.first(nb, 0, &mut scratch);
          f.is_some_and(|f: usize| f < i) || (t.out(f, nb) ^ t.out(Some(i), nb)) & mask == 0
        });
        if ok {
          let added = StateSet([new.0[0] & !cur.0[0], new.0[1] & !cur.0[1], new.0[2] & !cur.0[2], new.0[3] & !cur.0[3]]);
          t.mark(i, c, &added, true);
          t.cases[i].sets[c] = new;
          stats.widened += 1;
          stats.freed += (new == valid) as usize;
          break;
        }
      }
    }
  }
  remove(&mut t, &mut stats, &mut scratch);
  let out = t.cases.into_iter().zip(t.alive).filter(|x| x.1).map(|x| x.0).collect();
  (out, stats)
}

/// One vocabulary of sets for all the lists, and with a symmetry for all the directions.
///
/// Every set a transition tests in a neighbour costs a predicate. The lists are simplified one by
/// one and end up testing many sets that differ a little; here a set is given up if each of its
/// uses can take another set that is tested anyway (or no test at all) without changing what its
/// list defines, the rarest sets first. With a rotational symmetry a set at one cell and its
/// turned image at the turned cell count as one: the four directions then test the same things.
///
/// `what[l]` says what list `l` keeps: one bit or the state, for some centre states or for all
/// (as for [`simplify`]). Returns how many sets were given up.
pub fn unify(space: Space, lists: &mut [Vec<Case>], what: &[(Option<usize>, Option<StateSet>)], sym: Option<&Symmetry>, deadline: Instant) -> usize {
  let (n, k) = (space.n, space.k);
  let mut valid = StateSet::default();
  for v in 0..n {
    valid.insert(v as u8);
  }
  // a cell as the turned image of the lowest cell of its orbit: (that cell, turns)
  let frame: Vec<(usize, usize)> = (0..k)
    .map(|cell| match sym {
      Some(s) => (0..s.order).map(|t| ((0..k).find(|&b| s.turn_cell(b, t) == cell).unwrap(), t)).min().unwrap(),
      None => (cell, 0),
    })
    .collect();
  let order = sym.map_or(1, |s| s.order);
  let to_key = |cell: usize, set: &StateSet| -> (usize, StateSet) { (frame[cell].0, sym.map_or(*set, |s| s.turn_set(set, order - frame[cell].1))) };
  let from_key = |cell: usize, set: &StateSet| -> StateSet { sym.map_or(*set, |s| s.turn_set(set, frame[cell].1)) };
  let minus = |a: &StateSet, b: &StateSet| StateSet([a.0[0] & !b.0[0], a.0[1] & !b.0[1], a.0[2] & !b.0[2], a.0[3] & !b.0[3]]);

  let mut tabs: Vec<Tab> = lists.iter().map(|l| Tab::new(l.clone(), n, k)).collect();
  let masks: Vec<u8> = what.iter().map(|w| w.0.map_or(!0, |b| 1 << b)).collect();
  let lists_of = |c: &Case| -> Vec<Vec<u8>> { c.sets.iter().map(|s| s.iter().collect()).collect() };

  // Whether transition `i` of a table may test `new` instead of its set in `cell`: on the
  // neighbourhoods it gains it must not be first with another result, and on those it loses
  // what comes after it must give its result.
  let may = |t: &Tab, mask: u8, i: usize, cell: usize, new: &StateSet, scratch: &mut Vec<u64>| -> bool {
    scratch.resize(t.words, 0);
    let cur = t.cases[i].sets[cell];
    let (gained, lost) = (minus(new, &cur), minus(&cur, new));
    let bound = |nb: &[u8]| t.cases[i].eq.iter().all(|g| g[1..].iter().all(|&j| nb[j] == nb[g[0]]));
    if !gained.is_empty() {
      let mut sets = lists_of(&t.cases[i]);
      sets[cell] = gained.iter().collect();
      if product(&sets) > BUDGET {
        return false;
      }
      let ok = each(&sets, |nb| {
        if !bound(nb) {
          return true;
        }
        let f = t.first(nb, 0, scratch);
        f.is_some_and(|f: usize| f < i) || (t.out(f, nb) ^ t.out(Some(i), nb)) & mask == 0
      });
      if !ok {
        return false;
      }
    }
    if !lost.is_empty() {
      let mut sets = lists_of(&t.cases[i]);
      sets[cell] = lost.iter().collect();
      if product(&sets) > BUDGET {
        return false;
      }
      let ok = each(&sets, |nb| t.first(nb, 0, scratch) != Some(i) || (t.out(t.first(nb, i + 1, scratch), nb) ^ t.out(Some(i), nb)) & mask == 0);
      if !ok {
        return false;
      }
    }
    true
  };
  let set_to = |t: &mut Tab, i: usize, cell: usize, new: &StateSet| {
    let cur = t.cases[i].sets[cell];
    t.mark(i, cell, &minus(&cur, new), false);
    t.mark(i, cell, &minus(new, &cur), true);
    t.cases[i].sets[cell] = *new;
  };

  let mut scratch: Vec<u64> = Vec::new();
  let mut tried: Vec<(usize, StateSet)> = Vec::new();
  let mut given_up = 0;
  while Instant::now() < deadline {
    // who tests what
    let mut uses: FxMap<(usize, StateSet), Vec<(usize, usize, usize)>> = FxMap::default();
    for (l, t) in tabs.iter().enumerate() {
      for (i, c) in t.cases.iter().enumerate() {
        for cell in 1..k {
          if t.alive[i] && c.sets[cell] != valid && !c.eq.iter().any(|g| g.contains(&cell)) {
            uses.entry(to_key(cell, &c.sets[cell])).or_default().push((l, i, cell));
          }
        }
      }
    }
    let Some(key) = uses.iter().filter(|u| !tried.contains(u.0)).min_by_key(|u| (u.1.len(), u.0 .0, u.0 .1 .0)).map(|u| *u.0) else { break };
    tried.push(key);
    // what to test instead: nothing, or the sets most tested
    let mut others: Vec<(usize, StateSet)> = uses.iter().filter(|u| u.0 .0 == key.0 && *u.0 != key).map(|u| (u.1.len(), u.0 .1)).collect();
    others.sort_by(|a, b| b.0.cmp(&a.0).then(a.1 .0.cmp(&b.1 .0)));
    others.truncate(12);
    let mut undo: Vec<(usize, usize, usize, StateSet)> = Vec::new();
    let mut all = true;
    for &(l, i, cell) in &uses[&key] {
      let cur = tabs[l].cases[i].sets[cell];
      let choice = std::iter::once(valid).chain(others.iter().map(|o| from_key(cell, &o.1))).find(|new| may(&tabs[l], masks[l], i, cell, new, &mut scratch));
      match choice {
        Some(new) => {
          undo.push((l, i, cell, cur));
          set_to(&mut tabs[l], i, cell, &new);
        }
        None => {
          all = false;
          break;
        }
      }
    }
    if all {
      given_up += 1;
    } else {
      for (l, i, cell, old) in undo.into_iter().rev() {
        set_to(&mut tabs[l], i, cell, &old);
      }
    }
  }
  for (list, t) in lists.iter_mut().zip(tabs) {
    *list = t.cases.into_iter().zip(t.alive).filter(|x| x.1).map(|x| x.0).collect();
  }
  given_up
}
