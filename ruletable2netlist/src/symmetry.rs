//! Finding a rule's rotational symmetry from its function alone.
//!
//! Many rules look the same after a turn of the grid if the states are relabelled with it (an
//! arrow pointing east becomes an arrow pointing north). For a turn `pi` of the neighbours'
//! positions this module looks for a permutation `tau` of the states with
//!
//! ```text
//! next(tau c, tau x turned by pi) = tau next(c, x)
//! ```
//!
//! and reports for which centre states `c` that holds on every neighbourhood: a table may break
//! the symmetry for some (a fixed priority among simultaneous inputs, say) and keep it for the
//! rest. `tau` is found by backtracking on the neighbourhoods with at most one cell that is not
//! state 0, and then every centre state is checked on all neighbourhoods.

use crate::check::Space;
use ca_rule::{Interp, StateSet};

#[derive(Clone, Debug)]
pub struct Symmetry {
  /// turns that make a full one (4 for a quarter turn)
  pub order: usize,
  /// cell `i` goes to cell `pi[i]`; the centre, cell 0, stays
  pub pi: Vec<usize>,
  pub tau: Vec<u8>,
  /// centre states for which the rule is symmetric on every neighbourhood
  pub covariant: StateSet,
  /// `oriented[k]`: the centre states of orientation k. They are the states of the covariant
  /// orbits of `tau` that have `order` members; `oriented[k] = tau^k oriented[0]`
  pub oriented: Vec<StateSet>,
}

impl Symmetry {
  /// `tau^k` of a set of states.
  pub fn turn_set(&self, s: &StateSet, k: usize) -> StateSet {
    let mut cur = *s;
    for _ in 0..k % self.order {
      let mut next = StateSet::default();
      for v in cur.iter() {
        if (v as usize) < self.tau.len() {
          next.insert(self.tau[v as usize]);
        }
      }
      cur = next;
    }
    cur
  }
  /// `tau^k` of a state.
  pub fn turn(&self, v: u8, k: usize) -> u8 {
    (0..k % self.order).fold(v, |v, _| self.tau[v as usize])
  }
  /// The cell that cell `i` is at after `k` turns.
  pub fn turn_cell(&self, i: usize, k: usize) -> usize {
    (0..k % self.order).fold(i, |i, _| self.pi[i])
  }
  /// The orientation of a state, if it has one.
  pub fn orientation(&self, v: u8) -> Option<usize> {
    self.oriented.iter().position(|s| s.contains(v))
  }
  pub fn all_oriented(&self) -> StateSet {
    let mut s = StateSet::default();
    for o in &self.oriented {
      for i in 0..4 {
        s.0[i] |= o.0[i];
      }
    }
    s
  }
}

/// `f1[c][i][v]`: the next state of centre `c` with state `v` in cell `i` and 0 elsewhere.
fn single(space: Space, interp: &Interp) -> Vec<Vec<Vec<u8>>> {
  let mut scratch = vec![0u64; interp.words()];
  let mut nb = vec![0u8; space.k];
  (0..space.n)
    .map(|c| {
      (0..space.k)
        .map(|i| {
          (0..space.n)
            .map(|v| {
              nb.fill(0);
              nb[0] = c as u8;
              if i > 0 {
                nb[i] = v as u8;
              }
              interp.next(&nb, &mut scratch)
            })
            .collect()
        })
        .collect()
    })
    .collect()
}

/// A permutation `tau` (state 0 fixed) that is a symmetry on as many as possible of the
/// neighbourhoods with one cell other than 0. A table may break the symmetry for some centres,
/// so nothing is required to hold exactly here: a first guess pairs each state with the one
/// that does the same things one turn on, and exchanges of two images are made while they raise
/// the number of neighbourhoods on which the symmetry holds.
fn candidates(n: usize, k: usize, pi: &[usize], f1: &[Vec<Vec<u8>>]) -> Vec<Vec<u8>> {
  // What a state does at each cell, as the centre and as a neighbour, in numbers that a
  // relabelling keeps: a symmetry sends a state to one with the same numbers, turned by `pi`.
  let profile: Vec<Vec<i64>> = (0..n)
    .map(|s| {
      let mut p = vec![0i64; 3 * k];
      for i in 1..k {
        p[i] = (0..n).filter(|&v| f1[s][i][v] != f1[s][i][0]).count() as i64;
        p[k + i] = (0..n).filter(|&c| f1[c][i][s] != f1[c][i][0]).count() as i64;
        p[2 * k + i] = (0..n).map(|c| (0..n).filter(|&v| f1[c][i][v] as usize == s).count() as i64).sum();
      }
      p
    })
    .collect();
  let far = |s: usize, t: usize| -> i64 { (0..3).map(|q| (1..k).map(|i| (profile[s][q * k + i] - profile[t][q * k + pi[i]]).abs()).sum::<i64>()).sum() };
  // the closest pairs first; a state stays itself where that is as close as anything
  let mut pairs: Vec<(i64, bool, usize, usize)> = (1..n).flat_map(|s| (1..n).map(move |t| (s, t))).map(|(s, t)| (far(s, t), s != t, s, t)).collect();
  pairs.sort_unstable();
  let mut tau = vec![u8::MAX; n];
  let mut used = vec![false; n];
  tau[0] = 0;
  used[0] = true;
  for (_, _, s, t) in pairs {
    if tau[s] == u8::MAX && !used[t] {
      tau[s] = t as u8;
      used[t] = true;
    }
  }
  let held = |tau: &[u8]| -> usize {
    let mut h = 0;
    for c in 0..n {
      for i in 1..k {
        for v in 0..n {
          h += (f1[tau[c] as usize][pi[i]][tau[v] as usize] == tau[f1[c][i][v] as usize]) as usize;
        }
      }
    }
    h
  };
  let mut best = held(&tau);
  loop {
    let mut improved = false;
    for a in 1..n {
      for b in a + 1..n {
        tau.swap(a, b);
        let h = held(&tau);
        if h > best {
          best = h;
          improved = true;
        } else {
          tau.swap(a, b);
        }
      }
    }
    if !improved {
      break;
    }
  }
  if tau.iter().enumerate().all(|(i, &t)| i == t as usize) {
    return Vec::new();
  }
  vec![tau]
}

/// The centre states for which `tau` is a symmetry on every neighbourhood.
fn covariant(space: Space, interp: &Interp, pi: &[usize], tau: &[u8], threads: usize) -> StateSet {
  let total = space.total().unwrap();
  let per = total / space.n as u64;
  let fails: Vec<std::sync::atomic::AtomicBool> = (0..space.n).map(|_| std::sync::atomic::AtomicBool::new(false)).collect();
  let next = std::sync::atomic::AtomicUsize::new(0);
  std::thread::scope(|s| {
    for _ in 0..threads.max(1) {
      s.spawn(|| {
        let mut scratch = vec![0u64; interp.words()];
        let mut turned = vec![0u8; space.k];
        loop {
          let c = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
          if c >= space.n {
            break;
          }
          for idx in 0..per {
            let p = space.point(c as u64 * per + idx);
            let want = tau[interp.next(&p[..space.k], &mut scratch) as usize];
            for i in 0..space.k {
              turned[pi[i]] = tau[p[i] as usize];
            }
            if interp.next(&turned, &mut scratch) != want {
              fails[c].store(true, std::sync::atomic::Ordering::Relaxed);
              break;
            }
          }
        }
      });
    }
  });
  let mut set = StateSet::default();
  for c in 0..space.n {
    if !fails[c].load(std::sync::atomic::Ordering::Relaxed) {
      set.insert(c as u8);
    }
  }
  set
}

/// The rule's rotational symmetry, if it has one that gives some centre states an orientation.
/// Only rules whose neighbourhoods can all be enumerated (at most `limit`) are examined.
pub fn find(space: Space, interp: &Interp, limit: u64, threads: usize) -> Option<Symmetry> {
  if space.total().is_none_or(|t| t > limit) || space.k < 3 || space.n > 64 {
    return None;
  }
  let (n, k) = (space.n, space.k);
  let f1 = single(space, interp);
  let mut best: Option<Symmetry> = None;
  // quarter turns first, then half turns; both senses
  let nb = k - 1;
  for order in [4usize, 6, 2] {
    if nb % order != 0 {
      continue;
    }
    for step in [nb / order, nb - nb / order] {
      let pi: Vec<usize> = (0..k).map(|i| if i == 0 { 0 } else { 1 + (i - 1 + step) % nb }).collect();
      let cands = candidates(n, k, &pi, &f1);
      if std::env::var("R2N_DEBUG").is_ok() {
        eprintln!("symmetry: order {order} step {step}: {} candidates", cands.len());
      }
      for tau in cands {
        let cov = covariant(space, interp, &pi, &tau, threads);
        // states with an orientation: covariant orbits of full length
        let mut sym = Symmetry { order, pi: pi.clone(), tau: tau.clone(), covariant: cov, oriented: vec![StateSet::default(); order] };
        let mut seen = StateSet::default();
        for v in 0..n as u8 {
          if seen.contains(v) {
            continue;
          }
          let orbit: Vec<u8> = (0..order).map(|j| sym.turn(v, j)).collect();
          let full = sym.turn(v, order) == v && (1..order).all(|j| orbit[j] != v) && orbit.iter().all(|&o| cov.contains(o));
          for &o in &orbit {
            seen.insert(o);
          }
          if !full {
            continue;
          }
          // Orientation 0 goes to the member whose odd cell out is the lowest: the cell whose state
          // matters to this centre unlike the others' (the one an arrow points at, say). Members of
          // different orbits that point the same way then share an orientation. Without such a
          // cell: the member that, as a neighbour, acts from the lowest cells.
          let odd = |s: u8| -> usize {
            let cnt: Vec<i64> = (1..k).map(|i| (0..n).filter(|&v| f1[s as usize][i][v] != f1[s as usize][i][0]).count() as i64).collect();
            let far: Vec<i64> = cnt.iter().map(|&a| cnt.iter().map(|&b| (a - b).abs()).sum()).collect();
            let top = *far.iter().max().unwrap();
            if top > 0 && far.iter().filter(|&&f| f == top).count() == 1 {
              1 + far.iter().position(|&f| f == top).unwrap()
            } else {
              usize::MAX
            }
          };
          let acts = |s: u8| -> u32 {
            (1..k).filter(|&i| (0..n).any(|c| f1[c][i][s as usize] != f1[c][i][0])).fold(0u32, |m, i| m | 1 << i)
          };
          let first = (0..order).min_by_key(|&j| (odd(orbit[j]), acts(orbit[j]), j)).unwrap();
          for j in 0..order {
            sym.oriented[j].insert(orbit[(first + j) % order]);
          }
        }
        if std::env::var("R2N_DEBUG").is_ok() {
          let moved: Vec<String> = (0..n).filter(|&v| tau[v] as usize != v).map(|v| format!("{v}>{}", tau[v])).collect();
          eprintln!("  tau {} | covariant {} | oriented {:?}", moved.join(" "), cov.len(), sym.oriented.iter().map(|o| o.len()).collect::<Vec<_>>());
        }
        let better = best.as_ref().is_none_or(|b| (sym.all_oriented().len(), sym.covariant.len()) > (b.all_oriented().len(), b.covariant.len()));
        if better && !sym.oriented[0].is_empty() {
          best = Some(sym);
        }
      }
      if best.is_some() {
        return best;
      }
    }
  }
  best
}
