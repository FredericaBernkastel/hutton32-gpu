//! Algebraic factoring of sums of products: the divisors that many products share are found and
//! built once (fast extract, after Rajski and Vasudevamurthy).
//!
//! A sum of products is a list of cubes, a cube a sorted list of literals (`2 * variable +
//! complement`). Two kinds of divisor are extracted, the most profitable first:
//!
//! * a pair of literals that occurs in several cubes becomes one AND, used by all of them;
//! * a sum of two cubes `d1 + d2` that is left when the literals two cubes of a sum share are
//!   taken out (`b d1 + b d2 = b (d1 + d2)`) becomes one node, used wherever two cubes leave it.
//!
//! The second also factors a single sum, and it is what finds a condition such as "an excited
//! signal from any side" once for all the transitions that test it.

use crate::net::FxMap;
use ca_rule::StateSet;

pub type Cube = Vec<u32>;
pub type Sop = Vec<Cube>;

/// Divisors with more literals than this are not looked for.
const MAX_DIVISOR: usize = 10;

/// `a` without the literals of `b`, and the literals both have (all sorted).
fn split(a: &[u32], b: &[u32]) -> (Cube, Cube, Cube) {
  let (mut base, mut da, mut db) = (Vec::new(), Vec::new(), Vec::new());
  let (mut i, mut j) = (0, 0);
  while i < a.len() || j < b.len() {
    if j == b.len() || (i < a.len() && a[i] < b[j]) {
      da.push(a[i]);
      i += 1;
    } else if i == a.len() || b[j] < a[i] {
      db.push(b[j]);
      j += 1;
    } else {
      base.push(a[i]);
      i += 1;
      j += 1;
    }
  }
  (base, da, db)
}

/// Drop cubes that cannot be true, repeated cubes and cubes that contain another.
fn tidy(sop: &mut Sop) {
  for c in sop.iter_mut() {
    c.sort_unstable();
    c.dedup();
  }
  sop.retain(|c| !c.windows(2).any(|w| w[0] ^ 1 == w[1]));
  sop.sort();
  sop.dedup();
  let all = sop.clone();
  sop.retain(|c| !all.iter().any(|o| o.len() < c.len() && o.iter().all(|l| c.binary_search(l).is_ok())));
}

/// Extract shared divisors from `sops` (the roots). Variables from `vars` on are new nodes:
/// variable `vars + k` is `nodes[k]`, itself a sum of products over earlier variables. The roots
/// are rewritten in place to use them.
pub fn fast_extract(sops: &mut Vec<Sop>, vars: u32) -> Vec<Sop> {
  let roots = sops.len();
  for s in sops.iter_mut() {
    tidy(s);
  }
  // roots and new nodes together: sum r < roots is a root, sum roots + k is variable vars + k
  let mut all: Vec<Sop> = std::mem::take(sops);
  loop {
    // pairs of literals
    let mut pairs: FxMap<(u32, u32), u32> = FxMap::default();
    // sums of two cubes: what they save, and where they occur (sum, cube, cube)
    let mut doubles: FxMap<(Cube, Cube), (usize, Vec<(usize, usize, usize)>)> = FxMap::default();
    for (s, sop) in all.iter().enumerate() {
      for c in sop {
        for i in 0..c.len() {
          for j in i + 1..c.len() {
            *pairs.entry((c[i], c[j])).or_default() += 1;
          }
        }
      }
      for i in 0..sop.len() {
        for j in i + 1..sop.len() {
          let (base, d1, d2) = split(&sop[i], &sop[j]);
          if d1.is_empty() || d2.is_empty() || d1.len() + d2.len() > MAX_DIVISOR {
            continue;
          }
          let saves = base.len() + d1.len() + d2.len() - 1;
          let key = if d1 <= d2 { (d1, d2) } else { (d2, d1) };
          let e = doubles.entry(key).or_default();
          e.0 += saves;
          e.1.push((s, i, j));
        }
      }
    }
    let best_pair = pairs.iter().map(|(k, &n)| (n as i64 - 1, *k)).max();
    // a double's own gates are paid once; a cube serves one occurrence only
    let best_double = doubles
      .iter()
      .map(|(k, (saves, at))| (*saves as i64 - (k.0.len() + k.1.len()) as i64 + 1, at.len(), k))
      .filter(|d| d.0 > 0)
      .max_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(b.2.cmp(a.2)));
    let pair_gain = best_pair.map_or(0, |p| p.0);
    let double_gain = best_double.as_ref().map_or(0, |d| d.0);
    if pair_gain <= 0 && double_gain <= 0 {
      break;
    }
    let var = vars + (all.len() - roots) as u32;
    if double_gain >= pair_gain {
      let key = best_double.unwrap().2.clone();
      let at = doubles[&key].1.clone();
      let mut used: FxMap<(usize, usize), ()> = FxMap::default();
      let mut dead: Vec<(usize, usize)> = Vec::new();
      let mut applied = 0;
      for (s, i, j) in at {
        if used.contains_key(&(s, i)) || used.contains_key(&(s, j)) {
          continue;
        }
        used.insert((s, i), ());
        used.insert((s, j), ());
        let (mut base, _, _) = split(&all[s][i], &all[s][j]);
        base.push(var << 1);
        base.sort_unstable();
        all[s][i] = base;
        dead.push((s, j));
        applied += 1;
      }
      debug_assert!(applied > 0);
      dead.sort_unstable_by(|a, b| b.cmp(a));
      for (s, j) in dead {
        all[s].remove(j);
      }
      all.push(vec![key.0, key.1]);
    } else {
      let (a, b) = best_pair.unwrap().1;
      for sop in all.iter_mut() {
        for c in sop.iter_mut() {
          if c.binary_search(&a).is_ok() && c.binary_search(&b).is_ok() {
            c.retain(|&l| l != a && l != b);
            c.push(var << 1);
            c.sort_unstable();
          }
        }
      }
      all.push(vec![vec![a, b]]);
    }
    for s in all.iter_mut() {
      tidy(s);
    }
  }
  let nodes = all.split_off(roots);
  *sops = all;
  nodes
}

// ---- cubes of a set of states ------------------------------------------------------------------

/// A cube over the bits of a state: the bits in `mask` have the values in `value`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BitCube {
  pub mask: u8,
  pub value: u8,
}

fn low(t: &StateSet, n: usize) -> StateSet {
  // the half of an n-variable table where variable n - 1 is 0
  let bits = 1usize << (n - 1);
  let mut r = StateSet::default();
  for w in 0..4 {
    let lo = w * 64;
    if lo >= bits {
      break;
    }
    r.0[w] = if bits - lo >= 64 { t.0[w] } else { t.0[w] & ((1u64 << (bits - lo)) - 1) };
  }
  r
}

fn high(t: &StateSet, n: usize) -> StateSet {
  let bits = 1usize << (n - 1);
  let mut r = StateSet::default();
  if bits >= 64 {
    let shift = bits / 64;
    for w in 0..4 - shift {
      r.0[w] = t.0[w + shift];
    }
  } else {
    r.0[0] = t.0[0] >> bits;
  }
  low(&r, n)
}

fn full(n: usize) -> StateSet {
  let mut r = StateSet::default();
  let bits = 1usize << n;
  for w in 0..4 {
    let lo = w * 64;
    if lo < bits {
      r.0[w] = if bits - lo >= 64 { !0 } else { (1u64 << (bits - lo)) - 1 };
    }
  }
  r
}

fn join(lo: &StateSet, hi: &StateSet, n: usize) -> StateSet {
  let bits = 1usize << (n - 1);
  let mut r = *lo;
  if bits >= 64 {
    let shift = bits / 64;
    for w in 0..4 - shift {
      r.0[w + shift] |= hi.0[w];
    }
  } else {
    r.0[0] |= hi.0[0] << bits;
  }
  r
}

fn and(a: &StateSet, b: &StateSet) -> StateSet {
  StateSet([a.0[0] & b.0[0], a.0[1] & b.0[1], a.0[2] & b.0[2], a.0[3] & b.0[3]])
}
fn or(a: &StateSet, b: &StateSet) -> StateSet {
  StateSet([a.0[0] | b.0[0], a.0[1] | b.0[1], a.0[2] | b.0[2], a.0[3] | b.0[3]])
}
fn minus(a: &StateSet, b: &StateSet) -> StateSet {
  StateSet([a.0[0] & !b.0[0], a.0[1] & !b.0[1], a.0[2] & !b.0[2], a.0[3] & !b.0[3]])
}

/// An irredundant sum of cubes that covers `lower` and stays inside `upper` (Minato-Morreale),
/// over the lowest `n` variables; returns what the cubes cover.
fn isop(lower: &StateSet, upper: &StateSet, n: usize, out: &mut Vec<BitCube>) -> StateSet {
  if lower.is_empty() {
    return StateSet::default();
  }
  if *upper == full(n) {
    out.push(BitCube { mask: 0, value: 0 });
    return full(n);
  }
  let (l0, l1, u0, u1) = (low(lower, n), high(lower, n), low(upper, n), high(upper, n));
  let bit = 1u8 << (n - 1);
  let start = out.len();
  let f0 = isop(&minus(&l0, &u1), &u0, n - 1, out);
  for c in out[start..].iter_mut() {
    c.mask |= bit;
  }
  let mid = out.len();
  let f1 = isop(&minus(&l1, &u0), &u1, n - 1, out);
  for c in out[mid..].iter_mut() {
    c.mask |= bit;
    c.value |= bit;
  }
  let rest = or(&minus(&l0, &f0), &minus(&l1, &f1));
  let f2 = isop(&rest, &and(&u0, &u1), n - 1, out);
  join(&or(&f0, &f2), &or(&f1, &f2), n)
}

/// Cubes over the bits of a state that are true on `set` and false on the other states among
/// the first `states` codes; codes that are no state are free.
pub fn cubes_of(set: &StateSet, states: usize, bits: usize) -> Vec<BitCube> {
  let mut valid = StateSet::default();
  for v in 0..states {
    valid.insert(v as u8);
  }
  let lower = and(set, &valid);
  let upper = or(&lower, &minus(&full(bits), &valid));
  let mut out = Vec::new();
  let cover = isop(&lower, &upper, bits, &mut out);
  debug_assert!(and(&cover, &valid) == lower);
  out
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn factors_a_common_sum() {
    // p a + p b + q a + q b  ->  (p + q)(a + b), two divisors
    let (p, q, a, b) = (0u32, 2, 4, 6);
    let mut sops = vec![vec![vec![p, a], vec![p, b], vec![q, a], vec![q, b]]];
    let nodes = fast_extract(&mut sops, 4);
    let cost = |s: &Sop| s.iter().map(|c| c.len().saturating_sub(1)).sum::<usize>() + s.len().saturating_sub(1);
    let total: usize = sops.iter().chain(&nodes).map(cost).sum();
    assert!(total <= 3, "{sops:?} {nodes:?}");
  }

  #[test]
  fn cubes_cover_exactly() {
    let mut s = StateSet::default();
    for v in [13u8, 27, 28, 29, 31] {
      s.insert(v);
    }
    let cubes = cubes_of(&s, 32, 5);
    for v in 0..32u8 {
      assert_eq!(cubes.iter().any(|c| v & c.mask == c.value), s.contains(v));
    }
    // 20 states: codes 20 to 31 are free
    let mut t = StateSet::default();
    for v in 16..20u8 {
      t.insert(v);
    }
    assert_eq!(cubes_of(&t, 20, 5), vec![BitCube { mask: 16, value: 16 }]);
  }
}
