//! Restructuring a network through a cover of small cuts.
//!
//! The optimiser's replacements keep the structure they find: one node at a time is expressed
//! again over other nodes. Here the network is derived again. Every node is a function of a few
//! nodes below it (a cut, of at most six leaves); a set of cuts that covers the outputs with
//! little logic is chosen, and each chosen node is built from its function alone, by
//! decomposition. The result computes the same function on every input, usually with more nodes
//! and with other signals: the optimiser then goes down from a different place.

use crate::check::Rng;
use crate::net::{mask, node, not, FxMap, Kind, Lit, Net, FALSE, TRUE};

const VAR: [u64; 6] = [0xAAAA_AAAA_AAAA_AAAA, 0xCCCC_CCCC_CCCC_CCCC, 0xF0F0_F0F0_F0F0_F0F0, 0xFF00_FF00_FF00_FF00, 0xFFFF_0000_FFFF_0000, 0xFFFF_FFFF_0000_0000];

/// The function with variable `j` set to 0 and to 1 (neither depends on `j`).
fn cofactors(tt: u64, j: usize) -> (u64, u64) {
  let s = 1u32 << j;
  let lo = tt & !VAR[j];
  let hi = tt & VAR[j];
  (lo | lo << s, hi | hi >> s)
}

/// Gates to build a function by `build`, counted without what its parts share.
struct Sizes(FxMap<u64, (u32, u8)>);

/// How a function is taken apart: by one variable.
#[derive(Clone, Copy)]
enum Step {
  /// `x & f1`, `!x & f0`, `!x | f1`, `x | f0`, `x ^ f0`
  And1(usize),
  And0(usize),
  Or1(usize),
  Or0(usize),
  Xor(usize),
  /// `x ? f1 : f0`
  Mux(usize),
  /// `f0 ^ (x & (f0 ^ f1))`
  Diff0(usize),
  /// `f1 ^ (!x & (f0 ^ f1))`
  Diff1(usize),
}

impl Sizes {
  fn literal(tt: u64) -> Option<(usize, bool)> {
    (0..6).find_map(|j| if tt == VAR[j] { Some((j, false)) } else if tt == !VAR[j] { Some((j, true)) } else { None })
  }

  /// The size of a function and the code of its best first step.
  fn of(&mut self, tt: u64) -> u32 {
    if tt == 0 || tt == !0 || Self::literal(tt).is_some() {
      return 0;
    }
    if let Some(&(s, _)) = self.0.get(&tt) {
      return s;
    }
    let step = self.step(tt);
    let (f0, f1) = cofactors(tt, step.1);
    let s = match step.0 {
      0 | 2 => 1 + self.of(f1),
      1 | 3 | 4 => 1 + self.of(f0),
      5 => 3 + self.of(f0) + self.of(f1),
      6 => 2 + self.of(f0) + self.of(f0 ^ f1),
      _ => 2 + self.of(f1) + self.of(f0 ^ f1),
    };
    self.0.insert(tt, (s, (step.0 << 3 | step.1 as u8) as u8));
    s
  }

  /// The first step that gives the smallest count: (kind, variable).
  fn step(&mut self, tt: u64) -> (u8, usize) {
    fn offer(best: &mut Option<(u32, u8, usize)>, c: u32, kind: u8, j: usize) {
      if best.is_none_or(|b| c < b.0) {
        *best = Some((c, kind, j));
      }
    }
    let mut best: Option<(u32, u8, usize)> = None;
    for j in 0..6 {
      let (f0, f1) = cofactors(tt, j);
      if f0 == f1 {
        continue;
      }
      if f0 == 0 {
        offer(&mut best, 1 + self.of(f1), 0, j);
      } else if f1 == 0 {
        offer(&mut best, 1 + self.of(f0), 1, j);
      } else if f0 == !0 {
        offer(&mut best, 1 + self.of(f1), 2, j);
      } else if f1 == !0 {
        offer(&mut best, 1 + self.of(f0), 3, j);
      } else if f0 == !f1 {
        offer(&mut best, 1 + self.of(f0), 4, j);
      }
    }
    if let Some(b) = best {
      return (b.1, b.2);
    }
    for j in 0..6 {
      let (f0, f1) = cofactors(tt, j);
      if f0 == f1 {
        continue;
      }
      let (s0, s1, d) = (self.of(f0), self.of(f1), self.of(f0 ^ f1));
      offer(&mut best, 3 + s0 + s1, 5, j);
      offer(&mut best, 2 + s0 + d, 6, j);
      offer(&mut best, 2 + s1 + d, 7, j);
    }
    let b = best.expect("a function that depends on nothing is a constant");
    (b.1, b.2)
  }

  fn first(&mut self, tt: u64) -> Step {
    self.of(tt);
    let code = self.0[&tt].1;
    let j = (code & 7) as usize;
    match code >> 3 {
      0 => Step::And1(j),
      1 => Step::And0(j),
      2 => Step::Or1(j),
      3 => Step::Or0(j),
      4 => Step::Xor(j),
      5 => Step::Mux(j),
      6 => Step::Diff0(j),
      _ => Step::Diff1(j),
    }
  }
}

/// Gates for a function of the literals `vars`.
fn build(net: &mut Net, sizes: &mut Sizes, made: &mut FxMap<u64, Lit>, tt: u64, vars: &[Lit; 6]) -> Lit {
  if tt == 0 {
    return FALSE;
  }
  if tt == !0 {
    return TRUE;
  }
  if let Some((j, neg)) = Sizes::literal(tt) {
    return vars[j] ^ neg as u32;
  }
  if let Some(&l) = made.get(&tt) {
    return l;
  }
  if let Some(&l) = made.get(&!tt) {
    return not(l);
  }
  let l = match sizes.first(tt) {
    Step::And1(j) => {
      let f = build(net, sizes, made, cofactors(tt, j).1, vars);
      net.and(vars[j], f)
    }
    Step::And0(j) => {
      let f = build(net, sizes, made, cofactors(tt, j).0, vars);
      net.and(not(vars[j]), f)
    }
    Step::Or1(j) => {
      let f = build(net, sizes, made, cofactors(tt, j).1, vars);
      net.or(not(vars[j]), f)
    }
    Step::Or0(j) => {
      let f = build(net, sizes, made, cofactors(tt, j).0, vars);
      net.or(vars[j], f)
    }
    Step::Xor(j) => {
      let f = build(net, sizes, made, cofactors(tt, j).0, vars);
      net.xor(vars[j], f)
    }
    Step::Mux(j) => {
      let (f0, f1) = cofactors(tt, j);
      let (a, b) = (build(net, sizes, made, f1, vars), build(net, sizes, made, f0, vars));
      net.mux(vars[j], a, b)
    }
    Step::Diff0(j) => {
      let (f0, f1) = cofactors(tt, j);
      let (a, d) = (build(net, sizes, made, f0, vars), build(net, sizes, made, f0 ^ f1, vars));
      let x = net.and(vars[j], d);
      net.xor(a, x)
    }
    Step::Diff1(j) => {
      let (f0, f1) = cofactors(tt, j);
      let (a, d) = (build(net, sizes, made, f1, vars), build(net, sizes, made, f0 ^ f1, vars));
      let x = net.and(not(vars[j]), d);
      net.xor(a, x)
    }
  };
  made.insert(tt, l);
  l
}

#[derive(Clone)]
struct Cut {
  /// nodes, ascending
  leaves: Vec<u32>,
  /// the node as a function of the leaves: leaf i is variable i
  tt: u64,
  flow: f32,
}

/// A function of `from` as a function of `to`, a superset.
fn stretch(tt: u64, from: &[u32], to: &[u32]) -> u64 {
  if from == to {
    return tt;
  }
  let pos: Vec<usize> = from.iter().map(|l| to.iter().position(|x| x == l).unwrap()).collect();
  let mut out = 0u64;
  for m in 0..64u64 {
    let mut idx = 0;
    for (j, &p) in pos.iter().enumerate() {
      idx |= (m >> p & 1) << j;
    }
    out |= (tt >> idx & 1) << m;
  }
  out
}

fn union(a: &[u32], b: &[u32], k: usize) -> Option<Vec<u32>> {
  let mut out = Vec::with_capacity(k);
  let (mut i, mut j) = (0, 0);
  while i < a.len() || j < b.len() {
    let x = if j == b.len() || (i < a.len() && a[i] <= b[j]) {
      if j < b.len() && a[i] == b[j] {
        j += 1;
      }
      i += 1;
      a[i - 1]
    } else {
      j += 1;
      b[j - 1]
    };
    if out.len() == k {
      return None;
    }
    out.push(x);
  }
  Some(out)
}

/// The network built again from a cover by cuts of at most `k` leaves (2 to 6), `keep` cuts
/// remembered per node; `rng` varies which cuts are taken: a cut's logic counts for up to
/// `1 + jitter` times what it is, so that a small `jitter` chooses among nearly equal cuts and a
/// large one takes covers further from the best. The result computes what `net` computes.
pub fn recut(net: &Net, k: usize, keep: usize, jitter: f32, rng: &mut Rng) -> Net {
  let net = net.compact();
  let n = net.len();
  let k = k.clamp(2, 6);
  let refs = net.refs();
  let mut sizes = Sizes(FxMap::default());
  let mut cuts: Vec<Vec<Cut>> = vec![Vec::new(); n];
  // what a node costs as a leaf: the flow of its best cut
  let mut flow = vec![0f32; n];
  let mut best: Vec<Option<Cut>> = vec![None; n];
  for i in 1..n {
    let own = Cut { leaves: vec![i as u32], tt: VAR[0], flow: 0.0 };
    if i <= net.inputs {
      cuts[i] = vec![own];
      continue;
    }
    let x = net.nodes[i];
    let (a, b) = (node(x.a), node(x.b));
    let mut found: Vec<Cut> = Vec::new();
    for ca in &cuts[a] {
      for cb in &cuts[b] {
        let Some(leaves) = union(&ca.leaves, &cb.leaves, k) else { continue };
        if found.iter().any(|f| f.leaves == leaves) {
          continue;
        }
        let (ta, tb) = (stretch(ca.tt, &ca.leaves, &leaves) ^ mask(x.a), stretch(cb.tt, &cb.leaves, &leaves) ^ mask(x.b));
        let tt = if x.kind == Kind::And { ta & tb } else { ta ^ tb };
        let below: f32 = leaves.iter().map(|&l| flow[l as usize]).sum();
        let jitter = 1.0 + jitter * (rng.below(1000) as f32 / 1000.0);
        found.push(Cut { leaves, tt, flow: (sizes.of(tt) as f32 * jitter + below) / refs[i].max(1) as f32 });
      }
    }
    found.sort_by(|p, q| p.flow.total_cmp(&q.flow));
    found.truncate(keep.max(1));
    flow[i] = found[0].flow;
    best[i] = Some(found[0].clone());
    found.push(Cut { leaves: vec![i as u32], tt: VAR[0], flow: flow[i] });
    cuts[i] = found;
  }
  // the cover: from the outputs down
  let mut wanted = vec![false; n];
  for &o in net.outputs.iter().chain(&net.keep) {
    wanted[node(o)] = true;
  }
  for i in (net.inputs + 1..n).rev() {
    if wanted[i] {
      for &l in &best[i].as_ref().unwrap().leaves {
        wanted[l as usize] = true;
      }
    }
  }
  let mut out = Net::new(net.inputs);
  let mut lit: Vec<Lit> = (0..n).map(|i| (i << 1) as Lit).collect();
  for i in net.inputs + 1..n {
    if !wanted[i] {
      continue;
    }
    let c = best[i].as_ref().unwrap();
    let mut vars = [FALSE; 6];
    for (j, &l) in c.leaves.iter().enumerate() {
      vars[j] = lit[l as usize];
    }
    let mut made: FxMap<u64, Lit> = FxMap::default();
    lit[i] = build(&mut out, &mut sizes, &mut made, c.tt, &vars);
  }
  out.outputs = net.outputs.iter().map(|&o| lit[node(o)] ^ (o & 1)).collect();
  out.keep = net.keep.iter().map(|&o| lit[node(o)] ^ (o & 1)).collect();
  out.compact()
}

/// The network with its trees of AND nodes and of XOR nodes put together again. A tree is a node
/// with the nodes of the same kind below it that nothing else reads: an AND, or an XOR, of
/// several inputs, in the order the network happened to pair them. Here the pairs that most
/// trees have in common are made first, one node for all of them, and `rng` chooses among pairs
/// that are equally common. Never more nodes than before; the same function on every input.
pub fn regroup(net: &Net, rng: &mut Rng) -> Net {
  let net = net.compact();
  let n = net.len();
  let refs = net.refs();
  // a node is inside its reader's tree if that reader is its only one, of its kind, and reads
  // it plain
  let mut inside = vec![false; n];
  for i in net.inputs + 1..n {
    let x = net.nodes[i];
    for l in [x.a, x.b] {
      let c = node(l);
      if c > net.inputs && l & 1 == 0 && refs[c] == 1 && net.nodes[c].kind == x.kind {
        inside[c] = true;
      }
    }
  }
  // the inputs of every tree: literals of nodes outside it
  let mut trees: Vec<(usize, Kind, Vec<u32>, bool)> = Vec::new();
  for i in net.inputs + 1..n {
    if inside[i] {
      continue;
    }
    let kind = net.nodes[i].kind;
    let mut items: Vec<u32> = Vec::new();
    let mut stack = vec![i];
    while let Some(x) = stack.pop() {
      for l in [net.nodes[x].a, net.nodes[x].b] {
        if inside[node(l)] && l & 1 == 0 && net.nodes[node(l)].kind == kind {
          stack.push(node(l));
        } else {
          items.push(l);
        }
      }
    }
    items.sort_unstable();
    let mut constant = false;
    if kind == Kind::And {
      items.dedup();
      // a literal and its complement: false
      constant = items.windows(2).any(|w| w[0] ^ 1 == w[1]);
    } else {
      // XOR: complements are a constant, a node twice is nothing
      let mut out: Vec<u32> = Vec::new();
      for l in items {
        constant ^= l & 1 == 1;
        let l = l & !1;
        if out.last() == Some(&l) {
          out.pop();
        } else {
          out.push(l);
        }
      }
      items = out;
    }
    trees.push((i, kind, items, constant));
  }
  // pairs made once: item ids from `first` on stand for (kind, a, b)
  let first = (2 * n) as u32;
  let mut pairs: Vec<(Kind, u32, u32)> = Vec::new();
  for kind in [Kind::And, Kind::Xor] {
    loop {
      let mut count: FxMap<(u32, u32), u32> = FxMap::default();
      for t in trees.iter().filter(|t| t.1 == kind && !(kind == Kind::And && t.3)) {
        for x in 0..t.2.len() {
          for y in x + 1..t.2.len() {
            *count.entry((t.2[x], t.2[y])).or_default() += 1;
          }
        }
      }
      let most = count.values().copied().max().unwrap_or(0);
      if most < 2 {
        break;
      }
      let mut top: Vec<(u32, u32)> = count.iter().filter(|c| *c.1 == most).map(|c| *c.0).collect();
      top.sort_unstable();
      let (a, b) = top[rng.below(top.len())];
      let id = first + pairs.len() as u32;
      pairs.push((kind, a, b));
      for t in trees.iter_mut().filter(|t| t.1 == kind) {
        if t.2.contains(&a) && t.2.contains(&b) {
          t.2.retain(|&l| l != a && l != b);
          t.2.push(id);
          t.2.sort_unstable();
        }
      }
    }
  }
  let mut out = Net::new(net.inputs);
  let mut lit: Vec<Lit> = (0..n).map(|i| (i << 1) as Lit).collect();
  let mut made: Vec<Option<Lit>> = vec![None; pairs.len()];
  fn item(out: &mut Net, lit: &[Lit], pairs: &[(Kind, u32, u32)], made: &mut [Option<Lit>], first: u32, l: u32) -> Lit {
    if l < first {
      return lit[node(l)] ^ (l & 1);
    }
    let p = (l - first) as usize;
    if let Some(m) = made[p] {
      return m;
    }
    let (kind, a, b) = pairs[p];
    let (x, y) = (item(out, lit, pairs, made, first, a), item(out, lit, pairs, made, first, b));
    let m = out.gate(kind, x, y);
    made[p] = Some(m);
    m
  }
  for (root, kind, items, constant) in &trees {
    let mut acc = if *kind == Kind::And { TRUE } else { FALSE };
    if *kind == Kind::And && *constant {
      acc = FALSE;
    } else {
      // in a random order: other pairs than last time
      let mut order = items.clone();
      for i in (1..order.len()).rev() {
        order.swap(i, rng.below(i + 1));
      }
      for &l in &order {
        let x = item(&mut out, &lit, &pairs, &mut made, first, l);
        acc = out.gate(*kind, acc, x);
      }
      if *kind == Kind::Xor && *constant {
        acc = not(acc);
      }
    }
    lit[*root] = acc;
  }
  out.outputs = net.outputs.iter().map(|&o| lit[node(o)] ^ (o & 1)).collect();
  out.keep = net.keep.iter().map(|&o| lit[node(o)] ^ (o & 1)).collect();
  out.compact()
}

#[cfg(test)]
mod tests {
  use super::*;

  /// A random network of 8 inputs, restructured with every cut size: the same on all 256 inputs.
  #[test]
  fn the_function_stays() {
    let mut rng = Rng(7);
    for round in 0..20 {
      let mut net = Net::new(8);
      let mut lits: Vec<Lit> = (0..8).map(|i| net.input(i)).collect();
      for _ in 0..60 {
        let (a, b) = (lits[rng.below(lits.len())] ^ rng.below(2) as u32, lits[rng.below(lits.len())] ^ rng.below(2) as u32);
        let l = if rng.below(3) == 0 { net.xor(a, b) } else { net.and(a, b) };
        lits.push(l);
      }
      net.outputs = (0..4).map(|i| lits[lits.len() - 1 - i * 3] ^ (i as u32 & 1)).collect();
      let values = |n: &Net| -> Vec<[u64; 4]> {
        let w = 4;
        let mut buf = vec![0u64; n.len() * w];
        for i in 0..8 {
          for j in 0..w {
            buf[(1 + i) * w + j] = if i < 6 { VAR[i] } else { 0u64.wrapping_sub((j >> (i - 6) & 1) as u64) };
          }
        }
        n.simulate(w, &mut buf);
        n.outputs.iter().map(|&o| std::array::from_fn(|j| buf[node(o) * w + j] ^ mask(o))).collect()
      };
      let want = values(&net.compact());
      for k in 2..=6 {
        let new = recut(&net, k, 1 + round % 8, [0.0, 0.25, 2.0][round % 3], &mut rng);
        assert_eq!(values(&new), want, "cuts of {k}");
      }
      let new = regroup(&net, &mut rng);
      assert_eq!(values(&new), want, "regrouped");
      assert!(new.len() <= net.compact().len(), "regrouping added nodes");
    }
  }
}
