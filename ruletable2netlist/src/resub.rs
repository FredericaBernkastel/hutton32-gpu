//! Making a network smaller without changing what it computes: resubstitution with don't-cares.
//!
//! Every node is simulated on a sample of neighbourhoods. For a target node, the sample shows
//! where its value reaches an output (flip the node, re-evaluate what depends on it); elsewhere
//! its value is free. If a constant, another node, or one or two new gates over other nodes agree
//! with the target where it matters, the target is replaced, and the nodes only it used go away.
//! A replacement is kept only if the network gets smaller and still equals the table on every
//! neighbourhood of the checker; a neighbourhood that shows a difference joins the sample.

use crate::check::{Checker, Nb, Rng, Space};
use crate::net::{mask, node, FxMap, Kind, Lit, Net, FALSE, TRUE};
use ca_rule::table::{Case, StateSet};
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct Stats {
  pub sweeps: usize,
  /// replacements kept
  pub accepted: usize,
  /// checks against the table
  pub checks: usize,
  /// checks that found a difference
  pub refuted: usize,
}

impl Stats {
  pub fn add(&mut self, o: &Stats) {
    self.sweeps += o.sweeps;
    self.accepted += o.accepted;
    self.checks += o.checks;
    self.refuted += o.refuted;
  }
}

/// The neighbourhoods the nodes are simulated on.
#[derive(Clone)]
pub struct Sample {
  pub space: Space,
  pub pts: Vec<Nb>,
  /// the sample is every neighbourhood of the rule: what holds on it holds
  pub exact: bool,
}

impl Sample {
  /// Every neighbourhood of a small rule. Otherwise random ones and, for every transition,
  /// neighbourhoods it matches and the same with one cell changed: without them a transition
  /// that few neighbourhoods match would look like a constant.
  pub fn new(space: Space, cases: &[Case], rng: &mut Rng) -> Sample {
    if let Some(total) = space.total().filter(|&t| t <= 1 << 12) {
      return Sample { space, pts: (0..total).map(|i| space.point(i)).collect(), exact: true };
    }
    let mut pts: Vec<Nb> = (0..4096).map(|_| space.random(rng)).collect();
    let per = (6000 / cases.len().max(1)).clamp(1, 6);
    let mut valid = StateSet::default();
    for v in 0..space.n {
      valid.insert(v as u8);
    }
    let pick = |s: &StateSet, rng: &mut Rng| -> Option<u8> {
      let m: Vec<u8> = s.iter().filter(|&v| valid.contains(v)).collect();
      if m.is_empty() {
        None
      } else {
        Some(m[rng.below(m.len())])
      }
    };
    'cases: for c in cases {
      for _ in 0..per {
        let mut p = [0u8; 9];
        let mut set = [false; 9];
        for g in &c.eq {
          let mut s = StateSet([!0; 4]);
          for &m in g {
            for i in 0..4 {
              s.0[i] &= c.sets[m].0[i];
            }
          }
          let Some(v) = pick(&s, rng) else { continue 'cases };
          for &m in g {
            p[m] = v;
            set[m] = true;
          }
        }
        for i in 0..space.k {
          if !set[i] {
            let Some(v) = pick(&c.sets[i], rng) else { continue 'cases };
            p[i] = v;
          }
        }
        pts.push(p);
        let mut q = p;
        q[rng.below(space.k)] = rng.below(space.n) as u8;
        pts.push(q);
      }
    }
    let care = crate::check::CARE.load(std::sync::atomic::Ordering::Relaxed);
    if care != 0 {
      pts.retain(|p| care >> p[0] & 1 == 1);
    }
    Sample { space, pts, exact: false }
  }

  pub fn add(&mut self, pts: &[Nb]) {
    self.pts.extend_from_slice(pts);
  }

  /// Words a node's values take, and the planes of the inputs.
  fn planes(&self) -> (usize, Vec<u64>) {
    let w = self.pts.len().div_ceil(64).max(1);
    let mut pts = self.pts.clone();
    pts.resize(w * 64, *self.pts.last().unwrap());
    let mut out = vec![0u64; self.space.inputs() * w];
    self.space.pack(&pts, w, &mut out);
    (w, out)
  }
}

/// What may replace a target.
#[derive(Clone, Copy, Debug)]
enum Cand {
  L(Lit),
  /// `(a op b) ^ neg`
  G(Kind, Lit, Lit, bool),
  /// `(a op ((b op2 c) ^ inner)) ^ neg`
  G2(Kind, Lit, Kind, Lit, Lit, bool, bool),
  /// `a ? b : c`
  Mux(Lit, Lit, Lit),
}

impl Cand {
  fn lits(&self) -> ([Lit; 3], usize) {
    match *self {
      Cand::L(a) => ([a, 0, 0], 1),
      Cand::G(_, a, b, _) => ([a, b, 0], 2),
      Cand::G2(_, a, _, b, c, _, _) => ([a, b, c], 3),
      Cand::Mux(a, b, c) => ([a, b, c], 3),
    }
  }
  fn build(&self, net: &mut Net, m: impl Fn(Lit) -> Lit) -> Lit {
    match *self {
      Cand::L(a) => m(a),
      Cand::G(k, a, b, neg) => net.gate(k, m(a), m(b)) ^ neg as u32,
      Cand::G2(k, a, k2, b, c, inner, neg) => {
        let x = net.gate(k2, m(b), m(c)) ^ inner as u32;
        net.gate(k, m(a), x) ^ neg as u32
      }
      Cand::Mux(a, b, c) => net.mux(m(a), m(b), m(c)),
    }
  }
}

/// A literal that covers a set, and its ones outside it.
struct Cover {
  lit: Lit,
  x: Vec<u64>,
}

fn disjoint(a: &[u64], b: &[u64]) -> bool {
  a.iter().zip(b).all(|(x, y)| x & y == 0)
}

fn zero(a: &[u64]) -> bool {
  a.iter().all(|&x| x == 0)
}

/// A hash of `row & care` that is linear over XOR, so that hashes of sums are sums of hashes.
fn lhash(row: &[u64], care: &[u64]) -> u64 {
  let mut h = 0u64;
  for (j, (r, c)) in row.iter().zip(care).enumerate() {
    h ^= (r & c).rotate_left(((j * 7 + j / 64 * 13) & 63) as u32);
  }
  h
}

struct Engine {
  net: Net,
  w: usize,
  /// planes of the inputs on the sample
  inp: Vec<u64>,
  /// values of every node on the sample, `[node * w + word]`
  sig: Vec<u64>,
  refs: Vec<u32>,
  /// nodes the sweep has dealt with
  done: Vec<bool>,
  /// also take a replacement that leaves the size as it is, if it changes the network
  sideways: bool,
  /// keep a replacement only if it lowers the gates counted with their inverters (`gate_cost`)
  gate_aware: bool,
  gate_cost: usize,
  // per target
  mark: Vec<bool>,
  flip: Vec<u64>,
  care: Vec<u64>,
}

impl Engine {
  fn new(net: Net, sample: &Sample) -> Engine {
    let mut e = Engine { net, w: 0, inp: Vec::new(), sig: Vec::new(), refs: Vec::new(), done: Vec::new(), sideways: false, gate_aware: false, gate_cost: 0, mark: Vec::new(), flip: Vec::new(), care: Vec::new() };
    e.reload(sample);
    e
  }

  fn reload(&mut self, sample: &Sample) {
    let (w, inp) = sample.planes();
    self.w = w;
    self.inp = inp;
    self.resim();
  }

  fn resim(&mut self) {
    let (w, n) = (self.w, self.net.len());
    self.sig.clear();
    self.sig.resize(n * w, 0);
    self.sig[w..w + self.inp.len()].copy_from_slice(&self.inp);
    self.net.simulate(w, &mut self.sig);
    self.refs = self.net.refs();
    self.mark.resize(n, false);
    self.flip.resize(n * w, 0);
    self.care.resize(w, 0);
  }

  fn gates(&self) -> usize {
    self.net.len() - 1 - self.net.inputs
  }

  fn row(&self, i: usize) -> &[u64] {
    &self.sig[i * self.w..][..self.w]
  }

  /// Nodes that go away with the target, counted up to 4.
  fn mffc(&self, t: usize) -> usize {
    let mut dec: FxMap<usize, u32> = FxMap::default();
    let mut stack = vec![t];
    let mut count = 1;
    while let Some(x) = stack.pop() {
      let n = self.net.nodes[x];
      for l in [n.a, n.b] {
        let f = node(l);
        if f <= self.net.inputs {
          continue;
        }
        let d = dec.entry(f).or_default();
        *d += 1;
        if *d == self.refs[f] {
          count += 1;
          if count >= 4 {
            return count;
          }
          stack.push(f);
        }
      }
    }
    count
  }

  /// Where the target's value reaches an output: `self.care`; `self.mark` gets the nodes that
  /// depend on the target, which cannot be used to rebuild it.
  fn observe(&mut self, t: usize) {
    let (w, n) = (self.w, self.net.len());
    self.mark[..n].fill(false);
    self.mark[t] = true;
    for j in 0..w {
      self.flip[t * w + j] = !self.sig[t * w + j];
    }
    for i in t + 1..n {
      let x = self.net.nodes[i];
      let (a, b) = (node(x.a), node(x.b));
      if !self.mark[a] && !self.mark[b] {
        continue;
      }
      self.mark[i] = true;
      let (ma, mb) = (mask(x.a), mask(x.b));
      for j in 0..w {
        let va = if self.mark[a] { self.flip[a * w + j] } else { self.sig[a * w + j] };
        let vb = if self.mark[b] { self.flip[b * w + j] } else { self.sig[b * w + j] };
        self.flip[i * w + j] = if x.kind == Kind::And { (va ^ ma) & (vb ^ mb) } else { va ^ vb };
      }
    }
    self.care.fill(0);
    for &o in self.net.outputs.iter().chain(&self.net.keep) {
      let o = node(o);
      if self.mark[o] {
        for j in 0..w {
          self.care[j] |= self.flip[o * w + j] ^ self.sig[o * w + j];
        }
      }
    }
  }

  /// The network with the target replaced, if that is smaller; with it the sweep's marks.
  fn trial(&self, t: usize, c: &Cand) -> Option<(Net, Vec<bool>)> {
    let n = self.net.len();
    let inputs = self.net.inputs;
    let (cl, cn) = c.lits();
    // what stays in use
    let mut live = vec![false; n];
    let mut stack: Vec<usize> = self.net.outputs.iter().chain(&self.net.keep).map(|&o| node(o)).collect();
    while let Some(x) = stack.pop() {
      if live[x] {
        continue;
      }
      live[x] = true;
      if x == t {
        stack.extend(cl[..cn].iter().map(|&l| node(l)));
      } else if x > inputs {
        stack.push(node(self.net.nodes[x].a));
        stack.push(node(self.net.nodes[x].b));
      }
    }
    const UNSET: Lit = u32::MAX;
    let mut map = vec![UNSET; n];
    for (i, m) in map.iter_mut().enumerate().take(inputs + 1) {
      *m = (i << 1) as Lit;
    }
    let mut new = Net::new(inputs);
    for i in inputs + 1..n {
      if !live[i] || map[i] != UNSET {
        continue;
      }
      // the replacement may use nodes that come after the target: resolve on demand
      stack.clear();
      stack.push(i);
      while let Some(&x) = stack.last() {
        if map[x] != UNSET {
          stack.pop();
          continue;
        }
        let deps: ([Lit; 3], usize) = if x == t { (cl, cn) } else { ([self.net.nodes[x].a, self.net.nodes[x].b, 0], 2) };
        let mut ready = true;
        for &l in &deps.0[..deps.1] {
          if map[node(l)] == UNSET {
            stack.push(node(l));
            ready = false;
          }
        }
        if ready {
          let m = |l: Lit| map[node(l)] ^ (l & 1);
          map[x] = if x == t { c.build(&mut new, m) } else { new.gate(self.net.nodes[x].kind, m(self.net.nodes[x].a), m(self.net.nodes[x].b)) };
          stack.pop();
        }
      }
      if new.len() >= n + self.sideways as usize {
        return None;
      }
    }
    if new.len() >= n + self.sideways as usize {
      return None;
    }
    new.outputs = self.net.outputs.iter().map(|&o| map[node(o)] ^ (o & 1)).collect();
    new.keep = self.net.keep.iter().map(|&o| map[node(o)] ^ (o & 1)).collect();
    // gates that folded away may have left nodes nothing reads
    let (new, map2) = new.compact_map();
    if self.gate_aware && crate::map::quick_cost(&new) >= self.gate_cost {
      return None;
    }
    if new.len() == n && (0..n).all(|i| (new.nodes[i].kind, new.nodes[i].a, new.nodes[i].b) == (self.net.nodes[i].kind, self.net.nodes[i].a, self.net.nodes[i].b)) {
      return None;
    }
    let mut done = vec![false; new.len()];
    let at = |i: usize| node(map2[node(map[i])]);
    for i in inputs + 1..n {
      if live[i] && self.done[i] {
        done[at(i)] = true;
      }
    }
    done[at(t)] = true;
    Some((new, done))
  }

  /// The smallest network over the candidates, if one is smaller than the present one.
  fn best(&self, t: usize, cands: &[Cand]) -> Option<(Net, Vec<bool>)> {
    let mut best: Option<(Net, Vec<bool>)> = None;
    for c in cands.iter().take(12) {
      if let Some(r) = self.trial(t, c) {
        if best.as_ref().is_none_or(|b| r.0.len() < b.0.len()) {
          best = Some(r);
        }
      }
    }
    best
  }

  /// Literals of usable nodes that are 1 wherever `on` is: the `limit` with the fewest ones in
  /// `off`, and those ones. Literals with none are left out: they equal the target.
  fn covers(&self, on: &[u64], off: &[u64], limit: usize) -> Vec<Cover> {
    let w = self.w;
    let mut found: Vec<(u32, Lit)> = Vec::new();
    for d in 1..self.net.len() {
      if self.mark[d] {
        continue;
      }
      let r = self.row(d);
      let (mut ok0, mut ok1) = (true, true);
      for j in 0..w {
        ok0 &= on[j] & !r[j] == 0;
        ok1 &= on[j] & r[j] == 0;
        if !ok0 && !ok1 {
          break;
        }
      }
      if ok0 {
        let e: u32 = (0..w).map(|j| (r[j] & off[j]).count_ones()).sum();
        if e > 0 {
          found.push((e, (d << 1) as Lit));
        }
      }
      if ok1 {
        let e: u32 = (0..w).map(|j| (!r[j] & off[j]).count_ones()).sum();
        if e > 0 {
          found.push((e, (d << 1 | 1) as Lit));
        }
      }
    }
    found.sort_unstable();
    found.truncate(limit);
    found
      .into_iter()
      .map(|(_, l)| {
        let r = self.row(node(l));
        Cover { lit: l, x: (0..w).map(|j| (r[j] ^ mask(l)) & off[j]).collect() }
      })
      .collect()
  }

  /// Pairs of covers with no common one outside the set: their AND is the set.
  fn pairs(cv: &[Cover], max: usize) -> Vec<(Lit, Lit)> {
    let mut out = Vec::new();
    for i in 0..cv.len() {
      for j in i + 1..cv.len() {
        if disjoint(&cv[i].x, &cv[j].x) {
          out.push((cv[i].lit, cv[j].lit));
          if out.len() >= max {
            return out;
          }
        }
      }
    }
    out
  }

  /// Pairs of usable nodes whose XOR is `target` (or its complement: the flag) wherever `care`.
  fn xor_pairs(&self, target: &[u64], care: &[u64], max: usize) -> Vec<(Lit, Lit, bool)> {
    let w = self.w;
    let ht = lhash(target, care);
    let hc = lhash(care, care);
    let mut hs: Vec<(usize, u64)> = Vec::with_capacity(self.net.len());
    let mut map: FxMap<u64, u32> = FxMap::default();
    for d in 1..self.net.len() {
      if !self.mark[d] {
        let h = lhash(self.row(d), care);
        hs.push((d, h));
        map.entry(h).or_insert(d as u32);
      }
    }
    let mut out: Vec<(Lit, Lit, bool)> = Vec::new();
    for &(a, ha) in &hs {
      for neg in [false, true] {
        let key = ht ^ ha ^ if neg { hc } else { 0 };
        let Some(&b) = map.get(&key) else { continue };
        let b = b as usize;
        if b == a || out.iter().any(|o| (node(o.0), node(o.1)) == (a.min(b), a.max(b))) {
          continue;
        }
        let (ra, rb) = (self.row(a), self.row(b));
        let m = if neg { !0 } else { 0 };
        if (0..w).all(|j| (ra[j] ^ rb[j] ^ target[j] ^ m) & care[j] == 0) {
          out.push(((a.min(b) << 1) as Lit, (a.max(b) << 1) as Lit, neg));
          if out.len() >= max {
            return out;
          }
        }
      }
    }
    out
  }

  /// Try to replace node `t`. `effort`: 1 for a constant, a node or one gate; 2 and 3 add two
  /// gates.
  fn try_target(&mut self, t: usize, effort: u32) -> Option<(Net, Vec<bool>)> {
    let w = self.w;
    let room = self.mffc(t);
    self.observe(t);
    let care = self.care.clone();
    let tr: Vec<u64> = self.row(t).to_vec();
    let on: Vec<u64> = (0..w).map(|j| tr[j] & care[j]).collect();
    let off: Vec<u64> = (0..w).map(|j| !tr[j] & care[j]).collect();
    if zero(&on) {
      return self.trial(t, &Cand::L(FALSE));
    }
    if zero(&off) {
      return self.trial(t, &Cand::L(TRUE));
    }

    // another node
    let mut cands: Vec<Cand> = Vec::new();
    for d in 1..self.net.len() {
      if self.mark[d] {
        continue;
      }
      let r = self.row(d);
      let x = (r[0] ^ tr[0]) & care[0];
      if x != 0 && x != care[0] {
        continue;
      }
      if (0..w).all(|j| (r[j] ^ tr[j]) & care[j] == 0) {
        cands.push(Cand::L((d << 1) as Lit));
      } else if (0..w).all(|j| (r[j] ^ tr[j]) & care[j] == care[j]) {
        cands.push(Cand::L((d << 1 | 1) as Lit));
      }
      if cands.len() >= 4 {
        break;
      }
    }
    if let Some(r) = self.best(t, &cands) {
      return Some(r);
    }
    if room < 2 && !self.sideways {
      return None;
    }

    // one gate: AND of two covers (OR for the complement), XOR of two nodes
    cands.clear();
    let cv = [self.covers(&on, &off, 40), self.covers(&off, &on, 40)];
    for p in 0..2 {
      for (a, b) in Self::pairs(&cv[p], 6) {
        cands.push(Cand::G(Kind::And, a, b, p == 1));
      }
    }
    for (a, b, neg) in self.xor_pairs(&tr, &care, 6) {
      cands.push(Cand::G(Kind::Xor, a, b, neg));
    }
    if let Some(r) = self.best(t, &cands) {
      return Some(r);
    }
    if room + (self.sideways as usize) < 3 || effort < 2 {
      return None;
    }

    // two gates
    cands.clear();
    for p in 0..2 {
      let on = if p == 0 { &on } else { &off };
      let c = &cv[p];
      // a & b & c
      let m = c.len().min(24);
      let mut found = 0;
      'triples: for i in 0..m {
        for j in i + 1..m {
          let y: Vec<u64> = (0..w).map(|q| c[i].x[q] & c[j].x[q]).collect();
          for k in j + 1..m {
            if disjoint(&y, &c[k].x) {
              cands.push(Cand::G2(Kind::And, c[i].lit, Kind::And, c[j].lit, c[k].lit, false, p == 1));
              found += 1;
              if found >= 4 {
                break 'triples;
              }
            }
          }
        }
      }
      // a & (b | c) and a & (b ^ c): what is left to tell apart is the off-set inside a
      for a in c.iter().take(if effort >= 3 { 6 } else { 3 }) {
        let sub: Vec<u64> = a.x.clone();
        let inner = self.covers(&sub, on, 24);
        for (x, y) in Self::pairs(&inner, 2) {
          // x & y is 1 on the off-set inside a and 0 on the on-set
          cands.push(Cand::G2(Kind::And, a.lit, Kind::And, x, y, true, p == 1));
        }
        let care2: Vec<u64> = (0..w).map(|q| on[q] | sub[q]).collect();
        for (x, y, neg) in self.xor_pairs(on, &care2, 2) {
          cands.push(Cand::G2(Kind::And, a.lit, Kind::Xor, x, y, neg, p == 1));
        }
      }
    }
    if effort >= 3 {
      // a ^ (b & c): the nodes nearest the target
      let mut near: Vec<(u32, usize)> = Vec::new();
      let total: u32 = care.iter().map(|c| c.count_ones()).sum();
      for d in 1..self.net.len() {
        if !self.mark[d] {
          let r = self.row(d);
          let dist: u32 = (0..w).map(|j| ((r[j] ^ tr[j]) & care[j]).count_ones()).sum();
          let dist = dist.min(total - dist);
          if dist > 0 {
            near.push((dist, d));
          }
        }
      }
      near.sort_unstable();
      for &(_, d) in near.iter().take(6) {
        let r = self.row(d);
        let ron: Vec<u64> = (0..w).map(|j| (r[j] ^ tr[j]) & care[j]).collect();
        let roff: Vec<u64> = (0..w).map(|j| !(r[j] ^ tr[j]) & care[j]).collect();
        for (x, y) in Self::pairs(&self.covers(&ron, &roff, 24), 2) {
          cands.push(Cand::G2(Kind::Xor, (d << 1) as Lit, Kind::And, x, y, false, false));
        }
        for (x, y) in Self::pairs(&self.covers(&roff, &ron, 24), 2) {
          cands.push(Cand::G2(Kind::Xor, (d << 1) as Lit, Kind::And, x, y, true, false));
        }
      }
      // a ^ b ^ c
      if self.net.len() <= 900 {
        let ht = lhash(&tr, &care);
        let hc = lhash(&care, &care);
        let hs: Vec<(usize, u64)> = (1..self.net.len()).filter(|&d| !self.mark[d]).map(|d| (d, lhash(self.row(d), &care))).collect();
        let mut map: FxMap<u64, u32> = FxMap::default();
        for &(d, h) in &hs {
          map.entry(h).or_insert(d as u32);
        }
        let mut found = 0;
        'xor3: for (i, &(a, ha)) in hs.iter().enumerate() {
          for &(b, hb) in &hs[i + 1..] {
            for neg in [false, true] {
              let Some(&c) = map.get(&(ht ^ ha ^ hb ^ if neg { hc } else { 0 })) else { continue };
              let c = c as usize;
              if c == a || c == b {
                continue;
              }
              let (ra, rb, rc) = (self.row(a), self.row(b), self.row(c));
              let m = if neg { !0 } else { 0 };
              if (0..w).all(|j| (ra[j] ^ rb[j] ^ rc[j] ^ tr[j] ^ m) & care[j] == 0) {
                cands.push(Cand::G2(Kind::Xor, (a << 1) as Lit, Kind::Xor, (b << 1) as Lit, (c << 1) as Lit, false, neg));
                found += 1;
                if found >= 3 {
                  break 'xor3;
                }
              }
            }
          }
        }
      }
    }
    if let Some(r) = self.best(t, &cands) {
      return Some(r);
    }
    if room + (self.sideways as usize) < 4 {
      return None;
    }

    // a selection: one node where a nearby node is 1, another where it is 0
    cands.clear();
    let mut near: Vec<usize> = Vec::new();
    let mut queue = vec![t];
    let mut at = 0;
    while at < queue.len() && near.len() < 32 {
      let x = self.net.nodes[queue[at]];
      at += 1;
      for l in [x.a, x.b] {
        let f = node(l);
        if f != 0 && !near.contains(&f) {
          near.push(f);
          if f > self.net.inputs {
            queue.push(f);
          }
        }
      }
    }
    for &a in &near {
      if self.mark[a] {
        continue;
      }
      let ra = self.row(a);
      let halves = [(0..w).map(|j| care[j] & ra[j]).collect::<Vec<u64>>(), (0..w).map(|j| care[j] & !ra[j]).collect::<Vec<u64>>()];
      if zero(&halves[0]) || zero(&halves[1]) {
        continue;
      }
      // for each half a node that equals the target there
      let mut found: [Option<Lit>; 2] = [None, None];
      for (h, half) in halves.iter().enumerate() {
        let (ht, hc) = (lhash(&tr, half), lhash(half, half));
        for d in 1..self.net.len() {
          if self.mark[d] || d == a {
            continue;
          }
          let r = self.row(d);
          let hd = lhash(r, half);
          if hd == ht && (0..w).all(|j| (r[j] ^ tr[j]) & half[j] == 0) {
            found[h] = Some((d << 1) as Lit);
            break;
          }
          if hd == ht ^ hc && (0..w).all(|j| (r[j] ^ tr[j]) & half[j] == half[j]) {
            found[h] = Some((d << 1 | 1) as Lit);
            break;
          }
        }
        if found[h].is_none() {
          break;
        }
      }
      if let [Some(b), Some(c)] = found {
        cands.push(Cand::Mux((a << 1) as Lit, b, c));
        if cands.len() >= 4 {
          break;
        }
      }
    }
    self.best(t, &cands)
  }
}

/// One sweep over every node. Whether it ran to the end.
#[allow(clippy::too_many_arguments)]
fn sweep(e: &mut Engine, sample: &mut Sample, checker: &Checker, quick: &Option<Checker>, deadline: Instant, stats: &mut Stats, effort: u32, forward: bool, skip: &mut dyn FnMut() -> bool) -> bool {
  e.done = vec![false; e.net.len()];
  let mut snap = (e.net.clone(), e.done.clone());
  let (mut pending, mut batch) = (0usize, 1usize);
  let mut cursor: Option<usize> = None;
  let mut timed_out = false;
  loop {
    let n = e.net.len();
    let first = e.net.inputs + 1;
    let t = if timed_out {
      None
    } else if forward {
      (cursor.unwrap_or(first)..n).find(|&i| !e.done[i])
    } else {
      (first..cursor.map_or(n, |c| c.min(n))).rev().find(|&i| !e.done[i])
    };
    let mut verify = false;
    match t {
      Some(t) => {
        e.done[t] = true;
        cursor = Some(if forward { t + 1 } else { t });
        if !skip() {
          if let Some((net, done)) = e.try_target(t, effort) {
            e.net = net;
            e.done = done;
            e.resim();
            if e.gate_aware {
              e.gate_cost = crate::map::quick_cost(&e.net);
            }
            stats.accepted += 1;
            pending += 1;
            cursor = None;
            verify = !sample.exact && pending >= batch;
          }
        }
        timed_out = Instant::now() >= deadline;
      }
      None => {
        if pending == 0 || sample.exact {
          break;
        }
        verify = true;
      }
    }
    if verify {
      stats.checks += 1;
      let mut bad = match quick {
        Some(q) => q.check(&e.net, 16),
        None => Vec::new(),
      };
      if bad.is_empty() {
        bad = checker.check(&e.net, 16);
      }
      if bad.is_empty() {
        snap = (e.net.clone(), e.done.clone());
        batch = (batch * 2).min(64);
      } else {
        stats.refuted += 1;
        e.net = snap.0.clone();
        e.done = snap.1.clone();
        sample.add(&bad);
        e.reload(sample);
        batch = 1;
        cursor = None;
      }
      pending = 0;
    }
  }
  !timed_out
}

/// Replace the engine's network by the same function derived again from a cover by cuts
/// (`recut`): more nodes as a rule, and other signals for the sweeps to work with.
fn restructure(e: &mut Engine, checker: &Checker, rng: &mut Rng, log: &(dyn Fn(String) + Sync)) {
  let k = 4 + rng.below(3);
  // one time in three the trees of AND and of XOR nodes are paired again instead
  let trees = rng.below(3) == 0;
  let new = if trees { crate::recut::regroup(&e.net, rng) } else { crate::recut::recut(&e.net, k, 8, [0.25, 0.25, 0.6, 1.5][rng.below(4)], rng) };
  // it is the same function by construction; a difference would be an error in the code
  if !checker.check(&new, 1).is_empty() {
    log("restructuring changed the function: not used".to_string());
    return;
  }
  log(format!("restructured by {}: {} -> {} nodes", if trees { "trees".to_string() } else { format!("cuts of {k}") }, e.gates(), new.len() - 1 - new.inputs));
  e.net = new;
  e.resim();
}

/// Sweep the network until a sweep finds nothing; then, with `sideways` and until `until`, change
/// it without making it larger (replacements of equal size at a part of the nodes) and sweep again, keeping the
/// smallest network seen. Gives up early after many changes in a row that led nowhere.
#[allow(clippy::too_many_arguments)]
fn walk(net: Net, sample: &mut Sample, checker: &Checker, quick: &Option<Checker>, until: Instant, stats: &mut Stats, seed: u64, sideways: bool, log: &(dyn Fn(String) + Sync)) -> Net {
  // the second half of the time, or what is left when the nodes stop going down, counts the
  // inverters too
  let end = until;
  let until = if sideways { until - until.saturating_duration_since(Instant::now()) / 2 } else { until };
  let mut e = Engine::new(net, sample);
  let effort = |gates: usize| if gates > 6000 { 1 } else if gates > 1500 { 2 } else { 3 };
  let size = |n: &Net| n.len() - 1 - n.inputs;
  let mut rng = Rng(seed);
  // how many of a hundred changes are restructurings
  let recut = RECUT.load(std::sync::atomic::Ordering::Relaxed);
  let mut n = 0;
  let mut best = e.net.clone();
  let mut stale = 0;
  loop {
    // down to a network no sweep makes smaller
    let mut complete;
    loop {
      n += 1;
      stats.sweeps += 1;
      let before = e.gates();
      complete = sweep(&mut e, sample, checker, quick, until, stats, effort(before), n % 2 == 1, &mut || false);
      log(format!("sweep {n} (effort {}): {before} -> {} nodes, sample {}", effort(before), e.gates(), sample.pts.len()));
      if !complete || e.gates() >= before {
        break;
      }
    }
    if e.gates() < size(&best) {
      best = e.net.clone();
      stale = 0;
    } else if e.gates() > size(&best) {
      e.net = best.clone();
      e.resim();
    }
    if !complete || !sideways || e.gates() > 3000 || stale >= if recut > 0 { 600 } else { 60 } {
      break;
    }
    // sideways, or the whole network derived again from a cover by cuts
    stale += 1;
    if rng.below(100) < recut {
      restructure(&mut e, checker, &mut rng, log);
      continue;
    }
    e.sideways = true;
    complete = sweep(&mut e, sample, checker, quick, until, stats, 2, true, &mut || rng.below(3) != 0);
    e.sideways = false;
    if !complete {
      break;
    }
  }
  if e.gates() < size(&best) {
    best = e.net.clone();
  }
  if sideways && size(&best) <= 3000 && best.keep.is_empty() {
    // Fewest nodes is not fewest gates: a node read in both polarities costs an inverter. Sweeps
    // that keep a replacement only if nodes plus inverters go down, while they find any.
    // Then, as for the nodes: change the network at equal size, bring the nodes down again,
    // bring the gates down again, and keep the result if it has fewer gates.
    e.net = best.clone();
    e.resim();
    let start = crate::map::quick_cost(&e.net);
    let mut least = usize::MAX;
    let mut stale = 0;
    let mut round = 0;
    loop {
      e.sideways = true;
      e.gate_aware = true;
      e.gate_cost = crate::map::quick_cost(&e.net);
      let mut complete;
      loop {
        let before = e.gate_cost;
        n += 1;
        complete = sweep(&mut e, sample, checker, quick, end, stats, 3, n % 2 == 1, &mut || false);
        if !complete || e.gate_cost >= before {
          break;
        }
      }
      e.gate_aware = false;
      e.sideways = false;
      let cost = crate::map::quick_cost(&e.net);
      if cost < least {
        least = cost;
        best = e.net.clone();
        stale = 0;
      } else {
        // as many gates: go on from the new network; more: back to the best
        if cost > least {
          e.net = best.clone();
          e.resim();
        }
        stale += 1;
      }
      round += 1;
      if !complete || stale >= if recut > 0 { 400 } else { 40 } {
        break;
      }
      // change it, then nodes down again
      if rng.below(100) < recut {
        restructure(&mut e, checker, &mut rng, log);
        complete = true;
      } else {
        e.sideways = true;
        complete = sweep(&mut e, sample, checker, quick, end, stats, 2, true, &mut || rng.below(3) != 0);
        e.sideways = false;
      }
      while complete {
        let before = e.gates();
        n += 1;
        complete = sweep(&mut e, sample, checker, quick, end, stats, 3, n % 2 == 1, &mut || false);
        if e.gates() >= before {
          break;
        }
      }
      if !complete {
        break;
      }
    }
    log(format!("with inverters: {start} -> {least} gates in {round} rounds, {} nodes", best.len() - 1 - best.inputs));
  }
  best
}

/// Of a hundred changes, how many are restructurings (`Options::restructure`).
pub static RECUT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(60);

/// Mixed into the random choices of the search: another value, another walk.
pub static SEED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Optimise until `deadline`: down to a network no sweep makes smaller, and with `sideways` on
/// from there by changes of equal size. The network returned equals the table on every
/// neighbourhood of `checker`. One thread searches; the checks use them all, and they are most of
/// the work on a rule with many neighbourhoods, so searching on several threads gains nothing.
pub fn optimise(net: Net, sample: &mut Sample, checker: &Checker, deadline: Instant, sideways: bool, stats: &mut Stats, log: &(dyn Fn(String) + Sync)) -> Net {
  // a quick check that catches most differences before the full one
  let quick = if sample.exact || checker.points() <= 1 << 18 { None } else { Some(checker.subsample(1 << 18, 0x5eed)) };
  let net = net.compact();
  let seed = 0x5a17 ^ net.len() as u64 ^ SEED.load(std::sync::atomic::Ordering::Relaxed).wrapping_mul(0x9E37_79B9_7F4A_7C15);
  walk(net, sample, checker, &quick, deadline, stats, seed, sideways, log)
}
