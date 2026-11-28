//! Networks of two-input AND and XOR nodes with complemented edges, shared structurally: building
//! the same gate twice gives the same node, and constants fold as the network is built.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// A fast hash for small integer keys.
#[derive(Default, Clone, Copy)]
pub struct Fx(u64);

impl Hasher for Fx {
  fn finish(&self) -> u64 {
    self.0.rotate_left(26)
  }
  fn write(&mut self, bytes: &[u8]) {
    for &b in bytes {
      self.write_u64(b as u64);
    }
  }
  fn write_u64(&mut self, x: u64) {
    self.0 = (self.0.rotate_left(5) ^ x).wrapping_mul(0x517c_c1b7_2722_0a95);
  }
  fn write_u32(&mut self, x: u32) {
    self.write_u64(x as u64);
  }
  fn write_u8(&mut self, x: u8) {
    self.write_u64(x as u64);
  }
  fn write_usize(&mut self, x: usize) {
    self.write_u64(x as u64);
  }
}

pub type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

/// A node and whether it is complemented: `2 * node + complement`. Node 0 is the constant false.
pub type Lit = u32;
pub const FALSE: Lit = 0;
pub const TRUE: Lit = 1;

#[inline]
pub fn node(l: Lit) -> usize {
  (l >> 1) as usize
}
#[inline]
pub fn not(l: Lit) -> Lit {
  l ^ 1
}
/// All ones if the literal is complemented.
#[inline]
pub fn mask(l: Lit) -> u64 {
  0u64.wrapping_sub((l & 1) as u64)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Kind {
  Const,
  Input,
  And,
  Xor,
}

#[derive(Clone, Copy, Debug)]
pub struct Node {
  pub kind: Kind,
  pub a: Lit,
  pub b: Lit,
}

/// Nodes are in topological order: a node's inputs come before it. Node 0 is the constant, nodes
/// `1..=inputs` are the inputs.
#[derive(Clone)]
pub struct Net {
  pub nodes: Vec<Node>,
  pub inputs: usize,
  pub outputs: Vec<Lit>,
  /// Signals that are kept although no output needs them, for the optimiser to use: it may
  /// express other nodes with them at no cost, and those nothing ends up using are dropped by
  /// clearing this list.
  pub keep: Vec<Lit>,
  ands: FxMap<u64, u32>,
  xors: FxMap<u64, u32>,
}

impl Net {
  pub fn new(inputs: usize) -> Net {
    let mut nodes = vec![Node { kind: Kind::Const, a: 0, b: 0 }];
    nodes.extend((0..inputs).map(|_| Node { kind: Kind::Input, a: 0, b: 0 }));
    Net { nodes, inputs, outputs: Vec::new(), keep: Vec::new(), ands: FxMap::default(), xors: FxMap::default() }
  }

  pub fn input(&self, i: usize) -> Lit {
    debug_assert!(i < self.inputs);
    ((i + 1) << 1) as Lit
  }

  pub fn len(&self) -> usize {
    self.nodes.len()
  }

  pub fn and(&mut self, a: Lit, b: Lit) -> Lit {
    if a == FALSE || b == FALSE || a == not(b) {
      return FALSE;
    }
    if a == TRUE || a == b {
      return b;
    }
    if b == TRUE {
      return a;
    }
    let (a, b) = if a < b { (a, b) } else { (b, a) };
    let key = a as u64 | (b as u64) << 32;
    if let Some(&n) = self.ands.get(&key) {
      return n << 1;
    }
    let n = self.nodes.len() as u32;
    self.nodes.push(Node { kind: Kind::And, a, b });
    self.ands.insert(key, n);
    n << 1
  }

  pub fn or(&mut self, a: Lit, b: Lit) -> Lit {
    not(self.and(not(a), not(b)))
  }

  /// XOR nodes take plain inputs: complements move to the output.
  pub fn xor(&mut self, a: Lit, b: Lit) -> Lit {
    let c = (a ^ b) & 1;
    let (a, b) = (a & !1, b & !1);
    if a == b {
      return c;
    }
    if a == FALSE {
      return b ^ c;
    }
    if b == FALSE {
      return a ^ c;
    }
    let (a, b) = if a < b { (a, b) } else { (b, a) };
    let key = a as u64 | (b as u64) << 32;
    if let Some(&n) = self.xors.get(&key) {
      return n << 1 | c;
    }
    let n = self.nodes.len() as u32;
    self.nodes.push(Node { kind: Kind::Xor, a, b });
    self.xors.insert(key, n);
    n << 1 | c
  }

  /// `s ? t : e`
  pub fn mux(&mut self, s: Lit, t: Lit, e: Lit) -> Lit {
    if t == e {
      return t;
    }
    if t == not(e) {
      return self.xor(s, e);
    }
    match (t, e) {
      (TRUE, _) => self.or(s, e),
      (FALSE, _) => self.and(not(s), e),
      (_, TRUE) => self.or(not(s), t),
      (_, FALSE) => self.and(s, t),
      _ => {
        let x = self.and(s, t);
        let y = self.and(not(s), e);
        self.or(x, y)
      }
    }
  }

  pub fn gate(&mut self, kind: Kind, a: Lit, b: Lit) -> Lit {
    match kind {
      Kind::And => self.and(a, b),
      Kind::Xor => self.xor(a, b),
      _ => unreachable!(),
    }
  }

  /// The nodes some output depends on.
  pub fn live(&self) -> Vec<bool> {
    let mut live = vec![false; self.nodes.len()];
    for &o in self.outputs.iter().chain(&self.keep) {
      live[node(o)] = true;
    }
    for i in (self.inputs + 1..self.nodes.len()).rev() {
      if live[i] {
        live[node(self.nodes[i].a)] = true;
        live[node(self.nodes[i].b)] = true;
      }
    }
    live
  }

  /// AND and XOR nodes the outputs depend on (kept signals that no output needs do not count).
  pub fn gates(&self) -> usize {
    let mut plain = self.clone();
    plain.keep.clear();
    let live = plain.live();
    (self.inputs + 1..self.nodes.len()).filter(|&i| live[i]).count()
  }

  /// The same network without the nodes no output depends on.
  pub fn compact(&self) -> Net {
    self.compact_map().0
  }

  /// The same, and where every node went (meaningless for the nodes dropped).
  pub fn compact_map(&self) -> (Net, Vec<Lit>) {
    let live = self.live();
    let mut out = Net::new(self.inputs);
    let mut map: Vec<Lit> = (0..self.nodes.len()).map(|i| (i << 1) as Lit).collect();
    for i in self.inputs + 1..self.nodes.len() {
      if live[i] {
        let n = self.nodes[i];
        map[i] = out.gate(n.kind, map[node(n.a)] ^ (n.a & 1), map[node(n.b)] ^ (n.b & 1));
      }
    }
    out.outputs = self.outputs.iter().map(|&o| map[node(o)] ^ (o & 1)).collect();
    out.keep = self.keep.iter().map(|&o| map[node(o)] ^ (o & 1)).collect();
    (out, map)
  }

  /// Evaluate every node on `w` words: `buf[i * w..][..w]` is node i, and the rows of the inputs
  /// are filled in by the caller.
  pub fn simulate(&self, w: usize, buf: &mut [u64]) {
    buf[..w].fill(0);
    for i in self.inputs + 1..self.nodes.len() {
      let n = self.nodes[i];
      let (lo, hi) = buf.split_at_mut(i * w);
      let dst = &mut hi[..w];
      let a = &lo[node(n.a) * w..][..w];
      let b = &lo[node(n.b) * w..][..w];
      match n.kind {
        Kind::And => {
          let (ma, mb) = (mask(n.a), mask(n.b));
          for j in 0..w {
            dst[j] = (a[j] ^ ma) & (b[j] ^ mb);
          }
        }
        _ => {
          for j in 0..w {
            dst[j] = a[j] ^ b[j];
          }
        }
      }
    }
  }

  /// How many nodes and outputs read each node.
  pub fn refs(&self) -> Vec<u32> {
    let mut r = vec![0u32; self.nodes.len()];
    for i in self.inputs + 1..self.nodes.len() {
      r[node(self.nodes[i].a)] += 1;
      r[node(self.nodes[i].b)] += 1;
    }
    for &o in self.outputs.iter().chain(&self.keep) {
      r[node(o)] += 1;
    }
    r
  }
}
