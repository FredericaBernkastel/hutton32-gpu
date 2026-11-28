//! A rule as a network of two-input gates over bit-planes: element k of each input is bit k of a
//! cell's state for 32 or 64 cells at once (bit i = cell i). Evaluated on `u64` words on the CPU and
//! emitted as a WGSL function for the GPU kernels.
//!
//! The netlist itself ([`ca_rule::Netlist`]: inputs, gates, outputs, its text form) is shared with
//! the other crates of the workspace; `ruletable2netlist` compiles a rule table into one. What a
//! [`Network`] adds is the knowledge the engines need: which compiled fast path computes it
//! ([`Kind`]), the Life-like adder tree, and the checks against a [`Rule`].

pub use ca_rule::{Gate, Netlist, Op, Signal};
use rustc_hash::FxHashMap;

use super::{Neighbourhood, Pos, Rule};

/// Which network this is, for compiled fast paths.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
  /// the embedded 367-gate circuit of Hutton32b (`generated/hutton32b_planes.rs` computes it)
  Hutton32b,
  /// the embedded 597-gate circuit of Hutton32 (`generated/hutton32_planes.rs` computes it)
  Hutton32,
  /// the adder tree of a Life-like rule
  LifeLike(u16, u16, Neighbourhood),
  Other,
}

/// A netlist and the fast path that computes it.
#[derive(Clone, Debug)]
pub struct Network {
  net: Netlist,
  kind: Kind,
}

/// Builds a network gate by gate, folding constants and sharing identical gates.
struct Builder {
  gates: Vec<Gate>,
  memo: FxHashMap<Gate, Signal>,
}

impl Builder {
  fn new() -> Self {
    Builder { gates: Vec::new(), memo: FxHashMap::default() }
  }
  fn gate(&mut self, op: Op, a: Signal, b: Signal) -> Signal {
    use Signal::{One, Zero};
    match (op, a, b) {
      (Op::Not, Zero, _) => return One,
      (Op::Not, One, _) => return Zero,
      (Op::And, Zero, _) | (Op::And, _, Zero) => return Zero,
      (Op::And, One, x) | (Op::And, x, One) => return x,
      (Op::Or, One, _) | (Op::Or, _, One) => return One,
      (Op::Or, Zero, x) | (Op::Or, x, Zero) => return x,
      (Op::Xor, Zero, x) | (Op::Xor, x, Zero) => return x,
      (Op::Xor, One, x) | (Op::Xor, x, One) => return self.gate(Op::Not, x, Zero),
      _ => {}
    }
    let (a, b) = if op != Op::Not && format!("{b:?}") < format!("{a:?}") { (b, a) } else { (a, b) };
    let g = Gate { op, a, b: if op == Op::Not { Zero } else { b } };
    if let Some(&s) = self.memo.get(&g) {
      return s;
    }
    let s = Signal::Gate(self.gates.len() as u32);
    self.gates.push(g);
    self.memo.insert(g, s);
    s
  }
  fn not(&mut self, a: Signal) -> Signal {
    self.gate(Op::Not, a, Signal::Zero)
  }
  fn and(&mut self, a: Signal, b: Signal) -> Signal {
    self.gate(Op::And, a, b)
  }
  fn or(&mut self, a: Signal, b: Signal) -> Signal {
    self.gate(Op::Or, a, b)
  }
  fn xor(&mut self, a: Signal, b: Signal) -> Signal {
    self.gate(Op::Xor, a, b)
  }
}

impl Network {
  // ---- construction ---------------------------------------------------------------------------

  /// A network from a netlist (e.g. one compiled by `ruletable2netlist`).
  pub fn from_netlist(net: Netlist) -> Network {
    Network { net, kind: Kind::Other }
  }

  /// The same network, computed by the fast path `kind`.
  pub fn with_kind(mut self, kind: Kind) -> Network {
    self.kind = kind;
    self
  }

  /// Parse a netlist. The embedded Hutton32b circuit is recognised by its header comment.
  pub fn parse(text: &str) -> Result<Network, String> {
    let net = Netlist::parse(text)?;
    let kind = if net.gates.len() == 367 && net.bits == 5 && text.contains("Hutton32b") { Kind::Hutton32b } else { Kind::Other };
    Ok(Network { net, kind })
  }

  /// The adder tree of a Life-like rule: the live neighbours are counted by full and half adders,
  /// and the rule's B and S sets are decoded from the count.
  pub fn life_like(birth: u16, survive: u16, nbhd: Neighbourhood) -> Network {
    let mut b = Builder::new();
    let neighbours: Vec<Pos> = nbhd.fields().iter().copied().filter(|&p| p != Pos::C).collect();
    let mut inputs = vec![(Pos::C, 0u32)];
    inputs.extend(neighbours.iter().map(|&p| (p, 0u32)));
    let c = Signal::Input(0);
    let mut cols: Vec<Vec<Signal>> = vec![(1..inputs.len()).map(|i| Signal::Input(i as u16)).collect()];
    let mut j = 0;
    while j < cols.len() {
      while cols[j].len() > 1 {
        if cols.len() == j + 1 {
          cols.push(Vec::new());
        }
        if cols[j].len() >= 3 {
          let (x, y, z) = (cols[j].remove(0), cols[j].remove(0), cols[j].remove(0));
          let t = b.xor(x, y);
          let s = b.xor(t, z);
          let c1 = b.and(x, y);
          let c2 = b.and(t, z);
          let carry = b.or(c1, c2);
          cols[j].push(s);
          cols[j + 1].push(carry);
        } else {
          let (x, y) = (cols[j].remove(0), cols[j].remove(0));
          let s = b.xor(x, y);
          let carry = b.and(x, y);
          cols[j].push(s);
          cols[j + 1].push(carry);
        }
      }
      j += 1;
    }
    let count: Vec<Signal> = cols.iter().map(|c| c.first().copied().unwrap_or(Signal::Zero)).collect();
    let max = neighbours.len() as u32;
    let minterm = |k: u32, b: &mut Builder| -> Signal {
      let mut m = Signal::One;
      for (bit, &s) in count.iter().enumerate() {
        let lit = if k >> bit & 1 == 1 { s } else { b.not(s) };
        m = b.and(m, lit);
      }
      m
    };
    let set = |mask: u16, b: &mut Builder| -> Signal {
      let mut acc = Signal::Zero;
      for k in 0..=max {
        if mask >> k & 1 == 1 {
          let m = minterm(k, b);
          acc = b.or(acc, m);
        }
      }
      acc
    };
    let born = set(birth, &mut b);
    let keep = set(survive, &mut b);
    let nc = b.not(c);
    let x = b.and(c, keep);
    let y = b.and(nc, born);
    let out = b.or(x, y);
    Network { net: Netlist { bits: 1, inputs, gates: b.gates, outputs: vec![out] }, kind: Kind::LifeLike(birth, survive, nbhd) }
  }

  // ---- accessors ------------------------------------------------------------------------------

  /// The netlist (shared form of the workspace).
  pub fn netlist(&self) -> &Netlist {
    &self.net
  }
  pub fn bits(&self) -> u32 {
    self.net.bits
  }
  pub fn inputs(&self) -> &[(Pos, u32)] {
    &self.net.inputs
  }
  pub fn gates(&self) -> &[Gate] {
    &self.net.gates
  }
  pub fn outputs(&self) -> &[Signal] {
    &self.net.outputs
  }
  pub fn kind(&self) -> Kind {
    self.kind
  }
  /// The positions the network reads, in the order of [`Pos::ALL`].
  pub fn positions(&self) -> Vec<Pos> {
    self.net.positions()
  }
  /// Gates per operation: `(not, and, or, xor)`.
  pub fn counts(&self) -> (usize, usize, usize, usize) {
    self.net.counts()
  }
  /// Longest path from an input to an output, in gates.
  pub fn depth(&self) -> usize {
    self.net.depth()
  }

  // ---- evaluation -----------------------------------------------------------------------------

  /// Evaluate on 64 cells at once. `input(pos, bit)` is the plane of bit `bit` of the cells at
  /// `pos`; `out[k]` receives plane k of the next states. `scratch` is reused between calls.
  #[inline]
  pub fn eval64(&self, input: &dyn Fn(Pos, u32) -> u64, scratch: &mut Vec<u64>, out: &mut [u64]) {
    let n = &self.net;
    let ni = n.inputs.len();
    scratch.clear();
    scratch.extend(n.inputs.iter().map(|&(p, b)| input(p, b)));
    scratch.reserve(n.gates.len());
    for g in &n.gates {
      let v = |s: Signal, sc: &Vec<u64>| -> u64 {
        match s {
          Signal::Input(i) => sc[i as usize],
          Signal::Gate(j) => sc[ni + j as usize],
          Signal::Zero => 0,
          Signal::One => !0,
        }
      };
      let a = v(g.a, scratch);
      let r = match g.op {
        Op::Not => !a,
        Op::And => a & v(g.b, scratch),
        Op::Or => a | v(g.b, scratch),
        Op::Xor => a ^ v(g.b, scratch),
      };
      scratch.push(r);
    }
    for (k, &s) in n.outputs.iter().enumerate() {
      out[k] = match s {
        Signal::Input(i) => scratch[i as usize],
        Signal::Gate(j) => scratch[ni + j as usize],
        Signal::Zero => 0,
        Signal::One => !0,
      };
    }
  }

  /// Every signal on 32 cells at once: `words[i]` is input `i` ([`Network::inputs`]); the result
  /// holds the inputs, then gate `j` at `inputs().len() + j`. The outputs are the signals of
  /// [`Network::outputs`], read with [`Network::signal`].
  pub fn trace(&self, words: &[u32]) -> Vec<u32> {
    let n = &self.net;
    assert_eq!(words.len(), n.inputs.len(), "one word per input");
    let mut sc: Vec<u32> = words.to_vec();
    let ni = n.inputs.len();
    for g in &n.gates {
      let v = |s: Signal, sc: &Vec<u32>| -> u32 {
        match s {
          Signal::Input(i) => sc[i as usize],
          Signal::Gate(j) => sc[ni + j as usize],
          Signal::Zero => 0,
          Signal::One => !0,
        }
      };
      let a = v(g.a, &sc);
      let r = match g.op {
        Op::Not => !a,
        Op::And => a & v(g.b, &sc),
        Op::Or => a | v(g.b, &sc),
        Op::Xor => a ^ v(g.b, &sc),
      };
      sc.push(r);
    }
    sc
  }

  /// The word of signal `s` in a [`Network::trace`].
  pub fn signal(&self, trace: &[u32], s: Signal) -> u32 {
    match s {
      Signal::Input(i) => trace[i as usize],
      Signal::Gate(j) => trace[self.net.inputs.len() + j as usize],
      Signal::Zero => 0,
      Signal::One => !0,
    }
  }

  /// The input words of a [`Network::trace`] from the planes the inspector reads
  /// ([`crate::inspect::Planes`]: per plane, the words at `[C, N, NE, E, SE, S, SW, W, NW]`).
  pub fn trace_inputs(&self, planes: &[[u32; 9]]) -> Vec<u32> {
    self.net.inputs.iter().map(|&(p, b)| planes.get(b as usize).map(|w| w[Pos::ALL.iter().position(|&q| q == p).unwrap()]).unwrap_or(0)).collect()
  }

  /// Exhaustive check against the rule's table: the number of neighbourhoods on which the network
  /// differs from it. Neighbourhoods containing a code that is no state of the rule do not count.
  pub fn verify(&self, rule: &Rule) -> u64 {
    let table = rule.table().expect("verify needs a rule with a table");
    let nbhd = rule.neighbourhood();
    let bits = rule.planes();
    let fields = nbhd.fields();
    let n_idx = table.len();
    let states = rule.states();
    let mut scratch = Vec::new();
    let mut out = vec![0u64; self.net.bits as usize];
    let mut bad = 0u64;
    let mut base = 0usize;
    while base < n_idx {
      let lanes = (n_idx - base).min(64);
      let input = |p: Pos, b: u32| -> u64 {
        let f = match fields.iter().position(|&q| q == p) {
          Some(f) => f,
          None => return 0,
        };
        let mut w = 0u64;
        for i in 0..lanes {
          let idx = base + i;
          let state = (idx >> (bits as usize * f)) & ((1 << bits) - 1);
          w |= (((state >> b) & 1) as u64) << i;
        }
        w
      };
      self.eval64(&input, &mut scratch, &mut out);
      for i in 0..lanes {
        let idx = base + i;
        let valid = (0..fields.len()).all(|f| ((idx >> (bits as usize * f)) & ((1 << bits) - 1)) < states as usize);
        if !valid {
          continue;
        }
        let mut got = 0u32;
        for (k, w) in out.iter().enumerate() {
          got |= ((w >> i) & 1) as u32 * (1 << k);
        }
        if got != table[idx] as u32 {
          bad += 1;
        }
      }
      base += 64;
    }
    bad
  }

  // ---- WGSL and text --------------------------------------------------------------------------

  /// The network as a WGSL function `fn {name}(c: array<u32, B>, n: array<u32, B>, ...) -> array<u32, B>`
  /// with one parameter per position of `params` (bit i of element k = bit k of cell i).
  pub fn wgsl(&self, name: &str, params: &[Pos]) -> String {
    self.net.wgsl(name, params)
  }

  /// The netlist text of this network (the format [`Network::parse`] reads).
  pub fn to_netlist(&self) -> String {
    self.net.to_netlist()
  }
}
