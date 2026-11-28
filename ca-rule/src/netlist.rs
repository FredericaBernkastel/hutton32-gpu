//! A rule as a netlist of NOT, AND, OR and XOR gates over bit-planes: bit k of each cell's state
//! is one input, bit k of the next state one output, and a machine word carries one such bit for
//! 32 or 64 cells at once.
//!
//! Text form, one gate a line:
//!
//! ```text
//! INPUT c0 c1 c2 c3 c4 n0 ... w4
//! t0 = OR c0 c1
//! t2 = NOT c0
//! OUTPUT o0 t295
//! ```
//!
//! An input is a cell (`c n ne e se s sw w nw`) followed by a bit of its state.

use crate::nbhd::Pos;
use std::collections::HashMap;
use std::fmt::Write;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Op {
  Not,
  And,
  Or,
  Xor,
}

/// An input, a gate's output or a constant.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Signal {
  /// index into [`Netlist::inputs`]
  Input(u16),
  Gate(u32),
  Zero,
  One,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Gate {
  pub op: Op,
  pub a: Signal,
  /// unused for `Not`
  pub b: Signal,
}

/// Gates are in order: a gate reads inputs, constants and earlier gates.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Netlist {
  /// bits of a state
  pub bits: u32,
  /// input i is bit `.1` of the state of the cell at `.0`
  pub inputs: Vec<(Pos, u32)>,
  pub gates: Vec<Gate>,
  /// bit k of the next state
  pub outputs: Vec<Signal>,
}

impl Netlist {
  /// The inputs of a rule on these cells, cell by cell, bit 0 first: input `cell * bits + bit`.
  pub fn inputs_for(cells: &[Pos], bits: u32) -> Vec<(Pos, u32)> {
    cells.iter().flat_map(|&p| (0..bits).map(move |b| (p, b))).collect()
  }

  /// Parse the text form.
  pub fn parse(text: &str) -> Result<Netlist, String> {
    let mut inputs: Vec<(Pos, u32)> = Vec::new();
    let mut names: HashMap<String, Signal> = HashMap::new();
    let mut gates: Vec<Gate> = Vec::new();
    let mut outputs: Vec<(u32, Signal)> = Vec::new();
    for (no, raw) in text.lines().enumerate() {
      let line = raw.split('#').next().unwrap().trim();
      if line.is_empty() {
        continue;
      }
      let toks: Vec<&str> = line.split_whitespace().collect();
      let err = |m: &str| format!("line {}: {m}: {line}", no + 1);
      let sig = |n: &str, names: &HashMap<String, Signal>| match n {
        "0" | "ZERO" => Ok(Signal::Zero),
        "1" | "ONE" => Ok(Signal::One),
        _ => names.get(n).copied().ok_or_else(|| err("unknown signal")),
      };
      match toks[0] {
        "INPUT" => {
          for t in &toks[1..] {
            let split = t.find(|c: char| c.is_ascii_digit()).ok_or_else(|| err("input without a bit index"))?;
            let pos = Pos::from_name(&t[..split]).ok_or_else(|| err("unknown position"))?;
            let bit: u32 = t[split..].parse().map_err(|_| err("bad bit index"))?;
            names.insert(t.to_string(), Signal::Input(inputs.len() as u16));
            inputs.push((pos, bit));
          }
        }
        "OUTPUT" => {
          if toks.len() != 3 || !toks[1].starts_with('o') {
            return Err(err("bad OUTPUT line"));
          }
          let k: u32 = toks[1][1..].parse().map_err(|_| err("bad output index"))?;
          outputs.push((k, sig(toks[2], &names)?));
        }
        _ => {
          if toks.len() < 4 || toks[1] != "=" {
            return Err(err("expected `<name> = <OP> <a> [<b>]`"));
          }
          let g = match (toks[2], toks.len()) {
            ("NOT", 4) => Gate { op: Op::Not, a: sig(toks[3], &names)?, b: Signal::Zero },
            ("AND", 5) => Gate { op: Op::And, a: sig(toks[3], &names)?, b: sig(toks[4], &names)? },
            ("OR", 5) => Gate { op: Op::Or, a: sig(toks[3], &names)?, b: sig(toks[4], &names)? },
            ("XOR", 5) => Gate { op: Op::Xor, a: sig(toks[3], &names)?, b: sig(toks[4], &names)? },
            _ => return Err(err("unknown gate")),
          };
          names.insert(toks[0].to_string(), Signal::Gate(gates.len() as u32));
          gates.push(g);
        }
      }
    }
    outputs.sort_by_key(|o| o.0);
    if outputs.is_empty() || outputs.iter().enumerate().any(|(i, o)| o.0 != i as u32) {
      return Err("outputs must be o0 .. o(bits - 1)".into());
    }
    Ok(Netlist { bits: outputs.len() as u32, inputs, gates, outputs: outputs.into_iter().map(|o| o.1).collect() })
  }

  // ---- accessors ------------------------------------------------------------------------------

  pub fn bits(&self) -> u32 {
    self.bits
  }
  pub fn inputs(&self) -> &[(Pos, u32)] {
    &self.inputs
  }
  pub fn gates(&self) -> &[Gate] {
    &self.gates
  }
  pub fn outputs(&self) -> &[Signal] {
    &self.outputs
  }
  /// The positions the netlist reads, in the order of [`Pos::ALL`].
  pub fn positions(&self) -> Vec<Pos> {
    Pos::ALL.iter().copied().filter(|p| self.inputs.iter().any(|i| i.0 == *p)).collect()
  }
  /// Gates per operation: `(not, and, or, xor)`.
  pub fn counts(&self) -> (usize, usize, usize, usize) {
    let c = |o: Op| self.gates.iter().filter(|g| g.op == o).count();
    (c(Op::Not), c(Op::And), c(Op::Or), c(Op::Xor))
  }
  /// Longest path from an input to an output, in gates.
  pub fn depth(&self) -> usize {
    let mut d = vec![0usize; self.gates.len()];
    let at = |s: Signal, d: &[usize]| match s {
      Signal::Gate(g) => d[g as usize],
      _ => 0,
    };
    for i in 0..self.gates.len() {
      let g = self.gates[i];
      d[i] = 1 + at(g.a, &d).max(if g.op == Op::Not { 0 } else { at(g.b, &d) });
    }
    self.outputs.iter().map(|&s| at(s, &d)).max().unwrap_or(0)
  }
  /// `"367 gates (AND 165, OR 137, XOR 14, NOT 51), depth 25"`
  pub fn summary(&self) -> String {
    let (n, a, o, x) = self.counts();
    format!("{} gates (AND {a}, OR {o}, XOR {x}, NOT {n}), depth {}", self.gates.len(), self.depth())
  }

  // ---- evaluation -----------------------------------------------------------------------------

  /// Evaluate on 64 cells at once: `inputs[i]` is the plane of input i, `out[k]` receives plane k
  /// of the next states. `scratch` is reused between calls.
  #[inline]
  pub fn eval_words(&self, inputs: &[u64], scratch: &mut Vec<u64>, out: &mut [u64]) {
    scratch.clear();
    scratch.reserve(self.gates.len());
    let v = |s: Signal, sc: &Vec<u64>| -> u64 {
      match s {
        Signal::Input(i) => inputs[i as usize],
        Signal::Gate(j) => sc[j as usize],
        Signal::Zero => 0,
        Signal::One => !0,
      }
    };
    for g in &self.gates {
      let a = v(g.a, scratch);
      let r = match g.op {
        Op::Not => !a,
        Op::And => a & v(g.b, scratch),
        Op::Or => a | v(g.b, scratch),
        Op::Xor => a ^ v(g.b, scratch),
      };
      scratch.push(r);
    }
    for (k, &s) in self.outputs.iter().enumerate() {
      out[k] = v(s, scratch);
    }
  }

  /// The same with the inputs given by position: `input(pos, bit)` is the plane of bit `bit` of
  /// the cells at `pos`.
  #[inline]
  pub fn eval64(&self, input: &dyn Fn(Pos, u32) -> u64, scratch: &mut Vec<u64>, out: &mut [u64]) {
    let planes: Vec<u64> = self.inputs.iter().map(|&(p, b)| input(p, b)).collect();
    self.eval_words(&planes, scratch, out);
  }

  // ---- text -----------------------------------------------------------------------------------

  fn name_of(&self, x: Signal, index: bool, zero: &str, one: &str) -> String {
    match x {
      Signal::Input(i) => {
        let (p, k) = self.inputs[i as usize];
        if index {
          format!("{}[{k}]", p.name())
        } else {
          format!("{}{k}", p.name())
        }
      }
      Signal::Gate(g) => format!("t{g}"),
      Signal::Zero => zero.into(),
      Signal::One => one.into(),
    }
  }

  /// The text form (what [`Netlist::parse`] reads).
  pub fn to_netlist(&self) -> String {
    let mut s = String::from("INPUT");
    for (p, k) in &self.inputs {
      write!(s, " {}{k}", p.name()).unwrap();
    }
    s.push('\n');
    let n = |x| self.name_of(x, false, "0", "1");
    for (i, g) in self.gates.iter().enumerate() {
      match g.op {
        Op::Not => writeln!(s, "t{i} = NOT {}", n(g.a)).unwrap(),
        Op::And => writeln!(s, "t{i} = AND {} {}", n(g.a), n(g.b)).unwrap(),
        Op::Or => writeln!(s, "t{i} = OR {} {}", n(g.a), n(g.b)).unwrap(),
        Op::Xor => writeln!(s, "t{i} = XOR {} {}", n(g.a), n(g.b)).unwrap(),
      }
    }
    for (k, &o) in self.outputs.iter().enumerate() {
      writeln!(s, "OUTPUT o{k} {}", n(o)).unwrap();
    }
    s
  }

  fn body(&self, not: &str, zero: &str, one: &str) -> (String, Vec<String>) {
    let n = |x| self.name_of(x, true, zero, one);
    let mut s = String::new();
    for (i, g) in self.gates.iter().enumerate() {
      match g.op {
        Op::Not => writeln!(s, "  let t{i} = {not}{};", n(g.a)).unwrap(),
        Op::And => writeln!(s, "  let t{i} = {} & {};", n(g.a), n(g.b)).unwrap(),
        Op::Or => writeln!(s, "  let t{i} = {} | {};", n(g.a), n(g.b)).unwrap(),
        Op::Xor => writeln!(s, "  let t{i} = {} ^ {};", n(g.a), n(g.b)).unwrap(),
      }
    }
    (s, self.outputs.iter().map(|&o| n(o)).collect())
  }

  /// The netlist as a WGSL function `fn {name}(c: array<u32, B>, n: array<u32, B>, ...) -> array<u32, B>`
  /// with one parameter per position of `params` (bit i of element k = bit k of cell i).
  pub fn wgsl(&self, name: &str, params: &[Pos]) -> String {
    let b = self.bits;
    let sig: Vec<String> = params.iter().map(|p| format!("{}: array<u32, {b}>", p.name())).collect();
    let (body, outs) = self.body("~", "0u", "0xFFFFFFFFu");
    format!("fn {name}({}) -> array<u32, {b}> {{\n{body}  return array<u32, {b}>({});\n}}\n", sig.join(", "), outs.join(", "))
  }

  /// The same as a Rust function on `u64` words (64 cells).
  pub fn rust(&self, name: &str, params: &[Pos]) -> String {
    let b = self.bits;
    let sig: Vec<String> = params.iter().map(|p| format!("{}: [u64; {b}]", p.name())).collect();
    let (body, outs) = self.body("!", "0u64", "!0u64");
    format!("#[allow(clippy::all)]\npub fn {name}({}) -> [u64; {b}] {{\n{body}  [{}]\n}}\n", sig.join(", "), outs.join(", "))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn text_round_trip() {
    let text = "INPUT c0 c1 n0 n1\nt0 = XOR c0 n0\nt1 = NOT t0\nt2 = AND t1 c1\nt3 = OR t2 n1\nOUTPUT o0 t3\nOUTPUT o1 t0\n";
    let nl = Netlist::parse(text).unwrap();
    assert_eq!(nl.to_netlist(), text);
    assert_eq!(nl.counts(), (1, 1, 1, 1));
    assert_eq!(nl.depth(), 4);
    let (mut scratch, mut out) = (Vec::new(), [0u64; 2]);
    // c0, c1, n0, n1
    nl.eval_words(&[0b1100, 0b1010, 0b0110, 0b0001], &mut scratch, &mut out);
    assert_eq!(out[1] & 15, 0b1010);
    assert_eq!(out[0] & 15, (!0b1010u64 & 0b1010 | 0b0001) & 15);
  }
}
