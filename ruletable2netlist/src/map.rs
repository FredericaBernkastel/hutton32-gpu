//! Between the optimiser's AND/XOR networks and netlists of NOT, AND, OR and XOR gates.

use crate::net::{node, Kind, Net};
use ca_rule::{Gate, Netlist, Op, Pos, Signal};

/// Gates plus inverters of a choice of forms. An AND node is an AND gate that gives the node's
/// value (form 0), or an OR gate of the complemented inputs that gives its complement (form 1).
/// An XOR node reads what its inputs give (form 0), or the inverse of its first or second input
/// (forms 1 and 2), which complements what it gives. A node read in the polarity it does not
/// give costs one inverter, shared by all its readers.
fn cost(net: &Net, form: &[u8], pol: &mut [bool], inv: &mut [bool]) -> usize {
  inv.fill(false);
  let mut c = 0;
  for i in net.inputs + 1..net.len() {
    let n = net.nodes[i];
    c += 1;
    if n.kind == Kind::And {
      for l in [n.a, n.b] {
        // an AND gate reads the literal, an OR gate its complement
        let want = (l & 1 == 1) != (form[i] == 1);
        if want != pol[node(l)] {
          inv[node(l)] = true;
        }
      }
      pol[i] = form[i] == 1;
    } else {
      match form[i] {
        1 => inv[node(n.a)] = true,
        2 => inv[node(n.b)] = true,
        _ => {}
      }
      pol[i] = pol[node(n.a)] ^ pol[node(n.b)] ^ (form[i] != 0);
    }
  }
  for &o in &net.outputs {
    if node(o) != 0 && (o & 1 == 1) != pol[node(o)] {
      inv[node(o)] = true;
    }
  }
  c + inv.iter().filter(|&&x| x).count()
}

/// Gates of a network with its inverters, by a quick choice of forms (a first guess and two
/// descents): what the optimiser compares when it counts inverters too.
pub fn quick_cost(net: &Net) -> usize {
  let n = net.len();
  let first = net.inputs + 1;
  let mut form = vec![0u8; n];
  for i in first..n {
    let x = net.nodes[i];
    form[i] = (x.kind == Kind::And && x.a & 1 == 1 && x.b & 1 == 1) as u8;
  }
  let (mut pol, mut inv) = (vec![false; n], vec![false; n]);
  let mut cur = cost(net, &form, &mut pol, &mut inv);
  for _ in 0..2 {
    for i in first..n {
      if net.nodes[i].kind != Kind::And {
        continue;
      }
      form[i] ^= 1;
      let c = cost(net, &form, &mut pol, &mut inv);
      if c < cur {
        cur = c;
      } else {
        form[i] ^= 1;
      }
    }
  }
  cur
}

/// Map a network to gates, choosing the forms that need the fewest inverters: descents over one
/// node's form at a time, restarted from the best choice with a few forms changed, for a fixed
/// amount of work. Input
/// `cell * bits + bit` of the network is bit `bit` of the cell at `cells[cell]`.
pub fn to_netlist(net: &Net, cells: &[Pos], bits: usize) -> Netlist {
  let net = net.compact();
  let n = net.len();
  let first = net.inputs + 1;
  let mut form = vec![0u8; n];
  for i in first..n {
    let x = net.nodes[i];
    form[i] = (x.kind == Kind::And && x.a & 1 == 1 && x.b & 1 == 1) as u8;
  }
  let (mut pol, mut inv) = (vec![false; n], vec![false; n]);
  let mut cur = cost(&net, &form, &mut pol, &mut inv);
  let mut best = (cur, form.clone());
  // a fixed amount of work (node visits), so that a network always maps to the same netlist
  let work: u64 = std::env::var("R2N_MAP_WORK").ok().and_then(|v| v.parse().ok()).unwrap_or(600_000_000);
  let mut budget: u64 = work / n as u64;
  let mut rng = crate::check::Rng(n as u64);
  loop {
    if n == first {
      break;
    }
    // descend
    let mut improved = true;
    while improved {
      improved = false;
      for i in first..n {
        let choices: &[u8] = if net.nodes[i].kind == Kind::And { &[0, 1] } else { &[0, 1, 2] };
        for &f in choices {
          if f == form[i] {
            continue;
          }
          let keep = form[i];
          form[i] = f;
          let c = cost(&net, &form, &mut pol, &mut inv);
          budget = budget.saturating_sub(1);
          if c < cur {
            cur = c;
            improved = true;
          } else {
            form[i] = keep;
          }
        }
      }
      if budget == 0 {
        break;
      }
    }
    if cur < best.0 {
      best = (cur, form.clone());
    }
    if budget == 0 {
      break;
    }
    // restart near the best
    form.copy_from_slice(&best.1);
    for _ in 0..2 + rng.below(5) {
      let i = first + rng.below(n - first);
      form[i] = if net.nodes[i].kind == Kind::And { 1 - form[i] } else { rng.below(3) as u8 };
    }
    cur = cost(&net, &form, &mut pol, &mut inv);
  }
  let form = best.1;
  cost(&net, &form, &mut pol, &mut inv);

  let mut gates: Vec<Gate> = Vec::with_capacity(best.0);
  // what a node gives, and its inverter once one is made
  let mut native: Vec<Signal> = vec![Signal::Zero; n];
  let mut inverted: Vec<Option<Signal>> = vec![None; n];
  for i in 1..=net.inputs {
    native[i] = Signal::Input(i as u16 - 1);
  }
  fn read(gates: &mut Vec<Gate>, native: &[Signal], inverted: &mut [Option<Signal>], pol: &[bool], x: usize, want: bool) -> Signal {
    if want == pol[x] {
      return native[x];
    }
    if let Some(s) = inverted[x] {
      return s;
    }
    gates.push(Gate { op: Op::Not, a: native[x], b: Signal::Zero });
    let s = Signal::Gate(gates.len() as u32 - 1);
    inverted[x] = Some(s);
    s
  }
  for i in first..n {
    let x = net.nodes[i];
    let (na, nb) = (node(x.a), node(x.b));
    let g = if x.kind == Kind::And {
      let a = read(&mut gates, &native, &mut inverted, &pol, na, (x.a & 1 == 1) != (form[i] == 1));
      let b = read(&mut gates, &native, &mut inverted, &pol, nb, (x.b & 1 == 1) != (form[i] == 1));
      Gate { op: if form[i] == 1 { Op::Or } else { Op::And }, a, b }
    } else {
      let a = read(&mut gates, &native, &mut inverted, &pol, na, pol[na] != (form[i] == 1));
      let b = read(&mut gates, &native, &mut inverted, &pol, nb, pol[nb] != (form[i] == 2));
      Gate { op: Op::Xor, a, b }
    };
    gates.push(g);
    native[i] = Signal::Gate(gates.len() as u32 - 1);
  }
  let outputs = net
    .outputs
    .iter()
    .map(|&o| match o {
      0 => Signal::Zero,
      1 => Signal::One,
      _ => read(&mut gates, &native, &mut inverted, &pol, node(o), o & 1 == 1),
    })
    .collect();
  Netlist { bits: bits as u32, inputs: Netlist::inputs_for(cells, bits as u32), gates, outputs }
}

/// The network of a netlist written elsewhere, with its inputs in the order of `cells`.
pub fn to_net(nl: &Netlist, cells: &[Pos], bits: usize) -> Result<Net, String> {
  if nl.bits as usize != bits {
    return Err(format!("the netlist's states have {} bits, the rule's {bits}", nl.bits));
  }
  let mut net = Net::new(cells.len() * bits);
  let mut inputs = Vec::with_capacity(nl.inputs.len());
  for &(p, b) in &nl.inputs {
    let cell = cells.iter().position(|&q| q == p).ok_or_else(|| format!("the netlist reads {}, which is not in the rule's neighbourhood", p.name()))?;
    if b as usize >= bits {
      return Err(format!("the netlist reads bit {b} of a {bits}-bit state"));
    }
    inputs.push(net.input(cell * bits + b as usize));
  }
  let lit = |s: Signal, lits: &[u32]| match s {
    Signal::Zero => 0,
    Signal::One => 1,
    Signal::Input(i) => inputs[i as usize],
    Signal::Gate(i) => lits[i as usize],
  };
  let mut lits: Vec<u32> = Vec::with_capacity(nl.gates.len());
  for g in &nl.gates {
    let (a, b) = (lit(g.a, &lits), lit(g.b, &lits));
    lits.push(match g.op {
      Op::Not => a ^ 1,
      Op::And => net.and(a, b),
      Op::Or => net.or(a, b),
      Op::Xor => net.xor(a, b),
    });
  }
  net.outputs = nl.outputs.iter().map(|&s| lit(s, &lits)).collect();
  Ok(net.compact())
}
