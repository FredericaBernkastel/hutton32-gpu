//! Small tables compiled end to end: the netlist must give the table's next state on every
//! neighbourhood, with and without the optimiser.

use ruletable2netlist::{compile, Interp, Options, Table};

fn check(text: &str, opts: &Options) -> usize {
  let table = Table::parse("t", text).unwrap();
  let c = compile(&table, opts).unwrap();
  assert!(c.report.exhaustive);
  let cases = table.cases().unwrap();
  let (n, k, bits) = (table.states, table.cells(), table.bits());
  let interp = Interp::new(&cases, n, k);
  let (mut scratch, mut gates, mut out) = (vec![0u64; interp.words()], Vec::new(), vec![0u64; bits]);
  let mut nb = vec![0u8; k];
  for idx in 0..(n as u64).pow(k as u32) {
    let mut i = idx;
    for c in nb.iter_mut().rev() {
      *c = (i % n as u64) as u8;
      i /= n as u64;
    }
    let inputs: Vec<u64> = (0..k * bits).map(|j| (nb[j / bits] >> (j % bits) & 1) as u64).collect();
    c.netlist.eval_words(&inputs, &mut gates, &mut out);
    let got: u8 = (0..bits).map(|b| ((out[b] & 1) as u8) << b).sum();
    assert_eq!(got, interp.next(&nb, &mut scratch), "neighbourhood {nb:?}");
  }
  c.netlist.gates.len()
}

const SIGNALS: &str = "n_states:5\nneighborhood:vonNeumann\nsymmetries:rotate4\nvar a={0,1,2,3,4}\nvar b={0,1,2,3,4}\nvar c={0,1,2,3,4}\nvar h={3,4}\n\
0,h,a,b,c,1\n1,a,b,c,h,2\n2,a,a,b,c,a\n3,0,0,0,0,4\n4,a,b,c,a,0\n";

#[test]
fn raw_and_optimised_networks_equal_the_table() {
  let raw = check(SIGNALS, &Options { raw: true, ..Options::default() });
  let small = check(SIGNALS, &Options { seconds: 2.0, ..Options::default() });
  assert!(small <= raw, "{small} gates after optimisation, {raw} before");
}

#[test]
fn moore_permute() {
  // Life as a table: counts of live neighbours through `permute`
  let life = "n_states:2\nneighborhood:Moore\nsymmetries:permute\n0,1,1,1,0,0,0,0,0,1\n1,1,1,0,0,0,0,0,0,1\n1,1,1,1,0,0,0,0,0,1\n1,0,0,0,0,0,0,0,0,0\n1,1,0,0,0,0,0,0,0,0\n\
1,1,1,1,1,0,0,0,0,0\n1,1,1,1,1,1,0,0,0,0\n1,1,1,1,1,1,1,0,0,0\n1,1,1,1,1,1,1,1,0,0\n1,1,1,1,1,1,1,1,1,0\n";
  let gates = check(life, &Options { seconds: 3.0, ..Options::default() });
  assert!(gates < 400, "{gates} gates for Life");
}

/// Arrows that pass an excitation on: states 1..4 point east, north, west, south, 5..8 are the
/// same excited. An arrow is excited in the next generation if an excited arrow points at it
/// from a cell it does not point at. Written out for every direction, without symmetries.
const ARROWS: &str = "n_states:9\nneighborhood:vonNeumann\nsymmetries:none\n\
var a={0,1,2,3,4,5,6,7,8}\nvar b={0,1,2,3,4,5,6,7,8}\nvar c={0,1,2,3,4,5,6,7,8}\nvar d={0,1,2,3,4,5,6,7,8}\n\
var E={1,5}\nvar N={2,6}\nvar W={3,7}\nvar S={4,8}\n\
E,a,b,c,5,5\nE,8,b,c,d,5\nE,a,b,6,d,5\nE,a,b,c,d,1\n\
N,a,b,6,d,6\nN,a,7,c,d,6\nN,a,b,c,5,6\nN,a,b,c,d,2\n\
W,a,7,c,d,7\nW,8,b,c,d,7\nW,a,b,6,d,7\nW,a,b,c,d,3\n\
S,8,b,c,d,8\nS,a,7,c,d,8\nS,a,b,c,5,8\nS,a,b,c,d,4\n";

#[test]
fn symmetry_is_found_and_used() {
  use ruletable2netlist::{symmetry, Space, Style};
  let table = Table::parse("arrows", ARROWS).unwrap();
  let cases = table.cases().unwrap();
  let space = Space { n: table.states, k: table.cells(), bits: table.bits() };
  let interp = Interp::new(&cases, space.n, space.k);
  let sym = symmetry::find(space, &interp, 1 << 25, 2).expect("a quarter turn");
  assert_eq!(sym.order, 4);
  assert_eq!(sym.all_oriented().len(), 8);
  // every orientation holds one quiescent and one excited arrow that point the same way
  for o in &sym.oriented {
    let v: Vec<u8> = o.iter().collect();
    assert_eq!(v.len(), 2);
    assert_eq!(v[0] + 4, v[1], "{v:?}");
  }
  for style in [Style::Fold, Style::Factor, Style::Cover] {
    check(ARROWS, &Options { raw: true, style: Some(style), ..Options::default() });
  }
  let folded = check(ARROWS, &Options { seconds: 2.0, ..Options::default() });
  assert!(folded < 60, "{folded} gates");
}
