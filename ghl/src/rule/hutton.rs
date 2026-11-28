//! The two canonical 32-state rules of the Hutton32 family on the von Neumann neighbourhood:
//!
//! * **Hutton32**: Tim Hutton's `Hutton32.table` (Golly's rule table repository), which Golly's
//!   built-in Hutton32 computes as well;
//! * **Hutton32b**: `Hutton32b.rule` of the Hutton32-Computer repository: a 3×3 XNOR gate and two
//!   added rule sets.
//!
//! Both files are embedded (`generated/`). The tables are built from the transition function below,
//! a transliteration of the harness's `rule.rs`, which equals both files on all 2^25
//! neighbourhoods; a test parses the embedded files with the `@TABLE` loader and compares. Both
//! rules carry a gate network: Hutton32b its 367-gate hand-optimised circuit
//! (`generated/hutton32b.netlist`), Hutton32 the 597-gate circuit that `ruletable2netlist` compiled
//! from its table (`generated/hutton32.netlist`); a test checks each on all 2^25 neighbourhoods.

use std::sync::{Arc, OnceLock};

use super::{Inner, Inversion, Network, NetworkKind, Neighbourhood, Rule, Source};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Variant {
  Hutton32,
  Hutton32b,
}

/// The embedded canonical files.
pub const HUTTON32_TABLE: &str = include_str!("generated/Hutton32.table");
pub const HUTTON32B_RULE: &str = include_str!("generated/Hutton32b.rule");
pub const HUTTON32B_NETLIST: &str = include_str!("generated/hutton32b.netlist");
pub const HUTTON32_NETLIST: &str = include_str!("generated/hutton32.netlist");

/// The colours of the archived application (and of Hutton32b.rule's `@COLORS`).
pub const HUTTON32_COLORS: [[u8; 3]; 32] = [
  [0, 0, 0], [255, 0, 0], [255, 125, 0], [255, 150, 25], [255, 175, 50], [255, 200, 75], [255, 225, 100],
  [255, 250, 125], [251, 255, 0], [89, 89, 255], [106, 106, 255], [122, 122, 255], [139, 139, 255],
  [27, 176, 27], [36, 200, 36], [73, 255, 73], [106, 255, 106], [235, 36, 36], [255, 56, 56], [255, 73, 73],
  [255, 89, 89], [185, 56, 255], [191, 73, 255], [197, 89, 255], [203, 106, 255], [0, 255, 128],
  [255, 128, 64], [255, 255, 128], [33, 215, 215], [27, 176, 176], [24, 156, 156], [21, 137, 137],
];

/// The rule, built once per process.
pub fn rule(v: Variant) -> Rule {
  static H32: OnceLock<Rule> = OnceLock::new();
  static H32B: OnceLock<Rule> = OnceLock::new();
  let cell = match v {
    Variant::Hutton32 => &H32,
    Variant::Hutton32b => &H32B,
  };
  cell
    .get_or_init(|| {
      let network = match v {
        Variant::Hutton32 => Some(Arc::new(Network::parse(HUTTON32_NETLIST).expect("embedded netlist").with_kind(NetworkKind::Hutton32))),
        Variant::Hutton32b => Some(Arc::new(Network::parse(HUTTON32B_NETLIST).expect("embedded netlist"))),
      };
      Rule(Arc::new(Inner {
        name: match v {
          Variant::Hutton32 => "Hutton32".into(),
          Variant::Hutton32b => "Hutton32b".into(),
        },
        nbhd: Neighbourhood::VonNeumann,
        states: 32,
        bits: 5,
        table: Some(build_table(v)),
        tree: None,
        life_like: None,
        network,
        odd: None,
        inversion: Inversion::Never,
        palette: HUTTON32_COLORS.to_vec(),
        source: Source::Canonical,
      }))
    })
    .clone()
}

#[inline] fn is_ots(c: u32) -> bool { (9..=16).contains(&c) }
#[inline] fn is_sts(c: u32) -> bool { (17..=24).contains(&c) }
#[inline] fn is_ts(c: u32) -> bool { is_ots(c) || is_sts(c) }
#[inline] fn is_sensitized(c: u32) -> bool { (1..=8).contains(&c) }
#[inline] fn is_east(c: u32) -> bool { c == 9 || c == 13 || c == 17 || c == 21 }
#[inline] fn is_north(c: u32) -> bool { c == 10 || c == 14 || c == 18 || c == 22 }
#[inline] fn is_west(c: u32) -> bool { c == 11 || c == 15 || c == 19 || c == 23 }
#[inline] fn is_south(c: u32) -> bool { c == 12 || c == 16 || c == 20 || c == 24 }
#[inline] fn is_excited(c: u32) -> bool { (13..=16).contains(&c) || (21..=24).contains(&c) }
#[inline] fn dir(c: u32) -> u32 { c.wrapping_sub(9) % 4 }

#[inline]
fn output(c: u32, n: u32, s: u32, e: u32, w: u32) -> u32 {
  if is_east(c) { e } else if is_north(c) { n } else if is_west(c) { w } else if is_south(c) { s } else { 0 }
}
#[inline]
fn input(n: u32, s: u32, e: u32, w: u32) -> u32 {
  if is_east(w) && is_excited(w) { w }
  else if is_north(s) && is_excited(s) { s }
  else if is_west(e) && is_excited(e) { e }
  else if is_south(n) && is_excited(n) { n }
  else { 0 }
}
#[inline]
fn output_will_become_ots(c: u32, n: u32, s: u32, e: u32, w: u32) -> bool {
  let o = output(c, n, s, e, w);
  o == 8 || (o == 4 && is_excited(c)) || (o == 5 && !is_excited(c))
}
#[inline]
fn output_will_become_confluent(c: u32, n: u32, s: u32, e: u32, w: u32) -> bool {
  output(c, n, s, e, w) == 7 && is_excited(c)
}
#[inline]
fn output_will_become_sensitized(c: u32, n: u32, s: u32, e: u32, w: u32) -> bool {
  let out = output(c, n, s, e, w);
  (out == 0 && is_excited(c)) || out == 1 || out == 2 || out == 3 || (out == 4 && !is_ots(c))
}
#[inline]
fn excited_ots_to_us(c: u32, n: u32, s: u32, e: u32, w: u32) -> bool {
  ((n == 16 || n == 27 || n == 28 || n == 30 || n == 31) && !(c == 14 || c == 10))
    || ((s == 14 || s == 27 || s == 28 || s == 30 || s == 31) && !(c == 16 || c == 12))
    || ((e == 15 || e == 27 || e == 28 || e == 29 || e == 31) && !(c == 13 || c == 9))
    || ((w == 13 || w == 27 || w == 28 || w == 29 || w == 31) && !(c == 15 || c == 11))
}
#[inline]
fn excited_ots_arrow_to_us(c: u32, n: u32, s: u32, e: u32, w: u32) -> bool {
  (n == 16 && !(c == 14 || c == 10)) || (s == 14 && !(c == 16 || c == 12)) || (e == 15 && !(c == 13 || c == 9)) || (w == 13 && !(c == 15 || c == 11))
}
#[inline]
fn ots_arrow_to_us(n: u32, s: u32, e: u32, w: u32) -> bool {
  (is_ots(n) && is_south(n)) || (is_ots(s) && is_north(s)) || (is_ots(e) && is_west(e)) || (is_ots(w) && is_east(w))
}
#[inline]
fn excited_sts_to_us(c: u32, n: u32, s: u32, e: u32, w: u32) -> bool {
  ((n == 24 || n == 27 || n == 28 || n == 30 || n == 31) && !(c == 22 || c == 18))
    || ((s == 22 || s == 27 || s == 28 || s == 30 || s == 31) && !(c == 24 || c == 20))
    || ((e == 23 || e == 27 || e == 28 || e == 29 || e == 31) && !(c == 21 || c == 17))
    || ((w == 21 || w == 27 || w == 28 || w == 29 || w == 31) && !(c == 23 || c == 19))
}
#[inline]
fn excited_sts_arrow_to_us(c: u32, n: u32, s: u32, e: u32, w: u32) -> bool {
  (n == 24 && !(c == 22 || c == 18)) || (s == 22 && !(c == 24 || c == 20)) || (e == 23 && !(c == 21 || c == 17)) || (w == 21 && !(c == 23 || c == 19))
}
#[inline]
fn all_inputs_on(n: u32, s: u32, e: u32, w: u32) -> bool {
  !(n == 12 || s == 10 || e == 11 || w == 9) && (n == 16 || s == 14 || e == 15 || w == 13)
}
#[inline]
fn is_crossing(n: u32, s: u32, e: u32, w: u32) -> bool {
  let n_inputs = is_south(n) as u32 + is_east(w) as u32 + is_west(e) as u32 + is_north(s) as u32;
  let n_outputs = (is_ts(n) && !is_south(n)) as u32 + (is_ts(w) && !is_east(w)) as u32 + (is_ts(e) && !is_west(e)) as u32 + (is_ts(s) && !is_north(s)) as u32;
  n_inputs == 2 && n_outputs == 2
}
#[inline]
fn xnor_crossing(n: u32, s: u32, e: u32, w: u32) -> bool {
  let vertical = is_ots(n) && is_ots(s);
  let horizontal = is_ots(w) && is_ots(e);
  (vertical && is_south(n) && is_north(s) && (is_west(w) || is_north(w) || is_south(w)) && (is_east(e) || is_north(e) || is_south(e)))
    || (horizontal && is_east(w) && is_west(e) && (is_north(n) || is_west(n) || is_east(n)) && (is_south(s) || is_west(s) || is_east(s)))
}
#[inline]
fn xnor_output(n: u32, s: u32, e: u32, w: u32) -> bool {
  (e == 11 && w == 9) || (e == 15 && w == 13) || (n == 12 && s == 10) || (n == 16 && s == 14)
}
#[inline]
fn quiesce(c: u32) -> u32 {
  if is_excited(c) { c - 4 } else if (26..=31).contains(&c) { 25 } else { c }
}

/// The transition function of the canonical rule files, `(c, n, s, e, w) -> c'`.
pub fn transition(v: Variant, c: u32, n: u32, s: u32, e: u32, w: u32) -> u32 {
  let b = v == Variant::Hutton32b;
  let rot = |x: u32, k: u32, base: u32| -> u32 { (x.wrapping_sub(9).wrapping_add(k) % 4) + base };
  let fwd = |x: u32| -> u32 { 9 + dir(x) };
  let no_input = input(n, s, e, w) == 0;
  let conf_on = |n: u32, s: u32, e: u32, w: u32| all_inputs_on(n, s, e, w) || (b && xnor_crossing(n, s, e, w) && xnor_output(n, s, e, w));

  if is_ots(c) {
    if excited_sts_arrow_to_us(c, n, s, e, w) {
      0
    } else if excited_ots_to_us(c, n, s, e, w) {
      let o = output(c, n, s, e, w);
      if output_will_become_ots(c, n, s, e, w) || (is_sts(o) && !is_excited(o)) {
        0
      } else if output_will_become_confluent(c, n, s, e, w) {
        1
      } else {
        quiesce(c) + 4
      }
    } else if output_will_become_confluent(c, n, s, e, w) {
      0
    } else if is_excited(c) && output_will_become_sensitized(c, n, s, e, w) {
      quiesce(c) + 12
    } else {
      quiesce(c)
    }
  } else if is_sts(c) {
    let out = output(c, n, s, e, w);
    if is_excited(c) && is_sensitized(out) && ots_arrow_to_us(n, s, e, w) {
      if output_will_become_sensitized(c, n, s, e, w) {
        if excited_ots_arrow_to_us(c, n, s, e, w) { c.wrapping_sub(8) } else { c }
      } else if excited_ots_arrow_to_us(c, n, s, e, w) {
        0
      } else {
        quiesce(c).wrapping_sub(8)
      }
    } else if is_excited(c) && out == 0 {
      let stay = if b { excited_sts_to_us(c, n, s, e, w) } else { excited_sts_arrow_to_us(c, n, s, e, w) };
      if stay { c } else { quiesce(c) }
    } else if excited_ots_arrow_to_us(c, n, s, e, w) {
      0
    } else if excited_sts_to_us(c, n, s, e, w) {
      quiesce(c) + 4
    } else {
      quiesce(c)
    }
  } else if c == 0 {
    if excited_ots_arrow_to_us(c, n, s, e, w) {
      1
    } else if excited_sts_arrow_to_us(c, n, s, e, w) {
      quiesce(input(n, s, e, w)).wrapping_sub(8)
    } else {
      c
    }
  } else if c == 1 {
    if !excited_ots_arrow_to_us(c, n, s, e, w) { 2 } else { 3 }
  } else if c == 2 {
    if !excited_ots_arrow_to_us(c, n, s, e, w) { 4 } else { 5 }
  } else if c == 3 {
    if !excited_ots_arrow_to_us(c, n, s, e, w) { 6 } else { 7 }
  } else if c == 4 {
    if !excited_ots_arrow_to_us(c, n, s, e, w) { 8 } else { rot(quiesce(input(n, s, e, w)), 2, 9) }
  } else if c == 5 {
    if !excited_ots_arrow_to_us(c, n, s, e, w) {
      if no_input { 7 } else { rot(quiesce(input(n, s, e, w)), 3, 9) }
    } else {
      quiesce(input(n, s, e, w)).wrapping_add(8)
    }
  } else if c == 6 {
    if !excited_ots_arrow_to_us(c, n, s, e, w) { rot(quiesce(input(n, s, e, w)), 1, 17) } else { rot(quiesce(input(n, s, e, w)), 2, 17) }
  } else if c == 7 {
    if !excited_ots_arrow_to_us(c, n, s, e, w) {
      if no_input { 15 } else { rot(quiesce(input(n, s, e, w)), 3, 17) }
    } else {
      25
    }
  } else if c == 8 {
    if !excited_ots_arrow_to_us(c, n, s, e, w) {
      if no_input { 8 } else { fwd(input(n, s, e, w)) }
    } else {
      fwd(input(n, s, e, w).wrapping_add(1))
    }
  } else if c == 25 {
    if excited_sts_arrow_to_us(c, n, s, e, w) {
      0
    } else if b && xnor_crossing(n, s, e, w) {
      if xnor_output(n, s, e, w) { 26 } else { 25 }
    } else if is_crossing(n, s, e, w) {
      if (n == 16 || s == 14) && (e == 15 || w == 13) {
        31
      } else if n == 16 || s == 14 {
        30
      } else if e == 15 || w == 13 {
        29
      } else {
        25
      }
    } else if all_inputs_on(n, s, e, w) {
      26
    } else {
      25
    }
  } else if c == 26 {
    if excited_sts_arrow_to_us(c, n, s, e, w) { 0 } else if conf_on(n, s, e, w) { 28 } else { 27 }
  } else if c == 27 {
    if excited_sts_arrow_to_us(c, n, s, e, w) { 0 } else if conf_on(n, s, e, w) { 26 } else { 25 }
  } else if c == 28 {
    if excited_sts_arrow_to_us(c, n, s, e, w) { 0 } else if conf_on(n, s, e, w) { 28 } else { 27 }
  } else if c == 29 || c == 30 || c == 31 {
    if excited_sts_arrow_to_us(c, n, s, e, w) {
      0
    } else if b && (n == 0 || s == 0 || e == 0 || w == 0) {
      25
    } else if (n == 16 || s == 14) && (e == 15 || w == 13) {
      31
    } else if n == 16 || s == 14 {
      30
    } else if e == 15 || w == 13 {
      29
    } else {
      25
    }
  } else {
    c
  }
}

/// The 2^25-entry table in the index layout `w | s << 5 | e << 10 | n << 15 | c << 20`.
pub fn build_table(v: Variant) -> Vec<u8> {
  let mut lut = vec![0u8; 1 << 25];
  // rows of 32 entries share (c, n, s, e): split the work over the available threads
  let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(16);
  let chunk = lut.len() / threads;
  std::thread::scope(|sc| {
    for (t, part) in lut.chunks_mut(chunk).enumerate() {
      sc.spawn(move || {
        let base = t * chunk;
        for (j, out) in part.iter_mut().enumerate() {
          let i = (base + j) as u32;
          let (w, s, e, n, c) = (i & 31, (i >> 5) & 31, (i >> 10) & 31, (i >> 15) & 31, (i >> 20) & 31);
          let r = transition(v, c, n, s, e, w);
          debug_assert!(r < 32);
          *out = r as u8;
        }
      });
    }
  });
  lut
}
