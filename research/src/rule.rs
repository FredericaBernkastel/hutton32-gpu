//! CPU port of `archive/src/kernel/rules/hutton32{,b}.rule.wgsl`, in two versions.
//!
//! * `transition` follows the files as they are now. They equal the canonical rules on all 2^25
//!   neighbourhoods: Tim Hutton's `Hutton32.table` (which Golly's built-in Hutton32 reproduces)
//!   and `Hutton32b.rule` of the Hutton32-Computer repository.
//! * `transition_v1` follows the files as archived in commit 256ef91, before the corrections of
//!   paper.md section 10. It is kept so that the old behaviour stays reproducible.
//!
//! Both are line-by-line transliterations. WGSL `u32` arithmetic wraps, so every subtraction here
//! is `wrapping_sub`. Each is validated against its WGSL executed on the GPU (`lut` subcommand),
//! and the current one against the Golly tables as well, so the LUT, not this file, is the
//! ground truth.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Variant {
  Hutton32,
  Hutton32b,
}

impl Variant {
  pub fn name(self) -> &'static str {
    match self {
      Variant::Hutton32 => "hutton32",
      Variant::Hutton32b => "hutton32b",
    }
  }
  pub fn parse(s: &str) -> Option<Self> {
    match s {
      "hutton32" => Some(Variant::Hutton32),
      "hutton32b" => Some(Variant::Hutton32b),
      _ => None,
    }
  }
}

#[inline] fn is_ots(c: u32) -> bool { c >= 9 && c <= 16 }
#[inline] fn is_sts(c: u32) -> bool { c >= 17 && c <= 24 }
#[inline] fn is_ts(c: u32) -> bool { is_ots(c) || is_sts(c) }
#[inline] fn is_sensitized(c: u32) -> bool { c >= 1 && c <= 8 }
#[inline] fn is_east(c: u32) -> bool { c == 9 || c == 13 || c == 17 || c == 21 }
#[inline] fn is_north(c: u32) -> bool { c == 10 || c == 14 || c == 18 || c == 22 }
#[inline] fn is_west(c: u32) -> bool { c == 11 || c == 15 || c == 19 || c == 23 }
#[inline] fn is_south(c: u32) -> bool { c == 12 || c == 16 || c == 20 || c == 24 }
#[inline] fn is_excited(c: u32) -> bool { (c >= 13 && c <= 16) || (c >= 21 && c <= 24) }

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
  (n == 16 && !(c == 14 || c == 10))
    || (s == 14 && !(c == 16 || c == 12))
    || (e == 15 && !(c == 13 || c == 9))
    || (w == 13 && !(c == 15 || c == 11))
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
  (n == 24 && !(c == 22 || c == 18))
    || (s == 22 && !(c == 24 || c == 20))
    || (e == 23 && !(c == 21 || c == 17))
    || (w == 21 && !(c == 23 || c == 19))
}
#[inline]
fn all_inputs_on(n: u32, s: u32, e: u32, w: u32) -> bool {
  !(n == 12 || s == 10 || e == 11 || w == 9) && (n == 16 || s == 14 || e == 15 || w == 13)
}
#[inline]
fn is_crossing(n: u32, s: u32, e: u32, w: u32) -> bool {
  let n_inputs = is_south(n) as u32 + is_east(w) as u32 + is_west(e) as u32 + is_north(s) as u32;
  let n_outputs = (is_ts(n) && !is_south(n)) as u32
    + (is_ts(w) && !is_east(w)) as u32
    + (is_ts(e) && !is_west(e)) as u32
    + (is_ts(s) && !is_north(s)) as u32;
  n_inputs == 2 && n_outputs == 2
}
/// `ots_only`: the two inputs of the XNOR gate must be ordinary transmission states (the table,
/// and the current WGSL); v1 accepted special ones too.
#[inline]
fn xnor_crossing(n: u32, s: u32, e: u32, w: u32, ots_only: bool) -> bool {
  let vertical = !ots_only || (is_ots(n) && is_ots(s));
  let horizontal = !ots_only || (is_ots(w) && is_ots(e));
  (vertical && is_south(n) && is_north(s) && (is_west(w) || is_north(w) || is_south(w)) && (is_east(e) || is_north(e) || is_south(e)))
    || (horizontal && is_east(w) && is_west(e) && (is_north(n) || is_west(n) || is_east(n)) && (is_south(s) || is_west(s) || is_east(s)))
}
#[inline]
fn xnor_output(n: u32, s: u32, e: u32, w: u32) -> bool {
  (e == 11 && w == 9) || (e == 15 && w == 13) || (n == 12 && s == 10) || (n == 16 && s == 14)
}
#[inline]
fn quiesce(c: u32) -> u32 {
  if is_excited(c) { c - 4 } else if c >= 26 && c <= 31 { 25 } else { c }
}

/// The transition function of the current (canonical) rule files. Arguments follow the WGSL
/// signature `(c, n, s, e, w)`.
pub fn transition(v: Variant, c: u32, n: u32, s: u32, e: u32, w: u32) -> u32 {
  step(v, true, c, n, s, e, w)
}

/// The rule files as archived in commit 256ef91. Differs from `transition` in three places:
/// * a sensitised cell 5, 7 or 8 with no excited arrow pointing at it: `input()` is 0 and
///   `(0 - 9 + k) % 4` wraps in `u32` (-> 11, 19, 12). Golly's C++ (`jvnalgo.cpp`) computes it in
///   `int`, where the remainder is negative (-> 7, 15, 8); the current files return those;
/// * Hutton32b: the XNOR gate accepted special transmission states as inputs;
/// * Hutton32b: a crossing 29-31 reverted to 25 whenever `!is_crossing`, before the test for an
///   incoming excited STS. Now: only next to a ground cell, after that test.
pub fn transition_v1(v: Variant, c: u32, n: u32, s: u32, e: u32, w: u32) -> u32 {
  step(v, false, c, n, s, e, w)
}

fn step(v: Variant, canon: bool, c: u32, n: u32, s: u32, e: u32, w: u32) -> u32 {
  let b = v == Variant::Hutton32b;
  // ((x - 9 + k) % 4) + base
  let rot = |x: u32, k: u32, base: u32| -> u32 { (x.wrapping_sub(9).wrapping_add(k) % 4) + base };
  // 9 + dir(x)
  let fwd = |x: u32| -> u32 { 9 + dir(x) };
  // `if(input(n,s,e,w)==0u) {return ..;}` of the current files
  let no_input = canon && input(n, s, e, w) == 0;
  // `all_inputs_on(..) || (xnor_crossing(..) && xnor_output(..))` in the 26/27/28 branches
  let conf_on =
    |n: u32, s: u32, e: u32, w: u32| all_inputs_on(n, s, e, w) || (b && xnor_crossing(n, s, e, w, canon) && xnor_output(n, s, e, w));

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
    if !excited_ots_arrow_to_us(c, n, s, e, w) {
      8
    } else {
      rot(quiesce(input(n, s, e, w)), 2, 9)
    }
  } else if c == 5 {
    if !excited_ots_arrow_to_us(c, n, s, e, w) {
      if no_input { 7 } else { rot(quiesce(input(n, s, e, w)), 3, 9) }
    } else {
      quiesce(input(n, s, e, w)).wrapping_add(8)
    }
  } else if c == 6 {
    if !excited_ots_arrow_to_us(c, n, s, e, w) {
      rot(quiesce(input(n, s, e, w)), 1, 17)
    } else {
      rot(quiesce(input(n, s, e, w)), 2, 17)
    }
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
    } else if b && xnor_crossing(n, s, e, w, canon) {
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
    if b && !canon && !is_crossing(n, s, e, w) {
      25
    } else if excited_sts_arrow_to_us(c, n, s, e, w) {
      0
    } else if b && canon && (n == 0 || s == 0 || e == 0 || w == 0) {
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

/// LUT index layout, identical to `compute.wgsl`: `w | s<<5 | e<<10 | n<<15 | c<<20`.
#[inline(always)]
pub fn lut_index(c: u32, n: u32, s: u32, e: u32, w: u32) -> usize {
  (w | s << 5 | e << 10 | n << 15 | c << 20) as usize
}

pub const LUT_SIZE: usize = 1 << 25;

/// Build the 2^25-entry byte LUT of the current (canonical) rule.
pub fn build_lut(v: Variant) -> Vec<u8> {
  build_lut_of(v, true)
}
/// The LUT of the rule files as archived in commit 256ef91.
pub fn build_lut_v1(v: Variant) -> Vec<u8> {
  build_lut_of(v, false)
}
fn build_lut_of(v: Variant, canon: bool) -> Vec<u8> {
  let mut lut = vec![0u8; LUT_SIZE];
  for (i, out) in lut.iter_mut().enumerate() {
    let i = i as u32;
    let (w, s, e, n, c) = (i & 31, (i >> 5) & 31, (i >> 10) & 31, (i >> 15) & 31, (i >> 20) & 31);
    let r = step(v, canon, c, n, s, e, w);
    assert!(r < 32, "rule produced state {r} >= 32 for c={c} n={n} s={s} e={e} w={w}");
    *out = r as u8;
  }
  lut
}
