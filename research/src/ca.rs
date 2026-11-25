//! A cellular automaton rule as the harness sees it: a radius-1 neighbourhood, a number of states
//! and the transition function as a table over neighbourhood indices. Every rule of Golly's
//! HashLife family has this form (Moore, von Neumann and hexagonal neighbourhoods, up to 256
//! states); the table only has to fit in memory. The harness builds tables for the Life-like rules
//! (B/S notation) and for the Hutton32 family (von Neumann, 32 states, `rulespec`).
//!
//! Index layout: fields of `bits` bits in the order w, s, e, n, c, nw, ne, sw, se (field 0 is the
//! least significant). The von Neumann neighbourhood uses the first five fields, the hexagonal one
//! (Golly's: Moore without NE and SW) the fields w, s, e, n, c, nw, se, and Moore all nine. With
//! 5 bits per field the von Neumann layout is `rule::lut_index`, the one of the GPU kernels.

use std::ops::Deref;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Nbhd {
  VonNeumann,
  Hex,
  Moore,
}

impl Nbhd {
  /// Cells in the neighbourhood, the centre included.
  pub fn cells(self) -> u32 {
    match self {
      Nbhd::VonNeumann => 5,
      Nbhd::Hex => 7,
      Nbhd::Moore => 9,
    }
  }
  pub fn name(self) -> &'static str {
    match self {
      Nbhd::VonNeumann => "von Neumann",
      Nbhd::Hex => "hexagonal",
      Nbhd::Moore => "Moore",
    }
  }
}

pub struct Rule {
  pub name: String,
  pub nbhd: Nbhd,
  pub states: u32,
  /// bits per field of the table index
  pub bits: u32,
  pub table: Vec<u8>,
}

impl Deref for Rule {
  type Target = [u8];
  fn deref(&self) -> &[u8] {
    &self.table
  }
}

impl Rule {
  pub fn from_table(name: &str, nbhd: Nbhd, states: u32, bits: u32, table: Vec<u8>) -> Rule {
    assert_eq!(table.len(), 1usize << (bits * nbhd.cells()), "table size does not match the neighbourhood");
    Rule { name: name.to_string(), nbhd, states, bits, table }
  }
  /// A placeholder for engines that only convert patterns and never step.
  pub fn dummy() -> Rule {
    Rule::life_like(8, 12, Nbhd::Moore)
  }
  /// The 32-state von Neumann layout that the Hutton32 kernels on the GPU assume.
  pub fn is_vn32(&self) -> bool {
    self.nbhd == Nbhd::VonNeumann && self.bits == 5
  }
  /// Two states, outer-totalistic (Life-like).
  pub fn life_like_masks(&self) -> Option<(u16, u16)> {
    parse_life_like(&self.name).map(|(b, s, _)| (b, s))
  }

  /// A Life-like rule: bit k of `birth` / `survive` = born / survives with k live neighbours.
  pub fn life_like(birth: u16, survive: u16, nbhd: Nbhd) -> Rule {
    assert!(birth & 1 == 0, "B0 rules need Golly's strobing emulation, which the harness does not implement");
    let cells = nbhd.cells();
    let mut table = vec![0u8; 1 << cells];
    for (i, o) in table.iter_mut().enumerate() {
      let centre = (i >> 4) & 1;
      let others = ((i & 0xF) | ((i >> 5) << 4)).count_ones();
      *o = if centre == 1 { (survive >> others) & 1 } else { (birth >> others) & 1 } as u8;
    }
    Rule { name: life_like_name(birth, survive, nbhd), nbhd, states: 2, bits: 1, table }
  }
}

pub fn life_like_name(birth: u16, survive: u16, nbhd: Nbhd) -> String {
  let digits = |m: u16| (0..=8).filter(|k| m >> k & 1 == 1).map(|k| char::from(b'0' + k as u8)).collect::<String>();
  let suffix = match nbhd {
    Nbhd::Moore => "",
    Nbhd::Hex => "H",
    Nbhd::VonNeumann => "V",
  };
  format!("B{}/S{}{}", digits(birth), digits(survive), suffix)
}

/// Golly's B/S notation for outer-totalistic two-state rules: `B3/S23`, `b3s23`, `B2/S34H`
/// (hexagonal), `B2/SV` (von Neumann), and `Life`. Isotropic non-totalistic and MAP rules are not
/// recognised.
pub fn parse_life_like(s: &str) -> Option<(u16, u16, Nbhd)> {
  let t = s.trim().to_ascii_lowercase();
  if t == "life" || t == "conway" {
    return Some((1 << 3, 1 << 2 | 1 << 3, Nbhd::Moore));
  }
  let (t, nbhd) = if let Some(r) = t.strip_suffix('h') {
    (r, Nbhd::Hex)
  } else if let Some(r) = t.strip_suffix('v') {
    (r, Nbhd::VonNeumann)
  } else {
    (t.as_str(), Nbhd::Moore)
  };
  let t: String = t.chars().filter(|&c| c != '/').collect();
  let (bi, si) = (t.find('b')?, t.find('s')?);
  if bi != 0 && si != 0 {
    return None;
  }
  let (bs, ss) = if bi < si { (&t[bi + 1..si], &t[si + 1..]) } else { (&t[bi + 1..], &t[si + 1..bi]) };
  let max = nbhd.cells() - 1;
  let mask = |x: &str| -> Option<u16> {
    let mut m = 0u16;
    for ch in x.chars() {
      let d = ch.to_digit(10)?;
      if d > max {
        return None;
      }
      m |= 1 << d;
    }
    Some(m)
  };
  Some((mask(bs)?, mask(ss)?, nbhd))
}

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn parse() {
    assert_eq!(parse_life_like("B3/S23"), Some((8, 12, Nbhd::Moore)));
    assert_eq!(parse_life_like("b3s23"), Some((8, 12, Nbhd::Moore)));
    assert_eq!(parse_life_like("Life"), Some((8, 12, Nbhd::Moore)));
    assert_eq!(parse_life_like("B2/S34H"), Some((4, 24, Nbhd::Hex)));
    assert_eq!(parse_life_like("Hutton32"), None);
    assert_eq!(life_like_name(8, 12, Nbhd::Moore), "B3/S23");
  }
  #[test]
  fn life_table() {
    let r = Rule::life_like(8, 12, Nbhd::Moore);
    // a live centre with two live neighbours (w, s) survives; a dead one with three is born
    assert_eq!(r[0b1_0000 | 0b11], 1);
    assert_eq!(r[0b0_0000 | 0b111], 1);
    assert_eq!(r[0b1_0000 | 0b1], 0);
    assert_eq!(r[0b0_0000 | 0b11 | 1 << 5], 1);
  }
}
