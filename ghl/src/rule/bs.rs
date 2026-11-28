//! Golly's B/S notation for outer-totalistic two-state rules.

use super::Neighbourhood;

/// Parse B/S notation: `B3/S23`, `b3s23`, `B36/S23`, `B2/S34H` (hexagonal), `B2/SV` (von Neumann),
/// the older S/B form `23/3` (survival first), and `Life`. Returns `(birth, survive, neighbourhood)`
/// with bit k set for k live neighbours. Isotropic non-totalistic and MAP rules are not recognised.
pub fn parse_bs(s: &str) -> Option<(u16, u16, Neighbourhood)> {
  let t = s.trim().to_ascii_lowercase();
  if t == "life" || t == "conway" {
    return Some((1 << 3, 1 << 2 | 1 << 3, Neighbourhood::Moore));
  }
  let (t, nbhd) = if let Some(r) = t.strip_suffix('h') {
    (r.to_string(), Neighbourhood::Hexagonal)
  } else if let Some(r) = t.strip_suffix('v') {
    (r.to_string(), Neighbourhood::VonNeumann)
  } else {
    (t.clone(), Neighbourhood::Moore)
  };
  let t = t.trim_end_matches('/').to_string();
  let max = nbhd.neighbours() as u32;
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
  if t.contains('b') || t.contains('s') {
    let t: String = t.chars().filter(|&c| c != '/').collect();
    let (bi, si) = (t.find('b')?, t.find('s')?);
    if bi != 0 && si != 0 {
      return None;
    }
    let (bs, ss) = if bi < si { (&t[bi + 1..si], &t[si + 1..]) } else { (&t[bi + 1..], &t[si + 1..bi]) };
    return Some((mask(bs)?, mask(ss)?, nbhd));
  }
  // S/B: "23/3"
  let (s_part, b_part) = t.split_once('/')?;
  if s_part.chars().all(|c| c.is_ascii_digit()) && b_part.chars().all(|c| c.is_ascii_digit()) {
    return Some((mask(b_part)?, mask(s_part)?, nbhd));
  }
  None
}

/// The canonical name of a Life-like rule: `B3/S23`, `B2/S34H`, `B2/S3V`.
pub fn bs_name(birth: u16, survive: u16, nbhd: Neighbourhood) -> String {
  let digits = |m: u16| (0..=8).filter(|k| m >> k & 1 == 1).map(|k| char::from(b'0' + k as u8)).collect::<String>();
  let suffix = match nbhd {
    Neighbourhood::Moore => "",
    Neighbourhood::Hexagonal => "H",
    Neighbourhood::VonNeumann => "V",
    Neighbourhood::OneDimensional => "",
  };
  format!("B{}/S{}{}", digits(birth), digits(survive), suffix)
}

/// The table of a Life-like rule in the index layout of [`Neighbourhood::fields`] (one bit per
/// field; the centre is the field of [`super::Pos::C`]).
pub fn life_like_table(birth: u16, survive: u16, nbhd: Neighbourhood) -> Vec<u8> {
  let cells = nbhd.cells();
  let centre = nbhd.field_of(super::Pos::C).unwrap();
  let mut table = vec![0u8; 1 << cells];
  for (i, o) in table.iter_mut().enumerate() {
    let c = (i >> centre) & 1;
    let others = (i & !(1 << centre)).count_ones();
    *o = if c == 1 { (survive >> others) & 1 } else { (birth >> others) & 1 } as u8;
  }
  table
}

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn notation() {
    assert_eq!(parse_bs("B3/S23"), Some((8, 12, Neighbourhood::Moore)));
    assert_eq!(parse_bs("b3s23"), Some((8, 12, Neighbourhood::Moore)));
    assert_eq!(parse_bs("Life"), Some((8, 12, Neighbourhood::Moore)));
    assert_eq!(parse_bs("23/3"), Some((8, 12, Neighbourhood::Moore)));
    assert_eq!(parse_bs("B2/S34H"), Some((4, 24, Neighbourhood::Hexagonal)));
    assert_eq!(parse_bs("B2/SV"), Some((4, 0, Neighbourhood::VonNeumann)));
    assert_eq!(parse_bs("Hutton32"), None);
    assert_eq!(bs_name(8, 12, Neighbourhood::Moore), "B3/S23");
  }
  #[test]
  fn table() {
    let t = life_like_table(8, 12, Neighbourhood::Moore);
    // fields w s e n c nw ne sw se: a live centre (bit 4) with w and s survives
    assert_eq!(t[0b1_0000 | 0b11], 1);
    assert_eq!(t[0b0_0000 | 0b111], 1);
    assert_eq!(t[0b1_0000 | 0b1], 0);
    assert_eq!(t[0b0_0000 | 0b11 | 1 << 5], 1);
  }
}
