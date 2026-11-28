//! Golly's macrocell format (`[M2]`). Both of Golly's flavours are read: the multistate one (a
//! level-1 node lists four cell states) and the two-state one of the Life-like algorithms (level-3
//! leaves written as 8×8 bitmaps of `.`, `*` and `$`). Writing is [`crate::cpu::Engine::write_macrocell`].

use crate::{Error, Result};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum McNode {
  /// `(level, [nw, ne, sw, se])`: a level-1 node carries four cell states, a higher one node
  /// indices (1-based, 0 = empty)
  Inner(u32, [u64; 4]),
  /// a two-state 8×8 leaf (level 3): bit `8 * y + x` is the cell at `(x, y)`
  Leaf8(u64),
}

#[derive(Clone, Debug, Default)]
pub struct Macrocell {
  pub rule: String,
  /// the generation the file was saved at (`#G`), if any
  pub generation: Option<u64>,
  pub nodes: Vec<McNode>,
}

impl Macrocell {
  pub fn parse(text: &str) -> Result<Macrocell> {
    let mut rule = String::new();
    let mut generation = None;
    let mut nodes = Vec::new();
    for line in text.lines() {
      let t = line.trim();
      if t.is_empty() || t.starts_with('[') {
        continue;
      }
      if let Some(rest) = t.strip_prefix("#R") {
        rule = rest.trim().to_string();
        continue;
      }
      if let Some(rest) = t.strip_prefix("#G") {
        generation = rest.trim().parse().ok();
        continue;
      }
      if t.starts_with('#') {
        continue;
      }
      if t.starts_with(['.', '*', '$']) {
        let (mut x, mut y, mut bits) = (0u32, 0u32, 0u64);
        for ch in t.chars() {
          match ch {
            '.' => x += 1,
            '*' => {
              if x >= 8 || y >= 8 {
                return Err(Error::Pattern(format!("macrocell leaf larger than 8x8: {t}")));
              }
              bits |= 1 << (8 * y + x);
              x += 1;
            }
            '$' => {
              y += 1;
              x = 0;
            }
            _ => return Err(Error::Pattern(format!("bad macrocell leaf: {t}"))),
          }
        }
        nodes.push(McNode::Leaf8(bits));
        continue;
      }
      let v: Vec<u64> = t.split_whitespace().take(5).map(|x| x.parse()).collect::<std::result::Result<_, _>>().map_err(|_| Error::Pattern(format!("bad macrocell line: {t}")))?;
      if v.len() != 5 {
        return Err(Error::Pattern(format!("bad macrocell line: {t}")));
      }
      for &c in &v[1..] {
        if v[0] > 1 && c as usize > nodes.len() {
          return Err(Error::Pattern(format!("macrocell node refers forward: {t}")));
        }
      }
      nodes.push(McNode::Inner(v[0] as u32, [v[1], v[2], v[3], v[4]]));
    }
    // Golly reads a two-state macrocell without a rule line as Life
    if rule.is_empty() && nodes.iter().any(|n| matches!(n, McNode::Leaf8(_))) {
      rule = "B3/S23".to_string();
    }
    if nodes.is_empty() {
      return Err(Error::Pattern("empty macrocell".into()));
    }
    Ok(Macrocell { rule, generation, nodes })
  }
}
