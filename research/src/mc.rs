//! Golly macrocell (`[M2]`) reader and conversion into the reference engine. Both of Golly's
//! flavours are read: the multistate one (level-1 nodes carry four cell states) and the two-state
//! one of the Life-like algorithms (level-3 leaves written as 8×8 bitmaps of `.`, `*` and `$`).

use crate::hl::Engine;
use std::{fs, path::Path};

pub enum McNode {
  /// `(level, [nw, ne, sw, se])`: a level-1 node carries four cell states, a higher one node
  /// indices (1-based, 0 = empty)
  Inner(u32, [u64; 4]),
  /// a two-state 8×8 leaf (level 3): bit `8 * y + x` is the cell at `(x, y)`
  Leaf8(u64),
}

pub struct Macrocell {
  pub rule: String,
  pub nodes: Vec<McNode>,
}

pub fn read_mc(path: &Path) -> Macrocell {
  let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
  let mut rule = String::new();
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
    if t.starts_with('#') {
      continue;
    }
    if t.starts_with(['.', '*', '$']) {
      let (mut x, mut y, mut bits) = (0u32, 0u32, 0u64);
      for ch in t.chars() {
        match ch {
          '.' => x += 1,
          '*' => {
            assert!(x < 8 && y < 8, "macrocell leaf larger than 8x8: {t}");
            bits |= 1 << (8 * y + x);
            x += 1;
          }
          '$' => {
            y += 1;
            x = 0;
          }
          _ => panic!("bad macrocell leaf: {t}"),
        }
      }
      nodes.push(McNode::Leaf8(bits));
      continue;
    }
    let v: Vec<u64> = t.split_whitespace().take(5).map(|x| x.parse().expect("bad macrocell line")).collect();
    assert!(v.len() == 5, "bad macrocell line: {t}");
    nodes.push(McNode::Inner(v[0] as u32, [v[1], v[2], v[3], v[4]]));
  }
  // Golly reads a two-state macrocell without a rule line as Life
  if rule.is_empty() && nodes.iter().any(|n| matches!(n, McNode::Leaf8(_))) {
    rule = "B3/S23".to_string();
  }
  Macrocell { rule, nodes }
}

enum Conv {
  Cells(Vec<u8>),
  Id(u32),
}

impl<'a> Engine<'a> {
  /// A square of raw cells of side `2^d` as a node: raw cells below the base tile, a tile at it,
  /// a node of tiles above it.
  fn raw_to_conv(&mut self, cells: Vec<u8>, d: usize) -> Conv {
    use std::cmp::Ordering::*;
    match d.cmp(&self.a) {
      Less => Conv::Cells(cells),
      Equal => Conv::Id(self.tile(&cells)),
      Greater => {
        let side = 1usize << d;
        let h = side / 2;
        let mut ids = [0u32; 4];
        for q in 0..4 {
          let (ox, oy) = ((q & 1) * h, (q >> 1) * h);
          let mut sub = vec![0u8; h * h];
          for y in 0..h {
            sub[y * h..(y + 1) * h].copy_from_slice(&cells[(oy + y) * side + ox..(oy + y) * side + ox + h]);
          }
          let Conv::Id(id) = self.raw_to_conv(sub, d - 1) else { unreachable!() };
          ids[q] = id;
        }
        Conv::Id(self.intern(d, ids))
      }
    }
  }

  /// Convert a macrocell quadtree. Returns `(root id, level)` with `level >= a + 1`.
  pub fn from_mc(&mut self, mc: &Macrocell) -> (u32, usize) {
    let a = self.a;
    let mut conv: Vec<Conv> = Vec::with_capacity(mc.nodes.len());
    let mut last_level = 0usize;
    for node in &mc.nodes {
      let c = match *node {
        McNode::Leaf8(bits) => {
          last_level = 3;
          let cells = (0..64).map(|i| (bits >> i & 1) as u8).collect();
          self.raw_to_conv(cells, 3)
        }
        McNode::Inner(d, ch) => {
          let d = d as usize;
          last_level = d;
          if d <= a {
            // assemble raw cells of side 2^d
            let side = 1usize << d;
            let h = side / 2;
            let mut cells = vec![0u8; side * side];
            if d == 1 {
              for q in 0..4 {
                cells[(q >> 1) * 2 + (q & 1)] = ch[q] as u8;
              }
            } else {
              for q in 0..4 {
                if ch[q] == 0 {
                  continue;
                }
                let Conv::Cells(sub) = &conv[ch[q] as usize - 1] else { panic!("level mismatch in macrocell") };
                let (ox, oy) = ((q & 1) * h, (q >> 1) * h);
                for y in 0..h {
                  cells[(oy + y) * side + ox..(oy + y) * side + ox + h].copy_from_slice(&sub[y * h..(y + 1) * h]);
                }
              }
            }
            self.raw_to_conv(cells, d)
          } else {
            let z = self.empty(d - 1);
            let mut ids = [z; 4];
            for q in 0..4 {
              if ch[q] != 0 {
                let Conv::Id(id) = conv[ch[q] as usize - 1] else { panic!("level mismatch in macrocell") };
                ids[q] = id;
              }
            }
            Conv::Id(self.intern(d, ids))
          }
        }
      };
      conv.push(c);
    }
    assert!(!conv.is_empty(), "empty macrocell");
    let mut level = last_level;
    let mut id = match conv.pop().unwrap() {
      Conv::Id(id) => id,
      Conv::Cells(cells) => {
        // root smaller than a base tile: embed at the top-left of one
        let side = 1usize << level;
        let mut t = vec![0u8; self.tile_side * self.tile_side];
        for y in 0..side {
          t[y * self.tile_side..y * self.tile_side + side].copy_from_slice(&cells[y * side..(y + 1) * side]);
        }
        level = a;
        self.tile(&t)
      }
    };
    while level < a + 1 {
      let z = self.empty(level);
      id = self.intern(level + 1, [id, z, z, z]);
      level += 1;
    }
    (id, level)
  }
}
