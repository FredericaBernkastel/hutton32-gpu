//! Golly macrocell (`[M2]`, multistate) reader and conversion into the reference engine.

use crate::hl::Engine;
use std::{fs, path::Path};

pub struct Macrocell {
  pub rule: String,
  /// `(level, [nw, ne, sw, se])`; node `i` (1-based) is `nodes[i-1]`. Level-1 nodes carry four cell
  /// states, higher levels carry node indices (0 = empty).
  pub nodes: Vec<(u32, [u64; 4])>,
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
    let v: Vec<u64> = t.split_whitespace().take(5).map(|x| x.parse().expect("bad macrocell line")).collect();
    assert!(v.len() == 5, "bad macrocell line: {t}");
    nodes.push((v[0] as u32, [v[1], v[2], v[3], v[4]]));
  }
  Macrocell { rule, nodes }
}

enum Conv {
  Cells(Vec<u8>),
  Id(u32),
}

impl<'a> Engine<'a> {
  /// Convert a macrocell quadtree. Returns `(root id, level)` with `level >= a + 1`.
  pub fn from_mc(&mut self, mc: &Macrocell) -> (u32, usize) {
    let a = self.a;
    let mut conv: Vec<Conv> = Vec::with_capacity(mc.nodes.len());
    for &(d, ch) in &mc.nodes {
      let d = d as usize;
      let c = if d <= a {
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
        if d == a { Conv::Id(self.tile(&cells)) } else { Conv::Cells(cells) }
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
      };
      conv.push(c);
    }
    let (mut level, _) = mc.nodes.last().map(|&(d, _)| (d as usize, ())).expect("empty macrocell");
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
