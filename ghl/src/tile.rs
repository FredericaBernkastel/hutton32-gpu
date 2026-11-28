//! Base tiles: squares of `2^a` cells stored as bit-planes, plane after plane, the bits of a plane
//! row-major (bit `y * side + x` of the plane is the cell at `(x, y)`) and packed into `u32` words.
//! An 8×8 Hutton32b tile is 10 words (40 B), a 16×16 Life tile 8 words (32 B). The GPU tile store
//! holds tiles in exactly this layout.

/// An index into the tile store or a node table.
pub type TileId = u32;

/// The ID that is no node: a result not yet computed, an empty slot.
pub const NONE: u32 = u32::MAX;

/// Geometry of base tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileGeom {
  /// log2 of the side
  pub a: u32,
  pub side: usize,
  pub planes: u32,
  /// `u32` words per plane: `max(1, side^2 / 32)`
  pub wpp: usize,
}

impl TileGeom {
  pub fn new(a: u32, planes: u32) -> TileGeom {
    assert!((1..=6).contains(&a), "base tiles of 2x2 to 64x64 cells");
    let side = 1usize << a;
    TileGeom { a, side, planes, wpp: (side * side).div_ceil(32) }
  }
  /// `u32` words per tile.
  #[inline]
  pub fn words(&self) -> usize {
    self.planes as usize * self.wpp
  }
  /// Bytes per tile in the tile store.
  pub fn bytes(&self) -> usize {
    self.words() * 4
  }

  /// Pack `side * side` cells (row-major) into `out` (`words()` long).
  pub fn pack_into(&self, cells: &[u8], out: &mut [u32]) {
    out.fill(0);
    let s = self.side;
    for y in 0..s {
      for x in 0..s {
        let c = cells[y * s + x] as u32;
        if c != 0 {
          let bit = y * s + x;
          for k in 0..self.planes as usize {
            out[k * self.wpp + bit / 32] |= ((c >> k) & 1) << (bit % 32);
          }
        }
      }
    }
  }
  pub fn pack(&self, cells: &[u8]) -> Vec<u32> {
    let mut v = vec![0u32; self.words()];
    self.pack_into(cells, &mut v);
    v
  }
  /// Unpack into `side * side` cells.
  pub fn unpack_into(&self, words: &[u32], out: &mut [u8]) {
    let s = self.side;
    for (bit, o) in out.iter_mut().enumerate().take(s * s) {
      let mut c = 0u32;
      for k in 0..self.planes as usize {
        c |= ((words[k * self.wpp + bit / 32] >> (bit % 32)) & 1) << k;
      }
      *o = c as u8;
    }
  }
  pub fn unpack(&self, words: &[u32]) -> Vec<u8> {
    let mut v = vec![0u8; self.side * self.side];
    self.unpack_into(words, &mut v);
    v
  }
  #[inline]
  pub fn cell(&self, words: &[u32], x: usize, y: usize) -> u8 {
    let bit = y * self.side + x;
    let mut c = 0u32;
    for k in 0..self.planes as usize {
      c |= ((words[k * self.wpp + bit / 32] >> (bit % 32)) & 1) << k;
    }
    c as u8
  }
  /// Row `y` of plane `k` as the low `side` bits of a `u64` (bit x = cell x).
  #[inline]
  pub fn row(&self, words: &[u32], k: usize, y: usize) -> u64 {
    let s = self.side;
    let base = k * self.wpp;
    if s == 64 {
      let w0 = words[base + y * 2] as u64;
      let w1 = words[base + y * 2 + 1] as u64;
      w0 | w1 << 32
    } else if s == 32 {
      words[base + y] as u64
    } else {
      let bit = y * s;
      let w = words[base + bit / 32] as u64;
      (w >> (bit % 32)) & ((1u64 << s) - 1)
    }
  }
  /// Set row `y` of plane `k` from the low `side` bits of `v`.
  #[inline]
  pub fn set_row(&self, words: &mut [u32], k: usize, y: usize, v: u64) {
    let s = self.side;
    let base = k * self.wpp;
    if s == 64 {
      words[base + y * 2] = v as u32;
      words[base + y * 2 + 1] = (v >> 32) as u32;
    } else if s == 32 {
      words[base + y] = v as u32;
    } else {
      let bit = y * s;
      let m = ((1u64 << s) - 1) as u32;
      let w = &mut words[base + bit / 32];
      *w = (*w & !(m << (bit % 32))) | (((v as u32) & m) << (bit % 32));
    }
  }
  /// Is the tile all zero?
  pub fn is_empty(words: &[u32]) -> bool {
    words.iter().all(|&w| w == 0)
  }
}

/// Hash of a tile's words; the same function runs in WGSL (`gpu/kernels`).
#[inline]
pub fn tile_hash(words: &[u32]) -> u32 {
  let mut h: u32 = 0x9E37_79B9;
  for &w in words {
    h ^= w;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
  }
  h ^= h >> 16;
  h = h.wrapping_mul(0xC2B2_AE35);
  h ^ (h >> 16)
}

/// Hash of a node key (four child IDs); the same function runs in WGSL.
#[inline]
pub fn hash4(k: [u32; 4]) -> u32 {
  let mut h = k[0].wrapping_mul(0x9E3779B1);
  h = (h ^ (h >> 15)).wrapping_add(k[1].wrapping_mul(0x85EBCA77));
  h = (h ^ (h >> 13)).wrapping_add(k[2].wrapping_mul(0xC2B2AE3D));
  h = (h ^ (h >> 16)).wrapping_add(k[3].wrapping_mul(0x27D4EB2F));
  h = (h ^ (h >> 15)).wrapping_mul(0x2C1B3C6D);
  h ^ (h >> 12)
}

/// An open-addressing table in the GPU's layout, 8 words per entry `[k0, k1, k2, k3, value, 0, 0, 0]`,
/// at most half full (the harness's `build_table`). Returns the words, the slot mask and the mean
/// number of probes per insert.
pub fn build_table(entries: impl Iterator<Item = ([u32; 4], u32)>, count: usize, min_log2: u32) -> (Vec<u32>, u32, f64) {
  let mut log2 = min_log2;
  while (1usize << log2) < count * 2 {
    log2 += 1;
  }
  let mask = (1u32 << log2) - 1;
  let mut t = vec![0u32; 8usize << log2];
  for s in 0..(1usize << log2) {
    t[s * 8..s * 8 + 4].fill(u32::MAX);
  }
  let (mut probes, mut n) = (0u64, 0u64);
  for (k, v) in entries {
    let mut slot = hash4(k) & mask;
    loop {
      probes += 1;
      let b = slot as usize * 8;
      if t[b] == u32::MAX && t[b + 1] == u32::MAX {
        t[b..b + 4].copy_from_slice(&k);
        t[b + 4] = v;
        break;
      }
      slot = (slot + 1) & mask;
    }
    n += 1;
  }
  (t, mask, probes as f64 / n.max(1) as f64)
}

/// Displacement of every occupied slot of a table built by [`build_table`] from its home slot.
pub fn displacements(table: &[u32], mask: u32) -> impl Iterator<Item = Option<u32>> + '_ {
  (0..=mask).map(move |sl| {
    let b = sl as usize * 8;
    if table[b] == u32::MAX && table[b + 1] == u32::MAX {
      None
    } else {
      let home = hash4([table[b], table[b + 1], table[b + 2], table[b + 3]]) & mask;
      Some(sl.wrapping_sub(home) & mask)
    }
  })
}

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn pack_round_trip() {
    for (a, p) in [(2u32, 1u32), (3, 5), (4, 1), (5, 2), (6, 1)] {
      let g = TileGeom::new(a, p);
      let n = g.side * g.side;
      let cells: Vec<u8> = (0..n).map(|i| ((i * 7 + i / 3) % (1 << p)) as u8).collect();
      let w = g.pack(&cells);
      assert_eq!(g.unpack(&w), cells);
      for y in 0..g.side {
        for k in 0..p as usize {
          let r = g.row(&w, k, y);
          let want: u64 = (0..g.side).map(|x| (((cells[y * g.side + x] >> k) & 1) as u64) << x).sum();
          assert_eq!(r, want);
          let mut w2 = w.clone();
          g.set_row(&mut w2, k, y, r);
          assert_eq!(w2, w);
        }
      }
    }
  }
}
