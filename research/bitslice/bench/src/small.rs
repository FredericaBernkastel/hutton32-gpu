//! Form C, scheme (c) of `tables.py`: for every centre state the neighbour states fall into a
//! few classes (at most 6 per direction); the whole rule is 6779 (centre, class, class, class,
//! class) cases.  Four 1024-entry tables indexed by (centre, one neighbour) give mixed-radix
//! digits whose sum indexes one final table: about 15 KiB in total, no branches, five loads.

use crate::{ByteGrid, LUT_SIZE};

/// `dig[d][c << 5 | x]` is the digit of neighbour `x` at direction d (n, e, s, w) as seen by
/// centre `c`, already multiplied by its weight (the north digit also carries the centre's base
/// offset). The sum of the four digits indexes `fin`.
pub struct SmallTables {
  dig: [[u16; 1024]; 4],
  fin: Vec<u8>,
  pub cases: usize,
}

const DIR_SHIFT: [u32; 4] = [15, 10, 5, 0]; // n, e, s, w in the LUT index

pub fn build(lut: &[u8; LUT_SIZE]) -> SmallTables {
  let mut dig = [[0u16; 1024]; 4];
  let mut fin: Vec<u8> = Vec::new();
  for c in 0..32u32 {
    // classes of each neighbour as seen by this centre
    let mut cls = [[0usize; 32]; 4];
    let mut reps: [Vec<u32>; 4] = Default::default();
    for d in 0..4 {
      let sh = DIR_SHIFT[d];
      for x in 0..32u32 {
        let same = |r: u32| -> bool {
          (0..(1u32 << 15)).all(|rest| {
            // spread the 15 bits of the other three neighbours around the 5-bit hole at `sh`
            let lo = rest & ((1 << sh) - 1);
            let hi = (rest >> sh) << (sh + 5);
            let base = (c << 20) | hi | lo;
            lut[(base | x << sh) as usize] == lut[(base | r << sh) as usize]
          })
        };
        let found = reps[d].iter().position(|&r| same(r));
        cls[d][x as usize] = match found {
          Some(ci) => ci,
          None => {
            reps[d].push(x);
            reps[d].len() - 1
          }
        };
      }
    }
    let k: [usize; 4] = std::array::from_fn(|d| reps[d].len());
    let weight = [k[1] * k[2] * k[3], k[2] * k[3], k[3], 1];
    let base = fin.len();
    for d in 0..4 {
      for x in 0..32usize {
        let v = cls[d][x] * weight[d] + if d == 0 { base } else { 0 };
        dig[d][(c as usize) << 5 | x] = u16::try_from(v).expect("index fits u16");
      }
    }
    for &rn in &reps[0] {
      for &re in &reps[1] {
        for &rs in &reps[2] {
          for &rw in &reps[3] {
            fin.push(lut[(c << 20 | rn << 15 | re << 10 | rs << 5 | rw) as usize]);
          }
        }
      }
    }
  }
  let cases = fin.len();
  fin.resize(cases.next_power_of_two(), 0);
  SmallTables { dig, fin, cases }
}

impl SmallTables {
  pub fn bytes(&self) -> usize {
    4 * 1024 * 2 + self.cases
  }

  #[inline(always)]
  pub fn eval(&self, c: usize, n: usize, e: usize, s: usize, w: usize) -> u8 {
    let cb = (c & 31) << 5;
    let i = self.dig[0][cb | (n & 31)] as usize
      + self.dig[1][cb | (e & 31)] as usize
      + self.dig[2][cb | (s & 31)] as usize
      + self.dig[3][cb | (w & 31)] as usize;
    // SAFETY: the digits are constructed so that their sum is < cases <= fin.len()
    unsafe { *self.fin.get_unchecked(i) }
  }

  /// Number of table entries (of 2^25) that the small tables get wrong.
  pub fn check(&self, lut: &[u8; LUT_SIZE]) -> u64 {
    let mut bad = 0u64;
    for i in 0..LUT_SIZE {
      let (w, s, e, n, c) = (i & 31, (i >> 5) & 31, (i >> 10) & 31, (i >> 15) & 31, (i >> 20) & 31);
      if self.eval(c, n, e, s, w) != lut[i] {
        bad += 1;
      }
    }
    bad
  }
}

#[inline(never)]
pub fn step(t: &SmallTables, src: &ByteGrid, dst: &mut ByteGrid) {
  let (w, h, stride) = (src.w, src.h, src.stride);
  assert!(dst.w == w && dst.h == h && src.cells.len() == stride * (h + 2) && dst.cells.len() == src.cells.len());
  for y in 1..=h {
    let up = &src.cells[(y - 1) * stride..y * stride];
    let cur = &src.cells[y * stride..(y + 1) * stride];
    let dn = &src.cells[(y + 1) * stride..(y + 2) * stride];
    let out = &mut dst.cells[y * stride..(y + 1) * stride];
    for x in 1..=w {
      // SAFETY: 1 <= x <= w and every row slice has w + 2 elements
      unsafe {
        let v = t.eval(
          *cur.get_unchecked(x) as usize,
          *up.get_unchecked(x) as usize,
          *cur.get_unchecked(x + 1) as usize,
          *dn.get_unchecked(x) as usize,
          *cur.get_unchecked(x - 1) as usize,
        );
        *out.get_unchecked_mut(x) = v;
      }
    }
  }
}
