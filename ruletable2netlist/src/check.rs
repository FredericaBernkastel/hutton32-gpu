//! Checking a network against the table: on every neighbourhood of the rule where they can be
//! enumerated, on a fixed set of them otherwise.

use crate::net::{mask, node, Net};
use ca_rule::table::Interp;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// For diagnosis (`R2N_CARE`): the centre states a check looks at, as a bit set; 0 for all. A
/// network optimised under it is right for those centre states only.
pub static CARE: AtomicU64 = AtomicU64::new(0);
use std::sync::Mutex;

/// The states of a neighbourhood's cells, in the order of a transition line.
pub type Nb = [u8; 9];

/// Words evaluated together: 4,096 neighbourhoods.
const CW: usize = 64;

const PAT: [u64; 6] =
  [0xAAAA_AAAA_AAAA_AAAA, 0xCCCC_CCCC_CCCC_CCCC, 0xF0F0_F0F0_F0F0_F0F0, 0xFF00_FF00_FF00_FF00, 0xFFFF_0000_FFFF_0000, 0xFFFF_FFFF_0000_0000];

pub struct Rng(pub u64);

impl Rng {
  pub fn next(&mut self) -> u64 {
    self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = self.0;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
  }
  pub fn below(&mut self, n: usize) -> usize {
    (self.next() % n as u64) as usize
  }
}

/// The neighbourhoods of a rule: `k` cells of `n` states, `bits` bits a state. Input `c * bits + b`
/// of a network is bit `b` of cell `c`.
#[derive(Clone, Copy, Debug)]
pub struct Space {
  pub n: usize,
  pub k: usize,
  pub bits: usize,
}

impl Space {
  pub fn inputs(&self) -> usize {
    self.k * self.bits
  }
  /// How many neighbourhoods there are, if that fits 64 bits.
  pub fn total(&self) -> Option<u64> {
    (self.n as u64).checked_pow(self.k as u32)
  }
  /// Neighbourhood number `idx`: the last cell is the lowest digit.
  pub fn point(&self, mut idx: u64) -> Nb {
    let mut p = [0u8; 9];
    for i in (0..self.k).rev() {
      p[i] = (idx % self.n as u64) as u8;
      idx /= self.n as u64;
    }
    p
  }
  pub fn random(&self, rng: &mut Rng) -> Nb {
    let mut p = [0u8; 9];
    for c in p.iter_mut().take(self.k) {
      *c = rng.below(self.n) as u8;
    }
    p
  }
  /// Input planes of `64 * w` neighbourhoods: `out[i * w + j]` holds input i of neighbourhoods
  /// `64 j .. 64 j + 63`.
  pub fn pack(&self, pts: &[Nb], w: usize, out: &mut [u64]) {
    out[..self.inputs() * w].fill(0);
    for (j, p) in pts.iter().enumerate().take(64 * w) {
      for c in 0..self.k {
        let mut v = p[c];
        let mut i = c * self.bits;
        while v != 0 {
          if v & 1 == 1 {
            out[i * w + j / 64] |= 1 << (j % 64);
          }
          v >>= 1;
          i += 1;
        }
      }
    }
  }
}

enum Src {
  /// every neighbourhood of a rule whose states are a power of two: the planes are patterns
  Pow2,
  /// every neighbourhood, made as it is needed
  Count,
  /// stored planes, `[word * inputs + input]`
  Planes(Vec<u64>),
}

pub struct Checker<'a> {
  pub space: Space,
  interp: &'a Interp<'a>,
  src: Src,
  /// the neighbourhoods of a chosen set
  pts: Vec<Nb>,
  /// the number of neighbourhoods of a complete set
  total: u64,
  words: usize,
  /// the table's answers, `[word * bits + bit]`; made as needed if absent
  reference: Option<Vec<u64>>,
  pub exhaustive: bool,
  threads: usize,
}

impl<'a> Checker<'a> {
  /// Every neighbourhood of the rule. With `cache` the table's answers are computed once and
  /// kept (`bits / 8` bytes a neighbourhood).
  pub fn exhaustive(space: Space, interp: &'a Interp<'a>, cache: bool, threads: usize) -> Checker<'a> {
    let total = space.total().expect("too many neighbourhoods to enumerate");
    let words = total.div_ceil(64) as usize;
    let pow2 = space.n.is_power_of_two() && total >= 64;
    let mut c = Checker { space, interp, src: if pow2 { Src::Pow2 } else { Src::Count }, pts: Vec::new(), total, words, reference: None, exhaustive: true, threads };
    if cache {
      if !pow2 {
        let ni = space.inputs();
        let mut planes = vec![0u64; words * ni];
        c.each_chunk(&mut planes, ni, |c, start, cw, out| {
          let mut tmp = vec![0u64; ni * cw];
          let pts: Vec<Nb> = (0..cw * 64).map(|j| c.point_at((start * 64 + j) as u64)).collect();
          c.space.pack(&pts, cw, &mut tmp);
          for i in 0..ni {
            for j in 0..cw {
              out[j * ni + i] = tmp[i * cw + j];
            }
          }
        });
        c.src = Src::Planes(planes);
      }
      c.cache_reference(0);
    }
    c
  }

  /// A chosen set of neighbourhoods.
  pub fn sampled(space: Space, interp: &'a Interp<'a>, pts: Vec<Nb>, threads: usize) -> Checker<'a> {
    let mut c = Checker { space, interp, src: Src::Planes(Vec::new()), pts: Vec::new(), total: 0, words: 0, reference: Some(Vec::new()), exhaustive: false, threads };
    c.add(&pts);
    c
  }

  /// A set of `count` neighbourhoods drawn at random, checked against the same table.
  pub fn subsample(&self, count: usize, seed: u64) -> Checker<'a> {
    let mut rng = Rng(seed);
    let pts: Vec<Nb> = (0..count).map(|_| self.space.random(&mut rng)).collect();
    Checker::sampled(self.space, self.interp, pts, self.threads)
  }

  /// Add neighbourhoods to a chosen set.
  pub fn add(&mut self, pts: &[Nb]) {
    if self.exhaustive || pts.is_empty() {
      return;
    }
    let first = self.words;
    self.pts.extend_from_slice(pts);
    while self.pts.len() % 64 != 0 {
      self.pts.push(*pts.last().unwrap());
    }
    self.total += pts.len() as u64;
    self.words = self.pts.len() / 64;
    let (ni, w) = (self.space.inputs(), self.words - first);
    let mut tmp = vec![0u64; ni * w];
    self.space.pack(&self.pts[first * 64..], w, &mut tmp);
    if let Src::Planes(pl) = &mut self.src {
      pl.resize(self.words * ni, 0);
      for i in 0..ni {
        for j in 0..w {
          pl[(first + j) * ni + i] = tmp[i * w + j];
        }
      }
    }
    self.cache_reference(first);
  }

  /// Neighbourhoods checked.
  pub fn points(&self) -> u64 {
    self.total
  }

  fn point_at(&self, idx: u64) -> Nb {
    if self.exhaustive {
      self.space.point(idx.min(self.total - 1))
    } else {
      self.pts[idx as usize]
    }
  }

  /// Run `f(self, first word, words, slice)` over a `[word * stride]` array in chunks, in threads.
  fn each_chunk(&self, data: &mut [u64], stride: usize, f: impl Fn(&Checker, usize, usize, &mut [u64]) + Sync) {
    let chunks = Mutex::new(data.chunks_mut(CW * stride).enumerate());
    std::thread::scope(|s| {
      for _ in 0..self.threads.max(1) {
        s.spawn(|| loop {
          let next = chunks.lock().unwrap().next();
          let Some((i, out)) = next else { break };
          f(self, i * CW, out.len() / stride, out);
        });
      }
    });
  }

  /// The table's answers for words `start..start + cw`.
  fn answers(&self, start: usize, cw: usize, out: &mut [u64], scratch: &mut [u64]) {
    let (k, bits) = (self.space.k, self.space.bits);
    out[..cw * bits].fill(0);
    for j in 0..cw * 64 {
      let p = self.point_at((start * 64 + j) as u64);
      let mut v = self.interp.next(&p[..k], scratch);
      let mut b = 0;
      while v != 0 {
        if v & 1 == 1 {
          out[(j / 64) * bits + b] |= 1 << (j % 64);
        }
        v >>= 1;
        b += 1;
      }
    }
  }

  fn cache_reference(&mut self, first: usize) {
    let bits = self.space.bits;
    let mut r = self.reference.take().unwrap_or_default();
    r.resize(self.words * bits, 0);
    self.each_chunk(&mut r[first * bits..], bits, |c, start, cw, out| {
      let mut scratch = vec![0u64; c.interp.words()];
      c.answers(first + start, cw, out, &mut scratch);
    });
    self.reference = Some(r);
  }

  /// The rows of the inputs for words `start..start + cw`, as `Net::simulate` reads them.
  fn fill(&self, start: usize, cw: usize, buf: &mut [u64]) {
    let (k, bits, ni) = (self.space.k, self.space.bits, self.space.inputs());
    match &self.src {
      Src::Pow2 => {
        for c in 0..k {
          for b in 0..bits {
            let p = (k - 1 - c) * bits + b;
            let row = &mut buf[(1 + c * bits + b) * cw..][..cw];
            if p < 6 {
              row.fill(PAT[p]);
            } else {
              for (j, x) in row.iter_mut().enumerate() {
                *x = 0u64.wrapping_sub(((start + j) >> (p - 6) & 1) as u64);
              }
            }
          }
        }
      }
      Src::Planes(pl) => {
        for i in 0..ni {
          for j in 0..cw {
            buf[(1 + i) * cw + j] = pl[(start + j) * ni + i];
          }
        }
      }
      Src::Count => {
        let pts: Vec<Nb> = (0..cw * 64).map(|j| self.point_at((start * 64 + j) as u64)).collect();
        self.space.pack(&pts, cw, &mut buf[cw..]);
      }
    }
  }

  /// Neighbourhoods on which the network differs from the table, `max_fail` of them at most;
  /// none if the network is right on every neighbourhood of the set.
  pub fn check(&self, net: &Net, max_fail: usize) -> Vec<Nb> {
    let bits = self.space.bits;
    assert_eq!(net.inputs, self.space.inputs());
    assert_eq!(net.outputs.len(), bits);
    let chunks = self.words.div_ceil(CW);
    let care = CARE.load(Ordering::Relaxed);
    let shift = (self.space.k - 1) * bits;
    let next = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let fails: Mutex<Vec<Nb>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
      for _ in 0..self.threads.max(1) {
        s.spawn(|| {
          let mut buf = vec![0u64; net.len() * CW];
          let mut answers = vec![0u64; CW * bits];
          let mut scratch = vec![0u64; self.interp.words()];
          loop {
            let c = next.fetch_add(1, Ordering::Relaxed);
            if c >= chunks || stop.load(Ordering::Relaxed) {
              break;
            }
            let start = c * CW;
            let cw = CW.min(self.words - start);
            if care != 0 && self.exhaustive && matches!(self.src, Src::Pow2) && shift >= 6 && ((start << 6) >> shift) == (((start + cw) << 6) - 1) >> shift && care >> ((start << 6) >> shift) & 1 == 0 {
              // one centre state for the whole chunk, and not one that is looked at
              continue;
            }
            self.fill(start, cw, &mut buf);
            net.simulate(cw, &mut buf);
            let want: &[u64] = match &self.reference {
              Some(r) => &r[start * bits..(start + cw) * bits],
              None => {
                self.answers(start, cw, &mut answers, &mut scratch);
                &answers
              }
            };
            let mut found = 0;
            for j in 0..cw {
              let mut diff = 0u64;
              for (b, &o) in net.outputs.iter().enumerate() {
                diff |= buf[node(o) * cw + j] ^ mask(o) ^ want[j * bits + b];
              }
              if care != 0 && diff != 0 {
                if self.exhaustive && matches!(self.src, Src::Pow2) && shift >= 6 {
                  if care >> (((start + j) << 6) >> shift) & 1 == 0 {
                    diff = 0;
                  }
                } else {
                  let mut d = diff;
                  while d != 0 {
                    let t = d.trailing_zeros();
                    d &= d - 1;
                    let p = self.point_at(((start + j) * 64) as u64 + t as u64);
                    if care >> p[0] & 1 == 0 {
                      diff &= !(1 << t);
                    }
                  }
                }
              }
              if diff != 0 && found < 4 {
                found += 1;
                let p = self.point_at(((start + j) * 64) as u64 + diff.trailing_zeros() as u64);
                let mut f = fails.lock().unwrap();
                f.push(p);
                if f.len() >= max_fail {
                  stop.store(true, Ordering::Relaxed);
                }
              }
            }
          }
        });
      }
    });
    let mut f = fails.into_inner().unwrap();
    f.truncate(max_fail);
    f
  }
}
