//! Transitions made from the rule's function instead of taken from the table's text.
//!
//! A table lists its transitions as its author (or the script that wrote it) saw fit: with a
//! fixed priority among directions, with sets of states that differ from line to line. What the
//! rule computes is often far more regular. Here the states of each neighbour are first grouped
//! into classes: two states are in one class if no centre state and no other neighbours can tell
//! them apart. The function is then a table over (centre state, class of each neighbour), small
//! enough to hold, and covers are grown in it: from a neighbourhood that is not covered yet a
//! cube is widened as far as the function allows (every class of a cell first, which needs no
//! gate at all; then more centre states; then single classes), and where the rule has a
//! rotational symmetry the turned images of a cube are added with it, so that all directions are
//! tested in the same way. The cubes of a cover do not overlap with another result, so no
//! priority is left between them.

use crate::check::Space;
use crate::symmetry::Symmetry;
use ca_rule::table::Out;
use ca_rule::{Case, Interp, StateSet};

/// The classes of the states of each cell (cell 0, the centre, has none).
pub struct Classes {
  /// `of[cell][state]`
  pub of: Vec<Vec<u8>>,
  /// `members[cell][class]`
  pub members: Vec<Vec<StateSet>>,
}

/// The next state of every neighbourhood, the centre the highest digit.
pub fn function(space: Space, interp: &Interp, threads: usize) -> Vec<u8> {
  let total = space.total().unwrap() as usize;
  let mut lut = vec![0u8; total];
  let chunk = total.div_ceil(threads.max(1) * 8).max(1);
  let jobs = std::sync::Mutex::new(lut.chunks_mut(chunk).enumerate());
  std::thread::scope(|s| {
    for _ in 0..threads.max(1) {
      s.spawn(|| {
        let mut scratch = vec![0u64; interp.words()];
        loop {
          let job = jobs.lock().unwrap().next();
          let Some((i, out)) = job else { break };
          for (j, o) in out.iter_mut().enumerate() {
            let p = space.point((i * chunk + j) as u64);
            *o = interp.next(&p[..space.k], &mut scratch);
          }
        }
      });
    }
  });
  lut
}

/// Group the states of each neighbour: one class if nothing else can tell them apart.
pub fn classes(space: Space, lut: &[u8]) -> Classes {
  let (n, k) = (space.n, space.k);
  let mut of = vec![Vec::new(); k];
  let mut members = vec![Vec::new(); k];
  for cell in 1..k {
    let stride = n.pow((k - 1 - cell) as u32);
    // a fingerprint of everything the function does with each state in this cell
    let mut prints = vec![(0u64, 0u64); n];
    let block = stride * n;
    let mut salt = 0x9E37_79B9_7F4A_7C15u64;
    for base in (0..lut.len()).step_by(block) {
      for low in 0..stride {
        salt = salt.wrapping_mul(0xBF58_476D_1CE4_E5B9).wrapping_add(0x94D0_49BB_1331_11EB);
        for (v, p) in prints.iter_mut().enumerate() {
          let x = lut[base + v * stride + low] as u64 + 1;
          p.0 = p.0.wrapping_add(x.wrapping_mul(salt));
          p.1 ^= (x.wrapping_mul(salt | 1)).rotate_left((salt >> 58) as u32);
        }
      }
    }
    let mut seen: Vec<(u64, u64)> = Vec::new();
    of[cell] = prints
      .iter()
      .map(|p| match seen.iter().position(|s| s == p) {
        Some(c) => c as u8,
        None => {
          seen.push(*p);
          seen.len() as u8 - 1
        }
      })
      .collect();
    members[cell] = vec![StateSet::default(); seen.len()];
    for v in 0..n {
      members[cell][of[cell][v] as usize].insert(v as u8);
    }
  }
  Classes { of, members }
}

#[derive(Clone, PartialEq, Eq)]
struct Cube {
  /// centre states, as positions in the domain's list
  centres: u64,
  /// classes per neighbour
  sets: Vec<u32>,
}

const OFF: u8 = 0;
const ON: u8 = 1;

/// The function over (centre state of a list, class of each neighbour).
struct Domain<'a> {
  k: usize,
  cls: &'a Classes,
  centres: Vec<u8>,
  dims: Vec<usize>,
  stride: Vec<usize>,
  tuples: usize,
  f: Vec<u8>,
}

impl<'a> Domain<'a> {
  fn new(space: Space, lut: &[u8], cls: &'a Classes, centres: &StateSet) -> Option<Domain<'a>> {
    let (n, k) = (space.n, space.k);
    let centres: Vec<u8> = centres.iter().filter(|&c| (c as usize) < n).collect();
    let dims: Vec<usize> = (1..k).map(|c| cls.members[c].len()).collect();
    let tuples = dims.iter().try_fold(1usize, |p, &d| p.checked_mul(d))?;
    if centres.is_empty() || centres.len() > 64 || dims.iter().any(|&d| d > 32) || tuples.checked_mul(centres.len())? > 1 << 26 {
      return None;
    }
    let mut stride = vec![1usize; k - 1];
    for j in (0..k - 2).rev() {
      stride[j] = stride[j + 1] * dims[j + 1];
    }
    // a state of each class stands for it
    let reps: Vec<Vec<u8>> = (1..k).map(|c| cls.members[c].iter().map(|m| m.iter().next().unwrap()).collect()).collect();
    let mut f = vec![0u8; centres.len() * tuples];
    let mut a = vec![0usize; k - 1];
    for (ci, &c) in centres.iter().enumerate() {
      for t in 0..tuples {
        let mut r = t;
        for j in 0..k - 1 {
          a[j] = r / stride[j];
          r %= stride[j];
        }
        let mut idx = c as usize;
        for j in 0..k - 1 {
          idx = idx * n + reps[j][a[j]] as usize;
        }
        f[ci * tuples + t] = lut[idx];
      }
    }
    Some(Domain { k, cls, centres, dims, stride, tuples, f })
  }

  /// Visit the points of a cube until `visit` says no; whether it never did.
  fn each(&self, q: &Cube, visit: &mut dyn FnMut(usize) -> bool) -> bool {
    let lists: Vec<Vec<usize>> = (0..self.k - 1).map(|j| (0..self.dims[j]).filter(|&a| q.sets[j] >> a & 1 == 1).collect()).collect();
    if lists.iter().any(|l| l.is_empty()) {
      return true;
    }
    let mut at = vec![0usize; self.k - 1];
    for ci in (0..self.centres.len()).filter(|&ci| q.centres >> ci & 1 == 1) {
      at.fill(0);
      loop {
        let t: usize = (0..self.k - 1).map(|j| lists[j][at[j]] * self.stride[j]).sum();
        if !visit(ci * self.tuples + t) {
          return false;
        }
        // the next tuple of classes; done with this centre when the count wraps around
        let mut j = self.k - 1;
        let mut advanced = false;
        while j > 0 {
          j -= 1;
          at[j] += 1;
          if at[j] < lists[j].len() {
            advanced = true;
            break;
          }
          at[j] = 0;
        }
        if !advanced {
          break;
        }
      }
    }
    true
  }

  fn free_of_off(&self, tri: &[u8], q: &Cube) -> bool {
    self.each(q, &mut |p| tri[p] != OFF)
  }

  /// A cover of the ON points of `tri` by cubes without OFF points. `turns` gives, for a cube,
  /// its images under the rule's symmetry (those that exist in this domain).
  fn cover(&self, tri: &[u8], turns: &dyn Fn(&Cube) -> Vec<Cube>) -> Vec<Cube> {
    let full: Vec<u32> = self.dims.iter().map(|&d| if d == 32 { !0 } else { (1u32 << d) - 1 }).collect();
    let mut count = vec![0u16; tri.len()];
    let mut cubes: Vec<Cube> = Vec::new();
    // the sets in use per cell, with how often
    for p in 0..tri.len() {
      if tri[p] != ON || count[p] > 0 {
        continue;
      }
      let (ci, mut r) = (p / self.tuples, p % self.tuples);
      let mut q = Cube { centres: 1 << ci, sets: vec![0; self.k - 1] };
      for j in 0..self.k - 1 {
        q.sets[j] = 1 << (r / self.stride[j]);
        r %= self.stride[j];
      }
      // widen: all classes of a cell, then more centre states, then single classes
      let grow = |q: &mut Cube, j: usize, add: u32| -> bool {
        let mut slice = q.clone();
        slice.sets[j] = add;
        if self.free_of_off(tri, &slice) {
          q.sets[j] |= add;
          true
        } else {
          false
        }
      };
      for j in 0..self.k - 1 {
        let add = full[j] & !q.sets[j];
        grow(&mut q, j, add);
      }
      for c in 0..self.centres.len() {
        if q.centres >> c & 1 == 0 {
          let mut slice = q.clone();
          slice.centres = 1 << c;
          if self.free_of_off(tri, &slice) {
            q.centres |= 1 << c;
          }
        }
      }
      for j in 0..self.k - 1 {
        if q.sets[j] == full[j] {
          continue;
        }
        // a set other cubes test in this cell already, the most used first; else class by class
        let mut known: Vec<(usize, u32)> = Vec::new();
        for o in &cubes {
          let s = o.sets[j];
          if s != full[j] && s & q.sets[j] == q.sets[j] && s != q.sets[j] {
            match known.iter_mut().find(|x| x.1 == s) {
              Some(x) => x.0 += 1,
              None => known.push((1, s)),
            }
          }
        }
        known.sort_by(|a, b| b.0.cmp(&a.0).then(b.1.count_ones().cmp(&a.1.count_ones())));
        let have = cubes.iter().any(|o| o.sets[j] == q.sets[j]);
        let cur = q.sets[j];
        if known.iter().any(|x| grow(&mut q, j, x.1 & !cur)) || have {
          continue;
        }
        for a in 0..self.dims[j] {
          if q.sets[j] >> a & 1 == 0 {
            grow(&mut q, j, 1 << a);
          }
        }
      }
      let mut family = vec![q.clone()];
      for t in turns(&q) {
        if !family.contains(&t) && self.free_of_off(tri, &t) {
          family.push(t);
        }
      }
      for q in family {
        if cubes.contains(&q) {
          continue;
        }
        self.each(&q, &mut |p| {
          count[p] = count[p].saturating_add(1);
          true
        });
        cubes.push(q);
      }
    }
    // drop the cubes whose ON points all lie in other cubes, the last made first
    for i in (0..cubes.len()).rev() {
      if self.each(&cubes[i], &mut |p| tri[p] != ON || count[p] >= 2) {
        self.each(&cubes[i], &mut |p| {
          count[p] -= 1;
          true
        });
        cubes.remove(i);
      }
    }
    cubes
  }

  fn case(&self, q: &Cube, out: u8) -> Case {
    let mut sets = vec![StateSet::default(); self.k];
    for (ci, &c) in self.centres.iter().enumerate() {
      if q.centres >> ci & 1 == 1 {
        sets[0].insert(c);
      }
    }
    for j in 0..self.k - 1 {
      for a in 0..self.dims[j] {
        if q.sets[j] >> a & 1 == 1 {
          for i in 0..4 {
            sets[j + 1].0[i] |= self.cls.members[j + 1][a].0[i];
          }
        }
      }
    }
    Case { sets, eq: Vec::new(), out: Out::Const(out) }
  }

  /// The cube of sets of states, if they are centre states of this domain and whole classes.
  fn cube(&self, sets: &[StateSet]) -> Option<Cube> {
    let mut q = Cube { centres: 0, sets: vec![0; self.k - 1] };
    for c in sets[0].iter() {
      q.centres |= 1 << self.centres.iter().position(|&x| x == c)?;
    }
    for j in 0..self.k - 1 {
      let mut whole = StateSet::default();
      for v in sets[j + 1].iter() {
        let a = *self.cls.of[j + 1].get(v as usize)? as usize;
        q.sets[j] |= 1 << a;
        for i in 0..4 {
          whole.0[i] |= self.cls.members[j + 1][a].0[i];
        }
      }
      if whole != sets[j + 1] {
        return None;
      }
    }
    Some(q)
  }
}

/// The transitions for [`crate::build::build_folded`], made from the function: `canon` for the
/// centre states of orientation 0 (results as for that orientation), `rest[bit]` for the centre
/// states without orientation, one list per bit of the state, each transition setting the bit
/// or clearing it. `None` if the classes leave a table too large to hold.
pub fn covers(space: Space, lut: &[u8], sym: &Symmetry) -> Option<(Vec<Case>, Vec<Vec<Case>>)> {
  let (n, k) = (space.n, space.k);
  let cls = classes(space, lut);
  let oriented = sym.all_oriented();
  let mut others = StateSet::default();
  for v in (0..n as u8).filter(|&v| !oriented.contains(v)) {
    others.insert(v);
  }
  let mut rest: Vec<Vec<Case>> = vec![Vec::new(); space.bits];
  if let Some(d) = Domain::new(space, lut, &cls, &others) {
    // a cube turned by the symmetry: the centre's set and each cell's set relabelled, the cells moved
    let turns = |q: &Cube| -> Vec<Cube> {
      let c = d.case(q, 0);
      (1..sym.order)
        .filter_map(|t| {
          let mut sets = vec![StateSet::default(); k];
          for i in 0..k {
            sets[sym.turn_cell(i, t)] = sym.turn_set(&c.sets[i], t);
          }
          d.cube(&sets)
        })
        .collect()
    };
    for bit in 0..space.bits {
      for set in [true, false] {
        // set: the bit becomes 1 where the centre's is 0; clear: 0 where it is 1; where the
        // bit stays as it is either is free
        let tri: Vec<u8> = (0..d.f.len())
          .map(|p| {
            let c = d.centres[p / d.tuples] >> bit & 1 == 1;
            let v = d.f[p] >> bit & 1 == 1;
            if v != set {
              OFF
            } else if c == set {
              2
            } else {
              ON
            }
          })
          .collect();
        for q in d.cover(&tri, &turns) {
          rest[bit].push(d.case(&q, if set { 1 << bit } else { 0 }));
        }
      }
    }
  } else if !others.is_empty() {
    return None;
  }
  let mut canon: Vec<Case> = Vec::new();
  if !sym.oriented[0].is_empty() {
    let d = Domain::new(space, lut, &cls, &sym.oriented[0])?;
    let mut results: Vec<u8> = d.f.clone();
    results.sort_unstable();
    results.dedup();
    for v in results {
      let tri: Vec<u8> = (0..d.f.len()).map(|p| if d.f[p] != v { OFF } else if d.centres[p / d.tuples] == v { 2 } else { ON }).collect();
      for q in d.cover(&tri, &|_| Vec::new()) {
        canon.push(d.case(&q, v));
      }
    }
  }
  Some((canon, rest))
}
