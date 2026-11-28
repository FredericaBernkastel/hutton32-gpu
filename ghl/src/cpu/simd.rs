//! Four `u64` lanes as one value, for the host's brute force with AVX2 (Stage 5): the gate networks
//! are generic in their word type, and compiled for this one inside functions with
//! `#[target_feature(enable = "avx2")]` the compiler keeps the four lanes in one 256-bit register.

use std::ops::{BitAnd, BitOr, BitXor, Not};

/// A word type the networks evaluate on: 64 cells (`u64`) or 256 (`X4`).
pub trait Lanes: Copy + BitAnd<Output = Self> + BitOr<Output = Self> + BitXor<Output = Self> + Not<Output = Self> {
  const ZERO: Self;
  const ONES: Self;
  fn is_zero(self) -> bool;
}

impl Lanes for u64 {
  const ZERO: u64 = 0;
  const ONES: u64 = !0;
  #[inline(always)]
  fn is_zero(self) -> bool {
    self == 0
  }
}

/// Four lanes of 64 cells.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[repr(C, align(32))]
pub struct X4(pub [u64; 4]);

impl X4 {
  pub const ZERO: X4 = X4([0; 4]);
  pub const ONES: X4 = X4([!0; 4]);
}

macro_rules! op {
  ($tr:ident, $f:ident, $o:tt) => {
    impl $tr for X4 {
      type Output = X4;
      #[inline(always)]
      fn $f(self, r: X4) -> X4 {
        X4([self.0[0] $o r.0[0], self.0[1] $o r.0[1], self.0[2] $o r.0[2], self.0[3] $o r.0[3]])
      }
    }
  };
}
op!(BitAnd, bitand, &);
op!(BitOr, bitor, |);
op!(BitXor, bitxor, ^);

impl Not for X4 {
  type Output = X4;
  #[inline(always)]
  fn not(self) -> X4 {
    X4([!self.0[0], !self.0[1], !self.0[2], !self.0[3]])
  }
}

impl Lanes for X4 {
  const ZERO: X4 = X4::ZERO;
  const ONES: X4 = X4::ONES;
  #[inline(always)]
  fn is_zero(self) -> bool {
    (self.0[0] | self.0[1] | self.0[2] | self.0[3]) == 0
  }
}

/// Whether this CPU runs the AVX2 path.
pub fn has_avx2() -> bool {
  #[cfg(target_arch = "x86_64")]
  {
    std::is_x86_feature_detected!("avx2")
  }
  #[cfg(not(target_arch = "x86_64"))]
  {
    false
  }
}

/// The circuits of the canonical rules, generic in the word type (emitted from the netlists).
#[allow(clippy::all, unused_parens)]
pub mod circuits {
  use super::Lanes;
  include!("../rule/generated/circuits_generic.rs");
}
