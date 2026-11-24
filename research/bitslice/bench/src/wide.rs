//! The same generated gate network on 256-bit words (AVX2), 256 cells per operation.
//!
//! `step_planes.rs` is written for `u64`. Inside this module the name `u64` is deliberately
//! shadowed by a wrapper around `__m256i`, so the unmodified generated file can be included a
//! second time and compiles to vector code. Only built when AVX2 is enabled at compile time
//! (`RUSTFLAGS="-C target-cpu=native"`).

use std::arch::x86_64::*;
use std::ops::{BitAnd, BitOr, BitXor, Not};

#[allow(non_camel_case_types)]
#[derive(Clone, Copy)]
pub struct u64(pub __m256i);
pub type W = u64;

impl BitAnd for u64 {
  type Output = u64;
  #[inline(always)]
  fn bitand(self, o: u64) -> u64 {
    unsafe { u64(_mm256_and_si256(self.0, o.0)) }
  }
}
impl BitOr for u64 {
  type Output = u64;
  #[inline(always)]
  fn bitor(self, o: u64) -> u64 {
    unsafe { u64(_mm256_or_si256(self.0, o.0)) }
  }
}
impl BitXor for u64 {
  type Output = u64;
  #[inline(always)]
  fn bitxor(self, o: u64) -> u64 {
    unsafe { u64(_mm256_xor_si256(self.0, o.0)) }
  }
}
impl Not for u64 {
  type Output = u64;
  #[inline(always)]
  fn not(self) -> u64 {
    unsafe { u64(_mm256_xor_si256(self.0, _mm256_set1_epi32(-1))) }
  }
}

include!("../../step_planes.rs");
