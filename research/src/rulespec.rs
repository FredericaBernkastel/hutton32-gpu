//! Which transition function an experiment uses, and the Golly table loader.
//!
//! There are two rules, and each has one canonical table:
//!   * Hutton32:  Tim Hutton's `Hutton32.table` (Golly's rule table repository); Golly's built-in
//!     C++ rule of that name computes the same function;
//!   * Hutton32b: `doc/Hutton32b.rule` of the Hutton32-Computer repository (what Golly's
//!     RuleLoader runs).
//! The WGSL files in `archive/src/kernel/rules` equal them on all 2^25 neighbourhoods, and so does
//! the CPU port in `rule.rs` (`hutton32`, `hutton32b`). The same files as first archived
//! (commit 256ef91) did not; that version is kept as `hutton32-v1`, `hutton32b-v1`.

use crate::rule::{self, Variant, LUT_SIZE};
use std::path::{Path, PathBuf};

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum RuleSpec {
  /// the canonical rule: CPU port of the current WGSL rule files
  Wgsl(Variant),
  /// the WGSL rule files as archived in commit 256ef91 (paper.md, section 10)
  WgslV1(Variant),
  /// a Golly rule table: a `.rule` file with an `@TABLE` section, or a bare `.table`
  Table(PathBuf),
}

impl RuleSpec {
  pub fn parse(s: &str) -> Option<Self> {
    if let Some(p) = s.strip_prefix("table:") {
      return Some(RuleSpec::Table(p.into()));
    }
    match s {
      // older command lines: Golly's built-in Hutton32 is the canonical Hutton32
      "hutton32-golly" => Some(RuleSpec::Wgsl(Variant::Hutton32)),
      "hutton32-v1" => Some(RuleSpec::WgslV1(Variant::Hutton32)),
      "hutton32b-v1" => Some(RuleSpec::WgslV1(Variant::Hutton32b)),
      _ => Variant::parse(s).map(RuleSpec::Wgsl),
    }
  }
  pub fn name(&self) -> String {
    match self {
      RuleSpec::Wgsl(Variant::Hutton32) => "Hutton32".into(),
      RuleSpec::Wgsl(Variant::Hutton32b) => "Hutton32b".into(),
      RuleSpec::WgslV1(v) => format!("{} as archived in v1", v.name()),
      RuleSpec::Table(p) => format!("{} (Golly table)", p.file_stem().unwrap().to_string_lossy()),
    }
  }
  pub fn cache_name(&self) -> String {
    match self {
      RuleSpec::Wgsl(v) => format!("{}_canonical", v.name()),
      RuleSpec::WgslV1(v) => format!("{}_v1", v.name()),
      RuleSpec::Table(p) => format!("table_{}", p.file_stem().unwrap().to_string_lossy()),
    }
  }
  /// What a cached LUT of this rule depends on besides the harness itself: for a table, the
  /// bytes of the file. `load_lut` rebuilds the cache when this changes.
  pub fn source_stamp(&self) -> Option<String> {
    match self {
      RuleSpec::Table(p) => {
        let bytes = std::fs::read(p).unwrap_or_else(|e| panic!("failed to read {}: {e}", p.display()));
        // FNV-1a, 64 bit
        let hash = bytes.iter().fold(0xcbf29ce484222325u64, |h, &b| (h ^ b as u64).wrapping_mul(0x100000001b3));
        Some(format!("{:016x} {} bytes", hash, bytes.len()))
      }
      _ => None,
    }
  }
  pub fn build(&self) -> Vec<u8> {
    match self {
      RuleSpec::Wgsl(v) => rule::build_lut(*v),
      RuleSpec::WgslV1(v) => rule::build_lut_v1(*v),
      RuleSpec::Table(p) => lut_from_table(p),
    }
  }
}

/// Build a LUT from the `@TABLE` section of a Golly `.rule` file (von Neumann neighbourhood,
/// `symmetries:none`). Semantics follow `ruletable_algo.cpp`: transitions are `C,N,E,S,W,C'`;
/// a variable used more than once in a transition is bound (same value at every occurrence);
/// the first matching transition wins; unmatched neighbourhoods leave the cell unchanged.
pub fn lut_from_table(path: &Path) -> Vec<u8> {
  use std::collections::HashMap;
  let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {}: {e}", path.display()));
  // a bare `.table` file (Golly's rule table repository) has no sections: all of it is the table
  let mut in_table = !text.lines().any(|l| l.trim_start().starts_with('@'));
  let mut vars: HashMap<String, Vec<u8>> = HashMap::new();
  // expanded transitions: five input sets (32-bit masks) + output
  let mut rules: Vec<([u32; 5], u8)> = Vec::new();
  for raw in text.lines() {
    let line = raw.split('#').next().unwrap().trim();
    if line.starts_with('@') {
      in_table = line == "@TABLE";
      continue;
    }
    if !in_table || line.is_empty() {
      continue;
    }
    if let Some(v) = line.strip_prefix("n_states:") {
      assert_eq!(v.trim(), "32", "only 32-state tables are supported");
      continue;
    }
    if let Some(v) = line.strip_prefix("neighborhood:") {
      assert_eq!(v.trim(), "vonNeumann");
      continue;
    }
    if let Some(v) = line.strip_prefix("symmetries:") {
      assert_eq!(v.trim(), "none", "only symmetries:none is supported");
      continue;
    }
    if let Some(rest) = line.strip_prefix("var ") {
      let (name, set) = rest.split_once('=').expect("bad var line");
      let set = set.trim().trim_start_matches('{').trim_end_matches('}');
      let mut vals = Vec::new();
      for tok in set.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        match tok.parse::<u8>() {
          Ok(v) => vals.push(v),
          Err(_) => vals.extend(vars.get(tok).unwrap_or_else(|| panic!("unknown variable {tok}")).iter().copied()),
        }
      }
      vars.insert(name.trim().to_string(), vals);
      continue;
    }
    let toks: Vec<&str> = line.split(',').map(str::trim).collect();
    assert_eq!(toks.len(), 6, "bad transition: {line}");
    // bound variables: names that occur more than once among the six tokens
    let mut bound: Vec<&str> = Vec::new();
    for t in &toks {
      if vars.contains_key(*t) && toks.iter().filter(|x| *x == t).count() > 1 && !bound.contains(t) {
        bound.push(t);
      }
    }
    let mut idx = vec![0usize; bound.len()];
    loop {
      let value = |tok: &str| -> Vec<u8> {
        if let Some(bi) = bound.iter().position(|b| *b == tok) {
          vec![vars[tok][idx[bi]]]
        } else if let Some(v) = vars.get(tok) {
          v.clone()
        } else {
          vec![tok.parse::<u8>().unwrap_or_else(|_| panic!("bad token {tok} in {line}"))]
        }
      };
      let mut masks = [0u32; 5];
      for (i, m) in masks.iter_mut().enumerate() {
        for v in value(toks[i]) {
          *m |= 1 << v;
        }
      }
      let out = value(toks[5]);
      assert_eq!(out.len(), 1, "unbound output variable in {line}");
      rules.push((masks, out[0]));
      // next combination of bound variables (odometer)
      let mut i = 0;
      while i < bound.len() {
        idx[i] += 1;
        if idx[i] < vars[bound[i]].len() {
          break;
        }
        idx[i] = 0;
        i += 1;
      }
      if i == bound.len() {
        break;
      }
    }
  }
  assert!(!rules.is_empty(), "no @TABLE transitions found in {}", path.display());
  // per-position, per-state bitsets over rules
  let words = rules.len().div_ceil(64);
  let mut bits = vec![0u64; 5 * 32 * words];
  for (r, (masks, _)) in rules.iter().enumerate() {
    for pos in 0..5 {
      for st in 0..32 {
        if masks[pos] >> st & 1 == 1 {
          bits[(pos * 32 + st) * words + r / 64] |= 1 << (r % 64);
        }
      }
    }
  }
  let row = |pos: usize, st: u32| &bits[(pos * 32 + st as usize) * words..(pos * 32 + st as usize + 1) * words];
  let mut lut = vec![0u8; LUT_SIZE];
  for (i, out) in lut.iter_mut().enumerate() {
    let i = i as u32;
    let (w, s, e, n, c) = (i & 31, (i >> 5) & 31, (i >> 10) & 31, (i >> 15) & 31, (i >> 20) & 31);
    let (rc, rn, re, rs, rw) = (row(0, c), row(1, n), row(2, e), row(3, s), row(4, w));
    let mut res = c as u8;
    for k in 0..words {
      let m = rc[k] & rn[k];
      if m == 0 {
        continue;
      }
      let m = m & re[k] & rs[k] & rw[k];
      if m != 0 {
        res = rules[k * 64 + m.trailing_zeros() as usize].1;
        break;
      }
    }
    *out = res;
  }
  lut
}
