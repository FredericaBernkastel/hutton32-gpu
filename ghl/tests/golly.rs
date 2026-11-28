//! Final states against Golly 5.1's HashLife: the patterns of the harness's `head2head`,
//! `runlength` and `life-h2h` targets, recorded once with bgolly as fingerprints
//! (`fixtures/record_golly.py`, `fixtures/golly.tsv`), so these tests do not need Golly.
//!
//! `cargo test --test golly` runs the short runs; `cargo test --release --test golly -- --ignored`
//! runs all of them (minutes).

use std::{path::PathBuf, time::Instant};

use ghl::{Config, Pattern, Rule, Universe};

struct Fixture {
  file: String,
  rule: String,
  gens: u64,
  k: u32,
  population: u64,
  w: i64,
  h: i64,
  hash: u64,
  golly_s: f64,
}

fn fixtures() -> Vec<Fixture> {
  let text = include_str!("fixtures/golly.tsv");
  text
    .lines()
    .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    .map(|l| {
      let f: Vec<&str> = l.split('\t').collect();
      Fixture {
        file: f[0].into(),
        rule: f[1].into(),
        gens: f[2].parse().unwrap(),
        k: f[3].parse().unwrap(),
        population: f[4].parse().unwrap(),
        w: f[5].parse().unwrap(),
        h: f[6].parse().unwrap(),
        hash: u64::from_str_radix(f[7], 16).unwrap(),
        golly_s: f[8].parse().unwrap(),
      }
    })
    .collect()
}

fn run(f: &Fixture, config: Config) -> f64 {
  let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../research/patterns").join(&f.file);
  let loaded = Pattern::load(&path).unwrap();
  let rule = Rule::named(&f.rule).unwrap();
  let mut u = Universe::new(rule, config).unwrap();
  u.load(&loaded.pattern, (0, 0)).unwrap();
  let t = Instant::now();
  u.step(f.gens);
  let secs = t.elapsed().as_secs_f64();
  let fp = u.fingerprint();
  assert_eq!((fp.population, fp.w, fp.h, fp.hash), (f.population, f.w, f.h, f.hash), "{} after {} generations differs from Golly", f.file, f.gens);
  secs
}

/// Runs that take the host recursion well under a second.
#[test]
fn golly_short_runs() {
  for f in fixtures().iter().filter(|f| f.gens <= 16384) {
    run(f, Config::cpu_step(f.k));
  }
}

/// Every fixture, at Golly's step, with the time against Golly's (the CPU target of Stage 0 is
/// 0.26–0.97× Golly's speed).
#[test]
#[ignore]
fn golly_all_runs() {
  // GHL_FIXTURES=<substring>: only the fixtures whose file contains it
  let only = std::env::var("GHL_FIXTURES").ok();
  for f in fixtures().into_iter().filter(|f| only.as_deref().is_none_or(|o| o.split(',').any(|o| f.file.contains(o)))) {
    let s = run(&f, Config::cpu_step(f.k));
    eprintln!("{:<36} {:>9} gens  ghl {:>8.2} s  golly {:>6.2} s  ratio {:.2}", f.file, f.gens, s, f.golly_s, f.golly_s / s);
  }
}

/// The same final states with the scheduler's own step choice.
#[test]
#[ignore]
fn golly_adaptive_step() {
  for f in fixtures().iter().filter(|f| f.gens <= 1 << 20) {
    run(f, Config::cpu());
  }
}
