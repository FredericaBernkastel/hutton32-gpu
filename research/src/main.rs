//! Measurement harness for `paper.md`. Run `cargo run --release -- help`.
#![allow(dead_code)]

mod ca;
mod cmd_adapt;
mod cmd_bitslice;
mod cmd_brute2;
mod cmd_gpu;
mod cmd_life;
#[cfg(feature = "cubecl")]
mod cmd_life_cubecl;
mod cmd_insert;
mod cmd_lut;
mod cmd_overview;
mod cmd_sparse;
mod cmd_stats;
mod cmd_symmetry;
mod flat;
mod gpu;
mod hl;
mod mc;
mod pattern;
mod rule;
mod rulespec;
mod sim;

use std::path::PathBuf;

pub fn repo_root() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
}
/// `research/patterns`: every pattern the report uses (see its README).
pub fn patterns_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("patterns")
}
pub fn out_dir() -> PathBuf {
  let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("out");
  std::fs::create_dir_all(&p).unwrap();
  p
}

/// The LUT for a rule. The built-in rules are computed on every call (a third of a second), so
/// that no cache can outlive a change to `rule.rs`. A table's LUT takes much longer and is cached
/// in `out/`, next to a stamp of the table file's contents.
pub fn load_lut(spec: &rulespec::RuleSpec) -> ca::Rule {
  let Some(stamp) = spec.source_stamp() else {
    return spec.rule_from(spec.build());
  };
  let path = out_dir().join(format!("lut_{}.bin", spec.cache_name()));
  let stamp_path = path.with_extension("src");
  if std::fs::read_to_string(&stamp_path).is_ok_and(|s| s == stamp) {
    if let Ok(data) = std::fs::read(&path) {
      if data.len() == rule::LUT_SIZE {
        return spec.rule_from(data);
      }
    }
  }
  let lut = spec.build();
  std::fs::write(&path, &lut).unwrap();
  std::fs::write(&stamp_path, &stamp).unwrap();
  spec.rule_from(lut)
}

/// `--key value` lookup in argv.
pub fn opt(args: &[String], key: &str) -> Option<String> {
  args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}
pub fn opt_or<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> T {
  opt(args, key).map(|s| s.parse().unwrap_or_else(|_| panic!("bad value for {key}"))).unwrap_or(default)
}
pub fn flag(args: &[String], key: &str) -> bool {
  args.iter().any(|a| a == key)
}

/// `--repeats N` (default 1): time a measurement N times. With N > 1 it is run once more first,
/// untimed, as a warm-up. `once` must reset its own input and return the measured seconds; the
/// state it leaves behind is the last run's, so a verification after it checks the last run.
pub fn repeat_timed(args: &[String], mut once: impl FnMut() -> f64) -> Vec<f64> {
  let n: usize = opt_or(args, "--repeats", 1);
  if n > 1 {
    once();
  }
  (0..n.max(1)).map(|_| once()).collect()
}

/// Median of the run times, and a note on the spread of the rate `work / seconds` when there
/// was more than one run.
pub fn median_and_spread(secs: &[f64], work: f64, unit: &str) -> (f64, String) {
  let mut v = secs.to_vec();
  v.sort_by(|a, b| a.partial_cmp(b).unwrap());
  let n = v.len();
  let med = if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) };
  if n == 1 {
    return (med, String::new());
  }
  (med, format!(" | {n} runs after a warm-up, median of {unit}: min {:.4e}, max {:.4e}", work / v[n - 1], work / v[0]))
}

fn main() {
  let args: Vec<String> = std::env::args().skip(1).collect();
  let cmd = args.first().map(String::as_str).unwrap_or("help");
  let rest = if args.is_empty() { &args[..] } else { &args[1..] };
  match cmd {
    "lut" => cmd_lut::run(rest),
    "selftest" => cmd_stats::selftest(rest),
    "flat" => cmd_stats::flat(rest),
    "symmetry" => cmd_symmetry::run(rest),
    "root" => cmd_stats::root(rest),
    "fingerprint" => cmd_stats::fingerprint(rest),
    "export" => cmd_stats::export(rest),
    "ruletree" => cmd_stats::ruletree(rest),
    "rulediff" => cmd_stats::rulediff(rest),
    "adapt" => cmd_adapt::run(rest),
    "overview" => cmd_overview::run(rest),
    "gpu-baseline" => cmd_gpu::baseline(rest),
    "gpu-brute" => cmd_gpu::brute(rest),
    "gpu-sweep" => cmd_gpu::sweep(rest),
    "gpu-probe" => cmd_gpu::probe(rest),
    "gpu-latency" => cmd_gpu::latency(rest),
    "gpu-insert" => cmd_insert::run(rest),
    "gpu-brute2" => cmd_brute2::run(rest),
    "gpu-bitslice" => cmd_bitslice::run(rest),
    "gpu-window" => cmd_bitslice::window(rest),
    "gpu-tile" => cmd_bitslice::tile(rest),
    "gpu-sparse" => cmd_sparse::run(rest),
    "gpu-life" => cmd_life::run(rest),
    "life-soup" => cmd_life::soup_cmd(rest),
    #[cfg(feature = "cubecl")]
    "cubecl-life" => cmd_life_cubecl::run(rest),
    _ => {
      eprintln!(
        "{}",
        [
          "CPU (rule, reference engine, statistics):",
          "  lut [--rule hutton32|hutton32b] [--wgsl F] [--against R]   run a WGSL rule file on the GPU for all 2^25",
          "                                               neighbourhoods; compare with the CPU port or with rule R",
          "  rulediff --a R --b R                         compare two rules entry by entry",
          "  selftest [--pattern P]                       reference engine vs per-cell stepping",
          "  flat --pattern P [--lt L --a A --gens N]     miss-rate / learning curve of the flat tile CA",
          "  symmetry --pattern P [--a A --lt L --gens N --check N]  memo entries shared up to rotation / mirroring",
          "  root --pattern P [--k K --a A --gens N]      level-synchronous HashLife with batch statistics",
          "  fingerprint --pattern P                      population / bounding box / content hash",
          "  export --pattern P --out F [--as RULE]       write RLE",
          "  ruletree --rule R --name N [--out F]         write a Golly @TREE rule from a LUT",
          "  adapt --pattern P | --compose SPEC [--gens N --k K --rl R --a A]  adaptive subdivision / engine choice",
          "  overview [--out F]                           the data of the overview diagram (tools/ghl_diagram.py), as JSON",
          "GPU (all results are checked against the CPU):",
          "  gpu-baseline [--pattern P --reps N --gens G] archived kernel, verbatim structure",
          "  gpu-brute [--pattern P --reps N --gens G]    per-cell LUT kernel done properly",
          "  gpu-brute2 [--side N --pattern P --same-src] hostile content; full vs class-compressed LUT",
          "  gpu-bitslice [--side N --pattern P --wgsl F]  bit-plane kernel from a synthesised circuit (no table)",
          "  gpu-window [--windows N --pattern P --wgsl F]  32x32 windows, 8 generations in one dispatch (miss path)",
          "  gpu-tile [--k K --tw W --th H --side N --pattern P]  K generations per pass in workgroup-shared memory",
          "  gpu-probe [--log2-slots a,b,c]               random table-entry fetch rate vs table size (to 8 GiB)",
          "  gpu-insert [--log2-slots N --n N]            concurrent find-or-insert with 32-bit CAS",
          "  gpu-latency [--n N]                          dispatch / pass / submit / readback costs",
          "  gpu-sweep --pattern P [--lt L --reps N --cycles C]   dense universe: flat tile CA (memo hit path)",
          "  gpu-sparse --pattern P [--pitch F --reps N --export-mc F]  sparse universe: flat vs active blocks",
          "  gpu-life [--side N | --pattern P [--tiled | --margin M]] [--k K --tw W --th H] [--verify --fingerprint]",
          "                                               Life-like rules: bit-sliced Moore kernel, plain or shared tile",
          "  life-soup --side S [--density D --seed N] --out F   random square soup as RLE",
          "  cubecl-life [the flags of gpu-life] [--compiler wgsl|spirv --checked]",
          "                                               gpu-life ported to CubeCL (build with --features cubecl)",
          "rules (--rule, --a, --b, --against): hutton32 | hutton32b (canonical; the WGSL files in archive/src/kernel/rules),",
          "  hutton32-v1 | hutton32b-v1 (those files as archived in commit 256ef91), table:<Golly .rule or .table file>",
          "  (the canonical tables: table:rules/Hutton32.table, table:rules/Hutton32b.rule),",
          "  or a Life-like rule in B/S notation (B3/S23, B36/S23, B2/S34H, ...: Moore, hexagonal, von Neumann)",
          "patterns (--pattern): palette PNG, RLE or macrocell; the report's are in patterns/ (default: the counter preset)",
        ]
        .join("\n")
      );
    }
  }
}
