//! `lut`: execute a WGSL rule file on the GPU for all 2^25 neighbourhoods and compare the result
//! with the CPU port, or with any other rule the harness can load. Also reports structural facts
//! about the rule (neighbour classes).

use crate::{gpu::Gpu, opt, repo_root, rule};

/// naga 30 rejects two constructs that naga 0.10 (wgpu 0.14) accepted. These textual patches do
/// not change semantics; each is reported when applied.
fn modernize(src: &str) -> String {
  let mut s = src.to_string();
  // 1. mixing `||` and `&&` without parentheses is no longer valid WGSL.
  let a = "all_inputs_on(n,s,e,w) || xnor_crossing(n,s,e,w) && xnor_output(n,s,e,w)";
  let b = "all_inputs_on(n,s,e,w) || (xnor_crossing(n,s,e,w) && xnor_output(n,s,e,w))";
  if s.contains(a) {
    println!("  patch: parenthesised `a || b && c` ({} sites)", s.matches(a).count());
    s = s.replace(a, b);
  }
  // 2. a local `let output = output(..)` shadows the function it calls.
  let a = "let output = output(c,n,s,e,w);";
  if s.contains(a) {
    println!("  patch: renamed local `output` -> `out_state`");
    s = s.replace(a, "let out_state = output(c,n,s,e,w);");
    s = s.replace("is_sensitized(output) &&", "is_sensitized(out_state) &&");
    s = s.replace("is_excited(c) && output==0u", "is_excited(c) && out_state==0u");
  }
  s
}

/// `lut --rule hutton32b`: `archive/src/kernel/rules/hutton32b.rule.wgsl` against the CPU port.
/// `lut --rule hutton32b --wgsl <file> --against <rule spec>`: any WGSL rule file that defines
/// `fn hutton32b(c,n,s,e,w) -> u32`, executed on the GPU, against any rule the harness can load
/// (`table:<Golly table>`, or `hutton32b-v1` for the file as archived in commit 256ef91).
pub fn run(args: &[String]) {
  let variant = rule::Variant::parse(&opt(args, "--rule").unwrap_or("hutton32b".into())).expect("bad --rule");
  println!("rule: {}", variant.name());
  let against = opt(args, "--against").map(|s| crate::rulespec::RuleSpec::parse(&s).expect("bad --against"));

  let (lut, ref_name) = match &against {
    Some(spec) => (crate::load_lut(spec).table, spec.name()),
    None => {
      let t = std::time::Instant::now();
      let lut = rule::build_lut(variant);
      println!("CPU port: built 2^25 entries in {:.2}s", t.elapsed().as_secs_f64());
      std::fs::write(crate::out_dir().join(format!("lut_{}.bin", crate::rulespec::RuleSpec::Wgsl(variant).cache_name())), &lut).unwrap();
      (lut, "CPU port".to_string())
    }
  };

  // ---- WGSL on the GPU ----------------------------------------------------------------------
  let rule_path = opt(args, "--wgsl")
    .map(std::path::PathBuf::from)
    .unwrap_or_else(|| repo_root().join(format!("archive/src/kernel/rules/{}.rule.wgsl", variant.name())));
  let wgsl_name = if opt(args, "--wgsl").is_some() { rule_path.display().to_string() } else { format!("{}.rule.wgsl", variant.name()) };
  let rule_src = std::fs::read_to_string(&rule_path).expect("rule file");
  let rule_src = modernize(&rule_src);
  let wgsl = format!(
    r#"
@group(0) @binding(0) var<storage, read_write> lut: array<u32>;
{rule_src}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
  let word = id.x + id.y * 32768u;
  var acc = 0u;
  for (var k = 0u; k < 4u; k++) {{
    let offset = word * 4u + k;
    let w = offset & 0x1Fu;
    let s = (offset >> 5u) & 0x1Fu;
    let e = (offset >> 10u) & 0x1Fu;
    let n = (offset >> 15u) & 0x1Fu;
    let c = (offset >> 20u) & 0x1Fu;
    acc |= ({name}(c, n, s, e, w) & 0xFFu) << (k * 8u);
  }}
  lut[word] = acc;
}}
"#,
    name = variant.name()
  );
  let gpu = Gpu::new(opt(args, "--backend").as_deref());
  println!("GPU: {}", gpu.describe());
  let pipeline = gpu.pipeline("lut", &wgsl, "main");
  let buf = gpu.storage("lut", rule::LUT_SIZE as u64);
  let bg = gpu.bind(&pipeline, &[&buf]);
  let secs = gpu.timed(|| {
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    {
      let mut pass = enc.begin_compute_pass(&Default::default());
      pass.set_pipeline(&pipeline);
      pass.set_bind_group(0, &bg, &[]);
      // 2^23 words = 256 (workgroup) * 128 * 256
      pass.dispatch_workgroups(128, 256, 1);
    }
    gpu.queue.submit(Some(enc.finish()));
  });
  println!("GPU: {wgsl_name} evaluated 2^25 neighbourhoods in {:.3}s", secs);
  let gpu_lut = gpu.read(&buf, 0, rule::LUT_SIZE as u64);
  let mismatches = lut.iter().zip(&gpu_lut).filter(|(a, b)| a != b).count();
  println!("{ref_name} vs {wgsl_name}: {} mismatches out of {}", mismatches, rule::LUT_SIZE);
  if mismatches != 0 {
    // group by (centre, reference result, WGSL result), as `rulediff` does
    let mut groups: std::collections::BTreeMap<(u32, u8, u8), (u64, u32)> = Default::default();
    for (i, (a, b)) in lut.iter().zip(&gpu_lut).enumerate().filter(|(_, (a, b))| a != b) {
      let e = groups.entry(((i >> 20) as u32, *a, *b)).or_insert((0, i as u32));
      e.0 += 1;
    }
    println!("  centre  ref->  wgsl->     count   example (c n e s w)");
    for ((c, ra, rb), (count, ex)) in groups.iter().take(40) {
      println!(
        "  {:>6}  {:>5}  {:>6}  {:>8}   {} {} {} {} {}",
        c, ra, rb, count, (ex >> 20) & 31, (ex >> 15) & 31, (ex >> 10) & 31, (ex >> 5) & 31, ex & 31
      );
    }
    std::process::exit(1);
  }
  if against.is_some() {
    return;
  }

  // ---- structure of the rule ----------------------------------------------------------------
  let identity = (0..rule::LUT_SIZE).filter(|&i| lut[i] as usize == (i >> 20)).count();
  println!("entries with f(c,..)=c: {:.2}%", 100.0 * identity as f64 / rule::LUT_SIZE as f64);
  // neighbour equivalence classes: two values of a neighbour are equivalent if swapping them
  // never changes the output, for any centre and any other neighbours.
  for (name, shift) in [("w", 0u32), ("s", 5), ("e", 10), ("n", 15), ("c", 20)] {
    let mut class = [usize::MAX; 32];
    let mut reps: Vec<u32> = Vec::new();
    for v in 0..32u32 {
      'find: {
        for (ci, &r) in reps.iter().enumerate() {
          let mut same = true;
          for rest in 0..(1u32 << 20) {
            // spread the 20 "other" bits around the 5-bit hole at `shift`
            let lo = rest & ((1 << shift) - 1);
            let hi = (rest >> shift) << (shift + 5);
            let base = hi | lo;
            if lut[(base | v << shift) as usize] != lut[(base | r << shift) as usize] {
              same = false;
              break;
            }
          }
          if same {
            class[v as usize] = ci;
            break 'find;
          }
        }
        class[v as usize] = reps.len();
        reps.push(v);
      }
    }
    println!("position {name}: {} equivalence classes: {:?}", reps.len(), class);
  }
}
