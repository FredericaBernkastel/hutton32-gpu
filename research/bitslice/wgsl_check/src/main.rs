//! Checks that the generated `step_planes.wgsl` is accepted by naga 30 (parse + full validation)
//! when used from a compute shader. This does not execute the shader; the function's truth
//! table is checked by `verify.py` (which reads the same file as text).

const HARNESS: &str = r#"
@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst: array<u32>;

@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let i = id.x * 25u;
  let c = array<u32,5>(src[i], src[i + 1u], src[i + 2u], src[i + 3u], src[i + 4u]);
  let n = array<u32,5>(src[i + 5u], src[i + 6u], src[i + 7u], src[i + 8u], src[i + 9u]);
  let e = array<u32,5>(src[i + 10u], src[i + 11u], src[i + 12u], src[i + 13u], src[i + 14u]);
  let s = array<u32,5>(src[i + 15u], src[i + 16u], src[i + 17u], src[i + 18u], src[i + 19u]);
  let w = array<u32,5>(src[i + 20u], src[i + 21u], src[i + 22u], src[i + 23u], src[i + 24u]);
  let o = step_planes(c, n, e, s, w);
  let j = id.x * 5u;
  dst[j] = o[0]; dst[j + 1u] = o[1]; dst[j + 2u] = o[2]; dst[j + 3u] = o[3]; dst[j + 4u] = o[4];
}
"#;

fn main() {
  let path = std::env::args().nth(1).unwrap_or(concat!(env!("CARGO_MANIFEST_DIR"), "/../step_planes.wgsl").to_string());
  let func = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
  let source = format!("{func}\n{HARNESS}");
  let t = std::time::Instant::now();
  let module = match naga::front::wgsl::parse_str(&source) {
    Ok(m) => m,
    Err(e) => {
      eprintln!("PARSE ERROR:\n{}", e.emit_to_string(&source));
      std::process::exit(1);
    }
  };
  let mut validator = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty());
  match validator.validate(&module) {
    Ok(_) => {}
    Err(e) => {
      eprintln!("VALIDATION ERROR: {e:?}");
      std::process::exit(1);
    }
  }
  let f = module.functions.iter().find(|(_, f)| f.name.as_deref() == Some("step_planes")).expect("step_planes not found").1;
  println!(
    "{path}: naga {} parse + validate OK in {:.2}s; step_planes has {} arguments, {} expressions",
    "30.0.2",
    t.elapsed().as_secs_f64(),
    f.arguments.len(),
    f.expressions.len()
  );
}
