include!("src/lib.rs");

fn main() {
  println!("cargo:rerun-if-changed=src/kernel");

  if build_target::target_arch().unwrap() == build_target::Arch::WASM32 {
    use std::{env};
    use std::io::{Write, BufWriter};
    use std::fs::File;

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest_path = std::path::Path::new(&out_dir).join("gpu_kernel.wgsl");

    let data = compile_shader();

    let mut f = BufWriter::new(File::create(&dest_path).unwrap());
    write!(f, "{}", data).unwrap();
  }
}