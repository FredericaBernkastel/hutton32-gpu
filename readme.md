### Hutton32 CA simulation on GPU
Run with: `cargo run --release`

```
profiling for {
  device: GeForce GTX 1060 3GB
  simulation_dimm: 633x449,
  generations: 8192,
  iters/frame: 32
}

hutton32, naive branching -> 7.803s
hutton32, LUT 32MB        -> 5.695s (memory bound)
```

![](doc/scr.webp)

### Limitations
- Currently, `wgpu::Device::create_shader_module` will panic, if the source code is invalid. As such, it is impossible to implement a robust runtime shader recompilation.  
This must be addressed by wgpu developers.
- Even though `egui` supports compiling on `wasm32` target, currently `wgpu` is being emulated in browser using WebGL2. This enforces major restrictions on shader capabilities, most importantly lack of storage buffer type - thus rendering compute stage to be useless in any practical scenarios. This might change with the stabilization of [WebGPU](https://caniuse.com/webgpu), hence WebGL emulation layer no longer necessary - finally allowing us to perform scientific gpu computations both on native and web.