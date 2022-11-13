struct Uniforms {
  display_x_range: vec2<f32>,
  display_y_range: vec2<f32>,
  simulation_dimm: vec2<u32>,
};

struct VertexInput {
  @location(0) position: vec2<f32>
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) tex_coords: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> simulation_buffer: array<u32>;
@group(0) @binding(2) var<storage, read_write> hutton32_lut: array<atomic<u32>>;


//!define hutton32_colors

//!include ./src/kernel/util.wgsl
//!include ./src/kernel/vertex.wgsl
//!include ./src/kernel/fragment.wgsl
//!include ./src/kernel/compute.wgsl