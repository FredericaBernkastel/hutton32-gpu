//!define hutton32_colors
//!include ./kernel/hutton32.wgsl

struct Uniforms {
  display_x_range: vec2<f32>,
  display_y_range: vec2<f32>,

  simulation_dimm: vec2<u32>,

  time: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> simulation_buffer: array<u32>;
@group(0) @binding(2) var<storage, read_write> hutton32_lut: array<atomic<u32>>;


// vertex ---------------------

struct VertexInput {
  @location(0) position: vec2<f32>
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) tex_coords: vec2<f32>,
};

@vertex fn vs_main(
  //in: VertexInput,
  @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
  var out: VertexOutput;

  var vertices = array<vec2<f32>, 6> (
    vec2<f32>(-1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>( 1.0,  1.0),
    vec2<f32>(-1.0, -1.0),
    vec2<f32>( 1.0, -1.0)
  );

  let position = vertices[in_vertex_index];
  out.position = vec4<f32>(position.xy, 0.0, 1.0); // -1.0..1.0
  out.tex_coords = position.xy / 2.0 + 0.5; // 0.0..1.0
  return out;
}



// fragment ---------------------

fn sim_boundary_check(xy: vec2<u32>) -> bool {
  return
    xy.x >= 0u && xy.x < uniforms.simulation_dimm.x &&
    xy.y >= 0u && xy.y < uniforms.simulation_dimm.y;
}

fn get_cell(xy: vec2<u32>, parity: bool) -> u32 {
  /*                  simulation_buffer[i]: u32;

          parity == true                        parity == false

      unused                                unused
        |    next iteration                   |  current iteration
      - - -     |   current iteratiom       - - -     |  next iteratiom
      |    |    |    |                      |    |    |    |
     [u8] [u8] [u8] [u8]                   [u8] [u8] [u8] [u8]
  */
  let offset = xy.y * uniforms.simulation_dimm.x + xy.x;
 return
    //(u32(sim_boundary_check(xy)) * 0xffffffffu) & // if sim_boundary_check(xy)
    (simulation_buffer[offset] >> (u32(!parity) * 8u)) &
    0xffu;
}

fn set_cell(cell_record: u32, next: u32, parity: bool) -> u32 {
  let shift = u32(parity) * 8u;
  let next = next << shift;
  let current = cell_record & (0xffu << (8u - shift));
  return next | current;
}

@fragment fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let scale = vec2(
     uniforms.display_x_range[1] - uniforms.display_x_range[0],
    -uniforms.display_y_range[1] + uniforms.display_y_range[0]
  );

  let xy = (in.tex_coords * scale + vec2(uniforms.display_x_range[0], -uniforms.display_y_range[0]))
    / vec2<f32>(uniforms.simulation_dimm);
  let boundary = u32(xy.x >= 0.0 && xy.x <= 1.0 && xy.y >= 0.0 && xy.y <= 1.0);
  let pixel = vec2<u32>(xy * vec2<f32>(uniforms.simulation_dimm));
  let offset = pixel.y * uniforms.simulation_dimm.x + pixel.x;
  let parity = bool(simulation_buffer[offset] >> 16u);
  let cell = get_cell(pixel, parity) & (boundary * 0xffu);
  /*if (highlight_signals) {
    let offset = pixel.y * uniforms.simulation_dimm.x + pixel.x;
    let cell_diff = simulation_buffer[offset];
    let cell_diff = f32((cell_diff >> 8u) != (cell_diff & 0xffu));
    return vec4(max(vec3<f32>(hutton32_colors[cell]) / 255.0 * 0.1, vec3(cell_diff, 0.0, 0.0)), f32(boundary));
  }*/
  return vec4(vec3<f32>(hutton32_colors[cell]) / 255.0, f32(boundary));
}



// compute ---------------------

@compute @workgroup_size(1) fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let xy = vec2<i32>(global_id.xy);
  let offset = u32(xy.y) * uniforms.simulation_dimm.x + u32(xy.x);
  let current_record = simulation_buffer[offset];
  let parity = bool(current_record >> 16u);

  //let cell = game_of_life(xy, parity);
  let c = get_cell(vec2<u32>(xy), parity);
  let n = get_cell(vec2<u32>(xy + vec2( 0, -1)), parity);
  let s = get_cell(vec2<u32>(xy + vec2( 0,  1)), parity);
  let e = get_cell(vec2<u32>(xy + vec2( 1,  0)), parity);
  let w = get_cell(vec2<u32>(xy + vec2(-1,  0)), parity);

  //let cell = hutton32(c, n, s, e, w);
  let lut_offset = w | s << 5u | e << 10u | n << 15u | c << 20u;
  let cell = (hutton32_lut[lut_offset / 4u] >> ((lut_offset % 4u) * 8u)) & 0xFFu;

  let next_record = set_cell(current_record, cell, parity);
  simulation_buffer[offset] = next_record | (u32(!parity) << 16u);
}

@compute @workgroup_size(1) fn compute_lut(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let width = 8192u; // 2 ^ 13
  let offset = global_id.y * width + global_id.x;

  // c(5)-n(5)-e(5)-s(5)-w(5) = c'(8)
  let w = offset & 0x1Fu;
  let s = (offset >> 5u) & 0x1Fu;
  let e = (offset >> 10u) & 0x1Fu;
  let n = (offset >> 15u) & 0x1Fu;
  let c = (offset >> 20u) & 0x1Fu;
  var cell = hutton32(c, n, s, e, w);

  // u32            u32
  // u8 u8 u8 u8    u8 u8 u8 u8

  cell = cell << ((offset % 4u) * 8u);

  atomicOr(&hutton32_lut[offset / 4u], cell);
}