/*var<private> hutton32_colors: array<vec3<u32>, 32>
  = array<vec3<u32>, 32>(
    vec3<u32>(0, 0, 0),vec3<u32>(255, 0, 0),vec3<u32>(255, 125, 0),vec3<u32>(255, 150, 25),vec3<u32>(255, 175, 50),vec3<u32>(255, 200, 75),vec3<u32>(255, 225, 100),vec3<u32>(255, 250, 125),vec3<u32>(251, 255, 0),vec3<u32>(89, 89, 255),vec3<u32>(106, 106, 255),vec3<u32>(122, 122, 255),vec3<u32>(139, 139, 255),vec3<u32>(27, 176, 27),vec3<u32>(36, 200, 36),vec3<u32>(73, 255, 73),vec3<u32>(106, 255, 106),vec3<u32>(235, 36, 36),vec3<u32>(255, 56, 56),vec3<u32>(255, 73, 73),vec3<u32>(255, 89, 89),vec3<u32>(185, 56, 255),vec3<u32>(191, 73, 255),vec3<u32>(197, 89, 255),vec3<u32>(203, 106, 255),vec3<u32>(0, 255, 128),vec3<u32>(255, 128, 64),vec3<u32>(255, 255, 128),vec3<u32>(33, 215, 215),vec3<u32>(27, 176, 176),vec3<u32>(24, 156, 156),vec3<u32>(21, 137, 137),
  );*/
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

fn get_cell(xy: vec2<u32>) -> u32 {
  /*                  simulation_buffer[i]: u32;

          time % 2 == 0                           time % 2 != 0

      unused                                unused
        |    next iteration                   |  current iteration
      - - -     |   current iteratiom       - - -     |  next iteratiom
      |    |    |    |                      |    |    |    |
     [u8] [u8] [u8] [u8]                   [u8] [u8] [u8] [u8]
  */
  let offset = xy.y * uniforms.simulation_dimm.x + xy.x;
 return
    //(u32(sim_boundary_check(xy)) * 0xffffffffu) & // if sim_boundary_check(xy)
    (simulation_buffer[offset] >> (u32(uniforms.time % 2u != 0u) * 8u)) &
    0xffu;
}

fn set_cell(xy: vec2<u32>, state: u32) {
  let offset = xy.y * uniforms.simulation_dimm.x + xy.x;
  let shift = u32(uniforms.time % 2u == 0u) * 8u;
  let next = state << shift;
  let current = simulation_buffer[offset] & (0xffu << (8u - shift));
  simulation_buffer[offset] = next | current;
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
  //let offset = pixel.y * uniforms.simulation_dimm.x + pixel.x;
  //let cell = simulation_buffer[offset];
  let cell = get_cell(pixel) & (boundary * 0xffu);
  return vec4(vec3<f32>(hutton32_colors[cell]) / 255.0, f32(boundary));
}



// compute ---------------------

@compute @workgroup_size(1) fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let xy = vec2<i32>(global_id.xy);
  let cell = hutton32(
    get_cell(vec2<u32>(xy)),
    get_cell(vec2<u32>(xy + vec2( 0, -1))),
    get_cell(vec2<u32>(xy + vec2( 0,  1))),
    get_cell(vec2<u32>(xy + vec2( 1,  0))),
    get_cell(vec2<u32>(xy + vec2(-1,  0))),
  );
  set_cell(vec2<u32>(xy), cell);
}