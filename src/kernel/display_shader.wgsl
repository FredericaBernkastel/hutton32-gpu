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
  out.position = vec4<f32>(position, 0.0, 1.0); // -1.0..1.0
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
    uniforms.display_y_range[1] - uniforms.display_y_range[0]
  );

  let uv = in.tex_coords * scale + vec2(uniforms.display_x_range[0], uniforms.display_y_range[0]);
  let boundary = u32(uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0);
  let pixel = vec2<u32>(uv * vec2<f32>(uniforms.simulation_dimm));
  //let offset = pixel.y * uniforms.simulation_dimm.x + pixel.x;
  //let cell = simulation_buffer[offset];
  let cell = get_cell(pixel) & (boundary * 0xffu);
  return vec4(vec3(f32(cell & 1u)), f32(boundary));
}



// compute ---------------------

fn game_of_life(xy: vec2<u32>) {
  var moore_neighbourhood = array<vec2<i32>, 8> (
    vec2(-1, -1),
    vec2( 0, -1),
    vec2( 1, -1),
    vec2(-1,  0),
    vec2( 1,  0),
    vec2(-1,  1),
    vec2( 0,  1),
    vec2( 1,  1),
  );

  var neighbours = 0u;
  for (var i: i32 = 0; i < 8; i++) {
    neighbours += get_cell(vec2<u32>(vec2<i32>(xy) + moore_neighbourhood[i])) & 1u;
  }

  var cell = get_cell(xy);
  cell = u32(
    (cell == 1u && (neighbours == 2u || neighbours == 3u)) ||
    (cell == 0u && (neighbours == 3u))
  );
  set_cell(xy, cell);
}

@compute @workgroup_size(1) fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  //let uv = vec2<f32>(global_id.xy) / vec2<f32>(uniforms.simulation_dimm);
  game_of_life(global_id.xy);
  //let color = mandelbrot(uv) * 255.0;
  //simulation_buffer[offset] = u32(color);
}