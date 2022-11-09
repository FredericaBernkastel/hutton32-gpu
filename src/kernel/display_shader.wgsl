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
  let offset = xy.y * uniforms.simulation_dimm.x + xy.x;
  return
    (u32(sim_boundary_check(xy)) * 0xffffffffu) & // if sim_boundary_check(xy)
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
  if (uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0) {
    let uv = vec2<u32>(uv * vec2<f32>(uniforms.simulation_dimm));
    //let offset = uv.y * uniforms.simulation_dimm.x + uv.x;
    //let cell = simulation_buffer[offset];
    let cell = get_cell(uv);
    let color = vec4(vec3(f32(cell & 1u)), 1.0);
    return color;
  } else {
    return vec4(0.0);
  }
}



// compute ---------------------

fn mandelbrot(uv: vec2<f32>) -> f32 {
  var z = vec2(0.0, 0.0);
  let max_iter = 256;

  for (var i: i32 = 0; i < max_iter; i++) {
    z = vec2(z.x*z.x - z.y*z.y, 2.0 * z.x * z.y) + uv;
    if(length(z) > 2.0) {
      return f32(i) / f32(max_iter);
    }
  }

  return 1.0;
}

fn game_of_life(xy: vec2<u32>) {

  /* simulation_buffer[i]: u32; time % 2 == 0

      unused
        |    next iteration
      - - -     |   current iteratiom
      |    |    |    |
     [u8] [u8] [u8] [u8]
  */
  var moore_neighbourhood = array<vec2<i32>, 8> (
    vec2<i32>(-1, -1),
    vec2<i32>( 0, -1),
    vec2<i32>( 1, -1),
    vec2<i32>(-1,  0),
    vec2<i32>( 1,  0),
    vec2<i32>(-1,  1),
    vec2<i32>( 0,  1),
    vec2<i32>( 1,  1),
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
  game_of_life(vec2(global_id.xy));
  //let color = mandelbrot(uv) * 255.0;
  //simulation_buffer[offset] = u32(color);
}