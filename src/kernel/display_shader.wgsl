struct VertexInput {
  @location(0) position: vec2<f32>
};

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) tex_coords: vec2<f32>,
};

struct Uniforms {
  x_range: vec2<f32>,
  y_range: vec2<f32>,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

@vertex fn vs_main(
  //in: VertexInput,
  @builtin(vertex_index) in_vertex_index: u32,
) -> VertexOutput {
  var out: VertexOutput;

  var vertices: array<vec2<f32>, 6> = array<vec2<f32>, 6> (
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

fn mandelbrot(uv: vec2<f32>) -> f32 {
  var z = vec2(0.0, 0.0);
  let max_iter = 128;

  for (var i: i32 = 0; i < max_iter; i++) {
    z = vec2(z.x*z.x - z.y*z.y, 2.0 * z.x * z.y) + uv;
    if(length(z) > 1.0e+16) {
      return f32(i) / f32(max_iter);
    }
  }

  return 1.0;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
  let scale = vec2(
    uniforms.x_range[1] - uniforms.x_range[0],
    uniforms.y_range[1] - uniforms.y_range[0]
  );

  let xy = in.tex_coords * scale + vec2(uniforms.x_range[0], uniforms.y_range[0]);
  let color = mandelbrot(xy);
  return vec4(vec3(color), color);
}

@compute @workgroup_size(1) fn compute_main(@builtin(global_invocation_id) global_id: vec3<u32>) {

}