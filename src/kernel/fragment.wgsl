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
  var color = vec3<f32>(hutton32_colors[cell]) / 255.0;
  if ((cell >= 9u && cell <= 12u)) {
    //color = (color + vec3((color.x + color.y + color.z) / 3.0)) / 2.0;
    color = color * 0.85;
  }

  color = pow(color, vec3(2.2));
  /*if (highlight_signals) {
    let offset = pixel.y * uniforms.simulation_dimm.x + pixel.x;
    let cell_diff = simulation_buffer[offset];
    let cell_diff = f32((cell_diff >> 8u) != (cell_diff & 0xffu));
    return vec4(max(vec3<f32>(hutton32_colors[cell]) / 255.0 * 0.1, vec3(cell_diff, 0.0, 0.0)), f32(boundary));
  }*/
  return vec4(color, f32(boundary));
}