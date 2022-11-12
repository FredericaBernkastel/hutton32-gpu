//!include ./kernel/rules/hutton32.rule.wgsl

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