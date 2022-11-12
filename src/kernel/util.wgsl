fn sim_boundary_check(xy: vec2<u32>) -> bool {
  return
    xy.x >= 0u && xy.x < uniforms.simulation_dimm.x &&
    xy.y >= 0u && xy.y < uniforms.simulation_dimm.y;
}

fn get_cell(xy: vec2<u32>, parity: bool) -> u32 {
  /*                  simulation_buffer[i]: u32;

          parity == true                        parity == false

            next iteration                      current iteration
    unused      |   current iteratiom     unused      |  next iteratiom
      |         |    |                      |         |    |
     [u8] [u8] [u8] [u8]                   [u8] [u8] [u8] [u8]
           |                                     |
        parity                                parity
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
