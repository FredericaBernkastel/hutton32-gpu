fn game_of_life(xy: vec2<u32>, parity: bool) -> u32 {
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
    neighbours += get_cell(vec2<u32>(vec2<i32>(xy) + moore_neighbourhood[i]), parity) & 1u;
  }

  var cell = get_cell(xy, parity);
  return u32(
    (cell == 1u && (neighbours == 2u || neighbours == 3u)) ||
    (cell == 0u && (neighbours == 3u))
  );
}