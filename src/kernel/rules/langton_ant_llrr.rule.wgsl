struct CellState {
  color: u32, // [u2]
  has_ant: bool,
  ant_direction: u32, // [u2]
}

fn u32_to_cell_state(state: u32) -> CellState {
  // [ant_direction, u2] [has_ant, u1] [color, u2]
  return CellState(
    state & 3u,
    bool(state >> 2u),
    (state >> 3u) & 3u
  );
}

fn cell_state_to_u32(state: CellState) -> u32 {
  return state.color | u32(state.has_ant) << 2u | state.ant_direction << 3u;
}

fn new_direction(state: CellState, direction: u32) -> u32 {
  // let up = 0;
  // let right = 1;
  // let down = 2;
  // let left = 3;

  if (state.color == 0u || state.color == 1u) { // left rotation
    switch direction {
      case 0u: { return 3u; }
      case 1u: { return 0u; }
      case 2u: { return 1u; }
      case 3u: { return 2u; }
      default: { return 0u; }
    }
  }
  else if (state.color == 2u || state.color == 3u) { // right rotation
    return (direction + 1u) % 4u;
  } else {
    return 0u;
  }
}

fn langton_ant_llrr(c: u32, n: u32, s: u32, e: u32, w: u32) -> u32 {
  let c = u32_to_cell_state(c);
  let n = u32_to_cell_state(n);
  let s = u32_to_cell_state(s);
  let e = u32_to_cell_state(e);
  let w = u32_to_cell_state(w);

  let color = (c.color + u32(c.has_ant)) % 4u;
  let has_ant =
    (n.has_ant && n.ant_direction == 2u) ||
    (e.has_ant && e.ant_direction == 3u) ||
    (s.has_ant && s.ant_direction == 0u) ||
    (w.has_ant && w.ant_direction == 1u);

  var ant_direction = 0u;
  if (n.has_ant && n.ant_direction == 2u) {
    ant_direction = n.ant_direction;
  } else if (e.has_ant && e.ant_direction == 3u) {
    ant_direction = e.ant_direction;
  }
  else if (s.has_ant && s.ant_direction == 0u) {
    ant_direction = s.ant_direction;
  }
  else if (w.has_ant && w.ant_direction == 1u) {
    ant_direction = w.ant_direction;
  }

  if (has_ant) {
    ant_direction = new_direction(c, ant_direction);
  }

  return cell_state_to_u32(CellState (
    color, has_ant, ant_direction
  ));
}