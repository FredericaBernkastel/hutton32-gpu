//! The cells of a radius-1 neighbourhood and the neighbourhoods of Golly's HashLife family.

/// A cell of a radius-1 neighbourhood, relative to the centre (y grows downwards: north is y - 1).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub enum Pos {
  C,
  N,
  NE,
  E,
  SE,
  S,
  SW,
  W,
  NW,
}

impl Pos {
  /// `(dx, dy)` of this cell from the centre.
  pub fn offset(self) -> (i32, i32) {
    match self {
      Pos::C => (0, 0),
      Pos::N => (0, -1),
      Pos::NE => (1, -1),
      Pos::E => (1, 0),
      Pos::SE => (1, 1),
      Pos::S => (0, 1),
      Pos::SW => (-1, 1),
      Pos::W => (-1, 0),
      Pos::NW => (-1, -1),
    }
  }
  /// The lower-case name used in netlists and WGSL (`c`, `n`, `ne`, ...).
  pub fn name(self) -> &'static str {
    match self {
      Pos::C => "c",
      Pos::N => "n",
      Pos::NE => "ne",
      Pos::E => "e",
      Pos::SE => "se",
      Pos::S => "s",
      Pos::SW => "sw",
      Pos::W => "w",
      Pos::NW => "nw",
    }
  }
  pub fn from_name(s: &str) -> Option<Pos> {
    Some(match s {
      "c" => Pos::C,
      "n" => Pos::N,
      "ne" => Pos::NE,
      "e" => Pos::E,
      "se" => Pos::SE,
      "s" => Pos::S,
      "sw" => Pos::SW,
      "w" => Pos::W,
      "nw" => Pos::NW,
      _ => return None,
    })
  }
  pub const ALL: [Pos; 9] = [Pos::C, Pos::N, Pos::NE, Pos::E, Pos::SE, Pos::S, Pos::SW, Pos::W, Pos::NW];
}

/// The neighbourhoods of Golly's HashLife family.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Neighbourhood {
  VonNeumann,
  Moore,
  /// Golly's hexagonal neighbourhood on a square grid: Moore without NE and SW.
  Hexagonal,
  /// Left and right neighbours only: every row is an independent one-dimensional automaton.
  OneDimensional,
}

impl Neighbourhood {
  /// The cells of the neighbourhood in table-index order (field 0 is the least significant).
  pub fn fields(self) -> &'static [Pos] {
    match self {
      Neighbourhood::VonNeumann => &[Pos::W, Pos::S, Pos::E, Pos::N, Pos::C],
      Neighbourhood::Hexagonal => &[Pos::W, Pos::S, Pos::E, Pos::N, Pos::C, Pos::NW, Pos::SE],
      Neighbourhood::Moore => &[Pos::W, Pos::S, Pos::E, Pos::N, Pos::C, Pos::NW, Pos::NE, Pos::SW, Pos::SE],
      Neighbourhood::OneDimensional => &[Pos::W, Pos::E, Pos::C],
    }
  }
  /// Cells in the neighbourhood, the centre included.
  pub fn cells(self) -> usize {
    self.fields().len()
  }
  /// Neighbours, the centre excluded.
  pub fn neighbours(self) -> usize {
    self.cells() - 1
  }
  /// The order of a Golly `@TABLE` transition: the centre, then the neighbours clockwise from north.
  pub fn table_order(self) -> &'static [Pos] {
    match self {
      Neighbourhood::VonNeumann => &[Pos::C, Pos::N, Pos::E, Pos::S, Pos::W],
      Neighbourhood::Moore => &[Pos::C, Pos::N, Pos::NE, Pos::E, Pos::SE, Pos::S, Pos::SW, Pos::W, Pos::NW],
      Neighbourhood::Hexagonal => &[Pos::C, Pos::N, Pos::E, Pos::SE, Pos::S, Pos::W, Pos::NW],
      Neighbourhood::OneDimensional => &[Pos::C, Pos::W, Pos::E],
    }
  }
  /// Golly's name in `@TABLE` files.
  pub fn golly_name(self) -> &'static str {
    match self {
      Neighbourhood::VonNeumann => "vonNeumann",
      Neighbourhood::Moore => "Moore",
      Neighbourhood::Hexagonal => "hexagonal",
      Neighbourhood::OneDimensional => "oneDimensional",
    }
  }
  pub fn from_golly_name(s: &str) -> Option<Self> {
    Some(match s.trim().to_ascii_lowercase().as_str() {
      "vonneumann" => Neighbourhood::VonNeumann,
      "moore" => Neighbourhood::Moore,
      "hexagonal" => Neighbourhood::Hexagonal,
      "onedimensional" => Neighbourhood::OneDimensional,
      _ => return None,
    })
  }
  /// Position of `p` in [`Self::fields`], if it belongs to the neighbourhood.
  pub fn field_of(self, p: Pos) -> Option<usize> {
    self.fields().iter().position(|&q| q == p)
  }
}
