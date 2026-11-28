//! Rule formats shared by the crates of this workspace:
//!
//! * [`Pos`] and [`Neighbourhood`]: the cells of a radius-1 neighbourhood (von Neumann, Moore,
//!   hexagonal, one-dimensional), their order in a table index and in a Golly transition;
//! * [`Table`]: a Golly rule table (`@TABLE`), its transitions under its symmetries ([`Case`]) and
//!   their meaning ([`Interp`]: the next state of a neighbourhood);
//! * [`Netlist`]: a rule as NOT, AND, OR and XOR gates over the bits of a neighbourhood, with its
//!   text form, evaluation on machine words and emission as WGSL or Rust.
//!
//! `ruletable2netlist` turns a `Table` into a `Netlist`; `ghl` runs rules in either form.

pub mod nbhd;
pub mod netlist;
pub mod table;

pub use nbhd::{Neighbourhood, Pos};
pub use netlist::{Gate, Netlist, Op, Signal};
pub use table::{Case, Field, Interp, Out, StateSet, Table};

/// Bits a state of a rule with `states` states takes: states are coded in binary as numbered.
pub fn bits_for(states: usize) -> usize {
  (usize::BITS - (states.max(2) - 1).leading_zeros()) as usize
}
