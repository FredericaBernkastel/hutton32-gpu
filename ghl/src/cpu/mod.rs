//! The engines of the host: the recursion of paper Algorithm 2 on base tiles stored as bit-planes
//! ([`hl`]), the brute force on `u64` words that it uses below its floor ([`leaf`]), the positional
//! sweep that is the GPU hit path's oracle ([`flat`]), and the reclaiming of unreachable nodes
//! ([`reclaim`]).

pub mod flat;
pub mod hl;
pub mod leaf;
pub mod reclaim;
pub mod simd;

pub use flat::{Flat, SweepStats};
pub use hl::{fingerprint, grid_fingerprint, Block, BruteWave, Engine, EngineStats, FloorPolicy, Fingerprint, Level, LevelStats, Outcome, UniformFloor};
pub use leaf::{Leaf, PlaneGrid};
pub use reclaim::{reclaim, ReclaimStats, Remap};
