//! GHL: GPU HashLife for every rule of Golly's HashLife family.
//!
//! A [`Universe`] holds a rule and its cells and advances them with the engines of the paper:
//! the host recursion (HashLife evaluated level-synchronously, on base tiles stored as bit-planes),
//! the GPU's plain and shared-tile kernels, and the GPU's memo engine (the positional sweep of the
//! hit path and the level-synchronous miss path). A scheduler picks the engine per epoch (paper
//! Algorithm 1) and per region (Algorithm 3); every engine is bit-identical to per-cell stepping.
//!
//! ```no_run
//! use ghl::{Config, Pattern, Rule, Universe};
//! let rule = Rule::named("B3/S23")?;
//! let mut u = Universe::new(rule, Config::default())?;
//! let loaded = Pattern::load("glider.rle")?;
//! u.load(&loaded.pattern, (0, 0))?;
//! let report = u.step(1 << 20);
//! println!("{} gens/s, population {}", report.gens_per_s(), u.population());
//! # Ok::<(), ghl::Error>(())
//! ```
//!
//! What the engine does is visible while it runs: [`Universe::observer`] hands a UI thread the
//! counters of every epoch, the maps of the engine's data structures and the trace of a step
//! ([`inspect`]).

pub mod cpu;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod inspect;
pub mod io;
pub mod query;
pub mod rule;
pub mod sched;
pub mod tile;

mod error;
mod universe;

pub use error::{Error, Result};
pub use inspect::{Observer, Pending, Sample, StepTrace, Taps};
pub use io::{Format, Grid, Loaded, Macrocell, Pattern};
pub use query::{EngineKind, Fingerprint, Memory, Rect, Stats, StepReport, TileSummary};
pub use rule::{Neighbourhood, Network, Pos, Rule};
pub use sched::{Controller, Regime};
pub use tile::{TileId, NONE};
pub use universe::{Config, Core, Engines, Root, Universe};

#[cfg(feature = "gpu")]
pub use gpu::{GpuContext, GpuView};
