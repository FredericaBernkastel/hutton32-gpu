# ghl — GPU HashLife

A Rust library that runs every rule of Golly's HashLife family with the method of
[`paper.md`](../paper.md): bit-sliced brute force below a memo floor set by the rule's exchange rate
`rho`, a positional sweep of tile IDs as the hit path, and the level-synchronous recursion as the miss
path, on a GPU through wgpu and on the host. Every engine is bit-identical to per-cell stepping.

The roadmap is [`implementation.md`](../implementation.md); this crate implements it. Drawing, input
and everything else a user sees belong to a UI crate that depends on this one.

## Using it

```rust
use ghl::{Config, Pattern, Rule, Universe};

let rule = Rule::named("Hutton32b")?;                 // or "B3/S23", Rule::load("x.rule"), ...
let mut u = Universe::new(rule, Config::default())?;  // the GPU when there is one
let loaded = Pattern::load("counter.rle")?;
u.load(&loaded.pattern, (0, 0))?;
let report = u.step(1 << 20);                          // or u.step_for(frame_budget)
println!("{:.3e} cell-updates/s", report.cell_updates_per_s());

let obs = u.observer();                                // Clone + Send + Sync, for a UI thread
let sample = obs.sample();                             // the counters of the last epoch
let ids = obs.ids(ghl::Rect::new(0, 0, 256, 256), 3).wait();
# Ok::<(), ghl::Error>(())
```

`Config::cpu()` runs the host recursion only; `Config { regime: Some(Regime::Memo), .. }` pins an
engine; `Universe::with_gpu` shares a UI's `wgpu::Device` and `Queue`, and `gpu_view` hands it the
engine's buffers (the dense grid, or the ID grid, node table and tile store) to draw from in place.

wgpu's `WGPU_BACKEND` variable (`vulkan`, `dx12`, `metal`) restricts the backends a universe tries; a
backend whose driver or layers fail while the instance is created takes the process with it, and this
is how to avoid it.

Features: `gpu` (default; wgpu 30), `png` (the archived application's palette PNG), `compile`
(`Rule::load_compiled`: a `@TABLE` rule compiled into a gate network by the workspace's
`ruletable2netlist`).

## What is where

| Module | What it holds | Roadmap |
|---|---|---|
| `rule` | B/S notation (H, V, B0 conversion as Golly does), `@TABLE` and `@TREE` on every neighbourhood and symmetry, the canonical Hutton32 and Hutton32b tables, gate networks (the Life-like adder tree, the 367-gate Hutton32b circuit, the 597-gate Hutton32 circuit compiled by `ruletable2netlist`) | Stage 0, Stage 5 |
| `io` | RLE (extended states), macrocell read and write (both flavours), palette PNG | Stage 0 |
| `cpu` | the host recursion on bit-plane base tiles (`hl`), the brute force on `u64` words with an AVX2 path (`leaf`, `simd`), the positional sweep (`flat`), reclaiming (`reclaim`) | Stages 0, 2, 5 |
| `gpu` | device and calibration, WGSL generated from the rule (`kernels`): plain and shared-tile kernels for networks, byte-LUT and rule trees (`brute`); memo tables with the claim protocol (`memo`); the sweep with active blocks (`sweep`); the miss path (`miss`); grids and tables beyond one binding (`shard`) | Stages 1–3, 5 |
| `sched` | Algorithm 1 (regime from the active area, racing), Algorithm 3 and the price-based controller, the region model's prices (eq. 7), the step rule, the sweep tile (eq. 5), cycle detection | Stage 4 |
| `inspect` | `Observer`, `Sample` per epoch with its history, maps answered at once or at the next epoch boundary (`Pending`), `StepTrace` | Section 8 |
| `query`, `universe` | the interface of roadmap Section 7 | Section 7 |

## Checks

```
cargo test -p ghl --release                        # everything below except the long Golly runs
cargo test -p ghl --release --test golly -- --ignored   # final states of 2^20..2^24 generations against Golly 5.1
cargo test -p ghl --release --test controller -- --ignored --nocapture   # the controller's gates (model times)
cargo run  -p ghl --release --example overview     # the overview figure's data through the public API
python ghl/tests/fixtures/compare_overview.py      # ... key by key against the harness's overview.json
cargo bench -p ghl --bench stages                  # the stages' throughput targets (idle machine)
```

* `tests/stage0.rs`: the canonical networks on all 2^25 neighbourhoods, every Life-like network on all
  of its neighbourhoods, the harness's 22 self-test configurations, the universe against per-cell
  stepping under every step size and controller, B0 rules, edits, files, cycles, plain-cells blocks.
* `tests/golly.rs`: final states against fingerprints recorded once with bgolly
  (`fixtures/record_golly.py`, `fixtures/golly.tsv`).
* `tests/controller.rs`: the price-based controller in model times. `price_based_controller_acceptance`
  (the roadmap's three configurations: no universe more than 5 % above the uniform floor, the sums
  within 20 % of the engine per region chosen in hindsight) passes;
  `price_based_controller_tables_xxvii_xxviii` (the floor criterion at every configuration of those
  tables) fails on the Unit Life Cell, which is why the controller ships disabled;
  `price_based_controller_step_rule_scales` reports the other steps. `GHL_CONTROLLER_TRACES=1` writes
  every region's two prices per step to `research/out/controller_traces/`, on which
  `fixtures/controller_sim.py` simulates the rule for other constants; `GHL_CONTROLLER_CONFIGS`
  selects configurations.
* `tests/gpu.rs`: every rule class's kernels, the memo engine with the host learning and with the
  GPU's miss path, sparse universes on active blocks, sharded tables and grids, cycles, racing; all
  against the host recursion.
* `examples/overview.rs`: the keys `palette`, `patterns`, `structures`, `structures_life`, `sweep`,
  `table`, `table_life`, `learning`, `waves` and `controller` equal the harness's.

## Status

See the last section of [`implementation.md`](../implementation.md).
