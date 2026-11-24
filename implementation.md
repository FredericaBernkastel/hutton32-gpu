# GHL library: implementation roadmap

This document turns the algorithm of [`paper.md`](paper.md), GPU HashLife (GHL), into a library. It replaces the implementation plan of the previous version of the paper (§9 "Implementation plan" at commit `b7b9693`), keeps its stages and checks where they still hold, and folds in what the paper added since: the bit-plane kernel, the shared-tile kernel, the per-region controller and the choice of depth by active area.

The deliverable is a Rust library. Drawing, input, editing tools and everything else a user sees belong to a separate UI project that depends on it; Section 7 defines what that project gets.

Throughput targets are in cell-updates/s over the pattern's bounding box, with generations/s in parentheses, as in the paper. "Paper Table N" refers to the current `paper.md`; `reproduce.sh <target>` refers to `research/reproduce.sh`, whose harness commands serve as the reference implementation and the test oracle throughout.

## 1. Scope

**In the library**

* Rules: Hutton32 and Hutton32b from the canonical tables in `research/rules`, and any other radius-1, 32-state rule given as a Golly table.
* Pattern I/O: RLE (extended states), macrocell, and the archived application's palette PNG.
* Simulation: the full GHL algorithm (paper Algorithms 1–3) on a GPU through wgpu, and the host recursion on the CPU, with results bit-identical to per-cell stepping on every path.
* Queries a front end needs: generation, population, bounding box, cells of a region, per-tile summaries for zoomed-out views, fingerprints, engine statistics.
* Edits: setting cells and pasting patterns between steps.

**Not in the library**

* Windows, event loops, egui or any other GUI toolkit, rendering pipelines, colour schemes beyond the rule's palette, undo history, file dialogs.
* The measurement harness. `research/` stays as it is; the library copies what it needs and is tested against it.

## 2. What exists and what is reused

Paper Table IX lists where each component of GHL stands. In terms of code:

| Harness code (`research/`) | Status | In the library |
|---|---|---|
| `rule.rs`, `rulespec.rs` | CPU rule implementations, Golly table loader | `rule`: keep the table loader and the canonical rules only |
| `pattern.rs`, `mc.rs` | palette PNG, RLE reader and writer, macrocell reader | `io`: add a macrocell writer (exists in `cmd_sparse.rs`) |
| `sim.rs` | per-cell stepping | test oracle only |
| `hl.rs` | reference engine: hash-consed levels, memo floor, level-synchronous `solve`, root stepping, `result_at`, fingerprints | `cpu`: the host recursion; also the oracle for every GPU path |
| `flat.rs` | positional sweep on the CPU | test oracle for the GPU sweep |
| `bitslice/step_planes.wgsl`, `step_planes.rs`, `circuit.netlist` | the 367-gate circuit, verified on all $2^{25}$ neighbourhoods | `gpu::brute` and `cpu::leaf`, included as generated artifacts |
| `cmd_bitslice.rs` (`KERNEL`, `tile_kernel`) | plain bit-plane kernel; shared-tile kernel with $k$ generations per dispatch | `gpu::brute` |
| `cmd_gpu.rs` (`BRUTE_WGSL`) | byte-LUT kernel | `gpu::brute` fallback for rules without a circuit |
| `cmd_gpu.rs` (`SWEEP_WGSL`, `probe_wgsl`) | hit-path sweep; probe kernel sharded across bindings | `gpu::sweep`, `gpu::memo` |
| `cmd_insert.rs` | lock-free find-or-insert with 32-bit CAS (claim protocol) | `gpu::memo` |
| `cmd_sparse.rs` | active-block sweep, macrocell writer | `gpu::sweep`, `io` |
| `cmd_adapt.rs` | per-region floors and engines, cost model, controller (model only) | `sched`: the policy logic; the costs come from live counters instead of the model |
| `cmd_gpu.rs` (`gpu-latency`) | dispatch, pass, submit and read-back costs; fixed-capacity wave | `gpu::calibrate` (Section 5.2) |
| `cmd_bitslice.rs` (`WINDOW_KERNEL`), `cmd_brute2.rs` | one thread per window; class-compressed table | not used: both lose to the kernels above (paper Section VI-C) |

Not built anywhere yet: the GPU miss path as one pipeline (expand, evaluate on an atlas, intern, publish), overflow and replay, the racing of executors, table rebuild and cycle detection.

## 3. Crate layout

A new crate `ghl/` next to `research/`, in a workspace at the repository root (`research/` joins it unchanged; `archive/` stays out, it builds against wgpu 0.14).

```
ghl/
  Cargo.toml            wgpu 30, bytemuck, rustc-hash; image behind the "png" feature
  src/
    lib.rs              Universe, Config, errors; re-exports
    rule/               Rule, Golly table loader, canonical tables (include_bytes!), palette
    io/                 rle, macrocell (read + write), png (feature "png")
    tile.rs             TileId, base tiles of 8x8 cells as five bit-planes (40 B), content hashing
    cpu/
      hl.rs             host recursion: hash-consed levels, solve(), root stepping, result_at()
      leaf.rs           bit-sliced evaluation of the circuit on u64 words (AVX2 later)
      reclaim.rs        mark reachable nodes, rebuild
    gpu/
      context.rs        device, queue, limits; accepts an external Device/Queue
      shard.rs          buffers and tables larger than one 2 GiB binding
      brute.rs          plain bit-plane kernel, shared-tile kernel, byte-LUT fallback
      memo.rs           memo tables, claim protocol, probe, tile store
      sweep.rs          hit-path sweep over ID grids and active-block lists
      miss.rs           miss compaction, expand, atlas evaluation, intern, publish
      calibrate.rs      start-up micro-benchmarks for the scheduler's constants
      kernels/*.wgsl    one file per kernel; the circuit included from bitslice/
    sched/
      epoch.rs          Algorithm 1: epochs, regime choice, racing
      controller.rs     Algorithm 3: engine per 256x256 region
      sweep_tile.rs     raising and lowering the sweep tile
      cost.rs           costs from live counters and calibrated constants
    query.rs            region read-back, tile summaries, fingerprint, stats
  tests/                oracles against research/ code and recorded Golly fingerprints
  benches/              throughput targets of Section 5
```

Feature flags: `gpu` (default) and `png`. Without `gpu` the library runs the host recursion only, which matches Golly within 0.51–1.9× (paper Table XVIII) and is the right engine for idle universes anyway.

## 4. Stages

Each stage ends in something a UI can use and in checks that already exist in `research/`. Stage 1 depends only on Stage 0; Stages 2 to 4 build on each other; Stage 5 can start after Stage 2.

### Stage 0: foundation (CPU only)

* Workspace, crate skeleton, error type, `Config`.
* `rule`: load a Golly `@TABLE` `.rule` or bare `.table` into a $2^{25}$-byte table; embed the two canonical tables; expose the palette.
* `io`: RLE read and write, macrocell read and write, palette PNG behind `png`.
* `tile` and `cpu::hl`: port of `hl.rs` with 8×8 base tiles stored as bit-planes, the level-synchronous `solve`, root stepping and `result_at`; per-level statistics.
* `cpu::leaf`: the circuit on `u64` bit-planes (`bitslice/step_planes.rs`) as the host's brute-force step, replacing per-cell table lookups. On the CPU it is 2.0–3.5× the table on real content (paper Table XXIII); with it the host floor moves to 16×16 windows as well (paper Section VI-H), which is to be confirmed by measurement here.
* Public API for a CPU-only `Universe` (Section 7).

Done when:

* the rule implementation equals both tables on all $2^{25}$ neighbourhoods (`reproduce.sh correctness`);
* the engine agrees with per-cell stepping on the 22 configurations of `ca-research selftest`;
* final states match Golly's fingerprints for the ten patterns of `reproduce.sh head2head` at $2^{24}$ generations and for the replicator at 262,144. The fingerprints are recorded once with `bgolly` and committed as test fixtures, so the library's tests do not need Golly;
* on one thread the host recursion is within the 0.51–1.9× of Golly that the reference engine reaches (paper Table XVIII).

### Stage 1: plain cells on the GPU

The first GPU engine steps a dense grid without memoization. It replaces the archived kernel and is useful to a UI on its own.

* `gpu::context` with an optional external `wgpu::Device` and `Queue`, so the UI can share them.
* `gpu::brute`: the plain bit-plane kernel (five `u32` per 32 cells, zero ring, 8×8 workgroups, ping-pong buffers, all generations of a submit in one compute pass), the shared-tile kernel with $k = 16$ for small grids, and the byte-LUT kernel for rules without a circuit (Hutton32 today).
* Choice between shared tile and plain kernel: the crossover lies between $3 \cdot 10^{5}$ and $9.5 \cdot 10^{6}$ cells and was not located. Implement both and race them (Stage 4); until racing exists, use the shared tile below $3 \cdot 10^{5}$ cells.
* Grid growth: re-allocate with a margin when live cells reach the ring.

Targets (paper Table XII), cell-exact against the CPU after every benchmark:

| Workload | Target |
|---|---|
| Large grids, any content (Hutton32b) | $3.0 \cdot 10^{11}$ |
| Counter, 37,627 cells | $2.9\text{–}3.0 \cdot 10^{10}$ (770,000–785,000 gens/s) |
| `squares`, 274,950 cells | $1.5 \cdot 10^{11}$ (556,000–563,000 gens/s) |
| ROM, $9.5 \cdot 10^{6}$ cells | $3.9 \cdot 10^{11}$ (40,999 gens/s) |
| Hutton32, byte-LUT kernel, large grids | $1.86 \cdot 10^{11}$ |

Checks: `reproduce.sh bitplane-content bitplane-sizes tile`.

### Stage 2: the hit path, learned by the host

ID grids and memo tables move to VRAM; misses are still resolved by the host recursion.

* `gpu::memo`: open-addressed tables of 32-byte entries (four key IDs, value, spare), linear probing, a node's ID is its slot index; the tile store holds 8×8 base tiles as five bit-planes; tables larger than one binding are sharded by their top hash bits, as `gpu-probe` does with 8 × 1 GiB.
* `gpu::sweep`: one dispatch per sweep, gather four IDs, probe, write one ID; no atomics. Sparse universes run over a list of active blocks; a block joins when a border tile of a neighbour becomes non-empty.
* Learning on the host: run bursts of sweeps from a snapshot; a missed tile writes a sentinel; on a miss, restore the snapshot, let the host learn the affected region, upload the new entries and replay. The host recursion is the same code as Stage 0.
* Sweep tile: start at 16×16 and raise it while the hit rate stays above 40 % (paper equation (5)); every level shares the tables, so a change of level is a parallel re-key of the ID grid.
* `cpu::reclaim` is needed from here on: the reference engine never reclaims and grew to 148 M nodes (about 7 GB) on the replicator. Mark what is reachable from the current grids, rebuild, re-key.

Targets (paper Tables XV and XVII), tile-exact against the host's prediction:

| Workload | Target |
|---|---|
| 32×32 phase-shifted ROMs, $1.02 \cdot 10^{10}$ cells, 16×16 tiles | $8.6 \cdot 10^{12}$ (858 gens/s) |
| The same, 32×32 to 256×256 tiles | $2.1 \cdot 10^{14}$ to $6.8 \cdot 10^{16}$ (20,349 to 6,714,020 gens/s) |
| One ROM, 32×32 tiles | $1.5 \cdot 10^{13}$ (1,607,022 gens/s) |
| Sparse universe of 2304 counters, $1.03 \cdot 10^{10}$ cells, active blocks | $2.9 \cdot 10^{15}$ effective (278,648 gens/s); same final state as Golly |

Checks: `reproduce.sh sweep sweep-tiles sparse`.

### Stage 3: the miss path on the GPU

The host leaves the per-wave path. This is the stage the paper has not measured as a whole.

* Miss compaction: missed positions append to a miss list.
* Deduplicate and claim (`cmd_insert.rs`): claim word 0 (empty), $j+1$ (claimed in this batch by element $j$) or bit 31 (committed); a thread that finds a slot claimed by element $t$ compares against $t$'s key in the immutable batch input; a second dispatch commits. Measured at $0.76\text{–}1.29 \cdot 10^{9}$ requests/s with 0 unresolved, 0 wrong keys and 0 duplicate IDs over 67 M requests.
* Expand each distinct missing key into a $2s \times 2s$ window of an atlas; evaluate the atlas with the brute-force engine for $s/2$ generations; intern the central tile by content with the same claim protocol; publish the result ID and re-probe the missed positions.
* Level-synchronous recursion (paper Algorithm 2): 8×8 base tiles, missed 16×16 windows brute-forced (F16), every level above memoized, 16-generation steps while learning.
* Dispatch sizing: no indirect dispatch. Every stage is dispatched for a fixed capacity taken from asynchronously read, possibly stale miss counters; invocations beyond the live count return at once; a wave that overflows a capacity leaves a sentinel and is replayed from the last snapshot with a larger one.
* Atlas evaluation with the shared-tile kernel, since an atlas of missed windows is a small grid: about 12 µs instead of 29 µs for eight chained dispatches, which takes a wave from about 54 µs to about 37 µs (projected, not run).

Targets (projections, paper Table XXIV and Section VII-A):

| Quantity | Target |
|---|---|
| Miss-path wave of 16 fixed-capacity dispatches | 51–55 µs (measured empty) |
| Learning one ROM period | 0.1–0.5 s (the host takes 80–90 s) |
| Non-repeating logic, $\mu \approx 1.7$ % at 16×16 | $1.8 \cdot 10^{12}$ |
| Displays, $\mu \approx 3.5$ % | $1.0 \cdot 10^{12}$ |
| Hutton replicator, 262,144 generations | 5–15 s (Golly: 111 s) |

Checks: cell-exact against the host after every run; the GPU's own counters (requests, misses, inserts per level) equal the host recursion's for the same universe and steps, which is how `adapt` costs are validated.

### Stage 4: adaptivity

* Universe-wide floor from the active area (paper Table XXII): below $A_1 \approx 1.0 \cdot 10^{6}$ active cells no memo, plain cells; F32 up to $A_2 \approx 1.3 \cdot 10^{7}$; F16 above. Regions may only go coarser than the universe's floor.
* Per-region controller (paper Algorithm 3) on 256×256 regions: start at the floor; price coarser alternatives from the region's own statistics and switch when 15 % cheaper; price finer ones by subdividing one floor miss in sixteen; from plain cells, a two-step trial at most every 16 steps, the interval doubling up to 256 while trials fail. Plain-cells regions are stepped as cells under a region-level memo.
* Racing in the latency-bound regime (paper Algorithm 1, line 4): host recursion, shared-tile kernel and plain kernel each time one epoch; keep the fastest; re-time the others with exponential back-off. Switching costs a read-back (0.1–2.4 ms), so it happens only at epoch boundaries.
* Cycle detection: IDs are canonical, so a hash of the ID grid per epoch detects a globally periodic universe; from then on any generation is a lookup.

Targets (paper Tables XX and XXI, model): on the mixed universe the controller is 7 % faster than uniform F16 (32.6 against 35.0 ms model time) with 43 % less memory; on the ROM alone it is 1.9× faster; on logic it never leaves F16.

Checks: every policy ends in the same state (property test: random policy switches give the same fingerprint); `reproduce.sh floors engine-choice controller` give the model numbers the live counters must approach.

### Stage 5: scale and breadth

* Bit-plane grids beyond one 2 GiB binding: a $10^{5} \times 10^{5}$ universe as plain cells is 6.3 GB per buffer and fits a 16 GB card, but the kernel must span several bindings.
* Tables beyond one binding at full rate (`gpu-probe` runs 8 × 1 GiB).
* GPU table rebuild: mark entries reachable from the current grids, rebuild into a fresh table with the insert kernel, re-key the grids; all three are parallel passes. The memo is a cache: dropping it costs a relearning transient, never correctness.
* A circuit for Hutton32: edit the hand-written starting network in the blocks for an excited-STS, confluent or crossing centre, where the two rules differ (74,854 entries), then run the optimiser in `research/bitslice`, which works from any $2^{25}$-byte table.
* AVX2 for the host leaf: 6–8× the table on real content (paper Table XXIII).

## 5. Cross-cutting engineering

### 5.1 Rules from the measurements

* Keep a whole epoch inside one compute pass. A pass costs 15–30× a dispatch and a read-back 30–700× (paper Table I).
* No read-back in the per-wave path. The CPU reads counters without blocking and may act on values several waves old.
* No indirect dispatch: 260–490 µs per wave in wgpu against 51–55 µs at fixed capacity.
* Keep the width: one thread stepping a window alone is 21× slower than chained dispatches where latency matters (paper Table XIV). Generations per dispatch only through a shared tile.
* One binding holds less than 2 GiB on both backends; shard by the top hash bits.
* A probe costs the same from 128 MiB to 8 GiB, so VRAM is memo capacity: insert while memory lasts. A policy that stops memoizing at high miss rates would never learn a ROM, which misses 96 % for one period and 0 % afterwards.
* Exactness everywhere: keys compared in full, tiles by content, no probabilistic hashing.

### 5.2 Calibration

Every threshold in the paper comes from one RTX 5060 Ti. The library measures the constants it schedules with at start-up instead of hard-coding them: chained-dispatch time, fixed-capacity wave time, probe rate on a table larger than L2, brute-force rate and read-back latency, with short runs of the kernels of `gpu-latency` and `gpu-probe`. $A_1$, $A_2$, the controller's prices and the racing intervals derive from them. The values of the paper are the defaults when calibration is disabled. This is a proposal; the paper does not evaluate it.

### 5.3 Testing

* Oracles: per-cell stepping (`sim.rs`) for small cases, the host recursion for everything else, recorded Golly fingerprints for long runs.
* Exhaustive: rule tables and the circuit on all $2^{25}$ neighbourhoods, on the CPU and on the GPU.
* Kernels: cell-exact after random states (one step of $16384^{2}$ random cells samples 99.97 % of all neighbourhoods) and after full periods of the test patterns.
* Concurrency: the claim protocol under 4× duplication at $1.68 \cdot 10^{7}$ requests per batch, with 0 duplicate IDs.
* Determinism: the same universe and generation count give the same fingerprint under every engine, floor, tile size and policy sequence.
* Benchmarks: the targets of each stage, run on an otherwise idle machine; GPU throughput fell by up to 48 % during a build in the paper's measurements.

## 6. Risks

| Risk | Consequence | Mitigation |
|---|---|---|
| The GPU miss path is projected, not measured | Stage 3 may fall short of $\approx 10^{12}$ on non-repeating logic | Stage 2 already delivers the hit path; Stage 3's counters are checked against the host's, so the gap is visible per stage of the pipeline |
| Fixed capacities overflow often | replay cost dominates | capacities from the last epoch's counters with headroom; overflow counted and reported in `Stats` |
| Crossovers not located (shared tile against plain kernel; host against GPU for small batches) | wrong engine on mid-sized universes | racing (Stage 4) instead of fixed thresholds |
| Controller tuned on one composite universe | poor choices on other content | F16 is the safe default (worst case 2.6× behind); controller can be disabled in `Config` |
| Host memory growth during learning | out of memory on long novel runs | `cpu::reclaim` in Stage 2, before the host learns large universes |
| Hutton32 has no circuit | Hutton32 runs at the byte-LUT rate ($1.86 \cdot 10^{11}$), content-dependent | Stage 5 circuit |
| Constants differ on other GPUs | thresholds off | calibration at start-up |

## 7. Interface for the UI project

The UI depends on `ghl` and nothing in `ghl` depends on it.

```rust
pub struct Universe { /* engines, tables, scheduler */ }

impl Universe {
    pub fn new(rule: Rule, config: Config) -> Result<Self>;
    pub fn with_gpu(rule: Rule, config: Config, gpu: GpuContext) -> Result<Self>; // shared Device/Queue

    // content
    pub fn load(&mut self, pattern: &Pattern, origin: (i64, i64)) -> Result<()>;
    pub fn set_cells(&mut self, cells: &[((i64, i64), u8)]);
    pub fn clear(&mut self);
    pub fn save(&self, format: Format, out: &mut dyn std::io::Write) -> Result<()>;

    // time
    pub fn step(&mut self, generations: u64) -> StepReport;            // exactly this many
    pub fn step_for(&mut self, budget: std::time::Duration) -> StepReport; // as many as fit
    pub fn generation(&self) -> u64;

    // queries
    pub fn population(&self) -> u64;
    pub fn bounding_box(&self) -> Option<Rect>;
    pub fn read_region(&self, rect: Rect, out: &mut [u8]);              // one state per cell
    pub fn tile_summary(&self, rect: Rect, level: u32, out: &mut [TileSummary]); // zoomed-out views
    pub fn fingerprint(&self) -> Fingerprint;
    pub fn stats(&self) -> Stats;                                       // engines, hit rates, memory, overflows

    // zero-copy drawing (feature "gpu")
    pub fn gpu_view(&self) -> Option<GpuView>;                          // ID grid, tile store, layouts
}
```

* **Stepping and responsiveness.** `step_for` runs whole epochs until the budget is spent and returns the generation reached; a UI calls it once per frame or from its own thread. `Universe` is `Send`.
* **Edits.** Keys are tile contents, so editing cells never invalidates the memo: the edited tiles are re-interned and the rest of the universe keeps its IDs. Edits apply between steps.
* **Drawing.** With `gpu_view` a fragment shader reads a pixel's tile ID from the ID grid and then the tile store, so drawing costs follow the screen, not the universe. Zoomed out, `TileSummary` gives a per-tile mean colour and population, computed when a tile is interned. Without a shared device, `read_region` and `tile_summary` read back only the visible area.
* **Palette.** `Rule::palette()` gives the 32 colours of the archived application; anything else about appearance is the UI's.
