# GHL library: implementation roadmap

This document turns the algorithm of [`paper.md`](paper.md), GPU HashLife (GHL), into a library. It covers the whole method: the bit-sliced kernels, the hit path, the miss path, the choice of engine per region and per universe, for every rule of Golly's HashLife family.

The deliverable is a Rust library. Drawing, input, editing tools and everything else a user sees belong to a separate UI project that depends on it. Section 7 defines what that project gets, and Section 8 what it can look at: everything the overview figure at the head of the paper shows, read from the running engine.

Throughput targets are in cell-updates/s over the pattern's bounding box, with generations/s in parentheses, as in the paper. "Paper Table N" refers to the current `paper.md`; `reproduce.sh <target>` refers to `research/reproduce.sh`, whose harness commands serve as the reference implementation and the test oracle throughout.

## 1. Scope

**In the library**

* Rules: every rule of Golly's HashLife family. Life-like rules in B/S notation on the Moore, hexagonal and von Neumann neighbourhoods, with B0 rules through Golly's conversion into rules without B0 alternated between even and odd generations; Golly `@TABLE` and `@TREE` rules with up to 256 states on the Moore, von Neumann, hexagonal and one-dimensional neighbourhoods; Hutton32 and Hutton32b from the canonical tables in `research/rules`.
* Pattern I/O: RLE (extended states), macrocell, and the archived application's palette PNG.
* Simulation: the full GHL algorithm (paper Algorithms 1–3) on a GPU through wgpu, and the host recursion on the CPU, with results bit-identical to per-cell stepping on every path.
* Queries a front end needs: generation, population, bounding box, cells of a region, per-tile summaries for zoomed-out views, fingerprints.
* Introspection: the engine's own data structures and counters, read while it runs: tile IDs, memo tables, the outcome of a sweep, the waves of a step, the atlas of a floor wave, the engine per region, the calibrated rates (Section 8).
* Edits: setting cells and pasting patterns between steps.

**Not in the library**

* Windows, event loops, egui or any other GUI toolkit, rendering pipelines, colour schemes beyond the rule's palette, undo history, file dialogs.
* The measurement harness. `research/` stays as it is; the library copies what it needs and is tested against it.
* Golly's side of any comparison. The library reports its own rates; a UI that shows them against Golly ships recorded figures or runs `bgolly` itself.

## 2. What exists and what is reused

Paper Table IX lists where each component of GHL stands. In terms of code:

| Harness code (`research/`) | Status | In the library |
|---|---|---|
| `ca.rs` | a rule as a table over a radius-1 neighbourhood with its number of states; B/S notation | `rule`: the rule model, extended with rule trees for tables that do not fit |
| `rule.rs`, `rulespec.rs` | CPU rule implementations, Golly table loader (von Neumann, 32 states) | `rule`: keep the table loader, generalized to every neighbourhood and symmetry, and the canonical rules |
| `pattern.rs`, `mc.rs` | palette PNG, RLE reader and writer, macrocell reader (multistate and two-state) | `io`: add a macrocell writer (exists in `cmd_sparse.rs`, both flavours) |
| `sim.rs` | per-cell stepping | test oracle only |
| `hl.rs` | reference engine: hash-consed levels, memo floor, level-synchronous `solve`, root stepping, `result_at`, fingerprints | `cpu`: the host recursion; also the oracle for every GPU path |
| `flat.rs` | positional sweep on the CPU | test oracle for the GPU sweep |
| `bitslice/step_planes.wgsl`, `step_planes.rs`, `circuit.netlist` | the 367-gate circuit, verified on all $2^{25}$ neighbourhoods | `gpu::brute` and `cpu::leaf`, included as generated artifacts; the netlist also feeds `inspect` |
| `cmd_bitslice.rs` (`KERNEL`, `tile_kernel`) | plain bit-plane kernel; shared-tile kernel with $k$ generations per dispatch | `gpu::brute` |
| `cmd_life.rs` (`life_fn`, `PLAIN`, `tile_kernel`) | the Life-like adder tree, generated from the B and S sets; plain and shared-tile kernels on one plane | `gpu::brute` for Life-like rules |
| `cmd_gpu.rs` (`BRUTE_WGSL`) | byte-LUT kernel | `gpu::brute` fallback for rules without a circuit |
| `cmd_gpu.rs` (`SWEEP_WGSL`, `probe_wgsl`) | hit-path sweep; probe kernel sharded across bindings | `gpu::sweep`, `gpu::memo` |
| `cmd_insert.rs` | lock-free find-or-insert with 32-bit CAS (claim protocol) | `gpu::memo` |
| `cmd_sparse.rs` | active-block sweep, macrocell writer | `gpu::sweep`, `io` |
| `cmd_adapt.rs` | per-region floors and engines, cost model, controller (Algorithm 3), the region model: plain cells and the cheaper engine per region priced from a floor run (model only) | `sched`: the policy logic and the prices; the costs come from live counters instead of the model |
| `cmd_overview.rs` | the data of the overview figure, computed by the reference engine: cells, tile IDs and nodes per level, a memo table slot by slot, one sweep window by window, the waves of a step with the atlas of a floor wave, miss rates over time, the controller's regions | `inspect`: the same data from the running engine; `overview`'s JSON is its test oracle (Section 8) |
| `gpu.rs` (`Gpu::timed`, `pipeline_with`) | timing that submits pending uploads before the clock starts; shader modules with or without naga's runtime checks | `gpu::calibrate`, `gpu::context` |
| `cmd_gpu.rs` (`gpu-latency`) | dispatch, pass, submit and read-back costs; fixed-capacity wave | `gpu::calibrate` (Section 5.2) |
| `cmd_bitslice.rs` (`WINDOW_KERNEL`), `cmd_brute2.rs` | one thread per window; class-compressed table | not used: both lose to the kernels above (paper Section VI-C) |
| `cmd_life_cubecl.rs` | `gpu-life` ported to CubeCL 0.12.0-pre.2 | not used: kernels stay WGSL (Section 3) |
| `cmd_symmetry.rs` | memo entries shared up to rotation and mirroring | not used: Hutton32 and Hutton32b are not symmetric, and on Life the eight symmetries save 0–2 % of the memo entries at the floor, 20 % on the metapixel galaxy (`logs/2026-11-26/symmetry.txt`) |

Not built anywhere yet: the GPU miss path as one pipeline (expand, evaluate on an atlas, intern, publish), overflow and replay, plain cells stepped as blocks of regions, a controller that makes the region model's choice while the universe runs, the racing of executors, table rebuild and cycle detection.

## 3. Crate layout

A new crate `ghl/` next to `research/`, in a workspace at the repository root (`research/` joins it unchanged; `archive/` stays out, it builds against wgpu 0.14).

```
ghl/
  Cargo.toml            wgpu 30, bytemuck, rustc-hash; image behind the "png" feature
  src/
    lib.rs              Universe, Config, errors; re-exports
    rule/               Rule (neighbourhood, states, table or tree), loaders for B/S notation, @TABLE and @TREE,
                        B0 conversion, canonical tables (include_bytes!), palette, the rule's gate network
    io/                 rle, macrocell (read + write), png (feature "png")
    tile.rs             TileId, base tiles of the rule's size as b bit-planes, content hashing
    cpu/
      hl.rs             host recursion: hash-consed levels, solve(), root stepping, result_at()
      leaf.rs           bit-sliced evaluation of the rule's network on u64 words (AVX2 later)
      reclaim.rs        mark reachable nodes, rebuild
    gpu/
      context.rs        device, queue, limits; accepts an external Device/Queue
      shard.rs          buffers and tables larger than one 2 GiB binding
      brute.rs          per rule class: Life-like adder tree, synthesised circuits, table and tree fallbacks;
                        plain and shared-tile forms of each; plain cells over a mask of regions
      memo.rs           memo tables, claim protocol, probe, tile store
      sweep.rs          hit-path sweep over ID grids and active-block lists
      miss.rs           miss compaction, expand, atlas evaluation, intern, publish
      calibrate.rs      start-up micro-benchmarks for the scheduler's constants
      kernels/*.wgsl    one file per kernel; the circuit included from bitslice/
    sched/
      epoch.rs          Algorithm 1: epochs, regime choice, racing
      region.rs         the region grid (four steps wide), the step, the price of a stepped region (paper eq. 7)
      controller.rs     engine per region: Algorithm 3 and the price-based choice
      sweep_tile.rs     raising and lowering the sweep tile
      cost.rs           costs from live counters and calibrated constants
    inspect/            counters and their history, maps (IDs, outcomes, regions, memo), step traces (Section 8)
    query.rs            region read-back, tile summaries, fingerprint, stats
  tests/                oracles against research/ code, recorded Golly fingerprints, `overview` JSON
  benches/              throughput targets of Section 4
  examples/overview.rs  writes research/out/overview.json through the public API alone (Section 8.4)
```

Feature flags: `gpu` (default) and `png`. Without `gpu` the library runs the host recursion only, which runs at 0.26–0.97× Golly's speed (paper Table XXII); for idle universes and long runs of small ones that is the engine to use, so closing the gap to Golly is part of Stage 0.

**Kernels are WGSL.** Every kernel is WGSL source, generated or templated by Rust code and compiled by wgpu, as in the harness. What depends on the rule (the B and S sets, the circuit, the tile shape, $k$) is substituted before compilation, so changing the rule at run time is a new shader module, not a rebuild. Kernels that have passed their cell-exact tests may be compiled as trusted modules, without naga's index clamping and loop bounding: that is worth 18 % on the Life shared-tile kernel at $1024^{2}$ (714,984 to 841,863 gens/s) and nothing on the other kernels measured.

CubeCL is not used. `cmd_life_cubecl.rs` ports `gpu-life` to it kernel for kernel (`logs/2026-11-26/cubecl-life.txt`). On large grids it runs at the WGSL kernels' rate, and its SPIR-V path runs the shared tile 9–17 % faster than trusted WGSL. But where dispatch latency dominates it is 2.0–3.9× slower (the plain kernel at $256^{2}$ and $1024^{2}$), because every launch is a submission of its own and an epoch cannot be kept inside one compute pass (Section 5.1). It is also a pre-release, pinned to one exact version, with gaps met in the port: typed literals, no destructuring of tuples, a runtime trait that is not re-exported. The kernels are pure functions of words with the rule as a compile-time parameter, so they would port directly; the decision is to be revisited when CubeCL is stable and can chain launches in one pass.

## 4. Stages

Each stage ends in something a UI can use, in checks that already exist in `research/`, and in the introspection of what the stage adds (Section 8). Stage 1 depends only on Stage 0; Stages 2 to 4 build on each other; Stage 5 can start after Stage 2.

### Stage 0: foundation (CPU only)

* Workspace, crate skeleton, error type, `Config`.
* `rule`: parse B/S notation; load Golly `@TABLE` and `@TREE` rules on every neighbourhood into a table where it fits and a tree otherwise; convert B0 rules as Golly does; embed the two canonical Hutton tables; expose the palette and, where the rule has one, its gate network.
* `io`: RLE read and write, macrocell read and write, palette PNG behind `png`.
* `tile` and `cpu::hl`: port of `hl.rs` with base tiles stored as bit-planes (8×8 cells for Hutton32b, 16×16 for Life, from the rule's $\rho$), the level-synchronous `solve`, root stepping and `result_at`; per-level statistics.
* `cpu::leaf`: the rule's network on `u64` bit-planes (`bitslice/step_planes.rs` for Hutton32b, the adder tree for Life-like rules) as the host's brute-force step, replacing per-cell table lookups. On the CPU it is 2.0–3.5× the table on real content (paper Table XXXII); with it the host floor moves to 16×16 windows as well (paper Section VI-H), which is to be confirmed by measurement here.
* Public API for a CPU-only `Universe` (Section 7).
* `inspect` on the host recursion: IDs, tiles and nodes per level, per-level counters and their history, the trace of a step with its waves and the windows of a floor wave, the network and its signals on a chosen word.

Done when:

* the rule implementation equals both Hutton tables on all $2^{25}$ neighbourhoods (`reproduce.sh correctness`), and every Life-like rule's table equals its B/S definition on all $2^{9}$;
* the engine agrees with per-cell stepping on the 22 configurations of `ca-research selftest`;
* final states match Golly's fingerprints for the patterns of `reproduce.sh head2head`, `runlength` and `life-h2h` (Hutton32, Hutton32b and Life, $2^{20}$ to $2^{24}$ generations). The fingerprints are recorded once with `bgolly` and committed as test fixtures, so the library's tests do not need Golly;
* on one thread the host recursion is at least at the 0.26–0.97× of Golly's speed that the reference engine reaches (paper Table XXII), and closer to Golly on the idle displays, where the gap is widest;
* the inspector reproduces the keys `structures`, `structures_life` and `waves` of `ca-research overview` for the figure's universes (Section 8.4).

### Stage 1: plain cells on the GPU

The first GPU engine steps a dense grid without memoization. It replaces the archived kernel and is useful to a UI on its own.

* `gpu::context` with an optional external `wgpu::Device` and `Queue`, so the UI can share them.
* `gpu::brute`: for each rule class the plain kernel ($b$ `u32` per 32 cells, zero ring, 8×8 workgroups, ping-pong buffers, all generations of a submit in one compute pass) and the shared-tile kernel with $k = 16$ for small grids: the Life-like adder tree on one plane (WGSL generated from the B and S sets, as `cmd_life.rs` does), the synthesised circuit for Hutton32b, and the byte-LUT kernel for table rules without a circuit (Hutton32 today); a rule-tree kernel for tables that do not fit.
* `gpu::calibrate` measures the chosen kernel's rate at start-up; with the probe rate it gives the rule's $\rho$, which sets the floor and the base tile (paper Table II): 16×16 windows on 8×8 base tiles for Hutton32b, 32×32 on 16×16 for Life.
* Choice between shared tile and plain kernel: the crossover lies between $3 \cdot 10^{5}$ and $9.5 \cdot 10^{6}$ cells and was not located. Implement both and race them (Stage 4); until racing exists, use the shared tile below $3 \cdot 10^{5}$ cells.
* Grid growth: re-allocate with a margin when live cells reach the ring.
* `inspect`: the calibrated rates, the kernel in use with its tile shape and $k$, cell-updates and time per epoch.

Targets (paper Tables XIII and XVI), cell-exact against the CPU after every benchmark:

| Workload | Target |
|---|---|
| Large grids, any content (Hutton32b) | $3.0 \cdot 10^{11}$ |
| Counter, 37,627 cells | $3.0 \cdot 10^{10}$ (804,105 gens/s, shared tile) |
| `chaotic`, 19,932 cells | $1.8 \cdot 10^{10}$ (892,832 gens/s, shared tile) |
| `squares`, 274,950 cells | $1.57 \cdot 10^{11}$ (571,123 gens/s, shared tile) |
| ROM, $9.5 \cdot 10^{6}$ cells | $3.9 \cdot 10^{11}$ (40,859 gens/s) |
| Hutton32, byte-LUT kernel, large grids | $1.85 \cdot 10^{11}$ |
| Life, large grids, any content | $1.42 \cdot 10^{12}$ |
| Life, $256^{2}$ cells, shared tile | $1.37 \cdot 10^{11}$ (2,092,859 gens/s) |
| Life, 2048 × 2048 soup, 4096 generations | 50,018 gens/s; same final state as Golly |

Checks: `reproduce.sh bitplane-content bitplane-sizes tile life-kernels life-golly`.

### Stage 2: the hit path, learned by the host

ID grids and memo tables move to VRAM; misses are still resolved by the host recursion.

* `gpu::memo`: open-addressed tables of 32-byte entries (four key IDs, value, spare), linear probing, a node's ID is its slot index; the tile store holds the rule's base tiles as bit-planes; tables larger than one binding are sharded by their top hash bits, as `gpu-probe` does with 8 × 1 GiB.
* `gpu::sweep`: one dispatch per sweep, gather four IDs, probe, write one ID; no atomics. Sparse universes run over a list of active blocks; a block joins when a border tile of a neighbour becomes non-empty.
* Learning on the host: run bursts of sweeps from a snapshot; a missed tile writes a sentinel; on a miss, restore the snapshot, let the host learn the affected region, upload the new entries and replay. The host recursion is the same code as Stage 0.
* Sweep tile: start at 16×16 and raise it while the hit rate stays above 40 % (paper equation (5)); every level shares the tables, so a change of level is a parallel re-key of the ID grid.
* `cpu::reclaim` is needed from here on: the reference engine never reclaims and grew to 148 M nodes (about 7 GB) on the replicator. Mark what is reachable from the current grids, rebuild, re-key.
* `inspect`: the ID grid, the tile store and the memo tables as they lie in VRAM; the outcome of the last sweep per tile (the sentinel is the miss mask); windows, distinct windows and misses per sweep as a history.

Targets (paper Tables XVII, XVIII, XX and XXI), tile-exact against the host's prediction:

| Workload | Target |
|---|---|
| 32×32 phase-shifted ROMs, $1.02 \cdot 10^{10}$ cells, 16×16 tiles | $8.7 \cdot 10^{12}$ (857 gens/s) |
| The same, 32×32 to 256×256 tiles | $2.1 \cdot 10^{14}$ to $7.4 \cdot 10^{16}$ (20,908 to 7,259,759 gens/s) |
| One ROM, 32×32 tiles | $1.6 \cdot 10^{13}$ (1,647,369 gens/s) |
| Sparse universe of 2304 counters, $1.03 \cdot 10^{10}$ cells, active blocks | $2.9 \cdot 10^{15}$ effective (280,231 gens/s); same final state as Golly |
| 8×8 ROMs in 4 alignment classes, 16×16 tiles | $6.3 \cdot 10^{12}$ (9,949 gens/s), 9.85 M entries |
| Life: 32×32 phase-shifted Unit Life Cells, 32×32 to 256×256 tiles | $3.6 \cdot 10^{14}$ to $8.4 \cdot 10^{15}$ (851,582 to 20,024,161 gens/s) |
| Life: sparse universe of 2304 Unit Life Cells, active blocks | 418,939 gens/s, level with Golly's 414,457 over 16,384 generations; over $2^{20}$ Golly is 56× faster, so Stage 4's cycle detection decides this case |

Checks: `reproduce.sh sweep sweep-tiles unaligned sparse life-sweep life-sparse`; the inspector reproduces the keys `sweep`, `table`, `table_life` and `learning` of `overview`.

### Stage 3: the miss path on the GPU

The host leaves the per-wave path. This is the stage the paper has not measured as a whole.

* Miss compaction: missed positions append to a miss list.
* Deduplicate and claim (`cmd_insert.rs`): claim word 0 (empty), $j+1$ (claimed in this batch by element $j$) or bit 31 (committed); a thread that finds a slot claimed by element $t$ compares against $t$'s key in the immutable batch input; a second dispatch commits. Measured at $1.0\text{–}1.3 \cdot 10^{9}$ requests/s ($4.9 \cdot 10^{8}$ for the first batch into an empty table) with 0 unresolved, 0 wrong keys and 0 duplicate IDs over 67 M requests.
* Expand each distinct missing key into a $2s \times 2s$ window of an atlas; evaluate the atlas with the brute-force engine for $s/2$ generations; intern the central tile by content with the same claim protocol; publish the result ID and re-probe the missed positions.
* Level-synchronous recursion (paper Algorithm 2): the rule's base tiles and floor (F16 on 8×8 tiles for Hutton32b, F32 on 16×16 for Life), every level above memoized. While periodic content is new the step is one base tile's side, 8 generations for Hutton32b and 16 for Life (paper Table XXX): a step that does not divide a period multiplies the learning time, and the ROM takes 33 steps to recur whatever the step.
* Dispatch sizing: no indirect dispatch. Every stage is dispatched for a fixed capacity taken from asynchronously read, possibly stale miss counters; invocations beyond the live count return at once; a wave that overflows a capacity leaves a sentinel and is replayed from the last snapshot with a larger one.
* Atlas evaluation with the shared-tile kernel, since an atlas of missed windows is a small grid: a first generation costs a dispatch of 3.5 µs and each further one in the tile about 1.1 µs, which would take a wave from about 53 µs to about 37 µs (projected, not run).
* `inspect`: requests, distinct windows, misses and inserts per level and wave from the GPU's own counters; capacities and overflows; the atlas buffer of a wave with the positions of its windows.

Targets (projections, paper Table XXXIII and Section VII-A):

| Quantity | Target |
|---|---|
| Miss-path wave of 16 fixed-capacity dispatches | 50–55 µs (measured empty) |
| Learning one ROM period | 0.1–0.5 s (the host takes 84–88 s) |
| Non-repeating logic, $\mu \approx 1.7$ % at 16×16 | $1.8 \cdot 10^{12}$ |
| Displays, $\mu \approx 3.5$ % | $1.0 \cdot 10^{12}$ |
| Hutton replicator, 262,144 generations | 0.4 s of probes and 5–15 s of wave latency |

Checks: cell-exact against the host after every run; the GPU's own counters (requests, misses, inserts per level) equal the host recursion's for the same universe and steps, which is how `adapt` costs are validated and what makes a host trace a true picture of a GPU step (Section 8.2).

### Stage 4: adaptivity

* Universe-wide floor from the active area (paper Table XXXI): below $A_1$ no memo, plain cells; one level coarser than the rule's floor up to $A_2$; the rule's floor above. For Hutton32b $A_1 \approx 1.0 \cdot 10^{6}$ and $A_2 \approx 1.3 \cdot 10^{7}$ active cells; for another rule both follow from paper equation (6) with the calibrated rates. Regions may only go coarser than the universe's floor.
* Region and step from the region model (paper equations (7)–(9), Tables XXIX and XXX). The region is four steps wide, the smallest a step admits. The step is one base tile's side while periodic content is new (Hutton32b: 8 generations and 32×32 regions; Life: 16 and 64×64) and two to eight base tiles' sides where the memo engine carries the universe (Life's universes are cheapest at 64 generations and 256×256 regions). The step is universe-wide and changes at epoch boundaries. The rule for changing it is not in the paper: start at one base tile's side, lengthen while no region is in plain cells and the hashing above the floor dominates the epoch, shorten when plain cells are entered.
* Plain cells as blocks. Regions in plain cells are stepped together by the plain kernel over a mask of regions, so that neighbours share their halos, and are converted to tile IDs once per step. This is a requirement, not a refinement: stepped as isolated windows, which is all the reference engine can do, the choice per region costs 145.3 ms against 60.8 for uniform F16 on the Hutton32b universes (paper Section VI-G), and every gain below disappears.
* The price of both engines per region, from live counters. Requests and misses at and above the region level do not depend on the engine below it, so the floor's own region-level misses say which regions plain cells would step; at $q$ times their cell-updates (paper equation (7), with the calibrated rates) that is the price of plain cells without running them. The price of the floor is what the region's windows cost at and below the region level. `cmd_adapt.rs` computes both (`PlainModel`, `hindsight`) and asserts the first against every plain-cells run.
* The controller. Algorithm 3 is ported first (`cmd_adapt.rs`, `decide`): start at the floor; price coarser alternatives from the region's own statistics and switch when 15 % cheaper; price finer ones by subdividing one floor miss in sixteen; from plain cells, a two-step trial at most every 16 steps, the interval doubling up to 256 while trials fail. Its entry rule is its limit: it enters plain cells on bursts of misses that the floor would have learned, costs 49–55 % more than the engine per region chosen in hindsight on Hutton32b and loses to the uniform floor on Life (paper Table XXIX). Its replacement uses the two prices above, accumulated over the steps since a region's content last changed at the region level, and enters plain cells only when their price has stayed below the floor's over a span in which the floor could have learned; learning that other regions reuse is credited to the region that paid for it, because a price per region is 14–26 % too low for plain cells on a universe built from copies. This rule is a design, not a measured component.
* Racing in the latency-bound regime (paper Algorithm 1, line 4): host recursion, shared-tile kernel and plain kernel each time one epoch; keep the fastest; re-time the others with exponential back-off. Switching costs a read-back (0.1–2.4 ms), so it happens only at epoch boundaries.
* Cycle detection: IDs are canonical, so a hash of the ID grid per epoch detects a globally periodic universe; from then on any generation is a lookup. On recurring Life universes this is what lets GHL skip time as HashLife's doubling steps do; without it Golly is 9.2× faster on one Unit Life Cell and 56× on the sparse universe over $2^{20}$ generations (paper Table XI).
* `inspect`: the region grid with each region's engine, the steps spent in each, both prices and the time charged; the step; active area, $A_1$, $A_2$ and the floor chosen; the engines raced with their times; the period once a cycle is found.

Targets (model times, paper Tables XXVI, XXIX and XXX):

| Quantity | Target |
|---|---|
| Algorithm 3 as ported, mixed universe, 16-generation steps, 256×256 regions | 32.6 ms against 35.0 for uniform F16, with 43 % less memory; 1.9× faster on the ROM alone; never leaves F16 on logic |
| Engine per region in hindsight, Hutton32b universes, 16-generation steps, 64×64 regions | 32.1 ms (uniform F16: 65.8; Algorithm 3: 48.3) |
| The same at 8-generation steps, 32×32 regions | 25.1 ms |
| Engine per region in hindsight, Life universes, 64-generation steps, 256×256 regions | 3.16 ms (uniform F32 at 16-generation steps: 4.40) |
| The price-based controller | on no universe of paper Tables XXVII and XXVIII more than 5 % above the uniform floor; within 20 % of the hindsight sums above. Acceptance test, not a measurement; until it passes, the controller ships disabled and every region runs at the floor |

Checks: every policy ends in the same state (property test: random policy switches give the same fingerprint); `reproduce.sh floors engine-choice controller region-size region-size-life region-step region-model` give the model numbers the live counters must approach; the inspector reproduces the key `controller` of `overview`.

### Stage 5: scale and breadth

* Bit-plane grids beyond one 2 GiB binding: a $10^{5} \times 10^{5}$ universe as plain cells is 6.3 GB per buffer and fits a 16 GB card, but the kernel must span several bindings.
* Tables beyond one binding at full rate (`gpu-probe` runs 8 × 1 GiB).
* GPU table rebuild: mark entries reachable from the current grids, rebuild into a fresh table with the insert kernel, re-key the grids; all three are parallel passes. The memo is a cache: dropping it costs a relearning transient, never correctness.
* A circuit for Hutton32: edit the hand-written starting network in the blocks for an excited-STS, confluent or crossing centre, where the two rules differ (74,854 entries), then run the optimiser in `research/bitslice`, which works from any $2^{25}$-byte table.
* AVX2 for the host leaf: 6–8× the table on real content (paper Table XXXII).

## 5. Cross-cutting engineering

### 5.1 Rules from the measurements

* Keep a whole epoch inside one compute pass. A pass costs 15–30× a dispatch and a read-back 30–700× (paper Table I).
* No read-back in the per-wave path. The CPU reads counters without blocking and may act on values several waves old. Debug data obey the same rule (Section 8.1).
* No indirect dispatch: 230–490 µs per wave in wgpu against 50–55 µs at fixed capacity.
* Keep the width: one thread stepping a window alone is 22× slower than chained dispatches where latency matters (paper Table XV). Generations per dispatch only through a shared tile.
* One binding holds less than 2 GiB on both backends; shard by the top hash bits.
* A probe costs the same from 128 MiB to 8 GiB, so VRAM is memo capacity: insert while memory lasts. A policy that stops memoizing at high miss rates would never learn a ROM, which misses 96 % for one period and 0 % afterwards.
* Exactness everywhere: keys compared in full, tiles by content, no probabilistic hashing.
* An upload is not GPU work. `queue.write_buffer` runs at the next submit, so writes queued before a timed section are submitted before the clock starts (`Gpu::timed`); timed with the kernels, an upload of 100–240 MB made short runs look up to twice as slow.

### 5.2 Calibration

Every threshold in the paper comes from one RTX 5060 Ti. The library measures the constants it schedules with at start-up instead of hard-coding them: chained-dispatch time, fixed-capacity wave time, probe rate on a table larger than L2, insert rate, brute-force rate and read-back latency, with short runs of the kernels of `gpu-latency`, `gpu-probe` and `gpu-insert`. The rule's $\rho$, its floor and base tile, $A_1$, $A_2$, the price $q$ of a stepped region, the controller's prices and the racing intervals derive from them. The values of the paper are the defaults when calibration is disabled. The result is kept and exposed (Section 8), so a UI can show which constants a run is scheduled with. This is a proposal; the paper does not evaluate it.

### 5.3 Testing

* Oracles: per-cell stepping (`sim.rs`) for small cases, the host recursion for everything else, recorded Golly fingerprints for long runs, `ca-research overview` for the inspector.
* Exhaustive: rule tables and circuits on every neighbourhood where they can be enumerated ($2^{25}$ for the Hutton rules, $2^{9}$ for Life-like rules), on the CPU and on the GPU; random sampling against the rule tree beyond that.
* Kernels: cell-exact after random states (one step of $16384^{2}$ random cells samples 99.97 % of all $2^{25}$ neighbourhoods) and after full periods of the test patterns.
* Concurrency: the claim protocol under 4× duplication at $1.68 \cdot 10^{7}$ requests per batch, with 0 duplicate IDs.
* Determinism: the same universe and generation count give the same fingerprint under every engine, floor, tile size, step, region size and policy sequence, with every tap on and off.
* Benchmarks: the targets of each stage, on an otherwise idle machine. GPU rates repeat within a few percent; rates of the CPU and of Golly moved by up to 2× between two days of measurement (paper Appendix A), so a ratio compares runs of one day.

## 6. Risks

| Risk | Consequence | Mitigation |
|---|---|---|
| The GPU miss path is projected, not measured | Stage 3 may fall short of $\approx 10^{12}$ on non-repeating logic | Stage 2 already delivers the hit path; Stage 3's counters are checked against the host's, so the gap is visible per stage of the pipeline |
| Fixed capacities overflow often | replay cost dominates | capacities from the last epoch's counters with headroom; overflow counted and reported in `Stats` |
| Crossovers not located (shared tile against plain kernel; host against GPU for small batches) | wrong engine on mid-sized universes | racing (Stage 4) instead of fixed thresholds |
| The gains of plain cells per region exist only in the model | priced from the engine's counters, with regions as isolated windows, the choice per region loses to the uniform floor | plain cells stepped as blocks over a region mask (Stage 4); the uniform floor is the default until block stepping is measured |
| Algorithm 3 enters plain cells on bursts | 49–55 % above the choice in hindsight on Hutton32b, worse than the uniform floor on Life | the price-based controller; its acceptance test (Stage 4); disabled in `Config` until it passes |
| A price per region ignores that memo entries are shared | plain cells chosen where learning would have served other regions (metapixel galaxy: 1.25 ms against 1.22 for uniform F32) | credit shared learning to the region that paid; the acceptance test includes the galaxy |
| Host memory growth during learning | out of memory on long novel runs | `cpu::reclaim` in Stage 2, before the host learns large universes |
| Hutton32 has no circuit | Hutton32 runs at the byte-LUT rate ($1.85 \cdot 10^{11}$), content-dependent | Stage 5 circuit |
| Many-state Moore rules have neither a table that fits nor a circuit | they run on a rule-tree kernel, whose rate was not measured and sets their floor | calibration picks the floor from the measured rate; circuits synthesised per rule where the gate count is small |
| Recurring and sparse Life universes favour HashLife's doubling steps | the sweep is 9–56× slower than Golly there over $2^{20}$ generations | cycle detection (Stage 4); the host recursion as the racing candidate meanwhile |
| Constants differ on other GPUs | thresholds off | calibration at start-up |
| Introspection slows the engine or changes what it does | a UI that watches is not watching the real run | counters only in the fast path, maps at epoch boundaries, traces by replay; fingerprints equal with every tap on and off; overhead of the counters measured per stage (Section 8.1) |

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

    // introspection (Section 8)
    pub fn observer(&self) -> Observer;                                 // Clone + Send + Sync; reads while the engine steps
    pub fn set_taps(&mut self, taps: Taps);                             // which maps are kept current; none by default
    pub fn trace_step(&mut self) -> StepTrace;                          // the last step again, recorded wave by wave

    // zero-copy drawing (feature "gpu")
    pub fn gpu_view(&self) -> Option<GpuView>;                          // buffers and layouts, see Section 8.3
}
```

* **Stepping and responsiveness.** `step_for` runs whole epochs until the budget is spent and returns the generation reached; a UI calls it once per frame or from its own thread. `Universe` is `Send`.
* **Edits.** Keys are tile contents, so editing cells never invalidates the memo: the edited tiles are re-interned and the rest of the universe keeps its IDs. Edits apply between steps.
* **Drawing.** With `gpu_view` a fragment shader reads a pixel's tile ID from the ID grid and then the tile store, so drawing costs follow the screen, not the universe. Zoomed out, `TileSummary` gives a per-tile mean colour and population, computed when a tile is interned. Without a shared device, `read_region` and `tile_summary` read back only the visible area.
* **Palette.** `Rule::palette()` gives the 32 colours of the archived application; anything else about appearance is the UI's.
* **Reports.** `StepReport` carries the generations run, the cell-updates they stand for, the wall time and its split by engine, so a UI can show cell-updates/s and generations/s per engine without the inspector.

## 8. Introspection: the overview figure, live

The overview figure at the head of the paper (`research/figures/ghl_overview.svg`) explains GHL with data that the reference engine computes once, offline (`ca-research overview`, drawn by `tools/ghl_diagram.py`). A UI must be able to draw each of its panels from a universe that is running, at the user's viewport and at the current generation. The library therefore exposes the structures the engine already keeps, not a second computation of them.

### 8.1 Three kinds of data, three costs

| Kind | What | How it leaves the engine | Cost |
|---|---|---|---|
| Counters | requests, distinct windows, misses and inserts per level and wave; cell-updates and time per engine; overflows; the scheduler's choices | atomics that the kernels already maintain for dispatch sizing, read without blocking; published once per epoch as an immutable sample and kept in a ring as history | always on. No read-back in the per-wave path (Section 5.1); target below 1 % of an epoch, measured per stage |
| Maps | the ID grid, the outcome of the last sweep per tile, the engine per region, memo slots | buffers that exist in VRAM anyway. With a shared device the UI's shaders read them in place through `GpuView`; otherwise the visible rectangle is read back at an epoch boundary, when a tap is set and an observer has asked | nothing while no tap is set; one read-back (0.1–2.4 ms) and the bytes of the rectangle per request |
| Traces | one step wave by wave: every wave's windows, the atlas of a floor wave with its cells, the memo entry of a chosen window | `trace_step` replays the last step from its snapshot, the mechanism that overflow already needs, with recording on; exactness makes the replay the same step | one replay with a read-back per wave: milliseconds. For a paused or single-stepping UI; at full speed a UI shows counters and maps |

Three rules keep the picture honest. Debug data are read from the engine's own structures and never recomputed by a second implementation; the one exception is the signal trace of the rule's network, a pure function of the rule. Taps and traces must not change results: the determinism test of Section 5.3 runs with every tap on and off. And a CPU-only build returns the same types from the host recursion, whose counters equal the GPU's for the same universe and steps (Stage 3), so a UI written against the inspector works from Stage 0.

`Observer` is the handle for a UI thread. It is `Clone + Send + Sync`, holds the latest published sample behind an atomic pointer, and queues requests for maps that the engine answers at its next epoch boundary, so reading never blocks stepping and stepping never waits for the UI.

### 8.2 The panels and their data

The last column names the key of `research/out/overview.json` that holds the same data for the figure's universes; it is the test oracle of the row.

| Panel of the figure | What it draws | Library call | Kind | From stage | `overview.json` |
|---|---|---|---|---|---|
| Strip: the algorithm at a glance | which engine ran each step of Algorithm 1, on the GPU or the host | `Sample::regime`, `Sample::engines` | counters | 0 | — |
| 1. Universes | cells as pixels, colour by state | `read_region`, `tile_summary`, `GpuView` | map | 0; zero-copy from 2 | `patterns` |
| 1. Universes | how much is new: the share of probes that miss against the generations run | `Observer::history()`: per epoch the generation and, per level, windows, distinct windows, misses | counters | 0 (recursion), 2 (sweep) | `learning` |
| 2. Below the floor | the 32 cells of a word and their neighbours as bit-plane words | `read_planes(x, y)` | map | 0 | `patterns` |
| 2. Below the floor | the rule as a network of gates: 367 for Hutton32b, the adder tree for a Life-like rule | `Rule::network()`: inputs, gates, outputs, depth; `None` for a table or tree kernel | static | 0 | `bitslice/circuit.netlist`, `cmd_life.rs` |
| 2. Below the floor | on how many of the 32 cells each signal is 1, and that the outputs are the next generation | `Network::trace(&words)`: every signal as a word | pure function | 0 | checked by the figure itself |
| 2. Below the floor | brute force as measured | `Observer::calibration().r_b`, `Sample::engines` (kernel, tile shape, $k$, cell-updates, time) | counters | 1 | logs of `lut-kernel`, `bitplane-sizes`, `tile`, `life-kernels` |
| 3. Above the floor | the universe cut into tiles; cells become tile IDs | `ids(rect, level)`, `level_info(level)` (side, non-empty and distinct tiles) | map | 0 (host levels), 2 (ID grid in VRAM) | `structures`, `structures_life` |
| 3. Above the floor | a tile's content; a node's four children | `tile(level, id)` | map | 0 | `structures.levels[].tiles`, `.nodes` |
| 3. Above the floor | a memo entry: four IDs give one ID, and whether this sweep hit it | `memo_lookup(level, key)`, `outcomes(rect)` | map | 2 | `sweep` |
| 3. Above the floor | the memo table slot by slot: occupancy, probe lengths, load, bytes | `memo_stats(level)`, `memo_map(level, slots)` | counters, map | 0 (host tables), 2 | `table`, `table_life` |
| 4. A miss | one wave per level: windows asked for and missed, generations it stands for, dispatches | `StepTrace::waves`; totals per level in every `Sample` | trace, counters | 0 (host), 3 (GPU) | `waves.*.steps_logged[].waves`, `.totals` |
| 4. A miss | the atlas of a floor wave: the missed windows side by side, with their cells | `StepTrace::atlas(wave)` | trace | 0, 3 | `waves.*.steps_logged[].atlas` |
| 4. A miss | per universe: how deep the recursion may go | `Sample::depth`: active cells, $A_1$, $A_2$, the floor, the dispatch time, the engines raced with their times | counters | 4 | model: paper Table XXXI |
| 4. A miss | per region: memo or brute force | `regions(rect)`: engine, steps spent in each, both prices, time charged; `Sample::region_side`, `Sample::step`; the mode buffer in `GpuView` | counters, map | 4 | `controller` |
| 5. Why a floor | the measured rates, $\rho$, what a hit saves per window size, the floor | `Observer::calibration()`, `floor_table()` | static per run | 1 | logs through `tools/ghl_logs.py` |
| 5. Why a floor | GHL against Golly | the library's side only: `StepReport` rates by engine and run length | — | 0 | not library data (Section 1) |

### 8.3 Types

```rust
pub struct Observer { /* latest sample, request queue */ }

impl Observer {
    pub fn sample(&self) -> Arc<Sample>;                                // counters of the last finished epoch
    pub fn history(&self) -> Vec<Arc<Sample>>;                          // ring, thinned to log-spaced generations when full
    pub fn calibration(&self) -> Arc<Calibration>;                      // R_B per kernel, R_P by table size, R_I, latencies
    pub fn floor_table(&self) -> Vec<FloorRow>;                         // window, V, V / rho, memoized or not
    pub fn level_info(&self, level: u32) -> LevelInfo;

    // maps: filled at the next epoch boundary, or at once on a paused universe
    pub fn ids(&self, rect: Rect, level: u32) -> Pending<Vec<TileId>>;
    pub fn tile(&self, level: u32, id: TileId) -> Pending<Tile>;        // cells of a base tile, or four children
    pub fn outcomes(&self, rect: Rect) -> Pending<Vec<Outcome>>;        // Hit, Miss, Empty per tile of the last sweep
    pub fn regions(&self, rect: Rect) -> Pending<Vec<RegionInfo>>;
    pub fn memo_stats(&self, level: u32) -> MemoStats;                  // entries, slots, bytes, probe-length histogram
    pub fn memo_map(&self, level: u32, slots: Range<u64>) -> Pending<Vec<Slot>>; // empty, claimed or committed; displacement
    pub fn memo_lookup(&self, level: u32, key: [TileId; 4]) -> Pending<Option<MemoEntry>>;
    pub fn read_planes(&self, x: i64, y: i64) -> Pending<Planes>;       // a word of 32 cells and its neighbour words
}

bitflags! { pub struct Taps: u32 { const IDS; const OUTCOMES; const REGIONS; const MEMO; } }

pub struct Sample {                                                     // one epoch
    pub generation: u64, pub step: u32, pub region_side: u32,
    pub regime: Regime,                                                 // host recursion, plain cells, shared tile, memo
    pub engines: Vec<EngineUse>,                                        // kernel, tile shape, k, cell-updates, microseconds
    pub levels: Vec<LevelCounters>,                                     // requests, distinct, misses, inserts, entries
    pub depth: Depth,                                                   // active cells, A1, A2, floor, engines raced
    pub overflows: u32, pub memory: Memory,                             // bytes of grids, tables, tile store, atlas
}
pub struct RegionInfo { pub x: i64, pub y: i64, pub engine: Engine, pub steps: [u32; 2],
                        pub price_floor_ns: f64, pub price_cells_ns: f64, pub charged_ns: f64, pub switches: u32 }

pub struct StepTrace { pub generation: u64, pub step: u32, pub waves: Vec<Wave>, /* atlases */ }
pub struct Wave { pub level: u32, pub side: u32, pub generations: u32, pub requests: u64, pub distinct: u64,
                  pub misses: u64, pub inserts: u64, pub dispatches: u32, pub capacity: u64, pub micros: f32 }

impl Rule { pub fn network(&self) -> Option<&Network>; }
impl Network { pub fn trace(&self, words: &[u32]) -> Vec<u32>; }        // every signal, 32 cells at once
```

A `Pending<T>` resolves at the next epoch boundary (`poll`, or `wait` from another thread) and at once on a universe that is not stepping. Counters need none: they are in the sample.

`GpuView` grows with the stages: the ID grid and the tile store (Stage 2), then the sweep's outcome mask, the memo tables' claim words, the atlas, and the buffer of region engines, each with its layout and the lattice offset of the current phase. A UI on the shared device draws any of them as an overlay in its own shaders, at no cost to the engine.

### 8.4 Checks

* `examples/overview.rs` writes `overview.json` through the public API alone, for the universes of the figure. `tools/ghl_diagram.py` must draw the same figure from it as from the harness's file: key by key equal for `patterns`, `structures`, `sweep`, `table`, `learning`, `waves` and `controller`. This is the acceptance test of the inspector, and it grows with the stages as the last column of Section 8.2 says.
* Every map read from VRAM equals the host's for the same universe and generation: IDs after a re-key through the tile contents, outcomes against `flat.rs`, region engines against the scheduler's own table.
* A trace equals the counters: the waves of `trace_step` sum to the per-level totals of the sample for that step.
* Fingerprints and counters are equal with every tap on and off, and the overhead of the counters is reported next to each stage's throughput targets.

## 9. Status of the implementation

The crate `ghl/` implements every stage. What is verified is exactness: every engine ends in the state that per-cell stepping and the host recursion give, and the long runs in Golly's final states. The GPU tests pass on the Vulkan and DX12 backends of an RTX 5060 Ti. No throughput target has been measured yet; `benches/stages.rs` measures them, on an idle machine.

| Part | Built | Verified by |
|---|---|---|
| Stage 0 | rules (B/S with H, V and B0 as Golly converts it; `@TABLE` and `@TREE` on every neighbourhood and symmetry; both canonical tables), RLE, macrocell (both flavours, read and write), palette PNG; the host recursion on bit-plane base tiles, interning in the harness's order; the brute force on `u64` words; the CPU `Universe`; the inspector on the host | `tests/stage0.rs`: both Hutton networks on all $2^{25}$ neighbourhoods, every Life-like network on all of its own, the 22 configurations of `selftest`, the universe under every step and controller; `tests/golly.rs`: final states against fingerprints recorded with bgolly; all 29 patterns of `head2head`, `runlength` and `life-h2h`, $2^{20}$–$2^{24}$ generations among them, end in Golly's state, at Golly's step and at the scheduler's (the long runs with `--ignored`) |
| Stage 1 | WGSL generated from the rule: plain and shared-tile kernels for gate networks, the byte-LUT kernel, a rule-tree kernel; a start-up check of both kernels against the host; calibration of latencies, $R_B$, $R_P$ and $R_I$; grid growth on the GPU; an external `Device` and `Queue` | `tests/gpu.rs`: every rule class, both kernels, from both parities |
| Stage 2 | one node table for all levels (the level in the key, an ID its slot), a tile table and store; every insert, the host's included, through the claim protocol; the sweep over active blocks of 16×16 tiles; bursts from a snapshot with the host learning the misses; the sweep tile from equation (5), changed through the host; reclaiming | `tests/gpu.rs`: the memo engine with host learning on Hutton32b and Life, sparse universes |
| Stage 3 | the miss path inside the burst's compute pass: collect, claim and commit, the level-synchronous recursion with tag-deduplicated waves, the floor's atlas stepped by a shared-tile kernel in one dispatch, interning by content, publish and re-probe; fixed capacities with early exit, an overflow flag and replay with larger capacities | `tests/gpu.rs`: the same universes with the GPU's miss path, checked not to fall back to the host |
| Stage 4 | the regime from the active area with $A_1$ and $A_2$ from equation (6) (the work of plain cells scaled to the rule's calibrated $R_B$), racing with exponential back-off, the universe-wide floor; Algorithm 3, and the price-based controller, a rent rule on the floor's price against plain cells', which passes at the roadmap's three configurations and ships disabled (see below); plain cells stepped as one block over the mask of plain-cells regions, whose results the recursion takes at the region level; the step rule; cycle detection on the host's root and on the GPU's ID grid | `tests/stage0.rs` (controllers, blocks, cycles), `tests/gpu.rs` (racing, GPU cycles), `tests/api.rs` (observer, trace against counters, rule files), `tests/controller.rs` (the controller in model times: the acceptance test at the roadmap's three configurations passes; the floor criterion at every configuration of paper Tables XXVII and XXVIII fails on one universe; a report at the step rule's other steps) |
| Stage 5 | node tables in up to eight shards and dense grids in up to four, each kernel spanning them through accessors; the table rebuilt larger past 45 % load; the Hutton32 circuit (597 gates, compiled by `ruletable2netlist` and checked on all $2^{25}$ neighbourhoods); the host's brute force four words at a time with AVX2 | `tests/gpu.rs` (shards forced small), `cpu::leaf` tests (AVX2 against scalar and per-cell stepping) |
| Sections 7 and 8 | the interface of Section 7; `Observer`, `Sample` and its history, maps answered at once on a universe that is not stepping and at the next epoch boundary otherwise, `StepTrace` | `examples/overview.rs`: all ten keys of `overview.json` equal the harness's (`tests/fixtures/compare_overview.py`) |

Where the build differs from the text above, and why:

* **Workspace.** `ghl` sits next to `ca-rule` (neighbourhoods, Golly tables, netlists) and `ruletable2netlist`; `research/` stays outside with `archive/`. Cargo builds a workspace into one target directory, and the harness's scripts (`reproduce.sh`, `tools/*.py`) call `research/target/release/ca-research`, so joining the workspace would move the binary they run; `archive/` is a separate application on wgpu 0.14. The two crates exist because the rule compiler needed neighbourhoods, tables and netlists without wgpu: one bottom crate gives both it and `ghl` a single definition of each.
* **IDs.** On the GPU a node's ID is its slot, because the claim protocol creates IDs inside one dispatch, with no counter shared between threads. The host keeps the harness's IDs, one sequence per level in interning order, because they make the overview JSON equal to the harness's ID for ID, the acceptance test of Section 8.4. The two are translated through the tables' contents, which Section 8.4 prescribes for comparing VRAM maps with the host's anyway.
* **Plain cells as blocks on the host only.** Pricing a region needs its requests and misses at every level below it. The GPU's miss path deduplicates windows over the whole universe and keeps no position below the sweep level, so its counters cannot be charged to regions; the host's position-aware recursion (`result_at`) can, so the policy, its prices and the block stepping run there, and the GPU memo engine runs the uniform floor. The GPU follows once the miss path counts per region.
* **The price-based controller is a rent rule, without the credit.** The rule of Stage 4 as written (both prices summed since the region's content last changed at the region level, plain cells entered once their sum had stayed 15 % below the floor's over eight steps, the floor credited with what other regions saved by hitting its results) failed its acceptance test: 1.59 and 1.43 times the choice in hindsight on Hutton32b, 1.30 times on Life, and 1.22–1.38 times the floor on the Unit Life Cell, catacryst and gotts-dots. In plain cells only their own price can be observed, so the floor's sum stopped at its last value and the region left as soon as its own sum caught up with it; on the mixed universe the regions changed mode 2,072 times. Every region's two prices per step in the floor runs (`GHL_CONTROLLER_TRACES=1`) show what separates the cases at the roadmap's configurations: content that does not recur, the ROM's data or a soup, costs the floor four and a half to six times plain cells for as long as it lasts, while a floor that learns, as in Life's first steps or on the counter's bursts, costs two to three times for a few steps and less than plain cells afterwards. The rule built adds, at every step with a region-level miss in which the region has content, the floor's price minus three times that of plain cells to an excess bounded below; the region enters plain cells when the excess exceeds two plain-cells steps of a nominal region, and returns to the floor, its excess cleared, at the first step without a region-level miss. Steps without content count for nothing because a region at a growing pattern's front misses at the region level through its neighbours' cells while plain cells would cost it one insert: counted, those steps carried catacryst's front regions into plain cells just as content arrived. With it the controller costs 35.7 ms on the Hutton32b universes at 16-generation steps and 64×64 regions (the choice in hindsight: 31.9 as the floor run prices it, 32.1 as run; Algorithm 3: 48.3; uniform F16: 65.8), 28.5 ms at 8-generation steps and 32×32 regions (23.9; 25.1), and 3.26 ms on Life at 64-generation steps and 256×256 regions (3.19; 3.16), with no universe more than 0.1 % above the uniform floor. Its two constants were chosen by simulating the rule on those traces (`tests/fixtures/controller_sim.py`), so the test's universes do not test them independently; in that simulation the pairs around them pass as well (a premium of 2.5 with a threshold of 2 to 4, 3 with 1 to 3, 3.5 with 1 or 2). The credit changed nothing once the premium was in place (1.097 times the choice in hindsight on Hutton32b at 16-generation steps without it, 1.108 with it; 1.021 on Life either way), so it is not paid for: it costs a hash-map update per window.
* **The controller ships disabled.** It passes at the roadmap's three configurations but not at every configuration of Tables XXVII and XXVIII, whose universes the criterion names (`price_based_controller_tables_xxvii_xxviii`): at 16-generation steps with regions of 64×64 to 1024×1024 the sums stay within 1.17 times the choice in hindsight on both rules (Hutton32b 35.7–52.7 ms against 48.3–131.0 for Algorithm 3; Life 3.57–4.43 ms against 4.40–4.43 at the uniform floor and 4.41–7.23 for Algorithm 3), and every universe stays within 1.004 times the floor except the Unit Life Cell, at 1.11 times with 64×64 regions and 1.34 times with 128×128. At the step rule's other steps (`price_based_controller_step_rule_scales`) it costs 1.41 times the floor on the Unit Life Cell at 32-generation steps and 1.35 and 1.42 times on the mixed universe and the ROM at 64-generation steps with 256×256 regions, where the choice in hindsight is within 1 % of the floor. The failures share one shape. A region of the Unit Life Cell at 16-generation steps is new for 240 generations, one period, missing at every level and costing the floor up to 4.6 times plain cells, the price of a soup at that step; then its content recurs for the rest of the run, some 11,000 generations, over which the floor's learning pays. A 256×256 region of the ROM at 64-generation steps is new for 256 generations and then recurs to the end of the run. While the controller has to choose, both price exactly as content that never recurs, which the roadmap's configurations need it to put in plain cells (the soup at 64-generation steps, the ROM at 8 and 16): content that is new at every level costs the floor a fixed multiple of plain cells for a given step and region, 4.64 times for Hutton32b at 8-generation steps, 5.52 at 16 and 4.61 for Life at 16, and the Unit Life Cell's costliest region sits at it. A region in plain cells learns nothing below itself, so leaving them once the content recurs pays a period of learning again, which only regions that stay active for several periods repay. Telling those apart needs a sign of recurrence that plain cells can see, such as how many of a stepped region's nodes interning finds already present, and an estimate of how long a region stays active; neither is built. Waiting out a universe's first generations, when every pattern is new to the memo, does not separate them either: the ROM's regions are new from the first generation as well, and in simulation on the floor runs' traces no plain cells before generation 128 costs Hutton32b 1.65 times the choice in hindsight at 16-generation steps and 1.92 times at 8, where the Unit Life Cell needs 192 to 256 generations. Until a controller passes, `Config::controller` stays at the uniform floor, as the roadmap prescribes, and `Controller::Priced` enables this one. Two more reasons hold in this build whichever controller passes: a controller runs in host epochs and holds them to the step rule's steps, at most eight base tiles' sides, where the host recursion alone lengthens its step without bound, which long HashLife runs depend on; and it prices plain cells at the GPU's rate, the region model's and its tests', while this build steps them on the host at about a three-hundredth of it for the circuits (`Rates::paper`).
* **Table rebuild and sweep-tile changes through the host.** Both read the state back, replace the tables or the tile level, and upload again, through the batch claim per level that every upload already uses, so they are exact by construction. Both are rare: a rebuild doubles the table, and a change of sweep tile follows eight sweeps of evidence. The parallel GPU passes (mark from the grids, reinsert, re-key) are not built.
* **Host learning writes the end state.** After a missed burst the host's positional sweep has computed every window of it, so its end state is known; the ID grid is written once instead of replaying the burst on the GPU, with the same result by exactness. The new entries are uploaded all the same, so the next burst hits.
* **Re-probe by a second sweep.** The GPU publishes results and then sweeps the active blocks again, instead of re-probing only the missed positions. A sweep is one dispatch either way; the second costs one more probe per active window in a sweep that missed, and needs no kernel of its own. Re-probing from the miss list is the cheaper alternative, not built.
* **Capacities.** Every wave starts at 4,096 windows per level and doubles after an overflow, then replays; capacities never shrink. Sizing them from the last burst's counters, as the roadmap has it, is not built: it would save dispatch width after quiet bursts and cost replays on learning bursts.
* **Traces and taps on the GPU.** `trace_step` gives the last burst's counters per level without the atlas's cells: the atlas buffer is overwritten by the next floor wave inside the same compute pass, and a copy cannot be recorded inside a pass, so keeping a wave's atlas needs a buffer per wave. On the host every wave and atlas is recorded at no cost. Taps are accepted but change nothing: maps are answered on request, at once when the universe is not stepping and at the next epoch boundary otherwise, so none has to be kept current between requests; a UI on the shared device reads them in place through `gpu_view`.
* **Cycles on the GPU.** Exact detection needs the earlier grid itself, not only its hash, so grids are kept on the host, which bounds them at $2^{16}$ tiles. A period with displacement shows on the GPU as a shifted grid, which a grid compared in place cannot see; the host's root is a node, free of position, which is why the host also catches displaced periods (gliders, spaceships).
* **The Hutton32 circuit was compiled from the table.** `ruletable2netlist` compiles and optimises any `@TABLE` rule instead of a hand-edited start; it works for every table rule, not only Hutton32, and its 597-gate result is checked on all $2^{25}$ neighbourhoods here and by the compiler.
* **Depth thresholds per rule.** $A_1$ and $A_2$ come from equation (6) with the plain cells' work per region scaled to the rule's $R_B$ and the memo engines' work kept at the counter's, because the latter is dominated by probes, whose rate does not depend on the rule.
* **The overview example writes `research/out/overview_ghl.json`.** `research/out/overview.json` is the harness's output and the oracle it is compared with; writing over it would compare the file with itself. `--out` writes elsewhere.
* **`Observer::tile` returns `Pending<Option<Tile>>`.** An observer can name any ID, and a `Pending<Tile>` for one that does not exist would never resolve.
* **AVX2 covers the adder tree and the two Hutton circuits.** These are the networks the paper measures; other networks are interpreted gate by gate, where the interpretation, not the lane width, bounds the rate.
* **Calibration's $R_P$** is measured with the claim kernel finding keys already committed, the same probe loop as the sweep's with one compare-exchange path untaken, rather than with a separate read-only kernel.
