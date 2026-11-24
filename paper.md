# HashLife on GPU

**GPU-resident hash-consed tiles for von Neumann-neighbourhood cellular automata — theory, measurements and an implementation plan**

2026-11-24 · measured on RTX 5060 Ti 16 GB (driver 591.59, wgpu 30 / Vulkan), i5-6600K, 16 GB RAM, Windows 11 · baseline: Golly 5.1 `bgolly`

---

## 0. Summary

The project's GPU simulator tops out near $10^{9}$ cell-updates/s while Golly's HashLife does several $10^{9}$ on the same pattern and handles 100k × 100k sparse universes. This report explains both facts with measurements, derives a hybrid that keeps HashLife's memoization but stores it where a GPU can use it, and backs every load-bearing constant with a prototype in [`research/`](research/).

Three results carry the argument.

1. **The $2 \cdot 10^{9}$ ceiling is not a property of GPUs.** It comes from `@workgroup_size(1)`, one compute pass per generation and 4 bytes per cell. The same per-cell LUT algorithm with those three things fixed runs at **$1.86 \cdot 10^{11}$ cell-updates/s** (bit-exact against the CPU), 170× the archived kernel, and is then limited by memory bandwidth. With the table replaced by a 367-gate circuit over bit-planes (§13) it runs at **$3.0 \cdot 10^{11}$** on any content.
2. **HashLife's weakness on dense patterns is memoization granularity, not memoization.** Golly memoizes 4×4-cell blocks one generation at a time and pays about 13.7 bytes of RAM per *novel* cell-update. Memoizing 32×32 → 16×16 blocks over 8 generations costs 0.02–0.09 bytes per cell-update and one table probe per 2048 cell-updates. A GPU can afford to brute-force everything below that floor.
3. **A table probe on this GPU costs the same from 128 MiB to 8 GiB** ($3.7 \cdot 10^{9}$ fetches/s, no TLB cliff), so VRAM converts directly into memo capacity. This is the lever the question asked about.

The resulting design is HashLife with three changes: a *memo floor* (no memoization below 16×16 tiles; a brute-force kernel computes misses), a *positional sweep* (the universe is a flat array of tile IDs; one generation step of that array is a data-parallel gather + hash probe), and a *level-synchronous schedule* for the sparse hierarchy above it.

| Regime | Example (your patterns) | Golly HashLife | Archived GPU kernel | This work | Status |
|---|---|---|---|---|---|
| Dense, brute force only | 169–268 M cells, tiled counter | — | $1.08 \cdot 10^{9}$ | **$1.86 \cdot 10^{11}$** (byte LUT), **$3.0 \cdot 10^{11}$** (bit-planes, §13) | measured, verified |
| Dense, small and active | `squares.rle` 650×423 | $3.7\text{–}8.2 \cdot 10^{9}$ (13–30 k gens/s) | $1.16 \cdot 10^{9}$ (4.2 k gens/s) | $5.0 \cdot 10^{10}$ (177 k gens/s); **$1.6 \cdot 10^{11}$ (560 k gens/s)** with 16 generations per dispatch (§13.4) | measured, verified |
| Dense, circulating ROM | `lotr1.ANSI-1M` 3095×3059 | $2.4 \cdot 10^{7}$ (**2.5 gens/s**, GC thrash) | $1.18 \cdot 10^{9}$ (125 gens/s) | brute: $1.81 \cdot 10^{11}$ (19 k gens/s), $3.9 \cdot 10^{11}$ as bit-planes (41 k gens/s); memo hit path: **469 k gens/s** (16×16 tiles), **1.61 M gens/s** (32×32) | measured, verified |
| Dense, 100k × 100k | 1024 phase-shifted ROMs, $1.0 \cdot 10^{10}$ cells | not attempted (single ROM already thrashes) | does not fit VRAM | **858 gens/s** (16×16 tiles), **20 349 gens/s** (32×32) | measured, verified tile-exact |
| Sparse, 100k × 100k | 2304 phase-shifted counters, $1.0 \cdot 10^{10}$ cells | 4.3 k gens/s (first 16 k gens), 97 k gens/s (1 M gens) | does not fit VRAM | **279 k gens/s** | measured, same final state as Golly |
| Sparse, idle circuits | displays, ROM, demultiplexer | $10^{5}\text{–}10^{9}$ gens/s | — | CPU engine at 0.5–1.9× Golly; identical states | measured |
| Worst case, uniform random states | synthetic | — | — | **$3.0 \cdot 10^{11}$** as bit-planes; byte LUT $1.4 \cdot 10^{10}$ ($4.2 \cdot 10^{10}$ with a 2 MiB LUT) | measured, verified |

Against the stated targets: dense patterns gain 43–280× from the brute-force kernel alone and $10^{3}\text{–}10^{5}$× once content recurs; sparse patterns are matched on the CPU by construction (same algorithm, validated against Golly on 12 patterns) and exceeded on the GPU when the universe is large.

**Follow-up on adaptive tiles and engine choice (§12).** Letting the tile size follow recurrence takes the same 100k × 100k ROM universe from 858 to $6.7 \cdot 10^{6}$ generations per second (tiles of 16 to 256 cells, measured on the GPU). Subdivision should stop at 16×16 windows; below that hashing costs more than the cells. A second engine, plain cells under a coarse memo, is 2.6–4× faster on content that is new at every small scale and 3–74× slower on everything else, so it is an escape hatch, not a default. The signal that separates the two is novelty, not density.

**Follow-up on a table-free rule (§13).** Turning the rule table into plain bit operations is feasible and pays in its bit-sliced form. Canonical Hutton32b is 367 gates, checked on all $2^{25}$ neighbourhoods; applied to five bit-planes they run at $3.0 \cdot 10^{11}$ cell-updates/s on the GPU whatever the content (1.6–1.8× the LUT kernel on real patterns, 22× on random states) at 5 bits per cell. Applied to one cell at a time they lose to the table by 2.5–34×, and tables small enough for an L1 cache lose to bit-planes too. On large grids the kernel is memory-bound, so a smaller circuit adds nothing there. On small grids it is latency-bound, and a variant that steps a shared tile 16 generations per dispatch runs the single counter at 780 k generations/s, 2.9× the plain kernel and above Golly.

**What is not yet built.** The GPU numbers for memoized stepping are *hit-path* measurements: the memo table was learned by the CPU reference engine and the GPU then ran pure probe sweeps. The on-GPU *miss path* (compute a missing transition, hash-cons the result, insert it) is specified in §3.3 and its two critical primitives are measured (lock-free insert: $10^{9}$/s; brute kernel: $1.86 \cdot 10^{11}$/s), but it has not been assembled into one pipeline. §5 gives the projection and §9 the plan.

**The rules (§10).** Tim Hutton's `Hutton32.table` and your `Hutton32b.rule` are the two canonical tables, and Golly runs exactly those. The WGSL ports in `archive/` did not when this work started: `hutton32.rule.wgsl` differed from Hutton32 in 2.4 M of $2^{25}$ neighbourhoods and `hutton32b.rule.wgsl` from Hutton32b in 3.1 M, through three causes in nine source lines, which is why several of your Golly patterns evolved differently in the GPU app. Both files are now corrected in place and equal the tables on every neighbourhood.

---

## 1. The problem, with measurements

### 1.1 Setup

All experiments use the harness in `research/` (Rust, ~6 k lines; commands in Appendix A).

* **Rules.** Two rules, each with a canonical table: Hutton32 (Tim Hutton's `Hutton32.table`, which Golly's built-in rule reproduces) and Hutton32b (your `Hutton32b.rule`). The WGSL ports in `archive/` deviated from both when this work started and were corrected in place (§10). The harness's CPU port follows the corrected files and is checked three ways: against the WGSL executed on the GPU and against both tables, 0 mismatches out of 33 554 432 each. The ports as first archived remain available as `hutton32-v1` and `hutton32b-v1`.
* **Patterns.** Fifteen patterns from your collection (Golly's `Patterns/Hutton32`, a clone of the Hutton32-Computer repository) and two PNG presets of the archived app, all copied to `research/patterns` together with a list of where each is used. Patterns that come from Golly are run with the canonical rule named in their header, which is what Golly runs; both tables are in `research/rules`.
* **Reference engine.** `research/src/hl.rs` implements the proposed algorithm on the CPU. It agrees with per-cell stepping in 22 configurations, and with Golly — same population, bounding box and content hash — on 10 patterns after $2^{24}$ generations, on the Hutton replicator after 262 144 generations, and on a 98 304 × 104 448 universe.
* **GPU checks.** Every GPU throughput figure below was followed by a read-back and a cell-exact or tile-exact comparison with the CPU.

Throughput is given in cell-updates per second (cells × generations / second) over the bounding box.

### 1.2 Where the $2 \cdot 10^{9}$ ceiling comes from

The archived kernel, reproduced verbatim in structure (`gpu-baseline`):

| Grid | gens/s | cell-updates/s |
|---|---|---|
| 193 × 199 (default counter) | 23 892 | $9.2 \cdot 10^{8}$ |
| 635 × 451 (old counter) | 3 786 | $1.08 \cdot 10^{9}$ |
| 652 × 425 (`squares`) | 4 185 | $1.16 \cdot 10^{9}$ |
| 5080 × 3608 (18 M cells) | 58.8 | $1.08 \cdot 10^{9}$ |

The rate does not depend on grid size, so neither memory bandwidth nor the LUT is the limiter. Three structural costs explain it:

| Cause | Evidence |
|---|---|
| `@workgroup_size(1)`: one thread per workgroup | throughput is flat at $\approx 1.1 \cdot 10^{9}$ *workgroups*/s for every grid |
| One compute pass per generation | a pass costs 49–97 µs in wgpu; a chained dispatch inside one pass costs 3.4 µs (`gpu-latency`). This alone caps small patterns at ≈10–24 k gens/s |
| One `u32` per cell | 4× the memory traffic of a byte per cell |

With 4 cells per `u32`, 8×8 workgroups, ping-pong buffers with a zero ring and all generations of a submit in one pass (`gpu-brute`):

| Grid | gens/s | cell-updates/s | Limited by |
|---|---|---|---|
| 199 × 205 (default counter) | 274 544 | $1.1 \cdot 10^{10}$ | dispatch latency (3.6 µs/gen) |
| 658 × 431 (`squares`) | 177 445 | $5.0 \cdot 10^{10}$ | dispatch latency |
| 5128 × 3656 (18.7 M) | 9 354 | $1.75 \cdot 10^{11}$ | bandwidth |
| 15 384 × 10 968 (169 M) | 1 100 | **$1.86 \cdot 10^{11}$** | bandwidth: 371 GB/s of 448 GB/s |

Workgroup shape (8×8 … 64×4) and backend (Vulkan, DX12) change this by less than 2 %. The kernel is at 83 % of the card's memory bandwidth at 2 bytes of traffic per cell-update.

Content matters only for adversarial input (`gpu-brute2`, $8192^{2}$):

| Content | Full 32 MiB LUT | Class-compressed 2 MiB LUT |
|---|---|---|
| Tiled `chaotic.rle` | $1.8 \cdot 10^{11}$ | $1.3 \cdot 10^{11}$ |
| Uniform random states, decaying | $3.8 \cdot 10^{10}$ | $1.0 \cdot 10^{11}$ |
| Uniform random states, held constant | $1.4 \cdot 10^{10}$ | $4.3 \cdot 10^{10}$ |

Random states touch the whole LUT and miss the 32 MB L2 cache. The rule distinguishes only 15 classes of states per neighbour position (centre: 29), so the same function fits a 21-bit index; that table always fits L2. Real patterns, including your `chaotic.rle`, stay at $1.8 \cdot 10^{11}$ with the full LUT. §13 removes the table altogether, and with it the dependence on content.

### 1.3 Where HashLife stands

`bgolly` 5.1, RuleLoader with your `Hutton32b.rule` (or the native JvN algorithm for Hutton32), 8 GB memory limit, best step size:

| Pattern | Size | Golly | Notes |
|---|---|---|---|
| `advanced counter v2` | 191 × 197 | $5.8 \cdot 10^{5}$ gens/s ($2.2 \cdot 10^{10}$) | $5 \cdot 10^{4}$ gens/s at step 1 |
| `squares` | 650 × 423 | $1.3\text{–}3.0 \cdot 10^{4}$ gens/s ($3.7\text{–}8.2 \cdot 10^{9}$) | active display |
| `chaotic` (active phase) | 151 × 132 | $3.1\text{–}4.9 \cdot 10^{4}$ gens/s ($6\text{–}10 \cdot 10^{8}$) | 22 GCs in 16 k gens |
| `32x32 movie` (while playing) | 928 × 1513 | $\approx 1.0 \cdot 10^{4}$ gens/s | 524 k gens of activity (52 s), then idle |
| `lotr1.ANSI-1M` | 3095 × 3059 | **2.5 gens/s** ($2.4 \cdot 10^{7}$) | 18 GCs in 180 s; one step of 4096 never finished |
| Hutton replicator | grows to 6813 × 6691 | $2.4 \cdot 10^{3}$ gens/s | 34 GCs for 262 k gens |
| Idle circuits (displays, ROM, demux) | up to 2182 × 2620 | $10^{5}\text{–}10^{9}$ gens/s | fully memoized |

Your figures ($2 \cdot 10^{9}$ on the GPU, $4.3 \cdot 10^{9}$ for HashLife) match what this machine gives on `squares`-class patterns: $1.16 \cdot 10^{9}$ and $3.7\text{–}8.2 \cdot 10^{9}$.

`lotr1.ANSI-1M` is the instructive case. It is a 9.5 M-cell circulating ROM with period 528: every 16×16 block is different from every other at every moment of the period. A single CPU core doing plain per-cell LUT stepping manages $3.0 \cdot 10^{8}$ cell-updates/s on it, **13× faster than HashLife**; the archived GPU kernel does 125 gens/s and the fixed brute kernel 18 992 gens/s (cell-exact over a full period).

### 1.4 Two diagnoses

* The GPU baseline lost two orders of magnitude to dispatch structure.
* HashLife loses on dense content because of *what it chooses to remember*, which §2 quantifies.

---

## 2. Theory

### 2.1 The price of a probe

Let a tile of level $L$ be a $2^L \times 2^L$ block, $s = 2^L$. A $2s \times 2s$ window determines its central tile $s/2$ generations later (radius 1). Call that a **tile-step**; it is worth

$$V_L = s^2 \cdot \frac{s}{2} = 2^{3L-1} \text{ cell-updates} \qquad (256,\ 2048,\ 16\ 384,\ 131\ 072 \text{ for } L = 3, \ldots, 6).$$

There are three ways to obtain one: brute force (cost $V_L / R_B$), a memo hit (cost $1/R_P$), or a memo miss (cost $1/R_P$ plus the work to compute and store it). The rates were measured on this GPU:

| Constant | Value | Source |
|---|---|---|
| $R_B$, brute-force rate | $1.86 \cdot 10^{11}$ cell-updates/s | `gpu-brute` |
| $R_P$, random 32-byte table fetch, table ≤ 16 MiB | $3.2\text{–}3.5 \cdot 10^{10}$ /s | `gpu-probe` |
| $R_P$, table 32 MiB | $1.3 \cdot 10^{10}$ /s | |
| $R_P$, table 128 MiB … 8 GiB | **$3.7\text{–}4.0 \cdot 10^{9}$ /s, flat** | |
| $R_I$, concurrent find-or-insert (32-bit CAS) | $1.0\text{–}1.9 \cdot 10^{9}$ /s | `gpu-insert` |

The exchange rate **$\rho = R_B / R_P \approx 50$** says one probe into a large table costs as much as brute-forcing 50 cell-updates ($\rho \approx 5.5$ when the table fits L2). (The bit-plane kernel of §13 raises $R_B$ to $3.0 \cdot 10^{11}$ and $\rho$ to 80. The figures in §2–§12 are for the byte-LUT kernel; §13.6 restates the ones that move.) A memo level pays only if $V_L \gg \rho$:

| Tile | $V_L$ | $V_L / \rho$ = best possible gain over brute force |
|---|---|---|
| 8×8, 4 gens | 256 | 5× |
| 16×16, 8 gens | 2 048 | 41× |
| 32×32, 16 gens | 16 384 | 330× |
| 64×64, 32 gens | 131 072 | 2 600× |

So on this hardware a sweep that probes every tile should not use tiles smaller than 16×16. (Inside the recursion of §2.5 a smaller window is probed only when the one above it missed, and one more level pays; §12.3 measures it.) On a CPU core the same ratio comes out similar ($R_B \approx 3\text{–}4 \cdot 10^{8}$, $R_P \approx 10^{7}$ for an out-of-cache table, $\rho \approx 30\text{–}60$): **the optimal floor is not 2×2 on a CPU either.** Golly's generic HashLife memoizes 4×4 → 2×2 over one generation, worth $V = 4$, an order of magnitude below $\rho$. That is fine when everything hits and expensive when content is new.

### 2.2 Memory per memoized cell-update

A Golly `ghnode` is six pointers, 48 bytes (`ghashbase.h`), and a 2×2 leaf occupies a node slot too. The bottom memo entry covers 4 cell-updates, so novel content costs 12 B per cell-update there and, summing the geometric series over levels, **≈13.7 B per novel cell-update** overall. `maxmem` only triggers garbage collection, and the collector keeps what is reachable from the current root.

A tile-level entry is 32 B (four key IDs, a value, padding) and covers $V_L$ cell-updates; a new 16×16 tile adds 160 B of content (5 bits per cell).

| Scheme | Bytes per novel cell-update | One period of `lotr1.ANSI-1M` ($5 \cdot 10^{9}$ cell-updates) |
|---|---|---|
| Golly `ghashbase` | 13.7 | ≈ 70 GB |
| Memo at 16×16 tiles, entry only | 0.016 | 79 MB |
| Memo at 16×16 tiles, every result tile new | 0.094 | 473 MB |
| Memo at 32×32 tiles, entry only | 0.002 | 20 MB |

This is why HashLife "consumes all available RAM" on dense circuits and why the ROM never completes a period under an 8 GB limit, yet its whole period fits a tenth of this card's VRAM at a coarser grain. At 16×16 tiles a 16 GB card holds about $4 \cdot 10^{8}$ entries ($8 \cdot 10^{11}$ cell-updates) when result tiles recur, or $8 \cdot 10^{7}$ ($1.7 \cdot 10^{11}$ cell-updates) when every result tile is new.

### 2.3 The speedup law

With miss rate $\mu_L$ (fraction of tile-steps not found) and a miss costing $\kappa$ times the brute-force price of a tile-step, the speedup over brute force is

$$S_L = \frac{1}{\rho / V_L + \kappa \mu_L}.$$

Memoization at level $L$ wins while $\mu_L < (1 - \rho/V_L)/\kappa$. For the simplest miss path (evaluate the whole $2s \times 2s$ window for $s/2$ generations, no pyramid shrinking, plus two inserts) $\kappa \approx 4.5$ at $L = 4$, so the break-even miss rate is about 22 %; a region-merged miss path (§3.4) approaches $\kappa \approx 1.1$ and break-even near 85 %.

Measured miss rates of the flat tile automaton on your patterns (`flat`, second half of each run, so start-up transients are excluded):

| Pattern | 8×8 | 16×16 | 32×32 | 64×64 | 128×128 |
|---|---|---|---|---|---|
| `advanced counter v2` (262 k gens; period is 513 024) | 0.41 % | 1.70 % | 5.4 % | 9.4 % | 12.2 % |
| `32x32 movie`, playing | | 0.16 % | | | |
| `16x16 movie` | | 0.12 % | 0.56 % | | |
| `Segmented ROM (256k)` | | 0.07 % | 0.59 % | | |
| `component-basis` | | 0.09 % | 0.64 % | | |
| `squares` | | 3.5 % | 7.0 % | | |
| `chaotic` | 10.5 % | 25.5 % | 34.9 % | | |
| `lotr1.*` ROMs | | 96 % for 528 gens, then 0 | 0 after 1056 | | |
| displays, demultiplexer, `cpu`, `BIN2BCD` (idle) | | 0 | 0 | | |

The counter never repeats a global state within the run, yet 98.3 % of its 16×16 tile-steps are repeats. This is the redundancy HashLife exploits at its lowest levels; the point of §2.1–2.2 is that it can be exploited without storing anything smaller.

### 2.4 The renormalised automaton

Hash-consing gives every distinct tile a small integer ID. The memoized map

$$R_L : (\mathrm{ID}_{nw}, \mathrm{ID}_{ne}, \mathrm{ID}_{sw}, \mathrm{ID}_{se}) \mapsto \text{ID of the centred tile, } s/2 \text{ generations later}$$

is then itself a cellular automaton over tile IDs with a 2×2 neighbourhood on a lattice that shifts by half a tile each step (two alternating phases bring it back). Its alphabet and rule table are discovered lazily. Simulating the original CA for $s/2$ generations is **one gather of four IDs and one hash probe per tile**, regardless of $L$.

```
phase A (tiles at k·s)          phase B (tiles at k·s + s/2)        phase A
┌────┬────┬────┐                    ┌────┬────┐                 ┌────┬────┬────┐
│ a  │ b  │ c  │   R(a,b,d,e) ─►    │ p  │ q  │                 │    │    │    │
├────┼────┼────┤   R(b,c,e,f) ─►    ├────┼────┤  R(p,q,r,t) ─►  │    │ e' │    │   e' = e after s generations
│ d  │ e  │ f  │   …                │ r  │ t  │                 │    │    │    │
├────┼────┼────┤                    └────┴────┘                 └────┴────┴────┘
│ g  │ h  │ i  │
└────┴────┴────┘
```

This *positional sweep* is what a GPU is good at, and it has two properties plain HashLife lacks on dense content. Overlap is shared for free: every window is evaluated exactly once per sweep, so the recursion's thirteen sub-calls for eight results' worth of work disappear. And nothing above level L is hashed, so levels where every node would be unique cost nothing.

A flat array of IDs is also a far smaller representation of the universe than cells: 4 bytes per 256 or 1024 cells, 64× to 256× less than a byte per cell. A 100k × 100k universe is 151 MiB per buffer at 16×16 tiles, where a byte per cell would need 10 GB.

### 2.5 Level-synchronous HashLife

The sweep at one level is the bottom of a hierarchy. The full algorithm is HashLife's recursion run in a different order: instead of depth-first, each level processes one *batch* of distinct nodes per wave.

```
solve(level l, set Q of distinct nodes):
    M ← nodes of Q without a memoized result            # one probe pass
    if l is the floor:  brute-force every window in M   # one kernel
    else:
        S1 ← distinct { nine overlapping sub-windows of each m ∈ M }
        solve(l−1, S1)
        S2 ← distinct { four windows built from those nine results }
        solve(l−1, S2)
        result(m) ← node of the four results
```

Every line is a data-parallel pass over a batch. Tomas Rokicki sketched this "lockstep" order in pseudocode on the Golly list in 2014; no implementation or measurement of it was found. `research/src/hl.rs` is an implementation, validated against Golly, and it reports the batch sizes a GPU would see. The flat sweep is the special case where the top level is a positional array instead of a root node.

Two consequences shape the design.

* **Sequential depth.** With floor $a$, the bottom result advances $2^{a-1}$ generations, so $T$ generations need $T / 2^{a-1}$ dependent waves unless a higher level hits. Batches inside a wave are parallel; waves are not. Fixed per-wave cost therefore bounds any pattern whose misses are few but constant.
* **Regime by batch size.** Large batches belong on the GPU. A handful of nodes per wave is cheaper on the CPU, where a wave costs microseconds and no dispatch.

### 2.6 Latency constants

Measured with `gpu-latency` (wgpu 30, Vulkan, Windows):

| Operation | Cost |
|---|---|
| Dispatch chained inside one compute pass | 3.3–4.2 µs (typically 3.4) |
| New compute pass | 49–97 µs |
| Separate `queue.submit` | 150–460 µs |
| Dispatch + 16-byte read-back to the CPU | **0.1–2.4 ms** (five runs: 99 µs to 2 390 µs) |

A sweep of $n_d$ chained dispatches advancing $2^{L-1}$ generations cannot exceed $2^{L-1} / (n_d \cdot 3.4\ \mu\text{s})$ generations per second: $2.4 \cdot 10^{6}$ gens/s for a hit-only sweep at 16×16 tiles. Anything that needs the CPU inside the loop is one to three orders of magnitude slower, so the GPU must run autonomous stretches and the CPU must never sit in the per-wave path.

**Dispatches whose size is computed on the GPU.** The hit path has fixed sizes. The miss path does not: the number of misses, and then the number of distinct misses, are only known on the GPU. There are two ways to size its dispatches without the CPU, and they differ by a factor of five to nine in wgpu:

| Miss-path wave of 16 dispatches (probe, claim, commit, expand, 8 × evaluate, intern, commit, publish, re-probe) | Cost |
|---|---|
| Indirect dispatch. wgpu cannot write a buffer as storage and consume it as indirect arguments in one pass, so two dependent counts mean three passes | 260–490 µs |
| The same with `InstanceFlags::VALIDATION_INDIRECT_CALL` removed (each indirect call then costs 3.5 µs instead of ≈13 µs) | 135–165 µs |
| **Fixed capacity with early exit**: one pass, every stage dispatched for 4096 items, invocations beyond the live count return at once | **51–55 µs** |

An idle 64-thread workgroup costs about 1 ns once the 3.2 µs fixed cost is paid: a stage dispatched for 65 536 items takes 4.3–7 µs, for a million items 26–31 µs. So the learning path should use fixed capacities, chosen from miss counters that the CPU reads without blocking and that may be many waves stale; a wave that overflows its capacity leaves a sentinel in the affected tiles and is replayed from the last snapshot with a larger one. Every latency figure for the miss path in §5 and §12.6 assumes this scheme (16 dispatches ≈ 54 µs per wave). With indirect dispatch they would be 2.5–9× worse.

---

## 3. The algorithm

### 3.1 Data in VRAM

| Structure | Layout | Size |
|---|---|---|
| ID grids A, B | `u32` per tile, two phases | 4 B per 256 cells at $L = 4$ |
| Memo table per level | open addressing, linear probing, 32 B: 4 key IDs, value, spare | $2 \cdot 10^{8}$ entries in 6.4 GB |
| Claim words | one `atomic<u32>` per slot (insert protocol) | 4 B per slot |
| Tile store | 16×16 cells packed at 5 bits: 160 B, indexed by tile ID | only for the floor level |
| Active-block list | origins of blocks that are not known-empty | a few words per block |
| Atlas | scratch cell grid for windows being brute-forced | sized to the miss batch |

A node's ID is its slot index, so there is no separate ID allocator, and "children of ID x" is a direct read. Entries never move; reclaiming space is a rebuild (§3.6). wgpu caps one binding below 2 GiB on both backends, so large tables are sharded across bindings; `gpu-probe` runs 8 × 1 GiB this way at full speed.

### 3.2 Hit path

One dispatch per sweep. Each invocation gathers four IDs (out-of-range reads return the empty tile), hashes them, probes, and writes one ID. No atomics are involved because inserts happen in separate dispatches. This is the kernel measured in §4.2.

### 3.3 Miss path

A missed tile writes a sentinel and appends its index to a miss list. Then, all on the GPU:

1. **Deduplicate and claim.** Every miss runs find-or-insert on its key. WGSL has only 32-bit compare-exchange, so a 128-bit key cannot be installed atomically. The protocol avoids reading half-written keys altogether:
   * `claim[slot]` is 0 (empty), `j+1` (claimed in this batch by batch element j) or has bit 31 set (committed earlier);
   * a thread that finds a slot claimed by element t compares against element t's key **read from the immutable batch input**, never from the table;
   * committed slots are immutable and compared in place;
   * a second trivial dispatch marks this batch's claims committed.

   One dispatch resolves the whole batch, lock-free. `gpu-insert` runs it on 16.8 M requests per batch (4× duplication) into a 640 MiB table: $0.76\text{–}1.29 \cdot 10^{9}$ requests/s including the commit pass; across 67 M requests, 0 unresolved, 0 wrong keys, 0 duplicate IDs.
2. **Expand.** Each distinct missing key copies its four tiles into a $2s \times 2s$ window of the atlas.
3. **Evaluate.** The brute-force kernel runs $s/2$ generations over the atlas. Windows sit side by side; garbage from a neighbouring window travels one cell per generation and cannot reach the central tile.
4. **Intern.** The central tile is hash-consed by content with the same claim protocol (a pending owner's content is read from the atlas).
5. **Publish.** The result ID is written into the memo entry and the missed positions are re-probed.

When the floor is below the sweep level, a miss first descends one level (nine probes, then four) and only the floor's misses reach step 2; that is `solve` from §2.5.

### 3.4 Level control and raw regions

§12 measures these rules on your patterns and revises some of the thresholds.

* **Sweep level.** Start at 16×16. Raise the level while the measured hit rate stays near 1 (each level doubles generations per dispatch and quarters the tile count); lower it when $\kappa \mu$ exceeds the gain in §2.3. Tables for all levels are shared, so changing level is a parallel re-key of the ID grid.
* **Always insert while memory lasts.** A policy that stops memoizing at a high miss rate would never learn the ROM, whose miss rate is 96 % for exactly one period and 0 afterwards. Recurrence is detectable only within the horizon that fits in memory, which is another reason VRAM matters: for a $10^{7}$-cell dense circuit, 8 GB holds about 8 500 generations of completely new history.
* **Raw regions.** Where content has not recurred after the table has wrapped, keep the region as cells and step it with the brute kernel, probing again with exponential back-off. Contiguous misses are evaluated as one region with a halo rather than as isolated windows, which removes most of $\kappa$. The worst case is then the brute-force rate, not a multiple slower.

### 3.5 Sparse universes

* **Active blocks.** The key of four empty tiles maps to the empty tile, so a block that is empty with an empty one-block border stays empty and can be skipped exactly. The sweep dispatches over a list of active blocks; a block joins when a border tile of a neighbour becomes non-empty. Work is proportional to content. Measured in §4.3.
* **Hierarchy.** Idle or periodic regions hit at coarse levels. Raising the sweep level, or running `solve` from a coarse top grid, gives HashLife's behaviour on regular content: cost per step independent of the step size.
* **Host engine.** Small universes and idle circuits are faster on the CPU, where the same algorithm matches Golly (§4.4). The engine choice is by work per wave, and the two engines exchange state only at epoch boundaries because a read-back costs 0.1–2.4 ms.
* **Cycle detection.** IDs are canonical, so two ID grids are equal exactly when the universes are. A hash of the grid taken every epoch detects a globally periodic universe, after which any generation is a table lookup.

### 3.6 Memory management, rendering, I/O

* **Reclaiming.** When a table passes its load limit, mark entries reachable from the current grids, rebuild into a fresh table with the insert kernel, and re-key the grids. All three are parallel passes. The memo is a cache: dropping it costs a relearning transient, never correctness.
* **Rendering.** A pixel reads its tile ID, then the tile store. Zoomed out, a per-tile mean colour computed at intern time replaces the cells, so drawing cost follows the screen, not the universe.
* **Patterns.** A macrocell file is a hash-consed quadtree already; `research/src/mc.rs` and `cmd_sparse.rs` read and write it.
* **Exactness.** Keys are compared in full and tiles by content. There is no probabilistic hashing anywhere; results are bit-identical to per-cell stepping.

---

## 4. Measurements

### 4.1 Primitives

Collected from §1.2, §2.1 and §2.6: brute kernel $1.86 \cdot 10^{11}$ cell-updates/s ($3.0 \cdot 10^{11}$ as bit-planes, §13); table fetch $3.7 \cdot 10^{9}$/s out of cache and $3.4 \cdot 10^{10}$/s in cache, flat to 8 GiB; find-or-insert $10^{9}$/s; chained dispatch 3.4 µs; a 16-dispatch miss-path wave 51–55 µs with fixed capacities; read-back 0.1–2.4 ms.

### 4.2 Dense universes, memo hit path (`gpu-sweep`)

Universe: N × N copies of `lotr1.ANSI-1M`, each started at a different phase of its 528-generation period, so that no two large blocks are equal. The table was learned by the CPU engine over one period; the GPU ran sweeps; every tile was then compared with the CPU's prediction.

| Tile | Universe | Cells | Table | gens/s | cell-updates/s | probes/s | Check |
|---|---|---|---|---|---|---|---|
| 16×16 | 1 ROM, 3168 × 3136 | $9.9 \cdot 10^{6}$ | 256 MiB | **468 929** | $4.7 \cdot 10^{12}$ | $2.3 \cdot 10^{9}$ | 0 misses, 0 of 38 808 tiles wrong |
| 32×32 | 1 ROM, 3232 × 3200 | $1.0 \cdot 10^{7}$ | 64 MiB | **1 607 022** | $1.7 \cdot 10^{13}$ | $1.0 \cdot 10^{9}$ | 0 of 10 100 wrong |
| 16×16 | 8×8 ROMs, 25 344 × 25 088 | $6.4 \cdot 10^{8}$ | 256 MiB | 10 585 | $6.7 \cdot 10^{12}$ | $3.3 \cdot 10^{9}$ | 0 of 2.5 M wrong |
| 16×16 | 32×32 ROMs, 101 376 × 100 352 | $1.02 \cdot 10^{10}$ | 256 MiB | 848–858 | $8.6\text{–}8.7 \cdot 10^{12}$ | $4.2 \cdot 10^{9}$ | 0 of 39.7 M wrong |
| 32×32 | 32×32 ROMs, 103 424 × 102 400 | $1.06 \cdot 10^{10}$ | 64 MiB | **20 349** | $2.2 \cdot 10^{14}$ | $1.3 \cdot 10^{10}$ | 0 of 10.3 M wrong |

The single ROM is the pattern Golly runs at 2.5 gens/s; its rows are limited by dispatch latency, not by probes. The 100k × 100k dense universe occupies 151 MiB per ID buffer at 16×16 tiles (39 MiB at 32×32). At that size the gain over the brute kernel is 46× and 1160×, as §2.1 predicts from $V_L R_P / R_B$.

Learning this table took the CPU 80–90 s per ROM period (2 462 688 entries at 16×16). The projection for the GPU miss path is 0.1–0.5 s (§5).

### 4.3 Sparse universe (`gpu-sparse`)

Universe: 2304 copies of `advanced counter v2`, one per 2048 × 2176-cell block, each at a different phase of its 513 024-generation period: 98 304 × 104 448 cells, $5.5 \cdot 10^{7}$ live cells, 0.79 % of tiles non-empty. The same universe was exported as a macrocell file and run by Golly and by the CPU engine.

| Engine | 16 384 gens | Rate | Final state |
|---|---|---|---|
| Golly HashLife, step 16 | 15.95 s | 1 027 gens/s | |
| Golly HashLife, step 1024 | 3.88 s | 4 224 gens/s | population 55 478 381, hash `144be5c0cde9cb81` |
| Golly HashLife, step 16 384 | 3.79 s | 4 323 gens/s | |
| CPU reference engine, step 1024 | 2.78 s | 5 894 gens/s | identical |
| GPU, flat sweep (all 40 M tiles) | 1.81 s | 9 059 gens/s | 0 of 40.1 M tiles wrong |
| GPU, active blocks | 0.059 s | **278 648 gens/s** | identical to Golly's |

Over $2^{20}$ generations Golly averages 61 878 (step 1024) to 96 581 gens/s (step 65 536) as its memo warms. The GPU figure is a steady-state hit-path rate with the counter's cycle already learned (3.2 s on the CPU).

### 4.4 CPU head-to-head (`tools/head2head.py`)

Same pattern, same rule, step 1024, single thread. "Engine" is the unoptimized reference implementation (a `HashMap` per level, sort-based dedup).

| Pattern | Gens | Golly (s) | Engine (s) | Same final state |
|---|---|---|---|---|
| Segmented ROM (256k) | $2^{24}$ | 1.83 | 1.55 | yes |
| serial display (32×32) | $2^{24}$ | 0.16 | 0.28 | yes |
| parallel display (16×16) | $2^{24}$ | 0.019 | 0.010 | yes |
| 16×16 movie | $2^{24}$ | 3.48 | 3.17 | yes |
| efficient demultiplexer | $2^{24}$ | 0.024 | 0.030 | yes |
| cpu | $2^{24}$ | 0.017 | 0.020 | yes |
| BIN2BCD | $2^{24}$ | 0.047 | 0.060 | yes |
| advanced counter v2 | $2^{24}$ | 1.73 | 1.50 | yes |
| chaotic | $2^{24}$ | 5.27 | 7.95 | yes |
| 32×32 movie | $2^{24}$ | 52.5 | 55.6 | yes |
| squares | $2^{20}$ | 80.8 | 134.4 | yes |
| Hutton replicator (step 16 384) | 262 144 | 111.5 | 219.2 | yes |

The level-synchronous schedule with a 4×4 floor runs at 0.5–1.9× Golly's speed, faster on four of the twelve. It is the same algorithm doing a comparable number of node operations; the remaining gap is table engineering (Golly uses a hand-rolled chained table with prefetching, the engine a general-purpose `HashMap` and a sort per wave).

---

## 5. Projections for what is not built

The on-GPU miss path is a composition of measured parts. Per tile-step at 16×16 tiles:

$$t = \frac{1}{R_P} + \mu \left( \frac{4 V_L}{R_B} + \frac{2}{R_I} + t_\text{gather} \right) \approx 0.27\ \text{ns} + \mu \cdot 50\ \text{ns}$$

| Workload | $\mu$ at 16×16 | Projected cell-updates/s | vs Golly | vs archived GPU |
|---|---|---|---|---|
| ROM after its first period | 0 | $8.6 \cdot 10^{12}$ (measured, §4.2) | $3.6 \cdot 10^{5}$× | 8 000× |
| Segmented ROM, movies, component sheet | 0.07–0.16 % | $6\text{–}7 \cdot 10^{12}$ | | |
| Counter array, non-repeating | 1.7 % | $1.8 \cdot 10^{12}$ | 80× | 1 700× |
| `squares`-like displays | 3.5 % | $1.0 \cdot 10^{12}$ | 120–270× | 900× |
| `chaotic`-like | 25 % | $1.6 \cdot 10^{11}$ → use brute kernel, $1.8 \cdot 10^{11}$ | 180× | 170× |
| ROM during its first period | 96 % | $4 \cdot 10^{10}$ for 0.12 s | | |

The Golly column compares with Golly's measured rate on a single instance of the pattern (§1.3). The projections hold when a sweep contains enough tiles to hide dispatch latency, roughly $10^{5}$ tiles or $2.6 \cdot 10^{7}$ cells. Below that the latency bound of §2.6 applies: a sweep that has misses needs about 16 chained dispatches (probe, claim, commit, expand, 8 × evaluate, intern, commit, publish, re-probe). Dispatched with fixed capacities that is 51–55 µs per 8 generations as measured in §2.6, so **≈145 k gens/s for small active patterns**, slightly below the brute kernel's 177–275 k gens/s. Folding the eight evaluation dispatches into one helps only if the work stays spread over threads: one thread per window takes 630 µs per wave, while a workgroup sharing a tile does 8 generations in about 12 µs on a small grid (§13.4). The latter was not tried on an atlas; it would take a wave from about 54 µs to about 37 µs.

For the Hutton replicator the reference engine counts $1.45 \cdot 10^{9}$ node probes and 262 k waves for 262 144 generations. On the GPU that is 0.4 s of probes plus 5–15 s of wave latency against Golly's 111 s.

Three things would move these numbers and are not modelled: batches that overflow the atlas, table rebuilds, and the cost of activating blocks.

---

## 6. Verdict against the targets

| Target | Evidence | Basis |
|---|---|---|
| Dense ≥ 10× | Brute kernel: 43× (`squares`), 170× (large grids, byte LUT) and 280× (bit-planes, §13) over the archived kernel; 6–14× over Golly on `squares`, 7 700× on the ROM (16 000× as bit-planes) | measured, cell-exact |
| | Worst-case content, uniform random states: $3.0 \cdot 10^{11}$ as bit-planes (280×). The byte-LUT kernel falls to $1.4 \cdot 10^{10}$ there (13×) | measured, cell-exact |
| | Recurring content: $8.6 \cdot 10^{12}\text{–}2.2 \cdot 10^{14}$ at $10^{10}$ cells | measured hit path |
| | Non-repeating logic: $\approx 10^{12}$ | model on measured constants |
| Sparse ≥ HashLife | Same algorithm on the CPU, unoptimized: 0.5–1.9× Golly on 12 patterns, identical final states | measured |
| | 100k × 100k sparse on the GPU: 279 k gens/s vs 4.3–97 k | measured hit path |
| | Novel sparse activity (replicator): 5–15 s projected vs 111 s | model |

The closest case is a small universe (under $\sim 10^{5}$ cells) with constant novelty, such as a single counter: Golly runs it at $5.8 \cdot 10^{5}$ gens/s, the CPU engine at $6.5 \cdot 10^{5}$, the GPU brute kernel at $2.7 \cdot 10^{5}$. With 16 generations per dispatch on a shared tile the GPU reaches $7.8 \cdot 10^{5}$ (§13.4), 1.35× Golly. That margin is small and rests on one machine.

---

## 7. Alternatives considered

**FFT convolution.** The convolution theorem computes a weighted neighbour sum in O(N log N) per step independent of kernel radius, and SmoothLife and Lenia use it for radii of 10–30. Three things rule it out here. At radius 1 the stencil has five taps; one published benchmark has FFT Life 3× slower than direct convolution [21], and another has convolution-based Life in PyTorch 121× slower than a bit-packed CUDA kernel [22]. The rule must be a pointwise function of a linear statistic; for a non-totalistic 32-state rule the only such statistic is an injective base-32 code of the neighbourhood, 25 exact bits, which is precisely the LUT index the kernel already builds with five loads. And skipping generations in the Fourier domain needs linear (or solvable-group) structure — Rule 90 and its relatives — while prediction of general CA is P-complete. For uniform kernels Golly's Larger-than-Life uses running sums, which beat FFT at any radius.

**Bit-slicing.** Five bit-planes and a synthesised Boolean circuit make the kernel immune to content and cut traffic from 2 to 1.25 bytes per cell-update. The first version of this report estimated a gain of at most ≈1.6× and left it undone; it has since been built and measured, and is no longer an alternative but the recommended kernel. See §13.

**HashLife ported as is.** Depth-first recursion over a pointer structure has no batch to give a GPU, and the 2×2 floor is the wrong grain (§2.1). The one documented attempt (LaForest 2010) stopped at a 4×4 → 2×2 lookup table and reported 1.18× over plain Life.

**A larger direct table.** Two generations of a 32-state von Neumann cone are 13 cells, 65 bits. Only hash-consing shrinks the alphabet enough to tabulate multi-cell blocks.

---

## 8. Prior art and what is new

* **HashLife** (Gosper 1984) and Golly's `hlifealgo` / `ghashbase`. The generic version uses 2×2 leaves, 48-byte nodes, scalar rule evaluation and a mark-sweep collector.
* **lifelib** (Goucher): hashed quadtree with 16×16 bit-plane leaves and a bit-sliced SIMD leaf kernel, CPU only, single-threaded. The closest relative of the memo floor.
* **Lockstep HashLife** (Rokicki, golly-test 2014): the level-by-level order of §2.5 as pseudocode; no implementation was found. Rokicki in 2018: "no parallel versions of hashlife have been seen that outperform a single-threaded version."
* **HashLife on GPU** (LaForest, 2010): a $2^{16}$-entry table only; a static hash table with eviction was proposed and not built.
* **`gol_engines`** (das67333, 2026): the only measured parallel HashLife found; CPU work stealing, Life only, 31× to below 1× depending on the pattern.
* **GPU brute force**: bit-sliced Life at $10^{12}\text{–}10^{13}$ cell-updates/s; non-bit-sliced multi-state kernels near $10^{11}$, which is where the kernel of §1.2 sits. No throughput figure for a von Neumann-family rule on a GPU was found.
* **GPU hash tables** (WarpCore, BGHT, WarpSpeed, Hive): $2\text{–}4 \cdot 10^{9}$ lookups/s out of cache on larger cards, consistent with the $3.7 \cdot 10^{9}$ measured here.

Not found in the literature searched: a memo table of CA block transitions held in GPU memory; a bounded-depth HashLife that falls back to a brute-force kernel on a miss; a renormalised tile automaton swept positionally; a measured level-synchronous HashLife. The search could not reach forum threads behind logins.

---

## 9. Implementation plan

Each stage ends with a check that already exists in `research/`.

| Stage | Work | Result | Check |
|---|---|---|---|
| 0 | Canonical rules. Done in this revision: the WGSL rule files in `archive/src/kernel/rules` are corrected in place (§10). The new app takes them as they are, or builds its LUT from the Golly table as `rulespec.rs` does | GPU app runs the patterns as Golly does | `lut --rule … --against table:…` reports 0 mismatches |
| 1 | Replace the compute kernel with the bit-plane kernel of §13 (`research/bitslice/step_planes.wgsl`): five bits per cell, no table, zero ring, 8×8 workgroups, one pass per frame. The byte-LUT kernel (`gpu-brute`) is the fallback for rules without a circuit. Add RLE and macrocell loading | $3.0 \cdot 10^{11}$ on large grids for any content; default counter 24 k → 273 k gens/s, and → 780 k with the shared-tile variant for small grids (§13.4) | `gpu-bitslice --verify`, `gpu-tile --verify` |
| 2 | Tile sweep with a CPU learner: ID grids and memo table in VRAM, `gpu-sweep` kernel, misses resolved by the reference engine and uploaded. Run bursts of sweeps; on a miss, restore the snapshot, let the CPU learn the affected region, replay | $10^{12}\text{–}10^{14}$ once content recurs; 100k × 100k dense universes in VRAM | tile-exact comparison in `gpu-sweep` |
| 3 | Learner on the GPU: miss compaction, the claim protocol (`gpu-insert`), atlas evaluation, tile interning. Per §12: 8×8 base tiles, the recursion across levels with missed 16×16 windows brute-forced, 16-generation steps while learning | $\approx 2 \cdot 10^{7}$ misses/s; $\approx 10^{12}$ on non-repeating logic | `adapt` operation counts must match the GPU's counters |
| 4 | Sparse: active-block list (`gpu-sparse` kernel) with activation on border contact; sweep tile raised while it hits; plain cells under a region memo as second engine with the §12.5 controller; choice between host engine, plain kernel and recursion by active area (§12.6); cycle detection | 100k × 100k sparse at $10^{5}$ gens/s, $10^{6}\text{–}10^{7}$ once large tiles hit; CPU parity elsewhere | fingerprint equality with `bgolly` |
| 5 | Optional: tables beyond one binding, sharded as `gpu-probe` already does; circuits for further rules (§13.2) | memo capacity up to the whole card; the bit-plane kernel for rules other than Hutton32b | `gpu-probe`, `gpu-bitslice --verify` |

Engineering notes that came out of the measurements:

* Keep everything inside one compute pass per frame; a pass is 15–30× a dispatch, and a read-back 30–700×.
* **Generations per dispatch: keep the width.** A thread needs about half a microsecond per rule-circuit evaluation when evaluations follow one another, so one thread stepping a 32×32 window through 8 generations takes 630 µs however few windows there are, against 29 µs for eight chained dispatches. A workgroup that shares a tile and gives each thread one evaluation per generation pays about 1.1 µs per extra generation instead of 3.65 µs per dispatch: use it ($k = 8$ to 16) whenever the grid is small enough to be latency-bound, and the plain kernel otherwise (§13.4).
* **Decision for stage 3: no indirect dispatch.** The miss path's dispatch sizes depend on counts computed on the GPU. Sized with indirect calls, a wave costs 260–490 µs in wgpu (135–165 µs with indirect validation switched off); sized as fixed capacities with early exit it costs 51–55 µs, which is what this report's estimates assume (§2.6). Stage 3 therefore stays in wgpu: every miss-path stage is dispatched for a capacity taken from stale, asynchronously read miss counters, overflow is detected by a flag and repaired by replaying from a snapshot. Raw Vulkan would remove the pass boundary and make indirect dispatch viable, but it is not needed.
* One binding holds at most 2 GiB; shard tables by the top hash bits.
* On an 8 GB RTX 5060 the brute and out-of-cache probe rates should be about the same (same 448 GB/s bus; both are memory-bound) and memo capacity halves. Some listings give it a 24 MB L2 instead of 32 MB, which would lower the table size that still probes at the in-cache rate. None of this was measured on that card.
* The method needs nothing specific to Hutton32 above the kernel. Any radius-1 rule with a kernel (LUT, rule tree or circuit) plugs in below the floor.

---

## 10. The WGSL rule ports and the canonical rule tables

Two rules are in play, and each has one authoritative table:

| Rule | Canonical table | Verified identical to it: 0 of $2^{25}$ neighbourhoods differ |
|---|---|---|
| **Hutton32** (Tim Hutton, 2008) | `Hutton32.table` in Golly's rule table repository [25] | Golly's built-in `Hutton32`: the harness's transcription of `jvnalgo.cpp` with C integer semantics, which also reproduces `bgolly` on `squares` and on the Hutton replicator |
| **Hutton32b** (your modification: the 3×3 XNOR gate and two added sets of rules) | `doc/Hutton32b.rule` in the Hutton32-Computer repository [26] | the copy in Golly's `Rules` folder that `bgolly` loaded for the Hutton32b runs in this report; the two files differ by one blank line and their line endings. `research/rules` holds the published file, and `bgolly` now loads that |

The two rules themselves differ in 74 854 neighbourhoods, all with an excited STS, a confluent state or a crossing in the centre.

The WGSL files in `archive/src/kernel/rules` are meant to be ports of exactly these tables. As first archived (commit 256ef91, "v1" below) they were not. They have since been corrected in place and now are. `rulediff` compares $2^{25}$-entry LUTs entry by entry, and `lut` executes a WGSL rule file on the GPU for every neighbourhood and compares the result with a table:

| WGSL port | Canonical rule | Neighbourhoods that differ, v1 | Now |
|---|---|---|---|
| `hutton32.rule.wgsl` | Hutton32 | 2 430 000 (7.2 %) | 0 |
| `hutton32b.rule.wgsl` | Hutton32b | 3 082 794 (9.2 %) | 0 |

**Shared cause: three source lines, 2 430 000 neighbourhoods.** A sensitised cell in state 5, 7 or 8 with no excited arrow pointing at it. `input()` then returns 0 and v1 evaluates `(0 − 9 + k) % 4`: lines 301, 315 and 322 of the v1 `hutton32.rule.wgsl` (317, 331 and 338 of `hutton32b.rule.wgsl`), the last through `dir()`. Golly's C++ does this in `int`, where the remainder is negative. WGSL does it in `u32`, where the subtraction wraps around to a different residue:

| Centre | Canonical (C++ and table) | v1 port |
|---|---|---|
| 5 | 7 | 11 |
| 7 | 15 | 19 |
| 8 | **8** (stays) | **12** (becomes a south-pointing OTS) |

Each row is $30^4 = 810\ 000$ neighbourhoods. In the canonical rule a sensitised cell abandoned by its writer runs $1 \to 2 \to 4 \to 8$ and rests at 8; in the GPU app it turned into an arrow. The commented-out `{ return 8u; }` one line below was the canonical answer for state 8.

**Hutton32b only: 652 794 more.**

* 576 with centre 25. The table's XNOR gate takes ordinary transmission states as its two inputs ($e \in \lbrace 11, 15 \rbrace$ with $w \in \lbrace 9, 13 \rbrace$, or $n \in \lbrace 12, 16 \rbrace$ with $s \in \lbrace 10, 14 \rbrace$). The v1 `xnor_crossing()` tested direction only and so accepted special ones too. Example: $(c, n, e, s, w) = (25, 9, 15, 9, 17)$ is an ordinary crossing in the table (→ 29) and was an XNOR with a false output in v1 (→ 25).
* 652 218 with centre 29–31. v1 returned 25 as soon as `!is_crossing(…)`, and tested that before the incoming excited STS (lines 399–402). The table's `<29,12,9,0,13,25>` rules revert to 25 only when a neighbour is the ground state, and only after the STS test. In 357 021 of these the table destroys the cell (→ 0) and v1 kept it as 25; in the other 295 197 the table carries on as a crossing (→ 29, 30 or 31).

**The correction.** `research/tools/fix_rules.py` edited the two files in place: the three signed-remainder lines in each, and in `hutton32b.rule.wgsl` also the two lines of `xnor_crossing()`, the head of the 29–31 branch, and two spellings that current WGSL requires and naga 0.10 accepts as well (parentheses around `a || b && c`; a local that shadowed `output()`). Every edit asserts that it hits exactly the expected lines, and a second run reports each one as present. Executed on the GPU, each file now equals its canonical table on all 33 554 432 neighbourhoods, and so does the harness's CPU port of it. Both files also parse and validate under naga 0.10.1, the shader compiler of the archived app; the app itself was not rebuilt. The v1 behaviour stays available in the harness as `hutton32-v1` and `hutton32b-v1`, the v1 files in git.

**Which patterns notice.** `advanced counter v2`, `cpu` and both `lotr1` ROMs behave identically under v1 and under the canonical rule. Under v1 `BIN2BCD` differs from Golly by generation 2, and `Segmented ROM`, both displays, `16x16 movie` and the demultiplexer end in a different state after $2^{24}$ generations. Under the canonical rules every one of them matches Golly.

**What this report ran.** Every comparison with Golly and every pattern loaded from a Golly file used the canonical rule. The kernel throughput figures on the PNG presets (§1.2) were taken with the v1 rules, as the archived app then ran them; the harness now uses the canonical rules unless a `-v1` rule is asked for. The circuit of §13 is canonical Hutton32b: it was built and checked against the table, not against the WGSL.

---

## 11. Limitations

* **Hit path versus full pipeline.** The memoized GPU results use a table learned on the CPU and patterns that are periodic, replicated with phase offsets. They establish the steady-state rate and the correctness of the sweep. They do not establish learning speed on the GPU.
* **One machine, one rule.** All constants are from a single RTX 5060 Ti under wgpu on Windows. Pass, submit and read-back costs varied by 2–20× between runs of the same benchmark (§2.6 gives the ranges); the chained-dispatch and fixed-capacity figures were stable to 10 %.
* **Fixed-capacity dispatch is measured empty.** The 51–55 µs wave is sixteen dispatches that find almost nothing to do. The overflow-and-replay logic around it is designed, not built.
* **Miss rates are pattern statistics.** They were measured on fourteen of your patterns over $10^{4}\text{–}10^{6}$ generations. A circuit whose state space is explored more slowly would keep missing longer.
* **The reference engine is unoptimized** and holds every node forever; on the replicator it created 148 M nodes, an estimated 7 GB. The plan's reclaiming scheme is designed but not prototyped.
* **Golly baselines** are `bgolly` without rendering, 8 GB limit, best of the step sizes tried. Your GUI figures will be lower.
* **Literature claims** in §7–8 come from a survey run for this report; a few sources were read only as abstracts, and those are marked in the references.
* **§12 is simulation.** The adaptive-depth and engine-choice results count the reference engine's operations and price them with the measured GPU constants. Every policy was checked to produce the same universe, but the plain-cells mode is costed, not executed as cells, and the controller was tuned on the same composite it is reported on.
* **§13.** The circuit is for Hutton32b only and is not proven minimal. The bit-plane kernel is a benchmark on one card, measured up to $1.07 \cdot 10^{9}$ cells; buffers above about 300 MiB start slowly and the rates given for them are steady-state. The CPU figures come from the agent's benchmark, re-run once for this report; runs differ by up to 30 %.
* **Other load on the machine.** GPU throughput is sensitive to it. A first run of the kernel comparison of §13.3, taken while a build was running, showed the byte-LUT kernel up to 48 % low, and two entries in the 2 MiB column of the content table in §1.2 had been 18–46 % low for the same reason. Both tables now show repeats on an idle machine.

---

## 12. Follow-up: adaptive tile size and per-tile engine choice

Two refinements were proposed after the first version of this report: let tiles subdivide dynamically like a quadtree, and let a runtime heuristic choose between `memo` and `brute` per tile. This section restates both in the report's terms, measures them on your patterns, and revises the recommended configuration. The short version:

* Tile size should follow **novelty, not density**. Upwards (larger tiles where content recurs) the gain is large and nearly free: the same 100k × 100k ROM universe goes from 858 to 6.7 million generations per second as the tile grows from 16 to 256 cells. Downwards the useful range ends at 16×16 windows.
* The recursion **already is** a per-window memo/brute switch: a window that hits costs a probe, one that misses at the floor is brute-forced. An explicit second engine (plain cells under a coarse memo) is worth having for content that is new at every scale below the region: there it is 2.6–4× faster and uses 3–12× less memory. Everywhere else it is 3–74× slower.
* A controller that picks the engine per 256×256 region matches the best uniform setting on logic, is 1.9× faster on the ROM, and uses 43 % less memory on a mixed universe. Two earlier controller designs lost 16–40 % to their own experiments; the reasons are instructive and listed below.
* On a GPU the subdivision depth also sets the number of dispatches per generation, and that cost is shared by the whole universe. Below roughly $10^{6}$ active cells the plain kernel wins for that reason alone.

### 12.1 Three knobs and an engine

| Knob | Meaning | Golly `ghashbase` | §2–§4 of this report |
|---|---|---|---|
| Ceiling | largest window whose result is memoised | the root | 32×32 (flat sweep of 16×16 tiles) |
| Floor | smallest window that is memoised; a missed floor window is brute-forced | 4×4 | 32×32 |
| Step | generations per top-level step | chosen by the user | 8 |
| Engine below the floor | | scalar rule call | brute kernel on an atlas |

"Tiles of different sizes" is the level at which a window's result is obtained. Running the recursion of §2.5 from the root makes that level adaptive in both directions without any positional quadtree: a 4096×4096 window that hits is never opened, and a missed window is opened once more only where it missed. A literal quadtree of positional tiles runs into the sub-cycling problem of adaptive meshes (a small tile needs its large neighbour's state at times the neighbour never computes); the hash-consed recursion solves exactly that with its nine-then-four sub-windows.

`research/src/cmd_adapt.rs` (`adapt`) runs this recursion with a floor that can differ per 256×256 region and prices the operations with the constants of §2.1: 0.27 ns per find, 1 ns per insert, 5.4 ps per cell evaluation, every brute-forced window stepped whole.

### 12.2 Larger tiles

**Measured on the GPU** (`gpu-sweep`, 32×32 phase-shifted `lotr1.ANSI-1M` ROMs, hit path, every tile checked):

| Tile | Universe | gens/s | Memo entries | Table | ID buffer | Generations to learn |
|---|---|---|---|---|---|---|
| 16×16 | $1.02 \cdot 10^{10}$ cells | 858 | 2 462 688 | 256 MiB | 151 MiB | 528 |
| 32×32 | $1.06 \cdot 10^{10}$ | 20 349 | 628 903 | 64 MiB | 39 MiB | 1 056 |
| 64×64 | $1.16 \cdot 10^{10}$ | 191 061 | 162 691 | 16 MiB | 10 MiB | 2 112 |
| 128×128 | $1.36 \cdot 10^{10}$ | 1 149 321 | 43 759 | 4 MiB | 3 MiB | 4 224 |
| 256×256 | $1.83 \cdot 10^{10}$ | **6 714 020** | 12 475 | 1 MiB | 1 MiB | 8 448 |

Each level is worth 6–24×: 8× from the tile-step and the rest from the table sliding into the L2 cache. The price is the last column. The ROM's period is $528 = 2^4 \cdot 33$; a tile of side $2^L$ returns to the same alignment every $2^L$ generations and so sees the same content again only after $\operatorname{lcm}(528, 2^L)$ generations. Every level above 16×16 doubles the time to learn. Large tiles are something a region grows into.

**How much it helps when content is not periodic** (hash operations per generation, flat sweep at the given tile, recursion below it; `flat`):

| Pattern | 8×8 | 16×16 | 32×32 | 64×64 | 128×128 | 256×256 |
|---|---|---|---|---|---|---|
| `advanced counter v2` | 227 | 63 | 40 | 36 | 36 | 36 |
| `serial display`, idle | 66 781 | 8 634 | 1 277 | 336 | 212 | 196 |
| `32x32 movie`, playing | 67 149 | 9 171 | 1 907 | 1 002 | 887 | 872 |
| `chaotic` | 272 | 232 | 233 | 234 | 234 | 234 |
| ROM, first 528 gens | 88 837 | 92 164 | 92 590 | 95 132 | 95 139 | 95 140 |

The gain is whatever part of the universe recurs at the larger scale: empty and idle regions collapse (the display and the movie are mostly padding at small tiles), active logic saturates near 32–64 cells, novel content gains nothing. It never costs much. A level that never hits adds one failed probe and one insert per eight tile-steps below it, at most $1/7$ over all higher levels together, and by the model a level pays for itself once its hit rate exceeds $(p + \iota)/(8p + \iota)$, about 40 % with $p$ and $\iota$ the prices of a probe and an insert.

### 12.3 Smaller tiles

Model time for fixed floors ("F16" = brute-force missed 16×16 windows, memoise everything above), and for plain cells per region under a memo of region-sized windows:

| Content, horizon | Plain cells, ideal | F8 | F16 | F32 | F64 | F128 | F256 | Plain cells + coarse memo |
|---|---|---|---|---|---|---|---|---|
| counter, 16 384 gens (4×4 base tiles) | 5.8 ms | **0.77** | 0.89 | 1.91 | 4.68 | 11.4 | 33.9 | 10.5 |
| counter (8×8 base tiles) | 5.8 | — | **0.62** | 1.33 | | | | 9.4 |
| 8 perturbed `chaotic`, 8 192 gens (4×4) | 23.1 | **5.98** | 7.29 | 20.2 | 47.6 | 86.2 | 148 | 41.8 |
| same (8×8) | 23.1 | — | **5.21** | 14.0 | | | | |
| ROM, first 512 gens, all new (4×4) | 7.6 | 86.9 | 39.4 | 46.5 | | | | **10.0** |
| same (8×8) | 7.6 | — | 24.5 | 31.7 | | | | **8.6** |

"Ideal" is every non-empty region stepped as cells with no halo; the last column includes the halo and the conversion to IDs.

* **The floor belongs at 16×16 windows with 8×8 base tiles.** Going finer costs more in hashing than a window is worth: an 8×8 window is 0.7 ns of brute force, its bookkeeping 3.5 ns. With 4×4 base tiles F8 is 15 % ahead on logic and 2.2× behind on new content; 8×8 base tiles beat both.
* **Brute-forcing large isolated windows is the worst choice everywhere** (F64 and up). Each one carries its own halo (4× the cells) and the recursion asks for 13 of them where 8 would tile the result.
* §2.1 put the floor at 16×16 *tiles* (32×32 windows). That holds for a flat sweep, where every tile is probed every sweep. Inside the recursion a 16×16 window is probed only when the 32×32 window above it missed, so one more level pays.

### 12.4 Novelty, not density

Conditional miss rate per window size: of the windows requested because their parent missed, the fraction that missed too (`root`, 8×8 base tiles):

| Window | ROM, first 512 gens | counter | `chaotic` | `32x32 movie` | `squares` |
|---|---|---|---|---|---|
| 16×16 | 51 % | 21 % | 31 % | 33 % | 29 % |
| 32×32 | 99.7 % | 29 % | 57 % | 38 % | 35 % |
| 64×64 | 99.8 % | 36 % | 90 % | 45 % | 36 % |
| 128×128 | 99.9 % | 39 % | 93 % | 65 % | 51 % |
| 256×256 | 99.5 % | 45 % | 89 % | 74 % | 55 % |
| Live cells in bounding box | 86 % | 64 % | 80 % | 77 % | 35 % |

Four of the five are equally dense and want opposite treatment. Logic misses 20–55 % at every scale, so every level earns its keep. The ROM, while it is new, misses everything above 16×16: small tiles cannot help it until the whole period has been seen, at which point the largest tiles hit as well. For dense, locally unique content the right tile is the **largest** one, computed by brute force. "Smaller tiles where dense" is the right rule only for content that is dense *and* locally repetitive, and that is what a miss-driven recursion does on its own.

### 12.5 Choosing the engine

Model time and memory over 4096 generations, one pattern at a time, 8×8 base tiles (`tools/engine_choice.sh`). F = memoise everything, brute-force missed 16×16 or 32×32 windows. C = plain cells per 256×256 region with memo only above the region, at 64- or 16-generation steps.

| Content | F16 | F32 | C64 | C16 | Best vs worst |
|---|---|---|---|---|---|
| ROM `lotr1.PPMd` | 25.5 ms, 201 MiB | 33.5, 94 | 35.6, 63 | **9.8, 63** | C16 2.6× ahead of F16 |
| `chaotic` (perturbed) | **0.60**, 3.0 | 1.29, 1.6 | 2.36, 0.3 | 1.97, 0.7 | F16 3.3× |
| counter | **0.25**, 1.3 | 0.44, 0.6 | 4.24, 0.1 | 3.84, 0.3 | F16 15× |
| `32x32 movie` | **4.56**, 24.6 | 8.34, 12.8 | 30.1, 4.9 | 30.5, 6.1 | F16 6.6× |
| `squares` | **0.56**, 2.8 | 1.13, 1.5 | 10.7, 0.3 | 10.0, 0.7 | F16 18× |
| `serial display` | **2.37**, 10.5 | 6.14, 6.9 | 35.0, 1.3 | 36.7, 2.7 | F16 15× |
| `Segmented ROM` | **0.75**, 3.4 | 1.84, 2.6 | 21.0, 0.9 | 21.4, 1.3 | F16 28× |
| `16x16 movie` | **0.52**, 2.5 | 1.12, 1.5 | 38.4, 0.3 | 39.9, 0.9 | F16 74× |
| All eight | 35.1 | 53.8 | 177 | 154 | best per content: **19.4** |

Three things follow.

* **F16 is the safe default.** Its worst case is 2.6× behind; plain cells are up to 74× behind, because one moving signal forces a whole 256×256 region to be stepped.
* **The step matters as much as the engine.** The ROM learned through 64-generation steps needs 2112 generations (35.6 ms); through 16-generation steps, 528 (9.8 ms). A step that does not divide the period multiplies the learning time, as in §12.2. While a region is learning, its step should be short; it can be raised once the region hits.
* **Choosing per content is worth 1.8× on this mix**, all of it from the ROM.

**The controller that worked** (per 256×256 region, 16-generation steps):

1. Start at F16.
2. Coarser alternatives need no experiment. The statistics a region collects at its current depth price every shallower depth and plain cells exactly, because requests and misses above a level do not depend on what happens below it. Switch when the smoothed estimate is 15 % cheaper.
3. Finer alternatives are judged by subdividing one floor miss in sixteen and pricing the result. No whole-step trials.
4. From plain cells, return to F16 for a two-step trial at most every 16 steps, doubling up to 256 when the trial fails. The memo above the region catches whole-region recurrence in the meantime, so the trial only serves content that became regular at the fine scale.

| Universe (8×8 base tiles, 16-gen steps) | Uniform F16 | Controller |
|---|---|---|
| Mixed: ROM, 6 `chaotic`, 6 counters, movie, `squares`, display; 4.9 M live cells, 4096 gens | 35.0 ms, 257 MiB | **32.6 ms, 147 MiB** |
| ROM alone, 2112 gens | 24.7 ms, 200 MiB | **13.2 ms, 97 MiB** |
| counter, 16 384 gens | 0.67 ms | 0.67 ms (never leaves F16) |
| 8 perturbed `chaotic`, 8192 gens | 5.42 ms | 5.42 ms (never leaves F16) |

**Designs that did not work** (64-generation steps, 4×4 base tiles):

* *Walk down one level at a time from plain cells.* On the ROM's learning phase this cost 97.8 ms against 38.6 ms for plain cells and 40.4 ms for F16 throughout. The first step down from plain cells is a 256×256 brute window, the most expensive mode there is, so the experiment always failed and its cost was pure loss. Experiments must jump to a depth where being wrong is cheap.
* *Whole-step trials of the next finer level with a short back-off.* On the mixed universe: 64.1 ms, against 55.1 ms for uniform F16 and 46.8 ms for the best fixed mode per region. The novel phase lasted 33 steps; a few three-step trials at twice the cost consumed more than the gain available.
* *Sampled evidence, but still 64-generation steps*: 78.5 ms on the same universe. Plain cells learn the ROM four times slower at that step, and the controller kept leaving them. Mode boundaries also cost real work: a plain-cells region next to a memoised one pays a halo that a contiguous block would not, so flipping regions independently is worse than either uniform choice.

The practical trigger for plain cells is visible in §12.4: windows of 32×32 and larger missing nearly always, sustained, with the table growing. Memory is the stronger argument for it. During the ROM's first period F16 allocates 197 MiB and plain cells 16 MiB.

### 12.6 Latency couples every choice on a GPU

A missed floor window needs about seven dispatches plus one per brute-forced generation, and the levels above add roughly five per floor wave. With fixed-capacity dispatches at 3.4 µs each (§2.6; indirect dispatch would make every row below the first two 2.5–9× worse):

| Engine | Generations per floor wave | Dispatches per generation | Bound |
|---|---|---|---|
| Flat sweep, hits only, 16×16 tiles | 8 | 0.125 | $2.4 \cdot 10^{6}$ gens/s |
| Plain cells | 1 | ≈1.1 | $2.7 \cdot 10^{5}$ |
| F32 | 8 | ≈2.5 | $1.2 \cdot 10^{5}$ |
| F16 | 4 | ≈4 | $7.3 \cdot 10^{4}$ |
| F8 | 2 | ≈7 | $4.2 \cdot 10^{4}$ |

Running a window's generations inside one dispatch buys these bounds down only when a workgroup shares the work. One thread per window raises the time instead: 630 µs per wave against 29 µs for eight chained dispatches. A shared tile stepped 16 generations per dispatch lifts the plain-cells row from $2.7 \cdot 10^{5}$ to $5.6\text{–}7.8 \cdot 10^{5}$ on small grids (§13.4); the other rows and the thresholds below were not recomputed for it.

Work is local and latency is global: one region at a shallow floor makes the whole universe pay that floor's dispatches. The universe-wide choice is therefore the depth $f$ that minimises $L(f) + \sum_r w_r(f)$, with $L(f)$ the dispatch latency per generation at depth $f$ and $w_r(f)$ the work of region $r$; regions may only go coarser than that. With the counter's measured work per region (38 ns per generation at F16, 81 ns at F32, 575 ns as isolated plain cells):

| Active logic | Best engine on the GPU |
|---|---|
| below ≈ 0.7 M cells | plain kernel (or the CPU engine, §4.4) |
| 0.7–8 M cells | F32 |
| above ≈ 8 M cells | F16 |

This is consistent with §1.2, where the plain kernel already runs the 37 k-cell counter at $2.7 \cdot 10^{5}$ and the 275 k-cell `squares` at $1.8 \cdot 10^{5}$ generations per second, above the bounds in the table. The recursion itself has not been run on the GPU; its rows are the dispatch counts of §3.3 priced at the measured 3.4 µs.

### 12.7 Revised configuration

* Base tile 8×8; memoise every level above it; brute-force missed 16×16 windows (F16). Replaces the single-level 16×16 flat sweep of §3 as the default for large universes; the flat sweep remains the top of the hierarchy and the fastest path when everything hits.
* Raise the sweep tile while its hit rate stays above 40 %; expect each level to take twice as long to learn.
* Keep steps short (16 generations) while anything is being learned.
* Plain cells under a region-level memo as the second engine, entered on sustained novelty at 32×32 and above, left lazily.
* Choose the universe-wide floor from the size of the active area (§12.6).

In the plan of §9 this moves the recursion across levels from stage 5 into stage 3, and adds the plain-cells mode and the controller to stage 4. Nothing in stages 0–2 changes.

---

## 13. Follow-up: evaluating the rule without a table

A third refinement was proposed: pack the centre and its four neighbours into one 25-bit vector and turn the rule table into a series of plain bit operations on that vector. No branches, and no table, or one small enough for the L2 cache. A separate agent synthesised and checked the logic and measured it on the CPU; its exhaustive check and its CPU benchmark were re-run for this report, and every GPU figure below is from the harness on the final circuit. The short version:

* Canonical Hutton32b is **367 two-input gates**, depth 25, equal to the table on all $2^{25}$ neighbourhoods.
* Evaluated one cell at a time, as proposed, the gates lose to the table lookup by 2.5–34× (CPU). Applied to **bit-planes** the same gates update 32 cells per operation, and that is the fastest kernel in this report: **$3.0 \cdot 10^{11}$ cell-updates/s on any content**, 1.6–1.8× the byte-LUT kernel on real patterns and 22× on random states, at 5 bits per cell and with no 32 MiB table.
* On large grids that kernel is memory-bound: a circuit that returns its input unchanged is no faster. Synthesis was stopped at 367 gates for that reason.
* Small tables exist down to 10–15 KB at four or five loads per cell. They beat the big table only on hostile content and lose to bit-planes everywhere.
* Several generations per dispatch pay on small grids and only there, and only with the work spread over a workgroup. A tile held in workgroup-shared memory and stepped 16 generations per dispatch runs the single counter at **780 k generations/s** and `squares` at 560 k, 2.9× and 3.3× the plain kernel; on large grids the same kernel is 18–45 % slower. One thread stepping a whole window alone is 21× slower than chained dispatches.
* On one CPU core the circuit does $0.9\text{–}1.7 \cdot 10^{9}$ cell-updates/s in plain `u64` and $2.7\text{–}4.5 \cdot 10^{9}$ with AVX2: 2–8× the table on real content.

### 13.1 Three readings of the idea

| Form | What it is | Cost per cell-update | Measured |
|---|---|---|---|
| A. Scalar | the 25-bit vector in one register; a straight-line program of bit operations yields the 5 output bits | 364 32-bit operations | CPU: $1.5\text{–}2.5 \cdot 10^{7}$/s. The table lookup does $3.3\text{–}6.0 \cdot 10^{8}$ on a real pattern and $5\text{–}7 \cdot 10^{7}$ on random states |
| B. Bit-sliced | the same gate network; every operand is a word that holds one bit of 32 (or 64, or 256) different cells | 367 gates and 30 shift-and-merge operations per word: 12.4 operations per cell at 32 bits | GPU: $3.0 \cdot 10^{11}$/s (§13.3). CPU: $0.9\text{–}4.5 \cdot 10^{9}$/s (§13.5) |
| C. Small tables | the table factored through intermediate class codes | 4–5 loads from 10–15 KB | CPU: $2.6\text{–}4.5 \cdot 10^{8}$/s. GPU, the 2 MiB single-table variant of §1.2: $4.2 \cdot 10^{10}\text{–}1.3 \cdot 10^{11}$/s |

Form A is the proposal read literally, and it cannot win. A gate acts on single bits, so with one cell per register 364 operations buy one cell-update, where a table buys it with one load. The form pays when the other 31 bits of every register carry other cells, and that is form B. Both conditions of the proposal hold there: no branch, no table.

Form C answers the "or a small LUT" clause. The rule tells apart 15 classes of states per neighbour and 29 for the centre; given the centre, a neighbour has at most 6 classes, and the whole rule is 6 779 (centre, class, class, class, class) cases. That allows (`bitslice/tables.py`):

| Scheme | Index bits | Bytes | Loads per cell | Status |
|---|---|---|---|---|
| one table of class codes (§1.2) | 21 | 2 MiB | 1, plus 5 class lookups | measured on the GPU |
| `T5[Tn[c,n] + Te[c,e] + Ts[c,s] + Tw[c,w]]` | 10 (×4), 13 | 14 971 | 5, four of them independent | built, equal to the table on all $2^{25}$, benchmarked on the CPU |
| chain `T4[T3[T2[T1[c,n],s],e],w]` | 10, 12, 12, 12 | 10 592 | 4, each depending on the one before | sizes computed, not built |

Both small schemes fit an L1 cache, let alone L2. On the CPU the five-load form roughly equals the big table on real content and is 6–7× faster on random states, and is still 3–10× slower than bit-planes. On the GPU the big table is already bandwidth-bound on real content, so a small table could only repair hostile content, which the bit-plane kernel repairs better.

### 13.2 The circuit

`research/bitslice/circuit.netlist`, emitted as `step_planes.wgsl` (32-bit words) and `step_planes.rs` (64-bit): 165 AND, 137 OR, 14 XOR, 51 NOT; depth 25.

* **How it was made.** A network written by hand from the rule's structure: per neighbour the predicates the rule looks at (excited OTS or STS arrow at the centre, any arrow at the centre, a confluent state emitting towards it, ground), the state of the cell a transmission state points at through a 4:1 multiplexer, then one block per class of centre state. 473 gates. From there it is automatic: global resubstitution with observability don't-cares, ABC rewriting scripts run through Yosys, and an inverter placement solved as MaxSAT. 400 gates after 6 minutes, 367 after 24. Every intermediate network is kept only if it agrees with the table on all $2^{25}$ entries. It is not proven minimal. Optimisation was stopped when the GPU stopped noticing (§13.3).
* **Checks.** `verify.py` evaluates the netlist, the WGSL text and the Rust text on all $2^{25}$ neighbourhoods: 0 mismatches in every output bit, after first confirming that a deliberately corrupted gate is caught. On the GPU, one step of a $16384^{2}$ grid of uniformly random states samples $2.7 \cdot 10^{8}$ neighbourhoods, 99.97 % of all $2^{25}$: 0 mismatching cells. Patterns follow in §13.3.
* **Which rule.** Canonical Hutton32b (§10). The truth table is the LUT built from `Hutton32b.rule`; the WGSL port only served as a reading guide for the hand-written network, and where the two disagreed the table decided. The corrections of §10 therefore change nothing here: the corrected WGSL, the table and the circuit are one function on all $2^{25}$ neighbourhoods, and nothing needs to be regenerated.
* **Other rules.** Only the hand-written starting network is rule-specific; the optimiser works from any $2^{25}$-byte table. Hutton32 differs from Hutton32b in 74 854 entries, all in the blocks for an excited-STS, confluent or crossing centre; that edit has not been made. For an unrelated rule there is no generator of a starting network, and the byte-LUT kernel remains the fallback.

### 13.3 The bit-plane kernel on the GPU (`gpu-bitslice`)

Five `u32` per 32 cells, word k holding bit k of the state; rows are padded by one empty group on each side. An invocation loads its own group, the groups above and below, and the groups left and right for the two bits that cross a word boundary. Per plane it forms `e = (c >> 1) | (r << 31)` and `w = (c << 1) | (l >> 31)`, evaluates the circuit and stores five words. 8×8 workgroups, ping-pong buffers, all generations of a submit in one pass.

16 384 × 16 384 cells, the same content for every kernel. Each row was also run with a cell-exact comparison against the CPU: 0 mismatching cells throughout.

| Content | Byte LUT, 32 MiB | Byte LUT, 2 MiB | Bit-planes | Gain over the 32 MiB LUT |
|---|---|---|---|---|
| uniform random states, held constant | $1.3 \cdot 10^{10}$ | $4.2 \cdot 10^{10}$ | **$3.0 \cdot 10^{11}$** | 22× |
| tiled counter | $1.8 \cdot 10^{11}$ | $1.3 \cdot 10^{11}$ | **$2.9 \cdot 10^{11}$** | 1.6× |
| tiled `chaotic` | $1.8 \cdot 10^{11}$ | $1.3 \cdot 10^{11}$ | **$3.0 \cdot 10^{11}$** | 1.6× |
| tiled `lotr1.ANSI-1M` | $1.6 \cdot 10^{11}$ | $1.2 \cdot 10^{11}$ | **$3.0 \cdot 10^{11}$** | 1.8× |

Other sizes:

| Grid | Byte-LUT kernel | Bit-plane kernel |
|---|---|---|
| one counter, 224 × 205 | 274 544 gens/s | 270–273 k gens/s, cell-exact after 65 536 generations |
| `squares`, 672 × 431 | 177 445 gens/s | 169–175 k gens/s |
| one ROM, 3104 × 3067, a full period | 18 992 gens/s ($1.8 \cdot 10^{11}$) | **40 999 gens/s** ($3.9 \cdot 10^{11}$), cell-exact |
| $5.4 \cdot 10^{8}$ cells, 321 MiB per buffer | | $2.3 \cdot 10^{11}$ |
| $1.07 \cdot 10^{9}$ cells, 641 MiB per buffer | | $2.8 \cdot 10^{11}$ |

The two largest rows are steady-state rates over 256 and 512 generations. Buffers of that size start slowly: the first 32 generations after upload averaged $1.3\text{–}1.8 \cdot 10^{11}$.

**Where the limit is.**

* *Memory.* A cell-update reads 5 bits and writes 5: 1.25 bytes, against 2 for the byte kernel. $2.97 \cdot 10^{11}$ × 1.25 B is 371 GB/s; the byte kernel's $1.86 \cdot 10^{11}$ × 2 B is 372 GB/s. Both sit at 83 % of the card's 448 GB/s, and the gain on real content is exactly the saving in bytes.
* *Not the gates.* With a circuit that returns the centre unchanged, the same harness gives $3.0 \cdot 10^{11}$ at $16384^{2}$ and $2.8 \cdot 10^{11}$ at $32768^{2}$: the same as the real circuit. Cache-resident grids go to $4.3 \cdot 10^{11}$ ($4096^{2}$, where the identity circuit gives $3.8 \cdot 10^{11}$).
* *Headroom.* Evaluating the circuit $k$ times per invocation, each evaluation feeding the next, the pass rate at $16384^{2}$ is 3.0, 2.4, 1.4 and $0.76 \cdot 10^{11}$ for $k = 1, 2, 4, 8$. The gates alone therefore sustain $5.4\text{–}6.1 \cdot 10^{11}$ cell-evaluations/s, twice what memory lets through. Computing several generations per pass would cut the traffic, but neither way of doing it cashes the headroom in. Threads working alone need five evaluations per word for two generations instead of two, which would cap the rate near $2.3 \cdot 10^{11}$ (estimated). Threads sharing a tile were built and measured: $2.5 \cdot 10^{11}$ at best (§13.4).
* *Content.* None of it matters any more. The byte kernel's collapse on random states was a table that no longer fitted L2; there is no table.

### 13.4 Several generations per dispatch

Two kernels were built. They differ in who does the work, and they come out on opposite sides.

**One thread per window: no.** Earlier versions of §5 and §12.6 held out a kernel that steps a missed window through all its generations inside one dispatch as a way to halve the dispatch count of the miss path. `gpu-window` is that kernel: one invocation per 32×32 window, 8 generations in private arrays, the central 16×16 written out, verified against per-cell stepping.

| Windows per dispatch | Window kernel, time per dispatch | Atlas: 8 chained dispatches of the plain kernel |
|---|---|---|
| 1 | 619 µs | 29 µs |
| 64 | 623 µs | 31 µs |
| 1 024 | 627 µs | 60 µs |
| 4 096 | 639 µs | 115 µs |
| 16 384 | 2.8 ms | 0.5 ms |
| 262 144 | 41 ms (155 ns per window) | 7 ms (27 ns per window) |

The atlas column is the plain kernel's time per generation on a grid of that many windows, times eight: measured for $64^{2}$, $1024^{2}$ and $2048^{2}$ cells (3.65, 7.5 and 14 µs per generation), and 3.65 µs plus 27 ns per window for the rest. The window kernel loses at every batch size: by 21× where latency matters and by 6× where throughput does.

The reason is serial depth. One thread performs 240 circuit evaluations in sequence, and a GPU thread is slow in sequence. Measured directly on a 64×64 grid, with $k$ evaluations per invocation that depend on one another, a dispatch takes 3.65, 3.66, 5.7, 5.7, 7.7 and 18.0 µs for $k = 1, 2, 4, 8, 16, 32$: the first two hide in what a dispatch costs anyway, and beyond that an evaluation costs about half a microsecond even with all state in registers. In the window kernel, whose state lives in private arrays, it costs 2.6 µs.

**One workgroup per tile: yes, on small grids.** `gpu-tile` keeps the width. A workgroup owns a tile of bit-plane words. Each thread loads one word group of the tile or of its halo ($k$ rows above and below, one group left and right) into workgroup-shared memory, then all threads step $k$ generations there, one circuit evaluation per thread per generation with a barrier in between, and the tile is written once. A thread's depth is $k$ evaluations, not $k$ times the window. Cell-exact against the CPU for $k = 1, 2, 3, 4, 8$ and 16 on random states, the counter, `chaotic`, `squares` and a full ROM period.

| Grid | Plain kernel | Tile kernel, best setting | Gain |
|---|---|---|---|
| one counter, 224 × 205 | 273 k gens/s | **770–785 k gens/s** ($k = 16$, tiles of 224 × 16 cells; 20.5 µs per dispatch) | 2.9× |
| `squares`, 672 × 431 | 172 k gens/s | **556–563 k gens/s** ($k = 16$, tiles of 192 × 48) | 3.3× |
| one ROM, 3104 × 3067 | 41 k gens/s ($3.9 \cdot 10^{11}$) | 23 k gens/s ($2.2 \cdot 10^{11}$), $k = 4$ | 0.57× |
| $16384^{2}$, random states | $3.0 \cdot 10^{11}$ | $2.5 \cdot 10^{11}$ ($k = 3$; $1.7 \cdot 10^{11}$ at $k = 1$, $1.8 \cdot 10^{11}$ at $k = 8$) | 0.82× |

On a small grid a dispatch costs 3.65 µs whatever it does, and a further generation inside it costs about 1.1 µs, so sixteen generations take 20 µs instead of 58. That makes the GPU the fastest engine on the single counter: Golly runs it at 580 k generations/s and the CPU reference engine at 650 k (§6). On `squares` Golly does 13–30 k.

On a large grid the kernel loses. With $k = 1$ it does the plain kernel's job through shared memory and reaches 55 % of its rate, so the detour itself costs more than the bus traffic it saves at $k = 2$ to 4; beyond that the halo eats the tile. The plain kernel therefore stays for anything that is not latency-bound. The crossover was not located; it lies between $3 \cdot 10^{5}$ and $9.5 \cdot 10^{6}$ cells.

Rates depend on the tile shape by up to 1.6× at a given $k$ (`squares` at $k = 16$: 342 k, 373 k and 563 k for tiles 16, 32 and 48 rows high), and only a handful of shapes were tried. $k$ is limited to 32 by the one-word halo. An atlas of missed windows is a small grid too, so the miss path's eight evaluation dispatches should shrink to one the same way (about 12 µs against 29 µs); that was not run.

### 13.5 On the CPU

i5-6600K, one thread, cell-updates per second (`bitslice/bench`; all engines are compared cell by cell before timing). The ranges span the agent's runs and a re-run of each build for this report. The default build reproduced within 8 %; the AVX2 build came out 20–30 % lower on every engine, the table included.

| Engine | Uniform random states | Tiled counter |
|---|---|---|
| byte LUT, 32 MiB | $4.8\text{–}7.2 \cdot 10^{7}$ | $3.3\text{–}6.0 \cdot 10^{8}$ |
| form A, scalar circuit | $1.5\text{–}2.5 \cdot 10^{7}$ | $1.4\text{–}2.5 \cdot 10^{7}$ |
| form C, five loads from 15 KB | $3.2\text{–}4.5 \cdot 10^{8}$ | $2.6\text{–}4.4 \cdot 10^{8}$ |
| form B, `u64` | $0.9\text{–}1.7 \cdot 10^{9}$ ($2.3\text{–}2.8 \cdot 10^{9}$ on long rows with `target-cpu=native`) | $0.9\text{–}1.7 \cdot 10^{9}$ ($2.2\text{–}2.8 \cdot 10^{9}$) |
| form B, 256-bit AVX2 | $3.0\text{–}4.5 \cdot 10^{9}$ | $2.7\text{–}4.4 \cdot 10^{9}$ |

Within any one run, bit-planes are 2.0–3.5× the table on real content in portable `u64` code and 6–8× with AVX2, and 17–73× on random states. This raises a core's brute-force rate from $3\text{–}6 \cdot 10^{8}$ to $0.9\text{–}4.5 \cdot 10^{9}$ while an out-of-cache probe stays near $10^{7}$ per second, so the CPU's exchange rate $\rho$ moves from 30–60 to a few hundred. By §2.1 the CPU engine's floor then belongs at 16×16 tiles as well. That is an inference from the model: the reference engine of §4.4 uses a 4×4 floor and scalar evaluation, and was not rebuilt.

### 13.6 What changes in this report

* **$R_B$ and $\rho$.** The brute-force rate goes from $1.86 \cdot 10^{11}$ to $3.0 \cdot 10^{11}$ ($2.3\text{–}2.8 \cdot 10^{11}$ above $5 \cdot 10^{8}$ cells) and the exchange rate $\rho = R_B/R_P$ from 50 to 80. The best possible gain of a memo hit over brute force, $V_L/\rho$, becomes 3.2× at 8×8 tiles, 26× at 16×16, 205× at 32×32 and 1 640× at 64×64. The floor argument of §2.1 gets stronger: brute force became cheaper relative to a probe.
* **Worst case.** Dense content that never recurs runs at $3.0 \cdot 10^{11}$ whatever it is: 280× the archived kernel, where the byte kernel managed 13× on random states.
* **§12.5 re-priced** (`adapt --cell-rate 2.9e11`). No ranking changes. Over the eight contents uniform F16 costs 29.1 ms instead of 35.1 and the best engine per content 15.2 instead of 19.4. Plain cells lead F16 on the ROM by 3.0× instead of 2.6×; everywhere else F16 leads by 2.7–56× instead of 3.3–74×.
* **§12.6 thresholds.** The counter's work per generation becomes 33 ns at F16, 59 ns at F32 and 377 ns as plain cells. The plain kernel is then the best engine up to about 1.0 M active cells (was 0.7 M) and F32 up to about 13 M (was 8 M).
* **Small active patterns.** The plain bit-plane kernel changes nothing: 273 k against 275 k generations per second on the single counter, both waiting for the dispatch. The shared-tile variant does: 780 k on the counter and 560 k on `squares` (§13.4), which also raises the active area up to which plain cells are the best engine in §12.6.
* **Memory.** The 32 MiB table leaves VRAM and, more to the point, the L2 cache, which is then free for memo tables; a probe into a table that fits L2 is 9× cheaper than one that does not (§2.1). The two were not measured together. Cells cost 5 bits instead of 8: a 100k × 100k universe is 6.3 GB per buffer and fits the card as plain cells, where brute force would step it about 28 times a second. That size was not run; one binding holds 2 GiB, so the kernel would first have to span several.
* **Tile store.** A 16×16 tile is 160 bytes either way. Stored as five planes of sixteen 16-bit rows it is already in this kernel's format, so the miss path's expand and intern steps become word copies and the atlas is evaluated in place.
* **Plan.** Stage 1 of §9 uses this kernel, with the shared-tile variant for small grids. The one-thread-per-window kernel is dropped from §5, §9 and §12.6.

Not done: the kernel is a benchmark, not yet the app's kernel; there is no circuit for Hutton32; no small-table scheme other than the 2 MiB one was run on the GPU.

---

## Appendix A. Reproducing the numbers

Everything the commands need is in the repository: the harness in `research/`, every pattern in `research/patterns` (sources and rules in its README), and the two canonical tables in `research/rules`. The commands are bash and run from `research/`, in git-bash on Windows or in WSL or Linux (where `python` may be `python3`). They need Rust, Python 3 and a GPU with Vulkan or DX12. Golly is needed only for the Golly columns: `tools/time_golly.py` and `tools/head2head.py` take `bgolly` from `$BGOLLY`, from Golly 5.1's Windows install path, or from `PATH`, and use `research/rules` as Golly's rule folder. GPU figures should be taken on an otherwise idle machine (§11).

```bash
cd research
cargo build --release
R=./target/release/ca-research        # git-bash finds the .exe
P=patterns
BG="${BGOLLY:-/c/Program Files/golly-5.1-win-64bit/bgolly.exe}"   # only for the direct bgolly calls in §4.3
mkdir -p out/golly out/v1

# §1.1, §10: the rules
$R selftest                                                      # reference engine vs per-cell stepping
$R lut --rule hutton32b                                          # CPU port vs archive/src/kernel/rules/hutton32b.rule.wgsl run on the GPU
$R lut --rule hutton32  --against table:rules/Hutton32.table     # 0 mismatches: the WGSL file is Tim Hutton's table
$R lut --rule hutton32b --against table:rules/Hutton32b.rule     # 0 mismatches; leaves out/lut_table_Hutton32b.bin for §13
$R rulediff --a table:rules/Hutton32.table --b table:rules/Hutton32b.rule   # 74 854
$R rulediff --a hutton32-v1  --b hutton32                        # 2 430 000: the files as archived in 256ef91
$R rulediff --a hutton32b-v1 --b hutton32b                       # 3 082 794
git show 256ef91:archive/src/kernel/rules/hutton32b.rule.wgsl > out/v1/hutton32b.rule.wgsl
$R lut --rule hutton32b --wgsl out/v1/hutton32b.rule.wgsl --against hutton32b-v1   # 0: the v1 port is the v1 file
python tools/fix_rules.py                                        # reports every correction as present
# optional, needs the network: the committed tables are the upstream files
curl -sL --retry 3 https://gollygang.github.io/ruletablerepository/downloads/Hutton32.table | cmp - rules/Hutton32.table && echo same
curl -sL --retry 3 https://raw.githubusercontent.com/FredericaBernkastel/Hutton32-Computer/master/doc/Hutton32b.rule | cmp - rules/Hutton32b.rule && echo same

# §1.2: the archived kernel, then the byte-LUT kernel done properly
NEW=$P/presets/hutton32b_8bit_counter.png; OLD=$P/presets/hutton32_8bit_counter_old.png
$R gpu-baseline --pattern $NEW --gens 16384 --per-submit 272 --verify
$R gpu-baseline --pattern $OLD --rule hutton32 --gens 8192 --per-submit 32 --verify
$R gpu-baseline --pattern $P/misc/squares.rle --gens 8192 --per-submit 32 --verify
$R gpu-baseline --pattern $OLD --rule hutton32 --reps 8 --gens 512 --per-submit 32            # 5080 x 3608
$R gpu-brute    --pattern $NEW --gens 65536 --per-submit 1024 --verify
$R gpu-brute    --pattern $P/misc/squares.rle --gens 65536 --per-submit 1024 --verify
$R gpu-brute    --pattern $OLD --rule hutton32 --reps 8 --gens 2048 --per-submit 64 --verify  # 18.7 M cells
$R gpu-brute    --pattern $OLD --rule hutton32 --reps 24 --gens 512 --per-submit 32           # 169 M cells
for s in "8 8" "16 16" "32 8" "64 4"; do set -- $s; $R gpu-brute --pattern $OLD --rule hutton32 --reps 24 --gens 512 --per-submit 32 --wgx $1 --wgy $2; done
$R gpu-brute    --pattern $OLD --rule hutton32 --reps 24 --gens 512 --per-submit 32 --backend dx12
$R gpu-brute2   --side 8192 --gens 64 --pattern $P/misc/chaotic.rle --verify                 # content table: tiled chaotic
$R gpu-brute2   --side 8192 --gens 64 --verify                                               # random states, decaying
$R gpu-brute2   --side 8192 --gens 64 --same-src --verify                                    # random states, held constant

# §1.3: Golly HashLife (bgolly), load time subtracted; the last two lines are the GPU and single-core rates on the ROM
python tools/time_golly.py "$P/advanced counter v2.rle" 1048576 1 8 64 512 4096 32768 262144 --cells 37627
python tools/time_golly.py $P/misc/squares.rle 65536 8 64 1024 --algo JvN --cells 274950
python tools/time_golly.py $P/misc/chaotic.rle 16384 8 64 1024 --cells 19932
python tools/time_golly.py $P/misc/extra/lotr1.ANSI-1M.rle 65536 64 4096 --cells 9467605 --timeout 180
python tools/time_golly.py $P/replicators/Hutton-replicator.rle 1048576 1024 16384 262144 --algo JvN --timeout 300
for spec in "misc/Segmented ROM (256k).mc|RuleLoader|538740" "misc/serial display (32x32).mc|RuleLoader|1710912" \
            "32x32 movie/32x32 movie.mc|RuleLoader|1404064" "misc/parallel display (16x16).mc|JvN|612717" \
            "16x16 movie/16x16 movie.mc|JvN|5716840" "misc/efficient_demultiplexer.mc|JvN|578280"; do
  IFS='|' read -r f algo cells <<< "$spec"
  python tools/time_golly.py "$P/$f" 65536 64 1024 --algo $algo --cells $cells --timeout 120
done
$R gpu-baseline --pattern $P/misc/extra/lotr1.ANSI-1M.rle --gens 64 --per-submit 32
$R gpu-brute    --pattern $P/misc/extra/lotr1.ANSI-1M.rle --gens 528 --per-submit 264 --verify   # also prints the single-thread CPU rate

# §2.1, §2.6, §4.1: primitives
$R gpu-probe --log2-slots 10,14,16,18,19,20,21,22,24,25,26,27,28   # the 8 GiB table needs that much free VRAM
$R gpu-insert
$R gpu-latency

# §2.3: miss rates of the flat tile automaton (SUMMARY lines: second half of each run)
for a in 2 3 4; do for lt in 3 4 5 6 7; do
  if [ $lt -ge $a ]; then $R flat --pattern "$P/advanced counter v2.rle" --lt $lt --a $a --gens 262144 --quiet --summary | grep SUMMARY; fi
done; done
for f in "misc/Segmented ROM (256k).mc" "misc/serial display (32x32).mc" "misc/parallel display (16x16).mc" "16x16 movie/16x16 movie.mc" \
         "misc/efficient_demultiplexer.mc" "misc/chaotic.rle" "misc/squares.rle" "component-basis.rle" "misc/extra/lotr1.PPMd-244K.rle"; do
  for lt in 4 5; do $R flat --pattern "$P/$f" --lt $lt --a 2 --gens 16384 --quiet --summary | grep SUMMARY; done
done
$R flat --pattern "$P/32x32 movie/32x32 movie.mc" --lt 4 --a 2 --gens 131072 --summary
$R flat --pattern $P/misc/extra/lotr1.ANSI-1M.rle --lt 4 --a 2 --gens 4096 --summary
for f in "misc/weird designs/cpu.mc" "misc/BIN2BCD.rle"; do $R flat --pattern "$P/$f" --lt 4 --a 2 --gens 65536; done

# §2.5: batch statistics of the level-synchronous schedule
$R root --pattern $P/replicators/Hutton-replicator.rle --k 14 --gens 262144

# §4.2, §12.2: memo hit path on phase-shifted lotr1.ANSI-1M ROMs; each run first learns the table on the CPU (about 90 s per period)
ROM=$P/misc/extra/lotr1.ANSI-1M.rle
$R gpu-sweep --pattern $ROM --lt 4 --reps 1  --cycles 6600 --per-submit 512
$R gpu-sweep --pattern $ROM --lt 5 --reps 1  --cycles 6600 --per-submit 512
$R gpu-sweep --pattern $ROM --lt 4 --reps 8  --cycles 66
$R gpu-sweep --pattern $ROM --lt 4 --reps 32 --cycles 33 --per-submit 4
$R gpu-sweep --pattern $ROM --lt 5 --reps 32 --cycles 33 --per-submit 4
for lt in 6 7 8; do $R gpu-sweep --pattern $ROM --lt $lt --a 4 --reps 32 --cycles 264 --per-submit 66 --max-learn-gens 40000; done

# §4.3: sparse 100k x 100k universe of 2304 counters, then the same universe in Golly and in the CPU engine
$R gpu-sparse --pattern "$P/advanced counter v2.rle" --lt 4 --reps 48 --pitch 8 --cycles 1024 --export-mc out/golly/sparse100k.mc
python tools/time_golly.py out/golly/sparse100k.mc 16384 16 1024 16384 --timeout 600
python tools/time_golly.py out/golly/sparse100k.mc 1048576 1024 65536 --timeout 900
"$BG" -q -q -a RuleLoader -s rules/ -M 8000 -m 16384 -i 1024 -o out/golly/sparse100k_out.mc out/golly/sparse100k.mc
$R fingerprint --pattern out/golly/sparse100k_out.mc               # population 55478381, hash 144be5c0cde9cb81
$R root --pattern out/golly/sparse100k.mc --k 10 --gens 16384 --a 2   # the CPU engine ends in the same state

# §4.4: head-to-head at step 1024 (k = 10) or 16 384 (k = 14), one thread; the last column compares the final states
python tools/head2head.py 16777216 10 --a 2 --timeout 400 "$P/misc/Segmented ROM (256k).mc" "$P/misc/serial display (32x32).mc" \
  "$P/misc/parallel display (16x16).mc" "$P/16x16 movie/16x16 movie.mc" $P/misc/efficient_demultiplexer.mc "$P/misc/weird designs/cpu.mc" \
  $P/misc/BIN2BCD.rle "$P/advanced counter v2.rle" $P/misc/chaotic.rle "$P/32x32 movie/32x32 movie.mc"
python tools/head2head.py 1048576 10 --a 2 --timeout 600 $P/misc/squares.rle
python tools/head2head.py 262144 14 --a 2 --timeout 900 $P/replicators/Hutton-replicator.rle

# §10: the same with the engine on the v1 rules: NO for all but cpu and the counter
python tools/head2head.py 16777216 10 --a 2 --timeout 400 --v1 "$P/misc/Segmented ROM (256k).mc" "$P/misc/serial display (32x32).mc" \
  "$P/misc/parallel display (16x16).mc" "$P/16x16 movie/16x16 movie.mc" $P/misc/efficient_demultiplexer.mc "$P/misc/weird designs/cpu.mc" \
  $P/misc/BIN2BCD.rle "$P/advanced counter v2.rle"

# §12.2: model hash operations per generation by sweep tile (MODEL lines)
for spec in "advanced counter v2.rle|65536" "misc/chaotic.rle|16384" "misc/extra/lotr1.PPMd-244K.rle|528" \
            "misc/extra/lotr1.PPMd-244K.rle|8448" "32x32 movie/32x32 movie.mc|16384" "misc/serial display (32x32).mc|16384"; do
  IFS='|' read -r f g <<< "$spec"
  for lt in 3 4 5 6 7 8; do $R flat --pattern "$P/$f" --lt $lt --a 2 --gens $g --quiet | grep MODEL; done
done

# §12.3: fixed floors and plain cells (--a 2: 4x4 base tiles, --a 3: 8x8)
C=$P/misc/chaotic.rle
CH8="$C@64,64;$C@576,64~1;$C@1088,64~2;$C@1600,64~3;$C@64,576~4;$C@576,576~5;$C@1088,576~6;$C@1600,576~7"   # 8 perturbed copies
for a in 2 3; do
  $R adapt --pattern "$P/advanced counter v2.rle" --gens 16384 --a $a --policies fixed
  $R adapt --compose "$CH8" --gens 8192 --a $a --policies fixed
  $R adapt --pattern $P/misc/extra/lotr1.PPMd-244K.rle --gens 512 --a $a --policies fixed
done

# §12.4: conditional miss rate per window size (per level: requests in column 6, misses in column 7)
for spec in "misc/extra/lotr1.PPMd-244K.rle|512" "advanced counter v2.rle|16384" "misc/chaotic.rle|8192" \
            "32x32 movie/32x32 movie.mc|4096" "misc/squares.rle|4096"; do
  IFS='|' read -r f g <<< "$spec"
  $R root --pattern "$P/$f" --k 6 --gens $g --a 3
done

# §12.5: engine per content, then the controller
bash tools/engine_choice.sh $R 4096
$R adapt --compose "$(cat tools/composite_spec.txt)" --gens 4096 --a 3 --k 4 --init-floor 3 --policies "16x16,adaptive"
$R adapt --pattern $P/misc/extra/lotr1.PPMd-244K.rle --gens 2112 --a 3 --k 4 --init-floor 3 --policies "16x16,adaptive"
$R adapt --pattern "$P/advanced counter v2.rle" --gens 16384 --a 3 --k 4 --init-floor 3 --policies "16x16,adaptive"
$R adapt --compose "$CH8" --gens 8192 --a 3 --k 4 --init-floor 3 --policies "16x16,adaptive"

# §12.6: the counter's work per generation, which sets the thresholds
$R adapt --pattern "$P/advanced counter v2.rle" --gens 16384 --a 3

# §13: the circuit, the CPU benchmark and every GPU figure (on an idle machine)
python -m venv out/bitslice-venv
VPY=$PWD/out/bitslice-venv/bin/python; [ -e out/bitslice-venv/Scripts/python.exe ] && VPY=$PWD/out/bitslice-venv/Scripts/python
"$VPY" -m pip install numpy yowasp-yosys python-sat pillow
(cd bitslice && "$VPY" verify.py && "$VPY" scalar.py && "$VPY" tables.py)   # exhaustive checks; forms A and C
cargo run --release --manifest-path bitslice/bench/Cargo.toml                 # CPU: forms A, B, C against the byte LUT
RUSTFLAGS="-C target-cpu=native" cargo run --release --manifest-path bitslice/bench/Cargo.toml   # adds the AVX2 column
bash tools/bitslice_series.sh $R                                              # every GPU figure of §13: sections A-H, S and T
bash tools/engine_choice.sh $R 4096 --cell-rate 2.9e11                        # §12.5 at the bit-plane kernel's rate
$R adapt --pattern "$P/advanced counter v2.rle" --gens 16384 --a 3 --cell-rate 2.9e11
```

Single lines of the series, for a quick check:

```bash
$R gpu-bitslice --side 16384 --gens 64 --same-src --verify                    # random states, held constant
$R gpu-bitslice --side 3104 --gens 528 --pattern $P/misc/extra/lotr1.ANSI-1M.rle --verify
$R gpu-window --windows 64 --iters 4096
$R gpu-tile --k 16 --tw 7 --th 16 --side 200 --gens 65536 --pattern "$P/advanced counter v2.rle" --verify   # 16 generations per dispatch
$R gpu-tile --k 16 --tw 6 --th 48 --side 672 --gens 65536 --pattern $P/misc/squares.rle
$R gpu-tile --k 3 --side 16384 --gens 48 --same-src                           # large grid: slower than gpu-bitslice
```

The circuit is regenerated in `bitslice/` with `rule_circuit.py` (the hand-written network and its exhaustive check), `optimize.py 25 1` (25 minutes of optimisation), `optimize_more.py 6.5 7` and `emit.py`, each run with `"$VPY"`. The optimiser runs on a wall-clock budget, so a rerun does not reproduce the netlist gate for gate; `circuit.netlist` is the artifact.

The mixed universe of §12.5 is a `--compose` string of `file@x,y[+pre-run gens][~perturbation seed]` items, kept in `tools/composite_spec.txt` with paths relative to `research/`: the ROM at 512,512; six perturbed `chaotic` and six pre-run counters on a 512-cell pitch; `32x32 movie`, `squares` and `serial display` beside them. The three controller designs that §12.5 lists as not working were replaced in `cmd_adapt.rs` and cannot be rerun with the current harness.

`gpu-sweep` on the ROM spends about 90 s learning one period on the CPU before the GPU run. A pattern runs under the canonical rule named in its header (`cmd_stats.rs: variant_for`). The one comparison in this report that needs Golly's own copy of `Hutton32b.rule` is the check that it is the same rule as the committed one: `$R rulediff --a "table:<Golly folder>/Rules/Hutton32b.rule" --b table:rules/Hutton32b.rule` reports 0.

## Appendix B. Harness map

| File | Contents |
|---|---|
| `rule.rs`, `rulespec.rs` | CPU transliteration of both WGSL rule files, as they are now (canonical) and as first archived (`-v1`); loader for Golly tables (`.rule` with `@TABLE`, or a bare `.table`); rule selection |
| `cmd_lut.rs` | `lut`: a WGSL rule file executed on the GPU for all $2^{25}$ neighbourhoods, compared with the CPU port or with any other rule |
| `tools/fix_rules.py` | §10: the corrections to the WGSL rule files in `archive/`, applied in place, idempotent |
| `pattern.rs`, `mc.rs` | palette PNG, extended RLE, macrocell reader |
| `sim.rs` | per-cell LUT stepping (reference semantics) |
| `hl.rs` | the reference engine: hash-consed levels, memo floor, level-synchronous `solve`, root stepping, fingerprints |
| `flat.rs` | positional sweep (the renormalised tile automaton) on the CPU |
| `cmd_stats.rs` | `selftest`, `flat`, `root`, `fingerprint`, `rulediff`, `ruletree` |
| `cmd_gpu.rs` | `gpu-baseline`, `gpu-brute`, `gpu-sweep`, `gpu-probe`, `gpu-latency` |
| `cmd_brute2.rs` | hostile content; class-compressed LUT |
| `cmd_insert.rs` | lock-free find-or-insert with 32-bit CAS |
| `cmd_sparse.rs` | sparse universe, active-block sweep, macrocell writer |
| `cmd_adapt.rs` | §12: per-region floors and engines on the position-aware recursion (`hl.rs: result_at`), cost model, controller, `--compose` universes |
| `cmd_bitslice.rs` | §13: the bit-plane kernel (`gpu-bitslice`), the shared-tile kernel with several generations per dispatch (`gpu-tile`) and the one-thread-per-window kernel (`gpu-window`) |
| `bitslice/` | §13: the rule as a gate network. `rule_circuit.py` (hand-written network), `resub.py`, `abc_flow.py`, `xag.py`, `optimize.py` (optimiser), `emit.py`; artifacts `circuit.netlist`, `step_planes.wgsl`, `step_planes.rs`; `verify.py`; `scalar.py` and `tables.py` (forms A and C); `bench/` (CPU benchmark) |
| `tools/bitslice_series.sh` | every GPU measurement of §13, the shared-tile kernel included (section T) |
| `tools/time_golly.py`, `tools/head2head.py`, `tools/engine_choice.sh` | `bgolly` timing; head-to-head with a final-state check (`--v1`: engine on the v1 rules); engine-per-content table |
| `patterns/` | every pattern the report uses, with its source, rule and the sections that use it (`patterns/README.md`) |
| `rules/` | `Hutton32.table` and `Hutton32b.rule`, byte for byte as published; also `bgolly`'s rule folder |

## References

Sources marked † were read as abstract or secondary summary only.

1. R. W. Gosper, "Exploiting regularities in large cellular spaces", *Physica D* 10 (1984).
2. Golly source, `gollybase/ghashbase.{h,cpp}`, `hlifealgo.cpp`, `ruletable_algo.cpp`, `jvnalgo.cpp` — https://sourceforge.net/p/golly/code/ci/master/tree/gollybase/
3. T. Rokicki, "Life Algorithms", G4G13 (2018) — https://www.gathering4gardner.org/g4g13gift/math/RokickiTomas-GiftExchange-LifeAlgorithms-G4G13.pdf
4. T. Rokicki, golly-test list, "lockstep hashlife" (2014) — https://sourceforge.net/p/golly/mailman/message/32515364/
5. C. E. LaForest, "HashLife on GPU" (2010) — https://fpgacpu.ca/writings/HashLife-GPU.pdf
6. A. P. Goucher, lifelib — https://gitlab.com/apgoucher/lifelib
7. das67333, gol_engines — https://github.com/das67333/gol_engines
8. T. Fujita, K. Nakano, Y. Ito, bit-sliced Life on GPU, *IJFCS* 27 (2016) — https://doi.org/10.1142/S0129054116500404
9. Dijkstra & Brouwer, bit-packed Life in CUDA (2023) — https://binary-banter.github.io/game-of-life
10. M. Awad et al., "Better GPU Hash Tables" — https://arxiv.org/abs/2108.07232
11. D. Jünger et al., WarpCore — https://arxiv.org/abs/2009.07914
12. McCoy & Pandey, WarpSpeed — https://arxiv.org/abs/2509.16407
13. Polak et al., Hive — https://arxiv.org/abs/2510.15095
14. Z. Jia et al., GPU microarchitecture and TLB measurements — https://arxiv.org/abs/1903.07486
15. Maczan, WebGPU dispatch overhead measurements (2026) — https://arxiv.org/abs/2604.02344
16. N. Israeli, N. Goldenfeld, "Coarse-graining of cellular automata" — https://arxiv.org/abs/nlin/0508033
17. S. Rafler, SmoothLife — https://arxiv.org/abs/1111.1567 ; B. Chan, Lenia — https://arxiv.org/abs/1812.05433
18. A. D. Robison, "Fast computation of additive cellular automata", *Complex Systems* 1 (1987) — https://content.wolfram.com/sites/13/2018/02/01-1-15.pdf
19. C. Moore, "Predicting non-linear cellular automata quickly by decomposing them into linear ones" — https://arxiv.org/abs/patt-sol/9701008
20. D. Griffeath, C. Moore, "Life without Death is P-complete", *Complex Systems* 10 (1996) — https://content.wolfram.com/sites/13/2018/02/10-6-3.pdf
21. FFT versus direct convolution for Life, Julia benchmark — https://rivesunder.github.io/SortaSota/2021/09/27/faster_life_julia.html
22. Bit-packed CUDA versus PyTorch Life (121×) — https://www.boristhebrave.com/2025/09/11/accelerated-game-of-life-with-cuda-triton/
23. J. Signorini, 29-state von Neumann CA on a SIMD machine, *Supercomputing '89* † — https://dl.acm.org/doi/10.1145/76263.76282
24. E. Frankling, "auto": bit-plane and multi-state CA kernels in wgpu (2026) — https://erikfrankling.github.io/auto/
25. T. Hutton, `Hutton32.table`, Golly rule table repository — http://gollygang.github.io/ruletablerepository/downloads/Hutton32.table
26. `Hutton32b.rule`, Hutton32-Computer repository — https://github.com/FredericaBernkastel/Hutton32-Computer/blob/master/doc/Hutton32b.rule
