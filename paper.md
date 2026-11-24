# HashLife on GPU

*[Author names, affiliations and contact details: to be supplied by the authors.]*

**Abstract**—HashLife, as implemented in Golly, is the state of the art for long runs of large cellular automata, but it degrades by orders of magnitude on dense universes whose content does not repeat: on a dense universe of 9.5 million cells in the 32-state Hutton32b rule it reaches $2.4 \cdot 10^{7}$ cell-updates/s (2.5 generations/s). Brute-force GPU stepping avoids that failure but cannot exploit repetition. We measure, on a consumer GPU, the four rates that decide between the two (brute-force evaluation, random table probes, concurrent inserts and dispatch latency) and derive from them GPU HashLife (GHL), a single algorithm that memoizes only windows of 16×16 cells and larger, evaluates everything below that floor with a 367-gate bit-sliced circuit, runs the HashLife recursion level-synchronously as data-parallel waves under a positional sweep, and chooses its engine per region and per universe from measured novelty and active area. On an RTX 5060 Ti its components run at $3.0 \cdot 10^{11}$ cell-updates/s by brute force on any content and at up to $6.8 \cdot 10^{16}$ cell-updates/s ($6.7 \cdot 10^{6}$ generations/s) on a dense universe of $10^{10}$ cells whose content repeats. Against Golly 5.1, GHL is 1.3–43× faster on small dense universes; on a large dense universe it is $1.6 \cdot 10^{4}$× faster while the content is new and $6.4 \cdot 10^{5}$× once it repeats; on a sparse universe of $10^{5} \times 10^{5}$ cells it is 64× faster. On universes that are idle, which Golly memoizes completely, GHL's CPU implementation of the recursion runs at 0.57–1.9× Golly's speed. Final states are identical to Golly's throughout. The on-GPU miss path is specified and its primitives are measured, but it is not yet assembled; projections for it are reported separately.

**Index Terms**—Cellular automata, HashLife, memoization, hash consing, GPU computing, bit slicing, von Neumann neighbourhood.

---

## I. Introduction

HashLife [1] represents a universe as a hash-consed quadtree and memoizes the future of every node, so that regions which recur cost nothing to advance again. Golly [2] implements it for arbitrary rule tables, and for most patterns built in such rules it is the fastest simulator available: it advances idle circuits of up to millions of cells at up to $5.4 \cdot 10^{14}$ cell-updates/s ($10^{9}$ generations/s), because every block in them has been seen before.

This paper concerns two rules of the von Neumann family that Golly runs: Hutton32 [3], which Golly also implements natively [2], and Hutton32b [4], a variant with a 3×3 XNOR gate and two added rule sets in which the Hutton32-Computer project builds counters, ROMs, displays and a CPU. Both have 32 states and a five-cell neighbourhood, so a rule table has $2^{25}$ entries. Their patterns are dense. A circulating ROM of 9.5 million cells has a period of 528 generations, and every 16×16 block of it differs from every other at every moment of the period. On it Golly reaches $2.4 \cdot 10^{7}$ cell-updates/s (2.5 generations/s) and collects garbage 18 times in 180 s, while one CPU core stepping cells through a lookup table does $3.0 \cdot 10^{8}$ cell-updates/s, 13× more. Brute-force stepping on a GPU reaches $1.86 \cdot 10^{11}$ cell-updates/s on any pattern (Section III-B) but cannot exploit recurrence at all.

We ask what HashLife should look like when its memo table lives in GPU memory, and answer with measurements. The contributions are:

1. A cost model of memoized stepping in four measured rates (Section III). Its central quantity, the exchange rate $\rho$ between a table probe and a brute-force cell-update, is about 50 on the measured GPU, which places the optimal memo floor at 16×16-cell windows rather than HashLife's 2×2 leaves.
2. GPU HashLife (GHL), one algorithm for dense, sparse, novel and recurring content (Section IV), every parameter of which is set by a measurement (Table VIII).
3. A 367-gate bit-sliced circuit for Hutton32b, equal to the rule table on all $2^{25}$ neighbourhoods, which serves as GHL's brute-force engine at $3.0 \cdot 10^{11}$ cell-updates/s independent of content (Section IV-C).
4. An evaluation against Golly 5.1 on 13 patterns and a sparse composite universe with exact final-state comparisons that separates measured GPU components, a measured CPU implementation of the recursion and model projections (Sections V–VI).
5. A reproduction package in which every table is one command (Appendix A).

Section II reviews related work. Section III measures the costs and derives the model. Section IV presents GHL. Section V describes the evaluation, Section VI reports it, Section VII discusses projections and limitations, and Section VIII concludes.

## II. Related Work

### A. HashLife and Golly

Gosper's HashLife [1] memoizes, for each quadtree node of side $2s$, its centre $s/2$ generations later, and obtains a node's result from thirteen overlapping sub-results. Golly's generic implementation, [`ghashbase`] [2], uses 2×2 leaves, nodes of six pointers (48 B), scalar rule evaluation and a mark-and-sweep collector that runs when a memory limit is reached and keeps what is reachable from the current root. Its bottom memo entry, a 4×4 block advanced one generation, is worth four cell-updates. lifelib [5] uses 16×16 bit-plane leaves and a bit-sliced SIMD leaf kernel on a single CPU thread; it is the closest relative of the memo floor proposed here.

### B. Parallel HashLife

Rokicki sketched a level-by-level ("lockstep") order of the recursion in pseudocode in 2014 [6]; no implementation was found. In 2018 he observed that "no parallel versions of hashlife have been seen that outperform a single-threaded version" [7]. LaForest [8] used a $2^{16}$-entry 4×4 → 2×2 lookup table on a GPU and reported 1.18× over plain Life; a static hash table with eviction was proposed and not built. gol_engines [9] is the only measured parallel HashLife found: CPU work stealing, Life only, between 31× and below 1× depending on the pattern.

### C. Brute-force stepping on GPUs

Bit-sliced Life kernels on GPUs reach $10^{12}\text{–}10^{13}$ cell-updates/s [10], [11]. Bit-plane and multi-state kernels in wgpu are described in [12], and von Neumann's 29-state automaton was simulated on a SIMD machine in [13]. No throughput figure for a von Neumann-family rule on a GPU was found.

### D. Hash tables on GPUs

GPU hash tables reach $2\text{–}4 \cdot 10^{9}$ lookups/s out of cache on large cards [14]–[17]. Address-translation effects on GPUs are characterised in [18], and dispatch overheads in WebGPU are measured in [19].

### E. Approaches not pursued

FFT convolution computes neighbour sums in *O*(*N* log *N*) time independent of radius and is used by SmoothLife and Lenia for radii of 10–30 [20], [21]. At radius 1 it loses: FFT Life is 3× slower than direct convolution in one benchmark [22], and convolution-based Life in PyTorch is 121× slower than a bit-packed CUDA kernel in another [23]. For a non-totalistic 32-state rule the only linear statistic from which the next state follows is an injective code of the neighbourhood, which is the table index a direct kernel builds anyway. Generations can be skipped in closed form only for additive rules [24] and rules that decompose into linear ones [25]; predicting general automata is P-complete [26]. Coarse-graining [27] looks for automata on blocks of cells; the tile automaton of Section IV-D is an exact instance whose alphabet is discovered lazily.

Not found in the literature searched: a memo table of CA block transitions held in GPU memory; a bounded-depth HashLife that falls back to a brute-force kernel on a miss; a tile automaton swept positionally; a measured level-synchronous HashLife. The search could not reach forum threads behind logins.

## III. Measured Cost Model

### A. Platform

All figures were taken on an NVIDIA RTX 5060 Ti with 16 GB (driver 591.59, 448 GB/s, 32 MB L2), through wgpu 30 on Vulkan under Windows 11, with an Intel i5-6600K and 16 GB of RAM as host. Throughput is given throughout in cell-updates per second, cells × generations / s, where the cells are the pattern's bounding box. A universe composed of copies counts as one pattern and is measured as built at 16×16 tiles; tiled and random test grids count every cell. Padding that an engine adds (zero rings, tile alignment) is not counted, so two engines on the same workload are credited with the same cells and their rates compare directly. Where a pattern's speed is of interest, generations per second follow in parentheses. The replicator grows during a run; its rates are counted over its final bounding box and are therefore upper bounds. Empty cells inside the bounding box count too, so on sparse and idle universes, where memoization skips most of them, the rate is an effective one and can exceed the brute-force ceiling of $3.0 \cdot 10^{11}$; every such row states the cells it counts. Latency figures are given as time per generation.

### B. Four rates

Let a *tile* of level $L$ be a block of $s \times s$ cells, $s = 2^L$. A $2s \times 2s$ window determines its central tile $s/2$ generations later. Call that a *tile-step*; it is worth

$$V_L = s^2 \cdot \frac{s}{2} = 2^{3L-1} \text{ cell-updates} \qquad (1)$$

A tile-step can be obtained by brute force (cost $V_L / R_B$), by a memo hit (cost $1/R_P$), or by a miss (a probe plus the work to compute and store the result). Table I lists the four rates that price these options.

**TABLE I.** Cost constants measured on the RTX 5060 Ti

| Constant | Value | Measured with |
|---|---|---|
| $R_B$, brute-force evaluation, byte-LUT kernel | $1.86 \cdot 10^{11}$ cell-updates/s | `gpu-brute` |
| $R_B$, brute-force evaluation, bit-plane circuit | $3.0 \cdot 10^{11}$ cell-updates/s | `gpu-bitslice` |
| $R_P$, random 32-byte fetch, table ≤ 16 MiB | $3.2\text{–}3.5 \cdot 10^{10}$/s | `gpu-probe` |
| $R_P$, table of 32 MiB | $1.3 \cdot 10^{10}$/s | `gpu-probe` |
| $R_P$, table of 128 MiB … 8 GiB | **$3.7\text{–}4.0 \cdot 10^{9}$/s, flat** | `gpu-probe` |
| $R_I$, concurrent find-or-insert (32-bit CAS) | $1.0\text{–}1.9 \cdot 10^{9}$/s | `gpu-insert` |
| Dispatch chained inside one compute pass | 3.3–4.2 µs (typically 3.4) | `gpu-latency` |
| New compute pass | 49–97 µs | `gpu-latency` |
| Separate `queue.submit` | 150–460 µs | `gpu-latency` |
| Dispatch and 16-byte read-back to the CPU | 0.1–2.4 ms | `gpu-latency` |
| Miss-path wave of 16 dispatches, indirect dispatch (three passes) | 260–490 µs | `gpu-latency` |
| The same with wgpu's indirect-call validation removed | 135–165 µs | `gpu-latency` |
| The same at fixed capacity with early exit (one pass) | **51–55 µs** | `gpu-latency` |

The byte-LUT kernel (`gpu-brute`) looks up each cell's next state in the $2^{25}$-byte rule table. It stores four cells per `u32`, runs 8×8 workgroups on ping-pong buffers with a zero ring, and advances all generations of a submit inside one compute pass. It reaches $1.86 \cdot 10^{11}$ (371 GB/s of 448 GB/s), and workgroup shape (8×8 to 64×4) and backend (Vulkan, DX12) change this by less than 2 %. The bit-plane kernel of Section IV-C reaches $3.0 \cdot 10^{11}$ at 1.25 bytes of traffic per cell-update instead of 2.

The probe rate is the same from 128 MiB to 8 GiB: there is no address-translation cliff, so VRAM converts directly into memo capacity. wgpu caps one binding below 2 GiB, so large tables are sharded across bindings; `gpu-probe` runs 8 × 1 GiB this way at full speed.

### C. The exchange rate

The ratio

$$\rho = \frac{R_B}{R_P} \qquad (2)$$

is the number of cell-updates that cost as much as one probe. It is about 50 for a table larger than L2 with the byte-LUT kernel, 80 with the bit-plane kernel, and 5.5 when the table fits L2. A memo level pays only if $V_L \gg \rho$; Table II gives the best possible gain of a hit over brute force.

**TABLE II.** Value of a tile-step and best possible gain of a memo hit, $V_L / \rho$

| Tile | Generations | $V_L$ | $V_L / \rho$, byte LUT ($\rho = 50$) | $V_L / \rho$, bit-planes ($\rho = 80$) |
|---|---|---|---|---|
| 8×8 | 4 | 256 | 5× | 3.2× |
| 16×16 | 8 | 2,048 | 41× | 26× |
| 32×32 | 16 | 16,384 | 330× | 205× |
| 64×64 | 32 | 131,072 | 2,600× | 1,640× |

On a CPU core the ratio comes out similar ($R_B \approx 3\text{–}4 \cdot 10^{8}$ with a table, $R_P \approx 10^{7}$ out of cache, $\rho \approx 30\text{–}60$) and rises to a few hundred with bit-sliced evaluation (Section VI-H). Golly's bottom memo entry is worth $V = 4$, an order of magnitude below $\rho$. That costs nothing while everything hits and a great deal when content is new.

### D. Memory per memoized cell-update

A Golly node is 48 B and covers four cell-updates at the bottom level; summed over levels, novel content costs about 13.7 B per cell-update. A tile-level memo entry is 32 B (four key IDs, a value, padding) and covers $V_L$ cell-updates; a new 16×16 tile adds 160 B of content at 5 bits per cell.

**TABLE III.** Memory per novel cell-update

| Scheme | Bytes per novel cell-update | One period of [`lotr1.ANSI-1M`] ($5 \cdot 10^{9}$ cell-updates) |
|---|---|---|
| Golly [`ghashbase`] | 13.7 | ≈ 70 GB |
| Memo at 16×16 tiles, entry only | 0.016 | 79 MB |
| Memo at 16×16 tiles, every result tile new | 0.094 | 473 MB |
| Memo at 32×32 tiles, entry only | 0.002 | 20 MB |

This is why the ROM never completes a period in Golly under an 8 GB limit, yet its whole period fits a tenth of the GPU's memory at a coarser grain. At 16×16 tiles a 16 GB card holds about $4 \cdot 10^{8}$ entries ($8 \cdot 10^{11}$ cell-updates) when result tiles recur, or $8 \cdot 10^{7}$ ($1.7 \cdot 10^{11}$ cell-updates) when every result tile is new.

### E. The speedup law

With miss rate $\mu_L$ (fraction of tile-steps not found) and a miss costing $\kappa$ times the brute-force price of a tile-step, the speedup of memoized stepping over brute force is

$$S_L = \frac{1}{\rho / V_L + \kappa \mu_L} \qquad (3)$$

Memoization at level $L$ wins while $\mu_L < (1 - \rho/V_L)/\kappa$. For a miss that evaluates the whole $2s \times 2s$ window plus two inserts, $\kappa \approx 4.5$ at $L = 4$ and the break-even miss rate is about 22 %; evaluating contiguous misses as one region with a halo approaches $\kappa \approx 1.1$ and break-even near 85 %.

Table IV gives $\mu_L$ for the flat tile automaton of Section IV-D on the project's patterns, measured over the second half of each run so that start-up transients are excluded.

**TABLE IV.** Miss rate $\mu_L$ of the flat tile automaton by tile size

| Pattern | 8×8 | 16×16 | 32×32 | 64×64 | 128×128 |
|---|---|---|---|---|---|
| [`advanced counter v2`] (262 k gens; period 513,024) | 0.41 % | 1.70 % | 5.4 % | 9.4 % | 12.2 % |
| [`32x32 movie`], playing | | 0.16 % | | | |
| [`16x16 movie`] | | 0.12 % | 0.56 % | | |
| [`Segmented ROM (256k)`] | | 0.07 % | 0.59 % | | |
| [`component-basis`] | | 0.09 % | 0.64 % | | |
| [`squares`] | | 3.5 % | 7.0 % | | |
| [`chaotic`] | 10.5 % | 25.5 % | 34.9 % | | |
| [`lotr1.*`] ROMs | | 96 % for 528 gens, then 0 | 0 after 1,056 gens | | |
| [serial] and [parallel] displays, [demultiplexer], [`cpu`], [`BIN2BCD`] (idle) | | 0 | 0 | | |

The counter never repeats a global state within the run, yet 98.3 % of its 16×16 tile-steps are repeats. Inside the recursion a window is requested only when its parent missed. Table V gives the conditional miss rate per window size: of the windows requested because their parent missed, the fraction that missed too.

**TABLE V.** Conditional miss rate per window size (8×8 base tiles)

| Window | [ROM][lotr1.PPMd-244K.rle], first 512 gens | [counter] | [`chaotic`] | [`32x32 movie`] | [`squares`] |
|---|---|---|---|---|---|
| 16×16 | 51 % | 21 % | 31 % | 33 % | 29 % |
| 32×32 | 99.7 % | 29 % | 57 % | 38 % | 35 % |
| 64×64 | 99.8 % | 36 % | 90 % | 45 % | 36 % |
| 128×128 | 99.9 % | 39 % | 93 % | 65 % | 51 % |
| 256×256 | 99.5 % | 45 % | 89 % | 74 % | 55 % |
| Live cells in bounding box | 86 % | 64 % | 80 % | 77 % | 35 % |

Four of the five patterns are equally dense and want opposite treatment. Logic misses 20–55 % at every scale, so every level earns its keep. The ROM, while it is new, misses everything above 16×16: no tile size helps until the whole period has been seen, after which the largest tiles hit too. **The signal that selects an engine is novelty, not density.**

## IV. The GHL Algorithm

### A. Overview

GHL is HashLife with five changes, each forced by a measurement of Section III:

1. **A memo floor.** Windows below 16×16 cells are never memoized; a missed floor window is brute-forced (Table II).
2. **A bit-sliced brute-force engine.** Below the floor cells are stepped as five bit-planes by a gate network of the rule, at the same rate on any content (Section IV-C).
3. **A level-synchronous schedule.** The recursion runs level by level, one data-parallel wave per level, so that every stage is a GPU dispatch over a batch (Section IV-E).
4. **A positional sweep on top.** The top level is a flat array of tile IDs stepped as a tile automaton: one gather and one probe per tile (Section IV-D).
5. **Choice by novelty and active area.** Per region, the engine follows novelty (Table V); per universe, the depth follows the active area, because dispatch latency is paid once per generation by the whole universe (Section IV-F).

Algorithm 1 is the whole method; Algorithms 2 and 3 are its two non-trivial steps.

```
Algorithm 1  GHL: advance a universe
Input: universe as hash-consed tiles of 8x8 cells; tables in VRAM
 1  for each epoch (a fixed number of waves, all inside one compute pass per frame):
 2      A <- live cells in active blocks
 3      if A < A1:                                    # dispatch latency dominates
 4          run the fastest of {host HashLife recursion, shared-tile kernel (k = 16),
            plain bit-plane kernel}; time the others again with exponential back-off
 5      else:                                         # throughput dominates
 6          f <- 32x32 windows if A < A2 else 16x16 windows      # universe-wide floor
 7          for each wave of the epoch:
 8              HitSweep(top level s, active blocks)  # one gather + one probe per tile
 9              Solve(misses of the sweep, floor f)   # Algorithm 2
10              step regions in plain-cells mode as cells under a region-level memo
11          Controller(regions)                       # Algorithm 3
12          raise s while its hit rate stays above 40 %; lower it when it stops paying
13      hash the ID grid (global cycles); rebuild tables past their load limit
```

```
Algorithm 2  Solve(l, Q): level-synchronous HashLife with a memo floor
Input: level l, batch Q of distinct windows; every step is one dispatch at fixed capacity
 1  M <- windows of Q whose probe misses
 2  if l is the floor f:
 3      expand M into an atlas; step it by brute force on bit-planes; intern the centres
 4  else:
 5      S1 <- distinct nine overlapping sub-windows of each m in M;      Solve(l-1, S1)
 6      S2 <- distinct four windows built from those nine results;       Solve(l-1, S2)
 7      result(m) <- the node of the four results
 8  insert (M, results) with the claim protocol; commit; re-probe the missed positions
    on overflow of a fixed capacity: leave a sentinel, replay the wave with a larger one
```

```
Algorithm 3  Controller: engine per 256x256 region, 16-generation steps
 1  start every region at the floor f
 2  coarser alternatives: price them from the region's own statistics (requests and misses
    above a level do not depend on what happens below it); switch when 15 % cheaper
 3  finer alternatives: subdivide one floor miss in sixteen and price the result
 4  from plain cells: a two-step trial of f at most every 16 steps, the interval doubling
    up to 256 steps while trials fail
```

Fig. 1 shows the tile automaton that the sweep steps; Table VI the data it keeps on the GPU; Table VIII every parameter and the measurement that sets it.

```
phase A (tiles at k*s)          phase B (tiles at k*s + s/2)        phase A
+----+----+----+                    +----+----+                 +----+----+----+
| a  | b  | c  |   R(a,b,d,e) ->    | p  | q  |                 |    |    |    |
+----+----+----+   R(b,c,e,f) ->    +----+----+  R(p,q,r,t) ->  |    | e' |    |   e' = e after s generations
| d  | e  | f  |   ...              | r  | t  |                 |    |    |    |
+----+----+----+                    +----+----+                 +----+----+----+
| g  | h  | i  |
+----+----+----+
```

**Fig. 1.** The positional sweep. Each tile-step maps four tile IDs to the ID of the centred tile $s/2$ generations later; two alternating phases bring the lattice back.

### B. Data in VRAM

**TABLE VI.** Data structures in VRAM

| Structure | Layout | Size |
|---|---|---|
| ID grids A, B | one `u32` per tile of the sweep level, two phases | 4 B per 256 cells at 16×16 tiles |
| Memo table per level | open addressing, linear probing; 32 B entries: four key IDs, value, spare | $2 \cdot 10^{8}$ entries in 6.4 GB |
| Claim words | one `atomic<u32>` per slot | 4 B per slot |
| Tile store | 8×8 base tiles as five bit-planes, indexed by tile ID | 40 B per distinct base tile |
| Active-block list | origins of blocks that are not known to be empty | a few words per block |
| Atlas | scratch bit-plane grid for windows being brute-forced | sized to the miss batch |

A node's ID is its slot index, so there is no separate allocator and the children of an ID are a direct read. Entries never move; reclaiming space is a rebuild. A 100k × 100k universe is 151 MiB per ID buffer at 16×16 tiles, where a byte per cell would need 10 GB.

### C. Brute-force engine: the rule as a bit-sliced circuit

The rule tells apart 15 classes of states per neighbour and 29 for the centre. Starting from a network written by hand from the rule's structure (473 gates), global resubstitution with observability don't-cares, ABC rewriting scripts run through Yosys, and an inverter placement solved as MaxSAT reduce canonical Hutton32b to **367 two-input gates** (165 AND, 137 OR, 14 XOR, 51 NOT; depth 25) in 24 minutes. Every intermediate network was kept only if it agreed with the table on all $2^{25}$ entries; the final netlist, its WGSL emission and its Rust emission were each checked exhaustively, after a deliberately corrupted gate was confirmed to be caught. The network is not proven minimal.

Applied to bit-planes, the gates update 32 cells per operation. A cell is five bits in five `u32` words, one per bit; an invocation loads its group and the groups above and below, forms the east and west neighbours of each plane with two shifts and an OR, evaluates the network and stores five words. Table VII compares this form with the alternatives.

**TABLE VII.** Three ways to evaluate the rule without the $2^{25}$-byte table

| Form | What it is | Cost per cell-update | Measured |
|---|---|---|---|
| A. Scalar | the 25-bit neighbourhood in one register; a straight-line program yields the 5 output bits | 364 32-bit operations | CPU: $1.5\text{–}2.5 \cdot 10^{7}$/s; the table lookup does $3.3\text{–}6.0 \cdot 10^{8}$ on a real pattern |
| B. Bit-sliced | the same network; every operand holds one bit of 32, 64 or 256 cells | 12.4 operations per cell at 32 bits | GPU: $3.0 \cdot 10^{11}$/s; CPU: $0.9\text{–}4.5 \cdot 10^{9}$/s |
| C. Small tables | the table factored through class codes, `T5[Tn[c,n] + Te[c,e] + Ts[c,s] + Tw[c,w]]` | 5 loads from 14,971 B | CPU: $2.6\text{–}4.5 \cdot 10^{8}$/s |

Form A loses to the table by 2.5–34×; it pays only when the other bits of every register carry other cells, which is form B. Form C fits an L1 cache but is still 3–10× slower than bit-planes on the CPU. A one-table variant with a 21-bit class index (2 MiB) was measured on the GPU at $4.2 \cdot 10^{10}\text{–}1.3 \cdot 10^{11}$/s, and a four-load chain of 10,592 B was sized but not built.

GHL uses form B in two kernels. The **plain kernel** steps the grid once per dispatch with 8×8 workgroups. The **shared-tile kernel** gives a workgroup a tile of bit-plane words plus a halo of $k$ rows, loads it into workgroup-shared memory and steps it $k$ generations with one evaluation per thread per generation and a barrier between generations; $k$ is limited to 32 by the one-word halo.

### D. Hit path: the positional sweep

Hash-consing gives every distinct tile a small integer ID. The memoized map

$$R_L : (\mathrm{ID}_{nw}, \mathrm{ID}_{ne}, \mathrm{ID}_{sw}, \mathrm{ID}_{se}) \mapsto \mathrm{ID}_{c} \qquad (4)$$

where $\mathrm{ID}_{c}$ is the centred tile $s/2$ generations later, is itself a cellular automaton over tile IDs with a 2×2 neighbourhood on a lattice that shifts by half a tile each step (Fig. 1). Simulating the original automaton for $s/2$ generations is one gather of four IDs and one probe per tile, independent of $L$. Every window is evaluated once per sweep, so the overlap that costs HashLife thirteen sub-calls for eight results' worth of work is shared for free, and nothing above the sweep level is hashed.

The hit path is one dispatch per sweep with no atomics, because inserts happen in separate dispatches. A sweep of $n_d$ chained dispatches advancing $2^{L-1}$ generations costs at least $n_d \cdot 3.4\ \mu\text{s} / 2^{L-1}$ per generation, however small the universe: 0.42 µs for a hit-only sweep at 16×16 tiles.

For sparse universes the sweep runs over a list of active blocks. The key of four empty tiles maps to the empty tile, so a block that is empty with an empty one-block border stays empty and is skipped exactly; a block joins the list when a border tile of a neighbour becomes non-empty.

### E. Miss path: level-synchronous recursion

Algorithm 2 is HashLife's recursion in breadth-first order: instead of descending depth-first, each level processes one batch of distinct nodes per wave. With floor $a$ the bottom result advances $2^{a-1}$ generations, so $T$ generations need $T / 2^{a-1}$ dependent waves unless a higher level hits; batches inside a wave are parallel and waves are not. Fixed cost per wave therefore bounds every pattern whose misses are few but constant, and a handful of nodes per wave is cheaper on the host CPU, where a wave costs microseconds and no dispatch. Algorithm 1 accounts for both: it runs the recursion on the host when that is faster (line 4).

A missed window is resolved entirely on the GPU:

* **Deduplicate and claim.** WGSL has only 32-bit compare-exchange, so a 128-bit key cannot be installed atomically. Each slot has a claim word that is 0 (empty), $j+1$ (claimed in this batch by element $j$) or has bit 31 set (committed earlier). A thread that finds a slot claimed by element $t$ compares against element $t$'s key read from the immutable batch input, never from the table; committed slots are immutable and compared in place; a second dispatch marks the batch's claims committed. One dispatch resolves a whole batch without locks.
* **Expand, evaluate, intern.** Each distinct missing key copies its four tiles into a $2s \times 2s$ window of the atlas, the brute-force engine advances the atlas $s/2$ generations (garbage from a neighbouring window travels one cell per generation and cannot reach the central tile), and the central tile is hash-consed by content with the same claim protocol.
* **Publish.** The result ID is written into the memo entry and the missed positions are re-probed.

The number of misses, and then of distinct misses, is known only on the GPU. GHL does not size dispatches indirectly, which costs 260–490 µs per wave in wgpu (Table I). Every stage is dispatched for a fixed capacity taken from stale, asynchronously read miss counters, and invocations beyond the live count return at once: a wave of 16 dispatches then costs 51–55 µs, and an idle 64-thread workgroup costs about 1 ns once the 3.2 µs fixed cost of a dispatch is paid. A wave that overflows its capacity leaves a sentinel in the affected tiles and is replayed from the last snapshot with a larger capacity. The CPU never sits in the per-wave path.

### F. Choosing tile size, floor and engine

*Sweep tile.* A level that never hits adds one failed probe and one insert per eight tile-steps below it, at most $1/7$ over all higher levels together. With $p$ and $\iota$ the prices of a probe and an insert, a level pays for itself once its hit rate exceeds

$$h^{*} = \frac{p + \iota}{8p + \iota} \qquad (5)$$

about 40 %. GHL raises the sweep tile while its hit rate stays above that. Each level takes twice as long to learn as the one below: a tile of side $2^L$ sees the same content again only after $\mathrm{lcm}(528, 2^L)$ generations on the ROM.

*Floor.* Brute-forcing large isolated windows is the worst choice everywhere: each carries its own halo and the recursion asks for thirteen of them where eight would tile the result. Below 16×16 windows hashing costs more than the cells: an 8×8 window is 0.7 ns of brute force and 3.5 ns of bookkeeping. GHL therefore stores 8×8 base tiles and brute-forces missed 16×16 windows (F16); Table XIX gives the measurements.

*Engine per region.* The recursion is already a per-window switch between memo and brute force: a window that hits costs a probe, one that misses at the floor is brute-forced. A second engine pays only where content is new at every scale below the region: plain cells stepped per 256×256 region, with memo only above the region. Algorithm 3 enters it on sustained novelty at 32×32 windows and above and leaves it lazily. Two simpler controllers failed and shaped this one (Section VI-G): experiments must jump to a depth where being wrong is cheap, and the evidence for finer depths must be sampled, not obtained by whole-step trials. While anything is being learned, steps stay at 16 generations, because a step that does not divide a period multiplies the learning time.

*Depth per universe.* A missed floor window needs about seven dispatches plus one per brute-forced generation, and the levels above add about five per floor wave. Latency is global: one region at a shallow floor makes the whole universe pay that floor's dispatches. GHL therefore chooses the universe-wide floor $f$ that minimises

$$C(f) = L(f) + \sum_{r} w_r(f) \qquad (6)$$

with $L(f)$ the dispatch latency per generation at floor $f$ and $w_r(f)$ the work of region $r$; regions may only go coarser than $f$. With the counter's measured work per region this gives the thresholds $A_1$ and $A_2$ of Algorithm 1 (Table XXII).

**TABLE VIII.** Parameters of GHL and the measurements that set them

| Parameter | Value | Set by |
|---|---|---|
| Base tile | 8×8 cells, five bit-planes | floors (Table XIX) |
| Memo floor | 16×16 windows (F16); 32×32 (F32) for mid-sized active areas | $V_L / \rho$ (Table II), floors (Table XIX), depth (Table XXII) |
| Brute-force engine | 367-gate circuit on five bit-planes | Tables XII, XIII |
| Generations per dispatch on small grids | $k = 16$ in a shared tile | Table XII |
| Sweep tile | start at 16×16; raise while the hit rate exceeds 40 % | (5), Tables XV, XVI |
| Step while learning | 16 generations | Table XX |
| Engine region | 256×256 cells; plain cells on sustained novelty at ≥ 32×32 windows | Tables V, XX, XXI |
| $A_1$: below it, no memo | $\approx 1.0 \cdot 10^{6}$ active cells | Table XXII |
| $A_2$: F32 between $A_1$ and $A_2$, F16 above | $\approx 1.3 \cdot 10^{7}$ active cells | Table XXII |
| Miss-path dispatch | fixed capacity with early exit, one pass, ≈ 54 µs per 16-dispatch wave | Table I |
| Memo entries | 32 B, open addressing; tables sharded across bindings below 2 GiB | Table I |

### G. Exactness

Keys are compared in full and tiles by content. There is no probabilistic hashing anywhere, and the result of every engine is bit-identical to per-cell stepping. IDs are canonical, so two ID grids are equal exactly when the universes are; a hash of the grid taken every epoch detects a globally periodic universe. The memo is a cache: dropping it, for instance when a table is rebuilt from the entries reachable from the current grids, costs a relearning transient and never correctness.

## V. Evaluation Methodology

### A. Baseline

The state-of-the-art baseline is Golly 5.1's command-line `bgolly` without rendering, with an 8 GB memory limit and the best of several step sizes per pattern. Golly 5.1 is the official 64-bit Windows release of November 2026, built from branch `master` of Golly's repository [2]. The release is not tagged and its build commit is not recorded, but Golly's source is the same in every commit from 9a7079a8 ("updated version to 5.1", 2026-11-06) to 53bdb74e (2026-11-15), the head of `master`; the commits in between change only patterns and the change log. All references to Golly's source are to commit 53bdb74e. Hutton32b patterns run under RuleLoader with the published [`Hutton32b.rule`][Hutton32b.rule] [4] and Hutton32 patterns under Golly's native JvN algorithm, which is equal to [`Hutton32.table`] [3] on all $2^{25}$ neighbourhoods (Appendix B). Load time is subtracted. On the GPU, the byte-LUT brute-force kernel of Section III-B is the baseline.

### B. Patterns

Fifteen patterns of the Hutton32-Computer collection and two counters stored as palette PNG images, a [Hutton32b counter] of the same size as `advanced counter v2` (191×197) and a [Hutton32 counter] (633×449) that is tiled into the large brute-force grids, all listed with their rules and bounding boxes in [`README.md`]: an 8-bit counter ([`advanced counter v2`], 191×197, period 513,024), an active display ([`squares`], 650×423), a chaotic circuit ([`chaotic`], 151×132), two circulating ROMs ([`lotr1.ANSI-1M`], 3095×3059, and [`lotr1.PPMd-244K`], 1548×1461, period 528), circuits that are idle most of the time ([serial] and [parallel] displays, a [segmented ROM], a [demultiplexer], two movies, [16×16] and [32×32], a [CPU], a [BCD converter], a [component sheet]) and Tim Hutton's [replicator], which grows to 6813×6691. Large universes are built from phase-shifted copies, so that no two large blocks are equal.

### C. Verification

Every GPU throughput figure was followed by a read-back and a cell-exact or tile-exact comparison with the CPU. The CPU reference engine agrees with per-cell stepping in 22 configurations and with Golly (same population, bounding box and content hash) on 10 patterns after $2^{24}$ generations, on the replicator after 262,144 generations and on a 98,304 × 104,448 universe. The harness's CPU rule implementation and the circuit equal the canonical rule tables on all $2^{25}$ neighbourhoods (Appendix B).

### D. Implementation status and evidence classes

GHL is not yet one program. Table IX lists where each component runs in this evaluation. Results are labelled accordingly: **GPU** (measured on the GPU and verified), **CPU** (the same algorithm measured on the CPU reference engine) and **model** (operation counts of the reference engine priced with the constants of Table I).

**TABLE IX.** Implementation status of GHL's components

| Component | Status | Evidence |
|---|---|---|
| Plain and shared-tile bit-plane kernels | built, GPU | GPU |
| Positional hit-path sweep (dense) and active-block sweep (sparse) | built, GPU; table learned on the CPU | GPU |
| Claim protocol for find-or-insert | built, GPU benchmark | GPU |
| Fixed-capacity 16-dispatch wave | built as a benchmark of empty waves | GPU |
| Level-synchronous recursion with memo floor | built, CPU reference engine ([`hl.rs`]) | CPU |
| Per-region floors, plain-cells mode, controller, depth choice | operation counts on the CPU, priced with measured constants (`adapt`) | model |
| GPU miss path (expand, evaluate on atlas, intern) as one pipeline | specified, not built | model |
| Racing of executors, table rebuild, cycle detection | specified, not built | none |

## VI. Results

### A. The baseline

**TABLE X.** Golly 5.1 HashLife on the project's patterns (best step size)

| Pattern | Size | Golly, cell-updates/s (gens/s) | Notes |
|---|---|---|---|
| [`advanced counter v2`] | 191 × 197 | $2.2 \cdot 10^{10}$ ($5.8 \cdot 10^{5}$) | $1.9 \cdot 10^{9}$ ($5 \cdot 10^{4}$) at step 1 |
| [`squares`] | 650 × 423 | $3.7\text{–}8.2 \cdot 10^{9}$ ($1.3\text{–}3.0 \cdot 10^{4}$) | active display |
| [`chaotic`] (active phase) | 151 × 132 | $6.1\text{–}9.8 \cdot 10^{8}$ ($3.1\text{–}4.9 \cdot 10^{4}$) | 22 collections in 16 k gens |
| [`32x32 movie`] (while playing) | 928 × 1513 | $1.4 \cdot 10^{10}$ ($\approx 1.0 \cdot 10^{4}$) | 524 k gens of activity, then idle |
| [`lotr1.ANSI-1M`] | 3095 × 3059 | **$2.4 \cdot 10^{7}$** (2.5) | 18 collections in 180 s; one step of 4096 never finished |
| [Hutton replicator] | grows to 6813 × 6691 | at most $1.1 \cdot 10^{11}$ ($2.4 \cdot 10^{3}$) | final bounding box; 34 collections for 262 k gens |
| Idle circuits ([serial] and [parallel] displays, [segmented ROM], [demultiplexer], [`16x16 movie`], [`cpu`], [`BIN2BCD`]) | $1.2 \cdot 10^{5}\text{–}5.7 \cdot 10^{6}$ cells | $4.9 \cdot 10^{12}\text{–}5.4 \cdot 10^{14}$ ($4.8 \cdot 10^{6}\text{–}9.9 \cdot 10^{8}$) over $2^{24}$ gens; $1.3 \cdot 10^{11}\text{–}7.3 \cdot 10^{12}$ ($8.2 \cdot 10^{4}\text{–}1.2 \cdot 10^{7}$) over 65,536 gens | fully memoized; the rate grows with the run |

On the ROM a single CPU core stepping cells with the byte table does $3.0 \cdot 10^{8}$ cell-updates/s (32 gens/s), 13× Golly; the byte-LUT GPU kernel does $1.8 \cdot 10^{11}$ (18,992 gens/s).

### B. Overall comparison

Table XI places GHL against both baselines on every workload class. The GHL column gives the engine that Algorithm 1 selects for the workload and the rate of that engine.

**TABLE XI.** GHL against Golly 5.1 HashLife and the byte-LUT brute-force GPU kernel, in cell-updates/s (generations/s)

| Workload | Cells | Golly 5.1 | Brute-force GPU kernel (byte LUT) | GHL engine | GHL | GHL vs Golly | Evidence |
|---|---|---|---|---|---|---|---|
| [Counter] | 37,627 | $2.2 \cdot 10^{10}$ ($5.8 \cdot 10^{5}$) | $1.0 \cdot 10^{10}$ (274,544)ᵃ | shared tile, $k = 16$ | $2.9\text{–}3.0 \cdot 10^{10}$ (770,000–785,000) | 1.33–1.35× | GPU, cell-exact |
| [`squares`] | 274,950 | $3.7\text{–}8.2 \cdot 10^{9}$ ($1.3\text{–}3.0 \cdot 10^{4}$) | $4.9 \cdot 10^{10}$ (177,445) | shared tile, $k = 16$ | $1.5 \cdot 10^{11}$ (556,000–563,000) | 19–43× | GPU, cell-exact |
| [`chaotic`] | 19,932 | $6.1\text{–}9.8 \cdot 10^{8}$ ($3.1\text{–}4.9 \cdot 10^{4}$) | — | plain kernel | $\approx 5.4 \cdot 10^{9}$ ($\approx 2.7 \cdot 10^{5}$) | 5.5–8.7× | model: dispatch-bound rate of a small grid |
| [ROM][lotr1.ANSI-1M.rle], first period | $9.5 \cdot 10^{6}$ | $2.4 \cdot 10^{7}$ (2.5) | $1.8 \cdot 10^{11}$ (18,992) | plain cells (bit-planes) | $3.9 \cdot 10^{11}$ (40,999) | $1.6 \cdot 10^{4}$× | GPU, cell-exact over a period |
| [ROM][lotr1.ANSI-1M.rle], after one period | $9.5 \cdot 10^{6}$ | $2.4 \cdot 10^{7}$ (2.5) | $1.8 \cdot 10^{11}$ (18,992) | sweep, 32×32 tiles | $1.5 \cdot 10^{13}$ (1,607,022) | $6.4 \cdot 10^{5}$× | GPU hit path, tile-exact |
| 32×32 phase-shifted [ROMs] | $1.0 \cdot 10^{10}$ | not attempted | does not fit | sweep, 16×16 to 256×256 tiles | $8.6 \cdot 10^{12}$ (858) to $6.8 \cdot 10^{16}$ (6,714,020) | — | GPU hit path, tile-exact |
| 2304 [counters], sparse (0.79 % of tiles non-empty) | $1.0 \cdot 10^{10}$ | $4.4 \cdot 10^{13}$ (4,323) over the first 16,384 gens; $6.4\text{–}9.9 \cdot 10^{14}$ (61,878–96,581) over $2^{20}$ | does not fit | active-block sweep | $2.9 \cdot 10^{15}$ (278,648) | 64× / 2.9–4.5× | GPU hit path, same final state as Golly |
| Idle circuits | $1.2 \cdot 10^{5}\text{–}5.7 \cdot 10^{6}$ | $4.9 \cdot 10^{12}\text{–}5.4 \cdot 10^{14}$ ($4.8 \cdot 10^{6}\text{–}9.9 \cdot 10^{8}$) | — | host recursion | $5.8 \cdot 10^{12}\text{–}1.0 \cdot 10^{15}$ ($5.3 \cdot 10^{6}\text{–}1.7 \cdot 10^{9}$) | 0.57–1.9× | CPU, same final states |
| [Hutton replicator], 262,144 gens | at most $4.6 \cdot 10^{7}$ | at most $1.1 \cdot 10^{11}$ (2,351) | — | memoized recursion | at most $5.5 \cdot 10^{10}$ (1,196) on the CPU | 0.51× (CPU); 7–21× projected on the GPU | CPU; model |

ᵃ On the [Hutton32b PNG counter], a different counter of the same size; at this size every kernel waits for the dispatch, not for the content. The byte LUT needs 1 B per cell in each of two buffers, which does not fit $10^{10}$ cells in 16 GB. Idle circuits: $2^{24}$ generations at step 1024 (Table XVIII); the ratio is per pattern. Rates count every cell of the bounding box, empty ones included; the sparse universe and the idle circuits are therefore credited far above the brute-force ceiling of $3.0 \cdot 10^{11}$, and their rows compare engines on the same cells, not work done.

GHL is faster than Golly on every workload measured on the GPU, by 1.3× on the small active counter and by four to six orders of magnitude on dense novel content. The hit-path rates assume a table that the CPU learned (80–90 s per ROM period at 16×16 tiles, 3.2 s for the counter's cycle); Section VII-A projects the on-GPU learning time. Where Golly is at its best, on idle circuits and small universes with little novelty, GHL hands the work to the host recursion, which is HashLife and runs at Golly's speed within a factor of two.

### C. Brute-force engines

**TABLE XII.** Brute-force kernels, in cell-updates/s (generations/s for single patterns)

| Grid | Byte-LUT kernel | Bit-plane, plain | Bit-plane, shared tile |
|---|---|---|---|
| One [counter], 37,627 cells | $1.0 \cdot 10^{10}$ (274,544)ᵃ | $1.0 \cdot 10^{10}$ (270,000–273,000) | **$2.9\text{–}3.0 \cdot 10^{10}$** (770,000–785,000; $k = 16$, tiles of 224 × 16 cells) |
| [`squares`], 274,950 cells | $4.9 \cdot 10^{10}$ (177,445) | $4.6\text{–}4.8 \cdot 10^{10}$ (169,000–175,000) | **$1.5 \cdot 10^{11}$** (556,000–563,000; $k = 16$, tiles of 192 × 48) |
| One [ROM][lotr1.ANSI-1M.rle], $9.5 \cdot 10^{6}$ cells, a full period | $1.8 \cdot 10^{11}$ (18,992) | **$3.9 \cdot 10^{11}$** (40,999) | $2.2 \cdot 10^{11}$ (23,000; $k = 4$) |
| Tiled, 18 M cells | $1.75 \cdot 10^{11}$ | | |
| Tiled, 169 M cells | $1.86 \cdot 10^{11}$ | | |
| Random states, $16384^{2}$ cells | $1.3 \cdot 10^{10}$ | **$3.0 \cdot 10^{11}$** | $2.5 \cdot 10^{11}$ ($k = 3$) |
| Random states, $5.4 \cdot 10^{8}$ cells | | $2.3 \cdot 10^{11}$ | |
| Random states, $1.07 \cdot 10^{9}$ cells | | $2.8 \cdot 10^{11}$ | |

ᵃ On the Hutton32b PNG counter of the same size. The tiled grids are copies of the [Hutton32 counter]. Each kernel pads a single pattern differently (the counter is stepped as 199×205 cells by the byte-LUT kernel and 224×205 by the bit-plane kernels); only the pattern's bounding box is counted. The largest bit-plane rows are steady-state rates over 256 and 512 generations; the first 32 generations after upload averaged $1.3\text{–}1.8 \cdot 10^{11}$.

On small grids every kernel waits for the dispatch: 3.65 µs, whatever it does. A further generation inside a shared tile costs about 1.1 µs, so sixteen take 20 µs instead of 58, which makes the shared-tile kernel 2.9× the plain kernel on the counter and 3.3× on [`squares`]. Tile shape matters by up to 1.6× ([`squares`] at $k = 16$: $9.4 \cdot 10^{10}$, $1.0 \cdot 10^{11}$ and $1.5 \cdot 10^{11}$ cell-updates/s, or 342 k, 373 k and 563 k gens/s, for tiles 16, 32 and 48 rows high). On large grids the same kernel loses: at $k = 1$ it reaches 55 % of the plain kernel, so the detour through shared memory costs more than the traffic it saves; at $16384^{2}$ it does $1.7 \cdot 10^{11}$ at $k = 1$, $2.5 \cdot 10^{11}$ at $k = 3$ and $1.8 \cdot 10^{11}$ at $k = 8$, and on the ROM $2.2 \cdot 10^{11}$ (23 k gens/s, $k = 4$). The crossover lies between $3 \cdot 10^{5}$ and $9.5 \cdot 10^{6}$ cells and was not located, which is why Algorithm 1 races the two kernels.

**TABLE XIII.** Large grids ($16384^{2}$ cells), same content for every kernel; 0 mismatching cells in every row

| Content | Byte LUT, 32 MiB | Byte LUT, 2 MiB | Bit-planes | Gain over the 32 MiB LUT |
|---|---|---|---|---|
| uniform random states, held constant | $1.3 \cdot 10^{10}$ | $4.2 \cdot 10^{10}$ | **$3.0 \cdot 10^{11}$** | 22× |
| tiled [counter] | $1.8 \cdot 10^{11}$ | $1.3 \cdot 10^{11}$ | **$2.9 \cdot 10^{11}$** | 1.6× |
| tiled [`chaotic`] | $1.8 \cdot 10^{11}$ | $1.3 \cdot 10^{11}$ | **$3.0 \cdot 10^{11}$** | 1.6× |
| tiled [`lotr1.ANSI-1M`] | $1.6 \cdot 10^{11}$ | $1.2 \cdot 10^{11}$ | **$3.0 \cdot 10^{11}$** | 1.8× |

The byte-LUT kernel depends on content because random states touch the whole $2^{25}$-byte table and miss L2. At $8192^{2}$ it does $1.8 \cdot 10^{11}$ on tiled [`chaotic`], $3.8 \cdot 10^{10}$ on decaying random states and $1.4 \cdot 10^{10}$ on random states held constant; the 2 MiB class-compressed table, which always fits L2, gives $1.3 \cdot 10^{11}$, $1.0 \cdot 10^{11}$ and $4.3 \cdot 10^{10}$. The bit-plane kernel has no table and no such dependence.

It is memory-bound. A cell-update reads and writes five bits (1.25 B); $2.97 \cdot 10^{11}$ × 1.25 B is 371 GB/s, the same 83 % of the bus as the byte kernel's $1.86 \cdot 10^{11}$ × 2 B, so the gain on real content is exactly the saving in bytes. A circuit that returns its input unchanged gives the same rate ($3.0 \cdot 10^{11}$ at $16384^{2}$, $2.8 \cdot 10^{11}$ at $32768^{2}$); cache-resident grids reach $4.3 \cdot 10^{11}$ at $4096^{2}$ (identity circuit: $3.8 \cdot 10^{11}$). Chaining $k$ dependent evaluations per invocation gives pass rates of 3.0, 2.4, 1.4 and $0.76 \cdot 10^{11}$ for $k = 1, 2, 4, 8$, so the gates alone sustain $5.4\text{–}6.1 \cdot 10^{11}$ evaluations/s, twice what memory lets through. Synthesis was stopped at 367 gates for that reason.

A negative result constrains the miss path. Stepping a missed window through all its generations inside one dispatch with one thread per window (Table XIV) is slower at every batch size than eight chained dispatches of the plain kernel on an atlas, by 21× where latency matters and by 6× where throughput does. One thread performs 240 circuit evaluations in sequence: on a 64×64 grid a dispatch with $k$ dependent evaluations takes 3.65, 3.66, 5.7, 5.7, 7.7 and 18.0 µs for $k = 1, 2, 4, 8, 16, 32$, about half a microsecond per evaluation even with all state in registers, and 2.6 µs with state in private arrays. Generations per dispatch pay only when a workgroup shares the work.

**TABLE XIV.** One thread per 32×32 window, 8 generations per dispatch, against 8 chained dispatches of the plain kernel

| Windows per dispatch | Window kernel, time per dispatch | Plain kernel, 8 chained dispatches |
|---|---|---|
| 1 | 619 µs | 29 µs |
| 64 | 623 µs | 31 µs |
| 1,024 | 627 µs | 60 µs |
| 4,096 | 639 µs | 115 µs |
| 16,384 | 2.8 ms | 0.5 ms |
| 262,144 | 41 ms (155 ns per window) | 7 ms (27 ns per window) |

The right column was measured for $64^{2}$, $1024^{2}$ and $2048^{2}$ cells (3.65, 7.5 and 14 µs per generation) and is 3.65 µs plus 27 ns per window for the rest.

### D. Recurring content: the hit path

Universe: $N \times N$ copies of [`lotr1.ANSI-1M`], each started at a different phase of its 528-generation period. The CPU reference engine learned the table over one period (2,462,688 entries at 16×16 tiles); the GPU ran sweeps; every tile was then compared with the CPU's prediction.

**TABLE XV.** Memo hit path on phase-shifted ROMs (`gpu-sweep`); every tile checked against the CPU

| Tile | Universe | Cells counted | Table | Memo entries | cell-updates/s (gens/s) | Generations to learn |
|---|---|---|---|---|---|---|
| 16×16 | 1 ROM | $9.5 \cdot 10^{6}$ | 256 MiB | | $4.4 \cdot 10^{12}$ (**468,929**) | 528 |
| 32×32 | 1 ROM | $9.5 \cdot 10^{6}$ | 64 MiB | | $1.5 \cdot 10^{13}$ (**1,607,022**) | 1,056 |
| 16×16 | 8×8 ROMs, 25,344 × 25,088 | $6.4 \cdot 10^{8}$ | 256 MiB | | $6.7 \cdot 10^{12}$ (10,585) | 528 |
| 16×16 | 32×32 ROMs, 101,376 × 100,352 | $1.02 \cdot 10^{10}$ | 256 MiB | 2,462,688 | $8.6\text{–}8.7 \cdot 10^{12}$ (848–858) | 528 |
| 32×32 | 32×32 ROMs | $1.02 \cdot 10^{10}$ | 64 MiB | 628,903 | $2.1 \cdot 10^{14}$ (20,349) | 1,056 |
| 64×64 | 32×32 ROMs | $1.02 \cdot 10^{10}$ | 16 MiB | 162,691 | $1.9 \cdot 10^{15}$ (191,061) | 2,112 |
| 128×128 | 32×32 ROMs | $1.02 \cdot 10^{10}$ | 4 MiB | 43,759 | $1.2 \cdot 10^{16}$ (1,149,321) | 4,224 |
| 256×256 | 32×32 ROMs | $1.02 \cdot 10^{10}$ | 1 MiB | 12,475 | **$6.8 \cdot 10^{16}$** (6,714,020) | 8,448 |

Larger tiles pad every copy to a multiple of the tile, so the universe they step grows to $1.06 \cdot 10^{10}$, $1.16 \cdot 10^{10}$, $1.36 \cdot 10^{10}$ and $1.83 \cdot 10^{10}$ cells; only the $1.02 \cdot 10^{10}$ cells of the 16×16-tile universe are counted, and the single ROM is counted over its bounding box. Probe rates were $2.3 \cdot 10^{9}$/s and $1.0 \cdot 10^{9}$/s for the single ROM at 16×16 and 32×32 tiles, $3.3 \cdot 10^{9}$/s and $4.2 \cdot 10^{9}$/s for the 8×8 and 32×32 universes at 16×16, and $1.3 \cdot 10^{10}$/s for the 32×32 universe at 32×32. The rows with 16×16 and 32×32 tiles found 0 wrong tiles of 38,808, 10,100, 2.5 M, 39.7 M and 10.3 M. The single ROM is limited by dispatch latency, not by probes. At $10^{10}$ cells the gain over the byte-LUT brute kernel is 46× at 16×16 and 1,110× at 32×32, as $V_L R_P / R_B$ predicts. Each further level is worth 6–24×: 8× from the tile-step and the rest from the table sliding into L2. The ID buffer shrinks from 151 MiB at 16×16 tiles to 1 MiB at 256×256.

When content does not recur, larger sweep tiles cost little. Table XVI counts hash operations per generation for a flat sweep at the given tile with the recursion below it.

**TABLE XVI.** Hash operations per generation by sweep tile (`flat`, model)

| Pattern | 8×8 | 16×16 | 32×32 | 64×64 | 128×128 | 256×256 |
|---|---|---|---|---|---|---|
| [`advanced counter v2`] | 227 | 63 | 40 | 36 | 36 | 36 |
| [`serial display`], idle | 66,781 | 8,634 | 1,277 | 336 | 212 | 196 |
| [`32x32 movie`], playing | 67,149 | 9,171 | 1,907 | 1,002 | 887 | 872 |
| [`chaotic`] | 272 | 232 | 233 | 234 | 234 | 234 |
| [ROM][lotr1.PPMd-244K.rle], first 528 gens | 88,837 | 92,164 | 92,590 | 95,132 | 95,139 | 95,140 |

Empty and idle regions collapse, active logic saturates near 32–64 cells, and novel content gains nothing but loses little.

### E. Sparse universes

Universe: 2304 copies of [`advanced counter v2`], one per 2048 × 2176-cell block, each at a different phase of its period: 98,304 × 104,448 cells, $5.5 \cdot 10^{7}$ live cells, 0.79 % of tiles non-empty. The same universe was exported as a macrocell file and run by Golly and by the CPU engine.

**TABLE XVII.** Sparse 98,304 × 104,448 universe, 16,384 generations

| Engine | Time | cell-updates/s (gens/s) | Final state |
|---|---|---|---|
| Golly HashLife, step 16 | 15.95 s | $1.1 \cdot 10^{13}$ (1,027) | |
| Golly HashLife, step 1024 | 3.88 s | $4.3 \cdot 10^{13}$ (4,224) | population 55,478,381, hash `144be5c0cde9cb81` |
| Golly HashLife, step 16,384 | 3.79 s | $4.4 \cdot 10^{13}$ (4,323) | |
| CPU reference engine, step 1024 | 2.78 s | $6.1 \cdot 10^{13}$ (5,894) | identical |
| GPU, flat sweep (all 40 M tiles) | 1.81 s | $9.3 \cdot 10^{13}$ (9,059) | 0 of 40.1 M tiles wrong |
| **GHL, GPU active-block sweep** | 0.059 s | **$2.9 \cdot 10^{15}$** (278,648) | identical to Golly's |

Rates count all $1.03 \cdot 10^{10}$ cells of the universe, of which $5.5 \cdot 10^{7}$ are live; they are effective rates, not work done. Over $2^{20}$ generations Golly averages $6.4 \cdot 10^{14}$ (61,878 gens/s, step 1024) to $9.9 \cdot 10^{14}$ (96,581 gens/s, step 65,536) as its memo warms. The GPU figure is a steady-state hit-path rate with the counter's cycle already learned (3.2 s on the CPU).

### F. The recursion against Golly on the CPU

Table XVIII runs GHL's recursion (Algorithm 2 with a 4×4 floor) on one CPU thread against Golly at step 1024 (16,384 for the replicator). The engine is unoptimized: a general-purpose `HashMap` per level and a sort per wave, against Golly's hand-rolled chained table with prefetching.

**TABLE XVIII.** Same pattern, same rule, one thread

| Pattern | Gens | Golly (s) | Engine (s) | Same final state |
|---|---|---|---|---|
| [Segmented ROM (256k)] | $2^{24}$ | 1.83 | 1.55 | yes |
| [serial display (32×32)] | $2^{24}$ | 0.16 | 0.28 | yes |
| [parallel display (16×16)] | $2^{24}$ | 0.019 | 0.010 | yes |
| [16×16 movie] | $2^{24}$ | 3.48 | 3.17 | yes |
| [efficient demultiplexer] | $2^{24}$ | 0.024 | 0.030 | yes |
| [cpu] | $2^{24}$ | 0.017 | 0.020 | yes |
| [BIN2BCD] | $2^{24}$ | 0.047 | 0.060 | yes |
| [advanced counter v2] | $2^{24}$ | 1.73 | 1.50 | yes |
| [chaotic] | $2^{24}$ | 5.27 | 7.95 | yes |
| [32×32 movie] | $2^{24}$ | 52.5 | 55.6 | yes |
| [squares] | $2^{20}$ | 80.8 | 134.4 | yes |
| [Hutton replicator] (step 16,384) | 262,144 | 111.5 | 219.2 | yes |

The engine runs at 0.51–1.9× Golly's speed and is faster on four of the twelve. It is the same algorithm doing a comparable number of node operations; the gap is table engineering. On the replicator it counts $1.45 \cdot 10^{9}$ node probes and 262 k waves for 262,144 generations, and creates 148 M nodes, an estimated 7 GB, because it never reclaims.

### G. Choosing floor, engine and depth

These results count the reference engine's operations and price them with the constants of Table I: 0.27 ns per find, 1 ns per insert, 5.4 ps per cell evaluation, every brute-forced window stepped whole. Every policy was checked to produce the same universe.

**TABLE XIX.** Model time for fixed floors ("F16": missed 16×16 windows brute-forced, everything above memoized) and for plain cells per region under a memo of region-sized windows

| Content, horizon | Plain cells, ideal | F8 | F16 | F32 | F64 | F128 | F256 | Plain cells + coarse memo |
|---|---|---|---|---|---|---|---|---|
| [counter], 16,384 gens (4×4 base tiles) | 5.8 ms | **0.77** | 0.89 | 1.91 | 4.68 | 11.4 | 33.9 | 10.5 |
| [counter] (8×8 base tiles) | 5.8 | — | **0.62** | 1.33 | | | | 9.4 |
| 8 perturbed [`chaotic`], 8,192 gens (4×4) | 23.1 | **5.98** | 7.29 | 20.2 | 47.6 | 86.2 | 148 | 41.8 |
| same (8×8) | 23.1 | — | **5.21** | 14.0 | | | | |
| [ROM][lotr1.PPMd-244K.rle], first 512 gens, all new (4×4) | 7.6 | 86.9 | 39.4 | 46.5 | | | | **10.0** |
| same (8×8) | 7.6 | — | 24.5 | 31.7 | | | | **8.6** |

"Ideal" is every non-empty region stepped as cells with no halo; the last column includes the halo and the conversion to IDs. With 4×4 base tiles F8 is 15 % ahead on logic and 2.2× behind on new content; 8×8 base tiles with F16 beat both.

**TABLE XX.** Engine per content: model time (ms) and memory (MiB) over 4,096 generations, 8×8 base tiles. F = memoize everything, brute-force missed 16×16 or 32×32 windows; C = plain cells per 256×256 region with memo only above it, at 64- or 16-generation steps

| Content | F16 | F32 | C64 | C16 | Best vs worst |
|---|---|---|---|---|---|
| ROM [`lotr1.PPMd`] | 25.5 ms, 201 MiB | 33.5, 94 | 35.6, 63 | **9.8, 63** | C16 2.6× ahead of F16 |
| [`chaotic`] (perturbed) | **0.60**, 3.0 | 1.29, 1.6 | 2.36, 0.3 | 1.97, 0.7 | F16 3.3× |
| [counter] | **0.25**, 1.3 | 0.44, 0.6 | 4.24, 0.1 | 3.84, 0.3 | F16 15× |
| [`32x32 movie`] | **4.56**, 24.6 | 8.34, 12.8 | 30.1, 4.9 | 30.5, 6.1 | F16 6.6× |
| [`squares`] | **0.56**, 2.8 | 1.13, 1.5 | 10.7, 0.3 | 10.0, 0.7 | F16 18× |
| [`serial display`] | **2.37**, 10.5 | 6.14, 6.9 | 35.0, 1.3 | 36.7, 2.7 | F16 15× |
| [`Segmented ROM`] | **0.75**, 3.4 | 1.84, 2.6 | 21.0, 0.9 | 21.4, 1.3 | F16 28× |
| [`16x16 movie`] | **0.52**, 2.5 | 1.12, 1.5 | 38.4, 0.3 | 39.9, 0.9 | F16 74× |
| All eight | 35.1 | 53.8 | 177 | 154 | best per content: **19.4** |

F16 is the safe default: its worst case is 2.6× behind, while plain cells are up to 74× behind because one moving signal forces a whole 256×256 region to be stepped. The step matters as much as the engine: the ROM learned through 64-generation steps needs 2,112 generations (35.6 ms), through 16-generation steps 528 (9.8 ms). Choosing per content is worth 1.8× on this mix, all of it from the ROM. At the bit-plane kernel's rate ($2.9 \cdot 10^{11}$) no ranking changes: uniform F16 costs 29.1 ms instead of 35.1 and the best engine per content 15.2 instead of 19.4.

**TABLE XXI.** Algorithm 3 against uniform F16 (8×8 base tiles, 16-generation steps; model)

| Universe | Uniform F16 | GHL controller |
|---|---|---|
| [Mixed]: ROM, 6 [`chaotic`], 6 counters, movie, [`squares`], display; 4.9 M live cells, 4,096 gens | 35.0 ms, 257 MiB | **32.6 ms, 147 MiB** |
| [ROM][lotr1.PPMd-244K.rle] alone, 2,112 gens | 24.7 ms, 200 MiB | **13.2 ms, 97 MiB** |
| [counter], 16,384 gens | 0.67 ms | 0.67 ms (never leaves F16) |
| 8 perturbed [`chaotic`], 8,192 gens | 5.42 ms | 5.42 ms (never leaves F16) |

Three earlier controllers did not work (64-generation steps, 4×4 base tiles). Walking down one level at a time from plain cells cost 97.8 ms on the ROM's learning phase against 38.6 ms for plain cells and 40.4 ms for F16 throughout: the first step down is a 256×256 brute window, the most expensive mode there is, so the experiment always failed. Whole-step trials of the next finer level cost 64.1 ms on the mixed universe against 55.1 ms for uniform F16 and 46.8 ms for the best fixed mode per region. Sampled evidence with 64-generation steps cost 78.5 ms, because plain cells learn the ROM four times slower at that step and mode boundaries cost a halo that a contiguous block would not.

**TABLE XXII.** Dispatch time per generation of each engine (fixed-capacity dispatches at 3.4 µs), independent of the universe's size, and the active area at which each becomes the best engine

| Engine | Generations per floor wave | Dispatches per generation | Dispatch time per generation | Best for, byte-LUT pricing | Best for, bit-plane pricing |
|---|---|---|---|---|---|
| Flat sweep, hits only, 16×16 tiles | 8 | 0.125 | 0.42 µs | | |
| Plain cells | 1 | ≈1.1 | 3.7 µs | below ≈ 0.7 M active cells | below ≈ 1.0 M |
| F32 | 8 | ≈2.5 | 8.5 µs | 0.7–8 M | 1.0–13 M |
| F16 | 4 | ≈4 | 13.6 µs | above ≈ 8 M | above ≈ 13 M |
| F8 | 2 | ≈7 | 23.8 µs | | |

The thresholds use the counter's measured work per region and generation: 38 ns at F16, 81 ns at F32 and 575 ns as isolated plain cells, or 33, 59 and 377 ns at the bit-plane rate. The shared-tile kernel cuts the plain-cells dispatch time from 3.7 µs to 1.3–1.8 µs per generation on small grids (sixteen generations in one dispatch of about 20 µs); the other rows and the thresholds were not recomputed for it. The recursion itself has not run on the GPU; its rows are dispatch counts priced at the measured 3.4 µs.

### H. CPU kernels

The circuit also changes the host side. Table XXIII gives one i5-6600K thread ([`bench/`]; all engines compared cell by cell before timing; ranges span two sets of runs, which differed by up to 30 %).

**TABLE XXIII.** Cell-updates per second on one CPU thread

| Engine | Uniform random states | Tiled counter |
|---|---|---|
| byte LUT, 32 MiB | $4.8\text{–}7.2 \cdot 10^{7}$ | $3.3\text{–}6.0 \cdot 10^{8}$ |
| form A, scalar circuit | $1.5\text{–}2.5 \cdot 10^{7}$ | $1.4\text{–}2.5 \cdot 10^{7}$ |
| form C, five loads from 15 KB | $3.2\text{–}4.5 \cdot 10^{8}$ | $2.6\text{–}4.4 \cdot 10^{8}$ |
| form B, `u64` | $0.9\text{–}1.7 \cdot 10^{9}$ ($2.3\text{–}2.8 \cdot 10^{9}$ on long rows with `target-cpu=native`) | $0.9\text{–}1.7 \cdot 10^{9}$ ($2.2\text{–}2.8 \cdot 10^{9}$) |
| form B, 256-bit AVX2 | $3.0\text{–}4.5 \cdot 10^{9}$ | $2.7\text{–}4.4 \cdot 10^{9}$ |

Bit-planes are 2.0–3.5× the table on real content in portable `u64` code and 6–8× with AVX2, and 17–73× on random states. A core's brute-force rate rises to $0.9\text{–}4.5 \cdot 10^{9}$ while an out-of-cache probe stays near $10^{7}$/s, so the host's $\rho$ moves from 30–60 to a few hundred. By (3) the host recursion's floor then belongs at 16×16 tiles as well. That is an inference: the engine of Table XVIII uses a 4×4 floor and scalar evaluation and was not rebuilt.

## VII. Discussion

### A. Projected miss path

The on-GPU miss path is a composition of measured parts. Per tile-step at 16×16 tiles,

$$t = \frac{1}{R_P} + \mu \left( \frac{4 V_L}{R_B} + \frac{2}{R_I} + t_\text{gather} \right) \approx 0.27\ \text{ns} + \mu \cdot 50\ \text{ns} \qquad (7)$$

**TABLE XXIV.** Projected rates of memoized stepping with the miss path on the GPU (byte-LUT pricing)

| Workload | $\mu$ at 16×16 | Projected cell-updates/s | vs Golly | vs byte-LUT brute force |
|---|---|---|---|---|
| [ROM][lotr1.ANSI-1M.rle] after its first period | 0 | $8.6 \cdot 10^{12}$ (measured, Table XV) | $3.6 \cdot 10^{5}$× | 46× |
| [Segmented ROM], movies, component sheet | 0.07–0.16 % | $6\text{–}7 \cdot 10^{12}$ | | 32–38× |
| Counter array, non-repeating | 1.7 % | $1.8 \cdot 10^{12}$ | 80× | 9.7× |
| [`squares`]-like displays | 3.5 % | $1.0 \cdot 10^{12}$ | 120–270× | 5.4× |
| [`chaotic`]-like | 25 % | $1.6 \cdot 10^{11}$, so brute force: $1.8 \cdot 10^{11}$ | 180× | 1× |
| [ROM][lotr1.ANSI-1M.rle] during its first period | 96 % | $4 \cdot 10^{10}$ for 0.12 s | | 0.2× |

The Golly column compares with Golly's rate on a single instance of the pattern (Table X), the last column with the byte-LUT kernel's $1.86 \cdot 10^{11}$. The projections hold when a sweep contains enough tiles to hide dispatch latency, roughly $10^{5}$ tiles or $2.6 \cdot 10^{7}$ cells. Below that a sweep with misses costs about 54 µs per 8 generations, 6.8 µs per generation against 3.65 µs for one dispatch of the plain kernel, which is why Algorithm 1 does not memoize small universes. A shared tile on the atlas would take a wave from about 54 µs to about 37 µs; that was not run. On the GPU, learning one ROM period would take 0.1–0.5 s instead of the CPU's 80–90 s, and the replicator would take 0.4 s of probes plus 5–15 s of wave latency against Golly's 111 s. Batches that overflow the atlas, table rebuilds and the cost of activating blocks are not modelled.

### B. Where the margin is thin

The closest case is a small universe with constant novelty. On the single counter Golly does $2.2 \cdot 10^{10}$ cell-updates/s ($5.8 \cdot 10^{5}$ gens/s), the CPU engine $2.4 \cdot 10^{10}$ ($6.5 \cdot 10^{5}$), the plain GPU kernel $1.0 \cdot 10^{10}$ ($2.7 \cdot 10^{5}$) and the shared-tile kernel $2.9 \cdot 10^{10}$ ($7.8 \cdot 10^{5}$): a 1.35× lead that rests on one machine. Below about $10^{6}$ active cells every GPU engine waits for the dispatch, and the host recursion stays in Algorithm 1 for that reason.

### C. Limitations

* **Not one program.** The GPU results for memoized stepping are hit-path measurements on tables learned by the CPU, on periodic patterns replicated with phase offsets. They establish steady-state rates and correctness, not learning speed on the GPU. The fixed-capacity wave was measured empty; overflow and replay are designed, not built. The racing of executors, table rebuild and cycle detection are specified only.
* **Section VI-G is simulation.** Floors, engines and depth are costed from the reference engine's operation counts; the plain-cells mode is costed, not executed as cells, and the controller was tuned on the same composite it is reported on.
* **One machine, two rules.** All constants come from one RTX 5060 Ti under wgpu on Windows. Pass, submit and read-back costs varied by 2–20× between runs of the same benchmark; chained-dispatch and fixed-capacity figures were stable to 10 %. GPU throughput is sensitive to other load: a run of the kernel comparison taken during a build showed the byte-LUT kernel up to 48 % low, and the tables report repeats on an idle machine. On an 8 GB card of the same bus the brute and out-of-cache probe rates should be similar and memo capacity halves; this was not measured.
* **Miss rates are pattern statistics** measured on fourteen patterns over $10^{4}\text{–}10^{6}$ generations. A circuit whose state space is explored more slowly would keep missing longer.
* **The circuit** exists for Hutton32b only and is not proven minimal. Hutton32 differs from it in 74,854 neighbourhoods; that edit has not been made, and Hutton32 patterns use the byte-LUT kernel. The bit-plane kernel was measured up to $1.07 \cdot 10^{9}$ cells; one binding holds 2 GiB, so a $10^{10}$-cell universe as plain cells (6.3 GB per buffer) would need a kernel spanning several bindings.
* **Baselines.** Golly figures are `bgolly` without rendering at the best step size tried; interactive figures will be lower. The reference engine is unoptimized and never reclaims nodes.
* **Literature.** Sections II and VII rest on a survey run for this paper; sources marked † were read only as abstracts.

### D. Generality

Nothing in GHL above the brute-force engine is specific to Hutton32. Any radius-1 rule with a kernel (a table, a rule tree or a circuit) fits below the floor; only the hand-written starting network of the circuit is rule-specific, and the optimiser works from any $2^{25}$-byte table. For a rule without a circuit the byte-LUT kernel is the fallback, at $1.86 \cdot 10^{11}$ on real content.

## VIII. Conclusion

HashLife's weakness on dense content is the grain of its memoization, not memoization. On a GPU a probe costs as much as about fifty cell-updates of brute force, so memoizing below 16×16-cell windows loses and memoizing above them wins by up to $V_L / \rho$. GHL combines a memo floor at that grain, a 367-gate bit-sliced engine below it, a level-synchronous recursion and a positional sweep above it, and a choice of engine and depth driven by novelty and active area. Its measured components beat Golly 5.1 on every workload measured on the GPU, from 1.3× on a small active counter to $6.4 \cdot 10^{5}$× on a recurring ROM, and its recursion matches Golly within a factor of two on the CPU with identical final states. The remaining step is to assemble the on-GPU miss path, whose primitives and cost are measured here.

## Acknowledgment

*[To be completed by the authors. Funding: not stated. IEEE requires disclosure of AI-generated content: the authors must state where and how an AI assistant was used for this text and for the measurement harness, and confirm their review of it.]*

## References

Sources marked † were read as abstract or secondary summary only. Text in square brackets describes a source whose exact title was not recorded; those entries must be checked against the source before submission.

[1] R. W. Gosper, "Exploiting regularities in large cellular spaces," *Physica D*, vol. 10, 1984.  
[2] The Golly Gang, Golly 5.1, branch `master`, commit 53bdb74e48cd0792f87a2e3e6a5bb0c5d7d5af0e (2026-11-15); source files [`ghashbase.h`], [`ghashbase.cpp`], [`hlifealgo.cpp`], [`ruletable_algo.cpp`], [`jvnalgo.cpp`]. [Online]. Available: https://sourceforge.net/p/golly/code/ci/53bdb74e48cd0792f87a2e3e6a5bb0c5d7d5af0e/tree/  
[3] T. Hutton, "Hutton32.table," Golly rule table repository. [Online]. Available: http://gollygang.github.io/ruletablerepository/downloads/Hutton32.table  
[4] "Hutton32b.rule," Hutton32-Computer repository. [Online]. Available: https://github.com/FredericaBernkastel/Hutton32-Computer/blob/master/doc/Hutton32b.rule  
[5] A. P. Goucher, "lifelib." [Online]. Available: https://gitlab.com/apgoucher/lifelib  
[6] T. Rokicki, "lockstep hashlife," golly-test mailing list, 2014. [Online]. Available: https://sourceforge.net/p/golly/mailman/message/32515364/  
[7] T. Rokicki, "Life algorithms," G4G13, 2018. [Online]. Available: https://www.gathering4gardner.org/g4g13gift/math/RokickiTomas-GiftExchange-LifeAlgorithms-G4G13.pdf  
[8] C. E. LaForest, "HashLife on GPU," 2010. [Online]. Available: https://fpgacpu.ca/writings/HashLife-GPU.pdf  
[9] das67333, "gol_engines," 2026. [Online]. Available: https://github.com/das67333/gol_engines  
[10] T. Fujita, K. Nakano, and Y. Ito, [bit-sliced Life on the GPU], *Int. J. Found. Comput. Sci.*, vol. 27, 2016, doi: 10.1142/S0129054116500404.  
[11] Dijkstra and Brouwer, [bit-packed Life in CUDA], 2023. [Online]. Available: https://binary-banter.github.io/game-of-life  
[12] E. Frankling, "auto," [bit-plane and multi-state CA kernels in wgpu], 2026. [Online]. Available: https://erikfrankling.github.io/auto/  
[13] J. Signorini, [29-state von Neumann CA on a SIMD machine], in *Proc. Supercomputing ’89*. [Online]. Available: https://dl.acm.org/doi/10.1145/76263.76282 †  
[14] M. Awad *et al.*, "Better GPU hash tables," arXiv:2108.07232.  
[15] D. Jünger *et al.*, "WarpCore," arXiv:2009.07914.  
[16] McCoy and Pandey, "WarpSpeed," arXiv:2509.16407.  
[17] Polak *et al.*, "Hive," arXiv:2510.15095.  
[18] Z. Jia *et al.*, [GPU microarchitecture and TLB measurements], arXiv:1903.07486.  
[19] Maczan, [WebGPU dispatch overhead measurements], 2026, arXiv:2604.02344.  
[20] S. Rafler, [SmoothLife], arXiv:1111.1567.  
[21] B. Chan, [Lenia], arXiv:1812.05433.  
[22] [FFT versus direct convolution for Life, Julia benchmark], 2021. [Online]. Available: https://rivesunder.github.io/SortaSota/2021/09/27/faster_life_julia.html  
[23] [Bit-packed CUDA versus PyTorch Life], 2025. [Online]. Available: https://www.boristhebrave.com/2025/09/11/accelerated-game-of-life-with-cuda-triton/  
[24] A. D. Robison, "Fast computation of additive cellular automata," *Complex Systems*, vol. 1, 1987. [Online]. Available: https://content.wolfram.com/sites/13/2018/02/01-1-15.pdf  
[25] C. Moore, "Predicting non-linear cellular automata quickly by decomposing them into linear ones," arXiv:patt-sol/9701008.  
[26] D. Griffeath and C. Moore, "Life without Death is P-complete," *Complex Systems*, vol. 10, 1996. [Online]. Available: https://content.wolfram.com/sites/13/2018/02/10-6-3.pdf  
[27] N. Israeli and N. Goldenfeld, "Coarse-graining of cellular automata," arXiv:nlin/0508033.

## Appendix A. Reproducing the Results

Everything the tables need is in the repository: the harness in [`research/`], every pattern in [`patterns/`] (sources and rules in its README) and the two canonical tables in [`rules/`]. Each table is one command, run in bash (git-bash on Windows, WSL or Linux) from [`research/`]. [`reproduce.sh`] builds the harness first; run with no argument it lists the targets, and its source holds the exact harness commands behind each one. It needs Rust, Python 3 and a GPU with Vulkan or DX12. The Golly targets also need `bgolly`, which is taken from `$BGOLLY`, from Golly 5.1's Windows install path or from `PATH`, with [`rules/`] as its rule folder. GPU figures should be taken on an otherwise idle machine.

**TABLE XXV.** One command per table

| Table or claim | Command | Needs |
|---|---|---|
| Table I: $R_B$ (byte LUT), workgroup shapes, DX12 | `bash reproduce.sh lut-kernel` | |
| Table I: $R_P$, $R_I$ | `bash reproduce.sh probe` | |
| Table I: latency and the miss-path wave | `bash reproduce.sh latency` | |
| Table IV | `bash reproduce.sh miss-rates` | |
| Table V | `bash reproduce.sh cond-miss` | |
| Table VII: forms A and C | `bash reproduce.sh forms` | network (numpy) |
| Table VII, Section IV-C: the circuit against the table | `bash reproduce.sh circuit` | network (numpy) |
| Section V-C, Table XXVI: reference engine against per-cell stepping; CPU rules against both tables; the two rules compared | `bash reproduce.sh correctness` | |
| Table X | `bash reproduce.sh golly` | Golly |
| Table X: the ROM on one CPU core and the byte-LUT kernel | `bash reproduce.sh rom-kernels` | |
| Table XII: byte-LUT kernel | `bash reproduce.sh lut-kernel rom-kernels` | |
| Table XII: bit-plane kernel | `bash reproduce.sh bitplane-sizes` | |
| Table XII: shared tile | `bash reproduce.sh tile` | |
| Table XIII | `bash reproduce.sh bitplane-content` | |
| Section VI-C: the byte LUT at $8192^{2}$ | `bash reproduce.sh lut-content` | |
| Section VI-C: identity circuit, chained evaluations | `bash reproduce.sh bitplane-limit` | |
| Table XIV and the serial cost | `bash reproduce.sh window` | |
| Table XV: 16×16 and 32×32 tiles | `bash reproduce.sh sweep` | |
| Table XV: 64×64 to 256×256 tiles | `bash reproduce.sh sweep-tiles` | |
| Table XVI | `bash reproduce.sh sweep-ops` | |
| Table XVII | `bash reproduce.sh sparse` | Golly |
| Table XVIII | `bash reproduce.sh head2head` | Golly |
| Section VI-F: batch statistics of the replicator | `bash reproduce.sh batches` | |
| Table XIX | `bash reproduce.sh floors` | |
| Table XX | `bash reproduce.sh engine-choice` | |
| Table XXI | `bash reproduce.sh controller` | |
| Table XXII: work per generation of the counter | `bash reproduce.sh counter-work` | |
| Tables XX and XXII at the bit-plane rate | `bash reproduce.sh reprice` | |
| Table XXIII | `bash reproduce.sh cpu-bench` | |
| Appendix B: the committed tables are the published files | `bash reproduce.sh rules-upstream` | network |

`sweep` and `sweep-tiles` learn the ROM's table on the CPU before each GPU run, about 90 s per period at 16×16 tiles. Tables II, III, VIII, IX, XI and XXIV derive from or restate the others; Table VI and Algorithms 1–3 come from the design and the code. The circuit is regenerated in [`bitslice/`] with [`rule_circuit.py`], [`optimize.py`] `25 1` (25 minutes), [`optimize_more.py`] `6.5 7` and [`emit.py`], which also need `pip install yowasp-yosys python-sat`; the optimiser runs on a wall-clock budget, so a rerun does not reproduce the netlist gate for gate, and [`circuit.netlist`] is the artifact. The mixed universe of Table XXI is the `--compose` string in [`composite_spec.txt`] (items `file@x,y[+pre-run gens][~perturbation seed]`). The three controllers that Section VI-G reports as failed were replaced in [`cmd_adapt.rs`] and cannot be rerun.

## Appendix B. The Rule Tables

Each rule has one authoritative table (Table XXVI). The harness's CPU rule implementation and the circuit of Section IV-C equal these tables on all $2^{25}$ neighbourhoods; the circuit was built and checked against the Hutton32b table. The two rules differ in 74,854 neighbourhoods, all with an excited special transmission state, a confluent state or a crossing in the centre.

**TABLE XXVI.** Canonical rule tables

| Rule | Canonical table | Verified identical to it: 0 of $2^{25}$ neighbourhoods differ |
|---|---|---|
| **Hutton32** | [`Hutton32.table`] in Golly's rule table repository [3] | Golly's built-in Hutton32: the harness's transcription of [`jvnalgo.cpp`] [2] with C integer semantics, which also reproduces `bgolly` on [`squares`] and on the Hutton replicator |
| **Hutton32b** (a 3×3 XNOR gate and two added rule sets) | [`Hutton32b.rule`][Hutton32b.rule 2] in the Hutton32-Computer repository [4] | the copy in Golly's `Rules` folder, which differs from it only by one blank line and its line endings and was the copy `bgolly` loaded for the measurements; [`rules/`] holds the published file |

## Appendix C. Harness Map

| File | Contents |
|---|---|
| [`reproduce.sh`] | Appendix A: one target per table, with the exact harness commands behind it |
| [`rule.rs`], [`rulespec.rs`] | CPU implementations of both rules; loader for Golly tables; rule selection |
| [`pattern.rs`], [`mc.rs`] | palette PNG, extended RLE, macrocell reader |
| [`sim.rs`] | per-cell table stepping (reference semantics) |
| [`hl.rs`] | the reference engine: hash-consed levels, memo floor, level-synchronous `solve`, root stepping, fingerprints |
| [`flat.rs`] | positional sweep of the tile automaton on the CPU |
| [`cmd_stats.rs`] | `selftest`, `flat`, `root`, `fingerprint`, `rulediff`, `ruletree` |
| [`cmd_gpu.rs`] | `gpu-brute`, `gpu-sweep`, `gpu-probe`, `gpu-latency` |
| [`cmd_brute2.rs`] | hostile content; class-compressed table |
| [`cmd_insert.rs`] | lock-free find-or-insert with 32-bit CAS |
| [`cmd_sparse.rs`] | sparse universe, active-block sweep, macrocell writer |
| [`cmd_adapt.rs`] | per-region floors and engines on the position-aware recursion ([`hl.rs: result_at`]), cost model, controller, `--compose` universes |
| [`cmd_bitslice.rs`] | the plain bit-plane kernel (`gpu-bitslice`), the shared-tile kernel (`gpu-tile`) and the one-thread-per-window kernel (`gpu-window`) |
| [`bitslice/`] | the rule as a gate network: [`rule_circuit.py`], [`resub.py`], [`abc_flow.py`], [`xag.py`], [`optimize.py`], [`emit.py`]; artifacts [`circuit.netlist`], [`step_planes.wgsl`], [`step_planes.rs`]; [`verify.py`]; [`scalar.py`] and [`tables.py`] (forms A and C); [`bench/`] (CPU benchmark) |
| [`bitslice_series.sh`] | every GPU measurement of the bit-plane, window and shared-tile kernels |
| [`time_golly.py`], [`head2head.py`], [`engine_choice.sh`] | `bgolly` timing; head-to-head with a final-state check; engine-per-content table |
| [`patterns/`] | every pattern the paper uses, with its source, rule and the targets that use it |
| [`rules/`] | [`Hutton32.table`] and [`Hutton32b.rule`][Hutton32b.rule], byte for byte as published; also `bgolly`'s rule folder |

<!-- link targets -->
[`ghashbase`]: https://sourceforge.net/p/golly/code/ci/53bdb74e48cd0792f87a2e3e6a5bb0c5d7d5af0e/tree/gollybase/ghashbase.h
[`lotr1.ANSI-1M`]: research/patterns/misc/extra/lotr1.ANSI-1M.rle
[`advanced counter v2`]: research/patterns/advanced%20counter%20v2.rle
[`32x32 movie`]: research/patterns/32x32%20movie/32x32%20movie.mc
[`16x16 movie`]: research/patterns/16x16%20movie/16x16%20movie.mc
[`Segmented ROM (256k)`]: research/patterns/misc/Segmented%20ROM%20%28256k%29.mc
[`component-basis`]: research/patterns/component-basis.rle
[`squares`]: research/patterns/misc/squares.rle
[`chaotic`]: research/patterns/misc/chaotic.rle
[`lotr1.*`]: research/patterns/misc/extra/
[serial]: research/patterns/misc/serial%20display%20%2832x32%29.mc
[parallel]: research/patterns/misc/parallel%20display%20%2816x16%29.mc
[demultiplexer]: research/patterns/misc/efficient_demultiplexer.mc
[`cpu`]: research/patterns/misc/weird%20designs/cpu.mc
[`BIN2BCD`]: research/patterns/misc/BIN2BCD.rle
[lotr1.PPMd-244K.rle]: research/patterns/misc/extra/lotr1.PPMd-244K.rle
[counter]: research/patterns/advanced%20counter%20v2.rle
[Hutton32b.rule]: research/rules/Hutton32b.rule
[`Hutton32.table`]: research/rules/Hutton32.table
[Hutton32b counter]: research/patterns/presets/hutton32b_8bit_counter.png
[Hutton32 counter]: research/patterns/presets/hutton32_8bit_counter_old.png
[`README.md`]: research/patterns/README.md
[`lotr1.PPMd-244K`]: research/patterns/misc/extra/lotr1.PPMd-244K.rle
[segmented ROM]: research/patterns/misc/Segmented%20ROM%20%28256k%29.mc
[16×16]: research/patterns/16x16%20movie/16x16%20movie.mc
[32×32]: research/patterns/32x32%20movie/32x32%20movie.mc
[CPU]: research/patterns/misc/weird%20designs/cpu.mc
[BCD converter]: research/patterns/misc/BIN2BCD.rle
[component sheet]: research/patterns/component-basis.rle
[replicator]: research/patterns/replicators/Hutton-replicator.rle
[`hl.rs`]: research/src/hl.rs
[Hutton replicator]: research/patterns/replicators/Hutton-replicator.rle
[lotr1.ANSI-1M.rle]: research/patterns/misc/extra/lotr1.ANSI-1M.rle
[ROMs]: research/patterns/misc/extra/lotr1.ANSI-1M.rle
[counters]: research/patterns/advanced%20counter%20v2.rle
[Hutton32b PNG counter]: research/patterns/presets/hutton32b_8bit_counter.png
[`serial display`]: research/patterns/misc/serial%20display%20%2832x32%29.mc
[Segmented ROM (256k)]: research/patterns/misc/Segmented%20ROM%20%28256k%29.mc
[serial display (32×32)]: research/patterns/misc/serial%20display%20%2832x32%29.mc
[parallel display (16×16)]: research/patterns/misc/parallel%20display%20%2816x16%29.mc
[16×16 movie]: research/patterns/16x16%20movie/16x16%20movie.mc
[efficient demultiplexer]: research/patterns/misc/efficient_demultiplexer.mc
[BIN2BCD]: research/patterns/misc/BIN2BCD.rle
[advanced counter v2]: research/patterns/advanced%20counter%20v2.rle
[chaotic]: research/patterns/misc/chaotic.rle
[32×32 movie]: research/patterns/32x32%20movie/32x32%20movie.mc
[squares]: research/patterns/misc/squares.rle
[`lotr1.PPMd`]: research/patterns/misc/extra/lotr1.PPMd-244K.rle
[`Segmented ROM`]: research/patterns/misc/Segmented%20ROM%20%28256k%29.mc
[Mixed]: research/tools/composite_spec.txt
[`bench/`]: research/bitslice/bench/
[`ghashbase.h`]: https://sourceforge.net/p/golly/code/ci/53bdb74e48cd0792f87a2e3e6a5bb0c5d7d5af0e/tree/gollybase/ghashbase.h
[`ghashbase.cpp`]: https://sourceforge.net/p/golly/code/ci/53bdb74e48cd0792f87a2e3e6a5bb0c5d7d5af0e/tree/gollybase/ghashbase.cpp
[`hlifealgo.cpp`]: https://sourceforge.net/p/golly/code/ci/53bdb74e48cd0792f87a2e3e6a5bb0c5d7d5af0e/tree/gollybase/hlifealgo.cpp
[`ruletable_algo.cpp`]: https://sourceforge.net/p/golly/code/ci/53bdb74e48cd0792f87a2e3e6a5bb0c5d7d5af0e/tree/gollybase/ruletable_algo.cpp
[`jvnalgo.cpp`]: https://sourceforge.net/p/golly/code/ci/53bdb74e48cd0792f87a2e3e6a5bb0c5d7d5af0e/tree/gollybase/jvnalgo.cpp
[`research/`]: research/
[`patterns/`]: research/patterns/
[`rules/`]: research/rules/
[`reproduce.sh`]: research/reproduce.sh
[`bitslice/`]: research/bitslice/
[`rule_circuit.py`]: research/bitslice/rule_circuit.py
[`optimize.py`]: research/bitslice/optimize.py
[`optimize_more.py`]: research/bitslice/optimize_more.py
[`emit.py`]: research/bitslice/emit.py
[`circuit.netlist`]: research/bitslice/circuit.netlist
[`composite_spec.txt`]: research/tools/composite_spec.txt
[`cmd_adapt.rs`]: research/src/cmd_adapt.rs
[Hutton32b.rule 2]: https://github.com/FredericaBernkastel/Hutton32-Computer/blob/master/doc/Hutton32b.rule
[`rule.rs`]: research/src/rule.rs
[`rulespec.rs`]: research/src/rulespec.rs
[`pattern.rs`]: research/src/pattern.rs
[`mc.rs`]: research/src/mc.rs
[`sim.rs`]: research/src/sim.rs
[`flat.rs`]: research/src/flat.rs
[`cmd_stats.rs`]: research/src/cmd_stats.rs
[`cmd_gpu.rs`]: research/src/cmd_gpu.rs
[`cmd_brute2.rs`]: research/src/cmd_brute2.rs
[`cmd_insert.rs`]: research/src/cmd_insert.rs
[`cmd_sparse.rs`]: research/src/cmd_sparse.rs
[`hl.rs: result_at`]: research/src/hl.rs
[`cmd_bitslice.rs`]: research/src/cmd_bitslice.rs
[`resub.py`]: research/bitslice/resub.py
[`abc_flow.py`]: research/bitslice/abc_flow.py
[`xag.py`]: research/bitslice/xag.py
[`step_planes.wgsl`]: research/bitslice/step_planes.wgsl
[`step_planes.rs`]: research/bitslice/step_planes.rs
[`verify.py`]: research/bitslice/verify.py
[`scalar.py`]: research/bitslice/scalar.py
[`tables.py`]: research/bitslice/tables.py
[`bitslice_series.sh`]: research/tools/bitslice_series.sh
[`time_golly.py`]: research/tools/time_golly.py
[`head2head.py`]: research/tools/head2head.py
[`engine_choice.sh`]: research/tools/engine_choice.sh
