# Patterns used in `paper.md`

Every pattern the paper measures, copied byte for byte so that Appendix A of `paper.md` runs from this repository alone. Paths are relative to this folder; `reproduce.sh` runs from `research/` and refers to them as `patterns/…`.

A pattern runs under the rule named in its header (`cmd_stats.rs: variant_for`), which is the rule Golly runs it with; a two-state macrocell file without a rule line runs as Life (B3/S23), as Golly reads it. The canonical tables of Hutton32 and Hutton32b are in [`../rules`](../rules); Life is built from its B/S notation (`src/ca.rs`).

| File | Rule | Bounding box | Targets of `reproduce.sh` that use it |
|---|---|---|---|
| `advanced counter v2.rle` | Hutton32b | 191 × 197 | golly, miss-rates, sparse, head2head, sweep-ops, floors, cond-miss, engine-choice, controller, counter-work, bitplane-content, bitplane-sizes, tile, reprice |
| `component-basis.rle` | Hutton32b | 471 × 718 | miss-rates |
| `16x16 movie/16x16 movie.mc` | Hutton32 | 2182 × 2620 | golly, miss-rates, head2head, engine-choice, reprice |
| `32x32 movie/32x32 movie.mc` | Hutton32b | 928 × 1513 | golly, miss-rates, head2head, sweep-ops, cond-miss, engine-choice, controller, reprice |
| `misc/BIN2BCD.rle` | Hutton32b | 348 × 350 | miss-rates, head2head |
| `misc/chaotic.rle` | Hutton32b | 151 × 132 | lut-content, golly, miss-rates, head2head, sweep-ops, floors, cond-miss, engine-choice, controller, bitplane-content, tile, reprice |
| `misc/squares.rle` | Hutton32 | 650 × 423 | lut-kernel, golly, miss-rates, head2head, cond-miss, engine-choice, controller, bitplane-sizes, tile, reprice |
| `misc/Segmented ROM (256k).mc` | Hutton32b | 876 × 615 | golly, miss-rates, head2head, engine-choice, reprice |
| `misc/serial display (32x32).mc` | Hutton32b | 938 × 1824 | golly, miss-rates, head2head, sweep-ops, engine-choice, controller, reprice |
| `misc/parallel display (16x16).mc` | Hutton32 | 1141 × 537 | golly, miss-rates, head2head |
| `misc/efficient_demultiplexer.mc` | Hutton32 | 948 × 610 | golly, miss-rates, head2head |
| `misc/weird designs/cpu.mc` | Hutton32 | 384 × 430 | miss-rates, head2head |
| `misc/extra/lotr1.ANSI-1M.rle` | Hutton32b | 3095 × 3059 | golly, rom-kernels, miss-rates, sweep, sweep-tiles, bitplane-content, bitplane-sizes, tile |
| `misc/extra/lotr1.PPMd-244K.rle` | Hutton32b | 1548 × 1461 | miss-rates, sweep-ops, floors, cond-miss, engine-choice, controller, reprice |
| `replicators/Hutton-replicator.rle` | Hutton32 | 3129 × 4 at generation 0 | golly, batches, head2head |
| `presets/hutton32b_8bit_counter.rle` | Hutton32b | 191 × 197 | lut-kernel, cpu-bench; the default pattern of the harness and of `bitslice/bench` |
| `presets/hutton32_8bit_counter_old.rle` | Hutton32 | 633 × 449 | lut-kernel (tiled into the large grids) |
| `life/Unit-Life-Cell-512x512.rle` | B3/S23 | 511 × 511 | life-kernels, life-golly, life-h2h, life-sweep, life-sparse, life-floors |
| `life/metapixel-galaxy.mc` | B3/S23 | 30720 × 30720 | life-h2h |
| `life/ruler.mc` | B3/S23 (no rule line) | 169 × 116 | life-h2h |
| `life/wedge-grow.mc` | B3/S23 (no rule line) | 16193 × 15089 | life-h2h |
| `life/gotts-dots.mc` | B3/S23 (no rule line) | 187 × 39 | life-h2h |
| `life/metacatacryst.mc` | B3/S23 (no rule line) | 59739 × 14663 | life-h2h |
| `life/catacryst.mc` | B3/S23 (no rule line) | 2555 × 1772 | life-h2h |
| `life/mosquito5.mc` | B3/S23 (no rule line) | 2754 × 650 | life-h2h |
| `life/jagged.mc` | B3/S23 (no rule line) | 125 × 67 | life-h2h |
| `life/totalperiodic.mc` | B3/S23 | 56 × 60 | life-h2h |

Sources:

* `life/` comes from Golly 5.1's own pattern collection: `Patterns/Life/Signal-Circuitry/Unit-Life-Cell-512x512.rle`, `Patterns/HashLife/Metacell/metapixel-galaxy.mc.gz` (decompressed) and `Patterns/HashLife/Eventful/*.mc`. The random soups of the Life targets are not files: `ca-research life-soup` draws them from a fixed hash and a seed.
* Everything outside `presets/` and `life/` comes from the Hutton32 folder of the Golly installation the paper used (`Patterns/Hutton32`), which is a clone of the Hutton32-Computer repository at commit `f767939`. `misc/extra/` is not committed in that repository. `replicators/Hutton-replicator.rle` is Tim Hutton's and ships with Golly.
* `presets/*.rle` are the two palette-PNG presets of the archived app (`archive/doc/hutton32b_8bit_counter.png`, `archive/doc/hutton32_8bit_counter_old.png`; one pixel per cell) converted to Golly RLE with `ca-research export` (the first one earlier with `bitslice/png_to_rle.py`, which gives the same file). For each, `ca-research fingerprint` gives the PNG and the RLE the same population, bounding box and hash, and `tools/head2head.py` shows Golly and the reference engine in the same state after 65,536 generations, so Golly loads them as the harness does.
* The two presets are different patterns from `advanced counter v2`: the Hutton32b one has the same bounding box but 24,449 live cells against 24,058.
