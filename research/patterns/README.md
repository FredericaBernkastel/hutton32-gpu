# Patterns used in `paper.md`

Every pattern the paper measures, copied byte for byte so that Appendix A of `paper.md` runs from this repository alone. Paths are relative to this folder; `reproduce.sh` runs from `research/` and refers to them as `patterns/…`.

A pattern runs under the rule named in its header (`cmd_stats.rs: variant_for`), which is the rule Golly runs it with. The canonical tables of both rules are in [`../rules`](../rules).

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

Sources:

* Everything outside `presets/` comes from the Hutton32 folder of the Golly installation the paper used (`Patterns/Hutton32`), which is a clone of the Hutton32-Computer repository at commit `f767939`. `misc/extra/` is not committed in that repository. `replicators/Hutton-replicator.rle` is Tim Hutton's and ships with Golly.
* `presets/*.rle` are the two palette-PNG presets of the archived app (`archive/doc/hutton32b_8bit_counter.png`, `archive/doc/hutton32_8bit_counter_old.png`; one pixel per cell) converted to Golly RLE with `ca-research export` (the first one earlier with `bitslice/png_to_rle.py`, which gives the same file). For each, `ca-research fingerprint` gives the PNG and the RLE the same population, bounding box and hash, and `tools/head2head.py` shows Golly and the reference engine in the same state after 65,536 generations, so Golly loads them as the harness does.
* The two presets are different patterns from `advanced counter v2`: the Hutton32b one has the same bounding box but 24,449 live cells against 24,058.
