# Patterns used in `paper.md`

Every pattern the report measures, copied byte for byte so that Appendix A of `paper.md` runs from this repository alone. Paths are relative to this folder; the commands in Appendix A run from `research/` and refer to them as `patterns/…`.

A pattern runs under the rule named in its header (`cmd_stats.rs: variant_for`), which is the rule Golly runs it with. The canonical tables of both rules are in [`../rules`](../rules).

| File | Rule | Bounding box | Used in |
|---|---|---|---|
| `advanced counter v2.rle` | Hutton32b | 191 × 197 | §1.3, §2.3, §4.3 (sparse universe), §4.4, §10, §12, §13 |
| `component-basis.rle` | Hutton32b | 471 × 718 | §2.3 |
| `16x16 movie/16x16 movie.mc` | Hutton32 | 2182 × 2620 | §1.3, §2.3, §4.4, §10, §12.5 |
| `32x32 movie/32x32 movie.mc` | Hutton32b | 928 × 1513 | §1.3, §2.3, §4.4, §12 |
| `misc/BIN2BCD.rle` | Hutton32b | 348 × 350 | §2.3, §4.4, §10 |
| `misc/chaotic.rle` | Hutton32b | 151 × 132 | §1.2, §1.3, §2.3, §4.4, §12, §13 |
| `misc/squares.rle` | Hutton32 | 650 × 423 | §1.2, §1.3, §2.3, §4.4, §12, §13 |
| `misc/Segmented ROM (256k).mc` | Hutton32b | 876 × 615 | §1.3, §2.3, §4.4, §10, §12.5 |
| `misc/serial display (32x32).mc` | Hutton32b | 938 × 1824 | §1.3, §2.3, §4.4, §10, §12 |
| `misc/parallel display (16x16).mc` | Hutton32 | 1141 × 537 | §1.3, §2.3, §4.4, §10 |
| `misc/efficient_demultiplexer.mc` | Hutton32 | 948 × 610 | §1.3, §2.3, §4.4, §10 |
| `misc/weird designs/cpu.mc` | Hutton32 | 384 × 430 | §2.3, §4.4, §10 |
| `misc/extra/lotr1.ANSI-1M.rle` | Hutton32b | 3095 × 3059 | §1.3, §2.3, §4.2, §10, §12.2, §13 |
| `misc/extra/lotr1.PPMd-244K.rle` | Hutton32b | 1548 × 1461 | §2.3, §10, §12 |
| `replicators/Hutton-replicator.rle` | Hutton32 | 3129 × 4 at generation 0 | §1.1, §1.3, §4.4 |
| `presets/hutton32b_8bit_counter.png` | Hutton32b (no header; the harness default) | 191 × 197 | §1.2 |
| `presets/hutton32_8bit_counter_old.png` | Hutton32, given as `--rule hutton32` | 633 × 449 | §1.2 |
| `presets/hutton32b_8bit_counter.rle` | Hutton32b | 191 × 197 | §13.5 (default input of `bitslice/bench`) |

Sources:

* Everything outside `presets/` comes from the Hutton32 folder of the Golly installation the report used (`Patterns/Hutton32`), which is a clone of the Hutton32-Computer repository at commit `f767939`. `misc/extra/` is not committed in that repository. `replicators/Hutton-replicator.rle` is Tim Hutton's and ships with Golly.
* `presets/*.png` are presets of the archived app, identical to the files in `archive/doc`; the copies keep `research/` self-contained. One pixel is one cell; every colour is an entry of the app's palette.
* `presets/hutton32b_8bit_counter.rle` is `hutton32b_8bit_counter.png` converted by `bitslice/png_to_rle.py`; `ca-research fingerprint` gives both the same population, bounding box and hash.
