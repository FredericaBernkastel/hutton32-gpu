#!/usr/bin/env bash
# The engine region of Algorithm 3 at 2^rl cells, rl = 6..10 (64x64 to 1024x1024), at 16-generation
# steps: the controller on the universes of Table controller, and plain cells per region on the ROM
# and the counter (the C16 runs of Table engine). A region must be at least four times the step, so
# 64-generation steps start at 256x256. Model times: operation counts priced with the constants of
# Table I. Every floor run also prices plain cells in every region and the cheaper engine per region
# in hindsight (the region model); `hindsight` runs that assignment for regions up to 256x256 (above
# that the reference engine steps plain-cells regions cell by cell for minutes), and the plain-cells
# runs check the price. 20 to 35 minutes (35 with another sweep running beside it).
# usage (from research/): bash tools/region_sweep.sh <ca-research exe> [rl...]
R="$1"; shift
RLS=("$@"); [ $# -eq 0 ] && RLS=(6 7 8 9 10)
D="$(dirname "$0")"
P="$D/../patterns"
CTR="$P/advanced counter v2.rle"; CHA="$P/misc/chaotic.rle"; PPMD="$P/misc/extra/lotr1.PPMd-244K.rle"
CH8="$CHA@64,64;$CHA@576,64~1;$CHA@1088,64~2;$CHA@1600,64~3;$CHA@64,576~4;$CHA@576,576~5;$CHA@1088,576~6;$CHA@1600,576~7"
for rl in "${RLS[@]}"; do
  echo "=== region $((1 << rl))x$((1 << rl)) (--rl $rl)"
  POL="16x16,adaptive,hindsight"; [ "$rl" -gt 8 ] && POL="16x16,adaptive"
  "$R" adapt --compose "$(cat "$D/composite_spec.txt")" --gens 4096 --a 3 --k 4 --init-floor 3 --policies "$POL" --rl "$rl"
  "$R" adapt --pattern "$PPMD" --gens 2112 --a 3 --k 4 --init-floor 3 --policies "$POL" --rl "$rl"
  "$R" adapt --pattern "$CTR" --gens 16384 --a 3 --k 4 --init-floor 3 --policies "$POL" --rl "$rl"
  "$R" adapt --compose "$CH8" --gens 8192 --a 3 --k 4 --init-floor 3 --policies "$POL" --rl "$rl"
  echo "--- plain cells per region, 16-generation steps (C16), and the floor run that prices them"
  "$R" adapt --pattern "$PPMD" --gens 4096 --a 3 --k 4 --policies "16x16,plain" --rl "$rl"
  "$R" adapt --compose "$CTR@64,64+7000" --gens 4096 --a 3 --k 4 --policies "16x16,plain" --rl "$rl"
done
