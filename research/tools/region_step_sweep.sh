#!/usr/bin/env bash
# The region tied to the step: region = 4 x step, the smallest region the recursion admits, for steps
# of 8 to 64 generations on Hutton32b (8x8 base tiles) and of 16 to 256 on Life (16x16 base tiles; a
# step is at least a base tile's side). On each universe the floor run prices plain cells in every
# region and the cheaper engine per region in hindsight (the region model), and `hindsight` runs
# that assignment for regions up to 256x256 (as in the region sweeps). Model times: operation counts
# priced with the constants of Table I. 15 to 25 minutes (23 with another sweep running beside it).
# usage (from research/): bash tools/region_step_sweep.sh <ca-research exe> [k...]
R="$1"; shift
KS=("$@"); [ $# -eq 0 ] && KS=(3 4 5 6 7 8)
D="$(dirname "$0")"
P="$D/../patterns"
CTR="$P/advanced counter v2.rle"; CHA="$P/misc/chaotic.rle"; PPMD="$P/misc/extra/lotr1.PPMd-244K.rle"
CH8="$CHA@64,64;$CHA@576,64~1;$CHA@1088,64~2;$CHA@1600,64~3;$CHA@64,576~4;$CHA@576,576~5;$CHA@1088,576~6;$CHA@1600,576~7"
SOUP="$D/../out/life/soup512.rle"
[ -e "$SOUP" ] || "$R" life-soup --side 512 --seed 1 --out "$SOUP" >/dev/null
LIFE=(--a 4 --init-floor 4 --max-floor 5 --cell-rate 1.39e12)
for k in "${KS[@]}"; do
  rl=$((k + 2))
  HIND=",hindsight"; [ "$rl" -gt 8 ] && HIND=""
  echo "=== step $((1 << k)), region $((1 << rl))x$((1 << rl)) (--k $k --rl $rl)"
  if [ "$k" -le 6 ]; then
    echo "--- Hutton32b"
    H=(--a 3 --k "$k" --init-floor 3 --policies "16x16$HIND" --rl "$rl")
    "$R" adapt --compose "$(cat "$D/composite_spec.txt")" --gens 4096 "${H[@]}"
    "$R" adapt --pattern "$PPMD" --gens 2112 "${H[@]}"
    "$R" adapt --pattern "$CTR" --gens 16384 "${H[@]}"
    "$R" adapt --compose "$CH8" --gens 8192 "${H[@]}"
  fi
  if [ "$k" -ge 4 ]; then
    echo "--- Life"
    L=("${LIFE[@]}" --k "$k" --policies "32x32$HIND" --rl "$rl")
    "$R" adapt --pattern "$SOUP" --gens 1024 "${L[@]}"
    "$R" adapt --pattern "$P/life/Unit-Life-Cell-512x512.rle" --gens 11520 "${L[@]}"
    "$R" adapt --pattern "$P/life/metapixel-galaxy.mc" --gens 4096 "${L[@]}"
    "$R" adapt --pattern "$P/life/catacryst.mc" --gens 4096 "${L[@]}"
    "$R" adapt --pattern "$P/life/gotts-dots.mc" --gens 4096 "${L[@]}"
  fi
done
