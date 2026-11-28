#!/usr/bin/env bash
# The engine region of Algorithm 3 for Life, at 2^rl cells, rl = 6..10 (64x64 to 1024x1024), as GHL
# runs Life: 16x16 base tiles, floor F32 (F64 and plain cells as the controller's alternatives),
# 16-generation steps, priced at the Life kernel's rate as in Table lifefloors. One pattern per
# universe (adapt composes Hutton32b universes only). Plain cells per region alone (C16) only up to
# 256x256: above that the reference engine steps whole regions cell by cell for minutes; the region
# model prices them at every size from the floor run, and `hindsight` runs the cheaper engine per
# region. About 13 minutes.
# usage (from research/): bash tools/region_sweep_life.sh <ca-research exe> [rl...]
R="$1"; shift
RLS=("$@"); [ $# -eq 0 ] && RLS=(6 7 8 9 10)
D="$(dirname "$0")"
P="$D/../patterns/life"
SOUP="$D/../out/life/soup512.rle"
[ -e "$SOUP" ] || "$R" life-soup --side 512 --seed 1 --out "$SOUP" >/dev/null
LIFE=(--a 4 --k 4 --init-floor 4 --max-floor 5 --cell-rate 1.39e12)
for rl in "${RLS[@]}"; do
  echo "=== region $((1 << rl))x$((1 << rl)) (--rl $rl)"
  POL="32x32,adaptive,hindsight"; [ "$rl" -gt 8 ] && POL="32x32,adaptive"   # no cell-by-cell runs of regions above 256x256
  "$R" adapt --pattern "$SOUP" --gens 1024 "${LIFE[@]}" --policies "$POL" --rl "$rl"
  "$R" adapt --pattern "$P/Unit-Life-Cell-512x512.rle" --gens 11520 "${LIFE[@]}" --policies "$POL" --rl "$rl"
  "$R" adapt --pattern "$P/metapixel-galaxy.mc" --gens 4096 "${LIFE[@]}" --policies "$POL" --rl "$rl"
  "$R" adapt --pattern "$P/catacryst.mc" --gens 4096 "${LIFE[@]}" --policies "$POL" --rl "$rl"
  "$R" adapt --pattern "$P/gotts-dots.mc" --gens 4096 "${LIFE[@]}" --policies "$POL" --rl "$rl"
  if [ "$rl" -le 8 ]; then
    echo "--- plain cells per region, 16-generation steps (C16), and the floor run that prices them"
    "$R" adapt --pattern "$SOUP" --gens 1024 "${LIFE[@]}" --policies "32x32,plain" --rl "$rl"
    "$R" adapt --pattern "$P/Unit-Life-Cell-512x512.rle" --gens 11520 "${LIFE[@]}" --policies "32x32,plain" --rl "$rl"
  fi
done
