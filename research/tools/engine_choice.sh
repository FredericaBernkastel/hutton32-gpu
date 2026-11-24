#!/usr/bin/env bash
# Model cost of four engine configurations on each kind of content, one pattern at a time.
# usage: tools/engine_choice.sh <ca-research exe> <gens> [extra adapt options, e.g. --cell-rate 2.9e11]
#   F16 : memoise everything, brute-force missed 16x16 windows   (64-gen macro-steps)
#   F32 : memoise everything, brute-force missed 32x32 windows   (64-gen macro-steps)
#   C64 : plain cells per 256x256 region + memo above the region (64-gen macro-steps)
#   C16 : the same with 16-gen macro-steps
R="$1"; G="${2:-4096}"; EXTRA=("${@:3}")
P="D:/Program Files/golly-3.2-win/Patterns/Hutton32"
C="$P/misc/chaotic.rle"; K="$P/advanced counter v2.rle"
declare -a NAMES=("ROM lotr1.PPMd" "chaotic (perturbed)" "counter" "32x32 movie" "squares" "serial display" "Segmented ROM" "16x16 movie")
declare -a ARGS=(
  "--pattern|$P/misc/extra/lotr1.PPMd-244K.rle"
  "--compose|$C@64,64~3"
  "--compose|$K@64,64+7000"
  "--pattern|$P/32x32 movie/32x32 movie.mc"
  "--pattern|$P/misc/squares.rle"
  "--pattern|$P/misc/serial display (32x32).mc"
  "--pattern|$P/misc/Segmented ROM (256k).mc"
  "--pattern|$P/16x16 movie/16x16 movie.mc"
)
total() { grep -E "^$1" | awk '{print $(NF-3) "|" $(NF-1)}'; }
printf "%-22s %10s | %14s %14s %14s %14s\n" "content" "cells ms" "F16 ms/MiB" "F32 ms/MiB" "C64 ms/MiB" "C16 ms/MiB"
for i in "${!NAMES[@]}"; do
  IFS='|' read -r flag val <<< "${ARGS[$i]}"
  out6=$("$R" adapt "$flag" "$val" --gens "$G" --a 3 --k 6 --policies fixed "${EXTRA[@]}" 2>/dev/null)
  out4=$("$R" adapt "$flag" "$val" --gens "$G" --a 3 --k 4 --policies plain "${EXTRA[@]}" 2>/dev/null)
  nom=$(echo "$out6" | grep "^plain cells over" | awk '{print $8}')
  f16=$(echo "$out6" | total "fixed: brute at 16x16")
  f32=$(echo "$out6" | total "fixed: brute at 32x32")
  c64=$(echo "$out6" | total "fixed: plain cells")
  c16=$(echo "$out4" | total "fixed: plain cells")
  printf "%-22s %10s | %14s %14s %14s %14s\n" "${NAMES[$i]}" "$nom" "$f16" "$f32" "$c64" "$c16"
done
