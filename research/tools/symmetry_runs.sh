#!/usr/bin/env bash
# Memo entries shared up to rotation and mirroring (the `symmetry` command) on the patterns of
# the evaluation: Hutton32 and Hutton32b at 8x8 base tiles and 16x16 sweep tiles, Life at 16x16
# and 32x32, with a check of the shared results on a sample of each memo.
# usage (from research/): bash tools/symmetry_runs.sh <ca-research exe>
R="$1"
P="$(dirname "$0")/../patterns"
H="--a 3 --lt 4 --check 2000"
L="--a 4 --lt 5 --check 2000"
"$R" symmetry --pattern "$P/advanced counter v2.rle" --gens 16384 $H
"$R" symmetry --pattern "$P/misc/squares.rle" --gens 4096 $H
"$R" symmetry --pattern "$P/misc/chaotic.rle" --gens 8192 $H
"$R" symmetry --pattern "$P/32x32 movie/32x32 movie.mc" --gens 4096 $H
"$R" symmetry --pattern "$P/misc/extra/lotr1.PPMd-244K.rle" --gens 512 $H
"$R" symmetry --pattern "$P/misc/extra/lotr1.ANSI-1M.rle" --gens 528 $H
"$R" symmetry --pattern "$P/life/Unit-Life-Cell-512x512.rle" --gens 11520 $L
"$R" symmetry --pattern "$(dirname "$0")/../out/life/soup512.rle" --gens 4096 $L
"$R" symmetry --pattern "$(dirname "$0")/../out/life/soup2048.rle" --gens 4096 $L
for p in metapixel-galaxy gotts-dots catacryst wedge-grow totalperiodic ruler; do
  "$R" symmetry --pattern "$P/life/$p.mc" --gens 4096 $L
done
