#!/usr/bin/env bash
# Every GPU measurement of the bit-plane, window and shared-tile kernels, with the circuit in
# bitslice/step_planes.wgsl. reproduce.sh runs it section by section, one target per table.
# usage (from research/): bash tools/bitslice_series.sh ./target/release/ca-research [sections] [quick]
#   sections: letters to run, default ABCDEFGHST
#   quick:    skip the CPU verification of the pattern runs (the random-soup runs are still verified)
R="$1"; SECTIONS="${2:-ABCDEFGHST}"; QUICK="$3"
want() { case "$SECTIONS" in *"$1"*) return 0;; *) return 1;; esac; }
P="$(dirname "$0")/../patterns"
T=out/bitslice-tmp; mkdir -p "$T"
C=bitslice/step_planes.wgsl
SIG='fn step_planes(c: array<u32,5>, n: array<u32,5>, e: array<u32,5>, s: array<u32,5>, w: array<u32,5>) -> array<u32,5>'

# test circuits: identity, and k evaluations of the real circuit that depend on one another
echo "$SIG { return c; }" > "$T/identity.wgsl"
for k in 1 2 4 8 16 32; do
  {
    sed 's/fn step_planes(/fn step1(/' "$C"
    echo "$SIG {"
    echo "  var a = c; var b = n; var d = e; var f = s; var g = w;"
    for i in $(seq 1 $k); do echo "  { let t = step1(a, b, d, f, g); a = b; b = d; d = f; f = g; g = t; }"; done
    echo "  return g;"
    echo "}"
  } > "$T/chain$k.wgsl"
done

V="--verify"; [ "$QUICK" = quick ] && V=""
run() { # label, command...
  local label="$1"; shift
  "$@" 2>&1 | grep -E 'bit-plane kernel:|LUT|window kernel|tile kernel|circuit:' | grep -v 'wiring' | sed "s#^ *#[$label] #"
}

echo "=== circuit"; head -4 "$C"

if want A; then
echo "=== A. large grids (16384 x 16384): bit-plane kernel and byte-LUT kernel on the same content"
run "random, held constant | bit-plane" "$R" gpu-bitslice --side 16384 --gens 64 --same-src --verify
run "random, held constant | byte LUT " "$R" gpu-brute2   --side 16384 --gens 64 --same-src --verify
for item in "counter|$P/advanced counter v2.rle" "chaotic|$P/misc/chaotic.rle" "ROM|$P/misc/extra/lotr1.ANSI-1M.rle"; do
  name="${item%%|*}"; file="${item#*|}"
  run "$name tiled | bit-plane" "$R" gpu-bitslice --side 16384 --gens 64 --pattern "$file" $V
  run "$name tiled | byte LUT " "$R" gpu-brute2   --side 16384 --gens 64 --pattern "$file" $V
done
fi

if want B; then
echo "=== B. one ROM, one period of 528 generations"
run "ROM 3104x3067" "$R" gpu-bitslice --side 3104 --gens 528 --pattern "$P/misc/extra/lotr1.ANSI-1M.rle" --verify
fi

if want C; then
echo "=== C. small grids, bound by dispatch latency (three runs each)"
run "counter, verified" "$R" gpu-bitslice --side 200 --gens 65536 --pattern "$P/advanced counter v2.rle" --verify
for i in 1 2 3; do run "counter 224x205" "$R" gpu-bitslice --side 200 --gens 65536 --pattern "$P/advanced counter v2.rle"; done
for i in 1 2 3; do run "squares 672x431" "$R" gpu-bitslice --side 672 --gens 65536 --pattern "$P/misc/squares.rle"; done
fi

if want D; then
echo "=== D. 1.07e9 cells"
run "32768 x 32768 random" "$R" gpu-bitslice --side 32768 --gens 32 --same-src --verify
fi

if want E; then
echo "=== E. ceiling: the same harness with a circuit that returns the centre unchanged"
for side in 1024 2048 4096 16384; do
  g=512; [ $side = 16384 ] && g=64
  for i in 1 2; do
    run "side $side real    " "$R" gpu-bitslice --side $side --gens $g --same-src
    run "side $side identity" "$R" gpu-bitslice --side $side --gens $g --same-src --wgsl "$T/identity.wgsl"
  done
done
fi

if want F; then
echo "=== F. arithmetic headroom: k circuit evaluations per invocation, 16384 x 16384 (rate is per invocation pass)"
for k in 1 2 4 8; do for i in 1 2; do run "k=$k" "$R" gpu-bitslice --side 16384 --gens 64 --wgsl "$T/chain$k.wgsl"; done; done
fi

if want G; then
echo "=== G. serial cost: k dependent evaluations per invocation on a 64 x 64 grid (three runs each)"
for k in 1 2 4 8 16 32; do for i in 1 2 3; do run "k=$k" "$R" gpu-bitslice --side 64 --gens 65536 --wgsl "$T/chain$k.wgsl"; done; done
for i in 1 2 3; do run "identity" "$R" gpu-bitslice --side 64 --gens 65536 --wgsl "$T/identity.wgsl"; done
fi

if want H; then
echo "=== H. window kernel: 32x32 window, 8 generations, one dispatch"
for n in 1 64 1024 4096 16384 65536 262144; do
  it=$(( 4194304 / n )); [ $it -gt 4096 ] && it=4096; [ $it -lt 8 ] && it=8
  run "windows=$n dispatches=$it" "$R" gpu-window --windows $n --iters $it
done
fi

if want S; then
echo "=== S. size dependence, uniform random states held constant (two runs each)"
for item in "8192 128" "11584 64" "16384 64" "23168 32" "32768 32" "32768 128"; do
  set -- $item
  for i in 1 2; do run "side $1 gens $2" "$R" gpu-bitslice --side $1 --gens $2 --same-src; done
done
for i in 1 2; do run "side 32768 identity" "$R" gpu-bitslice --side 32768 --gens 32 --same-src --wgsl "$T/identity.wgsl"; done
fi

if want T; then
echo "=== T. shared tile: k generations per dispatch in workgroup-shared memory (gpu-tile)"
for k in 1 2 3 4 8; do run "random 1024x1024 k=$k" "$R" gpu-tile --k $k --side 1024 --gens 48 --verify; done
run "chaotic k=4, 3x9 groups" "$R" gpu-tile --k 4 --tw 3 --th 9 --side 700 --gens 64 --pattern "$P/misc/chaotic.rle" --verify
echo "--- small grids (plain kernel: counter 273k, squares 172k gens/s; section C)"
run "counter k=16, 7x16 groups" "$R" gpu-tile --k 16 --tw 7 --th 16 --side 200 --gens 65536 --pattern "$P/advanced counter v2.rle" --verify
for i in 1 2 3; do run "counter k=16, 7x16 groups" "$R" gpu-tile --k 16 --tw 7 --th 16 --side 200 --gens 65536 --pattern "$P/advanced counter v2.rle"; done
run "squares k=16, 6x48 groups" "$R" gpu-tile --k 16 --tw 6 --th 48 --side 672 --gens 16384 --pattern "$P/misc/squares.rle" --verify
for i in 1 2 3; do run "squares k=16, 6x48 groups" "$R" gpu-tile --k 16 --tw 6 --th 48 --side 672 --gens 65536 --pattern "$P/misc/squares.rle"; done
for th in 16 32; do for i in 1 2; do run "squares k=16, 6x$th groups" "$R" gpu-tile --k 16 --tw 6 --th $th --side 672 --gens 65536 --pattern "$P/misc/squares.rle"; done; done
echo "--- one ROM, one period (plain kernel: 41k gens/s; section B)"
for cfg in "2 6 28" "4 6 24" "8 6 24"; do
  set -- $cfg
  run "ROM k=$1, $2x$3 groups" "$R" gpu-tile --k $1 --tw $2 --th $3 --side 3104 --gens 528 --pattern "$P/misc/extra/lotr1.ANSI-1M.rle" --verify
done
echo "--- large grid (plain kernel: 3.0e11; section A)"
for k in 1 2 3 4 6 8; do run "random 16384^2 k=$k" "$R" gpu-tile --k $k --side 16384 --gens 48 --same-src; done
run "counter tiled 16384^2 k=3" "$R" gpu-tile --k 3 --side 16384 --gens 48 --pattern "$P/advanced counter v2.rle" $V
fi
