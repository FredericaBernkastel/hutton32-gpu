#!/usr/bin/env bash
# gpu-life (hand-written WGSL) against cubecl-life (its CubeCL port) on the grid sizes of the
# `life-kernels` target, plain kernel and shared tile. Needs a build with `--features cubecl`.
# Usage (from research/): bash tools/cubecl_compare.sh > logs/<date>/cubecl-life.txt
set -euo pipefail
R=target/release/ca-research.exe
[ -x "$R" ] || R=target/release/ca-research
run() { echo "=== $*"; "$R" "$@" 2>&1 | grep -E 'life kernel|verify|hash|^GPU' ; }

for sg in "256 65536" "1024 8192" "4096 2048" "16384 256" "32768 64"; do
  set -- $sg
  run gpu-life --side $1 --gens $2 --repeats 5
  for c in spirv wgsl; do run cubecl-life --side $1 --gens $2 --repeats 5 --compiler $c; done
done
for sg in "256 65536" "1024 8192"; do
  set -- $sg
  run gpu-life --side $1 --gens $2 --k 16 --tw 6 --th 32 --repeats 5
  for c in spirv wgsl; do run cubecl-life --side $1 --gens $2 --k 16 --tw 6 --th 32 --repeats 5 --compiler $c; done
done
# bounds-checked launches, and the final states against the CPU table
run cubecl-life --side 4096 --gens 2048 --repeats 5 --compiler spirv --checked
run cubecl-life --side 1024 --gens 8192 --k 16 --tw 6 --th 32 --repeats 5 --compiler spirv --checked
run gpu-life --side 512 --gens 1024 --k 16 --tw 6 --th 32 --verify
for c in spirv wgsl; do
  run cubecl-life --side 512 --gens 1024 --verify --compiler $c
  run cubecl-life --side 512 --gens 1024 --k 16 --tw 6 --th 32 --verify --compiler $c
  run cubecl-life --side 512 --gens 1024 --k 16 --tw 6 --th 32 --verify --compiler $c --checked
done
