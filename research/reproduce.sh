#!/usr/bin/env bash
# Reproduces the measured figures of paper.md, one target per table (paper Appendix A).
#
#   bash reproduce.sh <target>...     run one or more targets (from any directory)
#   bash reproduce.sh                 list the targets
#
# Runs in git-bash on Windows, in WSL or on Linux, and builds the harness first.
# Golly targets take bgolly from $BGOLLY, from Golly 5.1's Windows install path or from PATH;
# research/rules is its rule folder. Python is $PYTHON, else python3, else python.
set -u
cd "$(dirname "$0")" || exit 1

TARGETS='
1.1    reference engine vs per-cell stepping; CPU rule port vs the WGSL on the GPU and vs both tables
1.2a   the archived kernel (gpu-baseline)
1.2b   the byte-LUT kernel, workgroup shapes, DX12 (gpu-brute)
1.2c   content: full and class-compressed LUT (gpu-brute2)
1.3a   Golly HashLife per pattern (needs Golly)
1.3b   lotr1.ANSI-1M: archived kernel, byte-LUT kernel and one CPU core
2.1    probe and insert rates R_P, R_I (gpu-probe, gpu-insert; R_B is 1.2b)
2.3    miss rates of the flat tile automaton (flat)
2.5    batch statistics of the level-synchronous schedule, Hutton replicator (root)
2.6    latency constants and the miss-path wave (gpu-latency)
4.2    memo hit path on phase-shifted ROMs (gpu-sweep; learns ~90 s per period first)
4.3    sparse 100k x 100k universe: GPU, Golly, CPU engine (needs Golly)
4.4    CPU engine vs Golly, final states compared (needs Golly)
10a    WGSL ports vs the canonical tables, v1 vs now (lut, rulediff, fix_rules.py)
10b    which patterns notice the v1 rules (needs Golly)
10c    the committed tables are the published files (needs the network)
12.2a  sweep tiles 16x16 to 256x256 on 32x32 ROMs (gpu-sweep; long)
12.2b  hash operations per generation by sweep tile (flat)
12.3   fixed floors and plain cells (adapt)
12.4   conditional miss rate per window size (root)
12.5a  engine per content (tools/engine_choice.sh)
12.5b  the controller (adapt)
12.6   work per generation of the counter, which sets the thresholds (adapt)
13.1   forms A and C, checked on all 2^25 neighbourhoods (pip installs numpy)
13.2   the circuit vs the table on all 2^25 neighbourhoods (pip installs numpy)
13.3a  16384^2: bit-plane and byte-LUT kernels on the same content
13.3b  bit-plane kernel at other sizes
13.3c  where the limit is: identity circuit, k chained evaluations
13.4a  one thread per window vs chained dispatches; serial cost (gpu-window)
13.4b  shared tile, k generations per dispatch (gpu-tile)
13.5   CPU engines, default and target-cpu=native builds (bitslice/bench)
13.6   12.5a and 12.6 re-priced at the rate of the bit-plane kernel
'

R=./target/release/ca-research
P=patterns
CTR="$P/advanced counter v2.rle"
CHA=$P/misc/chaotic.rle
SQ=$P/misc/squares.rle
ROM=$P/misc/extra/lotr1.ANSI-1M.rle
PPMD=$P/misc/extra/lotr1.PPMd-244K.rle
NEW=$P/presets/hutton32b_8bit_counter.png
OLD=$P/presets/hutton32_8bit_counter_old.png
# eight perturbed copies of chaotic on a 512-cell pitch
CH8="$CHA@64,64;$CHA@576,64~1;$CHA@1088,64~2;$CHA@1600,64~3;$CHA@64,576~4;$CHA@576,576~5;$CHA@1088,576~6;$CHA@1600,576~7"

py() {
  if [ -z "${PY:-}" ]; then
    for c in ${PYTHON:-} python3 python; do
      if "$c" -c 'import sys; sys.exit(sys.version_info < (3, 8))' >/dev/null 2>&1; then PY=$c; break; fi
    done
    [ -n "${PY:-}" ] || { echo "no Python 3 found; set PYTHON" >&2; exit 1; }
  fi
  "$PY" "$@"
}

venv() { # a venv with numpy for the circuit checks; sets VPY
  local v=out/bitslice-venv
  [ -d $v ] || py -m venv $v || exit 1
  VPY=$PWD/$v/bin/python; [ -e $v/Scripts/python.exe ] && VPY=$PWD/$v/Scripts/python.exe
  "$VPY" -c 'import numpy' 2>/dev/null || "$VPY" -m pip install -q numpy || exit 1
}

lutbin() { # the 2^25-byte Hutton32b table that bitslice/ and bitslice/bench read
  [ -e out/lut_table_Hutton32b.bin ] || $R lut --rule hutton32b --against table:rules/Hutton32b.rule
}

bgolly() {
  local b=${BGOLLY:-} c
  for c in "/c/Program Files/golly-5.1-win-64bit/bgolly.exe" "$(command -v bgolly)"; do
    [ -z "$b" ] && [ -x "$c" ] && b=$c
  done
  [ -n "$b" ] || { echo "bgolly not found; set BGOLLY" >&2; exit 1; }
  "$b" "$@"
}

series() { bash tools/bitslice_series.sh $R "$1"; } # sections of the section-13 series

run() {
  case "$1" in
  1.1)
    $R selftest
    $R lut --rule hutton32b
    $R lut --rule hutton32  --against table:rules/Hutton32.table
    $R lut --rule hutton32b --against table:rules/Hutton32b.rule ;;
  1.2a)
    $R gpu-baseline --pattern $NEW --gens 16384 --per-submit 272 --verify
    $R gpu-baseline --pattern $OLD --rule hutton32 --gens 8192 --per-submit 32 --verify
    $R gpu-baseline --pattern $SQ --gens 8192 --per-submit 32 --verify
    $R gpu-baseline --pattern $OLD --rule hutton32 --reps 8 --gens 512 --per-submit 32 ;;
  1.2b)
    $R gpu-brute --pattern $NEW --gens 65536 --per-submit 1024 --verify
    $R gpu-brute --pattern $SQ --gens 65536 --per-submit 1024 --verify
    $R gpu-brute --pattern $OLD --rule hutton32 --reps 8 --gens 2048 --per-submit 64 --verify
    $R gpu-brute --pattern $OLD --rule hutton32 --reps 24 --gens 512 --per-submit 32
    for s in "8 8" "16 16" "32 8" "64 4"; do
      set -- $s; $R gpu-brute --pattern $OLD --rule hutton32 --reps 24 --gens 512 --per-submit 32 --wgx $1 --wgy $2
    done
    $R gpu-brute --pattern $OLD --rule hutton32 --reps 24 --gens 512 --per-submit 32 --backend dx12 ;;
  1.2c)
    $R gpu-brute2 --side 8192 --gens 64 --pattern $CHA --verify
    $R gpu-brute2 --side 8192 --gens 64 --verify
    $R gpu-brute2 --side 8192 --gens 64 --same-src --verify ;;
  1.3a)
    py tools/time_golly.py "$CTR" 1048576 1 8 64 512 4096 32768 262144 --cells 37627
    py tools/time_golly.py $SQ 65536 8 64 1024 --algo JvN --cells 274950
    py tools/time_golly.py $CHA 16384 8 64 1024 --cells 19932
    py tools/time_golly.py $ROM 65536 64 4096 --cells 9467605 --timeout 180
    py tools/time_golly.py $P/replicators/Hutton-replicator.rle 1048576 1024 16384 262144 --algo JvN --timeout 300
    for spec in "misc/Segmented ROM (256k).mc|RuleLoader|538740" "misc/serial display (32x32).mc|RuleLoader|1710912" \
                "32x32 movie/32x32 movie.mc|RuleLoader|1404064" "misc/parallel display (16x16).mc|JvN|612717" \
                "16x16 movie/16x16 movie.mc|JvN|5716840" "misc/efficient_demultiplexer.mc|JvN|578280"; do
      IFS='|' read -r f algo cells <<< "$spec"
      py tools/time_golly.py "$P/$f" 65536 64 1024 --algo $algo --cells $cells --timeout 120
    done ;;
  1.3b)
    $R gpu-baseline --pattern $ROM --gens 64 --per-submit 32
    $R gpu-brute --pattern $ROM --gens 528 --per-submit 264 --verify ;; # --verify also prints the one-core CPU rate
  2.1)
    $R gpu-probe --log2-slots 10,14,16,18,19,20,21,22,24,25,26,27,28 # the 8 GiB table needs that much free VRAM
    $R gpu-insert ;;
  2.3) # SUMMARY lines: the second half of each run
    for a in 2 3 4; do for lt in 3 4 5 6 7; do
      [ $lt -ge $a ] && $R flat --pattern "$CTR" --lt $lt --a $a --gens 262144 --quiet --summary | grep SUMMARY
    done; done
    for f in "misc/Segmented ROM (256k).mc" "misc/serial display (32x32).mc" "misc/parallel display (16x16).mc" \
             "16x16 movie/16x16 movie.mc" "misc/efficient_demultiplexer.mc" "misc/chaotic.rle" "misc/squares.rle" \
             "component-basis.rle" "misc/extra/lotr1.PPMd-244K.rle"; do
      for lt in 4 5; do $R flat --pattern "$P/$f" --lt $lt --a 2 --gens 16384 --quiet --summary | grep SUMMARY; done
    done
    $R flat --pattern "$P/32x32 movie/32x32 movie.mc" --lt 4 --a 2 --gens 131072 --summary
    $R flat --pattern $ROM --lt 4 --a 2 --gens 4096 --summary
    for f in "misc/weird designs/cpu.mc" "misc/BIN2BCD.rle"; do $R flat --pattern "$P/$f" --lt 4 --a 2 --gens 65536; done ;;
  2.5)
    $R root --pattern $P/replicators/Hutton-replicator.rle --k 14 --gens 262144 ;;
  2.6)
    $R gpu-latency ;;
  4.2)
    $R gpu-sweep --pattern $ROM --lt 4 --reps 1  --cycles 6600 --per-submit 512
    $R gpu-sweep --pattern $ROM --lt 5 --reps 1  --cycles 6600 --per-submit 512
    $R gpu-sweep --pattern $ROM --lt 4 --reps 8  --cycles 66
    $R gpu-sweep --pattern $ROM --lt 4 --reps 32 --cycles 33 --per-submit 4
    $R gpu-sweep --pattern $ROM --lt 5 --reps 32 --cycles 33 --per-submit 4 ;;
  4.3)
    mkdir -p out/golly
    $R gpu-sparse --pattern "$CTR" --lt 4 --reps 48 --pitch 8 --cycles 1024 --export-mc out/golly/sparse100k.mc
    py tools/time_golly.py out/golly/sparse100k.mc 16384 16 1024 16384 --timeout 600
    py tools/time_golly.py out/golly/sparse100k.mc 1048576 1024 65536 --timeout 900
    bgolly -q -q -a RuleLoader -s rules/ -M 8000 -m 16384 -i 1024 -o out/golly/sparse100k_out.mc out/golly/sparse100k.mc
    $R fingerprint --pattern out/golly/sparse100k_out.mc     # Golly's final state
    $R root --pattern out/golly/sparse100k.mc --k 10 --gens 16384 --a 2 ;; # the CPU engine's
  4.4) # the last column compares the final states
    py tools/head2head.py 16777216 10 --a 2 --timeout 400 "$P/misc/Segmented ROM (256k).mc" "$P/misc/serial display (32x32).mc" \
      "$P/misc/parallel display (16x16).mc" "$P/16x16 movie/16x16 movie.mc" $P/misc/efficient_demultiplexer.mc \
      "$P/misc/weird designs/cpu.mc" $P/misc/BIN2BCD.rle "$CTR" $CHA "$P/32x32 movie/32x32 movie.mc"
    py tools/head2head.py 1048576 10 --a 2 --timeout 600 $SQ
    py tools/head2head.py 262144 14 --a 2 --timeout 900 $P/replicators/Hutton-replicator.rle ;;
  10a)
    $R lut --rule hutton32  --against table:rules/Hutton32.table
    $R lut --rule hutton32b --against table:rules/Hutton32b.rule
    $R rulediff --a table:rules/Hutton32.table --b table:rules/Hutton32b.rule
    $R rulediff --a hutton32-v1  --b hutton32
    $R rulediff --a hutton32b-v1 --b hutton32b
    mkdir -p out/v1
    git show 256ef91:archive/src/kernel/rules/hutton32b.rule.wgsl > out/v1/hutton32b.rule.wgsl
    $R lut --rule hutton32b --wgsl out/v1/hutton32b.rule.wgsl --against hutton32b-v1 # the v1 port is the v1 file
    py tools/fix_rules.py ;;
  10b)
    py tools/head2head.py 16777216 10 --a 2 --timeout 400 --v1 "$P/misc/Segmented ROM (256k).mc" \
      "$P/misc/serial display (32x32).mc" "$P/misc/parallel display (16x16).mc" "$P/16x16 movie/16x16 movie.mc" \
      $P/misc/efficient_demultiplexer.mc "$P/misc/weird designs/cpu.mc" $P/misc/BIN2BCD.rle "$CTR" ;;
  10c)
    curl -sL --retry 3 https://gollygang.github.io/ruletablerepository/downloads/Hutton32.table \
      | cmp - rules/Hutton32.table && echo "rules/Hutton32.table: same as published"
    curl -sL --retry 3 https://raw.githubusercontent.com/FredericaBernkastel/Hutton32-Computer/master/doc/Hutton32b.rule \
      | cmp - rules/Hutton32b.rule && echo "rules/Hutton32b.rule: same as published" ;;
  12.2a)
    $R gpu-sweep --pattern $ROM --lt 4 --reps 32 --cycles 33 --per-submit 4
    $R gpu-sweep --pattern $ROM --lt 5 --reps 32 --cycles 33 --per-submit 4
    for lt in 6 7 8; do
      $R gpu-sweep --pattern $ROM --lt $lt --a 4 --reps 32 --cycles 264 --per-submit 66 --max-learn-gens 40000
    done ;;
  12.2b) # MODEL lines
    for spec in "advanced counter v2.rle|65536" "misc/chaotic.rle|16384" "misc/extra/lotr1.PPMd-244K.rle|528" \
                "misc/extra/lotr1.PPMd-244K.rle|8448" "32x32 movie/32x32 movie.mc|16384" "misc/serial display (32x32).mc|16384"; do
      IFS='|' read -r f g <<< "$spec"
      for lt in 3 4 5 6 7 8; do $R flat --pattern "$P/$f" --lt $lt --a 2 --gens $g --quiet | grep MODEL; done
    done ;;
  12.3) # --a 2: 4x4 base tiles, --a 3: 8x8
    for a in 2 3; do
      $R adapt --pattern "$CTR" --gens 16384 --a $a --policies fixed
      $R adapt --compose "$CH8" --gens 8192 --a $a --policies fixed
      $R adapt --pattern $PPMD --gens 512 --a $a --policies fixed
    done ;;
  12.4) # per level: requests in column 6, misses in column 7
    for spec in "misc/extra/lotr1.PPMd-244K.rle|512" "advanced counter v2.rle|16384" "misc/chaotic.rle|8192" \
                "32x32 movie/32x32 movie.mc|4096" "misc/squares.rle|4096"; do
      IFS='|' read -r f g <<< "$spec"
      $R root --pattern "$P/$f" --k 6 --gens $g --a 3
    done ;;
  12.5a)
    bash tools/engine_choice.sh $R 4096 ;;
  12.5b)
    $R adapt --compose "$(cat tools/composite_spec.txt)" --gens 4096 --a 3 --k 4 --init-floor 3 --policies "16x16,adaptive"
    $R adapt --pattern $PPMD --gens 2112 --a 3 --k 4 --init-floor 3 --policies "16x16,adaptive"
    $R adapt --pattern "$CTR" --gens 16384 --a 3 --k 4 --init-floor 3 --policies "16x16,adaptive"
    $R adapt --compose "$CH8" --gens 8192 --a 3 --k 4 --init-floor 3 --policies "16x16,adaptive" ;;
  12.6)
    $R adapt --pattern "$CTR" --gens 16384 --a 3 ;;
  13.1)
    lutbin; venv
    (cd bitslice && "$VPY" scalar.py && "$VPY" tables.py) ;;
  13.2)
    lutbin; venv
    (cd bitslice && "$VPY" verify.py) ;;
  13.3a) series A ;;
  13.3b) series BCDS ;;
  13.3c) series EFS ;;
  13.4a) series HGE ;;
  13.4b) series T ;;
  13.5)
    lutbin
    cargo run --release --manifest-path bitslice/bench/Cargo.toml
    RUSTFLAGS="-C target-cpu=native" cargo run --release --manifest-path bitslice/bench/Cargo.toml ;; # adds AVX2
  13.6)
    bash tools/engine_choice.sh $R 4096 --cell-rate 2.9e11
    $R adapt --pattern "$CTR" --gens 16384 --a 3 --cell-rate 2.9e11 ;;
  *)
    echo "unknown target: $1" >&2; return 2 ;;
  esac
}

if [ $# -eq 0 ]; then
  echo "usage: bash reproduce.sh <target>...   (GPU figures want an otherwise idle machine)"
  echo "$TARGETS"
  exit 0
fi
desc() { echo "$TARGETS" | awk -v t="$1" '$1 == t'; }
for t in "$@"; do
  [ -n "$(desc "$t")" ] || { echo "unknown target: $t (bash reproduce.sh lists them)" >&2; exit 2; }
done
cargo build --release -q || exit 1
for t in "$@"; do
  echo "=== $(desc "$t")"
  run "$t"
done
