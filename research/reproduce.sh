#!/usr/bin/env bash
# Reproduces the measured figures of paper.md: one target per table or claim (paper Appendix A).
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
correctness       reference engine vs per-cell stepping; CPU rules vs both tables; the two rules compared
lut-kernel       the byte-LUT kernel, workgroup shapes, DX12 (gpu-brute)
lut-content       content: full and class-compressed LUT (gpu-brute2)
golly             Golly HashLife per pattern and step size, peak memory (needs Golly; about 7 minutes)
golly-rom-long    Golly on the ROM at step 1 past one period, until its 8 GB limit (needs Golly; about 7 minutes)
golly-tree        Golly with Hutton32b as an @TREE rule instead of @TABLE (needs Golly)
runlength         counter and chaotic: Golly and the CPU engine as the run grows (needs Golly)
rom-kernels       lotr1.ANSI-1M: byte-LUT kernel and one CPU core
probe             probe and insert rates R_P, R_I (gpu-probe, gpu-insert; R_B is lut-kernel)
miss-rates        miss rates of the flat tile automaton (flat)
batches           batch statistics of the level-synchronous schedule, Hutton replicator (root)
latency           latency constants and the miss-path wave (gpu-latency)
sweep             memo hit path on phase-shifted ROMs (gpu-sweep; learns ~90 s per period first)
unaligned         copies at arbitrary cell offsets: table size and hit path against alignment classes (gpu-sweep)
sparse            sparse 100k x 100k universe: GPU, Golly, CPU engine (needs Golly)
head2head         CPU engine vs Golly, final states compared (needs Golly)
rules-upstream    the committed tables are the published files (needs the network)
sweep-tiles       sweep tiles 16x16 to 256x256 on 32x32 ROMs (gpu-sweep; long)
sweep-ops         hash operations per generation by sweep tile (flat)
floors            fixed floors and plain cells (adapt)
cond-miss         conditional miss rate per window size (root)
engine-choice     engine per content (tools/engine_choice.sh)
controller        the controller (adapt)
region-size       the controller region, 64x64 to 1024x1024 (tools/region_sweep.sh; about 15 minutes)
region-size-life  the same for Life (tools/region_sweep_life.sh; about 7 minutes)
counter-work      work per generation of the counter, which sets the thresholds (adapt)
forms             forms A and C, checked on all 2^25 neighbourhoods (pip installs numpy)
circuit           the circuit vs the table on all 2^25 neighbourhoods (pip installs numpy)
bitplane-content  16384^2: bit-plane and byte-LUT kernels on the same content
bitplane-sizes    bit-plane kernel at other sizes
bitplane-limit    where the limit is: identity circuit, k chained evaluations
window            one thread per window vs chained dispatches; serial cost (gpu-window)
tile              shared tile, k generations per dispatch (gpu-tile)
cpu-bench         CPU engines, default and target-cpu=native builds (bitslice/bench)
cpu-parallel      bit-sliced CPU forms on 4 threads: random states and the ROM (bitslice/bench)
reprice           engine-choice and counter-work re-priced at the rate of the bit-plane kernel
figures           the overview diagram and Figs. 2-4 as SVG in figures/ (overview, about a minute on the CPU; tools/ghl_diagram.py, tools/figures.py)
life-kernels      Life: bit-sliced Moore kernel by grid size, shared tile on small grids (gpu-life)
life-golly        Life: Golly HashLife and QuickLife on a soup and the Unit Life Cell; the soup on the GPU (needs Golly)
life-h2h          Life: the CPU engine vs Golly HashLife, final states compared (needs Golly)
life-sweep        Life: memo hit path on phase-shifted Unit Life Cells (gpu-sweep)
life-sparse       Life: sparse universe of Unit Life Cells: GPU, Golly, CPU engine (needs Golly)
life-floors       Life: fixed floors priced at the rate of the Life kernel and of the byte-LUT kernel (adapt)
'

R=./target/release/ca-research
P=patterns
CTR="$P/advanced counter v2.rle"
CHA=$P/misc/chaotic.rle
SQ=$P/misc/squares.rle
ROM=$P/misc/extra/lotr1.ANSI-1M.rle
PPMD=$P/misc/extra/lotr1.PPMd-244K.rle
ULC=$P/life/Unit-Life-Cell-512x512.rle
NEW=$P/presets/hutton32b_8bit_counter.rle
OLD=$P/presets/hutton32_8bit_counter_old.rle
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
  [ -e out/lut_table_Hutton32b.bin ] || $R rulediff --a hutton32b --b table:rules/Hutton32b.rule >/dev/null
}

bgolly() {
  local b=${BGOLLY:-} c
  for c in "/c/Program Files/golly-5.1-win-64bit/bgolly.exe" "$(command -v bgolly)"; do
    [ -z "$b" ] && [ -x "$c" ] && b=$c
  done
  [ -n "$b" ] || { echo "bgolly not found; set BGOLLY" >&2; exit 1; }
  "$b" "$@"
}

series() { REPEATS=5 bash tools/bitslice_series.sh $R "$1"; } # sections of the bit-plane kernel series, 5 runs each

run() {
  case "$1" in
  correctness)
    $R selftest
    $R rulediff --a hutton32  --b table:rules/Hutton32.table      # 0: the CPU rule is the table
    $R rulediff --a hutton32b --b table:rules/Hutton32b.rule      # 0
    $R rulediff --a table:rules/Hutton32.table --b table:rules/Hutton32b.rule ;; # the two rules differ in 74 854
  lut-kernel)
    $R gpu-brute --pattern "$CTR" --gens 65536 --per-submit 1024 --verify --repeats 5
    $R gpu-brute --pattern $NEW --gens 65536 --per-submit 1024 --verify --repeats 5
    $R gpu-brute --pattern $SQ --gens 65536 --per-submit 1024 --verify --repeats 5
    $R gpu-brute --pattern $OLD --reps 8 --gens 2048 --per-submit 64 --verify --repeats 5
    $R gpu-brute --pattern $OLD --reps 24 --gens 512 --per-submit 32 --repeats 5
    for s in "8 8" "16 16" "32 8" "64 4"; do
      set -- $s; $R gpu-brute --pattern $OLD --reps 24 --gens 512 --per-submit 32 --wgx $1 --wgy $2
    done
    $R gpu-brute --pattern $OLD --reps 24 --gens 512 --per-submit 32 --backend dx12 ;;
  lut-content)
    $R gpu-brute2 --side 8192 --gens 64 --pattern $CHA --verify --repeats 5
    $R gpu-brute2 --side 8192 --gens 64 --verify --repeats 5
    $R gpu-brute2 --side 8192 --gens 64 --same-src --verify --repeats 5 ;;
  golly)
    py tools/time_golly.py "$CTR" 1048576 1 8 64 512 4096 32768 262144 --cells 37627 --repeats 3
    py tools/time_golly.py $SQ 65536 8 64 1024 16384 --algo JvN --cells 274950 --repeats 3
    py tools/time_golly.py $CHA 16384 8 64 1024 --cells 19932 --repeats 3 --gc
    # the ROM: the rate over the first minute at each step size (a period needs about four)
    py tools/time_golly.py $ROM 1056 1 8 16 64 --cells 9467605 --timeout 60 ;;
  golly-rom-long) # about 7 minutes: step 1 until Golly stops at its 8 GB limit, past one period
    py tools/time_golly.py $ROM 1056 1 --cells 9467605 --timeout 600 ;;
  golly-tree)
    mkdir -p out/tree
    $R ruletree --rule hutton32b --name Hutton32b --out out/tree/Hutton32b.rule
    py tools/time_golly.py "$CTR" 1048576 1024 262144 --cells 37627 --repeats 3 --rules out/tree/
    py tools/time_golly.py $CHA 16384 64 1024 --cells 19932 --repeats 3 --rules out/tree/
    py tools/time_golly.py $ROM 1056 8 --cells 9467605 --timeout 60 --rules out/tree/ ;;
  runlength) # Golly and the CPU engine (8x8 base tiles) at step 1024 as the run grows; the GPU
             # shared-tile kernel's rate does not depend on the run length (tile target)
    for g in 16384 65536 262144 1048576 4194304 16777216; do
      py tools/head2head.py $g 10 --a 3 --timeout 300 "$CTR" $CHA
    done ;;
  rom-kernels)
    $R gpu-brute --pattern $ROM --gens 528 --per-submit 264 --verify --repeats 5 ;; # --verify also prints the one-core CPU rate
  probe)
    $R gpu-probe --log2-slots 10,14,16,18,19,20,21,22,24,25,26,27,28 # the 8 GiB table needs that much free VRAM
    $R gpu-insert ;;
  miss-rates) # SUMMARY lines: the second half of each run
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
  batches)
    $R root --pattern $P/replicators/Hutton-replicator.rle --k 14 --gens 262144 ;;
  latency)
    $R gpu-latency ;;
  sweep)
    $R gpu-sweep --pattern $ROM --lt 4 --reps 1  --cycles 6600 --per-submit 512 --repeats 5
    $R gpu-sweep --pattern $ROM --lt 5 --reps 1  --cycles 6600 --per-submit 512 --repeats 5
    $R gpu-sweep --pattern $ROM --lt 4 --reps 8  --cycles 66 --repeats 5
    $R gpu-sweep --pattern $ROM --lt 4 --reps 32 --cycles 33 --per-submit 4 --repeats 5
    $R gpu-sweep --pattern $ROM --lt 5 --reps 32 --cycles 33 --per-submit 4 --repeats 5 ;;
  unaligned) # 8x8 copies; class k is the ROM shifted by a pseudo-random cell offset inside a tile
    for k in 1 2 4; do
      $R gpu-sweep --pattern $ROM --lt 4 --reps 8 --cycles 66 --alignments $k --repeats 5
    done ;;
  sparse)
    mkdir -p out/golly
    $R gpu-sparse --pattern "$CTR" --lt 4 --reps 48 --pitch 8 --cycles 1024 --export-mc out/golly/sparse100k.mc --repeats 5
    py tools/time_golly.py out/golly/sparse100k.mc 16384 16 1024 16384 --timeout 600 --repeats 3
    py tools/time_golly.py out/golly/sparse100k.mc 1048576 1024 65536 --timeout 900 --repeats 3
    bgolly -q -q -a RuleLoader -s rules/ -M 8000 -m 16384 -i 1024 -o out/golly/sparse100k_out.mc out/golly/sparse100k.mc
    $R fingerprint --pattern out/golly/sparse100k_out.mc     # Golly's final state
    $R root --pattern out/golly/sparse100k.mc --k 10 --gens 16384 --a 2 ;; # the CPU engine's
  head2head) # 8x8 base tiles, as GHL; the last column compares the final states. The counter and
             # chaotic at 2^24 generations are in the runlength target
    py tools/head2head.py 16777216 10 --a 3 --timeout 300 "$P/misc/Segmented ROM (256k).mc" "$P/16x16 movie/16x16 movie.mc" \
      "$P/misc/serial display (32x32).mc" $P/misc/BIN2BCD.rle "$P/misc/parallel display (16x16).mc" \
      $P/misc/efficient_demultiplexer.mc "$P/misc/weird designs/cpu.mc"
    py tools/head2head.py 65536 10 --a 3 --timeout 300 "$P/32x32 movie/32x32 movie.mc" $SQ
    py tools/head2head.py 65536 12 --a 3 --timeout 300 $P/replicators/Hutton-replicator.rle ;;
  rules-upstream)
    curl -sL --retry 3 https://gollygang.github.io/ruletablerepository/downloads/Hutton32.table \
      | cmp - rules/Hutton32.table && echo "rules/Hutton32.table: same as published"
    curl -sL --retry 3 https://raw.githubusercontent.com/FredericaBernkastel/Hutton32-Computer/master/doc/Hutton32b.rule \
      | cmp - rules/Hutton32b.rule && echo "rules/Hutton32b.rule: same as published" ;;
  sweep-tiles)
    $R gpu-sweep --pattern $ROM --lt 4 --reps 32 --cycles 33 --per-submit 4 --repeats 5
    $R gpu-sweep --pattern $ROM --lt 5 --reps 32 --cycles 33 --per-submit 4 --repeats 5
    for lt in 6 7 8; do
      $R gpu-sweep --pattern $ROM --lt $lt --a 4 --reps 32 --cycles 264 --per-submit 66 --max-learn-gens 40000 --repeats 5
    done ;;
  sweep-ops) # MODEL lines
    for spec in "advanced counter v2.rle|65536" "misc/chaotic.rle|16384" "misc/extra/lotr1.PPMd-244K.rle|528" \
                "misc/extra/lotr1.PPMd-244K.rle|8448" "32x32 movie/32x32 movie.mc|16384" "misc/serial display (32x32).mc|16384"; do
      IFS='|' read -r f g <<< "$spec"
      for lt in 3 4 5 6 7 8; do $R flat --pattern "$P/$f" --lt $lt --a 2 --gens $g --quiet | grep MODEL; done
    done ;;
  floors) # --a 2: 4x4 base tiles, --a 3: 8x8
    for a in 2 3; do
      $R adapt --pattern "$CTR" --gens 16384 --a $a --policies fixed
      $R adapt --compose "$CH8" --gens 8192 --a $a --policies fixed
      $R adapt --pattern $PPMD --gens 512 --a $a --policies fixed
    done ;;
  cond-miss) # per level: requests in column 6, misses in column 7
    for spec in "misc/extra/lotr1.PPMd-244K.rle|512" "advanced counter v2.rle|16384" "misc/chaotic.rle|8192" \
                "32x32 movie/32x32 movie.mc|4096" "misc/squares.rle|4096"; do
      IFS='|' read -r f g <<< "$spec"
      $R root --pattern "$P/$f" --k 6 --gens $g --a 3
    done ;;
  engine-choice)
    bash tools/engine_choice.sh $R 4096 ;;
  controller)
    $R adapt --compose "$(cat tools/composite_spec.txt)" --gens 4096 --a 3 --k 4 --init-floor 3 --policies "16x16,adaptive"
    $R adapt --pattern $PPMD --gens 2112 --a 3 --k 4 --init-floor 3 --policies "16x16,adaptive"
    $R adapt --pattern "$CTR" --gens 16384 --a 3 --k 4 --init-floor 3 --policies "16x16,adaptive"
    $R adapt --compose "$CH8" --gens 8192 --a 3 --k 4 --init-floor 3 --policies "16x16,adaptive" ;;
  region-size)
    bash tools/region_sweep.sh $R ;;
  region-size-life)
    bash tools/region_sweep_life.sh $R ;;
  counter-work)
    $R adapt --pattern "$CTR" --gens 16384 --a 3 ;;
  forms)
    lutbin; venv
    (cd bitslice && "$VPY" scalar.py && "$VPY" tables.py) ;;
  circuit)
    lutbin; venv
    (cd bitslice && "$VPY" verify.py) ;;
  bitplane-content) series A ;;
  bitplane-sizes) series BCDS ;;
  bitplane-limit) series EFS ;;
  window) series HGE ;;
  tile) series T ;;
  cpu-bench)
    lutbin
    cargo run --release --manifest-path bitslice/bench/Cargo.toml
    RUSTFLAGS="-C target-cpu=native" cargo run --release --manifest-path bitslice/bench/Cargo.toml ;; # adds AVX2
  cpu-parallel) # i5-6600K: 4 cores; the ROM fills one 4096^2 grid
    lutbin
    RUSTFLAGS="-C target-cpu=native" cargo run --release --manifest-path bitslice/bench/Cargo.toml -- \
      --threads 4 --size 4096 --pattern $ROM ;;
  reprice)
    bash tools/engine_choice.sh $R 4096 --cell-rate 2.9e11
    $R adapt --pattern "$CTR" --gens 16384 --a 3 --cell-rate 2.9e11 ;;
  figures) # the diagram draws out/overview.json (the reference engine on the patterns), the logs and the two gate networks
    $R overview && py tools/ghl_diagram.py && py tools/figures.py ;;
  life-kernels) # random soups fill the grid; widths are multiples of 32
    $R gpu-life --side 512 --gens 1024 --verify
    $R gpu-life --side 512 --gens 1024 --k 16 --tw 6 --th 32 --verify
    for sg in "256 65536" "1024 8192" "4096 2048" "16384 256" "32768 64"; do
      set -- $sg; $R gpu-life --side $1 --gens $2 --repeats 5
    done
    for sg in "256 65536" "1024 8192"; do
      set -- $sg; $R gpu-life --side $1 --gens $2 --k 16 --tw 6 --th 32 --repeats 5
    done
    $R gpu-life --pattern "$ULC" --margin 64 --gens 11520 --repeats 5
    $R gpu-life --pattern "$ULC" --margin 64 --gens 11520 --k 16 --tw 6 --th 32 --repeats 5 --verify ;;
  life-golly)
    mkdir -p out/life
    $R life-soup --side 2048 --seed 1 --out out/life/soup2048.rle
    py tools/time_golly.py out/life/soup2048.rle 4096 64 1024 --algo HashLife --cells 4194304 --repeats 3 --timeout 120
    py tools/time_golly.py out/life/soup2048.rle 4096 1 64 --algo QuickLife --cells 4194304 --repeats 3 --timeout 120
    py tools/time_golly.py "$ULC" 1048576 1024 65536 1048576 --algo HashLife --cells 261121 --repeats 3
    py tools/time_golly.py "$ULC" 23040 1 64 --algo QuickLife --cells 261121 --repeats 3
    # the soup on the GPU: a margin of 2304 cells keeps every glider and spaceship of 4096 generations
    $R gpu-life --pattern out/life/soup2048.rle --margin 2304 --gens 4096 --repeats 5 --fingerprint
    $R gpu-life --pattern out/life/soup2048.rle --margin 2304 --gens 4096 --k 8 --tw 14 --th 48 --repeats 5 --fingerprint
    bgolly -q -q -a HashLife -M 8000 -m 4096 -i 1024 -o out/life/soup2048_4096.mc out/life/soup2048.rle
    $R fingerprint --pattern out/life/soup2048_4096.mc ;;  # Golly's final state
  life-h2h) # 8x8 base tiles, as for Hutton32b; the last column compares the final states
    py tools/head2head.py 16777216 10 --a 3 --timeout 300 $P/life/metapixel-galaxy.mc "$ULC" $P/life/ruler.mc       $P/life/wedge-grow.mc $P/life/gotts-dots.mc $P/life/metacatacryst.mc
    py tools/head2head.py 1048576 10 --a 3 --timeout 300 $P/life/catacryst.mc $P/life/mosquito5.mc $P/life/jagged.mc       $P/life/totalperiodic.mc ;;
  life-sweep) # GHL's floor for Life is 32x32 windows on 16x16 base tiles (--a 4)
    $R gpu-sweep --pattern "$ULC" --a 4 --lt 5 --reps 1 --cycles 3600 --per-submit 360 --repeats 5
    for lt in 5 6 7 8; do
      $R gpu-sweep --pattern "$ULC" --a 4 --lt $lt --reps 32 --cycles $((230400 >> lt)) --per-submit 64 --repeats 5
    done
    $R gpu-life --pattern "$ULC" --tiled --side 20480 --gens 256 --repeats 5 ;;  # brute force on 32x32 copies
  life-sparse)
    mkdir -p out/life
    $R gpu-sparse --pattern "$ULC" --a 4 --lt 5 --reps 48 --pitch 8 --cycles 512 --export-mc out/life/sparse_ulc.mc --repeats 5
    py tools/time_golly.py out/life/sparse_ulc.mc 16384 1024 16384 --algo HashLife --timeout 600 --repeats 3
    py tools/time_golly.py out/life/sparse_ulc.mc 1048576 1024 65536 --algo HashLife --timeout 900 --repeats 3
    bgolly -q -q -a HashLife -M 8000 -m 16384 -i 1024 -o out/life/sparse_ulc_out.mc out/life/sparse_ulc.mc
    $R fingerprint --pattern out/life/sparse_ulc_out.mc     # Golly's final state
    $R root --pattern out/life/sparse_ulc.mc --k 10 --gens 16384 --a 3 ;; # the CPU engine's
  life-floors) # --policies brute: fixed floors only (plain cells are costed in Table floors)
    $R life-soup --side 512 --seed 1 --out out/life/soup512.rle
    $R adapt --pattern out/life/soup512.rle --gens 1024 --a 3 --policies brute --max-floor 5 --cell-rate 1.39e12
    $R adapt --pattern out/life/soup512.rle --gens 1024 --a 3 --policies brute --max-floor 5
    $R adapt --pattern "$ULC" --gens 11520 --a 3 --policies brute --max-floor 5 --cell-rate 1.39e12 ;;
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
