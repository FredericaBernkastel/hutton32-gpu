#!/usr/bin/env bash
# Makes circuits like those in bitslice/compiled/: the two 32-state rules compiled from their
# tables and optimised, and the optimiser started from the 367-gate circuit. The search is bounded
# by time, so the circuits differ from run to run and with the load of the machine; each is equal
# to its table on all 2^25 neighbourhoods all the same. They go to out/ruletable2netlist/optimise;
# to use one, copy it into bitslice/compiled/ and run `rule-circuits`, which checks what is there.
# About 16 minutes with the default of 300 s each. A longer run gives a smaller circuit, and so
# does a further run from a result with another seed (--from NETLIST --seed N): the circuits kept
# in bitslice/compiled/ were made that way, in runs of 10 to 40 minutes.
# usage (from research/): bash tools/rule_circuits_optimise.sh <ruletable2netlist exe> [SECONDS]
X="$1"; S="${2:-300}"
D="$(dirname "$0")"
OUT="$D/../out/ruletable2netlist/optimise"; mkdir -p "$OUT/from367"
echo "=== the two tables compiled and optimised ($S s each)"
"$X" "$D/../rules/Hutton32b.rule" --seconds "$S" --quiet --out "$OUT" --lut "$D/../out/lut_table_Hutton32b.bin"
"$X" "$D/../rules/Hutton32.table" --seconds "$S" --quiet --out "$OUT" --lut "$D/../out/lut_table_Hutton32.bin"
echo "=== the optimiser started from the 367-gate circuit ($S s)"
"$X" "$D/../rules/Hutton32b.rule" --seconds "$S" --quiet --from "$D/../bitslice/circuit.netlist" --out "$OUT/from367" --lut "$D/../out/lut_table_Hutton32b.bin"
