#!/usr/bin/env bash
# Rule tables as gate networks, checked. Nothing here depends on time or on the machine's load.
#  1. The circuits in bitslice/compiled/ (compiled from the tables of Hutton32b and Hutton32 by
#     ../ruletable2netlist and optimised; tools/rule_circuits_optimise.sh makes such circuits) and
#     the 367-gate circuit: each is read back and compared with its rule table on all 2^25
#     neighbourhoods, with the table of next states the harness computes, and on a soup with Golly.
#  2. Every rule table Golly ships, compiled to the network as built from its transitions (no
#     simplification, no optimisation), compared with the table on every neighbourhood where the
#     rule has at most 2^30 of them and on four million drawn at random plus those its transitions
#     name otherwise; a soup is run by the table, by the network and by Golly.
#  In between, the two tables of step 1 compiled as built after the simplification of their
#  transitions (which runs to its end here, so that the result does not depend on the load).
# A soup is a 40x40 square of uniform random states run for 16 generations. About ten minutes,
# most of them for the two rules with about 2^30 neighbourhoods.
# usage (from research/): bash tools/rule_circuits.sh <ruletable2netlist exe> circuits|tables [golly rules dir]
#   circuits: step 1 and the two tables as built; tables: step 2
X="$1"; WHAT="$2"
D="$(dirname "$0")"
B=${BGOLLY:-}
for c in "/c/Program Files/golly-5.1-win-64bit/bgolly.exe" "$(command -v bgolly)"; do
  [ -z "$B" ] && [ -x "$c" ] && B=$c
done
G=(); [ -n "$B" ] && G=(--golly "$B")
RULES="${3:-$(dirname "${B:-/nonexistent/x}")/Rules}"
C="$D/../bitslice/compiled"

if [ "$WHAT" = circuits ]; then
echo "=== circuits read back and compared with their rule tables"
"$X" "$D/../rules/Hutton32b.rule" --quiet --check "$C/Hutton32b.netlist" --lut "$D/../out/lut_table_Hutton32b.bin" "${G[@]}"
"$X" "$D/../rules/Hutton32.table" --quiet --check "$C/Hutton32.netlist" --lut "$D/../out/lut_table_Hutton32.bin" "${G[@]}"
[ -e "$C/Hutton32b-from367.netlist" ] && "$X" "$D/../rules/Hutton32b.rule" --quiet --check "$C/Hutton32b-from367.netlist" --lut "$D/../out/lut_table_Hutton32b.bin" "${G[@]}"
"$X" "$D/../rules/Hutton32b.rule" --quiet --check "$D/../bitslice/circuit.netlist" --lut "$D/../out/lut_table_Hutton32b.bin" "${G[@]}"
echo "=== the two tables compiled as built, their transitions simplified first"
"$X" "$D/../rules/Hutton32b.rule" --quiet --raw --lut "$D/../out/lut_table_Hutton32b.bin"
"$X" "$D/../rules/Hutton32.table" --quiet --raw --lut "$D/../out/lut_table_Hutton32.bin"
elif [ -d "$RULES" ]; then
  echo "=== the rule tables Golly ships, compiled as built"
  for f in "$RULES"/*.rule; do
    grep -q '^@TABLE' "$f" || continue
    "$X" "$f" --quiet --raw --no-simplify "${G[@]}" 2>&1
  done
else
  echo "=== the rule tables Golly ships: skipped (no Rules folder at $RULES)"
fi
