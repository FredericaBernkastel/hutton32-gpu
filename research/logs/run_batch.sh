#!/usr/bin/env bash
# Runs reproduce.sh targets one after another, one log per target, on an otherwise idle machine.
# usage (from research/): bash logs/run_batch.sh <log dir> <target>...
dir="$1"; shift
mkdir -p "$dir"
for t in "$@"; do
  echo "$(date '+%F %T') start $t" >> "$dir/batch.txt"
  bash reproduce.sh "$t" > "$dir/$t.txt" 2>&1
  echo "$(date '+%F %T') done  $t (exit $?)" >> "$dir/batch.txt"
done
