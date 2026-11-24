"""Produce circuit.netlist (+ step_planes.wgsl / step_planes.rs / STATUS.txt).

Pipeline (every intermediate network is checked against all 2^25 truth-table entries):

  1. rule_circuit.py   hand-structured network (per-cell predicate decode, one block per centre class)
  2. resub.py          global resubstitution with observability don't-cares
  3. ABC               (via yowasp-yosys) AIG restructuring scripts, each followed by step 2 again
  4. xag.map_xag_exact NOT/AND/OR/XOR mapping, inverters minimised by MaxSAT (python-sat)

The best network by total gate count (NOT + AND + OR + XOR) is written out whenever it improves.

usage: python optimize.py [minutes=25] [seed=1]
"""
from __future__ import annotations

import subprocess
import sys
import time

import abc_flow
import resub
import rule_circuit
from abc_flow import COMPRESS2RS, MAP, RESYN2, RESYN2RS, run_abc
from xag import HERE, XAG, map_xag_exact

T0 = time.time()
BUDGET = float(sys.argv[1]) * 60 if len(sys.argv) > 1 else 25 * 60
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
LOG = open(HERE / "optimize.log", "a")


def log(msg):
    line = f"[{time.time() - T0:7.0f}s] {msg}"
    print(line, flush=True)
    LOG.write(line + "\n")
    LOG.flush()


def left():
    return BUDGET - (time.time() - T0)


best = {"total": None, "x": None}


def consider(x: XAG, how: str) -> bool:
    """Map, verify, and write the artifacts if this is the smallest network so far."""
    mapped = map_xag_exact(x, time_limit=20)
    proven = mapped.polarity_optimal
    nl = mapped.renamed()
    c = nl.counts()
    log(f"{how}: xag {x.count()} -> {c} depth {nl.depth()}/{nl.depth(False)}"
        f"{'' if proven else ' (inverter placement not proven optimal)'}")
    if best["total"] is not None and c["total"] >= best["total"]:
        return False
    bad = nl.verify()
    assert not any(bad), f"mismatches {bad}"
    best["total"], best["x"] = c["total"], x
    (HERE / "circuit.netlist").write_text(nl.to_text(
        "Hutton32b (Golly table) bit-sliced netlist. Inputs: bit k of the centre/N/E/S/W state.\n"
        "One gate per line: <name> = NOT a | AND a b | OR a b | XOR a b.  OUTPUT o<k> <signal>.\n"
        f"Produced by optimize.py ({how}); verified against all 2^25 entries of lut_table_Hutton32b.bin."),
        newline="\n")
    subprocess.run([sys.executable, str(HERE / "emit.py")], check=True, capture_output=True)
    (HERE / "STATUS.txt").write_text(
        f"step_planes.wgsl: {c['total']} gates (AND {c['AND']}, OR {c['OR']}, XOR {c['XOR']}, NOT {c['NOT']}), "
        f"depth {nl.depth()}; exhaustive check vs lut_table_Hutton32b.bin: 0 mismatches of 2^25; "
        f"source: {how}\n", newline="\n")
    log(f"  -> new best {c['total']} gates written")
    return True


def resub_until_stuck(x: XAG, seed: int, cap: float) -> XAG:
    eng = resub.Engine(x, seed=seed, log=lambda m: None)
    start = len(eng.gates)
    t = time.time()
    while True:
        remaining = min(cap - (time.time() - t), left())
        if remaining <= 5:
            break
        if eng.sweep(time_limit=remaining) == 0:
            break
    log(f"  resub: {start} -> {len(eng.gates)} nodes in {time.time() - t:.0f}s {eng.stats}")
    return eng.x


def main():
    log(f"=== optimize.py budget {BUDGET / 60:.0f} min, seed {SEED}")
    x = XAG()
    rule_circuit.build(x)
    consider(x, "hand-structured network")
    x = resub_until_stuck(x, SEED, 400)
    consider(x, "hand-structured network + resubstitution")

    scripts = [
        ("resyn2rs", f"strash; {RESYN2RS}; {MAP}"),
        ("compress2rs", f"strash; {COMPRESS2RS}; {MAP}"),
        ("dc2", f"strash; dc2; dc2; {MAP}"),
        ("resyn2", f"strash; {RESYN2}; {MAP}"),
        ("deepsyn", f"strash; &get -n; &deepsyn -T 30 -S {SEED}; &put; {MAP}"),
        ("balance-rewrite", f"strash; balance; rewrite -z; refactor -z; {MAP}"),
    ]
    rnd = 0
    while left() > 60:
        name, script = scripts[rnd % len(scripts)]
        rnd += 1
        base = best["x"]
        try:
            y = run_abc(base, script, f"opt_{name}")
        except Exception as ex:  # a failing ABC script must not lose the result so far
            log(f"ABC {name} failed: {str(ex)[:200]}")
            continue
        log(f"round {rnd}: ABC {name}: {resub.size(base)} -> {resub.size(y)} nodes ({y.abc_seconds:.0f}s)")
        if left() < 30:
            break
        y = resub_until_stuck(y, SEED + rnd, 240)
        consider(y, f"hand-structured network + resubstitution + ABC {name} + resubstitution (round {rnd})")
    log(f"=== done: best {best['total']} gates")


if __name__ == "__main__":
    main()
