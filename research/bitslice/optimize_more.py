"""Optional continuation of optimize.py: start from the current circuit.netlist and alternate
ABC `&deepsyn` (the one ABC script that helped after resubstitution) with resub.py.

usage: python optimize_more.py [minutes=6] [seed=1]
"""
from __future__ import annotations

import optimize as O
import resub
from abc_flow import MAP, run_abc
from xag import HERE, Netlist


def main():
    nl = Netlist.from_text((HERE / "circuit.netlist").read_text())
    assert not any(nl.verify())
    x = nl.to_xag()
    O.best["total"], O.best["x"] = nl.counts()["total"], x
    O.log(f"=== optimize_more.py: start from circuit.netlist, {O.best['total']} gates, {resub.size(x)} nodes")
    rnd = 0
    while O.left() > 90:
        rnd += 1
        base = O.best["x"]
        secs = int(max(20, min(60, O.left() / 4)))
        try:
            y = run_abc(base, f"strash; &get -n; &deepsyn -T {secs} -S {O.SEED + rnd}; &put; {MAP}", f"more_{rnd}")
        except Exception as ex:
            O.log(f"ABC deepsyn failed: {str(ex)[:200]}")
            break
        O.log(f"extra round {rnd}: ABC deepsyn -T {secs}: {resub.size(base)} -> {resub.size(y)} nodes ({y.abc_seconds:.0f}s)")
        if O.left() < 40:
            break
        y = O.resub_until_stuck(y, 100 + rnd, 200)
        O.consider(y, "hand-structured network + resubstitution + ABC (resyn2, deepsyn) + resubstitution "
                      f"(extra round {rnd})")
    O.log(f"=== done: best {O.best['total']} gates")


if __name__ == "__main__":
    main()
