"""Form C, "store cells pre-classified": is there a ONE-BYTE cell code q(state) such that the class
of a cell as seen from each side is a contiguous bit-field of q?

If so, a class-indexed table (5 + 4*4 = 21 index bits, 2 MiB) needs no per-neighbour mapping
at all: the index is built from shifts and masks of the stored bytes, exactly like today's
25-bit index, and the table emits the next cell already in that code.

Requirement: for each neighbour position d a 4-bit window of q must take different values on
states in different d-classes (15 classes), and a 5-bit window must separate the 29 centre
classes.  Solved as SAT (python-sat) for every choice of window positions until one works.

usage: python byte_code.py [seconds=120]     (writes byte_code.txt if a code is found)

Status: INCONCLUSIVE.  A 150 s run of an earlier version of this script (which also tried
layouts where two directions share a window) left all 28 layouts it reached undecided; the
search over the 480 layouts below was started and stopped before it finished.
"""
from __future__ import annotations

import itertools
import sys
import threading
import time

import numpy as np
from pysat.solvers import Solver

from xag import HERE, LUT_PATH

NAMES = "cnesw"


def classes(G5, pos):
    ax = NAMES.index(pos)
    m = np.moveaxis(G5, ax, 0).reshape(32, -1)
    ids, lab = {}, []
    for v in range(32):
        lab.append(ids.setdefault(m[v].tobytes(), len(ids)))
    return lab


def solve(cls, windows, budget):
    """windows: pos -> (lowest bit, width). Returns list of 32 codes or None / 'timeout'."""
    def var(s, b):
        return s * 8 + b + 1
    nxt = [32 * 8]
    dvar = {}
    s = Solver(name="g4")
    for b in range(8):
        s.add_clause([-var(0, b)])            # ground state = 0x00 (the zero ring stays zero)

    def diff(x, y, b):
        key = (x, y, b)
        if key not in dvar:
            nxt[0] += 1
            d = dvar[key] = nxt[0]
            s.add_clause([-d, var(x, b), var(y, b)])
            s.add_clause([-d, -var(x, b), -var(y, b)])
        return dvar[key]

    for pos, (lo, width) in windows.items():
        lab = cls[pos]
        for x in range(32):
            for y in range(x + 1, 32):
                if lab[x] != lab[y]:
                    s.add_clause([diff(x, y, b) for b in range(lo, lo + width)])
    timer = threading.Timer(budget, s.interrupt)
    timer.start()
    res = s.solve_limited(expect_interrupt=True)
    timer.cancel()
    out = None
    if res:
        model = set(l for l in s.get_model() if l > 0)
        out = [sum(1 << b for b in range(8) if var(st, b) in model) for st in range(32)]
    elif res is None:
        out = "timeout"
    s.delete()
    return out


def main():
    total = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
    G5 = np.fromfile(LUT_PATH, dtype=np.uint8).reshape(32, 32, 32, 32, 32)
    cls = {p: classes(G5, p) for p in NAMES}
    t0 = time.time()
    tried = timeouts = 0
    # two directions cannot share a window (their joint partition has more than 16 classes),
    # so only layouts with four different window positions are worth trying
    layouts = [(pc, perm) for perm in itertools.permutations(range(5), 4) for pc in range(4)]
    for pc, (pn, pe, ps, pw) in layouts:
        if time.time() - t0 > total:
            print(f"no code found in {total:.0f}s ({tried} of {len(layouts)} window layouts tried, "
                  f"{timeouts} of them undecided within the per-layout limit)")
            return
        windows = {"c": (pc, 5), "n": (pn, 4), "e": (pe, 4), "s": (ps, 4), "w": (pw, 4)}
        tried += 1
        code = solve(cls, windows, 2.0)
        if code == "timeout":
            timeouts += 1
            continue
        if code is None:
            continue
        if True:
            # independent check of the separation property
            for pos, (lo, width) in windows.items():
                field = [(q >> lo) & ((1 << width) - 1) for q in code]
                by_field = {}
                for st in range(32):
                    assert by_field.setdefault(field[st], cls[pos][st]) == cls[pos][st], "window does not separate"
            assert len(set(code)) == 32 and code[0] == 0
            lines = [
                "# One-byte cell code for Hutton32b in which every class is a bit-field (found by byte_code.py).",
                "# windows (lowest bit, width): " + ", ".join(f"{p}:{windows[p]}" for p in NAMES),
                "# table index = c_field | n_field<<5 | e_field<<9 | s_field<<13 | w_field<<17  (21 bits, 2 MiB of bytes)",
                "# state code(hex)",
            ] + [f"{st} 0x{q:02x}" for st, q in enumerate(code)]
            (HERE / "byte_code.txt").write_text("\n".join(lines) + "\n", newline="\n")
            print(f"found after {tried} layouts ({time.time() - t0:.1f}s): windows {windows}")
            print("codes:", " ".join(f"{q:02x}" for q in code))
            return
    print(f"no window layout admits a one-byte code ({tried} layouts, {timeouts} undecided)")


if __name__ == "__main__":
    main()
