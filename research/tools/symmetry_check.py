"""Is a 32-state von Neumann rule symmetric under the square's rotations and reflections once its
directional states are relabelled? Golly's `symmetries:` field cannot say so: it permutes the
neighbours' positions only, and both Hutton tables declare `symmetries:none`.

For a transform T of the grid with state map P, the rule f is equivariant if, on all 2^25
neighbourhoods, f(T(neighbourhood) with P applied to every cell) == P(f(neighbourhood)). Then a
tile and its transformed copy evolve into each other's transforms, and a memo may share them.
Mismatches are split by whether the neighbourhood or the result holds a sensitised state (1-8),
the states of construction, whose bit sequences name absolute directions.

Usage (from research/, numpy needed: out/bitslice-venv): python tools/symmetry_check.py [LUT...]
LUT index: w | s << 5 | e << 10 | n << 15 | c << 20 (rule.rs)."""
import sys

import numpy as np

LUTS = sys.argv[1:] or ["out/lut_table_Hutton32.bin", "out/lut_table_Hutton32b.bin"]

# directional states: four groups (OTS quiescent, OTS excited, STS quiescent, STS excited) of
# E, N, W, S = base + 0, 1, 2, 3; crossings 29 (horizontal excitation) and 30 (vertical)
GROUPS = (9, 13, 17, 21)


def state_map(dir_map, swap_crossings):
    p = np.arange(32, dtype=np.uint8)
    for b in GROUPS:
        for d in range(4):
            p[b + d] = b + dir_map[d]
    if swap_crossings:
        p[29], p[30] = 30, 29
    return p


E, N, W, S = 0, 1, 2, 3
# each transform: name, state map on directions, whether horizontal and vertical swap, and where
# the new neighbourhood's n, s, e, w come from (in terms of the old one)
TRANSFORMS = [
    ("rotate 90", {E: N, N: W, W: S, S: E}, True, ("e", "w", "s", "n")),     # counter-clockwise: east cell moves north
    ("rotate 180", {E: W, N: S, W: E, S: N}, False, ("s", "n", "w", "e")),
    ("rotate 270", {E: S, N: E, W: N, S: W}, True, ("w", "e", "n", "s")),
    ("mirror E-W", {E: W, N: N, W: E, S: S}, False, ("n", "s", "w", "e")),
    ("mirror N-S", {E: E, N: S, W: W, S: N}, False, ("s", "n", "e", "w")),
    ("transpose", {E: S, N: W, W: N, S: E}, True, ("w", "e", "s", "n")),     # reflect across the NW-SE diagonal
    ("anti-transpose", {E: N, N: E, W: S, S: W}, True, ("e", "w", "n", "s")),
]

idx = np.arange(1 << 25, dtype=np.uint32)
cells = {"w": idx & 31, "s": (idx >> 5) & 31, "e": (idx >> 10) & 31, "n": (idx >> 15) & 31, "c": idx >> 20}
sens = np.zeros(32, bool)
sens[1:9] = True
touches_construction = sens[cells["c"]] | sens[cells["n"]] | sens[cells["s"]] | sens[cells["e"]] | sens[cells["w"]]

for path in LUTS:
    lut = np.fromfile(path, dtype=np.uint8)
    assert lut.size == 1 << 25, path
    touches = touches_construction | sens[lut]
    print(f"== {path}: {int(touches.sum()):,} of {1 << 25:,} neighbourhoods touch a sensitised state")
    for name, dmap, swap, src in TRANSFORMS:
        p = state_map(dmap, swap)
        # the transform's own state map must also fix the geometry it claims: check it is a bijection
        assert len(set(p.tolist())) == 32
        n2, s2, e2, w2 = (p[cells[k]].astype(np.uint32) for k in src)
        c2 = p[cells["c"]].astype(np.uint32)
        j = w2 | s2 << 5 | e2 << 10 | n2 << 15 | c2 << 20
        bad = lut[j] != p[lut]
        nb, nt = int(bad.sum()), int((bad & ~touches).sum())
        print(f"  {name:15} mismatches {nb:10,}   without a sensitised state {nt:10,}")
        if nt:
            k = int(np.flatnonzero(bad & ~touches)[0])
            cc = {x: int(cells[x][k]) for x in "cnsew"}
            print(f"      e.g. c n s e w = {cc['c']} {cc['n']} {cc['s']} {cc['e']} {cc['w']} -> {int(lut[k])}; "
                  f"transformed -> {int(lut[j[k]])}, expected {int(p[lut[k]])}")
