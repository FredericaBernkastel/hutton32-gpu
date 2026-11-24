"""Form C: how small can a table-driven evaluation get?  Everything is computed from
research/out/lut_table_Hutton32b.bin (index = w | s<<5 | e<<10 | n<<15 | c<<20).

For a set B of positions, the *column multiplicity* mu(B) is the number of distinct functions
of the remaining positions obtained by fixing the positions in B in all possible ways.  A table
T1[B] -> id (mu(B) ids) followed by T2[id, rest] is then an exact two-stage evaluation.

usage: python tables.py            (prints a report; takes well under a minute)
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from xag import LUT_PATH

NAMES = "cnesw"   # axis order of the reshaped table
L1 = 128 * 1024   # the "128 KB L1" of the task statement
L2_NOTE = "L2"


def rows_unique(a: np.ndarray):
    """(number of distinct rows, inverse index) of a 2-D uint8 array."""
    a = np.ascontiguousarray(a)
    v = a.view(np.dtype((np.void, a.shape[1]))).ravel()
    _, inv = np.unique(v, return_inverse=True)
    return int(inv.max()) + 1, inv


def mu(G5, bound: str):
    axes = [NAMES.index(p) for p in bound]
    rest = [i for i in range(5) if i not in axes]
    m = np.transpose(G5, axes + rest).reshape(32 ** len(axes), -1)
    return rows_unique(m)


def fmt_bytes(b):
    if b < 1024:
        return f"{b} B"
    if b < 1024 * 1024:
        return f"{b / 1024:.1f} KiB"
    return f"{b / 1024 / 1024:.2f} MiB"


def fits(b):
    return "fits 128 KB L1" if b <= L1 else ("L2 only" if b <= 32 * 1024 * 1024 else "exceeds a 32 MB L2")


def idbytes(k):
    return 1 if k <= 256 else (2 if k <= 65536 else 4)


def main():
    G = np.fromfile(LUT_PATH, dtype=np.uint8)
    assert G.size == 1 << 25
    G5 = G.reshape(32, 32, 32, 32, 32)   # [c][n][e][s][w]

    print("== 1. classes per position (states the rule cannot tell apart at that position)")
    cls = {}
    for p in NAMES:
        k, inv = mu(G5, p)
        cls[p] = (k, inv)
        print(f"   {p}: {k} classes")
    k_c = cls["c"][0]
    dense = k_c * math.prod(cls[p][0] for p in "nesw")
    print(f"   single class-indexed table: {k_c} x 15^4 = {dense} entries dense "
          f"({fmt_bytes(dense)}), or 21 index bits = {fmt_bytes(1 << 21)} bit-packed  [{fits(1 << 21)}]")

    print("\n== 2. column multiplicity mu(B) for every set B of positions")
    MU = {}
    for r in range(1, 5):
        for combo in itertools.combinations(NAMES, r):
            b = "".join(combo)
            MU[b] = mu(G5, b)[0]
    for b, k in MU.items():
        print(f"   mu({b:4s}) = {k:7d}   ({math.ceil(math.log2(k)) if k > 1 else 0} bits)")

    print("\n== 3. neighbour pairs and the neighbour 4-tuple (what the task asked for)")
    for a, b in (("ns", "ew"), ("ne", "sw"), ("nw", "es")):
        ka, kb = MU[a], MU[b]
        ba, bb = math.ceil(math.log2(ka)), math.ceil(math.log2(kb))
        fin_dense = 32 * ka * kb
        fin_packed = 1 << (5 + ba + bb)
        st1 = 1024 * idbytes(ka) + 1024 * idbytes(kb)
        print(f"   pairs ({a}),({b}): {ka} and {kb} distinct behaviours "
              f"-> final table T[c,{a},{b}]: {5 + ba + bb} index bits = {fmt_bytes(fin_packed)} bit-packed, "
              f"{fmt_bytes(fin_dense)} dense; first-stage tables {fmt_bytes(st1)}  [{fits(fin_dense + st1)}]")
    k4 = MU["nesw"]
    print(f"   4-tuple (n,e,s,w): {k4} distinct behaviours ({math.ceil(math.log2(k4))} bits) "
          f"-> two-stage: T1[n,e,s,w] 2^20 x {idbytes(k4)} B = {fmt_bytes((1 << 20) * idbytes(k4))} "
          f"(or 15^4 x {idbytes(k4)} B = {fmt_bytes(15 ** 4 * idbytes(k4))} if the four cells arrive as class "
          f"codes), T2[c,id] = 32 x {k4} = {fmt_bytes(32 * k4)}")

    print("\n== 4. best exact schemes found by enumeration (raw 5-bit states as index parts)")
    # (a) two tables in sequence: T1[B] -> id, T2[id, rest]
    res = []
    for b, k in MU.items():
        rest = 5 - len(b)
        t1 = 32 ** len(b) * idbytes(k)
        t2 = k * 32 ** rest
        res.append((t1 + t2, b, k, t1, t2))
    res.sort()
    print("   (a) T1[B] -> id ; out = T2[id, rest]      (2 dependent loads)")
    for tot, b, k, t1, t2 in res[:6]:
        print(f"       B={b:4s} mu={k:6d}: T1 {fmt_bytes(t1):>10s} + T2 {fmt_bytes(t2):>10s} = {fmt_bytes(tot):>10s}  [{fits(tot)}]")
    # (b) two first-stage tables in parallel, then the final one
    res = []
    subsets = list(MU.keys())
    for b1 in subsets:
        for b2 in subsets:
            if b1 >= b2 or set(b1) & set(b2):
                continue
            rest = 5 - len(b1) - len(b2)
            k1, k2 = MU[b1], MU[b2]
            t1 = 32 ** len(b1) * idbytes(k1) + 32 ** len(b2) * idbytes(k2)
            t3 = k1 * k2 * 32 ** rest
            res.append((t1 + t3, b1, b2, k1, k2, t1, t3))
    res.sort()
    print("   (b) a = T1[B1], b = T2[B2] (independent) ; out = T3[a, b, rest]      (latency of 2 loads)")
    for tot, b1, b2, k1, k2, t1, t3 in res[:8]:
        print(f"       B1={b1:3s} ({k1:5d}) B2={b2:3s} ({k2:5d}): first stage {fmt_bytes(t1):>10s} + T3 {fmt_bytes(t3):>10s} "
              f"= {fmt_bytes(tot):>10s}  [{fits(tot)}]")

    print("\n== 5. schemes whose first-stage tables also see the centre (classes *given* c)")
    # per-centre multiplicities
    per_c_single = {p: [] for p in "nesw"}
    for c in range(32):
        F = G5[c]
        for i, p in enumerate("nesw"):
            m = np.moveaxis(F, i, 0).reshape(32, -1)
            per_c_single[p].append(rows_unique(m)[0])
    cases = [math.prod(per_c_single[p][c] for p in "nesw") for c in range(32)]
    print(f"   classes of one neighbour given c: max over c = "
          + ", ".join(f"{p}:{max(per_c_single[p])}" for p in "nesw"))
    print(f"   distinct cases sum_c prod_d k_d(c) = {sum(cases)}   (the whole rule is {sum(cases)} cases)")
    t1 = 4 * 1024 * 2
    print(f"   (c) four tables T_d[c, x_d] (1024 x u16 each, pre-scaled mixed-radix digit) ; out = T5[sum]: "
          f"{fmt_bytes(t1)} + {fmt_bytes(sum(cases))} = {fmt_bytes(t1 + sum(cases))}  [{fits(t1 + sum(cases))}]"
          f"   (4 independent loads + 3 adds + 1 dependent load)")
    for a, b in (("ns", "ew"), ("ne", "sw"), ("nw", "es")):
        ka, kb = [], []
        for c in range(32):
            F = G5[c]  # [n][e][s][w]
            ax = ["nesw".index(p) for p in a]
            bx = ["nesw".index(p) for p in b]
            m = np.transpose(F, ax + bx).reshape(1024, 1024)
            ka.append(rows_unique(m)[0])
            kb.append(rows_unique(m.T)[0])
        fin = sum(x * y for x, y in zip(ka, kb))
        ida, idb = idbytes(fin), idbytes(max(kb))
        t1 = 32768 * ida + 32768 * idb
        print(f"   (d) a = T1[c,{a}], b = T2[c,{b}] (2^15 entries each) ; out = T3[a + b]: pair classes given c: "
              f"max {max(ka)} / {max(kb)}; T3 = sum_c k_{a}(c)*k_{b}(c) = {fin} entries; "
              f"total {fmt_bytes(t1 + fin)}  [{fits(t1 + fin)}]")

    print("\n== 6. chain T1[c,n] -> T2[.,e] -> T3[.,s] -> T4[.,w]  (4 dependent loads), all orders of the neighbours")
    best = None
    for order in itertools.permutations("nesw"):
        tot, desc, b = 0, [], "c"
        ok = True
        prev = 32
        for i, p in enumerate(order):
            b2 = "".join(sorted(b + p, key=NAMES.index))
            k = MU[b2] if len(b2) < 5 else 32
            size = prev * 32 * (idbytes(k) if len(b2) < 5 else 1)
            tot += size
            desc.append(f"{prev}x32")
            prev = k
            b = b2
        if best is None or tot < best[0]:
            best = (tot, order, desc)
    print(f"   best order c,{','.join(best[1])}: tables {' + '.join(best[2])} entries = {fmt_bytes(best[0])}  [{fits(best[0])}]")


if __name__ == "__main__":
    main()
