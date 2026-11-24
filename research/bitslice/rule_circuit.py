"""Structure-aware gate network for Hutton32b (the Golly-table variant, lut_table_Hutton32b.bin).

The network is written by hand from the semantics of the rule and is only *accepted* by the
exhaustive check against the truth table (see `build_and_check`).  Layout:

  1. per-neighbour decode: the handful of predicates the rule looks at for each direction
     (arrow pointing at us, excited OTS/STS arrow at us, confluent output towards us, ...);
  2. centre decode (ground / sensitised / OTS / STS / confluent / crossing, direction, excitation);
  3. one block per centre class, each producing its contribution to the five output bits;
  4. OR of the contributions (the classes are mutually exclusive).

State encoding: 0 ground, 1-8 sensitised, 9-12 OTS quiescent E,N,W,S, 13-16 OTS excited,
17-20 STS quiescent, 21-24 STS excited, 25-28 confluent (25 + 2*now + next), 29/30/31 crossing
with horizontal / vertical / both signals.
"""
from __future__ import annotations

import sys

import numpy as np

from xag import XAG, Sim, map_xag_exact


def build(x: XAG) -> list[int]:
    AND, OR, XOR, NOT, MUX = x.AND, x.OR, x.XOR, x.NOT, x.MUX
    C = [x.inp(f"c{k}") for k in range(5)]
    N = [x.inp(f"n{k}") for k in range(5)]
    E = [x.inp(f"e{k}") for k in range(5)]
    S = [x.inp(f"s{k}") for k in range(5)]
    W = [x.inp(f"w{k}") for k in range(5)]

    # ------------------------------------------------------------------------------------------
    # 1. neighbour decode
    # ------------------------------------------------------------------------------------------
    def decode(b, lo1, lo0, vertical, is_north):
        """Predicates of one neighbour. `lo1 lo0` are the low state bits of an arrow in that
        cell that points at the centre (E=01 N=10 W=11 S=00)."""
        b0, b1, b2, b3, b4 = b
        l1 = b1 if lo1 else NOT(b1)
        l0 = b0 if lo0 else NOT(b0)
        lo = AND(l1, l0)
        d = {}
        if not is_north:
            # arrows E/N/W: upper bits b4 b3 b2 = 010 OTSq, 011 OTSx, 100 STSq, 101 STSx
            x43 = XOR(b4, b3)
            d["IN"] = AND(lo, x43)                     # any arrow pointing at us
            d["A"] = AND(d["IN"], b2)                  # excited arrow pointing at us
            d["EO"] = AND(d["A"], b3)                  # excited OTS arrow at us
            d["ES"] = AND(d["A"], b4)                  # excited STS arrow at us
            d["IO"] = AND(d["IN"], b3)                 # OTS arrow at us (either excitation)
            d["QO"] = AND(d["IO"], NOT(b2))            # quiescent OTS arrow at us
            z3 = AND(b3, NOT(b2), NOT(b1), NOT(b0))    # state 8 or 24
            d["OUT"] = AND(NOT(lo), XOR(x43, z3))      # transmission state not pointing at us
        else:
            # arrows S: upper bits 011 OTSq(12), 100 OTSx(16), 101 STSq(20), 110 STSx(24)
            d["IN"] = AND(lo, XOR(b4, AND(b3, b2)))
            d["A"] = AND(lo, b4, NOT(b2))
            d["EO"] = AND(d["A"], NOT(b3))
            d["ES"] = AND(d["A"], b3)
            d["QO"] = AND(lo, NOT(b4), b3, b2)
            d["IO"] = OR(d["QO"], d["EO"])
            d["OUT"] = AND(NOT(lo), XOR(b4, b3))
        # confluent state whose output reaches us: 27, 28, 31 and 30 (vertical) / 29 (horizontal)
        other = b0 if vertical else b1
        d["CX"] = AND(b4, b3, OR(AND(b1, b0), AND(b2, NOT(other))))
        d["Z"] = NOT(OR(b4, b3, b2, b1, b0))           # ground
        d["EC"] = OR(d["EO"], d["CX"])                 # ordinary excitation arriving from here
        d["SC"] = OR(d["ES"], d["CX"])                 # special excitation arriving from here
        return d

    dn = decode(N, 0, 0, True, True)
    de = decode(E, 1, 1, False, False)
    ds = decode(S, 1, 0, True, False)
    dw = decode(W, 0, 1, False, False)
    D = {"n": dn, "e": de, "s": ds, "w": dw}

    x.mark("neighbour decode (4 cells)")
    v_on = OR(dn["EO"], ds["EO"])           # excited OTS arrow from north/south
    h_on = OR(de["EO"], dw["EO"])           # ... from east/west
    EOany = OR(v_on, h_on)
    ESany = OR(dn["ES"], de["ES"], ds["ES"], dw["ES"])
    IOany = OR(dn["IO"], de["IO"], ds["IO"], dw["IO"])
    QOany = OR(dn["QO"], de["QO"], ds["QO"], dw["QO"])
    Zany = OR(dn["Z"], de["Z"], ds["Z"], dw["Z"])
    nES = NOT(ESany)
    nEO = NOT(EOany)

    # excited arrow pointing at us, priority W, S, E, N -> direction of that arrow (E=0 N=1 W=2 S=3)
    Aw, As, Ae, An = dw["A"], ds["A"], de["A"], dn["A"]
    has = OR(Aw, As, Ae, An)
    d0 = AND(NOT(Aw), OR(As, AND(An, NOT(Ae))))
    d1 = AND(NOT(Aw), NOT(As), OR(Ae, An))
    # the selected arrow is special (only read when an excited OTS arrow exists as well)
    iS = OR(dw["ES"], AND(NOT(Aw), OR(ds["ES"], AND(NOT(As), de["ES"]))))

    x.mark("neighbour aggregates, input direction")
    # ------------------------------------------------------------------------------------------
    # 2. centre decode
    # ------------------------------------------------------------------------------------------
    c0, c1, c2, c3, c4 = C
    nz = OR(c1, c0)
    cS = NOT(nz)                            # low bits 00
    lowE = AND(NOT(c1), c0)
    lowN = AND(c1, NOT(c0))
    lowW = AND(c1, c0)
    low = {"n": lowN, "e": lowE, "s": cS, "w": lowW}
    h00 = NOT(OR(c4, c3))                   # 0..7
    h00lo = AND(h00, NOT(c2))               # 0..3
    h00hi = AND(h00, c2)                    # 4..7
    isZero = AND(h00lo, cS)
    s123 = AND(h00lo, nz)
    is4 = AND(h00hi, cS)
    is5 = AND(h00hi, lowE)
    is6 = AND(h00hi, lowN)
    is7 = AND(h00hi, lowW)
    is8 = AND(NOT(c4), c3, NOT(c2), cS)
    cOTS = OR(AND(nz, NOT(c4), c3), AND(cS, OR(AND(NOT(c4), c3, c2), AND(c4, NOT(c3), NOT(c2)))))
    cSTS = OR(AND(nz, c4, NOT(c3)), AND(cS, c4, XOR(c3, c2)))
    cX = XOR(c2, cS)                        # excited (valid for transmission states)
    c43 = AND(c4, c3)
    c2528 = AND(c43, XOR(c2, nz))
    c2931 = AND(c43, c2, nz)
    is25 = AND(c2528, lowE)

    x.mark("centre decode")
    # ------------------------------------------------------------------------------------------
    # 3a. transmission states (9..24)
    # ------------------------------------------------------------------------------------------
    # state of the cell the centre points at
    o = [OR(AND(lowN, N[k]), AND(lowE, E[k]), AND(cS, S[k]), AND(lowW, W[k])) for k in range(5)]
    x.mark("pointed-at cell: 5-bit 4:1 mux")
    o_h0 = NOT(OR(o[4], o[3]))
    o_lo = OR(o[1], o[0])
    o_nlo = NOT(o_lo)
    o0123 = AND(o_h0, NOT(o[2]))
    o4567 = AND(o_h0, o[2])
    oZero = AND(o0123, o_nlo)
    o4 = AND(o4567, o_nlo)
    o7 = AND(o4567, o[1], o[0])
    o8 = AND(NOT(o[4]), o[3], NOT(o[2]), o_nlo)
    o1720 = AND(o[4], NOT(o[3]), XOR(o[2], o_lo))
    o1234 = OR(AND(o0123, o_lo), o4)
    o5678 = OR(AND(o4567, o_lo), o8)
    x.mark("pointed-at cell: predicates")
    # excitation arriving from a cell we do not point at
    INo = OR(*[AND(D[d]["EC"], NOT(low[d])) for d in "nesw"])
    SSo = OR(*[AND(D[d]["SC"], NOT(low[d])) for d in "nesw"])

    x.mark("excitation from the other three cells")
    # ordinary transmission state
    p_ret = OR(o8, o1720, AND(o4567, NOT(o[1]), XOR(o[0], cX)))      # 8, 17..20, 4 if excited else 5
    x7 = AND(cX, o7)
    killO = OR(ESany, AND(INo, p_ret), AND(x7, NOT(INo)))
    oneO = AND(cOTS, nES, INo, x7, NOT(p_ret))
    liveO = AND(cOTS, NOT(killO), NOT(AND(INo, x7)))
    x0123 = AND(cX, o0123)
    tOx_O = AND(liveO, INo)
    tSx_O = AND(liveO, NOT(INo), x0123)
    tOq_O = AND(liveO, NOT(INo), NOT(x0123))

    x.mark("OTS centre")
    # special transmission state
    mark = AND(cX, IOany)
    s_sens = AND(mark, o1234)
    s_fin = AND(mark, o5678)
    xo0 = AND(cX, oZero)
    plain = AND(NOT(s_sens), NOT(s_fin), OR(xo0, nEO))
    tOx_S = AND(cSTS, s_sens, EOany)
    tOq_S = AND(cSTS, s_fin, nEO)
    tSx_S = AND(cSTS, OR(AND(s_sens, nEO), AND(plain, SSo)))
    tSq_S = AND(cSTS, plain, NOT(SSo))

    x.mark("STS centre")
    tOq = OR(tOq_O, tOq_S)
    tOx = OR(tOx_O, tOx_S)
    tSq = tSq_S
    tSx = OR(tSx_O, tSx_S)
    liveTS = OR(tOq, tOx, tSq, tSx)
    ts_b0 = OR(AND(liveTS, c0), oneO)
    ts_b1 = AND(liveTS, c1)
    ts_b2 = OR(AND(nz, OR(tOx, tSx)), AND(cS, OR(tOq, tSq)))
    ts_b3 = OR(tOq, AND(nz, tOx), AND(cS, tSx))
    ts_b4 = OR(tSq, tSx, AND(cS, tOx))

    x.mark("transmission result bits")
    # ------------------------------------------------------------------------------------------
    # 3b. ground and sensitised states (0..8)
    # ------------------------------------------------------------------------------------------
    Eo = EOany
    g_one = AND(isZero, Eo)
    g_arr = AND(isZero, nEO, has)
    s4n = AND(is4, nEO)
    s4e = AND(is4, Eo)
    s5h = AND(is5, nEO, has)
    s5none = AND(is5, NOT(has))
    s5eo = AND(is5, Eo, NOT(iS))
    s5es = AND(is5, Eo, iS)
    s6h = AND(is6, nEO, has)
    s6none = AND(is6, NOT(has))
    s6e = AND(is6, Eo)
    s7h = AND(is7, nEO, has)
    s7none = AND(is7, NOT(has))
    s7e = AND(is7, Eo)
    s8h = AND(is8, nEO, has)
    s8none = AND(is8, NOT(has))
    s8e = AND(is8, Eo)
    k0 = OR(s5h, s6h, s7h, s8e)
    k1 = OR(s4e, s5h, s6e, s7h)
    rot = OR(g_arr, s4e, s5h, s5eo, s5es, s6h, s6e, s7h, s8h, s8e)
    r0 = XOR(d0, k0)
    r1 = XOR(d1, k1, AND(d0, k0))
    se_b0 = OR(AND(rot, NOT(r0)), g_one, AND(s123, Eo), s5none, s6none, s7none, s7e)
    se_b1 = OR(AND(rot, XOR(r1, r0)), AND(s123, c0), s5none, s7none)
    se_b2 = OR(AND(rot, r1, r0), AND(s123, c1), s5none, s7none)
    se_b3 = OR(g_arr, s4e, s5h, s8h, s8e, s5es, s4n, s7none, s7e, s8none)
    se_b4 = OR(s5eo, s5es, s6h, s6e, s7h, s6none, s7e)

    x.mark("ground + sensitised centre")
    # ------------------------------------------------------------------------------------------
    # 3c. confluent states (25..31)
    # ------------------------------------------------------------------------------------------
    allon = AND(EOany, NOT(QOany))
    XNh = AND(de["IO"], dw["IO"], dn["OUT"], ds["OUT"])
    XNv = AND(dn["IO"], ds["IO"], de["OUT"], dw["OUT"])
    xon = OR(AND(XNh, NOT(XOR(de["EO"], dw["EO"]))), AND(XNv, NOT(XOR(dn["EO"], ds["EO"]))))
    on = OR(allon, xon)
    XN = OR(XNh, XNv)
    # crossing: all four neighbours are transmission states, exactly two of them point at us
    a_, b_, c_, d_ = dn["IN"], ds["IN"], de["IN"], dw["IN"]
    p, q, g, k = XOR(a_, b_), XOR(c_, d_), AND(a_, b_), AND(c_, d_)
    two_in = OR(AND(p, q), AND(XOR(g, k), NOT(OR(p, q))))
    allTS = AND(*[OR(D[d]["IN"], D[d]["OUT"]) for d in "nesw"])
    cross = AND(two_in, allTS)
    crossRes = OR(AND(is25, cross, NOT(XN)), AND(c2931, NOT(Zany)))
    plainRes = AND(c2528, NOT(AND(is25, cross, NOT(XN))))
    zRes = AND(c2931, Zany)
    now = NOT(c0)
    cf_b2 = AND(nES, OR(AND(crossRes, EOany), AND(plainRes, now, on)))
    cf_b1 = AND(nES, OR(AND(crossRes, v_on), AND(plainRes, XOR(now, on))))
    cf_b0 = AND(nES, OR(AND(crossRes, OR(h_on, NOT(v_on))), AND(plainRes, NOT(on)), zRes))
    cf_hi = AND(c43, OR(c2, nz), nES)

    x.mark("confluent + crossing centre")
    # ------------------------------------------------------------------------------------------
    # 4. outputs
    # ------------------------------------------------------------------------------------------
    out = [
        OR(ts_b0, se_b0, cf_b0),
        OR(ts_b1, se_b1, cf_b1),
        OR(ts_b2, se_b2, cf_b2),
        OR(ts_b3, se_b3, cf_hi),
        OR(ts_b4, se_b4, cf_hi),
    ]
    x.mark("output ORs")
    x.outputs = out
    return out


def mismatch_report(x: XAG, outputs, limit=12) -> int:
    """Exhaustive comparison with the LUT; prints the first mismatches per centre state."""
    planes = x.simulate(outputs)
    ref = Sim.lut_planes()
    diff = np.zeros_like(planes[0])
    per_bit = []
    for k in range(5):
        dk = planes[k] ^ ref[k]
        per_bit.append(Sim.popcount(dk))
        diff |= dk
    total = Sim.popcount(diff)
    print(f"mismatching neighbourhoods: {total}   per output bit: {per_bit}")
    if total:
        bits = np.unpackbits(diff.view(np.uint8), bitorder="little")
        idx = np.flatnonzero(bits)
        got = np.zeros(idx.size, dtype=np.uint8)
        for k in range(5):
            pk = np.unpackbits(planes[k].view(np.uint8), bitorder="little")[idx]
            got |= pk << k
        lut = Sim.lut()
        cs = idx >> 20
        for c in np.unique(cs):
            sel = idx[cs == c]
            print(f"  centre {c}: {sel.size} mismatches")
            for i, gv in list(zip(sel, got[cs == c]))[:limit]:
                i = int(i)
                print(
                    f"    c={i >> 20 & 31:2d} n={i >> 15 & 31:2d} e={i >> 10 & 31:2d} s={i >> 5 & 31:2d} w={i & 31:2d}"
                    f"  circuit={int(gv):2d} lut={int(lut[i]):2d}"
                )
    return total


if __name__ == "__main__":
    x = XAG()
    outs = build(x)
    print("AND/XOR nodes per block (before any optimisation):")
    for label, n in x.section_sizes():
        print(f"  {n:4d}  {label}")
    print("total:", x.count())
    bad = mismatch_report(x, outs)
    if bad:
        sys.exit(1)
    nl = map_xag_exact(x)
    print("mapped to NOT/AND/OR/XOR (hand structure only):", nl.counts(), "depth", nl.depth(), "/", nl.depth(False))
    print("exhaustive check of the mapped netlist, mismatches per output bit:", nl.verify())
