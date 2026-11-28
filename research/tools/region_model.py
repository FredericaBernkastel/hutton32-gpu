"""The region model of paper.md (Section VI-G, "The region") against the sweeps that test it.

Reads the logs of the targets region-size, region-size-life and region-step (each from the newest
dated folder of research/logs that holds it) and prints, in this order:

  1. the checks: the price of plain cells taken from a floor run against the plain-cells run of the
     same universe, and the price of the cheaper engine per region against the run of that choice;
  2. the price of a stepped region, q = 1 + halo + conversion + memo, against its closed form;
  3. the novel area by region size and its exponent;
  4. the two tables of the paper as Markdown rows, with what the text quotes from them;
and writes research/figures/region_model.svg (Fig. 5).

Usage (from research/): python tools/region_model.py
"""
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ghl_logs  # noqa: E402
from svgplot import chart  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
NUM = r"([-+]?[0-9][0-9.]*(?:e[+-]?[0-9]+)?)"
# the constants `adapt` prices with (cmd_adapt.rs, Model): probes/s, ns per insert
R_P, INSERT_NS, PROBE_NS = 3.7e9, 1.0, 1e9 / 3.7e9
# conversion of a stepped region to IDs, in probes per base tile: a gather and 1.34 interned nodes
C_V = (PROBE_NS + 1.34 * (PROBE_NS + 0.5 * INSERT_NS)) / PROBE_NS
# the brute-force rates the sweeps price with, by base tile: Hutton32b at the byte-LUT rate (8x8
# base tiles), Life at its kernel's rate (16x16; --cell-rate in the sweep scripts)
R_B = {8: 1.86e11, 16: 1.39e12}
# the bit-plane kernel's rate (paper, Table consts), for the same counts at another exchange rate
R_B_PLANES = 3.0e11
SHORT = {"composite of 16 patterns": "mixed", "lotr1.PPMd-244K.rle": "ROM", "advanced counter v2.rle": "counter",
         "composite of 8 patterns": "chaotic", "composite of 1 patterns": "counter, pre-run", "soup512.rle": "soup",
         "Unit-Life-Cell-512x512.rle": "Unit Life Cell", "metapixel-galaxy.mc": "galaxy", "catacryst.mc": "catacryst",
         "gotts-dots.mc": "gotts-dots"}
REF_REGION = 256          # the region whose floor run is the reference row, as in Tables regions and liferegions


def parse(name):
    """One dict per `adapt` invocation of the log `name`."""
    out, b = [], None
    for line in ghl_logs.Logs().lines(name):
        m = re.match(r"universe: (.+?), (\d+)x(\d+) cells, population (\d+) \| base tile (\d+)x\d+, macro-step (\d+) gens, region (\d+)x\d+ \| (\d+) gens", line)
        if m:
            b = dict(universe=SHORT[m.group(1)], base=int(m.group(5)), step=int(m.group(6)), region=int(m.group(7)), gens=int(m.group(8)), rows={})
            out.append(b)
            continue
        if b is None:
            continue
        m = re.match(rf"(fixed: brute at \d+x\d+|fixed: plain cells|adaptive|hindsight per region)\s+\d+\s+\d+\s+{NUM}\s+{NUM}\s+{NUM}\s+{NUM}\s", line)
        if m:
            key = {"fixed: plain cells": "plain", "adaptive": "adaptive", "hindsight per region": "hindsight"}.get(m.group(1), "floor")
            if key == "floor" and "floor" in b["rows"]:
                continue                                   # the first fixed floor is the one the model prices from
            b["rows"][key] = dict(hash=float(m.group(3)), cells=float(m.group(4)), total=float(m.group(5)))
            continue
        m = re.match(rf"  plain cells in every \d+x\d+ region: hash {NUM} \+ cells {NUM} = {NUM} ms \| (\d+) of (\d+) non-empty region-steps stepped \| "
                     rf"per cell-update of a stepped region: halo \+{NUM}%, conversion \+{NUM}%, memo at and above the region \+{NUM}%", line)
        if m:
            g = [float(x) for x in m.groups()]
            b["model"] = dict(hash=g[0], cells=g[1], total=g[2], stepped=int(g[3]), nonempty=int(g[4]), halo=g[5] / 100, conv=g[6] / 100, memo=g[7] / 100)
            b["q"] = 1 + b["model"]["halo"] + b["model"]["conv"] + b["model"]["memo"]
            b["rho"] = R_B[b["base"]] / R_P
            continue
        m = re.match(rf"  the plain-cells run charged {NUM} ms: the model is off by {NUM} ms", line)
        if m:
            b["plain_off"] = float(m.group(2))
            continue
        m = re.match(rf"  the cheaper engine per region in hindsight: {NUM} ms, plain cells in (\d+) of (\d+) regions with work", line)
        if m:
            b["hind"] = dict(total=float(m.group(1)), cells=int(m.group(2)), regions=int(m.group(3)))
            continue
        m = re.match(rf"  run as assigned it charged {NUM} ms: the model is off by {NUM} ms", line)
        if m:
            b["hind_off"] = float(m.group(2))
            continue
        # the cross-check line: every policy priced from the engine's raw counters
        if re.match(r"  (brute at \d+x\d+|plain cells|adaptive|hindsight per region) [0-9.]+ ms( \||$)", line):
            b["counters"] = {}
            for part in line.strip().split(" | "):
                name_, v = part.rsplit(" ", 2)[0], float(part.rsplit(" ", 2)[1])
                key = {"plain cells": "plain", "adaptive": "adaptive", "hindsight per region": "hindsight"}.get(name_, "floor")
                b["counters"].setdefault(key, v)
    return out


def ms(v):
    """Model time as the tables of the paper print it: two decimals below 10 ms, one above."""
    return f"{v:.2f}" if v < 9.995 else f"{v:.1f}"


def best(vals):
    """Table cells for `vals`, the smallest in bold (ties all bold, as printed)."""
    shown = [ms(v) for v in vals]
    lo = min(float(x) for x in shown)
    return [f"**{x}**" if float(x) == lo else x for x in shown]


class Sweep:
    """The region sweep of one rule: every universe at every region, at 16-generation steps."""

    def __init__(self, rule, blocks):
        self.rule, self.blocks = rule, blocks
        self.unis = [u for u in dict.fromkeys(b["universe"] for b in blocks) if any(b["universe"] == u and "adaptive" in b["rows"] for b in blocks)]
        self.sizes = sorted({b["region"] for b in blocks})

    def get(self, u, s):
        return next(b for b in self.blocks if b["universe"] == u and b["region"] == s and "adaptive" in b["rows"])

    def ran(self, u, s):
        return "hindsight" in self.get(u, s)["rows"]

    def hind(self, u, s):
        b = self.get(u, s)
        return b["rows"]["hindsight"]["total"] if self.ran(u, s) else b["hind"]["total"]

    def floor(self, u):
        return self.get(u, REF_REGION)["rows"]["floor"]["total"]

    def sums(self, s):
        """(the cheaper engine per region, Algorithm 3, plain cells in every region), summed over the universes."""
        return (sum(self.hind(u, s) for u in self.unis), sum(self.get(u, s)["rows"]["adaptive"]["total"] for u in self.unis),
                sum(self.get(u, s)["model"]["total"] for u in self.unis))


def table_regionmodel(sweeps, name=lambda u: u):
    rows = ["| Region | | | | | | Sum | Algorithm 3, sum | Plain cells in every region, sum |", "|---|---|---|---|---|---|---|---|---|"]
    for sw in sweeps:
        pad = [""] * (5 - len(sw.unis))
        rows.append(f"| ***{sw.rule}*** | " + " | ".join([f"*{name(u)}*" for u in sw.unis] + pad) + " | | | |")
        cols = [best([sw.hind(u, s) for s in sw.sizes]) for u in sw.unis]
        tot = [sw.sums(s) for s in sw.sizes]
        hs, cs, ps = best([t[0] for t in tot]), best([t[1] for t in tot]), best([t[2] for t in tot])
        for i, s in enumerate(sw.sizes):
            mark = ["" if sw.ran(u, s) else "ᵃ" for u in sw.unis]
            rows.append(f"| {s}×{s} | " + " | ".join([c[i] + k for c, k in zip(cols, mark)] + pad) + f" | {hs[i]}{'ᵃ' if any(mark) else ''} | {cs[i]} | {ps[i]} |")
        rows.append(f"| Uniform F{2 * sw.blocks[0]['base']}, for reference | " + " | ".join([ms(sw.floor(u)) for u in sw.unis] + pad) + f" | {ms(sum(sw.floor(u) for u in sw.unis))} | | |")
    return rows


class Diagonal:
    """The sweep region = 4 x step of one rule."""

    def __init__(self, rule, blocks):
        self.rule, self.blocks = rule, blocks
        self.unis = list(dict.fromkeys(b["universe"] for b in blocks))
        self.steps = sorted({b["step"] for b in blocks})

    def at(self, u, t):
        return next(b for b in self.blocks if b["universe"] == u and b["step"] == t)

    def ran(self, u, t):
        return "hindsight" in self.at(u, t)["rows"]

    def hind(self, u, t):
        b = self.at(u, t)
        return b["rows"]["hindsight"]["total"] if self.ran(u, t) else b["hind"]["total"]

    def sums(self, t):
        """(uniform floor, the cheaper engine per region), summed over the universes."""
        return (sum(self.at(u, t)["rows"]["floor"]["total"] for u in self.unis), sum(self.hind(u, t) for u in self.unis))


def table_regionstep(diags, name=lambda u: u):
    rows = ["| Step, region | | | | | | Sum: uniform floor | Sum: engine per region |", "|---|---|---|---|---|---|---|---|"]
    for d in diags:
        pad = [""] * (5 - len(d.unis))
        b = d.blocks[0]["base"]
        rows.append(f"| ***{d.rule}***, {b}×{b} base tiles | " + " | ".join([f"*{name(u)}*" for u in d.unis] + pad) + " | | |")
        hcol = {u: best([d.hind(u, t) for t in d.steps]) for u in d.unis}
        fs, hs = best([d.sums(t)[0] for t in d.steps]), best([d.sums(t)[1] for t in d.steps])
        for i, t in enumerate(d.steps):
            mark = {u: "" if d.ran(u, t) else "ᵃ" for u in d.unis}
            rows.append(f"| {t}, {4 * t}×{4 * t} | " + " | ".join([f"{ms(d.at(u, t)['rows']['floor']['total'])} / {hcol[u][i]}{mark[u]}" for u in d.unis] + pad)
                        + f" | {fs[i]} | {hs[i]}{'ᵃ' if any(mark.values()) else ''} |")
    return rows


def load():
    hut, life, diag = parse("region-size.txt"), parse("region-size-life.txt"), parse("region-step.txt")
    sweeps = [Sweep("Hutton32b", hut), Sweep("Life", life)]
    diags = [Diagonal("Hutton32b", [b for b in diag if b["base"] == 8]), Diagonal("Life", [b for b in diag if b["base"] == 16])]
    return hut, life, diag, sweeps, diags


def main():
    hut, life, diag, sweeps, diags = load()

    # ---- 1. the checks -------------------------------------------------------------------
    print("1. checks")
    for rule, blocks in (("Hutton32b", hut), ("Life", life), ("region = 4 x step", diag)):
        po = [(b["universe"], b["region"], b["plain_off"]) for b in blocks if "plain_off" in b]
        ho = [(b["hind_off"] / b["rows"]["hindsight"]["total"], b["universe"], b["region"], b["step"]) for b in blocks if "hind_off" in b]
        if po:
            print(f"  {rule}: plain cells priced from the floor run against the plain-cells run, {len(po)} pairs: largest difference {max(abs(o) for _, _, o in po):.4f} ms")
        if ho:
            print(f"  {rule}: the cheaper engine per region, price against its run, {len(ho)} pairs: price below the run by at most {-100 * min(ho)[0]:.1f}% "
                  f"({min(ho)[1]} at {min(ho)[2]}, step {min(ho)[3]}), above it by at most {100 * max(ho)[0]:.1f}%")
            print("      per universe (price - run, % of the run, by region): " + "; ".join(
                f"{u} " + "/".join(f"{100 * r:+.1f}" for r, uu, _, _ in ho if uu == u) for u in dict.fromkeys(x[1] for x in ho)))

    # ---- 2. the price of a stepped region ------------------------------------------------
    print()
    print("2. q = 1 + halo + conversion + memo per cell-update of a stepped region (plain cells in every region)")
    print(f"   closed form: halo between 0 (inside a block) and 2(T-1)/s (1 + (2T-1)/(3s)) (alone); conversion = c_v rho / (b^2 T), c_v = {C_V:.2f}; memo = c_m rho / (s^2 T)")
    for rule, blocks in (("Hutton32b", hut), ("Life", life), ("region = 4 x step", diag)):
        print(f"  {rule}")
        for b in blocks:
            if "plain" in b["rows"]:
                continue
            m, s, t, rho = b["model"], b["region"], b["step"], b["rho"]
            alone = 2 * (t - 1) / s * (1 + (2 * t - 1) / (3 * s))
            conv = C_V * rho / (b["base"] ** 2 * t)
            b["c_m"] = m["memo"] * s * s * t / rho if m["memo"] >= 0.01 else float("nan")   # printed to 0.1 %
            assert -1e-3 <= m["halo"] <= alone + 1e-3, (b["universe"], s, m["halo"], alone)
            assert abs(m["conv"] - conv) < 2e-3, (b["universe"], s, m["conv"], conv)
            print(f"    {b['universe']:16} s={s:5} T={t:3} rho={rho:5.0f} b^3/rho={b['base'] ** 3 / rho:5.1f}  halo {m['halo']:.3f} (alone {alone:.3f})  conversion {m['conv']:.3f} (form {conv:.3f})"
                  f"  memo {m['memo']:.3f} (c_m {b['c_m']:5.1f}, {100 * m['stepped'] / m['nonempty']:5.1f}% of non-empty region-steps stepped)  q {b['q']:.2f}")

    # ---- 3. the novel area and its exponent ----------------------------------------------
    print()
    print("3. novel area Omega(s) = s^2 T x stepped region-steps at 16-generation steps; alpha = d ln Omega / d ln s between neighbouring sizes; what it must exceed for the smaller region to win: (halo + 2 memo) / q of the smaller")
    for sw in sweeps:
        print(f"  {sw.rule}")
        for u in sw.unis:
            bs = [sw.get(u, s) for s in sw.sizes]
            om = [s * s * b["model"]["stepped"] for s, b in zip(sw.sizes, bs)]
            al = [math.log2(om[i + 1] / om[i]) for i in range(len(om) - 1)]
            thr = [(b["model"]["halo"] + 2 * b["model"]["memo"]) / b["q"] for b in bs[:-1]]
            flags = "".join("." if a > t else "!" for a, t in zip(al, thr))
            print(f"    {u:16} alpha " + " ".join(f"{a:5.2f}" for a in al) + "   threshold " + " ".join(f"{x:5.2f}" for x in thr) + f"   {flags}   plain cells in every region: " + " ".join(ms(b["model"]["total"]) for b in bs))

    # ---- 4. the tables of the paper --------------------------------------------------------
    print()
    print("4. Table regionmodel: the cheaper engine per region, chosen from the floor run and then run (a: priced, not run), model ms")
    print("\n".join(table_regionmodel(sweeps)))
    for sw in sweeps:
        print(f"  {sw.rule}")
        print("    uniform floor by region (the root grows with it): " + ", ".join(f"{u} {'/'.join(ms(sw.get(u, s)['rows']['floor']['total']) for s in sw.sizes)}" for u in sw.unis))
        s0 = sw.sizes[0]
        print(f"    plain cells in n of m regions with work at {s0}: " + ", ".join(f"{u} {sw.get(u, s0)['hind']['cells']} of {sw.get(u, s0)['hind']['regions']}" for u in sw.unis))
        print("    Algorithm 3 above the cheaper engine per region, by region: " + ", ".join(f"{100 * (sw.sums(s)[1] / sw.sums(s)[0] - 1):+.0f}%" for s in sw.sizes))
        print("    per universe at the smallest region, Algorithm 3 / per region / floor: " + "; ".join(
            f"{u} {ms(sw.get(u, s0)['rows']['adaptive']['total'])} / {ms(sw.hind(u, s0))} / {ms(sw.floor(u))}" for u in sw.unis))
        print("    priced from the engine's raw counters instead, sums by region, floor / cheaper engine per region / Algorithm 3: " + ", ".join(
            f"{s}: {ms(sum(sw.get(u, s)['counters']['floor'] for u in sw.unis))} / "
            + (ms(sum(sw.get(u, s)['counters']['hindsight'] for u in sw.unis)) if all(sw.ran(u, s) for u in sw.unis) else "-")
            + f" / {ms(sum(sw.get(u, s)['counters']['adaptive'] for u in sw.unis))}" for s in sw.sizes))

    print()
    print("4b. Table regionstep: region = 4 x step; uniform floor / the cheaper engine per region (run), model ms")
    print("\n".join(table_regionstep(diags)))
    for d in diags:
        print(f"  {d.rule}")
        for t in d.steps:
            bs = [d.at(u, t) for u in d.unis]
            lhs = [(b["model"]["conv"] + 3 * b["model"]["memo"]) / b["q"] for b in bs]
            dur = [b["model"]["stepped"] / b["model"]["nonempty"] * b["gens"] / t for b in bs]
            print(f"    step {t:3}: q " + ", ".join(f"{u} {b['q']:.2f}" for u, b in zip(d.unis, bs)) + f" | (conversion + 3 memo)/q {min(lhs):.2f}-{max(lhs):.2f}")
            print("              plain cells in every region, priced: " + ", ".join(f"{u} {ms(b['model']['total'])}" for u, b in zip(d.unis, bs))
                  + " | steps a non-empty region is stepped, mean: " + ", ".join(f"{u} {x:.1f}" for u, x in zip(d.unis, dur)))
        print("    floor, hash part by step: " + "; ".join(f"{u} " + "/".join(ms(d.at(u, t)["rows"]["floor"]["hash"]) for t in d.steps) for u in d.unis))
        print("    priced from the engine's raw counters instead (every brute window isolated), sums floor / cheaper engine per region: " + ", ".join(
            f"step {t}: {ms(sum(d.at(u, t)['counters']['floor'] for u in d.unis))} / {ms(sum(d.at(u, t)['counters']['hindsight'] for u in d.unis))}" for t in d.steps if all(d.ran(u, t) for u in d.unis)))
        if d.blocks[0]["base"] == 8:
            # the same counts at the bit-plane kernel's rate: cells cost less, hashing the same
            k = R_B[8] / R_B_PLANES
            print(f"    at the bit-plane rate (rho = {R_B_PLANES / R_P:.0f}), ROM: " + ", ".join(
                f"step {t}: floor {ms(d.at('ROM', t)['rows']['floor']['hash'] + k * d.at('ROM', t)['rows']['floor']['cells'])}, "
                f"plain cells {ms(d.at('ROM', t)['model']['hash'] + k * d.at('ROM', t)['model']['cells'])}" for t in d.steps))

    # ---- Fig. 5: Algorithm 3 and the cheaper engine per region against the uniform floor ----
    # every curve is a sum over the rule's universes, divided by the uniform floor at 16-generation
    # steps (the reference rows of the tables); "step = side / 4" is the diagonal of Table regionstep
    series = []
    for sw, d in zip(sweeps, diags):
        ref = sum(sw.floor(u) for u in sw.unis)
        series += [(f"{sw.rule}: Algorithm 3", [(s, sw.sums(s)[1] / ref) for s in sw.sizes]),
                   (f"{sw.rule}: engine per region", [(s, sw.sums(s)[0] / ref) for s in sw.sizes]),
                   (f"{sw.rule}: per region, step = side / 4", [(4 * t, d.sums(t)[1] / ref) for t in d.steps])]
    os.makedirs(OUT, exist_ok=True)
    chart(os.path.join(OUT, "region_model.svg"), "region side (cells)", "model time / uniform floor at 16-generation steps", series, width=760, height=420,
          xticks=[32, 64, 128, 256, 512, 1024], yticks=[0.25, 0.5, 1, 2], dashed=[n for n, _ in series if "step =" in n])
    print()
    print("figure written to", os.path.abspath(os.path.join(OUT, "region_model.svg")))


if __name__ == "__main__":
    main()
