"""Simulate the price-based controller's rent rule on the price traces of the floor runs.

    GHL_CONTROLLER_TRACES=1 cargo test -p ghl --release --test controller -- --ignored
    python ghl/tests/fixtures/controller_sim.py [research/out/controller_traces]

Each trace row is one region in one macro-step of a floor run: the floor's price at and below the
region level, and what plain cells would have cost the region (`Policy::log`). The simulation runs
the rule of `sched::controller::Policy::decide_priced` region by region on those prices: at the
floor it pays the floor's price, in plain cells the cells' price. It ignores what a region in plain
cells changes for the rest (its memo is not learned, its neighbours' halos), which the acceptance
test itself includes. For every premium `rho` and threshold `b` it prints each configuration's sum
against the cheaper engine per region over the whole run (the static choice in hindsight) and the
worst universe against the uniform floor, and marks the pairs that meet the acceptance criteria:
every sum within 1.2 times the choice in hindsight, no universe above 1.05 times the floor.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROBE = GATHER = 1e9 / 3.7e9
INSERT = 1.0
ACCEPTANCE = ("Hutton32b_k4", "Hutton32b_k3", "Life_k6")


def nominal(cfg):
    """Plain cells for one step of a region, without halo, with its conversion to tile IDs."""
    rule, k = cfg.split("_")
    k = int(k[1:])
    s, t = 1 << (k + 2), 1 << k
    cell, b = (1e9 / 1.39e12, 16) if rule == "Life" else (1e9 / 1.86e11, 8)
    return s * s * t * cell + (s // b) ** 2 * (GATHER + 1.34 * (PROBE + 0.5 * INSERT))


def load(path):
    regions = defaultdict(list)
    common = 0.0
    for r in csv.DictReader(open(path)):
        tm = int(r["top_misses"])
        f = float(r["floor_ns"])
        c = (float(r["cells_ns"]) if r["cells_ns"] else 0.0) + tm * INSERT
        common += float(r["charged_ns"]) - f
        regions[(int(r["x"]), int(r["y"]))].append((tm, f, c, bool(r["cells_ns"])))
    return regions, common


def run(regions, rho, b, cref):
    total = 0.0
    for rows in regions.values():
        cells, excess = False, 0.0
        for tm, f, c, content in rows:
            total += c if cells else f
            if tm == 0:
                cells, excess = False, 0.0
            elif not cells and content:
                excess = max(excess + f - rho * c, -b * cref)
                if excess > b * cref:
                    cells, excess = True, 0.0
    return total


def main():
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "research/out/controller_traces")
    data = []
    for p in sorted(d.glob("*_floor.csv")):
        name = p.stem[: -len("_floor")]
        cfg = "_".join(name.split("_")[:2])
        if cfg in ACCEPTANCE:
            regions, common = load(p)
            floor = sum(r[1] for rows in regions.values() for r in rows) + common
            oracle = sum(min(sum(r[1] for r in rows), sum(r[2] for r in rows)) for rows in regions.values()) + common
            data.append((cfg, name, regions, common, floor, oracle))
    if not data:
        sys.exit(f"no floor traces of {', '.join(ACCEPTANCE)} in {d}")
    for rho in (2.0, 2.5, 3.0, 3.5, 4.0):
        for b in (1, 2, 3, 4):
            sums = defaultdict(lambda: [0.0, 0.0])
            worst = defaultdict(lambda: (0.0, ""))
            for cfg, name, regions, common, floor, oracle in data:
                t = run(regions, rho, b, nominal(cfg)) + common
                sums[cfg][0] += t
                sums[cfg][1] += oracle
                if t / floor > worst[cfg][0]:
                    worst[cfg] = (t / floor, name)
            ok = all(s[0] <= 1.2 * s[1] for s in sums.values()) and all(w[0] <= 1.05 for w in worst.values())
            cols = " | ".join(f"{cfg}: {sums[cfg][0] / sums[cfg][1]:.3f}x, worst {worst[cfg][0]:.3f}x floor" for cfg in ACCEPTANCE if cfg in sums)
            print(f"{'PASS' if ok else '    '} rho={rho} b={b} | {cols}")


if __name__ == "__main__":
    main()
