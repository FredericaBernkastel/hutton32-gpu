"""The measured numbers of the overview diagram, read from the raw logs in research/logs/<date>
(the output of reproduce.sh, see paper.md, Appendix A). Nothing is typed in by hand: every figure
is found by the line that reports it, and a missing line is an error.

    python tools/ghl_logs.py [logs dir]      print everything the diagram uses
"""
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "logs", "2026-11-25")
NUM = r"([0-9][0-9.]*(?:e[+-]?[0-9]+)?)"


class Logs:
    def __init__(self, directory=DEFAULT):
        self.dir = directory
        self._cache = {}

    def lines(self, name):
        if name not in self._cache:
            with open(os.path.join(self.dir, name), encoding="utf-8") as f:
                self._cache[name] = f.read().splitlines()
        return self._cache[name]

    def find(self, name, pattern, after=None, nth=0):
        """The groups of the `nth` line matching `pattern`, looking only below the first line
        that contains `after` if that is given."""
        lines = self.lines(name)
        start = 0
        if after is not None:
            start = next((i for i, l in enumerate(lines) if after in l), None)
            if start is None:
                raise KeyError(f"{name}: no line contains {after!r}")
        hits = [m for m in (re.search(pattern, l) for l in lines[start:]) if m]
        if len(hits) <= nth:
            raise KeyError(f"{name}: {pattern!r} not found" + (f" after {after!r}" if after else ""))
        g = hits[nth].groups()
        return [float(x) if re.fullmatch(NUM, x or "") else x for x in g]

    def all(self, name, pattern):
        return [[float(x) if re.fullmatch(NUM, x or "") else x for x in m.groups()]
                for m in (re.search(pattern, l) for l in self.lines(name)) if m]

    # ---- time_golly.py output: one block per pattern and algorithm ----------------------------
    def golly(self, name):
        """[{run, pattern, algo, step, gens_s, cells_s, timeout}] in file order; `run` counts the
        invocations of time_golly.py in the file."""
        out, pat, algo, run, last = [], None, None, -1, ""
        for l in self.lines(name):
            m = re.match(r"pattern (.+)$", l)
            if m:
                last = m.group(1).strip()
                continue
            m = re.match(r"load-only .*\(algo (\w+),", l)   # the line after time_golly's `pattern` line
            if m:
                pat, algo, run = os.path.basename(last), m.group(1), run + 1
                continue
            m = re.match(rf"step\s+(\d+):\s+{NUM}s net,\s+{NUM} gens/s(?:, {NUM} cell-updates/s)?", l)
            if m:
                out.append(dict(run=run, pattern=pat, algo=algo, step=int(m.group(1)), gens_s=float(m.group(3)),
                                cells_s=float(m.group(4)) if m.group(4) else None, timeout=False))
                continue
            m = re.match(rf"step\s+(\d+): TIMEOUT after (\d+)s at gen (\d+),\s+{NUM} gens/s, {NUM} cell-updates/s", l)
            if m:
                out.append(dict(run=run, pattern=pat, algo=algo, step=int(m.group(1)), gens_s=float(m.group(4)),
                                cells_s=float(m.group(5)), timeout=True, reached=int(m.group(3))))
        return out

    # ---- head2head.py output: blocks of rows `pattern, golly s, runs, engine s, runs, rates` ----
    def h2h(self, name):
        """A list of blocks; each block is {pattern: (golly gens/s, engine gens/s, same state)}."""
        blocks = []
        for l in self.lines(name):
            if l.startswith("pattern "):
                blocks.append({})
                continue
            m = re.match(r"(.+?)\s+[\d.]+\s+\d+\s+[\d.]+\s+\d+\s+(\d+)\s+(\d+)\s+(yes|NO)\s*$", l)
            if m and blocks:
                blocks[-1][m.group(1).strip()] = (float(m.group(2)), float(m.group(3)), m.group(4) == "yes")
        return blocks


# the patterns of the comparison, by the file names the logs give them
ROM, SQUARES, CHAOTIC, COUNTER = "lotr1.ANSI-1M.rle", "squares.rle", "chaotic.rle", "advanced counter v2.rle"
ULC, SOUP = "Unit-Life-Cell-512x512.rle", "soup2048.rle"
H32, LIFE = "[[Hutton32]] and [[Hutton32b]], 32 states", "[[Life]], 2 states"
WORDS = dict(enumerate("zero one two three four five six seven eight nine ten eleven twelve".split()))


def collect(logs=None):
    """Every measured number the diagram shows, as one dictionary."""
    L = logs or Logs()
    d = {}

    # ---- the four rates (paper Table I) ----
    lk = L.lines("lut-kernel.txt")
    i = next(i for i, l in enumerate(lk) if "x24 ->" in l and "8x8 workgroups" in l and "5 runs" in lk[i + 1])
    d["rb_lut"] = float(re.search(rf"= {NUM} cell-updates/s", lk[i + 1]).group(1))
    d["rb_planes"] = statistics.median(x[0] for x in L.all("bitplane-sizes.txt", rf"\[side 16384 gens 64\] bit-plane kernel: .* = {NUM} cell-updates/s"))
    lf = L.lines("life-kernels.txt")
    i = next(i for i, l in enumerate(lf) if "16384x16384" in l and "plain kernel" in l)
    d["rb_life"] = float(re.search(rf"= {NUM} cell-updates/s", lf[i + 1]).group(1))
    probes = [(int(a), int(b), c) for a, b, c in L.all("probe.txt", rf"^\s+(\d+)\s+(\d+) MiB\s+\d+\s+{NUM}\s")]
    d["probe_rows"] = probes
    # out of cache (128 MiB to 8 GiB): the median, to two figures as the paper states it
    d["rp"] = float(f"{statistics.median(r for _, mib, r in probes if mib >= 128):.1e}")
    d["rp_cache"] = statistics.median(r for log2, mib, r in probes if mib <= 16 and log2 >= 14)
    ins = {k: v for k, v in L.all("probe.txt", rf"^\s+([A-D]): .*?: {NUM} find-or-insert/s")}
    d["ri"] = (min(ins[k] for k in "BCD"), max(ins[k] for k in "BCD"))
    d["ri_first"] = ins["A"]
    d["dispatch_us"] = L.find("latency.txt", rf"chained dispatch, one pass:\s+{NUM} us each")[0]
    d["wave_us"] = L.find("latency.txt", rf"whole miss-path wave, 1 pass, 16 fixed dispatches.*:\s+{NUM} us")[0]
    d["cpu_lut"] = L.find("rom-kernels.txt", rf"CPU reference: {NUM} cell-updates/s single thread")[0]
    d["circuit_check"] = L.find("cpu-parallel.txt", r"exhaustive check of step_planes \(u64\) against lut_table_Hutton32b\.bin: (\d+) mismatches of (\d+) entries")
    d["cpu4"] = L.find("cpu-parallel.txt", rf"^lotr1\.ANSI-1M x1\s+\d+\s+{NUM}\s+{NUM}\s*$", after="bit-sliced forms on 4 threads")[1]

    # ---- Golly ----
    g = L.golly("golly.txt")
    rom = [r for r in g if r["pattern"] == ROM]
    best = max(rom, key=lambda r: r["cells_s"])
    d["rom_cells"] = 9467605                         # the ROM's bounding box, 3095 x 3059
    d["golly_rom_gens_s"] = best["cells_s"] / d["rom_cells"]
    d["golly_rom_cells_s"] = best["cells_s"]
    lg = L.golly("life-golly.txt")
    d["golly_soup_hl"] = max(r["gens_s"] for r in lg if r["pattern"] == SOUP and r["algo"] == "HashLife")
    ql = max((r for r in lg if r["pattern"] == SOUP and r["algo"] == "QuickLife"), key=lambda r: r["gens_s"])
    d["golly_soup_ql"], d["ql_cells_s"] = ql["gens_s"], ql["cells_s"]
    d["golly_ulc"] = max(r["gens_s"] for r in lg if r["pattern"] == ULC and r["algo"] == "HashLife")

    # ---- GPU components ----
    d["rom_planes_gens_s"] = L.find("bitplane-sizes.txt", rf"\[ROM 3104x3067\] bit-plane kernel: .* = {NUM} gens/s")[0]
    tl = "tile.txt"
    d["tile_counter"] = L.find(tl, rf"\[counter k=16, 7x16 groups\].* = {NUM} gens/s")[0]
    d["tile_squares"] = L.find(tl, rf"\[squares k=16, 6x48 groups, verified\].* = {NUM} gens/s")[0]
    d["tile_chaotic"] = L.find(tl, rf"\[chaotic k=16, 5x16 groups, verified\].* = {NUM} gens/s")[0]
    sw = L.lines("sweep.txt")
    i = next(i for i, l in enumerate(sw) if "tile 32x32" in l and "learned in" in l)
    d["rom_hit_learn_s"] = float(re.search(rf"learned in {NUM}s", sw[i]).group(1))
    j = next(j for j in range(i, len(sw)) if "universe: 1x1 copies" in sw[j])
    d["rom_hit_gens_s"] = float(re.search(rf"= {NUM} gens/s", sw[j + 1]).group(1))
    d["soup_gpu"] = L.find("life-golly.txt", rf"life kernel: 4096 gens .* = {NUM} gens/s", after="plain kernel")[0]
    ls = L.lines("life-sweep.txt")
    j = next(j for j, l in enumerate(ls) if "universe: 1x1 copies" in l)
    d["ulc_hit_gens_s"] = float(re.search(rf"= {NUM} gens/s", ls[j + 1]).group(1))
    for key, name in (("sparse", "sparse.txt"), ("sparse_ulc", "life-sparse.txt")):
        d[key + "_gpu"] = L.find(name, rf"active blocks : .* = {NUM} gens/s")[0]
        d[key + "_learn_s"] = L.find(name, rf"learned in {NUM}s")[0]
        d[key + "_file"] = L.find(name, r"^pattern (.+?) rule \S+: tile ")[0]
        nx, ny, cells = L.find(name, rf"universe: (\d+)x(\d+) copies .* cells \({NUM}\)")
        d[key + "_copies"], d[key + "_cells"] = int(nx) * int(ny), cells
        steps = L.golly(name)            # two runs: 16,384 generations, then 2^20
        d[key + "_golly"] = max(r["gens_s"] for r in steps if r["run"] == 0)
        d[key + "_golly_long"] = max(r["gens_s"] for r in steps if r["run"] == 1)

    # ---- the recursion on one CPU thread against Golly ----
    h = L.h2h("head2head.txt")
    d["idle"] = {k: v for k, v in h[0].items()}
    d["h2h_squares"] = h[1][SQUARES]
    d["h2h_movie"] = h[1]["32x32 movie.mc"]
    d["h2h_replicator"] = h[2]["Hutton-replicator.rle"]
    rl = L.h2h("runlength.txt")          # blocks: 2^14, 2^16, 2^18, 2^20, 2^22, 2^24 generations
    d["runlength"] = rl
    lh = L.h2h("life-h2h.txt")
    life = {}
    for b in lh:
        life.update(b)
    d["h2h_ulc"] = life.pop(ULC)
    d["life_eventful"] = life
    return d


def outcomes(d):
    """Rows of the comparison with Golly 5.1 (paper Table XI): GHL component rate over Golly's
    fastest run at the run length given. kind: 'gpu' measured on the GPU, 'cpu' the recursion on
    one CPU thread, 'learn' a GPU run with the CPU's learning time counted in, 'learn>' the same
    as a lower bound (Golly did not finish the run: its rate is that of its first minute)."""
    R = d["runlength"]
    g20, g24, g14 = R[3], R[5], R[0]
    rom_gens = 1056
    rom_learn = (rom_gens / d["golly_rom_gens_s"]) / (d["rom_hit_learn_s"] + rom_gens / d["rom_hit_gens_s"])
    sp_gens = 16384
    sp_learn = (sp_gens / d["sparse_golly"]) / (d["sparse_learn_s"] + sp_gens / d["sparse_gpu"])
    su_learn = (sp_gens / d["sparse_ulc_golly"]) / (d["sparse_ulc_learn_s"] + sp_gens / d["sparse_ulc_gpu"])
    idle = [e / g for g, e, _ in d["idle"].values()]
    ev = [e / g for g, e, _ in d["life_eventful"].values()]
    same = all(s for _, _, s in list(d["idle"].values()) + list(d["life_eventful"].values()))
    assert d["sparse_file"] == COUNTER and d["sparse_ulc_file"] == ULC

    def power(v):
        """6.0e10 as 6·10^{10}, 1.0e10 as 10^{10}."""
        e = int(f"{v:.0e}".split("e")[1])
        m = round(v / 10 ** e)
        return f"10^{{{e}}}" if m == 1 else f"{m}·10^{{{e}}}"

    # names: pattern files in backticks (the figure sets them in a monospaced font)
    rows = [
        dict(group=H32, name=f"`{ROM}`, first period", note=f"{d['rom_cells'] / 1e6:.1f} M cells, dense, all new", points=[("gpu", d["rom_planes_gens_s"] / d["golly_rom_gens_s"])]),
        dict(group=H32, name=f"`{ROM}`, once learned", note="the memo holds a period; 32×32 tiles", points=[("gpu", d["rom_hit_gens_s"] / d["golly_rom_gens_s"]), ("learn>", rom_learn)]),
        dict(group=H32, name=f"`{SQUARES}`", note="65,536 generations", points=[("gpu", d["tile_squares"] / d["h2h_squares"][0])]),
        dict(group=H32, name=f"`{CHAOTIC}`", note="16,384 and 2^{24} generations", points=[("gpu", d["tile_chaotic"] / g14[CHAOTIC][0]), ("cpu", g24[CHAOTIC][1] / g24[CHAOTIC][0])]),
        dict(group=H32, name=f"`{COUNTER}`", note="2^{20} and 2^{24} generations", points=[("gpu", d["tile_counter"] / g20[COUNTER][0]), ("cpu", g24[COUNTER][1] / g24[COUNTER][0])]),
        dict(group=H32, name=f"{d['sparse_copies']:,} × `{COUNTER}`", note=f"sparse, {power(d['sparse_cells'])} cells", points=[("gpu", d["sparse_gpu"] / d["sparse_golly"]), ("learn", sp_learn)]),
        dict(group=H32, name=f"{WORDS.get(len(idle), len(idle))} idle circuits", note="2^{24} generations", range=("cpu", min(idle), max(idle))),
        dict(group=LIFE, name="random soup, against QuickLife", note="2,048 × 2,048 cells, dense, chaotic", points=[("gpu", d["soup_gpu"] / d["golly_soup_ql"])]),
        dict(group=LIFE, name="random soup, against HashLife", note="the same run", points=[("gpu", d["soup_gpu"] / d["golly_soup_hl"])]),
        dict(group=LIFE, name=f"`{ULC}`", note="2^{20} and 2^{24} generations", points=[("gpu", d["ulc_hit_gens_s"] / d["golly_ulc"]), ("cpu", d["h2h_ulc"][1] / d["h2h_ulc"][0])]),
        dict(group=LIFE, name=f"{d['sparse_ulc_copies']:,} × `{ULC}`", note=f"sparse, {power(d['sparse_ulc_cells'])} cells", points=[("gpu", d["sparse_ulc_gpu"] / d["sparse_ulc_golly"]), ("learn", su_learn)]),
        dict(group=LIFE, name=f"{WORDS.get(len(ev), len(ev))} eventful patterns", note="2^{20} or 2^{24} generations", range=("cpu", min(ev), max(ev))),
    ]
    return rows, same


if __name__ == "__main__":
    d = collect(Logs(sys.argv[1]) if len(sys.argv) > 1 else None)
    for k, v in d.items():
        print(f"{k:22} {v}")
    rows, same = outcomes(d)
    print("\nagainst Golly (ratio > 1: the GHL component is faster); same final states:", same)
    for r in rows:
        if "range" in r:
            print(f"  {r['name']:44} {r['range'][0]}: {r['range'][1]:.3g} .. {r['range'][2]:.3g}   ({r['note']})")
        else:
            print(f"  {r['name']:44} " + ", ".join(f"{k}: {v:.4g}" for k, v in r["points"]) + f"   ({r['note']})")
    print(f"\nrho: byte LUT {d['rb_lut'] / d['rp']:.1f}, bit-planes {d['rb_planes'] / d['rp']:.1f}, Life {d['rb_life'] / d['rp']:.1f}")
