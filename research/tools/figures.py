"""Figures of paper.md, drawn from the medians in the newest dated folder of research/logs that holds each
target (paper.md, Appendix A; all of these from research/logs/2026-11-26): Fig. 2 is read from the logs,
Figs. 3 and 4 restate the tables their captions name. Writes SVG files to research/figures/.
Usage (from research/): python tools/figures.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgplot import chart  # noqa: E402
import ghl_logs  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)

# Fig. 3: brute-force kernels by grid size (cells counted as in Table brute)
CTR, CHA, SQ, ROM = 37627, 19932, 274950, 9467605
chart(
    os.path.join(OUT, "brute_by_size.svg"),
    "cells in the grid",
    "cell-updates/s",
    [
        ("32 states, byte LUT", [(CTR, 1.06e10), (SQ, 4.91e10), (ROM, 1.85e11), (1.87e7, 1.77e11), (1.69e8, 1.855e11)]),
        # 1024^2 to 4096^2: window.txt (section E, real circuit); the rest bitplane-sizes.txt
        ("32 states, bit-planes", [(CTR, 1.08e10), (SQ, 5.00e10), (1024**2, 1.47e11), (2048**2, 3.44e11), (4096**2, 4.55e11),
                               (ROM, 3.87e11), (8192**2, 2.96e11), (11584**2, 2.99e11), (16384**2, 3.03e11),
                               (23168**2, 3.05e11), (32768**2, 3.06e11)]),
        ("32 states, shared tile", [(CHA, 1.78e10), (CTR, 3.03e10), (SQ, 1.57e11), (ROM, 2.22e11), (16384**2, 2.45e11)]),
        # Life (Table lifebrute; logs life-kernels.txt): random soups filling the grid
        ("Life, plain", [(256**2, 3.05e10), (1024**2, 3.78e11), (4096**2, 1.43e12), (16384**2, 1.42e12), (32768**2, 1.46e12)]),
        ("Life, shared tile", [(256**2, 1.37e11), (1024**2, 7.57e11)]),
    ],
)

# Fig. 4: memo hit path by sweep tile on 32x32 phase-shifted ROMs (Table hitpath; 1.02e10 cells counted)
# and on 32x32 phase-shifted Unit Life Cells (Table lifehit)
chart(
    os.path.join(OUT, "hitpath_by_tile.svg"),
    "tile side (cells)",
    "generations/s",
    [
        ("hit path, 1.02·10¹⁰ cells", [(16, 857), (32, 20908), (64, 190267), (128, 1316642), (256, 7259759)]),
        ("hit path, one ROM", [(16, 473559), (32, 1647369)]),
        ("bit-plane brute force, one ROM", [(16, 40859), (32, 40859)]),
        # Life (Table lifehit; logs life-sweep.txt): 32x32 phase-shifted Unit Life Cells, 4.2e8 cells
        ("Life, 32x32 Unit Life Cells", [(32, 851582), (64, 3037102), (128, 7557567), (256, 20024161)]),
        ("Life, one Unit Life Cell", [(32, 2803002)]),
    ],
)
# Fig. 2: average rate over the whole run against its length (Table runlength), read from the logs:
# runlength.txt (one block per run length, Golly and the recursion) and tile.txt (the shared-tile kernel)
G = [16384, 65536, 262144, 2**20, 2**22, 2**24]
D = ghl_logs.collect()
RL = D["runlength"]
assert len(RL) == len(G)


def rl(pattern, side):
    """gens/s of Golly (side 0) or the recursion (side 1) for each run length of G."""
    return [(g, b[pattern][side]) for g, b in zip(G, RL)]



chart(
    os.path.join(OUT, "runlength.svg"),
    "generations in the run",
    "generations/s, averaged over the run",
    [
        ("Golly, counter", rl(ghl_logs.COUNTER, 0)),
        ("CPU recursion, counter", rl(ghl_logs.COUNTER, 1)),
        ("GPU shared tile, counter", [(G[0], D["tile_counter"]), (G[-1], D["tile_counter"])]),
        ("Golly, chaotic", rl(ghl_logs.CHAOTIC, 0)),
        ("CPU recursion, chaotic", rl(ghl_logs.CHAOTIC, 1)),
        ("GPU shared tile, chaotic", [(G[0], D["tile_chaotic"]), (G[-1], D["tile_chaotic"])]),
    ],
)
print("figures written to", os.path.abspath(OUT))
