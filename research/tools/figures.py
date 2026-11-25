"""Figures of paper.md, drawn from the medians reported in research/logs/2026-11-25 (see the tables
the captions name). Writes SVG files to research/figures/. Usage (from research/): python tools/figures.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from svgplot import chart  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(OUT, exist_ok=True)

# Fig. 3: brute-force kernels by grid size (cells counted as in Table brute)
CTR, CHA, SQ, ROM = 37627, 19932, 274950, 9467605
chart(
    os.path.join(OUT, "brute_by_size.svg"),
    "cells in the grid",
    "cell-updates/s",
    [
        ("32 states, byte LUT", [(CTR, 1.06e10), (SQ, 4.89e10), (ROM, 1.84e11), (1.87e7, 1.68e11), (1.69e8, 1.74e11)]),
        ("32 states, bit-planes", [(CTR, 1.07e10), (SQ, 4.96e10), (1024**2, 1.44e11), (2048**2, 3.48e11), (4096**2, 4.59e11),
                               (ROM, 3.91e11), (8192**2, 2.90e11), (11584**2, 2.96e11), (16384**2, 2.97e11),
                               (23168**2, 3.01e11), (32768**2, 3.02e11)]),
        ("32 states, shared tile", [(CHA, 1.79e10), (CTR, 3.02e10), (SQ, 1.56e11), (ROM, 2.21e11), (16384**2, 1.58e11)]),
        # Life (Table lifebrute; logs life-kernels.txt): random soups filling the grid
        ("Life, plain", [(256**2, 3.03e10), (1024**2, 3.71e11), (4096**2, 1.42e12), (16384**2, 1.41e12), (32768**2, 9.86e11)]),
        ("Life, shared tile", [(256**2, 1.37e11), (1024**2, 7.49e11)]),
    ],
)

# Fig. 4: memo hit path by sweep tile on 32x32 phase-shifted ROMs (Table hitpath; 1.02e10 cells counted)
# and on 32x32 phase-shifted Unit Life Cells (Table lifehit)
chart(
    os.path.join(OUT, "hitpath_by_tile.svg"),
    "tile side (cells)",
    "generations/s",
    [
        ("hit path, 1.02·10¹⁰ cells", [(16, 823), (32, 20686), (64, 189513), (128, 1305890), (256, 7190399)]),
        ("hit path, one ROM", [(16, 471546), (32, 1652238)]),
        ("bit-plane brute force, one ROM", [(16, 41321), (32, 41321)]),
        # Life (Table lifehit; logs life-sweep.txt): 32x32 phase-shifted Unit Life Cells, 4.2e8 cells
        ("Life, 32x32 Unit Life Cells", [(32, 841637), (64, 2981849), (128, 7493512), (256, 19260188)]),
        ("Life, one Unit Life Cell", [(32, 2775355)]),
    ],
)
# Fig. 2: average rate over the whole run against its length (Table runlength; logs runlength.txt, tile.txt)
G = [16384, 65536, 262144, 2**20, 2**22, 2**24]
chart(
    os.path.join(OUT, "runlength.svg"),
    "generations in the run",
    "generations/s, averaged over the run",
    [
        ("Golly, counter", list(zip(G, [265899, 363749, 482473, 648717, 2511205, 10068380]))),
        ("CPU recursion, counter", list(zip(G, [134959, 200692, 261503, 592048, 2331335, 8943556]))),
        ("GPU shared tile, counter", [(G[0], 801737), (G[-1], 801737)]),
        ("Golly, chaotic", list(zip(G, [50982, 30675, 93072, 327611, 1121131, 3154029]))),
        ("CPU recursion, chaotic", list(zip(G, [28561, 21995, 55357, 185205, 604759, 2031091]))),
        ("GPU shared tile, chaotic", [(G[0], 900168), (G[-1], 900168)]),
    ],
)
print("figures written to", os.path.abspath(OUT))
