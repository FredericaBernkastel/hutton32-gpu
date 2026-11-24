"""Convert a palette PNG of the archived app (one pixel per cell, colours = HUTTON32_COLORS in
archive/src/gpu/gpu_automata.rs) to Golly extended RLE, so that the Rust micro-benchmark needs
no image decoder.

usage: python png_to_rle.py <in.png> <out.rle>
"""
import sys

import numpy as np
from PIL import Image

COLORS = [
    (0, 0, 0), (255, 0, 0), (255, 125, 0), (255, 150, 25), (255, 175, 50), (255, 200, 75), (255, 225, 100),
    (255, 250, 125), (251, 255, 0), (89, 89, 255), (106, 106, 255), (122, 122, 255), (139, 139, 255),
    (27, 176, 27), (36, 200, 36), (73, 255, 73), (106, 255, 106), (235, 36, 36), (255, 56, 56), (255, 73, 73),
    (255, 89, 89), (185, 56, 255), (191, 73, 255), (197, 89, 255), (203, 106, 255), (0, 255, 128),
    (255, 128, 64), (255, 255, 128), (33, 215, 215), (27, 176, 176), (24, 156, 156), (21, 137, 137),
]


def state_token(s: int) -> str:
    if s == 0:
        return "."
    s -= 1
    return (chr(ord("p") + s // 24 - 1) if s >= 24 else "") + chr(ord("A") + s % 24)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    img = np.array(Image.open(src).convert("RGB"))
    h, w, _ = img.shape
    key = {c: i for i, c in enumerate(COLORS)}
    cells = np.zeros((h, w), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            cells[y, x] = key[tuple(int(v) for v in img[y, x])]   # KeyError = colour not in palette
    items, pending = [], 0
    for y in range(h):
        row = cells[y]
        nz = np.flatnonzero(row)
        if nz.size:
            if pending:
                items.append((pending, "$"))
                pending = 0
            x, end = 0, int(nz[-1]) + 1
            while x < end:
                run = 1
                while x + run < end and row[x + run] == row[x]:
                    run += 1
                items.append((run, state_token(int(row[x]))))
                x += run
        pending += 1
    body, line = [], ""
    for count, tok in items:
        item = (str(count) if count > 1 else "") + tok
        if len(line) + len(item) > 70:
            body.append(line)
            line = ""
        line += item
    body.append(line + "!")
    with open(dst, "w", newline="\n") as f:
        f.write(f"# converted from {src.replace(chr(92), '/').split('/')[-1]} by png_to_rle.py\n")
        f.write(f"x = {w}, y = {h}, rule = Hutton32b\n")
        f.write("\n".join(body) + "\n")
    print(f"{w}x{h}, {int((cells != 0).sum())} live cells, states used: {sorted(set(map(int, np.unique(cells))))}")


if __name__ == "__main__":
    main()
