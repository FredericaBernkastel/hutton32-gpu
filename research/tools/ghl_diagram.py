"""The overview diagram of GPU HashLife (GHL) at the head of paper.md, as one SVG without plotting
libraries. Everything it shows is data:

  * out/overview.json     patterns as cells, tile IDs, a memo table, one sweep, the waves of the
                          recursion, the controller (written by `ca-research overview`, which runs
                          the CPU reference engine);
  * logs/2026-11-25/      the measured rates and the comparison with Golly (tools/ghl_logs.py);
  * bitslice/circuit.netlist, src/cmd_life.rs    the two gate networks (tools/ghl_netlist.py).

While drawing, the script checks what it shows: the netlist against the harness's next generation
on every cell of the frame, the Life adder tree against a neighbour count, and the memo entry of
panel 3 against the stepped window.

Usage (from research/):  cargo run --release -- overview && python tools/ghl_diagram.py
Writes research/figures/ghl_overview.svg.
"""
import base64
import json
import math
import os
import re
import struct
import sys
import zlib
from xml.sax.saxutils import escape

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ghl_logs  # noqa: E402
import ghl_netlist  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
OUT = os.path.join(ROOT, "figures", "ghl_overview.svg")
DATA = os.path.join(ROOT, "out", "overview.json")
OPT_LOG = os.path.join(ROOT, "bitslice", "optimize.log")

W, M = 1200, 20
CW = W - 2 * M
FONT = "Helvetica, Arial, 'Liberation Sans', sans-serif"
INK, GREY, MID, LIGHT = "#1a1a1a", "#555555", "#8c8c8c", "#d6d6d6"
# Okabe-Ito
BLUE, SKY, ORANGE, GREEN, VERMIL, PURPLE = "#0072B2", "#56B4E9", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
GATE = {"AND": BLUE, "OR": ORANGE, "XOR": GREEN, "NOT": "#3a3a3a"}
B32 = bytes(range(256)).translate(bytes.maketrans(b"0123456789abcdefghijklmnopqrstuv", bytes(range(32))))


# ---- numbers -------------------------------------------------------------------------------------------

def thousands(n):
    return f"{int(round(n)):,}"


def sci(v, sig=2):
    """1.4·10^{4}: mantissa to `sig` figures, exponent as a superscript (markup of `rich`)."""
    e = math.floor(math.log10(abs(v)))
    m = round(v / 10 ** e, sig - 1)
    if m >= 10:
        m, e = m / 10, e + 1
    return f"{m:.{sig - 1}f}·10^{{{e}}}"


def times(v):
    """A ratio to two figures, as the paper prints it."""
    if v >= 1000:
        return sci(v) + "×"
    if v >= 100:
        return f"{v:.0f}×"
    if v >= 10:
        return f"{v:.0f}×"
    if v >= 1:
        return f"{v:.1f}×"
    return f"{v:.2f}×" if v >= 0.095 else f"{v:.3f}×"


def pct(v, digits=1):
    return f"{100 * v:.{digits}f} %"


def compact(n, room):
    """`n` in full if it fits `room` pixels at 9.5 px, else as 9.1k or 36k."""
    for s in (thousands(n), f"{n / 1000:.1f}k", f"{n / 1000:.0f}k"):
        if text_width(s, 9.5) + 3 <= room:
            return s
    return ""


# ---- text metrics and SVG ---------------------------------------------------------------------------------

# advance widths of Helvetica / Arial in 1/1000 em, regular and bold (measured in a browser)
_CHARS = " !\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~×·²³ρ→–≈≥≤µ°…"
_REG = ("278,278,355,556,556,889,667,191,333,333,389,584,278,333,278,278,556,556,556,556,556,556,556,556,556,556,278,278,584,584,584,556,1015,"
        "667,667,722,722,667,611,778,722,278,500,667,556,833,722,778,667,778,722,667,611,722,667,944,667,667,611,278,278,278,469,556,333,"
        "556,556,500,556,556,278,556,556,222,222,500,222,833,556,556,556,556,333,500,278,556,500,722,500,500,500,334,260,334,584,"
        "584,333,333,333,569,1000,556,549,549,549,576,400,1000")
_BOLD = ("278,333,474,556,556,889,722,238,333,333,389,584,278,333,278,278,556,556,556,556,556,556,556,556,556,556,333,333,584,584,584,611,975,"
         "722,722,722,722,667,611,778,722,278,556,722,611,833,722,778,667,778,722,667,611,722,667,944,667,667,611,333,278,333,584,556,333,"
         "556,611,556,611,556,333,611,611,278,278,556,278,889,611,611,611,611,389,556,333,611,556,778,556,556,500,389,280,389,584,"
         "584,333,333,333,619,1000,556,549,549,549,576,400,1000")
_ADV = {False: dict(zip(_CHARS, (int(v) for v in _REG.split(",")))), True: dict(zip(_CHARS, (int(v) for v in _BOLD.split(","))))}
assert len(_ADV[False]) == len(_CHARS) == len(_ADV[True])
_MARK = re.compile(r"([_^])\{([^}]*)\}")


def text_width(s, size, bold=False):
    adv = _ADV[bool(bold)]
    total, pos = 0.0, 0
    for m in _MARK.finditer(s):
        total += sum(adv.get(c, 556) for c in s[pos:m.start()])
        total += 0.72 * sum(adv.get(c, 556) for c in m.group(2))
        pos = m.end()
    total += sum(adv.get(c, 556) for c in s[pos:])
    return total * size / 1000


def wrap(s, size, width, bold=False):
    lines, cur = [], ""
    for word in s.split(" "):
        trial = f"{cur} {word}" if cur else word
        if cur and text_width(trial, size, bold) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur] if cur else lines


def rich(s, size):
    """Escape `s`; `^{x}` becomes a superscript and `_{x}` a subscript."""
    out, pos, cur = [], 0, 0.0
    small = round(size * 0.72, 1)
    for m in _MARK.finditer(s):
        seg = escape(s[pos:m.start()])
        if seg:
            out.append(f'<tspan dy="{-cur:.1f}">{seg}</tspan>' if cur else seg)
            cur = 0.0
        target = -0.36 * size if m.group(1) == "^" else 0.22 * size
        out.append(f'<tspan dy="{target - cur:.1f}" font-size="{small}">{escape(m.group(2))}</tspan>')
        cur = target
        pos = m.end()
    seg = escape(s[pos:])
    if seg:
        out.append(f'<tspan dy="{-cur:.1f}">{seg}</tspan>' if cur else seg)
    return "".join(out)


class Svg:
    def __init__(self):
        self.o = []

    def text(self, x, y, s, size=11, weight=None, anchor="start", fill=INK, halo=None):
        a = f' font-weight="{weight}"' if weight else ""
        a += f' text-anchor="{anchor}"' if anchor != "start" else ""
        a += f' stroke="{halo}" stroke-width="3.2" stroke-linejoin="round" paint-order="stroke"' if halo else ""
        self.o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"{a}>{rich(s, size)}</text>')

    def para(self, x, y, s, width, size=11, fill=GREY, lead=None, weight=None):
        """`s` wrapped to `width`, first baseline at y; returns the baseline after the last line."""
        lead = lead or round(size * 1.27, 1)
        for line in wrap(s, size, width, weight == "bold"):
            self.text(x, y, line, size=size, fill=fill, weight=weight)
            y += lead
        return y

    def rect(self, x, y, w, h, fill="none", stroke=None, sw=1, rx=0, dash=None, opacity=None):
        a = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        a += f' rx="{rx}"' if rx else ""
        a += f' stroke-dasharray="{dash}"' if dash else ""
        a += f' opacity="{opacity}"' if opacity is not None else ""
        self.o.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" fill="{fill}"{a}/>')

    def line(self, x1, y1, x2, y2, stroke=INK, sw=1, dash=None, marker=None, opacity=None):
        a = f' stroke-dasharray="{dash}"' if dash else ""
        a += f' marker-end="url(#ah-{marker})"' if marker else ""
        a += f' opacity="{opacity}"' if opacity is not None else ""
        self.o.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{sw}"{a}/>')

    def path(self, d, stroke=INK, sw=1, fill="none", opacity=None, dash=None, marker=None, join=None):
        a = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        a += f' opacity="{opacity}"' if opacity is not None else ""
        a += f' stroke-dasharray="{dash}"' if dash else ""
        a += f' marker-end="url(#ah-{marker})"' if marker else ""
        a += f' stroke-linejoin="{join}"' if join else ""
        self.o.append(f'<path d="{d}" fill="{fill}"{a}/>')

    def circle(self, x, y, r, fill=INK, stroke=None, sw=1, opacity=None):
        a = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        a += f' opacity="{opacity}"' if opacity is not None else ""
        self.o.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r}" fill="{fill}"{a}/>')

    def image(self, x, y, w, h, png):
        """A PNG at (x, y) scaled to w x h. Pictures of cells are written at a multiple of their
        display size, so the viewer's smooth scaling keeps them crisp at any zoom."""
        uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        self.o.append(f'<image x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" preserveAspectRatio="none" href="{uri}" xlink:href="{uri}"/>')

    def arrow(self, x1, y1, x2, y2, colour=GREY, sw=1.3, dash=None):
        self.line(x1, y1, x2, y2, stroke=colour, sw=sw, dash=dash, marker={INK: "ink", GREY: "grey", BLUE: "blue"}[colour])

    def badge(self, x, y, n, r=11.0, size=13.5):
        self.circle(x, y, r, fill=INK)
        self.text(x, y + size * 0.35, str(n), size=size, weight="bold", anchor="middle", fill="#ffffff")

    def cone(self, p1, p2, q1, q2, split, axis, inside="#ffffff", outside=MID, fill=("#ffffff", 0.15, "#000000", 0.05)):
        """A zoom cone from the edge p1-p2 of a region of a dark picture to the edge q1-q2 of its
        enlargement on the page. The picture ends at `split` along `axis` (0: x, 1: y): up to there
        the border lines are drawn in `inside`, beyond it in `outside`."""
        d = f"M{p1[0]:.1f},{p1[1]:.1f}L{q1[0]:.1f},{q1[1]:.1f}L{q2[0]:.1f},{q2[1]:.1f}L{p2[0]:.1f},{p2[1]:.1f}Z"
        self.path(d, stroke=None, fill=fill[0], opacity=fill[1])
        self.path(d, stroke=None, fill=fill[2], opacity=fill[3])
        for p, q in ((p1, q1), (p2, q2)):
            t = min(max((split - p[axis]) / (q[axis] - p[axis]), 0.0), 1.0)
            mx, my = p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])
            if inside and t > 0:
                self.line(p[0], p[1], mx, my, stroke=inside, sw=1, opacity=0.9)
            self.line(mx, my, q[0], q[1], stroke=outside, sw=1)

    def finish(self, height):
        head = [f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{height:.0f}" '
                f'viewBox="0 0 {W} {height:.0f}" font-family="{FONT}">', "<defs>"]
        for name, colour in (("ink", INK), ("grey", GREY), ("blue", BLUE)):
            head.append(f'<marker id="ah-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
                        f'<path d="M0,0.8 L10,5 L0,9.2 z" fill="{colour}"/></marker>')
        head += ["</defs>", f'<rect width="{W}" height="{height:.0f}" fill="#ffffff"/>']
        return "\n".join(head + self.o + ["</svg>"]) + "\n"


# ---- pictures of cells ----------------------------------------------------------------------------------

def _chunk(tag, body):
    return struct.pack(">I", len(body)) + tag + body + struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF)


def _png(w, h, colour_type, raw, palette=None):
    plte = _chunk(b"PLTE", b"".join(bytes(c) for c in palette)) if palette else b""
    return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, colour_type, 0, 0, 0)) + plte
            + _chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + _chunk(b"IEND", b""))


def png_indexed(rows, palette, scale=1):
    """A palette PNG of `rows` (bytes of colour indices), every cell `scale` x `scale` pixels."""
    raw = bytearray()
    for r in rows:
        line = bytearray(len(r) * scale)
        for k in range(scale):
            line[k::scale] = r
        raw += (b"\x00" + bytes(line)) * scale
    return _png(len(rows[0]) * scale, len(rows) * scale, 3, raw, palette)


def png_grey(rows):
    return _png(len(rows[0]), len(rows), 0, b"".join(b"\x00" + bytes(r) for r in rows))


def png_rgb(w, h, rgb):
    return _png(w, h, 2, b"".join(b"\x00" + rgb[y * w * 3:(y + 1) * w * 3] for y in range(h)))


def grid_rows(g, x0=0, y0=0, w=None, h=None):
    """Rows of states (bytes) of a grid of overview.json, or of a window of it."""
    gw = g["w"]
    cells = g["cells"].encode("ascii").translate(B32)
    w = gw - x0 if w is None else w
    h = g["h"] - y0 if h is None else h
    return [cells[(y0 + y) * gw + x0:(y0 + y) * gw + x0 + w] for y in range(h)]


def tile_rows(cells, side):
    b = cells.encode("ascii").translate(B32)
    return [b[y * side:(y + 1) * side] for y in range(side)]


def mosaic(tiles, side, cols, gap_index):
    """Square tiles of states side by side with one cell of `gap_index` between them."""
    n_rows = -(-len(tiles) // cols)
    out = [bytearray([gap_index]) * (cols * side + cols - 1) for _ in range(n_rows * side + n_rows - 1)]
    for k, t in enumerate(tiles):
        ox, oy = (k % cols) * (side + 1), (k // cols) * (side + 1)
        for r in range(side):
            out[oy + r][ox:ox + side] = t[r]
    return [bytes(r) for r in out]


LIFE_PALETTE = [(0, 0, 0), (255, 255, 255)]


# ---- shared drawing ---------------------------------------------------------------------------------------

def section(svg, y, n, title, sub=None, badge=None):
    """A numbered heading across the page; returns the y below it."""
    svg.line(M, y, W - M, y, stroke=LIGHT, sw=1)
    cy = y + 25
    svg.badge(M + 11, cy - 5.5, n)
    svg.text(M + 30, cy, title, size=17, weight="bold")
    if sub:
        svg.text(M + 30 + text_width(title, 17, True) + 12, cy, sub, size=12.5, fill=GREY)
    if badge:
        chips(svg, W - M, cy - 14, badge, right=True)
    return y + 42


def chips(svg, x, y, items, right=False, size=10.5):
    """Small labelled chips in a row: [(label, kind)], kind in gpu / cpu / model."""
    style = {"gpu": (INK, "#ffffff", INK, None), "cpu": ("#ffffff", INK, INK, None), "model": ("#ffffff", GREY, GREY, "3 2")}
    widths = [text_width(lab, size) + 16 for lab, _ in items]
    if right:
        x -= sum(widths) + 6 * (len(items) - 1)
    for (lab, kind), w in zip(items, widths):
        fill, ink, stroke, dash = style[kind]
        svg.rect(x, y, w, 17, fill=fill, stroke=stroke, sw=1, rx=8.5, dash=dash)
        svg.text(x + w / 2, y + 12.2, lab, size=size, anchor="middle", fill=ink)
        x += w + 6


def log_grid(svg, pos, lo, hi, a, b, vertical=True, major="#ffffff", minor="#c9c9c9", major_sw=1.0):
    """The grid of a logarithmic axis between 10^lo and 10^hi: a solid line at every power of ten
    and dashed lines at 2..9 times it, which bunch up towards the next power and so show the scale.
    `pos` maps a value to a coordinate; the lines run from a to b across the other axis."""
    solid, dashed = [], []
    for e in range(math.floor(lo), math.ceil(hi) + 1):
        for k in range(1, 10):
            v = k * 10.0 ** e
            if not 10 ** lo * 0.999 <= v <= 10 ** hi * 1.001:
                continue
            c = pos(v)
            (solid if k == 1 else dashed).append(f"M{c:.1f},{a:.1f}V{b:.1f}" if vertical else f"M{a:.1f},{c:.1f}H{b:.1f}")
    if dashed:
        svg.path("".join(dashed), stroke=minor, sw=0.6, dash="2 2")
    if solid and major:
        svg.path("".join(solid), stroke=major, sw=major_sw)


def bit_strip(svg, x, y, word, cell, h, on=INK, off="#ececec", bits=32):
    """A word as a row of cells: bit i is cell i."""
    svg.rect(x, y, bits * cell, h, fill=off)
    start = None
    for i in range(bits + 1):
        bit = i < bits and (word >> i) & 1
        if bit and start is None:
            start = i
        if not bit and start is not None:
            svg.rect(x + start * cell, y, (i - start) * cell, h, fill=on)
            start = None


def draw_network(svg, net, L, val, bits=32):
    """Wires shaded by the share of the cells on which the signal is 1, then the gates."""
    by_shade = {}
    for sig, (xa, ya), (xb, yb) in L.wires:
        k = round(bin(val[sig]).count("1") / bits * 8) / 8
        m = (xb - xa) * 0.5
        by_shade.setdefault(k, []).append(f"M{xa:.1f},{ya:.1f}C{xa + m:.1f},{ya:.1f} {xb - m:.1f},{yb:.1f} {xb:.1f},{yb:.1f}")
    for k in sorted(by_shade):
        v = int(round(198 - 172 * k))
        svg.path("".join(by_shade[k]), stroke=f"#{v:02x}{v:02x}{v:02x}", sw=0.75 + 0.35 * k)
    for g in net.order:
        x, y = L.pos[g]
        op = net.gates[g][0]
        if op == "NOT":
            svg.path(f"M{x - 2.4:.1f},{y - 2.3:.1f}L{x + 2.2:.1f},{y:.1f}L{x - 2.4:.1f},{y + 2.3:.1f}Z", stroke=None, fill=GATE[op])
        else:
            svg.circle(x, y, 3.0 if op == "XOR" else 2.6, fill=GATE[op], stroke="#ffffff", sw=0.5)


# ---- the algorithm in one strip ---------------------------------------------------------------------------------

def flow(svg, y):
    h = 28
    ya, yb = y + 4, y + 52

    def box(x, yy, w, n, label):
        svg.rect(x, yy, w, h, fill="#f1f1f1", rx=6)
        svg.badge(x + 16, yy + h / 2, n, r=8.5, size=11)
        svg.text(x + 31, yy + 18.6, label, size=12.5, weight="bold")

    a1, a2, a3, b3 = (M, 232), (302, 252), (616, 280), (946, 234)
    box(a1[0], ya, a1[1], 3, "the universe: a grid of tile IDs")
    box(a2[0], ya, a2[1], 3, "hit path: one probe per tile")
    box(a3[0], ya, a3[1], 3, "the grid s/2 generations on; repeat")
    box(a2[0], yb, a2[1], 4, "miss path: waves, one per level")
    box(a3[0], yb, a3[1], 4, "at the floor: an atlas of windows")
    box(b3[0], yb, b3[1], 2, "the rule as a circuit")
    svg.arrow(a1[0] + a1[1] + 5, ya + h / 2, a2[0] - 5, ya + h / 2)
    svg.arrow(a2[0] + a2[1] + 5, ya + h / 2, a3[0] - 5, ya + h / 2)
    svg.text((a2[0] + a2[1] + a3[0]) / 2, ya + h / 2 - 6, "hit", anchor="middle", fill=GREY)
    mx = a2[0] + a2[1] / 2
    svg.arrow(mx, ya + h + 3, mx, yb - 4)
    svg.text(mx + 7, (ya + h + yb) / 2 + 3.5, "miss", fill=GREY)
    svg.arrow(a2[0] + a2[1] + 5, yb + h / 2, a3[0] - 5, yb + h / 2)
    svg.arrow(a3[0] + a3[1] + 5, yb + h / 2, b3[0] - 5, yb + h / 2)
    px = b3[0] + b3[1] / 2
    svg.path(f"M{px:.1f},{yb - 3}V{ya + h / 2:.1f}H{a3[0] + a3[1] + 7:.1f}", stroke=GREY, sw=1.3, marker="grey")
    svg.text(px - 8, ya + h / 2 - 6, "results interned, memo filled", anchor="end", fill=GREY)
    svg.rect(a1[0], yb, a1[1], h, fill="#ffffff", stroke=GREY, sw=1, rx=6, dash="3 2")
    svg.badge(a1[0] + 16, yb + h / 2, 5, r=8.5, size=11)
    svg.text(a1[0] + 31, yb + 18.6, "measured rates set the floor", size=12.5, weight="bold")
    svg.arrow(a1[0] + a1[1] + 5, yb + h / 2, a2[0] - 5, yb + h / 2, dash="4 3")
    return yb + h + 14


# ---- 1. the universes -----------------------------------------------------------------------------------------

def log_bins(windows, misses, gens_per_sweep, factor=1.3):
    """Miss rate over growing spans of sweeps: [(generation at the end of the span, rate)]."""
    out, i, span = [], 0, 1.0
    n = len(windows)
    while i < n:
        j = min(n, max(i + 1, int(round(i + span))))
        out.append((j * gens_per_sweep, sum(misses[i:j]) / max(1, sum(windows[i:j]))))
        span *= factor
        i = j
    return out


def panel_universes(svg, y, D):
    P, pal = D["patterns"], D["palette"]
    learn = D["learning"]
    gap = 14
    cw = (CW - 5 * gap) / 6
    bh = 150.0
    sq = D["sweep"]
    mg = sq["margin"]
    sq0 = P["squares"]["gen0"]

    def ulc_png():
        g = P["ulc"]["gen0"]
        rows = grid_rows(g)
        n = 300
        f = g["w"] / n
        out = []
        for j in range(n):
            ya, yb = int(j * f), max(int(j * f) + 1, int((j + 1) * f))
            line = bytearray(n)
            for i in range(n):
                xa, xb = int(i * f), max(int(i * f) + 1, int((i + 1) * f))
                live = sum(sum(r[xa:xb]) for r in rows[ya:yb])
                line[i] = min(255, int(255 * min(1.0, 1.6 * live / ((xb - xa) * (yb - ya))) ** 0.55))
            out.append(line)
        return png_grey(out)

    rom, cnt, cha, soup = P["rom"], P["counter"]["gen0"], P["chaotic"], P["soup"]
    iw, ih = int(cw), int(bh)
    cards = [
        ("squares", "squares", [f"{sq0['w']} × {sq0['h']} cells · Hutton32", "computes and displays n^{2}", f"shown: generation {thousands(sq['generation'])}"],
         png_indexed(grid_rows(sq["cells"], mg, mg, sq0["w"], sq0["h"]), pal, 1), sq0["w"], sq0["h"]),
        ("counter", "8-bit counter", [f"{cnt['w']} × {cnt['h']} cells · Hutton32b", "dense circuit", "period 513,024 generations"], png_indexed(grid_rows(cnt), pal, 2), cnt["w"], cnt["h"]),
        ("chaotic", "chaotic", [f"{cha['gen0']['w']} × {cha['gen0']['h']} cells · Hutton32b", "dense, chaotic", f"shown: generation {thousands(cha['later_gen'])}"],
         png_indexed(grid_rows(cha["later"]), pal, 3), cha["later"]["w"], cha["later"]["h"]),
        ("rom", "circulating ROM", [f"{thousands(rom['w'])} × {thousands(rom['h'])} · Hutton32b", "dense, period 528", "shown: a detail, cell = pixel"],
         png_indexed(grid_rows(rom["crop"], 0, 0, iw, ih), pal, 2), iw, ih),
        ("soup", "random soup", [f"{thousands(soup['side'])} × {thousands(soup['side'])} cells · Life", "dense, chaotic", f"shown: a detail, generation {soup['view_gen']}"],
         png_indexed(grid_rows(soup["later"], 0, 0, iw, ih), LIFE_PALETTE, 2), iw, ih),
        ("ulc", "Unit Life Cell", [f"{P['ulc']['gen0']['w']} × {P['ulc']['gen0']['h']} cells · Life", "sparse", "period 5,760 generations"], ulc_png(), 1, 1),
    ]
    sy = y + bh + 68          # top of the curves
    sh = 34.0
    lo, hi = 1.0, 5.5         # log10 generations

    def gx(x0, gen):
        return x0 + cw * (math.log10(max(gen, 10)) - lo) / (hi - lo)

    for k, (key, name, lines, png, pw, ph) in enumerate(cards):
        x = M + k * (cw + gap)
        svg.rect(x, y, cw, bh, fill="#000000", rx=3)
        s = min(cw / pw, bh / ph)
        svg.image(x + (cw - pw * s) / 2, y + (bh - ph * s) / 2, pw * s, ph * s, png)
        svg.text(x, y + bh + 17, name, size=13, weight="bold")
        for q, line in enumerate(lines):
            assert text_width(line, 11) <= cw, line
            svg.text(x, y + bh + 31.5 + 13.5 * q, line, fill=GREY)
        # share of windows whose result is not in the memo, against generations run
        L = learn[key]
        pts = log_bins(L["windows"], L["misses"], L["gens_per_sweep"])
        svg.rect(x, sy, cw, sh, fill="#f5f5f5")
        log_grid(svg, lambda v: gx(x, v), lo, hi, sy, sy + sh)
        d, px = [], x
        for gen, r in pts:
            xx, yy = gx(x, gen), sy + sh * (1 - r)
            d.append(f"{px:.1f},{yy:.1f} {xx:.1f},{yy:.1f}")
            px = xx
        svg.path(f"M{x:.1f},{sy + sh:.1f} L" + " L".join(d) + f" L{px:.1f},{sy + sh:.1f} Z", stroke=None, fill=VERMIL, opacity=0.26)
        svg.path("M" + " L".join(d), stroke=VERMIL, sw=1.3, join="round")
        svg.line(x, sy + sh, x + cw, sy + sh, stroke=MID, sw=0.8)
        half = len(L["windows"]) // 2
        late = sum(L["misses"][half:]) / sum(L["windows"][half:])
        svg.text(x + cw - 3, sy + 11, "0 %" if late == 0 else pct(late, 0 if late >= 0.095 else 1), weight="bold", anchor="end", fill=VERMIL, halo="#f5f5f5")
        for e in (1, 3, 5):
            svg.text(gx(x, 10 ** e), sy + sh + 11.5, f"10^{{{e}}}" if e > 1 else "10", size=9.5, fill=GREY, anchor="start" if e == 1 else "middle")
    L = learn
    y = svg.para(M, sy + sh + 29, "Curves: the share of tile-steps not yet in the memo table, for tiles of 16×16 cells (Life: 32×32), against the generations run so far "
                 f"(log scale; the ROM's curve is that of a smaller ROM of the same kind, {thousands(L['rom']['w'])} × {thousands(L['rom']['h'])} cells, the soup's that of a "
                 f"{L['soup']['w']} × {L['soup']['h']} soup). The figure is the share over the second half of the run. "
                 "This novelty, not density, decides the engine: the ROM is new for one period and then entirely known, the soup stays new.", CW)
    return y - 4


# ---- 2. below the floor: the rule as a circuit ------------------------------------------------------------------

GROUPS = [("n", "north"), ("w", "west"), ("c", "centre"), ("e", "east"), ("s", "south")]
# Active cells from which a floor of 32x32 and of 16x16 windows pays for its dispatches: the model of
# paper.md, Table XXVII, at the bit-plane rate (the one figure pair here that no log file holds).
DEPTH_A1, DEPTH_A2 = 1.0e6, 1.3e7


def pick_cells(rows, nxt, width=32):
    """The 32 cells of the picture: the row segment in which most cells change, then most states."""
    best, where = None, None
    for y in range(1, len(rows) - 1):
        for x in range(1, len(rows[0]) - width - 1, 4):
            seg = rows[y][x:x + width]
            score = (sum(a != b for a, b in zip(seg, nxt[y][x:x + width])), len(set(seg) | set(rows[y - 1][x:x + width]) | set(rows[y + 1][x:x + width])))
            if best is None or score > best:
                best, where = score, (x, y)
    return where


def optimisation_steps():
    """(seconds, gates) at every new best network in bitslice/optimize.log (two runs, one after the other)."""
    out, base, last = [], 0.0, 0.0
    for line in open(OPT_LOG, encoding="utf-8"):
        m = re.match(r"\[\s*(\d+)s\]\s*(.*)", line)
        if not m:
            continue
        t, rest = float(m.group(1)), m.group(2)
        if rest.startswith("=== optimize_more"):
            base = last
        last = max(last, base + t)
        g = re.search(r"new best (\d+) gates", rest)
        if g:
            out.append((base + t, int(g.group(1))))
    return out


def panel_circuit(svg, y, D, logs):
    pal = D["palette"]
    hexpal = ["#%02x%02x%02x" % tuple(c) for c in pal]
    net = ghl_netlist.parse_netlist()
    sq = D["sweep"]
    rows, nxt = grid_rows(sq["cells"]), grid_rows(sq["cells_next"])
    gw = sq["cells"]["w"]
    # the network on every cell of the frame against the next generation by the rule table (the harness)
    got = ghl_netlist.step_grid(net, rows, gw)
    bad = sum(a != b for ra, rb in zip(got, nxt) for a, b in zip(ra, rb))
    assert bad == 0, f"the netlist differs from the rule on {bad} cells of the frame"
    cx, cy = pick_cells(rows, nxt)
    mask = (1 << 32) - 1
    seg = {"c": rows[cy][cx:cx + 32], "n": rows[cy - 1][cx:cx + 32], "s": rows[cy + 1][cx:cx + 32],
           "e": rows[cy][cx + 1:cx + 33], "w": rows[cy][cx - 1:cx + 31]}
    words = {f"{g}{k}": ghl_netlist.planes(seg[g], 5)[k] for g in seg for k in range(5)}
    val = net.evaluate(words, mask)
    out_states = bytes(sum(((val[net.outputs[f"o{k}"]] >> x) & 1) << k for k in range(5)) for x in range(32))
    assert out_states == nxt[cy][cx:cx + 32]

    cell, ph, pitch = 4.0, 7.0, 10.5        # width of a cell, height and pitch of a plane strip
    sx = M + 28                              # left edge of the strips
    top = y + 36
    gh = 9 + 5 * pitch + 14                  # one group: its cells, five planes, a gap
    in_y = {}

    def strip_cells(x, yy, states, h=6.0):
        svg.rect(x - 0.5, yy - 0.5, 32 * cell + 1, h + 1, fill="#000000")
        run, start = None, 0
        for i in range(33):
            s = states[i] if i < 32 else None
            if s != run:
                if run:
                    svg.rect(x + start * cell, yy, (i - start) * cell, h, fill=hexpal[run])
                run, start = s, i

    svg.text(M, y + 11, "32 cells of squares (marked in panel 3) and their four neighbours", size=11.5)
    svg.text(M, y + 24.5, "as 25 words, one bit-plane each", size=11.5, fill=GREY)
    for gi, (g, gname) in enumerate(GROUPS):
        gy = top + gi * gh
        svg.text(sx - 5, gy + 6, gname, size=10, anchor="end")
        strip_cells(sx, gy, seg[g])
        for k in range(5):
            py = gy + 10 + k * pitch
            bit_strip(svg, sx, py, words[f"{g}{k}"], cell, ph)
            svg.text(sx - 5, py + 6.4, f"{g}_{{{k}}}", size=9.5, anchor="end", fill=GREY)
            in_y[f"{g}{k}"] = py + ph / 2
    bottom = top + 5 * gh - 14
    # the network
    x_in = sx + 32 * cell + 6
    x_out = W - M - 252
    out_y = {f"o{k}": (top + bottom) / 2 - 2 * pitch + k * pitch for k in range(5)}
    L = ghl_netlist.layout(net, in_y, out_y, x_in, x_out, top - 6, bottom + 6)
    draw_network(svg, net, L, val)
    # the five output words and the cells they make
    ox = x_out + 6
    for k in range(5):
        bit_strip(svg, ox, out_y[f"o{k}"] - ph / 2, val[net.outputs[f"o{k}"]], cell, ph)
        svg.text(ox + 32 * cell + 5, out_y[f"o{k}"] + 3.2, f"o_{{{k}}}", size=9.5, fill=GREY)
    ny = out_y["o4"] + ph / 2 + 9
    strip_cells(ox, ny, out_states)
    svg.text(ox, ny + 21, "the 32 cells, one generation later", size=11.5)
    ty = svg.para(ox, ny + 35, f"Checked while drawing: on all {thousands(gw * sq['cells']['h'])} cells of this frame the network gives the next generation of the pattern's own "
                  f"rule table, {D['patterns']['squares']['rule']}.", 236)
    # how the network was found
    steps = optimisation_steps()
    ty = svg.para(ox, ty + 10, "Gates by minutes of optimisation (resubstitution, ABC rewriting, MaxSAT):", 236, fill=INK)
    tx0, tw_, th_ = ox + 28, 176.0, 34.0
    t1 = steps[-1][0]
    g_lo, g_hi = steps[-1][1] - 12, steps[0][1] + 8
    ty0 = ty - 2
    d, prev = [], None
    for t, g in steps + [(t1 * 1.04, steps[-1][1])]:
        xx, yy = tx0 + tw_ * t / (t1 * 1.04), ty0 + th_ * (g_hi - g) / (g_hi - g_lo)
        d.append(f"{xx:.1f},{prev:.1f} {xx:.1f},{yy:.1f}" if prev is not None else f"{xx:.1f},{yy:.1f}")
        prev = yy
    svg.path("M" + " L".join(d), stroke=INK, sw=1.4)
    svg.line(tx0, ty0 + th_ + 3, tx0 + tw_, ty0 + th_ + 3, stroke=MID, sw=0.8)
    for mnt in (0, 10, 20):
        xx = tx0 + tw_ * mnt * 60 / (t1 * 1.04)
        svg.line(xx, ty0 + th_ + 3, xx, ty0 + th_ + 7, stroke=MID, sw=0.8)
        svg.text(xx, ty0 + th_ + 17, str(mnt), size=9.5, anchor="middle", fill=GREY)
    svg.text(tx0 - 5, ty0 + th_ * (g_hi - steps[0][1]) / (g_hi - g_lo) + 3.5, str(steps[0][1]), size=10.5, weight="bold", anchor="end")
    svg.text(tx0 + tw_ + 5, ty0 + th_ * (g_hi - steps[-1][1]) / (g_hi - g_lo) + 3.5, str(steps[-1][1]), size=10.5, weight="bold")
    # the numbers of the network
    count = net.count()
    assert len(net.gates) == steps[-1][1]
    svg.text(ox, top + 22, f"{len(net.gates)} gates", size=25, weight="bold")
    svg.text(ox + text_width(f"{len(net.gates)} gates", 25, True) + 10, top + 22, f"depth {net.depth()}", size=14, fill=GREY)
    zero, entries = logs["circuit_check"]
    svg.para(ox, top + 42, f"replace the rule table of 2^{{25}} = {thousands(entries)} entries. Mismatches on all of them: {int(zero)}.", 236, size=11.5)
    lx = ox
    for op in ("AND", "OR", "NOT", "XOR"):
        if op == "NOT":
            svg.path(f"M{lx + 0.6:.1f},{top + 79.5:.1f}L{lx + 6.4:.1f},{top + 83:.1f}L{lx + 0.6:.1f},{top + 86.5:.1f}Z", stroke=None, fill=GATE[op])
        else:
            svg.circle(lx + 3.5, top + 83, 3.5, fill=GATE[op])
        lab = f"{count[op]} {op}"
        svg.text(lx + 11, top + 87, lab, size=11.5)
        lx += 11 + text_width(lab, 11.5) + 11
    # level axis
    ay = bottom + 20
    for lv in range(1, L.depth + 1):
        xx = x_in + L.dx * lv
        svg.line(xx, ay - 4, xx, ay, stroke=MID, sw=0.8)
        if lv in (1, 5, 10, 15, 20, 25):
            svg.text(xx, ay + 11, str(lv), size=9.5, fill=GREY, anchor="middle")
    svg.line(x_in + L.dx, ay, x_in + L.dx * L.depth, ay, stroke=MID, sw=0.8)
    svg.text(x_in + L.dx * L.depth + 14, ay + 11, "logic level", size=10, fill=GREY)
    svg.text(x_in + L.dx, ay + 27, "Every gate works on a 32-bit word, one bit per cell. Wire shade: on how many of the 32 cells the signal is 1 (light: none, dark: all). "
             "A signal with several consumers is one wire with branches.", fill=GREY)
    return ay + 36, net, (cx, cy)


# ---- 2b. the same for Life; what the networks buy ------------------------------------------------------------------

def panel_life(svg, y, D, logs):
    life = ghl_netlist.parse_life()
    g = D["patterns"]["soup"]["later"]
    side = g["w"]
    rows = grid_rows(g)
    mask = (1 << side) - 1
    ints = [ghl_netlist.planes(r, 1)[0] for r in rows]

    def words(c, n, s_, m):
        return {"n": n, "s": s_, "e": c >> 1, "w": (c << 1) & m, "ne": n >> 1, "nw": (n << 1) & m, "se": s_ >> 1, "sw": (s_ << 1) & m}

    def nxt(c, n, s_, m):
        v = life.evaluate(words(c, n, s_, m), m)
        return v["b1"] & ~v["b2"] & ~v["b3"] & (v["b0"] | c) & m      # born with 3, survives with 2 or 3

    best, where = -1, None
    for yy in range(1, side - 1):
        new = nxt(ints[yy], ints[yy - 1], ints[yy + 1], mask) ^ ints[yy]
        for x in range(2, side - 34, 2):
            score = bin((new >> x) & 0xFFFFFFFF).count("1")
            if score > best:
                best, where = score, (x, yy)
    cx, cy = where
    m32 = (1 << 32) - 1
    c, n, s_ = [(ints[cy + d] >> (cx - 1)) & ((1 << 34) - 1) for d in (0, -1, 1)]     # with one cell on each side
    w = {k: (v >> 1) & m32 for k, v in words(c, n, s_, (1 << 34) - 1).items()}
    cc = (c >> 1) & m32
    val = life.evaluate(w, m32)
    out = val["b1"] & ~val["b2"] & ~val["b3"] & (val["b0"] | cc) & m32
    for i in range(32):                    # against counting the neighbours
        live = sum(rows[cy + dy][cx + i + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy)
        assert ((out >> i) & 1) == (1 if live == 3 or (live == 2 and rows[cy][cx + i]) else 0)

    cell, ph, pitch = 4.0, 7.0, 10.5
    sx = M + 28
    svg.para(M, y + 11, f"Life-like rules (B/S notation) need one plane. A carry-save adder tree counts the live neighbours of 32 cells in {len(life.gates)} operations; "
             "the rule's birth and survival sets are decoded from the four count bits.", 620, size=11.5, fill=INK)
    top = y + 46
    order = ["nw", "n", "ne", "w", "e", "sw", "s", "se"]
    in_y = {}
    for k, name in enumerate(order):
        py = top + k * pitch
        bit_strip(svg, sx, py, w[name], cell, ph)
        svg.text(sx - 5, py + 6.4, name, size=9.5, anchor="end", fill=GREY)
        in_y[name] = py + ph / 2
    bottom = top + 7 * pitch + ph
    x_in, x_out = sx + 32 * cell + 6, sx + 32 * cell + 196
    out_y = {f"b{k}": (top + bottom) / 2 - 1.5 * pitch + k * pitch for k in range(4)}
    L = ghl_netlist.layout(life, in_y, out_y, x_in, x_out, top - 2, bottom + 2, gate_gap=9.0, wire_gap=4.0)
    draw_network(svg, life, L, val)
    ox = x_out + 6
    for k in range(4):
        bit_strip(svg, ox, out_y[f"b{k}"] - ph / 2, val[f"b{k}"], cell, ph)
        svg.text(ox + 32 * cell + 5, out_y[f"b{k}"] + 3.2, f"b_{{{k}}}", size=9.5, fill=GREY)
    svg.text(ox, out_y["b0"] - ph / 2 - 6, "neighbour count, four bits", size=10.5, fill=GREY)
    # decode and the cells
    dx = ox + 32 * cell + 24
    svg.arrow(dx, (top + bottom) / 2, dx + 24, (top + bottom) / 2)
    bx = dx + 32
    svg.text(bx, out_y["b0"] - ph / 2 - 6, "born with 3, survives with 2 or 3", size=10.5, fill=GREY)
    for k, (lab, word) in enumerate((("now", cc), ("next", out))):
        yy = (top + bottom) / 2 - 12 + k * 16
        svg.rect(bx - 0.5, yy - 0.5, 32 * cell + 1, 9, fill="#000000")
        bit_strip(svg, bx, yy, word, cell, 8, on="#ffffff", off="#000000")
        svg.text(bx + 32 * cell + 5, yy + 7.2, lab, size=9.5, fill=GREY)
    svg.text(bx, bottom + 5, f"32 cells of the soup, generation {D['patterns']['soup']['view_gen']}", size=10.5, fill=GREY)

    # measured rates of the engine forms
    rx = bx + 32 * cell + 46
    forms = [("CPU, one core, table lookup", logs["cpu_lut"], MID, 2), ("CPU, four cores, the circuit", logs["cpu4"], MID, 2),
             ("GPU, table lookup, real patterns", logs["rb_lut"], SKY, 3), ("GPU, the circuit, any content", logs["rb_planes"], BLUE, 2),
             ("GPU, the Life adder tree", logs["rb_life"], BLUE, 3)]
    lab_w = 176
    bx0, bw = rx + lab_w, W - M - rx - lab_w - 58
    lo, hi = 8.0, 12.5
    svg.text(rx, y + 11, "Brute force, measured, in cell-updates per second", size=11.5)
    log_grid(svg, lambda v: bx0 + bw * (math.log10(v) - lo) / (hi - lo), lo, hi, y + 22, y + 25 + 5 * 17.5 + 3, major="#dcdcdc", minor="#d0d0d0")
    for k, (lab, v, col, sig) in enumerate(forms):
        yy = y + 25 + k * 17.5
        svg.text(rx + lab_w - 8, yy + 10.5, lab, anchor="end")
        wv = bw * (math.log10(v) - lo) / (hi - lo)
        svg.rect(bx0, yy + 2, wv, 11, fill=col)
        svg.text(bx0 + wv + 5, yy + 11, sci(v, sig), weight="bold" if col == BLUE else None, halo="#ffffff")
    ay = y + 25 + 5 * 17.5 + 3
    svg.line(bx0, ay, bx0 + bw, ay, stroke=MID, sw=0.8)
    for e in range(8, 13):
        xx = bx0 + bw * (e - lo) / (hi - lo)
        svg.line(xx, ay, xx, ay + 4, stroke=MID, sw=0.8)
        svg.text(xx, ay + 14, f"10^{{{e}}}", size=9.5, fill=GREY, anchor="middle")
    return max(bottom + 5, ay + 14) + 14


# ---- 3. above the floor: tiles, IDs, memo; the hit path ---------------------------------------------------------------

def panel_tiles(svg, y, D, net, cells32):
    sw, pal = D["sweep"], D["palette"]
    T = D["table"]
    mg, s = sw["margin"], sw["tile"]
    nx, ny = sw["nx"], sw["ny"]
    g0 = D["patterns"]["squares"]["gen0"]
    pw, ph = g0["w"], g0["h"]
    rows = grid_rows(sw["cells"])
    x0 = M
    svg.rect(x0, y, pw, ph, fill="#000000")
    svg.image(x0, y, pw, ph, png_indexed([r[mg:mg + pw] for r in rows[mg:mg + ph]], pal, 2))
    # the lattice of phase A
    d = "".join(f"M{x0 + k * s},{y}v{ph}" for k in range(1, pw // s + 1)) + "".join(f"M{x0},{y + k * s}h{pw}" for k in range(1, ph // s + 1))
    svg.path(d, stroke="#ffffff", sw=0.5, opacity=0.16)
    # results that were not in the table: tiles of phase B, half a tile off the lattice
    missed = [((i - 1) * s + s // 2 - mg, (j - 1) * s + s // 2 - mg) for j in range(ny + 1) for i in range(nx + 1) if sw["miss"][j * (nx + 1) + i]]
    for mx, my in missed:
        xa, ya, xb, yb = max(mx, 0), max(my, 0), min(mx + s, pw), min(my + s, ph)
        if xb > xa and yb > ya:
            svg.rect(x0 + xa + 0.6, y + ya + 0.6, xb - xa - 1.2, yb - ya - 1.2, fill=VERMIL, opacity=0.32)
            svg.rect(x0 + xa + 0.6, y + ya + 0.6, xb - xa - 1.2, yb - ya - 1.2, stroke="#ff9d5c", sw=1.1)
    # the 32 cells that panel 2 sends through the circuit
    c32x, c32y = cells32[0] - mg, cells32[1] - mg
    if 0 <= c32x <= pw - 32 and 1 <= c32y < ph - 1:
        svg.rect(x0 + c32x - 1.5, y + c32y - 2.5, 35, 6, stroke="#ffffff", sw=1.2)
        left = c32x > pw / 2
        svg.text(x0 + c32x - 6 if left else x0 + c32x + 38, y + c32y + 4, "the 32 cells of panel 2", size=10.5, anchor="end" if left else "start", fill="#ffffff", halo="#000000")

    # the window that the panels on the right take apart: 4 x 4 tiles with many different tiles
    # around a result that was missing
    a, empty = sw["a_ids"], sw["empty"]
    miss_at = set(missed)
    best, zoom = None, None
    for ty in range(0, ph // s - 4):
        for tx in range(0, pw // s - 4):
            ids = [a[(ty + 2 + j) * nx + tx + 2 + i] for j in range(4) for i in range(4)]
            centre_missed = ((tx + 1) * s + s // 2, (ty + 1) * s + s // 2) in miss_at
            score = (len(set(ids)) - 3 * ids.count(empty), centre_missed, -abs(tx - 14) - abs(ty - 14))
            if best is None or score > best:
                best, zoom = score, (tx, ty)
    tx, ty = zoom
    rx = x0 + pw + 26
    zs, bs = 2.25, 36.0
    zy = y + 18
    zw = 4 * s * zs
    sqx, sqy, sqw = x0 + tx * s, y + ty * s, 4 * s
    svg.cone((sqx + sqw, sqy), (sqx + sqw, sqy + sqw), (rx, zy), (rx, zy + zw), x0 + pw, 0)
    svg.rect(sqx, sqy, sqw, sqw, stroke="#ffffff", sw=1.8)
    n_miss = sum(sw["miss"])
    cap = y + ph + 17
    svg.rect(x0, cap - 9, 10, 10, fill=VERMIL, opacity=0.55)
    svg.rect(x0 + 0.5, cap - 8.5, 9, 9, stroke=VERMIL, sw=1)
    cap = svg.para(x0 + 16, cap, f"squares at generation {thousands(sw['generation'])}, cut into {s}×{s}-cell tiles. One sweep advances {s // 2} generations with one probe per tile: "
                   f"{thousands(sw['windows'])} probes, {thousands(sw['unique'])} distinct keys, {n_miss} results not in the table ({pct(n_miss / sw['windows'])}, marked). "
                   "Novelty sits where the circuit is computing; everything else is a hit. The white frame is taken apart on the right.", pw - 16, size=11.5, fill=INK)

    # ---- cells become tile IDs ----
    svg.text(rx, y + 10, "Cells become tile IDs", size=12.5, weight="bold")
    zr = [r[mg + tx * s:mg + (tx + 4) * s] for r in rows[mg + ty * s:mg + (ty + 4) * s]]
    svg.rect(rx, zy, zw, zw, fill="#000000")
    svg.image(rx, zy, zw, zw, png_indexed(zr, pal, 5))
    svg.path("".join(f"M{rx + k * s * zs},{zy}v{zw}M{rx},{zy + k * s * zs}h{zw}" for k in range(1, 4)), stroke="#ffffff", sw=0.7, opacity=0.5)
    svg.rect(rx, zy, zw, zw, stroke="#ffffff", sw=1.8)
    svg.rect(rx - 0.9, zy - 0.9, zw + 1.8, zw + 1.8, stroke=MID, sw=0.8)
    svg.arrow(rx + zw + 6, zy + zw / 2, rx + zw + 28, zy + zw / 2)
    ix = rx + zw + 34
    ids = [[a[(ty + 2 + j) * nx + tx + 2 + i] for i in range(4)] for j in range(4)]
    flat = [q for r_ in ids for q in r_]
    seen = {}
    tints = ["#dbe9f6", "#fdebc8", "#d5efe6", "#f9dccb", "#efd9e8", "#e4e4e4"]
    for j in range(4):
        for i in range(4):
            v = ids[j][i]
            fill = "#f3f3f3" if v == empty else tints[seen.setdefault(v, len(seen)) % len(tints)] if flat.count(v) > 1 else "#ffffff"
            svg.rect(ix + i * bs, zy + j * bs, bs, bs, fill=fill, stroke=MID, sw=0.8)
            svg.text(ix + i * bs + bs / 2, zy + j * bs + bs / 2 + 3.4, "empty" if v == empty else str(v), size=9.5, anchor="middle", fill=GREY if v == empty else INK)
    nx_ = ix + 4 * bs + 14
    yy = svg.para(nx_, zy + 10, "A tile's ID is its slot in the tile store, so equal tiles have equal IDs and a universe is a grid of IDs: 4 bytes per tile.", W - M - nx_)
    svg.para(nx_, yy + 7, "Above the tiles every key is four IDs. Only IDs are hashed, whatever the rule.", W - M - nx_)

    # ---- one memo entry: the centre 2 x 2 tiles of the window and what they become ----
    my = zy + zw + 44
    gi, gj = tx + 2 + 2, ty + 2 + 2          # B(i, j) is the result of A(i-1..i, j-1..j)
    key = [a[(gj - 1) * nx + gi - 1], a[(gj - 1) * nx + gi], a[gj * nx + gi - 1], a[gj * nx + gi]]
    value = sw["b_ids"][gj * (nx + 1) + gi]
    was_miss = sw["miss"][gj * (nx + 1) + gi]
    win = [bytearray(2 * s) for _ in range(2 * s)]
    for q, kid in enumerate(key):
        t = tile_rows(sw["tiles"][str(kid)], s)
        for r_ in range(s):
            win[(q >> 1) * s + r_][(q & 1) * s:(q & 1) * s + s] = t[r_]
    res = tile_rows(sw["tiles"][str(value)], s)
    # the entry against the circuit: the window stepped s/2 generations has the value in its centre
    w_ = [bytes(r_) for r_ in win]
    for _ in range(s // 2):
        w_ = ghl_netlist.step_grid(net, w_, 2 * s)
    assert [r_[s // 2:s // 2 + s] for r_ in w_[s // 2:s // 2 + s]] == res, "memo entry differs from the stepped window"
    ms = 3.0
    wside = 2 * s * ms
    # the four centre tiles of the enlarged window, as cells: a cone down to the entry's window
    svg.cone((rx + s * zs, zy + 3 * s * zs), (rx + 3 * s * zs, zy + 3 * s * zs), (rx, my), (rx + wside, my), zy + zw, 1)
    svg.rect(rx + s * zs, zy + s * zs, 2 * s * zs, 2 * s * zs, stroke="#ffffff", sw=1.4, dash="4 2")
    svg.text(rx + wside + 22, my - 13, "A memo entry: four IDs give one ID", size=12.5, weight="bold")
    svg.rect(rx, my, wside, wside, fill="#000000")
    svg.image(rx, my, wside, wside, png_indexed([bytes(r_) for r_ in win], pal, 6))
    svg.path(f"M{rx + s * ms},{my}v{wside}M{rx},{my + s * ms}h{wside}", stroke="#ffffff", sw=0.7, opacity=0.5)
    svg.rect(rx - 0.7, my - 0.7, wside + 1.4, wside + 1.4, stroke=MID, sw=0.8)
    svg.rect(rx + s * ms / 2, my + s * ms / 2, s * ms, s * ms, stroke="#ffffff", sw=1.2, dash="3 2")
    ax = rx + wside + 8
    svg.arrow(ax, my + s * ms, ax + 46, my + s * ms)
    svg.text(ax + 22, my + s * ms - 6, f"{s // 2} gens", size=10.5, anchor="middle", fill=GREY)
    ox = ax + 54
    svg.rect(ox, my + s * ms / 2, s * ms, s * ms, fill="#000000")
    svg.image(ox, my + s * ms / 2, s * ms, s * ms, png_indexed(res, pal, 6))
    svg.rect(ox, my + s * ms / 2, s * ms, s * ms, stroke="#ffffff", sw=1.2, dash="3 2")
    svg.rect(ox - 0.7, my + s * ms / 2 - 0.7, s * ms + 1.4, s * ms + 1.4, stroke=MID, sw=0.8)
    svg.text(rx + wside / 2, my + wside + 13, "as cells", size=10.5, anchor="middle", fill=GREY)
    # the same as IDs: the four IDs framed in the grid above are the key
    svg.rect(ix + bs, zy + bs, 2 * bs, 2 * bs, stroke=BLUE, sw=1.8)
    bw_, bh_ = 52.0, 19.0
    ex = ox + s * ms + 44
    ky = my + s * ms - bh_
    svg.text(ex - 6, ky - 6, "the key: the four IDs framed above", size=10.5, fill=BLUE)
    for q, kid in enumerate(key):
        svg.rect(ex + (q & 1) * bw_, ky + (q >> 1) * bh_, bw_, bh_, fill="#ffffff", stroke=MID, sw=0.8)
        svg.text(ex + (q & 1) * bw_ + bw_ / 2, ky + (q >> 1) * bh_ + 13.2, str(kid), size=10, anchor="middle")
    svg.rect(ex, ky, 2 * bw_, 2 * bh_, stroke=BLUE, sw=1.6)
    vx = ex + 2 * bw_ + 30
    svg.arrow(ex + 2 * bw_ + 5, my + s * ms, vx - 5, my + s * ms, sw=1.2)
    svg.rect(vx, my + s * ms - bh_ / 2, bw_, bh_, fill="#dbe9f6", stroke=BLUE, sw=1)
    svg.text(vx + bw_ / 2, my + s * ms - bh_ / 2 + 13.2, str(value), size=10, anchor="middle")
    svg.text(vx + bw_ / 2, my + s * ms + bh_ / 2 + 13, "value", size=10.5, anchor="middle", fill=GREY)
    svg.text(ex + bw_, my + wside + 13, "as IDs: 32 bytes", size=10.5, anchor="middle", fill=GREY)
    ey = svg.para(rx, my + wside + 31, f"A {2 * s}×{2 * s} window and the {s}×{s} tile in its centre {s // 2} generations later. This entry was "
                  + ("missing" if was_miss else "a hit") + f" in this sweep. Checked while drawing: the circuit of panel 2, applied {s // 2} times, gives the same tile.", W - M - rx)

    # ---- the table in memory ----
    ty_ = ey + 22
    svg.text(rx, ty_ - 10, "The memo table, slot by slot", size=12.5, weight="bold")
    cols, rws, px = 128, 30, 2
    tpal = [(238, 238, 238), (0, 114, 178), (86, 180, 233), (230, 159, 0), (213, 94, 0)]
    m_ = T["map"]
    trows = [bytes(0 if ch == "." else min(B32[ord(ch)], 3) + 1 for ch in m_[r_ * cols:(r_ + 1) * cols]) for r_ in range(rws)]
    svg.image(rx, ty_, cols * px, rws * px, png_indexed(trows, tpal, 4))
    hist, ent = T["displacement"], T["entries"]
    lx = rx + cols * px + 14
    labels = [("empty", tpal[0], 1 - ent / T["slots"]), ("in its home slot", tpal[1], hist[0] / T["slots"]), ("1 slot further", tpal[2], hist[1] / T["slots"]),
              ("2 further", tpal[3], hist[2] / T["slots"]), ("3 or more", tpal[4], sum(hist[3:]) / T["slots"])]
    for k, (lab, col, share) in enumerate(labels):
        svg.rect(lx, ty_ + k * 12.6, 8, 8, fill="#%02x%02x%02x" % col)
        svg.text(lx + 12, ty_ + k * 12.6 + 8, f"{lab}: {pct(share, 0)}", size=10.5, fill=GREY)
    mib = T["slots"] * T["entry_bytes"] / 2 ** 20
    by = svg.para(rx, ty_ + rws * px + 15,
                  f"The first {thousands(cols * rws)} of {thousands(T['slots'])} slots ({mib:.0f} MiB) after {thousands(T['generations'])} generations: open addressing, linear probing, "
                  f"{T['avg_probes']:.2f} probes per insert. {sci(T['windows'], 3)} probes found {thousands(ent)} distinct windows; the floor windows under them took "
                  f"{pct(T['brute_cell_evals'] / T['delivered'])} of the cell-updates delivered. The rest were hits.", W - M - rx)
    return max(cap, by) + 2


# ---- 4. the miss path; choosing the engine -------------------------------------------------------------------------------

def miss_colour(n, top=5.0):
    """Light to dark vermillion by the logarithm of the batch size."""
    t = min(1.0, math.log10(n + 1) / top)
    a, b = (253, 236, 224), (158, 58, 0)
    return "#%02x%02x%02x" % tuple(int(round(a[k] + (b[k] - a[k]) * t)) for k in range(3)), t


def icicle(svg, x, y, w, run, step, row_h=15.0, label_w=66):
    """The waves of one step: a box per wave, its row the window size, its width the generations
    it stands for; the dark part of a box is the share of its windows that missed."""
    waves, floor, k = step["waves"], run["a"] + 1, run["k"]
    kids = {}
    for i, wv in enumerate(waves):
        kids.setdefault(wv[3], []).append(i)
    top_full = k + 2                       # above this level a window is resolved by one wave below it

    def units(i):
        ch = kids.get(i, [])
        return sum(units(c) for c in ch) if ch else 2 ** (min(waves[i][0], top_full) - floor)

    total = units(0)
    shown = [lv for lv in sorted({wv[0] for wv in waves}, reverse=True) if lv <= top_full]
    row = {lv: r + 1 for r, lv in enumerate(shown)}

    def draw(i, ux):
        lv, req, miss, _ = waves[i]
        u = units(i)
        if lv in row:
            bx, bw, by = x + label_w + w * ux / total, w * u / total, y + row[lv] * row_h
            col, t = miss_colour(miss)
            svg.rect(bx + 0.5, by + 0.5, bw - 1, row_h - 1, fill="#dcecf7")
            if miss:
                svg.rect(bx + 0.5, by + 0.5, max(1.5, (bw - 1) * miss / req), row_h - 1, fill=col)
            lab = compact(miss, bw - 1)
            if lab:
                svg.text(bx + bw / 2, by + row_h - 4.2, lab, size=9.5, anchor="middle", fill="#ffffff" if t > 0.55 and miss / req > 0.5 else INK)
        c0 = ux
        for c in kids.get(i, []):
            draw(c, c0)
            c0 += units(c)

    draw(0, 0)
    above = [wv for wv in waves if wv[0] > top_full]     # single waves, one per level
    svg.rect(x + label_w + 0.5, y + 0.5, w - 1, row_h - 1, fill="#f1f1f1")
    svg.text(x + label_w + w / 2, y + row_h - 4.2, f"{len(above)} waves, one per level: {' → '.join(thousands(wv[2]) for wv in above)} windows", size=9.5, anchor="middle", fill=GREY)
    svg.text(x + label_w - 6, y + row_h - 4.2, f"≥ {2 ** (top_full + 1)}", size=9.5, anchor="end", fill=GREY)
    for lv in shown:
        fl = lv == floor
        svg.text(x + label_w - 6, y + row[lv] * row_h + row_h - 4.2, f"{2 ** lv}×{2 ** lv}" + (" floor" if fl else ""), size=9.5, anchor="end",
                 fill=INK if fl else GREY, weight="bold" if fl else None)
    h = (len(shown) + 1) * row_h
    svg.line(x + label_w, y + h + 3, x + label_w + w, y + h + 3, stroke=MID, sw=0.8)
    for q in range(0, 5):
        xx = x + label_w + w * q / 4
        svg.line(xx, y + h + 3, xx, y + h + 7, stroke=MID, sw=0.8)
        svg.text(xx, y + h + 17, str(run["step_gens"] * q // 4), size=9.5, anchor="middle", fill=GREY)
    svg.text(x + label_w + w + 8, y + h + 17, "gens", size=9.5, fill=GREY)
    return h + 20, sum(wv[2] for wv in waves if wv[0] == floor), len(waves)


def panel_miss(svg, y, D, logs):
    pal = D["palette"]
    Wv = D["waves"]
    # ---- the waves ----
    x0, iw, lw = M, 366, 66
    col_w = lw + iw + 26
    svg.text(x0, y + 10, "One wave per level, each a batch for one dispatch", size=12.5, weight="bold")
    yy = svg.para(x0, y + 26, "A missed window asks for nine windows half its size, then for four more: two waves a level down. Each box is a wave; its number is the windows that missed, "
                  "its dark part their share of the distinct requests.", col_w)
    sq = Wv["squares"]
    step = sq["steps_logged"][-1]
    yy += 6
    svg.text(x0, yy, f"squares, the {sq['step_gens']} generations from {thousands(step['generation'])}: mostly known", size=11.5)
    h1, f1, n1 = icicle(svg, x0, yy + 6, iw, sq, step, label_w=lw)
    yy += h1 + 28
    rom = Wv["rom"]
    rstep = rom["steps_logged"][0]
    svg.text(x0, yy, f"a ROM, its first {rom['step_gens']} generations: all new", size=11.5)
    h2, f2, n2 = icicle(svg, x0, yy + 6, iw, rom, rstep, label_w=lw)
    yy += h2 + 20
    bottom1 = svg.para(x0, yy, f"{n1} and {n2} waves brute-force {thousands(f1)} and {thousands(f2)} floor windows. A wave is 16 dispatches of fixed capacity: "
                       f"{logs['wave_us']:.0f} µs measured for an empty one.", col_w)

    # ---- the atlas ----
    ax = x0 + col_w + 22
    at = step["atlas"]
    side, gens = at["side"], at["generations"]
    half = side // 2
    cols, rws, px = 7, 5, 2
    wins = at["windows"][:cols * rws]
    white = len(pal)
    palw = [tuple(c) for c in pal] + [(255, 255, 255)]
    mos = mosaic([tile_rows(wv["window"], side) for wv in wins], side, cols, white)
    mw, mh = len(mos[0]) * px, len(mos) * px
    res = mosaic([tile_rows(wv["result"], half) for wv in wins], half, cols, white)
    rw_, rh_ = len(res[0]) * px, len(res) * px
    atlas_w = mw + 40 + rw_
    svg.text(ax, y + 10, "At the floor: an atlas of missed windows", size=12.5, weight="bold")
    ay = svg.para(ax, y + 26, f"{cols * rws} of the {at['misses']} windows of one floor wave of squares.", atlas_w) + 2
    svg.image(ax, ay, mw, mh, png_indexed(mos, palw, 2 * px))
    cellp = (side + 1) * px
    q = side // 4 * px
    svg.path("".join(f"M{ax + (k % cols) * cellp + q},{ay + (k // cols) * cellp + q}h{half * px}v{half * px}h-{half * px}z" for k in range(len(wins))), stroke="#ffffff", sw=1.0, opacity=0.95)
    rx = ax + mw + 40
    svg.arrow(ax + mw + 6, ay + mh / 2, rx - 6, ay + mh / 2)
    svg.text(ax + mw + 20, ay + mh / 2 - 7, str(gens), size=10.5, anchor="middle", fill=GREY)
    svg.text(ax + mw + 20, ay + mh / 2 + 15, "gens", size=10.5, anchor="middle", fill=GREY)
    svg.image(rx, ay + (mh - rh_) / 2, rw_, rh_, png_indexed(res, palw, 2 * px))
    bottom2 = svg.para(ax, ay + mh + 17, f"Each missed {side}×{side} window is copied from its four {half}×{half} base tiles into a grid of bit-planes, the circuit of panel 2 steps the grid {gens} generations, "
                       "and the centre of every window (outlined) is interned as a base tile. The stages of a wave (claim, expand, evaluate, intern, publish) are each one dispatch over the "
                       "whole batch. Above the floor a miss costs probes and inserts only.", atlas_w)

    # depth per universe: the dispatches of a generation are paid once, whatever the universe's size
    dy = bottom2 + 13
    svg.text(ax, dy, "Per universe, dispatch latency picks the depth", size=12.5, weight="bold")
    dy = svg.para(ax, dy + 15, f"A generation costs its dispatches (from {logs['dispatch_us']:.1f} µs each) however small the universe, so few active cells cannot pay for a deep "
                  "recursion. The model's thresholds for the 32-state rule:", atlas_w)
    P = D["patterns"]
    marks = [("chaotic", P["chaotic"]["gen0"]["w"] * P["chaotic"]["gen0"]["h"]), ("counter", P["counter"]["gen0"]["w"] * P["counter"]["gen0"]["h"]),
             ("squares", P["squares"]["gen0"]["w"] * P["squares"]["gen0"]["h"]), ("ROM", P["rom"]["w"] * P["rom"]["h"])]
    dlo, dhi = 4.0, 8.0
    bar_y = dy + 16

    def DX(v):
        return ax + atlas_w * (math.log10(v) - dlo) / (dhi - dlo)

    bands = [(10 ** dlo, DEPTH_A1, "#e3e3e3", INK, "no memo: plain cells"), (DEPTH_A1, DEPTH_A2, SKY, INK, "floor 32×32"), (DEPTH_A2, 10 ** dhi, BLUE, "#ffffff", "floor 16×16")]
    for a_, b_, col, ink, lab in bands:
        svg.rect(DX(a_), bar_y, DX(b_) - DX(a_), 15, fill=col)
        svg.text((DX(a_) + DX(b_)) / 2, bar_y + 11, lab, size=10.5, anchor="middle", fill=ink)
    log_grid(svg, DX, dlo, dhi, bar_y + 15, bar_y + 26, major=MID, minor="#b9b9b9", major_sw=0.8)
    for e in range(4, 9):
        svg.text(DX(10 ** e), bar_y + 37, f"10^{{{e}}}", size=10, anchor="middle" if 4 < e < 8 else "start" if e == 4 else "end", fill=GREY)
    svg.text(DX(10 ** 6.5), bar_y + 37, "active cells", size=10, anchor="middle", fill=GREY)
    for k, (name, cells) in enumerate(marks):
        xx = DX(cells)
        svg.path(f"M{xx - 4:.1f},{bar_y - 7}L{xx + 4:.1f},{bar_y - 7}L{xx:.1f},{bar_y - 0.5}Z", stroke=None, fill=INK)
        # two universes of nearly the same size: their names go to either side
        before = k + 1 < len(marks) and DX(marks[k + 1][1]) - xx < 50
        after = k > 0 and xx - DX(marks[k - 1][1]) < 50
        svg.text(xx + 5 if before else xx - 5 if after else xx, bar_y - 10, name, size=10.5, anchor="end" if before else "start" if after else "middle")
    bottom2 = bar_y + 40

    # ---- the controller ----
    C = D["controller"]
    cx = ax + atlas_w + 26
    cw = W - M - cx
    th = C["thumb"]
    svg.text(cx, y + 10, "Engine per region, by novelty", size=12.5, weight="bold")
    cy = svg.para(cx, y + 26, f"A universe of 16 patterns, {thousands(C['w'])} × {thousands(C['h'])} cells.", cw) + 2
    ch = cw * C["h"] / C["w"]
    svg.rect(cx, cy, cw, ch, fill="#000000")
    lut = bytes(int(round(255 * (v / 255) ** 0.6)) for v in range(256))
    svg.image(cx, cy, cw, ch, png_rgb(th["w"], th["h"], bytes.fromhex(th["rgb"]).translate(lut)))
    reg = C["region"]
    sc = cw / C["w"]
    nrx, nry = math.ceil(C["w"] / reg), math.ceil(C["h"] / reg)
    cells_n = memo_n = 0
    for r_ in C["regions"]:
        if not (0 <= r_["x"] < nrx and 0 <= r_["y"] < nry) or r_["hash_ns"] + r_["cells_ns"] < 500:      # outside, or a region that only ever hit
            continue
        steps = dict((a_, b_) for a_, b_ in r_["steps"])
        is_cells = steps.get(0, 0) * 2 > sum(steps.values())
        cells_n += is_cells
        memo_n += not is_cells
        xa, ya = cx + r_["x"] * reg * sc, cy + r_["y"] * reg * sc
        wd, hd = min(reg, C["w"] - r_["x"] * reg) * sc, min(reg, C["h"] - r_["y"] * reg) * sc
        col = ORANGE if is_cells else SKY
        svg.rect(xa + 0.6, ya + 0.6, wd - 1.2, hd - 1.2, fill=col, opacity=0.5 if is_cells else 0.3)
        svg.rect(xa + 0.6, ya + 0.6, wd - 1.2, hd - 1.2, stroke=col, sw=0.9)
    ly = cy + ch + 16
    svg.rect(cx, ly - 9, 10, 10, fill=SKY)
    svg.text(cx + 15, ly, f"memo with the 16×16 floor: {memo_n} regions")
    svg.rect(cx, ly + 6, 10, 10, fill=ORANGE)
    svg.text(cx + 15, ly + 15, f"plain cells, memo only above: {cells_n} regions")
    bottom3 = svg.para(cx, ly + 33, f"{reg}×{reg}-cell regions, by the mode they spent most of the first {thousands(C['gens'])} generations in. Content that is new at every scale "
                       f"(the ROM while it is being learned, part of a movie) is stepped as cells; the rest keeps its memo. Model time {C['adaptive_ms']:.1f} ms; "
                       f"plain cells everywhere, {C['plain_ms']:.1f} ms.", cw)
    return max(bottom1, bottom2, bottom3) + 2


# ---- 5. why the floor is where it is; against Golly -----------------------------------------------------------------------

def panel_why(svg, y, D, logs):
    rp = logs["rp"]
    rb = [("32 states, table lookup", logs["rb_lut"], SKY, 3), ("32 states, the circuit", logs["rb_planes"], BLUE, 2), ("Life, the adder tree", logs["rb_life"], GREEN, 3)]
    x0, w0 = M, 452
    svg.text(x0, y + 10, "Measured rates put the memo floor", size=12.5, weight="bold")
    svg.para(x0, y + 26, "A probe costs as much as ρ = R_{B} / R_{P} cell-updates of brute force. The probe rate is the same for every rule; the brute-force rate is not. Rates per second:", w0)
    lo, hi = 8.7, 12.5
    ax0, aw = x0 + 4, w0 - 12

    def X(v):
        return ax0 + aw * (math.log10(v) - lo) / (hi - lo)

    ry = y + 120
    log_grid(svg, X, lo, hi, ry - 58, ry, major="#dcdcdc", minor="#d0d0d0")
    svg.line(ax0, ry, ax0 + aw, ry, stroke=INK, sw=1)
    for e in range(8, 13):
        for k in range(1, 10):
            v = k * 10.0 ** e
            if 10 ** lo <= v <= 10 ** hi:
                svg.line(X(v), ry, X(v), ry + (5 if k == 1 else 2.5), stroke=INK, sw=1 if k == 1 else 0.7)
    for e in range(9, 13):
        svg.text(X(10 ** e), ry + 17, f"10^{{{e}}}", size=10, anchor="middle", fill=GREY)
    ri = logs["ri"]
    svg.rect(X(ri[0]), ry - 5, X(ri[1]) - X(ri[0]), 10, fill=PURPLE)
    svg.text(X(ri[1]) + 2, ry - 25, "insert R_{I}", size=10.5, anchor="end", fill=PURPLE, halo="#ffffff")
    svg.text(X(ri[1]) + 2, ry - 12, f"{ri[0] / 1e9:.1f}–{ri[1] / 1e9:.1f}·10^{{9}}", size=10.5, anchor="end", fill=PURPLE, halo="#ffffff")
    svg.circle(X(rp), ry, 5, fill=VERMIL)
    svg.text(X(rp), ry + 33, "probe R_{P} = " + sci(rp), size=10.5, anchor="middle", fill=VERMIL)
    rho = []
    for k, (lab, v, col, sig) in enumerate(rb):
        svg.circle(X(v), ry, 5, fill=col)
        r_ = float(f"{v / rp:.2g}")
        rho.append(r_)
        by = ry - 14 - k * 19
        svg.path(f"M{X(rp):.1f},{ry - 8}V{by:.1f}H{X(v):.1f}V{ry - 8}", stroke=col, sw=1.3)
        svg.text((X(rp) + X(v)) / 2, by - 4.5, f"ρ = {r_:.0f}   {lab}", weight="bold", anchor="middle", fill=col, halo="#ffffff")
    svg.text(X(rb[1][1]), ry + 33, "brute force R_{B} = " + ", ".join(sci(v, sig) for _, v, _, sig in rb), size=10.5, anchor="middle", fill=GREY)

    # what a hit saves, by window: a w x w window gives its centre, (w/2) x (w/2) cells, w/4 generations later
    gy = ry + 82
    gh, gx0, gw = 104.0, x0 + 44, w0 - 196
    levels = [3, 4, 5, 6]                  # windows of side 2^l
    ylo, yhi = -1.25, 2.5

    def value(lv):
        return 2 ** (3 * lv - 4)           # w^3 / 16 cell-updates

    def GX(lv):
        return gx0 + gw * (lv - 3 + 0.35) / 3.7

    def GY(v):
        return gy + gh * (1 - (math.log10(v) - ylo) / (yhi - ylo))

    svg.text(x0, gy - 14, "What a hit saves, V / ρ: a w×w window gives its centre w/4 generations later, V = w^{3}/16", fill=GREY)
    log_grid(svg, GY, ylo, yhi, gx0, gx0 + gw, vertical=False, major="#dcdcdc", minor="#d0d0d0")
    for e in range(-1, 3):
        svg.text(gx0 - 6, GY(10 ** e) + 3.5, "0.1×" if e < 0 else f"{10 ** e}×", size=10, anchor="end", fill=GREY)
    for lv in levels:
        svg.text(GX(lv), gy + gh + 14, f"{2 ** lv}×{2 ** lv}", size=10, anchor="middle", fill=GREY)
    svg.text(gx0 - 6, gy + gh + 14, "window", size=10, anchor="end", fill=GREY)
    svg.line(gx0, GY(1), gx0 + gw, GY(1), stroke=INK, sw=1, dash="5 3")
    svg.text(gx0 + gw + 6, GY(1) + 3.5, "1×: a hit saves what a probe costs", size=10.5)
    floors = []
    for k, ((lab, v, col, _), r_) in enumerate(list(zip(rb, rho))[1:]):
        pts = [(GX(lv), GY(value(lv) / r_)) for lv in levels]
        svg.path("M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in pts), stroke=col, sw=1.6)
        fl = next(lv for lv in levels if value(lv) > r_)
        floors.append((fl, value(fl) / r_, value(fl - 1) / r_))
        for lv, (a, b) in zip(levels, pts):
            svg.circle(a, b, 5.2 if lv == fl else 3, fill=col if lv != fl else "#ffffff", stroke=col, sw=2 if lv == fl else 0)
        a, b = pts[levels.index(fl)]
        if k == 0:                         # the upper line's label above it, the lower line's below
            svg.text(a - 9, b - 8, f"{floors[-1][1]:.1f}×", weight="bold", anchor="end", fill=col, halo="#ffffff")
        else:
            svg.text(a + 9, b + 14, f"{floors[-1][1]:.1f}×", weight="bold", fill=col, halo="#ffffff")
        a, b = pts[levels.index(fl) - 1]
        svg.text(a + 8, b + 12, f"{floors[-1][2]:.2f}×", size=10, fill=col, halo="#ffffff")
        a, b = pts[-1]
        svg.text(a + 10, b + 4, lab, size=10.5, fill=col)
    (fh, gh_, bh_), (fl_, gl_, bl_) = floors
    bottom1 = svg.para(x0, gy + gh + 36, f"A window is worth a memo entry only if its result is worth more than the probe that finds it: from {2 ** fh}×{2 ** fh} windows on for the 32-state rule "
                       f"({gh_:.1f}×; {2 ** (fh - 1)}×{2 ** (fh - 1)} windows: {bh_:.2f}×) and from {2 ** fl_}×{2 ** fl_} for Life ({gl_:.1f}×; {2 ** (fl_ - 1)}×{2 ** (fl_ - 1)}: {bl_:.2f}×). "
                       "That is each rule's memo floor: a larger ρ puts it one level coarser. Golly memoizes down to 4×4 blocks, worth V = 4.", w0)

    # ---- against Golly ----
    rows, same = ghl_logs.outcomes(logs)
    assert same
    ox = x0 + w0 + 34
    ow = W - M - ox
    svg.text(ox, y + 10, "Against Golly 5.1 at its best step size: rate of the GHL component that ran, over Golly's", size=12.5, weight="bold")
    lab_w = 196
    px0, pw = ox + lab_w, ow - lab_w - 4
    lo, hi = -2.6, 6.75

    def PX(v):
        return px0 + pw * (math.log10(v) - lo) / (hi - lo)

    ty = y + 44
    rh = 24.0
    n = len(rows)
    one = PX(1)
    svg.rect(px0, ty - 4, one - px0, n * rh + 8, fill="#f3f3f3")
    svg.rect(one, ty - 4, px0 + pw - one, n * rh + 8, fill="#e9f2f9")
    log_grid(svg, PX, lo, hi, ty - 4, ty + n * rh + 4, minor="#c4ccd2")
    for e in range(-2, 7):
        xx = PX(10 ** e)
        svg.text(xx, ty + n * rh + 17, "1×" if e == 0 else f"10^{{{e}}}×" if e > 0 else ("0.1×" if e == -1 else "0.01×"), size=10, anchor="middle", fill=GREY)
    svg.line(one, ty - 4, one, ty + n * rh + 4, stroke=INK, sw=1.2)
    svg.text(one - 7, ty - 10, "Golly faster", size=10.5, anchor="end", fill=GREY)
    svg.text(one + 7, ty - 10, "GHL component faster", size=10.5, fill=BLUE)

    def shade_at(xx):
        return "#f3f3f3" if xx < one else "#e9f2f9"

    def mark(kind, xx, yy):
        if kind == "gpu":
            svg.circle(xx, yy, 5, fill=BLUE)
        elif kind == "cpu":
            svg.path(f"M{xx - 5.5:.1f},{yy:.1f}L{xx:.1f},{yy - 5.5:.1f}L{xx + 5.5:.1f},{yy:.1f}L{xx:.1f},{yy + 5.5:.1f}Z", stroke=INK, sw=1.4, fill="#ffffff")
        else:
            svg.circle(xx, yy, 4.3, fill="#ffffff", stroke=BLUE, sw=1.7)

    group, yy = None, ty
    for r_ in rows:
        cy = yy + rh / 2
        if r_["group"] != group:
            group = r_["group"]
            if yy > ty:
                svg.line(ox, yy, px0 + pw, yy, stroke=MID, sw=0.8)
            svg.text(ox, cy + 4, group, size=10.5, weight="bold", fill=GREY)
        svg.text(px0 - 9, cy - 0.5, r_["name"], size=11.5, anchor="end")
        svg.text(px0 - 9, cy + 10, r_["note"], size=9.5, anchor="end", fill=GREY)
        if "range" in r_:
            kind, a, b = r_["range"]
            svg.line(PX(a), cy, PX(b), cy, stroke=INK, sw=1.4)
            mark(kind, PX(a), cy)
            mark(kind, PX(b), cy)
            svg.text(PX(b) + 10, cy + 3.8, f"{a:.2f}–{b:.2f}×", size=10.5, halo=shade_at(PX(b) + 30))
        else:
            pts = r_["points"]
            xs = [PX(v) for _, v in pts]
            if len(pts) > 1:
                svg.line(min(xs), cy, max(xs), cy, stroke=MID, sw=1)
            for (kind, v), xx in zip(pts, xs):
                mark(kind.rstrip(">"), xx, cy)
            for pos, k in enumerate(sorted(range(len(pts)), key=lambda k: xs[k])):
                left = len(pts) > 1 and pos == 0
                kind, v = pts[k]
                if kind.endswith(">"):       # a lower bound: cut, not rounded, to two figures
                    unit = 10 ** (math.floor(math.log10(v)) - 1)
                    txt = "≥ " + times(math.floor(v / unit) * unit)
                else:
                    txt = times(v)
                svg.text(xs[k] - 10 if left else xs[k] + 10, cy + 3.8, txt, size=10.5, anchor="end" if left else "start",
                         halo=shade_at(xs[k] - 30 if left else xs[k] + 30))
        yy += rh
    ly = ty + n * rh + 38
    lx = ox
    for kind, lab in (("gpu", "GPU, measured and verified"), ("learn", "the same with the CPU's learning time counted"), ("cpu", "the recursion on one CPU thread")):
        mark(kind, lx + 6, ly - 4)
        svg.text(lx + 16, ly, lab, size=10.5, fill=GREY)
        lx += 16 + text_width(lab, 10.5) + 20
    bottom2 = svg.para(ox, ly + 17, "Golly leads where content recurs sparsely and runs are long; there GHL falls back to the recursion. Every GPU result was checked against the CPU, "
                       "and the CPU engine's final states equal Golly's.", ow, size=10.5)
    return max(bottom1, bottom2)


# ---- composition -----------------------------------------------------------------------------------------------

def build():
    D = json.load(open(DATA, encoding="utf-8"))
    logs = ghl_logs.collect()
    svg = Svg()
    svg.text(M, 34, "GPU HashLife (GHL)", size=26, weight="bold")
    tx = M + text_width("GPU HashLife (GHL)", 26, True) + 16
    svg.text(tx, 34, "HashLife above a memo floor, the rule as a bit-sliced circuit below it", size=14)
    svg.text(M, 53, "Every picture and number is drawn from the measurement harness. The components were measured separately: GHL has not yet run as one program.", fill=GREY)
    chips(svg, W - M, 18, [("measured on the GPU", "gpu"), ("CPU reference engine", "cpu"), ("model", "model")], right=True)
    y = flow(svg, 64)
    y = section(svg, y, 1, "Six universes, two rules", "real patterns; colour is the cell state")
    y = panel_universes(svg, y, D)
    y = section(svg, y + 10, 2, "Below the floor: the rule as a circuit", "Hutton32b, 32 states, as two-input gates on bit-planes", badge=[("built, measured on the GPU", "gpu")])
    y, net, cells32 = panel_circuit(svg, y, D, logs)
    y = panel_life(svg, y + 4, D, logs)
    y = section(svg, y + 10, 3, "Above the floor: tiles, IDs and a memo table", "the hit path: a gather of four IDs and one probe per tile",
                badge=[("hit path built on the GPU", "gpu"), ("picture: CPU reference engine", "cpu")])
    y = panel_tiles(svg, y, D, net, cells32)
    y = section(svg, y + 10, 4, "A miss: the recursion in waves, level by level", "and the engine per region",
                badge=[("CPU reference engine", "cpu"), ("controller: model", "model")])
    y = panel_miss(svg, y, D, logs)
    y = section(svg, y + 10, 5, "Why a floor, and what it buys", None, badge=[("rates and ratios: measured", "gpu"), ("floor: model", "model")])
    y = panel_why(svg, y, D, logs)
    return svg.finish(y + 6)


if __name__ == "__main__":
    out = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    print("wrote", os.path.abspath(OUT), f"({len(out) >> 10} KiB)")
