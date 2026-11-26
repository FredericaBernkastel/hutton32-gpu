"""The overview diagram of GPU HashLife (GHL) at the head of paper.md, as one SVG without plotting
libraries. Everything it shows is data:

  * out/overview.json     patterns as cells, tile IDs, a memo table, one sweep, the waves of the
                          recursion, the controller (written by `ca-research overview`, which runs
                          the CPU reference engine);
  * logs/<date>/         the measured rates and the comparison with Golly, each target from the
                          newest dated folder that holds it (tools/ghl_logs.py);
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
    """`n` in full if it fits `room` pixels at 9.5 px, else as 9.1k or 36k (k: thousand)."""
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
MONO = "'DejaVu Sans Mono', Menlo, Consolas, 'Liberation Mono', 'Courier New', monospace"
MONO_SCALE, MONO_ADV = 0.93, 602       # file names a little smaller; the advance of the widest of these fonts
for _t in _ADV.values():
    _t[" "] = _t[" "]
# markup of every text: ^{superscript} _{subscript} `file name` [[rule name]] **term**
_TOK = re.compile(r"([_^])\{([^}]*)\}|`([^`]*)`|\[\[([^\]]*)\]\]|\*\*([^*]*)\*\*")
_SPAN = re.compile(r"`[^`]*`|\[\[[^\]]*\]\]|\*\*[^*]*\*\*")


def rule_colour(name):
    """Two-state rules green, the 32-state rules blue, in the text as in the charts."""
    return GREEN if name.startswith(("Life", "B3")) else BLUE


def rule_name(rule):
    return "Life" if rule == "B3/S23" else rule


def fname(path):
    """A pattern as the paper names it: its file name, with the extension."""
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def _plain(s, bold):
    adv = _ADV[bool(bold)]
    return sum(adv.get(c, 556) for c in s)


def text_width(s, size, bold=False):
    total, pos = 0.0, 0
    for m in _TOK.finditer(s):
        total += _plain(s[pos:m.start()], bold)
        if m.group(1):
            total += 0.72 * _plain(m.group(2), bold)
        elif m.group(3) is not None:
            total += len(m.group(3)) * MONO_ADV * MONO_SCALE
        else:
            total += _plain(m.group(4) if m.group(4) is not None else m.group(5), True)
        pos = m.end()
    return (total + _plain(s[pos:], bold)) * size / 1000


def wrap(s, size, width, bold=False):
    """Lines no wider than `width`; a file name, a rule name or a term is never broken."""
    s = _SPAN.sub(lambda m: m.group(0).replace(" ", " "), s)
    lines, cur = [], ""
    for word in s.split(" "):
        trial = f"{cur} {word}" if cur else word
        if cur and text_width(trial, size, bold) > width:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    return lines + [cur] if cur else lines


def rich(s, size, neutral=True):
    """Escape `s` and apply the markup. File names are set in a monospaced font, rule names in bold
    italics and, in black or grey text (`neutral`), in their rule's colour, terms in bold."""
    out, pos, cur = [], 0, 0.0

    def emit(text, attrs=""):
        nonlocal cur
        if not text:
            return
        if cur:
            attrs += f' dy="{-cur:.1f}"'
            cur = 0.0
        out.append(f"<tspan{attrs}>{escape(text)}</tspan>" if attrs else escape(text))

    for m in _TOK.finditer(s):
        emit(s[pos:m.start()])
        if m.group(1):
            target = -0.36 * size if m.group(1) == "^" else 0.22 * size
            out.append(f'<tspan dy="{target - cur:.1f}" font-size="{round(size * 0.72, 1)}">{escape(m.group(2))}</tspan>')
            cur = target
        elif m.group(3) is not None:
            emit(m.group(3), f' font-family="{MONO}" font-size="{round(size * MONO_SCALE, 1)}"')
        elif m.group(4) is not None:
            emit(m.group(4), ' font-weight="bold" font-style="italic"' + (f' fill="{rule_colour(m.group(4))}"' if neutral else ""))
        else:
            emit(m.group(5), ' font-weight="bold"' + (f' fill="{INK}"' if neutral else ""))
        pos = m.end()
    emit(s[pos:])
    return "".join(out)


class Svg:
    def __init__(self):
        self.o = []

    def text(self, x, y, s, size=11, weight=None, anchor="start", fill=INK, halo=None):
        a = f' font-weight="{weight}"' if weight else ""
        a += f' text-anchor="{anchor}"' if anchor != "start" else ""
        a += f' stroke="{halo}" stroke-width="3.2" stroke-linejoin="round" paint-order="stroke"' if halo else ""
        self.o.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"{a}>{rich(s, size, fill in (INK, GREY))}</text>')

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

    def circle(self, x, y, r, fill=INK, stroke=None, sw=1, opacity=None, dash=None):
        a = f' stroke="{stroke}" stroke-width="{sw}"' if stroke else ""
        a += f' opacity="{opacity}"' if opacity is not None else ""
        a += f' stroke-dasharray="{dash}"' if dash else ""
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

def section(svg, y, n, title, sub=None, built=None):
    """A numbered heading across the page, with how far the panel's content is built; returns the y below it."""
    svg.line(M, y, W - M, y, stroke=LIGHT, sw=1)
    cy = y + 25
    svg.badge(M + 11, cy - 5.5, n)
    svg.text(M + 30, cy, title, size=17, weight="bold")
    if sub:
        svg.text(M + 30 + text_width(title, 17, True) + 12, cy, sub, size=12.5, fill=GREY)
    if built:
        status(svg, W - M, cy, built)
    return y + 42


def status_mark(svg, kind, x, y):
    """How far a part is built: on the GPU, a filled circle; in the CPU reference engine, a diamond;
    in the model only, a dashed circle. The comparison with Golly uses the same shapes."""
    if kind == "gpu":
        svg.circle(x, y, 4, fill=INK)
    elif kind == "cpu":
        svg.path(f"M{x - 4.6:.1f},{y:.1f}L{x:.1f},{y - 4.6:.1f}L{x + 4.6:.1f},{y:.1f}L{x:.1f},{y + 4.6:.1f}Z", stroke=INK, sw=1.3, fill="#ffffff")
    else:
        svg.circle(x, y, 3.7, fill="#ffffff", stroke=GREY, sw=1.2, dash="2 1.6")


def status(svg, x, y, items, size=11):
    """Status marks with their texts in a row that ends at x: [(kind, text)]."""
    widths = [13 + text_width(t, size) for _, t in items]
    x -= sum(widths) + 18 * (len(items) - 1)
    for (kind, t), w in zip(items, widths):
        status_mark(svg, kind, x + 4.6, y - size * 0.34)
        svg.text(x + 13, y, t, size=size, fill=GREY)
        x += w + 18


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

def pictogram(svg, kind, xr, yc):
    """A picture of a few pixels for a step of the algorithm, its right edge at xr, centred on yc."""
    light, mid = "#d2d2d2", "#8f8f8f"
    if kind == "tiles":            # a grid of tiles; the same shade is the same tile, hence the same ID
        for k, shade in enumerate((light, mid, light, mid, "#ededed", mid)):
            svg.rect(xr - 24 + (k % 3) * 8, yc - 8 + (k // 3) * 8, 7, 7, fill=shade)
    elif kind == "window":         # 2 x 2 tiles, their centre tile, and that tile later
        for k in range(4):
            svg.rect(xr - 40 + (k % 2) * 8, yc - 8 + (k // 2) * 8, 8, 8, fill="#ffffff", stroke=mid, sw=0.8)
        svg.rect(xr - 36, yc - 4, 8, 8, fill="#cfe5f5", stroke=INK, sw=1.2)
        svg.arrow(xr - 21, yc, xr - 11, yc, sw=1.0)
        svg.rect(xr - 9, yc - 4, 8, 8, fill="#cfe5f5", stroke=INK, sw=1.2)
    elif kind == "hit":            # found
        svg.circle(xr - 8, yc, 7.5, fill="#ffffff", stroke=BLUE, sw=1.3)
        svg.path(f"M{xr - 12:.1f},{yc + 0.2:.1f}l3,3.2l5.2,-6.4", stroke=BLUE, sw=1.6)
    elif kind == "floor":          # windows by size; below the dashed line nothing is memoized
        for x, side in ((34, 14), (17, 9)):
            svg.rect(xr - x, yc + 7 - side, side, side, fill="#ffffff", stroke=mid, sw=1)
        svg.line(xr - 6.5, yc - 9, xr - 6.5, yc + 9, stroke=INK, sw=1, dash="2 2")
        svg.rect(xr - 4.5, yc + 2.5, 4.5, 4.5, fill=light)
    elif kind == "waves":          # a window, the windows half its size, and theirs: one row per wave
        for row, n in enumerate((1, 2, 4)):
            w_ = (30 - (n - 1) * 1.5) / n
            for k in range(n):
                svg.rect(xr - 30 + k * (w_ + 1.5), yc - 8 + row * 6, w_, 4.5, fill=("#f3c3a5", "#e39a6d", VERMIL)[row])
    elif kind == "atlas":          # windows side by side, each with the centre that is kept
        for k in range(6):
            x, yy = xr - 26 + (k % 3) * 9, yc - 8.5 + (k // 3) * 9
            svg.rect(x, yy, 8, 8, fill="#3a3a3a")
            svg.rect(x + 2, yy + 2, 4, 4, fill="#ffffff")
    elif kind == "circuit":        # gates and wires
        pts = [(xr - 28, yc - 6, ORANGE), (xr - 28, yc + 6, BLUE), (xr - 16, yc - 1, BLUE), (xr - 16, yc + 7, ORANGE), (xr - 4, yc + 2, GREEN)]
        for a_, b_ in ((0, 2), (1, 2), (1, 3), (2, 4), (3, 4)):
            svg.line(pts[a_][0], pts[a_][1], pts[b_][0], pts[b_][1], stroke=mid, sw=0.9)
        for x, yy, col in pts:
            svg.circle(x, yy, 2.6, fill=col)


def flow(svg, y):
    """The algorithm as a chain of steps, each with the words the panels use, and the legend of the status marks."""
    gap = 22
    cw_ = (CW - 3 * gap) / 4
    xs = [M + k * (cw_ + gap) for k in range(4)]
    pad, size, lead = 10, 10.5, 13.2
    cards = {      # (row, column): (panel that shows it, title, text)
        (0, 0): (3, "The universe: a grid of tile IDs", "tiles",
                 "The cells are cut into square **tiles** of side s. Equal tiles get the same **ID**, so the GPU holds a grid of numbers, not cells."),
        (0, 1): (3, "Hit path: one probe per tile", "window",
                 "A **window** (2×2 tiles) fixes its centre tile s/2 generations ahead. The **memo** is a table of windows seen before; a **probe** looks one up."),
        (0, 2): (3, "A hit: nothing to compute", "hit",
                 "**Hit**: the window is in the memo. Its future centre tile is read off, no cell is touched, and the grid is s/2 generations on."),
        (1, 0): (5, "Measured rates set the floor", "floor",
                 "The **floor** is the smallest window that is memoized. Smaller ones cost more to look up than to compute."),
        (1, 1): (4, "Miss path: waves, one per level", "waves",
                 "**Miss**: not in the memo. The window is built from windows half its size, which may miss too. A **wave**: all misses of one size, as one batch."),
        (1, 2): (4, "At the floor: an atlas of windows", "atlas",
                 "Missed floor windows are not split. They are laid side by side, the atlas, and stepped cell by cell: **brute force**."),
        (1, 3): (2, "The rule as a circuit", "circuit",
                 "Brute force is a network of logic gates that updates 32 cells per operation. Its results go into the memo."),
    }
    lines = {k: wrap(v[3], size, cw_ - 2 * pad) for k, v in cards.items()}
    h = 30 + lead * max(len(v) for v in lines.values()) + 3
    ya, yb = y, y + h + 26
    for (row, col), (n, title, picture, _) in cards.items():
        x, yy = xs[col], (ya, yb)[row]
        config = (row, col) == (1, 0)          # not a step: what configures the steps
        svg.rect(x, yy, cw_, h, fill="#ffffff" if config else "#f3f3f3", stroke=GREY if config else None, sw=1, rx=6, dash="3 2" if config else None)
        svg.badge(x + pad + 8, yy + 15, n, r=8.5, size=11)
        svg.text(x + pad + 22, yy + 19.3, title, size=12.5, weight="bold")
        pictogram(svg, picture, x + cw_ - pad, yy + 15)
        for q, line in enumerate(lines[(row, col)]):
            svg.text(x + pad, yy + 38 + q * lead, line, size=size, fill=GREY)
    ma, mb = ya + h / 2, yb + h / 2
    for a_, b_, yy, dash in ((0, 1, ma, None), (1, 2, ma, None), (0, 1, mb, "4 3"), (1, 2, mb, None), (2, 3, mb, None)):
        svg.arrow(xs[a_] + cw_ + 4, yy, xs[b_] - 4, yy, dash=dash)
    svg.text(xs[1] + cw_ + gap / 2, ma - 7, "hit", size=10.5, anchor="middle", fill=GREY)
    mx = xs[1] + cw_ / 2
    svg.arrow(mx, ya + h + 4, mx, yb - 4)
    svg.text(mx + 7, (ya + h + yb) / 2 + 3.6, "miss", size=10.5, fill=GREY)
    # what brute force computed goes back into the memo
    by = yb + h
    svg.path(f"M{xs[3] + cw_ / 2:.1f},{by + 4:.1f}V{by + 14:.1f}H{mx:.1f}V{by + 5:.1f}", stroke=GREY, sw=1.3, marker="grey")
    svg.text((mx + xs[3] + cw_ / 2) / 2, by + 27, "the results are stored in the memo, and the probes that missed now hit", size=10.5, anchor="middle", fill=GREY)
    # the status marks of the panel headings
    lx = xs[3] + pad
    svg.text(lx, ya + 19.3, "How far each part is built", size=12.5, weight="bold")
    yy = ya + 38
    for kind, t in (("gpu", "built and measured on the GPU"), ("cpu", "the CPU reference engine: not yet on the GPU"), ("model", "model: operation counts at measured rates")):
        status_mark(svg, kind, lx + 4.6, yy - 3.6)
        svg.text(lx + 15, yy, t, size=size, fill=GREY)
        yy += lead
    return by + 40


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

    rom, cnt, cha, soup, ulc = P["rom"], P["counter"]["gen0"], P["chaotic"], P["soup"], P["ulc"]
    iw, ih = int(cw), int(bh)

    def size_rule(w, h, key):
        return f"{thousands(w)} × {thousands(h)} cells · [[{rule_name(P[key]['rule'])}]]"

    def title(key):
        return f"`{fname(P[key]['file'])}`"

    cards = [
        ("squares", title("squares"), [size_rule(sq0["w"], sq0["h"], "squares"), "computes and displays n^{2}", f"shown: generation {thousands(sq['generation'])}"],
         png_indexed(grid_rows(sq["cells"], mg, mg, sq0["w"], sq0["h"]), pal, 1), sq0["w"], sq0["h"]),
        ("counter", title("counter"), [size_rule(cnt["w"], cnt["h"], "counter"), "an 8-bit counter", "period 513,024 generations"], png_indexed(grid_rows(cnt), pal, 2), cnt["w"], cnt["h"]),
        ("chaotic", title("chaotic"), [size_rule(cha["gen0"]["w"], cha["gen0"]["h"], "chaotic"), "dense, chaotic", f"shown: generation {thousands(cha['later_gen'])}"],
         png_indexed(grid_rows(cha["later"]), pal, 3), cha["later"]["w"], cha["later"]["h"]),
        ("rom", title("rom"), [size_rule(rom["w"], rom["h"], "rom"), "a circulating ROM, period 528", "shown: a detail, cell = pixel"],
         png_indexed(grid_rows(rom["crop"], 0, 0, iw, ih), pal, 2), iw, ih),
        ("soup", "random soup", [size_rule(soup["side"], soup["side"], "soup"), f"dense, chaotic; made by `{soup['source']}`", f"shown: a detail, generation {soup['view_gen']}"],
         png_indexed(grid_rows(soup["later"], 0, 0, iw, ih), LIFE_PALETTE, 2), iw, ih),
        ("ulc", title("ulc"), [size_rule(ulc["gen0"]["w"], ulc["gen0"]["h"], "ulc"), "sparse; simulates one [[Life]] cell", "period 5,760 generations"], ulc_png(), 1, 1),
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
        assert text_width(name, 12, True) <= cw, name
        svg.text(x, y + bh + 17, name, size=12, weight="bold")
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
        if k == 0:
            svg.text(gx(x, 100), sy + sh + 11.5, "generations", size=9.5, fill=GREY, anchor="middle")
    L = learn
    y = svg.para(M, sy + sh + 29, "Curves: how much of each universe is new, as the share of probes that miss, against the generations run so far (log scale); tiles of 16×16 cells, "
                 f"for [[Life]] 32×32. The number is that share over the second half of the run. The ROM's curve is that of `{fname(L['rom']['file'])}`, a smaller ROM of the same kind "
                 f"({thousands(L['rom']['w'])} × {thousands(L['rom']['h'])} cells); the soup's, that of a {L['soup']['w']} × {L['soup']['h']} soup. "
                 "This novelty, not density, decides how a universe is best advanced: the ROM is new for one period and then entirely known; the soup stays new.", CW)
    return y - 4


# ---- 2. below the floor: the rule as a circuit ------------------------------------------------------------------

GROUPS = [("n", "north"), ("w", "west"), ("c", "centre"), ("e", "east"), ("s", "south")]
# Active cells from which a floor of 32x32 and of 16x16 windows pays for its dispatches: the model of
# paper.md, the table of dispatch time per engine ({T:bounds}), at the bit-plane rate (the one figure
# pair here that no log file holds).
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

    sqfile = fname(D["patterns"]["squares"]["file"])
    svg.text(M, y + 11, f"32 cells of `{sqfile}` (marked in panel 3) and their four neighbours.", size=11.5)
    svg.text(M, y + 24.5, "A state is 5 bits; bit k of 32 cells in a row is one 32-bit word: 25 words go in, 5 come out.", size=11.5, fill=GREY)
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
                  f"rule table, [[{D['patterns']['squares']['rule']}]].", 236)
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
    svg.text(x_in + L.dx, ay + 27, "Every gate works on a 32-bit word, so it serves 32 cells at once. Wire shade: on how many of the 32 cells the signal is 1 (light: none, dark: all). "
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
    svg.para(M, y + 11, f"A two-state rule such as [[Life]] needs one bit per cell. An adder tree counts the live neighbours of 32 cells in {len(life.gates)} operations; "
             "birth and survival are read off the four bits of the count.", 620, size=11.5, fill=INK)
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
    svg.para(bx, (top + bottom) / 2 + 36, f"32 cells of the random soup, generation {D['patterns']['soup']['view_gen']}", 32 * cell + 30, size=10.5)

    # brute force as measured, rule by rule: the table and the circuit on the CPU and on the GPU
    rx = bx + 32 * cell + 46
    groups = [("[[Hutton32]] and [[Hutton32b]], 32 states",
               [("CPU, one core, table lookup", logs["cpu_lut"], MID, 2), ("CPU, four cores, the circuit", logs["cpu4"], MID, 2),
                ("GPU, table lookup, real patterns", logs["rb_lut"], SKY, 3), ("GPU, the circuit, any content", logs["rb_planes"], BLUE, 2)]),
              ("[[Life]], 2 states",
               [("CPU, one thread, Golly's QuickLife", logs["ql_cells_s"], MID, 3), ("GPU, the adder tree", logs["rb_life"], GREEN, 3)])]
    lab_w = 186
    bx0, bw = rx + lab_w, W - M - rx - lab_w - 58
    lo, hi = 8.0, 12.5
    head, row = 16.0, 16.5
    svg.text(rx, y + 11, "Brute force as measured, in cell-updates per second", size=11.5)
    y0 = y + 20
    height = sum(head + row * len(rows_) for _, rows_ in groups) + 8 * (len(groups) - 1)
    log_grid(svg, lambda v: bx0 + bw * (math.log10(v) - lo) / (hi - lo), lo, hi, y0, y0 + height + 3, major="#dcdcdc", minor="#d0d0d0")
    yy = y0
    for gi, (gname, rows_) in enumerate(groups):
        if gi:
            svg.line(rx, yy + 1, W - M, yy + 1, stroke=MID, sw=0.8)       # one rule above the line, the other below
            yy += 8
        svg.rect(rx, yy + 1, text_width(gname, 11) + 8, head - 2, fill="#ffffff")
        svg.text(rx, yy + 12, gname, fill=GREY)
        yy += head
        for lab, v, col, sig in rows_:
            svg.text(rx + lab_w - 8, yy + 10.5, lab, anchor="end")
            wv = bw * (math.log10(v) - lo) / (hi - lo)
            svg.rect(bx0, yy + 2, wv, 11, fill=col)
            svg.text(bx0 + wv + 5, yy + 11, sci(v, sig), weight="bold" if col in (BLUE, GREEN) else None, halo="#ffffff")
            yy += row
    ay = yy + 3
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
    sqfile = fname(D["patterns"]["squares"]["file"])

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
    cap = svg.para(x0 + 16, cap, f"`{sqfile}` at generation {thousands(sw['generation'])}, cut into {s}×{s}-cell tiles. One pass over the grid, a sweep, advances it {s // 2} generations "
                   f"with one probe per tile: {thousands(sw['windows'])} probes for {thousands(sw['unique'])} distinct windows, {n_miss} misses ({pct(n_miss / sw['windows'])}, marked). "
                   "The misses sit where the circuit is computing; everything else is a hit. The white frame is taken apart on the right.", pw - 16, size=11.5, fill=INK)

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
    yy = svg.para(nx_, zy + 10, "A tile's ID is its place in the store of tiles, so equal tiles have equal IDs and a universe is a grid of IDs: 4 bytes per tile.", W - M - nx_)
    svg.para(nx_, yy + 7, "A window is then four IDs: the memo is looked up by IDs, never by cells, whatever the rule.", W - M - nx_)

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
    ey = svg.para(rx, my + wside + 31, f"A {2 * s}×{2 * s} window and the {s}×{s} tile in its centre {s // 2} generations later. In this sweep the entry was "
                  + ("not there yet: a miss" if was_miss else "there: a hit") + f". Checked while drawing: the circuit of panel 2, applied {s // 2} times, gives the same tile.", W - M - rx)

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
                  f"The first {thousands(cols * rws)} of {thousands(T['slots'])} slots ({mib:.0f} MiB) after {thousands(T['generations'])} generations of `{sqfile}`: a hash table "
                  f"with linear probing, {T['avg_probes']:.2f} probes per insert. {sci(T['windows'], 3)} probes met {thousands(ent)} distinct windows. The brute force under them "
                  f"amounts to {pct(T['brute_cell_evals'] / T['delivered'])} of the cell-updates delivered; the rest came out of the table.", W - M - rx)
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
    svg.text(x + label_w - 6, y + row_h - 4.2, "larger", size=9.5, anchor="end", fill=GREY)
    for lv in shown:
        fl = lv == floor
        svg.text(x + label_w - 6, y + row[lv] * row_h + row_h - 4.2, f"{2 ** lv}×{2 ** lv}" + (", floor" if fl else ""), size=9.5, anchor="end",
                 fill=INK if fl else GREY, weight="bold" if fl else None)
    h = (len(shown) + 1) * row_h
    svg.line(x + label_w, y + h + 3, x + label_w + w, y + h + 3, stroke=MID, sw=0.8)
    for q in range(0, 5):
        xx = x + label_w + w * q / 4
        svg.line(xx, y + h + 3, xx, y + h + 7, stroke=MID, sw=0.8)
        svg.text(xx, y + h + 17, str(run["step_gens"] * q // 4), size=9.5, anchor="middle", fill=GREY)
    svg.text(x + label_w + w + 8, y + h + 17, "gens", size=9.5, fill=GREY)
    svg.text(x + label_w - 6, y + h + 17, "window", size=9.5, anchor="end", fill=GREY)
    return h + 20, sum(wv[2] for wv in waves if wv[0] == floor), len(waves)


def panel_miss(svg, y, D, logs):
    pal = D["palette"]
    Wv = D["waves"]
    # ---- the waves ----
    x0, iw, lw = M, 366, 66
    col_w = lw + iw + 26
    svg.text(x0, y + 10, "One wave per level, each one batch for the GPU", size=12.5, weight="bold")
    yy = svg.para(x0, y + 26, "A missed window is built from nine windows half its size and then four more: two waves, one level down. Each box is a wave; its number is the windows that "
                  "missed in it, its dark part their share of the windows it asked for.", col_w)
    sq = Wv["squares"]
    step = sq["steps_logged"][-1]
    yy += 6
    svg.text(x0, yy, f"`{fname(sq['file'])}`, the {sq['step_gens']} generations from {thousands(step['generation'])}: mostly known", size=11.5)
    h1, f1, n1 = icicle(svg, x0, yy + 6, iw, sq, step, label_w=lw)
    yy += h1 + 28
    rom = Wv["rom"]
    rstep = rom["steps_logged"][0]
    svg.text(x0, yy, f"`{fname(rom['file'])}`, a ROM, its first {rom['step_gens']} generations: all new", size=11.5)
    h2, f2, n2 = icicle(svg, x0, yy + 6, iw, rom, rstep, label_w=lw)
    yy += h2 + 20
    bottom1 = svg.para(x0, yy, f"{n1} and {n2} waves end in brute force on {thousands(f1)} and {thousands(f2)} floor windows. A wave is 16 GPU calls (dispatches) of fixed size: "
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
    ay = svg.para(ax, y + 26, f"{cols * rws} of the {at['misses']} windows of one floor wave of `{fname(sq['file'])}`.", atlas_w) + 2
    svg.image(ax, ay, mw, mh, png_indexed(mos, palw, 2 * px))
    cellp = (side + 1) * px
    q = side // 4 * px
    svg.path("".join(f"M{ax + (k % cols) * cellp + q},{ay + (k // cols) * cellp + q}h{half * px}v{half * px}h-{half * px}z" for k in range(len(wins))), stroke="#ffffff", sw=1.0, opacity=0.95)
    rx = ax + mw + 40
    svg.arrow(ax + mw + 6, ay + mh / 2, rx - 6, ay + mh / 2)
    svg.text(ax + mw + 20, ay + mh / 2 - 7, str(gens), size=10.5, anchor="middle", fill=GREY)
    svg.text(ax + mw + 20, ay + mh / 2 + 15, "gens", size=10.5, anchor="middle", fill=GREY)
    svg.image(rx, ay + (mh - rh_) / 2, rw_, rh_, png_indexed(res, palw, 2 * px))
    bottom2 = svg.para(ax, ay + mh + 17, f"Each missed {side}×{side} window is copied from its four {half}×{half} tiles into one grid, the atlas. The circuit of panel 2 steps the whole grid "
                       f"{gens} generations, and the centre of every window (outlined) becomes a tile with an ID. Every stage of a wave is one GPU call over the whole batch. "
                       "Above the floor a miss costs probes and inserts only.", atlas_w)

    # depth per universe: the GPU calls of a generation are paid once, whatever the universe's size
    dy = bottom2 + 13
    svg.text(ax, dy, "Per universe: how deep the recursion may go", size=12.5, weight="bold")
    dy = svg.para(ax, dy + 15, f"A GPU call costs at least {logs['dispatch_us']:.1f} µs, however little it does, and every generation needs several. Few active cells cannot pay for them: "
                  "the smaller the universe, the coarser its floor. The model's thresholds for [[Hutton32b]]:", atlas_w)
    P = D["patterns"]
    marks = sorted((P[k]["gen0"]["w"] * P[k]["gen0"]["h"] if "gen0" in P[k] else P[k]["w"] * P[k]["h"], fname(P[k]["file"])) for k in ("chaotic", "counter", "squares", "rom"))
    dlo, dhi = 4.0, 8.0

    def DX(v):
        return ax + atlas_w * (math.log10(v) - dlo) / (dhi - dlo)

    # the names over their marks, in two rows where they would collide
    placed, ends = [], [ax - 99.0, ax - 99.0]
    for cells, name in marks:
        wd = text_width(f"`{name}`", 10.5)
        left = min(max(DX(cells) - wd / 2, ax), ax + atlas_w - wd)
        r_ = 0 if left >= ends[0] + 8 else 1
        ends[r_] = left + wd
        placed.append((cells, name, left, r_))
    for k, (cells, name, left, r_) in enumerate(placed):
        wd = text_width(f"`{name}`", 10.5)
        for cells2, _, _, r2 in placed:
            if r_ == 0 and r2 > 0 and left - 3 < DX(cells2) < left + wd + 3:
                placed[k] = (cells, name, DX(cells2) - 6 - wd, r_)
    rows_used = 1 + max(r_ for *_, r_ in placed)
    bar_y = dy + 4 + 13 * rows_used
    bands = [(10 ** dlo, DEPTH_A1, "#e3e3e3", INK, "no memo: brute force only"), (DEPTH_A1, DEPTH_A2, SKY, INK, "floor 32×32"), (DEPTH_A2, 10 ** dhi, BLUE, "#ffffff", "floor 16×16")]
    for a_, b_, col, ink, lab in bands:
        svg.rect(DX(a_), bar_y, DX(b_) - DX(a_), 15, fill=col)
        svg.text((DX(a_) + DX(b_)) / 2, bar_y + 11, lab, size=10.5, anchor="middle", fill=ink)
    log_grid(svg, DX, dlo, dhi, bar_y + 15, bar_y + 26, major=MID, minor="#b9b9b9", major_sw=0.8)
    for e in range(4, 9):
        svg.text(DX(10 ** e), bar_y + 37, f"10^{{{e}}}", size=10, anchor="middle" if 4 < e < 8 else "start" if e == 4 else "end", fill=GREY)
    svg.text(DX(10 ** 6.5), bar_y + 37, "active cells", size=10, anchor="middle", fill=GREY)
    for cells, name, left, r_ in placed:
        xx = DX(cells)
        ly_ = bar_y - 10 - 13 * r_
        if r_:
            svg.line(xx, ly_ + 3, xx, bar_y - 7, stroke=MID, sw=0.8)
        svg.path(f"M{xx - 4:.1f},{bar_y - 7}L{xx + 4:.1f},{bar_y - 7}L{xx:.1f},{bar_y - 0.5}Z", stroke=None, fill=INK)
        svg.text(left, ly_, f"`{name}`", size=10.5, halo="#ffffff")
    bottom2 = bar_y + 40

    # ---- the controller ----
    C = D["controller"]
    cx = ax + atlas_w + 26
    cw = W - M - cx
    th = C["thumb"]
    svg.text(cx, y + 10, "Per region: memo or brute force", size=12.5, weight="bold")
    cy = svg.para(cx, y + 26, f"A universe of {len(C['parts'])} patterns under [[{C['rule']}]], {thousands(C['w'])} × {thousands(C['h'])} cells.", cw) + 2
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
    # which file is where: a number at the corner of every part, and the key to the numbers
    files = []
    for part in C["parts"]:
        f_ = fname(part["file"])
        if f_ not in files:
            files.append(f_)
        px_, py_ = cx + part["x"] * sc, cy + part["y"] * sc
        svg.text(px_ + 2.5, py_ + 10, str(files.index(f_) + 1), size=10, weight="bold", fill="#ffffff", halo="#000000")
    count = {f_: sum(fname(part["file"]) == f_ for part in C["parts"]) for f_ in files}
    key = "   ".join(f"**{k + 1}**\u00a0`{f_}`" + (f"\u00a0×{count[f_]}" if count[f_] > 1 else "") for k, f_ in enumerate(files))
    ly = svg.para(cx, cy + ch + 14, key, cw, size=10.5, fill=GREY) + 5
    svg.rect(cx, ly - 9, 10, 10, fill=SKY)
    svg.text(cx + 15, ly, f"memo with the 16×16 floor: {memo_n} regions")
    svg.rect(cx, ly + 6, 10, 10, fill=ORANGE)
    svg.text(cx + 15, ly + 15, f"brute force, memo only above: {cells_n} regions")
    rom_file = fname(D["waves"]["rom"]["file"])
    bottom3 = svg.para(cx, ly + 33, f"{reg}×{reg}-cell regions, coloured by what they spent most of the first {thousands(C['gens'])} generations in. Where content is new at every scale "
                       f"(`{rom_file}` while it is being learned, part of the movie) every probe would miss, so the region is stepped by brute force. The rest keeps its memo. "
                       f"Model time for those generations: {C['adaptive_ms']:.1f} ms; with brute force everywhere, {C['plain_ms']:.1f} ms.", cw)
    return max(bottom1, bottom2, bottom3) + 2


# ---- 5. why the floor is where it is; against Golly -----------------------------------------------------------------------

def panel_why(svg, y, D, logs):
    rp = logs["rp"]
    rb = [("[[Hutton32]], table lookup", logs["rb_lut"], SKY, 3), ("[[Hutton32b]], the circuit", logs["rb_planes"], BLUE, 2), ("[[Life]], the adder tree", logs["rb_life"], GREEN, 3)]
    x0, w0 = M, 452
    svg.text(x0, y + 10, "Measured rates put the floor", size=12.5, weight="bold")
    svg.para(x0, y + 26, "A probe costs as much as ρ = R_{B} / R_{P} cell-updates of brute force: R_{B} is the rate of brute force, R_{P} that of probes. "
             "R_{P} is the same for every rule; R_{B} is not. Per second:", w0)
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
        return gx0 + gw * (lv - 3 + 0.6) / 3.95

    def GY(v):
        return gy + gh * (1 - (math.log10(v) - ylo) / (yhi - ylo))

    svg.text(x0, gy - 14, "What a hit saves, V / ρ. A w×w window gives its centre w/4 generations later: V = w^{3}/16 cell-updates", fill=GREY)
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
        if k == 0:                         # beside its own line: below it the other rule's line passes
            svg.text(a - 8, b + 3.5, f"{floors[-1][2]:.2f}×", size=10, anchor="end", fill=col, halo="#ffffff")
        else:
            svg.text(a + 8, b + 12, f"{floors[-1][2]:.2f}×", size=10, fill=col, halo="#ffffff")
        a, b = pts[-1]
        svg.text(a + 10, b + 4, lab, size=10.5, fill=col)
    (fh, gh_, bh_), (fl_, gl_, bl_) = floors
    bottom1 = svg.para(x0, gy + gh + 36, f"A window is worth a memo entry only if a hit saves more than the probe costs: from {2 ** fh}×{2 ** fh} windows on for [[Hutton32b]] "
                       f"({gh_:.1f}×; {2 ** (fh - 1)}×{2 ** (fh - 1)} windows: {bh_:.2f}×) and from {2 ** fl_}×{2 ** fl_} for [[Life]] ({gl_:.1f}×; {2 ** (fl_ - 1)}×{2 ** (fl_ - 1)}: {bl_:.2f}×). "
                       "That is each rule's floor: a larger ρ puts it one level coarser. Golly's HashLife memoizes down to 4×4 blocks, worth V = 4.", w0)

    # ---- against Golly ----
    rows, same = ghl_logs.outcomes(logs)
    assert same
    ox = x0 + w0 + 34
    ow = W - M - ox
    svg.text(ox, y + 10, "Against Golly 5.1, HashLife at its best step size: rate of the GHL part that ran, over Golly's", size=12.5, weight="bold")
    lab_w = 218
    px0, pw = ox + lab_w, ow - lab_w - 4
    lo, hi = -2.6, 6.75

    def PX(v):
        return px0 + pw * (math.log10(v) - lo) / (hi - lo)

    ty = y + 44
    rh, gh2 = 24.0, 17.0                    # a row; the heading of a group of rows
    groups = []
    for r_ in rows:
        if r_["group"] not in groups:
            groups.append(r_["group"])
    total = len(rows) * rh + len(groups) * gh2
    one = PX(1)
    svg.rect(px0, ty - 4, one - px0, total + 8, fill="#f3f3f3")
    svg.rect(one, ty - 4, px0 + pw - one, total + 8, fill="#e9f2f9")
    log_grid(svg, PX, lo, hi, ty - 4, ty + total + 4, minor="#c4ccd2")
    for e in range(-2, 7):
        xx = PX(10 ** e)
        svg.text(xx, ty + total + 17, "1×" if e == 0 else f"10^{{{e}}}×" if e > 0 else ("0.1×" if e == -1 else "0.01×"), size=10, anchor="middle", fill=GREY)
    svg.line(one, ty - 4, one, ty + total + 4, stroke=INK, sw=1.2)
    svg.text(one - 7, ty - 10, "Golly faster", size=10.5, anchor="end", fill=GREY)
    svg.text(one + 7, ty - 10, "GHL part faster", size=10.5, fill=BLUE)

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
                svg.line(ox, yy, px0 + pw, yy, stroke=MID, sw=0.8)       # one rule family above the line, the other below
            svg.rect(ox, yy + 1.5, pw + lab_w, gh2 - 3, fill="#ffffff", opacity=0.82)
            svg.text(ox, yy + 12.5, group, fill=GREY)
            yy += gh2
            cy = yy + rh / 2
        svg.text(px0 - 9, cy - 0.5, r_["name"], anchor="end")
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
                if left and xs[k] - 12 - text_width(txt, 10.5) < one < xs[k] - 8:
                    left = False
                svg.text(xs[k] - 10 if left else xs[k] + 10, cy + 3.8, txt, size=10.5, anchor="end" if left else "start",
                         halo=shade_at(xs[k] - 30 if left else xs[k] + 30))
        yy += rh
    ly = ty + total + 38
    lx = ox
    for kind, lab in (("gpu", "on the GPU, measured and verified"), ("learn", "the same, with the time to fill the memo on the CPU"), ("cpu", "the recursion on one CPU thread")):
        mark(kind, lx + 6, ly - 4)
        svg.text(lx + 16, ly, lab, size=10.5, fill=GREY)
        lx += 16 + text_width(lab, 10.5) + 18
    bottom2 = svg.para(ox, ly + 17, "Golly leads where content recurs sparsely and runs are long; there GHL falls back to the recursion on the CPU. Every GPU result was checked "
                       "against the CPU, and the CPU engine's final states equal Golly's.", ow, size=10.5)
    return max(bottom1, bottom2)


# ---- composition -----------------------------------------------------------------------------------------------

def build():
    D = json.load(open(DATA, encoding="utf-8"))
    logs = ghl_logs.collect()
    svg = Svg()
    svg.text(M, 34, "GPU HashLife (GHL)", size=26, weight="bold")
    tx = M + text_width("GPU HashLife (GHL)", 26, True) + 16
    svg.text(tx, 34, "HashLife above a memo floor, the rule as a circuit of logic gates below it", size=14)
    svg.text(W - M, 34, "Every picture and number is drawn from the measurement harness.", anchor="end", fill=GREY)
    svg.text(M, 53, "How a universe advances, and the words used below; the number on a step is the panel that shows it. "
             "The parts were measured separately: GHL has not yet run as one program.", fill=GREY)
    y = flow(svg, 64)
    y = section(svg, y, 1, "Six universes, two rules", "real patterns, named by their files; colour is the cell state")
    y = panel_universes(svg, y, D)
    y = section(svg, y + 10, 2, "Below the floor: the rule as a circuit", "[[Hutton32b]], 32 states, as a network of two-input gates", built=[("gpu", "built and measured on the GPU")])
    y, net, cells32 = panel_circuit(svg, y, D, logs)
    y = panel_life(svg, y + 4, D, logs)
    y = section(svg, y + 10, 3, "Above the floor: tiles, IDs and a memo table", "the hit path: four IDs read, one probe",
                built=[("gpu", "the hit path runs on the GPU"), ("cpu", "this picture: CPU reference engine")])
    y = panel_tiles(svg, y, D, net, cells32)
    y = section(svg, y + 10, 4, "A miss: the recursion in waves, level by level", "and how deep to go",
                built=[("cpu", "CPU reference engine"), ("model", "depth per universe and per region: model")])
    y = panel_miss(svg, y, D, logs)
    y = section(svg, y + 10, 5, "Why a floor, and what it buys", None, built=[("gpu", "rates and ratios measured"), ("model", "the floor: model")])
    y = panel_why(svg, y, D, logs)
    return svg.finish(y + 6)


if __name__ == "__main__":
    out = build()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(out)
    print("wrote", os.path.abspath(OUT), f"({len(out) >> 10} KiB)")
