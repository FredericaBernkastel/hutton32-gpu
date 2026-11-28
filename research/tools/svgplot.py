"""Minimal log-log line charts as standalone SVG (no plotting library needed).

    from svgplot import chart
    chart("out.svg", "x label", "y label", series=[("name", [(x, y), ...]), ...])

Ticks are the decades of the data unless `xticks` or `yticks` list them (then labelled as plain
numbers); `dashed` names series to draw with a dashed line.

Colours follow a fixed, colour-blind-safe order and every series also gets its own marker
shape, so the chart stays readable without colour. Text uses the reader's sans-serif font.
"""
import math

COLOURS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
MARKERS = ["circle", "square", "triangle", "diamond", "circle-open", "square-open", "triangle-open"]


def _ticks(lo, hi):
    a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    return [10.0 ** e for e in range(a, b + 1)]


def _fmt(v):
    e = int(round(math.log10(v)))
    return f"10<tspan baseline-shift='super' font-size='70%'>{e}</tspan>"


def _marker(kind, x, y, c):
    r = 4
    fill, stroke = (c, c) if not kind.endswith("open") else ("white", c)
    k = kind.replace("-open", "")
    if k == "circle":
        return f"<circle cx='{x:.1f}' cy='{y:.1f}' r='{r}' fill='{fill}' stroke='{stroke}' stroke-width='1.5'/>"
    if k == "square":
        return f"<rect x='{x - r:.1f}' y='{y - r:.1f}' width='{2 * r}' height='{2 * r}' fill='{fill}' stroke='{stroke}' stroke-width='1.5'/>"
    if k == "triangle":
        return f"<polygon points='{x:.1f},{y - r - 1:.1f} {x - r - 1:.1f},{y + r:.1f} {x + r + 1:.1f},{y + r:.1f}' fill='{fill}' stroke='{stroke}' stroke-width='1.5'/>"
    return f"<polygon points='{x:.1f},{y - r - 1:.1f} {x + r + 1:.1f},{y:.1f} {x:.1f},{y + r + 1:.1f} {x - r - 1:.1f},{y:.1f}' fill='{fill}' stroke='{stroke}' stroke-width='1.5'/>"


def chart(path, xlabel, ylabel, series, width=640, height=400, xlog=True, ylog=True, title=None, xticks=None, yticks=None, dashed=()):
    xs = [x for _, pts in series for x, _ in pts]
    ys = [y for _, pts in series for _, y in pts]
    xt, yt = xticks or _ticks(min(xs), max(xs)), yticks or _ticks(min(ys), max(ys))
    assert xt[0] <= min(xs) and max(xs) <= xt[-1] and yt[0] <= min(ys) and max(ys) <= yt[-1], "data outside the ticks"
    xf = _fmt if xticks is None else (lambda v: f"{v:g}")
    yf = _fmt if yticks is None else (lambda v: f"{v:g}")
    x0, x1, y0, y1 = xt[0], xt[-1], yt[0], yt[-1]
    # the right margin holds the legend: wide enough for its longest name (13 px Helvetica, ~7 px a character)
    ml, mr, mt, mb = 72, max(190, 44 + 7 * max(len(n) for n, _ in series) + 12), 20 if not title else 40, 52
    pw, ph = width - ml - mr, height - mt - mb
    X = lambda v: ml + pw * (math.log10(v) - math.log10(x0)) / (math.log10(x1) - math.log10(x0))
    Y = lambda v: mt + ph * (1 - (math.log10(v) - math.log10(y0)) / (math.log10(y1) - math.log10(y0)))
    o = [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}' "
         f"font-family='Helvetica, Arial, sans-serif' font-size='13'>",
         f"<rect width='{width}' height='{height}' fill='white'/>"]
    if title:
        o.append(f"<text x='{ml}' y='22' font-weight='bold'>{title}</text>")
    for v in xt:
        o.append(f"<line x1='{X(v):.1f}' y1='{mt}' x2='{X(v):.1f}' y2='{mt + ph}' stroke='#e4e4e4'/>")
        o.append(f"<text x='{X(v):.1f}' y='{mt + ph + 18}' text-anchor='middle'>{xf(v)}</text>")
    for v in yt:
        o.append(f"<line x1='{ml}' y1='{Y(v):.1f}' x2='{ml + pw}' y2='{Y(v):.1f}' stroke='#e4e4e4'/>")
        o.append(f"<text x='{ml - 8}' y='{Y(v) + 4:.1f}' text-anchor='end'>{yf(v)}</text>")
    o.append(f"<rect x='{ml}' y='{mt}' width='{pw}' height='{ph}' fill='none' stroke='#555'/>")
    o.append(f"<text x='{ml + pw / 2:.1f}' y='{height - 10}' text-anchor='middle'>{xlabel}</text>")
    o.append(f"<text transform='translate(18,{mt + ph / 2:.1f}) rotate(-90)' text-anchor='middle'>{ylabel}</text>")
    for i, (name, pts) in enumerate(series):
        c, m = COLOURS[i % len(COLOURS)], MARKERS[i % len(MARKERS)]
        pts = sorted(pts)
        dash = " stroke-dasharray='6 4'" if name in dashed else ""
        if len(pts) > 1:
            d = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in pts)
            o.append(f"<polyline points='{d}' fill='none' stroke='{c}' stroke-width='2'{dash}/>")
        o += [_marker(m, X(x), Y(y), c) for x, y in pts]
        ly = mt + 14 + 22 * i
        o.append(f"<line x1='{ml + pw + 14}' y1='{ly}' x2='{ml + pw + 38}' y2='{ly}' stroke='{c}' stroke-width='2'{dash}/>")
        o.append(_marker(m, ml + pw + 26, ly, c))
        o.append(f"<text x='{ml + pw + 44}' y='{ly + 4}'>{name}</text>")
    o.append("</svg>")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(o) + "\n")
