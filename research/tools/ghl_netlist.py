"""Gate networks for the overview diagram (tools/ghl_diagram.py): the Hutton32b circuit of
research/bitslice/circuit.netlist and the adder tree of the Life kernel (src/cmd_life.rs). They are
parsed from those files, evaluated bit-sliced on real cells, and laid out in layers for drawing.

    python tools/ghl_netlist.py        print the statistics of both networks
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
NETLIST = os.path.join(HERE, "..", "bitslice", "circuit.netlist")
LIFE_SRC = os.path.join(HERE, "..", "src", "cmd_life.rs")


class Net:
    """inputs: names; gates: name -> (op, [operands]) in evaluation order; outputs: name -> signal."""

    def __init__(self, inputs, gates, outputs):
        self.inputs, self.gates, self.outputs = list(inputs), dict(gates), dict(outputs)
        self.order = list(gates)

    def count(self):
        c = {}
        for op, _ in self.gates.values():
            c[op] = c.get(op, 0) + 1
        return c

    def levels(self):
        """Longest path from the inputs, every gate (NOT included) counting one level."""
        lvl = {i: 0 for i in self.inputs}
        for n in self.order:
            lvl[n] = 1 + max(lvl[a] for a in self.gates[n][1])
        return lvl

    def depth(self):
        lvl = self.levels()
        return max(lvl[s] for s in self.outputs.values())

    def evaluate(self, words, mask):
        """Every signal for the given input words; bit i of every word belongs to cell i."""
        v = dict(words)
        for n in self.order:
            op, a = self.gates[n]
            if op == "NOT":
                v[n] = ~v[a[0]] & mask
            elif op == "AND":
                v[n] = v[a[0]] & v[a[1]]
            elif op == "OR":
                v[n] = v[a[0]] | v[a[1]]
            elif op == "XOR":
                v[n] = v[a[0]] ^ v[a[1]]
            else:
                raise ValueError(op)
        return v


def parse_netlist(path=NETLIST):
    inputs, gates, outputs = [], {}, {}
    for line in open(path, encoding="utf-8"):
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        if t.startswith("INPUT"):
            inputs = t.split()[1:]
        elif t.startswith("OUTPUT"):
            _, o, s = t.split()
            outputs[o] = s
        else:
            name, rhs = t.split(" = ")
            p = rhs.split()
            gates[name] = (p[0], p[1:])
    return Net(inputs, gates, outputs)


def parse_life(path=LIFE_SRC):
    """The carry-save adder tree of `life_fn`: eight neighbour words in, the count b3 b2 b1 b0 out."""
    src = open(path, encoding="utf-8").read()
    body = src[src.index("// carry-save adder tree"):]
    body = body[body.index("\n"):body.index("let born")]
    gates = {}
    ops = {"&": "AND", "|": "OR", "^": "XOR"}

    def expr(tokens, name):
        """tokens of `a op b` where a and b are names or parenthesised `x op y`; returns the signal."""
        parts, i = [], 0
        while i < len(tokens):
            if tokens[i] == "(":
                j = tokens.index(")", i)
                parts.append(tokens[i + 1:j])
                i = j + 1
            else:
                parts.append(tokens[i])
                i += 1
        assert len(parts) == 3 and parts[1] in ops, tokens
        operands = []
        for k, p in ((0, parts[0]), (1, parts[2])):
            operands.append(expr(p, f"{name}.{'ab'[k]}") if isinstance(p, list) else p)
        gates[name] = (ops[parts[1]], operands)
        return name

    for name, rhs in re.findall(r"let (\w+) = ([^;]+);", body):
        expr(re.findall(r"\w+|[&|^()]", rhs), name)
    inputs = ["n", "s", "e", "w", "ne", "nw", "se", "sw"]
    used = {a for _, args in gates.values() for a in args}
    assert used - set(gates) == set(inputs), sorted(used - set(gates))
    return Net(inputs, gates, {b: b for b in ("b0", "b1", "b2", "b3")})


# ---- bit-sliced stepping of a grid of cells ---------------------------------------------------------

def planes(row, bits):
    """One integer per bit-plane of a row of states; bit x of plane k is bit k of cell x."""
    out = []
    for k in range(bits):
        table = bytes(48 + ((v >> k) & 1) for v in range(256))
        out.append(int(row[::-1].translate(table), 2) if row else 0)
    return out


def step_grid(net, rows, width):
    """One generation of `rows` (bytes of states, zero outside) through the von Neumann network
    `net`, whose inputs are c0.., n0.., e0.., s0.., w0.. and outputs o0.. (bit k of the state)."""
    bits = len(net.outputs)
    mask = (1 << width) - 1
    zero = [0] * bits
    pl = [planes(r, bits) for r in rows]
    out = []
    for y, c in enumerate(pl):
        n = pl[y - 1] if y > 0 else zero
        s = pl[y + 1] if y + 1 < len(pl) else zero
        words = {}
        for k in range(bits):
            words[f"c{k}"], words[f"n{k}"], words[f"s{k}"] = c[k], n[k], s[k]
            words[f"e{k}"] = c[k] >> 1                 # the east neighbour of cell x is cell x + 1
            words[f"w{k}"] = (c[k] << 1) & mask
        v = net.evaluate(words, mask)
        o = [format(v[net.outputs[f"o{k}"]], f"0{width}b")[::-1] for k in range(bits)]
        out.append(bytes(sum((o[k][x] == "1") << k for k in range(bits)) for x in range(width)))
    return out


# ---- layered layout ---------------------------------------------------------------------------------

def _isotonic(z):
    """Least-squares non-decreasing fit (pool adjacent violators)."""
    blocks = []
    for v in z:
        blocks.append([v, 1])
        while len(blocks) > 1 and blocks[-2][0] * blocks[-1][1] > blocks[-1][0] * blocks[-2][1]:
            s, c = blocks.pop()
            blocks[-1][0] += s
            blocks[-1][1] += c
    out = []
    for s, c in blocks:
        out += [s / c] * c
    return out


def _crossings(edges):
    """Crossings between two layers; edges are (position in the left layer, position in the right)."""
    import bisect
    seen, n = [], 0
    for _, v in sorted(edges):
        n += len(seen) - bisect.bisect_right(seen, v)
        bisect.insort(seen, v)
    return n


class Layout:
    """pos: signal or pin -> (x, y); wires: [(signal, (x1, y1), (x2, y2))], one per drawn segment;
    level: gate -> column; depth: number of gate columns; crossings: wire crossings in the drawing."""


def layout(net, in_y, out_y, x0, x1, top, bottom, gate_gap=5.4, wire_gap=2.3, sweeps=40):
    """Place the gates in columns by logic level. A signal with several consumers runs as one wire
    with a branch per consumer (a trunk point in every column it passes). `in_y` and `out_y` give
    the fixed heights of the input and output pins (name -> y)."""
    lvl = net.levels()
    depth = max(lvl.values())
    cons = {}
    for g in net.order:
        for a in net.gates[g][1]:
            cons.setdefault(a, []).append(g)
    is_out = {s: o for o, s in net.outputs.items()}
    for s in is_out:                       # output gates in the last column
        lvl[s] = depth

    def far(sig, without=None, at=None):
        m = depth + 1 if sig in is_out else 0
        for c in cons.get(sig, []):
            m = max(m, at if c == without else lvl[c])
        return m

    # shorten the wires: move every gate inside its slack to where the nets it touches are shortest
    for _ in range(12):
        moved = False
        for g in list(reversed(net.order)) + net.order:
            if g in is_out:
                continue
            args = net.gates[g][1]
            lo = 1 + max(lvl[a] for a in args)
            hi = min(lvl[c] for c in cons[g]) - 1
            if lo >= hi:
                continue
            best, best_cost = lvl[g], None
            for cand in range(lo, hi + 1):
                cost = (far(g) - cand) + sum(far(a, g, cand) - lvl[a] for a in set(args))
                if best_cost is None or cost < best_cost or (cost == best_cost and abs(cand - lvl[g]) < abs(best - lvl[g])):
                    best, best_cost = cand, cost
            if best != lvl[g]:
                lvl[g], moved = best, True
        if not moved:
            break

    # nodes per column: pins and gates by name, trunk points as (signal, column)
    cols = [[] for _ in range(depth + 2)]
    cols[0] = list(net.inputs)
    for g in net.order:
        cols[lvl[g]].append(g)
    cols[depth + 1] = list(net.outputs)
    left, right = {}, {}                   # node -> neighbours in the column before / after

    def link(a, b):
        right.setdefault(a, []).append(b)
        left.setdefault(b, []).append(a)

    for sig in net.inputs + net.order:
        end = far(sig)
        prev = sig
        targets = {}
        for c in cons.get(sig, []):
            targets.setdefault(lvl[c], []).append(c)
        if sig in is_out:
            targets.setdefault(depth + 1, []).append(is_out[sig])
        for col in range(lvl[sig] + 1, end + 1):
            for t in targets.get(col, []):
                link(prev, t)
            if col < end:
                node = (sig, col)
                cols[col].append(node)
                link(prev, node)
                prev = node

    # order the columns: barycentre sweeps, keeping the pins where they are
    rank = {}

    def rerank(c):
        n = len(cols[c])
        for i, node in enumerate(cols[c]):
            rank[node] = (i + 0.5) / n

    cols[0].sort(key=lambda n: in_y[n])
    cols[depth + 1].sort(key=lambda n: out_y[n])
    for c in range(depth + 2):
        rerank(c)

    def total():
        return sum(_crossings([(rank[a], rank[b]) for b in cols[c] for a in left.get(b, [])]) for c in range(1, depth + 2))

    best_order, best_x = [list(c) for c in cols], None
    for it in range(sweeps):
        for c in (range(1, depth + 1) if it % 2 == 0 else range(depth, 0, -1)):
            nb = left if it % 2 == 0 else right
            key = {n: (sum(rank[m] for m in nb[n]) / len(nb[n]) if nb.get(n) else rank[n]) for n in cols[c]}
            cols[c].sort(key=lambda n: key[n])
            rerank(c)
        x = total()
        if best_x is None or x < best_x:
            best_x, best_order = x, [list(c) for c in cols]
    cols = best_order
    for c in range(depth + 2):
        rerank(c)

    # heights: pull every node towards its neighbours, keep the order and a minimal spacing
    def gap(a, b):
        ga, gb = not isinstance(a, tuple), not isinstance(b, tuple)
        return gate_gap if ga and gb else wire_gap if not ga and not gb else (gate_gap + wire_gap) / 2

    y = {}
    for n in cols[0]:
        y[n] = in_y[n]
    for n in cols[depth + 1]:
        y[n] = out_y[n]
    offs = {}
    for c in range(1, depth + 1):
        o, acc = [0.0], 0.0
        for a, b in zip(cols[c], cols[c][1:]):
            acc += gap(a, b)
            o.append(acc)
        scale = min(1.0, (bottom - top) / acc) if acc > 0 else 1.0
        offs[c] = [v * scale for v in o]
        span = offs[c][-1]
        for i, n in enumerate(cols[c]):
            y[n] = (top + bottom - span) / 2 + offs[c][i]

    def trunk(a, b):
        return isinstance(a, tuple) and (isinstance(b, tuple) and a[0] == b[0] or b == a[0]) or isinstance(b, tuple) and a == b[0]

    for it in range(80):
        for c in (range(1, depth + 1) if it % 2 == 0 else range(depth, 0, -1)):
            want = []
            for n in cols[c]:
                num = den = 0.0
                for nb, w0 in ((left, 1.0 if it % 2 == 0 or it > 50 else 0.35), (right, 1.0 if it % 2 == 1 or it > 50 else 0.35)):
                    for m in nb.get(n, []):
                        w = w0 * (3.0 if trunk(n, m) else 1.0)
                        num += w * y[m]
                        den += w
                want.append(num / den if den else y[n])
            o = offs[c]
            z = _isotonic([w - d for w, d in zip(want, o)])
            hi = bottom - o[-1]
            for n, v, d in zip(cols[c], z, o):
                y[n] = min(max(v, top), hi) + d

    L = Layout()
    dx = (x1 - x0) / (depth + 1)
    L.pos = {n: (x0 + dx * c, y[n]) for c in range(depth + 2) for n in cols[c]}
    L.wires = [(a[0] if isinstance(a, tuple) else a, L.pos[a], L.pos[b]) for a in right for b in right[a]]
    L.level, L.depth, L.crossings, L.dx = lvl, depth, best_x, dx
    L.trunk_points = sum(len([n for n in c if isinstance(n, tuple)]) for c in cols)
    return L


if __name__ == "__main__":
    for name, net in (("Hutton32b", parse_netlist()), ("Life adder tree", parse_life())):
        c = net.count()
        print(f"{name}: {len(net.inputs)} inputs, {len(net.gates)} gates {c}, depth {net.depth()}, {len(net.outputs)} outputs")
    net = parse_netlist()
    L = layout(net, {n: i * 16.0 for i, n in enumerate(net.inputs)}, {o: 150 + 20.0 * i for i, o in enumerate(net.outputs)}, 0, 800, 0, 400)
    print("layout: crossings", L.crossings, "trunk points", L.trunk_points, "wire segments", len(L.wires))
