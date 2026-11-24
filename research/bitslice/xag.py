"""Small logic-network toolkit used by every script in this directory.

* `XAG`      - AND/XOR graph with complemented edges (construction, structural hashing).
* `Netlist`  - the final form: straight-line list of NOT/AND/OR/XOR gates (what the WGSL/Rust
               code is generated from), with a text format, statistics and exhaustive simulation.
* `Sim`      - exhaustive bit-parallel simulation over all 2^25 neighbourhoods
               (index = w | s<<5 | e<<10 | n<<15 | c<<20, exactly the LUT layout).
* `map_xag`  - XAG -> Netlist with inverter minimisation (choice of AND vs OR form per node).

Nothing here is rule-specific.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "out"
LUT_PATH = OUT_DIR / "lut_table_Hutton32b.bin"

POS = ["w", "s", "e", "n", "c"]  # LUT index order, least significant field first
INPUT_NAMES = [f"{p}{k}" for p in ["c", "n", "e", "s", "w"] for k in range(5)]
OUTPUT_NAMES = [f"o{k}" for k in range(5)]
NBITS = 25
NWORDS = 1 << (NBITS - 6)
ALL = np.uint64(0xFFFFFFFFFFFFFFFF)


# ------------------------------------------------------------------------------------------------
# exhaustive simulation
# ------------------------------------------------------------------------------------------------
class Sim:
    """Bit-planes of the 25 inputs over all 2^25 neighbourhoods, and of the reference LUT."""

    _inputs = None
    _lut = None
    _lut_planes = None

    @classmethod
    def index_bit(cls, j: int) -> np.ndarray:
        """Plane of bit `j` of the LUT index (bit p of word q <-> index 64*q + p)."""
        if j < 6:
            pat = 0
            for p in range(64):
                if (p >> j) & 1:
                    pat |= 1 << p
            return np.full(NWORDS, pat, dtype=np.uint64)
        q = np.arange(NWORDS, dtype=np.uint64)
        return np.where((q >> np.uint64(j - 6)) & np.uint64(1) == 1, ALL, np.uint64(0)).astype(np.uint64)

    @classmethod
    def inputs(cls) -> dict[str, np.ndarray]:
        if cls._inputs is None:
            d = {}
            for f, p in enumerate(POS):
                for k in range(5):
                    d[f"{p}{k}"] = cls.index_bit(5 * f + k)
            cls._inputs = d
        return cls._inputs

    @classmethod
    def lut(cls) -> np.ndarray:
        if cls._lut is None:
            lut = np.fromfile(LUT_PATH, dtype=np.uint8)
            assert lut.size == 1 << NBITS, f"unexpected LUT size {lut.size}"
            cls._lut = lut
        return cls._lut

    @classmethod
    def lut_planes(cls) -> list[np.ndarray]:
        if cls._lut_planes is None:
            lut = cls.lut()
            assert int(lut.max()) < 32
            cls._lut_planes = [
                np.packbits((lut >> k) & 1, bitorder="little").view(np.uint64).copy() for k in range(5)
            ]
        return cls._lut_planes

    @staticmethod
    def popcount(a: np.ndarray) -> int:
        return int(np.bitwise_count(a).sum())


# ------------------------------------------------------------------------------------------------
# XAG
# ------------------------------------------------------------------------------------------------
class XAG:
    """AND/XOR graph. A literal is `2*node + complement`; literal 0 is constant false."""

    def __init__(self):
        self.op: list[str] = ["const"]
        self.a: list[int] = [0]
        self.b: list[int] = [0]
        self.in_name: dict[int, str] = {}
        self.in_lit: dict[str, int] = {}
        self._hash: dict[tuple, int] = {}
        self.outputs: list[int] = []
        self.marks: list[tuple[str, int]] = []

    def mark(self, label: str) -> None:
        """Record a section boundary (used to report how many nodes each block creates)."""
        self.marks.append((label, len(self.op)))

    def section_sizes(self) -> list[tuple[str, int]]:
        res, prev = [], 1 + len(self.in_lit)
        for label, end in self.marks:
            res.append((label, end - prev))
            prev = end
        return res

    # -- construction -----------------------------------------------------------------------
    def inp(self, name: str) -> int:
        if name in self.in_lit:
            return self.in_lit[name]
        node = len(self.op)
        self.op.append("in")
        self.a.append(0)
        self.b.append(0)
        self.in_name[node] = name
        self.in_lit[name] = 2 * node
        return 2 * node

    def all_inputs(self):
        return {name: self.inp(name) for name in INPUT_NAMES}

    @staticmethod
    def NOT(x: int) -> int:
        return x ^ 1

    def _and2(self, x: int, y: int) -> int:
        if x > y:
            x, y = y, x
        if x == 0:
            return 0
        if x == 1:
            return y
        if x == y:
            return x
        if x == (y ^ 1):
            return 0
        key = ("and", x, y)
        r = self._hash.get(key)
        if r is None:
            node = len(self.op)
            self.op.append("and")
            self.a.append(x)
            self.b.append(y)
            r = self._hash[key] = 2 * node
        return r

    def _xor2(self, x: int, y: int) -> int:
        c = (x ^ y) & 1
        x &= ~1
        y &= ~1
        if x > y:
            x, y = y, x
        if x == y:
            return c
        if x == 0:
            return y ^ c
        key = ("xor", x, y)
        r = self._hash.get(key)
        if r is None:
            node = len(self.op)
            self.op.append("xor")
            self.a.append(x)
            self.b.append(y)
            r = self._hash[key] = 2 * node
        return r ^ c

    @staticmethod
    def _tree(f, xs, empty):
        xs = list(xs)
        if not xs:
            return empty
        while len(xs) > 1:
            nxt = [f(xs[i], xs[i + 1]) for i in range(0, len(xs) - 1, 2)]
            if len(xs) & 1:
                nxt.append(xs[-1])
            xs = nxt
        return xs[0]

    def AND(self, *xs: int) -> int:
        return self._tree(self._and2, xs, 1)

    def OR(self, *xs: int) -> int:
        return self._tree(self._and2, [x ^ 1 for x in xs], 1) ^ 1

    def XOR(self, *xs: int) -> int:
        return self._tree(self._xor2, xs, 0)

    def MUX(self, s: int, t: int, f: int) -> int:
        """s ? t : f"""
        return self.OR(self._and2(s, t), self._and2(s ^ 1, f))

    # -- analysis ---------------------------------------------------------------------------
    def live_nodes(self, outputs=None) -> list[int]:
        """Nodes in the cone of the outputs, in topological (index) order."""
        outputs = self.outputs if outputs is None else outputs
        seen = set()
        stack = [l >> 1 for l in outputs]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if self.op[n] in ("and", "xor"):
                stack.append(self.a[n] >> 1)
                stack.append(self.b[n] >> 1)
        return sorted(seen)

    def count(self, outputs=None) -> dict[str, int]:
        live = self.live_nodes(outputs)
        return {
            "and": sum(1 for n in live if self.op[n] == "and"),
            "xor": sum(1 for n in live if self.op[n] == "xor"),
        }

    def simulate(self, lits, inputs=None) -> list[np.ndarray]:
        """Exhaustive simulation; returns one plane per requested literal."""
        inputs = Sim.inputs() if inputs is None else inputs
        live = self.live_nodes(lits)
        refs = {n: 0 for n in live}
        for n in live:
            if self.op[n] in ("and", "xor"):
                refs[self.a[n] >> 1] += 1
                refs[self.b[n] >> 1] += 1
        for l in lits:
            refs[l >> 1] += 1
        val: dict[int, np.ndarray] = {}
        zero = np.zeros_like(next(iter(inputs.values())))

        def get(l):
            v = val[l >> 1]
            return ~v if l & 1 else v

        def release(l):
            n = l >> 1
            refs[n] -= 1
            if refs[n] == 0:
                del val[n]

        for n in live:
            o = self.op[n]
            if o == "const":
                val[n] = zero
            elif o == "in":
                val[n] = inputs[self.in_name[n]]
            else:
                x, y = get(self.a[n]), get(self.b[n])
                val[n] = (x & y) if o == "and" else (x ^ y)
                release(self.a[n])
                release(self.b[n])
        res = []
        for l in lits:
            res.append(get(l).copy() if True else None)
        return res

    def check(self, lit: int, expected_plane: np.ndarray, what: str = "") -> None:
        got = self.simulate([lit])[0]
        bad = Sim.popcount(got ^ expected_plane)
        assert bad == 0, f"check failed for {what}: {bad} mismatching neighbourhoods"

    # -- export for ABC -----------------------------------------------------------------------
    def to_blif(self, outputs=None, out_names=None, model="rule") -> str:
        outputs = self.outputs if outputs is None else outputs
        out_names = OUTPUT_NAMES if out_names is None else out_names
        live = self.live_nodes(outputs)
        ins = [n for n in live if self.op[n] == "in"]
        # keep every declared input (ABC must see all 25 even if one were unused)
        in_names = list(self.in_lit.keys())
        lines = [f".model {model}", ".inputs " + " ".join(in_names), ".outputs " + " ".join(out_names)]
        nm = {n: self.in_name[n] for n in ins}
        for n in live:
            if self.op[n] in ("and", "xor"):
                nm[n] = f"g{n}"
        for n in live:
            o = self.op[n]
            if o == "and":
                a, b = self.a[n], self.b[n]
                lines.append(f".names {nm[a >> 1]} {nm[b >> 1]} {nm[n]}")
                lines.append(f"{1 - (a & 1)}{1 - (b & 1)} 1")
            elif o == "xor":
                a, b = self.a[n], self.b[n]
                lines.append(f".names {nm[a >> 1]} {nm[b >> 1]} {nm[n]}")
                lines.append("10 1")
                lines.append("01 1")
        for l, name in zip(outputs, out_names):
            if l >> 1 == 0:
                lines.append(f".names {name}")
                if l & 1:
                    lines.append("1")
            else:
                lines.append(f".names {nm[l >> 1]} {name}")
                lines.append("0 1" if l & 1 else "1 1")
        lines.append(".end")
        return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------
# Netlist (NOT/AND/OR/XOR, straight-line)
# ------------------------------------------------------------------------------------------------
class Netlist:
    """`gates` is a topologically ordered list of (name, op, a, b) with op in NOT/AND/OR/XOR
    (`b` is None for NOT). `outputs` maps o0..o4 to a signal name (a gate, an input, or the
    constants "0"/"1")."""

    OPS = ("NOT", "AND", "OR", "XOR")

    def __init__(self, inputs=None):
        self.inputs: list[str] = list(INPUT_NAMES if inputs is None else inputs)
        self.gates: list[tuple[str, str, str, str | None]] = []
        self.outputs: dict[str, str] = {}

    # -- statistics ---------------------------------------------------------------------------
    def counts(self) -> dict[str, int]:
        c = {op: 0 for op in self.OPS}
        for _, op, _, _ in self.gates:
            c[op] += 1
        c["total"] = len(self.gates)
        return c

    def depth(self, count_not: bool = True) -> int:
        """Longest input-to-output path, in gates (optionally not counting NOT)."""
        d = {name: 0 for name in self.inputs}
        d["0"] = d["1"] = 0
        for name, op, a, b in self.gates:
            if op == "NOT":
                d[name] = d[a] + (1 if count_not else 0)
            else:
                d[name] = max(d[a], d[b]) + 1
        return max(d[s] for s in self.outputs.values())

    def max_live(self) -> int:
        """Maximum number of gate values simultaneously live in the given order (register
        pressure of the straight-line program)."""
        last = {}
        for i, (name, op, a, b) in enumerate(self.gates):
            last[a] = i
            if b is not None:
                last[b] = i
        for s in self.outputs.values():
            last[s] = len(self.gates)
        gate_names = {g[0] for g in self.gates}
        live = 0
        best = 0
        ends = {}
        for s, i in last.items():
            if s in gate_names:
                ends.setdefault(i, []).append(s)
        for i, (name, op, a, b) in enumerate(self.gates):
            live -= len(ends.get(i, []))
            live += 1
            best = max(best, live)
        return best

    # -- text format --------------------------------------------------------------------------
    def to_text(self, header: str = "") -> str:
        lines = []
        for h in header.splitlines():
            lines.append("# " + h if h else "#")
        lines.append("INPUT " + " ".join(self.inputs))
        for name, op, a, b in self.gates:
            lines.append(f"{name} = {op} {a}" + ("" if b is None else f" {b}"))
        for o, s in self.outputs.items():
            lines.append(f"OUTPUT {o} {s}")
        return "\n".join(lines) + "\n"

    @classmethod
    def from_text(cls, text: str) -> "Netlist":
        nl = cls(inputs=[])
        defined = {"0", "1"}
        for ln, raw in enumerate(text.splitlines(), 1):
            line = raw.split("#")[0].strip()
            if not line:
                continue
            tok = line.split()
            if tok[0] == "INPUT":
                nl.inputs += tok[1:]
                defined.update(tok[1:])
            elif tok[0] == "OUTPUT":
                assert len(tok) == 3 and tok[2] in defined, f"line {ln}: bad output"
                nl.outputs[tok[1]] = tok[2]
            else:
                assert len(tok) in (4, 5) and tok[1] == "=", f"line {ln}: bad gate line"
                name, op = tok[0], tok[2]
                assert op in cls.OPS, f"line {ln}: unknown op {op}"
                assert name not in defined, f"line {ln}: {name} redefined"
                args = tok[3:]
                assert len(args) == (1 if op == "NOT" else 2), f"line {ln}: wrong arity"
                for x in args:
                    assert x in defined, f"line {ln}: {x} used before definition"
                nl.gates.append((name, op, args[0], args[1] if len(args) > 1 else None))
                defined.add(name)
        return nl

    # -- simulation ---------------------------------------------------------------------------
    def simulate(self, inputs=None) -> dict[str, np.ndarray]:
        """Exhaustive simulation of the outputs."""
        inputs = Sim.inputs() if inputs is None else inputs
        some = next(iter(inputs.values()))
        val = {name: inputs[name] for name in self.inputs}
        val["0"] = np.zeros_like(some)
        val["1"] = ~np.zeros_like(some)
        last = {}
        for i, (name, op, a, b) in enumerate(self.gates):
            last[a] = i
            if b is not None:
                last[b] = i
        keep = set(self.outputs.values()) | set(self.inputs) | {"0", "1"}
        for i, (name, op, a, b) in enumerate(self.gates):
            if op == "NOT":
                val[name] = ~val[a]
            elif op == "AND":
                val[name] = val[a] & val[b]
            elif op == "OR":
                val[name] = val[a] | val[b]
            else:
                val[name] = val[a] ^ val[b]
            for x in (a, b):
                if x is not None and last.get(x) == i and x not in keep:
                    val.pop(x, None)
        return {o: val[s] for o, s in self.outputs.items()}

    def verify(self) -> list[int]:
        """Mismatch count per output bit against the reference LUT (all 2^25 entries)."""
        res = self.simulate()
        planes = Sim.lut_planes()
        return [Sim.popcount(res[f"o{k}"] ^ planes[k]) for k in range(5)]

    # -- conversion ---------------------------------------------------------------------------
    def to_xag(self) -> XAG:
        x = XAG()
        lit = {"0": 0, "1": 1}
        for name in self.inputs:
            lit[name] = x.inp(name)
        for name, op, a, b in self.gates:
            if op == "NOT":
                lit[name] = lit[a] ^ 1
            elif op == "AND":
                lit[name] = x.AND(lit[a], lit[b])
            elif op == "OR":
                lit[name] = x.OR(lit[a], lit[b])
            else:
                lit[name] = x.XOR(lit[a], lit[b])
        x.outputs = [lit[self.outputs[o]] for o in sorted(self.outputs)]
        return x

    def renamed(self, prefix: str = "t") -> "Netlist":
        """Rename gates to t0, t1, ... in order (inputs/outputs keep their names)."""
        nl = Netlist(self.inputs)
        ren = {n: n for n in self.inputs}
        ren["0"], ren["1"] = "0", "1"
        for i, (name, op, a, b) in enumerate(self.gates):
            ren[name] = f"{prefix}{i}"
            nl.gates.append((ren[name], op, ren[a], None if b is None else ren[b]))
        nl.outputs = {o: ren[s] for o, s in self.outputs.items()}
        return nl


# ------------------------------------------------------------------------------------------------
# XAG -> Netlist with inverter minimisation
# ------------------------------------------------------------------------------------------------
def _map_cost(x: XAG, live, outputs, F, want_plan=False):
    """Cost (#gates + #inverters) for native polarities F (node -> 0/1).

    AND node n = la & lb.  Native polarity 0: AND of the literals; native polarity 1 (the gate
    produces the complement of n): OR of the complemented literals.  XOR nodes produce
    polarity pa^pb for whatever polarities pa, pb of the fanins they read.
    A node needs an inverter iff somebody reads the polarity it does not produce natively.
    """
    need = {}  # node -> set of polarities read

    def req(node, pol):
        need.setdefault(node, set()).add(pol)

    plan = {}
    xors = []
    for n in live:
        o = x.op[n]
        if o == "and":
            f = F[n]
            a, b = x.a[n], x.b[n]
            pa, pb = (a & 1) ^ f, (b & 1) ^ f
            req(a >> 1, pa)
            req(b >> 1, pb)
            plan[n] = (pa, pb)
        elif o == "xor":
            xors.append(n)
    for l in outputs:
        if l >> 1:
            req(l >> 1, l & 1)

    def native(m):
        return F.get(m, 0)

    # xor nodes: parity of fanin polarities must equal native polarity of the node
    pending = []
    for n in xors:
        a, b = x.a[n] >> 1, x.b[n] >> 1
        if native(a) ^ native(b) == F[n]:
            req(a, native(a))
            req(b, native(b))
            plan[n] = (native(a), native(b))
        else:
            pending.append(n)
    # mismatching xor nodes: read the non-native polarity of one fanin, preferring one whose
    # inverter exists anyway
    changed = True
    while pending and changed:
        changed = False
        rest = []
        for n in pending:
            a, b = x.a[n] >> 1, x.b[n] >> 1
            ia = (1 ^ native(a)) in need.get(a, ())
            ib = (1 ^ native(b)) in need.get(b, ())
            if ia or ib:
                if ia:
                    pa, pb = 1 ^ native(a), native(b)
                else:
                    pa, pb = native(a), 1 ^ native(b)
                req(a, pa)
                req(b, pb)
                plan[n] = (pa, pb)
                changed = True
            else:
                rest.append(n)
        pending = rest
        if pending and not changed:
            n = pending.pop(0)
            a, b = x.a[n] >> 1, x.b[n] >> 1
            pa, pb = 1 ^ native(a), native(b)
            req(a, pa)
            req(b, pb)
            plan[n] = (pa, pb)
            changed = True
    inv = 0
    for m, pols in need.items():
        if (1 ^ native(m)) in pols:
            inv += 1
    gates = sum(1 for n in live if x.op[n] in ("and", "xor"))
    if want_plan:
        return gates + inv, plan, need
    return gates + inv


def map_xag(x: XAG, outputs=None, iters: int = 4000, seed: int = 1, restarts: int = 3) -> Netlist:
    """Map an XAG to NOT/AND/OR/XOR, choosing for every node whether to build it or its
    complement so that the number of NOT gates is small (local search)."""
    outputs = x.outputs if outputs is None else outputs
    live = x.live_nodes(outputs)
    gate_nodes = [n for n in live if x.op[n] in ("and", "xor")]
    rng = random.Random(seed)
    best_F, best_cost = None, None
    for r in range(restarts):
        F = {n: (0 if r == 0 else rng.randint(0, 1)) for n in gate_nodes}
        cost = _map_cost(x, live, outputs, F)
        stale = 0
        order = list(gate_nodes)
        for it in range(iters):
            improved = False
            rng.shuffle(order)
            for n in order:
                F[n] ^= 1
                c2 = _map_cost(x, live, outputs, F)
                if c2 < cost or (c2 == cost and rng.random() < 0.3):
                    if c2 < cost:
                        improved = True
                    cost = c2
                else:
                    F[n] ^= 1
            if not improved:
                stale += 1
                if stale >= 3:
                    break
            else:
                stale = 0
        if best_cost is None or cost < best_cost:
            best_cost, best_F = cost, dict(F)
    return emit_mapped(x, outputs, best_F)


def map_xag_exact(x: XAG, outputs=None, time_limit: float = 20.0) -> Netlist:
    """Like `map_xag`, but the polarity assignment is solved as MaxSAT (python-sat):
    minimise the number of nodes that are read in the polarity they do not produce natively.
    Core-guided RC2 is tried first (usually proves the optimum in well under a second); if it
    exceeds `time_limit`, the anytime LSU algorithm is run for `time_limit` seconds and its
    best assignment is used.  `nl.polarity_optimal` records whether optimality was proven."""
    from pysat.formula import WCNF

    outputs = x.outputs if outputs is None else outputs
    live = x.live_nodes(outputs)
    gate_nodes = [n for n in live if x.op[n] in ("and", "xor")]
    fv = {n: i + 1 for i, n in enumerate(gate_nodes)}          # F[n]
    iv = {}                                                     # I[m]
    nxt = [len(gate_nodes)]

    def inv(m):
        if m not in iv:
            nxt[0] += 1
            iv[m] = nxt[0]
        return iv[m]

    w = WCNF()

    def same_or_inv(m, fn_lit, e):
        """(F[m] xor fn xor e == 0) or I[m];  fn_lit is a literal for F[n] or None (constant 0)."""
        im = inv(m)
        fm = fv.get(m)  # None for inputs (native polarity 0)
        if fm is None and fn_lit is None:
            if e:
                w.append([im])
        elif fm is None:
            w.append([(fn_lit if e else -fn_lit), im])
        elif fn_lit is None:
            w.append([(fm if e else -fm), im])
        elif e == 0:
            w.append([-fm, fn_lit, im])
            w.append([fm, -fn_lit, im])
        else:
            w.append([fm, fn_lit, im])
            w.append([-fm, -fn_lit, im])

    for n in gate_nodes:
        a, b = x.a[n], x.b[n]
        if x.op[n] == "and":
            same_or_inv(a >> 1, fv[n], a & 1)
            same_or_inv(b >> 1, fv[n], b & 1)
        else:
            ma, mb = a >> 1, b >> 1
            ia, ib = inv(ma), inv(mb)
            va, vb, vn = fv.get(ma), fv.get(mb), fv[n]
            # forbid odd parity of (F[ma], F[mb], F[n]) unless an inverter is available
            for pa in (0, 1):
                for pb in (0, 1):
                    pn = 1 ^ pa ^ pb
                    if (va is None and pa) or (vb is None and pb):
                        continue
                    cl = [ia, ib]
                    if va is not None:
                        cl.append(-va if pa else va)
                    if vb is not None:
                        cl.append(-vb if pb else vb)
                    cl.append(-vn if pn else vn)
                    w.append(cl)
    for l in outputs:
        if l >> 1:
            same_or_inv(l >> 1, None, l & 1)
    for m, v in iv.items():
        w.append([-v], weight=1)

    # solve in a helper process (python-sat keeps the GIL, so a thread could not be timed out)
    import os
    import subprocess
    import sys
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".wcnf")
    os.close(fd)
    try:
        w.to_file(path)
        worker = str(HERE / "maxsat_worker.py")

        def parse(out):
            opt, model = False, None
            for line in out.splitlines():
                if line.startswith("optimal"):
                    opt = line.split()[1] == "1"
                elif line.startswith("v "):
                    model = [int(t) for t in line.split()[1:]]
            return opt, model

        model, optimal = None, False
        try:
            r = subprocess.run([sys.executable, worker, path, "rc2"], capture_output=True, text=True,
                               timeout=time_limit)
            if r.returncode == 0:
                optimal, model = parse(r.stdout)
        except subprocess.TimeoutExpired:
            pass
        if model is None:
            r = subprocess.run([sys.executable, worker, path, "lsu", str(time_limit)], capture_output=True,
                               text=True, timeout=time_limit * 10 + 120)
            assert r.returncode == 0, r.stderr
            optimal, model = parse(r.stdout)
            assert model is not None
    finally:
        os.unlink(path)
    val = {abs(v): v > 0 for v in model}
    F = {n: int(val.get(fv[n], False)) for n in gate_nodes}
    nl = emit_mapped(x, outputs, F)
    nl.polarity_optimal = optimal
    return nl


def emit_mapped(x: XAG, outputs, F) -> Netlist:
    live = x.live_nodes(outputs)
    _, plan, need = _map_cost(x, live, outputs, F, want_plan=True)
    nl = Netlist(list(x.in_lit.keys()))
    sig = {}  # (node, polarity) -> signal name
    counter = [0]

    def fresh():
        counter[0] += 1
        return f"t{counter[0] - 1}"

    def native(m):
        return F.get(m, 0)

    def get(m, pol):
        key = (m, pol)
        if key not in sig:
            # only the non-native polarity can be missing: build the inverter now
            assert (m, 1 ^ pol) in sig, "native signal must exist before its inverter"
            name = fresh()
            nl.gates.append((name, "NOT", sig[(m, 1 ^ pol)], None))
            sig[key] = name
        return sig[key]

    for n in live:
        o = x.op[n]
        if o == "in":
            sig[(n, 0)] = x.in_name[n]
        elif o == "and":
            pa, pb = plan[n]
            sa, sb = get(x.a[n] >> 1, pa), get(x.b[n] >> 1, pb)
            name = fresh()
            nl.gates.append((name, "AND" if F[n] == 0 else "OR", sa, sb))
            sig[(n, F[n])] = name
        elif o == "xor":
            pa, pb = plan[n]
            sa, sb = get(x.a[n] >> 1, pa), get(x.b[n] >> 1, pb)
            name = fresh()
            nl.gates.append((name, "XOR", sa, sb))
            sig[(n, F[n])] = name
    for k, l in enumerate(outputs):
        if l >> 1 == 0:
            nl.outputs[f"o{k}"] = "1" if l & 1 else "0"
        else:
            nl.outputs[f"o{k}"] = get(l >> 1, l & 1)
    return nl


# ------------------------------------------------------------------------------------------------
# BLIF reader for mapped netlists written by Yosys (`write_blif`) or ABC (`write_blif`)
# ------------------------------------------------------------------------------------------------
def read_blif_to_xag(text: str) -> XAG:
    """Read a combinational BLIF made of .names (any cover with <= 6 inputs) and/or
    .gate/.subckt lines for NOT/AND/OR/XOR/BUF (+ the Yosys `$_X_` names)."""
    # join continuation lines
    text = text.replace("\\\n", " ")
    inputs, outputs = [], []
    defs = {}  # signal -> ("names", ins, cover) | ("gate", type, ins)
    cur = None
    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        if line.startswith("."):
            cur = None
            tok = line.split()
            if tok[0] == ".inputs":
                inputs += tok[1:]
            elif tok[0] == ".outputs":
                outputs += tok[1:]
            elif tok[0] == ".names":
                *ins, out = tok[1:]
                cur = ("names", ins, [])
                defs[out] = cur
            elif tok[0] in (".gate", ".subckt"):
                gtype = tok[1].strip("$_").upper()
                pins = dict(p.split("=", 1) for p in tok[2:])
                out = pins.pop("Y")
                ins = [pins[k] for k in sorted(pins)]
                defs[out] = ("gate", gtype, ins)
            elif tok[0] in (".model", ".end"):
                pass
            else:
                raise ValueError(f"unsupported BLIF construct: {line}")
        else:
            assert cur is not None, f"stray line: {line}"
            cur[2].append(line.split())
    x = XAG()
    lit = {}
    for name in INPUT_NAMES:
        lit[name] = x.inp(name)
    for name in inputs:
        assert name in lit, f"unexpected input {name}"

    def build(sig):
        # iterative DFS
        stack = [sig]
        while stack:
            s = stack[-1]
            if s in lit:
                stack.pop()
                continue
            if s in ("$false", "$undef"):
                lit[s] = 0
                stack.pop()
                continue
            if s == "$true":
                lit[s] = 1
                stack.pop()
                continue
            d = defs[s]
            ins = d[1] if d[0] == "names" else d[2]
            missing = [i for i in ins if i not in lit]
            if missing:
                stack.extend(missing)
                continue
            stack.pop()
            if d[0] == "gate":
                g = d[1]
                v = [lit[i] for i in ins]
                if g == "NOT":
                    lit[s] = v[0] ^ 1
                elif g == "BUF":
                    lit[s] = v[0]
                elif g == "AND":
                    lit[s] = x.AND(*v)
                elif g == "OR":
                    lit[s] = x.OR(*v)
                elif g == "XOR":
                    lit[s] = x.XOR(*v)
                elif g == "NAND":
                    lit[s] = x.AND(*v) ^ 1
                elif g == "NOR":
                    lit[s] = x.OR(*v) ^ 1
                elif g == "XNOR":
                    lit[s] = x.XOR(*v) ^ 1
                elif g == "ZERO":
                    lit[s] = 0
                elif g == "ONE":
                    lit[s] = 1
                else:
                    raise ValueError(f"unknown gate {g}")
            else:
                cover = d[2]
                v = [lit[i] for i in ins]
                if not ins:
                    # constant: a line "1" means true
                    lit[s] = 1 if any(c == ["1"] for c in cover) else 0
                    continue
                on_rows = [c for c in cover if c[-1] == "1"]
                off_rows = [c for c in cover if c[-1] == "0"]
                assert not (on_rows and off_rows)
                rows = on_rows or off_rows
                # recognise xor/xnor of two inputs so that it stays one node
                pats = sorted(c[0] for c in rows)
                if len(ins) == 2 and pats == ["01", "10"]:
                    r = x.XOR(v[0], v[1])
                elif len(ins) == 2 and pats == ["00", "11"]:
                    r = x.XOR(v[0], v[1]) ^ 1
                else:
                    terms = []
                    for c in rows:
                        cube = c[0] if len(c) == 2 else ""
                        t = [v[i] ^ (1 if ch == "0" else 0) for i, ch in enumerate(cube) if ch != "-"]
                        terms.append(x.AND(*t))
                    r = x.OR(*terms) if terms else 0
                lit[s] = r if on_rows or not rows else r ^ 1
        return lit[sig]

    x.outputs = [build(o) for o in outputs]
    assert outputs == OUTPUT_NAMES, f"unexpected outputs {outputs}"
    return x
