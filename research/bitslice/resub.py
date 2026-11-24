"""Global resubstitution with observability don't-cares, specialised for this one function.

ABC's resub/mfs work on windows; here the whole circuit has only 25 inputs, so every candidate
can be checked against the complete truth table.  For a target node t:

  * care(t)   = the neighbourhoods on which flipping t changes some output (computed on a sample
                of neighbourhoods; the sample grows whenever it lets a wrong candidate through);
  * candidates = expressions with 0, 1 or 2 new gates over nodes outside the fanout cone of t
                (existing node / AND,OR,XOR of two / small trees of three) that agree with t on
                care(t) in the sample;
  * a candidate is accepted only if it makes the network smaller AND the rebuilt network
    reproduces all 2^25 entries of the truth table.

Cost function: number of AND/XOR nodes of the XAG (inverters are placed afterwards).
"""
from __future__ import annotations

import random
import sys
import time

import numpy as np

from xag import XAG, INPUT_NAMES, POS, Sim


# ------------------------------------------------------------------------------------------------
def compact(x: XAG, outputs=None) -> XAG:
    """Copy of the live part of `x` (nodes renumbered in topological order)."""
    return rebuild(x, x.outputs if outputs is None else outputs, None, None)


def rebuild(x: XAG, outputs, target, expr) -> XAG:
    """Copy `x`, replacing node `target` by `expr` (a tree of ('lit', l) / ('and', a, b) /
    ('xor', a, b) / ('not', a) over literals of `x`)."""
    y = XAG()
    for name in INPUT_NAMES:
        y.inp(name)
    memo = {0: 0}

    def ev(e):
        k = e[0]
        if k == "lit":
            return node(e[1] >> 1) ^ (e[1] & 1)
        if k == "not":
            return ev(e[1]) ^ 1
        if k == "and":
            return y.AND(ev(e[1]), ev(e[2]))
        return y.XOR(ev(e[1]), ev(e[2]))

    def node(n):
        r = memo.get(n)
        if r is not None:
            return r
        if n == target:
            r = ev(expr)
        else:
            o = x.op[n]
            if o == "in":
                r = y.inp(x.in_name[n])
            else:
                a, b = x.a[n], x.b[n]
                ra = node(a >> 1) ^ (a & 1)
                rb = node(b >> 1) ^ (b & 1)
                r = y.AND(ra, rb) if o == "and" else y.XOR(ra, rb)
        memo[n] = r
        return r

    y.outputs = [node(l >> 1) ^ (l & 1) for l in outputs]
    y.memo = memo  # old node -> new literal
    return y


def size(x: XAG) -> int:
    c = x.count()
    return c["and"] + c["xor"]


def full_check(x: XAG):
    """None if `x` reproduces the truth table, else an array of mismatching LUT indices."""
    planes = x.simulate(x.outputs)
    ref = Sim.lut_planes()
    diff = planes[0] ^ ref[0]
    for k in range(1, 5):
        diff |= planes[k] ^ ref[k]
    if not diff.any():
        return None
    return np.flatnonzero(np.unpackbits(diff.view(np.uint8), bitorder="little"))


# ------------------------------------------------------------------------------------------------
class Engine:
    def __init__(self, x: XAG, seed: int = 1, n_random: int = 6000, per_case: int = 3, log=print):
        self.rng = random.Random(seed)
        self.log = log
        self.x = compact(x)
        assert full_check(self.x) is None, "starting network is not equivalent to the truth table"
        self.points: list[int] = []
        self._init_points(n_random, per_case)
        self.prepare()
        self.stats = {"accepted": 0, "rejected_full": 0, "points_added": 0}

    # -- sample points ----------------------------------------------------------------------
    def _init_points(self, n_random, per_case):
        """Stratified sample: for every centre state and every combination of the neighbour
        classes that centre can tell apart, a few random concrete neighbourhoods."""
        lut5 = Sim.lut().reshape(32, 32, 32, 32, 32)  # [c][n][e][s][w]
        pts = set()
        rng = self.rng
        for c in range(32):
            f = lut5[c]
            groups = []
            for ax in range(4):
                m = np.moveaxis(f, ax, 0).reshape(32, -1)
                cls = {}
                for v in range(32):
                    cls.setdefault(m[v].tobytes(), []).append(v)
                groups.append(list(cls.values()))
            import itertools

            for combo in itertools.product(*groups):
                for _ in range(per_case):
                    n, e, s, w = (rng.choice(g) for g in combo)
                    pts.add(w | s << 5 | e << 10 | n << 15 | c << 20)
        for _ in range(n_random):
            pts.add(rng.getrandbits(25))
        self.points = sorted(pts)

    def add_points(self, idx):
        new = [int(i) for i in idx if int(i) not in self._pointset]
        self.points += new
        self._pointset.update(new)
        self.stats["points_added"] += len(new)
        self._input_sigs()

    def _input_sigs(self):
        pts = np.array(self.points, dtype=np.uint32)
        self.K = len(self.points)
        self.MASK = (1 << self.K) - 1
        self.in_sig = {}
        for f, p in enumerate(POS):
            for k in range(5):
                bits = ((pts >> (5 * f + k)) & 1).astype(np.uint8)
                self.in_sig[f"{p}{k}"] = int.from_bytes(np.packbits(bits, bitorder="little").tobytes(), "little")
        self._pointset = set(self.points)

    # -- per-network data ---------------------------------------------------------------------
    def prepare(self):
        """(Re)compute everything that depends on the current network."""
        if not hasattr(self, "in_sig"):
            self._input_sigs()
        x = self.x
        self.live = x.live_nodes()
        self.gates = [n for n in self.live if x.op[n] in ("and", "xor")]
        M = self.MASK
        sig = {0: 0}
        for n in self.live:
            o = x.op[n]
            if o == "in":
                sig[n] = self.in_sig[x.in_name[n]]
            elif o == "and":
                a, b = x.a[n], x.b[n]
                sig[n] = (sig[a >> 1] ^ (M if a & 1 else 0)) & (sig[b >> 1] ^ (M if b & 1 else 0))
            elif o == "xor":
                sig[n] = sig[x.a[n] >> 1] ^ sig[x.b[n] >> 1]
        self.sig = sig
        self.fanout = {n: [] for n in self.live}
        self.refs = {n: 0 for n in self.live}
        for n in self.gates:
            for l in (x.a[n], x.b[n]):
                self.fanout[l >> 1].append(n)
                self.refs[l >> 1] += 1
        for l in x.outputs:
            self.refs[l >> 1] += 1
        self.out_nodes = {l >> 1 for l in x.outputs}

    def lit_sig(self, l):
        return self.sig[l >> 1] ^ (self.MASK if l & 1 else 0)

    def mffc(self, t) -> int:
        """Number of gates that disappear when t is removed."""
        x = self.x
        refs = dict(self.refs)
        cnt = 0
        stack = [t]
        while stack:
            n = stack.pop()
            if x.op[n] not in ("and", "xor"):
                continue
            cnt += 1
            for l in (x.a[n], x.b[n]):
                m = l >> 1
                refs[m] -= 1
                if refs[m] == 0:
                    stack.append(m)
        return cnt

    def care(self, t):
        """(care mask over the sample, set of nodes in the fanout cone of t)."""
        x, sig, M = self.x, self.sig, self.MASK
        val = {t: sig[t] ^ M}
        for n in self.gates:
            if n <= t:
                continue
            a, b = x.a[n], x.b[n]
            na, nb = a >> 1, b >> 1
            if na in val or nb in val:
                va = val.get(na, sig[na]) ^ (M if a & 1 else 0)
                vb = val.get(nb, sig[nb]) ^ (M if b & 1 else 0)
                val[n] = (va & vb) if x.op[n] == "and" else (va ^ vb)
        c = 0
        for l in x.outputs:
            n = l >> 1
            if n in val:
                c |= val[n] ^ sig[n]
        return c, set(val.keys())

    # -- candidate generation -----------------------------------------------------------------
    def candidates(self, t, budget, limit_u=48, effort=2):
        """Yield expressions equal to t on its care set (in the sample), using at most `budget`
        new gates, cheapest first."""
        C, tfo = self.care(t)
        M = self.MASK
        T = self.sig[t]
        if C == 0:
            yield ("lit", 0)  # never observable
            return
        divs = [n for n in self.live if n not in tfo and n != 0]
        S = {n: self.sig[n] for n in divs}
        # dictionary: masked signature -> literal
        table = {}
        for n in divs:
            v = S[n] & C
            table.setdefault(v, 2 * n)
            table.setdefault(v ^ C, 2 * n + 1)
        TC = T & C
        # ---- 0 new gates ----
        if TC == 0:
            yield ("lit", 0)
        if TC == C:
            yield ("lit", 1)
        l = table.get(TC)
        if l is not None:
            yield ("lit", l)
        if budget < 1:
            return
        # ---- 1 new gate ----
        # xor
        for n in divs:
            l = table.get((T ^ S[n]) & C)
            if l is not None and (l >> 1) != n:
                yield ("xor", ("lit", 2 * n), ("lit", l))
                break
        # and / or (or = complemented and of complements)
        found_and = {}
        for neg in (0, 1):
            Tn = (T ^ (M if neg else 0)) & C     # function to realise as an AND of two literals
            off = C & ~Tn
            U = []
            for n in divs:
                s = S[n]
                if Tn & ~s == 0:
                    U.append((2 * n, s & off))
                if Tn & s == 0:
                    U.append((2 * n + 1, (s ^ M) & off))
            U.sort(key=lambda p: p[1].bit_count())
            U = U[:limit_u]
            found_and[neg] = U
            done = False
            for i in range(len(U)):
                la, ra = U[i]
                for j in range(i + 1, len(U)):
                    lb, rb = U[j]
                    if ra & rb == 0:
                        e = ("and", ("lit", la), ("lit", lb))
                        yield ("not", e) if neg else e
                        done = True
                        break
                if done:
                    break
        if budget < 2:
            return
        # ---- 2 new gates ----
        # xor(a, xor(b, c))
        if effort >= 2:
            dl = divs
            got = False
            for i, a in enumerate(dl):
                ta = T ^ S[a]
                for b in dl[i + 1:]:
                    l = table.get((ta ^ S[b]) & C)
                    if l is not None and (l >> 1) != a and (l >> 1) != b:
                        yield ("xor", ("lit", 2 * a), ("xor", ("lit", 2 * b), ("lit", l)))
                        got = True
                        break
                if got:
                    break
        for neg in (0, 1):
            Tn = (T ^ (M if neg else 0)) & C
            off = C & ~Tn
            U = found_and[neg]
            wrap = (lambda e: ("not", e)) if neg else (lambda e: e)
            # and(a, b, c)
            done = False
            for i in range(len(U)):
                la, ra = U[i]
                for j in range(i + 1, len(U)):
                    lb, rb = U[j]
                    rab = ra & rb
                    for k in range(j + 1, len(U)):
                        lc, rc = U[k]
                        if rab & rc == 0:
                            yield wrap(("and", ("lit", la), ("and", ("lit", lb), ("lit", lc))))
                            done = True
                            break
                    if done:
                        break
                if done:
                    break
            # and(a, xor(b, c)) and and(a, or(b, c)): g must equal Tn on C & a
            cnt = 0
            for la, ra in U[:24]:
                if ra == 0:
                    continue
                sa = self.lit_sig(la)
                C2 = C & sa
                T2 = Tn & C2
                tab2 = {}
                for n in divs:
                    v = S[n] & C2
                    tab2.setdefault(v, 2 * n)
                    tab2.setdefault(v ^ C2, 2 * n + 1)
                hit = False
                for n in divs:
                    if n == la >> 1:
                        continue
                    l = tab2.get((Tn ^ S[n]) & C2)
                    if l is not None and (l >> 1) != n and (l >> 1) != (la >> 1):
                        yield wrap(("and", ("lit", la), ("xor", ("lit", 2 * n), ("lit", l))))
                        hit = True
                        break
                # or(b, c): b, c inside T2 on C2, together covering T2
                V = []
                off2 = C2 & ~Tn
                for n in divs:
                    s = S[n]
                    if s & off2 == 0 and s & T2:
                        V.append((2 * n, s & T2))
                    sn = s ^ M
                    if sn & off2 == 0 and sn & T2:
                        V.append((2 * n + 1, sn & T2))
                V.sort(key=lambda p: -p[1].bit_count())
                V = V[:limit_u]
                done = False
                for i in range(len(V)):
                    lb, cb = V[i]
                    need = T2 & ~cb
                    for j in range(i + 1, len(V)):
                        lc, cc = V[j]
                        if need & ~cc == 0:
                            e = ("not", ("and", ("lit", lb ^ 1), ("lit", lc ^ 1)))
                            yield wrap(("and", ("lit", la), e))
                            done = True
                            break
                    if done:
                        break
                cnt += 1
        # xor(a, and(b, c)) with any polarities (covers xor(a, or(b, c)) too)
        if effort >= 2:
            hits = 0
            for a in divs:
                Ta = T ^ S[a]
                for neg in (0, 1):
                    Tn = (Ta ^ (M if neg else 0)) & C
                    if Tn == 0:
                        continue
                    off = C & ~Tn
                    U = []
                    for n in divs:
                        if n == a:
                            continue
                        s = S[n]
                        if Tn & ~s == 0:
                            U.append((2 * n, s & off))
                        elif Tn & s == 0:
                            U.append((2 * n + 1, (s ^ M) & off))
                    if len(U) < 2:
                        continue
                    U.sort(key=lambda p: p[1].bit_count())
                    U = U[:16]
                    done = False
                    for i in range(len(U)):
                        la, ra = U[i]
                        for j in range(i + 1, len(U)):
                            lb, rb = U[j]
                            if ra & rb == 0:
                                e = ("and", ("lit", la), ("lit", lb))
                                yield ("xor", ("lit", 2 * a + neg), e)
                                done = True
                                hits += 1
                                break
                        if done:
                            break
                if hits >= 3:
                    break

    # -- main loop ----------------------------------------------------------------------------
    def try_target(self, t, effort=2, allow_equal=False) -> bool:
        cur = len(self.gates)
        m = self.mffc(t)
        budget = min(2, m - 1 + (1 if allow_equal else 0))
        if budget < 0:
            return False
        for attempt in range(6):
            best = None
            for e in self.candidates(t, budget, effort=effort):
                y = rebuild(self.x, self.x.outputs, t, e)
                s = size(y)
                if s < cur or (allow_equal and s == cur and e[0] != "lit"):
                    if best is None or s < best[0]:
                        best = (s, e, y)
                        if s <= cur - m:   # cannot do better than freeing the whole cone
                            break
            if best is None:
                return False
            s, e, y = best
            bad = full_check(y)
            if bad is None:
                self.x = y
                self.prepare()
                self.stats["accepted"] += 1
                return True
            # the sample let a wrong candidate through: refine it and look again
            self.stats["rejected_full"] += 1
            pick = bad if bad.size <= 64 else np.array(self.rng.sample(list(map(int, bad)), 64))
            self.add_points(pick)
            self.prepare()
        return False

    def sweep(self, effort=2, allow_equal=False, time_limit=None) -> int:
        """One pass over all nodes, outputs first. Returns the number of accepted moves."""
        t0 = time.time()
        acc = 0
        tried = set()
        while True:
            todo = [n for n in reversed(self.gates) if n not in tried]
            if not todo:
                break
            t = todo[0]
            tried.add(t)
            if time_limit and time.time() - t0 > time_limit:
                break
            before = len(self.gates)
            if self.try_target(t, effort=effort, allow_equal=allow_equal):
                acc += 1
                self.log(f"    resub: {before} -> {len(self.gates)} gates "
                         f"(sample {self.K}, rejected {self.stats['rejected_full']}, {time.time() - t0:.0f}s)")
                # node ids changed: carry the set of already-visited nodes over
                memo = self.x.memo
                tried = {memo[n] >> 1 for n in tried if n in memo}
        return acc


def optimize(x: XAG, seed=1, effort=2, time_limit=None, log=print) -> XAG:
    eng = Engine(x, seed=seed, log=log)
    log(f"  resub start: {len(eng.gates)} gates, sample {eng.K} points")
    eng.sweep(effort=effort, time_limit=time_limit)
    log(f"  resub end: {len(eng.gates)} gates, stats {eng.stats}")
    return eng.x


if __name__ == "__main__":
    import rule_circuit
    from xag import map_xag_exact

    x = XAG()
    rule_circuit.build(x)
    y = optimize(x, time_limit=float(sys.argv[1]) if len(sys.argv) > 1 else 600)
    nl = map_xag_exact(y)
    print(y.count(), nl.counts(), nl.verify())
