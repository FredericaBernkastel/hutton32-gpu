"""Exhaustive check against all 2^25 entries of research/out/lut_table_Hutton32b.bin.

usage: python verify.py                      check circuit.netlist, step_planes.wgsl, step_planes.rs
       python verify.py <file> ...           check the given .netlist / .wgsl / .rs files

Each file is parsed on its own (the WGSL and Rust files are read as text: one `let` per gate),
evaluated bit-parallel on every neighbourhood (index = w | s<<5 | e<<10 | n<<15 | c<<20), and
compared with the table.  A self-test first confirms that the check can fail: a copy of the
netlist with one gate changed must be reported as wrong.

Exit status 0 only if every file has zero mismatches.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

from xag import Netlist, Sim, HERE, INPUT_NAMES

LET = re.compile(r"^\s*let\s+(\w+)\s*=\s*(.+?);\s*$")
UN = re.compile(r"^[~!]\s*(\w+)$")
BIN = re.compile(r"^(\w+)\s*([&|^])\s*(\w+)$")
OPS = {"&": "AND", "|": "OR", "^": "XOR"}


def parse_code(text: str, rust: bool) -> Netlist:
    """Read a generated step_planes function back into a Netlist."""
    nl = Netlist()
    assert "step_planes" in text
    ret = None
    for raw in text.splitlines():
        line = raw.split("//")[0].rstrip()
        if not line.strip():
            continue
        if not rust and "fn step_planes" not in line:
            line = re.sub(r"\b([cnesw])\[(\d)\]", r"\g<1>\g<2>", line)   # c[0] -> c0
        if rust:
            m = re.match(r"^\s*let \[(.+)\] = ([cnesw]);\s*$", line)
            if m:
                names = [t.strip() for t in m.group(1).split(",")]
                assert names == [f"{m.group(2)}{k}" for k in range(5)], line
                continue
            m = re.match(r"^\s*\[(.+)\]\s*$", line)
            if m:
                ret = [t.strip() for t in m.group(1).split(",")]
                continue
        else:
            m = re.match(r"^\s*return array<u32,5>\((.+)\);\s*$", line)
            if m:
                ret = [t.strip() for t in m.group(1).split(",")]
                continue
            if re.match(r"^\s*(let [cnesw]\d = [cnesw]\[\d\];\s*)+$", line):
                for p, k, q, j in re.findall(r"let ([cnesw])(\d) = ([cnesw])\[(\d)\];", line):
                    assert p == q and k == j, line
                continue
        m = LET.match(line)
        if m:
            name, rhs = m.group(1), m.group(2).strip()
            u = UN.match(rhs)
            b = BIN.match(rhs)
            if u:
                nl.gates.append((name, "NOT", u.group(1), None))
            elif b:
                nl.gates.append((name, OPS[b.group(2)], b.group(1), b.group(3)))
            else:
                raise ValueError(f"unrecognised statement: {line}")
            continue
        if re.match(r"^\s*(fn|pub fn) step_planes\(.*\{\s*$", line) or line.strip() in ("}",) or line.strip().startswith("#["):
            continue
        raise ValueError(f"unrecognised line: {line}")
    assert ret is not None and len(ret) == 5, "return statement not found"
    const = {"0u": "0", "0xffffffffu": "1", "0u64": "0", "!0u64": "1"}
    nl.outputs = {f"o{k}": const.get(r, r) for k, r in enumerate(ret)}
    # re-validate ordering / definitions through the text format
    return Netlist.from_text(nl.to_text())


def load(path: Path) -> Netlist:
    text = path.read_text()
    if path.suffix == ".wgsl":
        return parse_code(text, rust=False)
    if path.suffix == ".rs":
        return parse_code(text, rust=True)
    return Netlist.from_text(text)


def check(path: Path) -> bool:
    nl = load(path)
    assert nl.inputs == INPUT_NAMES, "unexpected input list"
    bad = nl.verify()
    c = nl.counts()
    ok = not any(bad)
    print(f"{path.name}: {c['total']} gates (AND {c['AND']}, OR {c['OR']}, XOR {c['XOR']}, NOT {c['NOT']}), "
          f"depth {nl.depth()} ({nl.depth(False)} without NOT): "
          f"mismatches per output bit {bad} of {1 << 25} entries -> {'PASS' if ok else 'FAIL'}")
    return ok


def self_test(path: Path) -> None:
    """The check must be able to fail: corrupt one gate and expect mismatches."""
    nl = load(path)
    for i in (len(nl.gates) // 2, len(nl.gates) - 1):
        name, op, a, b = nl.gates[i]
        if op == "NOT":
            continue
        broken = Netlist(nl.inputs)
        broken.gates = list(nl.gates)
        broken.gates[i] = (name, "XOR" if op != "XOR" else "AND", a, b)
        broken.outputs = dict(nl.outputs)
        bad = broken.verify()
        assert any(bad), "self-test failed: a corrupted netlist passed the check"
        print(f"self-test: gate {name} {op}->{broken.gates[i][1]} gives {sum(bad)} bit mismatches (detected)")
        return
    raise AssertionError("no gate to corrupt")


def main():
    files = [Path(a) for a in sys.argv[1:]] or [
        HERE / "circuit.netlist", HERE / "step_planes.wgsl", HERE / "step_planes.rs"]
    assert int(Sim.lut().size) == 1 << 25
    self_test(files[0])
    ok = True
    for f in files:
        ok &= check(f)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
