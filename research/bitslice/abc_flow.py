"""Driver for ABC (through yowasp-yosys, which bundles Yosys + ABC as WebAssembly).

`run_abc(xag, script)` writes the network as structural Verilog, lets Yosys hand it to ABC with
the given ABC script (which must end by mapping to the unit-cost library below), reads the
result back, and verifies it exhaustively against the truth table before returning it.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from xag import XAG, OUTPUT_NAMES, INPUT_NAMES, Sim, read_blif_to_xag, map_xag, HERE, OUT_DIR

WORK = HERE / "work"          # scratch files of Yosys/ABC (git-ignored)
YOSYS = OUT_DIR / "bitslice-venv" / "Scripts" / "yowasp-yosys.exe"

# Unit-cost library: every WGSL operator (~ & | ^) costs 1.
GENLIB = """\
GATE ZERO 0 Y=CONST0;
GATE ONE  0 Y=CONST1;
GATE BUF  0 Y=A;             PIN * NONINV  1 999 1 0 1 0
GATE NOT  1 Y=!A;            PIN * INV     1 999 1 0 1 0
GATE AND  1 Y=A*B;           PIN * NONINV  1 999 1 0 1 0
GATE OR   1 Y=A+B;           PIN * NONINV  1 999 1 0 1 0
GATE XOR  1 Y=(A*!B)+(!A*B); PIN * UNKNOWN 1 999 1 0 1 0
"""

# abc.rc aliases, spelled out (Yosys does not load abc.rc)
RESYN2 = "balance; rewrite; refactor; balance; rewrite; rewrite -z; balance; refactor -z; rewrite -z; balance"
RESYN2RS = (
    "balance; resub -K 6; rewrite; resub -K 6 -N 2; refactor; resub -K 8; balance; resub -K 8 -N 2; "
    "rewrite; resub -K 10; rewrite -z; resub -K 10 -N 2; balance; resub -K 12; refactor -z; "
    "resub -K 12 -N 2; rewrite -z; balance"
)
COMPRESS2RS = (
    "balance -l; resub -K 6 -l; rewrite -l; resub -K 6 -N 2 -l; refactor -l; resub -K 8 -l; balance -l; "
    "resub -K 8 -N 2 -l; rewrite -l; resub -K 10 -l; rewrite -z -l; resub -K 10 -N 2 -l; balance -l; "
    "resub -K 12 -l; refactor -z -l; resub -K 12 -N 2 -l; rewrite -z -l; balance -l"
)
MAP = "read_library unit.genlib; map -a"


def to_verilog(x: XAG, outputs=None) -> str:
    outputs = x.outputs if outputs is None else outputs
    live = x.live_nodes(outputs)
    nm = {}
    lines = ["module rule(" + ", ".join(f"input {n}" for n in INPUT_NAMES) + ", "
             + ", ".join(f"output {n}" for n in OUTPUT_NAMES) + ");"]
    for n in live:
        if x.op[n] == "in":
            nm[n] = x.in_name[n]

    def ref(l):
        if l >> 1 == 0:
            return "1'b1" if l & 1 else "1'b0"
        return ("~" if l & 1 else "") + nm[l >> 1]

    for n in live:
        if x.op[n] in ("and", "xor"):
            nm[n] = f"g{n}"
            sym = "&" if x.op[n] == "and" else "^"
            lines.append(f"  wire g{n} = {ref(x.a[n])} {sym} {ref(x.b[n])};")
    for l, name in zip(outputs, OUTPUT_NAMES):
        lines.append(f"  assign {name} = {ref(l)};")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def run_abc(x: XAG, script: str, tag: str, timeout: int = 3600, verify: bool = True) -> XAG:
    """Run an ABC script on the network; returns the (verified) result as an XAG."""
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "unit.genlib").write_text(GENLIB)
    (WORK / f"{tag}.v").write_text(to_verilog(x))
    (WORK / f"{tag}.abc").write_text(script.replace("; ", "\n") + "\n")
    out_blif = WORK / f"{tag}_out.blif"
    if out_blif.exists():
        out_blif.unlink()
    ys = (
        f"read_verilog {tag}.v\nhierarchy -top rule\nproc\ntechmap\n"
        f"abc -script {tag}.abc\nopt_clean -purge\nwrite_blif -icells {tag}_out.blif\n"
    )
    (WORK / f"{tag}.ys").write_text(ys)
    t = time.time()
    r = subprocess.run(
        [str(YOSYS), "-q", "-l", f"{tag}.log", f"{tag}.ys"],
        cwd=WORK, capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0 or not out_blif.exists():
        raise RuntimeError(f"yosys/abc failed for {tag}: rc={r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")
    y = read_blif_to_xag(out_blif.read_text())
    if verify:
        planes = y.simulate(y.outputs)
        ref = Sim.lut_planes()
        bad = [Sim.popcount(planes[k] ^ ref[k]) for k in range(5)]
        if any(bad):
            raise RuntimeError(f"ABC result for {tag} is WRONG: mismatches per bit {bad}")
    y.abc_seconds = time.time() - t
    return y


def cost(x: XAG, restarts: int = 2):
    nl = map_xag(x, restarts=restarts)
    return nl.counts()["total"], nl


if __name__ == "__main__":
    import rule_circuit

    x = XAG()
    rule_circuit.build(x)
    print("start:", x.count(), "mapped", cost(x)[0])
    scripts = {
        "map": f"strash; {MAP}",
        "dc2": f"strash; dc2; dc2; dc2; {MAP}",
        "resyn2": f"strash; {RESYN2}; {RESYN2}; {RESYN2}; {MAP}",
    }
    for tag, s in scripts.items():
        y = run_abc(x, s, tag)
        c, nl = cost(y)
        print(f"{tag:10s} xag={y.count()} mapped={c} {nl.counts()} depth={nl.depth()} ({y.abc_seconds:.1f}s)")
