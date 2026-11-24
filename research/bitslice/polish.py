"""Last step of the pipeline: redo the NOT/AND/OR/XOR mapping of circuit.netlist with a long
MaxSAT time limit (inverter placement), and rewrite the artifacts if the result is smaller.

usage: python polish.py [seconds=150]
"""
from __future__ import annotations

import subprocess
import sys

from xag import HERE, Netlist, map_xag_exact


def main():
    limit = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
    path = HERE / "circuit.netlist"
    text = path.read_text()
    old = Netlist.from_text(text)
    header = [l[2:] for l in text.splitlines() if l.startswith("# ")]
    x = old.to_xag()
    mapped = map_xag_exact(x, time_limit=limit)
    proven = mapped.polarity_optimal
    nl = mapped.renamed()
    bad = nl.verify()
    assert not any(bad), bad
    c_old, c = old.counts(), nl.counts()
    print(f"before: {c_old}  depth {old.depth()}")
    print(f"after : {c}  depth {nl.depth()}  inverter placement {'proven optimal' if proven else 'NOT proven optimal'}")
    if c["total"] < c_old["total"] or (c["total"] == c_old["total"] and proven):
        note = ("Inverter placement proven optimal for this AND/XOR structure (MaxSAT)." if proven
                else "Inverter placement: best found by MaxSAT within the time limit (not proven optimal).")
        header = [h for h in header if not h.startswith("Inverter placement")] + [note]
        path.write_text(nl.to_text("\n".join(header)), newline="\n")
        subprocess.run([sys.executable, str(HERE / "emit.py")], check=True)
        status = (HERE / "STATUS.txt").read_text().strip()
        src = status.split("source: ", 1)[1] if "source: " in status else "optimize.py"
        (HERE / "STATUS.txt").write_text(
            f"step_planes.wgsl: {c['total']} gates (AND {c['AND']}, OR {c['OR']}, XOR {c['XOR']}, NOT {c['NOT']}), "
            f"depth {nl.depth()}; exhaustive check vs lut_table_Hutton32b.bin: 0 mismatches of 2^25; "
            f"source: {src.replace(' + MaxSAT polish', '')} + MaxSAT polish\n", newline="\n")
        print("artifacts rewritten")
    else:
        print("no improvement; artifacts left as they are")


if __name__ == "__main__":
    main()
