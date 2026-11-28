"""Acceptance test of the inspector (roadmap Section 8.4): the overview JSON written through the
library's public API (`cargo run --release -p ghl --example overview`) against the harness's
(`ca-research overview`), key by key.

    python ghl/tests/fixtures/compare_overview.py [harness.json] [ghl.json]

Exit status 0 when every key of the figure is equal.
"""
import json, os, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
KEYS = ["palette", "patterns", "structures", "structures_life", "sweep", "table", "table_life", "learning", "waves", "controller"]


def diff(a, b, path, out, limit=12):
    if len(out) >= limit:
        return
    if type(a) != type(b):
        out.append(f"{path}: type {type(a).__name__} != {type(b).__name__}")
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}.{k}: only in {'harness' if k in a else 'ghl'}")
            else:
                diff(a[k], b[k], f"{path}.{k}", out, limit)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, f"{path}[{i}]", out, limit)
    elif a != b:
        sa, sb = str(a), str(b)
        out.append(f"{path}: {sa[:60]} != {sb[:60]}")


def main():
    h = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "research", "out", "overview.json")
    g = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "research", "out", "overview_ghl.json")
    a, b = json.load(open(h)), json.load(open(g))
    bad = 0
    for k in KEYS:
        if k not in a or k not in b:
            print(f"{k:<16} missing in {'harness' if k not in a else 'ghl'}")
            bad += 1
            continue
        out = []
        diff(a[k], b[k], k, out)
        print(f"{k:<16} {'equal' if not out else 'DIFFERS'}")
        for line in out:
            print("    " + line)
        bad += bool(out)
    print("all keys equal" if bad == 0 else f"{bad} keys differ")
    sys.exit(1 if bad else 0)


main()
