"""Record the final states of Golly's HashLife as fingerprints, the fixtures of `tests/golly.rs`.

For every pattern of the harness's `head2head`, `runlength` and `life-h2h` targets, bgolly runs it
for the given number of generations at the given step and writes the final state as a macrocell;
the harness (`research/target/release/ca-research fingerprint`, an implementation independent of
this library) prints its fingerprint. Run once, from the repository root, with Golly 5.1 installed:

    python ghl/tests/fixtures/record_golly.py

bgolly is taken from $BGOLLY or Golly 5.1's install path; the rule folder is research/rules.
"""
import os, re, shutil, subprocess, sys, time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
PAT = os.path.join(ROOT, "research", "patterns")
RULES = os.path.join(ROOT, "research", "rules") + os.sep
_WIN = r"C:\Program Files\golly-5.1-win-64bit\bgolly.exe"
BGOLLY = os.environ.get("BGOLLY") or (_WIN if os.path.exists(_WIN) else shutil.which("bgolly") or "bgolly")
HARNESS = os.path.join(ROOT, "research", "target", "release", "ca-research")
if os.path.exists(HARNESS + ".exe"):
    HARNESS += ".exe"
OUT = os.path.join(ROOT, "ghl", "tests", "fixtures", "golly.tsv")
TMP = os.path.join(ROOT, "research", "out", "golly", "fixture.mc")

# (pattern relative to research/patterns, generations, log2 of Golly's step)
RUNS = [
    # head2head
    ("misc/Segmented ROM (256k).mc", 1 << 24, 10),
    ("16x16 movie/16x16 movie.mc", 1 << 24, 10),
    ("misc/serial display (32x32).mc", 1 << 24, 10),
    ("misc/BIN2BCD.rle", 1 << 24, 10),
    ("misc/parallel display (16x16).mc", 1 << 24, 10),
    ("misc/efficient_demultiplexer.mc", 1 << 24, 10),
    ("misc/weird designs/cpu.mc", 1 << 24, 10),
    ("32x32 movie/32x32 movie.mc", 65536, 10),
    ("misc/squares.rle", 65536, 10),
    ("replicators/Hutton-replicator.rle", 65536, 12),
    # runlength: the shortest and the longest run of each
    ("advanced counter v2.rle", 16384, 10),
    ("advanced counter v2.rle", 1 << 20, 10),
    ("advanced counter v2.rle", 1 << 24, 10),
    ("misc/chaotic.rle", 16384, 10),
    ("misc/chaotic.rle", 1 << 20, 10),
    ("misc/chaotic.rle", 1 << 24, 10),
    # life-h2h
    ("life/metapixel-galaxy.mc", 1 << 24, 10),
    ("life/Unit-Life-Cell-512x512.rle", 1 << 24, 10),
    ("life/ruler.mc", 1 << 24, 10),
    ("life/wedge-grow.mc", 1 << 24, 10),
    ("life/gotts-dots.mc", 1 << 24, 10),
    ("life/metacatacryst.mc", 1 << 24, 10),
    ("life/catacryst.mc", 1 << 20, 10),
    ("life/mosquito5.mc", 1 << 20, 10),
    ("life/jagged.mc", 1 << 20, 10),
    ("life/totalperiodic.mc", 1 << 20, 10),
    # short runs, for the default test run
    ("life/Unit-Life-Cell-512x512.rle", 4096, 10),
    ("misc/squares.rle", 4096, 10),
    ("misc/chaotic.rle", 4096, 10),
]


def rule_of(path):
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            m = re.match(r"#R\s*(\S+)", line) or re.match(r"x\s*=.*rule\s*=\s*(\S+)", line)
            if m:
                return m.group(1)
            if line.startswith((".", "*", "$")):
                return "B3/S23"
            if i > 1000 or (line.strip() and not line.startswith(("#", "[M2]"))):
                break
    return "Hutton32b"


def algo_of(rule):
    if re.fullmatch(r"(?i)life|b\d*/?s\d*[hv]?|s\d*/?b\d*[hv]?", rule.strip()):
        return "HashLife"
    return "JvN" if rule.lower() == "hutton32" else "RuleLoader"


def main():
    os.makedirs(os.path.dirname(TMP), exist_ok=True)
    lines = ["# Golly 5.1's final states as fingerprints (tests/golly.rs); written by record_golly.py",
             "# pattern\trule\tgenerations\tstep_log2\tpopulation\twidth\theight\thash\tgolly_seconds"]
    for rel, gens, k in RUNS:
        path = os.path.join(PAT, rel)
        rule = rule_of(path)
        if os.path.exists(TMP):
            os.remove(TMP)
        t = time.perf_counter()
        cmd = [BGOLLY, "-q", "-q", "-a", algo_of(rule), "-s", RULES, "-M", "8000", "-m", str(gens), "-i", str(1 << k), "-o", TMP, path]
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        secs = time.perf_counter() - t
        if not os.path.exists(TMP):
            print(f"{rel}: bgolly wrote nothing: {p.stdout} {p.stderr}", file=sys.stderr)
            continue
        f = subprocess.run([HARNESS, "fingerprint", "--pattern", TMP], capture_output=True, text=True, timeout=1800)
        m = re.search(r"population (\d+) bbox (\d+)x(\d+) hash (\S+)", f.stdout + f.stderr)
        if not m:
            print(f"{rel}: no fingerprint: {f.stdout} {f.stderr}", file=sys.stderr)
            continue
        pop, w, h, hsh = m.groups()
        lines.append(f"{rel}\t{rule}\t{gens}\t{k}\t{pop}\t{w}\t{h}\t{hsh}\t{secs:.2f}")
        print(lines[-1], flush=True)
    with open(OUT, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}")


main()
