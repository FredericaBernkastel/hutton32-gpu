"""Head-to-head: Golly HashLife (bgolly) vs the reference engine (`ca-research root`), same pattern,
same step size, same number of generations. Also checks that both end in the same state.

usage: python head2head.py <gens> <k> [--a A] [--timeout S] [--min-time S] [--max-runs N] [--v1] <pattern> ...

Timing is symmetric: both sides are timed without writing any output, each run is repeated until the
runs add up to --min-time seconds (default 1) or --max-runs runs (default 5) were made, and the median
run is reported. Golly is timed with -q -q (no output), its load-only time subtracted;
the engine reports its own high-resolution CPU timer. A separate untimed Golly run writes the final
state, which is compared with the engine's by fingerprint.
--a A sets the engine's base tile (2: 4x4 cells, 3: 8x8 cells). Life-like (B/S) patterns run under
Golly's Life HashLife, Hutton32 under its JvN algorithm, everything else under RuleLoader.
--v1 runs the engine with the rule files as first archived (hutton32-v1 / hutton32b-v1).
bgolly is taken from $BGOLLY (default: Golly 5.1's install path, else bgolly on PATH); the rule
folder is research/rules.
"""
import os, re, shutil, statistics, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
_WIN_BGOLLY = r"C:\Program Files\golly-5.1-win-64bit\bgolly.exe"
BGOLLY = os.environ.get("BGOLLY") or (_WIN_BGOLLY if os.path.exists(_WIN_BGOLLY) else shutil.which("bgolly") or "bgolly")
RULES = os.path.join(HERE, "..", "rules") + os.sep
_EXE = os.path.join(HERE, "..", "target", "release", "ca-research")
ENGINE = _EXE + ".exe" if os.path.exists(_EXE + ".exe") else _EXE
OUT = os.path.join(HERE, "..", "out", "golly")


def rule_of(path):
    """The rule named in the pattern's header: macrocell "#R", or RLE "x = .., rule = ..", which
    can follow a long comment block."""
    with open(path, "r", errors="replace") as f:
        for i, line in enumerate(f):
            m = re.match(r"#R\s*(\S+)", line) or re.match(r"x\s*=.*rule\s*=\s*(\S+)", line)
            if m:
                return m.group(1)
            if line.startswith((".", "*", "$")):
                return "B3/S23"  # a two-state macrocell without a rule line: Golly reads it as Life
            if i > 1000 or (line.strip() and not line.startswith(("#", "[M2]"))):
                break  # past the header
    return "Hutton32b"


def golly_algo(rule):
    """Golly's algorithm for a rule: its Life HashLife for Life-like (B/S) rules, its built-in JvN
    family for Hutton32, RuleLoader (the published table) otherwise."""
    if re.fullmatch(r"(?i)life|b\d*/?s\d*[hv]?|s\d*/?b\d*[hv]?", rule.strip()):
        return "HashLife"
    return "JvN" if rule.lower() == "hutton32" else "RuleLoader"


def timed(cmd, timeout):
    t = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, errors="replace")
        return time.perf_counter() - t, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return None, ""


def repeat(measure, min_time, max_runs):
    """Run `measure()` (seconds or None) until min_time is reached or max_runs; median and count."""
    times = []
    while len(times) < max_runs and sum(times) < min_time:
        t = measure()
        if t is None:
            return None, len(times)
        times.append(t)
    return statistics.median(times), len(times)


def main():
    argv = sys.argv[1:]

    def take(flag, default):
        if flag in argv:
            i = argv.index(flag)
            v = argv[i + 1]
            del argv[i:i + 2]
            return v
        return default

    v1 = "--v1" in argv
    if v1:
        argv.remove("--v1")
    a = take("--a", "2")
    timeout = float(take("--timeout", "300"))
    min_time = float(take("--min-time", "1"))
    max_runs = int(take("--max-runs", "5"))
    gens, k = int(argv[0]), int(argv[1])
    os.makedirs(OUT, exist_ok=True)
    print(f"engine base tile: {1 << int(a)}x{1 << int(a)} cells; median of runs adding up to {min_time:g} s (at most {max_runs})")
    print(f"{'pattern':<34} {'golly s':>10} {'runs':>4} {'engine s':>10} {'runs':>4} {'golly gens/s':>13} {'engine gens/s':>14}  same-state")
    for pat in argv[2:]:
        rule = rule_of(pat)
        algo = golly_algo(rule)
        base = [BGOLLY, "-q", "-q", "-a", algo, "-s", RULES, "-M", "8000"]
        load, _ = timed(base + ["-m", "0", pat], timeout)

        def golly_once():
            t, _ = timed(base + ["-m", str(gens), "-i", str(1 << k), pat], timeout)
            return None if t is None else max(t - load, 1e-9)

        erule = ["--rule", rule.lower() + "-v1"] if v1 else []
        ecmd = [ENGINE, "root", "--pattern", pat, "--k", str(k), "--gens", str(gens), "--a", a] + erule
        eout_last = [""]

        def engine_once():
            t, out = timed(ecmd, timeout)
            if t is None:
                return None
            eout_last[0] = out
            cpu = re.findall(r"cpu\s+([\d.]+)s", out)
            return float(cpu[-1]) if cpu else None

        tg, ng = repeat(golly_once, min_time, max_runs)
        te, ne = repeat(engine_once, min_time, max_runs)
        same = "?"
        if tg is not None and te is not None:
            out_mc = os.path.join(OUT, "h2h.mc")
            if os.path.exists(out_mc):
                os.remove(out_mc)
            timed([BGOLLY, "-q", "-q", "-a", algo, "-s", RULES, "-M", "8000", "-m", str(gens), "-i", str(1 << k), "-o", out_mc, pat], timeout)
            _, fout = timed([ENGINE, "fingerprint", "--pattern", out_mc], timeout)
            gfp = re.search(r"population (\d+) bbox (\S+) hash (\S+)", fout)
            efp = re.search(r"final: gen \d+ population (\d+) bbox (\S+) hash (\S+)", eout_last[0])
            same = "yes" if gfp and efp and gfp.groups() == efp.groups() else "NO"
        name = os.path.basename(pat)[:33]
        gs = f"{tg:10.4f}" if tg is not None else "   timeout"
        es = f"{te:10.4f}" if te is not None else "   timeout"
        gr = f"{gens / tg:13.0f}" if tg is not None else "-".rjust(13)
        er = f"{gens / te:14.0f}" if te is not None else "-".rjust(14)
        print(f"{name:<34} {gs} {ng:4d} {es} {ne:4d} {gr} {er}  {same}", flush=True)


main()
