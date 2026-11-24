"""Head-to-head: Golly HashLife (bgolly) vs the reference engine (`ca-research root`), same pattern,
same step size, same number of generations. Also checks that both end in the same state.

usage: python head2head.py <gens> <k> [--a A] [--timeout S] <pattern> [<pattern> ...]
"""
import os, re, subprocess, sys, time

BGOLLY = r"C:\Program Files\golly-5.1-win-64bit\bgolly.exe"
RULES = r"D:/Program Files/golly-3.2-win/Rules/"
HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.join(HERE, "..", "target", "release", "ca-research.exe")
OUT = os.path.join(HERE, "..", "out", "golly")


def rule_of(path):
    with open(path, "r", errors="replace") as f:
        head = f.read(4000)
    m = re.search(r"#R\s*(\S+)", head) or re.search(r"rule\s*=\s*(\S+)", head)
    return m.group(1) if m else "Hutton32b"


def timed(cmd, timeout):
    t = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return time.perf_counter() - t, p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return None, ""


def main():
    argv = sys.argv[1:]

    def take(flag, default):
        if flag in argv:
            i = argv.index(flag)
            v = argv[i + 1]
            del argv[i:i + 2]
            return v
        return default

    a = take("--a", "2")
    timeout = float(take("--timeout", "300"))
    gens, k = int(argv[0]), int(argv[1])
    os.makedirs(OUT, exist_ok=True)
    print(f"{'pattern':<34} {'golly s':>9} {'engine s':>9} {'golly gens/s':>13} {'engine gens/s':>14}  same-state")
    for pat in argv[2:]:
        rule = rule_of(pat)
        algo = "JvN" if rule.lower() == "hutton32" else "RuleLoader"
        base = [BGOLLY, "-q", "-q", "-a", algo, "-s", RULES, "-M", "8000"]
        load, _ = timed(base + ["-m", "0", pat], timeout)
        out_mc = os.path.join(OUT, "h2h.mc")
        if os.path.exists(out_mc):
            os.remove(out_mc)
        tg, _ = timed(base + ["-m", str(gens), "-i", str(1 << k), "-o", out_mc, pat], timeout)
        te, eout = timed([ENGINE, "root", "--pattern", pat, "--k", str(k), "--gens", str(gens), "--a", a], timeout)
        same = "?"
        if tg is not None and te is not None:
            _, fout = timed([ENGINE, "fingerprint", "--pattern", out_mc], timeout)
            gfp = re.search(r"population (\d+) bbox (\S+) hash (\S+)", fout)
            efp = re.search(r"final: gen \d+ population (\d+) bbox (\S+) hash (\S+)", eout)
            same = "yes" if gfp and efp and gfp.groups() == efp.groups() else "NO"
        # engine cpu time excludes process start + LUT load; use its own timer for fairness too
        ecpu = re.findall(r"cpu\s+([\d.]+)s", eout)
        name = os.path.basename(pat)[:33]
        gs = f"{tg - load:9.3f}" if tg is not None else "  timeout"
        es = f"{float(ecpu[-1]):9.3f}" if (te is not None and ecpu) else "  timeout"
        gr = f"{gens / max(tg - load, 1e-6):13.0f}" if tg is not None else "-".rjust(13)
        er = f"{gens / max(float(ecpu[-1]), 1e-6):14.0f}" if (te is not None and ecpu) else "-".rjust(14)
        print(f"{name:<34} {gs} {es} {gr} {er}  {same}", flush=True)


main()
