"""Time bgolly (HashLife) on a pattern for a fixed number of generations at several step sizes.

usage: python time_golly.py <pattern> <gens> <step> [<step> ...]
         [--algo RuleLoader|JvN|RuleTree] [--maxmem MB] [--cells N] [--timeout S] [--rules DIR]
Prints wall-clock seconds (minus load-only time), generations/s, cell-updates/s (for --cells N)
and the number of garbage collections Golly reported.
"""
import subprocess, sys, time

BGOLLY = r"C:\Program Files\golly-5.1-win-64bit\bgolly.exe"
RULES = r"D:/Program Files/golly-3.2-win/Rules/"


def run(args, timeout):
    t = time.perf_counter()
    try:
        p = subprocess.run([BGOLLY] + args, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        return None, out
    dt = time.perf_counter() - t
    if p.returncode != 0:
        print("bgolly failed:", p.stdout[-400:], p.stderr[-400:])
    return dt, p.stdout


def main():
    argv = sys.argv[1:]

    def take(flag, default):
        if flag in argv:
            i = argv.index(flag)
            v = argv[i + 1]
            del argv[i:i + 2]
            return v
        return default

    algo = take("--algo", "RuleLoader")
    maxmem = take("--maxmem", "8000")
    cells = float(take("--cells", "0"))
    rules = take("--rules", RULES)
    timeout = float(take("--timeout", "600"))
    pattern, gens = argv[0], int(argv[1])
    steps = [int(s) for s in argv[2:]]
    base = ["-v", "-a", algo, "-s", rules, "-M", maxmem]
    load, _ = run(base + ["-m", "0", pattern], timeout)
    print(f"pattern {pattern}\nload-only {load:.3f}s (algo {algo}, maxmem {maxmem} MB)")
    for step in steps:
        dt, out = run(base + ["-m", str(gens), "-i", str(step), pattern], timeout)
        gcs = out.count("GC #")
        if dt is None:
            # report how far it got: last "<gen>: <pop>" line
            last = [l for l in out.splitlines() if ":" in l and l.split(":")[0].replace(",", "").strip().isdigit()]
            reached = int(last[-1].split(":")[0].replace(",", "")) if last else 0
            net = timeout - load
            line = f"step {step:>8}: TIMEOUT after {timeout:.0f}s at gen {reached}, {reached / net:12.1f} gens/s"
            if cells:
                line += f", {cells * reached / net:.3e} cell-updates/s"
            print(line + f", {gcs} GCs", flush=True)
            continue
        net = max(dt - load, 1e-9)
        line = f"step {step:>8}: {dt:8.3f}s total, {net:8.3f}s net, {gens / net:14.1f} gens/s"
        if cells:
            line += f", {cells * gens / net:.3e} cell-updates/s"
        print(line + f", {gcs} GCs", flush=True)


main()
