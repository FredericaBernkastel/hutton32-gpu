"""Time bgolly (HashLife) on a pattern for a fixed number of generations at several step sizes.

usage: python time_golly.py <pattern> <gens> <step> [<step> ...]
         [--algo RuleLoader|JvN|RuleTree] [--maxmem MB] [--cells N] [--timeout S] [--rules DIR]
         [--repeats N] [--gc]
Prints wall-clock seconds (minus load-only time), generations/s, cell-updates/s (for --cells N),
the peak working set of the bgolly process, the rate over the second half of the run and, with --gc,
the number of garbage collections Golly reported. Golly runs with -q (one progress line per step, no
population); --gc switches to -v, which also computes the population at every step and is slower at
small step sizes.
With --repeats N every step size is run N times and the median time is reported with the range.
bgolly is taken from $BGOLLY (default: Golly 5.1's install path, else bgolly on PATH); the default
rule folder is research/rules.
"""
import os, shutil, statistics, subprocess, sys, time

_WIN_BGOLLY = r"C:\Program Files\golly-5.1-win-64bit\bgolly.exe"
BGOLLY = os.environ.get("BGOLLY") or (_WIN_BGOLLY if os.path.exists(_WIN_BGOLLY) else shutil.which("bgolly") or "bgolly")
RULES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "rules") + os.sep


def peak_mib(proc):
    """Peak working set of a finished child process in MiB (None where it cannot be read)."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
        pmc = PMC()
        pmc.cb = ctypes.sizeof(PMC)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(wintypes.HANDLE(int(proc._handle)), ctypes.byref(pmc), pmc.cb)
        return pmc.PeakWorkingSetSize / 2**20 if ok else None
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024  # KiB on Linux; max over children so far
    except Exception:
        return None


PROGRESS = []  # (seconds since start, generation) of the last run's "<gen>: <pop>" lines


def run(args, timeout):
    """Returns (seconds, stdout, peak MiB, status): status is "ok", "timeout" or "exit N" (bgolly
    failed, e.g. at its memory limit). Progress lines are timestamped as they arrive, in PROGRESS."""
    import threading
    PROGRESS.clear()
    t = time.perf_counter()
    p = subprocess.Popen([BGOLLY] + args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    chunks = []  # (arrival time, bytes): read in large chunks so that bgolly never waits on the pipe

    def reader():
        fd = p.stdout.fileno()
        while True:
            b = os.read(fd, 1 << 16)
            if not b:
                break
            chunks.append((time.perf_counter() - t, b))

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    timed_out = False
    try:
        p.wait(timeout=timeout)
        dt = time.perf_counter() - t
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        timed_out = True
    th.join(5)
    # progress lines ("<gen>" with -q, "<gen>: <pop>" with -v) get the arrival time of the chunk
    # that completed them
    out, rest = [], b""
    for at, b in chunks:
        parts = (rest + b).split(b"\n")
        rest = parts.pop()
        for line in parts:
            s = line.decode(errors="replace")
            out.append(s)
            head = s.split(":")[0].replace(",", "").strip()
            if head.isdigit():
                PROGRESS.append((at, int(head)))
    out.append(rest.decode(errors="replace"))
    out = "\n".join(out)
    if timed_out:
        return timeout, out, peak_mib(p), "timeout"
    if p.returncode != 0:
        time.sleep(5)  # let the OS take back the failed process's memory before the next run
        return dt, out, peak_mib(p), f"exit {p.returncode}"
    return dt, out, peak_mib(p), "ok"


def late_rate(gens):
    """Generations per second over the second half of the last run (None if not enough progress lines)."""
    pts = [(t, g) for t, g in PROGRESS if g >= gens // 2]
    if len(pts) < 2 or pts[-1][1] == pts[0][1]:
        return None
    return (pts[-1][1] - pts[0][1]) / (pts[-1][0] - pts[0][0])


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
    repeats = int(take("--repeats", "1"))
    gc = "--gc" in argv
    if gc:
        argv.remove("--gc")
    pattern, gens = argv[0], int(argv[1])
    steps = [int(s) for s in argv[2:]]
    base = ["-v" if gc else "-q", "-a", algo, "-s", rules, "-M", maxmem]
    load, _, _, _ = run(base + ["-m", "0", pattern], timeout)
    print(f"pattern {pattern}\nload-only {load:.3f}s (algo {algo}, maxmem {maxmem} MB, rules {rules})")
    for step in steps:
        times, peaks, gcs, stopped, lates = [], [], 0, None, []
        # a run that would print more than 20,000 progress lines is timed fully quiet instead
        flags = base if gc or gens // step <= 20000 else ["-q"] + base
        for _ in range(repeats):
            dt, out, peak, status = run(flags + ["-m", str(gens), "-i", str(step), pattern], timeout)
            gcs = out.count("GC #")
            peaks.append(peak or 0.0)
            lr = late_rate(gens)
            if lr:
                lates.append(lr)
            if status != "ok":
                # report how far it got: the last progress line
                stopped = (status, dt, PROGRESS[-1][1] if PROGRESS else 0)
                break
            times.append(max(dt - load, 1e-9))
        mem = f", peak {max(peaks):.0f} MiB"
        if lates:
            mem += f", second half {statistics.median(lates):.1f} gens/s"
        if stopped is not None:
            status, dt, reached = stopped
            net = max(dt - load, 1e-9)
            line = (f"step {step:>8}: {'TIMEOUT' if status == 'timeout' else 'FAILED (' + status + ')'} "
                    f"after {dt:.0f}s at gen {reached}, {reached / net:12.1f} gens/s")
            if cells:
                line += f", {cells * reached / net:.3e} cell-updates/s"
            print(line + (f", {gcs} GCs" if gc else "") + mem, flush=True)
            continue
        net = statistics.median(times)
        line = f"step {step:>8}: {net:8.3f}s net, {gens / net:14.1f} gens/s"
        if cells:
            line += f", {cells * gens / net:.3e} cell-updates/s"
        if repeats > 1:
            line += f" (median of {repeats}; {gens / max(times):.1f}-{gens / min(times):.1f} gens/s)"
        print(line + (f", {gcs} GCs" if gc else "") + mem, flush=True)


main()
