"""Helper process for `xag.map_xag_exact` (kept separate so that it can be killed on timeout;
python-sat does not release the GIL inside RC2).

usage: maxsat_worker.py <file.wcnf> rc2          exact (core-guided); prints "o <cost>" + model
       maxsat_worker.py <file.wcnf> lsu <secs>   anytime (model-improving), stops after <secs>
"""
import sys
import threading

from pysat.formula import WCNF


def main():
    path, mode = sys.argv[1], sys.argv[2]
    w = WCNF(from_file=path)
    if mode == "rc2":
        from pysat.examples.rc2 import RC2

        with RC2(w) as s:
            model = s.compute()
            print("optimal 1")
            print("v " + " ".join(map(str, model)))
    else:
        from pysat.examples.lsu import LSU

        secs = float(sys.argv[3])
        lsu = LSU(w, solver="g4", expect_interrupt=True)
        timer = threading.Timer(secs, lsu.interrupt)
        timer.start()
        ok = lsu.solve()
        timer.cancel()
        assert ok
        print(f"optimal {int(bool(lsu.found_optimum()))}")
        print("v " + " ".join(map(str, lsu.get_model())))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
