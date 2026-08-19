"""probe_pytket_esd_fallback.py — why pytket-dqc falls back on the 64q suite.

bench_pytket_fair.py tries CoverEmbeddingSteinerDetached (ESD) first and drops
to PartitioningHeterogeneous (PH) when ESD returns nothing.  Table tab:timing
daggers four 64q circuits as fallbacks, but their reported times (325-479 s)
are far below the 1200 s ESD budget, so ESD cannot have exhausted it -- it must
raise.  This reports which, and with what, so the paper can say so accurately.

Contrast: on 200q/360q the sweep ran with --budget 900 and the reported times
(1453 s, 2065 s) exceed it, i.e. there ESD really did burn the budget.

Usage:  python3 code/probe_pytket_esd_fallback.py [--circuits qaoa,qnn,...]
                                                  [--budget 300]
"""

import sys, os, time, argparse, traceback

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from bench_pytket_fair import (SUITES, build_networks, _call_with_timeout,
                               _Timeout)

DEFAULT = "qaoa,qnn,random,multiplier"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default=DEFAULT)
    ap.add_argument("--budget", type=float, default=300.0,
                    help="seconds per ESD seed-0 attempt (the sweep gives the "
                         "whole 5-seed loop 1200 s)")
    args = ap.parse_args()

    from pytket.qasm import circuit_from_qasm
    from pytket_dqc.utils import DQCPass
    from pytket_dqc.distributors import CoverEmbeddingSteinerDetached

    s = SUITES["64q"]
    for cname in [c.strip() for c in args.circuits.split(",") if c.strip()]:
        path = os.path.join(s["circuit_dir"], cname + s["suffix"])
        circ = circuit_from_qasm(path, maxwidth=128)
        DQCPass().apply(circ)
        nets, _info = build_networks(s["arch"], 64)
        net = nets["C_physical"]

        t0 = time.perf_counter()
        try:
            d = _call_with_timeout(
                lambda: CoverEmbeddingSteinerDetached().distribute(
                    circ, net, seed=0), args.budget)
            print(f"{cname:<12} ESD COMPLETED cost={d.cost()} "
                  f"in {time.perf_counter()-t0:.1f}s", flush=True)
        except _Timeout:
            print(f"{cname:<12} ESD TIMEOUT after {time.perf_counter()-t0:.1f}s "
                  f"(budget {args.budget:.0f}s)", flush=True)
        except Exception as e:
            print(f"{cname:<12} ESD RAISED after {time.perf_counter()-t0:.1f}s: "
                  f"{type(e).__name__}: {str(e)[:200]}", flush=True)
            traceback.print_exc(limit=3)


if __name__ == "__main__":
    main()
