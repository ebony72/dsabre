"""sweep_params.py — one-at-a-time hyperparameter sweep behind `app:sensitivity`.

The appendix claims every continuous default sits on a flat region of its own
curve (within ~2% of the local minimum).  No script produced those numbers --
EXPERIMENTS.md lists Appendix B.2 as "script missing" -- so the figures in the
text predate the 2026-08-05..09 router changes (transactional recovery, the
teleport-legality fix, safe mode, the shared E_c, the legality floor).  This
script measures the claim against the current router.

Protocol is `benchmark.py`'s, matching `regen_tab_mech.py` exactly:
SabreLayout with per-core corners removed, best of 3 seeds, fwd->bwd->fwd
per seed, best layout by EPR, dSE (`dSABRE_BurstExt`) only.  Layouts depend
on (circuit, architecture, seed) and not on the config, so they are built
once per circuit and reused across every sweep point.

The default value of each parameter is included in its own grid, and routed
once per suite rather than once per grid.  Its geometric mean is checked
against the published Table III figure as a drift preflight: if `Full` does
not reproduce 15.3 (25q) / 173.3 (64q), the router generation has moved and
the sweep is not comparable with the table it annotates.

`w_link` is deliberately absent.  The inter-core edge weight is hardcoded to
10 in `architecture.py` (`Gr.add_edge(u, v, weight=10)`), not a HardwareConfig
field, so it cannot be swept without rebuilding the architecture -- see the
note printed at the end.

Usage
-----
    python3 code/sweep_params.py                       # all four params, both suites
    python3 code/sweep_params.py --param lookahead_decay
    python3 code/sweep_params.py --param weight_extended --values 0 0.25 1.0
    python3 code/sweep_params.py --suites 25q

Output: code/results/results_param_sweep.json, rewritten after every routed
cell so a killed run is still usable.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import warnings
from math import prod

warnings.filterwarnings("ignore")
sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from benchmark import SUITES, CANONICAL_CIRCUITS
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes

OUT = os.path.join(_HERE, "results", "results_param_sweep.json")

# Default first in each grid: it is the cell every other cell is measured
# against, and routing it first makes the drift preflight fail fast.
GRIDS = {
    "weight_extended": [0.25, 0.0, 0.1, 0.5, 1.0, 2.0],
    "cap_penalty":     [15.0, 0.0, 5.0, 30.0, 60.0],
    "lookahead_size":  [20, 5, 10, 40, 80],
    "lookahead_decay": [0.9, 0.5, 0.7, 1.0],
}
DEFAULTS = {k: v[0] for k, v in GRIDS.items()}

# Table III's dSABRE geometric means -- the drift check.
PUBLISHED_GMEAN = {"25q": 15.3, "64q": 173.3}


def gmean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return prod(xs) ** (1 / len(xs)) if xs else float("nan")


def load_qasm(path):
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def suite_circuits(sname):
    """(name, qc, dag, rev_dag) for each circuit of `sname`, cheapest first."""
    s = SUITES[sname]
    files = sorted(glob.glob(os.path.join(s["circuit_dir"], "*.qasm")))
    canon = CANONICAL_CIRCUITS.get(sname)
    if canon:
        files = [f for f in files
                 if os.path.basename(f).replace(s["suffix"], "") in canon]
    out = []
    for f in files:
        name = os.path.basename(f).replace(s["suffix"], "")
        qc, dag = load_qasm(f)
        out.append((name, qc, dag, circuit_to_dag(qc.reverse_ops()),
                    sum(1 for _ in dag.two_qubit_ops())))
    out.sort(key=lambda t: t[4])          # cheapest first: partial runs usable
    return out


def route(cfg, dag, rev_dag, layouts, arch):
    """Best EPR over the 3 seed layouts; None if every one aborts."""
    router = dSABRE_BurstExt(arch, cfg)
    best, aborts = None, 0
    for layout in layouts:
        m = run_sabre_passes(router, dag, rev_dag, layout)
        if m and not m.get("aborted"):
            if best is None or m["eprs"] < best["eprs"]:
                best = m
        else:
            aborts += 1
    return (None if best is None else best["eprs"]), aborts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--param", action="append", choices=sorted(GRIDS),
                    help="repeatable; default is all four")
    ap.add_argument("--values", nargs="+", type=float,
                    help="override the grid (single --param only)")
    ap.add_argument("--suites", default="25q,64q")
    args = ap.parse_args()

    params = args.param or sorted(GRIDS)
    if args.values:
        if len(params) != 1:
            raise SystemExit("--values needs exactly one --param")
        cast = int if params[0] == "lookahead_size" else float
        GRIDS[params[0]] = [cast(v) for v in args.values]
    suites = [s for s in args.suites.split(",") if s]

    doc = {"meta": {
        "date": time.strftime("%Y-%m-%d"),
        "purpose": "one-at-a-time hyperparameter sweep behind app:sensitivity",
        "router": "dSE (dsabre_ext.dSABRE_BurstExt)",
        "protocol": "SabreLayout corners-removed, best of 3 seeds, fwd->bwd->fwd",
        "note": "w_link is hardcoded in architecture.py, not a HardwareConfig "
                "field, so it is not swept here",
    }, "results": {}}
    t_start = time.time()

    for sname in suites:
        s = SUITES[sname]
        arch, hw = s["arch"], s["hw"]
        circuits = suite_circuits(sname)
        print(f"\n{'='*72}\n  {sname}: {len(circuits)} circuits, "
              f"{sum(len(GRIDS[p]) for p in params) - (len(params) - 1)} routed cells"
              f"\n{'='*72}", flush=True)

        # Layouts are config-independent: build once, reuse everywhere.
        layouts = {n: sabre_locked_boundary_layout(qc, dag, arch, seed=0)
                   for n, qc, dag, _rev, _cx in circuits}

        sres = doc["results"].setdefault(sname, {})

        # The shared default cell, routed once for all four grids.
        base_cfg = HardwareConfig(**hw.__dict__)
        base, base_ab = {}, {}
        print(f"{'default':<28}", end="", flush=True)
        for n, _qc, dag, rev, _cx in circuits:
            t0 = time.time()
            v, ab = route(base_cfg, dag, rev, layouts[n], arch)
            base[n], base_ab[n] = v, ab
            print(f" {n}={'ABORT' if v is None else v}({time.time()-t0:.0f}s)",
                  end="", flush=True)
        g_base = gmean(list(base.values()))
        pub = PUBLISHED_GMEAN[sname]
        drift = "" if abs(g_base - pub) < 0.05 * pub else \
                f"   [!! published says {pub} -- DRIFT]"
        print(f"\n  gmean = {g_base:.1f}  (published {pub}){drift}", flush=True)
        sres["_default"] = {"gmean": round(g_base, 2), "per_circuit": base,
                            "aborts": sum(base_ab.values())}
        json.dump(doc, open(OUT, "w"), indent=1)

        for p in params:
            sres[p] = {}
            for val in GRIDS[p]:
                if val == DEFAULTS[p]:
                    sres[p][str(val)] = {"gmean": round(g_base, 2),
                                         "delta_pct": 0.0,
                                         "per_circuit": base,
                                         "aborts": sum(base_ab.values()),
                                         "note": "default cell, routed once"}
                    print(f"  {p:<18} {val:<7} gmean={g_base:8.1f}  "
                          f"{'0.0':>7}%   (default)", flush=True)
                    json.dump(doc, open(OUT, "w"), indent=1)
                    continue
                cfg = HardwareConfig(**{**hw.__dict__, p: val})
                per, ab_tot, t0 = {}, 0, time.time()
                for n, _qc, dag, rev, _cx in circuits:
                    v, ab = route(cfg, dag, rev, layouts[n], arch)
                    per[n], ab_tot = v, ab_tot + ab
                g = gmean(list(per.values()))
                d = 100 * (g - g_base) / g_base
                n_fail = sum(1 for v in per.values() if v is None)
                sres[p][str(val)] = {"gmean": round(g, 2),
                                     "delta_pct": round(d, 1),
                                     "per_circuit": per,
                                     "aborts": ab_tot,
                                     "circuits_failed": n_fail}
                print(f"  {p:<18} {val:<7} gmean={g:8.1f}  {d:+7.1f}%"
                      + (f"   [{n_fail} circuit(s) abort]" if n_fail else "")
                      + f"   ({time.time()-t0:.0f}s)", flush=True)
                json.dump(doc, open(OUT, "w"), indent=1)

    print(f"\nDone in {time.time()-t_start:.0f}s -> {OUT}", flush=True)
    print("NOTE: w_link (inter-core edge weight) is hardcoded to 10 in "
          "architecture.py and is not a HardwareConfig field, so it is not "
          "part of this sweep.", flush=True)


if __name__ == "__main__":
    main()
