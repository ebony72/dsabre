"""verify_shared_ext.py — the shipped router reproduces the evaluated shared set.

`verify_router.py` proves the incremental rewrite is output-identical to the
frozen baseline, and runs in `local_ext_mode="taint"` to do it: the baseline
predates the shared intra-core extended set, so a diff in the default mode
would report a deliberate change as a regression.

This is the check for the other side.  It reruns the *default* router over
every suite and compares against the `shared` arm of
`results/results_local_ext_<suite>.json` -- the arm the adoption decision was
made on (`eval_local_ext_suites.py`, 2026-08-08).  A mismatch means the
router as shipped is no longer the router that was evaluated.

Two properties are checked per circuit:
  * EPR pairs and local SWAPs match the recorded `shared` arm exactly, and
  * the default router and an explicit `dSABRE_SharedExt` agree, which is what
    makes the two interchangeable.

Usage:  python3 verify_shared_ext.py [--suite 64q] [--quick]
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from dsabre_ext import dSABRE_BurstExt
from dsabre_local_bfs import dSABRE_SharedExt
from eval_local_ext_suites import SUITES, NUM_SL_SEEDS
from layout import sabre_locked_boundary_layout, run_sabre_passes

QUICK = ["25q", "36q", "64q"]


def load_qasm(path):
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def best(router, dag, rev_dag, layouts):
    out = None
    for layout in layouts:
        m = run_sabre_passes(router, dag, rev_dag, layout)
        if m and not m.get("aborted"):
            if out is None or m["eprs"] < out["eprs"]:
                out = m
    return None if out is None else (out["eprs"], out["ls"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="")
    ap.add_argument("--quick", action="store_true",
                    help="25q/36q/64q only; skip heavy-hex and 100q+")
    args = ap.parse_args()

    suites = ([args.suite] if args.suite
              else (QUICK if args.quick else list(SUITES)))
    ok = bad = missing = 0

    for su in suites:
        f = os.path.join(_HERE, "results", f"results_local_ext_{su}.json")
        if not os.path.exists(f):
            print(f"{su}: no recorded evaluation, skipping", flush=True)
            continue
        recorded = {r["circuit"]: r.get("shared") for r in
                    json.load(open(f))["results"]}
        s = SUITES[su]
        arch = s["arch"]()
        print(f"\n{'='*72}\n  {su}\n{'='*72}", flush=True)
        print(f"{'circuit':<12} {'expected':>14} {'default':>14} "
              f"{'SharedExt':>14}  verdict", flush=True)

        for cname, exp in sorted(recorded.items()):
            if not exp or exp.get("aborted"):
                continue
            path = os.path.join(s["d"], f"{cname}{s['sfx']}")
            if not os.path.exists(path):
                missing += 1
                continue
            qc, dag = load_qasm(path)
            rev_dag = circuit_to_dag(qc.reverse_ops())
            layouts = sabre_locked_boundary_layout(qc, dag, arch,
                                                   seed=0)[:NUM_SL_SEEDS]
            want = (exp["eprs"], exp["ls"])
            got_default = best(dSABRE_BurstExt(arch, s["hw"]), dag, rev_dag, layouts)
            got_shared = best(dSABRE_SharedExt(arch, s["hw"]), dag, rev_dag, layouts)
            good = got_default == want and got_shared == want
            ok += good
            bad += not good
            fmt = lambda t: "ABORT" if t is None else f"{t[0]}/{t[1]}"
            print(f"{cname:<12} {fmt(want):>14} {fmt(got_default):>14} "
                  f"{fmt(got_shared):>14}  {'ok' if good else 'MISMATCH'}",
                  flush=True)

    print(f"\n{'-'*72}")
    if missing:
        print(f"note: {missing} circuit files not found", flush=True)
    if bad:
        print(f"FAIL: {bad} of {ok+bad} circuits differ from the evaluated "
              f"shared arm.", flush=True)
        sys.exit(1)
    print(f"PASS: all {ok} circuits match the evaluated shared arm, and the "
          f"default router agrees with dSABRE_SharedExt throughout.", flush=True)


if __name__ == "__main__":
    main()
