"""ablate_local_ext_64q.py — intra-core extended set: topological scan vs BFS closure.

Runs the paper's 64q protocol (H-grid 2x3 4x4, SabreLayout corners-removed best
of 3 seeds, fwd -> bwd -> fwd passes, best of pass 1 / pass 3 by EPR) with two
routers that differ *only* in how the intra-core lookahead set `E_c` is built:

  scan  dSABRE_BFSExt          -- shipped: global topological sweep with taint
                                  propagation, Theta(N_r) per call
  bfs   dSABRE_LocalBFS        -- seeded Kahn closure of the front layer through
                                  core-local gates, O(L + M) per call

Both use the BFS-layer *inter*-core extended set, so the only axis under test is
the intra-core one.  Reports EPR pairs, local SWAPs, compile time, and the DAG
nodes each traversal examines.

Output: results/results_ablate_local_ext_64q.json, rewritten after every circuit.

Usage:  python3 ablate_local_ext_64q.py [--circuits ae,qft]
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from architecture import build_h_grid_architecture
from config import HardwareConfig
from dsabre_local_bfs import (dSABRE_LocalBFS, dSABRE_LocalKahnLex,
                              dSABRE_LocalScanCounted)
from layout import sabre_locked_boundary_layout, run_sabre_passes
from circuit_paths import circuits_path

CIRC_DIR = circuits_path("qasm_64")
SUFFIX = "_nativegates_ibm_qiskit_opt3_64.qasm"
OUT = os.path.join(_HERE, "results", "results_ablate_local_ext_64q.json")
CANON = {"ae", "ghz", "graphstate", "qft", "qnn", "random",
         "qpeexact", "qaoa", "multiplier"}

# CX counts of the published 64q suite (benchmark_circuits/README.md): abort on a mismatch rather
# than silently benchmarking a regenerated circuit.
EXPECTED_CX = {"ae": 1962, "ghz": 63, "graphstate": 64, "multiplier": 13040,
               "qaoa": 3920, "qft": 1966, "qnn": 8126, "qpeexact": 2139,
               "random": 1627}

_HW = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                     max_iterations=20000)
NUM_SL_SEEDS = 3


def load_qasm(path):
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def run_variant(router, dag, rev_dag, layouts):
    """Best-of-seeds under the paper's protocol; whole protocol is charged."""
    best, protocol_s = None, 0.0
    router.le_calls = router.le_visits = 0
    for layout in layouts:
        t0 = time.perf_counter()
        m = run_sabre_passes(router, dag, rev_dag, layout)
        protocol_s += time.perf_counter() - t0
        if m and not m.get("aborted"):
            if best is None or m["eprs"] < best["eprs"]:
                best = m
    if best is None:
        return {"aborted": True, "le_calls": router.le_calls,
                "le_visits": router.le_visits}
    return dict(eprs=best["eprs"], ls=best["ls"], cost=round(best["cost"], 3),
                teles=best.get("teles"), time_s=round(protocol_s, 3),
                time_seed_s=round(best["compile_time"], 3), aborted=False,
                backup_activations=best.get("backup_activations", 0),
                le_calls=router.le_calls, le_visits=router.le_visits)


ARMS = {
    "scan": dSABRE_LocalScanCounted,
    "bfs": dSABRE_LocalBFS,
    "lex": dSABRE_LocalKahnLex,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default="")
    ap.add_argument("--arms", default="scan,bfs",
                    help="comma-separated subset of " + ",".join(ARMS))
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            sys.exit(f"unknown arm {a!r}")

    # merge into whatever this file already holds, so arms can be run apart
    prior = {}
    if os.path.exists(OUT):
        with open(OUT) as fh:
            prior = {r["circuit"]: r for r in json.load(fh).get("results", [])}

    arch = build_h_grid_architecture(r=2, s=3, m=4)
    files = sorted(glob.glob(os.path.join(CIRC_DIR, "*.qasm")))
    files = [f for f in files
             if os.path.basename(f).replace(SUFFIX, "") in CANON]
    if args.circuits:
        want = set(args.circuits.split(","))
        files = [f for f in files
                 if os.path.basename(f).replace(SUFFIX, "") in want]

    hdr = f"{'circuit':<12} {'cx':>6}"
    for a in arms:
        hdr += (f" | {a+'_epr':>9} {a+'_ls':>8} {a+'_t':>8} {a+'_v/c':>9}")
    if len(arms) > 1:
        hdr += f" | {'dEPR':>7} {'dt':>7}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    records = []
    for f in files:
        cname = os.path.basename(f).replace(SUFFIX, "")
        qc, dag = load_qasm(f)
        n_cx = sum(1 for _ in dag.two_qubit_ops())
        if EXPECTED_CX.get(cname) not in (None, n_cx):
            print(f"{cname}: CX {n_cx} != published {EXPECTED_CX[cname]} -- "
                  f"circuit was regenerated, aborting", flush=True)
            sys.exit(1)
        rev_dag = circuit_to_dag(qc.reverse_ops())
        layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)[:NUM_SL_SEEDS]

        rec = dict(prior.get(cname, {}))
        rec.update(circuit=cname, cx=n_cx)
        for a in arms:
            rec[a] = run_variant(ARMS[a](arch, _HW), dag, rev_dag, layouts)

        def vpc(r):
            return r["le_visits"] / max(r["le_calls"], 1)

        def d(a, b):
            if a is None or b is None or a == 0:
                return "   ---"
            return f"{100*(b-a)/a:+.1f}%"

        row = f"{cname:<12} {n_cx:>6}"
        for a in arms:
            r = rec[a]
            row += (f" | {r.get('eprs','ABORT'):>9} {r.get('ls',''):>8} "
                    f"{r.get('time_s',0):>8.1f} {vpc(r):>9.1f}")
        if len(arms) > 1:
            f0, f1 = rec[arms[0]], rec[arms[-1]]
            row += (f" | {d(f0.get('eprs'), f1.get('eprs')):>7} "
                    f"{d(f0.get('time_s'), f1.get('time_s')):>7}")
        print(row, flush=True)

        records.append(rec)
        # circuits this run skipped keep whatever the file already had
        done = {r["circuit"] for r in records}
        merged = records + [v for k, v in prior.items() if k not in done]
        with open(OUT, "w") as fh:
            json.dump(dict(meta=dict(
                date=time.strftime("%Y-%m-%d"), suite="64q",
                arch="H-grid 2x3 4x4 (96 qubits)",
                layout="SabreLayout corners-removed, best of 3 seeds",
                pass_strategy="fwd -> bwd (reversed DAG) -> fwd; best of pass1/pass3",
                variants={
                    "scan": "dSABRE_LocalScanCounted -- shipped topological "
                            "sweep with taint propagation",
                    "bfs": "dSABRE_LocalBFS -- seeded Kahn closure over "
                           "core-local wires, FIFO order",
                    "lex": "dSABRE_LocalKahnLex -- the same closure, emitted "
                           "in Qiskit's lexicographic topological order",
                },
            ), results=merged), fh, indent=1)

    base = arms[0]
    ok = [r for r in records
          if all(not r[a].get("aborted") for a in arms)]
    if ok:
        print("-" * len(hdr), flush=True)
        for a in arms:
            e = sum(r[a]["eprs"] for r in ok)
            t = sum(r[a]["time_s"] for r in ok)
            v = sum(r[a]["le_visits"] for r in ok)
            be = sum(r[base]["eprs"] for r in ok)
            bt = sum(r[base]["time_s"] for r in ok)
            bv = sum(r[base]["le_visits"] for r in ok)
            same = sum(1 for r in ok if r[a]["eprs"] == r[base]["eprs"]
                       and r[a]["ls"] == r[base]["ls"])
            print(f"{a:<5} EPR {e:>6} ({100*(e-be)/be:+.1f}% vs {base}), "
                  f"time {t:>8.1f}s ({100*(t-bt)/bt:+.1f}%), "
                  f"E_c nodes {v:>12} ({bv/max(v,1):.1f}x fewer), "
                  f"identical (EPR,ls) to {base} on {same}/{len(ok)}",
                  flush=True)

    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
