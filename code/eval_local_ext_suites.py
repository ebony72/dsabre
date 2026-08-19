"""eval_local_ext_suites.py — full evaluation of the intra-core E_c construction.

Same three arms as `ablate_local_ext_64q.py`, over every suite the paper
reports.  The arms differ *only* in how `_get_local_extended` builds the
per-core intra-core lookahead set; the inter-core BFS set, the scorers, the
layout and the pass schedule are identical.

  sweep  dSABRE_LocalScanCounted  shipped: global topological sweep with taint
                                  propagation, Theta(N_r) per call
  fifo   dSABRE_LocalBFS          seeded closure of the front layer through
                                  core-local gates, popped FIFO, O(L+M)
  dag    dSABRE_LocalKahnLex      the same closure, popped in the host DAG
                                  library's own topological order

Protocol per suite is the paper's: sabre_locked_boundary_layout (3 SabreLayout
seeds) -> run_sabre_passes (fwd -> bwd -> fwd, best of pass 1 / pass 3), best
layout by EPR; the whole protocol is charged to `time_s`.

Output: results/results_local_ext_<suite>.json, rewritten after each circuit.

Usage:
  python3 eval_local_ext_suites.py --suite 25q
  python3 eval_local_ext_suites.py --suite hh-ring --arms sweep,fifo
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
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from architecture import (build_b_grid_architecture, build_h_grid_architecture,
                          build_heavy_hex_architecture)
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from dsabre_local_bfs import (dSABRE_LocalBFS, dSABRE_LocalKahnLex,
                              dSABRE_LocalScanCounted, dSABRE_SharedExt,
                              dSABRE_SharedExtK, dSABRE_SharedExt2Q)
from layout import sabre_locked_boundary_layout, run_sabre_passes
from circuit_paths import circuits_path

_C = circuits_path()
_HW_SMALL = HardwareConfig()
_HW_LARGE = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                           max_iterations=20000)
_HW_HUGE = HardwareConfig(deadlock_limit=200, max_backup_attempts=200,
                          max_iterations=50000)

# `circuits=None` means "every .qasm in the directory" (25q/36q, as
# benchmark.py globs them); a list is a whitelist, as the 64q suite and the
# large suites use.
SUITES = {
    "25q":     dict(d=f"{_C}/qasm_25", sfx="_nativegates_ibm_qiskit_opt3_25.qasm",
                    arch=lambda: build_b_grid_architecture(r=2, s=2, m=4),
                    hw=_HW_SMALL, circuits=None),
    "36q":     dict(d=f"{_C}/qasm_36", sfx="_nativegates_ibm_qiskit_opt3_36.qasm",
                    arch=lambda: build_b_grid_architecture(r=2, s=2, m=4),
                    hw=_HW_LARGE, circuits=None),
    "64q":     dict(d=f"{_C}/qasm_64", sfx="_nativegates_ibm_qiskit_opt3_64.qasm",
                    arch=lambda: build_h_grid_architecture(r=2, s=3, m=4),
                    hw=_HW_LARGE,
                    circuits=["ae", "ghz", "graphstate", "multiplier", "qaoa",
                              "qft", "qnn", "qpeexact", "random"]),
    "hh-ring": dict(d=f"{_C}/qasm_64", sfx="_nativegates_ibm_qiskit_opt3_64.qasm",
                    arch=lambda: build_heavy_hex_architecture(4, "ring"),
                    hw=_HW_LARGE,
                    circuits=["ae", "ghz", "graphstate", "qft", "qnn", "random"]),
    "hh-star": dict(d=f"{_C}/qasm_64", sfx="_nativegates_ibm_qiskit_opt3_64.qasm",
                    arch=lambda: build_heavy_hex_architecture(4, "star"),
                    hw=_HW_LARGE,
                    circuits=["ae", "ghz", "graphstate", "qft", "qnn", "random"]),
    "100q":    dict(d=f"{_C}/qasm_100", sfx="_nativegates_ibm_qiskit_opt3_100.qasm",
                    arch=lambda: build_h_grid_architecture(r=2, s=3, m=5),
                    hw=_HW_HUGE, circuits=["qft", "qpeexact"]),
    "200q":    dict(d=f"{_C}/qasm_200", sfx="_nativegates_ibm_qiskit_opt3_200.qasm",
                    arch=lambda: build_h_grid_architecture(r=4, s=3, m=5),
                    hw=_HW_HUGE, circuits=["qft", "qpeexact"]),
    "360q":    dict(d=f"{_C}/qasm_360", sfx="_nativegates_ibm_qiskit_opt3_360.qasm",
                    arch=lambda: build_h_grid_architecture(r=2, s=3, m=9),
                    hw=_HW_HUGE, circuits=["qft", "qpeexact"]),
}

# CX counts the published tables report.  A mismatch means the shared circuit
# directory was regenerated (see benchmark_circuits/README.md) and the run is not comparable.
EXPECTED_CX = {
    "64q": {"ae": 1962, "ghz": 63, "graphstate": 64, "multiplier": 13040,
            "qaoa": 3920, "qft": 1966, "qnn": 8126, "qpeexact": 2139,
            "random": 1627},
}
EXPECTED_CX["hh-ring"] = EXPECTED_CX["hh-star"] = EXPECTED_CX["64q"]

class _CountedDefault(dSABRE_BurstExt):
    """The shipped router verbatim, with the counter attributes the driver
    reads but no instrumentation inside `_get_local_extended`.

    Exists to price the `visits += 1` that `sweep` adds to the hot loop: the
    two must agree on every metric, and their times bound the counter's cost.
    """
    le_calls = 0
    le_visits = 0


ARMS = {
    "default": _CountedDefault,      # dSABRE_BurstExt, uninstrumented
    "sweep": dSABRE_LocalScanCounted,
    "fifo": dSABRE_LocalBFS,
    "dag": dSABRE_LocalKahnLex,
    "shared": dSABRE_SharedExt,     # E_c = E restricted to core c, |E| = L
    "sharedk": dSABRE_SharedExtK,   # ... with |E| = K*L
    "cached": dSABRE_SharedExt,     # same as shared; memo is now built in
    "shared2q": dSABRE_SharedExt2Q, # ... with dep counting 2Q layers only
}
NUM_SL_SEEDS = 3


def load_qasm(path):
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def run_arm(router, dag, rev_dag, layouts):
    best, protocol_s = None, 0.0
    router.le_calls = router.le_visits = 0
    if hasattr(router, 'ext_builds'):
        router.ext_builds = router.ext_reuses = 0
    aborts = 0
    for layout in layouts:
        t0 = time.perf_counter()
        m = run_sabre_passes(router, dag, rev_dag, layout)
        protocol_s += time.perf_counter() - t0
        if not m or m.get("aborted"):
            aborts += 1
            continue
        if best is None or m["eprs"] < best["eprs"]:
            best = m
    common = dict(le_calls=router.le_calls, le_visits=router.le_visits,
                  seed_aborts=aborts, time_s=round(protocol_s, 3))
    if hasattr(router, "ext_builds"):
        common.update(ext_builds=router.ext_builds,
                      ext_reuses=router.ext_reuses)
    if best is None:
        return dict(aborted=True, **common)
    return dict(eprs=best["eprs"], ls=best["ls"], cost=round(best["cost"], 3),
                teles=best.get("teles"),
                time_seed_s=round(best["compile_time"], 3), aborted=False,
                backup_activations=best.get("backup_activations", 0), **common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True, choices=list(SUITES))
    ap.add_argument("--arms", default="sweep,fifo,dag")
    ap.add_argument("--circuits", default="")
    args = ap.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            sys.exit(f"unknown arm {a!r}")

    s = SUITES[args.suite]
    out = os.path.join(_HERE, "results", f"results_local_ext_{args.suite}.json")
    prior = {}
    if os.path.exists(out):
        with open(out) as fh:
            prior = {r["circuit"]: r for r in json.load(fh).get("results", [])}

    arch = s["arch"]()
    files = sorted(glob.glob(os.path.join(s["d"], "*.qasm")))
    want = set(args.circuits.split(",")) if args.circuits else (
        set(s["circuits"]) if s["circuits"] else None)
    if want is not None:
        files = [f for f in files
                 if os.path.basename(f).replace(s["sfx"], "") in want]
    if not files:
        sys.exit(f"no circuits found in {s['d']}")

    hdr = f"{'circuit':<12} {'cx':>6}"
    for a in arms:
        hdr += f" | {a+'_epr':>9} {a+'_ls':>8} {a+'_t':>8} {a+'_v/c':>8}"
    print(f"suite {args.suite}: {len(files)} circuits, arms {arms}", flush=True)
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    records = []
    for f in files:
        cname = os.path.basename(f).replace(s["sfx"], "")
        qc, dag = load_qasm(f)
        n_cx = sum(1 for _ in dag.two_qubit_ops())
        exp = EXPECTED_CX.get(args.suite, {}).get(cname)
        if exp is not None and exp != n_cx:
            sys.exit(f"{cname}: CX {n_cx} != published {exp} -- circuit was "
                     f"regenerated, refusing to benchmark")
        rev_dag = circuit_to_dag(qc.reverse_ops())
        layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)[:NUM_SL_SEEDS]

        rec = dict(prior.get(cname, {}))
        rec.update(circuit=cname, cx=n_cx, qubits=qc.num_qubits)
        row = f"{cname:<12} {n_cx:>6}"
        for a in arms:
            r = rec[a] = run_arm(ARMS[a](arch, s["hw"]), dag, rev_dag, layouts)
            row += (f" | {r.get('eprs', 'ABORT'):>9} {r.get('ls', ''):>8} "
                    f"{r['time_s']:>8.1f} "
                    f"{r['le_visits']/max(r['le_calls'],1):>8.1f}")
        print(row, flush=True)

        records.append(rec)
        done = {r["circuit"] for r in records}
        merged = records + [v for k, v in prior.items() if k not in done]
        with open(out, "w") as fh:
            json.dump(dict(meta=dict(
                date=time.strftime("%Y-%m-%d"), suite=args.suite,
                layout="SabreLayout corners-removed, best of 3 seeds",
                pass_strategy="fwd -> bwd (reversed DAG) -> fwd; best of pass1/pass3",
                arms={
                    "default": "dSABRE_BurstExt -- the shipped router verbatim",
                    "sweep": "dSABRE_LocalScanCounted -- shipped topological sweep",
                    "fifo": "dSABRE_LocalBFS -- seeded closure, FIFO order",
                    "dag": "dSABRE_LocalKahnLex -- seeded closure, host DAG order",
                    "shared": "dSABRE_SharedExt -- E_c is the inter-core BFS "
                              "set restricted to core c, |E| = L",
                    "sharedk": "dSABRE_SharedExtK -- the same, |E| = K*L",
                    "shared2q": "dSABRE_SharedExt2Q -- the same, dep counts "
                                "two-qubit layers only",
                },
            ), results=merged), fh, indent=1)

    ok = [r for r in records if all(not r[a].get("aborted") for a in arms)]
    print("-" * len(hdr), flush=True)
    base = arms[0]
    for a in arms:
        e = sum(r[a]["eprs"] for r in ok)
        t = sum(r[a]["time_s"] for r in ok)
        v = sum(r[a]["le_visits"] for r in ok)
        be = sum(r[base]["eprs"] for r in ok)
        bt = sum(r[base]["time_s"] for r in ok)
        bv = sum(r[base]["le_visits"] for r in ok)
        same = sum(1 for r in ok if r[a]["eprs"] == r[base]["eprs"]
                   and r[a]["ls"] == r[base]["ls"])
        ab = sum(r[a].get("seed_aborts", 0) for r in records)
        print(f"{a:<6} EPR {e:>6} ({100*(e-be)/max(be,1):+.1f}%)  "
              f"time {t:>8.1f}s ({bt/max(t,1e-9):.1f}x)  "
              f"E_c nodes {v:>11} ({bv/max(v,1):.1f}x)  "
              f"=={base} on {same}/{len(ok)}  seed-aborts {ab}", flush=True)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
