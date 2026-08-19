"""
regen_ablation_64q_rest8.py — companion to regen_ablation_64q_full9.py.

That script re-runs multiplier; this one re-runs the other eight circuits of
the 64q suite, so every cell of Table V comes from one configuration.

Why this is needed.  `regen_ablation_corners.py` routes its ablations under a
bare `HardwareConfig()` (max_iterations=10000, deadlock_limit=50,
max_backup_attempts=50), whereas `benchmark.py` routes the 64q suite that
Table IV reports under

    HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                   max_iterations=20000)

Under the *baseline* configuration the difference is invisible: all six small
circuits record backup_activations = 0 in results_64q.json, so no limit binds
and the two budgets give identical counts.  Under the *ablated* configurations
that reasoning does not carry -- removing the capacity penalty is precisely
what provokes the cascading evictions the deadlock machinery exists to absorb,
so `no_cap_penalty` may well hit a limit that `full` never approaches, and any
row that does is measuring the iteration budget rather than the mechanism.

Output: code/results/results_ablate_mech_64q_rest8.json (written incrementally).

Usage:  python3 code/regen_ablation_64q_rest8.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace

sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from architecture import build_h_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes
from circuit_paths import circuits_path

CIRCUIT_DIR = circuits_path("qasm_64")
SUFFIX = "_nativegates_ibm_qiskit_opt3_64.qasm"
OUT = os.path.join(_HERE, "results", "results_ablate_mech_64q_rest8.json")

CIRCUITS = ["ghz", "graphstate", "qpeexact", "ae", "qft", "qaoa", "random", "qnn"]

# Published dSE counts and CX counts (results_64q.json / Table IV).
EXPECTED_FULL = {"ghz": 9, "graphstate": 16, "qpeexact": 271, "ae": 242,
                 "qft": 204, "qaoa": 529, "random": 721, "qnn": 508}
EXPECTED_CX = {"ghz": 63, "graphstate": 64, "qpeexact": 2139, "ae": 1962,
               "qft": 1966, "qaoa": 3920, "random": 1627, "qnn": 8126}

BASE_CFG = HardwareConfig(
    deadlock_limit=100, max_backup_attempts=100, max_iterations=20000
)
# "no_relief" removed: congestion relief was deleted from the router (see
# CHANGES_FROM_SUBMITTED.md).  EXPECTED_FULL above is STALE (measured with
# relief active) and needs re-measuring before this preflight can be trusted.
CONFIGS = [
    ("full",           BASE_CFG),
    ("no_lookahead",   replace(BASE_CFG, weight_extended=0.0)),
    ("no_cap_penalty", replace(BASE_CFG, cap_penalty=0.0)),
]


def load(name):
    qc = QuantumCircuit.from_qasm_file(os.path.join(CIRCUIT_DIR, name + SUFFIX))
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    dag = circuit_to_dag(qc)
    ncx = sum(1 for _ in dag.two_qubit_ops())
    if ncx != EXPECTED_CX[name]:
        raise SystemExit(
            f"ABORT: {name} has {ncx} CX, published table says "
            f"{EXPECTED_CX[name]}.  The circuit directory has drifted; see "
            f"benchmark_circuits/README.md on MQT Bench version drift.")
    return qc, dag, ncx


def best_of_three(cfg, qc, dag, rev_dag, arch):
    layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
    router = dSABRE_BurstExt(arch, cfg)
    best = None
    for layout in layouts:
        m = run_sabre_passes(router, dag, rev_dag, layout)
        if m and not m.get("aborted"):
            if best is None or m["eprs"] < best["eprs"]:
                best = m
    return best


def main():
    arch = build_h_grid_architecture(r=2, s=3, m=4)
    rows = []
    t_start = time.time()
    print(f"64q mechanism ablation under _HW_LARGE: {len(CIRCUITS)} circuits "
          f"x {len(CONFIGS)} configs\n", flush=True)

    for name in CIRCUITS:
        qc, dag, ncx = load(name)
        rev_dag = circuit_to_dag(qc.reverse_ops())
        print(f"── {name} ({ncx} CX) ──", flush=True)
        for label, cfg in CONFIGS:
            t0 = time.time()
            m = best_of_three(cfg, qc, dag, rev_dag, arch)
            dt = time.time() - t0
            exp = EXPECTED_FULL[name]
            if m is None:
                bad = f"  [!! published routes this at {exp} -- DRIFT]" \
                      if label == "full" else ""
                print(f"   {label:<16s} ABORT  ({dt:.0f}s){bad}", flush=True)
                rows.append(dict(circuit=name, cx=ncx, config=label,
                                 aborted=True, time_s=round(dt, 1)))
            else:
                note = ""
                if label == "full":
                    note = ("  [matches published]" if m["eprs"] == exp
                            else f"  [!! published says {exp} — DRIFT]")
                print(f"   {label:<16s} EPR={m['eprs']:6d}  SWAP={m['ls']:6d}"
                      f"  ({dt:.0f}s){note}", flush=True)
                rows.append(dict(circuit=name, cx=ncx, config=label,
                                 eprs=m["eprs"], ls=m["ls"], aborted=False,
                                 time_s=round(dt, 1)))
            with open(OUT, "w") as f:
                json.dump(dict(meta=dict(
                    date="2026-07-31",
                    suite="64q",
                    arch="H-grid 2x3 of 4x4 (96 physical)",
                    router="dSE (dsabre_ext.dSABRE_BurstExt)",
                    hw="deadlock_limit=100, max_backup_attempts=100, "
                       "max_iterations=20000 (benchmark.py _HW_LARGE)",
                    protocol="best of 3 SabreLayout seeds, fwd->bwd->fwd",
                    purpose="Table V ablation on the full 9-circuit 64q suite, "
                            "one HardwareConfig throughout",
                ), results=rows), f, indent=1)
        print("", flush=True)

    print(f"Done.  Total {time.time()-t_start:.0f}s  ->  {OUT}", flush=True)


if __name__ == "__main__":
    main()
