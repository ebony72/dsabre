"""
probe_compile_time.py — per-seed and total compilation time for dSABRE on the
64-qubit suite.

`benchmark.py` records `time_s` as the compile time of the *best* SabreLayout
seed, not the cost of the protocol that produced the reported EPR count.  That
understates dSABRE against `pytket-dqc`, whose recorded time is the whole
best-of-5 search.  The comparison in the appendix timing table needs the total.

The seed counts each tool actually requires differ, and the difference is real
rather than conventional:

  TeleSABRE   deterministic under optimize_initial with the Hungarian layout --
              all three seeds return identical EPR on all nine circuits -- so
              one invocation suffices and best-of-3 buys it nothing.
  dSABRE      three SabreLayout seeds give genuinely different results
              (AE: 242 / 258 / 263), so all three are needed.
  pytket-dqc  best of five seeds, per its own convention.

Output: code/results/results_compile_time_64q.json (written incrementally)

Usage:  python3 code/probe_compile_time.py
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.abspath(__file__))
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
OUT = os.path.join(_HERE, "results", "results_compile_time_64q.json")

# Cheapest first so a killed run still leaves usable rows.
CIRCUITS = ["ghz", "graphstate", "qpeexact", "ae", "qft", "random",
            "qaoa", "qnn", "multiplier"]

EXPECTED_CX = {"ghz": 63, "graphstate": 64, "qpeexact": 2139, "ae": 1962,
               "qft": 1966, "random": 1627, "qaoa": 3920, "qnn": 8126,
               "multiplier": 13040}
EXPECTED_EPR = {"ghz": 9, "graphstate": 16, "qpeexact": 271, "ae": 242,
                "qft": 204, "random": 721, "qaoa": 529, "qnn": 508,
                "multiplier": 1514}

HW = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                    max_iterations=20000)


def load(name):
    qc = QuantumCircuit.from_qasm_file(os.path.join(CIRCUIT_DIR, name + SUFFIX))
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    dag = circuit_to_dag(qc)
    ncx = sum(1 for _ in dag.two_qubit_ops())
    if ncx != EXPECTED_CX[name]:
        raise SystemExit(f"ABORT: {name} has {ncx} CX, expected "
                         f"{EXPECTED_CX[name]}; see benchmark_circuits/README.md on version drift.")
    return qc, dag, ncx


def main():
    arch = build_h_grid_architecture(r=2, s=3, m=4)
    rows = []
    print("dSABRE per-seed compile time, 64q suite "
          "(3 SabreLayout seeds, fwd->bwd->fwd each)\n", flush=True)
    for name in CIRCUITS:
        qc, dag, ncx = load(name)
        rev = circuit_to_dag(qc.reverse_ops())
        layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
        router = dSABRE_BurstExt(arch, HW)
        per_seed, eprs, best = [], [], None
        for layout in layouts:
            t0 = time.perf_counter()
            m = run_sabre_passes(router, dag, rev, layout)
            per_seed.append(round(time.perf_counter() - t0, 2))
            if m and not m.get("aborted"):
                eprs.append(m["eprs"])
                if best is None or m["eprs"] < best:
                    best = m["eprs"]
            else:
                eprs.append(None)
        total = round(sum(per_seed), 2)
        ok = (best == EXPECTED_EPR[name])
        note = "" if ok else f"  [!! expected EPR {EXPECTED_EPR[name]}, got {best}]"
        print(f"  {name:12} CX={ncx:6}  seeds={per_seed}  total={total:9.2f}s"
              f"  best_seed={min(per_seed):8.2f}s  EPR={eprs}{note}", flush=True)
        rows.append(dict(circuit=name, cx=ncx, per_seed_s=per_seed,
                         total_s=total, best_seed_s=min(per_seed),
                         eprs=eprs, best_epr=best, reproduces=ok))
        with open(OUT, "w") as f:
            json.dump(dict(meta=dict(
                date="2026-07-31", suite="64q",
                arch="H-grid 2x3 of 4x4 (96 physical)",
                router="dSE (dsabre_ext.dSABRE_BurstExt)",
                hw="deadlock_limit=100, max_backup_attempts=100, "
                   "max_iterations=20000",
                note="total_s is the cost of the full best-of-3 protocol; "
                     "benchmark.py's time_s records best_seed_s only",
            ), results=rows), f, indent=1)
    print(f"\nDone -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
