"""
regen_ablation_64q_full9.py — extend the 64q mechanism ablation from the
six-circuit common subset to the full nine-circuit suite.

`regen_ablation_corners.py` deliberately runs a fixed six-circuit core
(ae, ghz, graphstate, qft, qnn, random) so its geometric means stay
comparable with earlier ablation tables.  Table V of the paper now reports
the full suite instead, which needs qpeexact, qaoa and multiplier under
every configuration.

Two configurations already exist on all nine circuits from the same router
generation (unchanged since eb1826f, 2026-07-27) and are NOT re-run here:

    full          results/results_64q.json                (dSE column)
    no_hop_gain   results/nohop_regen/suites.log          (w_h = 0 sweep)

`full` is re-run on the three new circuits anyway, as a drift check against
results_64q.json: if it does not reproduce 271 / 529 / 1514 exactly, the
generations have diverged and the merge is invalid.

Cheap circuits run first so a partial result is usable; multiplier at
13,040 CX dominates wall time at roughly 20 min per configuration.

Output: code/results/results_ablate_mech_64q_extra.json  (written after
every circuit, not just at the end).

Usage:  python3 code/regen_ablation_64q_full9.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace

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
OUT = os.path.join(_HERE, "results", "results_ablate_mech_64q_multiplier.json")

# Cheapest first, so a killed job still leaves something usable.
NEW_CIRCUITS = ["multiplier"]

# Published dSE counts (results_64q.json) — the drift check for `full`.
EXPECTED_FULL = {"qpeexact": 271, "qaoa": 529, "multiplier": 1514}
EXPECTED_CX = {"qpeexact": 2139, "qaoa": 3920, "multiplier": 13040}

# benchmark.py routes the 64q suite under _HW_LARGE, not under bare defaults.
# regen_ablation_corners.py uses HardwareConfig(), whose limits happen never to
# bind on the six small circuits -- they reproduce results_64q.json exactly --
# but multiplier at 13,040 CX aborts under max_iterations=10000.  The ablation
# must use the same budget as the table it is ablating.
BASE_CFG = HardwareConfig(
    deadlock_limit=100, max_backup_attempts=100, max_iterations=20000
)
# "no_relief" removed: congestion relief was deleted from the router (see
# CHANGES_FROM_SUBMITTED.md), so this config no longer exists to ablate.
# EXPECTED_FULL below is STALE -- it was measured with relief active, and
# "full" now means something slightly different (relief-off, always).  Needs
# re-measuring against a fresh results_64q.json before this preflight can be
# trusted again.
CONFIGS = [
    ("full",           BASE_CFG),
    ("no_lookahead",   replace(BASE_CFG, weight_extended=0.0)),
    ("no_cap_penalty", replace(BASE_CFG, cap_penalty=0.0)),
]


def load(name):
    qf = os.path.join(CIRCUIT_DIR, name + SUFFIX)
    qc = QuantumCircuit.from_qasm_file(qf)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    dag = circuit_to_dag(qc)
    ncx = sum(1 for _ in dag.two_qubit_ops())
    if ncx != EXPECTED_CX[name]:
        raise SystemExit(
            f"ABORT: {name} has {ncx} CX, published table says "
            f"{EXPECTED_CX[name]}.  The circuit directory has drifted; see "
            f"CLAUDE.md on MQT Bench version drift.")
    return qc, dag, ncx


def best_of_three(cfg, qc, dag, rev_dag, arch):
    """Best EPR over 3 SabreLayout seeds, fwd->bwd->fwd — matches headline."""
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
    print(f"64q mechanism ablation, {len(NEW_CIRCUITS)} extra circuits "
          f"x {len(CONFIGS)} configs", flush=True)
    print(f"arch: H-grid 2x3 of 4x4 (96 physical)\n", flush=True)

    for name in NEW_CIRCUITS:
        qc, dag, ncx = load(name)
        rev_dag = circuit_to_dag(qc.reverse_ops())
        print(f"── {name} ({ncx} CX) ──", flush=True)
        for label, cfg in CONFIGS:
            t0 = time.time()
            m = best_of_three(cfg, qc, dag, rev_dag, arch)
            dt = time.time() - t0
            if m is None:
                bad = "  [!! published routes this at "\
                      f"{EXPECTED_FULL[name]} -- DRIFT]" if label == "full" else ""
                print(f"   {label:<16s} ABORT  ({dt:.0f}s){bad}", flush=True)
                rows.append(dict(circuit=name, cx=ncx, config=label,
                                 aborted=True, time_s=round(dt, 1)))
            else:
                note = ""
                if label == "full":
                    exp = EXPECTED_FULL[name]
                    note = ("  [matches published]" if m["eprs"] == exp
                            else f"  [!! published says {exp} — DRIFT]")
                print(f"   {label:<16s} EPR={m['eprs']:6d}  SWAP={m['ls']:6d}"
                      f"  ({dt:.0f}s){note}", flush=True)
                rows.append(dict(circuit=name, cx=ncx, config=label,
                                 eprs=m["eprs"], ls=m["ls"], aborted=False,
                                 time_s=round(dt, 1)))
            # Save after every single run, not just at the end.
            with open(OUT, "w") as f:
                json.dump(dict(meta=dict(
                    date="2026-07-31",
                    suite="64q",
                    arch="H-grid 2x3 of 4x4 (96 physical)",
                    router="dSE (dsabre_ext.dSABRE_BurstExt)",
                    protocol="best of 3 SabreLayout seeds, fwd->bwd->fwd",
                    purpose="extend Table V ablation from 6 to 9 circuits",
                    note="full/no_hop_gain on the other 6 circuits come from "
                         "regen_final/ablation.log and nohop_regen/suites.log, "
                         "same router generation (unchanged since 2026-07-27)",
                ), results=rows), f, indent=1)
        print("", flush=True)

    print(f"Done.  Total {time.time()-t_start:.0f}s  ->  {OUT}", flush=True)


if __name__ == "__main__":
    main()
