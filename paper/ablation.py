"""
ablation.py — A1: isolate the contribution of each dSABRE mechanism.

Configurations under test
-------------------------
  full            dSABRE  (relief ✓, recovery ✓, topo-E)
  no_relief       dSABRE  (relief ✗, recovery ✓, topo-E)
  no_recovery     dSABRE  (relief ✓, recovery ✗, topo-E)
  bare            dSABRE  (relief ✗, recovery ✗, topo-E)
  full_bfs        dSE     (relief ✓, recovery ✓, BFS-E)

For each circuit, all five variants run under TeleSABRE's Hungarian layout
(best of 3 seeds, 2 forward passes).  Reports EPRs and aborts.

Output:
  results/ablation.json  — full per-circuit per-variant table
  stdout                 — formatted summary table

Usage:
  python ablation.py --suite 25      # 25q only
  python ablation.py --suite all     # all (default)
"""

import sys, os, json, glob, time, argparse
from math import prod

sys.setrecursionlimit(50000)
sys.path.insert(0, os.path.dirname(__file__))

from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler import PassManager
from qiskit import QuantumCircuit

from architecture import build_b_grid_architecture, build_h_grid_architecture
from config import HardwareConfig
from router import General_dSABRE_Router
from dsabre_ext import dSABRE_BurstExt
from layout import run_passes
from benchmark import (
    SUITES, run_telesabre, p2v_to_layout, LAYOUT_PASSES, _RESULTS_DIR,
)


VARIANTS = [
    ("full",        "dS",  dict(enable_congestion_relief=True,  enable_deadlock_recovery=True)),
    ("no_relief",   "dS",  dict(enable_congestion_relief=False, enable_deadlock_recovery=True)),
    ("no_recovery", "dS",  dict(enable_congestion_relief=True,  enable_deadlock_recovery=False)),
    ("bare",        "dS",  dict(enable_congestion_relief=False, enable_deadlock_recovery=False)),
    ("full_bfs",    "dSE", dict(enable_congestion_relief=True,  enable_deadlock_recovery=True)),
]


def load_qasm(path):
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def run_suite(suite_name):
    s        = SUITES[suite_name]
    arch     = s["arch"]
    base_hw  = s["hw"]
    suffix   = s["suffix"]
    ts_dev   = s["ts_dev"]

    qasm_files = sorted(glob.glob(os.path.join(s["circuit_dir"], "*.qasm")))
    if not qasm_files:
        return []

    print(f"\n{'═'*92}")
    print(f"  Ablation — {suite_name}")
    print(f"{'═'*92}")
    hdr = f"{'circuit':<12}  {'TS_epr':>6}"
    for name, _, _ in VARIANTS:
        hdr += f"  {name:>11}"
    print(hdr)
    print("─" * len(hdr))

    records = []

    for qf in qasm_files:
        cname = os.path.basename(qf).replace(suffix, "")
        qc, dag  = load_qasm(qf)

        ts_result = run_telesabre(qf, ts_dev)
        if ts_result is None:
            row = f"{cname:<12}  {'---':>6}"
            for _ in VARIANTS:
                row += f"  {'---':>11}"
            print(row)
            continue
        layout = p2v_to_layout(ts_result["p2v"], dag)
        if len(layout) < qc.num_qubits:
            continue

        row = f"{cname:<12}  {ts_result['eprs']:>6}"
        rec = dict(circuit=cname, suite=suite_name, qubits=qc.num_qubits,
                   ts_epr=ts_result["eprs"], variants={})

        for name, router_kind, flags in VARIANTS:
            hw = HardwareConfig(
                cost_local_swap = base_hw.cost_local_swap,
                cost_teleport   = base_hw.cost_teleport,
                weight_extended = base_hw.weight_extended,
                lookahead_size  = base_hw.lookahead_size,
                lookahead_decay = base_hw.lookahead_decay,
                cap_penalty     = base_hw.cap_penalty,
                capacity_threshold = base_hw.capacity_threshold,
                hop_gain        = base_hw.hop_gain,
                relief_bonus    = base_hw.relief_bonus,
                demand_lookahead= base_hw.demand_lookahead,
                demand_threshold= base_hw.demand_threshold,
                congestion_threshold = base_hw.congestion_threshold,
                relief_space_req= base_hw.relief_space_req,
                max_iterations  = base_hw.max_iterations,
                deadlock_limit  = base_hw.deadlock_limit,
                max_backup_attempts = base_hw.max_backup_attempts,
                **flags,
            )
            router_cls = dSABRE_BurstExt if router_kind == "dSE" else General_dSABRE_Router
            router = router_cls(arch, hw)
            t0 = time.time()
            m  = run_passes(router, dag, layout, LAYOUT_PASSES)
            wall = time.time() - t0
            if m and not m.get("aborted"):
                rec["variants"][name] = dict(
                    eprs=m["eprs"], ls=m["ls"], time_s=round(m["compile_time"], 3),
                    aborted=False, wall_s=round(wall, 1),
                    relief_picks=m.get("relief_picks", 0),
                    backup_activations=m.get("backup_activations", 0),
                )
                row += f"  {m['eprs']:>11}"
            else:
                rec["variants"][name] = dict(aborted=True, wall_s=round(wall, 1),
                                              relief_picks=m.get("relief_picks", 0) if m else 0,
                                              backup_activations=m.get("backup_activations", 0) if m else 0)
                row += f"  {'ABORT':>11}"

        print(row)
        records.append(rec)

    # Geometric mean per variant (excluding aborts)
    print("─" * len(hdr))
    def gmean(lst):
        lst = [x for x in lst if x is not None and x > 0]
        return prod(lst) ** (1 / len(lst)) if lst else float("nan")
    summary = f"{'gmean':<12}  "
    summary += f"{gmean([r['ts_epr'] for r in records]):>6.1f}"
    for name, _, _ in VARIANTS:
        eprs = [r["variants"].get(name, {}).get("eprs") for r in records
                if not r["variants"].get(name, {}).get("aborted")]
        summary += f"  {gmean(eprs):>11.1f}"
    print(summary)
    print()

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["25", "36", "64", "all"], default="all")
    args = ap.parse_args()
    suites = []
    if args.suite in ("25", "all"): suites.append("25q")
    if args.suite in ("36", "all"): suites.append("36q")
    if args.suite in ("64", "all"): suites.append("64q")

    all_records = []
    t0 = time.time()
    for sname in suites:
        all_records += run_suite(sname)
    out = os.path.join(_RESULTS_DIR, "ablation.json")
    with open(out, "w") as f:
        json.dump({"variants": [v[0] for v in VARIANTS],
                   "records":  all_records}, f, indent=2)
    print(f"Saved → {out}  (wall {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
