"""
regen_ablation_corners.py — Regenerate Tables VIII (mechanism ablation)
and IX (pass-count study) using the corners-removed best-of-3 SabreLayout
protocol (same as the main results in Tables IV/V/VI), so all ablation
baselines align with the headline numbers.

Suites:
  - 25q: B-grid 2x2 4x4
  - 64q: H-grid 2x3 4x4

Mechanism configs (one knob off per row):
  full, no_lookahead (w_e=0), no_hop_gain (w_h=0),
  no_cap_penalty (c_p=0), no_relief

Pass-count configs: 1, 2, 3, 4 forward passes (no fwd/bwd/fwd here —
mirrors the original Table IX which sweeps pure-forward pass count).

Output: code/results/regen_ablation_corners.json
"""

import sys, os, glob, json, time
from math import prod
from dataclasses import replace

sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from architecture import build_b_grid_architecture, build_h_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_passes, run_sabre_passes


# ── Suites ────────────────────────────────────────────────────────────────────
SUITES = {
    "25q": dict(
        arch=build_b_grid_architecture(r=2, s=2, m=4),
        circuit_dir=os.path.expanduser("~/Documents/telesabre/circuits/qasm_25"),
        suffix="_nativegates_ibm_qiskit_opt3_25.qasm",
    ),
    "64q": dict(
        arch=build_h_grid_architecture(r=2, s=3, m=4),
        circuit_dir=os.path.expanduser("~/Documents/telesabre/circuits/qasm_64"),
        suffix="_nativegates_ibm_qiskit_opt3_64.qasm",
    ),
}

BASE_CFG = HardwareConfig()

MECH_CONFIGS = [
    ("full",           BASE_CFG),
    ("no_lookahead",   replace(BASE_CFG, weight_extended=0.0)),
    ("no_hop_gain",    replace(BASE_CFG, hop_gain=0.0)),
    ("no_cap_penalty", replace(BASE_CFG, cap_penalty=0.0)),
    ("no_relief",      replace(BASE_CFG, enable_congestion_relief=False)),
]

PASS_COUNTS = [1, 2, 3, 4]


def gmean(lst):
    lst = [x for x in lst if x is not None and x > 0]
    return prod(lst) ** (1 / len(lst)) if lst else float("nan")


def load_circuits(circuit_dir, suffix):
    circs = []
    for qf in sorted(glob.glob(os.path.join(circuit_dir, "*.qasm"))):
        cname = os.path.basename(qf).replace(suffix, "")
        qc = QuantumCircuit.from_qasm_file(qf)
        qc = qc.remove_final_measurements(inplace=False)
        qc = PassManager([RemoveBarriers()]).run(qc)
        dag = circuit_to_dag(qc)
        circs.append(dict(name=cname, qc=qc, dag=dag,
                          cx=sum(1 for _ in dag.two_qubit_ops())))
    return circs


def best_of_three(cfg, circuit, arch, passes=None, sabre_schedule=False):
    """Route a circuit under cfg, trying 3 SabreLayout seeds, returning best EPR.
    If sabre_schedule=True, use fwd->bwd->fwd (matches headline).
    Otherwise, run `passes` pure-forward passes (for pass-count study)."""
    layouts = sabre_locked_boundary_layout(circuit["qc"], circuit["dag"], arch, seed=0)
    router  = dSABRE_BurstExt(arch, cfg)
    best = None
    rev_dag = None
    if sabre_schedule:
        rev_dag = circuit_to_dag(circuit["qc"].reverse_ops())
    for layout in layouts:
        if sabre_schedule:
            m = run_sabre_passes(router, circuit["dag"], rev_dag, layout)
        else:
            m = run_passes(router, circuit["dag"], layout, passes)
        if m and not m.get("aborted"):
            if best is None or m["eprs"] < best["eprs"]:
                best = m
    return best


def run_mech(suite_label, circuits, arch):
    print(f"\n── Mechanism ablation [{suite_label}] (corners removed, best-of-3, fwd->bwd->fwd) ──", flush=True)
    out = []
    for label, cfg in MECH_CONFIGS:
        t0 = time.time()
        rows = []
        for c in circuits:
            m = best_of_three(cfg, c, arch, sabre_schedule=True)
            if m is None:
                rows.append(dict(circuit=c["name"], cx=c["cx"], aborted=True))
            else:
                rows.append(dict(circuit=c["name"], cx=c["cx"],
                                 eprs=m["eprs"], ls=m["ls"], aborted=False))
        eprs = [r["eprs"] for r in rows if not r.get("aborted")]
        lss  = [r["ls"]   for r in rows if not r.get("aborted")]
        gE, gS = gmean(eprs), gmean(lss)
        print(f"  {label:<16s}  gmEPR={gE:8.2f}  gmSWAP={gS:8.1f}  ({time.time()-t0:.1f}s)", flush=True)
        out.append(dict(label=label, suite=suite_label,
                        gmean_eprs=gE, gmean_ls=gS, circuits=rows))
    return out


def run_passes_sweep(suite_label, circuits, arch):
    print(f"\n── Pass-count sweep [{suite_label}] (corners removed, best-of-3) ──", flush=True)
    out = []
    for npasses in PASS_COUNTS:
        t0 = time.time()
        rows = []
        for c in circuits:
            m = best_of_three(BASE_CFG, c, arch, passes=npasses, sabre_schedule=False)
            if m is None:
                rows.append(dict(circuit=c["name"], cx=c["cx"], aborted=True))
            else:
                rows.append(dict(circuit=c["name"], cx=c["cx"],
                                 eprs=m["eprs"], ls=m["ls"], aborted=False))
        eprs = [r["eprs"] for r in rows if not r.get("aborted")]
        lss  = [r["ls"]   for r in rows if not r.get("aborted")]
        gE, gS = gmean(eprs), gmean(lss)
        print(f"  passes={npasses}  gmEPR={gE:8.2f}  gmSWAP={gS:8.1f}  ({time.time()-t0:.1f}s)", flush=True)
        out.append(dict(passes=npasses, suite=suite_label,
                        gmean_eprs=gE, gmean_ls=gS, circuits=rows))
    return out


def main():
    overall_t0 = time.time()
    results = {"meta": {"date": time.strftime("%Y-%m-%d"),
                        "protocol": "SabreLayout corners-removed, best of 3 seeds, 3 forward passes"},
               "mechanism_ablation": {},
               "passes_sweep": {}}

    for suite_label, info in SUITES.items():
        print(f"\n════ Suite {suite_label} ════", flush=True)
        circuits = load_circuits(info["circuit_dir"], info["suffix"])
        print(f"  {len(circuits)} circuits", flush=True)

        results["mechanism_ablation"][suite_label] = run_mech(suite_label, circuits, info["arch"])
        results["passes_sweep"][suite_label]       = run_passes_sweep(suite_label, circuits, info["arch"])

        outpath = os.path.join(_HERE, "results", "regen_ablation_corners.json")
        with open(outpath, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  → saved partial results to {outpath}", flush=True)

    print(f"\nDone.  Total: {time.time()-overall_t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
