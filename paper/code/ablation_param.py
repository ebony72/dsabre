"""
ablation_param.py — Option C parameter validation for dSABRE (BFS extended set).

Two-level study on the 25q suite:
  1. Mechanism ablation  — discrete on/off for each scoring component
  2. Sensitivity sweep   — one-at-a-time variation of weight_extended and cap_penalty

Reuses TeleSABRE layouts stored in results/results_25q.json — no TS re-runs.
Results saved to results/ablation_param.json.

Usage:
    cd paper && python3 ablation_param.py
"""

import sys, os, json, glob, time
from math import prod
from dataclasses import replace

sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler import PassManager

from architecture import build_b_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from layout import run_passes

# ── Architecture & defaults ────────────────────────────────────────────────────
ARCH        = build_b_grid_architecture(r=2, s=2, m=4)
BASE_CFG    = HardwareConfig()           # all defaults
LAYOUT_PASSES = 2
CIRCUIT_DIR = os.path.expanduser("~/Documents/telesabre/circuits/qasm_25")
SUFFIX      = "_nativegates_ibm_qiskit_opt3_25.qasm"
RESULTS_IN  = os.path.join(_HERE, "results", "results_25q.json")
RESULTS_OUT = os.path.join(_HERE, "results", "ablation_param.json")

# ── Configurations ─────────────────────────────────────────────────────────────
# Level 1: mechanism ablation (each disables exactly one component)
MECH_CONFIGS = [
    ("full",              BASE_CFG),
    ("no_lookahead",      replace(BASE_CFG, weight_extended=0.0)),
    ("no_hop_gain",       replace(BASE_CFG, hop_gain=0.0)),
    ("no_cap_penalty",    replace(BASE_CFG, cap_penalty=0.0)),
    ("no_relief",         replace(BASE_CFG, enable_congestion_relief=False)),
]

# Level 2: continuous sensitivity (one-at-a-time, others at default)
SWEEP_CONFIGS = [
    # weight_extended sweep
    *[(f"w_ext={w}", replace(BASE_CFG, weight_extended=w))
      for w in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]],
    # cap_penalty sweep
    *[(f"cap={c}", replace(BASE_CFG, cap_penalty=c))
      for c in [0.0, 5.0, 15.0, 30.0, 60.0]],
]


def gmean(lst):
    lst = [x for x in lst if x is not None and x > 0]
    return prod(lst) ** (1 / len(lst)) if lst else float("nan")


def pct(a, b):
    if b == 0 or b != b: return "   ---"
    return f"{100*(a-b)/b:+.1f}%"


def load_circuits():
    """Load QASM files and TS layouts from existing results JSON."""
    with open(RESULTS_IN) as f:
        existing = json.load(f)["results"]
    # Build name→p2v map from existing results
    ts_layouts = {r["circuit"]: r["ts"]["p2v"] for r in existing if r.get("ts")}

    circuits = []
    for qf in sorted(glob.glob(os.path.join(CIRCUIT_DIR, "*.qasm"))):
        cname = os.path.basename(qf).replace(SUFFIX, "")
        if cname not in ts_layouts:
            print(f"  [skip {cname}: no TS layout in results_25q.json]", flush=True)
            continue
        qc  = QuantumCircuit.from_qasm_file(qf)
        qc  = qc.remove_final_measurements(inplace=False)
        qc  = PassManager([RemoveBarriers()]).run(qc)
        dag = circuit_to_dag(qc)
        # Reconstruct layout dict: virtual qubit → physical index
        p2v   = ts_layouts[cname]
        qubits = dag.qubits
        layout = {qubits[v]: p for p, v in enumerate(p2v)
                  if v != -1 and v < len(qubits)}
        circuits.append(dict(name=cname, dag=dag, layout=layout,
                             cx=sum(1 for _ in dag.two_qubit_ops())))
    return circuits


def run_config(label, cfg, circuits):
    """Route all circuits under cfg; return list of per-circuit results."""
    router  = dSABRE_BurstExt(ARCH, cfg)
    results = []
    for c in circuits:
        m = run_passes(router, c["dag"], c["layout"], LAYOUT_PASSES)
        if m and not m.get("aborted"):
            results.append(dict(circuit=c["name"], cx=c["cx"],
                                eprs=m["eprs"], ls=m["ls"], aborted=False))
        else:
            results.append(dict(circuit=c["name"], cx=c["cx"], aborted=True))
    return results


def summarise(results):
    eprs = [r["eprs"] for r in results if not r.get("aborted")]
    lss  = [r["ls"]   for r in results if not r.get("aborted")]
    return gmean(eprs), gmean(lss)


def print_table(title, configs_results, baseline_epr):
    print(f"\n{'━'*62}", flush=True)
    print(f"  {title}", flush=True)
    print(f"{'━'*62}", flush=True)
    print(f"  {'Config':<22}  {'gmEPR':>8}  {'gmSWAP':>8}  {'ΔEPR':>8}", flush=True)
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}", flush=True)
    for label, results in configs_results:
        ge, gl = summarise(results)
        print(f"  {label:<22}  {ge:>8.1f}  {gl:>8.1f}  {pct(ge, baseline_epr):>8}", flush=True)
    print(flush=True)


def main():
    t0 = time.time()

    print("Loading circuits and TS layouts ...", flush=True)
    circuits = load_circuits()
    print(f"  {len(circuits)} circuits loaded.", flush=True)

    # ── Level 1: mechanism ablation ────────────────────────────────────────────
    print("\nLevel 1: mechanism ablation", flush=True)
    mech_results = []
    for label, cfg in MECH_CONFIGS:
        t1 = time.time()
        res = run_config(label, cfg, circuits)
        ge, _ = summarise(res)
        print(f"  {label:<22}  gmEPR={ge:.1f}  ({time.time()-t1:.1f}s)", flush=True)
        mech_results.append((label, res))

    baseline_epr, _ = summarise(mech_results[0][1])   # "full" config

    # ── Level 2: sensitivity sweeps ───────────────────────────────────────────
    print("\nLevel 2: sensitivity sweeps", flush=True)
    sweep_results = []
    for label, cfg in SWEEP_CONFIGS:
        t1 = time.time()
        res = run_config(label, cfg, circuits)
        ge, _ = summarise(res)
        print(f"  {label:<22}  gmEPR={ge:.1f}  ({time.time()-t1:.1f}s)", flush=True)
        sweep_results.append((label, res))

    # ── Print formatted summary tables ────────────────────────────────────────
    print_table("Mechanism ablation (25q)", mech_results, baseline_epr)

    # Split sweep by parameter
    w_ext_rows = [(l, r) for l, r in sweep_results if l.startswith("w_ext=")]
    cap_rows   = [(l, r) for l, r in sweep_results if l.startswith("cap=")]
    print_table("Sensitivity: weight_extended (25q)", w_ext_rows, baseline_epr)
    print_table("Sensitivity: cap_penalty (25q)",     cap_rows,   baseline_epr)

    # ── Save results ──────────────────────────────────────────────────────────
    def cfg_as_dict(cfg):
        return {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}

    payload = {
        "meta": {
            "date":   time.strftime("%Y-%m-%d"),
            "suite":  "25q",
            "router": "dSABRE_BurstExt (dsabre_ext.py)",
            "layout_passes": LAYOUT_PASSES,
            "note":   "TS layouts reused from results_25q.json",
        },
        "mechanism_ablation": [
            {"label": label, "config": cfg_as_dict(cfg),
             "circuits": res, "gmean_eprs": summarise(res)[0], "gmean_ls": summarise(res)[1]}
            for (label, cfg), (_, res) in zip(MECH_CONFIGS, mech_results)
        ],
        "sensitivity_sweeps": [
            {"label": label, "config": cfg_as_dict(cfg),
             "circuits": res, "gmean_eprs": summarise(res)[0], "gmean_ls": summarise(res)[1]}
            for (label, cfg), (_, res) in zip(SWEEP_CONFIGS, sweep_results)
        ],
    }
    with open(RESULTS_OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved → {RESULTS_OUT}", flush=True)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
