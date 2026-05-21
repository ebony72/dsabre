"""
ablate_hop_gain_360q.py — hop-gain ablation on 360-qubit QFT.

Runs dSABRE_BurstExt twice (hop_gain ON vs OFF) on the same 360q QFT
circuit and 486-physical H-grid (r=2, s=3, m=9), with best-of-3
SabreLayout seeds each.  Compares EPR and local-SWAP counts to test
whether g_hop matters on large-core architectures.

Output: code/results/ablate_hop_gain_360q.json
"""

import sys, os, json, glob, time

sys.setrecursionlimit(100000)
sys.path.insert(0, os.path.dirname(__file__))

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler import PassManager

from architecture import build_h_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes

_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

QASM = os.path.expanduser(
    "~/Documents/telesabre/circuits/qasm_360/qft_nativegates_ibm_qiskit_opt3_360.qasm"
)


def run_variant(enable_hop, qc, dag, rev_dag, arch):
    hw = HardwareConfig(deadlock_limit=200, max_backup_attempts=200,
                        max_iterations=50000, enable_hop_gain=enable_hop)
    router = dSABRE_BurstExt(arch, hw)
    sl_layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
    best = None
    per_seed = []
    for i, layout in enumerate(sl_layouts):
        t0 = time.perf_counter()
        m  = run_sabre_passes(router, dag, rev_dag, layout)
        elapsed = time.perf_counter() - t0
        if m and not m.get("aborted"):
            eprs, ls = m["eprs"], m["ls"]
            per_seed.append(dict(seed=i, eprs=eprs, ls=ls, time_s=round(elapsed, 2)))
            print(f"    sl_seed{i}: EPR={eprs}, SWAP={ls} ({elapsed:.1f}s)", flush=True)
            if best is None or eprs < best["eprs"]:
                best = dict(eprs=eprs, ls=ls, layout=f"sl_seed{i}",
                            time_s=round(elapsed, 2))
        else:
            per_seed.append(dict(seed=i, aborted=True, time_s=round(elapsed, 2)))
            print(f"    sl_seed{i}: ABORTED ({elapsed:.1f}s)", flush=True)
    return dict(best=best, seeds=per_seed)


def main():
    print(f"{'='*60}\n  hop-gain ablation: 360q QFT on H-grid 2x3 9x9 (486 phys)\n{'='*60}",
          flush=True)
    arch = build_h_grid_architecture(r=2, s=3, m=9)

    matches = glob.glob(QASM)
    if not matches:
        print(f"ERROR: no QASM at {QASM}")
        sys.exit(1)
    qc  = QuantumCircuit.from_qasm_file(matches[0])
    qc  = qc.remove_final_measurements(inplace=False)
    qc  = PassManager([RemoveBarriers()]).run(qc)
    dag     = circuit_to_dag(qc)
    rev_dag = circuit_to_dag(qc.reverse_ops())
    n_cx = sum(1 for _ in dag.two_qubit_ops())
    print(f"  qubits={qc.num_qubits}, CX={n_cx}", flush=True)

    t0 = time.perf_counter()
    print("\n  → dSE with enable_hop_gain=True (baseline)", flush=True)
    on = run_variant(True, qc, dag, rev_dag, arch)
    print("\n  → dSE with enable_hop_gain=False", flush=True)
    off = run_variant(False, qc, dag, rev_dag, arch)
    total = time.perf_counter() - t0

    if on["best"] and off["best"]:
        delta = 100 * (off["best"]["eprs"] - on["best"]["eprs"]) / on["best"]["eprs"]
        summary = (f"\nResult: hop-gain ON={on['best']['eprs']} EPR, "
                   f"OFF={off['best']['eprs']} EPR  (Δ {delta:+.2f}% when ablated)")
    else:
        summary = "\nResult: one or both variants aborted; see seeds."
    print(summary, flush=True)
    print(f"Total wall time: {total:.1f}s", flush=True)

    out = dict(
        meta=dict(
            date=time.strftime("%Y-%m-%d"),
            circuit="qft", qubits=qc.num_qubits, cx=n_cx,
            arch="H-grid 2x3 9x9 (486 phys)",
            protocol="best of 3 SabreLayout seeds, fwd-bwd-fwd",
            total_wall_s=round(total, 1),
        ),
        hop_gain_on=on,
        hop_gain_off=off,
    )
    out_path = os.path.join(_RESULTS_DIR, "ablate_hop_gain_360q.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
