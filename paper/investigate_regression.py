"""
investigate_regression.py — A4: why does dSABRE-base regress on ae/qft (25q)?

Compares routing behaviour for:
  - ae_25q  (TS=26 EPR, dS=40 EPR, dSE=27 EPR)
  - qft_25q (TS=46 EPR, dS=58 EPR, dSE=35 EPR)

For each circuit we route under TS's Hungarian layout and report:
  - Per-pass EPR / SWAP counts (single-pass + 2-pass results separately)
  - Mechanism instrumentation (relief_picks, backup_activations, force_make_room)
  - Number of inter-core gates in the front layer initially
  - Whether the regression is a high-EPR-count problem or a quality-of-teleport problem

Hypothesis to test
------------------
H1: dS picks too many teleports because the topological extended set
    over-represents far-future gates that dilute the immediate dF term.
H2: dS gets stuck in deadlock loops on these circuits and the recovery
    burns extra EPRs.
H3: dS selects suboptimal teleport directions (always moves q1 when q2
    would be better) due to scoring asymmetry.

The instrumentation will tell us which (if any) of these is the cause.
"""

import sys, os, json, time
sys.setrecursionlimit(50000)
sys.path.insert(0, os.path.dirname(__file__))

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler import PassManager

from architecture import build_b_grid_architecture
from config import HardwareConfig
from router import General_dSABRE_Router
from dsabre_ext import dSABRE_BurstExt
from layout import run_passes
from benchmark import run_telesabre, p2v_to_layout, SUITES

CIRCUITS = ["ae", "qft"]
SUITE = SUITES["25q"]


def load(circuit):
    qf = os.path.join(SUITE["circuit_dir"], circuit + SUITE["suffix"])
    qc = QuantumCircuit.from_qasm_file(qf)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc), qf


def main():
    arch = SUITE["arch"]
    hw   = SUITE["hw"]

    print(f"\n{'='*92}")
    print(f"  Investigating dS regression on 25q (vs TS and dSE)")
    print(f"{'='*92}\n")

    for cname in CIRCUITS:
        qc, dag, qf = load(cname)
        n2q = sum(1 for _ in dag.two_qubit_ops())
        ts_result = run_telesabre(qf, SUITE["ts_dev"])
        if ts_result is None:
            print(f"  {cname}: TS aborted; cannot compare")
            continue
        layout = p2v_to_layout(ts_result["p2v"], dag)
        ts_epr = ts_result["eprs"]

        # Front layer composition under TS layout
        front_2q = [n for n in dag.front_layer() if len(n.qargs) == 2]
        inter_count = sum(
            1 for n in front_2q
            if arch.core_of(layout.get(n.qargs[0]))
            != arch.core_of(layout.get(n.qargs[1]))
        )

        print(f"── {cname}  (qubits={qc.num_qubits}, 2q gates={n2q}) ──")
        print(f"     TS layout: front 2q gates = {len(front_2q)} (inter-core: {inter_count})")
        print(f"     TS EPRs   = {ts_epr}")

        for name, router_cls in [("dS", General_dSABRE_Router), ("dSE", dSABRE_BurstExt)]:
            router = router_cls(arch, hw)
            # Single pass
            t0 = time.time()
            m1, _ = router.route(dag, layout)
            t1 = time.time() - t0
            # Two-pass via run_passes (the benchmark configuration)
            mp = run_passes(router, dag, layout, 2)
            print(f"     {name:<4} 1-pass: EPR={m1['eprs']:>3}  SWAP={m1['ls']:>4}  "
                  f"R-cand={m1.get('relief_candidates',0):>3}  R-pick={m1.get('relief_picks',0):>2}  "
                  f"Backup={m1.get('backup_activations',0):>2}  F-room={m1.get('force_make_room',0):>2}  "
                  f"t={t1:.2f}s")
            if mp:
                print(f"     {name:<4} 2-pass: EPR={mp['eprs']:>3}  SWAP={mp['ls']:>4}  "
                      f"R-pick={mp.get('relief_picks',0):>2}  Backup={mp.get('backup_activations',0):>2}")
        print()


if __name__ == "__main__":
    main()
