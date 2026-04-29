"""
Ablation study: vary weight_burst in {0, 1, 2, 4, 8} on AE layout A.
w_beta=0 disables burst scoring, recovering plain teleport-distance behaviour.
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(__file__))

from architecture import build_b_grid_architecture
from burst_router import BurstDSABRE
from config import HardwareConfig
from main import locality_aware_layout, _run_passes

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

CIRCUIT = os.path.expanduser(
    "~/Documents/telesabre/circuits/qasm_25/"
    "ae_nativegates_ibm_qiskit_opt3_25.qasm"
)
R, S, M     = 2, 2, 4          # 2×2 grid of 4×4 cores → 64 physical qubits
LAYOUT_PASSES = 2
NUM_TRIALS    = 5               # layout-A random trials per w_beta value
BURST_NORM    = 8

def main():
    qc   = QuantumCircuit.from_qasm_file(CIRCUIT)
    dag  = circuit_to_dag(qc)
    arch = build_b_grid_architecture(r=R, s=S, m=M)
    hw   = HardwareConfig()

    print(f"{'w_beta':>8}  {'EPR_min':>8}  {'EPR_mean':>10}  {'burst_saves_mean':>17}")
    print("-" * 52)

    results = {}
    for wb in [0, 1, 2, 4, 8]:
        eprs   = []
        bsaves = []
        for trial in range(NUM_TRIALS):
            router = BurstDSABRE(arch, hw,
                                  weight_burst=float(wb),
                                  max_burst_normaliser=BURST_NORM)
            layout = locality_aware_layout(dag, arch, rng=random.Random(trial))
            m      = _run_passes(router, dag, layout, LAYOUT_PASSES)
            eprs.append(m["eprs"])
            bsaves.append(m.get("burst_saves", 0))

        best = min(eprs)
        mean = sum(eprs) / len(eprs)
        bs_m = sum(bsaves) / len(bsaves)
        results[wb] = (best, mean)
        print(f"{wb:>8}  {best:>8}  {mean:>10.1f}  {bs_m:>17.1f}")

    print()
    print("pgfplots coordinates (w_beta, EPR_min):")
    coords = " ".join(f"({wb},{v[0]})" for wb, v in sorted(results.items()))
    print(f"  {coords}")

if __name__ == "__main__":
    main()
