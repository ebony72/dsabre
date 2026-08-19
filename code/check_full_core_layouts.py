"""
check_full_core_layouts.py — does any experiment in the paper let SabreLayout
produce a "full core" (a core with zero free slots) in the INITIAL layout,
besides the 3x3-grid-of-3x3-cores stress architecture (64q_c33)?

By construction, `adaptive_corner_count()` (layout.py) picks k in {0..4}, the
largest value keeping usable-slot fill <= 80%, and `sabre_locked_boundary_layout`
then REMOVES the k most-remote corner qubits of every core from SabreLayout's
coupling map before it runs -- so whenever k >= 1, no core can ever end up
full: those k slots are structurally unreachable by the initial-layout pass.
Only k == 0 (no reservation at all) permits a core to be packed to capacity.

So the question "is a full core possible" reduces to "does k == 0 for this
architecture/qubit-count", which this script checks directly against the
architecture and qubit counts actually used in every table of the paper --
not just theoretically (via adaptive_corner_count), but empirically, by
building the real layout and reading per-core occupancy.

Usage:  python3 code/check_full_core_layouts.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from architecture import build_b_grid_architecture, build_h_grid_architecture, build_heavy_hex_architecture
from layout import adaptive_corner_count, sabre_locked_boundary_layout
from circuit_paths import circuits_path

CIRCUIT_DIR_25 = circuits_path("qasm_25")
CIRCUIT_DIR_36 = circuits_path("qasm_36")
CIRCUIT_DIR_64 = circuits_path("qasm_64")
CIRCUIT_DIR_100 = circuits_path("qasm_100")
CIRCUIT_DIR_200 = circuits_path("qasm_200")
CIRCUIT_DIR_360 = circuits_path("qasm_360")


def occ(layout, arch):
    o = [0] * arch.num_cores
    for p in layout.values():
        o[arch.core_of(p)] += 1
    return o


def report(name, arch, n_qubits, sample_layout_fn=None):
    spc = len(arch.core_qubits(0))
    k = adaptive_corner_count(arch, n_qubits)
    usable = arch.num_cores * (spc - k)
    fill = n_qubits / usable if usable else float("inf")
    full_possible = "YES (k=0)" if k == 0 else f"no (k={k} slots structurally reserved/core)"
    print(f"{name:<38} cores={arch.num_cores:>3} spc={spc:>3} nq={n_qubits:>4}  "
          f"k={k}  usable-fill={fill:5.1%}  full-core possible: {full_possible}")
    if sample_layout_fn is not None:
        try:
            qc, dag = sample_layout_fn()
            layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
            for i, L in enumerate(layouts):
                o = occ(L, arch)
                worst = max(o)
                flag = " <-- FULL CORE OBSERVED" if worst >= spc else ""
                print(f"    seed {i}: occupancy={o}  max={worst}/{spc}{flag}")
        except FileNotFoundError as e:
            print(f"    (circuit file not found, skipping empirical check: {e})")


def load_qasm(path):
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import RemoveBarriers
    qc = QuantumCircuit.from_qasm_file(path).remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def main():
    print("=" * 100)
    print("  MAIN-TEXT / SCALABILITY ARCHITECTURES (headline tables)")
    print("=" * 100)

    report("25q B-grid 2x2 of 4x4", build_b_grid_architecture(2, 2, 4), 25,
           lambda: load_qasm(os.path.join(CIRCUIT_DIR_25,
                             "ae_nativegates_ibm_qiskit_opt3_25.qasm")))
    report("36q B-grid 2x2 of 4x4", build_b_grid_architecture(2, 2, 4), 36,
           lambda: load_qasm(os.path.join(CIRCUIT_DIR_36,
                             "qaoa_nativegates_ibm_qiskit_opt3_36.qasm")))
    report("64q H-grid 2x3 of 4x4", build_h_grid_architecture(2, 3, 4), 64,
           lambda: load_qasm(os.path.join(CIRCUIT_DIR_64,
                             "multiplier_nativegates_ibm_qiskit_opt3_64.qasm")))
    report("100q H-grid 2x3 of 5x5", build_h_grid_architecture(2, 3, 5), 100,
           lambda: load_qasm(os.path.join(CIRCUIT_DIR_100,
                             "qft_nativegates_ibm_qiskit_opt3_100.qasm")))
    report("200q H-grid 3x4 of 5x5", build_h_grid_architecture(3, 4, 5), 200,
           lambda: load_qasm(os.path.join(CIRCUIT_DIR_200,
                             "qft_nativegates_ibm_qiskit_opt3_200.qasm")))
    report("360q H-grid 4x5 of 5x5", build_h_grid_architecture(4, 5, 5), 360,
           lambda: load_qasm(os.path.join(CIRCUIT_DIR_360,
                             "qft_nativegates_ibm_qiskit_opt3_360.qasm")))

    for topo in ("ring", "star"):
        report(f"64q heavy-hex {topo} (4x27q)", build_heavy_hex_architecture(4, topo), 64,
              lambda: load_qasm(os.path.join(CIRCUIT_DIR_64,
                                "multiplier_nativegates_ibm_qiskit_opt3_64.qasm")))

    print()
    print("=" * 100)
    print("  APPENDIX / ABLATION ARCHITECTURES (occupancy & freeslots stress tests)")
    print("=" * 100)

    report("64q_c33: 3x3 grid of 3x3", build_h_grid_architecture(3, 3, 3), 64,
          lambda: load_qasm(os.path.join(CIRCUIT_DIR_64,
                            "ghz_nativegates_ibm_qiskit_opt3_64.qasm")))
    report("64q_b243: B-grid 2x4 of 3x3", build_b_grid_architecture(2, 4, 3), 64,
          lambda: load_qasm(os.path.join(CIRCUIT_DIR_64,
                            "ghz_nativegates_ibm_qiskit_opt3_64.qasm")))
    report("25q_c223: B-grid 2x2 of 3x3", build_b_grid_architecture(2, 2, 3), 25,
          lambda: load_qasm(os.path.join(CIRCUIT_DIR_25,
                            "ghz_nativegates_ibm_qiskit_opt3_25.qasm")))
    report("25q_c133: B-grid 1x3 of 3x3", build_b_grid_architecture(1, 3, 3), 25,
          lambda: load_qasm(os.path.join(CIRCUIT_DIR_25,
                            "ghz_nativegates_ibm_qiskit_opt3_25.qasm")))

    print()
    print("=" * 100)
    print("  80q SUITE (hop-gain instrumentation, Section IV-D)")
    print("=" * 100)
    report("80q H-grid 2x5 of 4x4", build_h_grid_architecture(2, 5, 4), 80, None)


if __name__ == "__main__":
    main()
