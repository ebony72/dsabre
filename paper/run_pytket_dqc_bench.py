"""
Benchmark pytket-dqc (HypergraphPartitioning) on the 25q, 36q, and 64q circuit
suites using the same architectures as the main dSABRE paper.

  25q / 36q : B-grid 2×2 4×4 (4 QPUs)
  64q       : H-grid 2×3 4×4 (6 QPUs)

pytket-dqc uses gate teleportation (detached gates); each non-local CX gate
that cannot be merged with another costs 1 e-bit.  The cost() method on a
Distribution reports this e-bit count, which is directly comparable to the
EPR-pair metric used by dSABRE and TeleSABRE.

pytket-dqc assigns qubits to QPUs statically via KaHyPar hypergraph
partitioning and does not model intra-QPU SWAP routing; the SWAP column is
therefore omitted.

Usage:
    python run_pytket_dqc_bench.py          # all three suites
    python run_pytket_dqc_bench.py 25       # 25q only
    python run_pytket_dqc_bench.py 25 36    # specific suites
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
import time

logging.disable(logging.WARNING)

from pytket.qasm import circuit_from_qasm
from pytket_dqc.allocators import HypergraphPartitioning
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.utils import DQCPass

NUM_SEEDS = 5   # run this many random seeds; report best


# ── Architecture helpers ─────────────────────────────────────────────────────

def _even_partition(n: int, k: int) -> list[list[int]]:
    """Partition logical qubits 0..n-1 into k roughly-equal groups."""
    base, rem = divmod(n, k)
    groups, start = [], 0
    for i in range(k):
        end = start + base + (1 if i < rem else 0)
        groups.append(list(range(start, end)))
        start = end
    return groups


def bgrid_network(n_qubits: int) -> NISQNetwork:
    """B-grid 2×2 (4 QPUs) for 25q and 36q circuits."""
    # QPU layout:  0 1
    #              2 3
    coupling = [[0, 1], [2, 3], [0, 2], [1, 3]]
    slots = _even_partition(n_qubits, 4)
    return NISQNetwork(coupling, {i: slots[i] for i in range(4)})


def hgrid_network(n_qubits: int) -> NISQNetwork:
    """H-grid 2×3 (6 QPUs) for 64q circuits."""
    # QPU layout:  0 1 2
    #              3 4 5
    coupling = [[0, 1], [1, 2], [3, 4], [4, 5], [0, 3], [1, 4], [2, 5]]
    slots = _even_partition(n_qubits, 6)
    return NISQNetwork(coupling, {i: slots[i] for i in range(6)})


# ── Per-circuit runner ───────────────────────────────────────────────────────

def run_circuit(qasm_path: str, network: NISQNetwork) -> tuple[int, int, float]:
    """
    Return (best_ebit, cx_count, elapsed_seconds).
    Runs NUM_SEEDS seeds; reports the minimum e-bit count found.
    """
    circ = circuit_from_qasm(qasm_path, maxwidth=128)
    DQCPass().apply(circ)
    n_cx = circ.n_2qb_gates()

    alloc = HypergraphPartitioning()
    best_ebit = None
    t0 = time.perf_counter()
    for seed in range(NUM_SEEDS):
        try:
            dist = alloc.allocate(circ, network, seed=seed)
            cost = dist.cost()
            if best_ebit is None or cost < best_ebit:
                best_ebit = cost
        except Exception:
            pass
    elapsed = time.perf_counter() - t0

    return best_ebit if best_ebit is not None else -1, n_cx, elapsed


# ── Suite runner ─────────────────────────────────────────────────────────────

SUITES = {
    "25": {
        "dir": os.path.expanduser("~/Documents/telesabre/circuits/qasm_25"),
        "suffix": "_nativegates_ibm_qiskit_opt3_25.qasm",
        "network_fn": bgrid_network,
        "title": "25-qubit suite  (B-grid 2×2 4×4, 4 QPUs)",
    },
    "36": {
        "dir": os.path.expanduser("~/Documents/telesabre/circuits/qasm_36"),
        "suffix": "_nativegates_ibm_qiskit_opt3_36.qasm",
        "network_fn": bgrid_network,
        "title": "36-qubit suite  (B-grid 2×2 4×4, 4 QPUs)",
    },
    "64": {
        "dir": os.path.expanduser("~/Documents/telesabre/circuits/qasm_64"),
        "suffix": "_nativegates_ibm_qiskit_opt3_64.qasm",
        "network_fn": hgrid_network,
        "title": "64-qubit suite  (H-grid 2×3 4×4, 6 QPUs)",
    },
}


def run_suite(key: str) -> list[dict]:
    cfg = SUITES[key]
    files = sorted(glob.glob(os.path.join(cfg["dir"], "*.qasm")))
    if not files:
        print(f"No .qasm files in {cfg['dir']}")
        return []

    print(f"\n{cfg['title']}")
    print(f"  HypergraphPartitioning, best of {NUM_SEEDS} seeds")
    hdr = f"  {'circuit':<12}  {'CX':>6}  {'e-bits':>7}  {'time':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    ebits_list = []
    for qf in files:
        cname = os.path.basename(qf).replace(cfg["suffix"], "")
        from pytket.qasm import circuit_from_qasm as _cq
        tmp = _cq(qf, maxwidth=128)
        network = cfg["network_fn"](tmp.n_qubits)

        ebit, cx, elapsed = run_circuit(qf, network)
        ebits_list.append(ebit)
        rows.append({"circuit": cname, "cx": cx, "py_ebits": ebit, "py_time": elapsed})
        ebit_str = str(ebit) if ebit >= 0 else "FAIL"
        print(f"  {cname:<12}  {cx:>6}  {ebit_str:>7}  {elapsed:>5.1f}s")

    valid = [e for e in ebits_list if e >= 0]
    if valid:
        from math import prod
        gmean = prod(valid) ** (1.0 / len(valid))
        print(f"  {'gmean':<12}  {'':>6}  {gmean:>7.1f}")

    return rows


def main() -> None:
    keys = sys.argv[1:] if len(sys.argv) > 1 else ["25", "36", "64"]
    unknown = [k for k in keys if k not in SUITES]
    if unknown:
        print(f"Unknown suite(s): {unknown}. Choose from: 25, 36, 64")
        sys.exit(1)

    all_results = {}
    for k in keys:
        all_results[k] = run_suite(k)
    print()

    out_path = os.path.join(os.path.dirname(__file__), "results_pytket_dqc_bench.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
