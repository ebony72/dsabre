"""Deterministic temporal locality-greedy initial layout for dSABRE.

Native port of PORTER's ``compile_porter.build_initial_layout`` to the
dSABRE interfaces: takes a Qiskit DAGCircuit and a DistributedArchitecture,
returns a {qiskit Qubit: physical int} layout dict (same shape as
``layout.locality_aware_layout`` / ``sabre_locked_boundary_layout``).

Three deterministic steps, no seed:
  1. Every 2q gate contributes gamma**layer to its pair's weight, where
     layer is the gate's ASAP level in the 2q-only DAG (gamma=0.99).
  2. Qubits are visited in first-2q-gate-appearance order; each goes to the
     core of its heaviest already-placed neighbour if that core has headroom
     above ``min_free_per_core``, else to the core with most remaining room.
  3. Within each core, qubits by descending total weight take physical slots
     by ascending intra-core centrality (sum of local distances).

Candidate drop-in for dsabre/code/layout.py — costs ~1 ms, so it can also
just be appended to the sabre_locked_boundary_layout best-of portfolio.
"""
from collections import defaultdict


def temporal_locality_layout(dag, arch, min_free_per_core: int = 2,
                             gamma: float = 0.99) -> dict:
    qubits = list(dag.qubits)
    num_logical = len(qubits)
    qidx = {q: i for i, q in enumerate(qubits)}

    # 1. temporally discounted pair weights + first-appearance order
    level = defaultdict(int)
    pair_weight = defaultdict(float)
    first_seen = {}
    order = 0
    for node in dag.topological_op_nodes():
        if len(node.qargs) != 2:
            continue
        a, b = qidx[node.qargs[0]], qidx[node.qargs[1]]
        lay = max(level[a], level[b])
        level[a] = level[b] = lay + 1
        pair_weight[(min(a, b), max(a, b))] += gamma ** lay
        for q in (a, b):
            if q not in first_seen:
                first_seen[q] = order
                order += 1

    nbr_weight = {q: {} for q in range(num_logical)}
    for (a, b), w in pair_weight.items():
        nbr_weight[a][b] = w
        nbr_weight[b][a] = w

    # 2. greedy core assignment with per-core free-slot reservation
    core_order = list(range(arch.num_cores))
    remaining = {c: max(0, len(arch.core_qubits(c)) - min_free_per_core)
                 for c in core_order}
    if sum(remaining.values()) < num_logical:
        raise ValueError(
            f"cannot reserve {min_free_per_core} free slots per core "
            f"for {num_logical} logical qubits")

    visit = sorted(range(num_logical), key=lambda q: first_seen.get(q, 10**9 + q))
    core_of = {}
    core_to_logicals = {c: [] for c in core_order}
    for q in visit:
        best_core, best_w = None, -1.0
        for nbr, w in nbr_weight[q].items():
            c = core_of.get(nbr)
            if c is not None and remaining[c] > 0 and w > best_w:
                best_core, best_w = c, w
        if best_core is None:
            best_core = max((c for c in core_order if remaining[c] > 0),
                            key=lambda c: (remaining[c], -c))
        core_of[q] = best_core
        core_to_logicals[best_core].append(q)
        remaining[best_core] -= 1

    # 3. within-core: heaviest logical -> most central slot
    layout = {}
    for c in core_order:
        physical = list(arch.core_qubits(c))
        dist = arch.intra_dist[c]
        slots = sorted(physical,
                       key=lambda p: (sum(dist[p][o] for o in physical if o != p), p))
        logicals = sorted(core_to_logicals[c],
                          key=lambda q: (-sum(nbr_weight[q].values()), q))
        for p, q in zip(slots, logicals):
            layout[qubits[q]] = p
    return layout


if __name__ == "__main__":
    # Cross-check against PORTER's own builder on all three suites.
    # SFC lives beside the dsabre repo: <pyzoo>/SFC and <pyzoo>/dsabre/code.
    import os
    import sys
    _HERE = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, os.path.join(_HERE, "..", "..", "SFC"))
    import compare_routers as cr
    import compile_porter as cdr

    for suite in ["25", "64", "80"]:
        spec = cr.SUITES[suite]
        _, hw = cdr.hardware_for_name(spec["arch_name"])
        arch = cr.sfc_hardware_to_dsabre_architecture(hw)
        cr.verify_topology_match(hw, arch)
        for fname in spec["files"]:
            qasm_path = spec["circuit_dir"] / fname
            circuit, num_logical = cdr.parse_qasm2(qasm_path)
            qc, dag = cr.load_qasm(qasm_path)
            ref_lay = cdr.build_initial_layout(circuit, hw, num_logical,
                                               min_free_per_core=2)
            ref = {dag.qubits[q]: p
                   for q, p in ref_lay.logical_to_physical.items()
                   if q < num_logical}
            mine = temporal_locality_layout(dag, arch, min_free_per_core=2)
            status = "MATCH" if mine == ref else "DIFFER"
            n_diff = sum(1 for q in ref if mine.get(q) != ref[q])
            print(f"[{suite}q] {fname.split('_nativegates')[0]:12s} {status}"
                  + (f" ({n_diff}/{num_logical} differ)" if status == "DIFFER" else ""),
                  flush=True)
