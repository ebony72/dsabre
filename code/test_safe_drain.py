"""Regression test for `_safe_drain`, safe mode's escape hatch.

`_safe_drain` is the least-exercised path in the router by construction:
nothing reaches it until an instance is budget-starved enough that ordinary
routing and deadlock recovery have both run out, so no benchmark suite covered
it.  `random_100` (36 109 CX at 67% fill) was the first instance that did, and
it exposed a bug that turned into an `ITERATION_LIMIT` abort -- in safe mode,
where aborts are supposed to be impossible.

The bug: the drain stopped as soon as `_front_2q` came back empty.  Retiring
the front layer's 1-qubit gates can expose MORE 1-qubit gates, so the front is
briefly free of 2-qubit gates while the DAG is far from empty; the drain
reported failure on a circuit it had merely not finished draining.

The circuit below stacks three 1-qubit layers between each 2-qubit layer so
that state is hit repeatedly.  Run directly or under pytest.
"""
import copy
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from architecture import build_h_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BFSExt
from router import _RetiringDAG


def _alternating_circuit(n=12, blocks=6, singles=3):
    qc = QuantumCircuit(n)
    for _ in range(blocks):
        for _ in range(singles):
            for q in range(n):
                qc.h(q)
        for q in range(0, n - 1, 2):
            qc.cx(q, q + 1)
        for q in range(1, n - 1, 2):
            qc.cx(q, q + 1)
    return qc


def test_safe_drain_retires_everything():
    arch = build_h_grid_architecture(2, 3, 4)
    K, kap = arch.num_cores, len(arch.core_qubits(0))
    qc = _alternating_circuit()
    dag = circuit_to_dag(qc)

    # Two qubits per core, so most 2q gates are remote and the drain has to
    # use the guaranteed transaction rather than plain intra-core routing.
    phys = [arch.core_qubits(c)[i] for i in range(2) for c in range(K)]
    l2p = {q: p for q, p in zip(qc.qubits, phys)}
    p2l = {p: None for p in arch.Gr.nodes}
    for lq, p in l2p.items():
        p2l[p] = lq

    router = dSABRE_BFSExt(arch, HardwareConfig(
        safe_mode=True, tier1_floor=2, deadlock_limit=100,
        max_iterations=20000))
    real = copy.deepcopy(dag)
    router._init_dag_state(real)
    router._n_snapshots = router._n_rollbacks = router._n_wdag_rebuilds = 0
    router._t_snapshot = router._t_rollback = router._t_wdag_rebuild = 0.0
    wdag = _RetiringDAG(real, router)
    metrics = {"ls": 0, "teles": 0, "catcomms": 0, "eprs": 0, "cost": 0,
               "1q_gates": 0, "aborted": False, "backup_activations": 0,
               "force_make_room": 0, "relay_hops": 0, "safe_routes": 0,
               "safe_route_failed": 0, "trace": None}

    assert router._safe_drain(wdag, l2p, p2l, metrics), \
        "_safe_drain reported failure on a drainable circuit"
    assert router._remaining == 0, \
        f"{router._remaining} operations left after the drain"
    assert metrics["safe_route_failed"] == 0, \
        "a guaranteed transaction failed"

    occ = {c: 0 for c in range(K)}
    for p, lq in p2l.items():
        if lq is not None:
            occ[arch.core_of(p)] += 1
    free = [kap - occ[c] for c in range(K)]
    assert min(free) >= 2, f"capacity invariant broken on exit: {free}"
    return metrics, free


if __name__ == "__main__":
    m, free = test_safe_drain_retires_everything()
    print(f"drained to empty: eprs={m['eprs']} 1q={m['1q_gates']} "
          f"safe_routes={m['safe_routes']} failed={m['safe_route_failed']} "
          f"force_make_room={m['force_make_room']} exit_free={free}")
    print("PASS")
