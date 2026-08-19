"""
dSABRE_BurstExt — dSABRE with a BFS-layer extended lookahead set.

Motivation
----------
Vanilla dSABRE builds its extended-set E in pure topological order, which
interleaves gates from all wires arbitrarily.  For circuits with a "burst"
structure — a sequence of gates on the same qubit wire that becomes
inter-core after one teleportation — many of those follow-on gates are
pushed past the E window by unrelated gates on other wires, so dSABRE's
lookahead underestimates the benefit of teleporting there.

Algorithm
---------
Instead of topological order, we expand E layer by layer using a BFS over
the remaining DAG:

  1. Pre-compute the total op in-degree of every node (number of 2q/1q
     predecessors that have not yet been executed).
  2. Commit the front layer: decrement in-degrees of their successors.
  3. Collect the next BFS layer: nodes whose remaining in-degree hits zero.
  4. Within each layer, gates that share a qubit with the front layer
     ("burst-relevant" gates) are placed first; the rest follow in
     arrival order.
  5. Repeat until E reaches the requested size.

Design notes
------------
* Node identity: Qiskit's Rust-backed DAGCircuit returns a *fresh* Python
  wrapper object on every accessor call (front_layer, topological_op_nodes,
  successors, …), so id(node) is NOT a stable identity across calls.
  All tracking uses node._node_id, a stable integer assigned by the Rust layer.

* Correctness on multi-pass circuits: following quantum_successors along qubit
  wires without respecting cross-wire dependencies fails on circuits that visit
  the same wires twice (e.g. wstate's descending then ascending CX chain).
  The BFS approach only admits a gate once all its predecessors are committed,
  so second-pass gates never appear before first-pass gates complete.
"""

from _legacy_router_submitted import General_dSABRE_Router


class dSABRE_BurstExt(General_dSABRE_Router):
    """dSABRE with BFS-layer extended set and burst-qubit priority."""

    def _extended_2q(self, dag, front, size):
        front_nids   = {n._node_id for n in front}
        front_qubits = {q for n in front for q in n.qargs}

        # Total op in-degree for every node (counts only op-node predecessors).
        remaining = {
            n._node_id: sum(1 for p in dag.predecessors(n) if getattr(p, 'qargs', None))
            for n in dag.topological_op_nodes()
        }

        done = set()  # _node_ids already committed

        def commit(node):
            done.add(node._node_id)
            for succ in dag.successors(node):
                sid = getattr(succ, '_node_id', None)
                if sid is not None and getattr(succ, 'qargs', None) and sid not in done:
                    remaining[sid] -= 1

        for fn in front:
            commit(fn)

        ext     = []
        current = list(front)
        layer_depth = 1  # front layer = depth 0; first extended layer = depth 1

        while len(ext) < size:
            # Collect all nodes whose remaining in-degree just reached zero.
            nxt_map = {}
            for node in current:
                for succ in dag.successors(node):
                    sid = getattr(succ, '_node_id', None)
                    if (sid is not None
                            and getattr(succ, 'qargs', None)
                            and sid not in done
                            and sid not in nxt_map
                            and remaining[sid] == 0):
                        nxt_map[sid] = succ

            if not nxt_map:
                break

            next_layer = list(nxt_map.values())
            for n in next_layer:
                commit(n)

            # Within this BFS layer, gates sharing a qubit with the front layer
            # get soft priority (they are most likely to benefit immediately from
            # a teleportation that resolves a front-layer gate).
            layer_2q = [n for n in next_layer if len(n.qargs) == 2]
            priority = [n for n in layer_2q if front_qubits & set(n.qargs)]
            rest     = [n for n in layer_2q if n not in priority]

            for n in priority + rest:
                if len(ext) >= size:
                    break
                ext.append((n, layer_depth))  # all gates in same BFS layer share depth

            current = next_layer
            layer_depth += 1

        return ext
