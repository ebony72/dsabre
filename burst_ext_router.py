"""
dSABRE_BurstExt — dSABRE with a wire-balanced (round-robin) extended set.

Idea (Option A):
  Vanilla dSABRE builds the extended-set lookahead E in pure topological
  order, which interleaves wires arbitrarily.  For a teleport that primarily
  helps pending gates on one wire (the "burst" case), those gates are often
  pushed beyond E's window by gates on unrelated wires.

  This router replaces _extended_2q with a BFS-layer expansion that gives
  front-qubit gates soft priority within each layer.

  Design (two bugs fixed from the original wire-walk approach):

  (1) node identity — Qiskit's Rust-backed DAGCircuit creates a fresh Python
      wrapper object on every node-accessor call, so id(node) is not a
      stable identity.  All tracking uses node._node_id (a stable integer).

  (2) wire reuse — the original wire-walk followed quantum_successors along
      each qubit wire, which crosses DAG layers: for circuits with two passes
      over the same wires (e.g. w-state) a gate from the second pass would
      appear first in the extended set, ahead of gates that must execute
      before it.  A BFS layer expansion avoids this by only adding a gate
      when all its op-predecessors in the remaining DAG have been committed.

  Algorithm:
    1. BFS-layer expansion: pre-compute total op in-degrees, commit the
       front layer, then collect successive DAG layers (nodes whose remaining
       in-degree drops to zero).
    2. Within each layer, gates that touch a qubit already in the front layer
       are placed first (burst-relevant priority), then the rest follow in
       arrival order.
    3. Stop once the pool reaches `size` 2q gates.

  No new scoring term, no new hyperparameter, no ECH at routing time.
"""

from router import General_dSABRE_Router


class dSABRE_BurstExt(General_dSABRE_Router):

    def _extended_2q(self, dag, front, size):
        front_nids    = {n._node_id for n in front}
        front_qubits  = {q for n in front for q in n.qargs}

        # Pre-compute total op in-degrees (DAGInNode/DAGOutNode have qargs=None).
        remaining = {}
        for n in dag.topological_op_nodes():
            remaining[n._node_id] = sum(
                1 for p in dag.predecessors(n)
                if getattr(p, 'qargs', None)
            )

        done = set()   # committed _node_ids

        def commit(node):
            done.add(node._node_id)
            for succ in dag.successors(node):
                sid = getattr(succ, '_node_id', None)
                if sid is not None and getattr(succ, 'qargs', None) and sid not in done:
                    remaining[sid] -= 1

        for fn in front:
            commit(fn)

        # ── BFS layer expansion ───────────────────────────────────────────────
        ext     = []
        current = list(front)

        while len(ext) < size:
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

            # Within this layer give burst-qubit gates soft priority.
            layer_2q = [n for n in next_layer if len(n.qargs) == 2]
            priority = [n for n in layer_2q if front_qubits & set(n.qargs)]
            rest     = [n for n in layer_2q if n not in priority]
            for n in priority + rest:
                if len(ext) >= size:
                    break
                ext.append(n)

            current = next_layer

        return ext
