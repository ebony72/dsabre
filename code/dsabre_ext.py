"""dSABRE_BFSExt — dSABRE with a BFS-layer inter-core extended set ("bfs-ext").

Motivation
----------
`General_dSABRE_Router._top_ext` fills the inter-core extended set E in pure
topological order, which interleaves gates from all wires arbitrarily.  For
circuits with a "burst" structure — a run of gates on the same qubit wire
that becomes inter-core after one teleportation — many of those follow-on
gates are pushed past the E window by unrelated gates on other wires, so the
lookahead underestimates the benefit of teleporting there.

Algorithm
---------
Instead of topological order, E is expanded layer by layer by BFS over the
remaining DAG:

  1. Take each node's remaining in-degree (the number of unexecuted op
     predecessors) from the router's maintained table.
  2. Commit the front layer: decrement in-degrees of their successors.
  3. Collect the next BFS layer: nodes whose remaining in-degree hits zero.
  4. Within each layer, gates that share a qubit with the front layer
     ("burst-relevant" gates) are placed first; the rest follow in
     arrival order.
  5. Repeat until E reaches the requested size.

This is the production construction: every headline number in the paper is
`dSABRE_BFSExt` (called "dSE" in benchmark columns and "\\dSABRE{}" in the
paper).  `_top_ext` is run only where the two constructions are compared.

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

* Cost: O(|F| + L) per call, because the front layer seeds the BFS, every
  layer but the last is emitted in full (so their widths sum to at most L),
  and a layer is a matching on qubits so the truncated last one is O(n).
  This holds only because in-degrees are *carried* between calls by the
  router; rebuilding them here, as this file used to, made every call
  Theta(N) and the router quadratic in gate count.
"""

from router import General_dSABRE_Router


class dSABRE_BFSExt(General_dSABRE_Router):
    """dSABRE with BFS-layer extended set and burst-qubit priority."""

    def _build_inter_ext(self, dag, front, size):
        # `_inter_ext` memoises this on the working DAG's generation; override
        # the builder, not the memo.
        return self._bfs_ext(dag, front, size)

    def _bfs_ext(self, dag, front, size):
        """BFS-layer fill of the inter-core extended set.

        `self._indeg` is the router's maintained in-degree table and must not
        be mutated: this call's decrements go into the local `dec` overlay, so
        the effective in-degree of a node is `_indeg[nid] - dec[nid]`.
        """
        front_nids   = {n._node_id for n in front}
        front_qubits = {q for n in front for q in n.qargs}

        indeg = self._indeg
        dec = {}
        done = set()

        def commit(node):
            done.add(node._node_id)
            for succ in dag.successors(node):
                sid = getattr(succ, '_node_id', None)
                if sid is not None and getattr(succ, 'qargs', None) and sid not in done:
                    dec[sid] = dec.get(sid, 0) + 1

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
                            and indeg[sid] - dec.get(sid, 0) == 0):
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


# Back-compatible alias: the class was called dSABRE_BurstExt until 2026-08-07.
# Benchmark drivers and ablation scripts across the repo import that name.
dSABRE_BurstExt = dSABRE_BFSExt
