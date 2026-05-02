"""
dSABRE_BurstExt — dSABRE with a wire-balanced (round-robin) extended set.

Idea (Option A, the "simple" alternative to BurstDSABRE):
  Vanilla dSABRE builds the extended-set lookahead E in DAG topological order,
  which interleaves wires arbitrarily.  For a teleport that primarily helps
  pending gates on one wire (the "burst" case), those gates are often pushed
  beyond E's window by gates on unrelated wires.

  This router replaces _extended_2q with a wire-round-robin construction:
  for each qubit in the current front layer, walk that wire forward in the
  DAG and pick up its next 2q gates; cycle across wires until E is full.

  No new scoring term, no new hyperparameter, no ECH at routing time.
"""

from router import General_dSABRE_Router


class dSABRE_BurstExt(General_dSABRE_Router):

    def _extended_2q(self, wdag, front, size):
        front_set = {id(n) for n in front}

        # Collect qubits appearing in the current front layer.
        front_qubits = set()
        for n in front:
            for q in n.qargs:
                front_qubits.add(q)

        # Bucket every non-front 2q gate by each of its qubits, in DAG order.
        by_wire = {q: [] for q in front_qubits}
        for n in wdag.topological_op_nodes():
            if len(n.qargs) != 2 or id(n) in front_set:
                continue
            for q in n.qargs:
                if q in by_wire:
                    by_wire[q].append(n)

        # Round-robin across wires, skipping gates already added.
        seen = set()
        ext = []
        cursors = {q: 0 for q in front_qubits}
        active = list(front_qubits)
        while active and len(ext) < size:
            next_active = []
            for q in active:
                lst = by_wire[q]
                i = cursors[q]
                # Advance until we find an unseen gate on this wire.
                while i < len(lst) and id(lst[i]) in seen:
                    i += 1
                if i < len(lst):
                    n = lst[i]
                    seen.add(id(n))
                    ext.append(n)
                    cursors[q] = i + 1
                    next_active.append(q)
                    if len(ext) >= size:
                        break
            active = next_active

        return ext
