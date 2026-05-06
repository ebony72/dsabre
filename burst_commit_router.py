"""
dSABRE_BurstCommit — dSABRE with post-hoc harvest after each teleport.

Idea (Option B — "burst-commit"):
  Vanilla dSABRE applies a teleport and then returns to the top of the
  outer loop, where a fresh drain phase eventually executes newly-unblocked
  gates.  This wastes one full iteration per "free" gate.

  This router overrides _post_teleport to immediately drain all front-layer
  gates that become executable right after the teleport (same core, adjacent
  physical qubits).  The drain repeats until no more such gates remain, so a
  single teleport can harvest an entire burst of dependent gates in one shot.

  No new scoring term, no new hyperparameter, no ECH.
"""

from router import General_dSABRE_Router


class dSABRE_BurstCommit(General_dSABRE_Router):

    def _post_teleport(self, wdag, l2p, p2l, node_decay, metrics):
        arch = self.arch
        drained = True
        while drained:
            drained = False
            for node in self._front_2q(wdag):
                p1, p2 = l2p[node.qargs[0]], l2p[node.qargs[1]]
                if arch.core_of(p1) == arch.core_of(p2) and arch.Gr.has_edge(p1, p2):
                    wdag.remove_op_node(node)
                    node_decay[p1] = node_decay[p2] = 1.0
                    metrics["burst_harvested"] = metrics.get("burst_harvested", 0) + 1
                    drained = True
                    break  # front layer changed; restart scan
