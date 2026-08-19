"""dmax_counter.py — pytket-dqc's e-bit cost under a bounded EPR lifetime.

pytket-dqc charges one e-bit per edge of the Steiner tree spanning a
hyperedge's servers, and that link qubit then serves *every* gate of the
hyperedge downstream of it: an unbounded entanglement lifetime.  Bandini et
al. cap a pair at `D_max` gates.  This module re-scores an existing
`Distribution` under that cap.

The rule, per hyperedge:

  simple hyperedges (no H-embedding)
      A tree edge `e` serves every gate vertex of the hyperedge lying beyond
      it, i.e. on the far side from the shared qubit's home server; gates
      already at the home server need no link and age nothing.  Serving `g`
      gates on a pair good for `D_max` costs `ceil(g / D_max)`
      establishments.  At `D_max = inf` that is 1 per edge, which is
      `len(tree.edges)` -- pytket-dqc's own figure.

  embedded hyperedges (ALAP)
      The tool's own ALAP walk decides when a server joins `connected`,
      charging the path that connects it.  On top of that we age: each gate
      distributed to a server increments the counter of every live edge on
      its path home, and an edge whose counter passes `D_max` is
      re-established from its still-live neighbour for one more e-bit.  At
      `D_max = inf` no counter ever passes, so the walk's cost is the tool's.

`cost_under_dmax(dist, inf)` therefore reproduces `dist.cost()` exactly, and
`verify_reproduces_tool` asserts it hyperedge by hyperedge -- run that before
trusting any bounded number.
"""

from math import ceil, isclose, inf

import networkx as nx
from pytket import OpType
from pytket_dqc.utils import steiner_tree


def _establishments(gates_served: int, dmax) -> int:
    """Pairs consumed by one link serving `gates_served` gates."""
    if gates_served <= 0:
        return 1          # an edge on the tree is paid for even if idle
    if dmax == inf:
        return 1
    return ceil(gates_served / dmax)


def _simple_cost(dist, hyperedge, tree, dmax: float) -> int:
    """Cost of a hyperedge needing no H-embedding."""
    pmap = dist.placement.placement
    shared_qubit = dist.circuit.get_qubit_vertex(hyperedge)
    home = pmap[shared_qubit]
    gate_servers = [pmap[v] for v in hyperedge.vertices if v != shared_qubit]

    total = 0
    for u, v in tree.edges:
        stripped = tree.copy()
        stripped.remove_edge(u, v)
        near = nx.node_connected_component(stripped, home)
        beyond = sum(1 for s in gate_servers if s not in near)
        total += _establishments(beyond, dmax)
    return total


def _alap_cost(dist, hyperedge, tree, dmax: float) -> int:
    """Cost of an H-embedded hyperedge: the tool's ALAP walk, plus ageing.

    Mirrors ``Distribution.hyperedge_cost``'s embedded branch; the only
    additions are ``served`` and the re-establishment it triggers.
    """
    dist_circ = dist.circuit
    pmap = dist.placement.placement
    shared_qubit = dist_circ.get_qubit_vertex(hyperedge)
    home = pmap[shared_qubit]

    commands = dist_circ.get_hyperedge_subcircuit(hyperedge)
    vertices = sorted(hyperedge.vertices)
    assert vertices.pop(0) == shared_qubit

    cost = 0
    currently_h_embedding = False
    connected_servers = {home}
    served: dict[frozenset, int] = {}      # live edge -> gates since establishment

    def _edge_path(target):
        """Tree edges from home to `target`, as frozensets."""
        path = nx.shortest_path(tree, home, target)
        return [frozenset((a, b)) for a, b in zip(path, path[1:])]

    for command in commands:
        if command.op.type == OpType.H:
            currently_h_embedding = not currently_h_embedding

        elif command.op.type == OpType.Rz:
            assert (not currently_h_embedding
                    or isclose(command.op.params[0] % 1, 0)
                    or isclose(command.op.params[0] % 1, 1))

        elif command.op.type == OpType.CU1:
            if currently_h_embedding:
                q_vertices = [dist_circ.get_vertex_of_qubit(q)
                              for q in command.qubits]
                remote_vertex = [q for q in q_vertices if q != shared_qubit][0]
                remote_server = pmap[remote_vertex]
                connected_servers = connected_servers.intersection(
                    {home, remote_server})
                # Links to servers just disconnected are released.
                keep = set()
                for s in connected_servers:
                    keep.update(_edge_path(s))
                served = {e: n for e, n in served.items() if e in keep}

            elif command != dist_circ.get_gate_of_vertex(vertices[0]):
                pass                                   # D-embedded elsewhere
            else:
                gate_vertex = vertices.pop(0)
                gate_server = pmap[gate_vertex]

                if gate_server not in connected_servers:
                    best_path = None
                    for c_server in connected_servers:
                        p = nx.shortest_path(tree, c_server, gate_server)
                        if best_path is None or len(p) < len(best_path):
                            best_path = p
                    assert best_path is not None
                    connected_servers.update(best_path)
                    cost += len(best_path) - 1
                    for a, b in zip(best_path, best_path[1:]):
                        served[frozenset((a, b))] = 0

                # The gate is served by every live link on its path home.
                for e in _edge_path(gate_server):
                    if e not in served:
                        continue
                    served[e] += 1
                    if dmax != inf and served[e] > dmax:
                        cost += 1                      # re-establish the pair
                        served[e] = 1

    assert not vertices
    return cost


def hyperedge_cost_under_dmax(dist, hyperedge, dmax: float) -> int:
    if hyperedge.weight != 1:
        raise ValueError("only weight-1 hyperedges are supported")
    pmap = dist.placement.placement
    servers = [pmap[v] for v in hyperedge.vertices]
    tree = steiner_tree(dist.network.get_server_nx(), servers)
    if dist.circuit.requires_h_embedded_cu1(hyperedge):
        return _alap_cost(dist, hyperedge, tree, dmax)
    return _simple_cost(dist, hyperedge, tree, dmax)


def cost_under_dmax(dist, dmax: float) -> int:
    """Total e-bits for `dist` when one pair serves at most `dmax` gates."""
    return sum(hyperedge_cost_under_dmax(dist, h, dmax)
               for h in dist.circuit.hyperedge_list)


def costs_under_dmax(dist, dmaxes) -> dict:
    """`cost_under_dmax` for several bounds, walking the hyperedges once.

    The Steiner tree and the embedding test are the expensive parts and do
    not depend on `dmax`, so they are shared across the sweep.
    """
    pmap = dist.placement.placement
    server_graph = dist.network.get_server_nx()
    totals = {d: 0 for d in dmaxes}
    for h in dist.circuit.hyperedge_list:
        if h.weight != 1:
            raise ValueError("only weight-1 hyperedges are supported")
        tree = steiner_tree(server_graph, [pmap[v] for v in h.vertices])
        embedded = dist.circuit.requires_h_embedded_cu1(h)
        for d in dmaxes:
            totals[d] += (_alap_cost(dist, h, tree, d) if embedded
                          else _simple_cost(dist, h, tree, d))
    return totals


def verify_reproduces_tool(dist) -> tuple[bool, list]:
    """At dmax=inf the counter must equal pytket-dqc's own, hyperedge-wise."""
    bad = []
    for h in dist.circuit.hyperedge_list:
        mine, theirs = hyperedge_cost_under_dmax(dist, h, inf), dist.hyperedge_cost(h)
        if mine != theirs:
            bad.append((h, mine, theirs))
    return (not bad), bad
