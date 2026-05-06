"""
Initial layout strategies and multi-pass routing for dSABRE.

locality_aware_layout  — partition circuit qubits by interaction graph
                          community, then assign within each core by centrality.
_run_passes            — run up to N forward routing passes with early stopping.
"""

import random as _random
import time

import networkx as nx

from architecture import DistributedArchitecture


# ── Interaction graph ──────────────────────────────────────────────────────────

def _interaction_graph(dag) -> nx.Graph:
    """Weighted undirected graph: nodes = logical qubits, edges weighted by CX count."""
    G = nx.Graph()
    for node in dag.topological_op_nodes():
        if len(node.qargs) == 2:
            q1, q2 = node.qargs[0], node.qargs[1]
            if G.has_edge(q1, q2):
                G[q1][q2]["weight"] += 1
            else:
                G.add_edge(q1, q2, weight=1)
    for q in dag.qubits:
        if q not in G:
            G.add_node(q)
    return G


def _partition_qubits(ig: nx.Graph, num_cores: int, core_capacity: int):
    """Partition logical qubits into `num_cores` groups of at most `core_capacity`.

    Uses greedy modularity community detection when edges exist; falls back to
    degree-sorted round-robin for circuits with no 2-qubit interactions.
    Overflow qubits (from oversized communities) are reassigned to the most
    affine under-capacity core.
    """
    qubits = list(ig.nodes())
    if ig.number_of_edges() == 0:
        parts = [[] for _ in range(num_cores)]
        for i, q in enumerate(qubits):
            parts[i % num_cores].append(q)
        return parts

    try:
        from networkx.algorithms import community as nx_comm
        raw = list(nx_comm.greedy_modularity_communities(
            ig, weight="weight", cutoff=num_cores, best_n=num_cores
        ))
    except Exception:
        sorted_q = sorted(qubits, key=lambda q: ig.degree(q, weight="weight"), reverse=True)
        raw = [[] for _ in range(num_cores)]
        for i, q in enumerate(sorted_q):
            raw[i % num_cores].append(q)
        return raw

    parts = sorted([list(p) for p in raw], key=len, reverse=True)
    overflow = []
    for p in parts:
        while len(p) > core_capacity:
            victim = min(p, key=lambda q: sum(
                ig[q][nb].get("weight", 1) for nb in ig.neighbors(q) if nb in p
            ))
            p.remove(victim)
            overflow.append(victim)

    while len(parts) < num_cores:
        parts.append([])

    for q in overflow:
        best_part, best_aff = 0, -1
        for i, p in enumerate(parts):
            if len(p) >= core_capacity:
                continue
            aff = sum(ig[q][nb].get("weight", 1) for nb in ig.neighbors(q) if nb in p)
            if aff > best_aff:
                best_aff, best_part = aff, i
        parts[best_part].append(q)

    return parts


def _topology_aware_core_assignment(lq_list, ig: nx.Graph, arch: DistributedArchitecture, core_id: int):
    """Assign logical qubits to physical slots within one core.

    Strategy
    --------
    1. Rank physical qubits by intra-core centrality (lower sum-of-distances
       to all other core qubits = more central = better for high-degree qubits).
    2. Rank logical qubits by weighted degree in the interaction graph.
    3. Greedy: most-connected logical → most-central physical.
    4. Pairwise local search (≤20 rounds): swap two assignments if it strictly
       reduces total weighted intra-core interaction distance.
    """
    if not lq_list:
        return {}

    dist     = arch.intra_dist[core_id]
    all_phys = arch.core_qubits(core_id)

    phys_by_centrality = sorted(all_phys, key=lambda p: sum(dist[p][q] for q in all_phys))
    lq_by_importance   = sorted(
        lq_list,
        key=lambda q: ig.degree(q, weight="weight") if q in ig else 0,
        reverse=True,
    )
    mapping = dict(zip(lq_by_importance, phys_by_centrality[:len(lq_by_importance)]))

    sub_nodes  = set(lq_list)
    core_edges = [(u, v, d.get("weight", 1))
                  for u, v, d in ig.edges(data=True)
                  if u in sub_nodes and v in sub_nodes]
    if not core_edges:
        return mapping

    adj = {lq: [] for lq in lq_list}
    for u, v, w in core_edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    def swap_delta(a, b):
        pa, pb = mapping[a], mapping[b]
        d = 0.0
        for nbr, w in adj[a]:
            if nbr != b:
                d += w * (dist[pb][mapping[nbr]] - dist[pa][mapping[nbr]])
        for nbr, w in adj[b]:
            if nbr != a:
                d += w * (dist[pa][mapping[nbr]] - dist[pb][mapping[nbr]])
        return d

    for _ in range(20):
        improved = False
        lqs = list(mapping)
        for i in range(len(lqs)):
            for j in range(i + 1, len(lqs)):
                if swap_delta(lqs[i], lqs[j]) < 0:
                    mapping[lqs[i]], mapping[lqs[j]] = mapping[lqs[j]], mapping[lqs[i]]
                    improved = True
        if not improved:
            break

    return mapping


def locality_aware_layout(dag, arch: DistributedArchitecture, rng=None) -> dict:
    """Build an initial layout by partitioning qubits along the interaction graph.

    1. Compute the interaction graph (weighted by CX count).
    2. Partition logical qubits into `num_cores` communities.
    3. Assign each community to one core using topology-aware placement.
    4. Fill any remaining unassigned qubits into random free slots.

    Parameters
    ----------
    dag  : Qiskit DAGCircuit
    arch : DistributedArchitecture
    rng  : random.Random instance for reproducibility (None → module-level random)

    Returns
    -------
    {logical_qubit: physical_qubit}
    """
    if rng is None:
        rng = _random
    qubits_per_core = len(arch.core_qubits(0))
    ig    = _interaction_graph(dag)
    parts = _partition_qubits(ig, arch.num_cores, qubits_per_core)
    layout = {}
    for core_id, group in enumerate(parts):
        layout.update(_topology_aware_core_assignment(group, ig, arch, core_id))
    assigned = set(layout.values())
    free = [p for p in arch.data_qubits if p not in assigned]
    rng.shuffle(free)
    fp = iter(free)
    for lq in dag.qubits:
        if lq not in layout:
            layout[lq] = next(fp)
    return layout


# ── Multi-pass routing ─────────────────────────────────────────────────────────

def run_passes(router, dag, initial_layout: dict, layout_passes: int):
    """Run up to `layout_passes` forward routing passes with early stopping.

    After each pass the output layout becomes the input for the next pass.
    Stops early if EPR count does not improve.  Returns the metrics dict
    from the best pass, with `compile_time` set to total wall-clock time.
    Returns None if the first pass aborts.
    """
    current_layout = initial_layout
    prev_eprs      = float("inf")
    best_metrics   = None
    total_time     = 0.0

    for _ in range(layout_passes):
        t0 = time.perf_counter()
        metrics, final_layout = router.route(dag, current_layout)
        total_time += time.perf_counter() - t0

        if metrics["aborted"]:
            break
        if best_metrics is None or metrics["eprs"] < best_metrics["eprs"]:
            best_metrics = dict(metrics)
        if metrics["eprs"] >= prev_eprs:
            break
        prev_eprs      = metrics["eprs"]
        current_layout = final_layout

    if best_metrics is not None:
        best_metrics["compile_time"] = total_time
    return best_metrics
