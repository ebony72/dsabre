"""
Initial layout strategies and multi-pass routing for dSABRE.

sabre_locked_boundary_layout — SabreLayout on arch graph with corner nodes removed.
sabre_completed_boundary_layout — same, on a dist-k completed core graph, optionally
                               repacked to even per-core occupancy (2026-07-30).
locality_aware_layout        — partition circuit qubits by interaction graph
                               community, then assign within each core by centrality.
repack_to_budgets            — force a layout to an exact per-core occupancy profile.
run_passes                   — run up to N forward routing passes with early stopping.
run_sabre_passes             — fwd → bwd (reversed DAG) → fwd; pick best of pass1/pass3.
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


# ── SabreLayout with locked boundary ──────────────────────────────────────────

def _per_core_reserved_corner_nodes(arch: DistributedArchitecture, per_core: int) -> set:
    """The ``per_core`` least-useful corner physical qubits of each core:
    among the core's minimum-degree (corner) vertices in ``arch.Gr``, the
    ones with the largest total shortest-path distance to all other physical
    qubits. Deterministic (distance ties broken by ascending node id)."""
    if per_core <= 0:
        return set()
    dist = dict(nx.all_pairs_shortest_path_length(arch.Gr))
    reserved = set()
    for c in range(arch.num_cores):
        nodes = arch.core_qubits(c)
        degs = {n: arch.Gr.degree(n) for n in nodes}
        min_deg = min(degs.values())
        cands = [n for n in nodes if degs[n] == min_deg]
        if len(cands) < per_core:
            cands = sorted(nodes, key=lambda n: (degs[n], n))[:per_core]
        remoteness = {n: sum(dist[n].values()) for n in cands}
        chosen = sorted(cands, key=lambda n: (-remoteness[n], n))[:per_core]
        reserved.update(chosen)
    return reserved


def adaptive_corner_count(arch: DistributedArchitecture, num_qubits: int,
                          max_fill: float = 0.80, k_max: int = 4,
                          reserve: int = 1) -> int:
    """Fill-adaptive per-core reservation count: the LARGEST k (0..k_max) that
    keeps the usable-slot fill nq/(ncores*(slots_per_core-k)) at or below
    max_fill.

    Reserving per-core escape corners helps monotonically while head-room
    lasts (200q m=5 cores: k=4 beats k=2 on every circuit), but dense
    circuits regress once usable fill climbs toward ~90% (64q m=4 cores:
    k=4 -> 89% fill and ae/qft regress past k=2).  max_fill=0.80 reproduces
    the per-suite optima of the corner-count dose-response study:
    k=4 at 25/36/100/200q, k=2 at 64q.

    k is floored at 1 -- guaranteeing every core at least one reserved escape
    slot -- whenever that is physically feasible, i.e. whenever the spare
    capacity P-n covers one slot per core (P-n >= K, P = ncores*spc).  Below
    that floor a "full core" (zero free slots) is possible in the initial
    layout SabreLayout produces, which is what k>=1 exists to prevent; the
    80%-fill target is a performance tuning knob for k among {1..k_max}, not
    a license to fall back to zero reservation when reservation is
    affordable.  Only genuinely infeasible cases (P-n < K, e.g. the 25-qubit
    3-core "F=2<K=3" stress architecture) still return k=0.

    `reserve` raises that floor: `reserve=2` (what `config.safe_mode` needs --
    see SAFE_DSABRE.md) makes every core start with >=2 free slots, so the
    capacity invariant holds by construction rather than by coincidence.  It
    changes no published suite -- the fill rule already returns 4 everywhere
    except 64q, where it returns 2 -- but it stops a denser architecture from
    silently starting unsafe.  The floor is applied only when affordable
    (P-n >= reserve*K + 1), so callers must check the return value against
    `reserve` rather than assume it.
    """
    ncores = arch.num_cores
    spc = len(arch.core_qubits(0))
    physical = ncores * spc
    if reserve > 1 and (physical - num_qubits) >= reserve * ncores + 1:
        k_floor = reserve
    else:
        k_floor = 1 if (physical - num_qubits) >= ncores else 0
    for k in range(k_max, k_floor - 1, -1):
        usable = ncores * (spc - k)
        if usable > 0 and num_qubits / usable <= max_fill:
            return k
    return k_floor


def sabre_locked_boundary_layout(qc, dag, arch: DistributedArchitecture, seed: int = 0):
    """Run SabreLayout on the architecture graph with per-core reserved
    corners removed, using seeds [seed, seed+1, seed+2].

    Reserves the k most-remote corner nodes of EVERY core (k fill-adaptive,
    see `adaptive_corner_count`) as teleportation communication / escape
    slots, preventing SabreLayout from placing frequently-interacting qubits
    there.  The former whole-chip rule ("the 4 lowest-degree nodes") put all
    four reserved slots in core 0 under core-major numbering, leaving every
    other core with none and regularly fully packed.

    Returns a list of up to 3 candidate layouts ({logical_qubit: physical_qubit}).
    """
    from qiskit.transpiler import PassManager, CouplingMap
    from qiskit.transpiler.passes import SabreLayout

    k = adaptive_corner_count(arch, qc.num_qubits)
    corner_nodes = _per_core_reserved_corner_nodes(arch, per_core=k)

    reduced_nodes = [n for n in arch.Gr.nodes() if n not in corner_nodes]
    node_to_idx   = {n: i for i, n in enumerate(reduced_nodes)}
    reduced_edges  = [
        (node_to_idx[u], node_to_idx[v])
        for u, v in arch.Gr.edges()
        if u not in corner_nodes and v not in corner_nodes
    ]
    directed = reduced_edges + [(v, u) for u, v in reduced_edges]
    cm = CouplingMap(couplinglist=directed,
                     description=f"dsabre_{k}corners_per_core_removed")

    layouts = []
    for sd in [seed, seed + 1, seed + 2]:
        try:
            pm = PassManager([SabreLayout(cm, max_iterations=3, seed=sd, swap_trials=5)])
            transpiled = pm.run(qc)
            if transpiled.layout is None:
                continue
            tl = transpiled.layout
            virt_layout = (tl.initial_virtual_layout(filter_ancillas=True)
                           if hasattr(tl, "initial_virtual_layout") else tl.initial_layout)
            result = {}
            for virt_qubit, reduced_idx in virt_layout.get_virtual_bits().items():
                try:
                    bit_index = qc.find_bit(virt_qubit).index
                except Exception:
                    continue
                result[dag.qubits[bit_index]] = reduced_nodes[reduced_idx]
            assigned = set(result.values())
            free = [p for p in arch.data_qubits if p not in assigned]
            _random.Random(sd).shuffle(free)
            fp = iter(free)
            for lq in dag.qubits:
                if lq not in result:
                    result[lq] = next(fp)
            layouts.append(result)
        except Exception:
            continue
    return layouts


# ── Dist-k completion + even repack (2026-07-30 ablation) ──────────────────────

def _dist_k_intra_edges(arch: DistributedArchitecture, removed: set, k: int) -> list:
    """Per-core edges among non-``removed`` qubits at intra-core distance
    <= ``k``, plus the real inter-core links.  k=1 reproduces the bare grid
    (exactly the edges ``arch.Gr`` already encodes); k>=2 adds edges
    SabreLayout would not otherwise see.

    Motivation: ``CouplingMap`` is unweighted, so SabreLayout cannot tell that
    an inter-core hop costs ~10x an intra-core SWAP -- on the bare grid (k=1)
    it minimises total SWAP count treating both the same.  Completing a core
    towards a clique makes intra-core moves look relatively free, nudging the
    objective toward what the router actually pays for: inter-core crossings.
    """
    edges = []
    for c in range(arch.num_cores):
        nodes = [p for p in arch.core_qubits(c) if p not in removed]
        dist  = arch.intra_dist[c]
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                if dist[u][v] <= k:
                    edges.append((u, v))
    for u, v in arch.inter_core_links:
        if u not in removed and v not in removed:
            edges.append((u, v))
    return edges


def _even_core_budgets(arch: DistributedArchitecture, num_qubits: int) -> list:
    """As-even-as-possible per-core occupancy target summing to ``num_qubits``.

    Free slots split evenly (``F // num_cores`` each); any remainder goes to
    the highest-betweenness cores first.  The 2026-07-30 free-slot study
    found central/transit cores want slightly MORE head-room, never less --
    every centre-vs-periphery contrast favoured keeping the busiest cores
    less full, never more.
    """
    ncores = arch.num_cores
    spc    = len(arch.core_qubits(0))
    free_total = ncores * spc - num_qubits
    base, rem  = divmod(free_total, ncores)
    bc    = nx.betweenness_centrality(arch.core_graph)
    order = sorted(range(ncores),
                   key=lambda c: (-bc[c], -arch.core_graph.degree(c), c))
    frees = [base] * ncores
    for c in order[:rem]:
        frees[c] += 1
    return [spc - frees[c] for c in range(ncores)]


def repack_to_budgets(layout: dict, dag, arch: DistributedArchitecture, budgets) -> dict:
    """Re-map ``layout`` so core c holds exactly ``budgets[c]`` qubits.

    Two stages, both deterministic given ``layout``:

    1. *Core assignment.*  Starting from ``layout``'s core groups, repeatedly
       take the most over-budget core and move out its least-affine qubit
       (by interaction-graph weight) into the highest-affinity under-budget
       core (ties broken by core distance, then core id).  This keeps as much
       of the input's locality as the target profile allows.
    2. *Within-core placement.*  Each core's final group is placed on that
       core's physical slots by ``_topology_aware_core_assignment``, i.e. the
       most-connected logical qubits go to the most-central physical slots --
       which also tends to leave the LEAST-central slots (often exactly the
       corners a caller may have reserved) the ones left unassigned when a
       core is under budget.

    ``sum(budgets)`` must equal ``len(layout)``.
    """
    n = len(layout)
    if sum(budgets) != n:
        raise ValueError(f"budgets sum to {sum(budgets)}, need {n}")

    ig = _interaction_graph(dag)
    groups = {c: [] for c in range(arch.num_cores)}
    for lq, p in layout.items():
        groups[arch.core_of(p)].append(lq)

    def affinity(q, group):
        if q not in ig:
            return 0.0
        gs = set(group)
        return sum(ig[q][nb].get("weight", 1) for nb in ig.neighbors(q) if nb in gs)

    guard = 0
    while True:
        over = [c for c in range(arch.num_cores) if len(groups[c]) > budgets[c]]
        if not over:
            break
        guard += 1
        if guard > 4 * n:                       # cannot happen; cheap insurance
            raise RuntimeError("repack_to_budgets did not converge")
        c_over = max(over, key=lambda c: (len(groups[c]) - budgets[c], -c))
        victim = min(groups[c_over],
                     key=lambda q: (affinity(q, groups[c_over]), id(q)))
        under = [c for c in range(arch.num_cores) if len(groups[c]) < budgets[c]]
        c_under = max(under, key=lambda c: (affinity(victim, groups[c]),
                                            -arch.core_dist[c_over][c], -c))
        groups[c_over].remove(victim)
        groups[c_under].append(victim)

    result = {}
    for c in range(arch.num_cores):
        result.update(_topology_aware_core_assignment(groups[c], ig, arch, c))
    if len(result) != n:
        raise RuntimeError(f"repack produced {len(result)} of {n} placements")
    return result


def sabre_completed_boundary_layout(qc, dag, arch: DistributedArchitecture,
                                    seed: int = 0, dist_k: int = 2,
                                    even_repack: bool = True):
    """``sabre_locked_boundary_layout``, but SabreLayout runs on a dist-``dist_k``
    completed core graph and (by default) the result is repacked to even
    per-core occupancy.

    Same corner reservation as ``sabre_locked_boundary_layout`` (adaptive k
    corners removed per core); the only two changes:

    1. SabreLayout sees ``_dist_k_intra_edges(..., dist_k)`` instead of the
       bare grid -- see that function's docstring for why an unweighted
       CouplingMap needs this to have any chance of favouring intra-core
       SWAPs over inter-core teleports.
    2. If ``even_repack``, each candidate layout is passed through
       ``repack_to_budgets(..., _even_core_budgets(...))`` afterward.

    IMPORTANT -- these two knobs were validated SEPARATELY, not together.
    The 2026-07-30 ablation swept dist_k alone (no repack) and, on its own
    high-fill arm, a full per-core CLIQUE with even_repack; it never tried
    dist_k=2 stacked with even_repack, which is this function's default.
    Checking that exact combination against ``sabre_locked_boundary_layout``
    on the six-circuit ablation core, per suite (paired, both best-of-3):

      25q  (~39% fill): NET REGRESSION.  ghz 1->4 EPR, graphstate 2->6 EPR;
           ae/qft/random roughly tied.  even_repack forces balanced occupancy
           onto circuits (chain-shaped ones especially) that route best
           concentrated in one or two cores -- exactly the placement
           ``sabre_locked_boundary_layout`` already finds unforced.
      64q  (~67% fill): mixed, no clear net direction on the 4 circuits
           checked (ae 242->164 better, qft 204->246 worse, ghz/graphstate
           small losses/gains either way).
      64q on a 9-core 3x3 grid (~79% fill): NET WIN.  5 of 6 circuits
           improve, two of them (qft, graphstate) going from routing on only
           1 of 3 SabreLayout seeds to all 3; qnn, which the production
           layout cannot route AT ALL, routes on all 3 seeds at 3446 EPR.
           Total successful layouts across the suite: 15/18 vs 9/18.  The
           one exception is ae, which flips from marginal (1/3 seeds, 720
           EPR) to a complete failure (0/3) -- the sole regression in an
           otherwise clearly-better suite.

    So this combination is a genuine, fill-dependent trade, not a strict
    upgrade: worth it near/above ~75-80% fill, actively harmful below ~40%,
    a wash in between.  A caller who does not know the target fill ahead of
    time should probably pass ``even_repack=False`` (the more uniformly safe
    ~0.96-0.98x win measured for dist_k=2 alone) and reserve
    ``even_repack=True`` for the high-fill regime specifically.

    This is NOT wired into ``benchmark.py`` or any other driver: the existing
    paper tables are generated by ``sabre_locked_boundary_layout``, and
    neither knob here has been validated beyond the 6-circuit ablation core.

    Returns a list of up to 3 candidate layouts ({logical_qubit: physical_qubit}),
    same contract as ``sabre_locked_boundary_layout``.
    """
    from qiskit.transpiler import PassManager, CouplingMap
    from qiskit.transpiler.passes import SabreLayout

    k = adaptive_corner_count(arch, qc.num_qubits)
    corner_nodes = _per_core_reserved_corner_nodes(arch, per_core=k)

    reduced_nodes = [n for n in arch.Gr.nodes() if n not in corner_nodes]
    node_to_idx   = {n: i for i, n in enumerate(reduced_nodes)}
    completed_edges = _dist_k_intra_edges(arch, corner_nodes, dist_k)
    reduced_edges = [(node_to_idx[u], node_to_idx[v]) for u, v in completed_edges]
    directed = reduced_edges + [(v, u) for u, v in reduced_edges]
    cm = CouplingMap(couplinglist=directed,
                     description=f"dsabre_{k}corners_dist{dist_k}_completed")

    budgets = _even_core_budgets(arch, qc.num_qubits) if even_repack else None

    layouts = []
    for sd in [seed, seed + 1, seed + 2]:
        try:
            pm = PassManager([SabreLayout(cm, max_iterations=3, seed=sd, swap_trials=5)])
            transpiled = pm.run(qc)
            if transpiled.layout is None:
                continue
            tl = transpiled.layout
            virt_layout = (tl.initial_virtual_layout(filter_ancillas=True)
                           if hasattr(tl, "initial_virtual_layout") else tl.initial_layout)
            result = {}
            for virt_qubit, reduced_idx in virt_layout.get_virtual_bits().items():
                try:
                    bit_index = qc.find_bit(virt_qubit).index
                except Exception:
                    continue
                result[dag.qubits[bit_index]] = reduced_nodes[reduced_idx]
            assigned = set(result.values())
            free = [p for p in arch.data_qubits if p not in assigned]
            _random.Random(sd).shuffle(free)
            fp = iter(free)
            for lq in dag.qubits:
                if lq not in result:
                    result[lq] = next(fp)
            if budgets is not None:
                result = repack_to_budgets(result, dag, arch, budgets)
            layouts.append(result)
        except Exception:
            continue
    return layouts


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


def run_sabre_passes(router, dag, rev_dag, initial_layout: dict):
    """SABRE-style fwd → bwd (reversed DAG) → fwd routing.

    Pass 1: route `dag` from `initial_layout`        → fwd_final
    Pass 2: route `rev_dag` from fwd_final           → rev_final  (layout refinement)
    Pass 3: route `dag` from rev_final               → final

    Returns the better of pass-1 and pass-3 metrics (by EPR count), with
    `compile_time` set to the total wall-clock time across all three passes.
    Returns None if any pass aborts.
    """
    total_time = 0.0

    t0 = time.perf_counter()
    m1, fwd_final = router.route(dag, initial_layout)
    total_time += time.perf_counter() - t0
    if m1["aborted"]:
        return None

    t0 = time.perf_counter()
    m2, rev_final = router.route(rev_dag, fwd_final)
    total_time += time.perf_counter() - t0
    if m2["aborted"]:
        layout3 = fwd_final
    else:
        layout3 = rev_final

    t0 = time.perf_counter()
    m3, _ = router.route(dag, layout3)
    total_time += time.perf_counter() - t0
    if m3["aborted"]:
        best = m1
    else:
        best = m1 if m1["eprs"] <= m3["eprs"] else m3

    best = dict(best)
    best["compile_time"] = total_time
    return best
