import csv
import json
import os
import glob
import random
import time
import networkx as nx
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreLayout

from config import HardwareConfig
from architecture import build_b_grid_architecture, build_h_grid_architecture
from router import General_dSABRE_Router
from burst_router import BurstDSABRE

# =============================================================================
# Interaction graph + qubit partitioning (for locality_aware_layout)
# =============================================================================

def _interaction_graph(dag):
    G = nx.Graph()
    for node in dag.topological_op_nodes():
        if len(node.qargs) == 2:
            q1, q2 = node.qargs[0], node.qargs[1]
            if G.has_edge(q1, q2):
                G[q1][q2]["weight"] += 1
            else:
                G.add_edge(q1, q2, weight=1)
    for node in dag.qubits:
        if node not in G:
            G.add_node(node)
    return G

def _partition_qubits(interaction_G, num_cores, core_capacity):
    qubits = list(interaction_G.nodes())
    if interaction_G.number_of_edges() == 0:
        parts = [[] for _ in range(num_cores)]
        for i, q in enumerate(qubits):
            parts[i % num_cores].append(q)
        return parts
    try:
        from networkx.algorithms import community as nx_comm
        raw = list(nx_comm.greedy_modularity_communities(
            interaction_G, weight="weight", cutoff=num_cores, best_n=num_cores
        ))
    except Exception:
        sorted_q = sorted(qubits,
                          key=lambda q: interaction_G.degree(q, weight="weight"),
                          reverse=True)
        raw = [[] for _ in range(num_cores)]
        for i, q in enumerate(sorted_q):
            raw[i % num_cores].append(q)
        return raw
    parts = [list(p) for p in raw]
    parts.sort(key=len, reverse=True)
    overflow = []
    for p in parts:
        while len(p) > core_capacity:
            victim = min(p, key=lambda q: sum(
                interaction_G[q][nb].get("weight", 1)
                for nb in interaction_G.neighbors(q) if nb in p
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
            aff = sum(interaction_G[q][nb].get("weight", 1)
                      for nb in interaction_G.neighbors(q) if nb in p)
            if aff > best_aff:
                best_aff, best_part = aff, i
        parts[best_part].append(q)
    return parts

def _topology_aware_core_assignment(lq_list, interaction_G, arch, core_id):
    """
    Assign logical qubits to physical slots in a single core.

    1. Rank physical qubits by intra-core centrality (sum of distances to all
       other core qubits; lower = more central).
    2. Rank logical qubits by weighted degree in the interaction graph.
    3. Greedily pair most-connected logical with most-central physical.
    4. Pairwise local search (up to 20 rounds, O(n²·degree) per round):
       swap two assignments if it strictly reduces total weighted intra-core
       interaction distance, then repeat until no improvement.

    Returns {lq: phys}.
    """
    if not lq_list:
        return {}

    dist = arch.intra_dist[core_id]
    all_phys = arch.core_qubits(core_id)

    phys_by_centrality = sorted(
        all_phys,
        key=lambda p: sum(dist[p][q] for q in all_phys),
    )

    lq_by_importance = sorted(
        lq_list,
        key=lambda q: (interaction_G.degree(q, weight="weight")
                       if q in interaction_G else 0),
        reverse=True,
    )

    mapping = dict(zip(lq_by_importance, phys_by_centrality[:len(lq_by_importance)]))

    # Collect only intra-core interaction edges for the local search.
    sub_nodes = set(lq_list)
    core_edges = [
        (u, v, d.get("weight", 1))
        for u, v, d in interaction_G.edges(data=True)
        if u in sub_nodes and v in sub_nodes
    ]
    if not core_edges:
        return mapping

    # Adjacency list enables O(degree) incremental swap-delta evaluation.
    adj = {lq: [] for lq in lq_list}
    for u, v, w in core_edges:
        adj[u].append((v, w))
        adj[v].append((u, w))

    def swap_delta(a, b):
        """Cost change (<0 = improvement) of swapping physical slots of a and b."""
        pa, pb = mapping[a], mapping[b]
        d = 0.0
        for nbr, w in adj[a]:
            if nbr == b:
                continue   # edge (a,b) contributes zero: dist is symmetric
            pn = mapping[nbr]
            d += w * (dist[pb][pn] - dist[pa][pn])
        for nbr, w in adj[b]:
            if nbr == a:
                continue
            pn = mapping[nbr]
            d += w * (dist[pa][pn] - dist[pb][pn])
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


def locality_aware_layout(dag, arch, rng=None):
    if rng is None:
        rng = random
    qubits_per_core = len(arch.core_qubits(0))
    ig = _interaction_graph(dag)
    parts = _partition_qubits(ig, arch.num_cores, qubits_per_core)
    layout = {}
    for core_id, group in enumerate(parts):
        layout.update(_topology_aware_core_assignment(group, ig, arch, core_id))
    assigned = set(layout.values())
    free = [p for p in arch.data_qubits if p not in assigned]
    if rng is not None:
        rng.shuffle(free)
    fp = iter(free)
    for lq in dag.qubits:
        if lq not in layout:
            layout[lq] = next(fp)
    return layout

# =============================================================================
# Method B: SABRE locked boundary layout (corners removed)
# =============================================================================

def sabre_locked_boundary_layout(qc, dag, arch, seed=42):
    """
    Remove the four corner nodes of the monolithic architecture graph, run
    SabreLayout on the reduced graph, then map the result back to the full
    architecture (corner slots guaranteed free for dSABRE teleportation).
    Returns (layouts, sabre_layout_time).
    """
    degrees = dict(arch.Gr.degree())
    min_degree = min(degrees.values())
    corner_nodes = set(sorted(n for n, d in degrees.items() if d == min_degree)[:4])

    assert len(corner_nodes) == 4, (
        f"Expected 4 corner nodes, found {len(corner_nodes)}: {corner_nodes}"
    )

    reduced_nodes = [n for n in arch.Gr.nodes() if n not in corner_nodes]
    reduced_G = arch.Gr.subgraph(reduced_nodes)

    assert nx.is_connected(reduced_G), (
        f"Reduced graph disconnected after removing corners {corner_nodes}. "
        f"Components: {[list(c) for c in nx.connected_components(reduced_G)]}"
    )

    assert len(reduced_nodes) >= qc.num_qubits, (
        f"Reduced graph ({len(reduced_nodes)} nodes) smaller than "
        f"circuit ({qc.num_qubits} qubits)."
    )

    node_to_idx = {n: i for i, n in enumerate(reduced_nodes)}
    reduced_edges = [
        (node_to_idx[u], node_to_idx[v])
        for u, v in arch.Gr.edges()
        if u not in corner_nodes and v not in corner_nodes
    ]
    directed = reduced_edges + [(v, u) for u, v in reduced_edges]
    cm = CouplingMap(couplinglist=directed, description="dsabre_corners_removed")

    layouts = []
    t0 = time.perf_counter()
    for sd in [seed, seed + 1, seed + 2]:
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
        random.Random(sd).shuffle(free)
        fp = iter(free)
        for lq in dag.qubits:
            if lq not in result:
                result[lq] = next(fp)
        layouts.append(result)
    sabre_layout_time = time.perf_counter() - t0
    return layouts, sabre_layout_time

# =============================================================================
# Iterative routing passes
# =============================================================================

def _run_passes(router, dag, initial_layout, layout_passes):
    """
    Run up to `layout_passes` forward routing passes with early stopping.
    Returns the best metrics dict (compile_time = total wall-clock seconds).
    """
    current_layout = initial_layout
    prev_eprs = float("inf")
    best_metrics = None
    total_time = 0.0

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
        prev_eprs = metrics["eprs"]
        current_layout = final_layout

    if best_metrics is not None:
        best_metrics["compile_time"] = total_time
    return best_metrics

# =============================================================================
# Print helpers
# =============================================================================

def _row(m):
    """Format a metrics dict into a result row string."""
    burst = m.get("burst_saves", "-")
    burst_str = f"{burst:>6}" if isinstance(burst, int) else f"{'N/A':>6}"
    aborted = " [ABORTED]" if m.get("aborted") else ""
    return (
        f"EPRs={m['eprs']:4d}  "
        f"Teles={m['teles']:3d}  "
        f"BurstSaves={burst_str}  "
        f"CatComms={m.get('catcomms',0):3d}  "
        f"LocalSWAPs={m['ls']:4d}  "
        f"Cost={m['cost']:6.1f}  "
        f"Backups={m['backup_activations']:2d}  "
        f"Time={m['compile_time']*1000:7.1f}ms"
        f"{aborted}"
    )

def _summary_row(label, m, total_t, mark=""):
    burst = m.get("burst_saves", "N/A")
    burst_str = f"{burst:>6}" if isinstance(burst, int) else f"{'N/A':>6}"
    return (
        f"  {label:<26} {m['eprs']:>5}  {m['teles']:>5}  "
        f"{burst_str}  {m.get('catcomms',0):>8}  "
        f"{m['ls']:>10}  {m['cost']:>7.1f}  "
        f"{m['backup_activations']:>7}  "
        f"{m['compile_time']*1000:>13.1f}  "
        f"{total_t*1000:>13.1f}{mark}"
    )

# =============================================================================
# Export helpers
# =============================================================================

_CSV_FIELDS = [
    "circuit", "router", "strategy", "eprs", "teles", "burst_saves",
    "catcomms", "ls", "cost", "backup_activations",
    "aborted", "compile_time_ms", "total_time_ms",
]


def save_results_csv(rows: list, path: str) -> None:
    """
    Write benchmark result rows to a CSV file.

    Each element of `rows` is a dict with the keys listed in _CSV_FIELDS.
    Appends to an existing file (without re-writing the header) if it already
    exists; creates a new file with a header row otherwise.
    """
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def save_results_json(rows: list, path: str) -> None:
    """
    Write / merge benchmark result rows into a JSON file.

    If the file already exists and contains a JSON array, the new rows are
    appended before the file is re-serialised.  This lets you accumulate
    results across multiple benchmark runs without manual concatenation.
    """
    existing: list = []
    if os.path.exists(path):
        try:
            with open(path) as fh:
                existing = json.load(fh)
        except (json.JSONDecodeError, ValueError):
            existing = []
    combined = existing + rows
    with open(path, "w") as fh:
        json.dump(combined, fh, indent=2)


def save_trace(metrics: dict, path: str) -> None:
    """
    Write the routing trace (if present) from a metrics dict to a JSON file.

    The trace is a list of tuples:
      ("SWAP", p0, p1, core_id)
      ("TELE", virt, p_src, p_dst, src_core, next_core)
    Useful for debugging quality regressions or animating the routing process.
    """
    trace = metrics.get("trace")
    if trace is None:
        return
    with open(path, "w") as fh:
        json.dump([list(e) for e in trace], fh, indent=2)


def _metrics_to_row(m: dict, circuit: str, router: str,
                    strategy: str, total_t: float) -> dict:
    """Convert a metrics dict to a flat CSV/JSON row."""
    return {
        "circuit":            circuit,
        "router":             router,
        "strategy":           strategy,
        "eprs":               m.get("eprs", 0),
        "teles":              m.get("teles", 0),
        "burst_saves":        m.get("burst_saves", ""),
        "catcomms":           m.get("catcomms", 0),
        "ls":                 m.get("ls", 0),
        "cost":               m.get("cost", 0.0),
        "backup_activations": m.get("backup_activations", 0),
        "aborted":            int(m.get("aborted", False)),
        "compile_time_ms":    round(m.get("compile_time", 0.0) * 1000, 2),
        "total_time_ms":      round(total_t * 1000, 2),
    }


# =============================================================================
# Main benchmark loop
# =============================================================================

#from circuit_structure_metrics import profile_from_qiskit
#profile = profile_from_qiskit(qc, initial_layout, name=circuit_name)
#if profile["combined_score"] < 0.3:
#    # Skip BurstDSABRE — circuit has no burst/commutativity structure
#    router = dsabre
#else:
#    router = burst


if __name__ == "__main__":
    circuit_dir = os.path.expanduser("~/Documents/telesabre/circuits/qasm_25")
    qasm_files  = glob.glob(os.path.join(circuit_dir, "*.qasm"))

    if not qasm_files:
        print(f"No .qasm files found in {circuit_dir}.")
        exit(0)

    num_trials   = 3
    LAYOUT_PASSES = 2

    hw_config = HardwareConfig()

    # Tunable BurstDSABRE hyper-parameters
    BURST_WEIGHT     = 2.0
    BURST_NORMALISER = 8

    for qasm_file in sorted(qasm_files):
        circuit_name = os.path.basename(qasm_file)
        if circuit_name != "ae_nativegates_ibm_qiskit_opt3_25.qasm": continue
        print(f"\n{'='*80}")
        print(f"BENCHMARK: {circuit_name}")
        print(f"{'='*80}")

        try:
            qc      = QuantumCircuit.from_qasm_file(qasm_file)
            dag     = circuit_to_dag(qc)
            rev_qc  = qc.reverse_ops()
            rev_dag = circuit_to_dag(rev_qc)
            n_qubits = qc.num_qubits
        except Exception as e:
            print(f"  Failed to load {circuit_name}: {e}")
            continue
        if n_qubits == 0:
            continue

        r, s, m = 2, 2, 4
        arch = build_b_grid_architecture(r=r, s=s, m=m)

        #r, s, m = 2, 3, 4
        #arch = build_h_grid_architecture(r=r, s=s, m=m)
        
        total_phys = r * s * m * m

        print(f"  Circuit qubits  : {n_qubits}")
        print(f"  Architecture    : {r}x{s} grid of {m}x{m} cores "
              f"({total_phys} physical slots)")
        print(f"  Layout passes   : {LAYOUT_PASSES} | Trials (A & C): {num_trials}")
        print(f"  BurstDSABRE     : weight_burst={BURST_WEIGHT}  "
              f"max_burst_norm={BURST_NORMALISER}")

        if n_qubits > total_phys:
            print("  ERROR: Not enough physical qubits. Skipping.")
            continue

        # Instantiate both routers
        dsabre = General_dSABRE_Router(arch, hw_config)
        burst  = BurstDSABRE(arch, hw_config,
                              weight_burst=BURST_WEIGHT,
                              max_burst_normaliser=BURST_NORMALISER)

        # Collect best metrics and total wall-clock per (router × strategy)
        ROUTERS = [("dSABRE", dsabre), ("BurstDSABRE", burst)]
        best   = {(rn, ss): None for rn, _ in ROUTERS
                  for ss in ("A_random", "B_sabre_locked", "C_locality")}
        totalt = {(rn, ss): 0.0  for rn, _ in ROUTERS
                  for ss in ("A_random", "B_sabre_locked", "C_locality")}

        def _record(rname, strat, metrics):
            if metrics is None or metrics.get("aborted"):
                return
            totalt[(rname, strat)] += metrics["compile_time"]
            prev = best[(rname, strat)]
            if prev is None or metrics["eprs"] < prev["eprs"]:
                best[(rname, strat)] = dict(metrics)

        # ── Method A: random initial mapping ─────────────────────────────────
        print(f"\n  [Method A] Random initial mapping — {num_trials} trials")
        for t in range(num_trials):
            phys = list(arch.data_qubits)
            random.shuffle(phys)
            init_layout = {lq: p for lq, p in zip(dag.qubits, phys[:n_qubits])} 
            # this initial layout is not ideal as it fills the first cores while 
            # uses less qubits in the last cores 

            # Forward-backward SABRE-style layout refinement, then final route
            for rname, router in ROUTERS:
                current = init_layout.copy()
                total_t = 0.0
                aborted = False

                t0 = time.perf_counter()
                fwd_m, fwd_final = router.route(dag, current)
                total_t += time.perf_counter() - t0
                if fwd_m["aborted"]:
                    aborted = True
                else:
                    t0 = time.perf_counter()
                    rev_m, rev_final = router.route(rev_dag, fwd_final)
                    total_t += time.perf_counter() - t0
                    if rev_m["aborted"]:
                        aborted = True
                    else:
                        current = rev_final

                if aborted:
                    current = init_layout.copy()

                t0 = time.perf_counter()
                fm, _ = router.route(dag, current)
                total_t += time.perf_counter() - t0
                fm["compile_time"] = total_t

                print(f"    [{rname:<12}] Trial {t+1}: {_row(fm)}")
                _record(rname, "A_random", fm)

        # ── Method B: SABRE locked boundary layout ────────────────────────────
        print(f"\n  [Method B] SABRE locked boundary layout")
        try:
            candidates, layout_t = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
            per_cand_overhead = layout_t / max(len(candidates), 1)
            for i, cand_layout in enumerate(candidates):
                for rname, router in ROUTERS:
                    mb = _run_passes(router, dag, cand_layout, LAYOUT_PASSES)
                    if mb is not None:
                        mb["compile_time"] += per_cand_overhead
                        print(f"    [{rname:<12}] Candidate {i+1}: {_row(mb)}")
                        _record(rname, "B_sabre_locked", mb)
        except Exception as e:
            print(f"    [Method B failed: {e}]")

        # ── Method C: locality-aware layout ──────────────────────────────────
        print(f"\n  [Method C] Locality-aware layout — {num_trials} trials")
        for t in range(num_trials):
            try:
                loc_layout = locality_aware_layout(dag, arch,
                                                   rng=random.Random(t))
                for rname, router in ROUTERS:
                    mc = _run_passes(router, dag, loc_layout, LAYOUT_PASSES)
                    if mc is not None:
                        print(f"    [{rname:<12}] Trial {t+1}: {_row(mc)}")
                        _record(rname, "C_locality", mc)
            except Exception as e:
                print(f"    [Locality trial {t+1} failed: {e}]")

        # ── Summary table ─────────────────────────────────────────────────────
        HDR = (f"\n  {'Router':<14} {'Strategy':<14} {'EPRs':>5}  {'Teles':>5}  "
               f"{'Burst':>6}  {'CatComm':>8}  {'LocalSWAP':>10}  "
               f"{'Cost':>7}  {'Backup':>7}  "
               f"{'BestTime(ms)':>13}  {'TotalTime(ms)':>13}")
        print(HDR)
        print(f"  {'-'*118}")

        overall_best_eprs = float("inf")
        overall_best_row  = ""

        for strat_key, strat_label in [
            ("A_random",       "A (random)    "),
            ("B_sabre_locked", "B (SABRE lock)"),
            ("C_locality",     "C (locality)  "),
        ]:
            for rname, _ in ROUTERS:
                mm = best[(rname, strat_key)]
                tt = totalt[(rname, strat_key)]
                if mm is None:
                    print(f"  {rname:<14} {strat_label:<14} -- aborted / failed --")
                    continue
                mark = ""
                if mm["eprs"] < overall_best_eprs:
                    overall_best_eprs = mm["eprs"]
                    mark = "  ◄ best"
                    overall_best_row = f"{rname} / {strat_label.strip()}"
                print(_summary_row(f"{rname} / {strat_label}", mm, tt, mark))

        if overall_best_row:
            print(f"\n  >> Best result: {overall_best_row}  "
                  f"(EPRs = {overall_best_eprs})")
        else:
            print("\n  All methods aborted for this circuit.")

        # ── Export results ────────────────────────────────────────────────────
        export_rows = []
        for strat_key in ("A_random", "B_sabre_locked", "C_locality"):
            for rname, _ in ROUTERS:
                mm = best[(rname, strat_key)]
                tt = totalt[(rname, strat_key)]
                if mm is not None:
                    export_rows.append(
                        _metrics_to_row(mm, circuit_name, rname, strat_key, tt)
                    )
        if export_rows:
            save_results_csv(export_rows,  "results.csv")
            save_results_json(export_rows, "results.json")
