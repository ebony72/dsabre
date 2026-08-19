"""
bench_pytket_layout.py — dSABRE (dS + dSE) from pytket-dqc's KaHyPar layout.

For each circuit and suite:
  1. Run pytket-dqc HypergraphPartitioning (NUM_SEEDS seeds); record
     best e-bit cost and total wall time.
  2. Convert the best KaHyPar core assignment into a dSABRE physical
     layout using topology-aware intra-core assignment.
  3. Run both dS (General_dSABRE_Router) and dSE (dSABRE_BurstExt)
     from that layout using SABRE-style fwd/bwd/fwd passes (run_sabre_passes),
     matching the pass strategy of the main benchmark.

Suites: 25q, 36q, 64q  (B-grid 2x2 4x4 and H-grid 2x3 4x4)

Results saved to:  results/results_pytket_layout.json

Usage:
  python bench_pytket_layout.py           # all three suites
  python bench_pytket_layout.py 25 36     # specific suites
"""

from __future__ import annotations

import glob, json, logging, os, sys, time
import networkx as nx

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from pytket.qasm import circuit_from_qasm
from pytket_dqc.allocators import HypergraphPartitioning
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.utils import DQCPass

from architecture import build_b_grid_architecture, build_h_grid_architecture
from config import HardwareConfig
from router import General_dSABRE_Router
from dsabre_ext import dSABRE_BurstExt
from layout import run_sabre_passes
from circuit_paths import circuits_path

logging.disable(logging.WARNING)

_HERE        = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.environ.get("DSABRE_OUT_DIR") or os.path.join(_HERE, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

NUM_SEEDS = 5
_HW = HardwareConfig(deadlock_limit=100, max_backup_attempts=100, max_iterations=20000)


def _even_partition(n: int, k: int) -> list[list[int]]:
    base, rem = divmod(n, k)
    groups, start = [], 0
    for i in range(k):
        end = start + base + (1 if i < rem else 0)
        groups.append(list(range(start, end)))
        start = end
    return groups


SUITES = {
    "25": dict(
        circuit_dir = circuits_path("qasm_25"),
        suffix      = "_nativegates_ibm_qiskit_opt3_25.qasm",
        arch_fn     = lambda: build_b_grid_architecture(r=2, s=2, m=4),
        network_fn  = lambda n: NISQNetwork(
            [[0,1],[2,3],[0,2],[1,3]],
            {i: s for i, s in enumerate(_even_partition(n, 4))},
        ),
    ),
    "36": dict(
        circuit_dir = circuits_path("qasm_36"),
        suffix      = "_nativegates_ibm_qiskit_opt3_36.qasm",
        arch_fn     = lambda: build_b_grid_architecture(r=2, s=2, m=4),
        network_fn  = lambda n: NISQNetwork(
            [[0,1],[2,3],[0,2],[1,3]],
            {i: s for i, s in enumerate(_even_partition(n, 4))},
        ),
    ),
    "64": dict(
        circuit_dir = circuits_path("qasm_64"),
        suffix      = "_nativegates_ibm_qiskit_opt3_64.qasm",
        arch_fn     = lambda: build_h_grid_architecture(r=2, s=3, m=4),
        network_fn  = lambda n: NISQNetwork(
            [[0,1],[1,2],[3,4],[4,5],[0,3],[1,4],[2,5]],
            {i: s for i, s in enumerate(_even_partition(n, 6))},
        ),
    ),
}

# ~/Documents/telesabre/circuits/qasm_64 is shared across projects (see
# CLAUDE.md) and picked up two extra files (vqe_su2, wstate -- 36q-suite
# names re-scaled to 64 qubits by some other project) that are not part of
# dSABRE's published 9-circuit 64q suite; vqe_su2_64 also crashes pytket-dqc's
# DQCPass.  Whitelist rather than glob-all.
CANONICAL_CIRCUITS = {
    "64": {"ae", "ghz", "graphstate", "qft", "qnn", "random",
           "qpeexact", "qaoa", "multiplier"},
}


# ── Topology-aware intra-core layout from a core→qubits partition ────────────

def _interaction_graph(dag) -> nx.Graph:
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


def _topo_assign(lq_list, ig, arch, core_id) -> dict:
    """Topology-aware intra-core placement: high-degree → most-central slot."""
    if not lq_list:
        return {}
    dist     = arch.intra_dist[core_id]
    all_phys = arch.core_qubits(core_id)
    phys_by_centrality = sorted(all_phys,
        key=lambda p: sum(dist[p][q] for q in all_phys))
    lq_by_importance = sorted(lq_list,
        key=lambda q: ig.degree(q, weight="weight") if q in ig else 0,
        reverse=True)
    mapping = dict(zip(lq_by_importance, phys_by_centrality[:len(lq_by_importance)]))

    sub = set(lq_list)
    adj = {lq: [] for lq in lq_list}
    for u, v, d in ig.edges(data=True):
        if u in sub and v in sub:
            w = d.get("weight", 1)
            adj[u].append((v, w)); adj[v].append((u, w))

    def delta(a, b):
        pa, pb = mapping[a], mapping[b]; d = 0.0
        for nbr, w in adj[a]:
            if nbr != b: d += w * (dist[pb][mapping[nbr]] - dist[pa][mapping[nbr]])
        for nbr, w in adj[b]:
            if nbr != a: d += w * (dist[pa][mapping[nbr]] - dist[pb][mapping[nbr]])
        return d

    for _ in range(20):
        improved = False
        lqs = list(mapping)
        for i in range(len(lqs)):
            for j in range(i + 1, len(lqs)):
                if delta(lqs[i], lqs[j]) < 0:
                    mapping[lqs[i]], mapping[lqs[j]] = mapping[lqs[j]], mapping[lqs[i]]
                    improved = True
        if not improved:
            break
    return mapping


def placement_to_layout(placement_dict, dag, arch) -> dict:
    """Convert pytket-dqc qubit→server mapping to dSABRE logical→physical layout."""
    qubits = dag.qubits
    ig = _interaction_graph(dag)
    core_to_lq: dict[int, list] = {}
    for qubit_idx, server_id in placement_dict.items():
        core_to_lq.setdefault(server_id, []).append(qubits[qubit_idx])
    layout = {}
    for core_id, lq_list in core_to_lq.items():
        layout.update(_topo_assign(lq_list, ig, arch, core_id))
    # fill any unassigned qubits into remaining free slots
    assigned = set(layout.values())
    free = [p for p in arch.data_qubits if p not in assigned]
    fi = iter(free)
    for lq in dag.qubits:
        if lq not in layout:
            layout[lq] = next(fi)
    return layout


# ── Per-circuit runner ────────────────────────────────────────────────────────

def run_circuit(qasm_path, arch, network_fn, routers) -> dict:
    tk_circ = circuit_from_qasm(qasm_path, maxwidth=128)
    DQCPass().apply(tk_circ)
    n_tk = tk_circ.n_qubits

    qc  = QuantumCircuit.from_qasm_file(qasm_path)
    qc  = qc.remove_final_measurements(inplace=False)
    qc  = PassManager([RemoveBarriers()]).run(qc)
    dag     = circuit_to_dag(qc)
    rev_dag = circuit_to_dag(qc.reverse_ops())
    cx = sum(1 for _ in dag.two_qubit_ops())

    # pytket-dqc: best-of-NUM_SEEDS KaHyPar partition
    network = network_fn(n_tk)
    alloc   = HypergraphPartitioning()
    best_py_cost, best_placement = None, None
    py_t0 = time.perf_counter()
    for seed in range(NUM_SEEDS):
        try:
            dist = alloc.allocate(tk_circ, network, seed=seed)
            cost = dist.cost()
            if best_py_cost is None or cost < best_py_cost:
                best_py_cost = cost
                best_placement = {k: v for k, v in dist.placement.placement.items()
                                  if isinstance(k, int) and k < n_tk}
        except Exception:
            pass
    py_time = round(time.perf_counter() - py_t0, 3)

    if best_py_cost is None:
        return dict(cx=cx, py_ebits=None, py_time=py_time, routers={})

    layout = placement_to_layout(best_placement, dag, arch)

    router_results = {}
    for rkey, router in routers.items():
        t0 = time.perf_counter()
        m  = run_sabre_passes(router, dag, rev_dag, layout)
        elapsed = round(time.perf_counter() - t0, 3)
        if m and not m.get("aborted"):
            router_results[rkey] = dict(
                eprs=m["eprs"], ls=m["ls"], time_s=elapsed, aborted=False)
        else:
            router_results[rkey] = dict(aborted=True)

    return dict(cx=cx, py_ebits=best_py_cost, py_time=py_time,
                routers=router_results)


# ── Suite runner ──────────────────────────────────────────────────────────────

def run_suite(key: str) -> list[dict]:
    cfg  = SUITES[key]
    arch = cfg["arch_fn"]()
    routers = {
        "dS":  General_dSABRE_Router(arch, _HW),
        "dSE": dSABRE_BurstExt(arch, _HW),
    }
    files = sorted(glob.glob(os.path.join(cfg["circuit_dir"], "*.qasm")))
    canon = CANONICAL_CIRCUITS.get(key)
    if canon:
        files = [f for f in files
                 if os.path.basename(f).replace(cfg["suffix"], "") in canon]
    if not files:
        print(f"  [no .qasm files in {cfg['circuit_dir']}]", flush=True)
        return []

    print(f"\n  === {key}q suite ===", flush=True)
    records = []
    for qf in files:
        cname = os.path.basename(qf).replace(cfg["suffix"], "")
        print(f"  {cname}", flush=True)
        r = run_circuit(qf, arch, cfg["network_fn"], routers)
        r["circuit"] = cname; r["suite"] = key + "q"
        records.append(r)
        py_s = str(r["py_ebits"]) if r["py_ebits"] is not None else "FAIL"
        for rk, rv in r["routers"].items():
            if not rv.get("aborted"):
                delta = (f"{100*(rv['eprs'] - r['py_ebits'])/r['py_ebits']:+.1f}%"
                         if r["py_ebits"] else "---")
                print(f"    {rk}: EPR={rv['eprs']} (py={py_s}, {delta})", flush=True)
            else:
                print(f"    {rk}: ABORTED", flush=True)
    return records


def main():
    keys = sys.argv[1:] if len(sys.argv) > 1 else ["25", "36", "64"]
    unknown = [k for k in keys if k not in SUITES]
    if unknown:
        print(f"Unknown suite(s): {unknown}. Choose from: 25, 36, 64"); sys.exit(1)

    t0 = time.time()
    out_path = os.path.join(_RESULTS_DIR, "results_pytket_layout.json")
    all_records = {}

    def save():
        payload = dict(
            meta=dict(
                date=time.strftime("%Y-%m-%d"),
                description="dSABRE (dS + dSE) routed from pytket-dqc KaHyPar layout",
                layout="pytket-dqc HypergraphPartitioning, best of 5 seeds",
                pass_strategy="fwd -> bwd (reversed DAG) -> fwd; best of pass1/pass3",
                py_seeds=NUM_SEEDS,
                routers={
                    "dS":  "General_dSABRE_Router  (router.py)",
                    "dSE": "dSABRE_BurstExt        (dsabre_ext.py)",
                },
            ),
            results=all_records,
        )
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved (partial) → {out_path}", flush=True)

    for k in keys:
        all_records[k + "q"] = run_suite(k)
        save()   # persist after each suite so a later crash keeps earlier work

    print(f"Total wall time: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
