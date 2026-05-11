"""
Head-to-head comparison: pytket-dqc vs dSABRE (BFS extended set)
using pytket-dqc's own KaHyPar initial mapping for both routers.

For each circuit:
  1. Run pytket-dqc HypergraphPartitioning (NUM_SEEDS seeds); record
     best e-bit cost and total wall time.
  2. Convert the best placement (logical qubit → QPU) into a dSABRE
     physical layout using topology-aware intra-core assignment.
  3. Run dSABRE (BFS-extended-set, router_test.py) from that layout
     for LAYOUT_PASSES forward passes; record best EPR and wall time.

Architectures:
  25q / 36q : B-grid 2×2 4×4 (4 QPUs / 4 cores of 16 qubits)
  64q       : H-grid 2×3 4×4 (6 QPUs / 6 cores of 16 qubits)

pytket-dqc modification:
  pytket-dqc v0.0.1 assigns weight 0 to gate vertices in the KaHyPar
  hypergraph, but KaHyPar >=1.3.5 rejects zero-weight vertices.  We
  patch pytket_dqc/allocators/hypergraph_partitioning.py to assign
  weight 1 to all vertices and inflate each server's target block size
  by ceil(n_gates / k) so the capacity constraint still counts only
  qubit vertices effectively.  The 64q QASM loader additionally
  requires maxwidth=128 (pytket default is 32).

Usage:
    python run_pytket_dqc_vs_dsabre.py          # all three suites
    python run_pytket_dqc_vs_dsabre.py 25 36    # specific suites
"""

from __future__ import annotations

import glob
import json
import logging
import os
import sys
import time

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
from router_test import General_dSABRE_Router

logging.disable(logging.WARNING)

NUM_SEEDS     = 5
LAYOUT_PASSES = 2


# ── Inlined helpers (from main.py) ───────────────────────────────────────────

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


def _topology_aware_core_assignment(lq_list, interaction_G, arch, core_id):
    if not lq_list:
        return {}
    dist = arch.intra_dist[core_id]
    all_phys = arch.core_qubits(core_id)
    phys_by_centrality = sorted(
        all_phys, key=lambda p: sum(dist[p][q] for q in all_phys)
    )
    lq_by_importance = sorted(
        lq_list,
        key=lambda q: (interaction_G.degree(q, weight="weight")
                       if q in interaction_G else 0),
        reverse=True,
    )
    mapping = dict(zip(lq_by_importance, phys_by_centrality[:len(lq_by_importance)]))
    sub_nodes = set(lq_list)
    core_edges = [
        (u, v, d.get("weight", 1))
        for u, v, d in interaction_G.edges(data=True)
        if u in sub_nodes and v in sub_nodes
    ]
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
            if nbr == b:
                continue
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


def _run_passes(router, dag, initial_layout, layout_passes):
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


# ── Architecture + network helpers ───────────────────────────────────────────

def _even_partition(n: int, k: int) -> list[list[int]]:
    base, rem = divmod(n, k)
    groups, start = [], 0
    for i in range(k):
        end = start + base + (1 if i < rem else 0)
        groups.append(list(range(start, end)))
        start = end
    return groups


SUITES = {
    "25": {
        "dir":        os.path.expanduser("~/Documents/telesabre/circuits/qasm_25"),
        "suffix":     "_nativegates_ibm_qiskit_opt3_25.qasm",
        "arch_fn":    lambda: build_b_grid_architecture(r=2, s=2, m=4),
        "network_fn": lambda n: NISQNetwork(
            [[0,1],[2,3],[0,2],[1,3]],
            {i: s for i, s in enumerate(_even_partition(n, 4))}
        ),
        "title": "25-qubit suite  (B-grid 2×2 4×4, 4 QPUs / 4 cores)",
    },
    "36": {
        "dir":        os.path.expanduser("~/Documents/telesabre/circuits/qasm_36"),
        "suffix":     "_nativegates_ibm_qiskit_opt3_36.qasm",
        "arch_fn":    lambda: build_b_grid_architecture(r=2, s=2, m=4),
        "network_fn": lambda n: NISQNetwork(
            [[0,1],[2,3],[0,2],[1,3]],
            {i: s for i, s in enumerate(_even_partition(n, 4))}
        ),
        "title": "36-qubit suite  (B-grid 2×2 4×4, 4 QPUs / 4 cores)",
    },
    "64": {
        "dir":        os.path.expanduser("~/Documents/telesabre/circuits/qasm_64"),
        "suffix":     "_nativegates_ibm_qiskit_opt3_64.qasm",
        "arch_fn":    lambda: build_h_grid_architecture(r=2, s=3, m=4),
        "network_fn": lambda n: NISQNetwork(
            [[0,1],[1,2],[3,4],[4,5],[0,3],[1,4],[2,5]],
            {i: s for i, s in enumerate(_even_partition(n, 6))}
        ),
        "title": "64-qubit suite  (H-grid 2×3 4×4, 6 QPUs / 6 cores)",
    },
}


# ── Layout conversion ────────────────────────────────────────────────────────

def placement_to_dsabre_layout(placement_dict, dag, arch):
    qubits = dag.qubits
    ig = _interaction_graph(dag)
    core_to_lqubits: dict[int, list] = {}
    for qubit_idx, server_id in placement_dict.items():
        core_to_lqubits.setdefault(server_id, []).append(qubits[qubit_idx])
    layout: dict = {}
    for core_id, lq_list in core_to_lqubits.items():
        layout.update(_topology_aware_core_assignment(lq_list, ig, arch, core_id))
    return layout


# ── Per-circuit runner ───────────────────────────────────────────────────────

def run_circuit(qasm_path: str, arch, network_fn, router) -> dict:
    tk_circ = circuit_from_qasm(qasm_path, maxwidth=128)
    DQCPass().apply(tk_circ)
    n_qubits_tk = tk_circ.n_qubits

    qc = QuantumCircuit.from_qasm_file(qasm_path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    dag = circuit_to_dag(qc)
    cx = sum(1 for _ in dag.two_qubit_ops())

    network = network_fn(n_qubits_tk)
    alloc = HypergraphPartitioning()
    best_py_cost = None
    best_placement = None
    py_t0 = time.perf_counter()
    for seed in range(NUM_SEEDS):
        try:
            dist = alloc.allocate(tk_circ, network, seed=seed)
            cost = dist.cost()
            if best_py_cost is None or cost < best_py_cost:
                best_py_cost = cost
                best_placement = dict(dist.placement.placement)
        except Exception:
            pass
    py_time = time.perf_counter() - py_t0

    if best_py_cost is None:
        return {"cx": cx, "py_ebits": None, "py_time": py_time,
                "ds_eprs": None, "ds_time": 0.0}

    qubit_placement = {k: v for k, v in best_placement.items()
                       if isinstance(k, int) and k < n_qubits_tk}
    layout = placement_to_dsabre_layout(qubit_placement, dag, arch)

    ds_t0 = time.perf_counter()
    metrics = _run_passes(router, dag, layout, LAYOUT_PASSES)
    ds_time = time.perf_counter() - ds_t0

    ds_eprs = metrics["eprs"] if (metrics and not metrics.get("aborted")) else None
    return {
        "cx":       cx,
        "py_ebits": best_py_cost,
        "py_time":  py_time,
        "ds_eprs":  ds_eprs,
        "ds_time":  ds_time,
    }


# ── Suite runner ─────────────────────────────────────────────────────────────

def run_suite(key: str) -> list[dict]:
    cfg    = SUITES[key]
    arch   = cfg["arch_fn"]()
    hw     = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                            max_iterations=20000)
    router = General_dSABRE_Router(arch, hw)

    files = sorted(glob.glob(os.path.join(cfg["dir"], "*.qasm")))
    if not files:
        print(f"No .qasm files in {cfg['dir']}")
        return []

    print(f"\n{cfg['title']}")
    print(f"  pytket-dqc: best of {NUM_SEEDS} seeds; "
          f"dSABRE-BFS: {LAYOUT_PASSES} layout passes from pytket-dqc mapping")
    hdr = (f"  {'circuit':<12}  {'CX':>6}  "
           f"{'py-ebit':>8}  {'py-t':>6}  "
           f"{'dS-EPR':>7}  {'dS-t':>6}  "
           f"{'Δ(dS/py)':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    py_vals, ds_vals = [], []
    for qf in files:
        cname = os.path.basename(qf).replace(cfg["suffix"], "")
        r = run_circuit(qf, arch, cfg["network_fn"], router)
        r["circuit"] = cname
        rows.append(r)

        py_s = str(r["py_ebits"]) if r["py_ebits"] is not None else "FAIL"
        ds_s = str(r["ds_eprs"]) if r["ds_eprs"] is not None else "FAIL"

        if r["py_ebits"] is not None:
            py_vals.append(r["py_ebits"])
        if r["ds_eprs"] is not None:
            ds_vals.append(r["ds_eprs"])

        delta = (f"{100*(r['ds_eprs'] - r['py_ebits']) / r['py_ebits']:+.1f}%"
                 if r["py_ebits"] and r["ds_eprs"] else "     ---")

        print(f"  {cname:<12}  {r['cx']:>6}  "
              f"{py_s:>8}  {r['py_time']:>5.1f}s  "
              f"{ds_s:>7}  {r['ds_time']:>5.1f}s  "
              f"{delta:>9}")

    if py_vals and ds_vals and len(py_vals) == len(ds_vals):
        from math import prod
        gm_py = prod(py_vals) ** (1.0 / len(py_vals))
        gm_ds = prod(ds_vals) ** (1.0 / len(ds_vals))
        delta_gm = f"{100*(gm_ds - gm_py)/gm_py:+.1f}%"
        print(f"  {'gmean':<12}  {'':>6}  "
              f"{gm_py:>8.1f}  {'':>6}  "
              f"{gm_ds:>7.1f}  {'':>6}  "
              f"{delta_gm:>9}")

    return rows


def main() -> None:
    keys = sys.argv[1:] if len(sys.argv) > 1 else ["25", "36", "64"]
    unknown = [k for k in keys if k not in SUITES]
    if unknown:
        print(f"Unknown suite(s): {unknown}. Choose from: 25, 36, 64")
        sys.exit(1)

    all_results = {}
    for k in keys:
        all_results[k] = run_suite(k)
    print()

    out_path = os.path.join(os.path.dirname(__file__), "results_pytket_dqc_vs_dsabre.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
