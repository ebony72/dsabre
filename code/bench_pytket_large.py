"""
bench_pytket_large.py — pytket-dqc and dSABRE (from same pytket layout) for 100/200/360-qubit circuits.

For each circuit:
  1. Run pytket-dqc HypergraphPartitioning (NUM_SEEDS seeds); record best e-bit cost.
  2. Convert the best KaHyPar core assignment to a dSABRE physical layout
     (topology-aware intra-core assignment, same as bench_pytket_layout.py).
  3. Run dSE (dSABRE_BurstExt) from that layout using SABRE-style fwd/bwd/fwd passes.

Suites:
  100q  H-grid 2x3 5x5 (150 physical, 6 cores)
  200q  H-grid 4x3 5x5 (300 physical, 12 cores)
  360q  H-grid 2x3 9x9 (486 physical, 6 cores)

Results saved to: results/results_pytket_large.json
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

from architecture import build_h_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from layout import run_sabre_passes

logging.disable(logging.WARNING)

_HERE        = os.path.dirname(os.path.abspath(__file__))
_RESULTS_DIR = os.path.join(_HERE, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

NUM_SEEDS = 5
_HW = HardwareConfig(deadlock_limit=200, max_backup_attempts=200, max_iterations=50000)


def _h_grid_core_links(r: int, s: int) -> list[list[int]]:
    """Core-level inter-core links for an r×s H-grid (no diagonal offset needed here)."""
    links = []
    for cr in range(r):
        for cs in range(s):
            cid = cr * s + cs
            if cs + 1 < s:
                links.append([cid, cid + 1])
            if cr + 1 < r:
                links.append([cid, cid + s])
    return links


def _even_partition(n: int, k: int) -> list[list[int]]:
    base, rem = divmod(n, k)
    groups, start = [], 0
    for i in range(k):
        end = start + base + (1 if i < rem else 0)
        groups.append(list(range(start, end)))
        start = end
    return groups


SUITES = {
    "100q": dict(
        circuit_dir = os.path.expanduser("~/Documents/telesabre/circuits/qasm_100"),
        suffix      = "_nativegates_ibm_qiskit_opt3_100.qasm",
        arch        = build_h_grid_architecture(r=2, s=3, m=5),
        num_cores   = 6,
        core_links  = _h_grid_core_links(2, 3),
        circuits    = ["qft", "qpeexact"],
    ),
    "200q": dict(
        circuit_dir = os.path.expanduser("~/Documents/telesabre/circuits/qasm_200"),
        suffix      = "_nativegates_ibm_qiskit_opt3_200.qasm",
        arch        = build_h_grid_architecture(r=4, s=3, m=5),
        num_cores   = 12,
        core_links  = _h_grid_core_links(4, 3),
        circuits    = ["qft", "qpeexact"],
    ),
    "360q": dict(
        circuit_dir = os.path.expanduser("~/Documents/telesabre/circuits/qasm_360"),
        suffix      = "_nativegates_ibm_qiskit_opt3_360.qasm",
        arch        = build_h_grid_architecture(r=2, s=3, m=9),
        num_cores   = 6,
        core_links  = _h_grid_core_links(2, 3),
        circuits    = ["qft", "qpeexact"],
    ),
}


# ── Topology-aware intra-core placement (same as bench_pytket_layout.py) ──────

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
    qubits = dag.qubits
    ig = _interaction_graph(dag)
    core_to_lq: dict[int, list] = {}
    for qubit_idx, server_id in placement_dict.items():
        core_to_lq.setdefault(server_id, []).append(qubits[qubit_idx])
    layout = {}
    for core_id, lq_list in core_to_lq.items():
        layout.update(_topo_assign(lq_list, ig, arch, core_id))
    assigned = set(layout.values())
    free = [p for p in arch.data_qubits if p not in assigned]
    fi = iter(free)
    for lq in dag.qubits:
        if lq not in layout:
            layout[lq] = next(fi)
    return layout


# ── Per-circuit runner ────────────────────────────────────────────────────────

def run_circuit(cname, qasm_path, arch, num_cores, core_links, router) -> dict:
    n_logical = int(qasm_path.split("_")[-1].replace(".qasm", ""))
    tk_circ = circuit_from_qasm(qasm_path, maxwidth=n_logical + 10)
    DQCPass().apply(tk_circ)
    n_tk = tk_circ.n_qubits

    qc  = QuantumCircuit.from_qasm_file(qasm_path)
    qc  = qc.remove_final_measurements(inplace=False)
    qc  = PassManager([RemoveBarriers()]).run(qc)
    dag     = circuit_to_dag(qc)
    rev_dag = circuit_to_dag(qc.reverse_ops())
    cx = sum(1 for _ in dag.two_qubit_ops())
    print(f"    n={qc.num_qubits}, CX={cx}", flush=True)

    # pytket-dqc: best-of-NUM_SEEDS KaHyPar partition
    network = NISQNetwork(core_links,
                          {i: s for i, s in enumerate(_even_partition(n_tk, num_cores))})
    alloc = HypergraphPartitioning()
    best_py_cost, best_placement = None, None
    py_t0 = time.perf_counter()
    for seed in range(NUM_SEEDS):
        try:
            dist = alloc.allocate(tk_circ, network, seed=seed)
            cost = dist.cost()
            print(f"    py seed {seed}: e-bits={cost}", flush=True)
            if best_py_cost is None or cost < best_py_cost:
                best_py_cost = cost
                best_placement = {k: v for k, v in dist.placement.placement.items()
                                  if isinstance(k, int) and k < n_tk}
        except Exception as e:
            print(f"    py seed {seed}: FAILED ({e})", flush=True)
    py_time = round(time.perf_counter() - py_t0, 2)
    print(f"    pytket-dqc best: {best_py_cost} e-bits ({py_time}s)", flush=True)

    if best_py_cost is None or best_placement is None:
        return dict(circuit=cname, cx=cx, py_ebits=None, py_time=py_time,
                    dse=dict(aborted=True))

    layout = placement_to_layout(best_placement, dag, arch)

    t0 = time.perf_counter()
    m  = run_sabre_passes(router, dag, rev_dag, layout)
    dse_time = round(time.perf_counter() - t0, 2)
    if m and not m.get("aborted"):
        dse = dict(eprs=m["eprs"], ls=m["ls"], time_s=dse_time, aborted=False)
        delta = f"{100*(m['eprs'] - best_py_cost)/best_py_cost:+.1f}%"
        print(f"    dSE: EPR={m['eprs']}, SWAP={m['ls']} ({delta} vs py, {dse_time}s)", flush=True)
    else:
        dse = dict(aborted=True)
        print(f"    dSE: ABORTED ({dse_time}s)", flush=True)

    return dict(circuit=cname, cx=cx, py_ebits=best_py_cost, py_seed=None,
                py_time=py_time, dse=dse)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    keys = sys.argv[1:] if len(sys.argv) > 1 else list(SUITES.keys())
    unknown = [k for k in keys if k not in SUITES]
    if unknown:
        print(f"Unknown suite(s): {unknown}. Choose from: {list(SUITES)}"); sys.exit(1)

    t_total = time.time()
    all_results = {}

    for suite_key in keys:
        cfg    = SUITES[suite_key]
        arch   = cfg["arch"]
        router = dSABRE_BurstExt(arch, _HW)
        print(f"\n{'═'*60}\n  {suite_key}\n{'═'*60}", flush=True)

        suite_records = []
        for cname in cfg["circuits"]:
            pattern = os.path.join(cfg["circuit_dir"], f"{cname}{cfg['suffix']}")
            matches = glob.glob(pattern)
            if not matches:
                print(f"  [{cname}: no file at {pattern}]", flush=True)
                continue
            print(f"\n  Circuit: {cname}", flush=True)
            rec = run_circuit(cname, matches[0], arch,
                              cfg["num_cores"], cfg["core_links"], router)
            suite_records.append(rec)

        all_results[suite_key] = suite_records

    out_path = os.path.join(_RESULTS_DIR, "results_pytket_large.json")
    payload = dict(
        meta=dict(
            date=time.strftime("%Y-%m-%d"),
            description="pytket-dqc (5 seeds) and dSE from pytket layout for 100q/200q/360q",
            py_seeds=NUM_SEEDS,
        ),
        results=all_results,
    )
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved → {out_path}", flush=True)
    print(f"Total wall time: {time.time() - t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
