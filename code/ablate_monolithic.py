"""
ablate_monolithic.py — fix step (b)'s blindness: SabreLayout cannot see that
inter-core edges cost 10x, because CouplingMap is unweighted.  Instead of
weighting edges (impossible), reshape the GRAPH it sees: complete each core
so intra-core placement looks cheap relative to crossing a core boundary.

    dist-k completion   two non-reserved qubits of a core are connected iff
                        their intra-core distance is <= k
    k = 1               exactly the production reduced graph (grid edges)
    k = infinity        per-core clique: any intra-core move is 1 hop, so
                        SabreLayout's SWAP objective ~ counts inter-core
                        gate pairs — the EPR objective

Expected trade: larger k biases the layout toward minimizing core crossings
(fewer EPRs) but erases intra-core geometry, so the within-core placement
degrades (more local SWAPs).  The sweep measures where the balance sits.

`clique_even` additionally repacks the clique-SabreLayout output to the
uniform per-core occupancy of the free-slot study — the "allocate evenly, on
the completed graph" combination proposed 2026-07-30.

Arms (all -> identical fwd->bwd->fwd, best of 3 SabreLayout seeds, same
corner reservation as production):

    k1(prod)  k2  k3  k4  clique  [clique_even on 64q/64q_c33]

    (k3/k4 skipped on 3x3 cores, where diameter is 4 and they ~= clique)

Output: code/results/results_ablate_monolithic_{suite}.json  (per row)
Usage:  python3 code/ablate_monolithic.py --suite 25q 64q 64q_c33
"""

from __future__ import annotations

import argparse
import os
import random as _random
import time

from ablate_common import (ABLATION_CIRCUITS, RESULTS_DIR, SUITES, core_occupancy,
                           gmean, load_circuit, meta, repack_to_budgets,
                           route_layout_set, save_json, summarise)
from ablate_freeslots import build_profile
from layout import _per_core_reserved_corner_nodes, adaptive_corner_count
from dsabre_ext import dSABRE_BurstExt


# ── The artificial monolithic QPU ─────────────────────────────────────────────

def completion_edges(arch, reserved, k):
    """Edge list of the corner-removed monolithic QPU under dist-k completion.

    k=1 reproduces the production reduced graph (grid neighbours are exactly
    the dist-1 pairs); k=None is the per-core clique.  Inter-core links keep
    their real port endpoints (dropped if a port is reserved — production
    behaviour; no port is reserved on any suite arch, asserted below).
    """
    edges = []
    for c in range(arch.num_cores):
        nodes = [p for p in arch.core_qubits(c) if p not in reserved]
        d = arch.intra_dist[c]
        for i, u in enumerate(nodes):
            for v in nodes[i + 1:]:
                if k is None or d[u][v] <= k:
                    edges.append((u, v))
    for u, v in arch.inter_core_links:
        assert u not in reserved and v not in reserved, "reserved comm port"
        edges.append((u, v))
    return edges


def sabre_layout_on_edges(qc, dag, arch, nodes, edges, seed=0, n_seeds=3):
    """`sabre_layout_masked`, but on an explicit node/edge list."""
    from qiskit.transpiler import PassManager, CouplingMap
    from qiskit.transpiler.passes import SabreLayout

    node_to_idx = {n: i for i, n in enumerate(nodes)}
    directed = ([(node_to_idx[u], node_to_idx[v]) for u, v in edges]
                + [(node_to_idx[v], node_to_idx[u]) for u, v in edges])
    cm = CouplingMap(couplinglist=directed, description="dsabre_monolithic")

    layouts = []
    for sd in range(seed, seed + n_seeds):
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
                result[dag.qubits[bit_index]] = nodes[reduced_idx]
            assigned = set(result.values())
            free = [p for p in arch.data_qubits if p not in assigned]
            _random.Random(sd).shuffle(free)
            fp = iter(free)
            for lq in dag.qubits:
                if lq not in result:
                    result[lq] = next(fp)
            layouts.append(result)
        except Exception as e:                                    # noqa: BLE001
            print(f"    [sabre_layout_on_edges seed {sd} failed: {e}]", flush=True)
    return layouts


def arms_for(suite, arch):
    """[(name, k, even_repack)] — k=None means clique."""
    # intra-core diameter: 6 on 4x4 cores, 4 on 3x3 cores
    d0 = arch.intra_dist[0]
    diam = max(max(row.values()) for row in d0.values())
    arms = [("k1(prod)", 1, False), ("k2", 2, False)]
    if diam > 4:
        arms += [("k3", 3, False), ("k4", 4, False)]
    arms += [("clique", None, False)]
    if suite in ("64q", "64q_c33"):
        arms += [("clique_even", None, True)]
    return arms


def run_suite(suite: str):
    s = SUITES[suite]
    arch, hw, n = s["arch"], s["hw"], s["n_qubits"]
    spc = len(arch.core_qubits(0))
    router = dSABRE_BurstExt(arch, hw)
    out_path = os.path.join(RESULTS_DIR, f"results_ablate_monolithic_{suite}.json")
    if os.path.exists(out_path):
        raise SystemExit(f"ABORT: {out_path} exists; move it aside first.")

    kres = adaptive_corner_count(arch, n)
    reserved = _per_core_reserved_corner_nodes(arch, per_core=kres)
    nodes = [p for p in arch.Gr.nodes() if p not in reserved]
    arms = arms_for(suite, arch)
    uniform_frees = build_profile("uniform", arch, n)
    even_budgets = [spc - f for f in uniform_frees]

    print(f"\n{'='*104}\n  MONOLITHIC-QPU COMPLETION — {suite}  ({s['arch_name']})",
          flush=True)
    print(f"  corner reservation k={kres} ({len(reserved)} withheld); "
          f"even budgets={even_budgets}", flush=True)
    for name, k, ev in arms:
        ne = len(completion_edges(arch, reserved, k))
        print(f"  {name:<12} dist-k={'inf' if k is None else k:<4} "
              f"edges={ne:<5} even_repack={ev}", flush=True)
    print("=" * 104, flush=True)

    payload = dict(meta=meta("monolithic_completion", suite,
                             k_reserved=kres,
                             arms=[dict(name=nm, dist_k=k, even_repack=ev,
                                        n_edges=len(completion_edges(arch, reserved, k)))
                                   for nm, k, ev in arms],
                             even_budgets=even_budgets,
                             protocol="SabreLayout(3 seeds) on the completed "
                                      "graph -> [optional repack to uniform "
                                      "budgets] -> fwd->bwd->fwd, best EPR"),
                   results=[])

    circuits = {c: load_circuit(suite, c) for c in ABLATION_CIRCUITS}

    t_all = time.time()
    for name, k, even in arms:
        edges = completion_edges(arch, reserved, k)
        print(f"\n── {suite} | {name} " + "─" * 44, flush=True)
        print(f"{'circuit':<12} {'cx':>6} {'occ(first layout)':<30} {'p1ab':>5} "
              f"{'fail':>5} {'eprs':>7} {'ls':>7} {'t(s)':>8}", flush=True)
        eprs_list, ls_list = [], []
        for c in ABLATION_CIRCUITS:
            qc, dag, rev, ncx = circuits[c]
            t0 = time.perf_counter()
            layouts = sabre_layout_on_edges(qc, dag, arch, nodes, edges,
                                            seed=0, n_seeds=3)
            t_layout = time.perf_counter() - t0
            if even:
                layouts = [repack_to_budgets(L, dag, arch, even_budgets)
                           for L in layouts]
            occ = core_occupancy(layouts[0], arch) if layouts else None
            m, info = route_layout_set(router, dag, rev, layouts, pattern="fbf")
            row = summarise(m, extra=dict(
                suite=suite, arm=name, dist_k=("inf" if k is None else k),
                even_repack=even, circuit=c, cx=ncx, occupancy=occ,
                layout_time=round(t_layout, 2),
                n_layouts=info["n_layouts"],
                pass1_aborts=info["pass1_aborts"],
                layout_failures=info["layout_failures"],
                total_time=round(info["total_time"], 1)))
            payload["results"].append(row)
            save_json(out_path, payload)
            if row["aborted"]:
                print(f"{c:<12} {ncx:>6} {str(occ):<30} {info['pass1_aborts']:>4}/3 "
                      f"{info['layout_failures']:>4}/3 {'ABORT':>7}", flush=True)
            else:
                eprs_list.append(row["eprs"]); ls_list.append(row["ls"])
                print(f"{c:<12} {ncx:>6} {str(occ):<30} {info['pass1_aborts']:>4}/3 "
                      f"{info['layout_failures']:>4}/3 {row['eprs']:>7} "
                      f"{row['ls']:>7} {info['total_time']:>8.1f}", flush=True)
        print(f"{'GMEAN':<12} {'':>6} {'':<30} {'':>5} {'':>5} "
              f"{gmean(eprs_list):>7.1f} {gmean(ls_list):>7.1f}", flush=True)

    print(f"\nSaved → {out_path}   ({time.time()-t_all:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", nargs="+", choices=["25q", "64q", "64q_c33"],
                    required=True)
    a = ap.parse_args()
    for suite in a.suite:
        run_suite(suite)


if __name__ == "__main__":
    main()
