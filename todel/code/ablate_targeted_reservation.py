"""
ablate_targeted_reservation.py — spend the reservation budget per core instead
of uniformly, so ">=80% fill" stops meaning "no escape slots anywhere".

The production rule (`layout.adaptive_corner_count`) searches over a single
scalar k applied to every core, and returns the largest k whose GLOBAL usable
fill n/(K*(spc-k)) stays at or below max_fill=0.80.  The constraint is global
but the search is uniform, and that granularity mismatch is what makes the rule
collapse:

    3x3 grid of 3x3 cores, n=64:  81 slots, 0.80 budget permits withholding 1,
                                  but uniform k=1 costs 9 -> rule returns k=0
    H-grid 2x3 of 4x4,    n=64:  96 slots, budget permits 16,
                                  uniform k=2 spends only 12

Two observations from the 2026-07-29 ablations motivate spending it per core:

  * SabreLayout does not spread.  At k=0 on the 3x3 grid it placed ghz as
    [9,9,9,9,9,9,9,1,0] -- seven cores completely full beside two nearly
    empty -- so global fill says nothing about whether any individual core has
    head-room.  `_force_make_room` needs an outgoing link to a core with a free
    slot, and congestion relief needs `free >= relief_space_req` in the
    receiver; both are per-core properties.
  * max_fill=0.80 is miscalibrated.  It guards against over-constraining
    SabreLayout ("ae/qft regress past k=2") but the k-sweep put k3 (82% usable
    fill) at 0.927 and k4 (89%) at 0.966, both ahead of k2 at 64q.

Rule under test
---------------
    B = total_slots - ceil(n / max_fill)                    (clamped at >= 0)
spent round-robin over cores ranked by core-graph betweenness centrality
(transit cores first, ties by degree then id), at most k_max per core.  The
node chosen inside a core is the same one the production rule would pick: the
most chip-remote minimum-degree vertex of `arch.Gr`, so this varies only WHERE
the budget goes, never how a corner is identified.

max_fill is swept (0.80 / 0.85 / 0.90) because the budget, not the shape, is
what the earlier study found decisive.

On the 3x3 grid the headline metric is ABORTS, not EPR: at k=0 two of three
initial layouts for ae are unroutable and four of six circuits fail once
congestion relief is switched off.

Output: code/results/results_ablate_targeted_{suite}.json  (written per row)
Usage:  python3 code/ablate_targeted_reservation.py [--suite 64q|64q_c33|all]
"""

from __future__ import annotations

import argparse
import math
import os
import time

import networkx as nx

from ablate_common import (ABLATION_CIRCUITS, RESULTS_DIR, SUITES, best_over_layouts,
                           core_occupancy, gmean, load_circuit, meta, sabre_layout_masked,
                           save_json, summarise)
from layout import _per_core_reserved_corner_nodes, adaptive_corner_count
from dsabre_ext import dSABRE_BurstExt

MAX_FILLS = [0.80, 0.85, 0.90]
K_MAX = 4


# ── Budget allocation ─────────────────────────────────────────────────────────

def core_priority(arch):
    """Cores that most need an escape slot, first.

    Betweenness centrality on the core graph: a core with high betweenness sits
    on many inter-core shortest paths, so relocated qubits keep landing in it
    mid-route and it is the first place `_force_make_room` runs out of options.
    """
    bc = nx.betweenness_centrality(arch.core_graph)
    return sorted(range(arch.num_cores),
                  key=lambda c: (-bc[c], -arch.core_graph.degree(c), c))


def allocate(arch, n_qubits, max_fill, k_max=K_MAX):
    """Per-core reservation counts {core: k_c} under the global fill budget."""
    spc = len(arch.core_qubits(0))
    total = arch.num_cores * spc
    budget = max(0, total - math.ceil(n_qubits / max_fill))
    alloc = {c: 0 for c in range(arch.num_cores)}
    order = core_priority(arch)
    while budget > 0:
        spent = False
        for c in order:
            if budget == 0:
                break
            if alloc[c] < k_max:
                alloc[c] += 1
                budget -= 1
                spent = True
        if not spent:                       # every core at k_max
            break
    return alloc, budget


def reserved_nodes(arch, alloc):
    """Realise {core: k_c} as a node set, using the production per-core choice.

    `_per_core_reserved_corner_nodes(arch, per_core=j)` is nested (its j=1 pick
    is a subset of its j=2 pick, and so on), so taking each core's slice out of
    the j=k_c call reproduces exactly the nodes the production rule would pick
    for that core at that count.
    """
    out = set()
    for k in range(1, K_MAX + 1):
        full = _per_core_reserved_corner_nodes(arch, per_core=k)
        for c, kc in alloc.items():
            if kc == k:
                out |= {p for p in full if arch.core_of(p) == c}
    # Nesting holds only while a core has at least `per_core` minimum-degree
    # vertices; past that the production function falls back to a different
    # candidate ordering, and a core's slice could come out the wrong size.
    for c, kc in alloc.items():
        got = sum(1 for p in out if arch.core_of(p) == c)
        if got != kc:
            raise RuntimeError(f"core {c}: wanted {kc} reservations, realised {got} "
                               f"(min-degree candidate set too small for nesting)")
    keep = [n for n in arch.Gr.nodes() if n not in out]
    if not nx.is_connected(arch.Gr.subgraph(keep)):
        raise RuntimeError("targeted reservation disconnected the coupling map")
    return out


# ── Driver ────────────────────────────────────────────────────────────────────

def variants_for(suite):
    s = SUITES[suite]
    arch, n = s["arch"], s["n_qubits"]
    spc = len(arch.core_qubits(0))
    k_prod = adaptive_corner_count(arch, n)
    out = [("production", dict.fromkeys(range(arch.num_cores), k_prod),
            _per_core_reserved_corner_nodes(arch, per_core=k_prod),
            f"uniform k={k_prod} (adaptive_corner_count)")]
    for mf in MAX_FILLS:
        alloc, left = allocate(arch, n, mf)
        nodes = reserved_nodes(arch, alloc)
        out.append((f"targeted_{int(mf*100)}", alloc, nodes,
                    f"budget {arch.num_cores*spc - math.ceil(n/mf)} at max_fill={mf}"
                    f"{f', {left} unspent (k_max)' if left else ''}"))
    return out


def run_suite(suite: str):
    s = SUITES[suite]
    arch, hw, n = s["arch"], s["hw"], s["n_qubits"]
    router = dSABRE_BurstExt(arch, hw)
    out_path = os.path.join(RESULTS_DIR, f"results_ablate_targeted_{suite}.json")
    if os.path.exists(out_path):
        raise SystemExit(f"ABORT: {out_path} exists; move it aside first.")

    variants = variants_for(suite)
    print(f"\n{'='*100}\n  TARGETED RESERVATION — {suite}  ({s['arch_name']})\n{'='*100}",
          flush=True)
    prio = core_priority(arch)
    print(f"  core priority (betweenness first): {prio}", flush=True)
    for name, alloc, nodes, desc in variants:
        print(f"  {name:<14} {len(nodes):>3} withheld  alloc="
              f"{[alloc[c] for c in range(arch.num_cores)]}  | {desc}", flush=True)

    payload = dict(meta=meta("targeted_reservation", suite,
                             variants=[dict(name=nm, desc=d, n_reserved=len(nd),
                                            alloc=[al[c] for c in range(arch.num_cores)],
                                            reserved=sorted(nd))
                                       for nm, al, nd, d in variants],
                             core_priority=prio,
                             protocol="best of 3 SabreLayout seeds, fwd->bwd->fwd"),
                   results=[])

    circuits = {c: load_circuit(suite, c) for c in ABLATION_CIRCUITS}
    t_all = time.time()
    for name, alloc, nodes, desc in variants:
        print(f"\n── {suite} | {name} " + "─" * 44, flush=True)
        print(f"{'circuit':<12} {'cx':>6} {'occ':<32} {'eprs':>6} {'ls':>7} {'t(s)':>8}",
              flush=True)
        eprs_list, aborts = [], []
        for c in ABLATION_CIRCUITS:
            qc, dag, rev, ncx = circuits[c]
            layouts = sabre_layout_masked(qc, dag, arch, nodes, seed=0, n_seeds=3)
            occ = core_occupancy(layouts[0], arch) if layouts else None
            m = best_over_layouts(router, dag, rev, layouts, pattern="fbf")
            row = summarise(m, extra=dict(suite=suite, variant=name, circuit=c,
                                          cx=ncx, n_reserved=len(nodes),
                                          occupancy=occ))
            payload["results"].append(row)
            save_json(out_path, payload)
            if row["aborted"]:
                aborts.append(c)
                print(f"{c:<12} {ncx:>6} {str(occ):<32} {'ABORT':>6}", flush=True)
            else:
                eprs_list.append(row["eprs"])
                print(f"{c:<12} {ncx:>6} {str(occ):<32} {row['eprs']:>6} "
                      f"{row['ls']:>7} {row['time_s']:>8.1f}", flush=True)
        print(f"{'GMEAN':<12} {'':>6} {'':<32} {gmean(eprs_list):>6.1f}"
              f"   aborted: {', '.join(aborts) if aborts else 'none'}", flush=True)

    print(f"\nSaved → {out_path}   ({time.time()-t_all:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=["64q", "64q_c33", "all"], default="all")
    a = ap.parse_args()
    for suite in (["64q_c33", "64q"] if a.suite == "all" else [a.suite]):
        run_suite(suite)


if __name__ == "__main__":
    main()
