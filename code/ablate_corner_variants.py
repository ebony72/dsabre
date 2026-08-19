"""
ablate_corner_variants.py — how much of the corner-removal layout is the
*budget* (how many slots per core are withheld from SabreLayout) and how much
is the *shape* (which slots)?

The production rule reserves, in every core, the `k` most chip-remote of the
core's minimum-degree (corner) vertices, with k fill-adaptive: the largest
k <= 4 keeping usable fill nq/(K*(spc-k)) <= 0.80 — k=4 at 25q, k=2 at 64q.

Two families are swept here:

  budget  k0 k1 k2 k3 k4   the production rule at a fixed k.  Because
                           `adaptive_corner_count` only ever picks a k, this
                           sweep also is the dose-response of the max_fill
                           threshold: at 64q, max_fill 0.70/0.80/0.90 select
                           k = 1/2/4 respectively.

  shape   central / random / portadj / farport, all at the adaptive k, so the
          number of withheld slots is identical to production and only their
          placement differs.  `chip4` is the pre-2026-07 whole-chip rule (the 4
          lowest-degree nodes of the entire chip), which under core-major
          numbering puts all four reservations in core 0 and leaves every other
          core unprotected.

An earlier study (TODO.md, 2026-07-17) compared whole-layout alternatives
(spread / pack / locality-aware) and found reservation shape to be noise at
equal budgets; this one tests the same claim inside the corner rule itself,
where the budget can be held exactly fixed.

Shape variants are picked with a connectivity guard: a node is only withheld if
the reduced coupling graph stays connected, so SabreLayout always receives a
legal map.

Output: code/results/results_ablate_corners_{suite}.json  (written per row)
Usage:  python3 code/ablate_corner_variants.py [--suite 25q|64q|all]
"""

from __future__ import annotations

import argparse
import os
import random
import time

import networkx as nx

from ablate_common import (ABLATION_CIRCUITS, RESULTS_DIR, SUITES, best_over_layouts,
                           core_occupancy, gmean, load_circuit, meta, sabre_layout_masked,
                           save_json, summarise)
from layout import _per_core_reserved_corner_nodes, adaptive_corner_count
from dsabre_ext import dSABRE_BurstExt


# ── Reservation-set builders ──────────────────────────────────────────────────

def _connected_without(arch, removed) -> bool:
    keep = [n for n in arch.Gr.nodes() if n not in removed]
    return nx.is_connected(arch.Gr.subgraph(keep))


def _guarded_pick(arch, k, order_fn):
    """Withhold the first `k` nodes of each core in `order_fn(core)` order,
    skipping any node that would disconnect the reduced coupling graph."""
    removed = set()
    for c in range(arch.num_cores):
        taken = 0
        for nd in order_fn(c):
            if taken == k:
                break
            if _connected_without(arch, removed | {nd}):
                removed.add(nd)
                taken += 1
        if taken < k:
            print(f"    [core {c}: only {taken}/{k} nodes withheld — "
                  f"connectivity guard]", flush=True)
    return removed


def _central_order(arch):
    """Most-central-first: ascending sum of intra-core distance, tie by node id."""
    def order(c):
        nodes = arch.core_qubits(c)
        d = arch.intra_dist[c]
        return sorted(nodes, key=lambda n: (sum(d[n][m] for m in nodes), n))
    return order


def _port_dist_order(arch, farthest: bool):
    """Order a core's non-port nodes by distance to that core's comm ports."""
    def order(c):
        nodes = arch.core_qubits(c)
        ports = [p for p in nodes if p in arch.comm_qubits]
        d = arch.intra_dist[c]
        cands = [n for n in nodes if n not in arch.comm_qubits]
        sign = -1 if farthest else 1
        return sorted(cands, key=lambda n: (sign * min(d[n][p] for p in ports), n))
    return order


def _random_order(arch, seed):
    def order(c):
        nodes = list(arch.core_qubits(c))
        random.Random(seed * 1000 + c).shuffle(nodes)
        return nodes
    return order


def _chip4(arch):
    """Legacy whole-chip rule: the 4 lowest-degree nodes of the entire chip."""
    return set(sorted(arch.Gr.nodes(), key=lambda n: (arch.Gr.degree(n), n))[:4])


def variants_for(arch, n_qubits):
    """[(name, description, [removed_set per SabreLayout seed])]

    Most variants use one fixed reservation set for all three seeds; `random`
    pairs seed i with reservation draw i, so it still costs three layouts.
    """
    k = adaptive_corner_count(arch, n_qubits)
    out = []
    for kk in range(0, 5):
        out.append((f"k{kk}", f"production rule, k={kk}{' (= adaptive)' if kk == k else ''}",
                    [_per_core_reserved_corner_nodes(arch, per_core=kk)] * 3))
    out.append(("chip4", "legacy whole-chip 4 lowest-degree nodes", [_chip4(arch)] * 3))
    out.append((f"central_k{k}", f"k={k} most-central nodes per core",
                [_guarded_pick(arch, k, _central_order(arch))] * 3))
    out.append((f"portadj_k{k}", f"k={k} nodes nearest a comm port, per core",
                [_guarded_pick(arch, k, _port_dist_order(arch, farthest=False))] * 3))
    out.append((f"farport_k{k}", f"k={k} nodes farthest from a comm port, per core",
                [_guarded_pick(arch, k, _port_dist_order(arch, farthest=True))] * 3))
    out.append((f"random_k{k}", f"k={k} random nodes per core (draw i with seed i)",
                [_guarded_pick(arch, k, _random_order(arch, sd)) for sd in range(3)]))
    # Control for `random`: every other variant reuses ONE mask across the three
    # SabreLayout seeds, so `random` is the only arm that also gets mask
    # diversity.  These fix a single draw, isolating "is a random reservation as
    # good as the corner rule" from "do three different masks beat one".
    for sd in range(3):
        out.append((f"randfix{sd}_k{k}", f"k={k} random nodes per core, single draw {sd}",
                    [_guarded_pick(arch, k, _random_order(arch, sd))] * 3))
    return k, out


# ── Driver ────────────────────────────────────────────────────────────────────

def run_suite(suite: str, only=None, tag=""):
    s = SUITES[suite]
    arch, hw, n = s["arch"], s["hw"], s["n_qubits"]
    router = dSABRE_BurstExt(arch, hw)
    out_path = os.path.join(RESULTS_DIR,
                            f"results_ablate_corners_{suite}{'_' + tag if tag else ''}.json")
    if os.path.exists(out_path) and not tag:
        raise SystemExit(f"ABORT: {out_path} exists. Pass --tag to write a "
                         f"separate file rather than overwrite a completed run.")

    k_adaptive, variants = variants_for(arch, n)
    if only:
        variants = [v for v in variants if v[0] in only]
        if not variants:
            raise SystemExit(f"no variant matched {only}")
    print(f"\n{'='*92}\n  CORNER-RESERVATION VARIANTS — {suite}  ({s['arch_name']})"
          f"\n  adaptive k = {k_adaptive}\n{'='*92}", flush=True)
    for name, desc, sets in variants:
        print(f"  {name:<14} |{len(sets[0]):>3} withheld | {desc}", flush=True)
        print(f"                 reserved(seed0) = {sorted(sets[0])}", flush=True)

    payload = dict(meta=meta("corner_variants", suite,
                             k_adaptive=k_adaptive,
                             variants=[dict(name=nm, desc=d,
                                            n_reserved=len(st[0]),
                                            reserved_seed0=sorted(st[0]))
                                       for nm, d, st in variants],
                             protocol="best of 3 SabreLayout seeds, fwd->bwd->fwd"),
                   results=[])

    circuits = {c: load_circuit(suite, c) for c in ABLATION_CIRCUITS}

    t_all = time.time()
    for name, desc, sets in variants:
        print(f"\n── {suite} | {name} " + "─" * 40, flush=True)
        print(f"{'circuit':<12} {'cx':>6} {'occ':<28} {'eprs':>6} {'ls':>7} {'t(s)':>8}",
              flush=True)
        eprs_list, ls_list = [], []
        for c in ABLATION_CIRCUITS:
            qc, dag, rev, ncx = circuits[c]
            layouts = []
            for i, removed in enumerate(sets):
                layouts += sabre_layout_masked(qc, dag, arch, removed, seed=i, n_seeds=1)
            occ = core_occupancy(layouts[0], arch) if layouts else None
            m = best_over_layouts(router, dag, rev, layouts, pattern="fbf")
            row = summarise(m, extra=dict(suite=suite, variant=name, circuit=c,
                                          cx=ncx, n_reserved=len(sets[0]),
                                          occupancy=occ))
            payload["results"].append(row)
            save_json(out_path, payload)
            if row["aborted"]:
                print(f"{c:<12} {ncx:>6} {str(occ):<28} {'ABORT':>6}", flush=True)
            else:
                eprs_list.append(row["eprs"]); ls_list.append(row["ls"])
                print(f"{c:<12} {ncx:>6} {str(occ):<28} {row['eprs']:>6} "
                      f"{row['ls']:>7} {row['time_s']:>8.1f}", flush=True)
        print(f"{'GMEAN':<12} {'':>6} {'':<28} {gmean(eprs_list):>6.1f} "
              f"{gmean(ls_list):>7.1f}", flush=True)

    print(f"\nSaved → {out_path}   ({time.time()-t_all:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=["25q", "64q", "all"], default="all")
    ap.add_argument("--only", nargs="*", default=None,
                    help="run only these variant names")
    ap.add_argument("--tag", default="",
                    help="suffix for the output file (required to avoid "
                         "overwriting a completed run)")
    a = ap.parse_args()
    for suite in (["25q", "64q"] if a.suite == "all" else [a.suite]):
        run_suite(suite, only=a.only, tag=a.tag)


if __name__ == "__main__":
    main()
