"""
ablate_freeslots.py — how many free slots does dSE need, and where should
they live?

Let n = logical qubits, K = cores, spc = slots per core, F = K*spc - n free
slots.  Two questions from 2026-07-30:

  Q1 (routability)  What per-core free-slot minimum lets the router proceed
      without fail?  Ideally F >= K — one free slot per core on average —
      would suffice.  The router's own mechanics predict the shape of the
      answer: a scored teleport needs >= 1 free slot in the destination core
      (candidates into full cores are filtered out), `_force_make_room` can
      relieve a full core only through a DIRECT neighbour with a free slot
      (no recursive chains), and congestion relief needs a receiver with
      >= relief_space_req = 3.  So the hypothesis under test is that zero-free
      cores are survivable exactly while they stay adjacent to free capacity,
      and that a global F >= K guarantees nothing about the distribution.

  Q3 (distribution)  Given F, which cores should hold the frees?  Central
      (high-betweenness) cores carry the transit traffic, so both directions
      are plausible a priori: keep centres free as landing pads, or pack
      centres full because their slots are too valuable to idle.

Method: the production corner-removed SabreLayout (best of 3 seeds) is
repacked to an exact per-core occupancy = spc - frees_c via
`repack_to_budgets`, then routed with the production fwd->bwd->fwd protocol.
Reported per row: EPR of the best layout, plus how many of the 3 layouts
aborted their first forward pass (`p1ab`) and how many produced no result at
all (`fail`) — the routability metrics.

Architectures (same 25q/64q circuit suites, chips sized so F is small):

    25q_c133  1x3 of 3x3   27 slots  K=3  F=2   F < K frontier
    25q_c223  2x2 of 3x3   36 slots  K=4  F=11
    64q_b243  2x4 of 3x3   72 slots  K=8  F=8   F = K exactly
    64q_c33   3x3 of 3x3   81 slots  K=9  F=17
    64q       2x3 of 4x4   96 slots  K=6  F=32  production

Profiles (frees listed centre-first by core-graph betweenness):
    uniform          as even as possible
    floor1_*         every core keeps 1 free, the surplus parked in one core
    full1_*          one core starts full, rest even
    fullhalf_*       the K//2 most-/least-central cores start full
    conc_*           frees packed into as few cores as possible
    center_rich/poor frees proportional to betweenness / inverse betweenness

Output: code/results/results_ablate_freeslots_{suite}.json  (written per row)
Usage:  python3 code/ablate_freeslots.py --suite 25q_c133 [25q_c223 ...]
"""


from __future__ import annotations

import os as _os, sys as _sys
# This script lives in code/investigations/; the implementation, results/ and
# circuit families it uses are one level up, in code/.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import os
import time

import networkx as nx

from ablate_common import (ABLATION_CIRCUITS, RESULTS_DIR, SUITES, core_occupancy,
                           default_layouts, gmean, load_circuit, meta,
                           repack_to_budgets, route_layout_set, save_json, summarise)
from dsabre_ext import dSABRE_BurstExt


# ── Core priority: centre first ───────────────────────────────────────────────

def core_prio(arch):
    bc = nx.betweenness_centrality(arch.core_graph)
    return sorted(range(arch.num_cores),
                  key=lambda c: (-bc[c], -arch.core_graph.degree(c), c))


def _even(F, k):
    b, r = divmod(F, k)
    return [b + (1 if i < r else 0) for i in range(k)]


def _weighted(F, weights, cap):
    """Largest-remainder apportionment of F frees, capped at `cap` per core."""
    tot = sum(weights)
    raw = [F * w / tot for w in weights]
    out = [min(cap, int(x)) for x in raw]
    order = sorted(range(len(raw)), key=lambda i: raw[i] - int(raw[i]), reverse=True)
    i = 0
    while sum(out) < F:
        c = order[i % len(order)]
        if out[c] < cap:
            out[c] += 1
        i += 1
    return out


def build_profile(name, arch, n_qubits):
    """frees[core_id] for one named profile; None if infeasible on this arch."""
    K = arch.num_cores
    spc = len(arch.core_qubits(0))
    F = K * spc - n_qubits
    prio = core_prio(arch)
    frees = [0] * K

    def assign(cores, vals):
        for c, v in zip(cores, vals):
            frees[c] = v

    if name == "uniform":
        assign(prio, _even(F, K))
    elif name in ("floor1_center", "floor1_corner"):
        if F <= K:
            return None                      # floor 1 IS uniform (or infeasible)
        frees = [1] * K
        extra = F - K
        for c in (prio if name.endswith("center") else reversed(prio)):
            take = min(spc - frees[c], extra)
            frees[c] += take
            extra -= take
            if extra == 0:
                break
    elif name in ("full1_center", "full1_corner"):
        full = prio[0] if name.endswith("center") else prio[-1]
        rest = [c for c in prio if c != full]
        assign(rest, _even(F, K - 1))
    elif name in ("fullhalf_center", "fullhalf_corner"):
        z = K // 2
        full = prio[:z] if name.endswith("center") else prio[-z:]
        if any(v > spc for v in _even(F, K - z)):
            return None
        rest = [c for c in prio if c not in full]
        assign(rest, _even(F, K - z))
    elif name in ("conc_center", "conc_corner"):
        left = F
        for c in (prio if name.endswith("center") else reversed(prio)):
            take = min(spc, left)
            frees[c] = take
            left -= take
            if left == 0:
                break
    elif name in ("center_rich", "center_poor"):
        bc = nx.betweenness_centrality(arch.core_graph)
        mx = max(bc.values())
        if mx == 0:
            return None                      # symmetric core graph: no centre
        eps = 0.01
        w = [(bc[c] + eps) if name == "center_rich" else (mx - bc[c] + eps)
             for c in range(K)]
        frees = _weighted(F, w, spc)
    else:
        raise ValueError(name)

    assert sum(frees) == F and all(0 <= f <= spc for f in frees), (name, frees)
    if n_qubits - sum(spc - f for f in frees) != 0:
        raise AssertionError(name)
    return frees


PLANS = {
    "25q_c133": ["full1_center", "full1_corner", "conc_center", "conc_corner"],
    "25q_c223": ["uniform", "full1_center", "fullhalf_center", "conc_center"],
    "64q_b243": ["uniform", "full1_center", "full1_corner",
                 "fullhalf_center", "fullhalf_corner",
                 "conc_center", "conc_corner"],
    "64q_c33":  ["uniform", "floor1_center", "floor1_corner",
                 "full1_center", "full1_corner",
                 "fullhalf_center", "fullhalf_corner",
                 "conc_center", "conc_corner", "center_rich", "center_poor"],
    "64q":      ["uniform", "full1_center", "full1_corner",
                 "fullhalf_center", "fullhalf_corner",
                 "conc_center", "conc_corner", "center_rich", "center_poor"],
}


# ── Driver ────────────────────────────────────────────────────────────────────

def run_suite(suite: str):
    s = SUITES[suite]
    arch, hw, n = s["arch"], s["hw"], s["n_qubits"]
    K, spc = arch.num_cores, len(arch.core_qubits(0))
    F = K * spc - n
    router = dSABRE_BurstExt(arch, hw)
    out_path = os.path.join(RESULTS_DIR, f"results_ablate_freeslots_{suite}.json")
    if os.path.exists(out_path):
        raise SystemExit(f"ABORT: {out_path} exists; move it aside first.")

    prio = core_prio(arch)
    profs = []
    for name in PLANS[suite]:
        frees = build_profile(name, arch, n)
        if frees is None:
            print(f"  [skip {name}: infeasible/degenerate on this arch]", flush=True)
            continue
        profs.append((name, frees))

    print(f"\n{'='*100}\n  FREE-SLOT STUDY — {suite}  ({s['arch_name']})", flush=True)
    print(f"  K={K}  spc={spc}  n={n}  F={F}   centre-first priority: {prio}",
          flush=True)
    print("=" * 100, flush=True)
    for name, frees in profs:
        print(f"  {name:<16} frees={frees}  min={min(frees)}  zero-cores={frees.count(0)}",
              flush=True)

    payload = dict(meta=meta("freeslots", suite,
                             K=K, spc=spc, F=F, core_priority=prio,
                             profiles=[dict(name=nm, frees=fr,
                                            budgets=[spc - f for f in fr])
                                       for nm, fr in profs],
                             protocol="repack to exact budgets from production "
                                      "corner-removed SabreLayout (3 seeds), "
                                      "fwd->bwd->fwd per layout, best EPR; "
                                      "p1ab = layouts whose first forward pass "
                                      "aborted, fail = layouts with no result"),
                   results=[])

    circuits = {c: load_circuit(suite, c) for c in ABLATION_CIRCUITS}
    bases = {}
    for c, (qc, dag, rev, ncx) in circuits.items():
        bases[c] = default_layouts(qc, dag, arch, n, seed=0, n_seeds=3)

    t_all = time.time()
    for name, frees in profs:
        budgets = [spc - f for f in frees]
        print(f"\n── {suite} | {name}  frees={frees} " + "─" * 30, flush=True)
        print(f"{'circuit':<12} {'cx':>6} {'p1ab':>5} {'fail':>5} {'eprs':>7} "
              f"{'ls':>7} {'fmr':>5} {'t(s)':>8}", flush=True)
        eprs_list, total_p1, total_fail = [], 0, 0
        for c in ABLATION_CIRCUITS:
            qc, dag, rev, ncx = circuits[c]
            layouts = [repack_to_budgets(L, dag, arch, budgets) for L in bases[c]]
            occ = core_occupancy(layouts[0], arch)
            assert [spc - o for o in occ] == frees, (name, c, occ)
            m, info = route_layout_set(router, dag, rev, layouts, pattern="fbf")
            row = summarise(m, extra=dict(
                suite=suite, profile=name, frees=frees, min_free=min(frees),
                zero_cores=frees.count(0), circuit=c, cx=ncx,
                pass1_aborts=info["pass1_aborts"],
                layout_failures=info["layout_failures"],
                n_layouts=info["n_layouts"],
                total_time=round(info["total_time"], 1)))
            payload["results"].append(row)
            save_json(out_path, payload)
            total_p1 += info["pass1_aborts"]
            total_fail += info["layout_failures"]
            if row["aborted"]:
                print(f"{c:<12} {ncx:>6} {info['pass1_aborts']:>4}/3 "
                      f"{info['layout_failures']:>4}/3 {'ABORT':>7} {'':>7} {'':>5} "
                      f"{info['total_time']:>8.1f}", flush=True)
            else:
                eprs_list.append(row["eprs"])
                print(f"{c:<12} {ncx:>6} {info['pass1_aborts']:>4}/3 "
                      f"{info['layout_failures']:>4}/3 {row['eprs']:>7} "
                      f"{row['ls']:>7} {row['force_make_room']:>5} "
                      f"{info['total_time']:>8.1f}", flush=True)
        print(f"{'GMEAN':<12} {'':>6} {total_p1:>4}/18 {total_fail:>4}/18 "
              f"{gmean(eprs_list):>7.1f}", flush=True)

    print(f"\nSaved → {out_path}   ({time.time()-t_all:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", nargs="+", choices=list(PLANS), required=True)
    a = ap.parse_args()
    for suite in a.suite:
        run_suite(suite)


if __name__ == "__main__":
    main()
