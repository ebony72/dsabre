"""
ablate_occupancy.py — does hop gain earn its place once some cores are
actually congested?

DEPRECATION NOTE: this study originally covered congestion relief too.
Relief was removed from the router entirely (see CHANGES_FROM_SUBMITTED.md):
no architecture this study or any other tried ever showed it helping EPR
count or completion outside 64q_c33, and 64q_c33 has Multiplier failing
unconditionally regardless of relief.  The docstring below is kept as the
historical record of why the occupancy-skew regime was built; only the
hop-gain half of it still runs.

Motivation
----------
Hop gain can only discriminate between teleport destinations at core
distance >= 2.  On the production layouts the congestion regime this study
enters is rare in practice, which is why the post-layout-fix ablation found
relief worth roughly nothing on gmean (TODO.md, "Congestion-relief
contribution is now marginal") before it was removed outright.  That
measurement conflated "the mechanism does not work" with "the regime it
targets was never entered" -- this study enters the regime deliberately to
tell the two apart, which is why it survives for hop gain even though
relief is gone.

This study enters the regime deliberately: the initial mapping is repacked to a
prescribed per-core occupancy profile, from balanced up to cores at 100% fill,
and each profile is crossed with {full, no_hop_gain}.

Profiles are exact — `repack_to_budgets` moves the least-affine qubit out of
each over-budget core, so the realised occupancy equals the budget vector, and
the same repack runs for every profile including the balanced one.  The
untouched production layout is reported as `default` for reference; note it is
already skewed (25q: 12/12/1/0, i.e. two cores at 100% of their usable slots).

Output: code/results/results_ablate_occupancy_{suite}.json  (written per row)
Usage:  python3 code/ablate_occupancy.py [--suite 25q|64q|all]
"""


from __future__ import annotations

import os as _os, sys as _sys
# This script lives in code/investigations/; the implementation, results/ and
# circuit families it uses are one level up, in code/.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import os
import time

from ablate_common import (ABLATION_CIRCUITS, RESULTS_DIR, SUITES, best_over_layouts,
                           core_occupancy, default_layouts, gmean, load_circuit, meta,
                           repack_to_budgets, save_json, summarise)
from dsabre_ext import dSABRE_BurstExt


# ── Occupancy profiles ────────────────────────────────────────────────────────
# Budgets are listed in "hot order": position 0 is the hottest core.  The hot
# order ranks cores by core-graph degree (hubs first), so a congested core is a
# core that traffic actually has to cross, not a backwater.

def hot_order(arch):
    return sorted(range(arch.num_cores),
                  key=lambda c: (-arch.core_graph.degree(c), c))


def _spread(total, k):
    """`total` qubits over `k` cores, as evenly as possible, largest first."""
    base, rem = divmod(total, k)
    return [base + (1 if i < rem else 0) for i in range(k)]


def profiles_for(suite: str, arch):
    """[(name, peak_fill, budgets_in_hot_order), ...]"""
    n = SUITES[suite]["n_qubits"]
    K = arch.num_cores
    spc = len(arch.core_qubits(0))

    def one_hot(h):
        return [h] + _spread(n - h, K - 1)

    def multi_hot(h, count):
        return [h] * count + _spread(n - h * count, K - count)

    if suite == "64q_c33":
        # 64 logical on 81 slots leaves almost no room to vary occupancy: the
        # balanced profile already peaks at 8/9, one short of full.  Two rungs
        # are all this hardware supports.
        raw = [
            ("uniform", _spread(n, K)),          # [8,7,7,7,7,7,7,7,7] peak .89
            ("packed",  multi_hot(spc, 7)),      # [9]*7 + [1,0]       peak 1.0
        ]
    elif suite == "25q":
        raw = [
            ("uniform",   _spread(n, K)),          # [7,6,6,6]   peak 0.44
            ("hot_63",    one_hot(10)),            # [10,5,5,5]
            ("hot_75",    one_hot(12)),            # [12,5,4,4]
            ("hot_88",    one_hot(14)),            # [14,4,4,3]
            # 15/16 is the first rung where `congestion_threshold=1` is met, so
            # relief can fire from a statically-congested core rather than only
            # from one the routing itself fills up.
            ("hot_94",    one_hot(15)),            # [15,4,3,3]
            ("hot_100",   one_hot(16)),            # [16,3,3,3]
            ("twohot_81", [13, 12, 0, 0]),         # two loaded, two empty
            ("packed",    [16, 9, 0, 0]),          # fewest cores, hottest full
            ("graded",    [12, 7, 4, 2]),          # monotone occupancy gradient
        ]
    else:  # 64q — mean fill is already 67%, so a hot core must be >= 14
        raw = [
            ("uniform",     _spread(n, K)),        # [11,11,11,11,10,10] peak .69
            ("hot_88",      one_hot(14)),          # [14,10,10,10,10,10]
            ("hot_94",      one_hot(15)),          # first rung meeting free<=1
            ("hot_100",     one_hot(16)),          # [16,10,10,10,9,9]
            ("hot_100_leaf", None),                # 16 on a degree-2 core instead
            ("twohot_100",  multi_hot(16, 2)),     # both hubs full
            ("packed",      multi_hot(16, 4)),     # 4 cores full, 2 empty
            ("graded",      [16, 14, 12, 10, 7, 5]),
        ]

    order = hot_order(arch)
    out = []
    for name, budgets in raw:
        if name == "hot_100_leaf":
            # Same 16/10/10/10/9/9 shape, but the full core is the *least*
            # central one — isolates "how full" from "how much traffic crosses".
            leaf_first = sorted(range(K), key=lambda c: (arch.core_graph.degree(c), c))
            budgets = one_hot(16)
            per_core = [0] * K
            for c, b in zip(leaf_first, budgets):
                per_core[c] = b
        else:
            per_core = [0] * K
            for c, b in zip(order, budgets):
                per_core[c] = b
        assert sum(per_core) == n, (name, per_core)
        assert all(b <= spc for b in per_core), (name, per_core)
        out.append((name, round(max(per_core) / spc, 3), per_core))
    return out


# ── Mechanism configs ─────────────────────────────────────────────────────────
# Congestion relief was removed from the router (see CHANGES_FROM_SUBMITTED.md):
# no non-c33 architecture ever showed it helping EPR count or completion, and
# its one apparent benefit (the submitted paper's +23.4% at 64q) traced to
# node_decay, a separately-retired mechanism, not to relief itself.  Hop-gain
# was subsequently removed too, on the same evidence-collapsed basis (see
# CHANGES_FROM_SUBMITTED.md) -- the "no_relief"/"neither"/"no_hop_gain"
# configs this function used to return are all gone with their mechanisms;
# there is nothing left on this router to ablate, so this now always
# returns a single config.

def mech_configs(base):
    return [
        ("full", base),
    ]


# ── Driver ────────────────────────────────────────────────────────────────────

def run_suite(suite: str, only=None, tag=""):
    s = SUITES[suite]
    arch, hw, n = s["arch"], s["hw"], s["n_qubits"]
    profs = [("default", None, None)] + profiles_for(suite, arch)
    if only:
        profs = [p for p in profs if p[0] in only]
        if not profs:
            raise SystemExit(f"no profile matched {only}")
    configs = mech_configs(hw)
    out_path = os.path.join(
        RESULTS_DIR, f"results_ablate_occupancy_{suite}{'_' + tag if tag else ''}.json")
    if os.path.exists(out_path) and not tag:
        raise SystemExit(f"ABORT: {out_path} exists. Pass --tag to write a "
                         f"separate file rather than overwrite a completed run.")

    circuits = {}
    for c in ABLATION_CIRCUITS:
        qc, dag, rev, ncx = load_circuit(suite, c)
        circuits[c] = (qc, dag, rev, ncx)
    print(f"\n{'='*96}\n  OCCUPANCY-SKEW ABLATION — {suite}  ({s['arch_name']})\n{'='*96}", flush=True)
    for name, peak, budgets in profs:
        print(f"  profile {name:<13} peak_fill={peak if peak is not None else '(circuit-dependent)'}"
              f"  budgets={budgets}", flush=True)

    payload = dict(meta=meta("occupancy_skew", suite,
                             profiles=[dict(name=nm, peak_fill=pk, budgets=bd)
                                       for nm, pk, bd in profs],
                             configs=[c for c, _ in configs],
                             protocol="best of 3 SabreLayout seeds, fwd->bwd->fwd",
                             layout="production corner-removed SabreLayout, then "
                                    "repack_to_budgets to the profile"),
                   results=[])

    t_all = time.time()
    for prof_name, peak, budgets in profs:
        for cfg_name, cfg in configs:
            router = dSABRE_BurstExt(arch, cfg)
            hdr = f"\n── {suite} | {prof_name} | {cfg_name} " + "─" * 30
            print(hdr, flush=True)
            print(f"{'circuit':<12} {'occ':<26} {'eprs':>6} {'ls':>7} "
                  f"{'fmr':>6} {'t(s)':>8}", flush=True)
            eprs_list, ls_list = [], []
            for c in ABLATION_CIRCUITS:
                qc, dag, rev, ncx = circuits[c]
                base = default_layouts(qc, dag, arch, n, seed=0, n_seeds=3)
                if budgets is None:
                    layouts = base
                else:
                    layouts = [repack_to_budgets(L, dag, arch, budgets) for L in base]
                occ = core_occupancy(layouts[0], arch)
                m = best_over_layouts(router, dag, rev, layouts, pattern="fbf")
                row = summarise(m, extra=dict(suite=suite, profile=prof_name,
                                              peak_fill=peak, config=cfg_name,
                                              circuit=c, cx=ncx, occupancy=occ))
                payload["results"].append(row)
                save_json(out_path, payload)
                if row["aborted"]:
                    print(f"{c:<12} {str(occ):<26} {'ABORT':>6}", flush=True)
                else:
                    eprs_list.append(row["eprs"]); ls_list.append(row["ls"])
                    print(f"{c:<12} {str(occ):<26} {row['eprs']:>6} {row['ls']:>7} "
                          f"{row['force_make_room']:>6} {row['time_s']:>8.1f}", flush=True)
            print(f"{'GMEAN':<12} {'':<26} {gmean(eprs_list):>6.1f} {gmean(ls_list):>7.1f}",
                  flush=True)
    print(f"\nSaved → {out_path}   ({time.time()-t_all:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=["25q", "64q", "64q_c33", "all"], default="all")
    ap.add_argument("--profiles", nargs="*", default=None,
                    help="run only these occupancy profiles")
    ap.add_argument("--tag", default="",
                    help="suffix for the output file (required to avoid "
                         "overwriting a completed run)")
    a = ap.parse_args()
    for suite in (["25q", "64q"] if a.suite == "all" else [a.suite]):
        run_suite(suite, only=a.profiles, tag=a.tag)


if __name__ == "__main__":
    main()
