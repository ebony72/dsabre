"""
ablate_occupancy_64q_c33_k1.py — [HISTORICAL, FROZEN] full nine-circuit
re-run of the tight-machine congestion-relief ablation (`tab:regime`) under
the corrected corner-reservation policy: k is now floored at 1 whenever a
free slot per core is physically affordable (P-n >= K), rather than falling
back to k=0 ("no escape reservation at all") whenever k=0 happens to meet the
80%-fill target on its own, or whenever no k meets the target at all.

DEPRECATED: congestion relief was removed from the router after this script's
data fed into the finding that motivated the removal -- relief showed the
same completion pattern under k=0 and k=1 alike, and Multiplier fails
unconditionally under both regardless of relief.  See
CHANGES_FROM_SUBMITTED.md.  `no_relief`/`neither` have been dropped from
CONFIG_ORDER below since `mech_configs()` no longer produces them, and
hop-gain was subsequently removed from the router too, on the same
evidence-collapsed basis, dropping `no_hop_gain` as well.  The k>=1
floor in `layout.py::adaptive_corner_count` itself is unaffected by relief's
removal and remains in effect.

Why a fresh run rather than reusing results_ablate_occupancy_64q_c33*.json
----------------------------------------------------------------------------
`layout.py::adaptive_corner_count` was patched to add this floor.  For the
64q_c33 architecture (3x3 grid of 3x3 cores) that changes k from 0 to 1:
usable fill becomes 88.9% instead of the previous "no reservation" state, and
every core is now GUARANTEED at least one free slot in the initial layout --
verified directly: max observed occupancy across all seeds/circuits tested is
8/9, never 9/9.  Every existing 64q_c33 result (the original six-circuit run
from 2026-07-29 and its three-circuit completion) was generated under the OLD
k=0 policy and does not reflect this change.  This script re-runs the full
nine-circuit, two-layout, four-config matrix from scratch under the new
policy, so the two experiments (k=0 "no reservation at all" vs k=1 "minimal
reservation") can be compared directly rather than conflated.

No other architecture in the paper is affected by the floor: every headline
suite (25q/36q/64q H-grid/100q/200q/360q/heavy-hex) already gets k>=2 under
the old rule, so the floor never binds there -- verified by
check_full_core_layouts.py before this change was made.

Output: code/results/results_ablate_occupancy_64q_c33_k1.json
        (a fresh file; the k=0 files are untouched)

Usage:  python3 code/ablate_occupancy_64q_c33_k1.py [--configs full,no_relief]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from ablate_common import (RESULTS_DIR, SUITES, SUITE_CIRCUITS, best_over_layouts,
                           core_occupancy, default_layouts, load_circuit, meta,
                           repack_to_budgets, save_json, summarise)
from ablate_occupancy import mech_configs, profiles_for
from dsabre_ext import dSABRE_BurstExt
from layout import adaptive_corner_count

SUITE = "64q_c33"
# Cheapest-first so early feedback arrives fast; multiplier (13040 CX) last.
CIRCUIT_ORDER = ["ghz", "graphstate", "qpeexact", "ae", "qft", "random",
                 "qaoa", "qnn", "multiplier"]
CONFIG_ORDER = ["full"]   # no_relief/neither dropped: relief removed; no_hop_gain dropped: hop-gain removed
PROFILES = ["default", "uniform"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIG_ORDER),
                    help="comma-separated subset of " + ",".join(CONFIG_ORDER))
    ap.add_argument("--circuits", default=",".join(CIRCUIT_ORDER))
    args = ap.parse_args()

    want_cfg = [c.strip() for c in args.configs.split(",") if c.strip()]
    circuits_wanted = [c for c in CIRCUIT_ORDER
                       if c in [x.strip() for x in args.circuits.split(",")]]
    assert set(circuits_wanted) <= set(SUITE_CIRCUITS["64q"]), circuits_wanted

    s = SUITES[SUITE]
    arch, hw, n = s["arch"], s["hw"], s["n_qubits"]

    k_now = adaptive_corner_count(arch, n)
    print(f"adaptive_corner_count({SUITE}, {n}) = {k_now}  "
          f"(expect 1, was 0 before the floor)", flush=True)
    assert k_now == 1, f"expected k=1 under the new policy, got {k_now}"

    all_profs = [("default", None, None)] + profiles_for(SUITE, arch)
    profs = [p for p in all_profs if p[0] in PROFILES]
    assert len(profs) == len(PROFILES), [p[0] for p in all_profs]

    cfg_map = dict(mech_configs(hw))
    configs = [(name, cfg_map[name]) for name in CONFIG_ORDER if name in want_cfg]

    out_path = os.path.join(RESULTS_DIR, f"results_ablate_occupancy_{SUITE}_k1.json")
    if os.path.exists(out_path) and "--force" not in sys.argv:
        raise SystemExit(f"ABORT: {out_path} exists. Delete it or pass --force "
                         f"to overwrite -- this script always writes the same "
                         f"file, unlike the completion driver's --tag scheme.")

    print(f"\n{'='*96}", flush=True)
    print(f"  64q_c33 OCCUPANCY ABLATION UNDER k>=1 FLOOR — {s['arch_name']}",
          flush=True)
    print(f"  circuits: {circuits_wanted}", flush=True)
    print(f"  profiles: {[p[0] for p in profs]}", flush=True)
    print(f"  configs : {[c for c, _ in configs]}", flush=True)
    print(f"  out     : {out_path}", flush=True)
    print(f"{'='*96}\n", flush=True)

    loaded = {}
    for c in circuits_wanted:
        qc, dag, rev, ncx = load_circuit(SUITE, c)
        loaded[c] = (qc, dag, rev, ncx)
        print(f"  loaded {c:<12} {ncx:>6} CX  (preflight OK)", flush=True)

    payload = dict(
        meta=meta("occupancy_skew_k1floor", SUITE,
                  k=k_now,
                  profiles=[dict(name=nm, peak_fill=pk, budgets=bd)
                            for nm, pk, bd in profs],
                  configs=[c for c, _ in configs],
                  circuits=circuits_wanted,
                  protocol="best of 3 SabreLayout seeds, fwd->bwd->fwd",
                  layout="production corner-removed SabreLayout (k=1, floored "
                         "since P-n>=K on this architecture) then "
                         "repack_to_budgets to the profile",
                  note="Full nine-circuit re-run of results_ablate_occupancy_"
                       "64q_c33{,_extra3}.json under the corrected "
                       "adaptive_corner_count floor (layout.py).  Those files "
                       "used k=0; this one uses k=1, so every core keeps >=1 "
                       "free slot in the initial layout."),
        results=[])

    t_all = time.time()
    for cfg_name, cfg in configs:
        for prof_name, peak, budgets in profs:
            router = dSABRE_BurstExt(arch, cfg)
            print(f"\n── {SUITE} | {prof_name} | {cfg_name} " + "─" * 34, flush=True)
            print(f"{'circuit':<12} {'eprs':>7} {'ls':>8} "
                  f"{'fmr':>6} {'t(s)':>9}", flush=True)
            for c in circuits_wanted:
                qc, dag, rev, ncx = loaded[c]
                t0 = time.time()
                base = default_layouts(qc, dag, arch, n, seed=0, n_seeds=3)
                if budgets is not None:
                    occ0 = core_occupancy(base[0], arch)
                    assert min(occ0) is not None  # sanity: base has all cores present
                layouts = (base if budgets is None
                           else [repack_to_budgets(L, dag, arch, budgets) for L in base])
                occ = core_occupancy(layouts[0], arch)
                m = best_over_layouts(router, dag, rev, layouts, pattern="fbf")
                row = summarise(m, extra=dict(suite=SUITE, profile=prof_name,
                                              peak_fill=peak, config=cfg_name,
                                              circuit=c, cx=ncx, occupancy=occ))
                payload["results"].append(row)
                save_json(out_path, payload)
                dt = time.time() - t0
                if row["aborted"]:
                    print(f"{c:<12} {'ABORT':>7} {'':>8} {'':>6} {dt:>9.1f}",
                          flush=True)
                else:
                    print(f"{c:<12} {row['eprs']:>7} {row['ls']:>8} "
                          f"{row.get('force_make_room', 0):>6} {dt:>9.1f}", flush=True)

    print(f"\nDONE in {time.time() - t_all:.0f}s -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
