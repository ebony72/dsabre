"""
ablate_occupancy_complete64.py — [HISTORICAL, FROZEN] finish the 64q_c33
congestion-relief ablation on the three suite circuits the original run
omitted.

DEPRECATED: congestion relief was removed from the router after this script's
data (results_ablate_occupancy_64q_c33_extra3.json) fed directly into the
finding that motivated the removal -- relief's only demonstrated value was
completion on this architecture, which also fails unconditionally on
Multiplier regardless of relief, and its apparent submitted-paper benefit
traced to node_decay, not to any architecture property this script varies.
See CHANGES_FROM_SUBMITTED.md.  `no_relief`/`neither` have been dropped from
CONFIG_ORDER below since `mech_configs()` no longer produces them; hop-gain
was subsequently also removed from the router on the same evidence-collapsed
basis, so `no_hop_gain` is gone too and the script is kept runnable for
`full` only, mainly as a record of what was run.

Why
---
`results_ablate_occupancy_64q_c33.json` (2026-07-29) was produced by
`ablate_occupancy.py`, which iterates `ABLATION_CIRCUITS` — the historical
six-circuit subset (ae, ghz, graphstate, qft, qnn, random).  The 64q suite of
Table III is *nine* circuits; qpeexact, qaoa and multiplier were never run on
the 3x3-of-3x3 hardware.  The paper's tight-machine relief result therefore
rested on 6 of the 9 circuits while every other 64q ablation used all 9.

This driver runs exactly the missing cells with the same architecture, layout
protocol, profiles and mechanism configs as the original, and writes them to
`results_ablate_occupancy_64q_c33_extra3.json`.  Merge for reporting with
`analyze_regime64.py`; nothing in the original file is modified.

Ordering is chosen so the table-critical cells land first: configs run
full -> no_relief -> no_hop_gain -> neither, circuits cheapest-first
(qpeexact 2139 CX, qaoa 3920, multiplier 13040).  Rows are flushed to JSON as
they complete, so a kill at any point leaves a usable partial file.

Usage:  python3 code/ablate_occupancy_complete64.py [--configs full,no_relief]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from ablate_common import (RESULTS_DIR, SUITES, best_over_layouts, core_occupancy,
                           default_layouts, load_circuit, meta, repack_to_budgets,
                           save_json, summarise)
from ablate_occupancy import mech_configs, profiles_for
from dsabre_ext import dSABRE_BurstExt

SUITE = "64q_c33"
MISSING = ["qpeexact", "qaoa", "multiplier"]      # cheapest first
CONFIG_ORDER = ["full"]   # no_relief/neither dropped: relief removed; no_hop_gain dropped: hop-gain removed
# The published table reports the `default` (own) and `uniform` layouts only.
PROFILES = ["default", "uniform"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIG_ORDER),
                    help="comma-separated subset of " + ",".join(CONFIG_ORDER))
    ap.add_argument("--circuits", default=",".join(MISSING))
    args = ap.parse_args()

    want_cfg = [c.strip() for c in args.configs.split(",") if c.strip()]
    circuits_wanted = [c.strip() for c in args.circuits.split(",") if c.strip()]

    s = SUITES[SUITE]
    arch, hw, n = s["arch"], s["hw"], s["n_qubits"]

    all_profs = [("default", None, None)] + profiles_for(SUITE, arch)
    profs = [p for p in all_profs if p[0] in PROFILES]
    assert len(profs) == len(PROFILES), [p[0] for p in all_profs]

    cfg_map = dict(mech_configs(hw))
    configs = [(name, cfg_map[name]) for name in CONFIG_ORDER if name in want_cfg]

    out_path = os.path.join(RESULTS_DIR,
                            f"results_ablate_occupancy_{SUITE}_extra3.json")

    print(f"\n{'='*96}", flush=True)
    print(f"  COMPLETING 64q_c33 OCCUPANCY ABLATION — {s['arch_name']}", flush=True)
    print(f"  circuits: {circuits_wanted}", flush=True)
    print(f"  profiles: {[p[0] for p in profs]}", flush=True)
    print(f"  configs : {[c for c, _ in configs]}", flush=True)
    print(f"  out     : {out_path}", flush=True)
    print(f"{'='*96}\n", flush=True)

    # Preflight every circuit before spending hours on the first one.
    loaded = {}
    for c in circuits_wanted:
        qc, dag, rev, ncx = load_circuit(SUITE, c)
        loaded[c] = (qc, dag, rev, ncx)
        print(f"  loaded {c:<12} {ncx:>6} CX  (preflight OK)", flush=True)

    payload = dict(
        meta=meta("occupancy_skew_completion", SUITE,
                  profiles=[dict(name=nm, peak_fill=pk, budgets=bd)
                            for nm, pk, bd in profs],
                  configs=[c for c, _ in configs],
                  circuits=circuits_wanted,
                  protocol="best of 3 SabreLayout seeds, fwd->bwd->fwd",
                  layout="production corner-removed SabreLayout, then "
                         "repack_to_budgets to the profile",
                  note="Completes results_ablate_occupancy_64q_c33.json, whose "
                       "run covered only the six-circuit ABLATION_CIRCUITS "
                       "subset.  Same architecture, protocol and configs."),
        results=[])

    # Config-outer so that `full` and `no_relief` — the two columns the paper's
    # table reports — are complete across both layouts before the other two
    # configs start.  A kill after those two still yields a publishable table.
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
