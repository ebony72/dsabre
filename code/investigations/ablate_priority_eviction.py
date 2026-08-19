"""
ablate_priority_eviction.py — two more targeted mechanisms suggested by the
2026-08-03 commit-bonus/hard-lock findings: locking onto whoever moved last
doesn't help because (a) it doesn't favor gates that are actually cheap to
finish, and (b) it does nothing about the eviction side effect that causes
most of the "distance backslides" in the first place. This script ablates
each independently, plus their combination:

  cheapest_first_weight (config.py) -- score penalty proportional to a
      candidate's own gate's CURRENT qubit-pair phys_dist, so an
      almost-resolved gate's cheap last hop can cut ahead of a farther gate's
      expensive one instead of queuing behind it in raw-score order. A
      single-layout/single-pass probe on `ae` found weight in [0.1, 0.3] a
      sweet spot (0.3: -31.5% EPR, -15% SWAP, FEWER deadlock recoveries);
      weight >= 0.5 overshoots and starts hurting (the term dominates
      d_prep/hop_gain/dF/dE instead of merely tie-breaking among them).

  evict_distance_aware (config.py) -- `_evict` prefers a free slot that does
      not increase the evicted bystander qubit's OWN pending-gate distance,
      targeting the eviction side effect directly (97-99% of the "distance
      backslide" events traced on 2026-08-03 were exactly this).

Same production layout search + fwd-bwd-fwd protocol as benchmark.py.

Output: code/results/results_ablate_priority_eviction_{suite}.json
Usage:  python3 ablate_priority_eviction.py [--suite 64q]
"""

from __future__ import annotations

import os as _os, sys as _sys
# This script lives in code/investigations/; the implementation, results/ and
# circuit families it uses are one level up, in code/.
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))


import argparse
import os
import time

from ablate_common import (RESULTS_DIR, SUITE_CIRCUITS, SUITES, default_layouts,
                           gmean, load_circuit, meta, route_layout_set, save_json,
                           summarise)
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt

CONDITIONS = [
    ("baseline",        dict()),
    ("cheapest0.2",     dict(cheapest_first_weight=0.2)),
    ("cheapest0.3",     dict(cheapest_first_weight=0.3)),
    ("evict_aware",     dict(evict_distance_aware=True)),
    ("cheapest0.3+evict", dict(cheapest_first_weight=0.3, evict_distance_aware=True)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="64q")
    ap.add_argument("--circuits", default=None,
                    help="comma-separated override; defaults to the full suite table")
    args = ap.parse_args()

    suite    = args.suite
    s        = SUITES[suite]
    arch     = s["arch"]
    circuits = (args.circuits.split(",") if args.circuits
               else SUITE_CIRCUITS.get(suite, SUITE_CIRCUITS["64q"]))

    out_path = os.path.join(RESULTS_DIR, f"results_ablate_priority_eviction_{suite}.json")
    payload = dict(meta("cheapest_first_weight / evict_distance_aware sweep", suite,
                        circuits=circuits, conditions=[k for k, _ in CONDITIONS]),
                  rows={})

    col_w = 12
    hdr = f"{'circuit':<{col_w}}  {'cx':>6}"
    for key, _ in CONDITIONS:
        hdr += f"  {'epr@'+key:>14}  {'ls@'+key:>10}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for cname in circuits:
        qc, dag, rev_dag, n_cx = load_circuit(suite, cname)
        row_str = f"{cname:<{col_w}}  {n_cx:>6}"
        payload["rows"][cname] = dict(n_cx=n_cx, by_cond={})

        layouts = default_layouts(qc, dag, arch, s["n_qubits"], seed=0)

        for key, kw in CONDITIONS:
            hw = HardwareConfig(deadlock_limit=s["hw"].deadlock_limit,
                               max_backup_attempts=s["hw"].max_backup_attempts,
                               max_iterations=s["hw"].max_iterations,
                               **kw)
            router = dSABRE_BurstExt(arch, hw)
            t0 = time.perf_counter()
            best, info = route_layout_set(router, dag, rev_dag, layouts, pattern="fbf")
            elapsed = time.perf_counter() - t0
            result = summarise(best, extra=dict(condition=key, wall_s=round(elapsed, 2),
                                                pass1_aborts=info["pass1_aborts"],
                                                layout_failures=info["layout_failures"]))
            payload["rows"][cname]["by_cond"][key] = result
            if result.get("aborted"):
                row_str += f"  {'ABORT':>14}  {'---':>10}"
            else:
                row_str += f"  {result['eprs']:>14}  {result['ls']:>10}"
            save_json(out_path, payload)  # partial results after every cell

        print(row_str, flush=True)

    # ── Summary: EPR/SWAP delta vs baseline, per circuit and gmean ─────────────
    print("\n" + "=" * 60, flush=True)
    print(f"  {suite}: condition deltas vs baseline", flush=True)
    print("=" * 60, flush=True)
    for key, _ in CONDITIONS:
        if key == "baseline":
            continue
        epr_ratios, ls_ratios = [], []
        print(f"\n  condition={key} vs baseline:", flush=True)
        for cname in circuits:
            r0 = payload["rows"][cname]["by_cond"].get("baseline", {})
            r1 = payload["rows"][cname]["by_cond"].get(key, {})
            if r0.get("aborted") or r1.get("aborted") or not r0 or not r1:
                print(f"    {cname:<{col_w}}  ---", flush=True)
                continue
            epr0, epr1 = r0["eprs"], r1["eprs"]
            ls0, ls1 = r0["ls"], r1["ls"]
            epr_pct = 100 * (epr1 - epr0) / epr0 if epr0 else 0.0
            ls_pct  = 100 * (ls1 - ls0) / ls0 if ls0 else 0.0
            epr_ratios.append(epr1 / epr0 if epr0 else 1.0)
            ls_ratios.append(ls1 / ls0 if ls0 else 1.0)
            print(f"    {cname:<{col_w}}  epr {epr0:>6} -> {epr1:>6} ({epr_pct:+6.1f}%)"
                  f"   ls {ls0:>6} -> {ls1:>6} ({ls_pct:+6.1f}%)", flush=True)
        if epr_ratios:
            print(f"    {'GMEAN':<{col_w}}  epr x{gmean(epr_ratios):.4f}"
                  f"   ls x{gmean(ls_ratios):.4f}", flush=True)

    save_json(out_path, payload)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
