"""
ablate_backup_relay.py — does the invariant-preserving BFS-relay backup plan
(config.HardwareConfig.backup_relay_mode) beat the greedy direct-neighbour
_force_make_room approach it replaces, inside _backup_plan's cross-core hop?

Motivation (2026-08-03, see conversation log / router.py's "Invariant-preserving
relay" section): _backup_plan's old cross-core loop only checked that the next
core has >=1 free before hopping a stuck gate's qubit into it, which can drop
that core to exactly 0 -- and `_force_make_room`'s own rescue only looks at
DIRECT neighbours of the core needing room, giving up (silently making no
progress that backup_plan call) if none of them has any.  `backup_relay_mode`
does a full BFS over the (connected) core graph for the nearest core with
>=2 free and relays that surplus in, one hop at a time, guaranteeing every
core stays >=1 free -- PROVIDED that invariant already held when backup_plan
was invoked.  Empirically that precondition rarely holds on this suite (normal
routing's existing `_force_make_room` already lets cores hit 0 before
deadlock recovery ever fires), so this is not a formally-guaranteed win in
practice -- it is evaluated here purely as a more thorough (farther-searching)
heuristic replacement for the old direct-neighbour-only rescue.

Same production layout search + fwd-bwd-fwd protocol as benchmark.py.

Output: code/results/results_ablate_backup_relay_{suite}.json
Usage:  python3 ablate_backup_relay.py [--suite 64q]
"""
from __future__ import annotations

import argparse
import os
import time

from ablate_common import (RESULTS_DIR, SUITE_CIRCUITS, SUITES, default_layouts,
                           gmean, load_circuit, meta, route_layout_set, save_json,
                           summarise)
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt

CONDITIONS = [
    ("greedy", dict(backup_relay_mode=False)),
    ("relay",  dict(backup_relay_mode=True)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="64q")
    ap.add_argument("--circuits", default=None)
    args = ap.parse_args()

    suite    = args.suite
    s        = SUITES[suite]
    arch     = s["arch"]
    circuits = (args.circuits.split(",") if args.circuits
               else SUITE_CIRCUITS.get(suite, SUITE_CIRCUITS["64q"]))

    out_path = os.path.join(RESULTS_DIR, f"results_ablate_backup_relay_{suite}.json")
    payload = dict(meta("backup_relay_mode ablation", suite, circuits=circuits,
                        conditions=[k for k, _ in CONDITIONS]),
                  rows={})

    col_w = 12
    hdr = f"{'circuit':<{col_w}}  {'cx':>6}"
    for key, _ in CONDITIONS:
        hdr += f"  {'epr@'+key:>10}  {'ls@'+key:>8}  {'backups@'+key:>11}  {'relay@'+key:>8}"
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
                                                layout_failures=info["layout_failures"],
                                                relay_hops=best.get("relay_hops", 0) if best else None))
            payload["rows"][cname]["by_cond"][key] = result
            if result.get("aborted"):
                row_str += f"  {'ABORT':>10}  {'---':>8}  {'---':>11}  {'---':>8}"
            else:
                row_str += (f"  {result['eprs']:>10}  {result['ls']:>8}  "
                           f"{result['backup_activations']:>11}  {result.get('relay_hops', 0):>8}")
            save_json(out_path, payload)

        print(row_str, flush=True)

    print("\n" + "=" * 60, flush=True)
    print(f"  {suite}: relay vs greedy backup plan", flush=True)
    print("=" * 60, flush=True)
    epr_ratios, ls_ratios = [], []
    for cname in circuits:
        r0 = payload["rows"][cname]["by_cond"].get("greedy", {})
        r1 = payload["rows"][cname]["by_cond"].get("relay", {})
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
              f"   ls {ls0:>6} -> {ls1:>6} ({ls_pct:+6.1f}%)"
              f"   backups {r0['backup_activations']:>3} -> {r1['backup_activations']:>3}", flush=True)
    if epr_ratios:
        print(f"    {'GMEAN':<{col_w}}  epr x{gmean(epr_ratios):.4f}"
              f"   ls x{gmean(ls_ratios):.4f}", flush=True)

    save_json(out_path, payload)
    print(f"\nSaved {out_path}", flush=True)


if __name__ == "__main__":
    main()
