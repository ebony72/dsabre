"""
ablate_commitment.py — does "remembering" the gate a teleport was for and
softly favoring its continuation next iteration reduce EPR/SWAP cost?

Motivation (2026-08-03 trace, see CHANGES_FROM_SUBMITTED.md / conversation
log): instrumenting dSE's actual candidate scoring on the 64q suite showed
the router re-scores every competing inter-core front-layer gate from
scratch each iteration, with no memory of which one it was mid-move on.
Across ae/ghz/graphstate/multiplier/qaoa/qft/qpeexact/random/vqe_su2/wstate,
of the 1594 remote gates that needed >=2 teleport-relevant iterations to
clear: 90.8% had a DIFFERENT gate's teleport served while they sat pending,
and 35.6% saw their own qubit-pair distance actually INCREASE mid-flight
(98.6% of those from a different gate's eviction side effect, not their
own action) -- i.e. the router does not carry a remote gate to completion
once it starts moving it.

`config.HardwareConfig.commit_bonus` (default 0.0, no behavior change) makes
_generate_candidates discount the score of any candidate that continues
teleporting the SAME DAG node the previous iteration's teleport advanced,
so it tends to win ties/near-ties against competing gates without forcing
it to win against a genuinely better move. `commit_hard_lock=True` is the
hard variant: while the committed gate still has a legal candidate, no
competing gate's candidates are even generated (see its docstring for the
one-iteration release when the locked gate is temporarily stuck). This
script sweeps commit_bonus values AND the hard-lock condition over the 64q
suite (same production layout search + fwd-bwd-fwd protocol as
benchmark.py) and reports EPR/SWAP deltas against commit_bonus=0.

Output: code/results/results_ablate_commitment.json  (written after each row)
Usage:  python3 ablate_commitment.py [--bonuses 0,2,5,10,20,hard] [--suite 64q]
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


def _parse_bonus_token(tok: str):
    """'hard' -> ('hard', True); anything else -> (str(float), False)."""
    if tok.strip().lower() == "hard":
        return "hard", True
    return str(float(tok)), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="64q")
    ap.add_argument("--bonuses", default="0,2,5,10,20,hard")
    ap.add_argument("--circuits", default=None,
                    help="comma-separated override; defaults to the full suite table")
    args = ap.parse_args()

    suite   = args.suite
    tokens  = [_parse_bonus_token(x) for x in args.bonuses.split(",")]
    s       = SUITES[suite]
    arch    = s["arch"]
    circuits = (args.circuits.split(",") if args.circuits
               else SUITE_CIRCUITS.get(suite, SUITE_CIRCUITS["64q"]))

    out_path = os.path.join(RESULTS_DIR, f"results_ablate_commitment_{suite}.json")
    payload = dict(meta("commit_bonus / hard_lock sweep", suite, circuits=circuits,
                        conditions=[k for k, _ in tokens]),
                  rows={})

    col_w = 12
    hdr = f"{'circuit':<{col_w}}  {'cx':>6}"
    for key, _ in tokens:
        hdr += f"  {'epr@'+key:>9}  {'ls@'+key:>8}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    for cname in circuits:
        qc, dag, rev_dag, n_cx = load_circuit(suite, cname)
        row_str = f"{cname:<{col_w}}  {n_cx:>6}"
        payload["rows"][cname] = dict(n_cx=n_cx, by_bonus={})

        # Layout candidates are independent of commit_bonus: reuse across the sweep.
        layouts = default_layouts(qc, dag, arch, s["n_qubits"], seed=0)

        for key, is_hard in tokens:
            hw = HardwareConfig(deadlock_limit=s["hw"].deadlock_limit,
                               max_backup_attempts=s["hw"].max_backup_attempts,
                               max_iterations=s["hw"].max_iterations,
                               commit_bonus=0.0 if is_hard else float(key),
                               commit_hard_lock=is_hard)
            router = dSABRE_BurstExt(arch, hw)
            t0 = time.perf_counter()
            best, info = route_layout_set(router, dag, rev_dag, layouts, pattern="fbf")
            elapsed = time.perf_counter() - t0
            result = summarise(best, extra=dict(condition=key, wall_s=round(elapsed, 2),
                                                pass1_aborts=info["pass1_aborts"],
                                                layout_failures=info["layout_failures"]))
            payload["rows"][cname]["by_bonus"][key] = result
            if result.get("aborted"):
                row_str += f"  {'ABORT':>9}  {'---':>8}"
            else:
                row_str += f"  {result['eprs']:>9}  {result['ls']:>8}"
            save_json(out_path, payload)  # partial results after every cell

        print(row_str, flush=True)

    # ── Summary: EPR/SWAP delta vs commit_bonus=0, per circuit and gmean ───────
    print("\n" + "=" * 60, flush=True)
    print(f"  {suite}: condition deltas vs baseline (bonus=0.0)", flush=True)
    print("=" * 60, flush=True)
    keys = [k for k, _ in tokens]
    base_key = "0.0" if "0.0" in keys else keys[0]
    for key in keys:
        if key == base_key:
            continue
        epr_ratios, ls_ratios = [], []
        print(f"\n  condition={key} vs {base_key}:", flush=True)
        for cname in circuits:
            r0 = payload["rows"][cname]["by_bonus"].get(base_key, {})
            r1 = payload["rows"][cname]["by_bonus"].get(key, {})
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
