"""
ablate_sabrelayout.py — is the SabreLayout pass worth having at all, given
that the fwd->bwd->fwd protocol already refines the layout?

The production pipeline is: SabreLayout on the corner-removed coupling map
(best of 3 seeds) -> fwd->bwd->fwd routing.  But SABRE's own philosophy says
the bwd pass IS a layout optimiser — so maybe any corner-respecting start
would do, and SabreLayout is redundant with the protocol behind it.

Arms (every one routed with the identical fwd->bwd->fwd protocol):

    sabre3   production: SabreLayout on the corner-removed map, seeds 0-2,
             best EPR over the 3 starts
    sabre1   the same, seed 0 only — separates "SabreLayout is good" from
             "best-of-3 diversity is good"
    seq      no SabreLayout: logical qubit i on the i-th non-reserved
             physical slot in core-major order (fills cores one by one)
    rand3    no SabreLayout: 3 seeded shuffles of the non-reserved slots,
             best EPR over the 3 starts

Compute-matched comparisons: sabre1 vs seq (one start each) and sabre3 vs
rand3 (three starts each).  All arms respect the production corner
reservation (adaptive k), so the only variable is the placement algorithm.
On 64q_c33 the adaptive k is 0, so seq/rand place on ALL 81 slots — and
SabreLayout there packs cores solid anyway, so the comparison shows whether
its placement beats a shuffle even where its occupancy shape is pathological.

Output: code/results/results_ablate_sabrelayout_{suite}.json  (per row)
Usage:  python3 code/ablate_sabrelayout.py --suite 25q 64q 64q_c33
"""

from __future__ import annotations

import argparse
import os
import random as _random
import time

from ablate_common import (ABLATION_CIRCUITS, RESULTS_DIR, SUITES, core_occupancy,
                           default_layouts, gmean, load_circuit, meta,
                           route_layout_set, save_json, summarise)
from layout import _per_core_reserved_corner_nodes, adaptive_corner_count
from dsabre_ext import dSABRE_BurstExt

ARMS = ["sabre3", "sabre1", "seq", "rand3"]


def _free_slots(arch, reserved):
    return [p for p in sorted(arch.data_qubits) if p not in reserved]


def seq_layout(dag, arch, reserved):
    slots = _free_slots(arch, reserved)
    return {lq: slots[i] for i, lq in enumerate(dag.qubits)}


def rand_layouts(dag, arch, reserved, seeds=(0, 1, 2)):
    slots = _free_slots(arch, reserved)
    outs = []
    for sd in seeds:
        sl = slots[:]
        _random.Random(sd).shuffle(sl)
        outs.append({lq: sl[i] for i, lq in enumerate(dag.qubits)})
    return outs


def run_suite(suite: str):
    s = SUITES[suite]
    arch, hw, n = s["arch"], s["hw"], s["n_qubits"]
    router = dSABRE_BurstExt(arch, hw)
    out_path = os.path.join(RESULTS_DIR, f"results_ablate_sabrelayout_{suite}.json")
    if os.path.exists(out_path):
        raise SystemExit(f"ABORT: {out_path} exists; move it aside first.")

    k = adaptive_corner_count(arch, n)
    reserved = _per_core_reserved_corner_nodes(arch, per_core=k)

    print(f"\n{'='*100}\n  SABRELAYOUT VALUE — {suite}  ({s['arch_name']})", flush=True)
    print(f"  corner reservation: k={k}  ({len(reserved)} slots withheld from "
          f"every arm)\n{'='*100}", flush=True)

    payload = dict(meta=meta("sabrelayout_value", suite,
                             k_reserved=k, n_reserved=len(reserved),
                             arms=dict(
                                 sabre3="SabreLayout corner-removed, seeds 0-2",
                                 sabre1="SabreLayout corner-removed, seed 0",
                                 seq="sequential fill of non-reserved slots",
                                 rand3="3 seeded shuffles of non-reserved slots"),
                             fairness="sabre1 vs seq are 1-start arms; "
                                      "sabre3 vs rand3 are 3-start arms; all "
                                      "routed fwd->bwd->fwd"),
                   results=[])

    circuits = {c: load_circuit(suite, c) for c in ABLATION_CIRCUITS}

    t_all = time.time()
    for arm in ARMS:
        print(f"\n── {suite} | {arm} " + "─" * 40, flush=True)
        print(f"{'circuit':<12} {'cx':>6} {'occ(first layout)':<30} {'p1ab':>5} "
              f"{'fail':>5} {'eprs':>7} {'ls':>7} {'t(s)':>8}", flush=True)
        eprs_list = []
        for c in ABLATION_CIRCUITS:
            qc, dag, rev, ncx = circuits[c]
            if arm == "sabre3":
                layouts = default_layouts(qc, dag, arch, n, seed=0, n_seeds=3)
            elif arm == "sabre1":
                layouts = default_layouts(qc, dag, arch, n, seed=0, n_seeds=1)
            elif arm == "seq":
                layouts = [seq_layout(dag, arch, reserved)]
            else:
                layouts = rand_layouts(dag, arch, reserved)
            occ = core_occupancy(layouts[0], arch) if layouts else None
            m, info = route_layout_set(router, dag, rev, layouts, pattern="fbf")
            row = summarise(m, extra=dict(
                suite=suite, arm=arm, circuit=c, cx=ncx, occupancy=occ,
                n_layouts=info["n_layouts"],
                pass1_aborts=info["pass1_aborts"],
                layout_failures=info["layout_failures"],
                total_time=round(info["total_time"], 1)))
            payload["results"].append(row)
            save_json(out_path, payload)
            if row["aborted"]:
                print(f"{c:<12} {ncx:>6} {str(occ):<30} {info['pass1_aborts']:>4}/"
                      f"{info['n_layouts']} {info['layout_failures']:>4}/"
                      f"{info['n_layouts']} {'ABORT':>7}", flush=True)
            else:
                eprs_list.append(row["eprs"])
                print(f"{c:<12} {ncx:>6} {str(occ):<30} {info['pass1_aborts']:>4}/"
                      f"{info['n_layouts']} {info['layout_failures']:>4}/"
                      f"{info['n_layouts']} {row['eprs']:>7} {row['ls']:>7} "
                      f"{info['total_time']:>8.1f}", flush=True)
        print(f"{'GMEAN':<12} {'':>6} {'':<30} {'':>5} {'':>5} "
              f"{gmean(eprs_list):>7.1f}", flush=True)

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
