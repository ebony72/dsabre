"""
ablate_protocol.py — is the fwd -> bwd -> fwd protocol worth its 3x cost?

The production protocol routes the DAG forward, routes the *reversed* DAG from
the resulting layout (a layout-refinement pass whose own EPR count is discarded)
and routes forward again, keeping the better of the two forward passes.  Two
distinct things could be doing the work:

  (a) the reversal — a backward pass polishes the initial layout the way
      SABRE's bidirectional layout iteration does; or
  (b) mere repetition — any extra pass re-seeded from the previous final layout
      would help just as much.

Table IX's pass-count sweep varies only pure-forward repetition, so it cannot
separate the two.  This study runs both families at matched pass counts:

  f       1 pass    forward only
  ff      2 passes  forward, re-seeded from its own final layout
  fff     3 passes  ← the repetition control for fbf
  fbf     3 passes  ← production protocol
  fbfbf   5 passes  does a second reversal add anything

`fff` vs `fbf` is the comparison that isolates the backward pass; `f` vs `fbf`
is what the protocol is worth overall; `fbfbf` vs `fbf` says whether to keep
going.  Time is reported as the full best-of-3-layouts wall clock, so the
EPR-per-second trade is visible.

Output: code/results/results_ablate_protocol_{suite}.json  (written per row)
Usage:  python3 code/ablate_protocol.py [--suite 25q|64q|all]
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
                           default_layouts, gmean, load_circuit, meta, save_json,
                           summarise)
from dsabre_ext import dSABRE_BurstExt

PROTOCOLS = ["f", "ff", "fff", "fbf", "fbfbf"]


def run_suite(suite: str):
    s = SUITES[suite]
    arch, hw, n = s["arch"], s["hw"], s["n_qubits"]
    router = dSABRE_BurstExt(arch, hw)
    out_path = os.path.join(RESULTS_DIR, f"results_ablate_protocol_{suite}.json")

    print(f"\n{'='*88}\n  PASS-PROTOCOL STUDY — {suite}  ({s['arch_name']})\n{'='*88}", flush=True)
    payload = dict(meta=meta("pass_protocol", suite,
                             protocols=PROTOCOLS,
                             layout="production corner-removed SabreLayout, best of 3 seeds",
                             note="'b' passes route the reversed DAG and only refine the "
                                  "layout; reported EPR is the best forward pass"),
                   results=[])

    circuits = {c: load_circuit(suite, c) for c in ABLATION_CIRCUITS}
    layouts = {}
    for c, (qc, dag, rev, ncx) in circuits.items():
        layouts[c] = default_layouts(qc, dag, arch, n, seed=0, n_seeds=3)

    t_all = time.time()
    for pat in PROTOCOLS:
        print(f"\n── {suite} | protocol {pat} ({len(pat)} passes) " + "─" * 30, flush=True)
        print(f"{'circuit':<12} {'cx':>6} {'eprs':>6} {'ls':>7} {'t(s)':>8}", flush=True)
        eprs_list, ls_list, t_list = [], [], []
        for c in ABLATION_CIRCUITS:
            qc, dag, rev, ncx = circuits[c]
            m = best_over_layouts(router, dag, rev, layouts[c], pattern=pat)
            row = summarise(m, extra=dict(suite=suite, protocol=pat,
                                          passes=len(pat), circuit=c, cx=ncx))
            payload["results"].append(row)
            save_json(out_path, payload)
            if row["aborted"]:
                print(f"{c:<12} {ncx:>6} {'ABORT':>6}", flush=True)
            else:
                eprs_list.append(row["eprs"]); ls_list.append(row["ls"])
                t_list.append(row["time_s"])
                print(f"{c:<12} {ncx:>6} {row['eprs']:>6} {row['ls']:>7} "
                      f"{row['time_s']:>8.1f}", flush=True)
        print(f"{'GMEAN':<12} {'':>6} {gmean(eprs_list):>6.1f} {gmean(ls_list):>7.1f} "
              f"{gmean(t_list):>8.2f}", flush=True)

    print(f"\nSaved → {out_path}   ({time.time()-t_all:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", choices=["25q", "64q", "all"], default="all")
    a = ap.parse_args()
    for suite in (["25q", "64q"] if a.suite == "all" else [a.suite]):
        run_suite(suite)


if __name__ == "__main__":
    main()
