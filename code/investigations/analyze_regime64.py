"""
analyze_regime64.py — emit the tight-machine congestion-relief table
(`tab:regime` in the paper) over the FULL nine-circuit 64q suite.

Merges the original six-circuit run
    results/results_ablate_occupancy_64q_c33.json
with the completion run
    results/results_ablate_occupancy_64q_c33_extra3.json
and prints the LaTeX table body plus the completion tallies quoted in prose.

`--verify6` re-prints the six-circuit table exactly as published, so the merge
logic can be checked against the paper before the new rows are trusted.

Usage:  python3 code/analyze_regime64.py [--verify6]
"""

from __future__ import annotations

import argparse
import json
import os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
RESULTS = os.path.join(_HERE, "results")

SIX = ["ae", "ghz", "graphstate", "qft", "qnn", "random"]
NINE = SIX + ["qpeexact", "qaoa", "multiplier"]
# Table row order: cheap structured circuits first, then by CX.
ROW_ORDER = ["ghz", "graphstate", "ae", "qft", "random", "qpeexact",
             "qaoa", "qnn", "multiplier"]


def load(only_six: bool = False):
    rows = []
    files = [os.path.join(RESULTS, "results_ablate_occupancy_64q_c33.json")]
    if not only_six:
        extra = os.path.join(RESULTS,
                             "results_ablate_occupancy_64q_c33_extra3.json")
        if os.path.exists(extra):
            files.append(extra)
    for f in files:
        with open(f) as fh:
            rows += json.load(fh)["results"]
    cell = {}
    cx = {}
    for r in rows:
        cell[(r["circuit"], r["profile"], r["config"])] = r
        cx[r["circuit"]] = r["cx"]
    return cell, cx


def fmt(r):
    if r is None:
        return "--"
    if r.get("aborted"):
        return r"\textsc{a}"
    return str(r["eprs"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify6", action="store_true",
                    help="print the original six-circuit table for comparison")
    args = ap.parse_args()

    circuits = SIX if args.verify6 else NINE
    cell, cx = load(only_six=args.verify6)
    order = [c for c in ROW_ORDER if c in circuits]

    cols = [("default", "full"), ("default", "no_relief"),
            ("uniform", "full"), ("uniform", "no_relief")]

    print(f"\n%% tab:regime body — {len(order)} circuits"
          f"{' (VERIFY: original six)' if args.verify6 else ' (full 64q suite)'}")
    print("% BODY: code/results/results_ablate_occupancy_64q_c33"
          "{,_extra3}.json via analyze_regime64.py")
    missing = []
    for c in order:
        cells = [cell.get((c, p, g)) for p, g in cols]
        if any(x is None for x in cells):
            missing.append(c)
        vals = " & ".join(f"{fmt(x):>11s}" for x in cells)
        print(f"{c:<12} & {cx[c]:>5} & {vals} \\\\")

    tally = []
    for p, g in cols:
        done = sum(1 for c in order
                   if (r := cell.get((c, p, g))) is not None and not r.get("aborted"))
        tally.append(f"{done}/{len(order)}")
    print(f"\\cmidrule(lr){{1-6}}")
    print(f"\\textbf{{completed}} & & " +
          " & ".join(f"\\textbf{{{t}}}" if i in (0, 2) else t
                     for i, t in enumerate(tally)) + r" \\")

    if missing:
        print(f"\n!! INCOMPLETE — no data yet for: {', '.join(missing)}")

    # Prose figures
    print("\n── figures quoted in prose ──")
    for p, label in (("default", "own"), ("uniform", "uniform")):
        ok = [c for c in order if (r := cell.get((c, p, "full"))) and not r.get("aborted")]
        no = [c for c in order
              if (r := cell.get((c, p, "no_relief"))) and not r.get("aborted")]
        ab = [c for c in order
              if (r := cell.get((c, p, "no_relief"))) and r.get("aborted")]
        print(f"  layout {label:<8} relief: {len(ok)}/{len(order)} complete"
              f"   no_relief: {len(no)}/{len(order)} complete, {len(ab)} abort")
        print(f"      aborts without relief: {', '.join(ab) if ab else '(none)'}")

    # Rollback activity.  The paper quotes `backup_activations` (checkpoint
    # restores), maxed over both layouts with relief on -- not force_make_room.
    acts = [(c, p, (cell.get((c, p, "full")) or {}).get("backup_activations", 0) or 0)
            for c in order for p in ("default", "uniform")]
    live = [t for t in acts if t[2]]
    if live:
        c, p, v = max(live, key=lambda t: t[2])
        print(f"  checkpoint--rollback: max {v} activations on {c} "
              f"(layout {'own' if p == 'default' else p}); "
              f"nonzero in {len(live)}/{len(acts)} relief cells")

    # hop-gain completion effect, quoted in the paper for qnn
    print("\n── hop-gain on this machine (layout own) ──")
    for c in order:
        f_, n_ = cell.get((c, "default", "full")), cell.get((c, "default", "no_hop_gain"))
        if f_ and n_:
            print(f"  {c:<12} full={fmt(f_):>11s}   no_hop_gain={fmt(n_):>11s}")


if __name__ == "__main__":
    main()
