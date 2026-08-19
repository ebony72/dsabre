"""
analyze_telegate.py — merge the telegate study's result JSONs into one report.

Reads whichever of these exist in `results/` and prints a per-circuit EPR
table plus the geometric mean against the `off` (published router) anchor:

    telegate_64q.json            H-grid 2x3 of 4x4, 96 physical -- the paper's
                                 architecture; full score/bias/amortisation sweep
    telegate_64q_selective.json  same chip, the selective rule
    telegate_64q_c33.json        3x3 of 3x3, 81 physical -- capacity binds
    telegate_64q_b243.json       2x4 of 3x3, 72 physical -- F = K, capacity binds hard
    telegate_amortization_64q.json  gates served per teledata EPR pair

`gmean` is over circuits BOTH variants completed; a variant that aborts a
circuit the anchor routed is reported as an abort count rather than silently
dropped, because on the tight chips routability is the result.

Usage:  python3 analyze_telegate.py [--variants off,nat,b8,sel,tgmax]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
RESULTS = os.environ.get("DSABRE_OUT_DIR") or os.path.join(_HERE, "results")

from ablate_common import gmean

FILES = [
    ("64q  H-grid 2x3 of 4x4 (96 phys, 33% spare)", "telegate_64q.json"),
    ("64q  selective rule, same chip",              "telegate_64q_selective.json"),
    ("64q  3x3 of 3x3 (81 phys, capacity binds)",   "telegate_64q_c33.json"),
    ("64q  2x4 of 3x3 (72 phys, F = K)",            "telegate_64q_b243.json"),
]


def report(title, path, want=None):
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    res = d["results"]
    circuits = [c for c in d["meta"]["circuits"] if c in res]
    variants = [v for v in d["meta"]["variants"]
                if (want is None or v in want)
                and any(v in res[c] for c in circuits)]
    if not circuits or "off" not in variants:
        return
    print(f"\n{'='*(13 + 8*len(variants))}")
    print(f"{title}   [{d['meta'].get('mode', 'safe')} mode]")
    print(f"{'='*(13 + 8*len(variants))}")
    print(f"{'circuit':12s}" + "".join(f"{v:>8s}" for v in variants))
    print("-" * (12 + 8 * len(variants)))
    for c in circuits:
        line = f"{c:12s}"
        for v in variants:
            r = res[c].get(v)
            line += f"{'--':>8s}" if r is None else (
                f"{'ABORT':>8s}" if r["aborted"] else f"{r['eprs']:>8d}")
        print(line)
    print("-" * (12 + 8 * len(variants)))
    # gmean of the EPR ratio to `off`, over circuits both completed.
    line_g, line_n, line_a = f"{'gmean vs off':12s}", f"{'  n circuits':12s}", f"{'  aborts':12s}"
    for v in variants:
        ratios, aborts = [], 0
        for c in circuits:
            a, b = res[c].get("off"), res[c].get(v)
            if b is None:
                continue
            if b["aborted"]:
                aborts += 1
                continue
            if a is None or a["aborted"] or a["eprs"] == 0:
                continue
            ratios.append(b["eprs"] / a["eprs"])
        line_g += (f"{100*(gmean(ratios)-1):+7.1f}%" if ratios else f"{'--':>8s}")
        line_n += f"{len(ratios):>8d}"
        line_a += f"{aborts:>8d}"
    print(line_g); print(line_n); print(line_a)
    # `off` aborts are the other direction of the routability story.
    off_aborts = [c for c in circuits
                  if res[c].get("off") and res[c]["off"]["aborted"]]
    if off_aborts:
        print(f"\n  circuits the published router ABORTS here: {', '.join(off_aborts)}")
        for c in off_aborts:
            rescued = [v for v in variants
                       if v != "off" and res[c].get(v) and not res[c][v]["aborted"]]
            if rescued:
                print(f"    {c}: completed by "
                      + ", ".join(f"{v} ({res[c][v]['eprs']} EPR)" for v in rescued))


def amortization(path):
    if not os.path.exists(path):
        return
    d = json.load(open(path))
    t = d.get("totals")
    if not t:
        return
    print(f"\n{'='*64}\nWhat one teledata EPR pair buys (published router, 64q)\n{'='*64}")
    print(f"{'circuit':12s} {'EPR':>6s} {'gates/EPR':>10s} {'transit%':>9s} "
          f"{'=1 gate%':>9s} {'>=2 gates%':>11s}")
    for c, r in d["per_circuit"].items():
        n = r["hops"]
        print(f"{c:12s} {r['eprs']:6d} {r['gates_per_hop']:10.2f} "
              f"{100*r['transit']/n:8.1f}% {100*r['exactly_one']/n:8.1f}% "
              f"{100*r['two_or_more']/n:10.1f}%")
    n = t["hops"]
    print("-" * 60)
    print(f"{'SUITE':12s} {n:6d} {t['gates_per_hop']:10.2f} "
          f"{100*t['transit']/n:8.1f}% {100*t['exactly_one']/n:8.1f}% "
          f"{100*t['two_or_more']/n:10.1f}%")
    print("\n  A single-gate telegate serves exactly 1 gate per EPR pair.\n"
          f"  Teledata serves {t['gates_per_hop']:.1f} on this suite, so the "
          "telegate can only\n  compete on the "
          f"{100*(t['transit']+t['exactly_one'])/n:.0f}% of hops that serve at "
          "most one gate.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default=None,
                    help="comma-separated subset to print (default: all)")
    args = ap.parse_args()
    want = set(args.variants.split(",")) | {"off"} if args.variants else None
    for title, fn in FILES:
        report(title, os.path.join(RESULTS, fn), want)
    amortization(os.path.join(RESULTS, "telegate_amortization_64q.json"))


if __name__ == "__main__":
    main()
