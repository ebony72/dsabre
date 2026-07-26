"""
compare_hopgain.py — per-circuit effect of retiring the hop-gain term.

Compares results_hopgain_on/ (the snapshot taken before hop_gain was set to
0.0) against the regenerated results/ directory, for the dSE router that the
paper reports.  Prints per-circuit EPR deltas and per-suite geometric means so
the removal can be judged on the full suites rather than on the six-circuit
ablation subset.

Usage: python3 compare_hopgain.py
"""

import json, os
from math import prod

_HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(_HERE, "results_hopgain_on")
NEW = os.path.join(_HERE, "results")


def gmean(v):
    v = [x for x in v if x is not None and x > 0]
    return prod(v) ** (1 / len(v)) if v else float("nan")


def load(d, suite):
    p = os.path.join(d, f"results_{suite}.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return {r["circuit"]: r for r in json.load(f)["results"]}


def epr(rec, key="dSE"):
    if rec is None:
        return None
    r = rec["routers"].get(key, {})
    return None if r.get("aborted") else r.get("eprs")


def main():
    print(f"{'suite':>5}  {'circuit':<12} {'on':>7} {'off':>7} {'delta%':>8}",
          flush=True)
    print("-" * 46, flush=True)
    grand_on, grand_off = [], []
    for suite in ("25q", "36q", "64q"):
        old, new = load(OLD, suite), load(NEW, suite)
        if not old or not new:
            print(f"{suite:>5}  [missing]", flush=True)
            continue
        on_v, off_v = [], []
        for name in sorted(set(old) & set(new)):
            a, b = epr(old[name]), epr(new[name])
            if a is None or b is None:
                continue
            d = 100 * (b - a) / a if a else float("nan")
            flag = "  <-- worse" if d > 2 else ("  <-- better" if d < -2 else "")
            print(f"{suite:>5}  {name:<12} {a:>7} {b:>7} {d:>+7.1f}%{flag}",
                  flush=True)
            on_v.append(a); off_v.append(b)
        if on_v:
            go, gf = gmean(on_v), gmean(off_v)
            print(f"{suite:>5}  {'GMEAN':<12} {go:>7.1f} {gf:>7.1f} "
                  f"{100*(gf-go)/go:>+7.1f}%", flush=True)
            print("-" * 46, flush=True)
            grand_on += on_v; grand_off += off_v
    if grand_on:
        go, gf = gmean(grand_on), gmean(grand_off)
        print(f"{'ALL':>5}  {'GMEAN':<12} {go:>7.1f} {gf:>7.1f} "
              f"{100*(gf-go)/go:>+7.1f}%  ({len(grand_on)} circuits)",
              flush=True)
        print("\nPositive delta means the four-term score (hop gain removed) "
              "uses MORE EPR pairs.", flush=True)


if __name__ == "__main__":
    main()
