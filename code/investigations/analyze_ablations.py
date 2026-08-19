"""
analyze_ablations.py — summary tables for the 2026-07-29 dSE ablations.

Reads whatever `results/results_ablate_*.json` files exist (the drivers write
them row by row, so partial runs summarise fine) and prints:

  occupancy  gmean EPR per (profile x mechanism config), each config as a ratio
             to `full` at the same occupancy profile, plus how often congestion
             relief actually generated and won candidates.
  protocol   gmean EPR / SWAP / time per pass protocol, ratios to `fbf`.
  corners    gmean EPR per reservation variant, ratio to the adaptive default.

Usage: python3 code/analyze_ablations.py
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from ablate_common import RESULTS_DIR, gmean

SUITES = ["25q", "64q", "64q_c33"]


def _load(name):
    p = os.path.join(RESULTS_DIR, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def _by(rows, *keys):
    out = {}
    for r in rows:
        out.setdefault(tuple(r[k] for k in keys), []).append(r)
    return out


def _g(rows, field="eprs"):
    return gmean([r[field] for r in rows if not r.get("aborted")])


def _paired(rows_a, rows_b, field="eprs"):
    """(gmean_a, gmean_b, n) over the circuits both arms completed.

    Ratios must be paired: a run still in progress, or a circuit that aborted
    in one arm only, otherwise compares different circuit sets and can invert
    the sign of the effect.
    """
    a = {r["circuit"]: r[field] for r in rows_a if not r.get("aborted")}
    b = {r["circuit"]: r[field] for r in rows_b if not r.get("aborted")}
    common = sorted(set(a) & set(b))
    return gmean([a[c] for c in common]), gmean([b[c] for c in common]), len(common)


def _ratio(a, b):
    if not a or not b or b != b or b == 0:
        return float("nan")
    return a / b


def _fmt_ratio(x):
    return "  ---  " if x != x else f"{x:6.3f}"


# ── Occupancy ─────────────────────────────────────────────────────────────────

def report_occupancy(suite):
    d = _load(f"results_ablate_occupancy_{suite}.json")
    if not d:
        return
    rows = d["results"]
    profs = [p["name"] for p in d["meta"]["profiles"]]
    peak = {p["name"]: p["peak_fill"] for p in d["meta"]["profiles"]}
    configs = d["meta"]["configs"]
    grouped = _by(rows, "profile", "config")

    print(f"\n{'='*104}")
    print(f"  OCCUPANCY SKEW — {suite}   (dSE, gmean EPR over {len(d['meta']['circuits'])} circuits)")
    print(f"{'='*104}")
    ncirc = len(d["meta"]["circuits"])
    print(f"{'profile':<13} {'peak':>5} {'budgets':<24} {'n':>3} "
          + "".join(f"{c:>13}" for c in configs)
          + f"{'rel_cand':>10}{'rel_pick':>9}{'fmr':>7}")
    print(f"{'':<13} {'fill':>5} {'':<24} {'':>3} "
          + "".join(f"{'gmean  ×full':>13}" for _ in configs))
    print("-" * 112)

    default_full = grouped.get(("default", "full"), [])
    budgets = {p["name"]: p["budgets"] for p in d["meta"]["profiles"]}
    for prof in profs:
        full_rows = grouped.get((prof, "full"), [])
        if not full_rows:
            continue
        done = sum(1 for r in full_rows if not r.get("aborted"))
        line = (f"{prof:<13} {str(peak[prof] or '~'):>5} "
                f"{str(budgets[prof] or 'production'):<24} "
                f"{str(done) + ('' if done == ncirc else '!'):>3} ")
        for c in configs:
            rs = grouped.get((prof, c), [])
            if not rs:
                line += f"{'---':>13}"
                continue
            ga, gf, _ = _paired(rs, full_rows)
            line += f"{_g(rs):6.1f}{_fmt_ratio(_ratio(ga, gf)):>7}"
        rc = sum(r.get("relief_candidates", 0) for r in full_rows if not r.get("aborted"))
        rp = sum(r.get("relief_picks", 0) for r in full_rows if not r.get("aborted"))
        fm = sum(r.get("force_make_room", 0) for r in full_rows if not r.get("aborted"))
        line += f"{rc:>10}{rp:>9}{fm:>7}"
        gp, gd, nn = _paired(full_rows, default_full)
        if nn:
            line += f"   (skew ×{_ratio(gp, gd):.2f} vs default, n={nn})"
        print(line)

    # Routability.  The gmean column above is over each arm's OWN completed
    # circuits, so it is not comparable across columns when arms abort at
    # different rates — only the paired ×full ratio is.  A config that aborts
    # is a harder failure than a config that is a few percent worse, so the
    # abort counts get their own table rather than a footnote.
    aborts = {(p, c): sum(1 for r in grouped.get((p, c), []) if r.get("aborted"))
              for p in profs for c in configs}
    if any(aborts.values()):
        print(f"\n  circuits that FAILED TO ROUTE (of {ncirc})")
        print(f"  {'profile':<15}" + "".join(f"  {c:<24}" for c in configs))
        for p in profs:
            if not any(aborts[(p, c)] for c in configs):
                continue
            line = f"  {p:<15}"
            for c in configs:
                n_ab = aborts[(p, c)]
                names = ",".join(sorted(r["circuit"] for r in grouped.get((p, c), [])
                                        if r.get("aborted")))
                line += f"  {(str(n_ab) + ' ' + names) if n_ab else '-':<24}"
            print(line)

    # Per-circuit extremes for the two mechanisms
    for c in ("no_hop_gain", "no_relief"):
        print(f"\n  per-circuit {c}/full ratio  (>1 = mechanism helps)")
        hdr = f"  {'profile':<13}"
        circs = d["meta"]["circuits"]
        for cc in circs:
            hdr += f"{cc:>12}"
        print(hdr)
        for prof in profs:
            fr = {r["circuit"]: r for r in grouped.get((prof, "full"), [])
                  if not r.get("aborted")}
            ar = {r["circuit"]: r for r in grouped.get((prof, c), [])
                  if not r.get("aborted")}
            if not fr or not ar:
                continue
            line = f"  {prof:<13}"
            for cc in circs:
                if cc in fr and cc in ar and fr[cc]["eprs"]:
                    line += f"{ar[cc]['eprs']/fr[cc]['eprs']:>12.3f}"
                else:
                    line += f"{'---':>12}"
            print(line)


# ── Protocol ──────────────────────────────────────────────────────────────────

def report_protocol(suite):
    d = _load(f"results_ablate_protocol_{suite}.json")
    if not d:
        return
    rows = d["results"]
    prots = d["meta"]["protocols"]
    grouped = _by(rows, "protocol")

    print(f"\n{'='*104}")
    print(f"  PASS PROTOCOL — {suite}   (dSE, best of 3 SabreLayout seeds)")
    print(f"{'='*104}")
    ref_rows = grouped.get(("fbf",), [])
    ncirc = len(d["meta"]["circuits"])
    print(f"{'protocol':<10}{'passes':>7}{'n':>3}{'gmean EPR':>11}{'×fbf':>8}"
          f"{'gmean SWAP':>12}{'gmean t(s)':>12}{'×fbf t':>8}   per-circuit EPR")
    print("-" * 112)
    circs = d["meta"]["circuits"]
    for p in prots:
        rs = grouped.get((p,), [])
        if not rs:
            continue
        done = sum(1 for r in rs if not r.get("aborted"))
        ga, gb, n = _paired(rs, ref_rows)
        ta, tb, _ = _paired(rs, ref_rows, "time_s")
        per = {r["circuit"]: r["eprs"] for r in rs if not r.get("aborted")}
        per_s = " ".join(f"{c[:4]}={per.get(c, '--')}" for c in circs)
        print(f"{p:<10}{len(p):>7}{str(done) + ('' if done == ncirc else '!'):>3}"
              f"{_g(rs):>11.1f}{_ratio(ga, gb):>8.3f}"
              f"{_g(rs, 'ls'):>12.1f}{_g(rs, 'time_s'):>12.2f}"
              f"{_ratio(ta, tb):>8.2f}   {per_s}")
    print("\n  key contrasts (paired over circuits both arms completed):")
    for a, b, what in [("fff", "fbf", "backward pass vs equal-count repetition"),
                       ("f", "fbf", "whole protocol vs single pass"),
                       ("fbfbf", "fbf", "a second reversal")]:
        ga, gb, n = _paired(grouped.get((a,), []), grouped.get((b,), []))
        if ga == ga and gb == gb:
            print(f"    {a:>6} / {b:<6} = {ga/gb:6.3f}  (n={n})  ({what})")


# ── Corner variants ───────────────────────────────────────────────────────────

def report_corners(suite):
    d = _load(f"results_ablate_corners_{suite}.json")
    if not d:
        return
    rows = d["results"]
    k = d["meta"]["k_adaptive"]
    variants = [v["name"] for v in d["meta"]["variants"]]
    grouped = _by(rows, "variant")
    ref_name = f"k{k}"
    ref_rows = grouped.get((ref_name,), [])
    ncirc = len(d["meta"]["circuits"])

    print(f"\n{'='*104}")
    print(f"  CORNER RESERVATION — {suite}   (dSE; adaptive k = {k}, reference = {ref_name})")
    print(f"{'='*104}")
    print(f"{'variant':<14}{'withheld':>9}{'n':>3}{'gmean EPR':>11}{'×adaptive':>11}"
          f"{'gmean SWAP':>12}{'worst circuit':>22}")
    print("-" * 104)
    refper = {r["circuit"]: r["eprs"] for r in ref_rows if not r.get("aborted")}
    for v in variants:
        rs = grouped.get((v,), [])
        if not rs:
            continue
        g, gl = _g(rs), _g(rs, "ls")
        ga, gb, npair = _paired(rs, ref_rows)
        done = sum(1 for r in rs if not r.get("aborted"))
        nres = rs[0].get("n_reserved", "?")
        worst, worst_r = "-", 0.0
        for r in rs:
            if r.get("aborted"):
                worst, worst_r = r["circuit"] + " ABORT", 99.0
                continue
            b = refper.get(r["circuit"])
            if b:
                rr = r["eprs"] / b
                if rr > worst_r:
                    worst, worst_r = f"{r['circuit']} ×{rr:.2f}", rr
        print(f"{v:<14}{nres:>9}{str(done) + ('' if done == ncirc else '!'):>3}"
              f"{g:>11.1f}{_ratio(ga, gb):>11.3f}{gl:>12.1f}{worst:>22}")


# ── Free-slot study (Q1 routability + Q3 distribution, 2026-07-30) ────────────

FREESLOT_SUITES = ["25q_c133", "25q_c223", "64q_b243", "64q_c33", "64q"]


def report_freeslots(suite):
    d = _load(f"results_ablate_freeslots_{suite}.json")
    if not d:
        return
    rows = d["results"]
    grouped = _by(rows, "profile")
    profs = [p["name"] for p in d["meta"]["profiles"]]
    frees = {p["name"]: p["frees"] for p in d["meta"]["profiles"]}
    ncirc = len(d["meta"]["circuits"])
    ref_name = "uniform" if "uniform" in profs else profs[0]
    ref_rows = grouped.get((ref_name,), [])

    print(f"\n{'='*112}")
    print(f"  FREE SLOTS — {suite}   K={d['meta']['K']} spc={d['meta']['spc']} "
          f"F={d['meta']['F']}   (dSE, ref = {ref_name})")
    print(f"{'='*112}")
    print(f"{'profile':<17}{'frees (centre-first order: ' + str(d['meta']['core_priority']) + ')':<42}"
          f"{'min':>4}{'zero':>5}{'p1ab':>7}{'fail':>7}{'ABRT':>5}{'gmean':>8}{'×ref':>7}")
    print("-" * 112)
    for p in profs:
        rs = grouped.get((p,), [])
        if not rs:
            continue
        p1 = sum(r.get("pass1_aborts", 0) for r in rs)
        fl = sum(r.get("layout_failures", 0) for r in rs)
        nab = sum(1 for r in rs if r.get("aborted"))
        ga, gb, npair = _paired(rs, ref_rows)
        fr = frees[p]
        print(f"{p:<17}{str(fr):<42}{min(fr):>4}{fr.count(0):>5}"
              f"{p1:>4}/{3*ncirc:<3}{fl:>4}/{3*ncirc:<3}{nab:>5}"
              f"{_g(rs):>8.1f}{_fmt_ratio(_ratio(ga, gb)):>7}")
        if nab:
            names = ",".join(r["circuit"] for r in rs if r.get("aborted"))
            print(f"{'':<17}  ^ no result: {names}")


def report_sabrelayout(suite):
    d = _load(f"results_ablate_sabrelayout_{suite}.json")
    if not d:
        return
    rows = d["results"]
    grouped = _by(rows, "arm")
    arms = ["sabre3", "sabre1", "seq", "rand3"]
    ref_rows = grouped.get(("sabre3",), [])
    circs = d["meta"]["circuits"]

    print(f"\n{'='*112}")
    print(f"  SABRELAYOUT VALUE — {suite}   (k={d['meta']['k_reserved']} corners "
          f"reserved in every arm; all arms fwd->bwd->fwd)")
    print(f"{'='*112}")
    print(f"{'arm':<9}{'starts':>7}{'p1ab':>6}{'fail':>6}{'ABRT':>5}{'gmean EPR':>11}"
          f"{'×sabre3':>9}{'gmean t':>9}   per-circuit EPR")
    print("-" * 112)
    for a in arms:
        rs = grouped.get((a,), [])
        if not rs:
            continue
        nl = rs[0].get("n_layouts", "?")
        p1 = sum(r.get("pass1_aborts", 0) for r in rs)
        fl = sum(r.get("layout_failures", 0) for r in rs)
        nab = sum(1 for r in rs if r.get("aborted"))
        ga, gb, _n = _paired(rs, ref_rows)
        gt = gmean([r["total_time"] for r in rs
                    if not r.get("aborted") and r.get("total_time")])
        per = {r["circuit"]: (r["eprs"] if not r.get("aborted") else "A") for r in rs}
        per_s = " ".join(f"{c[:4]}={per.get(c, '--')}" for c in circs)
        print(f"{a:<9}{nl:>7}{p1:>6}{fl:>6}{nab:>5}{_g(rs):>11.1f}"
              f"{_fmt_ratio(_ratio(ga, gb)):>9}{gt:>9.1f}   {per_s}")
    print("\n  compute-matched contrasts:")
    for a, b, what in [("sabre1", "seq", "1-start: SabreLayout vs sequential"),
                       ("sabre3", "rand3", "3-start: SabreLayout vs random")]:
        ga, gb, n = _paired(grouped.get((b,), []), grouped.get((a,), []))
        if ga == ga and gb == gb:
            print(f"    {b:>6} / {a:<7} = {ga/gb:6.3f}  (n={n})  ({what})")


def report_monolithic(suite):
    d = _load(f"results_ablate_monolithic_{suite}.json")
    if not d:
        return
    rows = d["results"]
    grouped = _by(rows, "arm")
    arms = [a["name"] for a in d["meta"]["arms"]]
    ref_rows = grouped.get(("k1(prod)",), [])
    circs = d["meta"]["circuits"]

    print(f"\n{'='*112}")
    print(f"  MONOLITHIC COMPLETION — {suite}   (SabreLayout on dist-k completed "
          f"graph, k_reserved={d['meta']['k_reserved']}, all fwd->bwd->fwd)")
    print(f"{'='*112}")
    print(f"{'arm':<12}{'edges':>7}{'p1ab':>6}{'fail':>6}{'ABRT':>5}{'gmean EPR':>11}"
          f"{'×k1':>8}{'gmean SWAP':>12}   per-circuit EPR")
    print("-" * 112)
    for a in arms:
        rs = grouped.get((a,), [])
        if not rs:
            continue
        ne = next(x["n_edges"] for x in d["meta"]["arms"] if x["name"] == a)
        p1 = sum(r.get("pass1_aborts", 0) for r in rs)
        fl = sum(r.get("layout_failures", 0) for r in rs)
        nab = sum(1 for r in rs if r.get("aborted"))
        ga, gb, _n = _paired(rs, ref_rows)
        per = {r["circuit"]: (r["eprs"] if not r.get("aborted") else "A") for r in rs}
        per_s = " ".join(f"{c[:4]}={per.get(c, '--')}" for c in circs)
        print(f"{a:<12}{ne:>7}{p1:>6}{fl:>6}{nab:>5}{_g(rs):>11.1f}"
              f"{_fmt_ratio(_ratio(ga, gb)):>8}{_g(rs, 'ls'):>12.1f}   {per_s}")


def main():
    for suite in SUITES:
        report_occupancy(suite)
    for suite in SUITES:
        report_protocol(suite)
    for suite in SUITES:
        report_corners(suite)
    for suite in FREESLOT_SUITES:
        report_freeslots(suite)
    for suite in ["25q", "64q", "64q_c33"]:
        report_sabrelayout(suite)
    for suite in ["25q", "64q", "64q_c33"]:
        report_monolithic(suite)
    print()


if __name__ == "__main__":
    main()
