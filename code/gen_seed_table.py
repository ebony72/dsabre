"""gen_seed_table.py -- LaTeX rows for the per-seed sensitivity appendix.

Emits the body of the table backing Section IV-A's median-vs-best statement:
for every circuit of every suite, dSE's EPR count under each of the three
SabreLayout seeds, the best (what the main tables report), the median, and
TeleSABRE's count for reference.

Reads code/results/results_derived_deadlock{,_25q}.json (per-seed dSE counts,
written by probe_derived_deadlock.py) and the published results_*.json for
the TeleSABRE column.  Regenerate and paste into paper/appendices.tex.
"""

import json, os
from math import prod

_HERE = os.path.dirname(os.path.abspath(__file__))
_RES = os.path.join(_HERE, "results")

SUITES = [("25q", "results_25q", "25 logical qubits, B-grid"),
          ("36q", "results_36q", "36 logical qubits, B-grid"),
          ("64q", "results_64q", "64 logical qubits, H-grid"),
          ("100q", "results_100q", "100 logical qubits, H-grid $2{\\times}3$ of $5{\\times}5$"),
          ("200q", "results_200q", "200 logical qubits, H-grid $3{\\times}4$ of $5{\\times}5$"),
          ("360q", "results_360q", "360 logical qubits, H-grid $4{\\times}5$ of $5{\\times}5$")]

NAME = {"ae": "AE", "ghz": "GHZ", "graphstate": "Graphstate", "qft": "QFT",
        "qnn": "QNN", "random": "Random", "qpeexact": "QPEexact",
        "qaoa": "QAOA", "multiplier": "Multiplier", "bv": "BV", "dj": "DJ",
        "vqe_su2": "VQE-SU2", "wstate": "W-state"}


def gmean(xs):
    xs = [x for x in xs if x]
    return prod(xs) ** (1 / len(xs)) if xs else float("nan")


def main():
    per_seed = []
    for fn in ("results_derived_deadlock_25q.json", "results_derived_deadlock.json"):
        p = os.path.join(_RES, fn)
        if os.path.exists(p):
            per_seed += json.load(open(p))["results"]

    for suite, pub_file, caption in SUITES:
        pub = {r["circuit"]: (r.get("ts") or {}).get("eprs")
               for r in json.load(open(os.path.join(_RES, pub_file + ".json")))["results"]}
        rows = [r for r in per_seed if r["suite"] == suite]
        if not rows:
            continue
        print(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{{caption}}}}} \\\\")
        print("\\addlinespace[1pt]")
        ts_l, best_l, med_l = [], [], []
        for r in rows:
            a = r["arm_published"]
            seeds = [s["eprs"] if s else None for s in a["per_seed"]]
            ts = pub.get(r["circuit"])
            cells = " & ".join(str(s) if s is not None else "---" for s in seeds)
            print(f"{NAME.get(r['circuit'], r['circuit'])} & {cells} & "
                  f"{a['best']} & {a['median']:g} & "
                  f"{ts if ts else '---'} \\\\")
            if ts:
                ts_l.append(ts); best_l.append(a["best"]); med_l.append(a["median"])
        if ts_l:
            print("\\cmidrule(lr){1-7}")
            print(f"\\textbf{{gmean}} ({len(ts_l)}) & & & & "
                  f"\\textbf{{{gmean(best_l):.1f}}} & {gmean(med_l):.1f} & "
                  f"{gmean(ts_l):.1f} \\\\")
            print(f"% best {100*(gmean(best_l)/gmean(ts_l)-1):+.1f}%  "
                  f"median {100*(gmean(med_l)/gmean(ts_l)-1):+.1f}%")
        print("\\midrule")


if __name__ == "__main__":
    main()
