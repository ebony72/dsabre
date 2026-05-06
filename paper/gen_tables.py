"""
gen_tables.py — emit LaTeX table rows from results JSONs.

Outputs (in tables/):
  main_25q.tex      — body rows for 25q main table (TS vs dSABRE; EPR + SWAP only)
  main_36q.tex      — 36q main table
  main_64q.tex      — 64q main table
  ablation_all.tex  — BFS vs topological extended-set ablation (gmean per suite)
  cost_sens.tex     — cost-model sensitivity (used in §VI only)

Cost is reported only in cost_sens.tex; main tables show EPR + SWAP only.
The "dSABRE" column in main tables is the production router with the BFS
extended set as default.  The "dSABRE-topo" rows in the ablation table use
the same router but with topological-order extended set, isolating the
BFS contribution.
"""

import json, os
from math import prod

R = "/Users/sanjiangli/Documents/pyzoo/dsabre/paper/results"
OUT = "/Users/sanjiangli/Documents/pyzoo/dsabre/paper/tables"
os.makedirs(OUT, exist_ok=True)

LS_COST   = 3
EPR_COSTS = [10, 20, 50, 100]


def load(suite):
    with open(os.path.join(R, f"results_{suite}.json")) as f:
        return json.load(f)["results"]


def gmean(lst):
    lst = [x for x in lst if x is not None and x > 0]
    return prod(lst) ** (1 / len(lst)) if lst else float("nan")


def pct(a, b):
    if a is None or b is None or b == 0: return "---"
    return f"{100*(a-b)/b:+.1f}"


def main_table(suite, order):
    """Per-circuit + gmean for the §V main table (no cost column).

    Columns: Circuit | CX | TS_EPR | TS_LS | dSABRE_EPR | dSABRE_LS | Δ%EPR

    The "dSABRE" column is current results['routers']['dSE'] (router with
    the BFS-layer extended set, which is the default in the production
    router).
    """
    recs = sorted(load(suite),
                  key=lambda r: order.index(r["circuit"]) if r["circuit"] in order else 999)
    lines = []
    ts_e, ts_l = [], []
    de_e, de_l = [], []

    for r in recs:
        ts  = r["ts"]
        dSE = r["routers"].get("dSE", {})
        te  = ts["eprs"]      if ts else None
        tl  = ts.get("ts_ls") if ts else None
        ee  = dSE.get("eprs") if not dSE.get("aborted") else None
        el  = dSE.get("ls")   if not dSE.get("aborted") else None

        cells = [
            r["circuit"].replace("_", "\\_"),
            f"{r['cx']}",
            f"{te}" if te is not None else "---",
            f"{tl}" if tl is not None else "---",
            f"{ee}" if ee is not None else "---",
            f"{el}" if el is not None else "---",
            f"${pct(ee, te)}$" if ee is not None and te is not None else "---",
        ]
        lines.append(" & ".join(cells) + " \\\\")
        if te is not None: ts_e.append(te); ts_l.append(tl)
        if ee is not None: de_e.append(ee); de_l.append(el)

    gm_cells = [
        "\\textbf{gmean}", "",
        f"{gmean(ts_e):.1f}", f"{gmean(ts_l):.0f}",
        f"{gmean(de_e):.1f}", f"{gmean(de_l):.0f}",
        f"$\\mathbf{{{pct(gmean(de_e), gmean(ts_e))}}}$",
    ]
    lines.append("\\midrule")
    lines.append(" & ".join(gm_cells) + " \\\\")
    return "\n".join(lines)


def ablation_table(suites_order):
    """For each suite, gmean(EPR) / gmean(SWAP) for:
      - dSABRE-topo  (router with topological extended set, = old 'dS')
      - dSABRE       (router with BFS extended set,        = old 'dSE')
    plus % EPR reduction of BFS over topological.
    """
    rows = []
    for s in suites_order:
        recs = load(s)
        ds_e, ds_l, de_e, de_l = [], [], [], []
        for r in recs:
            dS  = r["routers"].get("dS",  {})
            dSE = r["routers"].get("dSE", {})
            if not dS.get("aborted")  and dS.get("eprs")  is not None:
                ds_e.append(dS["eprs"]); ds_l.append(dS["ls"])
            if not dSE.get("aborted") and dSE.get("eprs") is not None:
                de_e.append(dSE["eprs"]); de_l.append(dSE["ls"])
        cells = [
            f"\\textbf{{{s}}}",
            f"{gmean(ds_e):.1f}", f"{gmean(ds_l):.0f}",
            f"{gmean(de_e):.1f}", f"{gmean(de_l):.0f}",
            f"${pct(gmean(de_e), gmean(ds_e))}$",
            f"${pct(gmean(de_l), gmean(ds_l))}$",
        ]
        rows.append(" & ".join(cells) + " \\\\")
    return "\n".join(rows)


def cost_sens(suites_order):
    """For each c in EPR_COSTS, gmean(cost) per suite. Reports
    dSABRE/TS% (no separate "topo vs BFS" — that's the ablation table)."""
    rows = []
    for c in EPR_COSTS:
        cells = [str(c)]
        for s in suites_order:
            recs = load(s)
            ts_costs, de_costs = [], []
            for r in recs:
                ts  = r["ts"]
                dSE = r["routers"].get("dSE", {})
                if ts is None or ts.get("ts_ls") is None: continue
                if dSE.get("aborted"): continue
                ts_costs.append(ts["eprs"] * c + ts["ts_ls"] * LS_COST)
                de_costs.append(dSE["eprs"] * c + dSE["ls"]   * LS_COST)
            ts_g, de_g = gmean(ts_costs), gmean(de_costs)
            cells.append(f"$\\mathbf{{{pct(de_g, ts_g)}}}$")
        rows.append(" & ".join(cells) + " \\\\")
    return "\n".join(rows)


order_25 = ["ae", "ghz", "graphstate", "qft", "qnn", "random"]
order_36 = ["bv", "dj", "qaoa", "qpeexact", "vqe_su2", "wstate"]
order_64 = ["ae", "ghz", "graphstate", "qft", "qnn", "random"]


with open(os.path.join(OUT, "main_25q.tex"),    "w") as f: f.write(main_table("25q", order_25) + "\n")
with open(os.path.join(OUT, "main_36q.tex"),    "w") as f: f.write(main_table("36q", order_36) + "\n")
with open(os.path.join(OUT, "main_64q.tex"),    "w") as f: f.write(main_table("64q", order_64) + "\n")
with open(os.path.join(OUT, "ablation_all.tex"),"w") as f: f.write(ablation_table(["25q", "36q", "64q"]) + "\n")
with open(os.path.join(OUT, "cost_sens.tex"),   "w") as f: f.write(cost_sens(["25q", "36q", "64q"]) + "\n")


print("Generated tables:\n")
for fn in sorted(os.listdir(OUT)):
    print(f"=== {fn} ===")
    with open(os.path.join(OUT, fn)) as f:
        print(f.read())
    print()
