r"""
gen_main_merged.py — emit the merged three-panel main-results table.

The TCAD revision merges the separate 25q / 36q / 64q main tables into one
float to stay inside the page budget.  TeleSABRE and dSABRE columns are read
from results_{25,36,64}q.json; the pytket-dqc e-bit column is carried in the
TKET dict below, because pytket-dqc does not depend on any dSABRE setting and
is not re-run by benchmark.py.

Writes tables/main_merged.tex (the table body, to be \input or pasted).
"""

import json, os
from math import prod

_HERE = os.path.dirname(os.path.abspath(__file__))
R   = os.path.join(_HERE, "results")
OUT = os.path.join(_HERE, "tables")
os.makedirs(OUT, exist_ok=True)

# pytket-dqc e-bits, strongest completed distributor per cell.
# "d" marks cells where CoverEmbeddingSteinerDetached did not scale or embed
# and PartitioningHeterogeneous is reported instead.
TKET = {
    "25q": {"ae": 48, "ghz": 3, "graphstate": 4, "qft": 61, "qnn": 115,
            "random": 479},
    "36q": {"bv": 3, "dj": 3, "qaoa": 148, "qpeexact": 49, "vqe_su2": 9,
            "wstate": 6},
    "64q": {"ae": 141, "ghz": 5, "graphstate": 9, "qft": 181,
            "qnn": ("735", "d"), "random": ("968", "d"), "qpeexact": 96,
            "qaoa": 472, "multiplier": ("2957", "d")},
}

ORDER = {
    "25q": ["ae", "ghz", "graphstate", "qft", "qnn", "random"],
    "36q": ["bv", "dj", "qaoa", "qpeexact", "vqe_su2", "wstate"],
    "64q": ["ae", "ghz", "graphstate", "qft", "qnn", "random",
            "qpeexact", "qaoa", "multiplier"],
}

PANEL = {
    "25q": r"\emph{25 logical qubits, B-grid $2{\times}2$ of $4{\times}4$ cores (64 physical)}",
    "36q": r"\emph{36 logical qubits, same B-grid}",
    "64q": r"\emph{64 logical qubits, H-grid $2{\times}3$ of $4{\times}4$ cores (96 physical)}",
}


def gmean(v):
    v = [x for x in v if x is not None and x > 0]
    return prod(v) ** (1 / len(v)) if v else float("nan")


def pct(a, b):
    if a is None or b is None or b == 0:
        return None
    return 100 * (a - b) / b


def fmt_pct(a, b, bold=False):
    p = pct(a, b)
    if p is None:
        return "---"
    s = f"{p:+.1f}"
    return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"


def tket_cell(suite, name):
    v = TKET.get(suite, {}).get(name)
    if v is None:
        return "---", None
    if isinstance(v, tuple):
        return v[0] + "$^{\\ddagger}$", int(v[0])
    return str(v), int(v)


def panel(suite):
    with open(os.path.join(R, f"results_{suite}.json")) as f:
        recs = json.load(f)["results"]
    order = ORDER[suite]
    recs = [r for r in recs if r["circuit"] in order]
    recs.sort(key=lambda r: order.index(r["circuit"]))

    lines = [f"\\multicolumn{{9}}{{@{{}}l}}{{{PANEL[suite]}}} \\\\",
             "\\addlinespace[1pt]"]
    m_ts, m_de = [], []          # matched sets, for the TS gmean
    m_tsl, m_del = [], []
    all_de, all_del, all_tk = [], [], []

    for r in recs:
        name = r["circuit"]
        ts   = r["ts"]
        dse  = r["routers"].get("dSE", {})
        te   = ts["eprs"] if ts else None
        tl   = ts.get("ls") if ts else None
        ee   = dse.get("eprs") if not dse.get("aborted") else None
        el   = dse.get("ls") if not dse.get("aborted") else None
        tks, tkv = tket_cell(suite, name)

        label = name.replace("_", "\\_")
        if te is None:
            label += "$^{\\ast}$"

        lines.append(" & ".join([
            label, str(r["cx"]),
            str(te) if te is not None else "---",
            str(tl) if tl is not None else "---",
            str(ee) if ee is not None else "---",
            str(el) if el is not None else "---",
            tks,
            fmt_pct(ee, te), fmt_pct(ee, tkv),
        ]) + " \\\\")

        if te is not None and ee is not None:
            m_ts.append(te); m_de.append(ee)
            if tl is not None: m_tsl.append(tl)
            if el is not None: m_del.append(el)
        if ee is not None:
            all_de.append(ee)
            if el is not None: all_del.append(el)
            if tkv is not None: all_tk.append(tkv)

    g_ts, g_de = gmean(m_ts), gmean(m_de)
    lines.append("\\cmidrule(lr){1-9}")
    lines.append(" & ".join([
        f"\\textbf{{gmean}} ({len(m_ts)})", "",
        f"{g_ts:.1f}", f"{gmean(m_tsl):.0f}" if m_tsl else "---",
        f"{g_de:.1f}", f"{gmean(m_del):.0f}" if m_del else "---",
        "", fmt_pct(g_de, g_ts, bold=True), "",
    ]) + " \\\\")

    # A second gmean row over every circuit dSABRE routed, for the tket
    # comparison, whenever TeleSABRE did not complete the whole suite.
    if len(m_ts) != len(all_de) and all_tk:
        lines.append(" & ".join([
            f"\\textbf{{gmean}} ({len(all_de)})", "", "---", "---",
            f"{gmean(all_de):.1f}", f"{gmean(all_del):.0f}", f"{gmean(all_tk):.1f}",
            "---", fmt_pct(gmean(all_de), gmean(all_tk), bold=True),
        ]) + " \\\\")
    return "\n".join(lines)


def main():
    parts = []
    for i, suite in enumerate(["25q", "36q", "64q"]):
        if i:
            parts.append("\\midrule")
        parts.append(panel(suite))
    body = "\n".join(parts)
    path = os.path.join(OUT, "main_merged.tex")
    with open(path, "w") as f:
        f.write(body + "\n")
    print(body)
    print(f"\n→ {path}", flush=True)


if __name__ == "__main__":
    main()
