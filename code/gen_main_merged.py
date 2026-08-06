r"""
gen_main_merged.py — emit the merged three-panel main-results table.

The TCAD revision merges the separate 25q / 36q / 64q main tables into one
float to stay inside the page budget.  TeleSABRE and dSABRE columns are read
from results_{25,36,64}q.json.

The main table is a TeleSABRE-vs-dSABRE comparison only: those two share a
cost model, a physical architecture, and intra-core SWAP accounting, so the
columns are directly comparable.  pytket-dqc is compared separately in the
paper (Section IV-C, tab:fair), because its e-bit counts charge no intra-core
routing and assume communication capacity and entanglement lifetime the device
does not provide.  Set WITH_TKET=True to restore the two pytket-dqc columns;
the data path for them (PUBLISHED / _load_physical) is kept live either way.

Writes tables/main_merged.tex (the table body, to be \input or pasted).
"""

import json, os, glob
from math import prod

_HERE = os.path.dirname(os.path.abspath(__file__))
R   = os.path.join(_HERE, "results")
OUT = os.path.join(_HERE, "tables")
os.makedirs(OUT, exist_ok=True)

# pytket-dqc e-bits under the PHYSICAL network model (data capacity
# M-deg(c), communication capacity deg(c)) -- see bench_pytket_fair.py and
# Section IV-C of the paper.  Read from results_pytket_fair*.json so the table
# cannot drift from the measurement.  PUBLISHED holds the older permissive-model
# numbers, used only where the fair sweep has no entry (the large-circuit rows,
# whose architectures the fair sweep does not cover) and marked in the caption.
PUBLISHED = {
    "25q": {"ae": 48, "ghz": 3, "graphstate": 4, "qft": 61, "qnn": 115,
            "random": 479},
    "36q": {"bv": 3, "dj": 3, "qaoa": 148, "qpeexact": 49, "vqe_su2": 9,
            "wstate": 6},
    "64q": {"ae": 141, "ghz": 5, "graphstate": 9, "qft": 181,
            "qnn": ("735", "d"), "random": ("968", "d"), "qpeexact": 96,
            "qaoa": 472, "multiplier": ("2957", "d")},
    "100q": {"qft": 114},
    "200q": {"qft": ("447", "d")},
    "360q": {"qft": ("195", "d")},
}


def _load_physical():
    """Physical-model e-bits keyed by (suite, circuit), from the fair sweep."""
    out = {}
    for fn in sorted(glob.glob(os.path.join(R, "results_pytket_fair*.json"))):
        try:
            recs = json.load(open(fn))["results"]
        except Exception:
            continue
        for r in recs:
            c = r.get("C_physical", {}).get("ebits")
            if c is not None:
                out[(r["suite"], r["circuit"])] = (c, r.get("C_physical", {}).get("method"))
    return out


PHYSICAL = None   # populated in main()

TKET = PUBLISHED  # retained name for the fallback path

# Emit the two pytket-dqc columns (e-bits, vs. tket)?  Off: the main table is
# the like-for-like TeleSABRE comparison; pytket-dqc lives in tab:fair.
WITH_TKET = False
NCOL = 9 if WITH_TKET else 7

ORDER = {
    "25q": ["ae", "ghz", "graphstate", "qft", "qnn", "random"],
    "36q": ["bv", "dj", "qaoa", "qpeexact", "vqe_su2", "wstate"],
    "64q": ["ae", "ghz", "graphstate", "qft", "qnn", "random",
            "qpeexact", "qaoa", "multiplier"],
}

# The large-QFT rows are a different circuit family from the 25/64q panels:
# MQT-Bench's current QFT is a banded approximation (interaction range 19),
# whereas the v1.1.0 circuits used at 25 and 64 qubits are full-span.  See
# Section IV-G of the paper; they are deliberately not presented as one series.
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
    """Physical-model e-bits if the fair sweep measured them, else published."""
    if PHYSICAL and (suite, name) in PHYSICAL:
        v, method = PHYSICAL[(suite, name)]
        mark = "$^{\\ddagger}$" if method and method != "CoverEmbeddingSteinerDetached" else ""
        return f"{v}{mark}", int(v)
    v = PUBLISHED.get(suite, {}).get(name)
    if v is None:
        return "---", None
    if isinstance(v, tuple):
        return v[0] + "$^{\\ddagger}$$^{\\S}$", int(v[0])
    return f"{v}$^{{\\S}}$", int(v)


def panel(suite):
    with open(os.path.join(R, f"results_{suite}.json")) as f:
        recs = json.load(f)["results"]
    order = ORDER[suite]
    recs = [r for r in recs if r["circuit"] in order]
    recs.sort(key=lambda r: order.index(r["circuit"]))

    lines = [f"\\multicolumn{{{NCOL}}}{{@{{}}l}}{{{PANEL[suite]}}} \\\\",
             "\\addlinespace[1pt]"]
    m_ts, m_de = [], []          # matched sets, for the TS gmean
    m_tsl, m_del = [], []
    m_tk = []                    # tket values over the same matched circuits
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

        cells = [
            label, str(r["cx"]),
            str(te) if te is not None else "---",
            str(tl) if tl is not None else "---",
            str(ee) if ee is not None else "---",
            str(el) if el is not None else "---",
        ]
        cells += [tks, fmt_pct(ee, te), fmt_pct(ee, tkv)] if WITH_TKET \
            else [fmt_pct(ee, te)]
        lines.append(" & ".join(cells) + " \\\\")

        if te is not None and ee is not None:
            m_ts.append(te); m_de.append(ee)
            if tl is not None: m_tsl.append(tl)
            if el is not None: m_del.append(el)
            if tkv is not None: m_tk.append(tkv)
        if ee is not None:
            all_de.append(ee)
            if el is not None: all_del.append(el)
            if tkv is not None: all_tk.append(tkv)

    g_ts, g_de = gmean(m_ts), gmean(m_de)
    lines.append(f"\\cmidrule(lr){{1-{NCOL}}}")
    cells = [
        f"\\textbf{{gmean}} ({len(m_ts)})", "",
        f"{g_ts:.1f}", f"{gmean(m_tsl):.0f}" if m_tsl else "---",
        f"{g_de:.1f}", f"{gmean(m_del):.0f}" if m_del else "---",
    ]
    cells += [f"{gmean(m_tk):.1f}" if m_tk else "---",
              fmt_pct(g_de, g_ts, bold=True),
              fmt_pct(g_de, gmean(m_tk), bold=True) if m_tk else "---"] \
        if WITH_TKET else [fmt_pct(g_de, g_ts, bold=True)]
    lines.append(" & ".join(cells) + " \\\\")

    # A second gmean row over every circuit dSABRE routed, whenever TeleSABRE
    # did not complete the whole suite.  It is the dSABRE figure tab:fair
    # quotes for the pytket-dqc comparison.
    if len(m_ts) != len(all_de):
        cells = [f"\\textbf{{gmean}} ({len(all_de)})", "", "---", "---",
                 f"{gmean(all_de):.1f}", f"{gmean(all_del):.0f}"]
        cells += [f"{gmean(all_tk):.1f}" if all_tk else "---", "---",
                  fmt_pct(gmean(all_de), gmean(all_tk), bold=True) if all_tk
                  else "---"] if WITH_TKET else ["---"]
        lines.append(" & ".join(cells) + " \\\\")
    return "\n".join(lines)


def large_panel():
    """The QFT scalability rows.

    These come from results_scaling_b.json, the decomposed series: core size
    held at 5x5 while the core count grows 2x3 -> 3x4 -> 4x5, so logical qubits
    per core stay at 16.7/16.7/18.0 and the only variable is the core graph
    (diameter 3 -> 5 -> 7).  The earlier series mixed both axes -- it doubled
    the core count from 100q to 200q and then grew the cores at 360q, leaving
    60 qubits per core against 17, which for a banded QFT of range 19 changes
    the inherently-inter-core gate fraction from ~57% to 16.5% and made the
    three rows incomparable.
    """
    rows = [rf"\multicolumn{{{NCOL}}}{{@{{}}l}}{{\emph{{QFT scalability, "
            r"$5{\times}5$ cores throughout, core count $6\to12\to20$}} \\",
            r"\addlinespace[1pt]"]
    de, tk = [], []
    src = os.path.join(R, "results_scaling_b.json")
    recs = {x["label"]: x for x in json.load(open(src))["results"]} \
        if os.path.exists(src) else {}
    for suite in ("100q", "200q", "360q"):
        rec = recs.get(suite)
        if rec is None:
            continue
        ts = rec.get("ts") or {}
        dse = rec["routers"]["dSE"]
        te, tl = ts.get("eprs"), ts.get("ls", ts.get("ts_ls"))
        ee, el = dse.get("eprs"), dse.get("ls")
        tks, tkv = tket_cell(suite, "qft")
        label = suite + ("$^{\\ast}$" if te is None else "")
        cells = [
            label, str(rec["cx"]),
            str(te) if te is not None else "---",
            str(tl) if tl is not None else "---",
            str(ee), str(el),
        ]
        cells += [tks, fmt_pct(ee, te), fmt_pct(ee, tkv)] if WITH_TKET \
            else [fmt_pct(ee, te)]
        rows.append(" & ".join(cells) + " \\\\")
        if ee: de.append(ee)
        if tkv: tk.append(tkv)
    return "\n".join(rows)


def main():
    global PHYSICAL
    PHYSICAL = _load_physical()
    print(f'physical-model cells available: {len(PHYSICAL)}')
    parts = []
    for i, suite in enumerate(["25q", "36q", "64q"]):
        if i:
            parts.append("\\midrule")
        parts.append(panel(suite))
    lp = large_panel()
    if lp:
        parts.append("\\midrule")
        parts.append(lp)
    body = "\n".join(parts)
    path = os.path.join(OUT, "main_merged.tex")
    with open(path, "w") as f:
        f.write(body + "\n")
    print(body)
    print(f"\n→ {path}", flush=True)


if __name__ == "__main__":
    main()
