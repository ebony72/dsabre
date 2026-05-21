"""
verify_diff.py — compare results_verify/ vs results/ and emit verify_report.md.

Per file, walks the JSON and pulls out (circuit, metric) → numeric values from
both sides. Flags rows where any metric drifts more than thresholds. Classifies:
  match    — exact or within ±NOISE_PCT
  drift    — non-trivial numeric change (apply silently to appendix)
  material — > MATERIAL_PCT or sign change or new abort: PAUSE for review
"""
import json, os, sys, glob, math

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR  = os.path.join(HERE, "results")
VERIFY_DIR = os.path.join(HERE, "results_verify")
OUT = os.path.join(VERIFY_DIR, "verify_report.md")

NOISE_PCT    = 2.0   # ≤ 2% gmean swing → "match"
MATERIAL_PCT = 10.0  # > 10% drift on any cell → "material"

NUMERIC_KEYS = ("eprs", "ls", "teledata", "telegate", "cost")
# Skip paths under /routers/dS/ (topo extended-set, not in any appendix table).
SKIP_PATH_FRAGMENTS = ("/routers/dS/",)


def collect(obj, path=""):
    """Yield (key_path, value) pairs for numeric leaves under interesting keys."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            sub = f"{path}/{k}"
            if isinstance(v, (int, float)) and k in NUMERIC_KEYS:
                yield sub, v
            else:
                yield from collect(v, sub)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from collect(v, f"{path}[{i}]")


def index_records(data):
    """Return dict[(circuit, fill?, suite?) -> dict[metric_path -> value]]."""
    # Walk top-level lists named 'results' or 'rows'; fall back to dict of suites
    out = {}
    def add(rec, key_fields):
        cid = "/".join(str(rec.get(f, "")) for f in key_fields)
        cid = cid or "<root>"
        out.setdefault(cid, {})
        for p, v in collect(rec):
            out[cid][p] = v

    if isinstance(data, dict):
        if "results" in data and isinstance(data["results"], list):
            for rec in data["results"]:
                add(rec, ("suite", "circuit"))
            return out
        if "rows" in data and isinstance(data["rows"], list):
            for rec in data["rows"]:
                add(rec, ("suite", "circuit"))
            return out
        # nested-by-suite, e.g. dmaps_bench keys '25','36','64'
        for suite_k, suite_v in data.items():
            if isinstance(suite_v, list):
                for rec in suite_v:
                    rec2 = dict(rec); rec2.setdefault("suite", suite_k)
                    add(rec2, ("suite", "circuit"))
            elif isinstance(suite_v, dict):
                # e.g. mechanism_ablation / passes_sweep have nested lists
                for inner_k, inner_v in suite_v.items():
                    if isinstance(inner_v, list):
                        for rec in inner_v:
                            rec2 = dict(rec) if isinstance(rec, dict) else {"v": rec}
                            rec2.setdefault("suite", suite_k)
                            rec2.setdefault("label", inner_k)
                            add(rec2, ("suite", "label", "circuit"))
        if out: return out

    if isinstance(data, list):
        for rec in data:
            add(rec, ("suite", "circuit"))
        return out

    return out


def pct(new, old):
    if old in (None, 0): return None
    return 100.0 * (new - old) / old


def classify(diffs):
    severity = "match"
    for _, _, new, old in diffs:
        if old is None or new is None:
            if new != old: severity = max_sev(severity, "material")
            continue
        if old == 0 and new != 0: severity = max_sev(severity, "material"); continue
        if old == 0: continue
        p = abs(pct(new, old))
        if   p >  MATERIAL_PCT: severity = max_sev(severity, "material")
        elif p >  NOISE_PCT:    severity = max_sev(severity, "drift")
    return severity


def max_sev(a, b):
    order = {"match":0, "drift":1, "material":2}
    return a if order[a] >= order[b] else b


def diff_file(name):
    p_path = os.path.join(PAPER_DIR,  name)
    v_path = os.path.join(VERIFY_DIR, name)
    if not (os.path.exists(p_path) and os.path.exists(v_path)):
        return None
    paper  = json.load(open(p_path))
    verify = json.load(open(v_path))
    p_idx = index_records(paper)
    v_idx = index_records(verify)

    all_keys = sorted(set(p_idx) | set(v_idx))
    rows = []
    for k in all_keys:
        p_metrics = p_idx.get(k, {})
        v_metrics = v_idx.get(k, {})
        metrics = sorted(set(p_metrics) | set(v_metrics))
        diffs = []
        for m in metrics:
            if any(frag in m for frag in SKIP_PATH_FRAGMENTS): continue
            pv, vv = p_metrics.get(m), v_metrics.get(m)
            if pv == vv: continue
            diffs.append((k, m, vv, pv))
        if diffs:
            rows.append((k, diffs))
    severity = max((classify(r[1]) for r in rows), default="match")
    return rows, severity


def main():
    os.makedirs(VERIFY_DIR, exist_ok=True)
    files = sorted({os.path.basename(p) for p in glob.glob(os.path.join(PAPER_DIR, "*.json"))})
    lines = ["# Verify report\n",
             f"Comparing `{PAPER_DIR}` (paper) vs `{VERIFY_DIR}` (verify).\n",
             f"Thresholds: noise ≤ {NOISE_PCT}%, material > {MATERIAL_PCT}%.\n"]
    summary = []
    for fn in files:
        res = diff_file(fn)
        if res is None:
            lines.append(f"\n## {fn} — missing verify file\n")
            summary.append((fn, "missing"))
            continue
        rows, severity = res
        summary.append((fn, severity))
        lines.append(f"\n## {fn} — **{severity}** ({len(rows)} keys with diffs)\n")
        for k, diffs in rows:
            lines.append(f"\n### `{k}`")
            lines.append("| metric | paper | verify | Δ% |")
            lines.append("|---|---|---|---|")
            for _, m, vv, pv in diffs:
                p = pct(vv, pv) if isinstance(vv,(int,float)) and isinstance(pv,(int,float)) and pv else None
                ps = f"{p:+.1f}%" if p is not None else "—"
                lines.append(f"| `{m}` | {pv} | {vv} | {ps} |")
    # top summary
    lines.insert(3, "\n## Summary\n\n| file | status |\n|---|---|\n" +
                 "\n".join(f"| `{f}` | **{s}** |" for f, s in summary) + "\n")
    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {OUT}")
    for f, s in summary:
        print(f"  {s:>8}  {f}")


if __name__ == "__main__":
    main()
