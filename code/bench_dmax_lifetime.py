r"""bench_dmax_lifetime.py — the entanglement-lifetime sweep of Appendix J.

Re-scores pytket-dqc's own distributions under a bounded EPR lifetime
(`dmax_counter.py`), on the 64q suite.

The table exists to be read against Table~\ref{tab:fair}, so the distribution
scored here must be *the one that table reports* -- not a fresh search that
happens to use the same settings.  pytket-dqc's search is sensitive to how many
seeds finish inside the budget and to KaHyPar's own nondeterminism, so a re-run
finds a different (often better) answer: that is exactly how the published
D_max=inf column came to sit 14% below tab:fair.  We therefore load the
distribution `bench_pytket_fair.py` persisted, and check twice before reporting:

  1. `dist.cost()` equals the e-bit count recorded alongside it, and
  2. the counter at D_max=inf equals `dist.cost()`, hyperedge by hyperedge.

With both green the D_max=inf column *is* tab:fair's pytket-dqc column, by
construction.

Output: results/results_dmax_64q.json

Usage:
  python3 code/bench_dmax_lifetime.py
  python3 code/bench_dmax_lifetime.py --fair results_pytket_fair_v3_64q.json
"""

import sys, os, json, gzip, time, argparse
from math import inf, prod

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from dmax_counter import costs_under_dmax, verify_reproduces_tool

_RESULTS = os.path.join(_HERE, "results")

DMAX = [inf, 10, 5, 3, 1]
ORDERS = {
    "25q": ["ae", "ghz", "graphstate", "qft", "qnn", "random"],
    "36q": ["bv", "dj", "qaoa", "qpeexact", "vqe_su2", "wstate"],
    "64q": ["ae", "ghz", "graphstate", "qft", "qnn", "random",
            "qpeexact", "qaoa", "multiplier"],
}
MODEL = "A_published"        # the model tab:fair reports: pytket-dqc's
                             # default computation on its permissive network


def gmean(xs):
    return prod(xs) ** (1 / len(xs))


def load_distribution(rel_path):
    from pytket_dqc.circuits import Distribution
    path = rel_path if os.path.isabs(rel_path) else os.path.join(_HERE, rel_path)
    with gzip.open(path, "rt") as f:
        return Distribution.from_dict(json.load(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="64q", choices=sorted(ORDERS))
    ap.add_argument("--fair", default=None,
                    help="the sweep whose distributions are scored")
    ap.add_argument("--circuits", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ORDER = ORDERS[args.suite]
    args.fair = args.fair or f"results_pytket_fair_v3_{args.suite}.json"
    args.out = args.out or f"results_dmax_{args.suite}.json"
    MAIN = os.path.join(_RESULTS, f"results_{args.suite}.json")
    want = [c.strip() for c in (args.circuits or ",".join(ORDER)).split(",") if c.strip()]

    fair_path = os.path.join(_RESULTS, args.fair)
    fair = {r["circuit"]: r for r in json.load(open(fair_path))["results"]}
    fair_meta = json.load(open(fair_path))["meta"]
    ds = {r["circuit"]: r["routers"]["dSE"]["eprs"]
          for r in json.load(open(MAIN))["results"]}

    records, problems = [], []
    for cname in [c for c in ORDER if c in want]:
        t0 = time.perf_counter()
        entry = fair.get(cname)
        if entry is None or MODEL not in entry:
            problems.append(f"{cname}: no {MODEL} entry in {args.fair}")
            continue
        model = entry[MODEL]
        if not model.get("distribution"):
            problems.append(f"{cname}: {args.fair} saved no distribution -- "
                            f"re-run bench_pytket_fair.py to persist it")
            continue

        dist = load_distribution(model["distribution"])
        cost = dist.cost()
        ok, bad = verify_reproduces_tool(dist)
        if cost != model["ebits"]:
            problems.append(f"{cname}: reloaded {cost} e-bits, {args.fair} "
                            f"records {model['ebits']}")
        if not ok:
            problems.append(f"{cname}: counter != tool on {len(bad)} hyperedges")

        sweep = costs_under_dmax(dist, DMAX)
        row = dict(circuit=cname, method=model["method"],
                   ebits_fair=model["ebits"], ebits_reloaded=cost,
                   matches_fair=(cost == model["ebits"]),
                   counter_reproduces_tool=ok, n_bad_hyperedges=len(bad),
                   seed_loop_truncated=model.get("seed_loop_truncated"),
                   dsabre_eprs=ds[cname],
                   dmax={("inf" if D == inf else str(D)): sweep[D] for D in DMAX})
        records.append(row)
        print(f"  {cname:<11} {model['method'][:12]:<12} "
              f"e-bits={cost} ({'matches' if cost == model['ebits'] else 'MISMATCH'})"
              f"  counter@inf={'ok' if ok else 'BAD'}  {row['dmax']}"
              f"  ({time.perf_counter()-t0:.0f}s)", flush=True)

    out = os.path.join(_RESULTS, args.out)
    with open(out, "w") as f:
        json.dump(dict(meta=dict(
            date=time.strftime("%Y-%m-%d %H:%M:%S"), suite=args.suite,
            network_model=MODEL,
            scored_from=args.fair,
            source_sweep_meta=fair_meta,
            counter="dmax_counter.costs_under_dmax",
            note="distributions are loaded from disk, not re-searched, so the "
                 "D_max=inf column is tab:fair's pytket-dqc column by construction",
        ), results=records, problems=problems), f, indent=2)
    print(f"\nSaved → {out}", flush=True)

    if problems:
        print("\nPROBLEMS:", flush=True)
        for p in problems:
            print("  " + p, flush=True)
        return
    g_ds = gmean([r["dsabre_eprs"] for r in records])
    print(f"\ndSABRE gmean {g_ds:.1f}", flush=True)
    for D in DMAX:
        k = "inf" if D == inf else str(D)
        g = gmean([r["dmax"][k] for r in records])
        print(f"  D_max={k:<4} pytket {g:9.1f}   dSABRE {100*(g_ds-g)/g:+7.1f}%",
              flush=True)


if __name__ == "__main__":
    main()
