r"""
probe_mechanism_counters.py -- whole-search mechanism counters for the two
capacity designs, 64-qubit suite.

`tab:safemode` in the appendices reports per-mechanism counters for the
capacity-safe default against the score-only design it replaced, but two of
its rows conflate distinct quantities.  "Checkpoint--rollback activations"
carries `_backup_plan` activations -- a recovery invocation -- rather than
`metrics["rollbacks"]`, the transaction undo; and its dashes claim a mode
"does not have" a mechanism it does have and simply never uses.  Relay hops in
particular are reachable in score-only mode, through `_backup_plan`'s greedy
branch into `_route_gate_transaction` -> `_relay_room_to`, so a dash there is
unverified rather than true.

This probe re-measures both arms over every route of the search (3
SabreLayout seeds x fwd/bwd/fwd = 9 `route()` calls per circuit), on
`ablate_capacity.py`'s protocol so the EPR figures reproduce `tab:main`, and
reports each counter separately.  Counters are collected by wrapping
`route()`, since `run_sabre_passes` returns only the winning pass and its
metrics would miss the other two.

Output: code/results/results_mechanism_counters_64q.json

Usage:  python3 code/probe_mechanism_counters.py [suite ...]   (default 64q)
"""
import os, sys, glob, json, time
from dataclasses import replace

sys.setrecursionlimit(100000)
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from qiskit.converters import circuit_to_dag

import benchmark as B
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes

_OUT = os.path.join(_HERE, "results", "results_mechanism_counters_64q.json")

ARMS = {
    "score_only": dict(safe_mode=False, cap_penalty=15.0),
    "default":    dict(safe_mode=True, tier1_floor=2, cap_penalty=15.0),
}

COUNTERS = ("backup_activations", "rollbacks", "snapshots", "wdag_rebuilds",
            "relay_hops", "force_make_room", "safe_routes",
            "safe_route_failed", "iterations")


def run_arm(suite_name, override):
    s = B.SUITES[suite_name]
    arch = s["arch"]
    hw = replace(s["hw"], **override)
    router = dSABRE_BurstExt(arch, hw)

    # Wrap route() so every call in the search contributes, not just the pass
    # run_sabre_passes hands back.
    tally = {k: 0 for k in COUNTERS}
    n_routes = [0]
    raw_route = router.route

    def counting_route(dag, layout):
        m, final = raw_route(dag, layout)
        n_routes[0] += 1
        for k in COUNTERS:
            tally[k] += m.get(k, 0) or 0
        return m, final

    router.route = counting_route

    qasm_files = sorted(glob.glob(os.path.join(s["circuit_dir"], "*.qasm")))
    canon = B.CANONICAL_CIRCUITS.get(suite_name)
    if canon:
        qasm_files = [f for f in qasm_files
                      if os.path.basename(f).replace(s["suffix"], "") in canon]

    rows = []
    for qf in qasm_files:
        cname = os.path.basename(qf).replace(s["suffix"], "")
        qc, dag = B.load_qasm(qf)
        rev_dag = circuit_to_dag(qc.reverse_ops())
        before = dict(tally)
        best = None
        t0 = time.perf_counter()
        for lay in sabre_locked_boundary_layout(qc, dag, arch, seed=0):
            try:
                m = run_sabre_passes(router, dag, rev_dag, lay)
            except Exception as e:
                print(f"    {cname}: RAISED {type(e).__name__}: {e}", flush=True)
                m = None
            if m and not m.get("aborted"):
                best = m["eprs"] if best is None else min(best, m["eprs"])
        delta = {k: tally[k] - before[k] for k in COUNTERS}
        rows.append(dict(circuit=cname, eprs=best,
                         secs=round(time.perf_counter() - t0, 2), **delta))
        print(f"    {cname:11} eprs={best} " +
              " ".join(f"{k}={delta[k]}" for k in COUNTERS), flush=True)
    return rows, tally, n_routes[0]


if __name__ == "__main__":
    suites = sys.argv[1:] or ["64q"]
    out = json.load(open(_OUT)) if os.path.exists(_OUT) else {}
    for suite in suites:
        out.setdefault(suite, {})
        for label, override in ARMS.items():
            print(f"== {suite} / {label} ({override}) ==", flush=True)
            rows, totals, n_routes = run_arm(suite, override)
            out[suite][label] = dict(config=override, rows=rows,
                                     totals=totals, n_routes=n_routes)
            print(f"  TOTALS ({n_routes} routes): {totals}\n", flush=True)
            json.dump(out, open(_OUT, "w"), indent=2)
    print(f"Saved -> {_OUT}", flush=True)
