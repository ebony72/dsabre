"""
probe_telegate_amortization.py — what one teledata EPR pair actually buys.

A single-gate telegate serves exactly one remote gate per EPR pair, always.
A teledata hop also costs one pair, but the qubit it moves then executes
however many gates it has in the destination core before it moves again.  So
the whole comparison reduces to one measured quantity: the number of 2Q gates
a teledata hop amortises over on the suite in question.  Above 1 it wins,
below 1 it loses, and no amount of telegate scoring changes that.

This probe measures it directly from the published router's own operation
trace (telegate disabled, i.e. \\dSABRE{} exactly as in Table IV).

Definitions
-----------
A *residence* is a maximal interval during which one logical qubit sits in
one core, delimited by its teleports.  Residences opened by a teleport cost
one EPR pair; the initial residence (from the layout) is free and excluded.
`gates` is the number of 2Q gates involving that qubit executed during the
residence.

  gates = 0   a TRANSIT hop -- the qubit is on its way somewhere.  A one-hop
              telegate cannot replace these at all: nothing is executed here.
  gates = 1   the hop a telegate could have replaced one-for-one, at equal
              EPR cost.  Replacing it is a win only if the qubit's move was
              itself harmful later, never a direct saving.
  gates >= 2  the hop a telegate strictly LOSES to: the same one pair served
              `gates` gates, where telegate would have paid `gates` pairs.

Usage:
    python3 probe_telegate_amortization.py
    python3 probe_telegate_amortization.py --circuits ae,qft --suite 64q
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from ablate_common import (RESULTS_DIR, SUITES, SUITE_CIRCUITS, default_layouts,
                           load_circuit, save_json)
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt


def residences(trace, arch, initial_layout):
    """Replay `trace`, returning one record per teleport-opened residence.

    Each record is (qubit, core, gates_served, is_last_of_run).
    """
    where = {q: arch.core_of(p) for q, p in initial_layout.items()}
    # open[q] = [core, gates_served, opened_by_teleport]
    open_res = {q: [where[q], 0, False] for q in where}
    closed = []
    for ev in trace:
        if ev[0] == "TELE":
            _, q, _a, _b, _cs, cd = ev
            rec = open_res.get(q)
            if rec is not None and rec[2]:
                closed.append((q, rec[0], rec[1]))
            open_res[q] = [cd, 0, True]
        elif ev[0] == "GATE":
            _, q1, q2, _c = ev
            for q in (q1, q2):
                if q in open_res:
                    open_res[q][1] += 1
    for q, rec in open_res.items():
        if rec[2]:
            closed.append((q, rec[0], rec[1]))
    return closed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="64q")
    ap.add_argument("--circuits", default=None)
    ap.add_argument("--layouts", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    s = SUITES[args.suite]
    arch = s["arch"]
    circuits = (args.circuits.split(",") if args.circuits
                else SUITE_CIRCUITS[args.suite])
    out = args.out or os.path.join(RESULTS_DIR,
                                   f"telegate_amortization_{args.suite}.json")

    hw = HardwareConfig(deadlock_limit=None, max_backup_attempts=None,
                        max_iterations=None, trace_routing=True)

    print(f"{'circuit':11s} {'EPR':>5s} {'hops':>5s} {'gates/hop':>9s} "
          f"{'transit%':>8s} {'=1 gate%':>8s} {'>=2 gate%':>9s} {'max':>4s}",
          flush=True)
    payload = {"meta": dict(suite=args.suite, circuits=circuits,
                            router="dSE (dsabre_ext.dSABRE_BurstExt), "
                                   "telegate disabled",
                            definition="residence = interval a qubit spends in "
                                       "one core between its teleports; gates = "
                                       "2Q gates it executes there"),
               "per_circuit": {}}
    tot = Counter()
    for cname in circuits:
        qc, dag, rev_dag, ncx = load_circuit(args.suite, cname)
        layouts = default_layouts(qc, dag, arch, s["n_qubits"], seed=0,
                                  n_seeds=args.layouts)
        # fwd -> bwd -> fwd, run explicitly rather than through
        # `run_protocol_ex`, so each forward pass's trace stays paired with
        # the layout that pass actually started from (pass 3 starts from the
        # backward pass's output, not from L).
        best, best_entry = None, None
        r = dSABRE_BurstExt(arch, hw)
        for L in layouts:
            layout = L
            for step, d in (("f", dag), ("b", rev_dag), ("f", dag)):
                entry = layout
                m, out_layout = r.route(d, entry)
                if m["aborted"]:
                    continue
                layout = out_layout
                if step == "f" and (best is None or m["eprs"] < best["eprs"]):
                    best, best_entry = m, entry
        if best is None:
            print(f"{cname:11s}  ABORT", flush=True)
            continue
        recs = residences(best["trace"], arch, best_entry)
        served = [g for (_q, _c, g) in recs]
        hist = Counter(served)
        n = len(served)
        if n == 0:
            print(f"{cname:11s} {best['eprs']:5d}     0", flush=True)
            continue
        transit = sum(v for k, v in hist.items() if k == 0)
        one = hist.get(1, 0)
        many = n - transit - one
        row = dict(eprs=best["eprs"], hops=n,
                   gates_per_hop=round(sum(served) / n, 3),
                   transit=transit, exactly_one=one, two_or_more=many,
                   max_gates=max(served),
                   histogram={str(k): v for k, v in sorted(hist.items())})
        payload["per_circuit"][cname] = row
        tot["hops"] += n; tot["gates"] += sum(served)
        tot["transit"] += transit; tot["one"] += one; tot["many"] += many
        print(f"{cname:11s} {best['eprs']:5d} {n:5d} {row['gates_per_hop']:9.2f} "
              f"{100*transit/n:7.1f}% {100*one/n:7.1f}% {100*many/n:8.1f}% "
              f"{max(served):4d}", flush=True)

    if tot["hops"]:
        n = tot["hops"]
        print(f"\n{'SUITE':11s} {'':5s} {n:5d} {tot['gates']/n:9.2f} "
              f"{100*tot['transit']/n:7.1f}% {100*tot['one']/n:7.1f}% "
              f"{100*tot['many']/n:8.1f}%", flush=True)
        payload["totals"] = dict(hops=n, gates=tot["gates"],
                                 gates_per_hop=round(tot["gates"] / n, 3),
                                 transit=tot["transit"], exactly_one=tot["one"],
                                 two_or_more=tot["many"])
    save_json(out, payload)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
