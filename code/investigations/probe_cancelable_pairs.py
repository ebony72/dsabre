"""
probe_cancelable_pairs.py — trace-mining bound on the gain available from
cancelling "redundant" teleports (2026-08-04).

Question (follow-up to TELEPORT_COMMITMENT.md): if a post-hoc review pass
deleted every teleport that was not needed to bring a qubit from where its
previous gate executed to where its next gate executes, how many EPR pairs
could it save at most?

Method
------
Route each suite circuit with the production protocol (dSE, per-core adaptive
corner layout, 3 SabreLayout seeds, fwd->bwd->fwd; the reported pass is the
best forward pass by EPR).  With `trace_routing=True` the router now emits a
chronological stream of ("TELE", virt, p_src, p_comm_dst, src_core, dst_core),
("SWAP", ...) and ("GATE", q1, q2, core) events, truncated on checkpoint
rollback, so the stream is exactly the compiled output.

For each logical qubit, split its teleport history into windows delimited by
the 2q gates executed on it (1q gates do not pin location).  In the window
between gate i (executed in core X) and gate i+1 (executed in core Y), any
router whatsoever must spend at least core_dist(X, Y) teleport hops on that
qubit; the router actually spent `hops`.  The window's *excess* is
hops - core_dist(X, Y) >= 0 (every teleport is a single core-graph hop).

    excess_roundtrip : windows with core_dist = 0 and hops > 0 — the qubit
                       left its core and came back with no gate in between
                       (the pure "cancelable pair" case, A->B ... B->A).
    excess_detour    : windows with core_dist > 0 and hops > core_dist.
    excess_tail      : teleports after the qubit's last 2q gate (and, via the
                       same formula, before its first — anchor = initial core).

The sum is an OPTIMISTIC upper bound on the EPR a cancellation pass could
recover: it assumes every excess hop can be deleted, ignoring that those hops
may have been load-bearing as capacity relief (eviction room, force_make_room)
for *other* qubits' moves — deleting them could make the remaining schedule
illegal.  The true recoverable gain is <= this bound.

Excluded by design: redundant SWAP analysis (SWAPs cost no EPR, and the EPR
count is the headline metric) and moving gates' rendezvous cores (that is
re-routing, not cancellation).

Usage:  python3 probe_cancelable_pairs.py [25q] [64q]
Writes results/results_cancelable_pairs_{suite}.json incrementally.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from ablate_common import (SUITES, SUITE_CIRCUITS, RESULTS_DIR,
                           load_circuit, default_layouts, gmean)
from dsabre_ext import dSABRE_BurstExt


def route_best_traced(router, dag, rev_dag, layouts, pattern="fbf"):
    """run_protocol_ex over each layout, keeping the best forward pass's
    metrics (with trace) AND the layout that pass started from."""
    best = None          # (m, start_layout)
    for L in layouts:
        layout = L
        for step in pattern:
            d = dag if step == "f" else rev_dag
            start = dict(layout)
            m, final_layout = router.route(d, layout)
            if m["aborted"]:
                continue
            layout = final_layout
            if step == "f" and (best is None or m["eprs"] < best[0]["eprs"]):
                best = (m, start)
    return best


def analyse_trace(trace, start_layout, arch):
    """Per-qubit excess-teleport accounting over one routed pass."""
    pos    = {q: arch.core_of(p) for q, p in start_layout.items()}
    anchor = dict(pos)                   # core after the qubit's last 2q gate
    hops   = {q: 0 for q in pos}         # teleports since anchor
    cd     = arch.core_dist

    n_tele = 0
    n_gate = 0
    excess = {"roundtrip": 0, "detour": 0, "tail": 0}
    windows = {"total": 0, "roundtrip": 0, "detour": 0}
    per_qubit_seq = {q: [] for q in pos}  # for the strict-pair count

    for ev in trace:
        kind = ev[0]
        if kind == "TELE":
            _, virt, _p_src, _p_dst, src_c, dst_c = ev
            if virt is None:
                continue
            assert pos[virt] == src_c, (
                f"tracking drift: {virt} tracked in core {pos[virt]}, "
                f"teleport says {src_c}")
            pos[virt] = dst_c
            hops[virt] += 1
            per_qubit_seq[virt].append(("T", src_c, dst_c))
            n_tele += 1
        elif kind == "GATE":
            _, q1, q2, core = ev
            assert pos[q1] == pos[q2] == core, (
                f"tracking drift at gate: {pos[q1]}, {pos[q2]}, exec core {core}")
            n_gate += 1
            for q in (q1, q2):
                need = cd[anchor[q]][core]
                ex   = hops[q] - need
                if hops[q] > 0:
                    windows["total"] += 1
                if ex > 0:
                    if need == 0:
                        excess["roundtrip"] += ex
                        windows["roundtrip"] += 1
                    else:
                        excess["detour"] += ex
                        windows["detour"] += 1
                anchor[q] = core
                hops[q]   = 0
                per_qubit_seq[q].append(("G", core))
        # SWAPs are intra-core: they never change pos and cost no EPR.

    for q in pos:                        # teleports after the last gate on q
        excess["tail"] += hops[q]

    # Strict cancelable-PAIR count: immediately consecutive teleports of the
    # same qubit (no gate on it between) that exactly undo each other.
    strict_pairs = 0
    for q, seq in per_qubit_seq.items():
        i = 0
        while i + 1 < len(seq):
            a, b = seq[i], seq[i + 1]
            if a[0] == "T" and b[0] == "T" and a[1] == b[2] and a[2] == b[1]:
                strict_pairs += 1
                i += 2
            else:
                i += 1

    return dict(n_tele=n_tele, n_gate=n_gate, excess=excess,
                excess_total=sum(excess.values()),
                windows=windows, strict_pairs=strict_pairs)


def main():
    suites = [s for s in sys.argv[1:] if s in SUITES] or ["25q", "64q"]
    for suite in suites:
        s = SUITES[suite]
        arch, nq = s["arch"], s["n_qubits"]
        hw = replace(s["hw"], trace_routing=True)
        router = dSABRE_BurstExt(arch, hw)
        out_path = os.path.join(RESULTS_DIR, f"results_cancelable_pairs_{suite}.json")
        rows = {}
        print(f"=== {suite} ({s['arch_name']}) ===", flush=True)
        for cname in SUITE_CIRCUITS[suite]:
            t0 = time.perf_counter()
            qc, dag, rev_dag, n_cx = load_circuit(suite, cname)
            layouts = default_layouts(qc, dag, arch, nq, seed=0, n_seeds=3)
            best = route_best_traced(router, dag, rev_dag, layouts)
            if best is None:
                print(f"  {cname}: ALL PASSES ABORTED", flush=True)
                rows[cname] = dict(aborted=True)
            else:
                m, start = best
                a = analyse_trace(m["trace"], start, arch)
                reliable = (m["backup_activations"] == 0
                            and a["n_tele"] == m["teles"])
                bound_pct = 100.0 * a["excess_total"] / m["eprs"] if m["eprs"] else 0.0
                rows[cname] = dict(
                    eprs=m["eprs"], ls=m["ls"], teles=m["teles"], n_cx=n_cx,
                    backup_activations=m["backup_activations"],
                    force_make_room=m["force_make_room"],
                    trace_reliable=reliable,
                    **{k: v for k, v in a.items() if k != "n_tele"},
                    excess_bound_pct=round(bound_pct, 2),
                )
                print(f"  {cname}: eprs={m['eprs']} excess={a['excess_total']} "
                      f"(roundtrip={a['excess']['roundtrip']} "
                      f"detour={a['excess']['detour']} tail={a['excess']['tail']}) "
                      f"strict_pairs={a['strict_pairs']} "
                      f"bound={bound_pct:.1f}%"
                      f"{'' if reliable else '  [UNRELIABLE: recovery fired]'}"
                      f"  [{time.perf_counter()-t0:.0f}s]", flush=True)
            with open(out_path, "w") as f:
                json.dump(dict(meta=dict(study="cancelable_pairs", suite=suite,
                                         protocol="fbf best-forward, 3 layouts",
                                         date="2026-08-04"),
                               circuits=rows), f, indent=2)
        ok = [r for r in rows.values() if not r.get("aborted")]
        if ok:
            # If circuit i's EPR count shrank by its bound, the suite gmean
            # would shrink by gmean(1 - b_i): the bound on the reportable gain.
            g = gmean([1.0 - r["excess_bound_pct"] / 100.0 for r in ok])
            print(f"  {suite} gmean upper bound: {100.0 * (1.0 - g):.2f}% "
                  f"of gmean EPR removable (optimistic)", flush=True)


if __name__ == "__main__":
    main()
