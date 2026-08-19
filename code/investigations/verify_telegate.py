"""
verify_telegate.py — independent replay check of the telegate macro-action.

`telegate_router.py` validates its own preconditions inside the transaction,
which is exactly the kind of self-certification a reviewer should not have to
take on trust.  This script re-derives legality from the OUTSIDE: it routes
with `trace_routing=True` and replays the emitted operation stream against a
fresh copy of the initial layout, checking every operation against the
architecture rather than against the router's own bookkeeping.

Checked, per routed circuit:

  SWAP  u v c   u,v adjacent in intra[c]; both inside core c.
  TELE  q a b   a holds q; the source port (a's link partner endpoint) and b
                are the two ends of one physical link; b is empty; a is
                adjacent to that source port, which is itself empty.
  TGATE …       both link endpoints empty; each operand parked on a slot
                adjacent to its own endpoint, in its own core; the two
                endpoints are one physical link and their cores are
                NEIGHBOURS in the core graph (the one-hop restriction).
  GATE  q1 q2   either the two operands are adjacent in the chip graph (a
                genuine local gate) or the immediately preceding entry is a
                TGATE for this very pair (a remote gate).

  Gate stream   the multiset of executed 2Q gates equals the circuit's, and
                for every qubit the order of its gates matches the circuit's
                order on that wire -- i.e. the stream is a valid topological
                order of the 2Q DAG, with nothing dropped or invented.
  EPR count     metrics["eprs"] equals TELE + TGATE entries, and
                metrics["catcomms"] equals the TGATE entries.
  Occupancy     a telegate leaves every core's occupancy unchanged (the
                property that lets c_cap drop out of its score), and no
                physical qubit is ever doubly occupied.

Usage:
    python3 verify_telegate.py                 # 64q suite, default settings
    python3 verify_telegate.py --bias 4 --circuits ae,qft
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from ablate_common import SUITES, SUITE_CIRCUITS, load_circuit, default_layouts
from ablate_telegate import base_config
from telegate_router import dSABRE_Telegate, TelegateConfig


class ReplayError(AssertionError):
    pass


def replay(trace, arch, initial_layout, qc_pairs):
    """Replay `trace` from `initial_layout`; raise ReplayError on any breach.

    `qc_pairs` is the circuit's 2Q gate list as (q1, q2) in circuit order,
    used for the stream check.  Returns a dict of counts.
    """
    l2p = dict(initial_layout)
    p2l = {p: None for p in arch.Gr.nodes}
    for lq, p in l2p.items():
        if p2l[p] is not None:
            raise ReplayError(f"initial layout doubly occupies {p}")
        p2l[p] = lq

    links = set()
    for u, v in arch.inter_core_links:
        links.add((u, v)); links.add((v, u))

    occ0 = _occupancy(p2l, arch)
    executed = []
    n_swap = n_tele = n_tgate = 0
    prev = None

    for k, ev in enumerate(trace):
        kind = ev[0]

        if kind == "SWAP":
            _, u, v, c = ev
            if arch.core_of(u) != c or arch.core_of(v) != c:
                raise ReplayError(f"[{k}] SWAP {u},{v} claims core {c}")
            if not arch.intra[c].has_edge(u, v):
                raise ReplayError(f"[{k}] SWAP {u},{v} not an edge of core {c}")
            qu, qv = p2l[u], p2l[v]
            if qu is not None: l2p[qu] = v
            if qv is not None: l2p[qv] = u
            p2l[u], p2l[v] = qv, qu
            n_swap += 1

        elif kind == "TELE":
            _, q, a, b, cs, cd = ev
            if p2l[a] != q:
                raise ReplayError(f"[{k}] TELE of {q} from {a}, which holds {p2l[a]}")
            if p2l[b] is not None:
                raise ReplayError(f"[{k}] TELE destination port {b} is occupied")
            # The source endpoint is whichever port of core `cs` links to b.
            # Two placements are legal, and both are the PUBLISHED router's,
            # not anything the telegate introduces:
            #   * `_apply_teleport`: the qubit is staged on a slot adjacent
            #     to a FREE source port, which holds the local EPR half;
            #   * `_force_make_room`: the core is full, so the qubit sitting
            #     ON the source port is teleported straight off it (see that
            #     method's docstring).  Only reachable in score-only mode.
            ports = [u for (u, v) in links if v == b and arch.core_of(u) == cs]
            if not ports:
                raise ReplayError(f"[{k}] no link from core {cs} to port {b}")
            staged = any(p2l[u] is None and arch.intra[cs].has_edge(a, u)
                         for u in ports)
            on_port = a in ports
            if not (staged or on_port):
                raise ReplayError(
                    f"[{k}] TELE from {a}: not on a source endpoint and not "
                    f"adjacent to a free one (candidates {ports})")
            if arch.core_of(a) != cs or arch.core_of(b) != cd:
                raise ReplayError(f"[{k}] TELE cores {cs}->{cd} disagree with {a},{b}")
            p2l[a] = None; p2l[b] = q; l2p[q] = b
            n_tele += 1

        elif kind == "TGATE":
            _, virt, partner, s_src, port_src, port_dst, s_dst, cs, cd = ev
            if (port_src, port_dst) not in links:
                raise ReplayError(f"[{k}] TGATE {port_src}-{port_dst} is not a link")
            if arch.core_dist[cs][cd] != 1:
                raise ReplayError(
                    f"[{k}] TGATE spans core distance {arch.core_dist[cs][cd]}, "
                    f"not 1 -- the one-hop restriction is violated")
            if p2l[port_src] is not None or p2l[port_dst] is not None:
                raise ReplayError(f"[{k}] TGATE endpoint occupied "
                                  f"({port_src}:{p2l[port_src]}, {port_dst}:{p2l[port_dst]})")
            if p2l[s_src] != virt:
                raise ReplayError(f"[{k}] TGATE control {virt} not on {s_src}")
            if p2l[s_dst] != partner:
                raise ReplayError(f"[{k}] TGATE partner {partner} not on {s_dst}")
            if not arch.intra[cs].has_edge(s_src, port_src):
                raise ReplayError(f"[{k}] TGATE {s_src} not adjacent to {port_src}")
            if not arch.intra[cd].has_edge(s_dst, port_dst):
                raise ReplayError(f"[{k}] TGATE {s_dst} not adjacent to {port_dst}")
            # Occupancy is the whole point: the cat qubit is created on
            # `port_dst` and measured out inside this one action, so no
            # logical qubit changes core and neither core's occupancy moves.
            # That is what licenses dropping c_cap from its score, and it is
            # visible here as the absence of any p2l write.
            n_tgate += 1

        elif kind == "GATE":
            _, q1, q2, c = ev
            p1, p2 = l2p[q1], l2p[q2]
            local  = arch.Gr.has_edge(p1, p2)
            remote = (prev is not None and prev[0] == "TGATE"
                      and {prev[1], prev[2]} == {q1, q2})
            if not (local or remote):
                raise ReplayError(
                    f"[{k}] GATE {q1},{q2} on {p1},{p2}: not adjacent and not "
                    f"preceded by a matching TGATE")
            executed.append(frozenset((q1, q2)))

        else:
            raise ReplayError(f"[{k}] unknown trace entry {kind}")
        prev = ev

    # Map consistency after the whole replay, and net core flow: only TELE
    # moves a qubit between cores, so the difference from the entry profile
    # must sum to zero and no core may be over its physical capacity.
    for lq, p in l2p.items():
        if p2l[p] != lq:
            raise ReplayError(f"final maps disagree at {lq}/{p}")
    occ1 = _occupancy(p2l, arch)
    if sum(occ1) != sum(occ0):
        raise ReplayError(f"qubit count changed: {sum(occ0)} -> {sum(occ1)}")
    for c in range(arch.num_cores):
        if occ1[c] > len(arch.core_qubits(c)):
            raise ReplayError(f"core {c} over capacity: {occ1[c]}")

    _check_stream(executed, qc_pairs)
    return dict(swaps=n_swap, teleports=n_tele, telegates=n_tgate,
                gates=len(executed))


def _occupancy(p2l, arch):
    occ = [0] * arch.num_cores
    for p, q in p2l.items():
        if q is not None:
            occ[arch.core_of(p)] += 1
    return occ


def _check_stream(executed, qc_pairs):
    """The executed 2Q stream must be a permutation of the circuit's that
    preserves the per-wire order -- i.e. a valid topological order."""
    if len(executed) != len(qc_pairs):
        raise ReplayError(f"executed {len(executed)} 2Q gates, circuit has "
                          f"{len(qc_pairs)}")
    want = defaultdict(list)
    for q1, q2 in qc_pairs:
        want[q1].append(frozenset((q1, q2)))
        want[q2].append(frozenset((q1, q2)))
    got = defaultdict(list)
    for pair in executed:
        for q in pair:
            got[q].append(pair)
    if set(want) != set(got):
        raise ReplayError("executed stream touches a different qubit set")
    for q in want:
        if want[q] != got[q]:
            raise ReplayError(
                f"wire {q}: gate order differs from the circuit "
                f"(first divergence at index "
                f"{next(i for i, (a, b) in enumerate(zip(want[q], got[q])) if a != b)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="64q")
    ap.add_argument("--circuits", default=None)
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--amort", type=float, default=0.0)
    ap.add_argument("--layouts", type=int, default=2)
    args = ap.parse_args()

    s = SUITES[args.suite]
    arch = s["arch"]
    names = (args.circuits.split(",") if args.circuits
             else SUITE_CIRCUITS.get(args.suite, ["ae", "ghz", "graphstate",
                                                  "qft", "qnn", "random"]))
    # Safe mode on the roomy chips, score-only on the tight ones -- the same
    # choice `ablate_telegate.py` makes, so the runs verified here are the
    # runs that produced the numbers.  Score-only matters: it is the mode in
    # which `_apply_teleport`'s rollback paths and `_force_make_room` are
    # live, so a telegate has to coexist with them.
    cfg = TelegateConfig(**base_config(args.suite, arch), telegate=True,
                         telegate_bias=args.bias,
                         telegate_amort_weight=args.amort,
                         trace_routing=True)

    n_ok = n_tg = 0
    for cname in names:
        qc, dag, rev, ncx = load_circuit(args.suite, cname)
        pairs = [(n.qargs[0], n.qargs[1]) for n in dag.topological_op_nodes()
                 if len(n.qargs) == 2]
        layouts = default_layouts(qc, dag, arch, s["n_qubits"], seed=0,
                                  n_seeds=args.layouts)
        for i, L in enumerate(layouts):
            r = dSABRE_Telegate(arch, cfg)
            m, _ = r.route(dag, L)
            if m["aborted"]:
                print(f"  {cname} L{i}: ABORTED, skipped", flush=True)
                continue
            counts = replay(m["trace"], arch, L, pairs)
            if counts["teleports"] + counts["telegates"] != m["eprs"]:
                raise ReplayError(
                    f"{cname} L{i}: trace has "
                    f"{counts['teleports']}+{counts['telegates']} EPR events, "
                    f"metrics says {m['eprs']}")
            if counts["telegates"] != m["catcomms"]:
                raise ReplayError(f"{cname} L{i}: catcomms mismatch")
            if counts["swaps"] != m["ls"]:
                raise ReplayError(f"{cname} L{i}: swap count mismatch "
                                  f"{counts['swaps']} vs {m['ls']}")
            n_ok += 1
            n_tg += counts["telegates"]
            print(f"  OK  {cname:11s} L{i}  gates={counts['gates']:6d} "
                  f"swaps={counts['swaps']:6d} teledata={counts['teleports']:5d} "
                  f"telegate={counts['telegates']:5d}", flush=True)

    print(f"\n{n_ok} routings replayed clean, {n_tg} telegates verified "
          f"(bias={args.bias}, amort={args.amort}).")


if __name__ == "__main__":
    main()
