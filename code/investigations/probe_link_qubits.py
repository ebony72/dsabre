r"""
probe_link_qubits.py — why a pytket-dqc distribution needs many *simultaneous*
link qubits, and why that is not an artefact of gate ordering.

Section~\ref{sec:pytket} of the paper reports that 13 of 21 pytket-dqc
distributions cannot be materialised within the communication capacity the
device provides, and that AE at 25 qubits needs at least 19 link qubits per
core against the 2 available.  The obvious objection is that this looks like a
scheduling accident: if hyperedge H0 and hyperedge H1 both want a link into the
same server, why not finish H0, release the link, and only then start H1?

This probe answers that in two parts.

PART 1 — a five-gate circuit in which two links are provably required at once.

    Server A = {q0, q1}   Server B = {q2}   one inter-server link

        CZ(q0,q2)   CZ(q1,q2)   Rx(t, q2)   CZ(q0,q2)   CZ(q1,q2)

    Each control has exactly one CZ on each side of the Rx.  CZ and Rx on q2 do
    not commute, so the serialising reorder -- both of q0's gates, then both of
    q1's -- would place q0's two CZs adjacent, where they cancel.  The probe
    checks this by comparing unitaries: the reordered circuit is a *different
    operator*, so the reorder is not available at any price.  q0's hyperedge and
    q1's therefore interleave by construction and server B holds two link qubits
    simultaneously.

    The probe then prices the alternative.  pytket-dqc can repair a distribution
    to fit a bounded capacity (`allow_update=True`); doing so here splits the
    hyperedges and emits 4 starting processes instead of 2.  Two e-bits at two
    ports, or four e-bits at one -- and `Distribution.cost()` reports 2 for both,
    because it is computed from hypergraph structure and never consults
    `server_ebit_mem`.

PART 2 — the same effect at scale, measured on the real AE circuit.

    Reconstructs the published (model A) distribution of the 25-qubit AE circuit
    and reports, per server, the peak number of hyperedges live at once, plus
    what those hyperedges look like.  This reproduces the `A_required_ebit_mem`
    = 19 recorded in results_pytket_fair_25q.json and shows its cause: at the
    peak every one of the 19 concurrent links spans 6-31 gates, the widest open
    across 2161 of the circuit's 2218 commands.  None is a single-gate
    hyperedge.  Serialising them means breaking them, which is exactly what
    forfeits the amortisation the low e-bit count comes from.

Usage:  python3 code/probe_link_qubits.py
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

import numpy as np
from pytket import Circuit
from pytket.qasm import circuit_from_qasm
from pytket_dqc import HypergraphCircuit
from pytket_dqc.circuits.distribution import Distribution
from pytket_dqc.distributors import CoverEmbeddingSteinerDetached
from pytket_dqc.networks import NISQNetwork
from pytket_dqc.placement import Placement
from pytket_dqc.utils import DQCPass, ConstraintException
from pytket_dqc.utils.gateset import is_start_proc
from circuit_paths import circuits_path

# ── Part 1: the minimal forced example ───────────────────────────────────────

SQ_MIN = {0: [0, 1], 1: [2]}          # server A holds q0,q1 ; server B holds q2
COUPLING_MIN = [[0, 1]]


def _build(order_forced: bool) -> Circuit:
    c = Circuit(3)
    for q in range(3):
        c.H(q)
    if order_forced:                   # one CZ per control on each side of Rx
        c.CZ(0, 2); c.CZ(1, 2); c.Rx(0.37, 2); c.CZ(0, 2); c.CZ(1, 2)
    else:                              # the serialising reorder
        c.CZ(0, 2); c.CZ(0, 2); c.Rx(0.37, 2); c.CZ(1, 2); c.CZ(1, 2)
    DQCPass().apply(c)
    return c


def _place(hyp: HypergraphCircuit) -> Placement:
    """q0,q1 -> server 0 ; q2 -> server 1 ; every gate executes on server 1."""
    pl = {}
    for v in hyp.get_qubit_vertices():
        pl[v] = 0 if hyp.get_qubit_of_vertex(v).index[0] < 2 else 1
    for v in hyp.vertex_list:
        if not hyp.is_qubit_vertex(v):
            pl[v] = 1
    return Placement(pl)


def _min_ports(hyp, plac, upto=4):
    for k in range(1, upto + 1):
        net = NISQNetwork(COUPLING_MIN, SQ_MIN, server_ebit_mem={0: k, 1: k})
        try:
            pc = Distribution(hyp, plac, net).to_pytket_circuit(
                satisfy_bound=True, allow_update=False)
            return k, sum(1 for cmd in pc.get_commands() if is_start_proc(cmd))
        except ConstraintException:
            continue
    return None, None


def part1():
    print("=" * 74)
    print("PART 1  two link qubits required simultaneously, in five gates")
    print("=" * 74)

    forced, reordered = _build(True), _build(False)
    same = np.allclose(forced.get_unitary(), reordered.get_unitary())
    print("\n  circuit          CZ(q0,q2) CZ(q1,q2) Rx(q2) CZ(q0,q2) CZ(q1,q2)")
    print("  serialised as    CZ(q0,q2) CZ(q0,q2) Rx(q2) CZ(q1,q2) CZ(q1,q2)")
    print(f"\n  same unitary?  {same}"
          f"   <- the reorder is {'available' if same else 'NOT a valid rewrite'}")

    hyp = HypergraphCircuit(forced)
    plac = _place(hyp)
    dist = Distribution(hyp, plac, NISQNetwork(COUPLING_MIN, SQ_MIN))
    # Capture before any repair: to_pytket_circuit(allow_update=True) rewrites
    # the Distribution IN PLACE, so cost() re-queried afterwards returns the
    # repaired figure rather than the reported one.
    reported = dist.cost()
    k, n_start = _min_ports(hyp, plac)
    print(f"\n  Distribution.cost()            = {reported} e-bits")
    print(f"  minimum server_ebit_mem        = {k}")
    if n_start is not None:
        print(f"  starting processes at {k} ports  = {n_start}")

    net1 = NISQNetwork(COUPLING_MIN, SQ_MIN, server_ebit_mem={0: 1, 1: 1})
    try:
        repaired = Distribution(hyp, plac, net1)
        pc = repaired.to_pytket_circuit(satisfy_bound=True, allow_update=True)
        n = sum(1 for cmd in pc.get_commands() if is_start_proc(cmd))
        print(f"  repaired to 1 port             = {n} starting processes"
              f"  (cost() now {repaired.cost()}, mutated in place)")
        print(f"\n  => {reported} e-bits at {k} ports, or {n} e-bits at 1 port;"
              f" cost() reports {reported} before the repair is asked for.")
    except Exception as e:                                   # pragma: no cover
        print(f"  repair at 1 port: {type(e).__name__}")


# ── Part 2: the same effect on the real AE circuit ───────────────────────────

AE_QASM = circuits_path("qasm_25/"
                             "ae_nativegates_ibm_qiskit_opt3_25.qasm")
# The model-A network is taken from bench_pytket_fair.build_networks rather
# than restated here: every core offers all M of its qubits as data, with
# unbounded communication memory.  Hardcoding it once let this probe drift
# from the sweep when the network construction was corrected (2026-08-13).
DEVICE_PORTS = 2          # deg(c) on the 2x2 B-grid


def part2():
    print("\n" + "=" * 74)
    print("PART 2  AE at 25 qubits: where the 19 comes from")
    print("=" * 74)
    if not os.path.exists(AE_QASM):
        print(f"  SKIP: {AE_QASM} not found")
        return

    circ = circuit_from_qasm(AE_QASM)
    DQCPass().apply(circ)
    print(f"\n  AE 25q: {circ.n_qubits} qubits, {circ.n_gates} gates", flush=True)

    from bench_pytket_fair import SUITES, build_networks
    nets, info = build_networks(SUITES["25q"]["arch"], circ.n_qubits)
    net_a = nets["A_published"]
    print(f"  model-A network: server sizes "
          f"{[len(net_a.server_qubits[s]) for s in sorted(net_a.server_qubits)]}",
          flush=True)

    t0 = time.time()
    dist = CoverEmbeddingSteinerDetached().distribute(circ, net_a, seed=0)
    print(f"  distributed in {time.time()-t0:.1f}s, cost = {dist.cost()} e-bits",
          flush=True)

    hyp = dist.circuit
    v2c = hyp.get_vertex_to_command_index_map()
    spans = []
    for h in hyp.hyperedge_list:
        gv = [v for v in h.vertices if not hyp.is_qubit_vertex(v)]
        idx = [v2c[v] for v in gv if v in v2c]
        if not idx:
            continue
        qv = [v for v in h.vertices if hyp.is_qubit_vertex(v)][0]
        home = dist.placement.placement[qv]
        servers = {dist.placement.placement[v] for v in gv}
        spans.append((min(idx), max(idx), home, servers, len(gv)))

    sizes = sorted(n for *_, n in spans)
    print(f"\n  hyperedges: {len(spans)}   gates per hyperedge: "
          f"min={sizes[0]} median={sizes[len(sizes)//2]} max={sizes[-1]}")
    print(f"  covering more than one gate: {sum(1 for s in sizes if s > 1)}")

    # A link qubit is held on every server a hyperedge reaches that is not its
    # home, from the hyperedge's first gate to its last.
    events = defaultdict(list)
    for lo, hi, home, servers, _ in spans:
        for s in servers:
            if s != home:
                events[s].append((lo, +1))
                events[s].append((hi + 1, -1))

    print(f"\n  peak simultaneous link qubits (device provides {DEVICE_PORTS}):")
    peaks = {}
    for s in sorted(events):
        cur = peak = 0
        for _, d in sorted(events[s]):
            cur += d
            peak = max(peak, cur)
        peaks[s] = peak
        flag = "  <-- exceeds device" if peak > DEVICE_PORTS else ""
        print(f"    server {s}: {peak}{flag}")

    worst = max(peaks, key=peaks.get)
    live = [(lo, hi, n) for lo, hi, home, servers, n in spans
            if worst in servers and home != worst]
    best_t, best = None, -1
    for t in sorted({t for lo, hi, _ in live for t in (lo, hi)}):
        c = sum(1 for lo, hi, _ in live if lo <= t <= hi)
        if c > best:
            best, best_t = c, t
    at_peak = [(lo, hi, n) for lo, hi, n in live if lo <= best_t <= hi]
    multi = [x for x in at_peak if x[2] > 1]
    print(f"\n  at the peak on server {worst} (command {best_t}): {best} live links")
    print(f"    spanning >1 gate: {len(multi)} of {len(at_peak)}")
    print(f"    gates per hyperedge: "
          f"{sorted((n for *_, n in multi), reverse=True)}")
    print(f"    commands spanned:    "
          f"{sorted((hi - lo for lo, hi, _ in multi), reverse=True)}")
    print(f"\n  The widest is open across {max((hi-lo for lo,hi,_ in multi), default=0)}"
          f" of {circ.n_gates} commands.  Serialising these means splitting them,")
    print("  which forfeits exactly the amortisation the low e-bit count comes from.")


if __name__ == "__main__":
    part1()
    part2()
