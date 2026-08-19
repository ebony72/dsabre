r"""
probe_pytket_repair.py — does pytket-dqc's bounded-capacity repair work on our
device, and what does it cost?

Section~\ref{sec:pytket} reports that most pytket-dqc distributions cannot be
materialised within the deg(c) communication ports our cores provide.  That is
a statement about the distribution *as emitted*.  pytket-dqc does have a repair
path -- `to_pytket_circuit(satisfy_bound=True, allow_update=True)` -- which
rewrites a distribution until it fits a bounded `server_ebit_mem`.  The paper
footnotes that this call did not return within nine minutes on the 25-qubit
QFT, and reports realisability instead.

This probe measures the repair path directly, per circuit, so the claim can be
stated quantitatively rather than as a single anecdote:

  * does the repair return inside a budget?
  * if it does, how many e-bits does the executable circuit actually need,
    against the count pytket-dqc reports for the unrepaired distribution?

The distinction matters for how the comparison is worded.  "pytket-dqc has no
mechanism for bounded communication memory" would be false -- the mechanism
exists and works on small inputs.  What is true is narrower and checkable:
the distributors never optimise under the bound (passing a bounded network
leaves all 28 measured distributions byte-identical), the reported cost is the
unrepaired one, and the repair that makes a distribution executable both
raises that cost and, on these circuits, frequently does not return.

Output: code/results/results_pytket_repair.json  (written incrementally)

Usage:  python3 code/probe_pytket_repair.py [--budget 300] [--suite 25q]
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from architecture import build_b_grid_architecture, build_h_grid_architecture
from circuit_paths import circuits_path

OUT = os.path.join(_HERE, "results", "results_pytket_repair.json")

SUITES = {
    "25q": dict(arch=build_b_grid_architecture(2, 2, 4),
                circuit_dir=circuits_path("qasm_25"),
                suffix="_nativegates_ibm_qiskit_opt3_25.qasm"),
    "36q": dict(arch=build_b_grid_architecture(2, 2, 4),
                circuit_dir=circuits_path("qasm_36"),
                suffix="_nativegates_ibm_qiskit_opt3_36.qasm"),
    "64q": dict(arch=build_h_grid_architecture(2, 3, 4),
                circuit_dir=circuits_path("qasm_64"),
                suffix="_nativegates_ibm_qiskit_opt3_64.qasm"),
}


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def _with_timeout(fn, seconds):
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(int(max(1, seconds)))
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def core_layout(arch):
    """Server qubit sets and coupling from our architecture, physical model:
    a core offers M - deg(c) data slots and holds deg(c) link qubits."""
    import networkx as nx
    cg = arch.core_graph
    coupling = [[int(u), int(v)] for u, v in cg.edges()]
    deg = {c: cg.degree(c) for c in cg.nodes()}
    return coupling, deg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=float, default=300.0,
                    help="seconds allowed for each repair call")
    ap.add_argument("--suite", default="25q", choices=list(SUITES))
    args = ap.parse_args()

    from pytket.qasm import circuit_from_qasm
    from pytket_dqc.circuits.distribution import Distribution
    from pytket_dqc.distributors import CoverEmbeddingSteinerDetached
    from pytket_dqc.networks import NISQNetwork
    from pytket_dqc.utils import DQCPass
    from pytket_dqc.utils.gateset import is_start_proc

    s = SUITES[args.suite]
    arch = s["arch"]
    coupling, deg = core_layout(arch)
    n_cores = len(deg)
    m = arch.num_qubits // n_cores if hasattr(arch, "num_qubits") else 16

    rows = []
    import glob
    files = sorted(glob.glob(os.path.join(s["circuit_dir"], "*.qasm")))
    print(f"suite {args.suite}: {n_cores} cores, deg = {sorted(set(deg.values()))}, "
          f"repair budget {args.budget:.0f}s per circuit\n", flush=True)

    for qf in files:
        name = os.path.basename(qf).replace(s["suffix"], "")
        if not qf.endswith(s["suffix"]):
            continue
        circ = circuit_from_qasm(qf)
        DQCPass().apply(circ)
        n_log = circ.n_qubits
        # Physical model: even split of logical qubits across cores for the
        # placement, deg(c) link qubits per core for the bound.
        per = -(-n_log // n_cores)
        sq = {c: list(range(c * per, min((c + 1) * per, n_log)))
              for c in range(n_cores)}
        sq = {c: v for c, v in sq.items() if v}
        if len(sq) != n_cores:
            print(f"  {name:12} SKIP (cannot fill {n_cores} servers)", flush=True)
            continue

        row = dict(suite=args.suite, circuit=name, n_logical=n_log,
                   gates=circ.n_gates, cores=n_cores)
        try:
            dist = _with_timeout(
                lambda: CoverEmbeddingSteinerDetached().distribute(
                    circ, NISQNetwork(coupling, sq), seed=0), args.budget)
        except Exception as e:
            row["distribute"] = type(e).__name__
            rows.append(row)
            print(f"  {name:12} distribute: {type(e).__name__}", flush=True)
            continue

        reported = dist.cost()          # BEFORE any repair; repair mutates
        row["reported_ebits"] = reported

        bounded = NISQNetwork(coupling, sq, server_ebit_mem=dict(deg))
        t0 = time.perf_counter()
        try:
            rep = Distribution(dist.circuit, dist.placement, bounded)
            pc = _with_timeout(
                lambda: rep.to_pytket_circuit(satisfy_bound=True,
                                              allow_update=True), args.budget)
            dt = time.perf_counter() - t0
            n_start = sum(1 for c in pc.get_commands() if is_start_proc(c))
            row.update(repair="ok", repair_s=round(dt, 1),
                       repaired_ebits=n_start,
                       inflation=round(n_start / reported, 2) if reported else None)
            print(f"  {name:12} reported {reported:5} e-bits -> repaired "
                  f"{n_start:5} ({row['inflation']}x) in {dt:.0f}s", flush=True)
        except _Timeout:
            dt = time.perf_counter() - t0
            row.update(repair="timeout", repair_s=round(dt, 1))
            print(f"  {name:12} reported {reported:5} e-bits -> repair TIMEOUT "
                  f"after {dt:.0f}s", flush=True)
        except Exception as e:
            dt = time.perf_counter() - t0
            row.update(repair=type(e).__name__, repair_s=round(dt, 1))
            print(f"  {name:12} reported {reported:5} e-bits -> repair "
                  f"{type(e).__name__}", flush=True)

        rows.append(row)
        with open(OUT, "w") as f:
            json.dump(dict(meta=dict(
                date="2026-07-31", suite=args.suite,
                budget_s=args.budget,
                note="repair = to_pytket_circuit(satisfy_bound=True, "
                     "allow_update=True) against server_ebit_mem = deg(core); "
                     "reported_ebits captured BEFORE the repair, which mutates "
                     "the Distribution in place",
            ), results=rows), f, indent=1)

    print(f"\nDone -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
