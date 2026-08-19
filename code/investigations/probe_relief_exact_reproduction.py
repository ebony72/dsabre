"""
probe_relief_exact_reproduction.py — reproduce the submitted paper's Table VI
64q row exactly: Full=134.8, No congestion relief=166.4, +23.4%.

Why this script exists
-----------------------
`probe_relief_monolithic_layout.py` swapped in the submitted version's
initial-layout function (monolithic 4-corner removal) but kept the CURRENT
router and config -- and got the opposite sign (-2.8%, not +23.4%).  Diffing
`code/config.py` against the submitted commit (e1ba02b) found the router
changed too: the submitted version's `HardwareConfig` has
`enable_node_decay: bool = True`, and its `router.py` multiplies the intra-core
SWAP score by a per-node anti-oscillation penalty that the current router does
not have at all (fully removed, commit e1d951e "Drop node_decay").  The
submitted version also used smaller safety limits (max_iterations=10000,
deadlock_limit=50, max_backup_attempts=50) instead of the current probe's
20000/100/100.

This script uses the UNMODIFIED submitted-version router, dsabre_ext and
config (extracted verbatim via `git show e1ba02b:code/{router,dsabre_ext,
config}.py` into `_legacy_*_submitted.py`, imports patched to reference each
other rather than the current modules) together with the same monolithic
layout function, to check whether restoring node_decay (not just the layout)
is what reproduces the submitted number.  If it does, the submitted +23.4%
was never purely a layout-fix artifact -- part of it came from a mechanism
that was deliberately and separately retired.

Usage:  python3 code/probe_relief_exact_reproduction.py [--circuits ...]
"""

from __future__ import annotations

import argparse
import os
import random as _random
import sys
import time

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from architecture import build_h_grid_architecture
from _legacy_config_submitted import HardwareConfig as LegacyHardwareConfig
from _legacy_dsabre_ext_submitted import dSABRE_BurstExt as LegacyBurstExt
from circuit_paths import circuits_path

CIRCUIT_DIR = circuits_path("qasm_64")
SUFFIX = "_nativegates_ibm_qiskit_opt3_64.qasm"
SIX = ["ae", "ghz", "graphstate", "qft", "qnn", "random"]
EXPECTED_CX = {"ae": 1962, "ghz": 63, "graphstate": 64, "qft": 1966,
              "qnn": 8126, "random": 1627}


def _monolithic_corner_layout(qc, dag, arch, seed: int = 0):
    """Verbatim copy of the pre-2026-07-17 sabre_locked_boundary_layout
    (git show 6b1fc31^:code/layout.py) -- see probe_relief_monolithic_layout.py."""
    from qiskit.transpiler import PassManager, CouplingMap
    from qiskit.transpiler.passes import SabreLayout

    degrees = dict(arch.Gr.degree())
    min_degree = min(degrees.values())
    corner_nodes = set(sorted(n for n, d in degrees.items() if d == min_degree)[:4])
    reduced_nodes = [n for n in arch.Gr.nodes() if n not in corner_nodes]
    node_to_idx = {n: i for i, n in enumerate(reduced_nodes)}
    reduced_edges = [
        (node_to_idx[u], node_to_idx[v])
        for u, v in arch.Gr.edges()
        if u not in corner_nodes and v not in corner_nodes
    ]
    directed = reduced_edges + [(v, u) for u, v in reduced_edges]
    cm = CouplingMap(couplinglist=directed, description="dsabre_corners_removed_monolithic")

    layouts = []
    for sd in [seed, seed + 1, seed + 2]:
        try:
            pm = PassManager([SabreLayout(cm, max_iterations=3, seed=sd, swap_trials=5)])
            transpiled = pm.run(qc)
            if transpiled.layout is None:
                continue
            tl = transpiled.layout
            virt_layout = (tl.initial_virtual_layout(filter_ancillas=True)
                           if hasattr(tl, "initial_virtual_layout") else tl.initial_layout)
            result = {}
            for virt_qubit, reduced_idx in virt_layout.get_virtual_bits().items():
                try:
                    bit_index = qc.find_bit(virt_qubit).index
                except Exception:
                    continue
                result[dag.qubits[bit_index]] = reduced_nodes[reduced_idx]
            assigned = set(result.values())
            free = [p for p in arch.data_qubits if p not in assigned]
            _random.Random(sd).shuffle(free)
            fp = iter(free)
            for lq in dag.qubits:
                if lq not in result:
                    result[lq] = next(fp)
            layouts.append(result)
        except Exception:
            continue
    return layouts


def load_qasm(path):
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import RemoveBarriers
    qc = QuantumCircuit.from_qasm_file(path).remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def gmean(vals):
    vals = [v for v in vals if v is not None and v > 0]
    if not vals:
        return float("nan")
    from math import exp, log
    return exp(sum(log(v) for v in vals) / len(vals))


def run_legacy_passes(router, dag, rev_dag, layout):
    """fwd -> bwd -> fwd using the LEGACY router's own route() signature."""
    m1, fwd_final = router.route(dag, layout)
    if m1["aborted"]:
        return None
    m2, rev_final = router.route(rev_dag, fwd_final)
    layout3 = fwd_final if m2["aborted"] else rev_final
    m3, _ = router.route(dag, layout3)
    if m3["aborted"]:
        return m1 if not m1["aborted"] else None
    return m3 if m3["eprs"] <= m1["eprs"] else m1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", default=",".join(SIX))
    args = ap.parse_args()
    circuits = [c.strip() for c in args.circuits.split(",")]

    arch = build_h_grid_architecture(2, 3, 4)
    print(f"architecture: H-grid 2x3 of 4x4 (96 physical, 6 cores)", flush=True)

    # EXACT submitted-version defaults: node_decay on, tighter safety limits.
    hw = LegacyHardwareConfig(max_iterations=10000, deadlock_limit=50,
                              max_backup_attempts=50)
    hw_norelief = LegacyHardwareConfig(max_iterations=10000, deadlock_limit=50,
                                       max_backup_attempts=50,
                                       enable_congestion_relief=False)
    print(f"legacy config: enable_node_decay={hw.enable_node_decay}  "
          f"max_iterations={hw.max_iterations}  deadlock_limit={hw.deadlock_limit}  "
          f"max_backup_attempts={hw.max_backup_attempts}", flush=True)

    loaded = {}
    for c in circuits:
        qc, dag = load_qasm(os.path.join(CIRCUIT_DIR, c + SUFFIX))
        ncx = sum(1 for _ in dag.two_qubit_ops())
        if ncx != EXPECTED_CX[c]:
            raise SystemExit(f"ABORT: {c} has {ncx} CX, published table says {EXPECTED_CX[c]}.")
        loaded[c] = (qc, dag, ncx)
        print(f"  loaded {c:<12} {ncx:>6} CX  (preflight OK)", flush=True)

    print(f"\n{'='*90}\nROUTING under legacy router + legacy config + monolithic layout\n{'='*90}",
          flush=True)
    results = {"full": {}, "no_relief": {}}
    for cfg_name, cfg in (("full", hw), ("no_relief", hw_norelief)):
        router = LegacyBurstExt(arch, cfg)
        print(f"\n── {cfg_name} ──", flush=True)
        for c in circuits:
            qc, dag, ncx = loaded[c]
            from qiskit.converters import circuit_to_dag
            rev_dag = circuit_to_dag(qc.reverse_ops())
            layouts = _monolithic_corner_layout(qc, dag, arch, seed=0)
            t0 = time.time()
            best = None
            for L in layouts:
                m = run_legacy_passes(router, dag, rev_dag, L)
                if m is not None and (best is None or m["eprs"] < best["eprs"]):
                    best = m
            dt = time.time() - t0
            if best is None:
                print(f"{c:<12} {'ABORT':>7}  ({dt:.1f}s)", flush=True)
                results[cfg_name][c] = None
            else:
                print(f"{c:<12} {best['eprs']:>7} {best['ls']:>8}  ({dt:.1f}s)", flush=True)
                results[cfg_name][c] = best["eprs"]

    print(f"\n{'='*90}\nSUMMARY\n{'='*90}", flush=True)
    full_vals = [results["full"][c] for c in circuits]
    none_vals = [results["no_relief"][c] for c in circuits]
    print(f"completed: full={sum(1 for v in full_vals if v)}/{len(circuits)}  "
          f"no_relief={sum(1 for v in none_vals if v)}/{len(circuits)}", flush=True)
    if all(full_vals) and all(none_vals):
        gm_full, gm_none = gmean(full_vals), gmean(none_vals)
        print(f"gmean EPR: full={gm_full:.1f}  no_relief={gm_none:.1f}  "
              f"delta={100*(gm_none-gm_full)/gm_full:+.1f}%  "
              f"(submitted paper: full=134.8, no_relief=166.4, delta=+23.4%)", flush=True)
    else:
        print("not all circuits completed; see per-circuit table above", flush=True)


if __name__ == "__main__":
    main()
