"""
probe_relief_monolithic_layout.py — does congestion relief matter under the
SUBMITTED version's initial layout (whole-chip 4-corner removal), which the
submitted paper claimed gave relief a +23.4% gmean-EPR benefit at 64q?

Background
----------
The current `layout.py::sabre_locked_boundary_layout` uses PER-CORE
fill-adaptive corner reservation (fixed 2026-07-17, commit 6b1fc31).  Before
that fix, the function removed only the 4 globally-lowest-degree nodes of the
WHOLE architecture graph, sorted by node id ascending -- and under core-major
numbering those 4 nodes are always the 4 corners of core 0, so cores 1..K-1
get ZERO reserved escape slots and regularly come out fully packed.  That is
the layout the submitted manuscript's relief ablation ran under; the
9-of-9-non-failing search this probe follows up on (relief shows no benefit
on any architecture that itself completes reliably, outside the deliberately
capacity-starved 64q_c33 stress case, where Multiplier fails unconditionally
regardless of relief) found nothing under the CURRENT per-core layout.  This
checks whether the OLD layout -- on the real 96-physical 64q H-grid, not an
artificially shrunk machine -- reproduces the submitted +23.4% and, just as
importantly, whether it does so WITHOUT circuits failing unconditionally the
way Multiplier does on 64q_c33.

`_monolithic_corner_layout` below is the pre-fix `sabre_locked_boundary_layout`
copied verbatim from git commit 6b1fc31^ (`git show 6b1fc31^:code/layout.py`).
It is intentionally NOT restored into layout.py -- that function is correct
and used by every current headline number -- this is a standalone historical
reconstruction for this one comparison.

Usage:  python3 code/probe_relief_monolithic_layout.py [--circuits ...] [--full9]
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
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from layout import run_sabre_passes
from circuit_paths import circuits_path

CIRCUIT_DIR = circuits_path("qasm_64")
SUFFIX = "_nativegates_ibm_qiskit_opt3_64.qasm"

# The submitted paper's 64q suite (6 circuits); the current suite adds 3 more.
SIX = ["ae", "ghz", "graphstate", "qft", "qnn", "random"]
NINE = SIX + ["qpeexact", "qaoa", "multiplier"]
EXPECTED_CX = {"ae": 1962, "ghz": 63, "graphstate": 64, "qft": 1966,
              "qnn": 8126, "random": 1627,
              "qpeexact": 2139, "qaoa": 3920, "multiplier": 13040}


def _monolithic_corner_layout(qc, dag, arch, seed: int = 0):
    """Verbatim copy of the pre-2026-07-17 `sabre_locked_boundary_layout`
    (git show 6b1fc31^:code/layout.py) -- whole-chip 4-corner removal, not
    per-core.  Do not "fix" this; reproducing the old bug is the point."""
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
    return layouts, corner_nodes


def load_qasm(path):
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import RemoveBarriers
    qc = QuantumCircuit.from_qasm_file(path).remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def core_occupancy(layout, arch):
    occ = [0] * arch.num_cores
    for p in layout.values():
        occ[arch.core_of(p)] += 1
    return occ


def gmean(vals):
    vals = [v for v in vals if v is not None and v > 0]
    if not vals:
        return float("nan")
    from math import exp, log
    return exp(sum(log(v) for v in vals) / len(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full9", action="store_true",
                    help="run all 9 circuits (default: the original 6)")
    ap.add_argument("--circuits", default=None)
    args = ap.parse_args()

    circuits = ([c.strip() for c in args.circuits.split(",")] if args.circuits
               else (NINE if args.full9 else SIX))

    arch = build_h_grid_architecture(2, 3, 4)   # the real 64q suite: 6 cores of 4x4, 96 physical
    print(f"architecture: H-grid 2x3 of 4x4 (96 physical, 6 cores, {len(arch.inter_core_links)} links)",
          flush=True)

    hw = HardwareConfig(deadlock_limit=100, max_backup_attempts=100, max_iterations=20000)
    configs = {"full": hw, "no_relief": HardwareConfig(**{**hw.__dict__, "enable_congestion_relief": False})}

    loaded = {}
    for c in circuits:
        qc, dag = load_qasm(os.path.join(CIRCUIT_DIR, c + SUFFIX))
        ncx = sum(1 for _ in dag.two_qubit_ops())
        if ncx != EXPECTED_CX[c]:
            raise SystemExit(f"ABORT: {c} has {ncx} CX, published table says "
                             f"{EXPECTED_CX[c]}. See benchmark_circuits/README.md.")
        loaded[c] = (qc, dag, ncx)
        print(f"  loaded {c:<12} {ncx:>6} CX  (preflight OK)", flush=True)

    # Report the layout's per-core occupancy once (layout construction is
    # config-independent), to check the "5 of 6 cores unprotected" claim
    # directly before spending time on routing.
    print(f"\n{'='*90}\nINITIAL-LAYOUT OCCUPANCY UNDER THE MONOLITHIC (submitted-version) RULE\n{'='*90}",
          flush=True)
    print(f"core capacity = {len(arch.core_qubits(0))} each\n", flush=True)
    layouts_by_circuit = {}
    for c in circuits:
        qc, dag, ncx = loaded[c]
        t0 = time.time()
        layouts, corner_nodes = _monolithic_corner_layout(qc, dag, arch, seed=0)
        layouts_by_circuit[c] = layouts
        if not layouts:
            print(f"  {c:<12} LAYOUT CONSTRUCTION FAILED (no seed produced a layout)", flush=True)
            continue
        occ = core_occupancy(layouts[0], arch)
        full_cores = sum(1 for o in occ if o == len(arch.core_qubits(0)))
        print(f"  {c:<12} occupancy={occ}  full_cores={full_cores}/6  "
              f"reserved_corner_core={arch.core_of(next(iter(corner_nodes)))}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    print(f"\n{'='*90}\nROUTING: full vs no_relief, under the monolithic layout\n{'='*90}",
          flush=True)
    results = {cfg: {} for cfg in configs}
    for cfg_name, cfg in configs.items():
        router = dSABRE_BurstExt(arch, cfg)
        print(f"\n── {cfg_name} ──", flush=True)
        for c in circuits:
            qc, dag, ncx = loaded[c]
            layouts = layouts_by_circuit[c]
            if not layouts:
                print(f"{c:<12} NO LAYOUT", flush=True)
                results[cfg_name][c] = None
                continue
            from qiskit.converters import circuit_to_dag
            rev_dag = circuit_to_dag(qc.reverse_ops())
            t0 = time.time()
            best = None
            for L in layouts:
                m = run_sabre_passes(router, dag, rev_dag, L)
                if m and not m.get("aborted"):
                    if best is None or m["eprs"] < best["eprs"]:
                        best = m
            dt = time.time() - t0
            if best is None:
                print(f"{c:<12} {'ABORT':>7} {'':>8}  ({dt:.1f}s)", flush=True)
                results[cfg_name][c] = None
            else:
                print(f"{c:<12} {best['eprs']:>7} {best['ls']:>8}  ({dt:.1f}s)", flush=True)
                results[cfg_name][c] = best["eprs"]

    print(f"\n{'='*90}\nSUMMARY\n{'='*90}", flush=True)
    full_vals = [results["full"][c] for c in circuits]
    none_vals = [results["no_relief"][c] for c in circuits]
    n_full_ok = sum(1 for v in full_vals if v is not None)
    n_none_ok = sum(1 for v in none_vals if v is not None)
    print(f"completed: full={n_full_ok}/{len(circuits)}  no_relief={n_none_ok}/{len(circuits)}",
          flush=True)
    gm_full = gmean(full_vals)
    gm_none = gmean([v for v in none_vals if v is not None])
    print(f"gmean EPR (circuits that complete under BOTH): "
          f"full={gmean([f for f,n in zip(full_vals,none_vals) if f and n]):.1f}  "
          f"no_relief={gmean([n for f,n in zip(full_vals,none_vals) if f and n]):.1f}", flush=True)
    if all(v is not None for v in full_vals) and all(v is not None for v in none_vals):
        gm_delta = 100 * (gm_none - gm_full) / gm_full
        print(f"gmean EPR (all circuits, both complete): full={gm_full:.1f}  "
              f"no_relief={gm_none:.1f}  delta={gm_delta:+.1f}%  "
              f"(submitted paper claimed +23.4%)", flush=True)
    else:
        print("Not every circuit completed under both configs -- gmean over "
              "the full set is undefined; see per-circuit table above.", flush=True)


if __name__ == "__main__":
    main()
