"""
ablate_common.py — shared machinery for the 2026-07-29 dSABRE (dSE) ablations.

Three studies build on this module:

  ablate_occupancy.py       initial mappings with a controlled per-core
                            occupancy profile, crossed with hop-gain and
                            congestion-relief on/off.
  ablate_protocol.py        fwd / fwd-fwd / fwd-fwd-fwd / fwd-bwd-fwd /
                            fwd-bwd-fwd-bwd-fwd pass protocols.
  ablate_corner_variants.py corner-reservation count and shape variants.

All three are restricted to the 25q (B-grid 2x2 4x4) and 64q (H-grid 2x3 4x4)
suites, and all route with dSE (`dSABRE_BurstExt`) only — per the project convention, dS is
run only for the "topological extended set" row of the mechanism ablation.

Nothing here modifies the existing modules; `sabre_layout_masked` is a
generalisation of `layout.sabre_locked_boundary_layout` (arbitrary removed-node
set instead of the fixed per-core corner set) and reproduces it exactly when
handed `_per_core_reserved_corner_nodes(arch, adaptive_corner_count(...))`.
"""

from __future__ import annotations

import json
import os
import random as _random
import sys
import time
from math import prod

sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import networkx as nx

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from architecture import build_b_grid_architecture, build_h_grid_architecture
from config import HardwareConfig
from layout import (_interaction_graph, _topology_aware_core_assignment,
                    _per_core_reserved_corner_nodes, adaptive_corner_count)
from circuit_paths import circuits_path

RESULTS_DIR = os.environ.get("DSABRE_OUT_DIR") or os.path.join(_HERE, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ── Suites ────────────────────────────────────────────────────────────────────

SUITES = {
    "25q": dict(
        arch        = build_b_grid_architecture(r=2, s=2, m=4),
        circuit_dir = circuits_path("qasm_25"),
        suffix      = "_nativegates_ibm_qiskit_opt3_25.qasm",
        n_qubits    = 25,
        hw          = HardwareConfig(),
        arch_name   = "B-grid 2x2 4x4 (64 physical, 4 cores x 16)",
    ),
    "64q": dict(
        arch        = build_h_grid_architecture(r=2, s=3, m=4),
        circuit_dir = circuits_path("qasm_64"),
        suffix      = "_nativegates_ibm_qiskit_opt3_64.qasm",
        n_qubits    = 64,
        hw          = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                                     max_iterations=20000),
        arch_name   = "H-grid 2x3 4x4 (96 physical, 6 cores x 16)",
    ),
    # The 64q circuits on a 3x3 grid of 3x3 cores.  The point is the core
    # graph: diameter 4 against the H-grid's 3 and the B-grid's 2, so the
    # hop-gain term — which can only discriminate between candidate next-cores
    # at core distance >= 2 — finally has room to act.  The price is capacity:
    # 64 logical on 81 physical is 79% fill, `adaptive_corner_count` returns
    # k=0 (no reservation is affordable), and cores run near-full by
    # construction, which is also the regime congestion relief targets.
    "64q_c33": dict(
        arch        = build_b_grid_architecture(r=3, s=3, m=3),
        circuit_dir = circuits_path("qasm_64"),
        suffix      = "_nativegates_ibm_qiskit_opt3_64.qasm",
        n_qubits    = 64,
        hw          = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                                     max_iterations=20000),
        arch_name   = "3x3 grid of 3x3 cores (81 physical, 9 cores x 9, "
                      "core diameter 4)",
    ),
    # ── Free-slot study architectures (2026-07-30) ────────────────────────────
    # Same circuit suites on tighter chips, so F = K*spc - n is actually small.
    # 64q on 2x4 of 3x3: F = 8 = K exactly — the "one free slot per core on
    # average" ideal case of the routability question.
    "64q_b243": dict(
        arch        = build_b_grid_architecture(r=2, s=4, m=3),
        circuit_dir = circuits_path("qasm_64"),
        suffix      = "_nativegates_ibm_qiskit_opt3_64.qasm",
        n_qubits    = 64,
        hw          = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                                     max_iterations=20000),
        arch_name   = "B-grid 2x4 of 3x3 (72 physical, 8 cores x 9) — F=8=K",
    ),
    # 25q on 2x2 of 3x3: F = 11, K = 4 (cycle core graph, all cores symmetric).
    "25q_c223": dict(
        arch        = build_b_grid_architecture(r=2, s=2, m=3),
        circuit_dir = circuits_path("qasm_25"),
        suffix      = "_nativegates_ibm_qiskit_opt3_25.qasm",
        n_qubits    = 25,
        hw          = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                                     max_iterations=20000),
        arch_name   = "B-grid 2x2 of 3x3 (36 physical, 4 cores x 9) — F=11",
    ),
    # 25q on 1x3 of 3x3: F = 2 < K = 3 (path core graph) — at least one core
    # is full at every instant, the sub-K frontier of the routability question.
    "25q_c133": dict(
        arch        = build_b_grid_architecture(r=1, s=3, m=3),
        circuit_dir = circuits_path("qasm_25"),
        suffix      = "_nativegates_ibm_qiskit_opt3_25.qasm",
        n_qubits    = 25,
        hw          = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                                     max_iterations=20000),
        arch_name   = "B-grid 1x3 of 3x3 (27 physical, 3 cores x 9) — F=2<K=3",
    ),
}

# The six-circuit ablation core used by `regen_ablation_corners.py` (Table VI):
# keeps the geometric means comparable with the published ablation tables and
# keeps the 13k-CX multiplier out of a study it cannot change the sign of.
ABLATION_CIRCUITS = ["ae", "ghz", "graphstate", "qft", "qnn", "random"]

# The full circuit set of each suite as reported in Table III of the paper.
# ABLATION_CIRCUITS is the historical six-circuit subset; use these when an
# ablation is meant to share the main table's basis.
SUITE_CIRCUITS = {
    "25q": list(ABLATION_CIRCUITS),
    "64q": ABLATION_CIRCUITS + ["qpeexact", "qaoa", "multiplier"],
}

# Preflight against the published CX counts — a suite .qasm that has been
# regenerated with a newer MQT Bench emits a structurally different circuit
# (see benchmark_circuits/README.md, "Do not regenerate").
EXPECTED_CX = {
    "25q": {"ae": 558, "ghz": 24, "graphstate": 25,
            "qft": 580, "qnn": 1223, "random": 1124},
    # The 64q suite of Table III is nine circuits: the six above plus the three
    # class-completion additions.  ABLATION_CIRCUITS covers only the first six;
    # drivers that ablate the *whole* suite iterate SUITE_CIRCUITS["64q"].
    "64q": {"ae": 1962, "ghz": 63, "graphstate": 64,
            "qft": 1966, "qnn": 8126, "random": 1627,
            "qpeexact": 2139, "qaoa": 3920, "multiplier": 13040},
}
EXPECTED_CX["64q_c33"] = EXPECTED_CX["64q"]      # same circuits, other hardware
EXPECTED_CX["64q_b243"] = EXPECTED_CX["64q"]
EXPECTED_CX["25q_c223"] = EXPECTED_CX["25q"]
EXPECTED_CX["25q_c133"] = EXPECTED_CX["25q"]


def load_circuit(suite: str, cname: str):
    """Load one suite circuit; returns (qc, dag, rev_dag, n_cx).

    Aborts on a CX-count mismatch against EXPECTED_CX rather than silently
    benchmarking a regenerated circuit.
    """
    s = SUITES[suite]
    path = os.path.join(s["circuit_dir"], cname + s["suffix"])
    qc = QuantumCircuit.from_qasm_file(path).remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    dag = circuit_to_dag(qc)
    rev_dag = circuit_to_dag(qc.reverse_ops())
    n_cx = sum(1 for _ in dag.two_qubit_ops())
    exp = EXPECTED_CX[suite][cname]
    if n_cx != exp:
        raise SystemExit(
            f"ABORT: {suite}/{cname} has {n_cx} CX, published table says {exp}. "
            f"The circuit has been regenerated — see benchmark_circuits/README.md."
        )
    return qc, dag, rev_dag, n_cx


# ── Layout: SabreLayout over an arbitrary allowed-node subset ─────────────────

def sabre_layout_masked(qc, dag, arch, removed: set, seed: int = 0, n_seeds: int = 3):
    """`layout.sabre_locked_boundary_layout` with an arbitrary removed set.

    Runs SabreLayout on the architecture graph with `removed` physical qubits
    deleted, for seeds [seed .. seed+n_seeds-1].  Qubits SabreLayout leaves
    unplaced are filled into shuffled free slots, exactly as the original does.

    Returns a list of up to `n_seeds` layouts ({logical_qubit: physical_qubit}).
    """
    from qiskit.transpiler import PassManager as _PM, CouplingMap
    from qiskit.transpiler.passes import SabreLayout

    reduced_nodes = [n for n in arch.Gr.nodes() if n not in removed]
    node_to_idx = {n: i for i, n in enumerate(reduced_nodes)}
    reduced_edges = [(node_to_idx[u], node_to_idx[v])
                     for u, v in arch.Gr.edges()
                     if u not in removed and v not in removed]
    directed = reduced_edges + [(v, u) for u, v in reduced_edges]
    cm = CouplingMap(couplinglist=directed, description="dsabre_masked")

    layouts = []
    for sd in range(seed, seed + n_seeds):
        try:
            pm = _PM([SabreLayout(cm, max_iterations=3, seed=sd, swap_trials=5)])
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
        except Exception as e:                                    # noqa: BLE001
            print(f"    [sabre_layout_masked seed {sd} failed: {e}]", flush=True)
            continue
    return layouts


def default_layouts(qc, dag, arch, n_qubits, seed: int = 0, n_seeds: int = 3):
    """The paper's production layout: per-core adaptive corner reservation."""
    k = adaptive_corner_count(arch, n_qubits)
    removed = _per_core_reserved_corner_nodes(arch, per_core=k)
    return sabre_layout_masked(qc, dag, arch, removed, seed=seed, n_seeds=n_seeds)


# ── Layout: forcing a per-core occupancy profile ──────────────────────────────

def core_occupancy(layout: dict, arch) -> list:
    """Per-core count of occupied physical slots for a {lq: phys} layout."""
    occ = [0] * arch.num_cores
    for p in layout.values():
        occ[arch.core_of(p)] += 1
    return occ


def repack_to_budgets(base_layout: dict, dag, arch, budgets) -> dict:
    """Re-map `base_layout` so that core c holds exactly `budgets[c]` qubits.

    Two stages, both deterministic given `base_layout`:

    1. *Core assignment.*  Starting from the core groups of `base_layout`,
       repeatedly take the most over-budget core and move out the qubit with
       the least interaction weight inside it, into the under-budget core with
       the highest interaction weight for that qubit (ties broken by core
       distance, then core id).  This keeps as much of the baseline's locality
       as the profile allows.
    2. *Within-core placement.*  Each core's group is placed on that core's
       physical slots by `layout._topology_aware_core_assignment`, i.e. the
       most-connected logical qubits go to the most-central physical slots.

    `sum(budgets)` must equal the number of logical qubits, so the resulting
    occupancy equals `budgets` exactly.
    """
    n = len(base_layout)
    if sum(budgets) != n:
        raise ValueError(f"budgets sum to {sum(budgets)}, need {n}")

    ig = _interaction_graph(dag)
    groups = {c: [] for c in range(arch.num_cores)}
    for lq, p in base_layout.items():
        groups[arch.core_of(p)].append(lq)

    def affinity(q, group):
        if q not in ig:
            return 0.0
        gs = set(group)
        return sum(ig[q][nb].get("weight", 1) for nb in ig.neighbors(q) if nb in gs)

    guard = 0
    while True:
        over = [c for c in range(arch.num_cores) if len(groups[c]) > budgets[c]]
        if not over:
            break
        guard += 1
        if guard > 4 * n:                       # cannot happen; cheap insurance
            raise RuntimeError("repack_to_budgets did not converge")
        c_over = max(over, key=lambda c: (len(groups[c]) - budgets[c], -c))
        victim = min(groups[c_over],
                     key=lambda q: (affinity(q, groups[c_over]), id(q)))
        under = [c for c in range(arch.num_cores) if len(groups[c]) < budgets[c]]
        c_under = max(under, key=lambda c: (affinity(victim, groups[c]),
                                            -arch.core_dist[c_over][c], -c))
        groups[c_over].remove(victim)
        groups[c_under].append(victim)

    layout = {}
    for c in range(arch.num_cores):
        layout.update(_topology_aware_core_assignment(groups[c], ig, arch, c))
    if len(layout) != n:
        raise RuntimeError(f"repack produced {len(layout)} of {n} placements")
    return layout


# ── Routing protocols ─────────────────────────────────────────────────────────

def run_protocol_ex(router, dag, rev_dag, initial_layout: dict, pattern: str):
    """Route `pattern` (a string over {'f','b'}) starting from `initial_layout`.

    Each pass starts from the previous pass's final layout.  'b' passes route
    the reversed DAG and exist only to refine the layout, so their metrics are
    never reported: the returned result is the best *forward* pass by EPR count.
    A pass that aborts is skipped (the next pass continues from the layout
    before it); if every forward pass aborts, the best result is None.

    `pattern="fbf"` reproduces `layout.run_sabre_passes`.

    Returns (best_or_None, info) where info records per-pass abort flags —
    the routability studies need "did pass 1 fail from this exact layout"
    even when a later pass rescues the run.
    """
    layout = initial_layout
    total_time = 0.0
    best = None
    flags = []
    for step in pattern:
        d = dag if step == "f" else rev_dag
        t0 = time.perf_counter()
        m, final_layout = router.route(d, layout)
        total_time += time.perf_counter() - t0
        flags.append(bool(m["aborted"]))
        if m["aborted"]:
            continue
        layout = final_layout
        if step == "f" and (best is None or m["eprs"] < best["eprs"]):
            best = dict(m)
    info = dict(pass_aborts=flags,
                pass1_aborted=bool(flags and flags[0]),
                time=total_time)
    if best is not None:
        best["compile_time"] = total_time
        best["passes"] = len(pattern)
    return best, info


def run_protocol(router, dag, rev_dag, initial_layout: dict, pattern: str):
    """Back-compat wrapper: the best-forward-pass metrics dict, or None."""
    return run_protocol_ex(router, dag, rev_dag, initial_layout, pattern)[0]


def route_layout_set(router, dag, rev_dag, layouts, pattern="fbf"):
    """Run `pattern` from each candidate layout; keep the fewest-EPR result
    and aggregate per-layout failure statistics.

    Returns (best_or_None, info):
      info["pass1_aborts"]    how many layouts aborted their FIRST forward pass
                              (the routability event — later passes may rescue)
      info["layout_failures"] how many layouts produced no result at all
      info["total_time"]      wall clock across all layouts and passes
    """
    best, total_time, p1, fails = None, 0.0, 0, 0
    for L in layouts:
        m, inf = run_protocol_ex(router, dag, rev_dag, L, pattern)
        total_time += inf["time"]
        p1 += 1 if inf["pass1_aborted"] else 0
        if m is None:
            fails += 1
            continue
        if best is None or m["eprs"] < best["eprs"]:
            best = m
    if best is not None:
        best = dict(best)
        best["compile_time"] = total_time
    return best, dict(n_layouts=len(layouts), pass1_aborts=p1,
                      layout_failures=fails, total_time=total_time)


def best_over_layouts(router, dag, rev_dag, layouts, pattern="fbf"):
    """Run `pattern` from each candidate layout; keep the fewest-EPR result.

    `compile_time` accumulates over all candidate layouts, i.e. it is the real
    cost of the best-of-N protocol rather than of the winning layout alone.
    """
    best, total_time = None, 0.0
    for L in layouts:
        m = run_protocol(router, dag, rev_dag, L, pattern)
        if m is None:
            continue
        total_time += m["compile_time"]
        if best is None or m["eprs"] < best["eprs"]:
            best = m
    if best is not None:
        best = dict(best)
        best["compile_time"] = total_time
    return best


def summarise(m, extra=None) -> dict:
    """Flatten a router metrics dict into a JSON-friendly result row."""
    if m is None:
        return dict(aborted=True, **(extra or {}))
    row = dict(
        aborted            = False,
        eprs               = m["eprs"],
        ls                 = m["ls"],
        cost               = round(m.get("cost", 0.0), 1),
        time_s             = round(m["compile_time"], 3),
        backup_activations = m.get("backup_activations", 0),
        force_make_room    = m.get("force_make_room", 0),
    )
    row.update(extra or {})
    return row


# ── Reporting ─────────────────────────────────────────────────────────────────

def gmean(vals):
    vals = [v for v in vals if v is not None and v > 0]
    if not vals:
        return float("nan")
    return prod(vals) ** (1.0 / len(vals))


def save_json(path: str, payload: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def meta(study: str, suite: str, **kw) -> dict:
    s = SUITES[suite]
    # `circuits` defaults to the six-circuit subset but MUST be overridable:
    # a driver that runs a different set and inherits this default writes a
    # results file that misreports its own basis.
    circuits = kw.pop("circuits", ABLATION_CIRCUITS)
    return dict(
        date      = time.strftime("%Y-%m-%d"),
        study     = study,
        suite     = suite,
        arch      = s["arch_name"],
        router    = "dSE (dsabre_ext.dSABRE_BurstExt)",
        circuits  = circuits,
        **kw,
    )
