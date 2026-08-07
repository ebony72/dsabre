"""Verify the optimised default router is output-identical to the baseline.

Runs both routers over the published 25q/36q/64q suites under the exact
protocol `benchmark.py` uses -- 3 SabreLayout seeds x fwd->bwd->fwd -- and
compares, for EVERY pass of EVERY seed (not just the reported best):

  * all routing metrics (eprs, ls, teles, cost, 1q_gates, aborted,
    backup_activations, force_make_room, relay_hops, failure_log)
  * the final layout, qubit by qubit
  * on the 25q suite, the complete routing trace with `trace_routing=True`
    -- every SWAP, teleport, and gate execution in order

Any difference is reported and the exit status is non-zero.

Coverage gap: `_get_local_extended` has a second call site, in
`_fallback_local_swap`, which fires only when an inter-core front layer
yields no teleport candidate at all.  That never happens on any circuit of
these three suites (measured: 0 of 916 calls on 64q ae, 0 of 1089 on 64q
random), so the optimised override of it is exercised on the main path
only.  The two sites call the same function, so nothing is untested that is
not also unexercised -- but a circuit that does reach the fallback is not
covered by this check.

Usage:
  python3 verify_router.py                # all suites
  python3 verify_router.py --suite 25q
  python3 verify_router.py --no-trace     # skip the 25q trace pass
"""

import sys, os, glob, time, argparse

sys.setrecursionlimit(50000)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler import PassManager

from architecture import build_b_grid_architecture, build_h_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BFSExt
from _baseline_architecture import (
    build_b_grid_architecture as _base_b_grid,
    build_h_grid_architecture as _base_h_grid)
from _baseline_dsabre_ext import dSABRE_BurstExt as _BaselineRouter
from layout import sabre_locked_boundary_layout

_HW_SMALL = HardwareConfig()
_HW_LARGE = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                           max_iterations=20000)
_HW_XL = HardwareConfig(deadlock_limit=200, max_backup_attempts=200,
                        max_iterations=50000)          # bench_large.py's config

# The builder is kept rather than a built instance so the fast side can be
# given a HierarchicalArchitecture (item 4) while the baseline keeps the dense
# phys_dist -- routing over both must agree.
SUITES = {
    "25q": dict(circuit_dir=os.path.expanduser("~/Documents/telesabre/circuits/qasm_25"),
                suffix="_nativegates_ibm_qiskit_opt3_25.qasm",
                builder=build_b_grid_architecture, kw=dict(r=2, s=2, m=4),
                hw=_HW_SMALL),
    "36q": dict(circuit_dir=os.path.expanduser("~/Documents/telesabre/circuits/qasm_36"),
                suffix="_nativegates_ibm_qiskit_opt3_36.qasm",
                builder=build_b_grid_architecture, kw=dict(r=2, s=2, m=4),
                hw=_HW_LARGE),
    "64q": dict(circuit_dir=os.path.expanduser("~/Documents/telesabre/circuits/qasm_64"),
                suffix="_nativegates_ibm_qiskit_opt3_64.qasm",
                builder=build_h_grid_architecture, kw=dict(r=2, s=3, m=4),
                hw=_HW_LARGE),
    # 360 logical qubits on 486 physical, K=6, M=81 -- the regime where the
    # M factor in the intra bound (item 3) and the Theta(P^2) table (item 4)
    # actually bite.  bench_large.py's config, verbatim.
    "360q": dict(circuit_dir=os.path.expanduser("~/Documents/telesabre/circuits/qasm_360"),
                 suffix="_nativegates_ibm_qiskit_opt3_360.qasm",
                 builder=build_h_grid_architecture, kw=dict(r=2, s=3, m=9),
                 hw=_HW_XL),
}

# Mirrors benchmark.py: qasm_64/ is a shared directory and carries extra files.
CANONICAL_CIRCUITS = {
    "64q": {"ae", "ghz", "graphstate", "qft", "qnn", "random",
            "qpeexact", "qaoa", "multiplier"},
}

METRIC_KEYS = ("eprs", "ls", "teles", "catcomms", "cost", "1q_gates", "aborted",
               "backup_activations", "force_make_room", "relay_hops")


def load_qasm(path):
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def compare(mA, layA, mB, layB, label, diffs):
    """Compare one route() result pair; append human-readable diffs."""
    for k in METRIC_KEYS:
        a, b = mA.get(k), mB.get(k)
        if a != b:
            diffs.append(f"{label}: metric {k}: base={a!r} fast={b!r}")
    # failure_log entries are (reason, iteration, remaining, elapsed).  The
    # last field is wall-clock seconds -- a measurement, not a routing
    # decision, and necessarily different when one router is faster.  Compare
    # the decision fields only.
    def _fl(m):
        return [tuple(e[:3]) for e in (m.get("failure_log") or [])]
    if _fl(mA) != _fl(mB):
        diffs.append(f"{label}: failure_log differs: "
                     f"base={_fl(mA)} fast={_fl(mB)}")
    if layA != layB:
        n_bad = sum(1 for q in layA if layA.get(q) != layB.get(q))
        diffs.append(f"{label}: final layout differs on {n_bad} qubit(s)")
    tA, tB = mA.get("trace"), mB.get("trace")
    if tA is not None and tB is not None:
        if len(tA) != len(tB):
            diffs.append(f"{label}: trace length {len(tA)} vs {len(tB)}")
        else:
            for i, (x, y) in enumerate(zip(tA, tB)):
                if x != y:
                    diffs.append(f"{label}: trace diverges at op {i}: {x} vs {y}")
                    break


def three_passes(router, dag, rev_dag, layout):
    """fwd -> bwd -> fwd, returning each pass's (metrics, final layout).

    Mirrors `layout.run_sabre_passes` but keeps every pass so the comparison
    sees the intermediate states too, not only the reported winner.
    """
    out = []
    m1, fwd_final = router.route(dag, layout)
    out.append((m1, fwd_final))
    if m1["aborted"]:
        return out
    m2, rev_final = router.route(rev_dag, fwd_final)
    out.append((m2, rev_final))
    layout3 = fwd_final if m2["aborted"] else rev_final
    m3, f3 = router.route(dag, layout3)
    out.append((m3, f3))
    return out


def verify_suite(name, spec, trace=False, hier=True, seeds=None):
    hw = spec["hw"]
    base_builder = _base_b_grid if spec["builder"] is build_b_grid_architecture else _base_h_grid
    arch = base_builder(**spec["kw"])          # frozen: dense phys_dist
    arch_fast = spec["builder"](**spec["kw"]) if hier else arch
    suffix = spec["suffix"]
    files = sorted(glob.glob(os.path.join(spec["circuit_dir"], "*.qasm")))
    canon = CANONICAL_CIRCUITS.get(name)
    if canon:
        files = [f for f in files
                 if os.path.basename(f).replace(suffix, "") in canon]

    cfg = HardwareConfig(**{**hw.__dict__, "trace_routing": trace})
    base = _BaselineRouter(arch, cfg)
    fast = dSABRE_BFSExt(arch_fast, cfg)

    print(f"\n{'='*78}\n  {name}   ({len(files)} circuits, trace={trace}, "
          f"hierarchical_dphys={hier})\n{'='*78}", flush=True)
    print(f"{'circuit':<14} {'cx':>6} {'passes':>7} {'t_base':>8} {'t_fast':>8} "
          f"{'speedup':>8}  {'verdict':>10}", flush=True)

    all_diffs, tot_b, tot_f = [], 0.0, 0.0
    for qf in files:
        cname = os.path.basename(qf).replace(suffix, "")
        qc, dag = load_qasm(qf)
        rev_dag = circuit_to_dag(qc.reverse_ops())
        n_cx = sum(1 for _ in dag.two_qubit_ops())
        layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
        if seeds is not None:
            layouts = layouts[:seeds]

        diffs, npass = [], 0
        t0 = time.perf_counter()
        base_runs = [three_passes(base, dag, rev_dag, lay) for lay in layouts]
        t_b = time.perf_counter() - t0
        t0 = time.perf_counter()
        fast_runs = [three_passes(fast, dag, rev_dag, lay) for lay in layouts]
        t_f = time.perf_counter() - t0

        for si, (rb, rf) in enumerate(zip(base_runs, fast_runs)):
            if len(rb) != len(rf):
                diffs.append(f"{cname} seed{si}: pass count {len(rb)} vs {len(rf)}")
                continue
            for pi, ((mb, lb), (mf, lf)) in enumerate(zip(rb, rf)):
                npass += 1
                compare(mb, lb, mf, lf, f"{cname} seed{si} pass{pi+1}", diffs)

        tot_b += t_b; tot_f += t_f
        all_diffs += diffs
        print(f"{cname:<14} {n_cx:>6} {npass:>7} {t_b:>8.2f} {t_f:>8.2f} "
              f"{t_b/max(t_f,1e-9):>7.2f}x  "
              f"{'IDENTICAL' if not diffs else str(len(diffs))+' DIFFS':>10}",
              flush=True)
        for d in diffs[:5]:
            print(f"    ! {d}", flush=True)

    print(f"{'-'*78}\n{'TOTAL':<14} {'':>6} {'':>7} {tot_b:>8.2f} {tot_f:>8.2f} "
          f"{tot_b/max(tot_f,1e-9):>7.2f}x  "
          f"{'ALL IDENTICAL' if not all_diffs else str(len(all_diffs))+' DIFFS'}",
          flush=True)
    if hier:
        P = len(arch_fast.data_qubits)
        cached = arch_fast.phys_dist.cache_size()
        print(f"  hierarchical d_phys: {cached} pairs memoised over the whole "
              f"suite, vs {P*P} in the dense table ({100*cached/(P*P):.1f}%)",
              flush=True)
    return all_diffs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default=None, choices=list(SUITES))
    ap.add_argument("--no-trace", action="store_true")
    ap.add_argument("--dense-dphys", action="store_true",
                    help="give the fast router the dense phys_dist too, "
                         "isolating the router changes from item 4")
    ap.add_argument("--seeds", type=int, default=None,
                    help="cap SabreLayout seeds per circuit (identity is a "
                         "property of the router, so fewer seeds still test "
                         "it -- this only trades breadth for wall time)")
    args = ap.parse_args()

    # 360q at 3 seeds x 3 passes is hours of baseline; 1 seed still gives 3
    # independent pass comparisons per circuit.
    default_seeds = {"360q": 1}

    names = [args.suite] if args.suite else list(SUITES)
    failures = []
    for nm in names:
        # Trace comparison is the strictest check; run it where it is cheap.
        failures += verify_suite(nm, SUITES[nm],
                                 trace=(nm == "25q" and not args.no_trace),
                                 hier=not args.dense_dphys,
                                 seeds=args.seeds or default_seeds.get(nm))

    print()
    if failures:
        print(f"FAILED: {len(failures)} difference(s)")
        for d in failures[:40]:
            print("  " + d)
        sys.exit(1)
    print("PASS: the default router is output-identical to the frozen baseline "
          "on every pass of every seed of every circuit checked.")
