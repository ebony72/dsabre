"""probe_derived_deadlock.py -- can L_deadlock stop being hand-tuned?

Two questions, one job, because both need the same per-seed dSE runs:

1.  **Item 2 of the 0811 review.**  Table I's footnote called
    `L_deadlock` "the only per-suite parameter" (50 / 100 / 200 by suite).
    Arm A reruns each suite at its published hand-tuned value, arm B at a
    derived rule (`--rule arch|const`, see `bench`); if the two agree the
    paper can drop the footnote and claim zero per-suite parameters.

    **Result:** `--rule arch` agrees exactly on all eight suites, best and
    per-seed median alike, so it is what `deadlock_limit_for` and
    `deadlock_limit=None` now are.  `--rule const` (L=10) agrees through
    200 qubits and then loses the reported seed on `qpeexact_360`
    (1938 -> 2735 EPR), so it was rejected.

    Arm B also derives the other two recovery budgets, so that arm has NO
    hand-set constant at all:
      * `max_backup_attempts` -- route() already raises it to the unrouted
        gate count in safe mode, so it is derived already;
      * `max_iterations` -- set to `iterations_bound`, which is the worst case
        the theorem allows.  In safe mode exceeding it is not an abort (the
        route switches to draining with the guaranteed transaction), so the
        published 10000/20000/50000 are an effort knob rather than a
        correctness one -- but at the bound they cannot bind at all.

2.  **Item 6.**  The paper bounds the best-of-3 vs deterministic-TeleSABRE
    asymmetry only at 64q.  Recording every seed rather than the winner gives
    the median-over-seeds gap on all six suites.

TeleSABRE is not re-run: `L_deadlock` is a dSABRE parameter, so its column is
read from the published results JSONs.

Usage:
    python3 probe_derived_deadlock.py                 # all suites
    python3 probe_derived_deadlock.py --suite 64q     # one suite
"""

import sys, os, json, time, argparse
from math import prod
from statistics import median

sys.setrecursionlimit(100000)
_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # code/, one level up
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler import PassManager

from architecture import (build_b_grid_architecture, build_h_grid_architecture,
                          build_heavy_hex_architecture)
from config import HardwareConfig
from router import General_dSABRE_Router
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes

_RESULTS = os.path.join(_HERE, "results")
OUT = os.path.join(_RESULTS, "results_derived_deadlock.json")

# (circuit_dir, suffix, arch, published deadlock_limit, published max_iterations,
#  circuit whitelist or None)
SUITES = {
    "25q": dict(
        circuit_dir="~/Documents/telesabre/circuits/qasm_25",
        suffix="_nativegates_ibm_qiskit_opt3_25.qasm",
        arch=lambda: build_b_grid_architecture(r=2, s=2, m=4),
        pub_L=50, pub_iters=10000, pub_backups=50, circuits=None,
    ),
    "36q": dict(
        circuit_dir="~/Documents/telesabre/circuits/qasm_36",
        suffix="_nativegates_ibm_qiskit_opt3_36.qasm",
        arch=lambda: build_b_grid_architecture(r=2, s=2, m=4),
        pub_L=100, pub_iters=20000, pub_backups=100, circuits=None,
    ),
    "64q": dict(
        circuit_dir="~/Documents/telesabre/circuits/qasm_64",
        suffix="_nativegates_ibm_qiskit_opt3_64.qasm",
        arch=lambda: build_h_grid_architecture(r=2, s=3, m=4),
        pub_L=100, pub_iters=20000, pub_backups=100,
        circuits=["ae", "ghz", "graphstate", "qft", "qnn", "random",
                  "qpeexact", "qaoa", "multiplier"],
    ),
    "100q": dict(
        circuit_dir="~/Documents/telesabre/circuits/qasm_100",
        suffix="_nativegates_ibm_qiskit_opt3_100.qasm",
        arch=lambda: build_h_grid_architecture(r=2, s=3, m=5),
        pub_L=200, pub_iters=50000, pub_backups=200,
        circuits=["qft", "qpeexact"],
    ),
    "200q": dict(
        circuit_dir="~/Documents/telesabre/circuits/qasm_200",
        suffix="_nativegates_ibm_qiskit_opt3_200.qasm",
        arch=lambda: build_h_grid_architecture(r=3, s=4, m=5),
        pub_L=200, pub_iters=50000, pub_backups=200,
        circuits=["qft", "qpeexact"],
    ),
    "360q": dict(
        circuit_dir="~/Documents/telesabre/circuits/qasm_360",
        suffix="_nativegates_ibm_qiskit_opt3_360.qasm",
        arch=lambda: build_h_grid_architecture(r=4, s=5, m=5),
        pub_L=200, pub_iters=50000, pub_backups=200,
        circuits=["qft", "qpeexact"],
    ),
    # The architecture-independence suites (bench_heavyhex.py): non-grid at
    # both levels, so they are where a derived rule reading only the core and
    # core-graph diameters would be most likely to come apart from the
    # hand-tuned value.
    "hhex_ring": dict(
        circuit_dir="~/Documents/telesabre/circuits/qasm_64",
        suffix="_nativegates_ibm_qiskit_opt3_64.qasm",
        arch=lambda: build_heavy_hex_architecture(4, "ring"),
        pub_L=100, pub_iters=20000, pub_backups=100,
        circuits=["ae", "ghz", "graphstate", "qft", "qnn", "random"],
    ),
    "hhex_star": dict(
        circuit_dir="~/Documents/telesabre/circuits/qasm_64",
        suffix="_nativegates_ibm_qiskit_opt3_64.qasm",
        arch=lambda: build_heavy_hex_architecture(4, "star"),
        pub_L=100, pub_iters=20000, pub_backups=100,
        circuits=["ae", "ghz", "graphstate", "qft", "qnn", "random"],
    ),
}


def load_qasm(path):
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def gmean(xs):
    xs = [x for x in xs if x is not None and x > 0]
    return prod(xs) ** (1 / len(xs)) if xs else float("nan")


def run_arm(arch, hw, dag, rev_dag, layouts):
    """dSE over every SabreLayout seed; per-seed metrics, not just the winner."""
    router = dSABRE_BurstExt(arch, hw)
    per_seed = []
    for layout in layouts:
        m = run_sabre_passes(router, dag, rev_dag, layout)
        if m is None or m.get("aborted"):
            per_seed.append(None)
        else:
            per_seed.append(dict(
                eprs=m["eprs"], ls=m["ls"],
                iterations=m.get("iterations", 0),
                safe_routes=m.get("safe_routes", 0),
                safe_route_failed=m.get("safe_route_failed", 0),
                force_make_room=m.get("force_make_room", 0),
                compile_time=round(m.get("compile_time", 0.0), 3),
            ))
    ok = [s for s in per_seed if s]
    return dict(
        per_seed=per_seed,
        best=min((s["eprs"] for s in ok), default=None),
        median=median(sorted(s["eprs"] for s in ok)) if ok else None,
        aborts=sum(1 for s in per_seed if s is None),
        iterations=sum(s["iterations"] for s in ok),
        safe_routes=sum(s["safe_routes"] for s in ok),
        safe_route_failed=sum(s["safe_route_failed"] for s in ok),
        force_make_room=sum(s["force_make_room"] for s in ok),
        time_s=round(sum(s["compile_time"] for s in ok), 3),
    )


def bench(suite_name, spec, rule="const"):
    arch = spec["arch"]()
    cdir = os.path.expanduser(spec["circuit_dir"])
    suffix = spec["suffix"]
    names = spec["circuits"]
    if names is None:
        import glob
        names = sorted(os.path.basename(f).replace(suffix, "")
                       for f in glob.glob(os.path.join(cdir, "*.qasm")))

    # Two candidate derived rules, both free of per-suite tuning:
    #   "arch"   L = 4 * diam(core graph) * (diam(core) + 1) -- what
    #            `deadlock_limit_for` now is, and what `deadlock_limit=None`
    #            selects.  Reproduces the hand-tuned magnitudes
    #            (56/56/84/104/108/180/252 against 50/100/200).
    #   "const"  L = 10 flat, on the argument that in safe mode the stall
    #            window is pure waste (the checkpoint restore discards it).
    #            MEASURED AND REJECTED: EPR-identical through 200 qubits, but
    #            it loses the best-of-three seed on qpeexact_360.
    CONST_RULE_L = 10
    arch_L = General_dSABRE_Router.deadlock_limit_for(arch)
    derived_L = CONST_RULE_L if rule == "const" else arch_L
    other_L = arch_L if rule == "const" else CONST_RULE_L
    print(f"\n=== {suite_name}: published L={spec['pub_L']}, "
          f"derived L={derived_L} (rule={rule}; the other rule gives {other_L}) ===",
          flush=True)
    print(f"{'circuit':<12} {'cx':>6} | {'A best':>7} {'A med':>7} "
          f"| {'B best':>7} {'B med':>7} | {'dBest%':>7} {'A iters':>9} {'B iters':>9}",
          flush=True)

    rows = []
    for cname in names:
        qf = os.path.join(cdir, cname + suffix)
        if not os.path.exists(qf):
            print(f"  [missing {qf}]", flush=True)
            continue
        qc, dag = load_qasm(qf)
        rev_dag = circuit_to_dag(qc.reverse_ops())
        n_cx = sum(1 for _ in dag.two_qubit_ops())
        layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)

        hw_a = HardwareConfig(deadlock_limit=spec["pub_L"],
                              max_backup_attempts=spec["pub_backups"],
                              max_iterations=spec["pub_iters"])
        # Arm B: every recovery budget derived.  max_iterations at the
        # worst-case bound of Eq. iterations_bound, so it cannot bind.
        bound = General_dSABRE_Router.iterations_bound(n_cx, derived_L)
        hw_b = HardwareConfig(deadlock_limit=derived_L,
                              max_backup_attempts=n_cx + 1,
                              max_iterations=bound)

        a = run_arm(arch, hw_a, dag, rev_dag, layouts)
        b = run_arm(arch, hw_b, dag, rev_dag, layouts)
        d = (100.0 * (b["best"] - a["best"]) / a["best"]
             if a["best"] and b["best"] else None)
        print(f"{cname:<12} {n_cx:>6} | {str(a['best']):>7} {str(a['median']):>7} "
              f"| {str(b['best']):>7} {str(b['median']):>7} "
              f"| {('%+.1f' % d) if d is not None else '---':>7} "
              f"{a['iterations']:>9} {b['iterations']:>9}", flush=True)

        rows.append(dict(suite=suite_name, circuit=cname, cx=n_cx,
                         derived_L=derived_L, published_L=spec["pub_L"],
                         iterations_bound=bound, arm_published=a, arm_derived=b))
        _BY_SUITE[suite_name] = rows
        _save_partial()

    if rows:
        ab, bb = [r["arm_published"]["best"] for r in rows], [r["arm_derived"]["best"] for r in rows]
        am, bm = [r["arm_published"]["median"] for r in rows], [r["arm_derived"]["median"] for r in rows]
        print(f"{'gmean':<12} {'':>6} | {gmean(ab):>7.1f} {gmean(am):>7.1f} "
              f"| {gmean(bb):>7.1f} {gmean(bm):>7.1f} "
              f"| {100*(gmean(bb)/gmean(ab)-1):>+6.1f}%", flush=True)
        print(f"{'total':<12} {'':>6} | {sum(ab):>7} {'':>7} | {sum(bb):>7}", flush=True)
    return rows


_BY_SUITE = {}


def _save_partial():
    """Write after every circuit -- a killed run is still usable."""
    merged = [r for rows in _BY_SUITE.values() for r in rows]
    with open(OUT, "w") as f:
        json.dump({"meta": {"date": time.strftime("%Y-%m-%d"),
                            "what": "hand-tuned vs derived L_deadlock, per SabreLayout seed",
                            "router": "dSE (dSABRE_BurstExt), safe_mode default"},
                   "results": merged}, f, indent=1)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", action="append", choices=list(SUITES))
    ap.add_argument("--rule", default="const", choices=["const", "arch"])
    ap.add_argument("--out")
    args = ap.parse_args()
    if args.out:
        OUT = os.path.join(_RESULTS, args.out)
    todo = args.suite or list(SUITES)
    for name in todo:
        bench(name, SUITES[name], rule=args.rule)
        _save_partial()
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
