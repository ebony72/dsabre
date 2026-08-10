r"""
probe_cost_ratio_sweep.py — sweep cost_teleport (c_tele) at fixed
cost_local_swap (c_swap=3) across the three main suites, to check whether
dSABRE's margin over TeleSABRE survives a cost model that prices EPR pairs
less (or more) generously than the paper's own default (c_tele=10).

Only dSABRE is re-run: TeleSABRE's own cost model is a separate, fixed
baseline (its numbers are read from results_{suite}q.json, unchanged), so
this isolates dSABRE's robustness to the c_tele assumption rather than
re-tuning both tools together.

No producer script for this experiment survived in the repo (found while
auditing Section IV currency, 2026-08-10); this is a fresh implementation
of what dsabre.tex's sec:cost paragraph describes: "Sweeping c_tele from 10
to 100 at fixed c_swap=3 ... dSABRE saves X% at the teleport-friendly end
and Y% at the photonic-link end".

Protocol matches benchmark.py: SabreLayout corners-removed, 3 seeds,
fwd -> bwd (reversed DAG) -> fwd, best of pass 1/3, best EPR of the 3 seeds.

Output: code/results/results_cost_ratio_sweep.json
"""
import sys, os, json, time
from dataclasses import replace
from math import prod

sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from architecture import build_b_grid_architecture, build_h_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes
from benchmark import load_qasm, CANONICAL_CIRCUITS

_RESULTS_DIR = os.path.join(_HERE, "results")
OUT = os.path.join(_RESULTS_DIR, "results_cost_ratio_sweep.json")

SUITES = {
    "25q": dict(
        circuit_dir=os.path.expanduser("~/Documents/telesabre/circuits/qasm_25"),
        suffix="_nativegates_ibm_qiskit_opt3_25.qasm",
        arch=build_b_grid_architecture(r=2, s=2, m=4),
        hw=HardwareConfig(),
    ),
    "36q": dict(
        circuit_dir=os.path.expanduser("~/Documents/telesabre/circuits/qasm_36"),
        suffix="_nativegates_ibm_qiskit_opt3_36.qasm",
        arch=build_b_grid_architecture(r=2, s=2, m=4),
        hw=HardwareConfig(deadlock_limit=100, max_backup_attempts=100, max_iterations=20000),
    ),
    "64q": dict(
        circuit_dir=os.path.expanduser("~/Documents/telesabre/circuits/qasm_64"),
        suffix="_nativegates_ibm_qiskit_opt3_64.qasm",
        arch=build_h_grid_architecture(r=2, s=3, m=4),
        hw=HardwareConfig(deadlock_limit=100, max_backup_attempts=100, max_iterations=20000),
    ),
}

COST_TELE_VALUES = [10.0, 100.0]


def gmean(xs):
    xs = [x for x in xs if x is not None]
    return prod(xs) ** (1 / len(xs)) if xs else float("nan")


def main():
    payload = {"meta": {"date": time.strftime("%Y-%m-%d"),
                         "note": "dSABRE only, TS unchanged (read from results_{suite}q.json); "
                                 "cost_local_swap fixed at 3.0"},
               "results": {}}
    if os.path.exists(OUT):
        try:
            payload = json.load(open(OUT))
        except Exception:
            pass

    for suite_name, s in SUITES.items():
        canon = CANONICAL_CIRCUITS.get(suite_name)
        import glob
        qasm_files = sorted(glob.glob(os.path.join(s["circuit_dir"], "*.qasm")))
        if canon:
            qasm_files = [f for f in qasm_files
                          if os.path.basename(f).replace(s["suffix"], "") in canon]

        suite_out = payload["results"].setdefault(suite_name, {})
        print(f"\n=== {suite_name} ===", flush=True)
        for c_tele in COST_TELE_VALUES:
            key = str(c_tele)
            row = suite_out.setdefault(key, {})
            hw = replace(s["hw"], cost_teleport=c_tele, cost_local_swap=3.0)
            for qf in qasm_files:
                cname = os.path.basename(qf).replace(s["suffix"], "")
                if cname in row:
                    continue
                qc, dag = load_qasm(qf)
                rev_dag_qc = qc.reverse_ops()
                from qiskit.converters import circuit_to_dag
                rev_dag = circuit_to_dag(rev_dag_qc)
                router = dSABRE_BurstExt(s["arch"], hw)
                layouts = sabre_locked_boundary_layout(qc, dag, s["arch"], seed=0)
                best = None
                t0 = time.perf_counter()
                for layout in layouts:
                    m = run_sabre_passes(router, dag, rev_dag, layout)
                    if m and not m.get("aborted"):
                        if best is None or m["eprs"] < best["eprs"]:
                            best = m
                dt = time.perf_counter() - t0
                if best is not None:
                    row[cname] = dict(eprs=best["eprs"], ls=best["ls"])
                    print(f"  c_tele={c_tele:<6} {cname:<12} EPR={best['eprs']:<6} "
                          f"SWAP={best['ls']:<6} ({dt:.1f}s)", flush=True)
                else:
                    row[cname] = dict(aborted=True)
                    print(f"  c_tele={c_tele:<6} {cname:<12} ABORTED ({dt:.1f}s)", flush=True)
                json.dump(payload, open(OUT, "w"), indent=2)

    print(f"\nSaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
