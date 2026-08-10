r"""
probe_compile_time_scaling.py -- total (3-seed) dSABRE compile time for
QFT at 100q and 200q, matching probe_compile_time.py's convention for the
64q suite.

bench_large.py's run_dsabre() records `time_s` as the elapsed time of the
single *winning* SabreLayout seed, not the cost of the whole best-of-3
protocol that produced it -- the same understatement probe_compile_time.py
found and fixed for the 64q suite (see its docstring).  This script
re-times just the two circuits needed for a concrete, fair comparison
against TeleSABRE (whose reported time already sums every seed).

Output: code/results/results_compile_time_scaling.json
"""
import sys, os, json, time

sys.setrecursionlimit(100000)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from bench_large import SUITES, _HW
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes

OUT = os.path.join(_HERE, "results", "results_compile_time_scaling.json")

TARGETS = [("100q", "qft"), ("200q", "qft"), ("360q", "qft")]


def load(path):
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def main():
    rows = {}
    for suite_name, cname in TARGETS:
        s = SUITES[suite_name]
        qasm_path = os.path.join(s["circuit_dir"], f"{cname}{s['suffix']}")
        qc, dag = load(qasm_path)
        rev = circuit_to_dag(qc.reverse_ops())
        layouts = sabre_locked_boundary_layout(qc, dag, s["arch"], seed=0)
        router = dSABRE_BurstExt(s["arch"], _HW)
        per_seed, eprs, best = [], [], None
        for layout in layouts:
            t0 = time.perf_counter()
            m = run_sabre_passes(router, dag, rev, layout)
            per_seed.append(round(time.perf_counter() - t0, 2))
            if m and not m.get("aborted"):
                eprs.append(m["eprs"])
                if best is None or m["eprs"] < best:
                    best = m["eprs"]
            else:
                eprs.append(None)
        total = round(sum(per_seed), 2)
        print(f"{suite_name} {cname}: seeds={per_seed} total={total:.2f}s "
              f"best_seed={min(per_seed):.2f}s EPR={eprs}", flush=True)
        rows[f"{suite_name}_{cname}"] = dict(per_seed_s=per_seed, total_s=total,
                                              best_seed_s=min(per_seed),
                                              eprs=eprs, best_epr=best)
        json.dump(dict(meta=dict(date=time.strftime("%Y-%m-%d"),
                                  note="dSABRE total (3-seed) compile time, "
                                       "matching probe_compile_time.py's convention"),
                        results=rows), open(OUT, "w"), indent=2)
    print(f"\nSaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
