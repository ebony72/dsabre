"""
test_cost_score_64q.py — dSE-only run on the 64q suite (excluding multiplier)
to evaluate the cost-weighted teleport score (router.py `_generate_candidates`,
d_prep now priced at cfg.cost_local_swap instead of counted at weight 1).

Same protocol as benchmark.py's 64q suite (SabreLayout x3 seeds, fwd->bwd->fwd,
best of pass1/pass3, best layout by EPR), but dSE only and skipping TeleSABRE
(not needed to evaluate a router-internal scoring change). multiplier is
excluded per instruction (its EPR count dwarfs the rest of the suite -- see
CLAUDE.md -- and would swamp a gmean).

Writes results/test_cost_score_64q.json; does NOT touch results_64q.json,
which feeds the paper's tables.
"""
import glob
import json
import os
import sys
import time
import warnings
from math import prod

warnings.filterwarnings("ignore")
sys.setrecursionlimit(50000)
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from architecture import build_h_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes

CIRC_DIR = os.path.expanduser("~/Documents/telesabre/circuits/qasm_64")
SUFFIX = "_nativegates_ibm_qiskit_opt3_64.qasm"
OUT = os.path.join(_HERE, "results", "test_cost_score_64q.json")

CANONICAL_CIRCUITS_64Q = {"ae", "ghz", "graphstate", "qft", "qnn", "random",
                          "qpeexact", "qaoa"}  # multiplier excluded

_HW = HardwareConfig(deadlock_limit=100, max_backup_attempts=100, max_iterations=20000)


def load_qasm(path):
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def main():
    arch = build_h_grid_architecture(r=2, s=3, m=4)
    router = dSABRE_BurstExt(arch, _HW)

    qasm_files = sorted(glob.glob(os.path.join(CIRC_DIR, "*.qasm")))
    qasm_files = [f for f in qasm_files
                  if os.path.basename(f).replace(SUFFIX, "") in CANONICAL_CIRCUITS_64Q]

    baseline = {}
    baseline_path = os.path.join(_HERE, "results", "results_64q.json")
    if os.path.exists(baseline_path):
        d = json.load(open(baseline_path))
        for r in d["results"]:
            dse = r["routers"].get("dSE", {})
            if not dse.get("aborted"):
                baseline[r["circuit"]] = dict(eprs=dse.get("eprs"), ls=dse.get("ls"))

    col_w = 12
    hdr = (f"{'circuit':<{col_w}}  {'q':>3}  {'cx':>5}  {'epr':>6}  {'ls':>6}"
           f"  {'t':>6}  {'base_epr':>8}  {'base_ls':>7}  {'epr_d%':>7}  {'ls_d%':>7}")
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    records = []
    for qf in qasm_files:
        cname = os.path.basename(qf).replace(SUFFIX, "")
        qc, dag = load_qasm(qf)
        rev_dag = circuit_to_dag(qc.reverse_ops())
        n_qubits = qc.num_qubits
        n_cx = sum(1 for _ in dag.two_qubit_ops())

        sl_layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)

        best_m = None
        t0 = time.perf_counter()
        for layout in sl_layouts:
            m = run_sabre_passes(router, dag, rev_dag, layout)
            if m and not m.get("aborted"):
                if best_m is None or m["eprs"] < best_m["eprs"]:
                    best_m = m
        elapsed = time.perf_counter() - t0

        b = baseline.get(cname, {})

        def pct(new, old):
            if new is None or old is None or old == 0:
                return None
            return 100 * (new - old) / old

        if best_m is not None:
            epr, ls = best_m["eprs"], best_m["ls"]
            aborted = False
        else:
            epr, ls = None, None
            aborted = True

        epr_d = pct(epr, b.get("eprs"))
        ls_d = pct(ls, b.get("ls"))

        def fi(v): return str(v) if v is not None else "---"
        def ff(v): return f"{v:.2f}" if v is not None else "  ---"
        def fp(v): return f"{v:+.1f}%" if v is not None else "   ---"

        row = (f"{cname:<{col_w}}  {n_qubits:>3}  {n_cx:>5}  {fi(epr):>6}  {fi(ls):>6}"
               f"  {ff(elapsed):>6}  {fi(b.get('eprs')):>8}  {fi(b.get('ls')):>7}"
               f"  {fp(epr_d):>7}  {fp(ls_d):>7}")
        print(row, flush=True)

        records.append(dict(
            circuit=cname, qubits=n_qubits, cx=n_cx,
            eprs=epr, ls=ls, time_s=round(elapsed, 3), aborted=aborted,
            baseline_eprs=b.get("eprs"), baseline_ls=b.get("ls"),
            epr_pct_change=epr_d, ls_pct_change=ls_d,
        ))

        with open(OUT, "w") as f:
            json.dump(dict(meta=dict(
                date=time.strftime("%Y-%m-%d"),
                suite="64q", excluded="multiplier",
                router="dSE (dSABRE_BurstExt)",
                note="score = cost_local_swap * d_prep + cap - hop_gain - dF "
                     "- weight_extended * dE (cost-weighted d_prep, c_tele omitted "
                     "as a per-candidate constant)",
            ), results=records), f, indent=2)

    def gmean(lst):
        lst = [x for x in lst if x is not None and x > 0]
        return prod(lst) ** (1 / len(lst)) if lst else float("nan")

    print("-" * len(hdr), flush=True)
    eprs = [r["eprs"] for r in records if not r["aborted"]]
    lss = [r["ls"] for r in records if not r["aborted"]]
    b_eprs = [r["baseline_eprs"] for r in records if r["baseline_eprs"] is not None]
    b_lss = [r["baseline_ls"] for r in records if r["baseline_ls"] is not None]
    print(f"gmean epr: {gmean(eprs):.1f}  (baseline {gmean(b_eprs):.1f}, "
          f"{100*(gmean(eprs)-gmean(b_eprs))/gmean(b_eprs):+.1f}%)", flush=True)
    print(f"gmean ls:  {gmean(lss):.1f}  (baseline {gmean(b_lss):.1f}, "
          f"{100*(gmean(lss)-gmean(b_lss))/gmean(b_lss):+.1f}%)", flush=True)
    print(f"\nSaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
