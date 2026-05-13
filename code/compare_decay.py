"""
compare_decay.py — position-indexed vs DAG-depth decay on 25q/64q circuits.

Runs dSE (BFS extended set) with:
  OLD: discount = decay^i  (i = position in the extended list, 0-indexed)
  NEW: discount = decay^d  (d = actual DAG distance from the front layer)

Uses 3 SabreLayout seeds × fwd/bwd/fwd SABRE passes; picks best EPR per seed.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler import PassManager

from architecture import build_b_grid_architecture, build_h_grid_architecture
from config import HardwareConfig
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes


# ── OLD variant: replace actual depths with list-position indices ──────────────

class dSABRE_PosDec(dSABRE_BurstExt):
    """dSE with old position-indexed decay (decay^i, i = list position)."""

    def _extended_2q(self, dag, front, size):
        pairs = super()._extended_2q(dag, front, size)
        return [(g, i) for i, (g, _) in enumerate(pairs)]

    def _get_local_extended(self, wdag, front, core_ids, l2p):
        per_core = super()._get_local_extended(wdag, front, core_ids, l2p)
        return {ci: [(g, i) for i, (g, _) in enumerate(gates)]
                for ci, gates in per_core.items()}


# ── Suite definitions ──────────────────────────────────────────────────────────

SUITES = {
    "25q": dict(
        circuit_dir=os.path.expanduser("~/Documents/telesabre/circuits/qasm_25"),
        suffix="_nativegates_ibm_qiskit_opt3_25.qasm",
        arch=build_b_grid_architecture(r=2, s=2, m=4),
        hw=HardwareConfig(),
    ),
    "64q": dict(
        circuit_dir=os.path.expanduser("~/Documents/telesabre/circuits/qasm_64"),
        suffix="_nativegates_ibm_qiskit_opt3_64.qasm",
        arch=build_h_grid_architecture(r=2, s=3, m=4),
        hw=HardwareConfig(deadlock_limit=100, max_backup_attempts=100, max_iterations=20000),
    ),
}

_pm = PassManager([RemoveBarriers()])


def load_circuit(path):
    qc = QuantumCircuit.from_qasm_file(path)
    qc = _pm.run(qc)
    dag = circuit_to_dag(qc)
    rev_dag = circuit_to_dag(qc.reverse_ops())
    return qc, dag, rev_dag


def best_over_seeds(cls_instance, dag, rev_dag, qc, arch, num_seeds=3):
    layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
    best_eprs, best_ls = None, None
    for layout in layouts:
        m = run_sabre_passes(cls_instance, dag, rev_dag, layout)
        if m is None:
            continue
        if best_eprs is None or m["eprs"] < best_eprs:
            best_eprs, best_ls = m["eprs"], m["ls"]
    return best_eprs, best_ls


def bench_suite(suite_name, cfg):
    arch = cfg["arch"]
    hw   = cfg["hw"]
    cdir = cfg["circuit_dir"]
    suf  = cfg["suffix"]

    print(f"\n{'═'*68}", flush=True)
    print(f"  {suite_name}", flush=True)
    print(f"{'═'*68}", flush=True)
    print(f"  {'Circuit':<14}  {'EPR(old)':>9}  {'EPR(new)':>9}  {'ΔEPR%':>7}  {'LS(old)':>7}  {'LS(new)':>7}", flush=True)
    print(f"  {'-'*14}  {'-'*9}  {'-'*9}  {'-'*7}  {'-'*7}  {'-'*7}", flush=True)

    qasm_files = sorted(f for f in os.listdir(cdir) if f.endswith(suf))
    for fname in qasm_files:
        circ = fname.replace(suf, "")
        qc, dag, rev_dag = load_circuit(os.path.join(cdir, fname))

        r_old = dSABRE_PosDec(arch, hw)
        r_new = dSABRE_BurstExt(arch, hw)

        t0 = time.perf_counter()
        old_eprs, old_ls = best_over_seeds(r_old, dag, rev_dag, qc, arch)
        new_eprs, new_ls = best_over_seeds(r_new, dag, rev_dag, qc, arch)
        elapsed = time.perf_counter() - t0

        if old_eprs is None:
            print(f"  {circ:<14}  {'ABORT':>9}  {'ABORT':>9}", flush=True)
            continue

        pct = (new_eprs - old_eprs) / max(old_eprs, 1) * 100
        sign = "+" if pct >= 0 else ""
        print(f"  {circ:<14}  {old_eprs:>9}  {new_eprs:>9}  {sign}{pct:>6.1f}%  {old_ls:>7}  {new_ls:>7}  ({elapsed:.1f}s)", flush=True)

    print(flush=True)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--suite", choices=["25q", "64q"], default=None)
    args = p.parse_args()

    suites = [args.suite] if args.suite else list(SUITES.keys())
    for s in suites:
        bench_suite(s, SUITES[s])
    print("Done.")
