"""
Compare router.py (General_dSABRE_Router) vs router_test.py (LightDAG variant)
on the 36q benchmark circuits, B_grid_2_2_4_4 device.

For each circuit, both routers run on the same set of layouts (drawn once),
so any difference in EPR cost is purely algorithmic, not layout-luck.

Columns:
  cx     — 2q gate count
  orig   — EPR cost, router.py  (best over layouts)
  test   — EPR cost, router_test.py (same layouts, same seeds)
  diff   — test - orig  (negative = test is cheaper)
  %      — relative change  ((test-orig)/orig * 100)
  t_orig — router.py wall time (s, summed over all layout trials)
  t_test — router_test.py wall time (s)
  speedup— t_orig / t_test
"""

import sys, os, glob, random, time
sys.setrecursionlimit(50000)
sys.path.insert(0, os.path.dirname(__file__))

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import RemoveBarriers

from architecture import build_b_grid_architecture
from config import HardwareConfig
from main import locality_aware_layout, sabre_locked_boundary_layout, _run_passes

import router      as router_orig_mod
import router_test as router_test_mod

CIRCUIT_DIR = os.path.expanduser("~/Documents/telesabre/circuits/qasm_36")
NUM_TRIALS  = 3
LAYOUT_PASSES = 2

arch = build_b_grid_architecture(r=2, s=2, m=4)
hw   = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                      max_iterations=20000)

router_orig = router_orig_mod.General_dSABRE_Router(arch, hw)
router_test = router_test_mod.General_dSABRE_Router(arch, hw)


def collect_layouts(qc, dag):
    """Return a deterministic list of layouts used by run_36q_bench."""
    layouts = []
    # A: random seeds 100..102
    for t in range(NUM_TRIALS):
        layouts.append(locality_aware_layout(dag, arch, rng=random.Random(t + 100)))
    # B: SABRE-lock
    try:
        cands, _ = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
        layouts.extend(cands)
    except (Exception, RecursionError):
        pass
    # C: locality seeds 0..2
    for t in range(NUM_TRIALS):
        layouts.append(locality_aware_layout(dag, arch, rng=random.Random(t)))
    return layouts


def run_router(router, dag, layouts):
    best_epr = None
    total_t  = 0.0
    for ll in layouts:
        t0 = time.perf_counter()
        m  = _run_passes(router, dag, ll, LAYOUT_PASSES)
        total_t += time.perf_counter() - t0
        if m and not m.get("aborted"):
            if best_epr is None or m["eprs"] < best_epr:
                best_epr = m["eprs"]
    return best_epr, total_t


def main():
    qasm_files = sorted(glob.glob(os.path.join(CIRCUIT_DIR, "*.qasm")))
    if not qasm_files:
        print(f"No .qasm files in {CIRCUIT_DIR}"); return

    hdr = f"{'circuit':<12}  {'cx':>5}  {'orig':>5}  {'test':>5}  {'diff':>5}  {'%':>7}  {'t_orig':>7}  {'t_test':>7}  {'speedup':>7}"
    print(hdr)
    print("-" * len(hdr))

    sum_orig = sum_test = 0
    t_orig_total = t_test_total = 0.0

    for qf in qasm_files:
        cname = os.path.basename(qf).replace("_nativegates_ibm_qiskit_opt3_36.qasm", "")

        qc  = QuantumCircuit.from_qasm_file(qf)
        qc  = qc.remove_final_measurements(inplace=False)
        qc  = PassManager([RemoveBarriers()]).run(qc)
        dag = circuit_to_dag(qc)
        cx  = sum(1 for _ in dag.two_qubit_ops())

        layouts = collect_layouts(qc, dag)

        epr_orig, t_orig = run_router(router_orig, dag, layouts)
        epr_test, t_test = run_router(router_test, dag, layouts)

        def fmt(v):   return str(v)   if v is not None else "---"
        def fmtt(v):  return f"{v:.2f}"
        def fmtpct(a, b):
            if a is None or b is None or b == 0: return "    ---"
            return f"{100*(a-b)/b:+.1f}%"
        def fmtspd(a, b):
            if b == 0: return "    ---"
            return f"{a/b:.2f}x"

        diff = (epr_test - epr_orig) if (epr_orig is not None and epr_test is not None) else None
        print(f"{cname:<12}  {cx:>5}  {fmt(epr_orig):>5}  {fmt(epr_test):>5}  "
              f"{fmt(diff):>5}  {fmtpct(epr_test, epr_orig):>7}  "
              f"{fmtt(t_orig):>7}  {fmtt(t_test):>7}  {fmtspd(t_orig, t_test):>7}")

        if epr_orig is not None: sum_orig += epr_orig
        if epr_test is not None: sum_test += epr_test
        t_orig_total += t_orig
        t_test_total += t_test

    print("-" * len(hdr))
    pct_total = f"{100*(sum_test-sum_orig)/sum_orig:+.1f}%" if sum_orig else "---"
    spd_total = f"{t_orig_total/t_test_total:.2f}x" if t_test_total else "---"
    print(f"{'TOTAL':<12}  {'':>5}  {sum_orig:>5}  {sum_test:>5}  "
          f"{sum_test-sum_orig:>5}  {pct_total:>7}  "
          f"{t_orig_total:.2f}s   {t_test_total:.2f}s  {spd_total:>7}")


if __name__ == "__main__":
    main()
