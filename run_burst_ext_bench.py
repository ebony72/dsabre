"""
Compare four routers under TeleSABRE's initial layout (fair comparison):
  TS    — TeleSABRE (compiled C++ binary)
  dS    — vanilla dSABRE
  dSE   — dSABRE_BurstExt (wire-balanced extended set, Option A)
  Bu    — BurstDSABRE (full ECH + burst scoring)

Runs both 25q (B-grid 2x2 4x4) and 36q (B-grid 2x2 4x4) suites.
"""

import sys, os, json, glob, subprocess, tempfile, time

sys.setrecursionlimit(50000)
sys.path.insert(0, os.path.dirname(__file__))

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler import PassManager
from architecture import build_b_grid_architecture
from config import HardwareConfig
from router import General_dSABRE_Router
from burst_router import BurstDSABRE
from burst_ext_router import dSABRE_BurstExt
from main import _run_passes

TS_BIN  = os.path.expanduser("~/Documents/telesabre/telesabre")
TS_DEV  = os.path.expanduser("~/Documents/telesabre/devices/B_grid_2_2_4_4.json")

LAYOUT_PASSES = 2
NUM_TS_SEEDS  = 3

arch     = build_b_grid_architecture(r=2, s=2, m=4)
hw_25    = HardwareConfig()  # defaults, matches run_fair_comparison.py
hw_36    = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                          max_iterations=20000, max_burst_walk_depth=25)

dsabre_25  = General_dSABRE_Router(arch, hw_25)
dsabreE_25 = dSABRE_BurstExt(arch, hw_25)
burst_25   = BurstDSABRE(arch, hw_25, weight_burst=2.0, max_burst_normaliser=8)

dsabre_36  = General_dSABRE_Router(arch, hw_36)
dsabreE_36 = dSABRE_BurstExt(arch, hw_36)
burst_36   = BurstDSABRE(arch, hw_36, weight_burst=2.0, max_burst_normaliser=8)


def make_ts_config(seed: int, report_path: str) -> str:
    cfg = {"config": {
        "name": "ext_bench", "seed": seed,
        "energy_type": "extended-set",
        "usage_penalties_reset_interval": 5,
        "optimize_initial": True,
        "initial_layout_type": "hungarian",
        "teleport_bonus": 100, "telegate_bonus": 100,
        "safety_valve_iters": 100,
        "extended_set_size": 20, "extended_set_factor": 0.05,
        "inter_core_edge_weight": 2, "full_core_penalty": 10,
        "max_solving_deadlock_iterations": 1000,
        "gate_usage_penalty": 0.0, "swap_usage_penalty": 0.002,
        "teledata_usage_penaly": 0.005, "telegate_usage_penalty": 0.005,
        "init_layout_hun_min_free_gate": 5, "init_layout_hun_min_free_qubit": 4,
        "enable_passing_core_emptying_teleport_possibility": False,
        "max_iterations": 200000,
        "save_report": True, "report_filename": report_path,
        "required_successes": 1, "max_attempts": 10,
    }}
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, f); f.close()
    return f.name


def run_telesabre(qasm_path, seed):
    report_tmp = tempfile.mktemp(suffix=".json")
    cfg_path   = make_ts_config(seed, report_tmp)
    try:
        proc = subprocess.run([TS_BIN, cfg_path, TS_DEV, qasm_path],
                              capture_output=True, text=True, timeout=300)
        out = proc.stdout + proc.stderr
        td = tg = 0; ok = False
        for line in out.splitlines():
            s = line.strip()
            if "Teledata:" in s:   td = int(s.split(":")[1])
            elif "Telegate:" in s: tg = int(s.split(":")[1])
            elif "Success: true" in s: ok = True
        p2v = None
        if os.path.exists(report_tmp):
            with open(report_tmp) as rf:
                rep = json.load(rf)
            if rep.get("iterations"):
                p2v = rep["iterations"][0]["phys_to_virt"]
        return td, tg, p2v, ok
    finally:
        os.unlink(cfg_path)
        if os.path.exists(report_tmp): os.unlink(report_tmp)


def p2v_to_layout(p2v, dag):
    qubits = dag.qubits
    return {qubits[v]: p for p, v in enumerate(p2v)
            if v != -1 and v < len(qubits)}


def load_qasm(qf):
    qc = QuantumCircuit.from_qasm_file(qf)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def bench_suite(suite_name: str, circuit_dir: str, suffix: str,
                dsabre, dsabreE, burst):
    print(f"\n=== {suite_name} ===")
    qasm_files = sorted(glob.glob(os.path.join(circuit_dir, "*.qasm")))
    if not qasm_files:
        print(f"  no .qasm in {circuit_dir}"); return

    hdr = (f"{'circuit':<12}  {'cx':>5}  {'TS':>5}  "
           f"{'dS†':>5}  {'dSE†':>5}  {'Bu†':>5}  "
           f"{'dSE/dS%':>8}  {'Bu/dSE%':>8}")
    print(hdr); print("-" * len(hdr))

    for qf in qasm_files:
        cname = os.path.basename(qf).replace(suffix, "")
        t0 = time.time()
        qc, dag = load_qasm(qf)
        cx = sum(1 for _ in dag.two_qubit_ops())

        # TeleSABRE — capture best layout
        best_ts = best_p2v = None
        for seed in range(NUM_TS_SEEDS):
            td, tg, p2v, ok = run_telesabre(qf, seed)
            if ok:
                epr = td + tg  # 1 EPR pair per teledata; 1 EPR pair per telegate (cat-entangler protocol)
                if best_ts is None or epr < best_ts:
                    best_ts, best_p2v = epr, p2v

        # Run all 3 dSABRE-family routers under TS layout
        ds = dse = bu = None
        if best_p2v is not None:
            layout = p2v_to_layout(best_p2v, dag)
            if len(layout) >= qc.num_qubits:
                md  = _run_passes(dsabre,  dag, layout, LAYOUT_PASSES)
                mde = _run_passes(dsabreE, dag, layout, LAYOUT_PASSES)
                mb  = _run_passes(burst,   dag, layout, LAYOUT_PASSES)
                if md  and not md.get("aborted"):  ds  = md["eprs"]
                if mde and not mde.get("aborted"): dse = mde["eprs"]
                if mb  and not mb.get("aborted"):  bu  = mb["eprs"]

        def fmt(v): return str(v) if v is not None else "---"
        def pct(a, b):
            if a is None or b is None or b == 0: return "    ---"
            return f"{100*(a-b)/b:+.1f}%"

        elapsed = time.time() - t0
        print(f"{cname:<12}  {cx:>5}  {fmt(best_ts):>5}  "
              f"{fmt(ds):>5}  {fmt(dse):>5}  {fmt(bu):>5}  "
              f"{pct(dse, ds):>8}  {pct(bu, dse):>8}  ({elapsed:.0f}s)")


def main():
    bench_suite("25q suite",
                os.path.expanduser("~/Documents/telesabre/circuits/qasm_25"),
                "_nativegates_ibm_qiskit_opt3_25.qasm",
                dsabre_25, dsabreE_25, burst_25)
    bench_suite("36q suite",
                os.path.expanduser("~/Documents/telesabre/circuits/qasm_36"),
                "_nativegates_ibm_qiskit_opt3_36.qasm",
                dsabre_36, dsabreE_36, burst_36)


if __name__ == "__main__":
    main()
