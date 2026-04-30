"""
Compilation time comparison: TeleSABRE (C++) vs dSABRE (Python) vs BurstDSABRE (Python).
Runs on 25q circuits (fast); reports wall-clock seconds per circuit per router.
Each router is run once per circuit with a fixed seed/layout for reproducibility.
"""

import sys, os, json, glob, subprocess, tempfile, time, random

sys.path.insert(0, os.path.dirname(__file__))

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from architecture import build_b_grid_architecture
from config import HardwareConfig
from router import General_dSABRE_Router
from burst_router import BurstDSABRE
from main import locality_aware_layout, _run_passes

TS_BIN      = os.path.expanduser("~/Documents/telesabre/telesabre")
TS_DEV      = os.path.expanduser("~/Documents/telesabre/devices/B_grid_2_2_4_4.json")
CIRCUIT_DIR = os.path.expanduser("~/Documents/telesabre/circuits/qasm_25")

arch   = build_b_grid_architecture(r=2, s=2, m=4)
hw     = HardwareConfig()
dsabre = General_dSABRE_Router(arch, hw)
burst  = BurstDSABRE(arch, hw, weight_burst=2.0, max_burst_normaliser=8)

LAYOUT_PASSES = 2
TS_SEED = 0


def make_ts_config(seed: int, report_path: str) -> str:
    cfg = {"config": {
        "name": "timing", "seed": seed,
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
        "save_report": False, "report_filename": report_path,
        "required_successes": 1, "max_attempts": 10,
    }}
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, f); f.close()
    return f.name


def time_telesabre(qasm_path):
    cfg_path = make_ts_config(TS_SEED, "/tmp/ts_timing_report.json")
    try:
        t0 = time.perf_counter()
        proc = subprocess.run([TS_BIN, cfg_path, TS_DEV, qasm_path],
                              capture_output=True, text=True, timeout=120)
        return time.perf_counter() - t0
    finally:
        os.unlink(cfg_path)


def time_router(router, dag, layout):
    t0 = time.perf_counter()
    _run_passes(router, dag, layout, LAYOUT_PASSES)
    return time.perf_counter() - t0


def main():
    qasm_files = sorted(glob.glob(os.path.join(CIRCUIT_DIR, "*.qasm")))
    if not qasm_files:
        print(f"No .qasm files in {CIRCUIT_DIR}"); return

    hdr = f"{'circuit':<12}  {'CX':>5}  {'TS(s)':>7}  {'dS(s)':>7}  {'Bu(s)':>7}  {'dS/TS':>7}  {'Bu/TS':>7}"
    print(hdr)
    print("-" * len(hdr))

    for qf in qasm_files:
        cname = os.path.basename(qf).replace("_nativegates_ibm_qiskit_opt3_25.qasm", "")

        qc  = QuantumCircuit.from_qasm_file(qf)
        dag = circuit_to_dag(qc)
        cx  = sum(1 for _ in dag.two_qubit_ops())

        # Fixed layout for dSABRE and BurstDSABRE (locality, seed 0)
        layout = locality_aware_layout(dag, arch, rng=random.Random(0))

        ts_t  = time_telesabre(qf)
        ds_t  = time_router(dsabre, dag, layout)
        bu_t  = time_router(burst,  dag, layout)

        print(f"{cname:<12}  {cx:>5}  {ts_t:>7.2f}  {ds_t:>7.2f}  {bu_t:>7.2f}"
              f"  {ds_t/ts_t:>6.1f}x  {bu_t/ts_t:>6.1f}x")


if __name__ == "__main__":
    main()
