"""
Fair comparison: run BurstDSABRE with the SAME initial qubit mapping
that TeleSABRE uses (Hungarian + optimize_initial).

For each circuit:
  1. Run TeleSABRE binary with save_report=true → capture iteration-0
     phys_to_virt as the initial layout.
  2. Convert phys_to_virt to a Qiskit-qubit → physical-qubit dict.
  3. Run BurstDSABRE with that fixed initial layout (2 FBF passes).
  4. Also run dSABRE with the same layout for a 3-way comparison.

Reports best EPR over 3 TeleSABRE seeds.
"""

import sys, os, json, glob, subprocess, tempfile, random

sys.path.insert(0, os.path.dirname(__file__))

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from architecture import build_b_grid_architecture
from config import HardwareConfig
from router import General_dSABRE_Router
from burst_router import BurstDSABRE
from main import _run_passes

# ── Paths ──────────────────────────────────────────────────────────────────
TS_BIN      = os.path.expanduser("~/Documents/telesabre/telesabre")
TS_DEV      = os.path.expanduser("~/Documents/telesabre/devices/B_grid_2_2_4_4.json")
CIRCUIT_DIR = os.path.expanduser("~/Documents/telesabre/circuits/qasm_25")
REPORT_FILE = os.path.expanduser("~/Documents/telesabre/viewer/report.json")

BURST_WEIGHT = 2.0
BURST_NORM   = 8
LAYOUT_PASSES = 2
NUM_SEEDS     = 3   # TeleSABRE seeds to try; take best BurstDSABRE result

# ── Build architecture (matches B_grid_2_2_4_4) ────────────────────────────
arch    = build_b_grid_architecture(r=2, s=2, m=4)
hw      = HardwareConfig()
dsabre  = General_dSABRE_Router(arch, hw)
burst   = BurstDSABRE(arch, hw, weight_burst=BURST_WEIGHT,
                       max_burst_normaliser=BURST_NORM)


def make_ts_config(seed: int, report_path: str) -> str:
    """Write a per-run TeleSABRE config JSON; return its path."""
    cfg = {
        "config": {
            "name": "fair_cmp",
            "seed": seed,
            "energy_type": "extended-set",
            "usage_penalties_reset_interval": 5,
            "optimize_initial": True,
            "initial_layout_type": "hungarian",
            "teleport_bonus": 100,
            "telegate_bonus": 100,
            "safety_valve_iters": 100,
            "extended_set_size": 20,
            "extended_set_factor": 0.05,
            "inter_core_edge_weight": 2,
            "full_core_penalty": 10,
            "max_solving_deadlock_iterations": 1000,
            "gate_usage_penalty": 0.0,
            "swap_usage_penalty": 0.002,
            "teledata_usage_penaly": 0.005,
            "telegate_usage_penalty": 0.005,
            "init_layout_hun_min_free_gate": 5,
            "init_layout_hun_min_free_qubit": 4,
            "enable_passing_core_emptying_teleport_possibility": False,
            "max_iterations": 100000,
            "save_report": True,
            "report_filename": report_path,
            "required_successes": 1,
            "max_attempts": 10,
        }
    }
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, f); f.close()
    return f.name


def run_telesabre(qasm_path: str, seed: int):
    """Run TeleSABRE binary; return (teledata, telegate, initial_p2v, success)."""
    report_tmp = tempfile.mktemp(suffix=".json")
    cfg_path   = make_ts_config(seed, report_tmp)
    try:
        proc = subprocess.run(
            [TS_BIN, cfg_path, TS_DEV, qasm_path],
            capture_output=True, text=True, timeout=120
        )
        output = proc.stdout + proc.stderr
        # Parse summary
        teledata = telegate = 0
        success  = False
        for line in output.splitlines():
            s = line.strip()
            if "Teledata:" in s:
                teledata = int(s.split(":")[1].strip())
            elif "Telegate:" in s:
                telegate = int(s.split(":")[1].strip())
            elif "Success: true" in s:
                success = True

        # Extract initial layout from report (iteration 0)
        p2v = None
        if os.path.exists(report_tmp):
            with open(report_tmp) as rf:
                rep = json.load(rf)
            if rep.get("iterations"):
                p2v = rep["iterations"][0]["phys_to_virt"]

        return teledata, telegate, p2v, success
    finally:
        os.unlink(cfg_path)
        if os.path.exists(report_tmp):
            os.unlink(report_tmp)


def p2v_to_layout(p2v: list, dag) -> dict:
    """Convert TeleSABRE phys_to_virt list to {Qiskit_qubit: phys_id} dict."""
    qubits = dag.qubits
    v2p = {}
    for p, v in enumerate(p2v):
        if v != -1 and v < len(qubits):
            v2p[qubits[v]] = p
    return v2p


def main():
    qasm_files = sorted(glob.glob(os.path.join(CIRCUIT_DIR, "*.qasm")))
    if not qasm_files:
        print(f"No .qasm files in {CIRCUIT_DIR}"); return

    print(f"{'circuit':<12}  {'TS_EPR':>6}  {'dS_same':>7}  {'Bu_same':>7}  "
          f"{'Bu_own':>6}  {'Bu_vs_TS':>9}  {'Bu_vs_dS':>9}")
    print("-" * 72)

    # Previously measured BurstDSABRE best-layout results (from paper)
    bu_own_best = {
        "ae": 30, "qft": 31, "qnn": 48, "random": 160, "ghz": 1, "graphstate": 2
    }

    for qf in qasm_files:
        cname = os.path.basename(qf).replace("_nativegates_ibm_qiskit_opt3_25.qasm", "")
        qc  = QuantumCircuit.from_qasm_file(qf)
        dag = circuit_to_dag(qc)

        best_ts_epr = best_ds_same = best_bu_same = None
        best_p2v    = None

        for seed in range(NUM_SEEDS):
            td, tg, p2v, ok = run_telesabre(qf, seed)
            if not ok or p2v is None:
                continue
            ts_epr = td + 2 * tg
            if best_ts_epr is None or ts_epr < best_ts_epr:
                best_ts_epr = ts_epr
                best_p2v    = p2v

            # BurstDSABRE and dSABRE with this exact layout
            layout = p2v_to_layout(p2v, dag)
            if len(layout) < qc.num_qubits:
                continue  # layout incomplete

            md = _run_passes(dsabre, dag, layout, LAYOUT_PASSES)
            mb = _run_passes(burst,  dag, layout, LAYOUT_PASSES)

            if not md.get("aborted") and (best_ds_same is None or md["eprs"] < best_ds_same):
                best_ds_same = md["eprs"]
            if not mb.get("aborted") and (best_bu_same is None or mb["eprs"] < best_bu_same):
                best_bu_same = mb["eprs"]

        bu_own = bu_own_best.get(cname, "?")

        def fmt(v): return str(v) if v is not None else "FAIL"
        def pct(a, b):
            if a is None or b is None or b == 0: return "  ---"
            return f"{100*(a-b)/b:+.1f}%"

        print(f"{cname:<12}  {fmt(best_ts_epr):>6}  {fmt(best_ds_same):>7}  "
              f"{fmt(best_bu_same):>7}  {fmt(bu_own):>6}  "
              f"{pct(best_bu_same, best_ts_epr):>9}  "
              f"{pct(best_bu_same, best_ds_same):>9}")

    print()
    print("Columns:")
    print("  TS_EPR   — TeleSABRE EPR (teledata + 2×telegate), best of 3 seeds")
    print("  dS_same  — dSABRE with TeleSABRE's initial layout + 2 FBF passes")
    print("  Bu_same  — BurstDSABRE with TeleSABRE's initial layout + 2 FBF passes")
    print("  Bu_own   — BurstDSABRE with its own best layout (from paper)")
    print("  Bu_vs_TS — Bu_same relative to TS_EPR (negative = Bu wins)")
    print("  Bu_vs_dS — Bu_same relative to dS_same (negative = Bu wins)")


if __name__ == "__main__":
    main()
