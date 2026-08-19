"""
Fill-ratio sweep: QFT at n = 16, 32, 48, 60 on the same 64-qubit B-grid.

Tests the hypothesis that DMapS's relative EPR cost (DMapS / dSE) shrinks
toward 1 as fill ratio n / 64 climbs, because dSABRE's free-slot-teleport
advantage disappears when there are few free slots left.

Outputs:
    code/results/results_fill_sweep_dse.json   (dSE numbers, run in main env)
    code/results/results_fill_sweep_dmaps.json (DMapS numbers, run in dmaps env)

Run dSE side from the main repo env, DMapS side from the dmaps conda env:
    python code/run_fill_sweep.py dse
    /opt/anaconda3/envs/dmaps/bin/python code/run_fill_sweep.py dmaps
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from circuit_paths import circuits_path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RESULTS = os.environ.get("DSABRE_OUT_DIR") or os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)

QFT_SIZES = [24, 36, 48, 56]   # fill ratios 37.5 / 56 / 75 / 87.5 % on 64-phys
NUM_SEEDS = 5
QASM_DIR = circuits_path("qasm_fill_sweep")


def make_qft_qasm(n: int, path: str):
    """Generate a QFT_n circuit decomposed to {sx, rz, cx, x} without
    aggressive coupling-free optimisation (which can collapse QFT-16 to a
    trivial form).  We compile at optimization_level=1 so the CX count
    scales as O(n^2) as expected."""
    from qiskit.circuit.library import QFT
    from qiskit import transpile
    from qiskit.qasm2 import dumps

    qc = QFT(n, do_swaps=True, inverse=False, approximation_degree=0)
    qc = transpile(qc.decompose(reps=4),
                   basis_gates=["sx", "rz", "cx", "x"],
                   optimization_level=1,
                   seed_transpiler=0)
    with open(path, "w") as f:
        f.write(dumps(qc))


def ensure_qasms():
    os.makedirs(QASM_DIR, exist_ok=True)
    paths = {}
    for n in QFT_SIZES:
        p = os.path.join(QASM_DIR, f"qft_n{n}.qasm")
        if not os.path.exists(p):
            print(f"  generating {p}", flush=True)
            make_qft_qasm(n, p)
        paths[n] = p
    return paths


# ── dSE side (main qiskit env) ───────────────────────────────────────────────
def run_dse(paths):
    import random
    from qiskit import QuantumCircuit
    from qiskit.converters import circuit_to_dag
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import RemoveBarriers
    from architecture import build_b_grid_architecture
    from config import HardwareConfig
    from dsabre_ext import dSABRE_BurstExt
    from layout import sabre_locked_boundary_layout, run_sabre_passes

    arch = build_b_grid_architecture(2, 2, 4)
    hw = HardwareConfig(deadlock_limit=100, max_backup_attempts=100, max_iterations=20000)
    router = dSABRE_BurstExt(arch, hw)

    rows = []
    for n, qasm in paths.items():
        qc = QuantumCircuit.from_qasm_file(qasm).remove_final_measurements(inplace=False)
        qc = PassManager([RemoveBarriers()]).run(qc)
        dag = circuit_to_dag(qc)
        rev_dag = circuit_to_dag(qc.reverse_ops())
        n_cx = sum(1 for _ in dag.two_qubit_ops())

        sl_layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
        best = None
        t0 = time.perf_counter()
        for layout in sl_layouts:
            m = run_sabre_passes(router, dag, rev_dag, layout)
            if m and not m.get("aborted"):
                if best is None or m["eprs"] < best["eprs"]:
                    best = m
        t = time.perf_counter() - t0
        if best is None:
            print(f"  dSE qft_n{n}: FAIL", flush=True)
            rows.append({"n": n, "cx": n_cx, "eprs": -1, "ls": -1, "time_s": t})
            continue
        rows.append({"n": n, "cx": n_cx, "eprs": best["eprs"], "ls": best["ls"],
                     "time_s": t})
        print(f"  dSE qft_n{n:<2}  cx={n_cx:<5}  eprs={best['eprs']:<5}  ls={best['ls']:<5}  t={t:.1f}s",
              flush=True)
    return rows


# ── DMapS side (dmaps conda env) ─────────────────────────────────────────────
def run_dmaps(paths):
    import contextlib
    sys.path.insert(0, os.path.expanduser("~/Documents/DMapS/src"))
    from pathlib import Path
    from architecture import build_b_grid_architecture
    from run_dmaps_bench import _arch_to_dmaps_json, derive_metrics, _silence_fd
    from router.multi_mode_routing import MultiModeRouting

    arch = build_b_grid_architecture(2, 2, 4)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tf:
        dev = tf.name
    qubit_chip = _arch_to_dmaps_json(arch, dev)

    rows = []
    for n, qasm in paths.items():
        best = None
        elapsed = 0.0
        seed_log = []
        for seed in range(NUM_SEEDS):
            t0 = time.perf_counter()
            try:
                with _silence_fd():
                    _, _, mapped = MultiModeRouting().cpa_map_car_rout(
                        origin_qasm_fn=Path(qasm),
                        chip_info_fn=Path(dev),
                        chip_type="zcz",
                        intra_chip_all2all=False,
                    )
                m = derive_metrics(mapped, qubit_chip)
                ok = True
            except Exception as e:
                m = {"epr": -1, "local_swap": -1, "remote_cx": -1,
                     "remote_swap": -1, "local_cx": -1, "overall": -1,
                     "error": repr(e)}
                ok = False
            dt = time.perf_counter() - t0
            elapsed += dt
            m["seed"] = seed; m["seed_time"] = dt
            seed_log.append(m)
            if ok and (best is None or m["overall"] < best["overall"]):
                best = m
        if best is None:
            best = seed_log[0]
        best = {k: v for k, v in best.items() if k != "seeds"}
        best["total_time"] = elapsed
        best["n"] = n
        rows.append(best)
        print(f"  DMapS qft_n{n:<2}  epr={best['epr']:<5}  ls={best['local_swap']:<5}  "
              f"rCX={best['remote_cx']:<4}  rSWAP={best['remote_swap']:<4}  t={elapsed:.1f}s",
              flush=True)
    os.unlink(dev)
    return rows


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("dse", "dmaps"):
        print("usage: run_fill_sweep.py {dse|dmaps}", file=sys.stderr)
        sys.exit(1)
    side = sys.argv[1]
    paths = ensure_qasms()

    if side == "dse":
        rows = run_dse(paths)
        out = os.path.join(RESULTS, "results_fill_sweep_dse.json")
    else:
        rows = run_dmaps(paths)
        out = os.path.join(RESULTS, "results_fill_sweep_dmaps.json")

    with open(out, "w") as f:
        json.dump({"sizes": QFT_SIZES, "device": "B-grid 2x2 4x4 (64 phys)",
                   "rows": rows}, f, indent=2)
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
