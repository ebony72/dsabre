"""
bench_extend_64q.py — run the three suite-extension circuits
(qpeexact, qaoa, multiplier @ 64q) through all three tools under the
paper's exact 64q protocols (2026-07-20).

Protocols (identical to the existing Table VI rows):
  dSABRE  : sabre_locked_boundary_layout (3 SabreLayout seeds) ->
            run_sabre_passes (fwd/bwd/fwd, best of pass1/pass3), best
            layout by EPR; dS and dSE; HardwareConfig(deadlock_limit=100,
            max_backup_attempts=100, max_iterations=20000).
  TeleSABRE: seeds 0-2, 300 s per-seed timeout, optimize_initial +
            hungarian layout, H_grid_2_3_4_4.json; best seed by EPR.
  pytket-dqc: network = bench_pytket_layout.py's 64q setting (6 cores,
            H-grid coupling, server_qubits = even_partition(n, 6));
            CoverEmbeddingSteinerDetached best-of-5 seeds with a 1200 s
            per-seed alarm (the paper's "20-minute budget"), falling back
            to PartitioningHeterogeneous best-of-5 when ESD fails or
            exceeds budget (the Table VI ddagger convention).

Output: results/results_extend_64q.json — one record per circuit, saved
after each circuit completes.  Merge into the canonical results files is
done separately.
"""
import glob
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import warnings

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
from router import General_dSABRE_Router
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes

CIRCUITS = ["qpeexact", "qaoa", "multiplier"]
CIRC_DIR = os.path.expanduser("~/Documents/telesabre/circuits/qasm_64")
SUFFIX = "_nativegates_ibm_qiskit_opt3_64.qasm"
TS_BIN = os.path.expanduser("~/Documents/telesabre/telesabre")
TS_DEV = os.path.expanduser("~/Documents/telesabre/devices/H_grid_2_3_4_4.json")
OUT = os.path.join(_HERE, "results", "results_extend_64q.json")

NUM_TS_SEEDS = 3
NUM_PY_SEEDS = 5
ESD_BUDGET_S = 1200
COUPLING = [[0, 1], [1, 2], [3, 4], [4, 5], [0, 3], [1, 4], [2, 5]]

_HW = HardwareConfig(deadlock_limit=100, max_backup_attempts=100, max_iterations=20000)


# ── TeleSABRE (identical to benchmark.py) ────────────────────────────────────

def _ts_config(seed, report_path):
    cfg = {"config": {
        "name": "bench", "seed": seed,
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
        "init_layout_hun_min_free_gate": 5, "init_layout_hun_free_qubit": 4,
        "enable_passing_core_emptying_teleport_possibility": False,
        "max_iterations": 200000,
        "save_report": True, "report_filename": report_path,
        "required_successes": 1, "max_attempts": 10,
    }}
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, f)
    f.close()
    return f.name


def run_telesabre(qasm_path):
    best = None
    for seed in range(NUM_TS_SEEDS):
        rpt = tempfile.mktemp(suffix=".json")
        cfg = _ts_config(seed, rpt)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [TS_BIN, cfg, TS_DEV, qasm_path],
                capture_output=True, text=True, timeout=300,
            )
        except subprocess.TimeoutExpired:
            os.unlink(cfg)
            print(f"    TS seed {seed}: timeout>300s", flush=True)
            continue
        elapsed = time.perf_counter() - t0
        out = proc.stdout + proc.stderr
        td = tg = ls_ts = 0
        ok = False
        for line in out.splitlines():
            s = line.strip()
            if "Teledata:" in s:
                td = int(s.split(":")[1])
            elif "Telegate:" in s:
                tg = int(s.split(":")[1])
            elif "Swaps:" in s:
                ls_ts = int(s.split(":")[1])
            elif "Success: true" in s:
                ok = True
        os.unlink(cfg)
        if os.path.exists(rpt):
            os.unlink(rpt)
        print(f"    TS seed {seed}: ok={ok} epr={td+tg} swaps={ls_ts} "
              f"({elapsed:.0f}s)", flush=True)
        if ok:
            eprs = td + tg
            if best is None or eprs < best["eprs"]:
                best = dict(eprs=eprs, teledata=td, telegate=tg, ls=ls_ts,
                            seed=seed, time_s=round(elapsed, 2))
    return best


# ── pytket-dqc (Table VI protocol) ───────────────────────────────────────────

class _Timeout(Exception):
    pass


def _alarm(fn, s):
    old = signal.signal(signal.SIGALRM,
                        lambda a, b: (_ for _ in ()).throw(_Timeout()))
    signal.alarm(s)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _even_partition(n, k):
    base, rem = divmod(n, k)
    groups, start = [], 0
    for i in range(k):
        end = start + base + (1 if i < rem else 0)
        groups.append(list(range(start, end)))
        start = end
    return groups


def run_pytket(qasm_path):
    from pytket.qasm import circuit_from_qasm
    from pytket_dqc import NISQNetwork
    from pytket_dqc.distributors import (
        CoverEmbeddingSteinerDetached,
        PartitioningHeterogeneous,
    )
    from pytket_dqc.utils import DQCPass

    circ = circuit_from_qasm(qasm_path, maxwidth=128)
    DQCPass().apply(circ)
    net = NISQNetwork(
        COUPLING, {i: s for i, s in enumerate(_even_partition(circ.n_qubits, 6))}
    )

    results = {}
    esd_best, esd_fail = None, 0
    for seed in range(NUM_PY_SEEDS):
        t0 = time.time()
        try:
            d = _alarm(
                lambda: CoverEmbeddingSteinerDetached().distribute(
                    circ, net, seed=seed),
                ESD_BUDGET_S,
            )
            cost = d.cost()
            print(f"    py esd seed {seed}: {cost} ({time.time()-t0:.0f}s)",
                  flush=True)
            if esd_best is None or cost < esd_best:
                esd_best = cost
        except _Timeout:
            esd_fail += 1
            print(f"    py esd seed {seed}: timeout>{ESD_BUDGET_S}s", flush=True)
            if esd_fail >= 2:
                print("    py esd: skipping remaining seeds", flush=True)
                break
        except Exception as e:
            esd_fail += 1
            print(f"    py esd seed {seed}: {repr(e)[:80]}", flush=True)
            if esd_fail >= 2:
                break
    results["esd"] = esd_best

    php_best = None
    for seed in range(NUM_PY_SEEDS):
        t0 = time.time()
        try:
            d = _alarm(
                lambda: PartitioningHeterogeneous().distribute(
                    circ, net, seed=seed),
                ESD_BUDGET_S,
            )
            cost = d.cost()
            print(f"    py php seed {seed}: {cost} ({time.time()-t0:.0f}s)",
                  flush=True)
            if php_best is None or cost < php_best:
                php_best = cost
        except Exception as e:
            print(f"    py php seed {seed}: {repr(e)[:80]}", flush=True)
    results["php"] = php_best

    if esd_best is not None:
        results["py_ebits"] = esd_best
        results["py_method"] = "CoverEmbeddingSteinerDetached"
    else:
        results["py_ebits"] = php_best
        results["py_method"] = "PartitioningHeterogeneous"
    return results


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    arch = build_h_grid_architecture(r=2, s=3, m=4)
    routers = {
        "dS": General_dSABRE_Router(arch, _HW),
        "dSE": dSABRE_BurstExt(arch, _HW),
    }

    records = []
    if os.path.exists(OUT):
        records = json.load(open(OUT))["results"]
        done = {r["circuit"] for r in records}
    else:
        done = set()

    for cname in CIRCUITS:
        if cname in done:
            print(f"skip {cname} (done)", flush=True)
            continue
        qf = os.path.join(CIRC_DIR, cname + SUFFIX)
        assert os.path.exists(qf), qf
        print(f"=== {cname} ===", flush=True)

        qc = QuantumCircuit.from_qasm_file(qf)
        qc = qc.remove_final_measurements(inplace=False)
        qc = PassManager([RemoveBarriers()]).run(qc)
        dag = circuit_to_dag(qc)
        rev_dag = circuit_to_dag(qc.reverse_ops())
        n_cx = sum(1 for _ in dag.two_qubit_ops())
        print(f"  qubits={qc.num_qubits} cx={n_cx}", flush=True)

        ts = run_telesabre(qf)

        sl_layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)
        router_results = {}
        for k, router in routers.items():
            best_m = None
            for li, layout in enumerate(sl_layouts):
                t0 = time.perf_counter()
                m = run_sabre_passes(router, dag, rev_dag, layout)
                el = time.perf_counter() - t0
                got = (m["eprs"] if m and not m.get("aborted") else "abort")
                print(f"    {k} layout {li}: {got} ({el:.0f}s)", flush=True)
                if m and not m.get("aborted"):
                    if best_m is None or m["eprs"] < best_m["eprs"]:
                        best_m = m
            if best_m is not None:
                router_results[k] = dict(
                    eprs=best_m["eprs"], ls=best_m["ls"],
                    time_s=round(best_m["compile_time"], 3), aborted=False)
            else:
                router_results[k] = {"aborted": True}

        py = run_pytket(qf)

        rec = dict(circuit=cname, suite="64q", cx=n_cx, ts=ts,
                   routers=router_results, **py)
        records.append(rec)
        payload = dict(
            meta=dict(
                date=time.strftime("%Y-%m-%d"),
                description="64q suite extension: qpeexact/qaoa/multiplier",
                protocols="identical to Table VI rows; see module docstring",
            ),
            results=records,
        )
        with open(OUT, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  saved -> {OUT}", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
