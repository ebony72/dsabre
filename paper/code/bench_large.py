"""
bench_large.py — dSABRE vs dSABRE_BurstExt vs TeleSABRE on 100/200/360-qubit circuits.

Suites:
  100q  ~/Documents/telesabre/circuits/qasm_100/  H-grid 2x3 5x5 (150 physical qubits)
  200q  ~/Documents/telesabre/circuits/qasm_200/  H-grid 4x3 5x5 (300 physical qubits)
  360q  ~/Documents/telesabre/circuits/qasm_360/  H-grid 2x3 9x9 (486 physical qubits)

Protocol (same for all suites):
  1. Run TeleSABRE with seeds 0-2 (timeout 600s each); pick best-EPR seed.
  2. For dS and dSE: run from TS layout (if TS converged) + locality layouts (seeds 0-2).
  3. Report best EPR across all layout attempts for each router.
  4. Save paper/results/results_{suite}.json in the same format as results_64q.json.

Usage:
  python bench_large.py                  # all suites
  python bench_large.py --suite 100      # 100q only
  python bench_large.py --suite 200      # 200q only
  python bench_large.py --suite 360      # 360q only
"""

import sys, os, json, glob, subprocess, tempfile, time, argparse, random
from math import prod

sys.setrecursionlimit(100000)
sys.path.insert(0, os.path.dirname(__file__))

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler import PassManager

from architecture import build_h_grid_architecture
from config import HardwareConfig
from router import General_dSABRE_Router
from dsabre_ext import dSABRE_BurstExt
from layout import locality_aware_layout, run_passes


def p2v_to_layout(p2v, dag):
    qubits = dag.qubits
    return {qubits[v]: p for p, v in enumerate(p2v)
            if v != -1 and v < len(qubits)}

# ── Paths ──────────────────────────────────────────────────────────────────────
TS_BIN       = os.path.expanduser("~/Documents/telesabre/telesabre")
_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

_HW = HardwareConfig(deadlock_limit=200, max_backup_attempts=200, max_iterations=50000)

SUITES = {
    "100q": dict(
        circuit_dir = os.path.expanduser("~/Documents/telesabre/circuits/qasm_100"),
        suffix      = "_nativegates_ibm_qiskit_opt3_100.qasm",
        arch        = build_h_grid_architecture(r=2, s=3, m=5),
        ts_dev      = os.path.expanduser("~/Documents/telesabre/devices/H_grid_2_3_5_5.json"),
        ts_timeout  = 300,
        circuits    = ["qft", "qpeexact"],
    ),
    "200q": dict(
        circuit_dir = os.path.expanduser("~/Documents/telesabre/circuits/qasm_200"),
        suffix      = "_nativegates_ibm_qiskit_opt3_200.qasm",
        arch        = build_h_grid_architecture(r=4, s=3, m=5),
        ts_dev      = os.path.expanduser("~/Documents/telesabre/devices/H_grid_4_3_5_5.json"),
        ts_timeout  = 600,
        circuits    = ["qft", "qpeexact"],
    ),
    "360q": dict(
        circuit_dir = os.path.expanduser("~/Documents/telesabre/circuits/qasm_360"),
        suffix      = "_nativegates_ibm_qiskit_opt3_360.qasm",
        arch        = build_h_grid_architecture(r=2, s=3, m=9),
        ts_dev      = os.path.expanduser("~/Documents/telesabre/devices/H_grid_2_3_9_9.json"),
        ts_timeout  = 600,
        circuits    = ["qft", "qpeexact"],
    ),
}

NUM_TS_SEEDS      = 3
NUM_LAYOUT_SEEDS  = 3
LAYOUT_PASSES     = 2


def _ts_config(seed, report_path, ts_name, max_iter=200000):
    cfg = {"config": {
        "name": ts_name, "seed": seed,
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
        "max_iterations": max_iter,
        "save_report": True, "report_filename": report_path,
        "required_successes": 1, "max_attempts": 10,
    }}
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, f); f.close()
    return f.name


def run_telesabre(qasm_path, ts_dev, timeout=300):
    """Run TeleSABRE seeds 0–(NUM_TS_SEEDS-1); return best result dict or None."""
    best = None
    for seed in range(NUM_TS_SEEDS):
        rpt = tempfile.mktemp(suffix=".json")
        cfg = _ts_config(seed, rpt, "bench")
        t0  = time.perf_counter()
        try:
            proc = subprocess.run(
                [TS_BIN, cfg, ts_dev, qasm_path],
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"    TS seed {seed}: timeout ({timeout}s)")
            os.unlink(cfg)
            continue
        elapsed = time.perf_counter() - t0

        out = proc.stdout + proc.stderr
        td = tg = ts_ls = 0; ok = False
        for line in out.splitlines():
            s = line.strip()
            if   "Teledata:" in s:       td    = int(s.split(":")[1])
            elif "Telegate:" in s:       tg    = int(s.split(":")[1])
            elif "Swaps:" in s:          ts_ls = int(s.split(":")[1])
            elif "Success: true" in s:   ok    = True

        p2v = None
        if os.path.exists(rpt):
            try:
                with open(rpt) as rf:
                    rep = json.load(rf)
                if rep.get("iterations"):
                    p2v = rep["iterations"][0]["phys_to_virt"]
            except Exception:
                pass
        os.unlink(cfg)
        if os.path.exists(rpt): os.unlink(rpt)

        if ok and p2v is not None:
            eprs = td + tg
            print(f"    TS seed {seed}: EPR={eprs}, SWAP={ts_ls} ({elapsed:.1f}s)")
            if best is None or eprs < best["eprs"]:
                best = dict(eprs=eprs, teledata=td, telegate=tg, ts_ls=ts_ls,
                            seed=seed, time_s=round(elapsed, 2), p2v=p2v)
        else:
            print(f"    TS seed {seed}: FAILED ({elapsed:.1f}s)")
    return best


def run_dsabre(router_key, router, dag, arch, ts_result, label):
    """Try TS layout (if available) + locality layouts; return best (eprs, ls) result."""
    layouts = []
    if ts_result is not None:
        layouts.append(("ts_layout", p2v_to_layout(ts_result["p2v"], dag)))
    for seed in range(NUM_LAYOUT_SEEDS):
        rng = random.Random(seed)
        loc = locality_aware_layout(dag, arch, rng=rng)
        layouts.append((f"locality_seed{seed}", loc))

    best = None
    for layout_name, layout in layouts:
        t0 = time.perf_counter()
        m  = run_passes(router, dag, layout, LAYOUT_PASSES)
        elapsed = time.perf_counter() - t0
        if m and not m.get("aborted"):
            eprs = m["eprs"]; ls = m["ls"]
            print(f"    {label} {layout_name}: EPR={eprs}, SWAP={ls} ({elapsed:.1f}s)")
            if best is None or eprs < best["eprs"]:
                best = dict(eprs=eprs, ls=ls, layout=layout_name,
                            time_s=round(elapsed, 2), aborted=False)
        else:
            print(f"    {label} {layout_name}: ABORTED ({elapsed:.1f}s)")
    if best is None:
        best = dict(aborted=True)
    return best


def bench_suite(suite_name, s):
    arch       = s["arch"]
    ts_dev     = s["ts_dev"]
    suffix     = s["suffix"]
    ts_timeout = s["ts_timeout"]
    target_circuits = s["circuits"]

    routers = {
        "dS":  General_dSABRE_Router(arch, _HW),
        "dSE": dSABRE_BurstExt(arch, _HW),
    }

    records = []
    print(f"\n{'═'*60}\n  {suite_name}\n{'═'*60}")

    for cname in target_circuits:
        pattern = os.path.join(s["circuit_dir"], f"{cname}{suffix}")
        matches = glob.glob(pattern)
        if not matches:
            print(f"  [{cname}: no file found at {pattern}]")
            continue
        qasm_path = matches[0]

        print(f"\n  Circuit: {cname}")
        qc  = QuantumCircuit.from_qasm_file(qasm_path)
        qc  = qc.remove_final_measurements(inplace=False)
        qc  = PassManager([RemoveBarriers()]).run(qc)
        dag = circuit_to_dag(qc)
        n_qubits = qc.num_qubits
        n_cx     = sum(1 for _ in dag.two_qubit_ops())
        print(f"    n={n_qubits}, CX={n_cx}")

        print("  → TeleSABRE")
        ts_result = run_telesabre(qasm_path, ts_dev, timeout=ts_timeout)

        router_results = {}
        for rkey, router in routers.items():
            print(f"  → {rkey}")
            router_results[rkey] = run_dsabre(rkey, router, dag, arch, ts_result, rkey)

        # Build record matching results_64q.json format
        rec = dict(
            suite=suite_name, circuit=cname, qubits=n_qubits, cx=n_cx,
            ts=ts_result,
            routers=router_results,
        )
        records.append(rec)

        # Quick summary
        ts_epr = ts_result["eprs"] if ts_result else None
        for rkey in ["dS", "dSE"]:
            r = router_results.get(rkey, {})
            if not r.get("aborted") and ts_epr:
                delta = 100 * (r["eprs"] - ts_epr) / ts_epr
                print(f"    {rkey}: EPR={r['eprs']} ({delta:+.1f}% vs TS)")

    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=["100", "200", "360", "all"], default="all")
    args = parser.parse_args()

    run_suites = []
    if args.suite in ("100", "all"): run_suites.append("100q")
    if args.suite in ("200", "all"): run_suites.append("200q")
    if args.suite in ("360", "all"): run_suites.append("360q")

    t_total = time.time()
    all_records = {}
    for sname in run_suites:
        records = bench_suite(sname, SUITES[sname])
        all_records[sname] = records
        out_path = os.path.join(_RESULTS_DIR, f"results_{sname}.json")
        payload = dict(
            meta=dict(
                date=time.strftime("%Y-%m-%d"),
                suite=sname,
                layout_passes=LAYOUT_PASSES,
                ts_seeds=NUM_TS_SEEDS,
                locality_seeds=NUM_LAYOUT_SEEDS,
                routers={
                    "dS":  "General_dSABRE_Router  (router.py)",
                    "dSE": "dSABRE_BurstExt        (dsabre_ext.py)",
                },
            ),
            results=records,
        )
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved → {out_path}")

    print(f"\nTotal wall time: {time.time() - t_total:.0f}s")


if __name__ == "__main__":
    main()
