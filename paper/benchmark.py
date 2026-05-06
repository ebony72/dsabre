"""
benchmark.py — dSABRE vs dSABRE_BurstExt vs TeleSABRE evaluation.

Covers three circuit suites:
  25q  ~/Documents/telesabre/circuits/qasm_25/   B-grid 2x2 4x4  (64 qubits)
  36q  ~/Documents/telesabre/circuits/qasm_36/   B-grid 2x2 4x4  (64 qubits)
  64q  ~/Documents/telesabre/circuits/qasm_64/   H-grid 2x3 4x4  (96 qubits)

Protocol (same for all suites):
  1. Run TeleSABRE with seeds 0–2; pick the seed with fewest EPRs.
  2. Extract TS's initial layout (phys_to_virt from the JSON report).
  3. Route with dS and dSE under that layout (LAYOUT_PASSES=2).
  4. Print a formatted table and save results/results_<suite>.json.

Usage:
  python benchmark.py                # all suites
  python benchmark.py --suite 25     # 25q only
  python benchmark.py --suite 36     # 36q only
  python benchmark.py --suite 64     # 64q only
"""

import sys, os, json, glob, subprocess, tempfile, time, argparse
from math import prod

sys.setrecursionlimit(50000)
sys.path.insert(0, os.path.dirname(__file__))

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler import PassManager

from architecture import build_b_grid_architecture, build_h_grid_architecture
from config import HardwareConfig
from router import General_dSABRE_Router
from dsabre_ext import dSABRE_BurstExt
from layout import locality_aware_layout, run_passes

# ── Paths ──────────────────────────────────────────────────────────────────────
TS_BIN      = os.path.expanduser("~/Documents/telesabre/telesabre")
_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

# ── Hardware configs ───────────────────────────────────────────────────────────
_HW_SMALL = HardwareConfig()                          # 25q default
_HW_LARGE = HardwareConfig(                           # 36q / 64q
    deadlock_limit=100, max_backup_attempts=100, max_iterations=20000
)

# ── Suite definitions ──────────────────────────────────────────────────────────
SUITES = {
    "25q": dict(
        circuit_dir = os.path.expanduser("~/Documents/telesabre/circuits/qasm_25"),
        suffix      = "_nativegates_ibm_qiskit_opt3_25.qasm",
        arch        = build_b_grid_architecture(r=2, s=2, m=4),
        ts_dev      = os.path.expanduser("~/Documents/telesabre/devices/B_grid_2_2_4_4.json"),
        hw          = _HW_SMALL,
    ),
    "36q": dict(
        circuit_dir = os.path.expanduser("~/Documents/telesabre/circuits/qasm_36"),
        suffix      = "_nativegates_ibm_qiskit_opt3_36.qasm",
        arch        = build_b_grid_architecture(r=2, s=2, m=4),
        ts_dev      = os.path.expanduser("~/Documents/telesabre/devices/B_grid_2_2_4_4.json"),
        hw          = _HW_LARGE,
    ),
    "64q": dict(
        circuit_dir = os.path.expanduser("~/Documents/telesabre/circuits/qasm_64"),
        suffix      = "_nativegates_ibm_qiskit_opt3_64.qasm",
        arch        = build_h_grid_architecture(r=2, s=3, m=4),
        ts_dev      = os.path.expanduser("~/Documents/telesabre/devices/H_grid_2_3_4_4.json"),
        hw          = _HW_LARGE,
    ),
}

LAYOUT_PASSES = 2
NUM_TS_SEEDS  = 3

# ── TeleSABRE helpers ──────────────────────────────────────────────────────────

def _ts_config(seed: int, report_path: str, ts_name: str) -> str:
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
        "max_iterations": 200000,
        "save_report": True, "report_filename": report_path,
        "required_successes": 1, "max_attempts": 10,
    }}
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(cfg, f); f.close()
    return f.name


def run_telesabre(qasm_path: str, ts_dev: str) -> dict | None:
    """Run TeleSABRE with seeds 0–(NUM_TS_SEEDS-1); return best result or None."""
    best = None
    for seed in range(NUM_TS_SEEDS):
        rpt = tempfile.mktemp(suffix=".json")
        cfg = _ts_config(seed, rpt, "bench")
        t0  = time.perf_counter()
        try:
            proc = subprocess.run(
                [TS_BIN, cfg, ts_dev, qasm_path],
                capture_output=True, text=True, timeout=300,
            )
        except subprocess.TimeoutExpired:
            os.unlink(cfg)
            continue
        elapsed = time.perf_counter() - t0

        out = proc.stdout + proc.stderr
        td = tg = 0; ok = False
        for line in out.splitlines():
            s = line.strip()
            if   "Teledata:" in s: td = int(s.split(":")[1])
            elif "Telegate:" in s: tg = int(s.split(":")[1])
            elif "Success: true" in s: ok = True

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
            if best is None or eprs < best["eprs"]:
                best = dict(eprs=eprs, teledata=td, telegate=tg,
                            seed=seed, time_s=round(elapsed, 2), p2v=p2v)
    return best


def p2v_to_layout(p2v: list, dag) -> dict:
    qubits = dag.qubits
    return {qubits[v]: p for p, v in enumerate(p2v)
            if v != -1 and v < len(qubits)}


def load_qasm(path: str):
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


# ── Per-suite benchmark ────────────────────────────────────────────────────────

def bench_suite(suite_name: str, s: dict) -> list:
    arch    = s["arch"]
    hw      = s["hw"]
    ts_dev  = s["ts_dev"]
    suffix  = s["suffix"]

    routers = {
        "dS":  General_dSABRE_Router(arch, hw),
        "dSE": dSABRE_BurstExt(arch, hw),
    }
    rkeys = list(routers.keys())

    qasm_files = sorted(glob.glob(os.path.join(s["circuit_dir"], "*.qasm")))
    if not qasm_files:
        print(f"  [no .qasm files in {s['circuit_dir']}]")
        return []

    col_w = 12
    hdr   = f"{'circuit':<{col_w}}  {'q':>3}  {'cx':>5}  {'TS_epr':>7}  {'TS_t':>5}"
    for k in rkeys:
        hdr += f"  {k+'_epr':>8}  {k+'_ls':>7}  {k+'_t':>6}"
    for k in rkeys:
        hdr += f"  {k+'/TS%':>8}"
    print(f"\n{'═'*len(hdr)}")
    print(f"  {suite_name}")
    print(f"{'═'*len(hdr)}")
    print(hdr)
    print("─" * len(hdr))

    records = []

    for qf in qasm_files:
        cname     = os.path.basename(qf).replace(suffix, "")
        t_wall    = time.time()
        qc, dag   = load_qasm(qf)
        n_qubits  = qc.num_qubits
        n_cx      = sum(1 for _ in dag.two_qubit_ops())

        ts_result = run_telesabre(qf, ts_dev)

        router_results = {}
        if ts_result is not None:
            layout = p2v_to_layout(ts_result["p2v"], dag)
            if len(layout) >= n_qubits:
                for k, router in routers.items():
                    m = run_passes(router, dag, layout, LAYOUT_PASSES)
                    if m and not m.get("aborted"):
                        router_results[k] = dict(
                            eprs=m["eprs"], ls=m["ls"],
                            time_s=round(m["compile_time"], 3), aborted=False,
                            # Mechanism instrumentation:
                            relief_candidates  = m.get("relief_candidates", 0),
                            relief_picks       = m.get("relief_picks", 0),
                            backup_activations = m.get("backup_activations", 0),
                            force_make_room    = m.get("force_make_room", 0),
                        )
                    else:
                        router_results[k] = {"aborted": True}

        def fmt_i(v): return str(v)   if v is not None else "---"
        def fmt_f(v): return f"{v:.2f}" if v is not None else "  ---"
        def pct(a, b):
            if a is None or b is None or b == 0: return "    ---"
            return f"{100*(a-b)/b:+.1f}%"

        ts_epr  = ts_result["eprs"]   if ts_result else None
        ts_time = ts_result["time_s"] if ts_result else None

        row = (f"{cname:<{col_w}}  {n_qubits:>3}  {n_cx:>5}"
               f"  {fmt_i(ts_epr):>7}  {fmt_f(ts_time):>5}")
        for k in rkeys:
            r   = router_results.get(k, {})
            epr = r.get("eprs")   if not r.get("aborted") else None
            ls  = r.get("ls")     if not r.get("aborted") else None
            t   = r.get("time_s") if not r.get("aborted") else None
            row += f"  {fmt_i(epr):>8}  {fmt_i(ls):>7}  {fmt_f(t):>6}"
        for k in rkeys:
            r   = router_results.get(k, {})
            epr = r.get("eprs") if not r.get("aborted") else None
            row += f"  {pct(epr, ts_epr):>8}"

        elapsed_wall = time.time() - t_wall
        print(row + f"  ({elapsed_wall:.0f}s)")

        records.append(dict(
            suite=suite_name, circuit=cname, qubits=n_qubits, cx=n_cx,
            ts=ts_result, routers=router_results,
        ))

    # Geometric mean summary
    def gmean(lst):
        lst = [x for x in lst if x is not None and x > 0]
        return prod(lst) ** (1 / len(lst)) if lst else float("nan")

    ts_eprs = [r["ts"]["eprs"] for r in records if r["ts"]]
    print("─" * len(hdr))
    summary = f"{'gmean':<{col_w}}  {'':>3}  {'':>5}  {gmean(ts_eprs):>7.1f}  {'':>5}"
    for k in rkeys:
        eprs = [r["routers"].get(k, {}).get("eprs") for r in records
                if not r["routers"].get(k, {}).get("aborted")]
        lss  = [r["routers"].get(k, {}).get("ls") for r in records
                if not r["routers"].get(k, {}).get("aborted")]
        ts   = [r["routers"].get(k, {}).get("time_s") for r in records
                if not r["routers"].get(k, {}).get("aborted")]
        summary += f"  {gmean(eprs):>8.1f}  {gmean(lss):>7.1f}  {gmean(ts):>6.2f}"
    for k in rkeys:
        eprs = [r["routers"].get(k, {}).get("eprs") for r in records
                if not r["routers"].get(k, {}).get("aborted")]
        summary += f"  {pct(gmean(eprs), gmean(ts_eprs)):>8}"
    print(summary)

    return records


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=["25", "36", "64", "all"], default="all",
                        help="Which circuit suite to run (default: all)")
    args = parser.parse_args()

    run_suites = []
    if args.suite in ("25", "all"): run_suites.append("25q")
    if args.suite in ("36", "all"): run_suites.append("36q")
    if args.suite in ("64", "all"): run_suites.append("64q")

    t_total = time.time()
    for sname in run_suites:
        records = bench_suite(sname, SUITES[sname])
        out_path = os.path.join(_RESULTS_DIR, f"results_{sname}.json")
        payload = dict(
            meta=dict(
                date          = time.strftime("%Y-%m-%d"),
                suite         = sname,
                arch          = ("B-grid 2x2 4x4 (64 qubits)" if sname != "64q"
                                 else "H-grid 2x3 4x4 (96 qubits)"),
                layout_passes = LAYOUT_PASSES,
                ts_seeds      = NUM_TS_SEEDS,
                routers       = {
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
