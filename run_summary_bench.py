"""
run_summary_bench.py — Comprehensive comparison: TeleSABRE vs three dSABRE variants.

Routers under test
──────────────────
  TS    TeleSABRE (compiled C++ binary, reference baseline)
  dS    General_dSABRE_Router  (router.py  — production, Qiskit DAGCircuit)
  dSf   General_dSABRE_Router  (router_test.py — LightDAG, avoids deepcopy)
  dSE   dSABRE_BurstExt        (burst_ext_router.py — BFS-layer extended set,
                                  Option A, built on router.py)

Hardware
────────
  B-grid 2×2 4×4  (4 cores, 16 physical qubits per core = 64 total)

Circuit suites
──────────────
  25q  ~/Documents/telesabre/circuits/qasm_25/
  36q  ~/Documents/telesabre/circuits/qasm_36/

Protocol
────────
  1. Run TS with seeds 0-2; pick the seed that gives the fewest EPRs.
  2. Extract TS's best initial layout (phys_to_virt from the JSON report).
  3. Run all three dSABRE-family routers under that layout (LAYOUT_PASSES=2).
  4. Record per-circuit: EPRs, local SWAPs, compile time (s).
  5. Write full JSON to results_summary.json for permanent record.
  6. Print formatted table to stdout.

Run:
  python run_summary_bench.py
  python run_summary_bench.py --suite 25    # 25q only
  python run_summary_bench.py --suite 36    # 36q only
"""

import sys, os, json, glob, subprocess, tempfile, time, argparse
from math import prod

sys.setrecursionlimit(50000)
sys.path.insert(0, os.path.dirname(__file__))

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler import PassManager

from architecture import build_b_grid_architecture
from config import HardwareConfig

# ── router imports ────────────────────────────────────────────────────────────
import router as _router_mod
import router_test as _router_test_mod
from burst_ext_router import dSABRE_BurstExt
from main import _run_passes

# ── hardware / config ─────────────────────────────────────────────────────────
TS_BIN = os.path.expanduser("~/Documents/telesabre/telesabre")
TS_DEV = os.path.expanduser("~/Documents/telesabre/devices/B_grid_2_2_4_4.json")

LAYOUT_PASSES = 2
NUM_TS_SEEDS  = 3

arch   = build_b_grid_architecture(r=2, s=2, m=4)
hw_25  = HardwareConfig()
hw_36  = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                        max_iterations=20000, max_burst_walk_depth=25)
hw_64  = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                        max_iterations=20000, max_burst_walk_depth=25)

SUITES = {
    "25q": {
        "dir":    os.path.expanduser("~/Documents/telesabre/circuits/qasm_25"),
        "suffix": "_nativegates_ibm_qiskit_opt3_25.qasm",
        "hw":     hw_25,
    },
    "36q": {
        "dir":    os.path.expanduser("~/Documents/telesabre/circuits/qasm_36"),
        "suffix": "_nativegates_ibm_qiskit_opt3_36.qasm",
        "hw":     hw_36,
    },
    "64q": {
        "dir":    os.path.expanduser("~/Documents/telesabre/circuits/qasm_64"),
        "suffix": "_nativegates_ibm_qiskit_opt3_64.qasm",
        "hw":     hw_64,
    },
}

# ── TeleSABRE helpers ─────────────────────────────────────────────────────────

def _ts_config(seed: int, report_path: str) -> str:
    cfg = {"config": {
        "name": "summary_bench", "seed": seed,
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


def run_telesabre(qasm_path: str):
    """Run TS with multiple seeds; return best result dict."""
    best = None
    for seed in range(NUM_TS_SEEDS):
        rpt = tempfile.mktemp(suffix=".json")
        cfg = _ts_config(seed, rpt)
        t0  = time.perf_counter()
        try:
            proc = subprocess.run(
                [TS_BIN, cfg, TS_DEV, qasm_path],
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
                best = {
                    "eprs": eprs, "teledata": td, "telegate": tg,
                    "seed": seed, "time_s": round(elapsed, 2),
                    "p2v": p2v,
                }
    return best


def p2v_to_layout(p2v, dag):
    qubits = dag.qubits
    return {qubits[v]: p for p, v in enumerate(p2v)
            if v != -1 and v < len(qubits)}


def load_qasm(qf: str):
    qc = QuantumCircuit.from_qasm_file(qf)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


# ── per-suite benchmark ────────────────────────────────────────────────────────

def bench_suite(suite_name: str, hw: HardwareConfig, circuit_dir: str, suffix: str):
    qasm_files = sorted(glob.glob(os.path.join(circuit_dir, "*.qasm")))
    if not qasm_files:
        print(f"  [no .qasm files in {circuit_dir}]")
        return []

    routers = {
        "dS":  _router_mod.General_dSABRE_Router(arch, hw),
        "dSf": _router_test_mod.General_dSABRE_Router(arch, hw),
        "dSE": dSABRE_BurstExt(arch, hw),
    }

    # ── header ──
    col_w = 12
    rkeys = list(routers.keys())
    hdr  = f"{'circuit':<{col_w}}  {'q':>3}  {'cx':>5}  {'TS_epr':>7}  {'TS_t':>5}"
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
        cname  = os.path.basename(qf).replace(suffix, "")
        t_wall = time.time()
        qc, dag = load_qasm(qf)
        n_qubits = qc.num_qubits
        n_cx     = sum(1 for _ in dag.two_qubit_ops())

        # TeleSABRE
        ts_result = run_telesabre(qf)

        rec = {
            "suite":   suite_name,
            "circuit": cname,
            "qubits":  n_qubits,
            "cx":      n_cx,
            "ts": ts_result,
            "routers": {},
        }

        router_results = {}
        if ts_result is not None:
            layout = p2v_to_layout(ts_result["p2v"], dag)
            if len(layout) >= n_qubits:
                for k, router in routers.items():
                    m = _run_passes(router, dag, layout, LAYOUT_PASSES)
                    if m and not m.get("aborted"):
                        router_results[k] = {
                            "eprs":      m["eprs"],
                            "ls":        m["ls"],
                            "time_s":    round(m["compile_time"], 3),
                            "aborted":   False,
                        }
                    else:
                        router_results[k] = {"aborted": True}

        rec["routers"] = router_results

        def fmt_int(v): return str(v) if v is not None else "---"
        def fmt_f(v):   return f"{v:.2f}" if v is not None else "  ---"
        def pct(a, b):
            if a is None or b is None or b == 0: return "    ---"
            return f"{100*(a-b)/b:+.1f}%"

        ts_epr  = ts_result["eprs"]  if ts_result else None
        ts_time = ts_result["time_s"] if ts_result else None

        row = (f"{cname:<{col_w}}  {n_qubits:>3}  {n_cx:>5}"
               f"  {fmt_int(ts_epr):>7}  {fmt_f(ts_time):>5}")
        for k in rkeys:
            r = router_results.get(k, {})
            epr = r.get("eprs") if not r.get("aborted") else None
            ls  = r.get("ls")   if not r.get("aborted") else None
            t   = r.get("time_s") if not r.get("aborted") else None
            row += f"  {fmt_int(epr):>8}  {fmt_int(ls):>7}  {fmt_f(t):>6}"
        for k in rkeys:
            r = router_results.get(k, {})
            epr = r.get("eprs") if not r.get("aborted") else None
            row += f"  {pct(epr, ts_epr):>8}"

        elapsed_wall = time.time() - t_wall
        print(row + f"  ({elapsed_wall:.0f}s)")
        records.append(rec)

    # ── geometric mean summary ──────────────────────────────────────────────
    def gmean(lst):
        lst = [x for x in lst if x is not None and x > 0]
        return prod(lst) ** (1/len(lst)) if lst else float("nan")

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
        summary += (f"  {gmean(eprs):>8.1f}  {gmean(lss):>7.1f}  {gmean(ts):>6.2f}")
    for k in rkeys:
        eprs = [r["routers"].get(k, {}).get("eprs") for r in records
                if not r["routers"].get(k, {}).get("aborted")]
        summary += f"  {pct(gmean(eprs), gmean(ts_eprs)):>8}"
    print(summary)

    return records


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["25", "36", "64", "all"], default="all")
    args = parser.parse_args()

    run_suites = []
    if args.suite in ("25", "all"): run_suites.append("25q")
    if args.suite in ("36", "all"): run_suites.append("36q")
    if args.suite in ("64", "all"): run_suites.append("64q")

    all_records = []
    t_total = time.time()

    for sname in run_suites:
        s = SUITES[sname]
        records = bench_suite(sname, s["hw"], s["dir"], s["suffix"])
        all_records.extend(records)

    # ── persist results ───────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(__file__), "results_summary.json")
    payload = {
        "meta": {
            "date":          time.strftime("%Y-%m-%d"),
            "arch":          "B-grid 2x2 4x4 (4 cores, 16 qubits/core)",
            "layout_passes": LAYOUT_PASSES,
            "ts_seeds":      NUM_TS_SEEDS,
            "routers": {
                "dS":  "General_dSABRE_Router  (router.py, Qiskit DAGCircuit)",
                "dSf": "General_dSABRE_Router  (router_test.py, LightDAG)",
                "dSE": "dSABRE_BurstExt        (burst_ext_router.py, BFS-layer extended set)",
            },
            "hw_25q": {
                "lookahead_size": hw_25.lookahead_size,
                "deadlock_limit": hw_25.deadlock_limit,
                "max_backup_attempts": hw_25.max_backup_attempts,
                "max_iterations": hw_25.max_iterations,
            },
            "hw_36q": {
                "lookahead_size": hw_36.lookahead_size,
                "deadlock_limit": hw_36.deadlock_limit,
                "max_backup_attempts": hw_36.max_backup_attempts,
                "max_iterations": hw_36.max_iterations,
                "max_burst_walk_depth": hw_36.max_burst_walk_depth,
            },
            "hw_64q": {
                "lookahead_size": hw_64.lookahead_size,
                "deadlock_limit": hw_64.deadlock_limit,
                "max_backup_attempts": hw_64.max_backup_attempts,
                "max_iterations": hw_64.max_iterations,
                "max_burst_walk_depth": hw_64.max_burst_walk_depth,
            },
        },
        "results": all_records,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)

    elapsed = time.time() - t_total
    print(f"\nResults saved → {out_path}  (total wall time {elapsed:.0f}s)")


if __name__ == "__main__":
    main()
