"""
run_64q_bench.py — dS / dSf / dSE on 64q circuits, H_grid_2_3_4_4 device
(6 cores × 16 qubits = 96 physical qubits; 32 qubits of slack for routing).

B_grid_2_2_4_4 (64 qubits) causes TeleSABRE segfaults and dSABRE deadlocks
on 64q circuits — no routing slack.  H_grid_2_3_4_4 is the correct device.

Protocol: same as run_summary_bench.py
  1. Run TS with seeds 0-2; pick the seed that gives the fewest EPRs.
  2. Extract TS's best initial layout (phys_to_virt from JSON report).
  3. Run dS, dSf, dSE under that layout (LAYOUT_PASSES=2).
  4. Print table + save results_64q.json.
"""

import sys, os, json, glob, subprocess, tempfile, time
from math import prod

sys.setrecursionlimit(50000)
sys.path.insert(0, os.path.dirname(__file__))

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler import PassManager

from architecture import build_h_grid_architecture
from config import HardwareConfig
import router as _router_mod
import router_test as _router_test_mod
from burst_ext_router import dSABRE_BurstExt
from main import _run_passes

TS_BIN = os.path.expanduser("~/Documents/telesabre/telesabre")
TS_DEV = os.path.expanduser("~/Documents/telesabre/devices/H_grid_2_3_4_4.json")

CIRCUIT_DIR   = os.path.expanduser("~/Documents/telesabre/circuits/qasm_64")
SUFFIX        = "_nativegates_ibm_qiskit_opt3_64.qasm"
LAYOUT_PASSES = 2
NUM_TS_SEEDS  = 3

arch = build_h_grid_architecture(r=2, s=3, m=4)
hw   = HardwareConfig(deadlock_limit=100, max_backup_attempts=100,
                      max_iterations=20000, max_burst_walk_depth=25)

routers = {
    "dS":  _router_mod.General_dSABRE_Router(arch, hw),
    "dSf": _router_test_mod.General_dSABRE_Router(arch, hw),
    "dSE": dSABRE_BurstExt(arch, hw),
}


def _ts_config(seed, report_path):
    cfg = {"config": {
        "name": "bench64", "seed": seed,
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


def run_telesabre(qasm_path):
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
                best = {"eprs": eprs, "teledata": td, "telegate": tg,
                        "seed": seed, "time_s": round(elapsed, 2), "p2v": p2v}
    return best


def p2v_to_layout(p2v, dag):
    qubits = dag.qubits
    return {qubits[v]: p for p, v in enumerate(p2v)
            if v != -1 and v < len(qubits)}


def load_qasm(qf):
    qc = QuantumCircuit.from_qasm_file(qf)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def main():
    qasm_files = sorted(glob.glob(os.path.join(CIRCUIT_DIR, "*.qasm")))
    if not qasm_files:
        print(f"No .qasm files in {CIRCUIT_DIR}"); return

    rkeys = list(routers.keys())
    col_w = 12
    hdr   = f"{'circuit':<{col_w}}  {'q':>3}  {'cx':>5}  {'TS_epr':>7}  {'TS_t':>5}"
    for k in rkeys:
        hdr += f"  {k+'_epr':>8}  {k+'_ls':>7}  {k+'_t':>6}"
    for k in rkeys:
        hdr += f"  {k+'/TS%':>8}"
    print(f"\n{'═'*len(hdr)}")
    print("  64q — H_grid_2_3_4_4 (6 cores × 16 qubits = 96 physical)")
    print(f"{'═'*len(hdr)}")
    print(hdr)
    print("─" * len(hdr))

    records = []

    for qf in qasm_files:
        cname  = os.path.basename(qf).replace(SUFFIX, "")
        t_wall = time.time()
        qc, dag = load_qasm(qf)
        n_qubits = qc.num_qubits
        n_cx     = sum(1 for _ in dag.two_qubit_ops())

        ts_result = run_telesabre(qf)

        rec = {"circuit": cname, "qubits": n_qubits, "cx": n_cx,
               "ts": ts_result, "routers": {}}

        router_results = {}
        if ts_result is not None:
            layout = p2v_to_layout(ts_result["p2v"], dag)
            if len(layout) >= n_qubits:
                for k, router in routers.items():
                    m = _run_passes(router, dag, layout, LAYOUT_PASSES)
                    if m and not m.get("aborted"):
                        router_results[k] = {"eprs": m["eprs"], "ls": m["ls"],
                                             "time_s": round(m["compile_time"], 3),
                                             "aborted": False}
                    else:
                        router_results[k] = {"aborted": True}

        rec["routers"] = router_results

        def fmt_int(v): return str(v) if v is not None else "---"
        def fmt_f(v):   return f"{v:.2f}" if v is not None else "  ---"
        def pct(a, b):
            if a is None or b is None or b == 0: return "    ---"
            return f"{100*(a-b)/b:+.1f}%"

        ts_epr  = ts_result["eprs"]   if ts_result else None
        ts_time = ts_result["time_s"] if ts_result else None

        row = (f"{cname:<{col_w}}  {n_qubits:>3}  {n_cx:>5}"
               f"  {fmt_int(ts_epr):>7}  {fmt_f(ts_time):>5}")
        for k in rkeys:
            r   = router_results.get(k, {})
            epr = r.get("eprs")   if not r.get("aborted") else None
            ls  = r.get("ls")     if not r.get("aborted") else None
            t   = r.get("time_s") if not r.get("aborted") else None
            row += f"  {fmt_int(epr):>8}  {fmt_int(ls):>7}  {fmt_f(t):>6}"
        for k in rkeys:
            r   = router_results.get(k, {})
            epr = r.get("eprs") if not r.get("aborted") else None
            row += f"  {pct(epr, ts_epr):>8}"

        elapsed_wall = time.time() - t_wall
        print(row + f"  ({elapsed_wall:.0f}s)")
        records.append(rec)

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
        summary += f"  {gmean(eprs):>8.1f}  {gmean(lss):>7.1f}  {gmean(ts):>6.2f}"
    for k in rkeys:
        eprs = [r["routers"].get(k, {}).get("eprs") for r in records
                if not r["routers"].get(k, {}).get("aborted")]
        summary += f"  {pct(gmean(eprs), gmean(ts_eprs)):>8}"
    print(summary)

    out_path = os.path.join(os.path.dirname(__file__), "results_64q.json")
    payload = {
        "meta": {"date": time.strftime("%Y-%m-%d"), "arch": "H-grid 2x3 4x4 (6 cores, 16 qubits/core)",
                 "layout_passes": LAYOUT_PASSES, "ts_seeds": NUM_TS_SEEDS,
                 "routers": {"dS": "General_dSABRE_Router (router.py)",
                             "dSf": "General_dSABRE_Router (router_test.py, LightDAG)",
                             "dSE": "dSABRE_BurstExt (burst_ext_router.py)"}},
        "results": records,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
