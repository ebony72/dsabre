"""
bench_bgrid_8links.py — Benchmark enhanced B-grid (8 inter-core links) vs original (4 links).

Enhanced B-grid adds 4 parallel inter-core links to the standard 2x2 4x4 B-grid:
  Original links: [7,20], [13,33], [30,50], [43,56]
  New links:      [11,24], [14,34], [39,52], [29,49]
  (parallel to originals, no corner nodes involved)

Runs 25q and 36q circuit suites on the enhanced architecture with all three compilers
(dS, dSE, TeleSABRE) and saves results for comparison with original B-grid.

Usage:
  python bench_bgrid_8links.py
  python bench_bgrid_8links.py --suite 25
  python bench_bgrid_8links.py --suite 36
"""

import sys, os, json, glob, subprocess, tempfile, time, argparse
from math import prod

sys.setrecursionlimit(50000)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler import PassManager

from architecture import DistributedArchitecture, build_b_grid_architecture
import networkx as nx
from config import HardwareConfig
from router import General_dSABRE_Router
from dsabre_ext import dSABRE_BurstExt
from layout import sabre_locked_boundary_layout, run_sabre_passes
from circuit_paths import circuits_path

_HERE        = os.path.dirname(os.path.abspath(__file__))
TS_BIN       = os.path.expanduser("~/Documents/telesabre/telesabre")
_RESULTS_DIR = os.environ.get("DSABRE_OUT_DIR") or os.path.join(_HERE, "results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

TS_DEV_8LINKS = os.path.expanduser(
    "~/Documents/telesabre/devices/B_grid_2_2_4_4_8links.json"
)


def build_b_grid_8links(r: int = 2, s: int = 2, m: int = 4) -> DistributedArchitecture:
    """B-grid with 8 inter-core links: original 4 + 4 parallel (no corners)."""
    base = build_b_grid_architecture(r, s, m)
    extra_links = [
        (11, 24),  # C0->C1: row 2 (parallel to [7,20] at row 1)
        (14, 34),  # C0->C2: col 2 (parallel to [13,33] at col 1)
        (39, 52),  # C2->C3: row 1 (parallel to [43,56] at row 2)
        (29, 49),  # C1->C3: col 1 (parallel to [30,50] at col 2)
    ]
    return DistributedArchitecture(base.intra, base.inter_core_links + extra_links)


_HW_SMALL = HardwareConfig()
_HW_LARGE = HardwareConfig(
    deadlock_limit=100, max_backup_attempts=100, max_iterations=20000
)

SUITES = {
    "25q": dict(
        circuit_dir=circuits_path("qasm_25"),
        suffix="_nativegates_ibm_qiskit_opt3_25.qasm",
        arch=build_b_grid_8links(),
        ts_dev=TS_DEV_8LINKS,
        hw=_HW_SMALL,
    ),
    "36q": dict(
        circuit_dir=circuits_path("qasm_36"),
        suffix="_nativegates_ibm_qiskit_opt3_36.qasm",
        arch=build_b_grid_8links(),
        ts_dev=TS_DEV_8LINKS,
        hw=_HW_LARGE,
    ),
}

NUM_TS_SEEDS = 3
NUM_SL_SEEDS = 3


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
    best = None
    for seed in range(NUM_TS_SEEDS):
        rpt = tempfile.mktemp(suffix=".json")
        cfg = _ts_config(seed, rpt, "bench_8links")
        t0 = time.perf_counter()
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
        td = tg = ls_ts = 0; ok = False
        for line in out.splitlines():
            s = line.strip()
            if   "Teledata:" in s: td    = int(s.split(":")[1])
            elif "Telegate:" in s: tg    = int(s.split(":")[1])
            elif "Swaps:"    in s: ls_ts = int(s.split(":")[1])
            elif "Success: true" in s: ok = True

        os.unlink(cfg)
        if os.path.exists(rpt): os.unlink(rpt)

        if ok:
            eprs = td + tg
            if best is None or eprs < best["eprs"]:
                best = dict(eprs=eprs, teledata=td, telegate=tg, ls=ls_ts,
                            seed=seed, time_s=round(elapsed, 2))
    return best


def load_qasm(path: str):
    qc = QuantumCircuit.from_qasm_file(path)
    qc = qc.remove_final_measurements(inplace=False)
    qc = PassManager([RemoveBarriers()]).run(qc)
    return qc, circuit_to_dag(qc)


def bench_suite(suite_name: str, s: dict) -> list:
    arch   = s["arch"]
    hw     = s["hw"]
    ts_dev = s["ts_dev"]
    suffix = s["suffix"]

    routers = {
        "dS":  General_dSABRE_Router(arch, hw),
        "dSE": dSABRE_BurstExt(arch, hw),
    }
    rkeys = list(routers.keys())

    qasm_files = sorted(glob.glob(os.path.join(s["circuit_dir"], "*.qasm")))
    if not qasm_files:
        print(f"  [no .qasm files in {s['circuit_dir']}]", flush=True)
        return []

    col_w = 12
    hdr = f"{'circuit':<{col_w}}  {'q':>3}  {'cx':>5}  {'TS_epr':>7}  {'TS_t':>5}"
    for k in rkeys:
        hdr += f"  {k+'_epr':>8}  {k+'_ls':>7}  {k+'_t':>6}"
    for k in rkeys:
        hdr += f"  {k+'/TS%':>8}"
    print(f"\n{'═'*len(hdr)}", flush=True)
    print(f"  {suite_name} [8-link B-grid]", flush=True)
    print(f"{'═'*len(hdr)}", flush=True)
    print(hdr, flush=True)
    print("─" * len(hdr), flush=True)

    records = []

    for qf in qasm_files:
        cname    = os.path.basename(qf).replace(suffix, "")
        t_wall   = time.time()
        qc, dag  = load_qasm(qf)
        rev_dag  = circuit_to_dag(qc.reverse_ops())
        n_qubits = qc.num_qubits
        n_cx     = sum(1 for _ in dag.two_qubit_ops())

        ts_result = run_telesabre(qf, ts_dev)

        sl_layouts = sabre_locked_boundary_layout(qc, dag, arch, seed=0)

        router_results = {}
        for k, router in routers.items():
            best_m = None
            for layout in sl_layouts:
                m = run_sabre_passes(router, dag, rev_dag, layout)
                if m and not m.get("aborted"):
                    if best_m is None or m["eprs"] < best_m["eprs"]:
                        best_m = m
            if best_m is not None:
                router_results[k] = dict(
                    eprs=best_m["eprs"], ls=best_m["ls"],
                    time_s=round(best_m["compile_time"], 3), aborted=False,
                    backup_activations = best_m.get("backup_activations", 0),
                    force_make_room    = best_m.get("force_make_room", 0),
                    safe_routes        = best_m.get("safe_routes", 0),
                    safe_route_failed  = best_m.get("safe_route_failed", 0),
                    relay_hops         = best_m.get("relay_hops", 0),
                )
            else:
                router_results[k] = {"aborted": True}

        def fmt_i(v): return str(v)    if v is not None else "---"
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
        print(row + f"  ({elapsed_wall:.0f}s)", flush=True)

        records.append(dict(
            suite=suite_name, circuit=cname, qubits=n_qubits, cx=n_cx,
            ts=ts_result, routers=router_results,
        ))

    def gmean(lst):
        lst = [x for x in lst if x is not None and x > 0]
        return prod(lst) ** (1 / len(lst)) if lst else float("nan")

    ts_eprs = [r["ts"]["eprs"] for r in records if r["ts"]]
    print("─" * len(hdr), flush=True)
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
    print(summary, flush=True)

    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=["25", "36", "all"], default="all")
    args = parser.parse_args()

    run_suites = []
    if args.suite in ("25", "all"): run_suites.append("25q")
    if args.suite in ("36", "all"): run_suites.append("36q")

    t_total = time.time()
    all_records = {}
    for sname in run_suites:
        records = bench_suite(sname, SUITES[sname])
        all_records[sname] = records
        out_path = os.path.join(_RESULTS_DIR, f"results_{sname}_8links.json")
        payload = dict(
            meta=dict(
                date          = time.strftime("%Y-%m-%d"),
                suite         = sname,
                arch          = "B-grid 2x2 4x4 (64 qubits) — 8 inter-core links",
                extra_links   = [[11,24],[14,34],[39,52],[29,49]],
                layout        = "SabreLayout corners-removed, best of 3 seeds",
                pass_strategy = "fwd -> bwd (reversed DAG) -> fwd; best of pass1/pass3",
                ts_seeds      = NUM_TS_SEEDS,
                sl_seeds      = NUM_SL_SEEDS,
                routers       = {
                    "dS":  "General_dSABRE_Router  (router.py)",
                    "dSE": "dSABRE_BurstExt        (dsabre_ext.py)",
                },
            ),
            results=records,
        )
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nSaved → {out_path}", flush=True)

    print(f"\nTotal wall time: {time.time() - t_total:.0f}s", flush=True)


if __name__ == "__main__":
    main()
